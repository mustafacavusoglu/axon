use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use tokio::sync::watch;

use crate::metrics;
use crate::session::pool::SessionPool;

pub mod circuit_breaker;
pub mod config_parser;
pub use config_parser::ModelConfig;

use circuit_breaker::CircuitBreaker;

static CIRCUIT_BREAKER: std::sync::OnceLock<Mutex<CircuitBreaker>> = std::sync::OnceLock::new();

fn get_circuit_breaker() -> &'static Mutex<CircuitBreaker> {
    CIRCUIT_BREAKER.get_or_init(|| Mutex::new(CircuitBreaker::new()))
}

pub async fn load_all_models(repo_path: &Path, pool: &SessionPool) {
    let repo = repo_path.to_path_buf();
    let pool = pool.clone();

    let result = tokio::task::spawn_blocking(move || load_all_models_sync(&repo, &pool)).await;

    if let Err(e) = result {
        tracing::error!(error = %e, "model loading task panicked");
    }
}

fn load_all_models_sync(repo_path: &Path, pool: &SessionPool) {
    if !repo_path.is_dir() {
        tracing::warn!(path = %repo_path.display(), "model repository not found");
        return;
    }

    let entries = match std::fs::read_dir(repo_path) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!(error = %e, "failed to read model repository");
            return;
        }
    };

    for entry in entries.flatten() {
        let model_dir = entry.path();
        if !model_dir.is_dir() {
            continue;
        }

        let model_name = match model_dir.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        if !is_valid_model_name(&model_name) {
            tracing::warn!(name = %model_name, "skipping model with invalid name");
            continue;
        }

        let config = load_model_config(&model_dir);
        let platform = config
            .as_ref()
            .map(|c| c.platform.clone())
            .unwrap_or_else(|| "onnxruntime_onnx".to_string());

        let concurrency = config
            .as_ref()
            .and_then(|c| c.instance_groups.first())
            .map(|ig| ig.count as u32)
            .unwrap_or(4);

        let versions = discover_versions(&model_dir);
        if versions.is_empty() {
            tracing::warn!(model = %model_name, "no model versions found");
            continue;
        }

        for version in versions {
            let cb_key = format!("{model_name}@v{version}");

            if let Ok(cb) = get_circuit_breaker().lock() {
                if cb.is_open(&cb_key) {
                    tracing::debug!(model = %model_name, version, "circuit open, skipping");
                    metrics::record_circuit_breaker_trip();
                    continue;
                }
            }

            if platform == "script" {
                let script_file = model_dir
                    .join(version.to_string())
                    .join("model.rhai");
                if !script_file.exists() {
                    tracing::warn!(model = %model_name, version, "model.rhai not found, skipping");
                    continue;
                }

                let load_start = std::time::Instant::now();
                match pool.load_script_model(&model_name, version, &script_file, concurrency) {
                    Ok(_) => {
                        let load_secs = load_start.elapsed().as_secs_f64();
                        metrics::record_model_load_duration(&model_name, load_secs);
                        metrics::set_model_ready(&model_name, version);
                        if let Ok(mut cb) = get_circuit_breaker().lock() {
                            cb.record_success(&cb_key);
                        }
                    }
                    Err(e) => {
                        tracing::error!(model = %model_name, version, error = %e, "failed to load script model");
                        metrics::record_model_load_error(&model_name);
                        if let Ok(mut cb) = get_circuit_breaker().lock() {
                            cb.record_failure(&cb_key);
                        }
                    }
                }
            } else if platform == "ensemble" {
                let config = match config.as_ref() {
                    Some(c) => c,
                    None => {
                        tracing::warn!(model = %model_name, "ensemble model missing config");
                        continue;
                    }
                };

                let load_start = std::time::Instant::now();
                match pool.load_ensemble_model(&model_name, version, config, concurrency) {
                    Ok(_) => {
                        let load_secs = load_start.elapsed().as_secs_f64();
                        metrics::record_model_load_duration(&model_name, load_secs);
                        metrics::set_model_ready(&model_name, version);
                        if let Ok(mut cb) = get_circuit_breaker().lock() {
                            cb.record_success(&cb_key);
                        }
                    }
                    Err(e) => {
                        tracing::error!(model = %model_name, version, error = %e, "failed to load ensemble model");
                        metrics::record_model_load_error(&model_name);
                        if let Ok(mut cb) = get_circuit_breaker().lock() {
                            cb.record_failure(&cb_key);
                        }
                    }
                }
            } else {
                let model_file = model_dir.join(version.to_string()).join("model.onnx");
                if !model_file.exists() {
                    tracing::warn!(model = %model_name, version, "model.onnx not found, skipping");
                    continue;
                }

                let load_start = std::time::Instant::now();
                match pool.load_model(&model_name, version, &model_file, concurrency) {
                    Ok(_) => {
                        let load_secs = load_start.elapsed().as_secs_f64();
                        metrics::record_model_load_duration(&model_name, load_secs);
                        metrics::set_model_ready(&model_name, version);
                        if let Ok(mut cb) = get_circuit_breaker().lock() {
                            cb.record_success(&cb_key);
                        }
                    }
                    Err(e) => {
                        tracing::error!(model = %model_name, version, error = %e, "failed to load model");
                        metrics::record_model_load_error(&model_name);
                        if let Ok(mut cb) = get_circuit_breaker().lock() {
                            cb.record_failure(&cb_key);
                        }
                    }
                }
            }
        }
    }
}

pub async fn poll_loop(
    repo_path: PathBuf,
    pool: SessionPool,
    interval_secs: u64,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
    interval.tick().await;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                load_all_models(&repo_path, &pool).await;
                metrics::set_models_count(pool.model_count() as i64);
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow_and_update() {
                    tracing::info!("poll loop shutting down");
                    return;
                }
            }
        }
    }
}

fn discover_versions(model_dir: &Path) -> Vec<u32> {
    let mut versions = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if let Some(n) = name.to_str() {
                if let Ok(v) = n.parse::<u32>() {
                    if v > 0 && entry.path().is_dir() {
                        versions.push(v);
                    }
                }
            }
        }
    }
    versions.sort();
    versions
}

fn load_model_config(model_dir: &Path) -> Option<ModelConfig> {
    let yaml_path = model_dir.join("config.yaml");
    if yaml_path.exists() {
        match std::fs::read(&yaml_path) {
            Ok(content) => match config_parser::parse_model_config_yaml(&content) {
                Ok(cfg) => return Some(cfg),
                Err(e) => {
                    tracing::warn!(path = %yaml_path.display(), error = %e, "failed to parse config.yaml, falling back to config.pbtxt");
                }
            },
            Err(e) => {
                tracing::warn!(path = %yaml_path.display(), error = %e, "failed to read config.yaml");
            }
        }
    }

    let config_path = model_dir.join("config.pbtxt");
    if !config_path.exists() {
        return None;
    }
    match std::fs::read(&config_path) {
        Ok(content) => match config_parser::parse_model_config(&content) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                tracing::warn!(path = %config_path.display(), error = %e, "failed to parse config");
                None
            }
        },
        Err(e) => {
            tracing::warn!(path = %config_path.display(), error = %e, "failed to read config");
            None
        }
    }
}

fn is_valid_model_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        && !name.starts_with('.')
        && !name.contains("..")
}
