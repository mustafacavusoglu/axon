use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::sync::watch;

use crate::metrics;
use crate::session::pool::SessionPool;

pub mod config_parser;
pub use config_parser::ModelConfig;

pub async fn load_all_models(repo_path: &Path, pool: &SessionPool) {
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
            let model_file = model_dir.join(version.to_string()).join("model.onnx");
            if !model_file.exists() {
                tracing::warn!(model = %model_name, version, "model.onnx not found, skipping");
                continue;
            }

            match pool.load_model(&model_name, version, &model_file, concurrency) {
                Ok(_) => {}
                Err(e) => {
                    tracing::error!(model = %model_name, version, error = %e, "failed to load model");
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
