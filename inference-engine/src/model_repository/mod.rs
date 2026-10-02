use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use tokio::sync::watch;

use crate::metrics;
use crate::session::device::ExecutionConfig;
use crate::session::pool::SessionPool;
use crate::session::runner::ModelRunner;

pub mod circuit_breaker;
pub mod config_parser;
pub use config_parser::ModelConfig;

use circuit_breaker::CircuitBreaker;

static CIRCUIT_BREAKER: std::sync::OnceLock<Mutex<CircuitBreaker>> = std::sync::OnceLock::new();

fn get_circuit_breaker() -> &'static Mutex<CircuitBreaker> {
    CIRCUIT_BREAKER.get_or_init(|| Mutex::new(CircuitBreaker::new()))
}

/// Everything needed to (re)load models from the repository.
#[derive(Clone)]
pub struct Repository {
    pub path: PathBuf,
    pub pool: SessionPool,
    pub default_concurrency: u32,
    pub exec: Arc<ExecutionConfig>,
}

impl Repository {
    /// Scans the whole repository and loads new or changed models. With
    /// `prune`, model versions whose directories disappeared are unloaded.
    pub async fn load_all(&self, prune: bool) {
        let repo = self.clone();
        let result = tokio::task::spawn_blocking(move || repo.load_all_sync(prune)).await;
        if let Err(e) = result {
            tracing::error!(error = %e, "model loading task panicked");
        }
        metrics::set_models_count(self.pool.model_count() as i64);
    }

    /// Loads (or reloads) a single model from the repository, all versions
    /// or just `version`. Used by the explicit model-control API.
    pub async fn load_one(&self, name: String, version: Option<u32>) -> anyhow::Result<()> {
        let repo = self.clone();
        let result =
            tokio::task::spawn_blocking(move || repo.load_model_dir(&name, version, true)).await;
        metrics::set_models_count(self.pool.model_count() as i64);
        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(e) => anyhow::bail!("model loading task panicked: {e}"),
        }
    }

    fn load_all_sync(&self, prune: bool) {
        if !self.path.is_dir() {
            tracing::warn!(path = %self.path.display(), "model repository not found");
            return;
        }

        let entries = match std::fs::read_dir(&self.path) {
            Ok(e) => e,
            Err(e) => {
                tracing::error!(error = %e, "failed to read model repository");
                return;
            }
        };

        let mut present: HashSet<(String, u32)> = HashSet::new();
        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let Some(model_name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !is_valid_model_name(&model_name) {
                tracing::warn!(name = %model_name, "skipping model with invalid name");
                continue;
            }
            match self.load_model_dir(&model_name, None, false) {
                Ok(versions) => {
                    present.extend(versions.into_iter().map(|v| (model_name.clone(), v)));
                }
                Err(e) => {
                    tracing::warn!(model = %model_name, error = %e, "skipping model");
                }
            }
        }

        if prune {
            for (name, version, _) in self.pool.list_models() {
                if !present.contains(&(name.clone(), version))
                    && self.pool.unload_model(&name, version).is_ok()
                {
                    metrics::clear_model(&name, version);
                }
            }
        }
    }

    /// Loads all versions (or one) of `model_name`. Returns the versions that
    /// exist on disk, whether or not (re)loading them succeeded, so a broken
    /// update never unloads the previously working model.
    fn load_model_dir(
        &self,
        model_name: &str,
        only_version: Option<u32>,
        force: bool,
    ) -> anyhow::Result<Vec<u32>> {
        if !is_valid_model_name(model_name) {
            anyhow::bail!("invalid model name '{model_name}'");
        }
        let model_dir = self.path.join(model_name);
        if !model_dir.is_dir() {
            anyhow::bail!("model '{model_name}' not found in repository");
        }

        let config = load_model_config(&model_dir).map(Arc::new);
        let platform = config
            .as_ref()
            .map(|c| c.platform.clone())
            .unwrap_or_else(|| "onnxruntime_onnx".to_string());
        let group = config.as_ref().and_then(|c| c.instance_groups.first());
        let concurrency = group
            .map(|ig| ig.count)
            .filter(|c| *c > 0)
            .map(|c| c as u32)
            .unwrap_or(self.default_concurrency)
            .clamp(1, 1024);
        let placement = self.exec.placement(
            group.map(|g| g.kind.as_str()),
            group.map(|g| g.gpus.as_slice()).unwrap_or(&[]),
        );
        let config_mtime = newest_mtime(&[
            model_dir.join("config.yaml"),
            model_dir.join("config.pbtxt"),
        ]);

        let mut versions = discover_versions(&model_dir);
        if let Some(v) = only_version {
            if !versions.contains(&v) {
                anyhow::bail!("version {v} of model '{model_name}' not found");
            }
            versions = vec![v];
        }
        if versions.is_empty() {
            anyhow::bail!("no model versions found");
        }

        for &version in &versions {
            let version_dir = model_dir.join(version.to_string());
            let fingerprint = newest_mtime_in_dir(&version_dir).max(config_mtime);
            if !force && self.pool.is_current(model_name, version, fingerprint) {
                continue;
            }

            let cb_key = format!("{model_name}@v{version}");
            if !force
                && get_circuit_breaker()
                    .lock()
                    .is_ok_and(|cb| cb.is_open(&cb_key))
            {
                tracing::debug!(model = %model_name, version, "circuit open, skipping");
                metrics::record_circuit_breaker_trip();
                continue;
            }

            let load_start = std::time::Instant::now();
            let result: anyhow::Result<ModelRunner> = match platform.as_str() {
                "script" => {
                    let script_file = version_dir.join("model.rhai");
                    if !script_file.exists() {
                        tracing::warn!(model = %model_name, version, "model.rhai not found, skipping");
                        continue;
                    }
                    ModelRunner::load_rhai(&script_file, self.pool.clone(), concurrency as usize)
                }
                "ensemble" => match config.as_ref() {
                    Some(c) => {
                        ModelRunner::load_ensemble(c, self.pool.clone(), concurrency as usize)
                    }
                    None => {
                        tracing::warn!(model = %model_name, "ensemble model missing config");
                        continue;
                    }
                },
                _ => {
                    let model_file = version_dir.join("model.onnx");
                    if !model_file.exists() {
                        tracing::warn!(model = %model_name, version, "model.onnx not found, skipping");
                        continue;
                    }
                    let effective =
                        cap_for_external_data(model_name, version, &version_dir, concurrency);
                    ModelRunner::load_onnx(&model_file, effective as usize, &self.exec, &placement)
                }
            };

            match result {
                Ok(runner) => {
                    let device = runner.device_label();
                    self.pool
                        .insert(model_name, version, runner, config.clone(), fingerprint);
                    metrics::record_model_load_duration(
                        model_name,
                        load_start.elapsed().as_secs_f64(),
                    );
                    metrics::set_model_ready(model_name, version);
                    tracing::info!(
                        target: "axon::console",
                        name = model_name,
                        version,
                        instances = concurrency,
                        device = %device,
                        platform = %platform,
                        "model loaded"
                    );
                    if let Ok(mut cb) = get_circuit_breaker().lock() {
                        cb.record_success(&cb_key);
                    }
                }
                Err(e) => {
                    tracing::error!(model = %model_name, version, error = %e, "failed to load model");
                    metrics::record_model_load_error(model_name);
                    if let Ok(mut cb) = get_circuit_breaker().lock() {
                        cb.record_failure(&cb_key);
                    }
                    if force {
                        return Err(e);
                    }
                }
            }
        }

        Ok(versions)
    }
}

/// If external data files are present (e.g. bert.onnx.data) cap concurrency
/// at 2: loading N sessions × a large .data file causes an apparent hang.
fn cap_for_external_data(
    model_name: &str,
    version: u32,
    version_dir: &Path,
    concurrency: u32,
) -> u32 {
    let has_external = std::fs::read_dir(version_dir)
        .map(|mut d| {
            d.any(|e| {
                e.ok()
                    .and_then(|e| e.file_name().into_string().ok())
                    .is_some_and(|n| n.ends_with(".data"))
            })
        })
        .unwrap_or(false);
    if !has_external {
        return concurrency;
    }
    let cap = concurrency.clamp(1, 2);
    if concurrency > cap {
        tracing::info!(
            model = %model_name,
            version,
            original = concurrency,
            capped = cap,
            "external data detected — capping concurrency to avoid load hang"
        );
    }
    cap
}

pub async fn poll_loop(repo: Repository, interval_secs: u64, mut shutdown: watch::Receiver<bool>) {
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs.max(1)));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    interval.tick().await;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                repo.load_all(true).await;
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

fn newest_mtime(paths: &[PathBuf]) -> Option<SystemTime> {
    paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        .max()
}

/// Newest mtime among the files directly inside `dir` (model.onnx, external
/// weights, model.rhai, tokenizer.json, vocab files, ...).
fn newest_mtime_in_dir(dir: &Path) -> Option<SystemTime> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| e.metadata().and_then(|m| m.modified()).ok())
        .max()
}

pub fn load_model_config(model_dir: &Path) -> Option<ModelConfig> {
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

pub fn is_valid_model_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        && !name.starts_with('.')
        && !name.contains("..")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_validation() {
        assert!(is_valid_model_name("bert-base_v1.2"));
        assert!(!is_valid_model_name(""));
        assert!(!is_valid_model_name("../etc"));
        assert!(!is_valid_model_name("a/b"));
        assert!(!is_valid_model_name(".hidden"));
        assert!(!is_valid_model_name("a..b"));
        assert!(!is_valid_model_name("modèle"));
        assert!(!is_valid_model_name(&"x".repeat(129)));
    }
}
