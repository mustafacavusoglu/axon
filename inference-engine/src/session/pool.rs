use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use dashmap::DashMap;
use tokio::sync::Semaphore;

use crate::session::runner::ModelRunner;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Ready,
}

pub struct ModelSession {
    pub name: String,
    pub version: u32,
    pub state: SessionState,
    pub runner: Arc<ModelRunner>,
}

impl ModelSession {
    pub fn concurrency(&self) -> &Arc<Semaphore> {
        self.runner.concurrency_semaphore()
    }
}

fn model_key(name: &str, version: u32) -> String {
    format!("{}@v{}", name, version)
}

#[derive(Clone)]
pub struct SessionPool {
    sessions: Arc<DashMap<String, Arc<ModelSession>>>,
}

impl SessionPool {
    pub fn new(num_threads: usize) -> anyhow::Result<Self> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("infer-worker-{}", i))
            .build_global()
            .context("failed to build rayon thread pool")?;

        tracing::info!(num_threads, "rayon thread pool initialised");

        Ok(Self {
            sessions: Arc::new(DashMap::new()),
        })
    }

    pub fn load_model(
        &self,
        name: &str,
        version: u32,
        model_path: &Path,
        concurrency: u32,
    ) -> anyhow::Result<Arc<ModelSession>> {
        let key = model_key(name, version);

        if let Some(existing) = self.sessions.get(&key) {
            if existing.state == SessionState::Ready {
                return Ok(existing.clone());
            }
        }

        if !model_path.exists() {
            anyhow::bail!("model file not found: {}", model_path.display());
        }

        let count = if concurrency > 0 {
            concurrency as usize
        } else {
            4
        };

        let runner = ModelRunner::load_onnx(model_path, count)?;

        let session = Arc::new(ModelSession {
            name: name.to_string(),
            version,
            state: SessionState::Ready,
            runner: Arc::new(runner),
        });

        self.sessions.insert(key, session.clone());
        tracing::info!(name, version, instances = count, "model loaded");
        Ok(session)
    }

    pub fn load_script_model(
        &self,
        name: &str,
        version: u32,
        script_path: &Path,
        concurrency: u32,
    ) -> anyhow::Result<Arc<ModelSession>> {
        let key = model_key(name, version);

        if let Some(existing) = self.sessions.get(&key) {
            if existing.state == SessionState::Ready {
                return Ok(existing.clone());
            }
        }

        if !script_path.exists() {
            anyhow::bail!("script file not found: {}", script_path.display());
        }

        let count = if concurrency > 0 {
            concurrency as usize
        } else {
            4
        };

        let runner = ModelRunner::load_rhai(script_path, self.clone(), count)?;

        let session = Arc::new(ModelSession {
            name: name.to_string(),
            version,
            state: SessionState::Ready,
            runner: Arc::new(runner),
        });

        self.sessions.insert(key, session.clone());
        tracing::info!(name, version, instances = count, "script model loaded");
        Ok(session)
    }

    pub fn unload_model(&self, name: &str, version: u32) -> anyhow::Result<()> {
        let key = model_key(name, version);
        match self.sessions.remove(&key) {
            Some(_) => {
                tracing::info!(name, version, "model unloaded");
                Ok(())
            }
            None => anyhow::bail!("model not found: {}", key),
        }
    }

    pub fn get(&self, name: &str, version: u32) -> Option<Arc<ModelSession>> {
        let key = model_key(name, version);
        self.sessions.get(&key).map(|r| r.clone())
    }

    pub fn get_latest(&self, name: &str) -> Option<Arc<ModelSession>> {
        let mut latest: Option<Arc<ModelSession>> = None;
        for entry in self.sessions.iter() {
            let s = entry.value();
            if s.name == name && s.state == SessionState::Ready {
                if latest.as_ref().map_or(true, |l| s.version > l.version) {
                    latest = Some(s.clone());
                }
            }
        }
        latest
    }

    pub fn list_models(&self) -> Vec<(String, u32, SessionState)> {
        self.sessions
            .iter()
            .map(|entry| {
                let s = entry.value();
                (s.name.clone(), s.version, s.state)
            })
            .collect()
    }

    pub fn model_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn get_versions(&self, name: &str) -> Vec<u32> {
        let mut versions: Vec<u32> = self
            .sessions
            .iter()
            .filter(|e| e.value().name == name)
            .map(|e| e.value().version)
            .collect();
        versions.sort();
        versions
    }

    pub fn all_model_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .sessions
            .iter()
            .map(|e| e.value().name.clone())
            .collect();
        names.sort();
        names.dedup();
        names
    }
}
