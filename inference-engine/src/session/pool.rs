use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicU64};

use anyhow::Context;
use dashmap::DashMap;
use tokio::sync::Semaphore;

use crate::session::runner::ModelRunner;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Loading,
    Ready,
    Unloading,
    Error,
}

pub struct ModelSession {
    pub name: String,
    pub version: u32,
    pub state: SessionState,
    pub memory_bytes: AtomicU64,
    pub runner: Arc<ModelRunner>,
    pub concurrency: Arc<Semaphore>,
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
        model_path: &PathBuf,
        concurrency: u32,
    ) -> anyhow::Result<Arc<ModelSession>> {
        let key = model_key(name, version);

        if let Some(existing) = self.sessions.get(&key) {
            if existing.state == SessionState::Ready {
                return Ok(existing.clone());
            }
        }

        let model_file = model_path.join("model.onnx");
        let runner = ModelRunner::load(&model_file)?;

        let count = if concurrency > 0 { concurrency } else { 1 };
        let session = Arc::new(ModelSession {
            name: name.to_string(),
            version,
            state: SessionState::Ready,
            memory_bytes: AtomicU64::new(runner.estimate_memory()),
            runner: Arc::new(runner),
            concurrency: Arc::new(Semaphore::new(count as usize)),
        });

        self.sessions.insert(key, session.clone());
        tracing::info!(name, version, "model loaded");
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

    pub fn get(&self, name: &str, version: u32) -> anyhow::Result<Arc<ModelSession>> {
        let key = model_key(name, version);
        self.sessions
            .get(&key)
            .map(|r| r.clone())
            .ok_or_else(|| anyhow::anyhow!("model not loaded: {}", key))
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
}
