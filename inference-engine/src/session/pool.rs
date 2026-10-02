use std::sync::Arc;
use std::time::SystemTime;

use dashmap::DashMap;
use tokio::sync::Semaphore;

use crate::model_repository::config_parser::ModelConfig;
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
    /// Parsed model config, cached at load time so metadata requests never
    /// touch the filesystem.
    pub config: Option<Arc<ModelConfig>>,
    /// Change-detection fingerprint (newest mtime of the files the model was
    /// built from). Used by the repository poller to skip unchanged models.
    pub fingerprint: Option<SystemTime>,
}

impl ModelSession {
    pub fn concurrency(&self) -> &Arc<Semaphore> {
        self.runner.concurrency_semaphore()
    }
}

fn model_key(name: &str, version: u32) -> String {
    format!("{name}@v{version}")
}

#[derive(Clone)]
pub struct SessionPool {
    sessions: Arc<DashMap<String, Arc<ModelSession>>>,
    pub cpu_semaphore: Arc<Semaphore>,
}

impl SessionPool {
    pub fn new(num_threads: usize) -> anyhow::Result<Self> {
        let cpu_limit = if num_threads > 0 {
            num_threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        };
        tracing::info!(
            cpu_limit,
            "session pool initialised (CPU semaphore + ONNX Runtime threading)"
        );

        Ok(Self {
            sessions: Arc::new(DashMap::new()),
            cpu_semaphore: Arc::new(Semaphore::new(cpu_limit)),
        })
    }

    /// Returns true when `name@version` is loaded and was built from files
    /// with the given fingerprint, i.e. a reload would be a no-op.
    pub fn is_current(&self, name: &str, version: u32, fingerprint: Option<SystemTime>) -> bool {
        let key = model_key(name, version);
        // Copy the value out so the shard guard is released immediately.
        let existing = self.sessions.get(&key).map(|r| r.fingerprint);
        matches!(existing, Some(fp) if fp.is_some() && fp == fingerprint)
    }

    /// Atomically publishes a freshly built runner. The previous session (if
    /// any) keeps serving in-flight requests through its own `Arc` and is
    /// dropped once they finish, so a reload never produces a 404 window and a
    /// failed reload never removes a working model.
    pub fn insert(
        &self,
        name: &str,
        version: u32,
        runner: ModelRunner,
        config: Option<Arc<ModelConfig>>,
        fingerprint: Option<SystemTime>,
    ) -> Arc<ModelSession> {
        let session = Arc::new(ModelSession {
            name: name.to_string(),
            version,
            state: SessionState::Ready,
            runner: Arc::new(runner),
            config,
            fingerprint,
        });
        self.sessions
            .insert(model_key(name, version), session.clone());
        session
    }

    pub fn unload_model(&self, name: &str, version: u32) -> anyhow::Result<()> {
        let key = model_key(name, version);
        match self.sessions.remove(&key) {
            Some(_) => {
                tracing::info!(target: "axon::console", name, version, "model unloaded");
                Ok(())
            }
            None => anyhow::bail!("model not found: {key}"),
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
            if s.name == name
                && s.state == SessionState::Ready
                && latest.as_ref().is_none_or(|l| s.version > l.version)
            {
                latest = Some(s.clone());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::runner::ModelRunner;

    fn dummy_runner() -> ModelRunner {
        ModelRunner::Noop(Arc::new(Semaphore::new(1)))
    }

    #[test]
    fn reload_replaces_without_deadlock() {
        let pool = SessionPool::new(1).unwrap();
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + std::time::Duration::from_secs(1);

        pool.insert("m", 1, dummy_runner(), None, Some(t0));
        assert!(pool.is_current("m", 1, Some(t0)));
        assert!(!pool.is_current("m", 1, Some(t1)));

        let old = pool.get("m", 1).unwrap();
        pool.insert("m", 1, dummy_runner(), None, Some(t1));
        let new = pool.get("m", 1).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        assert_eq!(pool.model_count(), 1);
    }

    #[test]
    fn missing_fingerprint_is_never_current() {
        let pool = SessionPool::new(1).unwrap();
        pool.insert("m", 1, dummy_runner(), None, None);
        assert!(!pool.is_current("m", 1, None));
    }

    #[test]
    fn latest_version_wins() {
        let pool = SessionPool::new(1).unwrap();
        pool.insert("m", 1, dummy_runner(), None, None);
        pool.insert("m", 3, dummy_runner(), None, None);
        pool.insert("m", 2, dummy_runner(), None, None);
        assert_eq!(pool.get_latest("m").unwrap().version, 3);
        assert_eq!(pool.get_versions("m"), vec![1, 2, 3]);
        pool.unload_model("m", 3).unwrap();
        assert_eq!(pool.get_latest("m").unwrap().version, 2);
    }
}
