use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub socket_path: PathBuf,
    pub num_threads: usize,
    pub arena_size_mb: usize,
    pub inference_timeout_ms: u64,
}

impl EngineConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            socket_path: std::env::var("SOCKET_PATH")
                .unwrap_or_else(|_| "/run/inference.sock".into())
                .into(),
            num_threads: std::env::var("NUM_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n| n > 0)
                .unwrap_or_else(num_cpus::get_physical),
            arena_size_mb: std::env::var("ARENA_SIZE_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096),
            inference_timeout_ms: std::env::var("INFERENCE_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30000),
        })
    }
}
