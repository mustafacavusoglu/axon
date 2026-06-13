use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "axon-server",
    version = "0.3.5",
    about = "Axon Inference Server — CPU ONNX serving"
)]
pub struct ServerConfig {
    #[arg(long, default_value = "/models")]
    pub model_repository: PathBuf,

    #[arg(long, default_value = "none", help = "Model control mode: none, poll")]
    pub model_control_mode: String,

    #[arg(long, default_value_t = 30)]
    pub repository_poll_secs: u64,

    #[arg(long, default_value_t = 8000)]
    pub http_port: u16,

    #[arg(long, default_value_t = 8001)]
    pub grpc_port: u16,

    #[arg(long, default_value_t = 8002)]
    pub metrics_port: u16,

    #[arg(long, default_value_t = 30000)]
    pub inference_timeout_ms: u64,

    #[arg(long, default_value_t = 0, help = "Worker threads (0 = auto-detect)")]
    pub num_threads: usize,

    #[arg(long, default_value_t = 4)]
    pub concurrency_per_model: u32,

    #[arg(
        long,
        default_value = "info",
        help = "Log level: trace, debug, info, warn, error"
    )]
    pub log_level: String,

    #[arg(
        long,
        default_value = "/tmp/logs/axon",
        help = "Log directory for file output (daily rotation)"
    )]
    pub log_dir: PathBuf,
}
