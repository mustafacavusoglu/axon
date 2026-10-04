use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;

use crate::session::device::{parse_gpu_ids, DevicePreference, ExecutionConfig};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "axon-server",
    version,
    about = "Axon Inference Server — ONNX serving on CPU and GPU"
)]
pub struct ServerConfig {
    #[arg(long, default_value = "/models")]
    pub model_repository: PathBuf,

    #[arg(
        long,
        default_value = "none",
        value_parser = ["none", "poll", "explicit"],
        help = "Model control mode: none, poll (auto-reload), explicit (load/unload API enabled)"
    )]
    pub model_control_mode: String,

    #[arg(long, default_value_t = 30)]
    pub repository_poll_secs: u64,

    #[arg(
        long,
        env = "AXON_HOST",
        default_value = "0.0.0.0",
        help = "Address the HTTP, gRPC and metrics servers bind to"
    )]
    pub host: String,

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
        env = "AXON_DEVICE",
        default_value = "cpu",
        help = "Default execution device: cpu, auto (GPU if available), cuda, tensorrt"
    )]
    pub device: DevicePreference,

    #[arg(
        long,
        env = "AXON_GPU_DEVICE_IDS",
        default_value = "0",
        value_parser = validate_gpu_ids,
        help = "Comma separated GPU ids; model instances are spread round-robin"
    )]
    pub gpu_device_ids: String,

    #[arg(
        long,
        env = "AXON_GPU_MEM_LIMIT_MB",
        default_value_t = 0,
        help = "Per-session GPU memory arena limit in MiB (0 = unlimited)"
    )]
    pub gpu_mem_limit_mb: usize,

    #[arg(long, env = "AXON_TRT_FP16", help = "Enable FP16 kernels for TensorRT")]
    pub trt_fp16: bool,

    #[arg(
        long,
        env = "AXON_TRT_CACHE_DIR",
        help = "Directory for TensorRT engine/timing caches (speeds up restarts)"
    )]
    pub trt_cache_dir: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = 1,
        help = "ONNX Runtime intra-op threads per session instance"
    )]
    pub intra_op_threads: usize,

    #[arg(
        long,
        help = "ONNX session creation timeout in seconds (default: 10 CPU, 120 CUDA, 900 TensorRT)"
    )]
    pub model_load_timeout_secs: Option<u64>,

    #[arg(
        long,
        env = "AXON_UI",
        help = "Serve the built-in web UI (model list + inference playground) at /ui"
    )]
    pub ui: bool,

    #[arg(
        long,
        env = "AXON_API_KEY",
        hide_env_values = true,
        help = "Require 'Authorization: Bearer <key>' on all non-health endpoints"
    )]
    pub api_key: Option<String>,

    #[arg(
        long,
        default_value = "info",
        help = "Log level: trace, debug, info, warn, error"
    )]
    pub log_level: String,

    #[arg(
        long,
        default_value = "/tmp/logs/axon",
        help = "Log directory for file output"
    )]
    pub log_dir: PathBuf,

    #[arg(
        long,
        default_value = "7d",
        help = "Log rotation: <count><unit> e.g. 7d (daily, keep 7), 24h (hourly, keep 24)"
    )]
    pub log_rotation: String,
}

fn validate_gpu_ids(s: &str) -> Result<String, String> {
    parse_gpu_ids(s).map(|_| s.to_string())
}

impl ServerConfig {
    pub fn execution_config(&self) -> ExecutionConfig {
        ExecutionConfig {
            default_device: self.device,
            gpu_device_ids: parse_gpu_ids(&self.gpu_device_ids).unwrap_or_else(|_| vec![0]),
            gpu_mem_limit: self.gpu_mem_limit_mb.saturating_mul(1024 * 1024),
            trt_fp16: self.trt_fp16,
            trt_cache_dir: self.trt_cache_dir.clone(),
            intra_op_threads: self.intra_op_threads.max(1),
            load_timeout: self.model_load_timeout_secs.map(Duration::from_secs),
        }
    }

    /// API key, ignoring empty values (e.g. `AXON_API_KEY=` in compose files).
    pub fn api_key(&self) -> Option<String> {
        self.api_key.clone().filter(|k| !k.trim().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gpu_flags() {
        let cfg = ServerConfig::try_parse_from([
            "axon-server",
            "--device=tensorrt",
            "--gpu-device-ids=0,1",
            "--gpu-mem-limit-mb=2048",
            "--trt-fp16",
        ])
        .unwrap();
        let exec = cfg.execution_config();
        assert_eq!(exec.default_device, DevicePreference::TensorRt);
        assert_eq!(exec.gpu_device_ids, vec![0, 1]);
        assert_eq!(exec.gpu_mem_limit, 2048 * 1024 * 1024);
        assert!(exec.trt_fp16);
    }

    #[test]
    fn rejects_unknown_control_mode() {
        assert!(
            ServerConfig::try_parse_from(["axon-server", "--model-control-mode=yolo"]).is_err()
        );
    }

    #[test]
    fn defaults_are_cpu() {
        let cfg = ServerConfig::try_parse_from(["axon-server"]).unwrap();
        assert_eq!(cfg.device, DevicePreference::Cpu);
        assert!(cfg.api_key().is_none());
        assert!(!cfg.ui);
    }

    #[test]
    fn ui_flag() {
        let cfg = ServerConfig::try_parse_from(["axon-server", "--ui"]).unwrap();
        assert!(cfg.ui);
    }
}
