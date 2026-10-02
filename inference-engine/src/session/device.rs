//! Execution device selection (CPU / CUDA / TensorRT).
//!
//! The server-wide default comes from `--device`; a model can override it via
//! its `instance_group.kind` (Triton semantics):
//!
//! | kind        | effect                                                    |
//! |-------------|-----------------------------------------------------------|
//! | `KIND_CPU`  | always CPU                                                |
//! | `KIND_GPU`  | GPU required (TensorRT if `--device=tensorrt`, else CUDA) |
//! | `KIND_AUTO` | server default (`--device`)                               |
//!
//! `--device=auto` tries CUDA and silently falls back to CPU when no GPU (or
//! no GPU build of ONNX Runtime) is available. Explicit `cuda`/`tensorrt` (and
//! `KIND_GPU`) fail the model load instead of silently running on CPU.

use std::path::PathBuf;
use std::time::Duration;

use ort::ep::{self, ArenaExtendStrategy, ExecutionProviderDispatch};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Cpu,
    Auto,
    Cuda,
    TensorRt,
}

impl std::str::FromStr for DevicePreference {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            "cuda" | "gpu" => Ok(Self::Cuda),
            "tensorrt" | "trt" => Ok(Self::TensorRt),
            other => Err(format!(
                "invalid device '{other}' (expected cpu, auto, cuda, tensorrt)"
            )),
        }
    }
}

/// Concrete device a session is created on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(i32),
    TensorRt(i32),
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Device::Cpu)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{id}"),
            Device::TensorRt(id) => write!(f, "trt:{id}"),
        }
    }
}

/// Placement for one model: which EP family and which GPUs its instances are
/// spread over (round-robin).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub preference: DevicePreference,
    pub gpu_ids: Vec<i32>,
}

impl Placement {
    /// Device for the `index`-th session instance of the model.
    pub fn device_for_instance(&self, index: usize) -> Device {
        let id = if self.gpu_ids.is_empty() {
            0
        } else {
            self.gpu_ids[index % self.gpu_ids.len()]
        };
        match self.preference {
            DevicePreference::Cpu => Device::Cpu,
            DevicePreference::Auto | DevicePreference::Cuda => Device::Cuda(id),
            DevicePreference::TensorRt => Device::TensorRt(id),
        }
    }

    /// Whether a GPU failure may silently fall back to CPU.
    pub fn allows_cpu_fallback(&self) -> bool {
        self.preference == DevicePreference::Auto
    }
}

/// Server-wide ONNX Runtime execution settings.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    pub default_device: DevicePreference,
    pub gpu_device_ids: Vec<i32>,
    /// Per-session GPU memory arena limit in bytes (0 = unlimited).
    pub gpu_mem_limit: usize,
    pub trt_fp16: bool,
    pub trt_cache_dir: Option<PathBuf>,
    pub intra_op_threads: usize,
    /// Session creation timeout; `None` = derive from the device.
    pub load_timeout: Option<Duration>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            default_device: DevicePreference::Cpu,
            gpu_device_ids: vec![0],
            gpu_mem_limit: 0,
            trt_fp16: false,
            trt_cache_dir: None,
            intra_op_threads: 1,
            load_timeout: None,
        }
    }
}

impl ExecutionConfig {
    /// Resolves the placement for a model from its `instance_group` kind and
    /// optional `gpus` list.
    pub fn placement(&self, kind: Option<&str>, gpus: &[i32]) -> Placement {
        let kind = kind.unwrap_or("KIND_AUTO").trim().to_ascii_uppercase();
        let preference = match kind.as_str() {
            "KIND_CPU" => DevicePreference::Cpu,
            "KIND_GPU" => match self.default_device {
                DevicePreference::TensorRt => DevicePreference::TensorRt,
                _ => DevicePreference::Cuda,
            },
            _ => self.default_device,
        };
        let gpu_ids = if gpus.is_empty() {
            self.gpu_device_ids.clone()
        } else {
            gpus.to_vec()
        };
        Placement {
            preference,
            gpu_ids,
        }
    }

    pub fn load_timeout_for(&self, device: Device) -> Duration {
        self.load_timeout.unwrap_or(match device {
            Device::Cpu => Duration::from_secs(10),
            // cuDNN/cuBLAS initialisation and TensorRT engine builds are slow.
            Device::Cuda(_) => Duration::from_secs(120),
            Device::TensorRt(_) => Duration::from_secs(900),
        })
    }

    /// Execution providers to register for `device` (CPU is always the
    /// implicit last fallback inside ONNX Runtime).
    pub fn execution_providers(&self, device: Device) -> Vec<ExecutionProviderDispatch> {
        match device {
            Device::Cpu => vec![],
            Device::Cuda(id) => vec![self.cuda_ep(id)],
            Device::TensorRt(id) => {
                let mut trt = ep::TensorRT::default()
                    .with_device_id(id)
                    .with_fp16(self.trt_fp16);
                if let Some(dir) = &self.trt_cache_dir {
                    let dir = dir.display().to_string();
                    trt = trt
                        .with_engine_cache(true)
                        .with_engine_cache_path(&dir)
                        .with_timing_cache(true)
                        .with_timing_cache_path(&dir);
                }
                // Nodes TensorRT cannot handle go to CUDA, then CPU.
                vec![trt.build().error_on_failure(), self.cuda_ep(id)]
            }
        }
    }

    fn cuda_ep(&self, id: i32) -> ExecutionProviderDispatch {
        let mut cuda = ep::CUDA::default()
            .with_device_id(id)
            // Several sessions share one GPU; grow the arena only as needed
            // instead of doubling, which otherwise exhausts VRAM quickly.
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            // Exhaustive cuDNN search re-benchmarks every new input shape,
            // which is pathological for dynamic-shape NLP models.
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic);
        if self.gpu_mem_limit > 0 {
            cuda = cuda.with_memory_limit(self.gpu_mem_limit);
        }
        cuda.build().error_on_failure()
    }
}

/// Parses a comma separated GPU id list (`"0,1"`).
pub fn parse_gpu_ids(s: &str) -> Result<Vec<i32>, String> {
    let ids: Vec<i32> = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            p.parse::<i32>()
                .ok()
                .filter(|v| *v >= 0)
                .ok_or_else(|| format!("invalid GPU id '{p}'"))
        })
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err("GPU id list is empty".to_string());
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(dev: DevicePreference) -> ExecutionConfig {
        ExecutionConfig {
            default_device: dev,
            gpu_device_ids: vec![0, 1],
            ..Default::default()
        }
    }

    #[test]
    fn kind_cpu_always_cpu() {
        let p = cfg(DevicePreference::Cuda).placement(Some("KIND_CPU"), &[]);
        assert_eq!(p.device_for_instance(0), Device::Cpu);
    }

    #[test]
    fn kind_gpu_uses_cuda_or_trt() {
        let p = cfg(DevicePreference::Cpu).placement(Some("KIND_GPU"), &[]);
        assert_eq!(p.preference, DevicePreference::Cuda);
        assert!(!p.allows_cpu_fallback());
        let p = cfg(DevicePreference::TensorRt).placement(Some("kind_gpu"), &[]);
        assert_eq!(p.device_for_instance(0), Device::TensorRt(0));
    }

    #[test]
    fn auto_follows_server_default() {
        let p = cfg(DevicePreference::Auto).placement(None, &[]);
        assert!(p.allows_cpu_fallback());
        assert_eq!(p.device_for_instance(1), Device::Cuda(1));
        let p = cfg(DevicePreference::Cpu).placement(Some("KIND_AUTO"), &[]);
        assert_eq!(p.device_for_instance(0), Device::Cpu);
    }

    #[test]
    fn instances_round_robin_over_gpus() {
        let p = cfg(DevicePreference::Cuda).placement(Some("KIND_GPU"), &[2, 3]);
        let ids: Vec<Device> = (0..4).map(|i| p.device_for_instance(i)).collect();
        assert_eq!(
            ids,
            vec![
                Device::Cuda(2),
                Device::Cuda(3),
                Device::Cuda(2),
                Device::Cuda(3)
            ]
        );
    }

    #[test]
    fn parse_device_and_ids() {
        assert_eq!(
            "GPU".parse::<DevicePreference>(),
            Ok(DevicePreference::Cuda)
        );
        assert_eq!(
            "trt".parse::<DevicePreference>(),
            Ok(DevicePreference::TensorRt)
        );
        assert!("tpu".parse::<DevicePreference>().is_err());
        assert_eq!(parse_gpu_ids("0, 2"), Ok(vec![0, 2]));
        assert!(parse_gpu_ids("-1").is_err());
        assert!(parse_gpu_ids("").is_err());
    }

    #[test]
    fn load_timeout_defaults() {
        let c = ExecutionConfig::default();
        assert_eq!(c.load_timeout_for(Device::Cpu), Duration::from_secs(10));
        assert!(c.load_timeout_for(Device::TensorRt(0)) > c.load_timeout_for(Device::Cuda(0)));
    }
}
