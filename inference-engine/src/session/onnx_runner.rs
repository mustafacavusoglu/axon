use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, Value};
use parking_lot::Mutex;
use tokio::sync::Semaphore;

use super::device::{Device, ExecutionConfig, Placement};
use super::types::{InferenceOutput, InputTensor, TensorData};

pub struct OnnxRunner {
    sessions: Vec<Mutex<Session>>,
    devices: Vec<Device>,
    next: AtomicUsize,
    semaphore: Arc<Semaphore>,
}

impl OnnxRunner {
    pub fn load(
        model_path: &Path,
        concurrency: usize,
        exec: &ExecutionConfig,
        placement: &Placement,
    ) -> anyhow::Result<Self> {
        let count = concurrency.max(1);
        tracing::info!(
            path = %model_path.display(),
            instances = count,
            device = ?placement.preference,
            "loading ONNX model"
        );

        let mut sessions = Vec::with_capacity(count);
        let mut devices = Vec::with_capacity(count);
        // Once a GPU instance fails under `--device=auto`, place the remaining
        // instances on CPU too instead of paying the failure timeout again.
        let mut cpu_fallback = false;

        for i in 0..count {
            let wanted = if cpu_fallback {
                Device::Cpu
            } else {
                placement.device_for_instance(i)
            };
            let session = match Self::create_session_with_timeout(model_path, wanted, exec) {
                Ok(s) => (s, wanted),
                Err(e) if wanted.is_gpu() && placement.allows_cpu_fallback() => {
                    tracing::warn!(
                        target: "axon::console",
                        path = %model_path.display(),
                        device = %wanted,
                        error = %e,
                        "GPU session unavailable, falling back to CPU"
                    );
                    cpu_fallback = true;
                    (
                        Self::create_session_with_timeout(model_path, Device::Cpu, exec)?,
                        Device::Cpu,
                    )
                }
                Err(e) => return Err(e),
            };
            sessions.push(Mutex::new(session.0));
            devices.push(session.1);
        }

        tracing::info!(
            path = %model_path.display(),
            instances = count,
            devices = ?devices.iter().map(|d| d.to_string()).collect::<Vec<_>>(),
            "ONNX sessions created"
        );

        Ok(Self {
            sessions,
            devices,
            next: AtomicUsize::new(0),
            semaphore: Arc::new(Semaphore::new(count)),
        })
    }

    /// Human readable placement, e.g. `cuda:0` or `cuda:0,cuda:1`.
    pub fn device_label(&self) -> String {
        let mut labels: Vec<String> = self.devices.iter().map(|d| d.to_string()).collect();
        labels.dedup();
        labels.join(",")
    }

    fn create_session_with_timeout(
        model_path: &Path,
        device: Device,
        exec: &ExecutionConfig,
    ) -> anyhow::Result<Session> {
        use std::sync::mpsc;

        let timeout = exec.load_timeout_for(device);
        let path = model_path.to_path_buf();
        let thread_exec = exec.clone();
        let (tx, rx) = mpsc::channel();

        std::thread::Builder::new()
            .name("axon-onnx-load".to_string())
            .spawn(move || {
                let result = Self::create_session(&path, device, &thread_exec);
                let _ = tx.send(result);
            })?;

        match rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // The loader thread cannot be cancelled; it finishes (or hangs)
                // in the background and its result is dropped.
                anyhow::bail!(
                    "ONNX session load on {device} timed out after {}s: {}",
                    timeout.as_secs(),
                    model_path.display()
                )
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!(
                    "ONNX session load thread panicked: {}",
                    model_path.display()
                )
            }
        }
    }

    fn create_session(
        model_path: &Path,
        device: Device,
        exec: &ExecutionConfig,
    ) -> anyhow::Result<Session> {
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create session builder: {e}"))?;

        let providers = exec.execution_providers(device);
        if !providers.is_empty() {
            builder = builder
                .with_execution_providers(providers)
                .map_err(|e| anyhow::anyhow!("failed to register {device} provider: {e}"))?;
        }

        // External data (bert.onnx.data etc.) is resolved automatically by ONNX
        // Runtime relative to the directory of model_path — no extra config needed.
        // Level1 only: higher levels have been observed to hang on AMD64.
        builder
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("failed to set optimization level: {e}"))?
            .with_intra_threads(exec.intra_op_threads.max(1))
            .map_err(|e| anyhow::anyhow!("failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to load ONNX model {} on {device}: {e}",
                    model_path.display()
                )
            })
    }

    pub fn concurrency_semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }

    pub fn run(&self, inputs: Vec<(String, InputTensor)>) -> anyhow::Result<InferenceOutput> {
        let mut session_inputs: Vec<(String, Value)> = Vec::with_capacity(inputs.len());

        for (name, tensor) in inputs {
            let value: Value = match tensor {
                InputTensor::F32(data, shape) => Tensor::from_array((shape, data))
                    .map_err(|e| anyhow::anyhow!("fp32 input '{name}': {e}"))?
                    .into(),
                InputTensor::I32(data, shape) => Tensor::from_array((shape, data))
                    .map_err(|e| anyhow::anyhow!("int32 input '{name}': {e}"))?
                    .into(),
                InputTensor::I64(data, shape) => Tensor::from_array((shape, data))
                    .map_err(|e| anyhow::anyhow!("int64 input '{name}': {e}"))?
                    .into(),
                InputTensor::String(data, shape) => {
                    let array =
                        ndarray::ArrayD::<String>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    Tensor::from_string_array(&array)
                        .map_err(|e| anyhow::anyhow!("string input '{name}': {e}"))?
                        .into()
                }
            };
            session_inputs.push((name, value));
        }

        // Start probing at a rotating offset so load spreads evenly across
        // instances (and therefore across GPUs).
        let n = self.sessions.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed) % n;
        let mut session = (0..n)
            .find_map(|i| self.sessions[(start + i) % n].try_lock())
            .unwrap_or_else(|| self.sessions[start].lock());

        let outputs = session
            .run(session_inputs)
            .map_err(|e| anyhow::anyhow!("onnxruntime: {e}"))?;

        let mut results = Vec::with_capacity(outputs.len());
        for (name, value) in outputs.iter() {
            let (shape, data) = extract_output(name, &value)?;
            results.push((name.to_string(), shape, data));
        }

        Ok(results)
    }
}

fn extract_output(
    name: &str,
    value: &ort::value::ValueRef<'_>,
) -> anyhow::Result<(Vec<i64>, TensorData)> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Ok((shape.to_vec(), TensorData::F32(data.to_vec())));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
        return Ok((shape.to_vec(), TensorData::I64(data.to_vec())));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
        return Ok((shape.to_vec(), TensorData::I32(data.to_vec())));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<f64>() {
        // FP64 is not part of the public datatype set; narrow to FP32.
        return Ok((
            shape.to_vec(),
            TensorData::F32(data.iter().map(|&v| v as f32).collect()),
        ));
    }
    if let Ok((shape, data)) = value.try_extract_strings() {
        return Ok((shape.to_vec(), TensorData::String(data)));
    }

    if let Ok(maps) = value.try_extract_sequence::<ort::value::DynValueTypeMarker>() {
        if !maps.is_empty() {
            return extract_tree_sequence(name, &maps);
        }
    }

    Err(anyhow::anyhow!(
        "unsupported output tensor type for '{name}'"
    ))
}

/// Upper bound on the class dimension reconstructed from ZipMap outputs, so a
/// model emitting a huge class id cannot trigger a giant allocation.
const MAX_ZIPMAP_CLASSES: i64 = 1 << 20;

fn extract_tree_sequence(
    name: &str,
    maps: &[ort::value::ValueRef<'_, ort::value::DynValueTypeMarker>],
) -> anyhow::Result<(Vec<i64>, TensorData)> {
    if maps.is_empty() {
        return Err(anyhow::anyhow!("empty sequence output for '{name}'"));
    }

    let rows: Vec<HashMap<i64, f32>> = maps
        .iter()
        .map(|m| {
            m.try_extract_map::<i64, f32>()
                .map_err(|e| anyhow::anyhow!("failed to extract map element from '{name}': {e}"))
        })
        .collect::<anyhow::Result<_>>()?;

    let max_key = rows
        .iter()
        .flat_map(|r| r.keys().copied())
        .max()
        .unwrap_or(0);
    if max_key >= MAX_ZIPMAP_CLASSES {
        anyhow::bail!("class id {max_key} in '{name}' exceeds {MAX_ZIPMAP_CLASSES}");
    }
    let class_dim = (max_key + 1).max(1) as usize;

    let mut flat_probs = vec![0.0f32; rows.len() * class_dim];
    for (row_idx, row) in rows.into_iter().enumerate() {
        for (k, v) in row {
            if k >= 0 {
                flat_probs[row_idx * class_dim + k as usize] = v;
            }
        }
    }

    let shape = vec![maps.len() as i64, class_dim as i64];
    Ok((shape, TensorData::F32(flat_probs)))
}
