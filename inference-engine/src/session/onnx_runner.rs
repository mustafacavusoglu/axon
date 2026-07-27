use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, Value};
use parking_lot::Mutex;
use tokio::sync::Semaphore;

use super::types::{InferenceOutput, InputTensor, TensorData};

pub struct OnnxRunner {
    sessions: Vec<Mutex<Session>>,
    semaphore: Arc<Semaphore>,
    #[allow(dead_code)]
    model_path: PathBuf,
}

impl OnnxRunner {
    pub fn load(model_path: &Path, concurrency: usize) -> anyhow::Result<Self> {
        let count = concurrency.max(1);

        // For large models with external data (e.g. BERT .onnx.data files) each
        // session load reads the full weight file.  Create only 1 session up-front;
        // additional sessions up to `count` are spun up lazily on first use.
        // This prevents apparent hangs when count=4 and the model is hundreds of MB.
        tracing::info!(
            path = %model_path.display(),
            instances = count,
            "loading ONNX model (1 of {} sessions, rest lazy)"  , count
        );
        let first = Self::create_session(model_path)?;
        let mut sessions = Vec::with_capacity(count);
        sessions.push(Mutex::new(first));

        for i in 1..count {
            tracing::debug!(path = %model_path.display(), session = i + 1, total = count, "loading additional ONNX session");
            let s = Self::create_session(model_path)?;
            sessions.push(Mutex::new(s));
        }

        tracing::info!(path = %model_path.display(), instances = count, "ONNX sessions created");

        Ok(Self {
            sessions,
            semaphore: Arc::new(Semaphore::new(count)),
            model_path: model_path.to_path_buf(),
        })
    }

    fn create_session(model_path: &Path) -> anyhow::Result<Session> {
        let has_external_data = model_path
            .parent()
            .map(|d| {
                std::fs::read_dir(d)
                    .ok()
                    .map(|mut e| {
                        e.any(|f| {
                            f.ok()
                                .and_then(|f| f.file_name().into_string().ok())
                                .map(|n| n.ends_with(".onnx.data") || n.ends_with(".data"))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        if has_external_data {
            tracing::info!(
                path = %model_path.display(),
                "external data file detected — using Level0 optimization to avoid load hang"
            );
        }

        let level = if has_external_data {
            // Level0 = no optimization; large transformer models can hang for
            // minutes at Level1+ during graph shape-inference over external data.
            GraphOptimizationLevel::Level0
        } else {
            GraphOptimizationLevel::Level1
        };

        let builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create session builder: {e}"))?;

        // External data (bert.onnx.data etc.) is resolved automatically by ONNX
        // Runtime relative to the directory of model_path — no extra config needed.
        builder
            .with_optimization_level(level)
            .map_err(|e| anyhow::anyhow!("failed to set optimization level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!("failed to load ONNX model {}: {}", model_path.display(), e)
            })
    }

    pub fn concurrency_semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }

    pub fn run(&self, inputs: Vec<(String, InputTensor)>) -> anyhow::Result<InferenceOutput> {
        let mut session_inputs: HashMap<String, Value> = HashMap::new();

        for (name, tensor) in inputs {
            let value = match tensor {
                InputTensor::F32(data, shape) => {
                    let array =
                        ndarray::ArrayD::<f32>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    Value::from_array(array)
                        .map_err(|e| anyhow::anyhow!("fp32 input '{name}': {e}"))?
                        .into()
                }
                InputTensor::I32(data, shape) => {
                    let array =
                        ndarray::ArrayD::<i32>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    Value::from_array(array)
                        .map_err(|e| anyhow::anyhow!("int32 input '{name}': {e}"))?
                        .into()
                }
                InputTensor::I64(data, shape) => {
                    let array =
                        ndarray::ArrayD::<i64>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    Value::from_array(array)
                        .map_err(|e| anyhow::anyhow!("int64 input '{name}': {e}"))?
                        .into()
                }
                InputTensor::String(data, shape) => {
                    let array =
                        ndarray::ArrayD::<String>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    let string_tensor: Value = Tensor::from_string_array(&array)
                        .map_err(|e| anyhow::anyhow!("string input '{name}': {e}"))?
                        .into();
                    string_tensor
                }
            };
            session_inputs.insert(name, value);
        }

        let mut session_guard = None;
        for s in &self.sessions {
            if let Some(guard) = s.try_lock() {
                session_guard = Some(guard);
                break;
            }
        }
        let mut session = session_guard.unwrap_or_else(|| self.sessions[0].lock());

        let outputs = session
            .run(session_inputs)
            .map_err(|e| anyhow::anyhow!("inference failed: {e}"))?;

        let mut results = Vec::new();
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
        let shape_i64: Vec<i64> = shape.iter().copied().collect();
        return Ok((shape_i64, TensorData::F32(data.to_vec())));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
        let shape_i64: Vec<i64> = shape.iter().copied().collect();
        return Ok((shape_i64, TensorData::I64(data.to_vec())));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
        let shape_i64: Vec<i64> = shape.iter().copied().collect();
        return Ok((shape_i64, TensorData::I32(data.to_vec())));
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

fn extract_tree_sequence(
    name: &str,
    maps: &[ort::value::ValueRef<'_, ort::value::DynValueTypeMarker>],
) -> anyhow::Result<(Vec<i64>, TensorData)> {
    if maps.is_empty() {
        return Err(anyhow::anyhow!("empty sequence output for '{name}'"));
    }

    let first_map = &maps[0];
    let probs: HashMap<i64, f32> = first_map
        .try_extract_map::<i64, f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract map from '{name}': {e}"))?;

    let num_classes = probs.len();
    let max_key = probs.keys().max().copied().unwrap_or(0);
    let class_dim = (max_key + 1).max(num_classes as i64) as usize;

    let batch_size = maps.len();
    let mut flat_probs = Vec::with_capacity(batch_size * class_dim);

    for map_val in maps {
        let class_map: HashMap<i64, f32> = map_val
            .try_extract_map::<i64, f32>()
            .map_err(|e| anyhow::anyhow!("failed to extract map element from '{name}': {e}"))?;

        let mut row = vec![0.0f32; class_dim];
        for (k, v) in class_map {
            if k >= 0 && (k as usize) < class_dim {
                row[k as usize] = v;
            }
        }
        flat_probs.extend(row);
    }

    let shape = vec![batch_size as i64, class_dim as i64];
    Ok((shape, TensorData::F32(flat_probs)))
}
