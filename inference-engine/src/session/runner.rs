use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, Value};
use parking_lot::Mutex;
use tokio::sync::Semaphore;

pub enum InputTensor {
    F32(Vec<f32>, Vec<usize>),
    I32(Vec<i32>, Vec<usize>),
    I64(Vec<i64>, Vec<usize>),
    String(Vec<String>, Vec<usize>),
}

pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl TensorData {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::F32(data) => {
                let mut buf = Vec::with_capacity(data.len() * 4);
                for f in data {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                buf
            }
            TensorData::I32(data) => {
                let mut buf = Vec::with_capacity(data.len() * 4);
                for i in data {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
                buf
            }
            TensorData::I64(data) => {
                let mut buf = Vec::with_capacity(data.len() * 8);
                for i in data {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
                buf
            }
        }
    }

    pub fn dtype_str(&self) -> &'static str {
        match self {
            TensorData::F32(_) => "FP32",
            TensorData::I32(_) => "INT32",
            TensorData::I64(_) => "INT64",
        }
    }
}

pub struct ModelRunner {
    sessions: Vec<Mutex<Session>>,
    semaphore: Arc<Semaphore>,
    model_path: PathBuf,
}

impl ModelRunner {
    pub fn load(model_path: &Path, concurrency: usize) -> anyhow::Result<Self> {
        let count = concurrency.max(1);
        let mut sessions = Vec::with_capacity(count);

        for _ in 0..count {
            let session = Self::create_session(model_path)?;
            sessions.push(Mutex::new(session));
        }

        tracing::info!(path = %model_path.display(), instances = count, "ONNX sessions created");

        Ok(Self {
            sessions,
            semaphore: Arc::new(Semaphore::new(count)),
            model_path: model_path.to_path_buf(),
        })
    }

    fn create_session(model_path: &Path) -> anyhow::Result<Session> {
        let builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create session builder: {}", e))?;

        builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("failed to set optimization level: {}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("failed to set intra threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!("failed to load ONNX model {}: {}", model_path.display(), e)
            })
    }

    pub fn concurrency_semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }

    pub fn run(
        &self,
        inputs: Vec<(String, InputTensor)>,
    ) -> anyhow::Result<Vec<(String, Vec<i64>, TensorData)>> {
        let mut session_inputs: HashMap<String, Value> = HashMap::new();

        for (name, tensor) in inputs {
            let value = match tensor {
                InputTensor::F32(data, shape) => {
                    let array =
                        ndarray::ArrayD::<f32>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    Value::from_array(array)
                        .map_err(|e| anyhow::anyhow!("fp32 input '{}': {}", name, e))?
                        .into()
                }
                InputTensor::I32(data, shape) => {
                    let array =
                        ndarray::ArrayD::<i32>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    Value::from_array(array)
                        .map_err(|e| anyhow::anyhow!("int32 input '{}': {}", name, e))?
                        .into()
                }
                InputTensor::I64(data, shape) => {
                    let array =
                        ndarray::ArrayD::<i64>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    Value::from_array(array)
                        .map_err(|e| anyhow::anyhow!("int64 input '{}': {}", name, e))?
                        .into()
                }
                InputTensor::String(data, shape) => {
                    let array =
                        ndarray::ArrayD::<String>::from_shape_vec(ndarray::IxDyn(&shape), data)?;
                    let string_tensor: Value = Tensor::from_string_array(&array)
                        .map_err(|e| anyhow::anyhow!("string input '{}': {}", name, e))?
                        .into();
                    string_tensor
                }
            };
            session_inputs.insert(name, value);
        }

        // Find an available session (round-robin via try_lock)
        let mut session_guard = None;
        for s in &self.sessions {
            if let Some(guard) = s.try_lock() {
                session_guard = Some(guard);
                break;
            }
        }
        // Fallback: block on first session if all are busy
        let mut session = session_guard.unwrap_or_else(|| self.sessions[0].lock());

        let outputs = session
            .run(session_inputs)
            .map_err(|e| anyhow::anyhow!("inference failed: {}", e))?;

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
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        return Ok((shape_i64, TensorData::F32(data.to_vec())));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        return Ok((shape_i64, TensorData::I64(data.to_vec())));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        return Ok((shape_i64, TensorData::I32(data.to_vec())));
    }

    if let Ok(maps) = value.try_extract_sequence::<ort::value::DynValueTypeMarker>() {
        if !maps.is_empty() {
            return extract_tree_sequence(name, &maps);
        }
    }

    Err(anyhow::anyhow!(
        "unsupported output tensor type for '{}'",
        name
    ))
}

fn extract_tree_sequence(
    name: &str,
    maps: &[ort::value::ValueRef<'_, ort::value::DynValueTypeMarker>],
) -> anyhow::Result<(Vec<i64>, TensorData)> {
    if maps.is_empty() {
        return Err(anyhow::anyhow!("empty sequence output for '{}'", name));
    }

    let first_map = &maps[0];
    let probs: HashMap<i64, f32> = first_map
        .try_extract_map::<i64, f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract map from '{}': {}", name, e))?;

    let num_classes = probs.len();
    let max_key = probs.keys().max().copied().unwrap_or(0);
    let class_dim = (max_key + 1).max(num_classes as i64) as usize;

    let batch_size = maps.len();
    let mut flat_probs = Vec::with_capacity(batch_size * class_dim);

    for map_val in maps {
        let class_map: HashMap<i64, f32> = map_val
            .try_extract_map::<i64, f32>()
            .map_err(|e| anyhow::anyhow!("failed to extract map element from '{}': {}", name, e))?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_to_bytes_f32() {
        let td = TensorData::F32(vec![1.0, 2.0]);
        let bytes = td.to_bytes();
        assert_eq!(bytes.len(), 8);
        let v1 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(v1, 1.0);
    }

    #[test]
    fn test_tensor_data_to_bytes_i64() {
        let td = TensorData::I64(vec![42]);
        let bytes = td.to_bytes();
        assert_eq!(bytes.len(), 8);
        let v = i64::from_le_bytes(bytes.try_into().unwrap());
        assert_eq!(v, 42);
    }
}
