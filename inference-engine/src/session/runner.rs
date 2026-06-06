use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ndarray::ArrayD;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl TensorData {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::F32(data) => data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect(),
            TensorData::I64(data) => data.iter().flat_map(|i| i.to_le_bytes().to_vec()).collect(),
        }
    }
}

pub struct ModelRunner {
    session: Mutex<Session>,
}

impl ModelRunner {
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create session builder: {}", e))?;

        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("failed to set optimization level: {}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("failed to set intra threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!("failed to load ONNX model {}: {}", model_path.display(), e)
            })?;

        tracing::info!(
            path = %model_path.display(),
            "ONNX session created"
        );

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    pub fn estimate_memory(&self) -> u64 {
        0
    }

    pub fn run(
        &self,
        inputs: Vec<(String, ArrayD<f32>)>,
    ) -> anyhow::Result<Vec<(String, Vec<i64>, TensorData)>> {
        let mut session_inputs: HashMap<String, Value> = HashMap::new();

        for (name, array) in &inputs {
            let shape: Vec<usize> = array.shape().to_vec();
            let data: Vec<f32> = array.iter().copied().collect();
            let tensor = ndarray::ArrayD::<f32>::from_shape_vec(
                ndarray::IxDyn(&shape),
                data,
            )?;
            let value = Value::from_array(tensor)
                .map_err(|e| anyhow::anyhow!("failed to create ort value for '{}': {}", name, e))?;
            session_inputs.insert(name.clone(), value.into());
        }

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(session_inputs)
            .map_err(|e| anyhow::anyhow!("inference failed: {}", e))?;

        let mut results = Vec::new();
        for (name, value) in outputs.iter() {
            match extract_output(name, &value) {
                Ok((shape, data)) => results.push((name.to_string(), shape, data)),
                Err(e) => return Err(e),
            }
        }

        Ok(results)
    }
}

fn extract_output(name: &str, value: &ort::value::ValueRef<'_>) -> anyhow::Result<(Vec<i64>, TensorData)> {
    let f32_res = value.try_extract_tensor::<f32>();
    let i64_res = value.try_extract_tensor::<i64>();

    if let Ok((shape, data)) = f32_res {
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        return Ok((shape_i64, TensorData::F32(data.to_vec())));
    }
    if let Ok((shape, data)) = i64_res {
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        return Ok((shape_i64, TensorData::I64(data.to_vec())));
    }

    let err_str = format!("{}", f32_res.err().unwrap());
    if err_str.contains("Sequence") {
        return extract_tree_sequence(name, value);
    }

    Err(anyhow::anyhow!(
        "failed to extract tensor for '{}': f32={}, i64={}",
        name,
        err_str,
        i64_res.err().map(|e| format!("{}", e)).unwrap_or_default()
    ))
}

fn extract_tree_sequence(name: &str, value: &ort::value::ValueRef<'_>) -> anyhow::Result<(Vec<i64>, TensorData)> {
    let maps = value.try_extract_sequence::<ort::value::DynValueTypeMarker>()
        .map_err(|e| anyhow::anyhow!("failed to extract sequence from '{}': {}", name, e))?;

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

    for map_val in &maps {
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
    fn test_load_onnx_model() {
        let path = std::path::Path::new(
            "/Users/mustafacavusoglu/workspace/axon/local_models/model_repository/lgbm_breast_cancer/1/model.onnx"
        );
        if !path.exists() {
            eprintln!("SKIP: model file not found");
            return;
        }
        let runner = ModelRunner::load(path);
        assert!(runner.is_ok(), "Failed to load model: {:?}", runner.err());
    }
}
