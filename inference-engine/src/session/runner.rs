use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ndarray::ArrayD;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

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

        let input_names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        let output_names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();

        tracing::info!(
            path = %model_path.display(),
            ?input_names,
            ?output_names,
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
    ) -> anyhow::Result<Vec<(String, ArrayD<f32>)>> {
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
            let (shape, data) = value.try_extract_tensor::<f32>().map_err(|e| {
                anyhow::anyhow!("failed to extract tensor for output '{}': {}", name, e)
            })?;
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let array = ArrayD::from_shape_vec(
                ndarray::IxDyn(&shape_usize),
                data.to_vec(),
            )?;
            results.push((name.to_string(), array));
        }

        Ok(results)
    }
}
