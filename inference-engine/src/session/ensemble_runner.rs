use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use tokio::sync::Semaphore;

use crate::model_repository::config_parser::{KeyValue, ModelConfig};
use crate::session::pool::SessionPool;
use crate::session::types::{InferenceOutput, InputTensor, TensorData};

struct Step {
    model_name: String,
    /// Pinned version, or `None` for the latest loaded version.
    model_version: Option<u32>,
    /// (step input name, ensemble tensor name, last use → may be moved)
    inputs: Vec<(String, String, bool)>,
    output_map: Vec<KeyValue>,
}

pub struct EnsembleRunner {
    pool: SessionPool,
    steps: Vec<Step>,
    outputs: Vec<String>,
    semaphore: Arc<Semaphore>,
}

impl EnsembleRunner {
    pub fn load(
        config: &ModelConfig,
        pool: SessionPool,
        concurrency: usize,
    ) -> anyhow::Result<Self> {
        let scheduling = config
            .ensemble_scheduling
            .as_ref()
            .context("ensemble config missing ensemble_scheduling")?;

        if scheduling.steps.is_empty() {
            anyhow::bail!("ensemble must have at least one step");
        }
        if let Some(step) = scheduling
            .steps
            .iter()
            .find(|s| s.model_name == config.name)
        {
            anyhow::bail!("ensemble '{}' references itself as a step", step.model_name);
        }

        let output_names: Vec<String> = config.outputs.iter().map(|o| o.name.clone()).collect();

        // A tensor can be moved (instead of cloned) into the last step that
        // reads it, unless it is also a final ensemble output.
        let mut last_use: HashMap<&str, (usize, usize)> = HashMap::new();
        for (si, step) in scheduling.steps.iter().enumerate() {
            for (ii, kv) in step.input_map.iter().enumerate() {
                last_use.insert(kv.value.as_str(), (si, ii));
            }
        }

        let steps = scheduling
            .steps
            .iter()
            .enumerate()
            .map(|(si, s)| Step {
                model_name: s.model_name.clone(),
                model_version: u32::try_from(s.model_version).ok().filter(|v| *v > 0),
                inputs: s
                    .input_map
                    .iter()
                    .enumerate()
                    .map(|(ii, kv)| {
                        let movable = last_use.get(kv.value.as_str()) == Some(&(si, ii))
                            && !output_names.contains(&kv.value);
                        (kv.key.clone(), kv.value.clone(), movable)
                    })
                    .collect(),
                output_map: s.output_map.clone(),
            })
            .collect();

        Ok(Self {
            pool,
            steps,
            outputs: output_names,
            semaphore: Arc::new(Semaphore::new(concurrency.max(1))),
        })
    }

    pub fn run(&self, inputs: Vec<(String, InputTensor)>) -> anyhow::Result<InferenceOutput> {
        let mut tensor_map: HashMap<String, InputTensor> = inputs.into_iter().collect();

        for step in &self.steps {
            let mut step_inputs = Vec::with_capacity(step.inputs.len());
            for (key, value, movable) in &step.inputs {
                let tensor = if *movable {
                    tensor_map.remove(value)
                } else {
                    tensor_map.get(value).cloned()
                }
                .with_context(|| {
                    format!(
                        "missing tensor '{value}' required by step '{}' input '{key}'",
                        step.model_name
                    )
                })?;
                step_inputs.push((key.clone(), tensor));
            }

            let session = match step.model_version {
                Some(v) => self.pool.get(&step.model_name, v),
                None => self.pool.get_latest(&step.model_name),
            }
            .with_context(|| format!("model '{}' not found in pool", step.model_name))?;

            let outputs = session.runner.run(step_inputs)?;

            for (output_name, shape, data) in outputs {
                if let Some(kv) = step.output_map.iter().find(|kv| kv.key == output_name) {
                    let shape_usize: Vec<usize> =
                        shape.iter().map(|s| (*s).max(0) as usize).collect();
                    let tensor = match data {
                        TensorData::F32(d) => InputTensor::F32(d, shape_usize),
                        TensorData::I32(d) => InputTensor::I32(d, shape_usize),
                        TensorData::I64(d) => InputTensor::I64(d, shape_usize),
                        TensorData::String(d) => InputTensor::String(d, shape_usize),
                    };
                    tensor_map.insert(kv.value.clone(), tensor);
                }
            }
        }

        let mut result = Vec::with_capacity(self.outputs.len());
        for output_name in &self.outputs {
            let tensor = tensor_map
                .remove(output_name)
                .with_context(|| format!("missing final output tensor '{output_name}'"))?;

            let to_i64 = |s: Vec<usize>| s.into_iter().map(|n| n as i64).collect::<Vec<i64>>();
            let (shape, data) = match tensor {
                InputTensor::F32(d, s) => (to_i64(s), TensorData::F32(d)),
                InputTensor::I32(d, s) => (to_i64(s), TensorData::I32(d)),
                InputTensor::I64(d, s) => (to_i64(s), TensorData::I64(d)),
                InputTensor::String(d, s) => (to_i64(s), TensorData::String(d)),
            };
            result.push((output_name.clone(), shape, data));
        }

        Ok(result)
    }

    pub fn concurrency_semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }
}
