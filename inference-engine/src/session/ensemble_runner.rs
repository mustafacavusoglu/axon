use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use tokio::sync::Semaphore;

use crate::model_repository::config_parser::{KeyValue, ModelConfig};
use crate::session::pool::SessionPool;
use crate::session::types::{InferenceOutput, InputTensor, TensorData};

pub struct EnsembleRunner {
    pool: SessionPool,
    steps: Vec<(String, Vec<KeyValue>, Vec<KeyValue>)>,
    #[allow(dead_code)]
    inputs: Vec<String>,
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

        let steps: Vec<_> = scheduling
            .steps
            .iter()
            .map(|s| {
                (
                    s.model_name.clone(),
                    s.input_map.clone(),
                    s.output_map.clone(),
                )
            })
            .collect();

        let input_names: Vec<String> = config.inputs.iter().map(|i| i.name.clone()).collect();
        let output_names: Vec<String> = config.outputs.iter().map(|o| o.name.clone()).collect();

        Ok(Self {
            pool,
            steps,
            inputs: input_names,
            outputs: output_names,
            semaphore: Arc::new(Semaphore::new(concurrency.max(1))),
        })
    }

    pub fn run(&self, inputs: Vec<(String, InputTensor)>) -> anyhow::Result<InferenceOutput> {
        let mut tensor_map: HashMap<String, InputTensor> = inputs.into_iter().collect();

        for (model_name, input_map, output_map) in &self.steps {
            let mut step_inputs = Vec::new();
            for kv in input_map {
                let tensor = tensor_map.get(&kv.value).with_context(|| {
                    format!(
                        "missing tensor '{}' required by step '{}' input '{}'",
                        kv.value, model_name, kv.key
                    )
                })?;
                step_inputs.push((kv.key.clone(), tensor.clone()));
            }

            let session = self
                .pool
                .get_latest(model_name)
                .with_context(|| format!("model '{model_name}' not found in pool"))?;

            let outputs = session.runner.run(step_inputs)?;

            for (output_name, shape, data) in outputs {
                if let Some(kv) = output_map.iter().find(|kv| kv.key == output_name) {
                    let shape_usize: Vec<usize> = shape.iter().map(|s| *s as usize).collect();
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

        let mut result = Vec::new();
        for output_name in &self.outputs {
            let tensor = tensor_map
                .get(output_name)
                .with_context(|| format!("missing final output tensor '{output_name}'"))?;

            let (shape, data) = match tensor {
                InputTensor::F32(d, s) => (
                    s.iter().map(|n| *n as i64).collect(),
                    TensorData::F32(d.clone()),
                ),
                InputTensor::I32(d, s) => (
                    s.iter().map(|n| *n as i64).collect(),
                    TensorData::I32(d.clone()),
                ),
                InputTensor::I64(d, s) => (
                    s.iter().map(|n| *n as i64).collect(),
                    TensorData::I64(d.clone()),
                ),
                InputTensor::String(d, s) => (
                    s.iter().map(|n| *n as i64).collect(),
                    TensorData::String(d.clone()),
                ),
            };
            result.push((output_name.clone(), shape, data));
        }

        Ok(result)
    }

    pub fn concurrency_semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }
}
