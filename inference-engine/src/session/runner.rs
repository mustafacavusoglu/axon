use std::path::Path;
use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::model_repository::config_parser::ModelConfig;
use crate::session::pool::SessionPool;

use super::ensemble_runner::EnsembleRunner;
use super::onnx_runner::OnnxRunner;
use super::rhai_runner::RhaiRunner;
use super::types::{InferenceOutput, InputTensor};

pub enum ModelRunner {
    Onnx(Arc<OnnxRunner>),
    Rhai(Arc<RhaiRunner>),
    Ensemble(Arc<EnsembleRunner>),
}

impl ModelRunner {
    pub fn load_onnx(model_path: &Path, concurrency: usize) -> anyhow::Result<Self> {
        let runner = OnnxRunner::load(model_path, concurrency)?;
        Ok(ModelRunner::Onnx(Arc::new(runner)))
    }

    pub fn load_rhai(script_path: &Path, pool: SessionPool, concurrency: usize) -> anyhow::Result<Self> {
        let runner = RhaiRunner::load(script_path, pool, concurrency)?;
        Ok(ModelRunner::Rhai(Arc::new(runner)))
    }

    pub fn load_ensemble(config: &ModelConfig, pool: SessionPool, concurrency: usize) -> anyhow::Result<Self> {
        let runner = EnsembleRunner::load(config, pool, concurrency)?;
        Ok(ModelRunner::Ensemble(Arc::new(runner)))
    }

    pub fn concurrency_semaphore(&self) -> &Arc<Semaphore> {
        match self {
            ModelRunner::Onnx(r) => r.concurrency_semaphore(),
            ModelRunner::Rhai(r) => r.concurrency_semaphore(),
            ModelRunner::Ensemble(r) => r.concurrency_semaphore(),
        }
    }

    pub fn run(&self, inputs: Vec<(String, InputTensor)>) -> anyhow::Result<InferenceOutput> {
        match self {
            ModelRunner::Onnx(r) => r.run(inputs),
            ModelRunner::Rhai(r) => r.run(inputs),
            ModelRunner::Ensemble(r) => r.run(inputs),
        }
    }
}
