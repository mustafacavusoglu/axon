use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::model_repository::config_parser::ModelConfig;
use crate::session::pool::SessionPool;

use super::device::{ExecutionConfig, Placement};
use super::ensemble_runner::EnsembleRunner;
use super::onnx_runner::OnnxRunner;
use super::rhai_runner::RhaiRunner;
use super::types::{InferenceOutput, InputTensor};

/// Maximum nesting of model calls (ensemble steps and BLS `infer()` calls).
/// Guards against a script or ensemble that (indirectly) calls itself, which
/// would otherwise overflow the stack and abort the whole process.
const MAX_CALL_DEPTH: usize = 8;

thread_local! {
    static CALL_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct DepthGuard;

impl DepthGuard {
    fn enter() -> anyhow::Result<Self> {
        CALL_DEPTH.with(|d| {
            let depth = d.get();
            if depth >= MAX_CALL_DEPTH {
                anyhow::bail!(
                    "model call depth limit ({MAX_CALL_DEPTH}) exceeded — recursive ensemble/BLS?"
                );
            }
            d.set(depth + 1);
            Ok(DepthGuard)
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        CALL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

pub enum ModelRunner {
    Onnx(Arc<OnnxRunner>),
    Rhai(Arc<RhaiRunner>),
    Ensemble(Arc<EnsembleRunner>),
    #[cfg(test)]
    Noop(Arc<Semaphore>),
}

impl ModelRunner {
    pub fn load_onnx(
        model_path: &Path,
        concurrency: usize,
        exec: &ExecutionConfig,
        placement: &Placement,
    ) -> anyhow::Result<Self> {
        let runner = OnnxRunner::load(model_path, concurrency, exec, placement)?;
        Ok(ModelRunner::Onnx(Arc::new(runner)))
    }

    pub fn load_rhai(
        script_path: &Path,
        pool: SessionPool,
        concurrency: usize,
    ) -> anyhow::Result<Self> {
        let runner = RhaiRunner::load(script_path, pool, concurrency)?;
        Ok(ModelRunner::Rhai(Arc::new(runner)))
    }

    pub fn load_ensemble(
        config: &ModelConfig,
        pool: SessionPool,
        concurrency: usize,
    ) -> anyhow::Result<Self> {
        let runner = EnsembleRunner::load(config, pool, concurrency)?;
        Ok(ModelRunner::Ensemble(Arc::new(runner)))
    }

    pub fn platform_name(&self) -> &'static str {
        match self {
            ModelRunner::Onnx(_) => "onnxruntime",
            ModelRunner::Rhai(_) => "script",
            ModelRunner::Ensemble(_) => "ensemble",
            #[cfg(test)]
            ModelRunner::Noop(_) => "noop",
        }
    }

    /// Where the model executes (`cpu`, `cuda:0`, ...).
    pub fn device_label(&self) -> String {
        match self {
            ModelRunner::Onnx(r) => r.device_label(),
            _ => "cpu".to_string(),
        }
    }

    pub fn concurrency_semaphore(&self) -> &Arc<Semaphore> {
        match self {
            ModelRunner::Onnx(r) => r.concurrency_semaphore(),
            ModelRunner::Rhai(r) => r.concurrency_semaphore(),
            ModelRunner::Ensemble(r) => r.concurrency_semaphore(),
            #[cfg(test)]
            ModelRunner::Noop(s) => s,
        }
    }

    pub fn run(&self, inputs: Vec<(String, InputTensor)>) -> anyhow::Result<InferenceOutput> {
        let _depth = DepthGuard::enter()?;
        match self {
            ModelRunner::Onnx(r) => r.run(inputs),
            ModelRunner::Rhai(r) => r.run(inputs),
            ModelRunner::Ensemble(r) => r.run(inputs),
            #[cfg(test)]
            ModelRunner::Noop(_) => Ok(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_guard_limits_and_resets() {
        let guards: Vec<DepthGuard> = (0..MAX_CALL_DEPTH)
            .map(|_| DepthGuard::enter().unwrap())
            .collect();
        assert!(DepthGuard::enter().is_err());
        drop(guards);
        assert!(DepthGuard::enter().is_ok());
    }
}
