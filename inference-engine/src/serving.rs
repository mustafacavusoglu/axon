//! Shared request handling for the HTTP and gRPC front-ends: authentication,
//! admission control and the blocking inference call.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::metrics;
use crate::model_repository::Repository;
use crate::session::pool::{ModelSession, SessionPool};
use crate::session::types::{InferenceOutput, InputTensor};

/// Upper bound on any single tensor dimension.
pub const MAX_DIM_SIZE: i64 = 1_000_000;
/// Upper bound on the element count of a single input tensor.
pub const MAX_TENSOR_ELEMENTS: usize = 100_000_000;
/// Upper bound on the number of input tensors in one request.
pub const MAX_INPUTS: usize = 64;

pub struct ServeContext {
    pub repo: Repository,
    pub inference_timeout: Duration,
    pub api_key: Option<Arc<str>>,
    /// Load/unload API enabled (`--model-control-mode=explicit`).
    pub explicit_model_control: bool,
}

impl ServeContext {
    pub fn pool(&self) -> &SessionPool {
        &self.repo.pool
    }

    /// Checks an `Authorization` header value against the configured key.
    /// Always succeeds when no key is configured.
    pub fn authorized(&self, authorization: Option<&str>) -> bool {
        let Some(expected) = self.api_key.as_deref() else {
            return true;
        };
        let Some(value) = authorization else {
            return false;
        };
        let token = value
            .strip_prefix("Bearer ")
            .or_else(|| value.strip_prefix("bearer "))
            .unwrap_or("")
            .trim();
        constant_time_eq(token.as_bytes(), expected.as_bytes())
    }
}

/// Length-independent comparison so the key cannot be recovered byte by byte
/// through response timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= (x ^ y) as usize;
    }
    diff == 0
}

/// Validates a request shape and returns it as `usize` dims plus the element
/// count.
pub fn validate_shape(shape: &[i64]) -> anyhow::Result<(Vec<usize>, usize)> {
    if shape.len() > 16 {
        anyhow::bail!("too many dimensions: {} (max 16)", shape.len());
    }
    let mut dims = Vec::with_capacity(shape.len());
    let mut total: usize = 1;
    for &d in shape {
        if !(0..=MAX_DIM_SIZE).contains(&d) {
            anyhow::bail!("dimension out of range: {d} (max {MAX_DIM_SIZE})");
        }
        dims.push(d as usize);
        total = total
            .checked_mul(d as usize)
            .ok_or_else(|| anyhow::anyhow!("shape product overflow: {shape:?}"))?;
    }
    if total > MAX_TENSOR_ELEMENTS {
        anyhow::bail!("tensor too large: {total} elements (max {MAX_TENSOR_ELEMENTS})");
    }
    Ok((dims, total))
}

#[derive(Debug)]
pub enum InferError {
    /// Model concurrency limit reached (HTTP 429 / RESOURCE_EXHAUSTED).
    Busy,
    /// Compute admission failed or timed out (HTTP 503 / UNAVAILABLE).
    Unavailable,
    /// Deadline exceeded (HTTP 504 / DEADLINE_EXCEEDED).
    Timeout(Duration),
    /// Model execution failed (HTTP 500 / INTERNAL).
    Failed(String),
}

impl InferError {
    pub fn status_label(&self) -> &'static str {
        match self {
            InferError::Busy => "429",
            InferError::Unavailable => "503",
            InferError::Timeout(_) => "504",
            InferError::Failed(_) => "500",
        }
    }
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferError::Busy => write!(f, "model concurrency limit reached"),
            InferError::Unavailable => write!(f, "compute admission failed"),
            InferError::Timeout(d) => write!(f, "inference timed out after {}ms", d.as_millis()),
            InferError::Failed(e) => write!(f, "inference failed: {e}"),
        }
    }
}

/// Decrements the in-flight compute gauge when the blocking task finishes
/// (not when the client gives up), so the gauge reflects real GPU/CPU work.
struct ComputeGuard(String);

impl ComputeGuard {
    fn new(model: &str) -> Self {
        metrics::inc_inflight_compute(model);
        ComputeGuard(model.to_string())
    }
}

impl Drop for ComputeGuard {
    fn drop(&mut self) {
        metrics::dec_inflight_compute(&self.0);
    }
}

/// How to obtain the per-model concurrency permit.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Fail fast with `Busy` when all instances are in use.
    Reject,
    /// Wait for a free instance until the deadline.
    Wait,
}

/// Runs one inference with admission control and a deadline. Records request
/// and latency metrics.
pub async fn execute(
    ctx: &ServeContext,
    session: &Arc<ModelSession>,
    model_name: &str,
    inputs: Vec<(String, InputTensor)>,
    deadline: Instant,
    admission: Admission,
) -> Result<InferenceOutput, InferError> {
    let result = execute_inner(ctx, session, model_name, inputs, deadline, admission).await;
    match &result {
        Ok(_) => metrics::record_request(model_name, "200"),
        Err(e) => metrics::record_request(model_name, e.status_label()),
    }
    result
}

async fn execute_inner(
    ctx: &ServeContext,
    session: &Arc<ModelSession>,
    model_name: &str,
    inputs: Vec<(String, InputTensor)>,
    deadline: Instant,
    admission: Admission,
) -> Result<InferenceOutput, InferError> {
    let timeout = ctx.inference_timeout;
    let remaining = || deadline.saturating_duration_since(Instant::now());

    let model_permit = match admission {
        Admission::Reject => session
            .concurrency()
            .clone()
            .try_acquire_owned()
            .map_err(|_| InferError::Busy)?,
        Admission::Wait => {
            match tokio::time::timeout(remaining(), session.concurrency().clone().acquire_owned())
                .await
            {
                Ok(Ok(p)) => p,
                Ok(Err(_)) => return Err(InferError::Unavailable),
                Err(_) => return Err(InferError::Timeout(timeout)),
            }
        }
    };

    let queue_start = Instant::now();
    let cpu_permit = match tokio::time::timeout(
        remaining(),
        ctx.pool().cpu_semaphore.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(_)) => return Err(InferError::Unavailable),
        Err(_) => {
            tracing::warn!(model = %model_name, "compute admission timed out");
            return Err(InferError::Unavailable);
        }
    };
    metrics::record_queue_wait(model_name, queue_start.elapsed().as_secs_f64());

    let compute = ComputeGuard::new(model_name);
    let start = Instant::now();
    let runner = session.runner.clone();
    let handle = tokio::task::spawn_blocking(move || {
        // Permits are released only when compute really finishes, even if
        // the caller timed out, so overload cannot pile up hidden work.
        let _model_guard = model_permit;
        let _cpu_guard = cpu_permit;
        let _compute = compute;
        runner.run(inputs)
    });

    let outputs = match tokio::time::timeout(remaining(), handle).await {
        Ok(Ok(Ok(out))) => out,
        Ok(Ok(Err(e))) => {
            tracing::error!(model = %model_name, error = %e, "inference failed");
            return Err(InferError::Failed(e.to_string()));
        }
        Ok(Err(e)) => {
            tracing::error!(model = %model_name, error = %e, "inference task failed");
            return Err(InferError::Failed("inference task panicked".to_string()));
        }
        Err(_) => {
            tracing::warn!(model = %model_name, timeout_ms = timeout.as_millis(), "inference timed out");
            return Err(InferError::Timeout(timeout));
        }
    };

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics::record_latency(model_name, latency_ms);
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_works() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret2"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn shape_validation() {
        assert_eq!(validate_shape(&[2, 3]).unwrap(), (vec![2, 3], 6));
        assert_eq!(validate_shape(&[]).unwrap(), (vec![], 1));
        assert_eq!(validate_shape(&[0, 5]).unwrap().1, 0);
        assert!(validate_shape(&[-1]).is_err());
        assert!(validate_shape(&[MAX_DIM_SIZE + 1]).is_err());
        assert!(validate_shape(&[1_000_000, 1_000_000]).is_err());
        assert!(validate_shape(&[1; 17]).is_err());
    }
}
