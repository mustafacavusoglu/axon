use std::path::PathBuf;
use std::time::{Duration, Instant};

use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;

use crate::config::EngineConfig;
use crate::session::runner::{InputTensor, TensorData};
use crate::arena::TensorArena;
use crate::metrics;
use crate::session::pool::SessionPool;

pub mod engine {
    tonic::include_proto!("inference.engine.v1");
}

use engine::inference_engine_server::{InferenceEngine, InferenceEngineServer};
use engine::{
    BatchInferRequest, BatchInferResponse, HealthRequest, HealthResponse, InferOutput,
    LoadModelRequest, LoadModelResponse, ModelStatusRequest, ModelStatusResponse,
    UnloadModelRequest, UnloadModelResponse,
};

pub struct InferenceEngineImpl {
    pool: SessionPool,
    start_time: Instant,
    timeout: Duration,
}

impl InferenceEngineImpl {
    pub fn new(pool: SessionPool, _arena: TensorArena, timeout_ms: u64) -> Self {
        Self {
            pool,
            start_time: Instant::now(),
            timeout: Duration::from_millis(timeout_ms),
        }
    }
}

#[tonic::async_trait]
impl InferenceEngine for InferenceEngineImpl {
    async fn batch_infer(
        &self,
        request: tonic::Request<BatchInferRequest>,
    ) -> Result<tonic::Response<BatchInferResponse>, tonic::Status> {
        let req = request.into_inner();
        let start = Instant::now();

        let session = self.pool.get(&req.model_name, req.version).map_err(|e| {
            tonic::Status::not_found(format!("model not found: {}", e))
        })?;

        let _permit = session.concurrency.acquire().await.map_err(|e| {
            tonic::Status::resource_exhausted(format!("model concurrency limit reached: {}", e))
        });

        let inputs: Vec<(String, InputTensor)> = req
            .inputs
            .iter()
            .map(|inp| {
                let shape: Vec<usize> = inp.shape.iter().map(|&d| d as usize).collect();
                let dtype = engine::DataType::try_from(inp.dtype()).unwrap_or(engine::DataType::TypeInvalid);

                match dtype {
                    engine::DataType::TypeFp32 => {
                        let floats = parse_f32(&inp.data, &shape, &inp.name)?;
                        Ok((inp.name.clone(), InputTensor::F32(floats, shape)))
                    }
                    engine::DataType::TypeInt32 => {
                        let ints = parse_i32(&inp.data, &shape, &inp.name)?;
                        Ok((inp.name.clone(), InputTensor::I32(ints, shape)))
                    }
                    engine::DataType::TypeInt64 => {
                        let ints = parse_i64(&inp.data, &shape, &inp.name)?;
                        Ok((inp.name.clone(), InputTensor::I64(ints, shape)))
                    }
                    engine::DataType::TypeString => {
                        if shape.len() == 1 && shape[0] == 1 {
                            let s = String::from_utf8(inp.data.clone())
                                .map_err(|e| tonic::Status::invalid_argument(format!(
                                    "invalid utf8 string for '{}': {}", inp.name, e
                                )))?;
                            Ok((inp.name.clone(), InputTensor::String(vec![s], shape)))
                        } else {
                            Err(tonic::Status::invalid_argument(format!(
                                "multi-element string tensor not yet supported for '{}'", inp.name
                            )))
                        }
                    }
                    _ => Err(tonic::Status::invalid_argument(format!(
                        "unsupported dtype {:?} for input '{}'",
                        dtype, inp.name
                    ))),
                }
            })
            .collect::<Result<Vec<_>, tonic::Status>>()?;

        let timeout = self.timeout;
        let infer_future = tokio::task::spawn_blocking({
            let runner = session.runner.clone();
            move || runner.run(inputs)
        });

        let outputs = tokio::time::timeout(timeout, infer_future)
            .await
            .map_err(|_| tonic::Status::deadline_exceeded(format!(
                "inference timed out after {}ms", timeout.as_millis()
            )))?
            .map_err(|e| tonic::Status::internal(format!("spawn error: {}", e)))?
            .map_err(|e| tonic::Status::internal(format!("inference error: {}", e)))?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        metrics::inc_requests();
        metrics::record_latency(&req.model_name, latency_ms);

        let response_outputs: Vec<InferOutput> = outputs
            .into_iter()
            .map(|(name, shape, tensor_data)| {
                let bytes = tensor_data.to_bytes();
                let dtype = match tensor_data {
                    TensorData::F32(_) => engine::DataType::TypeFp32,
                    TensorData::I32(_) => engine::DataType::TypeInt32,
                    TensorData::I64(_) => engine::DataType::TypeInt64,
                };
                InferOutput {
                    name,
                    shape,
                    data: bytes,
                    dtype: dtype.into(),
                }
            })
            .collect();

        Ok(tonic::Response::new(BatchInferResponse {
            outputs: response_outputs,
            latency_ms,
        }))
    }

    async fn load_model(
        &self,
        request: tonic::Request<LoadModelRequest>,
    ) -> Result<tonic::Response<LoadModelResponse>, tonic::Status> {
        let req = request.into_inner();
        let model_path = PathBuf::from(&req.model_path);
        let concurrency = req.concurrency;

        let pool = self.pool.clone();
        let name = req.name.clone();
        let name_for_err = name.clone();
        let version = req.version;

        match tokio::task::spawn_blocking(move || {
            pool.load_model(&name, version, &model_path, concurrency)
        }).await {
            Ok(Ok(_)) => Ok(tonic::Response::new(LoadModelResponse {
                success: true,
                error: String::new(),
            })),
            Ok(Err(e)) => {
                tracing::error!(name=%name_for_err, version=version, error=%e, "model load failed");
                Ok(tonic::Response::new(LoadModelResponse {
                    success: false,
                    error: e.to_string(),
                }))
            }
            Err(e) => {
                tracing::error!(name=%name_for_err, version=version, error=%e, "spawn_blocking failed");
                Ok(tonic::Response::new(LoadModelResponse {
                    success: false,
                    error: e.to_string(),
                }))
            }
        }
    }

    async fn unload_model(
        &self,
        request: tonic::Request<UnloadModelRequest>,
    ) -> Result<tonic::Response<UnloadModelResponse>, tonic::Status> {
        let req = request.into_inner();
        match self.pool.unload_model(&req.name, req.version) {
            Ok(_) => Ok(tonic::Response::new(UnloadModelResponse { success: true })),
            Err(e) => {
                tracing::warn!(name=%req.name, version=req.version, error=%e, "model unload failed");
                Ok(tonic::Response::new(UnloadModelResponse { success: false }))
            }
        }
    }

    async fn model_status(
        &self,
        request: tonic::Request<ModelStatusRequest>,
    ) -> Result<tonic::Response<ModelStatusResponse>, tonic::Status> {
        let req = request.into_inner();
        let session = self.pool.get(&req.name, req.version).map_err(|e| {
            tonic::Status::not_found(format!("model not found: {}", e))
        })?;

        let status_str = match session.state {
            crate::session::pool::SessionState::Loading => "LOADING",
            crate::session::pool::SessionState::Ready => "READY",
            crate::session::pool::SessionState::Unloading => "UNLOADING",
            crate::session::pool::SessionState::Error => "ERROR",
        };

        Ok(tonic::Response::new(ModelStatusResponse {
            name: session.name.clone(),
            version: session.version,
            state: status_str.to_string(),
            platform: "onnxruntime_onnx".to_string(),
            memory_bytes: session.memory_bytes.load(std::sync::atomic::Ordering::Acquire),
        }))
    }

    async fn healthcheck(
        &self,
        _request: tonic::Request<HealthRequest>,
    ) -> Result<tonic::Response<HealthResponse>, tonic::Status> {
        Ok(tonic::Response::new(HealthResponse {
            healthy: true,
            uptime_sec: self.start_time.elapsed().as_secs(),
        }))
    }
}

fn parse_f32(data: &[u8], shape: &[usize], name: &str) -> Result<Vec<f32>, tonic::Status> {
    let elem_count = data.len() / 4;
    let total: usize = shape.iter().product();
    if total != elem_count {
        return Err(tonic::Status::invalid_argument(format!(
            "shape product {} != data elements {} for {}",
            total, elem_count, name
        )));
    }
    let mut result = Vec::with_capacity(elem_count);
    for chunk in data.chunks_exact(4) {
        result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(result)
}

fn parse_i32(data: &[u8], shape: &[usize], name: &str) -> Result<Vec<i32>, tonic::Status> {
    let elem_count = data.len() / 4;
    let total: usize = shape.iter().product();
    if total != elem_count {
        return Err(tonic::Status::invalid_argument(format!(
            "shape product {} != data elements {} for {}",
            total, elem_count, name
        )));
    }
    let mut result = Vec::with_capacity(elem_count);
    for chunk in data.chunks_exact(4) {
        result.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(result)
}

fn parse_i64(data: &[u8], shape: &[usize], name: &str) -> Result<Vec<i64>, tonic::Status> {
    let elem_count = data.len() / 8;
    let total: usize = shape.iter().product();
    if total != elem_count {
        return Err(tonic::Status::invalid_argument(format!(
            "shape product {} != data elements {} for {}",
            total, elem_count, name
        )));
    }
    let mut result = Vec::with_capacity(elem_count);
    for chunk in data.chunks_exact(8) {
        result.push(i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(result)
}

pub async fn serve(
    socket_path: std::path::PathBuf,
    config: EngineConfig,
    pool: SessionPool,
    arena: TensorArena,
) -> anyhow::Result<()> {
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let uds = UnixListener::bind(&socket_path)?;
    let svc = InferenceEngineImpl::new(pool, arena, config.inference_timeout_ms);

    tracing::info!(path = %socket_path.display(), "gRPC server listening");

    Server::builder()
        .add_service(InferenceEngineServer::new(svc))
        .serve_with_incoming(UnixListenerStream::new(uds))
        .await?;

    Ok(())
}
