use std::path::PathBuf;
use std::time::Instant;

use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;

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
}

impl InferenceEngineImpl {
    pub fn new(pool: SessionPool, _arena: TensorArena) -> Self {
        Self {
            pool,
            start_time: Instant::now(),
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

        let inputs: Vec<(String, ndarray::ArrayD<f32>)> = req
            .inputs
            .iter()
            .map(|inp| {
                let dtype = inp.dtype();
                match engine::DataType::try_from(dtype).unwrap_or(engine::DataType::TypeInvalid) {
                    engine::DataType::TypeFp32 => {
                        let elem_count = inp.data.len() / 4;
                        let mut floats = Vec::with_capacity(elem_count);
                        for chunk in inp.data.chunks_exact(4) {
                            floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                        }
                        let shape: Vec<usize> = inp.shape.iter().map(|&d| d as usize).collect();
                        let total: usize = shape.iter().product();
                        if total != floats.len() {
                            return Err(tonic::Status::invalid_argument(format!(
                                "shape product {} != data len {} for input {}",
                                total,
                                floats.len(),
                                inp.name
                            )));
                        }
                        let array = ndarray::ArrayD::from_shape_vec(
                            ndarray::IxDyn(&shape),
                            floats,
                        )
                        .map_err(|e| tonic::Status::invalid_argument(format!("shape error: {}", e)))?;
                        Ok((inp.name.clone(), array))
                    }
                    other => Err(tonic::Status::invalid_argument(format!(
                        "unsupported dtype {:?}",
                        other
                    ))),
                }
            })
            .collect::<Result<Vec<_>, tonic::Status>>()?;

        let outputs = tokio::task::spawn_blocking({
            let runner = session.runner.clone();
            move || runner.run(inputs)
        })
        .await
        .map_err(|e| tonic::Status::internal(format!("spawn error: {}", e)))?
        .map_err(|e| tonic::Status::internal(format!("inference error: {}", e)))?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        metrics::inc_requests();
        metrics::record_latency(&req.model_name, latency_ms);

        let response_outputs: Vec<InferOutput> = outputs
            .into_iter()
            .map(|(name, array)| {
                let shape: Vec<i64> = array.shape().iter().map(|&d| d as i64).collect();
                let data: Vec<f32> = array.iter().copied().collect();
                let bytes: Vec<u8> = data
                    .into_iter()
                    .flat_map(|f| f.to_le_bytes().to_vec())
                    .collect();
                InferOutput {
                    name,
                    shape,
                    data: bytes,
                    dtype: engine::DataType::TypeFp32.into(),
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

        match self.pool.load_model(&req.name, req.version, &model_path).await {
            Ok(_) => Ok(tonic::Response::new(LoadModelResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => {
                tracing::error!(name=%req.name, version=req.version, error=%e, "model load failed");
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

pub async fn serve(
    socket_path: std::path::PathBuf,
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
    let svc = InferenceEngineImpl::new(pool, arena);

    tracing::info!(path = %socket_path.display(), "gRPC server listening");

    Server::builder()
        .add_service(InferenceEngineServer::new(svc))
        .serve_with_incoming(UnixListenerStream::new(uds))
        .await?;

    Ok(())
}
