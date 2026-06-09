use std::path::PathBuf;
use std::time::Instant;

use tokio::sync::watch;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use crate::metrics;
use crate::session::pool::SessionPool;
use crate::session::types::InputTensor;

pub mod kfs {
    tonic::include_proto!("inference.kfs");
}

use kfs::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};
use kfs::{
    InferInput, InferOutput, ModelInferRequest, ModelInferResponse, ModelMetadataRequest,
    ModelMetadataResponse, ModelReadyRequest, ModelReadyResponse, ServerLiveRequest,
    ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest,
    ServerReadyResponse, TensorMetadata,
};

struct KfsService {
    pool: SessionPool,
    repo_path: PathBuf,
    inference_timeout: std::time::Duration,
}

pub async fn serve(
    port: u16,
    pool: SessionPool,
    repo_path: PathBuf,
    inference_timeout_ms: u64,
    mut shutdown: watch::Receiver<bool>,
) {
    let svc = KfsService {
        pool,
        repo_path,
        inference_timeout: std::time::Duration::from_millis(inference_timeout_ms),
    };

    let addr = format!("0.0.0.0:{port}").parse().unwrap();

    tracing::info!(port, "gRPC server listening");

    Server::builder()
        .add_service(GrpcInferenceServiceServer::new(svc))
        .serve_with_shutdown(addr, async move {
            let _ = shutdown.wait_for(|v| *v).await;
        })
        .await
        .ok();
}

#[tonic::async_trait]
impl GrpcInferenceService for KfsService {
    async fn server_live(
        &self,
        _request: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _request: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse {
            ready: self.pool.model_count() > 0,
        }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let req = request.into_inner();
        let version: u32 = req.version.parse().unwrap_or(0);

        let ready = if version > 0 {
            self.pool.get(&req.name, version).is_some()
        } else {
            self.pool.get_latest(&req.name).is_some()
        };

        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn server_metadata(
        &self,
        _request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: "axon-server".to_string(),
            version: "0.3.0".to_string(),
        }))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let req = request.into_inner();
        let versions = self.pool.get_versions(&req.name);
        if versions.is_empty() {
            return Err(Status::not_found(format!("model '{}' not found", req.name)));
        }

        let config_path = self.repo_path.join(&req.name).join("config.pbtxt");
        let (inputs, outputs, platform) = if let Ok(content) = std::fs::read(&config_path) {
            if let Ok(cfg) = crate::model_repository::config_parser::parse_model_config(&content) {
                let ins: Vec<TensorMetadata> = cfg
                    .inputs
                    .iter()
                    .map(|t| TensorMetadata {
                        name: t.name.clone(),
                        datatype: t.data_type.as_str().to_string(),
                        shape: t.dims.clone(),
                    })
                    .collect();
                let outs: Vec<TensorMetadata> = cfg
                    .outputs
                    .iter()
                    .map(|t| TensorMetadata {
                        name: t.name.clone(),
                        datatype: t.data_type.as_str().to_string(),
                        shape: t.dims.clone(),
                    })
                    .collect();
                let p = cfg.platform.clone();
                (ins, outs, p)
            } else {
                (vec![], vec![], "onnxruntime_onnx".to_string())
            }
        } else {
            (vec![], vec![], "onnxruntime_onnx".to_string())
        };

        Ok(Response::new(ModelMetadataResponse {
            name: req.name,
            versions: versions.iter().map(|v| v.to_string()).collect(),
            platform,
            inputs,
            outputs,
        }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let req = request.into_inner();

        let session = if req.model_version.is_empty() || req.model_version == "0" {
            self.pool.get_latest(&req.model_name)
        } else {
            let v: u32 = req
                .model_version
                .parse()
                .map_err(|_| Status::invalid_argument("invalid model version"))?;
            self.pool.get(&req.model_name, v)
        };

        let session = session.ok_or_else(|| {
            metrics::record_request(&req.model_name, "404");
            Status::not_found(format!("model '{}' not found or not ready", req.model_name))
        })?;

        let queue_start = Instant::now();
        let _permit = session.concurrency().acquire().await.map_err(|_| {
            metrics::record_request(&req.model_name, "503");
            Status::resource_exhausted("concurrency limit")
        })?;
        metrics::record_queue_wait(&req.model_name, queue_start.elapsed().as_secs_f64());
        metrics::inc_inflight(&req.model_name);

        let inputs = parse_grpc_inputs(&req.inputs).inspect_err(|_s| {
            metrics::dec_inflight(&req.model_name);
            metrics::record_request(&req.model_name, "400");
        })?;

        let start = Instant::now();
        let runner = session.runner.clone();
        let infer_future = tokio::task::spawn_blocking(move || runner.run(inputs));

        let outputs = match tokio::time::timeout(self.inference_timeout, infer_future).await {
            Ok(Ok(Ok(out))) => out,
            Ok(Ok(Err(e))) => {
                metrics::dec_inflight(&req.model_name);
                metrics::record_request(&req.model_name, "500");
                return Err(Status::internal(format!("inference error: {e}")));
            }
            Ok(Err(e)) => {
                metrics::dec_inflight(&req.model_name);
                metrics::record_request(&req.model_name, "500");
                return Err(Status::internal(format!("task join error: {e}")));
            }
            Err(_) => {
                metrics::dec_inflight(&req.model_name);
                metrics::record_request(&req.model_name, "504");
                return Err(Status::deadline_exceeded(format!(
                    "inference timed out after {}ms",
                    self.inference_timeout.as_millis()
                )));
            }
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        metrics::dec_inflight(&req.model_name);
        metrics::record_request(&req.model_name, "200");
        metrics::record_latency(&req.model_name, latency_ms);

        let response_outputs: Vec<InferOutput> = outputs
            .into_iter()
            .map(|(name, shape, tensor_data)| {
                let raw_data = tensor_data.to_bytes();
                let datatype = tensor_data.dtype_str().to_string();
                InferOutput {
                    name,
                    shape,
                    datatype,
                    raw_data,
                }
            })
            .collect();

        Ok(Response::new(ModelInferResponse {
            id: req.id,
            model_name: req.model_name,
            model_version: session.version.to_string(),
            outputs: response_outputs,
        }))
    }
}

#[allow(clippy::result_large_err)]
fn parse_grpc_inputs(inputs: &[InferInput]) -> Result<Vec<(String, InputTensor)>, Status> {
    let mut result = Vec::with_capacity(inputs.len());

    for inp in inputs {
        let shape: Vec<usize> = inp.shape.iter().map(|&d| d as usize).collect();
        let total: usize = shape
            .iter()
            .copied()
            .reduce(|a, b| a.saturating_mul(b))
            .unwrap_or(0);

        let tensor = match inp.datatype.as_str() {
            "FP32" | "FLOAT32" => {
                if inp.raw_data.len() % 4 != 0 {
                    return Err(Status::invalid_argument(format!(
                        "FP32 data for '{}' not aligned to 4 bytes",
                        inp.name
                    )));
                }
                let elem_count = inp.raw_data.len() / 4;
                if elem_count != total {
                    return Err(Status::invalid_argument(format!(
                        "shape product {} != data elements {} for '{}'",
                        total, elem_count, inp.name
                    )));
                }
                let floats: Vec<f32> = inp
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                InputTensor::F32(floats, shape)
            }
            "INT32" => {
                if inp.raw_data.len() % 4 != 0 {
                    return Err(Status::invalid_argument(format!(
                        "INT32 data for '{}' not aligned to 4 bytes",
                        inp.name
                    )));
                }
                let elem_count = inp.raw_data.len() / 4;
                if elem_count != total {
                    return Err(Status::invalid_argument(format!(
                        "shape product {} != data elements {} for '{}'",
                        total, elem_count, inp.name
                    )));
                }
                let ints: Vec<i32> = inp
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                InputTensor::I32(ints, shape)
            }
            "INT64" => {
                if inp.raw_data.len() % 8 != 0 {
                    return Err(Status::invalid_argument(format!(
                        "INT64 data for '{}' not aligned to 8 bytes",
                        inp.name
                    )));
                }
                let elem_count = inp.raw_data.len() / 8;
                if elem_count != total {
                    return Err(Status::invalid_argument(format!(
                        "shape product {} != data elements {} for '{}'",
                        total, elem_count, inp.name
                    )));
                }
                let ints: Vec<i64> = inp
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                InputTensor::I64(ints, shape)
            }
            "BYTES" | "STRING" => {
                let s = String::from_utf8(inp.raw_data.clone()).map_err(|e| {
                    Status::invalid_argument(format!("invalid UTF-8 for '{}': {}", inp.name, e))
                })?;
                InputTensor::String(vec![s], shape)
            }
            _ => {
                return Err(Status::invalid_argument(format!(
                    "unsupported datatype '{}' for input '{}'",
                    inp.datatype, inp.name
                )));
            }
        };

        result.push((inp.name.clone(), tensor));
    }

    Ok(result)
}
