use std::sync::Arc;
use std::time::Instant;

use tokio::sync::watch;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use crate::metrics;
use crate::serving::{self, Admission, InferError, ServeContext, MAX_INPUTS};
use crate::session::types::InputTensor;

// Generated code; newer clippy flags tonic's async_trait output.
#[allow(unknown_lints, clippy::double_must_use)]
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

/// Matches the HTTP body limit (tonic's default is 4 MiB).
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

struct KfsService {
    ctx: Arc<ServeContext>,
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    ctx: Arc<ServeContext>,
    mut shutdown: watch::Receiver<bool>,
) {
    if let Ok(addr) = listener.local_addr() {
        tracing::info!(%addr, "gRPC server listening");
    }

    let svc = GrpcInferenceServiceServer::new(KfsService { ctx })
        .max_decoding_message_size(MAX_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_MESSAGE_BYTES);

    if let Err(e) = Server::builder()
        .add_service(svc)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
            let _ = shutdown.wait_for(|v| *v).await;
        })
        .await
    {
        tracing::error!(error = %e, "gRPC server failed");
    }
}

impl KfsService {
    #[allow(clippy::result_large_err)]
    fn authorize<T>(&self, request: &Request<T>) -> Result<(), Status> {
        let auth = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok());
        if self.ctx.authorized(auth) {
            Ok(())
        } else {
            Err(Status::unauthenticated("missing or invalid API key"))
        }
    }
}

impl From<InferError> for Status {
    fn from(e: InferError) -> Self {
        match e {
            InferError::Busy => Status::resource_exhausted(e.to_string()),
            InferError::Unavailable => Status::unavailable(e.to_string()),
            InferError::Timeout(_) => Status::deadline_exceeded(e.to_string()),
            InferError::Failed(_) => Status::internal(e.to_string()),
        }
    }
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
            ready: self.ctx.pool().model_count() > 0,
        }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        self.authorize(&request)?;
        let req = request.into_inner();
        let version: u32 = req.version.parse().unwrap_or(0);

        let ready = if version > 0 {
            self.ctx.pool().get(&req.name, version).is_some()
        } else {
            self.ctx.pool().get_latest(&req.name).is_some()
        };

        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn server_metadata(
        &self,
        request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        self.authorize(&request)?;
        Ok(Response::new(ServerMetadataResponse {
            name: "axon-server".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        self.authorize(&request)?;
        let req = request.into_inner();
        let session = match req.version.parse::<u32>() {
            Ok(v) if v > 0 => self.ctx.pool().get(&req.name, v),
            _ => self.ctx.pool().get_latest(&req.name),
        }
        .ok_or_else(|| Status::not_found(format!("model '{}' not found", req.name)))?;

        let to_meta = |defs: &[crate::model_repository::config_parser::TensorDef]| {
            defs.iter()
                .map(|t| TensorMetadata {
                    name: t.name.clone(),
                    datatype: t.data_type.as_str().to_string(),
                    shape: t.dims.clone(),
                })
                .collect::<Vec<_>>()
        };
        let cfg = session.config.as_deref();

        Ok(Response::new(ModelMetadataResponse {
            versions: self
                .ctx
                .pool()
                .get_versions(&req.name)
                .iter()
                .map(|v| v.to_string())
                .collect(),
            name: req.name,
            platform: session.platform(),
            inputs: cfg.map(|c| to_meta(&c.inputs)).unwrap_or_default(),
            outputs: cfg.map(|c| to_meta(&c.outputs)).unwrap_or_default(),
        }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        self.authorize(&request)?;
        let req = request.into_inner();
        let request_start = Instant::now();

        let session = if req.model_version.is_empty() || req.model_version == "0" {
            self.ctx.pool().get_latest(&req.model_name)
        } else {
            let v: u32 = req
                .model_version
                .parse()
                .map_err(|_| Status::invalid_argument("invalid model version"))?;
            self.ctx.pool().get(&req.model_name, v)
        };

        let session = session.ok_or_else(|| {
            metrics::record_request(&req.model_name, "404");
            Status::not_found(format!("model '{}' not found or not ready", req.model_name))
        })?;

        let inputs = parse_grpc_inputs(req.inputs).inspect_err(|_| {
            metrics::record_request(&req.model_name, "400");
        })?;

        let outputs = serving::execute(
            &self.ctx,
            &session,
            &req.model_name,
            inputs,
            request_start + self.ctx.inference_timeout,
            Admission::Reject,
        )
        .await?;

        tracing::info!(
            model = %req.model_name,
            total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
            "inference completed"
        );

        let response_outputs: Vec<InferOutput> = outputs
            .into_iter()
            .map(|(name, shape, tensor_data)| InferOutput {
                name,
                shape,
                datatype: tensor_data.dtype_str().to_string(),
                raw_data: tensor_data.to_bytes(),
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

/// Decodes little-endian fixed-width elements, checking alignment and the
/// element count against the declared shape.
#[allow(clippy::result_large_err)]
fn decode_fixed<const N: usize, T>(
    inp: &InferInput,
    total: usize,
    from_le: fn([u8; N]) -> T,
) -> Result<Vec<T>, Status> {
    if !inp.raw_data.len().is_multiple_of(N) {
        return Err(Status::invalid_argument(format!(
            "{} data for '{}' not aligned to {N} bytes",
            inp.datatype, inp.name
        )));
    }
    let elem_count = inp.raw_data.len() / N;
    if elem_count != total {
        return Err(Status::invalid_argument(format!(
            "shape product {total} != data elements {elem_count} for '{}'",
            inp.name
        )));
    }
    Ok(inp
        .raw_data
        .as_chunks::<N>()
        .0
        .iter()
        .map(|c| from_le(*c))
        .collect())
}

#[allow(clippy::result_large_err)]
fn parse_grpc_inputs(inputs: Vec<InferInput>) -> Result<Vec<(String, InputTensor)>, Status> {
    if inputs.len() > MAX_INPUTS {
        return Err(Status::invalid_argument(format!(
            "too many inputs: {} (max {MAX_INPUTS})",
            inputs.len()
        )));
    }
    let mut result = Vec::with_capacity(inputs.len());

    for inp in inputs {
        let (shape, total) = serving::validate_shape(&inp.shape)
            .map_err(|e| Status::invalid_argument(format!("input '{}': {e}", inp.name)))?;

        let tensor = match inp.datatype.as_str() {
            "FP32" | "FLOAT32" => {
                let floats = decode_fixed::<4, f32>(&inp, total, f32::from_le_bytes)?;
                if floats.iter().any(|v| !v.is_finite()) {
                    return Err(Status::invalid_argument(format!(
                        "non-finite FP32 value in '{}'",
                        inp.name
                    )));
                }
                InputTensor::F32(floats, shape)
            }
            "INT32" => InputTensor::I32(
                decode_fixed::<4, i32>(&inp, total, i32::from_le_bytes)?,
                shape,
            ),
            "INT64" => InputTensor::I64(
                decode_fixed::<8, i64>(&inp, total, i64::from_le_bytes)?,
                shape,
            ),
            "BYTES" | "STRING" => {
                if total != 1 {
                    return Err(Status::invalid_argument(format!(
                        "BYTES input '{}' must have exactly one element",
                        inp.name
                    )));
                }
                let s = String::from_utf8(inp.raw_data).map_err(|e| {
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

        result.push((inp.name, tensor));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(datatype: &str, shape: Vec<i64>, raw: Vec<u8>) -> InferInput {
        InferInput {
            name: "x".to_string(),
            shape,
            datatype: datatype.to_string(),
            raw_data: raw,
        }
    }

    #[test]
    fn decodes_fp32() {
        let raw: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let parsed = parse_grpc_inputs(vec![input("FP32", vec![1, 2], raw)]).unwrap();
        match &parsed[0].1 {
            InputTensor::F32(d, s) => {
                assert_eq!(d, &vec![1.0, 2.0]);
                assert_eq!(s, &vec![1, 2]);
            }
            _ => panic!("expected FP32"),
        }
    }

    #[test]
    fn rejects_invalid() {
        let nan: Vec<u8> = f32::NAN.to_le_bytes().to_vec();
        assert!(parse_grpc_inputs(vec![input("FP32", vec![1], nan)]).is_err());
        assert!(parse_grpc_inputs(vec![input("FP32", vec![-1], vec![0; 4])]).is_err());
        assert!(parse_grpc_inputs(vec![input("INT64", vec![1], vec![0; 4])]).is_err());
        assert!(parse_grpc_inputs(vec![input("INT32", vec![2], vec![0; 4])]).is_err());
        assert!(parse_grpc_inputs(vec![input("BYTES", vec![2], b"ab".to_vec())]).is_err());
    }
}
