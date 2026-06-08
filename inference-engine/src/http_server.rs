use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::metrics;
use crate::model_repository;
use crate::session::pool::SessionPool;
use crate::session::types::InputTensor;

struct AppState {
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
    let state = Arc::new(AppState {
        pool,
        repo_path,
        inference_timeout: std::time::Duration::from_millis(inference_timeout_ms),
    });

    let app = Router::new()
        .route("/v2/health/live", get(health_live))
        .route("/v2/health/ready", get(health_ready))
        .route("/v2", get(server_metadata))
        .route("/v2/models", get(list_models))
        .route("/v2/models/{model_name}", get(model_metadata))
        .route(
            "/v2/models/{model_name}/versions/{version}",
            get(model_version_metadata),
        )
        .route("/v2/models/{model_name}/infer", post(infer))
        .route(
            "/v2/models/{model_name}/versions/{version}/infer",
            post(infer_version),
        )
        .route("/v2/models/{model_name}/load", post(load_model))
        .route("/v2/models/{model_name}/unload", post(unload_model))
        .route("/v2/repository/index", post(repository_index))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .expect("failed to bind HTTP port");

    tracing::info!(port, "HTTP server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown.wait_for(|v| *v).await;
        })
        .await
        .ok();
}

async fn health_live() -> Json<serde_json::Value> {
    Json(serde_json::json!({"live": true}))
}

async fn health_ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.pool.model_count() > 0 {
        (StatusCode::OK, Json(serde_json::json!({"ready": true})))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"ready": false})))
    }
}

#[derive(Serialize)]
struct ServerMetadataResponse {
    name: String,
    version: String,
    extensions: Vec<String>,
}

async fn server_metadata() -> Json<ServerMetadataResponse> {
    Json(ServerMetadataResponse {
        name: "axon-server".to_string(),
        version: "0.2.0".to_string(),
        extensions: vec![],
    })
}

#[derive(Serialize)]
struct ModelEntry {
    name: String,
    version: String,
    state: String,
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<Vec<ModelEntry>> {
    let models = state.pool.list_models();
    let entries: Vec<ModelEntry> = models
        .into_iter()
        .map(|(name, version, st)| ModelEntry {
            name,
            version: version.to_string(),
            state: format!("{:?}", st),
        })
        .collect();
    Json(entries)
}

#[derive(Serialize)]
struct ModelMetadataResponse {
    name: String,
    versions: Vec<String>,
    platform: String,
    inputs: Vec<TensorMetadataResponse>,
    outputs: Vec<TensorMetadataResponse>,
}

#[derive(Serialize)]
struct TensorMetadataResponse {
    name: String,
    datatype: String,
    shape: Vec<i64>,
}

async fn model_metadata(
    State(state): State<Arc<AppState>>,
    AxumPath(model_name): AxumPath<String>,
) -> Result<Json<ModelMetadataResponse>, StatusCode> {
    let versions = state.pool.get_versions(&model_name);
    if versions.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }

    let config = load_config_for_model(&state.repo_path, &model_name);

    let (inputs, outputs) = match config {
        Some(ref cfg) => {
            let ins: Vec<TensorMetadataResponse> = cfg
                .inputs
                .iter()
                .map(|t| TensorMetadataResponse {
                    name: t.name.clone(),
                    datatype: t.data_type.as_str().to_string(),
                    shape: t.dims.clone(),
                })
                .collect();
            let outs: Vec<TensorMetadataResponse> = cfg
                .outputs
                .iter()
                .map(|t| TensorMetadataResponse {
                    name: t.name.clone(),
                    datatype: t.data_type.as_str().to_string(),
                    shape: t.dims.clone(),
                })
                .collect();
            (ins, outs)
        }
        None => (vec![], vec![]),
    };

    let platform = config
        .as_ref()
        .map(|c| c.platform.clone())
        .unwrap_or_else(|| "onnxruntime_onnx".to_string());

    Ok(Json(ModelMetadataResponse {
        name: model_name,
        versions: versions.iter().map(|v| v.to_string()).collect(),
        platform,
        inputs,
        outputs,
    }))
}

async fn model_version_metadata(
    State(state): State<Arc<AppState>>,
    AxumPath((model_name, version)): AxumPath<(String, String)>,
) -> Result<Json<ModelMetadataResponse>, StatusCode> {
    let v: u32 = version.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
    if state.pool.get(&model_name, v).is_none() {
        return Err(StatusCode::NOT_FOUND);
    }

    let config = load_config_for_model(&state.repo_path, &model_name);
    let (inputs, outputs) = match config {
        Some(ref cfg) => {
            let ins: Vec<TensorMetadataResponse> = cfg
                .inputs
                .iter()
                .map(|t| TensorMetadataResponse {
                    name: t.name.clone(),
                    datatype: t.data_type.as_str().to_string(),
                    shape: t.dims.clone(),
                })
                .collect();
            let outs: Vec<TensorMetadataResponse> = cfg
                .outputs
                .iter()
                .map(|t| TensorMetadataResponse {
                    name: t.name.clone(),
                    datatype: t.data_type.as_str().to_string(),
                    shape: t.dims.clone(),
                })
                .collect();
            (ins, outs)
        }
        None => (vec![], vec![]),
    };

    let platform = config
        .as_ref()
        .map(|c| c.platform.clone())
        .unwrap_or_else(|| "onnxruntime_onnx".to_string());

    Ok(Json(ModelMetadataResponse {
        name: model_name,
        versions: vec![version],
        platform,
        inputs,
        outputs,
    }))
}

#[derive(Deserialize)]
struct InferRequest {
    #[serde(default)]
    id: String,
    inputs: Vec<InferInputRequest>,
}

#[derive(Deserialize)]
struct InferInputRequest {
    name: String,
    shape: Vec<i64>,
    datatype: String,
    data: serde_json::Value,
}

#[derive(Serialize)]
struct InferResponse {
    id: String,
    model_name: String,
    model_version: String,
    outputs: Vec<InferOutputResponse>,
}

#[derive(Serialize)]
struct InferOutputResponse {
    name: String,
    shape: Vec<i64>,
    datatype: String,
    data: Vec<f64>,
}

async fn infer(
    State(state): State<Arc<AppState>>,
    AxumPath(model_name): AxumPath<String>,
    Json(req): Json<InferRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let session = state
        .pool
        .get_latest(&model_name)
        .ok_or(StatusCode::NOT_FOUND)?;
    run_inference(state, session, model_name, req).await
}

async fn infer_version(
    State(state): State<Arc<AppState>>,
    AxumPath((model_name, version)): AxumPath<(String, String)>,
    Json(req): Json<InferRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let v: u32 = version.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
    let session = state
        .pool
        .get(&model_name, v)
        .ok_or(StatusCode::NOT_FOUND)?;
    run_inference(state, session, model_name, req).await
}

async fn run_inference(
    state: Arc<AppState>,
    session: std::sync::Arc<crate::session::pool::ModelSession>,
    model_name: String,
    req: InferRequest,
) -> Result<impl IntoResponse, StatusCode> {
    let queue_start = Instant::now();
    let _permit = session
        .concurrency()
        .acquire()
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "concurrency limit reached");
            metrics::record_request(&model_name, "503");
            StatusCode::SERVICE_UNAVAILABLE
        })?;
    metrics::record_queue_wait(&model_name, queue_start.elapsed().as_secs_f64());
    metrics::inc_inflight(&model_name);

    let inputs = parse_http_inputs(&req.inputs).map_err(|e| {
        tracing::warn!(error = %e, "bad request");
        metrics::dec_inflight(&model_name);
        metrics::record_request(&model_name, "400");
        StatusCode::BAD_REQUEST
    })?;

    let start = Instant::now();
    let runner = session.runner.clone();
    let infer_future = tokio::task::spawn_blocking(move || runner.run(inputs));

    let outputs = match tokio::time::timeout(state.inference_timeout, infer_future).await {
        Ok(Ok(Ok(out))) => out,
        Ok(Ok(Err(e))) => {
            metrics::dec_inflight(&model_name);
            tracing::error!(error = %e, "inference failed");
            metrics::record_request(&model_name, "500");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
        Ok(Err(e)) => {
            metrics::dec_inflight(&model_name);
            tracing::error!(error = %e, "spawn_blocking failed");
            metrics::record_request(&model_name, "500");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
        Err(_) => {
            metrics::dec_inflight(&model_name);
            tracing::warn!(model = %model_name, timeout_ms = state.inference_timeout.as_millis(), "inference timed out");
            metrics::record_request(&model_name, "504");
            return Err(StatusCode::GATEWAY_TIMEOUT);
        }
    };

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics::dec_inflight(&model_name);
    metrics::record_request(&model_name, "200");
    metrics::record_latency(&model_name, latency_ms);

    let response_outputs: Vec<InferOutputResponse> = outputs
        .into_iter()
        .map(|(name, shape, tensor_data)| {
            let dtype = tensor_data.dtype_str().to_string();
            let data = match tensor_data {
                crate::session::types::TensorData::F32(d) => {
                    d.into_iter().map(|v| v as f64).collect()
                }
                crate::session::types::TensorData::I32(d) => {
                    d.into_iter().map(|v| v as f64).collect()
                }
                crate::session::types::TensorData::I64(d) => {
                    d.into_iter().map(|v| v as f64).collect()
                }
            };
            InferOutputResponse {
                name,
                shape,
                datatype: dtype,
                data,
            }
        })
        .collect();

    Ok(Json(InferResponse {
        id: req.id,
        model_name: model_name.clone(),
        model_version: session.version.to_string(),
        outputs: response_outputs,
    }))
}

fn parse_http_inputs(inputs: &[InferInputRequest]) -> anyhow::Result<Vec<(String, InputTensor)>> {
    let mut result = Vec::with_capacity(inputs.len());

    for inp in inputs {
        let shape: Vec<usize> = inp.shape.iter().map(|&d| d as usize).collect();
        let total: usize = shape.iter().copied().reduce(|a, b| a.saturating_mul(b)).unwrap_or(0);

        let tensor = match inp.datatype.as_str() {
            "FP32" | "FLOAT32" => {
                let data = extract_f64_array(&inp.data, total)?;
                let floats: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
                InputTensor::F32(floats, shape)
            }
            "INT32" => {
                let data = extract_f64_array(&inp.data, total)?;
                let ints: Vec<i32> = data.into_iter().map(|v| v as i32).collect();
                InputTensor::I32(ints, shape)
            }
            "INT64" => {
                let data = extract_f64_array(&inp.data, total)?;
                let ints: Vec<i64> = data.into_iter().map(|v| v as i64).collect();
                InputTensor::I64(ints, shape)
            }
            "BYTES" | "STRING" => {
                let strings: Vec<String> = if let Some(arr) = inp.data.as_array() {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                } else {
                    vec![]
                };
                InputTensor::String(strings, shape)
            }
            _ => anyhow::bail!("unsupported datatype: {}", inp.datatype),
        };

        result.push((inp.name.clone(), tensor));
    }

    Ok(result)
}

fn extract_f64_array(value: &serde_json::Value, expected: usize) -> anyhow::Result<Vec<f64>> {
    match value {
        serde_json::Value::Array(arr) => {
            let flat = flatten_json_array(arr);
            if flat.len() != expected && expected > 0 {
                anyhow::bail!(
                    "data length {} doesn't match shape product {}",
                    flat.len(),
                    expected
                );
            }
            Ok(flat)
        }
        _ => anyhow::bail!("expected array for tensor data"),
    }
}

fn flatten_json_array(arr: &[serde_json::Value]) -> Vec<f64> {
    let mut result = Vec::new();
    for v in arr {
        match v {
            serde_json::Value::Number(n) => {
                result.push(n.as_f64().unwrap_or(0.0));
            }
            serde_json::Value::Array(inner) => {
                result.extend(flatten_json_array(inner));
            }
            _ => {}
        }
    }
    result
}

#[derive(Deserialize)]
struct LoadRequest {
    #[serde(default)]
    version: Option<u32>,
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    AxumPath(model_name): AxumPath<String>,
    body: Option<Json<LoadRequest>>,
) -> StatusCode {
    let version = body.and_then(|b| b.version).unwrap_or(1);
    let model_file = state
        .repo_path
        .join(&model_name)
        .join(version.to_string())
        .join("model.onnx");

    if !model_file.exists() {
        return StatusCode::NOT_FOUND;
    }

    match state.pool.load_model(&model_name, version, &model_file, 4) {
        Ok(_) => {
            metrics::set_models_count(state.pool.model_count() as i64);
            StatusCode::OK
        }
        Err(e) => {
            tracing::error!(model = %model_name, version, error = %e, "load failed");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

#[derive(Deserialize)]
struct UnloadRequest {
    #[serde(default)]
    version: Option<u32>,
}

async fn unload_model(
    State(state): State<Arc<AppState>>,
    AxumPath(model_name): AxumPath<String>,
    body: Option<Json<UnloadRequest>>,
) -> StatusCode {
    let version = body.and_then(|b| b.version).unwrap_or(1);
    match state.pool.unload_model(&model_name, version) {
        Ok(_) => {
            metrics::set_models_count(state.pool.model_count() as i64);
            StatusCode::OK
        }
        Err(_) => StatusCode::NOT_FOUND,
    }
}

#[derive(Serialize)]
struct RepoModelEntry {
    name: String,
    state: String,
}

async fn repository_index(State(state): State<Arc<AppState>>) -> Json<Vec<RepoModelEntry>> {
    let names = state.pool.all_model_names();
    let entries: Vec<RepoModelEntry> = names
        .into_iter()
        .map(|name| RepoModelEntry {
            name,
            state: "READY".to_string(),
        })
        .collect();
    Json(entries)
}

fn load_config_for_model(
    repo_path: &PathBuf,
    model_name: &str,
) -> Option<model_repository::ModelConfig> {
    let config_path = repo_path.join(model_name).join("config.pbtxt");
    let content = std::fs::read(&config_path).ok()?;
    crate::model_repository::config_parser::parse_model_config(&content).ok()
}
