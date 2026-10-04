use std::sync::Arc;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Path as AxumPath, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::model_repository::config_parser::TensorDef;
use crate::model_repository::is_valid_model_name;
use crate::serving::{self, Admission, InferError, ServeContext, MAX_INPUTS};
use crate::session::pool::ModelSession;
use crate::session::types::{InferenceOutput, InputTensor, TensorData};

const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_BATCH_SIZE: usize = 128;
const MAX_JSON_DEPTH: usize = 8;

type Ctx = State<Arc<ServeContext>>;

/// JSON error body (`{"error": "..."}`), as used by KServe v2.
struct ApiError(StatusCode, String);

impl ApiError {
    fn new(status: StatusCode, msg: impl Into<String>) -> Self {
        ApiError(status, msg.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

impl From<InferError> for ApiError {
    fn from(e: InferError) -> Self {
        let status = match e {
            InferError::Busy => StatusCode::TOO_MANY_REQUESTS,
            InferError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            InferError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
            InferError::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        ApiError(status, e.to_string())
    }
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    ctx: Arc<ServeContext>,
    mut shutdown: watch::Receiver<bool>,
) {
    let protected = Router::new()
        .route("/v2", get(server_metadata))
        .route("/v2/models", get(list_models))
        .route("/v2/models/{model_name}", get(model_metadata))
        .route("/v2/models/{model_name}/ready", get(model_ready))
        .route(
            "/v2/models/{model_name}/versions/{version}",
            get(model_version_metadata),
        )
        .route(
            "/v2/models/{model_name}/versions/{version}/ready",
            get(model_version_ready),
        )
        .route("/v2/models/{model_name}/infer", post(infer))
        .route(
            "/v2/models/{model_name}/versions/{version}/infer",
            post(infer_version),
        )
        .route("/v2/models/{model_name}/infer_batch", post(infer_batch))
        .route(
            "/v2/models/{model_name}/versions/{version}/infer_batch",
            post(infer_batch_version),
        )
        .route("/v2/models/{model_name}/load", post(load_model))
        .route("/v2/models/{model_name}/unload", post(unload_model))
        .route("/v2/repository/index", post(repository_index))
        .route_layer(middleware::from_fn_with_state(ctx.clone(), require_api_key));

    // Health endpoints stay unauthenticated for orchestrator probes.
    let mut app = Router::new()
        .route("/v2/health/live", get(health_live))
        .route("/v2/health/ready", get(health_ready))
        .merge(protected);

    // The UI assets are static and secret-free, so they are served without
    // the API key (a browser navigation cannot send a Bearer header); every
    // API call the UI makes carries the key the user typed in.
    if ctx.ui_enabled {
        app = app.merge(crate::ui::router());
    }

    let app = app
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(ctx);

    if let Ok(addr) = listener.local_addr() {
        tracing::info!(%addr, "HTTP server listening");
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown.wait_for(|v| *v).await;
        })
        .await
        .ok();
}

async fn require_api_key(State(ctx): Ctx, req: Request, next: Next) -> Response {
    let auth = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    if !ctx.authorized(auth) {
        return (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(serde_json::json!({ "error": "unauthorized" })),
        )
            .into_response();
    }
    next.run(req).await
}

async fn health_live() -> Json<serde_json::Value> {
    Json(serde_json::json!({"live": true}))
}

async fn health_ready(State(ctx): Ctx) -> impl IntoResponse {
    if ctx.pool().model_count() > 0 {
        (StatusCode::OK, Json(serde_json::json!({"ready": true})))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"ready": false})),
        )
    }
}

#[derive(Serialize)]
struct ServerMetadataResponse {
    name: &'static str,
    version: &'static str,
    extensions: Vec<String>,
}

async fn server_metadata() -> Json<ServerMetadataResponse> {
    Json(ServerMetadataResponse {
        name: "axon-server",
        version: env!("CARGO_PKG_VERSION"),
        extensions: vec![],
    })
}

#[derive(Serialize)]
struct ModelEntry {
    name: String,
    version: String,
    state: String,
    platform: &'static str,
    /// Where the model executes (`cpu`, `cuda:0`, ...).
    device: String,
}

async fn list_models(State(ctx): Ctx) -> Json<Vec<ModelEntry>> {
    let mut models = ctx.pool().list_models();
    models.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    Json(
        models
            .into_iter()
            .map(|(name, version, st)| {
                let session = ctx.pool().get(&name, version);
                ModelEntry {
                    platform: session
                        .as_ref()
                        .map(|s| s.runner.platform_name())
                        .unwrap_or("unknown"),
                    device: session
                        .as_ref()
                        .map(|s| s.runner.device_label())
                        .unwrap_or_default(),
                    name,
                    version: version.to_string(),
                    state: format!("{st:?}"),
                }
            })
            .collect(),
    )
}

#[derive(Serialize)]
struct ModelMetadataResponse {
    name: String,
    versions: Vec<String>,
    platform: String,
    device: String,
    inputs: Vec<TensorMetadataResponse>,
    outputs: Vec<TensorMetadataResponse>,
}

#[derive(Serialize)]
struct TensorMetadataResponse {
    name: String,
    datatype: String,
    shape: Vec<i64>,
}

fn tensor_metadata(defs: &[TensorDef]) -> Vec<TensorMetadataResponse> {
    defs.iter()
        .map(|t| TensorMetadataResponse {
            name: t.name.clone(),
            datatype: t.data_type.as_str().to_string(),
            shape: t.dims.clone(),
        })
        .collect()
}

fn metadata_response(
    name: String,
    versions: Vec<String>,
    session: &ModelSession,
) -> ModelMetadataResponse {
    let cfg = session.config.as_deref();
    ModelMetadataResponse {
        name,
        versions,
        platform: cfg
            .map(|c| c.platform.clone())
            .unwrap_or_else(|| "onnxruntime_onnx".to_string()),
        device: session.runner.device_label(),
        inputs: cfg.map(|c| tensor_metadata(&c.inputs)).unwrap_or_default(),
        outputs: cfg.map(|c| tensor_metadata(&c.outputs)).unwrap_or_default(),
    }
}

fn not_found(model: &str) -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        format!("model '{model}' not found or not ready"),
    )
}

fn parse_version(version: &str) -> Result<u32, ApiError> {
    version
        .parse()
        .map_err(|_| ApiError::new(StatusCode::BAD_REQUEST, "invalid model version"))
}

async fn model_metadata(
    State(ctx): Ctx,
    AxumPath(model_name): AxumPath<String>,
) -> Result<Json<ModelMetadataResponse>, ApiError> {
    let session = ctx
        .pool()
        .get_latest(&model_name)
        .ok_or_else(|| not_found(&model_name))?;
    let versions = ctx
        .pool()
        .get_versions(&model_name)
        .iter()
        .map(|v| v.to_string())
        .collect();
    Ok(Json(metadata_response(model_name, versions, &session)))
}

async fn model_version_metadata(
    State(ctx): Ctx,
    AxumPath((model_name, version)): AxumPath<(String, String)>,
) -> Result<Json<ModelMetadataResponse>, ApiError> {
    let v = parse_version(&version)?;
    let session = ctx
        .pool()
        .get(&model_name, v)
        .ok_or_else(|| not_found(&model_name))?;
    Ok(Json(metadata_response(model_name, vec![version], &session)))
}

async fn model_ready(State(ctx): Ctx, AxumPath(model_name): AxumPath<String>) -> StatusCode {
    if ctx.pool().get_latest(&model_name).is_some() {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn model_version_ready(
    State(ctx): Ctx,
    AxumPath((model_name, version)): AxumPath<(String, String)>,
) -> StatusCode {
    match version.parse::<u32>() {
        Ok(v) if ctx.pool().get(&model_name, v).is_some() => StatusCode::OK,
        Ok(_) => StatusCode::NOT_FOUND,
        Err(_) => StatusCode::BAD_REQUEST,
    }
}

#[derive(Deserialize)]
struct InferRequest {
    #[serde(default)]
    id: String,
    inputs: Vec<InferInputRequest>,
}

#[derive(Deserialize)]
struct BatchInferRequest {
    #[serde(default)]
    id: String,
    requests: Vec<InferRequest>,
}

#[derive(Serialize)]
struct BatchInferResponse {
    id: String,
    model_name: String,
    model_version: String,
    responses: Vec<BatchInferItem>,
}

#[derive(Serialize)]
struct BatchInferItem {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    outputs: Option<Vec<InferOutputResponse>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
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

/// Output data serialised straight from the typed vectors, avoiding an
/// intermediate `serde_json::Value` per element.
#[derive(Serialize)]
#[serde(untagged)]
enum OutputData {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Str(Vec<String>),
}

#[derive(Serialize)]
struct InferOutputResponse {
    name: String,
    shape: Vec<i64>,
    datatype: &'static str,
    data: OutputData,
}

fn to_output_responses(outputs: InferenceOutput) -> Vec<InferOutputResponse> {
    outputs
        .into_iter()
        .map(|(name, shape, tensor_data)| {
            let datatype = tensor_data.dtype_str();
            let data = match tensor_data {
                TensorData::F32(d) => OutputData::F32(d),
                TensorData::I32(d) => OutputData::I32(d),
                TensorData::I64(d) => OutputData::I64(d),
                TensorData::String(d) => OutputData::Str(d),
            };
            InferOutputResponse {
                name,
                shape,
                datatype,
                data,
            }
        })
        .collect()
}

async fn infer(
    State(ctx): Ctx,
    AxumPath(model_name): AxumPath<String>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, ApiError> {
    let session = ctx
        .pool()
        .get_latest(&model_name)
        .ok_or_else(|| not_found(&model_name))?;
    run_inference(ctx, session, model_name, req).await
}

async fn infer_version(
    State(ctx): Ctx,
    AxumPath((model_name, version)): AxumPath<(String, String)>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, ApiError> {
    let v = parse_version(&version)?;
    let session = ctx
        .pool()
        .get(&model_name, v)
        .ok_or_else(|| not_found(&model_name))?;
    run_inference(ctx, session, model_name, req).await
}

async fn infer_batch(
    State(ctx): Ctx,
    AxumPath(model_name): AxumPath<String>,
    Json(req): Json<BatchInferRequest>,
) -> Result<Json<BatchInferResponse>, ApiError> {
    let session = ctx
        .pool()
        .get_latest(&model_name)
        .ok_or_else(|| not_found(&model_name))?;
    run_batch_inference(ctx, session, model_name, req).await
}

async fn infer_batch_version(
    State(ctx): Ctx,
    AxumPath((model_name, version)): AxumPath<(String, String)>,
    Json(req): Json<BatchInferRequest>,
) -> Result<Json<BatchInferResponse>, ApiError> {
    let v = parse_version(&version)?;
    let session = ctx
        .pool()
        .get(&model_name, v)
        .ok_or_else(|| not_found(&model_name))?;
    run_batch_inference(ctx, session, model_name, req).await
}

async fn run_batch_inference(
    ctx: Arc<ServeContext>,
    session: Arc<ModelSession>,
    model_name: String,
    req: BatchInferRequest,
) -> Result<Json<BatchInferResponse>, ApiError> {
    if req.requests.len() > MAX_BATCH_SIZE {
        tracing::warn!(
            model = %model_name,
            count = req.requests.len(),
            max = MAX_BATCH_SIZE,
            "batch size exceeds limit"
        );
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("batch size exceeds {MAX_BATCH_SIZE}"),
        ));
    }

    let batch_start = Instant::now();
    let deadline = batch_start + ctx.inference_timeout;

    // Items run concurrently, bounded by the model's instance count and the
    // global compute semaphore; each waits for a free instance until the
    // shared batch deadline.
    let mut tasks = tokio::task::JoinSet::new();
    for (index, single_req) in req.requests.into_iter().enumerate() {
        let ctx = ctx.clone();
        let session = session.clone();
        let model_name = model_name.clone();
        tasks.spawn(async move {
            let result = match parse_http_inputs(&single_req.inputs) {
                Ok(inputs) => serving::execute(
                    &ctx,
                    &session,
                    &model_name,
                    inputs,
                    deadline,
                    Admission::Wait,
                )
                .await
                .map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            };
            match result {
                Ok(out) => BatchInferItem {
                    index,
                    outputs: Some(to_output_responses(out)),
                    error: None,
                },
                Err(e) => BatchInferItem {
                    index,
                    outputs: None,
                    error: Some(e),
                },
            }
        });
    }

    let mut responses = Vec::with_capacity(tasks.len());
    while let Some(item) = tasks.join_next().await {
        match item {
            Ok(item) => responses.push(item),
            Err(e) => tracing::error!(error = %e, "batch item task failed"),
        }
    }
    responses.sort_by_key(|r| r.index);

    tracing::info!(
        model = %model_name,
        batch = responses.len(),
        total_ms = batch_start.elapsed().as_secs_f64() * 1000.0,
        "batch inference completed"
    );

    Ok(Json(BatchInferResponse {
        id: req.id,
        model_name,
        model_version: session.version.to_string(),
        responses,
    }))
}

async fn run_inference(
    ctx: Arc<ServeContext>,
    session: Arc<ModelSession>,
    model_name: String,
    req: InferRequest,
) -> Result<Json<InferResponse>, ApiError> {
    let request_start = Instant::now();
    let deadline = request_start + ctx.inference_timeout;

    let inputs = parse_http_inputs(&req.inputs).map_err(|e| {
        tracing::warn!(model = %model_name, error = %e, "bad request");
        crate::metrics::record_request(&model_name, "400");
        ApiError::new(StatusCode::BAD_REQUEST, e.to_string())
    })?;

    let outputs = serving::execute(
        &ctx,
        &session,
        &model_name,
        inputs,
        deadline,
        Admission::Reject,
    )
    .await?;

    tracing::info!(
        model = %model_name,
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "inference completed"
    );

    Ok(Json(InferResponse {
        id: req.id,
        model_name,
        model_version: session.version.to_string(),
        outputs: to_output_responses(outputs),
    }))
}

fn parse_http_inputs(inputs: &[InferInputRequest]) -> anyhow::Result<Vec<(String, InputTensor)>> {
    if inputs.len() > MAX_INPUTS {
        anyhow::bail!("too many inputs: {} (max {MAX_INPUTS})", inputs.len());
    }
    let mut result = Vec::with_capacity(inputs.len());

    for inp in inputs {
        let (shape, total) = serving::validate_shape(&inp.shape)
            .map_err(|e| anyhow::anyhow!("input '{}': {e}", inp.name))?;
        let name = inp.name.as_str();

        let tensor = match inp.datatype.as_str() {
            "FP32" | "FLOAT32" => {
                let data = flatten_numbers(&inp.data, total, name, |n| {
                    let v = n.as_f64().filter(|v| v.is_finite())?;
                    let f = v as f32;
                    f.is_finite().then_some(f)
                })?;
                InputTensor::F32(data, shape)
            }
            "INT32" => {
                let data = flatten_numbers(&inp.data, total, name, |n| {
                    as_integer(n).and_then(|v| i32::try_from(v).ok())
                })?;
                InputTensor::I32(data, shape)
            }
            "INT64" => {
                let data = flatten_numbers(&inp.data, total, name, as_integer)?;
                InputTensor::I64(data, shape)
            }
            "BYTES" | "STRING" => {
                let mut strings = Vec::with_capacity(total.min(1 << 16));
                collect_strings(&inp.data, 0, &mut strings, name)?;
                if strings.len() != total {
                    anyhow::bail!(
                        "input '{name}': {} strings don't match shape product {total}",
                        strings.len()
                    );
                }
                InputTensor::String(strings, shape)
            }
            other => anyhow::bail!("input '{name}': unsupported datatype {other}"),
        };

        result.push((inp.name.clone(), tensor));
    }

    Ok(result)
}

/// Integer value of a JSON number; floats are accepted only when integral.
/// Uses `as_i64` first so INT64 values beyond 2^53 keep full precision.
fn as_integer(n: &serde_json::Number) -> Option<i64> {
    n.as_i64().or_else(|| {
        let f = n.as_f64()?;
        (f.fract() == 0.0 && f >= i64::MIN as f64 && f < i64::MAX as f64).then_some(f as i64)
    })
}

/// Flattens a (possibly nested) JSON number array straight into `Vec<T>`,
/// rejecting non-numeric or out-of-range elements.
fn flatten_numbers<T>(
    value: &serde_json::Value,
    expected: usize,
    name: &str,
    convert: impl Fn(&serde_json::Number) -> Option<T>,
) -> anyhow::Result<Vec<T>> {
    fn walk<T>(
        v: &serde_json::Value,
        depth: usize,
        out: &mut Vec<T>,
        limit: usize,
        name: &str,
        convert: &impl Fn(&serde_json::Number) -> Option<T>,
    ) -> anyhow::Result<()> {
        match v {
            serde_json::Value::Number(n) => {
                if out.len() >= limit {
                    anyhow::bail!("input '{name}': more data than shape product {limit}");
                }
                let value = convert(n)
                    .ok_or_else(|| anyhow::anyhow!("input '{name}': invalid value {n}"))?;
                out.push(value);
                Ok(())
            }
            serde_json::Value::Array(items) => {
                if depth >= MAX_JSON_DEPTH {
                    anyhow::bail!("input '{name}': data nested deeper than {MAX_JSON_DEPTH}");
                }
                for item in items {
                    walk(item, depth + 1, out, limit, name, convert)?;
                }
                Ok(())
            }
            other => anyhow::bail!("input '{name}': expected number, got {other}"),
        }
    }

    if !value.is_array() {
        anyhow::bail!("input '{name}': expected array for tensor data");
    }
    let mut out = Vec::with_capacity(expected);
    walk(value, 0, &mut out, expected, name, &convert)?;
    if out.len() != expected {
        anyhow::bail!(
            "input '{name}': data length {} doesn't match shape product {expected}",
            out.len()
        );
    }
    Ok(out)
}

fn collect_strings(
    v: &serde_json::Value,
    depth: usize,
    out: &mut Vec<String>,
    name: &str,
) -> anyhow::Result<()> {
    match v {
        serde_json::Value::String(s) => {
            out.push(s.clone());
            Ok(())
        }
        serde_json::Value::Array(items) if depth < MAX_JSON_DEPTH => {
            for item in items {
                collect_strings(item, depth + 1, out, name)?;
            }
            Ok(())
        }
        serde_json::Value::Array(_) => {
            anyhow::bail!("input '{name}': data nested deeper than {MAX_JSON_DEPTH}")
        }
        other => anyhow::bail!("input '{name}': expected string, got {other}"),
    }
}

#[derive(Deserialize)]
struct VersionRequest {
    #[serde(default)]
    version: Option<u32>,
}

fn require_model_control(ctx: &ServeContext) -> Result<(), ApiError> {
    if ctx.explicit_model_control {
        Ok(())
    } else {
        Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "model control API requires --model-control-mode=explicit",
        ))
    }
}

async fn load_model(
    State(ctx): Ctx,
    AxumPath(model_name): AxumPath<String>,
    body: Option<Json<VersionRequest>>,
) -> Result<StatusCode, ApiError> {
    require_model_control(&ctx)?;
    if !is_valid_model_name(&model_name) {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "invalid model name"));
    }
    let version = body.and_then(|b| b.version);
    ctx.repo
        .load_one(model_name.clone(), version)
        .await
        .map_err(|e| {
            tracing::error!(model = %model_name, error = %e, "load failed");
            ApiError::new(StatusCode::BAD_REQUEST, e.to_string())
        })?;
    Ok(StatusCode::OK)
}

async fn unload_model(
    State(ctx): Ctx,
    AxumPath(model_name): AxumPath<String>,
    body: Option<Json<VersionRequest>>,
) -> Result<StatusCode, ApiError> {
    require_model_control(&ctx)?;
    let versions = match body.and_then(|b| b.version) {
        Some(v) => vec![v],
        None => ctx.pool().get_versions(&model_name),
    };
    if versions.is_empty() {
        return Err(not_found(&model_name));
    }
    for v in versions {
        ctx.pool()
            .unload_model(&model_name, v)
            .map_err(|e| ApiError::new(StatusCode::NOT_FOUND, e.to_string()))?;
        crate::metrics::clear_model(&model_name, v);
    }
    crate::metrics::set_models_count(ctx.pool().model_count() as i64);
    Ok(StatusCode::OK)
}

#[derive(Serialize)]
struct RepoModelEntry {
    name: String,
    state: &'static str,
}

async fn repository_index(State(ctx): Ctx) -> Json<Vec<RepoModelEntry>> {
    Json(
        ctx.pool()
            .all_model_names()
            .into_iter()
            .map(|name| RepoModelEntry {
                name,
                state: "READY",
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(datatype: &str, shape: Vec<i64>, data: serde_json::Value) -> InferInputRequest {
        InferInputRequest {
            name: "x".to_string(),
            shape,
            datatype: datatype.to_string(),
            data,
        }
    }

    #[test]
    fn parses_nested_fp32() {
        let inputs = vec![input(
            "FP32",
            vec![2, 2],
            serde_json::json!([[1, 2.5], [3, 4]]),
        )];
        let parsed = parse_http_inputs(&inputs).unwrap();
        match &parsed[0].1 {
            InputTensor::F32(d, s) => {
                assert_eq!(d, &vec![1.0, 2.5, 3.0, 4.0]);
                assert_eq!(s, &vec![2, 2]);
            }
            _ => panic!("expected FP32"),
        }
    }

    #[test]
    fn int64_keeps_precision() {
        let big = 9_007_199_254_740_993i64; // 2^53 + 1
        let inputs = vec![input("INT64", vec![1], serde_json::json!([big]))];
        match &parse_http_inputs(&inputs).unwrap()[0].1 {
            InputTensor::I64(d, _) => assert_eq!(d[0], big),
            _ => panic!("expected INT64"),
        }
    }

    #[test]
    fn rejects_bad_data() {
        let cases = vec![
            input("FP32", vec![2], serde_json::json!([1.0])),
            input("FP32", vec![1], serde_json::json!([1.0, 2.0])),
            input("FP32", vec![1], serde_json::json!(["a"])),
            input("FP32", vec![1], serde_json::json!([1e300])),
            input("INT32", vec![1], serde_json::json!([1.5])),
            input("INT32", vec![1], serde_json::json!([3_000_000_000i64])),
            input("BYTES", vec![2], serde_json::json!(["a"])),
            input("BYTES", vec![1], serde_json::json!([1])),
            input("FP32", vec![-1], serde_json::json!([1.0])),
            input("FP16", vec![1], serde_json::json!([1.0])),
            input("FP32", vec![1], serde_json::json!([[[[[[[[[[1.0]]]]]]]]]])),
        ];
        for c in cases {
            assert!(parse_http_inputs(&[c]).is_err());
        }
    }

    #[test]
    fn output_serialization_is_typed() {
        let out = to_output_responses(vec![(
            "y".to_string(),
            vec![2],
            TensorData::I64(vec![1, 2]),
        )]);
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(
            json,
            serde_json::json!([{"name": "y", "shape": [2], "datatype": "INT64", "data": [1, 2]}])
        );
    }
}
