use std::sync::OnceLock;
use std::time::Instant;

use axum::routing::get;
use axum::Router;
use prometheus::{
    Encoder, GaugeVec, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Registry,
    TextEncoder,
};
use tokio::sync::watch;

static REGISTRY: OnceLock<Registry> = OnceLock::new();
static REQUESTS_TOTAL: OnceLock<IntCounterVec> = OnceLock::new();
static ACTIVE_MODELS: OnceLock<IntGauge> = OnceLock::new();
static MODEL_INFO: OnceLock<IntGaugeVec> = OnceLock::new();
static INFER_LATENCY: OnceLock<HistogramVec> = OnceLock::new();
static INFLIGHT_REQUESTS: OnceLock<IntGaugeVec> = OnceLock::new();
static QUEUE_WAIT_LATENCY: OnceLock<HistogramVec> = OnceLock::new();
static MODEL_LOAD_DURATION: OnceLock<HistogramVec> = OnceLock::new();
static MODEL_LOAD_ERRORS: OnceLock<IntCounterVec> = OnceLock::new();
static UPTIME_SECONDS: OnceLock<GaugeVec> = OnceLock::new();
static CIRCUIT_BREAKER_TRIPS: OnceLock<IntCounter> = OnceLock::new();

static START_TIME: OnceLock<Instant> = OnceLock::new();

fn registry() -> &'static Registry {
    REGISTRY.get_or_init(|| {
        let r = Registry::new();

        // Request counter with model + HTTP status labels
        let c = IntCounterVec::new(
            prometheus::opts!(
                "axon_requests_total",
                "total inference requests by model and status"
            ),
            &["model", "status"],
        )
        .expect("valid descriptor");
        r.register(Box::new(c.clone())).expect("unique name");
        REQUESTS_TOTAL.set(c).ok();

        // Number of loaded models
        let g = IntGauge::new("axon_models_loaded", "number of currently loaded models")
            .expect("valid descriptor");
        r.register(Box::new(g.clone())).expect("unique name");
        ACTIVE_MODELS.set(g).ok();

        // Per-model info gauge (for Grafana model inventory)
        let m = IntGaugeVec::new(
            prometheus::opts!(
                "axon_model_info",
                "loaded model versions (1=ready, 0=unloaded)"
            ),
            &["model", "version"],
        )
        .expect("valid descriptor");
        r.register(Box::new(m.clone())).expect("unique name");
        MODEL_INFO.set(m).ok();

        // Inference latency histogram
        let h = HistogramVec::new(
            prometheus::histogram_opts!(
                "axon_inference_duration_seconds",
                "inference duration in seconds",
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            &["model"],
        )
        .expect("valid descriptor");
        r.register(Box::new(h.clone())).expect("unique name");
        INFER_LATENCY.set(h).ok();

        // In-flight requests per model
        let inf = IntGaugeVec::new(
            prometheus::opts!(
                "axon_inflight_requests",
                "number of currently processing requests per model"
            ),
            &["model"],
        )
        .expect("valid descriptor");
        r.register(Box::new(inf.clone())).expect("unique name");
        INFLIGHT_REQUESTS.set(inf).ok();

        // Queue/semaphore wait time
        let qw = HistogramVec::new(
            prometheus::histogram_opts!(
                "axon_queue_wait_seconds",
                "time spent waiting for concurrency permit",
                vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            ),
            &["model"],
        )
        .expect("valid descriptor");
        r.register(Box::new(qw.clone())).expect("unique name");
        QUEUE_WAIT_LATENCY.set(qw).ok();

        // Model load duration
        let mld = HistogramVec::new(
            prometheus::histogram_opts!(
                "axon_model_load_duration_seconds",
                "time to load a model from disk",
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
            ),
            &["model"],
        )
        .expect("valid descriptor");
        r.register(Box::new(mld.clone())).expect("unique name");
        MODEL_LOAD_DURATION.set(mld).ok();

        // Model load errors
        let mle = IntCounterVec::new(
            prometheus::opts!(
                "axon_model_load_errors_total",
                "number of model load failures"
            ),
            &["model"],
        )
        .expect("valid descriptor");
        r.register(Box::new(mle.clone())).expect("unique name");
        MODEL_LOAD_ERRORS.set(mle).ok();

        // Server uptime
        let up = GaugeVec::new(
            prometheus::opts!("axon_server_info", "server metadata"),
            &["version"],
        )
        .expect("valid descriptor");
        r.register(Box::new(up.clone())).expect("unique name");
        UPTIME_SECONDS.set(up).ok();

        // Circuit breaker trips
        let cbt = IntCounter::new(
            "axon_circuit_breaker_trips_total",
            "number of times circuit breaker opened for a model",
        )
        .expect("valid descriptor");
        r.register(Box::new(cbt.clone())).expect("unique name");
        CIRCUIT_BREAKER_TRIPS.set(cbt).ok();

        r
    })
}

pub fn init() {
    registry();
    START_TIME.get_or_init(Instant::now);
    if let Some(up) = UPTIME_SECONDS.get() {
        up.with_label_values(&["0.3.0"]).set(1.0);
    }
}

// ── Request metrics ───────────────────────────────────────

pub fn record_request(model: &str, status: &str) {
    if let Some(c) = REQUESTS_TOTAL.get() {
        c.with_label_values(&[model, status]).inc();
    }
}

#[allow(dead_code)]
pub fn inc_requests() {
    record_request("", "200");
}

pub fn record_latency(model: &str, ms: f64) {
    if let Some(h) = INFER_LATENCY.get() {
        h.with_label_values(&[model]).observe(ms / 1000.0);
    }
}

pub fn inc_inflight(model: &str) {
    if let Some(g) = INFLIGHT_REQUESTS.get() {
        g.with_label_values(&[model]).inc();
    }
}

pub fn dec_inflight(model: &str) {
    if let Some(g) = INFLIGHT_REQUESTS.get() {
        g.with_label_values(&[model]).dec();
    }
}

pub fn record_queue_wait(model: &str, wait_secs: f64) {
    if let Some(h) = QUEUE_WAIT_LATENCY.get() {
        h.with_label_values(&[model]).observe(wait_secs);
    }
}

// ── Model lifecycle metrics ───────────────────────────────

pub fn set_models_count(n: i64) {
    if let Some(g) = ACTIVE_MODELS.get() {
        g.set(n);
    }
}

pub fn set_model_ready(model: &str, version: u32) {
    if let Some(m) = MODEL_INFO.get() {
        m.with_label_values(&[model, &version.to_string()]).set(1);
    }
}

#[allow(dead_code)]
pub fn clear_model(model: &str, version: u32) {
    if let Some(m) = MODEL_INFO.get() {
        m.with_label_values(&[model, &version.to_string()]).set(0);
    }
}

pub fn record_model_load_duration(model: &str, secs: f64) {
    if let Some(h) = MODEL_LOAD_DURATION.get() {
        h.with_label_values(&[model]).observe(secs);
    }
}

pub fn record_model_load_error(model: &str) {
    if let Some(c) = MODEL_LOAD_ERRORS.get() {
        c.with_label_values(&[model]).inc();
    }
}

pub fn record_circuit_breaker_trip() {
    if let Some(c) = CIRCUIT_BREAKER_TRIPS.get() {
        c.inc();
    }
}

// ── Metrics endpoint ──────────────────────────────────────

fn metrics_text() -> String {
    let r = registry();
    let mut buf = Vec::new();
    let encoder = TextEncoder::new();
    if encoder.encode(&r.gather(), &mut buf).is_err() {
        return String::from("# error encoding metrics\n");
    }
    String::from_utf8(buf).unwrap_or_else(|_| String::from("# utf8 error\n"))
}

async fn metrics_handler() -> String {
    metrics_text()
}

pub async fn serve_metrics(port: u16, mut shutdown: watch::Receiver<bool>) {
    let app = Router::new().route("/metrics", get(metrics_handler));

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("failed to bind metrics port");

    tracing::info!(port, "metrics server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown.wait_for(|v| *v).await;
        })
        .await
        .ok();
}
