use std::sync::OnceLock;

use axum::routing::get;
use axum::Router;
use prometheus::{Encoder, HistogramVec, IntCounter, IntGauge, Registry, TextEncoder};
use tokio::sync::watch;

static REGISTRY: OnceLock<Registry> = OnceLock::new();
static INFER_REQUESTS: OnceLock<IntCounter> = OnceLock::new();
static ACTIVE_MODELS: OnceLock<IntGauge> = OnceLock::new();
static INFER_LATENCY: OnceLock<HistogramVec> = OnceLock::new();

fn registry() -> &'static Registry {
    REGISTRY.get_or_init(|| {
        let r = Registry::new();

        let c = IntCounter::new("axon_requests_total", "total inference requests")
            .expect("metric creation should not fail for a valid descriptor");
        r.register(Box::new(c.clone()))
            .expect("metric registration should not fail for unique names");
        INFER_REQUESTS.set(c).ok();

        let g = IntGauge::new("axon_models_loaded", "number of loaded models")
            .expect("metric creation should not fail for a valid descriptor");
        r.register(Box::new(g.clone()))
            .expect("metric registration should not fail for unique names");
        ACTIVE_MODELS.set(g).ok();

        let h = HistogramVec::new(
            prometheus::histogram_opts!(
                "axon_inference_latency_ms",
                "inference latency in ms",
                vec![0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0]
            ),
            &["model"],
        )
        .expect("metric creation should not fail for a valid descriptor");
        r.register(Box::new(h.clone()))
            .expect("metric registration should not fail for unique names");
        INFER_LATENCY.set(h).ok();

        r
    })
}

pub fn inc_requests() {
    if let Some(c) = INFER_REQUESTS.get() {
        c.inc();
    }
}

pub fn set_models_count(n: i64) {
    if let Some(g) = ACTIVE_MODELS.get() {
        g.set(n);
    }
}

pub fn record_latency(model: &str, ms: f64) {
    if let Some(h) = INFER_LATENCY.get() {
        h.with_label_values(&[model]).observe(ms);
    }
}

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

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
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
