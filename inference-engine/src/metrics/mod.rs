use prometheus::{Encoder, IntCounter, IntGauge, HistogramVec, Registry, TextEncoder};
use std::sync::OnceLock;

static REGISTRY: OnceLock<Registry> = OnceLock::new();
static INFER_REQUESTS: OnceLock<IntCounter> = OnceLock::new();
static ACTIVE_MODELS: OnceLock<IntGauge> = OnceLock::new();
static INFER_LATENCY: OnceLock<HistogramVec> = OnceLock::new();

pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(|| {
        let r = Registry::new();

        let c = IntCounter::new("inference_engine_requests_total", "total inference requests")
            .unwrap();
        r.register(Box::new(c.clone())).unwrap();
        INFER_REQUESTS.set(c).ok();

        let g = IntGauge::new("inference_engine_models_loaded", "number of loaded models")
            .unwrap();
        r.register(Box::new(g.clone())).unwrap();
        ACTIVE_MODELS.set(g).ok();

        let h = HistogramVec::new(
            prometheus::histogram_opts!(
                "inference_engine_latency_ms",
                "Rust execution latency in ms",
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]
            ),
            &["model"],
        )
        .unwrap();
        r.register(Box::new(h.clone())).unwrap();
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

pub fn metrics_text() -> String {
    let r = registry();
    let mut buf = Vec::new();
    let encoder = TextEncoder::new();
    encoder.encode(&r.gather(), &mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}
