mod server;
mod session;
mod arena;
mod config;
mod metrics;

use std::time::Duration;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

use config::EngineConfig;
use session::pool::SessionPool;

fn init_tracing() -> Option<SdkTracerProvider> {
    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .or_else(|_| std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
        .ok()?;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(3))
        .build()
        .ok()?;

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();

    let tracer = provider.tracer("inference-engine");
    let _ = opentelemetry::global::set_tracer_provider(provider.clone());

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().json().with_current_span(true))
        .with(otel_layer)
        .init();

    Some(provider)
}

fn main() -> anyhow::Result<()> {
    let provider = init_tracing();

    if provider.is_none() {
        tracing_subscriber::registry()
            .with(EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")))
            .with(tracing_subscriber::fmt::layer().json().with_current_span(true))
            .init();
    }

    let config = EngineConfig::from_env()?;
    let pool = SessionPool::new(config.num_threads)?;
    let arena = arena::TensorArena::new(config.arena_size_mb * 1024 * 1024);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.num_threads / 2)
        .enable_all()
        .build()?;

    let socket = config.socket_path.clone();
    tracing::info!(
        socket = %socket.display(),
        threads = config.num_threads,
        arena_mb = config.arena_size_mb,
        timeout_ms = config.inference_timeout_ms,
        "starting inference engine"
    );

    rt.block_on(server::serve(socket, config, pool, arena))?;

    if let Some(p) = provider {
        let _ = p.shutdown();
    }

    Ok(())
}
