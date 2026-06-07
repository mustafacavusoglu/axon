mod arena;
mod config;
mod grpc_server;
mod http_server;
mod metrics;
mod model_repository;
mod session;

use std::time::Duration;

use clap::Parser;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tokio::signal;
use tokio::sync::watch;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use config::ServerConfig;
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

    let tracer = provider.tracer("axon-server");
    let _ = opentelemetry::global::set_tracer_provider(provider.clone());

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with(tracing_subscriber::fmt::layer().json().with_current_span(true))
        .with(otel_layer)
        .init();

    Some(provider)
}

fn main() -> anyhow::Result<()> {
    let config = ServerConfig::parse();

    let provider = init_tracing();
    if provider.is_none() {
        tracing_subscriber::registry()
            .with(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .with(tracing_subscriber::fmt::layer().json().with_current_span(true))
            .init();
    }

    let worker_threads = if config.num_threads > 0 {
        config.num_threads
    } else {
        num_cpus::get_physical()
    };

    let pool = SessionPool::new(worker_threads)?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tracing::info!(
            model_repository = %config.model_repository.display(),
            http_port = config.http_port,
            grpc_port = config.grpc_port,
            metrics_port = config.metrics_port,
            model_control_mode = %config.model_control_mode,
            threads = worker_threads,
            "starting axon-server"
        );

        metrics::init();
        model_repository::load_all_models(&config.model_repository, &pool).await;
        metrics::set_models_count(pool.model_count() as i64);

        let poll_handle = if config.model_control_mode == "poll" {
            let poll_pool = pool.clone();
            let poll_repo = config.model_repository.clone();
            let poll_interval = config.repository_poll_secs;
            let poll_shutdown = shutdown_rx.clone();
            Some(tokio::spawn(async move {
                model_repository::poll_loop(poll_repo, poll_pool, poll_interval, poll_shutdown)
                    .await;
            }))
        } else {
            None
        };

        let http_handle = {
            let http_pool = pool.clone();
            let http_repo = config.model_repository.clone();
            let rx = shutdown_rx.clone();
            tokio::spawn(http_server::serve(
                config.http_port,
                http_pool,
                http_repo,
                rx,
            ))
        };

        let grpc_handle = {
            let grpc_pool = pool.clone();
            let grpc_repo = config.model_repository.clone();
            let rx = shutdown_rx.clone();
            tokio::spawn(grpc_server::serve(
                config.grpc_port,
                grpc_pool,
                grpc_repo,
                rx,
            ))
        };

        let metrics_handle = {
            let rx = shutdown_rx.clone();
            tokio::spawn(metrics::serve_metrics(config.metrics_port, rx))
        };

        signal::ctrl_c().await.ok();
        tracing::info!("shutdown signal received, draining...");
        let _ = shutdown_tx.send(true);

        if let Some(h) = poll_handle {
            let _ = h.await;
        }
        let _ = tokio::time::timeout(Duration::from_secs(30), http_handle).await;
        let _ = tokio::time::timeout(Duration::from_secs(30), grpc_handle).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), metrics_handle).await;

        tracing::info!("axon-server stopped");
    });

    if let Some(p) = provider {
        let _ = p.shutdown();
    }

    Ok(())
}
