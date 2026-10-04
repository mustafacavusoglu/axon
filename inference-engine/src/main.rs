mod config;
mod grpc_server;
mod http_server;
mod metrics;
mod model_repository;
mod serving;
mod session;
mod ui;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
#[cfg(not(unix))]
use tokio::signal;
#[cfg(unix)]
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::watch;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

use config::ServerConfig;
use model_repository::Repository;
use session::pool::SessionPool;

struct LogGuards {
    _stdout: tracing_appender::non_blocking::WorkerGuard,
    _file: tracing_appender::non_blocking::WorkerGuard,
}

fn parse_rotation(s: &str) -> (tracing_appender::rolling::Rotation, usize) {
    let s = s.trim();
    let (num_str, unit) = s.split_at(s.len().saturating_sub(1));
    let count = num_str.parse::<usize>().unwrap_or(7);
    let rotation = match unit {
        "h" => tracing_appender::rolling::Rotation::HOURLY,
        "d" => tracing_appender::rolling::Rotation::DAILY,
        _ => tracing_appender::rolling::Rotation::DAILY,
    };
    (rotation, count)
}

fn init_tracing(config: &ServerConfig) -> (Option<SdkTracerProvider>, LogGuards) {
    let log_level: tracing::Level = config.log_level.parse().unwrap_or(tracing::Level::INFO);
    let (rotation, max_files) = parse_rotation(&config.log_rotation);

    std::fs::create_dir_all(&config.log_dir).ok();
    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(rotation)
        .filename_prefix("axon-server")
        .filename_suffix("json")
        .max_log_files(max_files)
        .build(&config.log_dir)
        .expect("failed to create log appender");
    let (file_writer, file_guard) = tracing_appender::non_blocking(file_appender);
    let (stdout_writer, stdout_guard) = tracing_appender::non_blocking(std::io::stdout());

    let console_filter = Targets::new().with_target("axon::console", tracing::Level::INFO);

    let file_filter = Targets::new()
        .with_default(log_level)
        .with_target("hyper", tracing::Level::WARN)
        .with_target("tower", tracing::Level::WARN)
        .with_target("tonic", tracing::Level::WARN)
        .with_target("h2", tracing::Level::WARN);

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(stdout_writer)
        .with_target(false)
        .with_level(false)
        .compact()
        .with_filter(console_filter);

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(file_writer)
        .json()
        .with_current_span(true)
        .with_filter(file_filter);

    let otel_provider = init_otel();
    if let Some(ref provider) = otel_provider {
        let tracer = provider.tracer("axon-server");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        tracing_subscriber::registry()
            .with(stdout_layer)
            .with(file_layer)
            .with(otel_layer)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(stdout_layer)
            .with(file_layer)
            .init();
    }

    let guards = LogGuards {
        _stdout: stdout_guard,
        _file: file_guard,
    };
    (otel_provider, guards)
}

fn init_otel() -> Option<SdkTracerProvider> {
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

    opentelemetry::global::set_tracer_provider(provider.clone());
    Some(provider)
}

fn print_startup_table(config: &ServerConfig, pool: &SessionPool) {
    let version = env!("CARGO_PKG_VERSION");
    let models = pool.list_models();
    let width = 62;

    println!();
    println!("  ╔{}╗", "═".repeat(width));
    println!(
        "  ║{:^width$}║",
        format!("axon-server v{version}"),
        width = width
    );
    println!("  ╠{}╣", "═".repeat(width));
    println!("  ║{:<width$}║", "  Endpoints", width = width);
    println!(
        "  ║{:<width$}║",
        format!("    HTTP     http://{}:{}", config.host, config.http_port),
        width = width
    );
    println!(
        "  ║{:<width$}║",
        format!("    gRPC     {}:{}", config.host, config.grpc_port),
        width = width
    );
    println!(
        "  ║{:<width$}║",
        format!(
            "    Metrics  http://{}:{}/metrics",
            config.host, config.metrics_port
        ),
        width = width
    );
    println!(
        "  ║{:<width$}║",
        format!("    Logs     {}", config.log_dir.display()),
        width = width
    );
    println!(
        "  ║{:<width$}║",
        format!("    Device   {:?}", config.device),
        width = width
    );
    println!("  ╠{}╣", "═".repeat(width));
    println!("  ║{:<width$}║", "  Models", width = width);
    println!(
        "  ║{:<width$}║",
        "    Name                     Ver  Platform     Device",
        width = width
    );
    println!(
        "  ║{:<width$}║",
        "    ─────────────────────────────────────────────────────",
        width = width
    );

    if models.is_empty() {
        println!("  ║{:<width$}║", "    (no models loaded)", width = width);
    } else {
        let mut sorted = models;
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        for (name, version, _state) in &sorted {
            let session = pool.get(name, *version);
            let platform = session
                .as_ref()
                .map(|s| s.runner.platform_name())
                .unwrap_or("unknown");
            let device = session
                .as_ref()
                .map(|s| s.runner.device_label())
                .unwrap_or_default();
            let line = format!("    {name:<25} {version:<4} {platform:<12} {device}");
            println!("  ║{line:<width$}║");
        }
    }

    println!("  ╚{}╝", "═".repeat(width));
    println!();
}

fn main() -> anyhow::Result<()> {
    let config = ServerConfig::parse();
    let (provider, _guards) = init_tracing(&config);

    let inference_threads = if config.num_threads > 0 {
        config.num_threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    };

    let tokio_threads = (inference_threads / 2).clamp(2, 8);
    let blocking_threads = inference_threads.max(4);
    let pool = SessionPool::new(inference_threads)?;
    let exec = config.execution_config();
    tracing::info!(
        target: "axon::console",
        device = ?exec.default_device,
        gpus = ?exec.gpu_device_ids,
        "execution device configured"
    );

    let repo = Repository {
        path: config.model_repository.clone(),
        pool: pool.clone(),
        default_concurrency: config.concurrency_per_model,
        exec: Arc::new(exec),
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(tokio_threads)
        .max_blocking_threads(blocking_threads)
        // Rhai BLS scripts and nested ensemble/BLS calls recurse on the
        // blocking threads; the 2 MiB default is too tight.
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        metrics::init();

        let bind = |port: u16| -> anyhow::Result<SocketAddr> {
            format!("{}:{port}", config.host)
                .parse::<SocketAddr>()
                .map_err(|e| anyhow::anyhow!("invalid bind address {}:{port}: {e}", config.host))
        };
        let http_listener = tokio::net::TcpListener::bind(bind(config.http_port)?).await?;
        let grpc_listener = tokio::net::TcpListener::bind(bind(config.grpc_port)?).await?;
        let metrics_listener = tokio::net::TcpListener::bind(bind(config.metrics_port)?).await?;

        let ctx = Arc::new(serving::ServeContext {
            repo: repo.clone(),
            inference_timeout: Duration::from_millis(config.inference_timeout_ms),
            api_key: config.api_key().map(Arc::from),
            explicit_model_control: config.model_control_mode == "explicit",
            ui_enabled: config.ui,
        });
        if ctx.ui_enabled {
            tracing::info!(
                target: "axon::console",
                "web UI enabled at http://{}:{}/ui",
                config.host,
                config.http_port
            );
        }
        if ctx.api_key.is_none() {
            tracing::warn!(
                target: "axon::console",
                "no --api-key set: inference and model-control endpoints are unauthenticated"
            );
        }

        let http_handle = tokio::spawn(http_server::serve(
            http_listener,
            ctx.clone(),
            shutdown_rx.clone(),
        ));
        let grpc_handle = tokio::spawn(grpc_server::serve(
            grpc_listener,
            ctx.clone(),
            shutdown_rx.clone(),
        ));
        let metrics_handle = tokio::spawn(metrics::serve_metrics(
            metrics_listener,
            shutdown_rx.clone(),
        ));

        let poll_handle = if config.model_control_mode == "poll" {
            Some(tokio::spawn(model_repository::poll_loop(
                repo.clone(),
                config.repository_poll_secs,
                shutdown_rx.clone(),
            )))
        } else {
            None
        };

        tracing::info!(
            target: "axon::console",
            "loading models from {}",
            config.model_repository.display()
        );

        let load_repo = repo.clone();
        let load_config = config.clone();
        tokio::spawn(async move {
            load_repo.load_all(false).await;
            print_startup_table(&load_config, &load_repo.pool);
        });

        #[cfg(unix)]
        {
            let mut sigterm =
                signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
            let mut sigint =
                signal(SignalKind::interrupt()).expect("failed to install SIGINT handler");
            tokio::select! {
                _ = sigterm.recv() => {
                    tracing::info!(target: "axon::console", "SIGTERM received, draining...");
                }
                _ = sigint.recv() => {
                    tracing::info!(target: "axon::console", "SIGINT received, draining...");
                }
            }
        }
        #[cfg(not(unix))]
        {
            signal::ctrl_c().await.ok();
            tracing::info!(target: "axon::console", "shutdown signal received, draining...");
        }
        let _ = shutdown_tx.send(true);

        if let Some(h) = poll_handle {
            let _ = h.await;
        }
        let _ = tokio::time::timeout(Duration::from_secs(30), http_handle).await;
        let _ = tokio::time::timeout(Duration::from_secs(30), grpc_handle).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), metrics_handle).await;

        tracing::info!(target: "axon::console", "axon-server stopped");
        anyhow::Ok(())
    })?;

    if let Some(p) = provider {
        let _ = p.shutdown();
    }

    Ok(())
}
