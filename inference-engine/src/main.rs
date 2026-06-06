mod server;
mod session;
mod arena;
mod config;
mod metrics;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use config::EngineConfig;
use session::pool::SessionPool;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer()
            .json()
            .with_current_span(true))
        .init();

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

    Ok(())
}
