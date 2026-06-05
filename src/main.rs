#![allow(dead_code)]

mod api;
mod config;
mod embeddings;
mod storage;

use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    dotenvy::dotenv().ok();
    let config = config::Config::from_env();

    let embedder = embeddings::from_config(&config);
    let store = Box::new(
        storage::pgvector::PgVectorStore::new(&config.database_url)
            .await
            .expect("failed to connect to postgres"),
    );

    let state = api::AppState::new(
        embedder,
        store,
        config.hybrid_alpha,
        config.decay_half_life_days,
        config.sync_turn_min_importance,
        config.session_strategy,
        config.prefetch_cadence,
        config.sync_turn_cadence,
        config.conclusion_cadence,
        config.context_tokens,
        config.base_context_cadence,
        config.rerank_enabled,
    );
    let app = api::router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
