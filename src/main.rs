#![allow(dead_code)]

mod api;
mod config;
mod consolidation_worker;
mod embeddings;
mod reminder_worker;
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
    let store: Box<dyn storage::MemoryStore> = Box::new(
        storage::pgvector::PgVectorStore::new(&config.database_url)
            .await
            .expect("failed to connect to postgres"),
    );
    store.initialize().await.expect("failed to run migrations");

    let openrouter_key = config.openrouter_api_key.clone().unwrap_or_default();
    let state = api::AppState::new(
        embedder,
        store,
        config.hybrid_alpha,
        config.decay_half_life_days,
        config.sync_turn_min_importance,
        config.prefetch_cadence,
        config.sync_turn_cadence,
        config.conclusion_cadence,
        config.context_tokens,
        config.base_context_cadence,
        config.rerank_enabled,
        config.batch_size,
        config.batch_idle_seconds,
        config.fresh_ttl_hours,
        config.consolidation_cadence,
        config.consolidation_model,
        openrouter_key.clone(),
        config.prune_deep_importance,
        config.prune_deep_access_count,
        config.prune_deep_age_days,
        config.prune_consolid_insight,
        config.prune_consolid_age_days,
    );
    reminder_worker::start(state.clone());
    if config.consolidation_cadence > 0 && config.openrouter_api_key.is_some() {
        consolidation_worker::start(state.clone());
    }

    let app = api::router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
