mod handlers;
mod state;

pub use state::AppState;

use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

pub fn router(state: AppState) -> Router {
    let state = Arc::new(state);
    Router::new()
        .route("/health", get(handlers::health))
        .route("/initialize", post(handlers::initialize))
        .route("/prefetch", post(handlers::prefetch))
        .route("/sync_turn", post(handlers::sync_turn))
        .route("/on_session_end", post(handlers::on_session_end))
        .route("/tool_schemas", get(handlers::tool_schemas))
        .route("/handle_tool_call", post(handlers::handle_tool_call))
        .route("/export", post(handlers::export_memories))
        .route("/import", post(handlers::import_memories))
        .route("/delete", post(handlers::delete_memory))
        .route("/compact", post(handlers::compact))
        .route("/snapshot", get(handlers::snapshot))
        .route("/link", post(handlers::memory_link))
        .route("/unlink", post(handlers::memory_unlink))
        .route("/graph", post(handlers::memory_graph))
        .route("/memories/list", post(handlers::list_memories))
        .route("/memories/get", post(handlers::get_memory))
        .route("/memories/update", post(handlers::update_memory))
        .route("/memories/feedback", post(handlers::add_feedback))
        .route("/backup", post(handlers::backup))
        .route("/restore", post(handlers::restore))
        .route("/profile/status", post(handlers::profile_status))
        .route("/session/conclude", post(handlers::conclude))
        .route("/openapi.json", get(handlers::openapi))
        .route("/reminders/due", get(handlers::reminders_due))
        .route("/remind", post(handlers::remind))
        .route("/deep_search", post(handlers::deep_search))
        .route("/memory_stats", get(handlers::memory_stats))
        .route("/stats/detailed", post(handlers::detailed_stats))
        .route("/flush", post(handlers::flush))
        .with_state(state)
        .layer(CorsLayer::permissive())
}
