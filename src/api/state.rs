use crate::config::SessionStrategy;
use crate::embeddings::EmbeddingProvider;
use crate::storage::MemoryStore;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub embedder: Arc<dyn EmbeddingProvider>,
    pub store: Arc<dyn MemoryStore>,
    pub hybrid_alpha: f32,
    pub decay_half_life_days: f64,
    pub sync_turn_min_importance: f32,
    pub session_strategy: SessionStrategy,
    pub prefetch_cadence: u64,
    pub sync_turn_cadence: u64,
    pub conclusion_cadence: u64,
    pub context_tokens: Option<usize>,
    pub base_context_cadence: u64,
    pub rerank_enabled: bool,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedder: Box<dyn EmbeddingProvider>,
        store: Box<dyn MemoryStore>,
        hybrid_alpha: f32,
        decay_half_life_days: f64,
        sync_turn_min_importance: f32,
        session_strategy: SessionStrategy,
        prefetch_cadence: u64,
        sync_turn_cadence: u64,
        conclusion_cadence: u64,
        context_tokens: Option<usize>,
base_context_cadence: u64,
        rerank_enabled: bool,
    ) -> Self {
        Self {
            embedder: Arc::from(embedder),
            store: Arc::from(store),
            hybrid_alpha,
            decay_half_life_days,
            sync_turn_min_importance,
            session_strategy,
            prefetch_cadence,
            sync_turn_cadence,
            conclusion_cadence,
            context_tokens,
            base_context_cadence,
            rerank_enabled,
        }
    }
}