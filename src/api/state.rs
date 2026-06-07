use crate::embeddings::EmbeddingProvider;
use crate::storage::MemoryStore;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct ConversationTurn {
    pub user: String,
    pub assistant: String,
    pub importance: f32,
    pub tags: Value,
    pub metadata: Value,
}

#[derive(Clone)]
pub struct ConversationBuffer {
    pub turns: Vec<ConversationTurn>,
    pub accumulated_embedding: Option<Vec<f32>>,
    pub last_turn_at: DateTime<Utc>,
    pub conversation_id: String,
    pub turn_offset: usize,
    pub session_batch_count: usize,
    pub latest_tags: Value,
    pub latest_metadata: Value,
}

impl ConversationBuffer {
    pub fn new() -> Self {
        Self {
            turns: vec![],
            accumulated_embedding: None,
            last_turn_at: Utc::now(),
            conversation_id: uuid::Uuid::new_v4().to_string(),
            turn_offset: 0,
            session_batch_count: 0,
            latest_tags: serde_json::json!({}),
            latest_metadata: serde_json::json!({}),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn combined_content(&self) -> String {
        self.turns
            .iter()
            .map(|t| format!("user: {}\nassistant: {}", t.user, t.assistant))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn mean_importance(&self) -> f32 {
        if self.turns.is_empty() {
            return 0.0;
        }
        self.turns.iter().map(|t| t.importance).sum::<f32>() / self.turns.len() as f32
    }
}

#[derive(Clone)]
pub struct AppState {
    pub embedder: Arc<dyn EmbeddingProvider>,
    pub store: Arc<dyn MemoryStore>,
    pub hybrid_alpha: f32,
    pub decay_half_life_days: f64,
    pub sync_turn_min_importance: f32,
    pub prefetch_cadence: u64,
    pub sync_turn_cadence: u64,
    pub conclusion_cadence: u64,
    pub context_tokens: Option<usize>,
    pub base_context_cadence: u64,
    pub rerank_enabled: bool,
    pub batch_size: usize,
    pub batch_idle_seconds: u64,
    pub fresh_ttl_hours: u64,
    pub consolidation_cadence: u64,
    pub consolidation_model: String,
    pub openrouter_api_key: String,
    pub prune_deep_importance: f32,
    pub prune_deep_access_count: i32,
    pub prune_deep_age_days: i64,
    pub prune_consolid_insight: f32,
    pub prune_consolid_age_days: i64,
    pub buffers: Arc<Mutex<HashMap<String, ConversationBuffer>>>,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedder: Box<dyn EmbeddingProvider>,
        store: Box<dyn MemoryStore>,
        hybrid_alpha: f32,
        decay_half_life_days: f64,
        sync_turn_min_importance: f32,
        prefetch_cadence: u64,
        sync_turn_cadence: u64,
        conclusion_cadence: u64,
        context_tokens: Option<usize>,
        base_context_cadence: u64,
        rerank_enabled: bool,
        batch_size: usize,
        batch_idle_seconds: u64,
        fresh_ttl_hours: u64,
        consolidation_cadence: u64,
        consolidation_model: String,
        openrouter_api_key: String,
        prune_deep_importance: f32,
        prune_deep_access_count: i32,
        prune_deep_age_days: i64,
        prune_consolid_insight: f32,
        prune_consolid_age_days: i64,
    ) -> Self {
        Self {
            embedder: Arc::from(embedder),
            store: Arc::from(store),
            hybrid_alpha,
            decay_half_life_days,
            sync_turn_min_importance,
            prefetch_cadence,
            sync_turn_cadence,
            conclusion_cadence,
            context_tokens,
            base_context_cadence,
            rerank_enabled,
            batch_size,
            batch_idle_seconds,
            fresh_ttl_hours,
            consolidation_cadence,
            consolidation_model,
            openrouter_api_key,
            prune_deep_importance,
            prune_deep_access_count,
            prune_deep_age_days,
            prune_consolid_insight,
            prune_consolid_age_days,
            buffers: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}
