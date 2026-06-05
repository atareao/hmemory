pub mod pgvector;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;

#[derive(Clone)]
pub struct MemoryRecord {
    pub id: i64,
    pub session_id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub tags: Value,
    pub metadata: Value,
    pub importance: f32,
    pub source: String,
    pub category: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub feedback_positive: i32,
    pub feedback_negative: i32,
    pub score: f64,
}

#[derive(Clone)]
pub enum TagMatchMode {
    All,
    Any,
}

#[derive(Clone)]
pub struct SearchFilters {
    pub query: String,
    pub tag_filter: Option<Value>,
    pub tag_match_mode: TagMatchMode,
    pub category_filter: Option<String>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub min_importance: Option<f32>,
    pub alpha: f32,
    pub limit: usize,
    pub session_id: Option<String>,
    pub decay_half_life_days: f64,
    pub level: RetrievalLevel,
}

#[derive(Clone, Copy)]
pub enum RetrievalLevel {
    Summary,
    Overview,
    Details,
}

#[derive(Clone)]
pub struct SessionStatus {
    pub session_id: String,
    pub strategy: String,
    pub path_or_repo: String,
    pub turn_count: i64,
    pub is_cold: bool,
    pub created_at: DateTime<Utc>,
    pub last_active_at: DateTime<Utc>,
}

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    async fn store(
        &self,
        session_id: &str,
        content: &str,
        embedding: &[f32],
        tags: &Value,
        metadata: &Value,
        importance: f32,
        source: &str,
        category: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn search(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn delete_session(
        &self,
        session_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn list_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn delete_by_id(&self, id: i64) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn list_memories(
        &self,
        session_id: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<MemoryRecord>, u64), Box<dyn std::error::Error + Send + Sync>>;
    async fn get_by_id(
        &self,
        id: i64,
    ) -> Result<Option<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    async fn update_memory(
        &self,
        id: i64,
        content: Option<String>,
        tags: Option<Value>,
        metadata: Option<Value>,
        importance: Option<f32>,
        category: Option<String>,
        source: Option<String>,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>;
    async fn add_feedback(
        &self,
        id: i64,
        useful: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn backup(
        &self,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn restore(
        &self,
        records: &[MemoryRecord],
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
    async fn resolve_session(
        &self,
        session_id: &str,
        strategy: &str,
        path_or_repo: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_session_status(
        &self,
        session_id: &str,
    ) -> Result<SessionStatus, Box<dyn std::error::Error + Send + Sync>>;
    async fn increment_turn_count(
        &self,
        session_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn get_base_context(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn create_conclusion(
        &self,
        session_id: &str,
        content: &str,
        category: &str,
        confidence: f32,
        source_turn_ids: &[i64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}