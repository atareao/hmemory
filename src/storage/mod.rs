pub mod pgvector;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: i64,
    pub profile: String,
    pub content: String,
    #[serde(skip)]
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
    pub expires_at: Option<DateTime<Utc>>,
    pub immortal: bool,
    pub score: f64,
    pub access_count: i32,
    pub last_accessed_at: Option<DateTime<Utc>>,
    pub trust_score: f32,
    pub event_date: Option<DateTime<Utc>>,
    pub reminder_interval: Option<String>,
    pub reminder_at: Option<DateTime<Utc>>,
    pub reminder_sent: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryRecordFresh {
    pub id: i64,
    pub profile: String,
    pub content: String,
    #[serde(skip)]
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
    pub expires_at: Option<DateTime<Utc>>,
    pub immortal: bool,
    pub score: f64,
    pub access_count: i32,
    pub last_accessed_at: Option<DateTime<Utc>>,
    pub trust_score: f32,
    pub event_date: Option<DateTime<Utc>>,
    pub reminder_interval: Option<String>,
    pub reminder_at: Option<DateTime<Utc>>,
    pub reminder_sent: bool,
    pub conversation_id: String,
    pub turn_range: String,
    pub user_msg: String,
    pub assistant_msg: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryRecordConsolid {
    pub id: i64,
    pub profile: String,
    pub summary: String,
    #[serde(skip)]
    pub embedding: Vec<f32>,
    pub source_ids: Vec<i64>,
    pub depth: String,
    pub tags: Value,
    pub metadata: Value,
    pub insight_score: f32,
    pub importance: f32,
    pub created_at: DateTime<Utc>,
    pub last_consolidated_at: DateTime<Utc>,
    pub access_count: i32,
    pub score: f64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    pub id_a: i64,
    pub id_b: i64,
    pub relation_type: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub memory: MemoryRecord,
    pub edges: Vec<GraphEdge>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub target_id: i64,
    pub relation_type: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AssociativeResult {
    pub seed: MemoryRecord,
    pub related: Vec<MemoryRecord>,
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
    pub profile: Option<String>,
    pub decay_half_life_days: f64,
    pub level: RetrievalLevel,
    pub include_deep: bool,
}

#[derive(Clone, Copy)]
pub enum RetrievalLevel {
    Summary,
    Overview,
    Details,
}

#[derive(Clone)]
pub struct ProfileStatus {
    pub profile: String,
    pub turn_count: i64,
    pub is_cold: bool,
    pub created_at: DateTime<Utc>,
    pub last_active_at: DateTime<Utc>,
}

#[derive(Clone, Serialize)]
pub struct MemoryStats {
    pub fresh_count: u64,
    pub deep_count: u64,
    pub consolid_count: u64,
    pub per_profile: Vec<ProfileTierCounts>,
}

#[derive(Clone, Serialize)]
pub struct ProfileTierCounts {
    pub profile: String,
    pub fresh: u64,
    pub deep: u64,
    pub consolid: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DetailedStatsRequest {
    pub profile: Option<String>,
}

#[derive(Clone, Serialize)]
pub struct DetailedMemoryStats {
    pub tier_counts: TierCounts,
    pub total_memories: u64,
    pub total_profiles: u64,
    pub total_links: u64,
    pub importance: NumericStats,
    pub trust_score: NumericStats,
    pub access_count: NumericStats,
    pub content_length: NumericStats,
    pub oldest_memory: Option<DateTime<Utc>>,
    pub newest_memory: Option<DateTime<Utc>>,
    pub category_distribution: Vec<LabelCount>,
    pub top_tags: Vec<LabelCount>,
    pub source_distribution: Vec<LabelCount>,
    pub feedback_positive: i64,
    pub feedback_negative: i64,
    pub immortal_count: u64,
    pub mortal_count: u64,
    pub expired_count: u64,
    pub consolid_depth_distribution: Vec<LabelCount>,
    pub reminders_active: u64,
    pub reminders_sent: u64,
    pub per_profile: Vec<ProfileDetailedStats>,
}

#[derive(Clone, Serialize)]
pub struct TierCounts {
    pub fresh: u64,
    pub deep: u64,
    pub consolid: u64,
}

#[derive(Clone, Serialize)]
pub struct NumericStats {
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    pub median: f64,
    pub p95: f64,
    pub count: u64,
}

#[derive(Clone, Serialize)]
pub struct LabelCount {
    pub label: String,
    pub count: u64,
}

#[derive(Clone, Serialize)]
pub struct ProfileDetailedStats {
    pub profile: String,
    pub tier_counts: TierCounts,
    pub importance_avg: f64,
    pub top_category: Option<String>,
    pub feedback_positive: i64,
    pub feedback_negative: i64,
    pub memory_count: u64,
}

#[derive(Clone, Serialize)]
pub struct CompactReport {
    pub merged_pairs: Vec<(i64, i64, f64)>,
    pub archived_ids: Vec<i64>,
    pub decayed_ids: Vec<i64>,
}

#[derive(Clone, Serialize)]
pub struct SnapshotDiff {
    pub added: Vec<MemoryRecord>,
    pub removed: Vec<MemoryRecord>,
    pub changed: Vec<(MemoryRecord, MemoryRecord)>,
}

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    async fn store(
        &self,
        profile: &str,
        content: &str,
        embedding: &[f32],
        tags: &Value,
        metadata: &Value,
        importance: f32,
        source: &str,
        category: &str,
        expires_at: Option<DateTime<Utc>>,
        immortal: bool,
        event_date: Option<DateTime<Utc>>,
        reminder_interval: Option<&str>,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>>;
    async fn find_duplicate(
        &self,
        embedding: &[f32],
        threshold: f32,
    ) -> Result<Option<(i64, f32)>, Box<dyn std::error::Error + Send + Sync>>;
    async fn search(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn track_access(
        &self,
        ids: &[i64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn compact(
        &self,
        dry_run: bool,
        similarity_threshold: f32,
        importance_threshold: f32,
    ) -> Result<CompactReport, Box<dyn std::error::Error + Send + Sync>>;
    async fn snapshot(&self)
    -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn diff_snapshots(
        &self,
        snapshot_a: &[MemoryRecord],
        snapshot_b: &[MemoryRecord],
    ) -> Result<SnapshotDiff, Box<dyn std::error::Error + Send + Sync>>;
    async fn rollback_to(
        &self,
        snapshot: &[MemoryRecord],
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
    async fn delete_by_profile(
        &self,
        profile: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn list_by_profile(
        &self,
        profile: &str,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn delete_by_id(&self, id: i64) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn list_memories(
        &self,
        profile: Option<&str>,
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
    async fn backup(&self) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn restore(
        &self,
        records: &[MemoryRecord],
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_profile_status(
        &self,
        profile: &str,
    ) -> Result<ProfileStatus, Box<dyn std::error::Error + Send + Sync>>;
    async fn increment_turn_count(
        &self,
        profile: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn get_base_context(
        &self,
        profile: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn create_conclusion(
        &self,
        profile: &str,
        content: &str,
        category: &str,
        confidence: f32,
        source_turn_ids: &[i64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn link_memories(
        &self,
        id_a: i64,
        id_b: i64,
        relation_type: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn unlink_memories(
        &self,
        id_a: i64,
        id_b: i64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn get_relations(
        &self,
        id: i64,
    ) -> Result<Vec<MemoryLink>, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_memory_graph(
        &self,
        id: i64,
        depth: u32,
    ) -> Result<Vec<GraphNode>, Box<dyn std::error::Error + Send + Sync>>;
    async fn associative_search(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
        depth: u32,
    ) -> Result<Vec<AssociativeResult>, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_due_reminders(
        &self,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;
    async fn mark_reminder_sent(
        &self,
        id: i64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn set_reminder(
        &self,
        id: i64,
        event_date: Option<DateTime<Utc>>,
        reminder_interval: Option<String>,
        reminder_at: Option<DateTime<Utc>>,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>;

    // === 3-table storage extensions ===

    async fn store_deep(
        &self,
        profile: &str,
        content: &str,
        embedding: &[f32],
        tags: &Value,
        metadata: &Value,
        importance: f32,
        source: &str,
        category: &str,
        expires_at: Option<DateTime<Utc>>,
        event_date: Option<DateTime<Utc>>,
        reminder_interval: Option<&str>,
        conversation_id: &str,
        turn_range: &str,
        user_msg: &str,
        assistant_msg: &str,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>>;

    async fn store_fresh(
        &self,
        profile: &str,
        content: &str,
        embedding: &[f32],
        tags: &Value,
        metadata: &Value,
        importance: f32,
        source: &str,
        category: &str,
        expires_at: Option<DateTime<Utc>>,
        event_date: Option<DateTime<Utc>>,
        reminder_interval: Option<&str>,
        conversation_id: &str,
        turn_range: &str,
        user_msg: &str,
        assistant_msg: &str,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>>;

    async fn search_fresh(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;

    async fn search_consolid(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;

    async fn search_deep(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;

    async fn store_consolid(
        &self,
        profile: &str,
        summary: &str,
        embedding: &[f32],
        source_ids: &[i64],
        tags: &Value,
        metadata: &Value,
        importance: f32,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>>;

    async fn prune_fresh(
        &self,
        ttl_hours: u64,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;

    async fn get_new_deep_since(
        &self,
        since: DateTime<Utc>,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;

    async fn get_new_fresh_since(
        &self,
        profile: &str,
        since: DateTime<Utc>,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>>;

    async fn prune_deep(
        &self,
        min_importance: f32,
        min_access_count: i32,
        max_age_days: i64,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;

    async fn prune_consolid(
        &self,
        min_insight_score: f32,
        max_age_days: i64,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;

    async fn schedule_review(
        &self,
        id: i64,
        importance: f32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    async fn memory_stats(&self) -> Result<MemoryStats, Box<dyn std::error::Error + Send + Sync>>;

    async fn detailed_stats(
        &self,
        profile: Option<&str>,
    ) -> Result<DetailedMemoryStats, Box<dyn std::error::Error + Send + Sync>>;

    async fn get_consolid_by_depth_since(
        &self,
        depth: &str,
        since: DateTime<Utc>,
    ) -> Result<Vec<MemoryRecordConsolid>, Box<dyn std::error::Error + Send + Sync>>;
}
