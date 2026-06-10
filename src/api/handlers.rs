use axum::{Json, extract::State};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

use super::AppState;
use crate::storage::{
    DetailedStatsRequest, MemoryRecord, RetrievalLevel, SearchFilters, TagMatchMode,
};

fn compute_importance(text: &str) -> f32 {
    let mut score = 0.2f32;
    if text.len() > 100 {
        score += 0.1;
    }
    if text.len() > 300 {
        score += 0.1;
    }
    if text.contains("```") {
        score += 0.15;
    }
    let num_count = text.chars().filter(|c| c.is_ascii_digit()).count();
    if num_count > 5 {
        score += 0.1;
    }
    let keywords = [
        "project",
        "config",
        "build",
        "deploy",
        "api",
        "error",
        "fix",
        "implement",
        "database",
        "server",
        "function",
        "feature",
        "bug",
        "version",
        "v0.",
        "release",
        "merge",
        "commit",
        "pr",
        "todo",
        "refactor",
    ];
    for kw in &keywords {
        if text.to_lowercase().contains(kw) {
            score += 0.05;
        }
    }
    let lower = text.to_lowercase().trim().to_string();
    let greetings = [
        "hello", "hi", "hey", "thanks", "ok", "sure", "yeah", "yep", "nope", "no", "yes",
        "goodbye", "bye", "👍", "done",
    ];
    for g in &greetings {
        if lower == *g || lower.starts_with(&format!("{g} ")) || lower.ends_with(&format!(" {g}")) {
            score -= 0.15;
            break;
        }
    }
    score.clamp(0.0, 1.0)
}

fn parse_tag_match_mode(s: Option<&str>) -> TagMatchMode {
    match s {
        Some("any") => TagMatchMode::Any,
        _ => TagMatchMode::All,
    }
}

fn parse_level(s: Option<&str>) -> RetrievalLevel {
    match s {
        Some("summary") => RetrievalLevel::Summary,
        Some("overview") => RetrievalLevel::Overview,
        _ => RetrievalLevel::Details,
    }
}

fn parse_datetime(s: Option<&str>) -> Option<DateTime<Utc>> {
    s.and_then(|v| {
        DateTime::parse_from_rfc3339(v)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    })
}

// ── Health ──────────────────────────────────────────────────

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

// ── Initialize ──────────────────────────────────────────────

#[derive(Deserialize)]
pub struct InitializeRequest {
    pub profile: String,
}

pub async fn initialize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InitializeRequest>,
) -> Json<Value> {
    match state.store.initialize().await {
        Ok(()) => Json(json!({ "ok": true, "profile": req.profile })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Prefetch (search) ───────────────────────────────────────

#[derive(Deserialize)]
pub struct PrefetchRequest {
    pub query: String,
    pub profile: Option<String>,
    pub limit: Option<usize>,
    pub tag_filter: Option<Value>,
    pub tag_match_mode: Option<String>,
    pub category_filter: Option<String>,
    pub created_after: Option<String>,
    pub created_before: Option<String>,
    pub min_importance: Option<f32>,
    pub level: Option<String>,
    pub include_deep: Option<bool>,
}

#[derive(Serialize)]
pub struct PrefetchMemory {
    pub id: i64,
    pub content: String,
    pub tags: Value,
    pub source: String,
    pub category: String,
    pub created_at: DateTime<Utc>,
    pub score: f64,
}

pub async fn prefetch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PrefetchRequest>,
) -> Json<Value> {
    let limit = req.limit.unwrap_or(5);
    let embedding = match state.embedder.embed(&req.query).await {
        Ok(v) => v,
        Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
    };
    let filters = SearchFilters {
        query: req.query.clone(),
        tag_filter: req.tag_filter.clone(),
        tag_match_mode: parse_tag_match_mode(req.tag_match_mode.as_deref()),
        category_filter: req.category_filter.clone(),
        created_after: parse_datetime(req.created_after.as_deref()),
        created_before: parse_datetime(req.created_before.as_deref()),
        min_importance: req.min_importance,
        alpha: state.hybrid_alpha,
        limit,
        profile: req.profile.clone(),
        decay_half_life_days: state.decay_half_life_days,
        level: parse_level(req.level.as_deref()),
        include_deep: req.include_deep.unwrap_or(false),
    };
    match state.store.search(&embedding, &filters).await {
        Ok(records) => {
            let ids: Vec<i64> = records.iter().map(|r| r.id).collect();
            if !ids.is_empty() {
                let _ = state.store.track_access(&ids).await;
            }
            let memories: Vec<PrefetchMemory> = records
                .into_iter()
                .map(|r| PrefetchMemory {
                    id: r.id,
                    content: r.content,
                    tags: r.tags,
                    source: r.source,
                    category: r.category,
                    created_at: r.created_at,
                    score: r.score,
                })
                .collect();
            Json(json!({ "ok": true, "memories": memories }))
        }
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Deep Search (explicit memories_deep) ────────────────────

#[derive(Deserialize)]
pub struct DeepSearchRequest {
    pub query: String,
    pub profile: Option<String>,
    pub limit: Option<usize>,
    pub min_importance: Option<f32>,
    pub tags: Option<Value>,
    pub tag_match_mode: Option<String>,
    pub alpha: Option<f32>,
}

pub async fn deep_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeepSearchRequest>,
) -> Json<Value> {
    let limit = req.limit.unwrap_or(10);
    let embedding = match state.embedder.embed(&req.query).await {
        Ok(v) => v,
        Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
    };
    let filters = SearchFilters {
        query: req.query.clone(),
        tag_filter: req.tags.clone(),
        tag_match_mode: parse_tag_match_mode(req.tag_match_mode.as_deref()),
        category_filter: None,
        created_after: None,
        created_before: None,
        min_importance: req.min_importance,
        alpha: req.alpha.unwrap_or(state.hybrid_alpha),
        limit,
        profile: req.profile.clone(),
        decay_half_life_days: state.decay_half_life_days,
        level: RetrievalLevel::Details,
        include_deep: true,
    };
    match state.store.search_deep(&embedding, &filters).await {
        Ok(records) => {
            let results: Vec<Value> = records
                .into_iter()
                .map(|r| {
                    json!({
                        "id": r.id, "content": r.content, "tags": r.tags,
                        "source": r.source, "category": r.category,
                        "created_at": r.created_at, "score": r.score,
                        "importance": r.importance, "immortal": r.immortal
                    })
                })
                .collect();
            Json(json!({ "ok": true, "results": results }))
        }
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    dot / (na.sqrt() * nb.sqrt()).max(1e-10)
}

async fn flush_buffer(
    state: &Arc<AppState>,
    profile: &str,
    buffer: &crate::api::state::ConversationBuffer,
    is_last: bool,
) {
    if buffer.is_empty() {
        return;
    }
    let content = buffer.combined_content();
    let importance = buffer.mean_importance();
    let importance = if buffer.session_batch_count == 0 {
        importance * 1.10
    } else {
        importance
    };
    let importance = if is_last {
        importance * 1.10
    } else {
        importance
    };
    let batch_emb = buffer
        .accumulated_embedding
        .clone()
        .unwrap_or_else(|| vec![0.0f32; 4]);
    let cid = &buffer.conversation_id;
    let turn_count = buffer.turns.len();
    let turn_range = format!(
        "{}-{}",
        buffer.turn_offset + 1,
        buffer.turn_offset + turn_count
    );
    let tags = &buffer.latest_tags;
    let metadata = &buffer.latest_metadata;
    let user_msg = &buffer.turns[0].user;
    let assistant_msg = &buffer
        .turns
        .last()
        .map(|t| &t.assistant)
        .cloned()
        .unwrap_or_default();

    if let Ok(id) = state
        .store
        .store_fresh(
            profile,
            &content,
            &batch_emb,
            tags,
            metadata,
            importance,
            "hermes",
            "general",
            None,
            None,
            None,
            cid,
            &turn_range,
            user_msg,
            assistant_msg,
        )
        .await
    {
        let _ = state.store.schedule_review(id, importance).await;
    }

    if let Ok(id) = state
        .store
        .store_deep(
            profile,
            &content,
            &batch_emb,
            tags,
            metadata,
            importance,
            "hermes",
            "general",
            None,
            None,
            None,
            cid,
            &turn_range,
            user_msg,
            assistant_msg,
        )
        .await
    {
        let _ = state.store.schedule_review(id, importance).await;
    }
}

// ── Sync Turn (buffered) ─────────────────────────────────────

#[derive(Deserialize)]
pub struct SyncTurnRequest {
    pub user: String,
    pub assistant: String,
    pub profile: String,
    pub tags: Option<Value>,
    pub metadata: Option<Value>,
    pub importance: Option<f32>,
    pub source: Option<String>,
    pub category: Option<String>,
}

pub async fn sync_turn(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SyncTurnRequest>,
) -> Json<Value> {
    let tags = req.tags.unwrap_or(json!({}));
    let metadata = req.metadata.unwrap_or(json!({}));
    let _importance = req.importance.unwrap_or(0.0);
    let _source = req.source.unwrap_or_else(|| "hermes".to_string());
    let _category = req.category.unwrap_or_else(|| "general".to_string());
    let combined = format!("{}\n{}", req.user, req.assistant);
    let computed = compute_importance(&combined);
    if computed < state.sync_turn_min_importance {
        return Json(json!({ "ok": true, "stored": 0, "skipped": true, "importance": computed }));
    }

    let combined_embedding = match state.embedder.embed(&combined).await {
        Ok(v) => v,
        Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
    };

    let profile = &req.profile;
    let mut buffers = state.buffers.lock().await;
    let buffer = buffers
        .entry(profile.clone())
        .or_insert_with(crate::api::state::ConversationBuffer::new);

    if !buffer.is_empty() {
        let elapsed = (Utc::now() - buffer.last_turn_at).num_seconds() as u64;
        if elapsed >= state.batch_idle_seconds {
            let batch_count = buffer.session_batch_count;
            flush_buffer(&state, profile, buffer, false).await;
            *buffer = crate::api::state::ConversationBuffer::new();
            buffer.session_batch_count = batch_count + 1;
            buffer.conversation_id = uuid::Uuid::new_v4().to_string();
        }
    }

    if !buffer.is_empty() {
        if let Some(ref acc) = buffer.accumulated_embedding {
            let sim = cosine_similarity(&combined_embedding, acc);
            if sim < 0.65 && buffer.turns.len() >= 2 {
                let batch_count = buffer.session_batch_count;
                flush_buffer(&state, profile, buffer, false).await;
                *buffer = crate::api::state::ConversationBuffer::new();
                buffer.session_batch_count = batch_count + 1;
                buffer.conversation_id = uuid::Uuid::new_v4().to_string();
            }
        }
    }

    buffer.turns.push(crate::api::state::ConversationTurn {
        user: req.user,
        assistant: req.assistant,
        importance: computed,
        tags: tags.clone(),
        metadata: metadata.clone(),
    });
    buffer.last_turn_at = Utc::now();
    buffer.latest_tags = tags;
    buffer.latest_metadata = metadata;

    match &mut buffer.accumulated_embedding {
        None => buffer.accumulated_embedding = Some(combined_embedding),
        Some(avg) => {
            let n = buffer.turns.len() as f32;
            for (a, b) in avg.iter_mut().zip(&combined_embedding) {
                *a += (b - *a) / n;
            }
        }
    }

    if buffer.turns.len() >= state.batch_size {
        let batch_count = buffer.session_batch_count;
        flush_buffer(&state, profile, buffer, false).await;
        *buffer = crate::api::state::ConversationBuffer::new();
        buffer.session_batch_count = batch_count + 1;
    }

    if let Err(e) = state.store.increment_turn_count(profile).await {
        tracing::warn!("failed to increment turn count: {e}");
    }

    Json(json!({ "ok": true, "buffered": buffer.turns.len(), "importance": computed }))
}

// ── On Session End ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct SessionEndRequest {
    pub profile: String,
}

pub async fn on_session_end(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SessionEndRequest>,
) -> Json<Value> {
    let mut buffers = state.buffers.lock().await;
    if let Some(buffer) = buffers.get(&req.profile) {
        if !buffer.is_empty() {
            flush_buffer(&state, &req.profile, buffer, true).await;
        }
    }
    buffers.remove(&req.profile);
    Json(json!({ "ok": true }))
}

// ── Tool Schemas ────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

fn memory_search_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_search".into(),
        description: "Search memories by semantic similarity".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 5},
                "tag_filter": {"type": "object", "description": "Filter by tags"},
                "tag_match_mode": {"type": "string", "enum": ["all", "any"], "default": "all"},
                "category_filter": {"type": "string", "description": "Filter by category"},
                "profile": {"type": "string", "description": "Profile to search in (omit for current profile, empty string for all profiles)", "optional": true},
                "include_deep": {"type": "boolean", "description": "Also search deep historical memory (false = fresh + consolidated only)", "optional": true}
            },
            "required": ["query"]
        }),
    }
}

fn memory_add_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_add".into(),
        description: "Add a new memory".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content"},
                "tags": {"type": "object", "description": "Tags"},
                "importance": {"type": "number", "description": "Importance score 0-1"},
                "source": {"type": "string", "description": "Source identifier"},
                "category": {"type": "string", "description": "Category"},
                "event_date": {"type": "string", "description": "RFC3339 datetime for scheduling / event date", "optional": true},
                "reminder": {"type": "string", "description": "Relative interval like '30m','1h','2d' or absolute RFC3339 datetime", "optional": true},
                "reminder_in": {"type": "string", "description": "From-now interval like '30m','2h','1d' (e.g. '2h' = reminder in 2 hours)", "optional": true}
            },
            "required": ["content"]
        }),
    }
}

fn memory_list_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_list".into(),
        description: "List memories with pagination".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results", "default": 20},
                "offset": {"type": "integer", "description": "Offset", "default": 0},
                "profile": {"type": "string", "description": "Profile to list (omit for current profile, empty string for all profiles)", "optional": true}
            },
            "required": []
        }),
    }
}

fn memory_get_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_get".into(),
        description: "Get a memory by ID".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Memory ID"}
            },
            "required": ["id"]
        }),
    }
}

fn memory_update_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_update".into(),
        description: "Update a memory".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Memory ID"},
                "content": {"type": "string", "description": "New content"},
                "tags": {"type": "object", "description": "New tags"},
                "importance": {"type": "number", "description": "New importance score"},
                "category": {"type": "string", "description": "New category"},
                "source": {"type": "string", "description": "New source"}
            },
            "required": ["id"]
        }),
    }
}

fn memory_export_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_export".into(),
        description: "Export memories (optionally filtered by profile)".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "profile": {"type": "string", "description": "Profile to export (omit for all profiles)", "optional": true}
            },
            "required": []
        }),
    }
}

fn memory_import_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_import".into(),
        description: "Import memories".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "tags": {"type": "object"},
                            "importance": {"type": "number"},
                            "source": {"type": "string"},
                            "category": {"type": "string"}
                        },
                        "required": ["content"]
                    }
                }
            },
            "required": ["memories"]
        }),
    }
}

fn memory_delete_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_delete".into(),
        description: "Delete a memory by ID".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Memory ID"}
            },
            "required": ["id"]
        }),
    }
}

fn memory_backup_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_backup".into(),
        description: "Backup all memories".into(),
        parameters: json!({
            "type": "object",
            "properties": {},
            "required": []
        }),
    }
}

fn memory_restore_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_restore".into(),
        description: "Restore memories from backup".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "profile": {"type": "string"},
                            "content": {"type": "string"},
                            "embedding": {"type": "array", "items": {"type": "number"}},
                            "tags": {"type": "object"},
                            "metadata": {"type": "object"},
                            "importance": {"type": "number"},
                            "source": {"type": "string"},
                            "category": {"type": "string"},
                            "created_at": {"type": "string"},
                            "updated_at": {"type": "string"},
                            "feedback_positive": {"type": "integer"},
                            "feedback_negative": {"type": "integer"}
                        },
                        "required": ["profile", "content"]
                    }
                }
            },
            "required": ["memories"]
        }),
    }
}

fn memory_compact_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_compact".into(),
        description: "Compact and prune memories: merge similar pairs, archive low-importance ones. Use dry_run=true to preview.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "description": "Preview changes without applying", "default": true},
                "similarity_threshold": {"type": "number", "description": "Cosine similarity threshold for merging (default 0.9)", "default": 0.9},
                "importance_threshold": {"type": "number", "description": "Archive memories below this importance (default 0.1)", "default": 0.1}
            },
            "required": []
        }),
    }
}

fn memory_snapshot_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_snapshot".into(),
        description: "Create a full snapshot of all memories as JSON".into(),
        parameters: json!({
            "type": "object",
            "properties": {},
            "required": []
        }),
    }
}

fn memory_diff_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_diff".into(),
        description: "Diff two memory snapshots to see what changed".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "snapshot_a": {"type": "array", "items": {"type": "object"}, "description": "First snapshot"},
                "snapshot_b": {"type": "array", "items": {"type": "object"}, "description": "Second snapshot"}
            },
            "required": ["snapshot_a", "snapshot_b"]
        }),
    }
}

fn memory_rollback_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_rollback".into(),
        description: "Rollback database to match a given snapshot".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "snapshot": {"type": "array", "items": {"type": "object"}, "description": "Target snapshot to restore to"}
            },
            "required": ["snapshot"]
        }),
    }
}

fn memory_link_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_link".into(),
        description: "Create a link between two memories".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id_a": {"type": "integer", "description": "First memory ID"},
                "id_b": {"type": "integer", "description": "Second memory ID"},
                "relation_type": {"type": "string", "enum": ["extends", "contradicts", "supersedes", "related"], "description": "Type of relation"}
            },
            "required": ["id_a", "id_b", "relation_type"]
        }),
    }
}

fn memory_unlink_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_unlink".into(),
        description: "Remove a link between two memories".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id_a": {"type": "integer", "description": "First memory ID"},
                "id_b": {"type": "integer", "description": "Second memory ID"}
            },
            "required": ["id_a", "id_b"]
        }),
    }
}

fn memory_graph_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_graph".into(),
        description: "Get the subgraph of connected memories".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Memory ID"},
                "depth": {"type": "integer", "description": "Max traversal depth", "default": 2}
            },
            "required": ["id"]
        }),
    }
}

fn memory_associative_search_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_associative_search".into(),
        description: "Associative search: find top-3 seeds then re-search with each as query"
            .into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "depth": {"type": "integer", "description": "Results per seed", "default": 2},
                "limit": {"type": "integer", "description": "Seed results", "default": 3},
                "category_filter": {"type": "string", "description": "Filter by category"},
                "profile": {"type": "string", "description": "Profile to search in (omit for all profiles)", "optional": true}
            },
            "required": ["query"]
        }),
    }
}

fn memory_reminders_due_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_reminders_due".into(),
        description: "Get all due reminders (reminder_at <= now and not yet sent)".into(),
        parameters: json!({
            "type": "object",
            "properties": {},
            "required": []
        }),
    }
}

fn memory_feedback_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_feedback".into(),
        description: "Mark a memory as helpful or unhelpful".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Memory ID"},
                "useful": {"type": "boolean", "description": "True=helpful, False=unhelpful"}
            },
            "required": ["id", "useful"]
        }),
    }
}

fn memory_remind_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_remind".into(),
        description: "Set a reminder for a memory".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer", "description": "Memory ID"},
                "event_date": {"type": "string", "description": "RFC3339 event date (optional, for relative intervals)", "optional": true},
                "reminder_interval": {"type": "string", "description": "Relative interval like '30m','1h','2d' or absolute RFC3339 datetime", "optional": true},
                "reminder_at": {"type": "string", "description": "Absolute RFC3339 datetime for the reminder to fire", "optional": true},
                "reminder_in": {"type": "string", "description": "From-now interval like '30m','2h','1d' (e.g. '2h' = reminder in 2 hours)", "optional": true}
            },
            "required": ["memory_id"]
        }),
    }
}

fn memory_deep_search_schema() -> ToolSchema {
    ToolSchema {
        name: "memory_deep_search".into(),
        description: "Search only deep historical memory (memories_deep table)".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
                "profile": {"type": "string", "description": "Profile to search", "optional": true},
                "min_importance": {"type": "number", "description": "Minimum importance 0-1", "optional": true},
                "tags": {"type": "object", "description": "Tag filter", "optional": true},
                "tag_match_mode": {"type": "string", "enum": ["all", "any"], "default": "all", "optional": true},
                "alpha": {"type": "number", "description": "Hybrid search weight 0-1 (0=BM25, 1=vector)", "optional": true}
            },
            "required": ["query"]
        }),
    }
}

pub async fn tool_schemas() -> Json<Vec<ToolSchema>> {
    Json(vec![
        memory_search_schema(),
        memory_add_schema(),
        memory_list_schema(),
        memory_get_schema(),
        memory_update_schema(),
        memory_export_schema(),
        memory_import_schema(),
        memory_delete_schema(),
        memory_backup_schema(),
        memory_restore_schema(),
        memory_compact_schema(),
        memory_snapshot_schema(),
        memory_diff_schema(),
        memory_rollback_schema(),
        memory_link_schema(),
        memory_unlink_schema(),
        memory_graph_schema(),
        memory_associative_search_schema(),
        memory_deep_search_schema(),
        memory_reminders_due_schema(),
        memory_feedback_schema(),
        memory_remind_schema(),
    ])
}

// ── Handle Tool Call ────────────────────────────────────────

#[derive(Deserialize)]
pub struct ToolCallRequest {
    pub tool_name: String,
    pub args: Value,
    pub profile: String,
}

pub async fn handle_tool_call(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ToolCallRequest>,
) -> Json<Value> {
    // Normalize: accept both "hmemory_X" and "memory_X" from different Hermes plugin versions
    let tool_name = req
        .tool_name
        .strip_prefix("hmemory_")
        .unwrap_or(&req.tool_name);
    match tool_name {
        "memory_search" => {
            let query = req
                .args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let limit = req.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
            let tag_filter = req.args.get("tag_filter").cloned();
            let tag_match_mode =
                parse_tag_match_mode(req.args.get("tag_match_mode").and_then(|v| v.as_str()));
            let category_filter = req
                .args
                .get("category_filter")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let embedding = match state.embedder.embed(&query).await {
                Ok(v) => v,
                Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
            };
            let level = parse_level(req.args.get("level").and_then(|v| v.as_str()));
            let search_profile: Option<String> =
                match req.args.get("profile").and_then(|v| v.as_str()) {
                    Some(p) if !p.is_empty() => Some(p.to_string()),
                    _ => None,
                };
            let filters = SearchFilters {
                query: query.clone(),
                tag_filter,
                tag_match_mode,
                category_filter,
                created_after: None,
                created_before: None,
                min_importance: None,
                alpha: state.hybrid_alpha,
                limit,
                profile: search_profile,
                decay_half_life_days: state.decay_half_life_days,
                level,
                include_deep: req
                    .args
                    .get("include_deep")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            };
            match state.store.search(&embedding, &filters).await {
                Ok(records) => {
                    let ids: Vec<i64> = records.iter().map(|r| r.id).collect();
                    if !ids.is_empty() {
                        let _ = state.store.track_access(&ids).await;
                    }
                    let results: Vec<Value> = records
                        .into_iter()
                        .map(|r| {
                            json!({
                                "id": r.id, "content": r.content, "tags": r.tags,
                                "source": r.source, "category": r.category,
                                "created_at": r.created_at, "score": r.score
                            })
                        })
                        .collect();
                    Json(json!({ "ok": true, "results": results }))
                }
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_add" => {
            let content = req
                .args
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let tags = req.args.get("tags").cloned().unwrap_or(json!({}));
            let importance = req
                .args
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;
            let source = req
                .args
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("tool")
                .to_string();
            let category = req
                .args
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("general")
                .to_string();
            let metadata = req.args.get("metadata").cloned().unwrap_or(json!({}));
            let event_date = req
                .args
                .get("event_date")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc));
            let reminder = req
                .args
                .get("reminder")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let reminder_in = req
                .args
                .get("reminder_in")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let reminder_final = if let Some(ri) = &reminder_in {
                let (num_str, unit) = ri.split_at(ri.len().saturating_sub(1));
                let num: i64 = num_str.parse().unwrap_or(0);
                let delta = match unit {
                    "m" => chrono::TimeDelta::minutes(num),
                    "h" => chrono::TimeDelta::hours(num),
                    "d" => chrono::TimeDelta::days(num),
                    _ => chrono::TimeDelta::zero(),
                };
                Some((Utc::now() + delta).to_rfc3339())
            } else {
                reminder
            };
            let embedding = match state.embedder.embed(&content).await {
                Ok(v) => v,
                Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
            };
            if let Ok(Some((dup_id, sim))) = state.store.find_duplicate(&embedding, 0.85).await {
                return Json(
                    json!({ "ok": false, "warning": format!("Ya existe (ID {}, {:.0}% match). No duplicado.", dup_id, sim * 100.0), "duplicate_id": dup_id }),
                );
            }
            match state
                .store
                .store(
                    &req.profile,
                    &content,
                    &embedding,
                    &tags,
                    &metadata,
                    importance,
                    &source,
                    &category,
                    None,
                    false,
                    event_date,
                    reminder_final.as_deref(),
                )
                .await
            {
                Ok(_id) => Json(json!({ "ok": true })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_list" => {
            let limit = req.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
            let offset = req.args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let list_profile: Option<&str> = match req.args.get("profile").and_then(|v| v.as_str())
            {
                Some(p) if !p.is_empty() => Some(p),
                _ => None,
            };
            match state.store.list_memories(list_profile, limit, offset).await {
                Ok((records, total)) => {
                    let memories: Vec<Value> = records.into_iter().map(|r| json!({
                        "id": r.id, "profile": r.profile, "content": r.content,
                        "tags": r.tags, "source": r.source, "category": r.category,
                        "importance": r.importance, "created_at": r.created_at, "updated_at": r.updated_at,
                        "feedback_positive": r.feedback_positive, "feedback_negative": r.feedback_negative
                    })).collect();
                    Json(json!({ "ok": true, "memories": memories, "total": total }))
                }
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_get" => {
            let id = match req.args.get("id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id" })),
            };
            match state.store.get_by_id(id).await {
                Ok(Some(record)) => Json(json!({ "ok": true,
                    "id": record.id, "profile": record.profile, "content": record.content,
                    "tags": record.tags, "source": record.source, "category": record.category,
                    "importance": record.importance, "created_at": record.created_at, "updated_at": record.updated_at,
                    "feedback_positive": record.feedback_positive, "feedback_negative": record.feedback_negative,
                    "score": record.score
                })),
                Ok(None) => Json(json!({ "ok": false, "error": "not found" })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_update" => {
            let id = match req.args.get("id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id" })),
            };
            let content = req
                .args
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let tags = req.args.get("tags").cloned();
            let metadata = req.args.get("metadata").cloned();
            let importance = req
                .args
                .get("importance")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32);
            let category = req
                .args
                .get("category")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let source = req
                .args
                .get("source")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            match state
                .store
                .update_memory(id, content, tags, metadata, importance, category, source)
                .await
            {
                Ok(updated) => Json(json!({ "ok": true, "updated": updated })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_backup" => match state.store.backup().await {
            Ok(records) => {
                let memories: Vec<Value> = records.into_iter().map(|r| json!({
                        "id": r.id, "profile": r.profile, "content": r.content,
                        "embedding": r.embedding, "tags": r.tags, "metadata": r.metadata,
                        "importance": r.importance, "source": r.source, "category": r.category,
                        "created_at": r.created_at, "updated_at": r.updated_at,
                        "feedback_positive": r.feedback_positive, "feedback_negative": r.feedback_negative,
                        "score": r.score
                    })).collect();
                Json(json!({ "ok": true, "memories": memories }))
            }
            Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
        },
        "memory_restore" => {
            let memories_arr = match req.args.get("memories").and_then(|v| v.as_array()) {
                Some(arr) => arr,
                None => return Json(json!({ "ok": false, "error": "missing memories array" })),
            };
            let records: Vec<MemoryRecord> = memories_arr
                .iter()
                .filter_map(|v| {
                    let profile = v.get("profile")?.as_str()?.to_string();
                    let content = v.get("content")?.as_str()?.to_string();
                    let embedding = v
                        .get("embedding")?
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|n| n.as_f64())
                                .map(|n| n as f32)
                                .collect()
                        })
                        .unwrap_or_default();
                    let tags = v.get("tags").cloned().unwrap_or(json!({}));
                    let metadata = v.get("metadata").cloned().unwrap_or(json!({}));
                    let importance =
                        v.get("importance").and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
                    let source = v
                        .get("source")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let category = v
                        .get("category")
                        .and_then(|s| s.as_str())
                        .unwrap_or("general")
                        .to_string();
                    let created_at = v
                        .get("created_at")
                        .and_then(|s| s.as_str())
                        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now);
                    let updated_at = v
                        .get("updated_at")
                        .and_then(|s| s.as_str())
                        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now);
                    let feedback_positive = v
                        .get("feedback_positive")
                        .and_then(|n| n.as_i64())
                        .unwrap_or(0) as i32;
                    let feedback_negative = v
                        .get("feedback_negative")
                        .and_then(|n| n.as_i64())
                        .unwrap_or(0) as i32;
                    Some(MemoryRecord {
                        id: 0,
                        profile,
                        content,
                        embedding,
                        tags,
                        metadata,
                        importance,
                        source,
                        category,
                        created_at,
                        updated_at,
                        feedback_positive,
                        feedback_negative,
                        expires_at: None,
                        immortal: false,
                        score: 0.0,
                        access_count: 0,
                        last_accessed_at: None,
                        trust_score: 0.5,
                        event_date: None,
                        reminder_interval: None,
                        reminder_at: None,
                        reminder_sent: false,
                    })
                })
                .collect();
            match state.store.restore(&records).await {
                Ok(count) => Json(json!({ "ok": true, "restored": count })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_export" => {
            let export_profile = req
                .args
                .get("profile")
                .and_then(|v| v.as_str())
                .filter(|p| !p.is_empty());
            let records = if let Some(p) = export_profile {
                match state.store.list_by_profile(p).await {
                    Ok(r) => r,
                    Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
                }
            } else {
                match state.store.list_memories(None, 100000, 0).await {
                    Ok((r, _)) => r,
                    Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
                }
            };
            let memories: Vec<Value> = records.into_iter().map(|r| json!({
                    "id": r.id, "profile": r.profile, "content": r.content,
                    "tags": r.tags, "source": r.source, "category": r.category,
                    "importance": r.importance, "created_at": r.created_at, "updated_at": r.updated_at,
                    "feedback_positive": r.feedback_positive, "feedback_negative": r.feedback_negative
                })).collect();
            Json(json!({ "ok": true, "memories": memories }))
        }
        "memory_import" => {
            let memories_arr = match req.args.get("memories").and_then(|v| v.as_array()) {
                Some(arr) => arr,
                None => return Json(json!({ "ok": false, "error": "missing memories array" })),
            };
            let mut imported = 0u64;
            for item in memories_arr {
                let content = match item.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c.to_string(),
                    None => continue,
                };
                let tags = item.get("tags").cloned().unwrap_or(json!({}));
                let importance = item
                    .get("importance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;
                let source = item
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("import")
                    .to_string();
                let category = item
                    .get("category")
                    .and_then(|v| v.as_str())
                    .unwrap_or("general")
                    .to_string();
                let metadata = item.get("metadata").cloned().unwrap_or(json!({}));
                let embedding = match state.embedder.embed(&content).await {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if state
                    .store
                    .store(
                        &req.profile,
                        &content,
                        &embedding,
                        &tags,
                        &metadata,
                        importance,
                        &source,
                        &category,
                        None,
                        false,
                        None,
                        None,
                    )
                    .await
                    .is_ok()
                {
                    imported += 1;
                }
            }
            Json(json!({ "ok": true, "imported": imported }))
        }
        "memory_delete" => {
            let id = match req.args.get("id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id" })),
            };
            match state.store.delete_by_id(id).await {
                Ok(()) => Json(json!({ "ok": true })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_compact" => {
            let dry_run = req
                .args
                .get("dry_run")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let similarity = req
                .args
                .get("similarity_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.9) as f32;
            let importance = req
                .args
                .get("importance_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            match state.store.compact(dry_run, similarity, importance).await {
                Ok(report) => Json(
                    json!({ "ok": true, "dry_run": dry_run, "merged": report.merged_pairs.len(), "archived": report.archived_ids.len(), "report": report }),
                ),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_snapshot" => match state.store.snapshot().await {
            Ok(snap) => Json(json!({ "ok": true, "snapshot": snap })),
            Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
        },
        "memory_diff" => {
            let snapshot_a_val = match req.args.get("snapshot_a") {
                Some(v) => v,
                None => return Json(json!({ "ok": false, "error": "missing snapshot_a" })),
            };
            let snapshot_b_val = match req.args.get("snapshot_b") {
                Some(v) => v,
                None => return Json(json!({ "ok": false, "error": "missing snapshot_b" })),
            };
            let snapshot_a: Vec<MemoryRecord> =
                serde_json::from_value(snapshot_a_val.clone()).unwrap_or_default();
            let snapshot_b: Vec<MemoryRecord> =
                serde_json::from_value(snapshot_b_val.clone()).unwrap_or_default();
            match state.store.diff_snapshots(&snapshot_a, &snapshot_b).await {
                Ok(diff) => Json(
                    json!({ "ok": true, "added": diff.added.len(), "removed": diff.removed.len(), "changed": diff.changed.len(), "diff": diff }),
                ),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_rollback" => {
            let snapshot_val = match req.args.get("snapshot") {
                Some(v) => v,
                None => return Json(json!({ "ok": false, "error": "missing snapshot" })),
            };
            let snapshot: Vec<MemoryRecord> =
                serde_json::from_value(snapshot_val.clone()).unwrap_or_default();
            match state.store.rollback_to(&snapshot).await {
                Ok(count) => Json(json!({ "ok": true, "restored": count })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_link" => {
            let id_a = match req.args.get("id_a").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id_a" })),
            };
            let id_b = match req.args.get("id_b").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id_b" })),
            };
            let relation_type = req
                .args
                .get("relation_type")
                .and_then(|v| v.as_str())
                .unwrap_or("related");
            match state.store.link_memories(id_a, id_b, relation_type).await {
                Ok(()) => Json(json!({ "ok": true })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_unlink" => {
            let id_a = match req.args.get("id_a").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id_a" })),
            };
            let id_b = match req.args.get("id_b").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id_b" })),
            };
            match state.store.unlink_memories(id_a, id_b).await {
                Ok(()) => Json(json!({ "ok": true })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_graph" => {
            let id = match req.args.get("id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id" })),
            };
            let depth = req.args.get("depth").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
            match state.store.get_memory_graph(id, depth).await {
                Ok(nodes) => Json(json!({ "ok": true, "nodes": nodes })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_associative_search" => {
            let query = req
                .args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let depth = req.args.get("depth").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
            let limit = req.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
            let category_filter = req
                .args
                .get("category_filter")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let embedding = match state.embedder.embed(&query).await {
                Ok(v) => v,
                Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
            };
            let search_profile: Option<String> =
                match req.args.get("profile").and_then(|v| v.as_str()) {
                    Some(p) if !p.is_empty() => Some(p.to_string()),
                    _ => None,
                };
            let seed_filters = SearchFilters {
                query: query.clone(),
                tag_filter: None,
                tag_match_mode: TagMatchMode::All,
                category_filter,
                created_after: None,
                created_before: None,
                min_importance: None,
                alpha: state.hybrid_alpha,
                limit,
                profile: search_profile,
                decay_half_life_days: state.decay_half_life_days,
                level: RetrievalLevel::Details,
                include_deep: req
                    .args
                    .get("include_deep")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            };
            match state
                .store
                .associative_search(&embedding, &seed_filters, depth)
                .await
            {
                Ok(results) => Json(json!({ "ok": true, "results": results })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_reminders_due" => match state.store.get_due_reminders(50).await {
            Ok(records) => {
                let reminders: Vec<Value> = records
                    .into_iter()
                    .map(|r| {
                        json!({
                            "id": r.id,
                            "profile": r.profile,
                            "content": r.content,
                            "importance": r.importance,
                            "event_date": r.event_date,
                            "reminder_interval": r.reminder_interval,
                            "reminder_at": r.reminder_at,
                        })
                    })
                    .collect();
                Json(json!({ "ok": true, "reminders": reminders }))
            }
            Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
        },
        "memory_feedback" => {
            let id = match req.args.get("id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id" })),
            };
            let useful = req
                .args
                .get("useful")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            match state.store.add_feedback(id, useful).await {
                Ok(()) => Json(json!({ "ok": true })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_remind" => {
            let memory_id = match req.args.get("memory_id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing memory_id" })),
            };
            let event_date = req
                .args
                .get("event_date")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc));
            let reminder_interval = req
                .args
                .get("reminder_interval")
                .and_then(|v| v.as_str().map(String::from));
            let reminder_at = req
                .args
                .get("reminder_at")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc));
            let reminder_in = req
                .args
                .get("reminder_in")
                .and_then(|v| v.as_str().map(String::from));
            if reminder_interval.is_none() && reminder_at.is_none() && reminder_in.is_none() {
                return Json(
                    json!({ "ok": false, "error": "reminder_interval, reminder_at, or reminder_in required" }),
                );
            }
            let reminder_at = if let (None, Some(ri)) = (&reminder_at, &reminder_in) {
                let (num_str, unit) = ri.split_at(ri.len().saturating_sub(1));
                let num: i64 = num_str.parse().unwrap_or(0);
                let delta = match unit {
                    "m" => chrono::TimeDelta::minutes(num),
                    "h" => chrono::TimeDelta::hours(num),
                    "d" => chrono::TimeDelta::days(num),
                    _ => chrono::TimeDelta::zero(),
                };
                Some(Utc::now() + delta)
            } else {
                reminder_at
            };
            match state
                .store
                .set_reminder(memory_id, event_date, reminder_interval, reminder_at)
                .await
            {
                Ok(true) => Json(json!({ "ok": true })),
                Ok(false) => Json(json!({ "ok": false, "error": "Memory not found" })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_deep_search" => {
            let query = req
                .args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let limit = req.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            let embedding = match state.embedder.embed(&query).await {
                Ok(v) => v,
                Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
            };
            let search_profile: Option<String> =
                match req.args.get("profile").and_then(|v| v.as_str()) {
                    Some(p) if !p.is_empty() => Some(p.to_string()),
                    _ => None,
                };
            let filters = SearchFilters {
                query: query.clone(),
                tag_filter: req.args.get("tags").cloned(),
                tag_match_mode: parse_tag_match_mode(
                    req.args.get("tag_match_mode").and_then(|v| v.as_str()),
                ),
                category_filter: None,
                created_after: None,
                created_before: None,
                min_importance: req
                    .args
                    .get("min_importance")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
                alpha: req
                    .args
                    .get("alpha")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(state.hybrid_alpha as f64) as f32,
                limit,
                profile: search_profile,
                decay_half_life_days: state.decay_half_life_days,
                level: RetrievalLevel::Details,
                include_deep: true,
            };
            match state.store.search_deep(&embedding, &filters).await {
                Ok(records) => {
                    let results: Vec<Value> = records
                        .into_iter()
                        .map(|r| {
                            json!({
                                "id": r.id, "content": r.content, "tags": r.tags,
                                "source": r.source, "category": r.category,
                                "created_at": r.created_at, "score": r.score,
                                "importance": r.importance, "immortal": r.immortal
                            })
                        })
                        .collect();
                    Json(json!({ "ok": true, "results": results }))
                }
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        _ => Json(json!({ "ok": false, "error": format!("unknown tool: {tool_name}") })),
    }
}

// ── Export ───────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ExportRequest {
    pub profile: String,
}

pub async fn export_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> Json<Value> {
    match state.store.list_by_profile(&req.profile).await {
        Ok(records) => {
            let memories: Vec<Value> = records.into_iter().map(|r| json!({
                "id": r.id, "profile": r.profile, "content": r.content,
                "embedding": r.embedding, "tags": r.tags, "metadata": r.metadata,
                "importance": r.importance, "source": r.source, "category": r.category,
                "created_at": r.created_at, "updated_at": r.updated_at,
                "feedback_positive": r.feedback_positive, "feedback_negative": r.feedback_negative,
                "score": r.score
            })).collect();
            Json(json!({ "ok": true, "memories": memories }))
        }
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Import ───────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ImportMemoryItem {
    pub content: String,
    pub tags: Option<Value>,
    pub metadata: Option<Value>,
    pub importance: Option<f32>,
    pub source: Option<String>,
    pub category: Option<String>,
}

#[derive(Deserialize)]
pub struct ImportRequest {
    pub profile: String,
    pub memories: Vec<ImportMemoryItem>,
}

pub async fn import_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ImportRequest>,
) -> Json<Value> {
    let mut imported = 0u64;
    for item in &req.memories {
        let tags = item.tags.as_ref().unwrap_or(&json!({})).clone();
        let metadata = item.metadata.as_ref().unwrap_or(&json!({})).clone();
        let importance = item.importance.unwrap_or(0.5);
        let source = item.source.clone().unwrap_or_else(|| "import".to_string());
        let category = item
            .category
            .clone()
            .unwrap_or_else(|| "general".to_string());
        let embedding = match state.embedder.embed(&item.content).await {
            Ok(v) => v,
            Err(_) => continue,
        };
        if state
            .store
            .store(
                &req.profile,
                &item.content,
                &embedding,
                &tags,
                &metadata,
                importance,
                &source,
                &category,
                None,
                false,
                None,
                None,
            )
            .await
            .is_ok()
        {
            imported += 1;
        }
    }
    Json(json!({ "ok": true, "imported": imported }))
}

// ── Delete ───────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct DeleteRequest {
    pub id: i64,
    pub profile: String,
}

pub async fn delete_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeleteRequest>,
) -> Json<Value> {
    match state.store.delete_by_id(req.id).await {
        Ok(()) => Json(json!({ "ok": true })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Memories CRUD ───────────────────────────────────────────

#[derive(Deserialize)]
pub struct ListMemoriesRequest {
    pub profile: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

pub async fn list_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ListMemoriesRequest>,
) -> Json<Value> {
    let limit = req.limit.unwrap_or(20);
    let offset = req.offset.unwrap_or(0);
    match state
        .store
        .list_memories(req.profile.as_deref(), limit, offset)
        .await
    {
        Ok((records, total)) => {
            let memories: Vec<Value> = records.into_iter().map(|r| json!({
                "id": r.id, "profile": r.profile, "content": r.content,
                "tags": r.tags, "source": r.source, "category": r.category,
                "importance": r.importance, "created_at": r.created_at, "updated_at": r.updated_at,
                "feedback_positive": r.feedback_positive, "feedback_negative": r.feedback_negative
            })).collect();
            Json(json!({ "ok": true, "memories": memories, "total": total }))
        }
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct GetMemoryRequest {
    pub id: i64,
}

pub async fn get_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GetMemoryRequest>,
) -> Json<Value> {
    match state.store.get_by_id(req.id).await {
        Ok(Some(record)) => Json(json!({ "ok": true,
            "id": record.id, "profile": record.profile, "content": record.content,
            "tags": record.tags, "source": record.source, "category": record.category,
            "importance": record.importance, "created_at": record.created_at, "updated_at": record.updated_at,
            "feedback_positive": record.feedback_positive, "feedback_negative": record.feedback_negative,
            "score": record.score
        })),
        Ok(None) => Json(json!({ "ok": false, "error": "not found" })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct UpdateMemoryRequest {
    pub id: i64,
    pub content: Option<String>,
    pub tags: Option<Value>,
    pub metadata: Option<Value>,
    pub importance: Option<f32>,
    pub category: Option<String>,
    pub source: Option<String>,
}

pub async fn update_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Json<Value> {
    match state
        .store
        .update_memory(
            req.id,
            req.content,
            req.tags,
            req.metadata,
            req.importance,
            req.category,
            req.source,
        )
        .await
    {
        Ok(updated) => Json(json!({ "ok": true, "updated": updated })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct AddFeedbackRequest {
    pub id: i64,
    pub useful: bool,
}

pub async fn add_feedback(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddFeedbackRequest>,
) -> Json<Value> {
    match state.store.add_feedback(req.id, req.useful).await {
        Ok(()) => Json(json!({ "ok": true })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Backup ───────────────────────────────────────────────────

pub async fn backup(State(state): State<Arc<AppState>>, Json(_req): Json<Value>) -> Json<Value> {
    match state.store.backup().await {
        Ok(records) => {
            let memories: Vec<Value> = records.into_iter().map(|r| json!({
                "id": r.id, "profile": r.profile, "content": r.content,
                "embedding": r.embedding, "tags": r.tags, "metadata": r.metadata,
                "importance": r.importance, "source": r.source, "category": r.category,
                "created_at": r.created_at, "updated_at": r.updated_at,
                "feedback_positive": r.feedback_positive, "feedback_negative": r.feedback_negative,
                "score": r.score
            })).collect();
            Json(json!({ "ok": true, "memories": memories }))
        }
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Restore ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RestoreMemoryItem {
    pub profile: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub tags: Option<Value>,
    pub metadata: Option<Value>,
    pub importance: Option<f32>,
    pub source: Option<String>,
    pub category: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub feedback_positive: Option<i32>,
    pub feedback_negative: Option<i32>,
}

#[derive(Deserialize)]
pub struct RestoreRequest {
    pub memories: Vec<RestoreMemoryItem>,
}

pub async fn restore(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RestoreRequest>,
) -> Json<Value> {
    let now = Utc::now();
    let records: Vec<MemoryRecord> = req
        .memories
        .into_iter()
        .map(|item| {
            let created_at = item
                .created_at
                .as_deref()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(now);
            let updated_at = item
                .updated_at
                .as_deref()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(now);
            MemoryRecord {
                id: 0,
                profile: item.profile,
                content: item.content,
                embedding: item.embedding,
                tags: item.tags.unwrap_or(json!({})),
                metadata: item.metadata.unwrap_or(json!({})),
                importance: item.importance.unwrap_or(0.0),
                source: item.source.unwrap_or_default(),
                category: item.category.unwrap_or_else(|| "general".to_string()),
                created_at,
                updated_at,
                feedback_positive: item.feedback_positive.unwrap_or(0),
                feedback_negative: item.feedback_negative.unwrap_or(0),
                expires_at: None,
                immortal: false,
                score: 0.0,
                access_count: 0,
                last_accessed_at: None,
                trust_score: 0.5,
                event_date: None,
                reminder_interval: None,
                reminder_at: None,
                reminder_sent: false,
            }
        })
        .collect();
    match state.store.restore(&records).await {
        Ok(count) => Json(json!({ "ok": true, "restored": count })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct RemindRequest {
    pub memory_id: i64,
    pub event_date: Option<String>,
    pub reminder_interval: Option<String>,
    pub reminder_at: Option<String>,
    pub reminder_in: Option<String>,
}

pub async fn remind(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RemindRequest>,
) -> Json<Value> {
    let event_date = req
        .event_date
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));
    let reminder_at = req
        .reminder_at
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));

    if req.reminder_interval.is_none() && reminder_at.is_none() && req.reminder_in.is_none() {
        return Json(
            json!({ "ok": false, "error": "Se requiere reminder_interval, reminder_at, o reminder_in" }),
        );
    }

    let reminder_at = if let (None, Some(ri)) = (&reminder_at, &req.reminder_in) {
        let (num_str, unit) = ri.split_at(ri.len().saturating_sub(1));
        let num: i64 = num_str.parse().unwrap_or(0);
        let delta = match unit {
            "m" => chrono::TimeDelta::minutes(num),
            "h" => chrono::TimeDelta::hours(num),
            "d" => chrono::TimeDelta::days(num),
            _ => chrono::TimeDelta::zero(),
        };
        Some(Utc::now() + delta)
    } else {
        reminder_at
    };

    match state
        .store
        .set_reminder(
            req.memory_id,
            event_date,
            req.reminder_interval,
            reminder_at,
        )
        .await
    {
        Ok(true) => Json(json!({ "ok": true })),
        Ok(false) => Json(json!({ "ok": false, "error": "Memory not found" })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

pub async fn reminders_due(State(state): State<Arc<AppState>>) -> Json<Value> {
    match state.store.get_due_reminders(50).await {
        Ok(records) => {
            let reminders: Vec<Value> = records
                .into_iter()
                .map(|r| {
                    json!({
                        "id": r.id,
                        "profile": r.profile,
                        "content": r.content,
                        "importance": r.importance,
                        "event_date": r.event_date,
                        "reminder_interval": r.reminder_interval,
                        "reminder_at": r.reminder_at,
                    })
                })
                .collect();
            Json(json!({ "ok": true, "reminders": reminders }))
        }
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Compact ───────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CompactRequest {
    pub dry_run: Option<bool>,
    pub similarity_threshold: Option<f32>,
    pub importance_threshold: Option<f32>,
}

pub async fn compact(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompactRequest>,
) -> Json<Value> {
    let dry_run = req.dry_run.unwrap_or(true);
    let similarity = req.similarity_threshold.unwrap_or(0.9);
    let importance = req.importance_threshold.unwrap_or(0.1);
    match state.store.compact(dry_run, similarity, importance).await {
        Ok(report) => Json(
            json!({ "ok": true, "dry_run": dry_run, "merged": report.merged_pairs.len(), "archived": report.archived_ids.len(), "report": report }),
        ),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Snapshot ──────────────────────────────────────────────────

pub async fn snapshot(State(state): State<Arc<AppState>>) -> Json<Value> {
    match state.store.snapshot().await {
        Ok(snap) => Json(json!({ "ok": true, "snapshot": snap })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Link / Graph ──────────────────────────────────────────────

#[derive(Deserialize)]
pub struct LinkRequest {
    pub id_a: i64,
    pub id_b: i64,
    pub relation_type: Option<String>,
}

pub async fn memory_link(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LinkRequest>,
) -> Json<Value> {
    let rtype = req.relation_type.as_deref().unwrap_or("related");
    match state.store.link_memories(req.id_a, req.id_b, rtype).await {
        Ok(()) => Json(json!({ "ok": true })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct UnlinkRequest {
    pub id_a: i64,
    pub id_b: i64,
}

pub async fn memory_unlink(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnlinkRequest>,
) -> Json<Value> {
    match state.store.unlink_memories(req.id_a, req.id_b).await {
        Ok(()) => Json(json!({ "ok": true })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct GraphRequest {
    pub id: i64,
    pub depth: Option<u32>,
}

pub async fn memory_graph(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GraphRequest>,
) -> Json<Value> {
    let depth = req.depth.unwrap_or(2);
    match state.store.get_memory_graph(req.id, depth).await {
        Ok(nodes) => Json(json!({ "ok": true, "nodes": nodes })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── OpenAPI ─────────────────────────────────────────────────

pub async fn openapi() -> Json<Value> {
    Json(serde_json::from_str(include_str!("../../openapi.json")).unwrap())
}

// ── Profile Management ────────────────────────────────────────

#[derive(Deserialize)]
pub struct ProfileStatusRequest {
    pub profile: String,
}

pub async fn profile_status(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProfileStatusRequest>,
) -> Json<Value> {
    match state.store.get_profile_status(&req.profile).await {
        Ok(status) => Json(json!({
            "ok": true,
            "profile": status.profile,
            "turn_count": status.turn_count,
            "is_cold": status.is_cold,
            "created_at": status.created_at,
            "last_active_at": status.last_active_at,
        })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct ConcludeRequest {
    pub profile: String,
    pub content: String,
    pub category: Option<String>,
    pub confidence: Option<f32>,
    pub source_turn_ids: Option<Vec<i64>>,
}

pub async fn conclude(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ConcludeRequest>,
) -> Json<Value> {
    let category = req.category.unwrap_or_else(|| "insight".to_string());
    let confidence = req.confidence.unwrap_or(0.5);
    let source_turn_ids = req.source_turn_ids.unwrap_or_default();
    match state
        .store
        .create_conclusion(
            &req.profile,
            &req.content,
            &category,
            confidence,
            &source_turn_ids,
        )
        .await
    {
        Ok(()) => Json(json!({ "ok": true })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Memory Stats ───────────────────────────────────────────────

pub async fn memory_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    match state.store.memory_stats().await {
        Ok(stats) => Json(json!(stats)),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Detailed Stats ────────────────────────────────────────────

pub async fn detailed_stats(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DetailedStatsRequest>,
) -> Json<Value> {
    match state.store.detailed_stats(req.profile.as_deref()).await {
        Ok(stats) => Json(json!(stats)),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Flush (manual buffer flush) ────────────────────────────────

#[derive(Deserialize)]
pub struct FlushRequest {
    pub profile: String,
}

pub async fn flush(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FlushRequest>,
) -> Json<Value> {
    let mut buffers = state.buffers.lock().await;
    if let Some(buffer) = buffers.get(&req.profile) {
        if !buffer.is_empty() {
            flush_buffer(&state, &req.profile, buffer, true).await;
        }
    }
    buffers.remove(&req.profile);
    Json(json!({ "ok": true, "flushed": true }))
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Mutex;
    use tower::ServiceExt;

    use crate::embeddings::EmbeddingProvider;
    use crate::storage::{
        AssociativeResult, CompactReport, DetailedMemoryStats, GraphNode, MemoryLink, MemoryRecord,
        MemoryStats, MemoryStore, NumericStats, ProfileStatus, SnapshotDiff, TierCounts,
    };

    #[derive(Default)]
    struct MockStore {
        memories: Mutex<Vec<MemoryRecord>>,
        next_id: Mutex<i64>,
    }

    #[async_trait]
    impl MemoryStore for MockStore {
        async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
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
            _expires_at: Option<DateTime<Utc>>,
            _immortal: bool,
            _event_date: Option<DateTime<Utc>>,
            _reminder_interval: Option<&str>,
        ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
            let mut next = self.next_id.lock().unwrap();
            *next += 1;
            let id = *next;
            let record = MemoryRecord {
                id,
                profile: profile.to_string(),
                content: content.to_string(),
                embedding: embedding.to_vec(),
                tags: tags.clone(),
                metadata: metadata.clone(),
                importance,
                source: source.to_string(),
                category: category.to_string(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                feedback_positive: 0,
                feedback_negative: 0,
                expires_at: None,
                immortal: false,
                score: 0.0,
                access_count: 0,
                last_accessed_at: None,
                trust_score: 0.5,
                event_date: None,
                reminder_interval: None,
                reminder_at: None,
                reminder_sent: false,
            };
            self.memories.lock().unwrap().push(record);
            Ok(id)
        }
        async fn find_duplicate(
            &self,
            _embedding: &[f32],
            _threshold: f32,
        ) -> Result<Option<(i64, f32)>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(None)
        }
        async fn search(
            &self,
            _embedding: &[f32],
            _filters: &SearchFilters,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn track_access(
            &self,
            _ids: &[i64],
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn compact(
            &self,
            _dry_run: bool,
            _similarity_threshold: f32,
            _importance_threshold: f32,
        ) -> Result<CompactReport, Box<dyn std::error::Error + Send + Sync>> {
            Ok(CompactReport {
                merged_pairs: vec![],
                archived_ids: vec![],
                decayed_ids: vec![],
            })
        }
        async fn snapshot(
            &self,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn diff_snapshots(
            &self,
            _snapshot_a: &[MemoryRecord],
            _snapshot_b: &[MemoryRecord],
        ) -> Result<SnapshotDiff, Box<dyn std::error::Error + Send + Sync>> {
            Ok(SnapshotDiff {
                added: vec![],
                removed: vec![],
                changed: vec![],
            })
        }
        async fn rollback_to(
            &self,
            _snapshot: &[MemoryRecord],
        ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
            Ok(0)
        }
        async fn delete_by_profile(
            &self,
            _profile: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn list_by_profile(
            &self,
            _profile: &str,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn delete_by_id(
            &self,
            _id: i64,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn list_memories(
            &self,
            _profile: Option<&str>,
            _limit: usize,
            _offset: usize,
        ) -> Result<(Vec<MemoryRecord>, u64), Box<dyn std::error::Error + Send + Sync>> {
            let mems = self.memories.lock().unwrap().clone();
            let total = mems.len() as u64;
            Ok((mems, total))
        }
        async fn get_by_id(
            &self,
            id: i64,
        ) -> Result<Option<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self
                .memories
                .lock()
                .unwrap()
                .iter()
                .find(|m| m.id == id)
                .cloned())
        }
        async fn update_memory(
            &self,
            _id: i64,
            _content: Option<String>,
            _tags: Option<Value>,
            _metadata: Option<Value>,
            _importance: Option<f32>,
            _category: Option<String>,
            _source: Option<String>,
        ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
            Ok(true)
        }
        async fn add_feedback(
            &self,
            _id: i64,
            _useful: bool,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn backup(
            &self,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn restore(
            &self,
            records: &[MemoryRecord],
        ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
            let count = records.len() as u64;
            self.memories.lock().unwrap().extend_from_slice(records);
            Ok(count)
        }
        async fn get_profile_status(
            &self,
            _profile: &str,
        ) -> Result<ProfileStatus, Box<dyn std::error::Error + Send + Sync>> {
            Ok(ProfileStatus {
                profile: _profile.to_string(),
                turn_count: 0,
                is_cold: true,
                created_at: Utc::now(),
                last_active_at: Utc::now(),
            })
        }
        async fn increment_turn_count(
            &self,
            _profile: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn get_base_context(
            &self,
            _profile: &str,
            _limit: usize,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn create_conclusion(
            &self,
            _profile: &str,
            _content: &str,
            _category: &str,
            _confidence: f32,
            _source_turn_ids: &[i64],
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn link_memories(
            &self,
            _id_a: i64,
            _id_b: i64,
            _relation_type: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn unlink_memories(
            &self,
            _id_a: i64,
            _id_b: i64,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn get_relations(
            &self,
            _id: i64,
        ) -> Result<Vec<MemoryLink>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![])
        }
        async fn get_memory_graph(
            &self,
            _id: i64,
            _depth: u32,
        ) -> Result<Vec<GraphNode>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![])
        }
        async fn associative_search(
            &self,
            _embedding: &[f32],
            _filters: &SearchFilters,
            _depth: u32,
        ) -> Result<Vec<AssociativeResult>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![])
        }
        async fn get_due_reminders(
            &self,
            _limit: usize,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![])
        }
        async fn mark_reminder_sent(
            &self,
            _id: i64,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn set_reminder(
            &self,
            _id: i64,
            _event_date: Option<DateTime<Utc>>,
            _reminder_interval: Option<String>,
            _reminder_at: Option<DateTime<Utc>>,
        ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
            Ok(true)
        }

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
            _expires_at: Option<DateTime<Utc>>,
            _event_date: Option<DateTime<Utc>>,
            _reminder_interval: Option<&str>,
            _conversation_id: &str,
            _turn_range: &str,
            _user_msg: &str,
            _assistant_msg: &str,
        ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
            let mut next = self.next_id.lock().unwrap();
            *next += 1;
            let id = *next;
            self.memories.lock().unwrap().push(MemoryRecord {
                id,
                profile: profile.to_string(),
                content: content.to_string(),
                embedding: embedding.to_vec(),
                tags: tags.clone(),
                metadata: metadata.clone(),
                importance,
                source: source.to_string(),
                category: category.to_string(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                feedback_positive: 0,
                feedback_negative: 0,
                expires_at: None,
                immortal: false,
                score: 0.0,
                access_count: 0,
                last_accessed_at: None,
                trust_score: 0.5,
                event_date: None,
                reminder_interval: None,
                reminder_at: None,
                reminder_sent: false,
            });
            Ok(id)
        }

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
            _expires_at: Option<DateTime<Utc>>,
            _event_date: Option<DateTime<Utc>>,
            _reminder_interval: Option<&str>,
            _conversation_id: &str,
            _turn_range: &str,
            _user_msg: &str,
            _assistant_msg: &str,
        ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
            self.store(
                profile, content, embedding, tags, metadata, importance, source, category, None,
                false, None, None,
            )
            .await
        }

        async fn search_fresh(
            &self,
            _embedding: &[f32],
            _filters: &SearchFilters,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }

        async fn search_consolid(
            &self,
            _embedding: &[f32],
            _filters: &SearchFilters,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![])
        }

        async fn search_deep(
            &self,
            _embedding: &[f32],
            _filters: &SearchFilters,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }

        async fn store_consolid(
            &self,
            _profile: &str,
            _summary: &str,
            _embedding: &[f32],
            _source_ids: &[i64],
            _tags: &Value,
            _metadata: &Value,
            _importance: f32,
        ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
            Ok(0)
        }

        async fn prune_fresh(
            &self,
            _ttl_hours: u64,
        ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
            Ok(0)
        }

        async fn get_new_deep_since(
            &self,
            _since: DateTime<Utc>,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }

        async fn get_new_fresh_since(
            &self,
            _profile: &str,
            _since: DateTime<Utc>,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }

        async fn prune_deep(
            &self,
            _min_importance: f32,
            _min_access_count: i32,
            _max_age_days: i64,
        ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
            Ok(0)
        }

        async fn prune_consolid(
            &self,
            _min_insight_score: f32,
            _max_age_days: i64,
        ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
            Ok(0)
        }

        async fn schedule_review(
            &self,
            _id: i64,
            _importance: f32,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }

        async fn memory_stats(
            &self,
        ) -> Result<MemoryStats, Box<dyn std::error::Error + Send + Sync>> {
            Ok(MemoryStats {
                fresh_count: 0,
                deep_count: 0,
                consolid_count: 0,
                per_profile: vec![],
            })
        }

        async fn detailed_stats(
            &self,
            _profile: Option<&str>,
        ) -> Result<DetailedMemoryStats, Box<dyn std::error::Error + Send + Sync>> {
            Ok(DetailedMemoryStats {
                tier_counts: TierCounts {
                    fresh: 0,
                    deep: 0,
                    consolid: 0,
                },
                total_memories: 0,
                total_profiles: 0,
                total_links: 0,
                importance: NumericStats {
                    min: 0.0,
                    max: 0.0,
                    avg: 0.0,
                    median: 0.0,
                    p95: 0.0,
                    count: 0,
                },
                trust_score: NumericStats {
                    min: 0.0,
                    max: 0.0,
                    avg: 0.0,
                    median: 0.0,
                    p95: 0.0,
                    count: 0,
                },
                access_count: NumericStats {
                    min: 0.0,
                    max: 0.0,
                    avg: 0.0,
                    median: 0.0,
                    p95: 0.0,
                    count: 0,
                },
                content_length: NumericStats {
                    min: 0.0,
                    max: 0.0,
                    avg: 0.0,
                    median: 0.0,
                    p95: 0.0,
                    count: 0,
                },
                oldest_memory: None,
                newest_memory: None,
                category_distribution: vec![],
                top_tags: vec![],
                source_distribution: vec![],
                feedback_positive: 0,
                feedback_negative: 0,
                immortal_count: 0,
                mortal_count: 0,
                expired_count: 0,
                consolid_depth_distribution: vec![],
                reminders_active: 0,
                reminders_sent: 0,
                per_profile: vec![],
            })
        }

        async fn get_consolid_by_depth_since(
            &self,
            _depth: &str,
            _since: DateTime<Utc>,
        ) -> Result<
            Vec<crate::storage::MemoryRecordConsolid>,
            Box<dyn std::error::Error + Send + Sync>,
        > {
            Ok(vec![])
        }
    }

    struct MockEmbedder;

    #[async_trait]
    impl EmbeddingProvider for MockEmbedder {
        async fn embed(
            &self,
            _text: &str,
        ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![0.1; 4])
        }
    }

    fn test_app() -> Router {
        let state = AppState::new(
            Box::new(MockEmbedder),
            Box::new(MockStore::default()),
            0.5,
            30.0,
            0.3,
            1,
            1,
            10,
            None,
            5,
            false,
            20,
            300,
            24,
            3600,
            "gpt-4o-mini".to_string(),
            "".to_string(),
            0.05,
            0,
            90,
            0.05,
            60,
        );
        crate::api::router(state)
    }

    #[tokio::test]
    async fn test_health() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body, json!({"status": "ok"}));
    }

    #[tokio::test]
    async fn test_tool_schemas() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/tool_schemas")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Vec<ToolSchema> = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body.len() >= 8);
        assert!(body.iter().any(|s| s.name == "memory_search"));
    }

    #[tokio::test]
    async fn test_initialize() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/initialize")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "profile": "test-session"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["profile"], "test-session");
    }

    #[tokio::test]
    async fn test_prefetch() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/prefetch")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "query": "test query"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_sync_turn() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/sync_turn")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "user": "hello",
                            "assistant": "hi there",
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body["ok"] == true);
    }

    #[tokio::test]
    async fn test_sync_turn_conservative_skip() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/sync_turn")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "user": "ok",
                            "assistant": "sure",
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["skipped"], true);
    }

    #[tokio::test]
    async fn test_on_session_end() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/on_session_end")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_search() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "memory_search",
                            "args": {"query": "find something"},
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_add() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "memory_add",
                            "args": {"content": "important note", "tags": {"project": "test"}},
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_unknown() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "nonexistent_tool",
                            "args": {},
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], false);
    }

    #[tokio::test]
    async fn test_health_404() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 404);
    }

    #[tokio::test]
    async fn test_export() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/export")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_import() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/import")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "profile": "s1",
                            "memories": [{"content": "imported memory"}]
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_delete_memory() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/delete")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "id": 1,
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_export() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "memory_export",
                            "args": {},
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_delete() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "memory_delete",
                            "args": {"id": 1},
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_import() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "memory_import",
                            "args": {"memories": [{"content": "imported via tool"}]},
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_prefetch_with_tags() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/prefetch")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "query": "test",
                            "tag_filter": {"project": "test"},
                            "tag_match_mode": "any"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_import_with_metadata() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/import")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "profile": "s1",
                            "memories": [{
                                "content": "memory with metadata",
                                "metadata": {"source": "test"},
                                "importance": 0.8
                            }]
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_sync_turn_with_metadata() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/sync_turn")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "user": "hello world",
                            "assistant": "hi there",
                            "profile": "s1",
                            "tags": {"project": "test"},
                            "metadata": {"source": "manual"},
                            "importance": 0.7,
                            "source": "test",
                            "category": "testing"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body["ok"] == true);
    }

    #[tokio::test]
    async fn test_list_memories() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories/list")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "profile": "s1",
                            "limit": 10,
                            "offset": 0
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_get_memory() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories/get")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "id": 1
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body["ok"] == true || (body["ok"] == false && body["error"] == "not found"));
    }

    #[tokio::test]
    async fn test_update_memory() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories/update")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "id": 1,
                            "content": "updated content",
                            "importance": 0.9
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_backup() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/backup")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_restore() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/restore")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "memories": [{
                                "profile": "s1",
                                "content": "restored memory",
                                "embedding": [0.1, 0.2, 0.3, 0.4],
                                "tags": {},
                                "metadata": {},
                                "importance": 0.5,
                                "source": "test",
                                "category": "general",
                                "created_at": "2024-01-01T00:00:00Z",
                                "updated_at": "2024-01-01T00:00:00Z"
                            }]
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_feedback() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/memories/feedback")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "id": 1,
                            "useful": true
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_add_with_event_date() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "memory_add",
                            "args": {
                                "content": "doctor appointment",
                                "event_date": "2026-06-10T14:00:00Z",
                                "reminder": "30m"
                            },
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_add_with_reminder_absolute() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "memory_add",
                            "args": {
                                "content": "standup meeting",
                                "event_date": "2026-06-11T09:00:00Z",
                                "reminder": "2026-06-11T08:55:00Z"
                            },
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn test_handle_tool_call_reminders_due() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/handle_tool_call")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "tool_name": "memory_reminders_due",
                            "args": {},
                            "profile": "s1"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["reminders"], json!([]));
    }

    #[tokio::test]
    async fn test_reminders_due_endpoint() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/reminders/due")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["reminders"], json!([]));
    }

    #[tokio::test]
    async fn test_tool_schemas_contains_reminders_due() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/tool_schemas")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Vec<ToolSchema> = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body.iter().any(|s| s.name == "memory_reminders_due"));
    }

    #[tokio::test]
    async fn test_memory_add_schema_has_event_fields() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/tool_schemas")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body: Vec<ToolSchema> = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let add_schema = body.iter().find(|s| s.name == "memory_add").unwrap();
        assert!(
            add_schema.parameters["properties"]
                .get("event_date")
                .is_some()
        );
        assert!(
            add_schema.parameters["properties"]
                .get("reminder")
                .is_some()
        );
    }
}
