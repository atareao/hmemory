use axum::{Json, extract::State};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

use super::AppState;
use crate::storage::{MemoryRecord, RetrievalLevel, SearchFilters, TagMatchMode};

fn compute_importance(text: &str) -> f32 {
    let mut score = 0.2f32;
    if text.len() > 100 { score += 0.1; }
    if text.len() > 300 { score += 0.1; }
    if text.contains("```") { score += 0.15; }
    let num_count = text.chars().filter(|c| c.is_ascii_digit()).count();
    if num_count > 5 { score += 0.1; }
    let keywords = ["project", "config", "build", "deploy", "api", "error", "fix", "implement", "database", "server", "function", "feature", "bug", "version", "v0.", "release", "merge", "commit", "pr", "todo", "refactor"];
    for kw in &keywords {
        if text.to_lowercase().contains(kw) { score += 0.05; }
    }
    let lower = text.to_lowercase().trim().to_string();
    let greetings = ["hello", "hi", "hey", "thanks", "ok", "sure", "yeah", "yep", "nope", "no", "yes", "goodbye", "bye", "👍", "done"];
    for g in &greetings {
        if lower == *g || lower.starts_with(&format!("{g} ")) || lower.ends_with(&format!(" {g}")) {
            score -= 0.15; break;
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
    s.and_then(|v| DateTime::parse_from_rfc3339(v).ok().map(|dt| dt.with_timezone(&Utc)))
}

// ── Health ──────────────────────────────────────────────────

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

// ── Initialize ──────────────────────────────────────────────

#[derive(Deserialize)]
pub struct InitializeRequest {
    pub session_id: String,
}

pub async fn initialize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InitializeRequest>,
) -> Json<Value> {
    match state.store.initialize().await {
        Ok(()) => Json(json!({ "ok": true, "session_id": req.session_id })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Prefetch (search) ───────────────────────────────────────

#[derive(Deserialize)]
pub struct PrefetchRequest {
    pub query: String,
    pub session_id: Option<String>,
    pub limit: Option<usize>,
    pub tag_filter: Option<Value>,
    pub tag_match_mode: Option<String>,
    pub category_filter: Option<String>,
    pub created_after: Option<String>,
    pub created_before: Option<String>,
    pub min_importance: Option<f32>,
    pub level: Option<String>,
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
        session_id: req.session_id.clone(),
        decay_half_life_days: state.decay_half_life_days,
        level: parse_level(req.level.as_deref()),
    };
    match state.store.search(&embedding, &filters).await {
        Ok(records) => {
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

// ── Sync Turn ───────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SyncTurnRequest {
    pub user: String,
    pub assistant: String,
    pub session_id: String,
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
    let importance = req.importance.unwrap_or(0.0);
    let source = req.source.unwrap_or_else(|| "hermes".to_string());
    let category = req.category.unwrap_or_else(|| "general".to_string());
    let combined = format!("{}\n{}", req.user, req.assistant);
    let computed = compute_importance(&combined);
    if computed < state.sync_turn_min_importance {
        return Json(json!({ "ok": true, "stored": 0, "skipped": true, "importance": computed }));
    }
    let user_embedding = match state.embedder.embed(&req.user).await {
        Ok(v) => v,
        Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
    };
    let assistant_embedding = match state.embedder.embed(&req.assistant).await {
        Ok(v) => v,
        Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
    };
    if let Err(e) = state.store.store(&req.session_id, &format!("user: {}", req.user), &user_embedding, &tags, &metadata, computed.max(importance), &source, &category).await {
        return Json(json!({ "ok": false, "error": e.to_string() }));
    }
    if let Err(e) = state.store.store(&req.session_id, &format!("assistant: {}", req.assistant), &assistant_embedding, &tags, &metadata, computed.max(importance), &source, &category).await {
        return Json(json!({ "ok": false, "error": e.to_string() }));
    }
    Json(json!({ "ok": true, "stored": 2, "importance": computed }))
}

// ── On Session End ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct SessionEndRequest {
    pub session_id: String,
}

pub async fn on_session_end(
    _state: State<Arc<AppState>>,
    Json(_req): Json<SessionEndRequest>,
) -> Json<Value> {
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
                "category_filter": {"type": "string", "description": "Filter by category"}
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
                "category": {"type": "string", "description": "Category"}
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
                "offset": {"type": "integer", "description": "Offset", "default": 0}
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
        description: "Export all memories for a session".into(),
        parameters: json!({
            "type": "object",
            "properties": {},
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
                            "session_id": {"type": "string"},
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
                        "required": ["session_id", "content"]
                    }
                }
            },
            "required": ["memories"]
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
    ])
}

// ── Handle Tool Call ────────────────────────────────────────

#[derive(Deserialize)]
pub struct ToolCallRequest {
    pub tool_name: String,
    pub args: Value,
    pub session_id: String,
}

pub async fn handle_tool_call(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ToolCallRequest>,
) -> Json<Value> {
    // Normalize: accept both "hmemory_X" and "memory_X" from different Hermes plugin versions
    let tool_name = req.tool_name.strip_prefix("hmemory_").unwrap_or(&req.tool_name);
    match tool_name {
        "memory_search" => {
            let query = req.args.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let limit = req.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
            let tag_filter = req.args.get("tag_filter").cloned();
            let tag_match_mode = parse_tag_match_mode(req.args.get("tag_match_mode").and_then(|v| v.as_str()));
            let category_filter = req.args.get("category_filter").and_then(|v| v.as_str()).map(|s| s.to_string());
            let embedding = match state.embedder.embed(&query).await {
                Ok(v) => v,
                Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
            };
            let level = parse_level(req.args.get("level").and_then(|v| v.as_str()));
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
                session_id: Some(req.session_id.clone()),
                decay_half_life_days: state.decay_half_life_days,
                level,
            };
            match state.store.search(&embedding, &filters).await {
                Ok(records) => {
                    let results: Vec<Value> = records.into_iter().map(|r| json!({
                        "id": r.id, "content": r.content, "tags": r.tags,
                        "source": r.source, "category": r.category,
                        "created_at": r.created_at, "score": r.score
                    })).collect();
                    Json(json!({ "ok": true, "results": results }))
                }
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_add" => {
            let content = req.args.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let tags = req.args.get("tags").cloned().unwrap_or(json!({}));
            let importance = req.args.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
            let source = req.args.get("source").and_then(|v| v.as_str()).unwrap_or("tool").to_string();
            let category = req.args.get("category").and_then(|v| v.as_str()).unwrap_or("general").to_string();
            let metadata = req.args.get("metadata").cloned().unwrap_or(json!({}));
            let embedding = match state.embedder.embed(&content).await {
                Ok(v) => v,
                Err(e) => return Json(json!({ "ok": false, "error": e.to_string() })),
            };
            match state.store.store(&req.session_id, &content, &embedding, &tags, &metadata, importance, &source, &category).await {
                Ok(()) => Json(json!({ "ok": true })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_list" => {
            let limit = req.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
            let offset = req.args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            match state.store.list_memories(Some(&req.session_id), limit, offset).await {
                Ok((records, total)) => {
                    let memories: Vec<Value> = records.into_iter().map(|r| json!({
                        "id": r.id, "session_id": r.session_id, "content": r.content,
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
                Ok(Some(record)) => Json(json!({ "ok": true, "memory": {
                    "id": record.id, "session_id": record.session_id, "content": record.content,
                    "tags": record.tags, "source": record.source, "category": record.category,
                    "importance": record.importance, "created_at": record.created_at, "updated_at": record.updated_at,
                    "feedback_positive": record.feedback_positive, "feedback_negative": record.feedback_negative,
                    "score": record.score
                }})),
                Ok(None) => Json(json!({ "ok": false, "error": "not found" })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_update" => {
            let id = match req.args.get("id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return Json(json!({ "ok": false, "error": "missing id" })),
            };
            let content = req.args.get("content").and_then(|v| v.as_str()).map(|s| s.to_string());
            let tags = req.args.get("tags").cloned();
            let metadata = req.args.get("metadata").cloned();
            let importance = req.args.get("importance").and_then(|v| v.as_f64()).map(|v| v as f32);
            let category = req.args.get("category").and_then(|v| v.as_str()).map(|s| s.to_string());
            let source = req.args.get("source").and_then(|v| v.as_str()).map(|s| s.to_string());
            match state.store.update_memory(id, content, tags, metadata, importance, category, source).await {
                Ok(updated) => Json(json!({ "ok": true, "updated": updated })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_backup" => {
            match state.store.backup().await {
                Ok(records) => {
                    let memories: Vec<Value> = records.into_iter().map(|r| json!({
                        "id": r.id, "session_id": r.session_id, "content": r.content,
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
        "memory_restore" => {
            let memories_arr = match req.args.get("memories").and_then(|v| v.as_array()) {
                Some(arr) => arr,
                None => return Json(json!({ "ok": false, "error": "missing memories array" })),
            };
            let records: Vec<MemoryRecord> = memories_arr.iter().filter_map(|v| {
                let session_id = v.get("session_id")?.as_str()?.to_string();
                let content = v.get("content")?.as_str()?.to_string();
                let embedding = v.get("embedding")?.as_array().map(|a| a.iter().filter_map(|n| n.as_f64()).map(|n| n as f32).collect()).unwrap_or_default();
                let tags = v.get("tags").cloned().unwrap_or(json!({}));
                let metadata = v.get("metadata").cloned().unwrap_or(json!({}));
                let importance = v.get("importance").and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
                let source = v.get("source").and_then(|s| s.as_str()).unwrap_or("").to_string();
                let category = v.get("category").and_then(|s| s.as_str()).unwrap_or("general").to_string();
                let created_at = v.get("created_at").and_then(|s| s.as_str()).and_then(|s| DateTime::parse_from_rfc3339(s).ok()).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(Utc::now);
                let updated_at = v.get("updated_at").and_then(|s| s.as_str()).and_then(|s| DateTime::parse_from_rfc3339(s).ok()).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(Utc::now);
                let feedback_positive = v.get("feedback_positive").and_then(|n| n.as_i64()).unwrap_or(0) as i32;
                let feedback_negative = v.get("feedback_negative").and_then(|n| n.as_i64()).unwrap_or(0) as i32;
                Some(MemoryRecord {
                    id: 0, session_id, content, embedding, tags, metadata, importance, source, category,
                    created_at, updated_at, feedback_positive, feedback_negative, score: 0.0,
                })
            }).collect();
            match state.store.restore(&records).await {
                Ok(count) => Json(json!({ "ok": true, "restored": count })),
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
        }
        "memory_export" => {
            match state.store.list_by_session(&req.session_id).await {
                Ok(records) => {
                    let memories: Vec<Value> = records.into_iter().map(|r| json!({
                        "id": r.id, "session_id": r.session_id, "content": r.content,
                        "tags": r.tags, "source": r.source, "category": r.category,
                        "importance": r.importance, "created_at": r.created_at, "updated_at": r.updated_at,
                        "feedback_positive": r.feedback_positive, "feedback_negative": r.feedback_negative
                    })).collect();
                    Json(json!({ "ok": true, "memories": memories }))
                }
                Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
            }
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
                let importance = item.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                let source = item.get("source").and_then(|v| v.as_str()).unwrap_or("import").to_string();
                let category = item.get("category").and_then(|v| v.as_str()).unwrap_or("general").to_string();
                let metadata = item.get("metadata").cloned().unwrap_or(json!({}));
                let embedding = match state.embedder.embed(&content).await {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if state.store.store(&req.session_id, &content, &embedding, &tags, &metadata, importance, &source, &category).await.is_ok() {
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
        _ => Json(json!({ "ok": false, "error": format!("unknown tool: {tool_name}") })),
    }
}

// ── Export ───────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ExportRequest {
    pub session_id: String,
}

pub async fn export_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> Json<Value> {
    match state.store.list_by_session(&req.session_id).await {
        Ok(records) => {
            let memories: Vec<Value> = records.into_iter().map(|r| json!({
                "id": r.id, "session_id": r.session_id, "content": r.content,
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
    pub session_id: String,
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
        let category = item.category.clone().unwrap_or_else(|| "general".to_string());
        let embedding = match state.embedder.embed(&item.content).await {
            Ok(v) => v,
            Err(_) => continue,
        };
        if state.store.store(&req.session_id, &item.content, &embedding, &tags, &metadata, importance, &source, &category).await.is_ok() {
            imported += 1;
        }
    }
    Json(json!({ "ok": true, "imported": imported }))
}

// ── Delete ───────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct DeleteRequest {
    pub id: i64,
    pub session_id: String,
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
    pub session_id: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

pub async fn list_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ListMemoriesRequest>,
) -> Json<Value> {
    let limit = req.limit.unwrap_or(20);
    let offset = req.offset.unwrap_or(0);
    match state.store.list_memories(req.session_id.as_deref(), limit, offset).await {
        Ok((records, total)) => {
            let memories: Vec<Value> = records.into_iter().map(|r| json!({
                "id": r.id, "session_id": r.session_id, "content": r.content,
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
        Ok(Some(record)) => Json(json!({ "ok": true, "memory": {
            "id": record.id, "session_id": record.session_id, "content": record.content,
            "tags": record.tags, "source": record.source, "category": record.category,
            "importance": record.importance, "created_at": record.created_at, "updated_at": record.updated_at,
            "feedback_positive": record.feedback_positive, "feedback_negative": record.feedback_negative,
            "score": record.score
        }})),
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
    match state.store.update_memory(req.id, req.content, req.tags, req.metadata, req.importance, req.category, req.source).await {
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

pub async fn backup(
    State(state): State<Arc<AppState>>,
    Json(_req): Json<Value>,
) -> Json<Value> {
    match state.store.backup().await {
        Ok(records) => {
            let memories: Vec<Value> = records.into_iter().map(|r| json!({
                "id": r.id, "session_id": r.session_id, "content": r.content,
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
    pub session_id: String,
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
    let records: Vec<MemoryRecord> = req.memories.into_iter().map(|item| {
        let created_at = item.created_at.as_deref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(now);
        let updated_at = item.updated_at.as_deref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(now);
        MemoryRecord {
            id: 0,
            session_id: item.session_id,
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
            score: 0.0,
        }
    }).collect();
    match state.store.restore(&records).await {
        Ok(count) => Json(json!({ "ok": true, "restored": count })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── OpenAPI ─────────────────────────────────────────────────

pub async fn openapi() -> Json<Value> {
    Json(serde_json::from_str(include_str!("../../openapi.json")).unwrap())
}

// ── Session Management ───────────────────────────────────────

#[derive(Deserialize)]
pub struct ResolveSessionRequest {
    pub session_id: String,
    pub strategy: Option<String>,
    pub path_or_repo: Option<String>,
}

pub async fn resolve_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ResolveSessionRequest>,
) -> Json<Value> {
    let strategy = req.strategy.unwrap_or_else(|| format!("{:?}", state.session_strategy));
    let path_or_repo = req.path_or_repo.unwrap_or_default();
    match state.store.resolve_session(&req.session_id, &strategy, &path_or_repo).await {
        Ok(sid) => Json(json!({ "ok": true, "session_id": sid })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct SessionStatusRequest {
    pub session_id: String,
}

pub async fn session_status(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SessionStatusRequest>,
) -> Json<Value> {
    match state.store.get_session_status(&req.session_id).await {
        Ok(status) => Json(json!({
            "ok": true,
            "session_id": status.session_id,
            "strategy": status.strategy,
            "path_or_repo": status.path_or_repo,
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
    pub session_id: String,
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
    match state.store.create_conclusion(&req.session_id, &req.content, &category, confidence, &source_turn_ids).await {
        Ok(()) => Json(json!({ "ok": true })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::SessionStatus;
    use async_trait::async_trait;
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Mutex;
    use tower::ServiceExt;

    use crate::embeddings::EmbeddingProvider;
    use crate::storage::{MemoryRecord, MemoryStore};

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
            session_id: &str,
            content: &str,
            embedding: &[f32],
            tags: &Value,
            metadata: &Value,
            importance: f32,
            source: &str,
            category: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let mut next = self.next_id.lock().unwrap();
            *next += 1;
            let record = MemoryRecord {
                id: *next,
                session_id: session_id.to_string(),
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
                score: 0.0,
            };
            self.memories.lock().unwrap().push(record);
            Ok(())
        }
        async fn search(
            &self,
            _embedding: &[f32],
            _filters: &SearchFilters,
        ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn delete_session(&self, _session_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn list_by_session(&self, _session_id: &str) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn delete_by_id(&self, _id: i64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn list_memories(&self, _session_id: Option<&str>, _limit: usize, _offset: usize) -> Result<(Vec<MemoryRecord>, u64), Box<dyn std::error::Error + Send + Sync>> {
            let mems = self.memories.lock().unwrap().clone();
            let total = mems.len() as u64;
            Ok((mems, total))
        }
        async fn get_by_id(&self, id: i64) -> Result<Option<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().iter().find(|m| m.id == id).cloned())
        }
        async fn update_memory(&self, _id: i64, _content: Option<String>, _tags: Option<Value>, _metadata: Option<Value>, _importance: Option<f32>, _category: Option<String>, _source: Option<String>) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
            Ok(true)
        }
        async fn add_feedback(&self, _id: i64, _useful: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn backup(&self) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn restore(&self, records: &[MemoryRecord]) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
            let count = records.len() as u64;
            self.memories.lock().unwrap().extend_from_slice(records);
            Ok(count)
        }
        async fn resolve_session(&self, _session_id: &str, _strategy: &str, _path_or_repo: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
            Ok(_session_id.to_string())
        }
        async fn get_session_status(&self, _session_id: &str) -> Result<SessionStatus, Box<dyn std::error::Error + Send + Sync>> {
            Ok(SessionStatus {
                session_id: _session_id.to_string(),
                strategy: "per-session".into(),
                path_or_repo: "".into(),
                turn_count: 0,
                is_cold: true,
                created_at: Utc::now(),
                last_active_at: Utc::now(),
            })
        }
        async fn increment_turn_count(&self, _session_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
        async fn get_base_context(&self, _session_id: &str, _limit: usize) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.memories.lock().unwrap().clone())
        }
        async fn create_conclusion(&self, _session_id: &str, _content: &str, _category: &str, _confidence: f32, _source_turn_ids: &[i64]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
    }

    struct MockEmbedder;

    #[async_trait]
    impl EmbeddingProvider for MockEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
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
            crate::config::SessionStrategy::PerSession,
            1,
            1,
            10,
            None,
            5,
            false,
        );
        crate::api::router(state)
    }

    #[tokio::test]
    async fn test_health() {
        let app = test_app();
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert_eq!(body, json!({"status": "ok"}));
    }

    #[tokio::test]
    async fn test_tool_schemas() {
        let app = test_app();
        let response = app
            .oneshot(Request::builder().uri("/tool_schemas").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Vec<ToolSchema> = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "session_id": "test-session"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["session_id"], "test-session");
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "query": "test query"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "user": "hello",
                        "assistant": "hi there",
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "user": "ok",
                        "assistant": "sure",
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "tool_name": "memory_search",
                        "args": {"query": "find something"},
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "tool_name": "memory_add",
                        "args": {"content": "important note", "tags": {"project": "test"}},
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "tool_name": "nonexistent_tool",
                        "args": {},
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], false);
    }

    #[tokio::test]
    async fn test_health_404() {
        let app = test_app();
        let response = app
            .oneshot(Request::builder().uri("/nonexistent").body(Body::empty()).unwrap())
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "session_id": "s1",
                        "memories": [{"content": "imported memory"}]
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "id": 1,
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "tool_name": "memory_export",
                        "args": {},
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "tool_name": "memory_delete",
                        "args": {"id": 1},
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "tool_name": "memory_import",
                        "args": {"memories": [{"content": "imported via tool"}]},
                        "session_id": "s1"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "query": "test",
                        "tag_filter": {"project": "test"},
                        "tag_match_mode": "any"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "session_id": "s1",
                        "memories": [{
                            "content": "memory with metadata",
                            "metadata": {"source": "test"},
                            "importance": 0.8
                        }]
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "user": "hello world",
                        "assistant": "hi there",
                        "session_id": "s1",
                        "tags": {"project": "test"},
                        "metadata": {"source": "manual"},
                        "importance": 0.7,
                        "source": "test",
                        "category": "testing"
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "session_id": "s1",
                        "limit": 10,
                        "offset": 0
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "id": 1
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "id": 1,
                        "content": "updated content",
                        "importance": 0.9
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "memories": [{
                            "session_id": "s1",
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
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
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
                    .body(Body::from(serde_json::to_vec(&json!({
                        "id": 1,
                        "useful": true
                    })).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert_eq!(body["ok"], true);
    }
}