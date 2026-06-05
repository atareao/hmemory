use super::{MemoryRecord, MemoryStore, RetrievalLevel, SearchFilters, SessionStatus, TagMatchMode};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;

// For the json! macro used in create_conclusion
use serde_json::json;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicBool, Ordering};

fn admin_db_url(url: &str) -> String {
    let pos = url.rfind('/').unwrap_or(url.len());
    let base = &url[..pos];
    format!("{}/postgres", base.trim_end_matches('/'))
}

fn target_db_name(url: &str) -> String {
    url.rsplit('/').next().unwrap_or("hmemory").to_string()
}

pub struct PgVectorStore {
    pool: PgPool,
    paradedb_available: AtomicBool,
}

fn vec_to_pgstring(v: &[f32]) -> String {
    let inner: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", inner.join(","))
}

fn parse_vector_text(s: &str) -> Option<Vec<f32>> {
    let s = s.trim();
    let s = s.strip_prefix('[').or_else(|| s.strip_prefix('{'))?;
    let s = s.strip_suffix(']').or_else(|| s.strip_suffix('}'))?;
    if s.is_empty() {
        return Some(Vec::new());
    }
    s.split(',')
        .map(|part| part.trim().parse::<f32>().ok())
        .collect()
}

impl PgVectorStore {
    pub async fn new(database_url: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if let Err(e) = PgPoolOptions::new()
            .max_connections(1)
            .connect(database_url)
            .await
        {
            let should_create = e
                .as_database_error()
                .and_then(|d| d.code())
                .is_some_and(|c| c == "3D000");

            if should_create {
                let admin_url = admin_db_url(database_url);
                let db_name = target_db_name(database_url);
                let admin_pool = PgPoolOptions::new()
                    .max_connections(1)
                    .connect(&admin_url)
                    .await?;
                sqlx::query(&format!("CREATE DATABASE \"{}\"", db_name))
                    .execute(&admin_pool)
                    .await?;
                admin_pool.close().await;
            }
        }

        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;
        let store = Self {
            pool,
            paradedb_available: AtomicBool::new(false),
        };
        // Auto-migrate: add new columns if they don't exist
        for col in [
            "ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '{}'",
            "ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'",
            "ADD COLUMN IF NOT EXISTS importance REAL DEFAULT 0.0",
            "ADD COLUMN IF NOT EXISTS source TEXT DEFAULT ''",
            "ADD COLUMN IF NOT EXISTS category TEXT DEFAULT ''",
            "ADD COLUMN IF NOT EXISTS feedback_positive INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS feedback_negative INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()",
        ] {
            let _ = sqlx::query(&format!("ALTER TABLE memories {col}"))
                .execute(&store.pool)
                .await;
        }
        Ok(store)
    }

        #[allow(clippy::too_many_arguments)]
    fn row_to_record(
        id: i64,
        session_id: String,
        content: String,
        embedding_str: String,
        tags_str: String,
        metadata_str: String,
        importance: f32,
        source: String,
        category: String,
        created_at: Option<DateTime<Utc>>,
        updated_at: Option<DateTime<Utc>>,
        feedback_positive: i32,
        feedback_negative: i32,
        score: f64,
    ) -> MemoryRecord {
        MemoryRecord {
            id,
            session_id,
            content,
            embedding: parse_vector_text(&embedding_str).unwrap_or_default(),
            tags: serde_json::from_str(&tags_str).unwrap_or(Value::Null),
            metadata: serde_json::from_str(&metadata_str).unwrap_or(Value::Null),
            importance,
            source,
            category,
            created_at: created_at.unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
            updated_at: updated_at.unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
            feedback_positive,
            feedback_negative,
            score,
        }
    }

    fn build_tag_keys(filter: &Option<Value>) -> Vec<String> {
        match filter {
            Some(Value::Object(m)) => m.keys().cloned().collect(),
            _ => vec![],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_search_where(
        session_id: Option<&str>,
        filter_tag_json: &Option<String>,
        tag_match_mode: &TagMatchMode,
        tag_keys: &[String],
        category_filter: &Option<String>,
        created_after: &Option<DateTime<Utc>>,
        created_before: &Option<DateTime<Utc>>,
        min_imp: &Option<f32>,
        start_param: usize,
    ) -> (String, Vec<String>) {
        let mut clauses = Vec::new();
        let mut params = Vec::new();
        let mut pi = start_param;

        if let Some(sid) = session_id {
            clauses.push(format!("AND session_id = ${pi}"));
            params.push(sid.to_string());
            pi += 1;
        }

        if let Some(tj) = filter_tag_json {
            match tag_match_mode {
                TagMatchMode::All => {
                    clauses.push(format!("AND (${pi}::jsonb IS NULL OR tags @> ${pi}::jsonb)"));
                    params.push(tj.clone());
                    pi += 1;
                }
                TagMatchMode::Any if !tag_keys.is_empty() => {
                    clauses.push(format!("AND (${pi}::jsonb IS NULL OR tags ?| ${next}::text[])", next = pi + 1));
                    params.push(tj.clone());
                    let keys_str = tag_keys.join(",");
                    params.push(keys_str);
                    pi += 2;
                }
                TagMatchMode::Any => {
                    // no keys to match, skip filter
                }
            }
        }

        if let Some(cf) = category_filter {
            clauses.push(format!("AND (${pi}::text IS NULL OR category = ${pi})"));
            params.push(cf.clone());
            pi += 1;
        }

        if let Some(ca) = created_after {
            clauses.push(format!("AND created_at >= ${pi}::timestamptz"));
            params.push(ca.to_rfc3339());
            pi += 1;
        }

        if let Some(cb) = created_before {
            clauses.push(format!("AND created_at <= ${pi}::timestamptz"));
            params.push(cb.to_rfc3339());
            pi += 1;
        }

        if let Some(mi) = min_imp {
            clauses.push(format!("AND importance >= ${pi}::real"));
            params.push(mi.to_string());
        }

        (clauses.join(" "), params)
    }

    fn vec_score_expr(decay: f64, vec_param: usize) -> String {
        if decay <= 0.0 {
            format!("(1 - (embedding <=> ${vec_param}::vector)) AS vec_score")
        } else {
            format!(
                "(1 - (embedding <=> ${vec_param}::vector)) * \
                 POW(0.5, GREATEST(0, EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0 / {decay})) AS vec_score"
            )
        }
    }

    const SELECT_COLS: &'static str = r#"
        SELECT id, session_id, content, embedding::text,
               COALESCE(tags::text, '{}') AS tags,
               COALESCE(metadata::text, '{}') AS metadata,
               COALESCE(importance, 0.0) AS importance,
               COALESCE(source, '') AS source,
               COALESCE(category, '') AS category,
               created_at, updated_at,
               COALESCE(feedback_positive, 0) AS feedback_positive,
               COALESCE(feedback_negative, 0) AS feedback_negative
    "#;

    async fn run_vector_search(
        &self,
        session_id: Option<&str>,
        embedding: &[f32],
        filters: &SearchFilters,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let vec_str = vec_to_pgstring(embedding);
        let tag_json = filters.tag_filter.as_ref().map(|v| v.to_string());
        let tag_keys = Self::build_tag_keys(&filters.tag_filter);

        let (where_clause, where_params) = Self::build_search_where(
            session_id,
            &tag_json,
            &filters.tag_match_mode,
            &tag_keys,
            &filters.category_filter,
            &filters.created_after,
            &filters.created_before,
            &filters.min_importance,
            2, // $1 = vec_str
        );

        let score_expr = Self::vec_score_expr(filters.decay_half_life_days, 1);
        let last_param = 2 + where_params.len();
        let sql = format!(
            "{} , {} FROM memories WHERE 1=1 {} ORDER BY vec_score DESC LIMIT ${last_param}",
            Self::SELECT_COLS,
            score_expr,
            where_clause,
        );

        let mut query = sqlx::query_as::<_, (i64, String, String, String, String, String, f32, String, String, Option<DateTime<Utc>>, Option<DateTime<Utc>>, i32, i32, f64)>(&sql)
            .bind(&vec_str);

        for p in &where_params {
            query = query.bind(p);
        }
        query = query.bind(limit as i64);

        let rows = query.fetch_all(&self.pool).await?;

        Ok(rows
            .into_iter()
            .map(|(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, score)| {
                Self::row_to_record(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, score)
            })
            .collect())
    }

    async fn run_bm25_search(
        &self,
        session_id: Option<&str>,
        query: &str,
        filters: &SearchFilters,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        if query.is_empty() {
            return Ok(vec![]);
        }

        let tag_json = filters.tag_filter.as_ref().map(|v| v.to_string());
        let tag_keys = Self::build_tag_keys(&filters.tag_filter);

        // BM25: content ||| $query_param, pdb.score(id) as bm25_score
        let bm25_param: u32 = if session_id.is_some() { 2 } else { 1 };
        let bm25_start = bm25_param as usize + 1;
        let (where_clause, where_params) = Self::build_search_where(
            session_id,
            &tag_json,
            &filters.tag_match_mode,
            &tag_keys,
            &filters.category_filter,
            &filters.created_after,
            &filters.created_before,
            &filters.min_importance,
            bm25_start,
        );

        let last_param = bm25_param + where_params.len() as u32 + 1;
        let sql = format!(
            "{} , pdb.score(id) AS bm25_score FROM memories WHERE 1=1 AND content ||| ${} {} ORDER BY bm25_score DESC LIMIT ${}",
            Self::SELECT_COLS,
            bm25_param,
            where_clause,
            last_param,
        );

        let mut query_builder = sqlx::query_as::<_, (i64, String, String, String, String, String, f32, String, String, Option<DateTime<Utc>>, Option<DateTime<Utc>>, i32, i32, f64)>(&sql);

        if let Some(sid) = session_id {
            query_builder = query_builder.bind(sid);
        }
        query_builder = query_builder.bind(query);
        for p in &where_params {
            query_builder = query_builder.bind(p);
        }
        query_builder = query_builder.bind(limit as i64);

        let rows = query_builder.fetch_all(&self.pool).await?;

        Ok(rows
            .into_iter()
            .map(|(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, score)| {
                Self::row_to_record(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, score)
            })
            .collect())
    }

    fn fuse_results(
        vector_results: Vec<MemoryRecord>,
        bm25_results: Vec<MemoryRecord>,
        alpha: f32,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let alpha_f64 = alpha as f64;
        let mut map: HashMap<i64, (f64, f64, MemoryRecord)> = HashMap::new();

        for r in vector_results {
            let vec_score = r.score;
            let mut base = r;
            base.score = 0.0;
            map.insert(base.id, (vec_score, 0.0, base));
        }

        for r in bm25_results {
            let id = r.id;
            let bm25_score = r.score;
            let content = r.content;
            let session_id = r.session_id;
            let embedding = r.embedding;
            let tags = r.tags;
            let metadata = r.metadata;
            let importance = r.importance;
            let source = r.source;
            let category = r.category;
            let created_at = r.created_at;
            let updated_at = r.updated_at;
            let feedback_positive = r.feedback_positive;
            let feedback_negative = r.feedback_negative;

            match map.entry(id) {
                Entry::Occupied(mut occ) => {
                    let (_, existing_bm25, record) = occ.get_mut();
                    *existing_bm25 = bm25_score;
                    record.content = content;
                    record.tags = tags;
                    record.metadata = metadata;
                    record.importance = importance;
                    record.source = source;
                    record.category = category;
                    record.created_at = created_at;
                    record.updated_at = updated_at;
                    record.feedback_positive = feedback_positive;
                    record.feedback_negative = feedback_negative;
                    record.embedding = embedding;
                }
                Entry::Vacant(vac) => {
                    vac.insert((
                        0.0,
                        bm25_score,
                        MemoryRecord {
                            id,
                            session_id,
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
                            score: 0.0,
                        },
                    ));
                }
            }
        }

        let mut result: Vec<MemoryRecord> = map
            .into_values()
            .map(|(vec_score, bm25_score, mut record)| {
                record.score = alpha_f64 * vec_score + (1.0 - alpha_f64) * bm25_score;
                record
            })
            .collect();

        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(limit);
        result
    }
}

#[async_trait]
impl MemoryStore for PgVectorStore {
    async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await?;

        match sqlx::query("CREATE EXTENSION IF NOT EXISTS paradedb")
            .execute(&self.pool)
            .await
        {
            Ok(_) => {
                tracing::info!("ParadeDB available, BM25 search enabled");
                self.paradedb_available.store(true, Ordering::Relaxed);
            }
            Err(e) => {
                tracing::warn!("ParadeDB not available, BM25 search disabled: {e}");
            }
        }

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1536) NOT NULL,
                tags JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                importance REAL DEFAULT 0.0,
                source TEXT DEFAULT '',
                category TEXT DEFAULT '',
                feedback_positive INTEGER DEFAULT 0,
                feedback_negative INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL DEFAULT 'per-session',
                path_or_repo TEXT NOT NULL DEFAULT '',
                turn_count BIGINT DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT now(),
                last_active_at TIMESTAMPTZ DEFAULT now()
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Additive migration for existing tables
        for col in [
            "ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '{}'",
            "ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'",
            "ADD COLUMN IF NOT EXISTS importance REAL DEFAULT 0.0",
            "ADD COLUMN IF NOT EXISTS source TEXT DEFAULT ''",
            "ADD COLUMN IF NOT EXISTS category TEXT DEFAULT ''",
            "ADD COLUMN IF NOT EXISTS feedback_positive INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS feedback_negative INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()",
        ] {
            sqlx::query(&format!("ALTER TABLE memories {col}"))
                .execute(&self.pool)
                .await?;
        }

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories (session_id)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN (tags)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
        )
        .execute(&self.pool)
        .await?;

        if self.paradedb_available.load(Ordering::Relaxed) {
            sqlx::query(
                r#"CREATE INDEX IF NOT EXISTS memories_bm25_idx ON memories
                   USING bm25 (id, content, tags, source, session_id)
                   WITH (key_field='id')"#,
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

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
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let vec_str = vec_to_pgstring(embedding);
        let tags_str = tags.to_string();
        let metadata_str = metadata.to_string();

        sqlx::query(
            r#"
            INSERT INTO memories (session_id, content, embedding, tags, metadata, importance, source, category)
            VALUES ($1, $2, $3::vector, $4::jsonb, $5::jsonb, $6, $7, $8)
            "#,
        )
        .bind(session_id)
        .bind(content)
        .bind(&vec_str)
        .bind(&tags_str)
        .bind(&metadata_str)
        .bind(importance)
        .bind(source)
        .bind(category)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn search(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = filters.limit;
        let session_id = filters.session_id.as_deref();

        let vector_results = self
            .run_vector_search(session_id, embedding, filters, limit * 2)
            .await?;

        let bm25_results = if self.paradedb_available.load(Ordering::Relaxed)
            && filters.alpha < 1.0
            && !filters.query.is_empty()
        {
            self.run_bm25_search(session_id, &filters.query, filters, limit * 2)
                .await?
        } else {
            vec![]
        };

        let adjusted_limit = match filters.level {
            RetrievalLevel::Summary => 1.min(limit),
            RetrievalLevel::Overview => 3.min(limit),
            RetrievalLevel::Details => limit,
        };

        Ok(Self::fuse_results(
            vector_results,
            bm25_results,
            filters.alpha,
            adjusted_limit,
        ))
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("DELETE FROM memories WHERE session_id = $1")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn list_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query_as::<_, (i64, String, String, String, String, String, f32, String, String, Option<DateTime<Utc>>, Option<DateTime<Utc>>, i32, i32)>(
            r#"
            SELECT id, session_id, content, embedding::text,
                   COALESCE(tags::text, '{}') AS tags,
                   COALESCE(metadata::text, '{}') AS metadata,
                   COALESCE(importance, 0.0) AS importance,
                   COALESCE(source, '') AS source,
                   COALESCE(category, '') AS category,
                   created_at, updated_at,
                   COALESCE(feedback_positive, 0) AS feedback_positive,
                   COALESCE(feedback_negative, 0) AS feedback_negative
            FROM memories
            WHERE session_id = $1
            ORDER BY id DESC
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative)| {
                Self::row_to_record(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, 0.0)
            })
            .collect())
    }

    async fn delete_by_id(&self, id: i64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("DELETE FROM memories WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn list_memories(
        &self,
        session_id: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<MemoryRecord>, u64), Box<dyn std::error::Error + Send + Sync>> {
        let (where_clause, params) = match session_id {
            Some(sid) => ("WHERE session_id = $1".to_string(), vec![sid.to_string()]),
            None => ("".to_string(), vec![]),
        };

        let count_sql = format!("SELECT COUNT(*) FROM memories {}", where_clause);
        let mut count_query = sqlx::query_scalar::<_, i64>(&count_sql);
        for p in &params {
            count_query = count_query.bind(p);
        }
        let total: i64 = count_query.fetch_one(&self.pool).await.unwrap_or(0);

        let data_sql = if session_id.is_some() {
            format!(
                "{} FROM memories WHERE session_id = $1 ORDER BY id DESC LIMIT $2 OFFSET $3",
                Self::SELECT_COLS,
            )
        } else {
            format!(
                "{} FROM memories ORDER BY id DESC LIMIT $1 OFFSET $2",
                Self::SELECT_COLS,
            )
        };

        let mut data_query = sqlx::query_as::<_, (i64, String, String, String, String, String, f32, String, String, Option<DateTime<Utc>>, Option<DateTime<Utc>>, i32, i32)>(&data_sql);
        if let Some(sid) = session_id {
            data_query = data_query.bind(sid);
        }
        data_query = data_query.bind(limit as i64);
        data_query = data_query.bind(offset as i64);

        let rows = data_query.fetch_all(&self.pool).await?;

        let records = rows
            .into_iter()
            .map(|(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative)| {
                Self::row_to_record(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, 0.0)
            })
            .collect();

        Ok((records, total as u64))
    }

    async fn get_by_id(
        &self,
        id: i64,
    ) -> Result<Option<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let sql = format!("{} FROM memories WHERE id = $1", Self::SELECT_COLS);
        let rows = sqlx::query_as::<_, (i64, String, String, String, String, String, f32, String, String, Option<DateTime<Utc>>, Option<DateTime<Utc>>, i32, i32)>(&sql)
            .bind(id)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().next().map(
            |(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative)| {
                Self::row_to_record(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, 0.0)
            },
        ))
    }

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
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let mut sets = Vec::new();
        let mut params: Vec<String> = Vec::new();
        let mut pi = 1usize;

        if let Some(ref c) = content {
            sets.push(format!("content = ${pi}"));
            params.push(c.clone());
            pi += 1;
        }
        if let Some(ref t) = tags {
            sets.push(format!("tags = ${pi}::jsonb"));
            params.push(t.to_string());
            pi += 1;
        }
        if let Some(ref m) = metadata {
            sets.push(format!("metadata = ${pi}::jsonb"));
            params.push(m.to_string());
            pi += 1;
        }
        if let Some(ref i) = importance {
            sets.push(format!("importance = ${pi}"));
            params.push(i.to_string());
            pi += 1;
        }
        if let Some(ref c) = category {
            sets.push(format!("category = ${pi}"));
            params.push(c.clone());
            pi += 1;
        }
        if let Some(ref s) = source {
            sets.push(format!("source = ${pi}"));
            params.push(s.clone());
            pi += 1;
        }

        if sets.is_empty() {
            return Ok(false);
        }

        sets.push("updated_at = now()".to_string());
        let sql = format!(
            "UPDATE memories SET {} WHERE id = ${}",
            sets.join(", "),
            pi
        );

        let mut query = sqlx::query(&sql);
        for p in &params {
            query = query.bind(p);
        }
        query = query.bind(id);

        let result = query.execute(&self.pool).await?;
        Ok(result.rows_affected() > 0)
    }

    async fn add_feedback(
        &self,
        id: i64,
        useful: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let col = if useful { "feedback_positive" } else { "feedback_negative" };
        let sql = format!("UPDATE memories SET {col} = {col} + 1, updated_at = now() WHERE id = $1");
        sqlx::query(&sql).bind(id).execute(&self.pool).await?;
        Ok(())
    }

    async fn backup(
        &self,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let sql = format!("{} FROM memories ORDER BY id", Self::SELECT_COLS);
        let rows = sqlx::query_as::<_, (i64, String, String, String, String, String, f32, String, String, Option<DateTime<Utc>>, Option<DateTime<Utc>>, i32, i32)>(&sql)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows
            .into_iter()
            .map(|(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative)| {
                Self::row_to_record(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, 0.0)
            })
            .collect())
    }

    async fn restore(
        &self,
        records: &[MemoryRecord],
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut count = 0u64;
        for r in records {
            let vec_str = vec_to_pgstring(&r.embedding);
            let tags_str = r.tags.to_string();
            let metadata_str = r.metadata.to_string();
            let result = sqlx::query(
                r#"
                INSERT INTO memories (session_id, content, embedding, tags, metadata, importance, source, category, created_at, updated_at)
                VALUES ($1, $2, $3::vector, $4::jsonb, $5::jsonb, $6, $7, $8, $9, $10)
                "#,
            )
            .bind(&r.session_id)
            .bind(&r.content)
            .bind(&vec_str)
            .bind(&tags_str)
            .bind(&metadata_str)
            .bind(r.importance)
            .bind(&r.source)
            .bind(&r.category)
            .bind(r.created_at)
            .bind(r.updated_at)
            .execute(&self.pool)
            .await;
            if result.is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    async fn resolve_session(
        &self,
        session_id: &str,
        strategy: &str,
        path_or_repo: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            r#"INSERT INTO sessions (session_id, strategy, path_or_repo, created_at, last_active_at, turn_count)
               VALUES ($1, $2, $3, now(), now(), 0)
               ON CONFLICT (session_id) DO UPDATE SET last_active_at = now()"#,
        )
        .bind(session_id)
        .bind(strategy)
        .bind(path_or_repo)
        .execute(&self.pool)
        .await?;
        Ok(session_id.to_string())
    }

    async fn get_session_status(
        &self,
        session_id: &str,
    ) -> Result<SessionStatus, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query_as::<_, (String, String, String, i64, Option<DateTime<Utc>>, Option<DateTime<Utc>>)>(
            "SELECT session_id, strategy, path_or_repo, turn_count, created_at, last_active_at FROM sessions WHERE session_id = $1",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| format!("session {session_id} not found"))?;

        let (sid, strategy, path_or_repo, turn_count, created_at, last_active_at) = row;
        let last_active = last_active_at.unwrap_or_else(Utc::now);
        let is_cold = turn_count < 3
            || (Utc::now() - last_active).num_minutes() > 60;

        Ok(SessionStatus {
            session_id: sid,
            strategy,
            path_or_repo,
            turn_count,
            is_cold,
            created_at: created_at.unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
            last_active_at: last_active,
        })
    }

    async fn increment_turn_count(
        &self,
        session_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE sessions SET turn_count = turn_count + 1, last_active_at = now() WHERE session_id = $1")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_base_context(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query_as::<_, (i64, String, String, String, String, String, f32, String, String, Option<DateTime<Utc>>, Option<DateTime<Utc>>, i32, i32)>(
            r#"
            SELECT id, session_id, content, embedding::text,
                   COALESCE(tags::text, '{}') AS tags,
                   COALESCE(metadata::text, '{}') AS metadata,
                   COALESCE(importance, 0.0) AS importance,
                   COALESCE(source, '') AS source,
                   COALESCE(category, '') AS category,
                   created_at, updated_at,
                   COALESCE(feedback_positive, 0) AS feedback_positive,
                   COALESCE(feedback_negative, 0) AS feedback_negative
            FROM memories
            WHERE session_id = $1 AND importance >= 0.5
            ORDER BY created_at DESC
            LIMIT $2
            "#,
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative)| {
                Self::row_to_record(id, session_id, content, embedding_str, tags_str, metadata_str, importance, source, category, created_at, updated_at, feedback_positive, feedback_negative, 0.0)
            })
            .collect())
    }

    async fn create_conclusion(
        &self,
        session_id: &str,
        content: &str,
        category: &str,
        confidence: f32,
        source_turn_ids: &[i64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let vec_str = vec_to_pgstring(&[]);
        let metadata = json!({ "conclusion": true, "confidence": confidence, "source_turn_ids": source_turn_ids });

        sqlx::query(
            r#"
            INSERT INTO memories (session_id, content, embedding, tags, metadata, importance, source, category)
            VALUES ($1, $2, $3::vector, $4::jsonb, $5::jsonb, $6, $7, $8)
            "#,
        )
        .bind(session_id)
        .bind(content)
        .bind(&vec_str)
        .bind(json!({"type": "conclusion"}).to_string())
        .bind(metadata.to_string())
        .bind(confidence)
        .bind("conclusion")
        .bind(category)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_to_pgstring_roundtrip() {
        let v = vec![1.0, 2.5, -3.0, 0.0, 42.0];
        let s = vec_to_pgstring(&v);
        let parsed = parse_vector_text(&s).unwrap();
        assert_eq!(parsed.len(), v.len());
        for (a, b) in parsed.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vec_to_pgstring_empty() {
        let s = vec_to_pgstring(&[]);
        assert_eq!(s, "[]");
        assert_eq!(parse_vector_text("[]").unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_parse_vector_text_bracket_variants() {
        let expected = vec![1.0, 2.0, 3.0];
        assert_eq!(parse_vector_text("[1,2,3]").unwrap(), expected);
        assert_eq!(parse_vector_text("{1,2,3}").unwrap(), expected);
        assert_eq!(parse_vector_text("[1.0,2.0,3.0]").unwrap(), expected);
    }

    #[test]
    fn test_parse_vector_text_malformed() {
        assert!(parse_vector_text("").is_none());
        assert!(parse_vector_text("[").is_none());
        assert!(parse_vector_text("no-brackets").is_none());
        assert!(parse_vector_text("[]]").is_none());
    }

    fn make_rec(id: i64, content: &str, score: f64) -> MemoryRecord {
        MemoryRecord {
            id,
            session_id: "s".into(),
            content: content.into(),
            embedding: vec![],
            tags: Value::Null,
            metadata: Value::Null,
            importance: 0.0,
            source: "".into(),
            category: "".into(),
            created_at: DateTime::from_timestamp(0, 0).unwrap(),
            updated_at: DateTime::from_timestamp(0, 0).unwrap(),
            feedback_positive: 0,
            feedback_negative: 0,
            score,
        }
    }

    #[test]
    fn test_fuse_results_both_sources() {
        let alpha = 0.5f32;
        let limit = 5;

        let vec_records = vec![
            make_rec(1, "vec1", 0.9),
            make_rec(2, "vec2", 0.7),
        ];

        let bm25_records = vec![
            make_rec(1, "vec1", 0.6),
            make_rec(3, "bm25_only", 0.8),
        ];

        let fused = PgVectorStore::fuse_results(vec_records, bm25_records, alpha, limit);

        assert_eq!(fused.len(), 3);
        assert!((fused.iter().find(|r| r.id == 1).unwrap().score - 0.75).abs() < 1e-6);
        assert!((fused.iter().find(|r| r.id == 2).unwrap().score - 0.35).abs() < 1e-6);
        assert!((fused.iter().find(|r| r.id == 3).unwrap().score - 0.40).abs() < 1e-6);

        for w in fused.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_fuse_results_pure_vector() {
        let alpha = 1.0f32;
        let limit = 5;
        let vec_records = vec![make_rec(1, "a", 0.9)];
        let bm25_records = vec![make_rec(2, "b", 0.6)];
        let fused = PgVectorStore::fuse_results(vec_records, bm25_records, alpha, limit);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].id, 1);
        assert!((fused[0].score - 0.9).abs() < 1e-6);
        assert_eq!(fused[1].id, 2);
        assert!((fused[1].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_results_pure_bm25() {
        let alpha = 0.0f32;
        let limit = 5;
        let vec_records = vec![make_rec(1, "a", 0.9)];
        let bm25_records = vec![make_rec(2, "b", 0.6)];
        let fused = PgVectorStore::fuse_results(vec_records, bm25_records, alpha, limit);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].id, 2);
        assert!((fused[0].score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_results_limit() {
        let alpha = 0.5f32;
        let limit = 1;
        let vec_records = vec![make_rec(1, "a", 0.9), make_rec(2, "b", 0.8)];
        let fused = PgVectorStore::fuse_results(vec_records, vec![], alpha, limit);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, 1);
    }
}