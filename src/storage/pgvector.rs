use super::{
    CompactReport, DetailedMemoryStats, LabelCount, MemoryLink, MemoryRecord, MemoryRecordConsolid,
    MemoryStats, MemoryStore, NumericStats, ProfileDetailedStats, ProfileStatus, ProfileTierCounts,
    SearchFilters, SnapshotDiff, TagMatchMode, TierCounts,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;

use serde_json::json;
use sqlx::PgPool;
use sqlx::Row;
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
    if s.ends_with(']') {
        let s = &s[..s.len() - 1];
        if s.is_empty() {
            return Some(Vec::new());
        }
        s.split(',')
            .map(|part| part.trim().parse::<f32>().ok())
            .collect()
    } else if s.ends_with('}') {
        let s = &s[..s.len() - 1];
        if s.is_empty() {
            return Some(Vec::new());
        }
        s.split(',')
            .map(|part| part.trim().parse::<f32>().ok())
            .collect()
    } else {
        None
    }
}

fn row_to_record_from_row(row: &sqlx::postgres::PgRow) -> MemoryRecord {
    MemoryRecord {
        id: row.get("id"),
        profile: row.get("profile"),
        content: row.get("content"),
        embedding: parse_vector_text(row.get::<&str, _>("embedding")).unwrap_or_default(),
        tags: serde_json::from_str(row.get::<&str, _>("tags")).unwrap_or(Value::Null),
        metadata: serde_json::from_str(row.get::<&str, _>("metadata")).unwrap_or(Value::Null),
        importance: row.get("importance"),
        source: row.get("source"),
        category: row.get("category"),
        created_at: row
            .get::<Option<DateTime<Utc>>, _>("created_at")
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
        updated_at: row
            .get::<Option<DateTime<Utc>>, _>("updated_at")
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
        feedback_positive: row.get("feedback_positive"),
        feedback_negative: row.get("feedback_negative"),
        expires_at: row.try_get("expires_at").ok().flatten(),
        immortal: row.try_get("immortal").unwrap_or(false),
        score: row.try_get("score").unwrap_or(0.0),
        access_count: row.try_get("access_count").unwrap_or(0),
        last_accessed_at: row.try_get("last_accessed_at").ok().flatten(),
        trust_score: row.try_get("trust_score").unwrap_or(0.5),
        event_date: row.try_get("event_date").ok().flatten(),
        reminder_interval: row.try_get("reminder_interval").ok(),
        reminder_at: row.try_get("reminder_at").ok().flatten(),
        reminder_sent: row.try_get("reminder_sent").unwrap_or(false),
    }
}

fn parse_interval_delta(interval: &str) -> Option<chrono::TimeDelta> {
    let (num_str, unit) = interval.split_at(interval.len().saturating_sub(1));
    let num: i64 = num_str.parse().ok()?;
    match unit {
        "m" => Some(chrono::TimeDelta::minutes(num)),
        "h" => Some(chrono::TimeDelta::hours(num)),
        "d" => Some(chrono::TimeDelta::days(num)),
        _ => None,
    }
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
                let _ = sqlx::query(&format!("CREATE DATABASE \"{}\"", db_name))
                    .execute(&admin_pool)
                    .await;
                admin_pool.close().await;
            }
        }

        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        Ok(Self {
            pool,
            paradedb_available: AtomicBool::new(false),
        })
    }

    fn build_tag_keys(filter: &Option<Value>) -> Vec<String> {
        match filter {
            Some(Value::Object(m)) => m.keys().cloned().collect(),
            _ => vec![],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_search_where(
        profile: Option<&str>,
        filter_tag_json: &Option<String>,
        tag_match_mode: &TagMatchMode,
        tag_keys: &[String],
        category_filter: &Option<String>,
        created_after: &Option<DateTime<Utc>>,
        created_before: &Option<DateTime<Utc>>,
        min_importance: &Option<f32>,
        pi: usize,
    ) -> (String, Vec<String>) {
        let mut clauses = Vec::new();
        let mut params = Vec::new();

        if let Some(sid) = profile {
            clauses.push(format!("AND profile = ${pi}"));
            params.push(sid.to_string());
        }

        if let Some(tj) = filter_tag_json {
            if *tj != "null" {
                match tag_match_mode {
                    TagMatchMode::All => {
                        clauses.push(format!(
                            "AND (${pi}::jsonb IS NULL OR tags @> ${pi}::jsonb)"
                        ));
                        params.push(tj.clone());
                    }
                    TagMatchMode::Any if !tag_keys.is_empty() => {
                        clauses.push(format!(
                            "AND (${pi}::jsonb IS NULL OR tags ?| ${pi}::jsonb)"
                        ));
                        let keys_str = tag_keys.join(",");
                        params.push(format!("{{{}}}", keys_str));
                    }
                    TagMatchMode::Any => {}
                }
            }
        }

        if let Some(cf) = category_filter {
            if !cf.is_empty() {
                clauses.push(format!("AND (${pi}::text IS NULL OR category = ${pi})"));
                params.push(cf.clone());
            }
        }
        if let Some(ca) = created_after {
            clauses.push(format!("AND created_at >= ${pi}::timestamptz"));
            params.push(ca.to_rfc3339());
        }
        if let Some(cb) = created_before {
            clauses.push(format!("AND created_at <= ${pi}::timestamptz"));
            params.push(cb.to_rfc3339());
        }
        if let Some(mi) = min_importance {
            clauses.push(format!("AND importance >= ${pi}::real"));
            params.push(mi.to_string());
        }

        let where_clause = clauses.join(" ");
        (where_clause, params)
    }

    fn vec_score_expr(decay: f64, vec_param: usize) -> String {
        if decay <= 0.0 {
            format!("(1 - (embedding <=> ${vec_param}::vector)) AS score")
        } else {
            format!(
                "(1 - (embedding <=> ${vec_param}::vector)) * \
                 POW(0.5, GREATEST(0, EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0 / {decay})) AS score"
            )
        }
    }

    const SELECT_COLS: &'static str = r#"
        SELECT id, profile, content, embedding::text,
               COALESCE(tags::text, '{}') AS tags,
               COALESCE(metadata::text, '{}') AS metadata,
               COALESCE(importance, 0.0) AS importance,
               COALESCE(source, '') AS source,
               COALESCE(category, '') AS category,
               created_at, updated_at,
               COALESCE(feedback_positive, 0) AS feedback_positive,
               COALESCE(feedback_negative, 0) AS feedback_negative,
               COALESCE(access_count, 0) AS access_count,
               last_accessed_at,
               COALESCE(trust_score, 0.5) AS trust_score,
               expires_at,
               COALESCE(immortal, false) AS immortal,
               event_date,
               reminder_interval,
               reminder_at,
               COALESCE(reminder_sent, false) AS reminder_sent
    "#;

    async fn run_vector_search_on(
        &self,
        table: &str,
        profile: Option<&str>,
        embedding: &[f32],
        filters: &SearchFilters,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let vec_str = vec_to_pgstring(embedding);
        let tag_json = filters.tag_filter.as_ref().map(|v| v.to_string());
        let tag_keys = Self::build_tag_keys(&filters.tag_filter);

        let (where_clause, where_params) = Self::build_search_where(
            profile,
            &tag_json,
            &filters.tag_match_mode,
            &tag_keys,
            &filters.category_filter,
            &filters.created_after,
            &filters.created_before,
            &filters.min_importance,
            2,
        );

        let score_expr = Self::vec_score_expr(filters.decay_half_life_days, 1);
        let last_param = 2 + where_params.len();
        let sql = format!(
            "{} , {} FROM {} WHERE 1=1 {} ORDER BY score DESC LIMIT ${last_param}",
            Self::SELECT_COLS,
            score_expr,
            table,
            where_clause,
        );

        let mut query = sqlx::query(&sql).bind(&vec_str);
        for p in &where_params {
            query = query.bind(p);
        }
        query = query.bind(limit as i64);

        let rows = query.fetch_all(&self.pool).await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn run_bm25_search_on(
        &self,
        table: &str,
        profile: Option<&str>,
        query_text: &str,
        filters: &SearchFilters,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        if query_text.is_empty() {
            return Ok(vec![]);
        }

        let tag_json = filters.tag_filter.as_ref().map(|v| v.to_string());
        let tag_keys = Self::build_tag_keys(&filters.tag_filter);

        let bm25_param: u32 = if profile.is_some() { 2 } else { 1 };
        let bm25_start = bm25_param as usize + 1;
        let (where_clause, where_params) = Self::build_search_where(
            profile,
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
            "{} , pdb.score(id)::double precision AS score FROM {} WHERE 1=1 AND content ||| ${} {} ORDER BY score DESC LIMIT ${}",
            Self::SELECT_COLS,
            table,
            bm25_param,
            where_clause,
            last_param,
        );

        let mut query_builder = sqlx::query(&sql);
        if let Some(sid) = profile {
            query_builder = query_builder.bind(sid);
        }
        query_builder = query_builder.bind(query_text);
        for p in &where_params {
            query_builder = query_builder.bind(p);
        }
        query_builder = query_builder.bind(limit as i64);

        let rows = query_builder.fetch_all(&self.pool).await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    fn compute_reminder_at(
        event_date: Option<DateTime<Utc>>,
        interval: &str,
    ) -> Option<DateTime<Utc>> {
        if let Ok(dt) = DateTime::parse_from_rfc3339(interval) {
            return Some(dt.with_timezone(&Utc));
        }
        let (num_str, unit) = interval.split_at(interval.len().saturating_sub(1));
        let num: i64 = num_str.parse().ok()?;
        let delta = match unit {
            "m" => chrono::TimeDelta::minutes(num),
            "h" => chrono::TimeDelta::hours(num),
            "d" => chrono::TimeDelta::days(num),
            _ => return None,
        };
        Some(event_date? - delta)
    }

    fn compute_reminder_in(interval: &str) -> Option<DateTime<Utc>> {
        let delta = parse_interval_delta(interval)?;
        Some(Utc::now() + delta)
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
            match map.entry(id) {
                Entry::Occupied(mut occ) => {
                    let (_, existing_bm25, record) = occ.get_mut();
                    *existing_bm25 = bm25_score;
                    record.content = r.content;
                    record.tags = r.tags;
                    record.metadata = r.metadata;
                    record.importance = r.importance;
                    record.source = r.source;
                    record.category = r.category;
                    record.created_at = r.created_at;
                    record.updated_at = r.updated_at;
                    record.feedback_positive = r.feedback_positive;
                    record.feedback_negative = r.feedback_negative;
                    record.embedding = r.embedding;
                }
                Entry::Vacant(vac) => {
                    vac.insert((0.0, bm25_score, r));
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

        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result.truncate(limit);
        result
    }

    async fn reconsolidate_to_fresh(
        &self,
        record: &MemoryRecord,
        immortal: bool,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        let exists: bool =
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM memories_fresh WHERE id = $1)")
                .bind(record.id)
                .fetch_one(&self.pool)
                .await
                .unwrap_or(false);
        if exists {
            if immortal {
                sqlx::query("UPDATE memories_fresh SET immortal = true WHERE id = $1")
                    .bind(record.id)
                    .execute(&self.pool)
                    .await
                    .ok();
            }
            return Ok(record.id);
        }
        let expires = Utc::now() + chrono::TimeDelta::hours(24);
        let vec_str = vec_to_pgstring(&record.embedding);
        let tags_str = record.tags.to_string();
        let meta_str = record.metadata.to_string();

        let result = sqlx::query(
            r#"INSERT INTO memories_fresh
               (id, profile, content, embedding, tags, metadata, importance, source, category,
                expires_at, immortal, created_at, updated_at, conversation_id, turn_range, user_msg, assistant_msg)
               VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6::jsonb, $7, $8, $9,
                       $10, $11, $12, $13, 'reconsolidated', '', '', '')
               ON CONFLICT (id) DO UPDATE SET expires_at = EXCLUDED.expires_at, importance = EXCLUDED.importance, immortal = EXCLUDED.immortal
               RETURNING id"#,
        )
        .bind(record.id)
        .bind(&record.profile)
        .bind(&record.content)
        .bind(&vec_str)
        .bind(&tags_str)
        .bind(&meta_str)
        .bind(record.importance)
        .bind(&record.source)
        .bind(&record.category)
        .bind(expires)
        .bind(immortal)
        .bind(record.created_at)
        .bind(Utc::now())
        .fetch_optional(&self.pool)
        .await?;

        match result {
            Some(row) => Ok(row.get::<i64, _>("id")),
            None => Ok(record.id),
        }
    }

    async fn set_immortal(&self, id: i64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE memories_deep SET immortal = true WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn do_schedule_review(
        &self,
        id: i64,
        importance: f32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let intervals: &[&str] = if importance >= 0.9 {
            &["1h", "24h", "7d", "30d"]
        } else if importance >= 0.8 {
            &["1h", "24h", "7d"]
        } else if importance >= 0.7 {
            &["1h", "24h"]
        } else {
            return Ok(());
        };
        let schedule_json = serde_json::json!({
            "review_schedule": intervals,
            "review_stage": 0,
            "next_review": intervals.first().copied().unwrap_or("1h"),
        });
        sqlx::query(
            "UPDATE memories_deep SET reminder_interval = $1, reminder_at = now() + $2::interval, metadata = metadata || $3::jsonb WHERE id = $4",
        )
        .bind(intervals[0])
        .bind(intervals[0])
        .bind(schedule_json.to_string())
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
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

        // Rename legacy memories table to memories_deep
        sqlx::query("ALTER TABLE IF EXISTS memories RENAME TO memories_deep")
            .execute(&self.pool)
            .await
            .ok();

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories_deep (
                id BIGSERIAL PRIMARY KEY,
                profile TEXT NOT NULL,
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
                updated_at TIMESTAMPTZ DEFAULT now(),
                expires_at TIMESTAMPTZ DEFAULT NULL,
                immortal BOOLEAN DEFAULT false,
                access_count INTEGER DEFAULT 0,
                last_accessed_at TIMESTAMPTZ DEFAULT NULL,
                trust_score REAL DEFAULT 0.5,
                event_date TIMESTAMPTZ DEFAULT NULL,
                reminder_interval TEXT DEFAULT NULL,
                reminder_at TIMESTAMPTZ DEFAULT NULL,
                reminder_sent BOOLEAN DEFAULT FALSE
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create memories_fresh (ephemeral working memory)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories_fresh (
                id BIGSERIAL PRIMARY KEY,
                profile TEXT NOT NULL,
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
                updated_at TIMESTAMPTZ DEFAULT now(),
                expires_at TIMESTAMPTZ DEFAULT NULL,
                immortal BOOLEAN DEFAULT false,
                access_count INTEGER DEFAULT 0,
                last_accessed_at TIMESTAMPTZ DEFAULT NULL,
                trust_score REAL DEFAULT 0.5,
                event_date TIMESTAMPTZ DEFAULT NULL,
                reminder_interval TEXT DEFAULT NULL,
                reminder_at TIMESTAMPTZ DEFAULT NULL,
                reminder_sent BOOLEAN DEFAULT FALSE,
                conversation_id TEXT NOT NULL DEFAULT '',
                turn_range TEXT NOT NULL DEFAULT '',
                user_msg TEXT NOT NULL DEFAULT '',
                assistant_msg TEXT NOT NULL DEFAULT ''
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create memories_consolid (compressed summaries)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories_consolid (
                id BIGSERIAL PRIMARY KEY,
                profile TEXT NOT NULL,
                summary TEXT NOT NULL,
                embedding vector(1536) NOT NULL,
                source_ids BIGINT[] DEFAULT '{}',
                depth TEXT NOT NULL DEFAULT 'shallow',
                tags JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                insight_score REAL DEFAULT 0.0,
                importance REAL DEFAULT 0.0,
                created_at TIMESTAMPTZ DEFAULT now(),
                last_consolidated_at TIMESTAMPTZ DEFAULT now(),
                access_count INTEGER DEFAULT 0
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Migration: rename session_id to profile if the old column exists
        sqlx::query(
            r#"
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='memories_deep' AND column_name='session_id') THEN
                    ALTER TABLE memories_deep RENAME COLUMN session_id TO profile;
                END IF;
            END $$;
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Add missing columns to memories_deep
        for col in [
            "ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '{}'",
            "ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'",
            "ADD COLUMN IF NOT EXISTS importance REAL DEFAULT 0.0",
            "ADD COLUMN IF NOT EXISTS source TEXT DEFAULT ''",
            "ADD COLUMN IF NOT EXISTS category TEXT DEFAULT ''",
            "ADD COLUMN IF NOT EXISTS feedback_positive INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS feedback_negative INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()",
            "ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ DEFAULT NULL",
            "ADD COLUMN IF NOT EXISTS immortal BOOLEAN DEFAULT false",
            "ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ DEFAULT NULL",
            "ADD COLUMN IF NOT EXISTS trust_score REAL DEFAULT 0.5",
            "ADD COLUMN IF NOT EXISTS event_date TIMESTAMPTZ DEFAULT NULL",
            "ADD COLUMN IF NOT EXISTS reminder_interval TEXT DEFAULT NULL",
            "ADD COLUMN IF NOT EXISTS reminder_at TIMESTAMPTZ DEFAULT NULL",
            "ADD COLUMN IF NOT EXISTS reminder_sent BOOLEAN DEFAULT FALSE",
        ] {
            let _ = sqlx::query(&format!("ALTER TABLE memories_deep {col}"))
                .execute(&self.pool)
                .await;
        }

        // Profile stats table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS profile_stats (
                profile TEXT PRIMARY KEY,
                turn_count BIGINT DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT now(),
                last_active_at TIMESTAMPTZ DEFAULT now()
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Memory links table
        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS memory_links (
                id_a BIGINT NOT NULL,
                id_b BIGINT NOT NULL,
                relation_type TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (id_a, id_b)
            )"#,
        )
        .execute(&self.pool)
        .await?;

        // Indexes on memories_deep
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_deep_profile ON memories_deep (profile)",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_deep_tags ON memories_deep USING GIN (tags)",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_deep_category ON memories_deep (category)",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_deep_embedding ON memories_deep USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
        )
        .execute(&self.pool)
        .await?;

        // Indexes on memories_fresh
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_fresh_profile ON memories_fresh (profile)",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_fresh_tags ON memories_fresh USING GIN (tags)",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_fresh_created ON memories_fresh (created_at)",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_fresh_embedding ON memories_fresh USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
        )
        .execute(&self.pool)
            .await?;

        // Indexes on memories_consolid
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_consolid_profile ON memories_consolid (profile)")
            .execute(&self.pool)
            .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memories_consolid_embedding ON memories_consolid USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
        )
        .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_links_a ON memory_links (id_a)")
            .execute(&self.pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_links_b ON memory_links (id_b)")
            .execute(&self.pool)
            .await?;

        // BM25 indexes if ParadeDB available
        if self.paradedb_available.load(Ordering::Relaxed) {
            for tbl in &["memories_deep", "memories_fresh"] {
                let idx_name = format!("{}_bm25_idx", tbl);
                let _ = sqlx::query(&format!(
                    r#"CREATE INDEX IF NOT EXISTS {} ON {}
                       USING bm25 (id, content, tags, source, profile)
                       WITH (key_field='id')"#,
                    idx_name, tbl
                ))
                .execute(&self.pool)
                .await;
            }
        }

        Ok(())
    }

    async fn find_duplicate(
        &self,
        embedding: &[f32],
        threshold: f32,
    ) -> Result<Option<(i64, f32)>, Box<dyn std::error::Error + Send + Sync>> {
        let vec_str = vec_to_pgstring(embedding);
        let row = sqlx::query(
            "SELECT id, (1 - (embedding <=> $1::vector))::real AS sim FROM memories_deep WHERE (1 - (embedding <=> $1::vector))::real >= $2 ORDER BY sim DESC LIMIT 1",
        )
        .bind(&vec_str)
        .bind(threshold as f64)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| (r.get::<i64, _>("id"), r.get::<f32, _>("sim"))))
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
        expires_at: Option<DateTime<Utc>>,
        immortal: bool,
        event_date: Option<DateTime<Utc>>,
        reminder_interval: Option<&str>,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        let vec_str = vec_to_pgstring(embedding);
        let tags_str = tags.to_string();
        let metadata_str = metadata.to_string();

        let reminder_at = if let Some(ri) = reminder_interval {
            Self::compute_reminder_at(event_date, ri)
        } else {
            None
        };

        let result = sqlx::query(
            r#"
            INSERT INTO memories_deep (profile, content, embedding, tags, metadata, importance, source, category, expires_at, immortal, event_date, reminder_interval, reminder_at)
            VALUES ($1, $2, $3::vector, $4::jsonb, $5::jsonb, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
            "#,
        )
        .bind(profile)
        .bind(content)
        .bind(&vec_str)
        .bind(&tags_str)
        .bind(&metadata_str)
        .bind(importance)
        .bind(source)
        .bind(category)
        .bind(expires_at)
        .bind(immortal)
        .bind(event_date)
        .bind(reminder_interval)
        .bind(reminder_at)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.get("id"))
    }

    async fn search(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = filters.limit;
        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::new();

        let mut fresh = self.search_fresh(embedding, filters).await?;
        for r in &mut fresh {
            r.score = r.score * 1.15;
        }
        for r in fresh {
            if seen.insert(r.id) {
                results.push(r);
            }
        }

        let consolid = self.search_consolid(embedding, filters).await?;
        for r in consolid {
            if seen.insert(r.id) {
                results.push(r);
            }
        }

        const WARM_THRESHOLD: f64 = 0.70;
        const HOT_THRESHOLD: f64 = 0.85;
        const NUCLEAR_THRESHOLD: f64 = 0.95;

        if filters.include_deep {
            let deep = self.search_deep(embedding, filters).await?;
            for r in &deep {
                if seen.insert(r.id) {
                    let mut record = r.clone();
                    if record.score > WARM_THRESHOLD {
                        record.score *= 1.2;
                    }
                    results.push(record);
                }
                if r.score > NUCLEAR_THRESHOLD {
                    self.reconsolidate_to_fresh(r, true).await.ok();
                    self.set_immortal(r.id).await.ok();
                } else if r.score > HOT_THRESHOLD {
                    self.reconsolidate_to_fresh(r, false).await.ok();
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    async fn track_access(
        &self,
        ids: &[i64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if ids.is_empty() {
            return Ok(());
        }
        let id_list: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        let sql = format!(
            "UPDATE memories_deep SET access_count = access_count + 1, last_accessed_at = now() WHERE id IN ({})",
            id_list.join(",")
        );
        sqlx::query(&sql).execute(&self.pool).await?;
        Ok(())
    }

    async fn compact(
        &self,
        dry_run: bool,
        similarity_threshold: f32,
        importance_threshold: f32,
    ) -> Result<CompactReport, Box<dyn std::error::Error + Send + Sync>> {
        let mut report = CompactReport {
            merged_pairs: vec![],
            archived_ids: vec![],
            decayed_ids: vec![],
        };

        let candidates = sqlx::query_as::<_, (i64, String, String, f32)>(
            r#"SELECT id, content, embedding::text, importance
               FROM memories_deep WHERE importance < $1"#,
        )
        .bind(importance_threshold)
        .fetch_all(&self.pool)
        .await?;

        if !dry_run {
            for (id, _, _, _) in &candidates {
                let _ = sqlx::query("UPDATE memories_deep SET importance = importance * 0.5, metadata = jsonb_set(COALESCE(metadata, '{}'), '{archived}', 'true'::jsonb) WHERE id = $1")
                    .bind(id)
                    .execute(&self.pool)
                    .await;
            }
        }
        report.archived_ids = candidates.iter().map(|(id, _, _, _)| *id).collect();

        let mut processed = std::collections::HashSet::new();
        let all_ids: Vec<(i64, String)> = sqlx::query_as::<_, (i64, String)>(
            "SELECT id, embedding::text FROM memories_deep ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;

        for item in &all_ids {
            if processed.contains(&item.0) {
                continue;
            }
            let id_a = item.0;
            if let Some(emb_a) = parse_vector_text(&item.1).map(|v| vec_to_pgstring(&v)) {
                let similar = sqlx::query_as::<_, (i64, f64)>(
                    r#"SELECT id, 1 - (embedding <=> $1::vector) AS sim
                       FROM memories_deep WHERE id > $2 AND 1 - (embedding <=> $1::vector) >= $3
                       ORDER BY sim DESC LIMIT 1"#,
                )
                .bind(&emb_a)
                .bind(id_a)
                .bind(similarity_threshold as f64)
                .fetch_optional(&self.pool)
                .await?;

                if let Some((id_b, sim)) = similar {
                    processed.insert(id_b);
                    report.merged_pairs.push((id_a, id_b, sim));
                    if !dry_run {
                        if let Ok((merged_content,)) = sqlx::query_as::<_, (String,)>(&format!(
                            r#"SELECT string_agg(content, E'\n---\n') FROM memories_deep WHERE id IN ({}, {})"#,
                            id_a, id_b
                        ))
                        .fetch_one(&self.pool)
                        .await {
                            if let Ok((emb_str,)) = sqlx::query_as::<_, (String,)>(
                                "SELECT embedding::text FROM memories_deep WHERE id = $1 LIMIT 1",
                            )
                            .bind(id_a)
                            .fetch_one(&self.pool)
                            .await {
                                let _ = sqlx::query("DELETE FROM memories_deep WHERE id = $1")
                                    .bind(id_b)
                                    .execute(&self.pool)
                                    .await;
                                let _ = sqlx::query("UPDATE memories_deep SET content = $1, embedding = $2::vector WHERE id = $3")
                                    .bind(&merged_content)
                                    .bind(&emb_str)
                                    .bind(id_a)
                                    .execute(&self.pool)
                                    .await;
                            }
                        }
                    }
                }
            }
        }

        Ok(report)
    }

    async fn snapshot(
        &self,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let sql = format!("{} FROM memories_deep ORDER BY id", Self::SELECT_COLS);
        let rows = sqlx::query(&sql).fetch_all(&self.pool).await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn diff_snapshots(
        &self,
        snapshot_a: &[MemoryRecord],
        snapshot_b: &[MemoryRecord],
    ) -> Result<SnapshotDiff, Box<dyn std::error::Error + Send + Sync>> {
        use std::collections::HashMap;
        let map_a: HashMap<i64, &MemoryRecord> = snapshot_a.iter().map(|r| (r.id, r)).collect();
        let map_b: HashMap<i64, &MemoryRecord> = snapshot_b.iter().map(|r| (r.id, r)).collect();

        let added: Vec<MemoryRecord> = snapshot_b
            .iter()
            .filter(|r| !map_a.contains_key(&r.id))
            .cloned()
            .collect();
        let removed: Vec<MemoryRecord> = snapshot_a
            .iter()
            .filter(|r| !map_b.contains_key(&r.id))
            .cloned()
            .collect();
        let changed: Vec<(MemoryRecord, MemoryRecord)> = snapshot_a
            .iter()
            .filter_map(|a| {
                map_b.get(&a.id).and_then(|b| {
                    if a.content != b.content || a.importance != b.importance {
                        Some((a.clone(), (*b).clone()))
                    } else {
                        None
                    }
                })
            })
            .collect();

        Ok(SnapshotDiff {
            added,
            removed,
            changed,
        })
    }

    async fn rollback_to(
        &self,
        snapshot: &[MemoryRecord],
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let current_ids: Vec<i64> = sqlx::query_scalar("SELECT id FROM memories_deep ORDER BY id")
            .fetch_all(&self.pool)
            .await?;
        let snapshot_ids: std::collections::HashSet<i64> = snapshot.iter().map(|r| r.id).collect();

        let to_delete: Vec<String> = current_ids
            .iter()
            .filter(|id| !snapshot_ids.contains(id))
            .map(|id| id.to_string())
            .collect();

        if !to_delete.is_empty() {
            let id_list = to_delete.join(",");
            sqlx::query(&format!(
                "DELETE FROM memories_deep WHERE id IN ({id_list})"
            ))
            .execute(&self.pool)
            .await?;
        }

        let mut restored = 0u64;
        for rec in snapshot {
            let emb_str = vec_to_pgstring(&rec.embedding);
            let tags_str = rec.tags.to_string();
            let meta_str = rec.metadata.to_string();
            let result = sqlx::query(
                r#"INSERT INTO memories_deep (id, profile, content, embedding, tags, metadata, importance, source, category, created_at, updated_at)
                   VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6::jsonb, $7, $8, $9, $10, $11)
                   ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, importance = EXCLUDED.importance"#,
            )
            .bind(rec.id)
            .bind(&rec.profile)
            .bind(&rec.content)
            .bind(&emb_str)
            .bind(&tags_str)
            .bind(&meta_str)
            .bind(rec.importance)
            .bind(&rec.source)
            .bind(&rec.category)
            .bind(rec.created_at)
            .bind(rec.updated_at)
            .execute(&self.pool)
            .await?;
            restored += result.rows_affected();
        }
        Ok(restored)
    }

    async fn delete_by_profile(
        &self,
        profile: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("DELETE FROM memories_deep WHERE profile = $1")
            .bind(profile)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn list_by_profile(
        &self,
        profile: &str,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(&format!(
            "{} FROM memories_deep WHERE profile = $1 ORDER BY id DESC",
            Self::SELECT_COLS
        ))
        .bind(profile)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn delete_by_id(&self, id: i64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("DELETE FROM memories_deep WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn list_memories(
        &self,
        profile: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<MemoryRecord>, u64), Box<dyn std::error::Error + Send + Sync>> {
        let (where_clause, params) = match profile {
            Some(sid) => ("WHERE profile = $1".to_string(), vec![sid.to_string()]),
            None => ("".to_string(), vec![]),
        };

        let count_sql = format!("SELECT COUNT(*) FROM memories_deep {}", where_clause);
        let mut count_query = sqlx::query_scalar::<_, i64>(&count_sql);
        for p in &params {
            count_query = count_query.bind(p);
        }
        let total: i64 = count_query.fetch_one(&self.pool).await.unwrap_or(0);

        let data_sql = if profile.is_some() {
            format!(
                "{} FROM memories_deep WHERE profile = $1 ORDER BY id DESC LIMIT $2 OFFSET $3",
                Self::SELECT_COLS,
            )
        } else {
            format!(
                "{} FROM memories_deep ORDER BY id DESC LIMIT $1 OFFSET $2",
                Self::SELECT_COLS,
            )
        };

        let mut data_query = sqlx::query(&data_sql);
        if let Some(sid) = profile {
            data_query = data_query.bind(sid);
        }
        data_query = data_query.bind(limit as i64);
        data_query = data_query.bind(offset as i64);

        let rows = data_query.fetch_all(&self.pool).await?;
        let records = rows.iter().map(row_to_record_from_row).collect();
        Ok((records, total as u64))
    }

    async fn get_by_id(
        &self,
        id: i64,
    ) -> Result<Option<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let sql = format!("{} FROM memories_deep WHERE id = $1", Self::SELECT_COLS);
        let rows = sqlx::query(&sql).bind(id).fetch_all(&self.pool).await?;
        Ok(rows.first().map(row_to_record_from_row))
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
        let mut sets: Vec<String> = Vec::new();
        let mut pi = 1;

        if content.is_some() {
            sets.push(format!("content = ${}", pi));
            pi += 1;
        }
        if tags.is_some() {
            sets.push(format!("tags = ${}::jsonb", pi));
            pi += 1;
        }
        if metadata.is_some() {
            sets.push(format!("metadata = ${}::jsonb", pi));
            pi += 1;
        }
        if importance.is_some() {
            sets.push(format!("importance = ${}", pi));
            pi += 1;
        }
        if category.is_some() {
            sets.push(format!("category = ${}", pi));
            pi += 1;
        }
        if source.is_some() {
            sets.push(format!("source = ${}", pi));
            pi += 1;
        }

        if sets.is_empty() {
            return Ok(false);
        }

        let sql = format!(
            "UPDATE memories_deep SET {} WHERE id = ${}",
            sets.join(", "),
            pi
        );
        let mut query = sqlx::query(&sql);

        if let Some(ref c) = content {
            query = query.bind(c);
        }
        if let Some(ref t) = tags {
            query = query.bind(t.to_string());
        }
        if let Some(ref m) = metadata {
            query = query.bind(m.to_string());
        }
        if let Some(i) = importance {
            query = query.bind(i);
        }
        if let Some(ref c) = category {
            query = query.bind(c);
        }
        if let Some(ref s) = source {
            query = query.bind(s);
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
        let col = if useful {
            "feedback_positive"
        } else {
            "feedback_negative"
        };
        let sql = format!(
            "UPDATE memories_deep SET {} = {} + 1 WHERE id = $1",
            col, col
        );
        sqlx::query(&sql).bind(id).execute(&self.pool).await?;
        Ok(())
    }

    async fn backup(&self) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let sql = format!("{} FROM memories_deep ORDER BY id", Self::SELECT_COLS);
        let rows = sqlx::query(&sql).fetch_all(&self.pool).await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn restore(
        &self,
        records: &[MemoryRecord],
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut count = 0u64;
        for rec in records {
            let emb_str = vec_to_pgstring(&rec.embedding);
            let tags_str = rec.tags.to_string();
            let meta_str = rec.metadata.to_string();
            let result = sqlx::query(
                "INSERT INTO memories_deep (profile, content, embedding, tags, metadata, importance, source, category, created_at, updated_at) VALUES ($1, $2, $3::vector, $4::jsonb, $5::jsonb, $6, $7, $8, $9, $10)",
            )
            .bind(&rec.profile)
            .bind(&rec.content)
            .bind(&emb_str)
            .bind(&tags_str)
            .bind(&meta_str)
            .bind(rec.importance)
            .bind(&rec.source)
            .bind(&rec.category)
            .bind(rec.created_at)
            .bind(rec.updated_at)
            .execute(&self.pool)
            .await?;
            count += result.rows_affected();
        }
        Ok(count)
    }

    async fn get_profile_status(
        &self,
        profile: &str,
    ) -> Result<ProfileStatus, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query_as::<_, (String, i64, DateTime<Utc>, DateTime<Utc>)>(
            "SELECT profile, turn_count, created_at, last_active_at FROM profile_stats WHERE profile = $1",
        )
        .bind(profile)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some((p, turn_count, created_at, last_active_at)) => Ok(ProfileStatus {
                profile: p,
                turn_count,
                is_cold: turn_count == 0,
                created_at,
                last_active_at,
            }),
            None => Ok(ProfileStatus {
                profile: profile.to_string(),
                turn_count: 0,
                is_cold: true,
                created_at: Utc::now(),
                last_active_at: Utc::now(),
            }),
        }
    }

    async fn increment_turn_count(
        &self,
        profile: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO profile_stats (profile, turn_count, last_active_at) VALUES ($1, 1, now()) ON CONFLICT (profile) DO UPDATE SET turn_count = profile_stats.turn_count + 1, last_active_at = now()",
        )
        .bind(profile)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_base_context(
        &self,
        profile: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(
            &format!(
                "{} FROM memories_deep WHERE profile = $1 ORDER BY importance DESC, created_at DESC LIMIT $2",
                Self::SELECT_COLS
            ),
        )
        .bind(profile)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn create_conclusion(
        &self,
        profile: &str,
        content: &str,
        category: &str,
        confidence: f32,
        source_turn_ids: &[i64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let emb = vec![0.0f32; 1536];
        let vec_str = vec_to_pgstring(&emb);
        let metadata = json!({
            "confidence": confidence,
            "source_turn_ids": source_turn_ids,
            "type": "conclusion"
        });
        sqlx::query(
            "INSERT INTO memories_deep (profile, content, embedding, metadata, importance, source, category) VALUES ($1, $2, $3::vector, $4::jsonb, $5, $6, $7)",
        )
        .bind(profile)
        .bind(content)
        .bind(&vec_str)
        .bind(&metadata.to_string())
        .bind(0.8f32)
        .bind("system")
        .bind(category)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn link_memories(
        &self,
        id_a: i64,
        id_b: i64,
        relation_type: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            r#"INSERT INTO memory_links (id_a, id_b, relation_type)
               VALUES ($1, $2, $3)
               ON CONFLICT (id_a, id_b) DO UPDATE SET relation_type = EXCLUDED.relation_type"#,
        )
        .bind(id_a)
        .bind(id_b)
        .bind(relation_type)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn unlink_memories(
        &self,
        id_a: i64,
        id_b: i64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "DELETE FROM memory_links WHERE (id_a = $1 AND id_b = $2) OR (id_a = $2 AND id_b = $1)",
        )
        .bind(id_a)
        .bind(id_b)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_relations(
        &self,
        id: i64,
    ) -> Result<Vec<MemoryLink>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(
            r#"SELECT id_a, id_b, relation_type, created_at
               FROM memory_links WHERE id_a = $1 OR id_b = $1
               ORDER BY created_at DESC"#,
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| MemoryLink {
                id_a: r.get("id_a"),
                id_b: r.get("id_b"),
                relation_type: r.get("relation_type"),
                created_at: r.get("created_at"),
            })
            .collect())
    }

    async fn get_memory_graph(
        &self,
        id: i64,
        depth: u32,
    ) -> Result<Vec<super::GraphNode>, Box<dyn std::error::Error + Send + Sync>> {
        let mut visited = std::collections::HashSet::new();
        let mut nodes = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((id, 0u32));

        while let Some((current_id, current_depth)) = queue.pop_front() {
            if current_depth > depth || !visited.insert(current_id) {
                continue;
            }
            let memory = self
                .get_by_id(current_id)
                .await?
                .ok_or_else(|| format!("Memory {current_id} not found"))?;
            let link_rows = sqlx::query_as::<_, (i64, i64, String, DateTime<Utc>)>(
                r#"SELECT id_a, id_b, relation_type, created_at
                   FROM memory_links WHERE id_a = $1 OR id_b = $1
                   ORDER BY created_at DESC"#,
            )
            .bind(current_id)
            .fetch_all(&self.pool)
            .await?;

            let mut edges = Vec::new();
            for (id_a, id_b, relation_type, _) in link_rows {
                let target_id = if id_a == current_id { id_b } else { id_a };
                edges.push(super::GraphEdge {
                    target_id,
                    relation_type,
                });
                if current_depth + 1 <= depth {
                    queue.push_back((target_id, current_depth + 1));
                }
            }
            nodes.push(super::GraphNode { memory, edges });
        }
        Ok(nodes)
    }

    async fn associative_search(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
        _depth: u32,
    ) -> Result<Vec<super::AssociativeResult>, Box<dyn std::error::Error + Send + Sync>> {
        let _limit = filters.limit;
        let _profile = filters.profile.as_deref();

        let seeds = self.search_deep(embedding, filters).await?;

        let mut results = Vec::new();
        for seed in seeds {
            let link_rows = sqlx::query_as::<_, (i64, i64, String)>(
                r#"SELECT id_a, id_b, relation_type FROM memory_links WHERE id_a = $1 OR id_b = $1"#,
            )
            .bind(seed.id)
            .fetch_all(&self.pool)
            .await?;

            let mut related_ids: Vec<i64> = Vec::new();
            for (id_a, id_b, _) in link_rows {
                let target = if id_a == seed.id { id_b } else { id_a };
                related_ids.push(target);
            }

            let mut related = Vec::new();
            for rid in related_ids {
                if let Some(rec) = self.get_by_id(rid).await? {
                    related.push(rec);
                }
            }

            results.push(super::AssociativeResult { seed, related });
        }
        Ok(results)
    }

    async fn get_due_reminders(
        &self,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let sql = format!(
            "{} FROM memories_deep WHERE reminder_at IS NOT NULL AND reminder_at <= now() AND NOT reminder_sent ORDER BY reminder_at ASC LIMIT $1",
            Self::SELECT_COLS
        );
        let rows = sqlx::query(&sql)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn mark_reminder_sent(
        &self,
        id: i64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE memories_deep SET reminder_sent = TRUE WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn set_reminder(
        &self,
        id: i64,
        event_date: Option<DateTime<Utc>>,
        reminder_interval: Option<String>,
        reminder_at: Option<DateTime<Utc>>,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let reminder_at = match (&reminder_interval, &reminder_at, event_date) {
            (Some(ri), None, ed) => Self::compute_reminder_at(ed, ri),
            (_, Some(ra), _) => Some(*ra),
            _ => None,
        };
        let result = sqlx::query(
            "UPDATE memories_deep SET event_date = $1, reminder_interval = $2, reminder_at = $3, reminder_sent = FALSE, updated_at = now() WHERE id = $4",
        )
        .bind(event_date)
        .bind(&reminder_interval)
        .bind(reminder_at)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

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
        _conversation_id: &str,
        _turn_range: &str,
        _user_msg: &str,
        _assistant_msg: &str,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        self.store(
            profile,
            content,
            embedding,
            tags,
            metadata,
            importance,
            source,
            category,
            expires_at,
            false,
            event_date,
            reminder_interval,
        )
        .await
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
        expires_at: Option<DateTime<Utc>>,
        event_date: Option<DateTime<Utc>>,
        reminder_interval: Option<&str>,
        conversation_id: &str,
        turn_range: &str,
        user_msg: &str,
        assistant_msg: &str,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        let vec_str = vec_to_pgstring(embedding);
        let tags_str = tags.to_string();
        let metadata_str = metadata.to_string();
        let actual_expires =
            expires_at.unwrap_or_else(|| Utc::now() + chrono::TimeDelta::hours(24));

        let reminder_at = if let Some(ri) = reminder_interval {
            Self::compute_reminder_at(event_date, ri)
        } else {
            None
        };

        let result = sqlx::query(
            r#"INSERT INTO memories_fresh (profile, content, embedding, tags, metadata, importance, source, category, expires_at, event_date, reminder_interval, reminder_at, conversation_id, turn_range, user_msg, assistant_msg)
               VALUES ($1, $2, $3::vector, $4::jsonb, $5::jsonb, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
               RETURNING id"#,
        )
        .bind(profile)
        .bind(content)
        .bind(&vec_str)
        .bind(&tags_str)
        .bind(&metadata_str)
        .bind(importance)
        .bind(source)
        .bind(category)
        .bind(actual_expires)
        .bind(event_date)
        .bind(reminder_interval)
        .bind(reminder_at)
        .bind(conversation_id)
        .bind(turn_range)
        .bind(user_msg)
        .bind(assistant_msg)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.get("id"))
    }

    async fn search_fresh(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = filters.limit;
        let profile = filters.profile.as_deref();
        let vec_results = self
            .run_vector_search_on("memories_fresh", profile, embedding, filters, limit * 2)
            .await?;
        let bm25_results =
            if self.paradedb_available.load(Ordering::Relaxed) && !filters.query.is_empty() {
                self.run_bm25_search_on(
                    "memories_fresh",
                    profile,
                    &filters.query,
                    filters,
                    limit * 2,
                )
                .await?
            } else {
                vec![]
            };
        Ok(Self::fuse_results(
            vec_results,
            bm25_results,
            filters.alpha,
            limit,
        ))
    }

    async fn search_consolid(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = filters.limit;
        let profile = filters.profile.as_deref();
        let vec_str = vec_to_pgstring(embedding);

        let (where_clause, where_params) = match profile {
            Some(p) => ("WHERE profile = $1".to_string(), vec![p.to_string()]),
            None => ("".to_string(), vec![]),
        };

        let last_param = 1 + where_params.len();
        let sql = format!(
            r#"SELECT id, profile, summary AS content, embedding::text,
               COALESCE(tags::text, '{{}}') AS tags,
               COALESCE(metadata::text, '{{}}') AS metadata,
               importance, 'consolidated' AS source, '' AS category,
               created_at, last_consolidated_at AS updated_at,
               0 AS feedback_positive, 0 AS feedback_negative,
               COALESCE(access_count, 0) AS access_count,
               NULL AS last_accessed_at,
               0.5 AS trust_score,
               NULL AS expires_at,
               false AS immortal,
               NULL AS event_date,
               NULL AS reminder_interval,
               NULL AS reminder_at,
               false AS reminder_sent,
{}
       FROM memories_consolid
       {} ORDER BY score DESC LIMIT ${}"#,
            Self::vec_score_expr(filters.decay_half_life_days, 1),
            where_clause,
            last_param + 1,
        );

        let mut query = sqlx::query(&sql).bind(&vec_str);
        for p in &where_params {
            query = query.bind(p);
        }
        query = query.bind(limit as i64);
        let rows = query.fetch_all(&self.pool).await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn search_deep(
        &self,
        embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = filters.limit;
        let profile = filters.profile.as_deref();
        let vec_results = self
            .run_vector_search_on("memories_deep", profile, embedding, filters, limit * 2)
            .await?;
        let bm25_results = if self.paradedb_available.load(Ordering::Relaxed)
            && !filters.query.is_empty()
        {
            self.run_bm25_search_on("memories_deep", profile, &filters.query, filters, limit * 2)
                .await?
        } else {
            vec![]
        };
        Ok(Self::fuse_results(
            vec_results,
            bm25_results,
            filters.alpha,
            limit,
        ))
    }

    async fn store_consolid(
        &self,
        profile: &str,
        summary: &str,
        embedding: &[f32],
        source_ids: &[i64],
        tags: &Value,
        metadata: &Value,
        importance: f32,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        let vec_str = vec_to_pgstring(embedding);
        let tags_str = tags.to_string();
        let meta_str = metadata.to_string();

        let depth = metadata
            .get("depth")
            .and_then(|v| v.as_str())
            .unwrap_or("shallow");

        let result = sqlx::query(
            r#"INSERT INTO memories_consolid (profile, summary, embedding, source_ids, tags, metadata, importance, depth)
               VALUES ($1, $2, $3::vector, $4::bigint[], $5::jsonb, $6::jsonb, $7, $8)
               RETURNING id"#,
        )
        .bind(profile)
        .bind(summary)
        .bind(&vec_str)
        .bind(source_ids)
        .bind(&tags_str)
        .bind(&meta_str)
        .bind(importance)
        .bind(depth)
        .fetch_one(&self.pool)
        .await?;

        Ok(result.get("id"))
    }

    async fn prune_fresh(
        &self,
        ttl_hours: u64,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let result = sqlx::query(
            &format!("DELETE FROM memories_fresh WHERE created_at < now() - make_interval(hours => {ttl_hours})"),
        )
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn get_new_deep_since(
        &self,
        since: DateTime<Utc>,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let sql = format!(
            "{} FROM memories_deep WHERE created_at > $1 ORDER BY created_at ASC",
            Self::SELECT_COLS
        );
        let rows = sqlx::query(&sql).bind(since).fetch_all(&self.pool).await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn get_new_fresh_since(
        &self,
        profile: &str,
        since: DateTime<Utc>,
    ) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(
            r#"SELECT id, profile, content, embedding::text,
               COALESCE(tags::text, '{}') AS tags,
               COALESCE(metadata::text, '{}') AS metadata,
               COALESCE(importance, 0.0) AS importance,
               COALESCE(source, '') AS source,
               COALESCE(category, '') AS category,
               created_at, updated_at,
               COALESCE(feedback_positive, 0) AS feedback_positive,
               COALESCE(feedback_negative, 0) AS feedback_negative,
               COALESCE(access_count, 0) AS access_count,
               last_accessed_at,
               COALESCE(trust_score, 0.5) AS trust_score,
               expires_at,
               COALESCE(immortal, false) AS immortal,
               event_date,
               reminder_interval,
               reminder_at,
               COALESCE(reminder_sent, false) AS reminder_sent
               FROM memories_fresh WHERE profile = $1 AND created_at > $2
               ORDER BY created_at ASC"#,
        )
        .bind(profile)
        .bind(since)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row_to_record_from_row).collect())
    }

    async fn prune_deep(
        &self,
        min_importance: f32,
        min_access_count: i32,
        max_age_days: i64,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let result = sqlx::query(
            "DELETE FROM memories_deep WHERE importance < $1 AND access_count <= $2 AND created_at < now() - make_interval(days => $3) AND NOT immortal",
        )
        .bind(min_importance)
        .bind(min_access_count)
        .bind(max_age_days)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn prune_consolid(
        &self,
        min_insight_score: f32,
        max_age_days: i64,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let result = sqlx::query(
            "DELETE FROM memories_consolid WHERE insight_score < $1 AND created_at < now() - make_interval(days => $2)",
        )
        .bind(min_insight_score)
        .bind(max_age_days)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn schedule_review(
        &self,
        id: i64,
        importance: f32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.do_schedule_review(id, importance).await
    }

    async fn memory_stats(&self) -> Result<MemoryStats, Box<dyn std::error::Error + Send + Sync>> {
        let fresh: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories_fresh")
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);
        let deep: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories_deep")
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);
        let consolid: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories_consolid")
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);

        let profile_rows: Vec<(String, i64, i64, i64)> = sqlx::query_as(
            r#"SELECT
                COALESCE(f.profile, d.profile, c.profile) AS profile,
                COALESCE(f.cnt, 0) AS fresh,
                COALESCE(d.cnt, 0) AS deep,
                COALESCE(c.cnt, 0) AS consolid
             FROM (SELECT profile, COUNT(*) cnt FROM memories_fresh GROUP BY profile) f
             FULL JOIN (SELECT profile, COUNT(*) cnt FROM memories_deep GROUP BY profile) d USING (profile)
             FULL JOIN (SELECT profile, COUNT(*) cnt FROM memories_consolid GROUP BY profile) c USING (profile)"#,
        )
            .fetch_all(&self.pool)
            .await
            .unwrap_or_default();

        let per_profile = profile_rows
            .into_iter()
            .map(|(profile, fresh, deep, consolid)| ProfileTierCounts {
                profile,
                fresh: fresh as u64,
                deep: deep as u64,
                consolid: consolid as u64,
            })
            .collect();

        Ok(MemoryStats {
            fresh_count: fresh as u64,
            deep_count: deep as u64,
            consolid_count: consolid as u64,
            per_profile,
        })
    }

    async fn detailed_stats(
        &self,
        profile: Option<&str>,
    ) -> Result<DetailedMemoryStats, Box<dyn std::error::Error + Send + Sync>> {
        let pool = &self.pool;

        let deep_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memories_deep WHERE ($1::text IS NULL OR profile = $1)",
        )
        .bind(profile)
        .fetch_one(pool)
        .await
        .unwrap_or(0);

        let fresh_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memories_fresh WHERE ($1::text IS NULL OR profile = $1)",
        )
        .bind(profile)
        .fetch_one(pool)
        .await
        .unwrap_or(0);

        let consolid_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memories_consolid WHERE ($1::text IS NULL OR profile = $1)",
        )
        .bind(profile)
        .fetch_one(pool)
        .await
        .unwrap_or(0);

        let total_profiles: i64 =
            sqlx::query_scalar("SELECT COUNT(DISTINCT profile) FROM memories_deep")
                .fetch_one(pool)
                .await
                .unwrap_or(0);

        let total_links: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_links")
            .fetch_one(pool)
            .await
            .unwrap_or(0);

        let deep_numeric: Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, i64, String, String)> =
            sqlx::query_as(
                r#"SELECT
                    COALESCE(MIN(importance)::double precision, 0) AS imp_min,
                    COALESCE(AVG(importance)::double precision, 0) AS imp_avg,
                    COALESCE(MAX(importance)::double precision, 0) AS imp_max,
                    COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY importance)::double precision, 0) AS imp_median,
                    COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY importance)::double precision, 0) AS imp_p95,
                    COALESCE(MIN(trust_score)::double precision, 0) AS ts_min,
                    COALESCE(AVG(trust_score)::double precision, 0) AS ts_avg,
                    COALESCE(MAX(trust_score)::double precision, 0) AS ts_max,
                    COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trust_score)::double precision, 0) AS ts_median,
                    COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY trust_score)::double precision, 0) AS ts_p95,
                    COALESCE(MIN(access_count)::double precision, 0) AS ac_min,
                    COALESCE(AVG(access_count)::double precision, 0) AS ac_avg,
                    COALESCE(MAX(access_count)::double precision, 0) AS ac_max,
                    COALESCE(COUNT(*)::bigint, 0) AS cnt,
                    COALESCE(MIN(created_at)::text, '') AS oldest,
                    COALESCE(MAX(created_at)::text, '') AS newest
                 FROM memories_deep
                 WHERE ($1::text IS NULL OR profile = $1)"#,
            )
            .bind(profile)
            .fetch_optional(pool)
            .await
            .unwrap_or(None);

        let content_len: Option<(f64, f64, f64, f64, f64)> = sqlx::query_as(
            r#"SELECT
                COALESCE(MIN(char_length(content))::double precision, 0),
                COALESCE(AVG(char_length(content))::double precision, 0),
                COALESCE(MAX(char_length(content))::double precision, 0),
                COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY char_length(content))::double precision, 0),
                COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY char_length(content))::double precision, 0)
             FROM memories_deep
             WHERE ($1::text IS NULL OR profile = $1)"#,
        )
        .bind(profile)
        .fetch_optional(pool)
        .await
        .unwrap_or(None);

        let access_stats: Option<(f64, f64)> = sqlx::query_as(
            r#"SELECT
                COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY access_count)::double precision, 0),
                COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY access_count)::double precision, 0)
             FROM memories_deep
             WHERE ($1::text IS NULL OR profile = $1)"#,
        )
        .bind(profile)
        .fetch_optional(pool)
        .await
        .unwrap_or(None);

        let feedback: Option<(i64, i64)> = sqlx::query_as(
            r#"SELECT
                COALESCE(SUM(feedback_positive), 0)::bigint,
                COALESCE(SUM(feedback_negative), 0)::bigint
             FROM memories_deep
             WHERE ($1::text IS NULL OR profile = $1)"#,
        )
        .bind(profile)
        .fetch_optional(pool)
        .await
        .unwrap_or(None);

        let immortal_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memories_deep WHERE immortal = true AND ($1::text IS NULL OR profile = $1)",
        )
        .bind(profile)
        .fetch_one(pool)
        .await
        .unwrap_or(0);

        let expired_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memories_deep WHERE expires_at IS NOT NULL AND expires_at < NOW() AND ($1::text IS NULL OR profile = $1)",
        )
        .bind(profile)
        .fetch_one(pool)
        .await
        .unwrap_or(0);

        let reminders_active: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memories_deep WHERE reminder_interval IS NOT NULL AND ($1::text IS NULL OR profile = $1)",
        )
        .bind(profile)
        .fetch_one(pool)
        .await
        .unwrap_or(0);

        let reminders_sent: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memories_deep WHERE reminder_sent = true AND ($1::text IS NULL OR profile = $1)",
        )
        .bind(profile)
        .fetch_one(pool)
        .await
        .unwrap_or(0);

        let categories: Vec<(String, i64)> = sqlx::query_as(
            r#"SELECT category, COUNT(*)::bigint AS cnt
             FROM memories_deep
             WHERE category != '' AND ($1::text IS NULL OR profile = $1)
             GROUP BY category
             ORDER BY cnt DESC
             LIMIT 20"#,
        )
        .bind(profile)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        let sources: Vec<(String, i64)> = sqlx::query_as(
            r#"SELECT source, COUNT(*)::bigint AS cnt
             FROM memories_deep
             WHERE source != '' AND ($1::text IS NULL OR profile = $1)
             GROUP BY source
             ORDER BY cnt DESC
             LIMIT 20"#,
        )
        .bind(profile)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        let tags: Vec<(String, i64)> = sqlx::query_as(
            r#"SELECT tag, COUNT(*)::bigint AS cnt
             FROM memories_deep, LATERAL jsonb_object_keys(tags) AS tag
             WHERE ($1::text IS NULL OR profile = $1)
             GROUP BY tag
             ORDER BY cnt DESC
             LIMIT 20"#,
        )
        .bind(profile)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        let depths: Vec<(String, i64)> = sqlx::query_as(
            r#"SELECT depth, COUNT(*)::bigint AS cnt
             FROM memories_consolid
             WHERE ($1::text IS NULL OR profile = $1)
             GROUP BY depth
             ORDER BY cnt DESC"#,
        )
        .bind(profile)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        let profile_rows: Vec<(String, i64, i64, i64, f64, String, i64, i64)> = sqlx::query_as(
            r#"SELECT
                d.profile,
                COALESCE(d.cnt, 0) AS deep,
                COALESCE(f.cnt, 0) AS fresh,
                COALESCE(c.cnt, 0) AS consolid,
                COALESCE(d.imp_avg, 0)::double precision AS importance_avg,
                COALESCE(d.top_cat, '') AS top_category,
                COALESCE(d.fb_pos, 0)::bigint AS feedback_positive,
                COALESCE(d.fb_neg, 0)::bigint AS feedback_negative
             FROM (
                SELECT profile, COUNT(*) AS cnt,
                    AVG(importance) AS imp_avg,
                    (SELECT category FROM memories_deep d2 WHERE d2.profile = d1.profile AND d2.category != '' GROUP BY category ORDER BY COUNT(*) DESC LIMIT 1) AS top_cat,
                    SUM(feedback_positive) AS fb_pos,
                    SUM(feedback_negative) AS fb_neg
                FROM memories_deep d1
                WHERE ($1::text IS NULL OR profile = $1)
                GROUP BY profile
             ) d
             LEFT JOIN (SELECT profile, COUNT(*) AS cnt FROM memories_fresh WHERE ($1::text IS NULL OR profile = $1) GROUP BY profile) f ON d.profile = f.profile
             LEFT JOIN (SELECT profile, COUNT(*) AS cnt FROM memories_consolid WHERE ($1::text IS NULL OR profile = $1) GROUP BY profile) c ON d.profile = c.profile
             ORDER BY d.cnt DESC"#,
        )
        .bind(profile)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        let per_profile: Vec<ProfileDetailedStats> = profile_rows
            .into_iter()
            .map(
                |(p, deep, fresh, consolid, imp_avg, top_cat, fb_pos, fb_neg)| {
                    ProfileDetailedStats {
                        profile: p,
                        tier_counts: TierCounts {
                            fresh: fresh as u64,
                            deep: deep as u64,
                            consolid: consolid as u64,
                        },
                        importance_avg: imp_avg,
                        top_category: if top_cat.is_empty() {
                            None
                        } else {
                            Some(top_cat)
                        },
                        feedback_positive: fb_pos,
                        feedback_negative: fb_neg,
                        memory_count: (deep + fresh + consolid) as u64,
                    }
                },
            )
            .collect();

        let (
            imp_min,
            imp_avg,
            imp_max,
            imp_median,
            imp_p95,
            ts_min,
            ts_avg,
            ts_max,
            ts_median,
            ts_p95,
            ac_min,
            ac_avg,
            ac_max,
            cnt,
            oldest_str,
            newest_str,
        ) = deep_numeric.unwrap_or((
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            String::new(),
            String::new(),
        ));

        let (cl_min, cl_avg, cl_max, cl_median, cl_p95) =
            content_len.unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0));
        let (ac_median, ac_p95) = access_stats.unwrap_or((0.0, 0.0));
        let (fb_pos, fb_neg) = feedback.unwrap_or((0, 0));

        let oldest = chrono::DateTime::parse_from_rfc3339(&oldest_str)
            .ok()
            .map(|dt| dt.with_timezone(&Utc));
        let newest = chrono::DateTime::parse_from_rfc3339(&newest_str)
            .ok()
            .map(|dt| dt.with_timezone(&Utc));

        let deep_count_u = deep_count as u64;
        let fresh_count_u = fresh_count as u64;
        let consolid_count_u = consolid_count as u64;

        Ok(DetailedMemoryStats {
            tier_counts: TierCounts {
                fresh: fresh_count_u,
                deep: deep_count_u,
                consolid: consolid_count_u,
            },
            total_memories: deep_count_u + fresh_count_u + consolid_count_u,
            total_profiles: total_profiles as u64,
            total_links: total_links as u64,
            importance: NumericStats {
                min: imp_min,
                max: imp_max,
                avg: imp_avg,
                median: imp_median,
                p95: imp_p95,
                count: cnt as u64,
            },
            trust_score: NumericStats {
                min: ts_min,
                max: ts_max,
                avg: ts_avg,
                median: ts_median,
                p95: ts_p95,
                count: cnt as u64,
            },
            access_count: NumericStats {
                min: ac_min,
                max: ac_max,
                avg: ac_avg,
                median: ac_median,
                p95: ac_p95,
                count: cnt as u64,
            },
            content_length: NumericStats {
                min: cl_min,
                max: cl_max,
                avg: cl_avg,
                median: cl_median,
                p95: cl_p95,
                count: cnt as u64,
            },
            oldest_memory: oldest,
            newest_memory: newest,
            category_distribution: categories
                .into_iter()
                .map(|(label, count)| LabelCount {
                    label,
                    count: count as u64,
                })
                .collect(),
            top_tags: tags
                .into_iter()
                .map(|(label, count)| LabelCount {
                    label,
                    count: count as u64,
                })
                .collect(),
            source_distribution: sources
                .into_iter()
                .map(|(label, count)| LabelCount {
                    label,
                    count: count as u64,
                })
                .collect(),
            feedback_positive: fb_pos,
            feedback_negative: fb_neg,
            immortal_count: immortal_count as u64,
            mortal_count: (deep_count_u.saturating_sub(immortal_count as u64)),
            expired_count: expired_count as u64,
            consolid_depth_distribution: depths
                .into_iter()
                .map(|(label, count)| LabelCount {
                    label,
                    count: count as u64,
                })
                .collect(),
            reminders_active: reminders_active as u64,
            reminders_sent: reminders_sent as u64,
            per_profile,
        })
    }

    async fn get_consolid_by_depth_since(
        &self,
        depth: &str,
        since: DateTime<Utc>,
    ) -> Result<Vec<MemoryRecordConsolid>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(
            r#"SELECT id, profile, summary, embedding::text,
               COALESCE(tags::text, '{}') AS tags,
               COALESCE(metadata::text, '{}') AS metadata,
               importance, source_ids, depth,
               insight_score, created_at, last_consolidated_at,
               COALESCE(access_count, 0) AS access_count,
               0.0 AS score
               FROM memories_consolid
               WHERE depth = $1 AND created_at > $2
               ORDER BY created_at ASC"#,
        )
        .bind(depth)
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|row| {
                let embedding_str: &str = row.get("embedding");
                MemoryRecordConsolid {
                    id: row.get("id"),
                    profile: row.get("profile"),
                    summary: row.get("summary"),
                    embedding: parse_vector_text(embedding_str).unwrap_or_default(),
                    source_ids: row.get("source_ids"),
                    depth: row.get("depth"),
                    tags: serde_json::from_str(row.get::<&str, _>("tags")).unwrap_or(Value::Null),
                    metadata: serde_json::from_str(row.get::<&str, _>("metadata"))
                        .unwrap_or(Value::Null),
                    insight_score: row.get("insight_score"),
                    importance: row.get("importance"),
                    created_at: row.get("created_at"),
                    last_consolidated_at: row.get("last_consolidated_at"),
                    access_count: row.get("access_count"),
                    score: 0.0,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_rec(id: i64, content: &str, score: f64) -> MemoryRecord {
        MemoryRecord {
            id,
            profile: "test".into(),
            content: content.into(),
            embedding: vec![],
            tags: Value::Null,
            metadata: Value::Null,
            importance: 0.5,
            source: "test".into(),
            category: "general".into(),
            created_at: DateTime::from_timestamp(0, 0).unwrap(),
            updated_at: DateTime::from_timestamp(0, 0).unwrap(),
            feedback_positive: 0,
            feedback_negative: 0,
            expires_at: None,
            immortal: false,
            score,
            access_count: 0,
            last_accessed_at: None,
            trust_score: 0.5,
            event_date: None,
            reminder_interval: None,
            reminder_at: None,
            reminder_sent: false,
        }
    }

    #[test]
    fn test_fuse_results_both_sources() {
        let limit = 5;
        let alpha = 0.5;
        let vec_records = vec![make_rec(1, "vector A", 0.9), make_rec(2, "vector B", 0.8)];
        let bm25_records = vec![make_rec(2, "bm25 B", 0.85), make_rec(3, "bm25 C", 0.7)];
        let fused = PgVectorStore::fuse_results(vec_records, bm25_records, alpha, limit);
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].id, 2);
        assert!((fused[0].score - 0.825).abs() < 0.001);
    }

    #[test]
    fn test_fuse_results_pure_vector() {
        let limit = 5;
        let alpha = 0.5;
        let vec_records = vec![make_rec(1, "vector A", 0.9), make_rec(2, "vector B", 0.8)];
        let bm25_records = vec![];
        let fused = PgVectorStore::fuse_results(vec_records, bm25_records, alpha, limit);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_fuse_results_pure_bm25() {
        let limit = 5;
        let alpha = 0.5;
        let vec_records = vec![];
        let bm25_records = vec![make_rec(1, "bm25 A", 0.9)];
        let fused = PgVectorStore::fuse_results(vec_records, bm25_records, alpha, limit);
        assert_eq!(fused.len(), 1);
    }

    #[test]
    fn test_fuse_results_limit() {
        let limit = 1;
        let alpha = 0.5;
        let vec_records = vec![make_rec(1, "v1", 0.9), make_rec(2, "v2", 0.8)];
        let fused = PgVectorStore::fuse_results(vec_records, vec![], alpha, limit);
        assert_eq!(fused.len(), 1);
    }

    #[test]
    fn test_compute_reminder_at_relative_minutes() {
        let event_date = Some(DateTime::from_timestamp(1000, 0).unwrap());
        let result = PgVectorStore::compute_reminder_at(event_date, "30m");
        assert!(result.is_some());
        assert_eq!(
            result.unwrap(),
            DateTime::from_timestamp(1000 - 30 * 60, 0).unwrap()
        );
    }

    #[test]
    fn test_compute_reminder_at_absolute() {
        let result = PgVectorStore::compute_reminder_at(None, "2026-06-07T10:00:00Z");
        assert!(result.is_some());
        assert_eq!(
            result.unwrap(),
            DateTime::parse_from_rfc3339("2026-06-07T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc)
        );
    }

    #[test]
    fn test_compute_reminder_at_relative_hours() {
        let event_date = Some(DateTime::from_timestamp(1000, 0).unwrap());
        let result = PgVectorStore::compute_reminder_at(event_date, "1h");
        assert!(result.is_some());
        assert_eq!(result.unwrap().timestamp(), 1000 - 3600);
    }

    #[test]
    fn test_compute_reminder_at_relative_days() {
        let event_date = Some(DateTime::from_timestamp(1000, 0).unwrap());
        let result = PgVectorStore::compute_reminder_at(event_date, "2d");
        assert!(result.is_some());
        assert_eq!(result.unwrap().timestamp(), 1000 - 2 * 86400);
    }

    #[test]
    fn test_compute_reminder_at_no_event_date() {
        let result = PgVectorStore::compute_reminder_at(None, "30m");
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_reminder_at_invalid_interval() {
        let event_date = Some(DateTime::from_timestamp(1000, 0).unwrap());
        let result = PgVectorStore::compute_reminder_at(event_date, "garbage");
        assert!(result.is_none());
    }

    #[test]
    fn test_vec_to_pgstring_roundtrip() {
        let v = vec![0.1, 0.2, 0.3];
        let s = vec_to_pgstring(&v);
        let parsed = parse_vector_text(&s).unwrap();
        assert!((parsed[0] - 0.1).abs() < 1e-6);
        assert!((parsed[1] - 0.2).abs() < 1e-6);
        assert!((parsed[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_vec_to_pgstring_empty() {
        let v: Vec<f32> = vec![];
        let s = vec_to_pgstring(&v);
        assert_eq!(s, "[]");
        let parsed = parse_vector_text(&s).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_parse_vector_text_bracket_variants() {
        assert_eq!(
            parse_vector_text("[1.0,2.0,3.0]"),
            Some(vec![1.0, 2.0, 3.0])
        );
        assert_eq!(
            parse_vector_text("{1.0,2.0,3.0}"),
            Some(vec![1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn test_parse_vector_text_empty() {
        assert_eq!(parse_vector_text("[]"), Some(vec![]));
    }

    #[test]
    fn test_parse_vector_text_malformed() {
        assert_eq!(parse_vector_text(""), None);
    }

    #[test]
    fn test_parse_vector_text_whitespace() {
        assert_eq!(
            parse_vector_text("  [1.0, 2.0, 3.0]  "),
            Some(vec![1.0, 2.0, 3.0])
        );
    }
}
