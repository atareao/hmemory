use crate::api::AppState;
use chrono::Utc;
use serde_json::json;

pub fn start(state: AppState) {
    let cadence = state.consolidation_cadence;
    let ttl = state.fresh_ttl_hours;
    let model = state.consolidation_model.clone();
    let api_key = state.openrouter_api_key.clone();
    let store = state.store.clone();
    let embedder = state.embedder.clone();
    let prune_deep_importance = state.prune_deep_importance;
    let prune_deep_access_count = state.prune_deep_access_count;
    let prune_deep_age_days = state.prune_deep_age_days;
    let prune_consolid_insight = state.prune_consolid_insight;
    let prune_consolid_age_days = state.prune_consolid_age_days;

    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(cadence)).await;
            if let Err(e) = run_consolidation(
                &*store,
                &*embedder,
                ttl,
                &model,
                &api_key,
                prune_deep_importance,
                prune_deep_access_count,
                prune_deep_age_days,
                prune_consolid_insight,
                prune_consolid_age_days,
            )
            .await
            {
                tracing::warn!("consolidation worker failed: {e}");
            }
        }
    });
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    dot / (na.sqrt() * nb.sqrt()).max(1e-10)
}

fn semantic_cluster(
    records: &[crate::storage::MemoryRecord],
    threshold: f32,
) -> Vec<Vec<&crate::storage::MemoryRecord>> {
    if records.is_empty() {
        return vec![];
    }
    let mut groups: Vec<Vec<&crate::storage::MemoryRecord>> = Vec::new();
    for rec in records {
        let mut assigned = false;
        for group in &mut groups {
            let centroid = group_centroid(group);
            if cosine_similarity(&rec.embedding, &centroid) >= threshold {
                group.push(rec);
                assigned = true;
                break;
            }
        }
        if !assigned {
            groups.push(vec![rec]);
        }
    }
    groups
}

fn group_centroid(group: &[&crate::storage::MemoryRecord]) -> Vec<f32> {
    if group.is_empty() {
        return vec![];
    }
    let dim = group[0].embedding.len();
    let mut sum = vec![0.0f32; dim];
    for rec in group {
        for (s, e) in sum.iter_mut().zip(&rec.embedding) {
            *s += e;
        }
    }
    let n = group.len() as f32;
    for s in &mut sum {
        *s /= n;
    }
    sum
}

#[allow(clippy::too_many_arguments)]
async fn run_consolidation(
    store: &dyn crate::storage::MemoryStore,
    embedder: &dyn crate::embeddings::EmbeddingProvider,
    ttl_hours: u64,
    model: &str,
    api_key: &str,
    prune_deep_importance: f32,
    prune_deep_access_count: i32,
    prune_deep_age_days: i64,
    prune_consolid_insight: f32,
    prune_consolid_age_days: i64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let since = Utc::now() - chrono::TimeDelta::hours(ttl_hours as i64);

    let fresh_records = store.get_new_fresh_since("", since).await?;

    // Level 1: Shallow consolidation — semantic clustering via embedding cosim
    if !fresh_records.is_empty() {
        let groups = semantic_cluster(&fresh_records, 0.85);
        for records in groups.iter().filter(|g| g.len() >= 2) {
            let combined: Vec<String> = records.iter().map(|r| r.content.clone()).collect();
            if combined.is_empty() {
                continue;
            }
            let profile = records[0].profile.clone();
            let tags = records[0].tags.clone();
            let importance: f32 =
                records.iter().map(|r| r.importance).sum::<f32>() / records.len() as f32;
            let source_ids: Vec<i64> = records.iter().map(|r| r.id).collect();
            let summary = match call_llm_summarize(&combined.join("\n---\n"), model, api_key).await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("LLM summarization failed (skipping): {e}");
                    continue;
                }
            };
            if summary.len() < 20 {
                continue;
            }
            let embedding = match embedder.embed(&summary).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("embedding failed: {e}");
                    continue;
                }
            };
            let emotional = detect_emotional_tag(&combined.join(" "));
            let mut meta = json!({"source": "consolidation_worker", "depth": "shallow"});
            if let Some(ev) = emotional {
                meta["emotional_valence"] = json!(ev.0);
                meta["emotional_arousal"] = json!(ev.1);
            }
            let _ = store
                .store_consolid(
                    &profile,
                    &summary,
                    &embedding,
                    &source_ids,
                    &tags,
                    &meta,
                    importance,
                )
                .await;
            tracing::info!(
                "shallow consolidated {} fresh -> summary (len={})",
                source_ids.len(),
                summary.len()
            );
        }
    }

    // Level 2: Daily consolidation — group existing shallow consolid items by profile
    let since_daily = Utc::now() - chrono::TimeDelta::hours(24);
    let daily_deep = store.get_new_deep_since(since_daily).await?;
    if daily_deep.len() >= 5 {
        let by_profile: std::collections::HashMap<String, Vec<&crate::storage::MemoryRecord>> =
            daily_deep
                .iter()
                .fold(std::collections::HashMap::new(), |mut acc, r| {
                    acc.entry(r.profile.clone()).or_default().push(r);
                    acc
                });
        for (profile, records) in by_profile {
            if records.len() < 5 {
                continue;
            }
            let combined: Vec<String> = records.iter().map(|r| r.content.clone()).collect();
            let summary = match call_llm_summarize(&combined.join("\n---\n"), model, api_key).await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("daily LLM summarization failed: {e}");
                    continue;
                }
            };
            if summary.len() < 20 {
                continue;
            }
            let embedding = match embedder.embed(&summary).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("daily embedding failed: {e}");
                    continue;
                }
            };
            let source_ids: Vec<i64> = records.iter().map(|r| r.id).collect();
            let importance =
                records.iter().map(|r| r.importance).sum::<f32>() / records.len() as f32;
            let tags = records[0].tags.clone();
            let mut meta = json!({"source": "consolidation_worker", "depth": "daily"});
            if let Some(emotional) = detect_emotional_tag(&combined.join(" ")) {
                meta["emotional_valence"] = json!(emotional.0);
                meta["emotional_arousal"] = json!(emotional.1);
            }
            let _ = store
                .store_consolid(
                    &profile,
                    &summary,
                    &embedding,
                    &source_ids,
                    &tags,
                    &meta,
                    importance,
                )
                .await;
            tracing::info!("daily consolidated {} deep -> summary", source_ids.len());
        }
    }

    // Level 3: Weekly consolidation — group daily consolid summaries into weekly insights
    let since_weekly = Utc::now() - chrono::TimeDelta::days(7);
    let weekly_consolid = store
        .get_consolid_by_depth_since("daily", since_weekly)
        .await?;
    if weekly_consolid.len() >= 3 {
        let by_profile: std::collections::HashMap<
            String,
            Vec<&crate::storage::MemoryRecordConsolid>,
        > = weekly_consolid
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, r| {
                acc.entry(r.profile.clone()).or_default().push(r);
                acc
            });
        for (profile, records) in by_profile {
            if records.len() < 3 {
                continue;
            }
            let combined: Vec<String> = records.iter().map(|r| r.summary.clone()).collect();
            let summary = match call_llm_summarize(
                &format!(
                    "Weekly digest of daily consolidations:\n{}",
                    combined.join("\n---\n")
                ),
                model,
                api_key,
            )
            .await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("weekly LLM summarization failed: {e}");
                    continue;
                }
            };
            if summary.len() < 20 {
                continue;
            }
            let embedding = match embedder.embed(&summary).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("weekly embedding failed: {e}");
                    continue;
                }
            };
            let source_ids: Vec<i64> = records.iter().map(|r| r.id).collect();
            let importance =
                records.iter().map(|r| r.importance).sum::<f32>() / records.len() as f32;
            let tags = records[0].tags.clone();
            let mut meta = json!({"source": "consolidation_worker", "depth": "weekly"});
            if let Some(emotional) = detect_emotional_tag(&combined.join(" ")) {
                meta["emotional_valence"] = json!(emotional.0);
                meta["emotional_arousal"] = json!(emotional.1);
            }
            let _ = store
                .store_consolid(
                    &profile,
                    &summary,
                    &embedding,
                    &source_ids,
                    &tags,
                    &meta,
                    importance,
                )
                .await;
            tracing::info!("weekly consolidated {} daily -> insight", source_ids.len());
        }
    }

    // Level 4: Monthly consolidation — group weekly summaries into monthly insights
    let since_monthly = Utc::now() - chrono::TimeDelta::days(30);
    let monthly_consolid = store
        .get_consolid_by_depth_since("weekly", since_monthly)
        .await?;
    if monthly_consolid.len() >= 2 {
        let by_profile: std::collections::HashMap<
            String,
            Vec<&crate::storage::MemoryRecordConsolid>,
        > = monthly_consolid
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, r| {
                acc.entry(r.profile.clone()).or_default().push(r);
                acc
            });
        for (profile, records) in by_profile {
            let combined: Vec<String> = records.iter().map(|r| r.summary.clone()).collect();
            let summary = match call_llm_summarize(
                &format!(
                    "Monthly digest of weekly summaries:\n{}",
                    combined.join("\n---\n")
                ),
                model,
                api_key,
            )
            .await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("monthly LLM summarization failed: {e}");
                    continue;
                }
            };
            if summary.len() < 20 {
                continue;
            }
            let embedding = match embedder.embed(&summary).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("monthly embedding failed: {e}");
                    continue;
                }
            };
            let source_ids: Vec<i64> = records.iter().map(|r| r.id).collect();
            let importance =
                records.iter().map(|r| r.importance).sum::<f32>() / records.len() as f32;
            let tags = records[0].tags.clone();
            let mut meta = json!({"source": "consolidation_worker", "depth": "monthly"});
            if let Some(emotional) = detect_emotional_tag(&combined.join(" ")) {
                meta["emotional_valence"] = json!(emotional.0);
                meta["emotional_arousal"] = json!(emotional.1);
            }
            let _ = store
                .store_consolid(
                    &profile,
                    &summary,
                    &embedding,
                    &source_ids,
                    &tags,
                    &meta,
                    importance,
                )
                .await;
            tracing::info!(
                "monthly consolidated {} weekly -> insight",
                source_ids.len()
            );
        }
    }

    // Level 5: Yearly consolidation — group monthly summaries into yearly insights
    let since_yearly = Utc::now() - chrono::TimeDelta::days(365);
    let yearly_consolid = store
        .get_consolid_by_depth_since("monthly", since_yearly)
        .await?;
    if yearly_consolid.len() >= 2 {
        let by_profile: std::collections::HashMap<
            String,
            Vec<&crate::storage::MemoryRecordConsolid>,
        > = yearly_consolid
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, r| {
                acc.entry(r.profile.clone()).or_default().push(r);
                acc
            });
        for (profile, records) in by_profile {
            let combined: Vec<String> = records.iter().map(|r| r.summary.clone()).collect();
            let summary = match call_llm_summarize(
                &format!(
                    "Yearly digest of monthly summaries:\n{}",
                    combined.join("\n---\n")
                ),
                model,
                api_key,
            )
            .await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("yearly LLM summarization failed: {e}");
                    continue;
                }
            };
            if summary.len() < 20 {
                continue;
            }
            let embedding = match embedder.embed(&summary).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("yearly embedding failed: {e}");
                    continue;
                }
            };
            let source_ids: Vec<i64> = records.iter().map(|r| r.id).collect();
            let importance =
                records.iter().map(|r| r.importance).sum::<f32>() / records.len() as f32;
            let tags = records[0].tags.clone();
            let mut meta = json!({"source": "consolidation_worker", "depth": "yearly"});
            if let Some(emotional) = detect_emotional_tag(&combined.join(" ")) {
                meta["emotional_valence"] = json!(emotional.0);
                meta["emotional_arousal"] = json!(emotional.1);
            }
            let _ = store
                .store_consolid(
                    &profile,
                    &summary,
                    &embedding,
                    &source_ids,
                    &tags,
                    &meta,
                    importance,
                )
                .await;
            tracing::info!(
                "yearly consolidated {} monthly -> insight",
                source_ids.len()
            );
        }
    }

    // Active forgetting: prune low-value memories
    let pruned_deep = store
        .prune_deep(
            prune_deep_importance,
            prune_deep_access_count,
            prune_deep_age_days,
        )
        .await
        .unwrap_or(0);
    let pruned_consolid = store
        .prune_consolid(prune_consolid_insight, prune_consolid_age_days)
        .await
        .unwrap_or(0);
    if pruned_deep > 0 || pruned_consolid > 0 {
        tracing::info!(
            "active forgetting: pruned {} deep, {} consolid",
            pruned_deep,
            pruned_consolid
        );
    }

    Ok(())
}

fn detect_emotional_tag(text: &str) -> Option<(f32, f32)> {
    let keywords = [
        ("excelente", 0.8, 0.7),
        ("genial", 0.7, 0.6),
        ("perfecto", 0.9, 0.5),
        ("mal", -0.7, 0.6),
        ("error", -0.6, 0.7),
        ("problema", -0.5, 0.6),
        ("urgente", -0.3, 0.9),
        ("importante", 0.0, 0.8),
        ("me encanta", 0.9, 0.6),
        ("odio", -0.9, 0.8),
        ("feliz", 0.8, 0.5),
        ("triste", -0.7, 0.4),
        ("frustrado", -0.6, 0.8),
        ("éxito", 0.8, 0.5),
    ];
    let lower = text.to_lowercase();
    for (word, val, aro) in &keywords {
        if lower.contains(word) {
            return Some((*val, *aro));
        }
    }
    None
}

async fn call_llm_summarize(
    text: &str,
    model: &str,
    api_key: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if api_key.is_empty() {
        return Err("no API key configured".into());
    }
    let system_prompt = "You are a memory consolidation assistant. Extract the key insights, decisions, and important details from the conversation turns below. Produce a concise single-paragraph summary that captures what was discussed and decided. Omit small talk and repetition.";
    let body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    });
    let client = reqwest::Client::new();
    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;
    let data: serde_json::Value = resp.json().await?;
    let content = data["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| format!("unexpected OpenRouter response: {}", data))?
        .to_string();
    Ok(content)
}
