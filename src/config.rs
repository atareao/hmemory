use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub port: u16,
    pub database_url: String,
    pub embedding_provider: EmbeddingProviderKind,
    pub openrouter_api_key: Option<String>,
    pub openrouter_model: String,
    pub ollama_base_url: String,
    pub ollama_model: String,
    pub hybrid_alpha: f32,
    pub decay_half_life_days: f64,
    pub sync_turn_min_importance: f32,
    pub prefetch_cadence: u64,
    pub sync_turn_cadence: u64,
    pub conclusion_cadence: u64,
    pub context_tokens: Option<usize>,
    pub base_context_cadence: u64,
    pub rerank_enabled: bool,
    pub rerank_model: String,
    pub batch_size: usize,
    pub batch_idle_seconds: u64,
    pub fresh_ttl_hours: u64,
    pub consolidation_cadence: u64,
    pub consolidation_model: String,
    pub prune_deep_importance: f32,
    pub prune_deep_access_count: i32,
    pub prune_deep_age_days: i64,
    pub prune_consolid_insight: f32,
    pub prune_consolid_age_days: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmbeddingProviderKind {
    OpenRouter,
    Ollama,
}

impl Config {
    pub fn from_env() -> Self {
        let provider = env::var("EMBEDDING_PROVIDER")
            .unwrap_or_else(|_| panic!("EMBEDDING_PROVIDER must be set (openrouter|ollama)"));

        let embedding_provider = match provider.as_str() {
            "openrouter" => EmbeddingProviderKind::OpenRouter,
            "ollama" => EmbeddingProviderKind::Ollama,
            other => {
                panic!("unknown EMBEDDING_PROVIDER '{other}'; expected 'openrouter' or 'ollama'")
            }
        };

        Self {
            port: env::var("PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgres://localhost:5432/hmemory".to_string()),
            embedding_provider,
            openrouter_api_key: env::var("OPENROUTER_API_KEY").ok(),
            openrouter_model: env::var("OPENROUTER_MODEL")
                .unwrap_or_else(|_| "openai/text-embedding-3-small".to_string()),
            ollama_base_url: env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            ollama_model: env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".to_string()),
            hybrid_alpha: env::var("HYBRID_ALPHA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.5),
            decay_half_life_days: env::var("DECAY_HALF_LIFE_DAYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30.0),
            sync_turn_min_importance: env::var("SYNC_TURN_MIN_IMPORTANCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.3),
            prefetch_cadence: env::var("PREFETCH_CADENCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            sync_turn_cadence: env::var("SYNC_TURN_CADENCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            conclusion_cadence: env::var("CONCLUSION_CADENCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            context_tokens: env::var("CONTEXT_TOKENS").ok().and_then(|v| v.parse().ok()),
            base_context_cadence: env::var("BASE_CONTEXT_CADENCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            rerank_enabled: env::var("RERANK_ENABLED")
                .ok()
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            rerank_model: env::var("RERANK_MODEL")
                .unwrap_or_else(|_| "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string()),
            batch_size: env::var("BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            batch_idle_seconds: env::var("BATCH_IDLE_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
            fresh_ttl_hours: env::var("FRESH_TTL_HOURS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(48),
            consolidation_cadence: env::var("CONSOLIDATION_CADENCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600),
            consolidation_model: env::var("CONSOLIDATION_MODEL")
                .unwrap_or_else(|_| "deepseek/deepseek-chat".to_string()),
            prune_deep_importance: env::var("PRUNE_DEEP_IMPORTANCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.05),
            prune_deep_access_count: env::var("PRUNE_DEEP_ACCESS_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            prune_deep_age_days: env::var("PRUNE_DEEP_AGE_DAYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(90),
            prune_consolid_insight: env::var("PRUNE_CONSOLID_INSIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.05),
            prune_consolid_age_days: env::var("PRUNE_CONSOLID_AGE_DAYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_env_defaults() {
        temp_env::with_vars(
            [
                ("EMBEDDING_PROVIDER", Some("ollama")),
                ("PORT", None),
                ("DATABASE_URL", None),
                ("OPENROUTER_API_KEY", None),
                ("OPENROUTER_MODEL", None),
                ("OLLAMA_BASE_URL", None),
                ("OLLAMA_MODEL", None),
            ],
            || {
                let cfg = Config::from_env();
                assert_eq!(cfg.port, 8080);
                assert_eq!(cfg.embedding_provider, EmbeddingProviderKind::Ollama);
                assert_eq!(cfg.ollama_base_url, "http://localhost:11434");
                assert_eq!(cfg.ollama_model, "nomic-embed-text");
                assert!(cfg.openrouter_api_key.is_none());
            },
        );
    }

    #[test]
    fn test_from_env_openrouter() {
        temp_env::with_vars(
            [
                ("EMBEDDING_PROVIDER", Some("openrouter")),
                ("PORT", Some("9090")),
                ("OPENROUTER_API_KEY", Some("sk-test")),
                ("OPENROUTER_MODEL", Some("custom-model")),
                ("OLLAMA_BASE_URL", None),
                ("OLLAMA_MODEL", None),
            ],
            || {
                let cfg = Config::from_env();
                assert_eq!(cfg.port, 9090);
                assert_eq!(cfg.embedding_provider, EmbeddingProviderKind::OpenRouter);
                assert_eq!(cfg.openrouter_api_key, Some("sk-test".into()));
                assert_eq!(cfg.openrouter_model, "custom-model");
            },
        );
    }

    #[test]
    #[should_panic(expected = "EMBEDDING_PROVIDER must be set")]
    fn test_from_env_missing_provider() {
        temp_env::with_vars([("EMBEDDING_PROVIDER", None::<&str>)], || {
            Config::from_env();
        });
    }

    #[test]
    #[should_panic(expected = "unknown EMBEDDING_PROVIDER")]
    fn test_from_env_invalid_provider() {
        temp_env::with_vars([("EMBEDDING_PROVIDER", Some("invalid"))], || {
            Config::from_env();
        });
    }
}
