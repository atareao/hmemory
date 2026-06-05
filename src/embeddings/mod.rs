pub mod ollama;
pub mod openrouter;

use async_trait::async_trait;

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str)
    -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>;
}

pub fn from_config(config: &crate::config::Config) -> Box<dyn EmbeddingProvider> {
    match config.embedding_provider {
        crate::config::EmbeddingProviderKind::OpenRouter => {
            Box::new(openrouter::OpenRouterEmbedder::new(
                config.openrouter_api_key.clone().unwrap_or_default(),
                config.openrouter_model.clone(),
            ))
        }
        crate::config::EmbeddingProviderKind::Ollama => Box::new(ollama::OllamaEmbedder::new(
            config.ollama_base_url.clone(),
            config.ollama_model.clone(),
        )),
    }
}
