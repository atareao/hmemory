use super::EmbeddingProvider;
use async_trait::async_trait;
use serde::Deserialize;

pub struct OllamaEmbedder {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Vec<f32>,
}

impl OllamaEmbedder {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            base_url,
            model,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbedder {
    async fn embed(
        &self,
        text: &str,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let body = serde_json::json!({
            "model": self.model,
            "prompt": text,
        });

        let url = format!("{}/api/embeddings", self.base_url.trim_end_matches('/'));
        let resp = self.client.post(&url).json(&body).send().await?;
        let result: OllamaEmbeddingResponse = resp.error_for_status()?.json().await?;
        Ok(result.embedding)
    }
}
