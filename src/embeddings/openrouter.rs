use super::EmbeddingProvider;
use async_trait::async_trait;
use serde::Deserialize;

pub struct OpenRouterEmbedder {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl OpenRouterEmbedder {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenRouterEmbedder {
    async fn embed(
        &self,
        text: &str,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });

        let resp = self
            .client
            .post("https://openrouter.ai/api/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        let result: EmbeddingResponse = resp.error_for_status()?.json().await?;
        Ok(result
            .data
            .into_iter()
            .next()
            .ok_or("empty embedding response")?
            .embedding)
    }
}
