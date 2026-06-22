//! OpenRouter embeddings client.
//!
//! Implements [`EmbeddingService`] via OpenRouter's OpenAI-compatible
//! `POST /embeddings`. Default model `openai/text-embedding-3-small` at 1536
//! dims; the dimension is configurable and sent as the `dimensions` request
//! field so the configured value is authoritative (the response length is
//! checked against it). `EmbeddingTask` is ignored — OpenRouter/OpenAI
//! embeddings do not use Jina-style task prefixes.
//!
//! Loud-degrade contract (project-wide): every error returns
//! `AlzinaError::Search { degraded: true, .. }` so a caller never silently
//! drops a source on an embedding outage.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use base::error::{AlzinaError, SearchDetail};
use base::search::{EmbeddingService, EmbeddingTask};
use base::AlzinaResult;

use super::DEFAULT_BASE_URL;

const HTTP_TIMEOUT_SECS: u64 = 120;
/// Max inputs per request. OpenAI-compatible endpoints accept arrays; keep the
/// batch bounded (mirrors the Jina client).
const MAX_BATCH_SIZE: usize = 128;

/// Default embedding model — balanced cost/quality, 1536 dims.
pub const DEFAULT_MODEL: &str = "openai/text-embedding-3-small";
/// Default embedding dimension for [`DEFAULT_MODEL`].
pub const DEFAULT_DIMENSIONS: usize = 1536;

fn degraded(message: impl Into<String>, reason: impl Into<String>) -> AlzinaError {
    AlzinaError::Search(SearchDetail {
        message: message.into(),
        degraded: true,
        degradation_reason: Some(reason.into()),
    })
}

/// OpenRouter embeddings client.
pub struct OpenRouterEmbeddingService {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimensions: usize,
    base_url: String,
}

impl OpenRouterEmbeddingService {
    /// Construct a client pointed at the production OpenRouter endpoint.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimensions: usize,
    ) -> AlzinaResult<Self> {
        Self::build(api_key.into(), model.into(), dimensions, DEFAULT_BASE_URL.to_string())
    }

    /// Test-only constructor accepting a custom base URL.
    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimensions: usize,
        base_url: impl Into<String>,
    ) -> AlzinaResult<Self> {
        Self::build(api_key.into(), model.into(), dimensions, base_url.into())
    }

    fn build(
        api_key: String,
        model: String,
        dimensions: usize,
        base_url: String,
    ) -> AlzinaResult<Self> {
        if api_key.is_empty() {
            return Err(degraded(
                "OpenRouterEmbeddingService: api_key is empty",
                "No OPENROUTER_API_KEY configured",
            ));
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
            .build()
            .map_err(|e| degraded(format!("reqwest client build: {e}"), format!("{e}")))?;
        Ok(Self { client, api_key, model, dimensions, base_url })
    }

    async fn embed_inner(&self, inputs: &[String]) -> AlzinaResult<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        let url = format!("{}/embeddings", self.base_url);

        for chunk in inputs.chunks(MAX_BATCH_SIZE) {
            let body = json!({
                "model": self.model,
                "input": chunk,
                "dimensions": self.dimensions,
            });
            let resp = self
                .client
                .post(&url)
                .bearer_auth(&self.api_key)
                .header("HTTP-Referer", "https://github.com/axiotic/espigue")
                .header("X-Title", "espigue")
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    tracing::warn!(error = %e, "OpenRouter embeddings request failed (network)");
                    degraded(
                        format!("OpenRouter embeddings request failed: {e}"),
                        format!("OpenRouter embeddings unavailable: {e}"),
                    )
                })?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
                tracing::warn!(status = %status, body = %text, "OpenRouter embeddings non-2xx");
                let reason = match status.as_u16() {
                    429 => "OpenRouter embeddings rate-limited (429)".to_string(),
                    401 | 403 => "OpenRouter embeddings auth failed (check OPENROUTER_API_KEY)".to_string(),
                    _ => format!("OpenRouter embeddings returned {status}"),
                };
                return Err(degraded(
                    format!("OpenRouter embeddings HTTP {status}: {text}"),
                    reason,
                ));
            }

            let parsed: EmbeddingResponse = resp.json().await.map_err(|e| {
                degraded(
                    format!("OpenRouter embeddings decode failed: {e}"),
                    format!("OpenRouter embeddings returned invalid JSON: {e}"),
                )
            })?;

            // Reorder defensively by the response `index` — never trust server
            // ordering for a positional caller.
            let mut rows = parsed.data;
            rows.sort_by_key(|d| d.index);
            for d in rows {
                if d.embedding.len() != self.dimensions {
                    return Err(degraded(
                        format!(
                            "OpenRouter embeddings dim mismatch: got {}, expected {}",
                            d.embedding.len(),
                            self.dimensions
                        ),
                        "embedding dimension mismatch (model/dim misconfigured)",
                    ));
                }
                out.push(d.embedding);
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl EmbeddingService for OpenRouterEmbeddingService {
    async fn embed(&self, text: &str, _task: EmbeddingTask) -> AlzinaResult<Vec<f32>> {
        let mut v = self.embed_inner(std::slice::from_ref(&text.to_string())).await?;
        v.pop()
            .ok_or_else(|| degraded("OpenRouter embeddings returned no vector", "empty response"))
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        _task: EmbeddingTask,
    ) -> AlzinaResult<Vec<Vec<f32>>> {
        self.embed_inner(texts).await
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingRow>,
}

#[derive(Deserialize)]
struct EmbeddingRow {
    index: usize,
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn embed_returns_vector_of_configured_dim() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(json!({ "model": "m", "dimensions": 3 })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "index": 0, "embedding": [0.1, 0.2, 0.3] }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let svc = OpenRouterEmbeddingService::with_base_url("k", "m", 3, server.uri()).unwrap();
        let v = svc.embed("hello", EmbeddingTask::Query).await.expect("embed ok");
        assert_eq!(v, vec![0.1, 0.2, 0.3]);
        assert_eq!(svc.dimensions(), 3);
    }

    #[tokio::test]
    async fn embed_batch_reorders_by_index() {
        let server = MockServer::start().await;
        // Server returns rows out of order; we must restore input order.
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    { "index": 1, "embedding": [0.9, 0.9] },
                    { "index": 0, "embedding": [0.1, 0.1] }
                ]
            })))
            .mount(&server)
            .await;

        let svc = OpenRouterEmbeddingService::with_base_url("k", "m", 2, server.uri()).unwrap();
        let out = svc
            .embed_batch(&["a".into(), "b".into()], EmbeddingTask::Passage)
            .await
            .expect("batch ok");
        assert_eq!(out, vec![vec![0.1, 0.1], vec![0.9, 0.9]]);
    }

    #[tokio::test]
    async fn dim_mismatch_is_loud_degrade() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "index": 0, "embedding": [0.1, 0.2] }]
            })))
            .mount(&server)
            .await;

        // Configured dim 5 but server returns 2 → must error, not silently pass.
        let svc = OpenRouterEmbeddingService::with_base_url("k", "m", 5, server.uri()).unwrap();
        let err = svc.embed("x", EmbeddingTask::Query).await.expect_err("dim mismatch errors");
        match err {
            AlzinaError::Search(d) => assert!(d.degraded && d.message.contains("dim mismatch")),
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_input_no_network_call() {
        let server = MockServer::start().await; // no mock mounted
        let svc = OpenRouterEmbeddingService::with_base_url("k", "m", 3, server.uri()).unwrap();
        let out = svc.embed_batch(&[], EmbeddingTask::Passage).await.expect("empty ok");
        assert!(out.is_empty());
    }

    #[test]
    fn empty_key_errors() {
        assert!(OpenRouterEmbeddingService::new("", "m", 1536).is_err());
    }
}
