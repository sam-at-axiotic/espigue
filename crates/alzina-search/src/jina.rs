//! Jina v3 embedding HTTP client.
//!
//! Implements [`EmbeddingService`](alzina_core::EmbeddingService) over the
//! Jina AI embeddings REST API. Used by the hybrid search service to convert
//! query and passage text into dense vectors.
//!
//! ## Endpoint
//!
//! `POST https://api.jina.ai/v1/embeddings` with bearer-token auth.
//!
//! ## Task prefixes
//!
//! Jina v3 uses task-specific prompts:
//! - [`EmbeddingTask::Passage`] -> `"retrieval.passage"` (indexing)
//! - [`EmbeddingTask::Query`] -> `"retrieval.query"` (searching)
//!
//! ## Batching
//!
//! Inputs above 128 are split into chunks of 128 (Jina's per-call limit) and
//! the resulting vectors are concatenated in input order.
//!
//! ## AC-1 (loud degradation)
//!
//! Every error path returns `AlzinaError::Search(SearchDetail { degraded: true,
//! degradation_reason: Some(...) })` so callers (e.g. `HybridSearchService`)
//! can surface a clear reason to the agent. Each degradation is logged at
//! `tracing::warn!`. Rate limiting is NOT done in-client — callers decide
//! backoff policy on top of the surfaced 429.

use alzina_core::error::{AlzinaError, AlzinaResult, SearchDetail};
use alzina_core::search::{EmbeddingService, EmbeddingTask};
use async_trait::async_trait;

/// Default Jina API base URL. Tests override via [`JinaEmbeddingService::with_base_url`].
const DEFAULT_BASE_URL: &str = "https://api.jina.ai/v1";

/// Maximum number of inputs accepted by Jina in a single `/embeddings` call.
const MAX_BATCH_SIZE: usize = 128;

/// HTTP timeout for the underlying `reqwest::Client`.
const HTTP_TIMEOUT_SECS: u64 = 30;

/// Jina v3 embedding service.
///
/// Construct with [`JinaEmbeddingService::new`] for production. Tests can use
/// [`JinaEmbeddingService::with_base_url`] to point at a `wiremock::MockServer`.
pub struct JinaEmbeddingService {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimensions: usize,
    /// Test affordance — production callers always go through `new(...)`.
    base_url: String,
}

impl JinaEmbeddingService {
    /// Construct a new Jina embedding client pointed at the production endpoint
    /// (`https://api.jina.ai/v1`).
    ///
    /// Returns a degraded `Search` error if `api_key` is empty or if the
    /// underlying `reqwest::Client` fails to build.
    pub fn new(api_key: String, model: String, dimensions: usize) -> AlzinaResult<Self> {
        if api_key.is_empty() {
            return Err(AlzinaError::Search(SearchDetail {
                message: "JinaEmbeddingService: api_key is empty".into(),
                degraded: true,
                degradation_reason: Some("No JINA_API_KEY configured".into()),
            }));
        }
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(HTTP_TIMEOUT_SECS))
            .build()
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("reqwest client build: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!("reqwest client build failed: {e}")),
                })
            })?;
        Ok(Self {
            client,
            api_key,
            model,
            dimensions,
            base_url: DEFAULT_BASE_URL.into(),
        })
    }

    /// Test-only constructor that accepts a custom base URL (e.g. a
    /// `wiremock::MockServer` URI). Production code should always use
    /// [`Self::new`].
    pub fn with_base_url(
        api_key: String,
        model: String,
        dimensions: usize,
        base_url: String,
    ) -> AlzinaResult<Self> {
        let mut svc = Self::new(api_key, model, dimensions)?;
        svc.base_url = base_url;
        Ok(svc)
    }
}

/// Private deserialisation envelope for Jina's `/embeddings` response.
#[derive(serde::Deserialize)]
struct JinaResponse {
    data: Vec<JinaEmbeddingItem>,
}

#[derive(serde::Deserialize)]
struct JinaEmbeddingItem {
    /// Index of the input this embedding corresponds to. Used to defend against
    /// out-of-order responses (Jina docs do not promise ordering).
    #[serde(default)]
    index: usize,
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingService for JinaEmbeddingService {
    async fn embed(&self, text: &str, task: EmbeddingTask) -> AlzinaResult<Vec<f32>> {
        let mut result = self.embed_batch(&[text.to_string()], task).await?;
        if result.is_empty() {
            return Err(AlzinaError::Search(SearchDetail {
                message: "Jina returned empty data".into(),
                degraded: true,
                degradation_reason: Some("Jina API returned no embeddings".into()),
            }));
        }
        Ok(result.remove(0))
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        task: EmbeddingTask,
    ) -> AlzinaResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Chunk over the per-call limit; concatenate results in input order.
        if texts.len() > MAX_BATCH_SIZE {
            let mut out = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(MAX_BATCH_SIZE) {
                let part = self.embed_batch_one_call(chunk, task).await?;
                out.extend(part);
            }
            return Ok(out);
        }

        self.embed_batch_one_call(texts, task).await
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl JinaEmbeddingService {
    /// Single round-trip to the Jina API for up to [`MAX_BATCH_SIZE`] inputs.
    async fn embed_batch_one_call(
        &self,
        texts: &[String],
        task: EmbeddingTask,
    ) -> AlzinaResult<Vec<Vec<f32>>> {
        debug_assert!(texts.len() <= MAX_BATCH_SIZE);
        debug_assert!(!texts.is_empty());

        let task_str = match task {
            EmbeddingTask::Passage => "retrieval.passage",
            EmbeddingTask::Query => "retrieval.query",
        };

        let body = serde_json::json!({
            "model": self.model,
            "task": task_str,
            "dimensions": self.dimensions,
            "embedding_type": "float",
            "input": texts,
        });

        let url = format!("{}/embeddings", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(
                    error = %e,
                    "Jina API request failed (network); search will degrade to FTS5-only"
                );
                AlzinaError::Search(SearchDetail {
                    message: format!("Jina request failed: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!(
                        "Jina API unavailable: {e}, falling back to FTS5"
                    )),
                })
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            tracing::warn!(status = %status, body = %body_text, "Jina API non-2xx response");
            let reason = match status.as_u16() {
                429 => "Jina rate-limited (429), falling back to FTS5".to_string(),
                401 | 403 => {
                    "Jina auth failed (check JINA_API_KEY), falling back to FTS5".to_string()
                }
                _ => format!("Jina returned {status}, falling back to FTS5"),
            };
            return Err(AlzinaError::Search(SearchDetail {
                message: format!("Jina HTTP {status}: {body_text}"),
                degraded: true,
                degradation_reason: Some(reason),
            }));
        }

        let parsed: JinaResponse = resp.json().await.map_err(|e| {
            tracing::warn!(error = %e, "Jina API response JSON decode failed");
            AlzinaError::Search(SearchDetail {
                message: format!("Jina response decode failed: {e}"),
                degraded: true,
                degradation_reason: Some(format!(
                    "Jina returned invalid JSON: {e}, falling back to FTS5"
                )),
            })
        })?;

        // Validate count matches the inputs we sent.
        if parsed.data.len() != texts.len() {
            let msg = format!(
                "Jina returned {} embeddings for {} inputs",
                parsed.data.len(),
                texts.len()
            );
            tracing::warn!(
                expected = texts.len(),
                actual = parsed.data.len(),
                "Jina embedding count mismatch"
            );
            return Err(AlzinaError::Search(SearchDetail {
                message: msg.clone(),
                degraded: true,
                degradation_reason: Some(format!(
                    "Jina embedding count/length mismatch: {msg}, falling back to FTS5"
                )),
            }));
        }

        // Sort by index defensively — docs do not promise ordering.
        let mut data = parsed.data;
        data.sort_by_key(|item| item.index);

        // Validate dimensionality of every embedding.
        let expected_dim = self.dimensions;
        let mut out = Vec::with_capacity(data.len());
        for (i, item) in data.into_iter().enumerate() {
            if item.embedding.len() != expected_dim {
                let msg = format!(
                    "Jina embedding[{i}] has dimension {} (expected {expected_dim})",
                    item.embedding.len()
                );
                tracing::warn!(
                    expected = expected_dim,
                    actual = item.embedding.len(),
                    "Jina embedding dimension mismatch"
                );
                return Err(AlzinaError::Search(SearchDetail {
                    message: msg.clone(),
                    degraded: true,
                    degradation_reason: Some(format!(
                        "Jina embedding dimension mismatch: {msg}, falling back to FTS5"
                    )),
                }));
            }
            out.push(item.embedding);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    /// Build a JSON Jina response with `count` embeddings each of `dim` length.
    /// Each embedding is `index = i`, `embedding = vec![0.1; dim]`.
    fn mock_response(count: usize, dim: usize) -> serde_json::Value {
        let data: Vec<serde_json::Value> = (0..count)
            .map(|i| {
                json!({
                    "object": "embedding",
                    "index": i,
                    "embedding": vec![0.1f32; dim],
                })
            })
            .collect();
        json!({
            "model": "jina-embeddings-v3",
            "object": "list",
            "usage": {"total_tokens": 4, "prompt_tokens": 4},
            "data": data,
        })
    }

    /// Responder that returns `count` embeddings sized to the SERVICE's
    /// expected dimension. Used for batching tests where input length varies.
    struct EchoResponder {
        dim: usize,
    }

    impl Respond for EchoResponder {
        fn respond(&self, req: &Request) -> ResponseTemplate {
            let parsed: serde_json::Value =
                serde_json::from_slice(&req.body).expect("test request body must parse as JSON");
            let inputs = parsed
                .get("input")
                .and_then(|v| v.as_array())
                .expect("test request must have an 'input' array");
            let count = inputs.len();
            ResponseTemplate::new(200).set_body_json(mock_response(count, self.dim))
        }
    }

    #[tokio::test]
    async fn embed_returns_vector_on_200() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_response(1, 1024)))
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .expect("service builds");

        let v = svc
            .embed("hello", EmbeddingTask::Query)
            .await
            .expect("embed succeeds");
        assert_eq!(v.len(), 1024);
        for (i, x) in v.iter().enumerate() {
            assert!(
                (x - 0.1).abs() < 1e-6,
                "embedding[{i}] = {x}, expected ~0.1"
            );
        }
    }

    #[tokio::test]
    async fn embed_batch_chunks_above_128() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(EchoResponder { dim: 1024 })
            .expect(2)
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .unwrap();

        let texts: Vec<String> = (0..200).map(|i| format!("text-{i}")).collect();
        let out = svc
            .embed_batch(&texts, EmbeddingTask::Passage)
            .await
            .expect("batch succeeds");
        assert_eq!(out.len(), 200);
        for v in &out {
            assert_eq!(v.len(), 1024);
        }
        // Mock's `.expect(2)` is verified on Drop of the MockServer.
    }

    #[tokio::test]
    async fn embed_returns_search_error_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .unwrap();

        let err = svc
            .embed("hello", EmbeddingTask::Query)
            .await
            .expect_err("must error on 429");
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded, "must be flagged degraded");
                let reason = detail
                    .degradation_reason
                    .expect("degradation_reason must be set");
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("rate-limited") || lower.contains("429"),
                    "reason should mention rate limiting/429: got {reason:?}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn embed_returns_search_error_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .unwrap();

        let err = svc
            .embed("hello", EmbeddingTask::Query)
            .await
            .expect_err("must error on 401");
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded);
                let reason = detail.degradation_reason.expect("reason set");
                assert!(
                    reason.to_lowercase().contains("auth"),
                    "reason should mention auth: got {reason:?}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn embed_returns_search_error_on_500() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .unwrap();

        let err = svc
            .embed("hello", EmbeddingTask::Query)
            .await
            .expect_err("must error on 500");
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded);
                let reason = detail.degradation_reason.expect("reason set");
                assert!(
                    reason.contains("500"),
                    "reason should mention status 500: got {reason:?}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn embed_returns_search_error_on_dimension_mismatch() {
        let server = MockServer::start().await;
        // Service expects 1024 but server returns 512-dim vectors.
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_response(1, 512)))
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .unwrap();

        let err = svc
            .embed("hello", EmbeddingTask::Query)
            .await
            .expect_err("must error on dim mismatch");
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded);
                let reason = detail.degradation_reason.expect("reason set");
                assert!(
                    reason.to_lowercase().contains("dimension"),
                    "reason should mention dimension: got {reason:?}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn embed_returns_search_error_on_count_mismatch() {
        let server = MockServer::start().await;
        // Send 3 inputs; mock returns only 2 embeddings.
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_response(2, 1024)))
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .unwrap();

        let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let err = svc
            .embed_batch(&texts, EmbeddingTask::Passage)
            .await
            .expect_err("must error on count mismatch");
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded);
                let reason = detail.degradation_reason.expect("reason set");
                let lower = reason.to_lowercase();
                assert!(
                    lower.contains("count")
                        || lower.contains("length")
                        || lower.contains("mismatch"),
                    "reason should mention count/length/mismatch: got {reason:?}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn task_prefix_passage_serialised_correctly() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(json!({"task": "retrieval.passage"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_response(1, 1024)))
            .expect(1)
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .unwrap();

        let _ = svc
            .embed("hello", EmbeddingTask::Passage)
            .await
            .expect("embed succeeds");
        // Mock expectation verified on Drop.
    }

    #[tokio::test]
    async fn task_prefix_query_serialised_correctly() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(json!({"task": "retrieval.query"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_response(1, 1024)))
            .expect(1)
            .mount(&server)
            .await;

        let svc = JinaEmbeddingService::with_base_url(
            "test_key".into(),
            "jina-embeddings-v3".into(),
            1024,
            server.uri(),
        )
        .unwrap();

        let _ = svc
            .embed("hello", EmbeddingTask::Query)
            .await
            .expect("embed succeeds");
    }

    #[test]
    fn new_with_empty_api_key_errors() {
        let res = JinaEmbeddingService::new("".into(), "jina-embeddings-v3".into(), 1024);
        let err = match res {
            Ok(_) => panic!("expected error for empty api_key, got Ok"),
            Err(e) => e,
        };
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded);
                let reason = detail.degradation_reason.expect("reason set");
                assert!(
                    reason.to_lowercase().contains("jina_api_key")
                        || reason.to_lowercase().contains("api_key")
                        || reason.to_lowercase().contains("api key"),
                    "reason should mention api_key: got {reason:?}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }
}
