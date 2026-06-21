//! Jina cross-encoder reranker HTTP client.
//!
//! A bi-encoder (the embedding model in [`jina`](crate::jina)) ranks by cosine
//! similarity — good recall, but it cannot separate the *methodological core* of
//! a query from a topical false-friend that merely shares surface vocabulary
//! (e.g. "multi-agent debate" for phishing detection vs. for consensus quality).
//! A cross-encoder reads the query and each candidate *together*, so it scores
//! those false-friends low. This client calls Jina's `/rerank` endpoint to get
//! that precision signal over the fused candidate list.
//!
//! ## Endpoint
//!
//! `POST https://api.jina.ai/v1/rerank` with bearer-token auth — the SAME
//! `JINA_API_KEY` as the embedder. No new infra.
//!
//! ## Loud degradation (project-wide contract)
//!
//! Every error path returns `AlzinaError::Search(SearchDetail { degraded: true,
//! .. })` and logs at `tracing::warn!`. The caller keeps the un-reranked hits
//! and folds the reason — a reranker outage must never drop a source silently.

use alzina_core::error::{AlzinaError, AlzinaResult, SearchDetail};

/// Default Jina API base URL. Tests override via [`JinaRerankService::with_base_url`].
const DEFAULT_BASE_URL: &str = "https://api.jina.ai/v1";

/// HTTP timeout for the underlying `reqwest::Client`.
const HTTP_TIMEOUT_SECS: u64 = 30;

/// One reranked candidate: the index into the input `documents` slice and the
/// cross-encoder relevance score (Jina returns it in `[0.0, 1.0]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankResult {
    /// Index into the `documents` slice passed to [`JinaRerankService::rerank`].
    pub index: usize,
    /// Cross-encoder relevance score, higher is more relevant.
    pub score: f32,
}

/// Jina cross-encoder reranker client.
///
/// Construct with [`JinaRerankService::new`] for production. Tests use
/// [`JinaRerankService::with_base_url`] to point at a `wiremock::MockServer`.
pub struct JinaRerankService {
    client: reqwest::Client,
    api_key: String,
    model: String,
    /// Test affordance — production callers always go through `new(...)`.
    base_url: String,
}

impl JinaRerankService {
    /// Construct a reranker client pointed at the production endpoint.
    ///
    /// Returns a degraded `Search` error if `api_key` is empty or the
    /// underlying `reqwest::Client` fails to build.
    pub fn new(api_key: String, model: String) -> AlzinaResult<Self> {
        if api_key.is_empty() {
            return Err(AlzinaError::Search(SearchDetail {
                message: "JinaRerankService: api_key is empty".into(),
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
            base_url: DEFAULT_BASE_URL.into(),
        })
    }

    /// Test-only constructor that accepts a custom base URL.
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> AlzinaResult<Self> {
        let mut svc = Self::new(api_key, model)?;
        svc.base_url = base_url;
        Ok(svc)
    }

    /// Rerank `documents` against `query`.
    ///
    /// Returns one [`RerankResult`] per input document, **sorted by score
    /// descending** (Jina's default ordering), each carrying its original index
    /// into `documents`. An empty `documents` slice returns an empty vec without
    /// a network call.
    ///
    /// Every error is a loud-degrade `Search` error — the caller keeps the
    /// un-reranked order and folds the reason.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> AlzinaResult<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": false,
        });

        let url = format!("{}/rerank", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "Jina rerank request failed (network)");
                AlzinaError::Search(SearchDetail {
                    message: format!("Jina rerank request failed: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!("Jina rerank unavailable: {e}")),
                })
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            tracing::warn!(status = %status, body = %body_text, "Jina rerank non-2xx response");
            let reason = match status.as_u16() {
                429 => "Jina rerank rate-limited (429)".to_string(),
                401 | 403 => "Jina rerank auth failed (check JINA_API_KEY)".to_string(),
                _ => format!("Jina rerank returned {status}"),
            };
            return Err(AlzinaError::Search(SearchDetail {
                message: format!("Jina rerank HTTP {status}: {body_text}"),
                degraded: true,
                degradation_reason: Some(reason),
            }));
        }

        let parsed: RerankResponse = resp.json().await.map_err(|e| {
            tracing::warn!(error = %e, "Jina rerank response JSON decode failed");
            AlzinaError::Search(SearchDetail {
                message: format!("Jina rerank response decode failed: {e}"),
                degraded: true,
                degradation_reason: Some(format!("Jina rerank returned invalid JSON: {e}")),
            })
        })?;

        // Defend against out-of-range indices (an index past our input would
        // corrupt the caller's reorder). Drop any such row loudly rather than
        // panic on a later get().
        let n = documents.len();
        let mut out = Vec::with_capacity(parsed.results.len());
        for r in parsed.results {
            if r.index >= n {
                tracing::warn!(
                    index = r.index,
                    n,
                    "Jina rerank returned out-of-range index — dropping that result"
                );
                continue;
            }
            out.push(RerankResult {
                index: r.index,
                score: r.relevance_score,
            });
        }

        // Jina returns results sorted by score descending, but sort defensively
        // — the contract this fn promises must not depend on server ordering.
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }
}

/// Private deserialisation envelope for Jina's `/rerank` response.
#[derive(serde::Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(serde::Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Build a Jina rerank response from `(index, score)` pairs in any order.
    fn mock_response(pairs: &[(usize, f32)]) -> serde_json::Value {
        let results: Vec<serde_json::Value> = pairs
            .iter()
            .map(|(i, s)| json!({ "index": i, "relevance_score": s }))
            .collect();
        json!({
            "model": "jina-reranker-v2-base-multilingual",
            "usage": { "total_tokens": 10 },
            "results": results,
        })
    }

    #[tokio::test]
    async fn rerank_sorts_descending_and_keeps_indices() {
        let server = MockServer::start().await;
        // Server returns out-of-order, low-score-first to prove we sort.
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(mock_response(&[(0, 0.10), (2, 0.90), (1, 0.50)])),
            )
            .mount(&server)
            .await;

        let svc = JinaRerankService::with_base_url(
            "k".into(),
            "jina-reranker-v2-base-multilingual".into(),
            server.uri(),
        )
        .unwrap();

        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let out = svc.rerank("q", &docs).await.expect("rerank succeeds");
        assert_eq!(out.len(), 3);
        // Sorted descending by score.
        assert_eq!(out[0].index, 2);
        assert_eq!(out[1].index, 1);
        assert_eq!(out[2].index, 0);
        assert!((out[0].score - 0.90).abs() < 1e-6);
    }

    #[tokio::test]
    async fn empty_documents_no_network_call() {
        // No mock mounted: any HTTP call would 404 and error. Empty input must
        // short-circuit to Ok(empty) without touching the network.
        let server = MockServer::start().await;
        let svc = JinaRerankService::with_base_url(
            "k".into(),
            "jina-reranker-v2-base-multilingual".into(),
            server.uri(),
        )
        .unwrap();
        let out = svc.rerank("q", &[]).await.expect("empty is Ok");
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn out_of_range_index_dropped_not_panicked() {
        let server = MockServer::start().await;
        // index 5 is past the 2-doc input — must be dropped, not indexed into.
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mock_response(&[(0, 0.8), (5, 0.9)])),
            )
            .mount(&server)
            .await;

        let svc = JinaRerankService::with_base_url(
            "k".into(),
            "jina-reranker-v2-base-multilingual".into(),
            server.uri(),
        )
        .unwrap();
        let docs = vec!["a".to_string(), "b".to_string()];
        let out = svc.rerank("q", &docs).await.expect("rerank succeeds");
        assert_eq!(out.len(), 1, "out-of-range index must be dropped");
        assert_eq!(out[0].index, 0);
    }

    #[tokio::test]
    async fn rerank_429_is_loud_degrade() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let svc = JinaRerankService::with_base_url(
            "k".into(),
            "jina-reranker-v2-base-multilingual".into(),
            server.uri(),
        )
        .unwrap();
        let err = svc
            .rerank("q", &["a".to_string()])
            .await
            .expect_err("429 must error");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
                let r = d.degradation_reason.unwrap().to_lowercase();
                assert!(r.contains("rate-limited") || r.contains("429"), "got {r}");
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rerank_sends_query_and_model() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .and(body_partial_json(json!({
                "model": "jina-reranker-v2-base-multilingual",
                "query": "consensus mechanisms",
                "return_documents": false,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_response(&[(0, 0.5)])))
            .expect(1)
            .mount(&server)
            .await;

        let svc = JinaRerankService::with_base_url(
            "k".into(),
            "jina-reranker-v2-base-multilingual".into(),
            server.uri(),
        )
        .unwrap();
        let _ = svc
            .rerank("consensus mechanisms", &["doc".to_string()])
            .await
            .expect("rerank succeeds");
    }

    #[test]
    fn empty_api_key_errors() {
        let res = JinaRerankService::new("".into(), "jina-reranker-v2-base-multilingual".into());
        assert!(res.is_err(), "empty key must error");
    }
}
