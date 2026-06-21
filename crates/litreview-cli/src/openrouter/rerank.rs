//! OpenRouter reranking.
//!
//! OpenRouter's `POST /rerank` is the Cohere shape — the same request/response
//! contract as the client in `alzina_search::rerank` (`{model, query,
//! documents}` → `{results: [{index, relevance_score}]}`). We reuse that client
//! via its `with_base_url` constructor, pointed at OpenRouter with a Cohere
//! rerank model. The provider-neutral rename of the upstream `JinaRerankService`
//! is deferred to the public extraction (Phase 5).

use alzina_core::AlzinaResult;
use alzina_search::JinaRerankService;

use super::DEFAULT_BASE_URL;

/// Default OpenRouter rerank model — fast, cheap Cohere cross-encoder.
pub const DEFAULT_RERANK_MODEL: &str = "cohere/rerank-4-fast";

/// Build an OpenRouter-backed reranker with the default model.
pub fn openrouter_reranker(api_key: &str) -> AlzinaResult<JinaRerankService> {
    openrouter_reranker_with_model(api_key, DEFAULT_RERANK_MODEL)
}

/// Build an OpenRouter-backed reranker with an explicit model id (e.g.
/// `cohere/rerank-4-pro`, `nvidia/llama-nemotron-rerank-vl-1b-v2:free`).
pub fn openrouter_reranker_with_model(
    api_key: &str,
    model: &str,
) -> AlzinaResult<JinaRerankService> {
    JinaRerankService::with_base_url(
        api_key.to_string(),
        model.to_string(),
        DEFAULT_BASE_URL.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn openrouter_rerank_parses_cohere_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "cohere/rerank-4-fast",
                "results": [
                    { "index": 0, "relevance_score": 0.1 },
                    { "index": 1, "relevance_score": 0.9 }
                ]
            })))
            .mount(&server)
            .await;

        // Same client we use in production, pointed at the mock.
        let svc = JinaRerankService::with_base_url(
            "k".into(),
            DEFAULT_RERANK_MODEL.into(),
            server.uri(),
        )
        .unwrap();
        let out = svc
            .rerank("q", &["a".into(), "b".into()])
            .await
            .expect("rerank ok");
        // Sorted descending by score → index 1 first.
        assert_eq!(out[0].index, 1);
        assert!((out[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn helper_builds() {
        assert!(openrouter_reranker("k").is_ok());
        assert!(openrouter_reranker("").is_err());
    }
}
