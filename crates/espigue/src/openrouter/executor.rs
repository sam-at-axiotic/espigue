//! OpenRouter chat-completions executor.
//!
//! Implements the [`AgentExecutor`] seam by calling OpenRouter's
//! OpenAI-compatible `POST /chat/completions`. Replaces the TypeScript sidecar +
//! Claude Agent SDK path entirely — the `instruction` is the fully-rendered TTD
//! prompt, so this is a thin chat shim. `agent_id` / `task` are governance-only
//! and ignored here.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use base::error::AlzinaError;
use base::identity::AgentId;
use base::AlzinaResult;
use orchestration::{AgentExecutor, SamplingParams};

use super::DEFAULT_BASE_URL;
use super::retry::{PostFailure, RetryPolicy, post_json_with_retry};

/// HTTP timeout. Synthesis stages can be slow on large prompts, so this is
/// generous.
const HTTP_TIMEOUT_SECS: u64 = 300;
/// Connect-phase timeout. Without it, a SYN blackhole burns the full total
/// timeout and then reads as a post-send failure — non-retryable for chat —
/// even though the request never reached the server. A connect timeout
/// fails fast as a connect error, which always retries.
const CONNECT_TIMEOUT_SECS: u64 = 15;

/// OpenRouter chat-completions executor.
pub struct OpenRouterExecutor {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenRouterExecutor {
    /// Construct an executor pointed at the production OpenRouter endpoint.
    pub fn new(api_key: impl Into<String>) -> AlzinaResult<Self> {
        Self::build(api_key.into(), DEFAULT_BASE_URL.to_string())
    }

    /// Test-only constructor that accepts a custom base URL (e.g. a wiremock
    /// server).
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> AlzinaResult<Self> {
        Self::build(api_key.into(), base_url.into())
    }

    fn build(api_key: String, base_url: String) -> AlzinaResult<Self> {
        if api_key.is_empty() {
            return Err(AlzinaError::Orchestration(
                "OpenRouterExecutor: api_key is empty (set OPENROUTER_API_KEY)".into(),
            ));
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .build()
            .map_err(|e| AlzinaError::Orchestration(format!("reqwest client build: {e}")))?;
        Ok(Self { client, api_key, base_url })
    }

    async fn chat(
        &self,
        model: &str,
        instruction: &str,
        sampling: Option<SamplingParams>,
    ) -> AlzinaResult<String> {
        self.chat_with_system(model, None, instruction, sampling).await
    }

    async fn chat_with_system(
        &self,
        model: &str,
        system: Option<&str>,
        instruction: &str,
        sampling: Option<SamplingParams>,
    ) -> AlzinaResult<String> {
        let messages = match system.filter(|s| !s.trim().is_empty()) {
            Some(sys) => json!([
                { "role": "system", "content": sys },
                { "role": "user", "content": instruction },
            ]),
            None => json!([{ "role": "user", "content": instruction }]),
        };
        let mut body = json!({
            "model": model,
            "messages": messages,
        });
        if let Some(s) = sampling {
            body["temperature"] = json!(s.temperature);
            body["top_p"] = json!(s.top_p);
            if s.top_k > 0 {
                body["top_k"] = json!(s.top_k);
            }
        }

        let url = format!("{}/chat/completions", self.base_url);
        let parsed: ChatCompletion = post_json_with_retry(
            &self.client,
            &url,
            &self.api_key,
            &body,
            "chat",
            &RetryPolicy::chat(),
        )
        .await
        .map_err(|f| match f {
            PostFailure::Transport(e) => {
                AlzinaError::Orchestration(format!("OpenRouter chat request failed: {e}"))
            }
            PostFailure::Http { status, body } => {
                AlzinaError::Orchestration(format!("OpenRouter chat HTTP {status}: {body}"))
            }
            PostFailure::Decode { source, body } => AlzinaError::Orchestration(format!(
                "OpenRouter chat response decode failed after retries: {source}; body: {body}"
            )),
        })?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| AlzinaError::Orchestration("OpenRouter chat returned no choices".into()))
    }
}

#[async_trait]
impl AgentExecutor for OpenRouterExecutor {
    async fn execute(
        &self,
        _agent_id: &AgentId,
        instruction: &str,
        model: &str,
        _task: &str,
    ) -> AlzinaResult<String> {
        self.chat(model, instruction, None).await
    }

    async fn execute_with_sampling(
        &self,
        _agent_id: &AgentId,
        instruction: &str,
        model: &str,
        _task: &str,
        sampling: Option<SamplingParams>,
    ) -> AlzinaResult<String> {
        self.chat(model, instruction, sampling).await
    }

    async fn execute_with_system(
        &self,
        _agent_id: &AgentId,
        system: &str,
        instruction: &str,
        model: &str,
        _task: &str,
    ) -> AlzinaResult<String> {
        self.chat_with_system(model, Some(system), instruction, None).await
    }
}

#[derive(Deserialize)]
struct ChatCompletion {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn agent() -> AgentId {
        AgentId::new("test-agent")
    }

    #[tokio::test]
    async fn execute_returns_assistant_content() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer k"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "role": "assistant", "content": "hello world" } }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let out = exec
            .execute(&agent(), "say hi", "google/gemini-2.5-flash", "graph_draft")
            .await
            .expect("execute succeeds");
        assert_eq!(out, "hello world");
    }

    #[tokio::test]
    async fn execute_sends_model_and_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "model": "anthropic/claude-opus-4",
                "messages": [{ "role": "user", "content": "the prompt" }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "content": "ok" } }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let _ = exec
            .execute(&agent(), "the prompt", "anthropic/claude-opus-4", "synthesis_merger")
            .await
            .expect("execute succeeds");
    }

    #[tokio::test]
    async fn execute_with_system_sends_system_role() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "messages": [
                    { "role": "system", "content": "the instructions" },
                    { "role": "user", "content": "the data" }
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "content": "ok" } }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let _ = exec
            .execute_with_system(&agent(), "the instructions", "the data", "m", "synthesis_merger")
            .await
            .expect("execute_with_system succeeds");
    }

    #[tokio::test]
    async fn sampling_threads_temperature_and_top_p() {
        let server = MockServer::start().await;
        // Values exact in binary floating point so the JSON matcher is not
        // tripped by f32→f64 representation drift.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({ "temperature": 0.5, "top_p": 0.5, "top_k": 40 })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "content": "ok" } }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let _ = exec
            .execute_with_sampling(
                &agent(),
                "p",
                "m",
                "graph_draft",
                Some(SamplingParams { temperature: 0.5, top_p: 0.5, top_k: 40 }),
            )
            .await
            .expect("execute succeeds");
    }

    #[tokio::test]
    async fn persistent_429_errors_after_bounded_retries() {
        let server = MockServer::start().await;
        // Retry-After 0 keeps the test fast while exercising the header path.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "0")
                    .set_body_string("rate limited"),
            )
            .expect(3) // 1 initial + 2 retries, then give up
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let err = exec
            .execute(&agent(), "p", "m", "graph_draft")
            .await
            .expect_err("persistent 429 must still error");
        match err {
            AlzinaError::Orchestration(m) => assert!(m.contains("429"), "got {m}"),
            other => panic!("expected Orchestration error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn transient_429_recovers_on_retry() {
        let server = MockServer::start().await;
        // First request: 429. Mounted first and limited to one use, so the
        // retry falls through to the 200 mock below.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "content": "recovered" } }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let out = exec
            .execute(&agent(), "p", "m", "graph_draft")
            .await
            .expect("one 429 then 200 must succeed");
        assert_eq!(out, "recovered");
    }

    #[tokio::test]
    async fn transient_503_recovers_on_retry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).insert_header("retry-after", "0"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "content": "recovered" } }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let out = exec
            .execute(&agent(), "p", "m", "graph_draft")
            .await
            .expect("one 503 then 200 must succeed");
        assert_eq!(out, "recovered");
    }

    #[tokio::test]
    async fn bad_request_is_not_retried() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("not a valid model ID"))
            .expect(1) // fatal on first sight — retrying a 400 cannot succeed
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let err = exec
            .execute(&agent(), "p", "m", "graph_draft")
            .await
            .expect_err("400 must error immediately");
        match err {
            AlzinaError::Orchestration(m) => {
                assert!(m.contains("400"), "got {m}");
                assert!(m.contains("not a valid model ID"), "got {m}");
            }
            other => panic!("expected Orchestration error, got {other:?}"),
        }
    }

    #[test]
    fn empty_key_errors() {
        assert!(OpenRouterExecutor::new("").is_err());
    }
}
