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

/// HTTP timeout. Synthesis stages can be slow on large prompts, so this is
/// generous.
const HTTP_TIMEOUT_SECS: u64 = 300;

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
        let mut body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": instruction }],
        });
        if let Some(s) = sampling {
            body["temperature"] = json!(s.temperature);
            body["top_p"] = json!(s.top_p);
            if s.top_k > 0 {
                body["top_k"] = json!(s.top_k);
            }
        }

        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            // OpenRouter attribution headers (optional but recommended).
            .header("HTTP-Referer", "https://github.com/axiotic/espigue")
            .header("X-Title", "espigue")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "OpenRouter chat request failed (network)");
                AlzinaError::Orchestration(format!("OpenRouter chat request failed: {e}"))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            tracing::warn!(status = %status, body = %text, "OpenRouter chat non-2xx response");
            return Err(AlzinaError::Orchestration(format!(
                "OpenRouter chat HTTP {status}: {text}"
            )));
        }

        let parsed: ChatCompletion = resp.json().await.map_err(|e| {
            AlzinaError::Orchestration(format!("OpenRouter chat response decode failed: {e}"))
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
    async fn non_2xx_is_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let exec = OpenRouterExecutor::with_base_url("k", server.uri()).unwrap();
        let err = exec
            .execute(&agent(), "p", "m", "graph_draft")
            .await
            .expect_err("429 must error");
        match err {
            AlzinaError::Orchestration(m) => assert!(m.contains("429"), "got {m}"),
            other => panic!("expected Orchestration error, got {other:?}"),
        }
    }

    #[test]
    fn empty_key_errors() {
        assert!(OpenRouterExecutor::new("").is_err());
    }
}
