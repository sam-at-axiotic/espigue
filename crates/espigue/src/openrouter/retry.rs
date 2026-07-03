//! Bounded retry for OpenRouter POSTs.
//!
//! Generation is the most expensive surface in a run: by the time Stage 3
//! fires, retrieval and every earlier LLM call have already been paid for, so
//! a single transient 429/5xx must not discard the run. Bounded attempts,
//! server `Retry-After` honoured, exponential backoff otherwise. Non-429 4xx
//! (auth, bad request) surface immediately — retrying those cannot succeed.
//!
//! Billing note: a non-streaming generation that times out client-side may
//! still complete and bill server-side. [`RetryPolicy::retry_post_send`]
//! decides whether such post-send transport failures retry; connect-phase
//! failures always retry (the request never reached the server).

use std::time::Duration;

/// Per-caller retry policy for [`post_json_with_retry`].
pub(crate) struct RetryPolicy {
    /// Total attempts per call (1 initial + retries).
    pub max_attempts: u32,
    /// Base for exponential backoff when the server sends no Retry-After.
    pub backoff_base: Duration,
    /// Cap on any single wait, including a server-provided Retry-After.
    pub max_wait: Duration,
    /// Retry transport errors that may have fired AFTER the request reached
    /// the server (e.g. a client timeout mid-response). Chat sets this false:
    /// a 300s-timeout generation likely completed and billed server-side, so
    /// retrying can pay for the same slow call up to three times. Embeddings
    /// calls are cheap and fast, so they keep it true.
    pub retry_post_send: bool,
}

impl RetryPolicy {
    /// Policy for `POST /chat/completions` — post-send failures do not retry.
    pub(crate) const fn chat() -> Self {
        Self {
            max_attempts: 3,
            backoff_base: Duration::from_secs(2),
            max_wait: Duration::from_secs(60),
            retry_post_send: false,
        }
    }

    /// Policy for `POST /embeddings` — cheap calls, post-send retries allowed.
    pub(crate) const fn embeddings() -> Self {
        Self {
            retry_post_send: true,
            ..Self::chat()
        }
    }
}

/// Terminal failure after retries are exhausted or a fatal status.
#[derive(Debug)]
pub(crate) enum PostFailure {
    /// Transport error (connect failure, or a post-send failure the policy
    /// does not retry).
    Transport(reqwest::Error),
    /// Final HTTP error status, with response body text.
    Http {
        status: reqwest::StatusCode,
        body: String,
    },
}

/// POST `body` to `url` with OpenRouter auth + attribution headers, retrying
/// transient failures per `policy`. Returns the first 2xx response.
///
/// Transient = HTTP 429, any 5xx, a connect-phase transport error, or (when
/// `policy.retry_post_send`) any transport error. Anything else is fatal on
/// first sight. `what` labels log lines ("chat", "embeddings").
pub(crate) async fn post_json_with_retry(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
    what: &str,
    policy: &RetryPolicy,
) -> Result<reqwest::Response, PostFailure> {
    let mut attempt: u32 = 1;
    loop {
        let sent = client
            .post(url)
            .bearer_auth(api_key)
            // OpenRouter attribution headers (optional but recommended).
            .header("HTTP-Referer", "https://github.com/axiotic/espigue")
            .header("X-Title", "espigue")
            .json(body)
            .send()
            .await;

        let mut wait_hint: Option<Duration> = None;
        match sent {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) => {
                let status = resp.status();
                let transient = status.as_u16() == 429 || status.is_server_error();
                wait_hint = parse_retry_after(&resp);
                if !transient || attempt >= policy.max_attempts {
                    let body_text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
                    tracing::warn!(
                        status = %status,
                        attempt,
                        what,
                        body = %body_text,
                        "OpenRouter HTTP failure (terminal)"
                    );
                    return Err(PostFailure::Http {
                        status,
                        body: body_text,
                    });
                }
                tracing::warn!(status = %status, attempt, what, "OpenRouter transient HTTP failure; retrying");
            }
            Err(e) => {
                // Connect-phase failures never reached the server and are
                // always safe to retry; anything later is policy-gated
                // because the server may have processed (and billed) the
                // request.
                let retryable = e.is_connect() || policy.retry_post_send;
                if !retryable || attempt >= policy.max_attempts {
                    tracing::warn!(error = %e, attempt, what, "OpenRouter request failed (transport, terminal)");
                    return Err(PostFailure::Transport(e));
                }
                tracing::warn!(error = %e, attempt, what, "OpenRouter request failed (transport); retrying");
            }
        }

        // Jitter breaks the lockstep herd: without it, N concurrent callers
        // that hit the same 429 with no Retry-After all sleep exactly
        // backoff_base and re-fire simultaneously. A server-provided
        // Retry-After is used as sent — that pacing is the server's call.
        let wait = wait_hint
            .unwrap_or_else(|| policy.backoff_base * 2u32.saturating_pow(attempt - 1) + jitter())
            .min(policy.max_wait);
        tokio::time::sleep(wait).await;
        attempt += 1;
    }
}

/// 0–1s of clock-derived jitter (subsecond nanos), avoiding a rand
/// dependency. Concurrent tasks reach here at different instants, which is
/// all the decorrelation the backoff herd needs.
fn jitter() -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    Duration::from_millis(u64::from(nanos % 1000))
}

/// Parse an integer-seconds `Retry-After` header. HTTP-date values are rare
/// on rate limiters and fall back to exponential backoff.
fn parse_retry_after(resp: &reqwest::Response) -> Option<Duration> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fast_policy(retry_post_send: bool) -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            backoff_base: Duration::from_millis(10),
            max_wait: Duration::from_millis(50),
            retry_post_send,
        }
    }

    fn short_timeout_client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_millis(100))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn post_send_timeout_is_not_retried_when_policy_forbids() {
        let server = MockServer::start().await;
        // The server accepts the request, then stalls past the client
        // timeout — a post-send failure that may already be billed.
        Mock::given(method("POST"))
            .and(path("/x"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(500)))
            .expect(1) // chat policy: exactly one attempt
            .mount(&server)
            .await;

        let err = post_json_with_retry(
            &short_timeout_client(),
            &format!("{}/x", server.uri()),
            "k",
            &json!({}),
            "test",
            &fast_policy(false),
        )
        .await
        .expect_err("timeout must fail");
        assert!(matches!(err, PostFailure::Transport(_)));
    }

    #[tokio::test]
    async fn backoff_branch_recovers_without_retry_after_header() {
        let server = MockServer::start().await;
        // 429 with NO Retry-After header → the jittered exponential-backoff
        // branch (fast here because max_wait caps the jitter at 50ms).
        Mock::given(method("POST"))
            .and(path("/x"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/x"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let resp = post_json_with_retry(
            &reqwest::Client::new(),
            &format!("{}/x", server.uri()),
            "k",
            &json!({}),
            "test",
            &fast_policy(false),
        )
        .await
        .expect("must recover via the backoff branch");
        assert!(resp.status().is_success());
    }

    #[tokio::test]
    async fn connect_refused_retries_even_under_chat_policy() {
        // Bind a port, then drop the listener: connection refused. A
        // connect-phase failure never reached the server, so it retries even
        // with retry_post_send=false. No server exists to count requests;
        // the elapsed lower bound (two backoff waits) is the retry evidence.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/x", listener.local_addr().unwrap());
        drop(listener);

        let started = std::time::Instant::now();
        let err = post_json_with_retry(
            &reqwest::Client::new(),
            &url,
            "k",
            &json!({}),
            "test",
            &fast_policy(false),
        )
        .await
        .expect_err("no server must fail");
        assert!(matches!(err, PostFailure::Transport(_)));
        assert!(
            started.elapsed() >= Duration::from_millis(20),
            "expected at least two backoff waits (retries happened)"
        );
    }

    #[tokio::test]
    async fn post_send_timeout_retries_when_policy_allows() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/x"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(500)))
            .expect(3) // embeddings policy: all attempts spent
            .mount(&server)
            .await;

        let err = post_json_with_retry(
            &short_timeout_client(),
            &format!("{}/x", server.uri()),
            "k",
            &json!({}),
            "test",
            &fast_policy(true),
        )
        .await
        .expect_err("persistent timeout must still fail");
        assert!(matches!(err, PostFailure::Transport(_)));
    }
}
