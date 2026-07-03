//! Bounded retry for OpenRouter POSTs.
//!
//! Generation is the most expensive surface in a run: by the time Stage 3
//! fires, retrieval and every earlier LLM call have already been paid for, so
//! a single transient 429/5xx must not discard the run. Mirrors the
//! `search::lit_gateway` retry semantics: bounded attempts, server
//! `Retry-After` honoured, exponential backoff otherwise. Non-429 4xx (auth,
//! bad request) surface immediately — retrying those cannot succeed.

use std::time::Duration;

/// Total attempts per call (1 initial + 2 retries).
const MAX_ATTEMPTS: u32 = 3;
/// Base for exponential backoff when the server sends no Retry-After.
const BACKOFF_BASE_SECS: u64 = 2;
/// Cap on any single wait, including a server-provided Retry-After.
const MAX_WAIT: Duration = Duration::from_secs(60);

/// Terminal failure after retries are exhausted or a fatal status.
pub(crate) enum PostFailure {
    /// Transport error (the final attempt never got a response).
    Transport(reqwest::Error),
    /// Final HTTP error status, with response body text.
    Http {
        status: reqwest::StatusCode,
        body: String,
    },
}

/// POST `body` to `url` with OpenRouter auth + attribution headers, retrying
/// transient failures. Returns the first 2xx response.
///
/// Transient = HTTP 429, any 5xx, or a transport error. Anything else is
/// fatal on first sight. `what` labels log lines ("chat", "embeddings").
pub(crate) async fn post_json_with_retry(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
    what: &str,
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
                if !transient || attempt >= MAX_ATTEMPTS {
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
                if attempt >= MAX_ATTEMPTS {
                    tracing::warn!(error = %e, attempt, what, "OpenRouter request failed (network, terminal)");
                    return Err(PostFailure::Transport(e));
                }
                tracing::warn!(error = %e, attempt, what, "OpenRouter request failed (network); retrying");
            }
        }

        let wait = wait_hint
            .unwrap_or_else(|| Duration::from_secs(BACKOFF_BASE_SECS.pow(attempt)))
            .min(MAX_WAIT);
        tokio::time::sleep(wait).await;
        attempt += 1;
    }
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
