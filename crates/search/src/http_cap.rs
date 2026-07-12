//! Capped HTTP response body reads (extends the T-lq4-02 DoS guard crate-wide).
//!
//! `pdf_fetch::fetch_pdf_bytes` caps PDF downloads at 20 MB. Every other
//! network reader in this crate used `resp.text()` / `resp.json()`, which
//! buffer the whole body with no size bound — only the client timeout limited
//! how much a hostile or broken server (ar5iv full-text HTML especially) could
//! make us hold in memory during a multi-hour run.
//!
//! [`read_body_capped`] closes that class:
//!
//! 1. If the server declares a `Content-Length` above the cap, reject before
//!    reading anything (mirrors `fetch_pdf_bytes`).
//! 2. Otherwise stream chunks and abort as soon as the running total would
//!    pass the cap — peak memory is bounded by the cap regardless of what
//!    the server sends (chunked responses included, which the post-read
//!    check style cannot bound).
//!
//! ## Loud-degrade contract
//!
//! The helpers return a plain [`BodyCapError`]; each call site wraps it in its
//! existing error type (`AlzinaError::Search { degraded: true, … }` or
//! `S2CallError`) under the same message patterns as a transport read failure,
//! so retryability classes are unchanged. Over-cap errors name the endpoint
//! (`what`) and the cap.

/// Cap for full-document bodies: ar5iv HTML renders and Jina embedding
/// batches (128 inputs × up to 2048-dim float JSON runs to several MB).
/// 20 MB — matches the PDF download cap in `pdf_fetch` (T-lq4-02).
pub(crate) const LARGE_BODY_MAX_BYTES: usize = 20_000_000;

/// Cap for structured API bodies: arXiv Atom XML, S2 graph/search JSON, and
/// Jina rerank JSON. Legitimate responses are well under 1 MB; 8 MB leaves an
/// order-of-magnitude margin while still bounding a hostile stream.
pub(crate) const API_BODY_MAX_BYTES: usize = 8_000_000;

/// Error from a capped body read.
#[derive(Debug)]
pub(crate) enum BodyCapError {
    /// The body (declared via `Content-Length`, or streamed) exceeded `cap`.
    TooLarge { what: &'static str, cap: usize },
    /// Transport error from the underlying stream. `Display` delegates to the
    /// reqwest error so existing `{e}` interpolations keep their wording.
    Read(reqwest::Error),
}

impl std::fmt::Display for BodyCapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { what, cap } => {
                write!(f, "{what} response body exceeds {cap} byte cap (T-lq4-02)")
            }
            Self::Read(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BodyCapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TooLarge { .. } => None,
            Self::Read(e) => Some(e),
        }
    }
}

/// Read a response body, aborting once it exceeds `cap` bytes.
///
/// Behaviour for bodies at or under the cap is byte-identical to
/// `resp.bytes()`: the chunks are concatenated unmodified.
pub(crate) async fn read_body_capped(
    mut resp: reqwest::Response,
    cap: usize,
    what: &'static str,
) -> Result<Vec<u8>, BodyCapError> {
    // Pre-check: reject a declared over-cap length without reading anything.
    if resp.content_length().is_some_and(|len| len as usize > cap) {
        return Err(BodyCapError::TooLarge { what, cap });
    }
    let mut buf: Vec<u8> =
        Vec::with_capacity(resp.content_length().map_or(0, |l| (l as usize).min(cap)));
    while let Some(chunk) = resp.chunk().await.map_err(BodyCapError::Read)? {
        if buf.len() + chunk.len() > cap {
            return Err(BodyCapError::TooLarge { what, cap });
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Capped equivalent of `resp.text()`.
///
/// This crate builds reqwest without the `charset` feature, so `text()` is
/// exactly `String::from_utf8_lossy` over the raw bytes — replicated here so
/// bodies at or under the cap decode identically.
pub(crate) async fn read_text_capped(
    resp: reqwest::Response,
    cap: usize,
    what: &'static str,
) -> Result<String, BodyCapError> {
    let bytes = read_body_capped(resp, cap, what).await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn serve(body: Vec<u8>) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/body"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
            .mount(&server)
            .await;
        server
    }

    async fn get(server: &MockServer) -> reqwest::Response {
        reqwest::Client::new()
            .get(format!("{}/body", server.uri()))
            .send()
            .await
            .expect("send")
    }

    #[tokio::test]
    async fn under_cap_passes_through_byte_identical() {
        let body: Vec<u8> = (0u8..=255).cycle().take(10_000).collect();
        let server = serve(body.clone()).await;
        let got = read_body_capped(get(&server).await, 10_001, "test").await.expect("under cap");
        assert_eq!(got, body);
    }

    #[tokio::test]
    async fn exact_cap_boundary_passes() {
        let body = vec![7u8; 4096];
        let server = serve(body.clone()).await;
        let got = read_body_capped(get(&server).await, 4096, "test")
            .await
            .expect("a body exactly at the cap is allowed");
        assert_eq!(got, body);
    }

    #[tokio::test]
    async fn over_cap_rejected_naming_what_and_cap() {
        let server = serve(vec![7u8; 4097]).await;
        let err = read_body_capped(get(&server).await, 4096, "test-endpoint")
            .await
            .expect_err("over cap must error");
        let msg = err.to_string();
        assert!(msg.contains("test-endpoint"), "names what: {msg}");
        assert!(msg.contains("4096"), "names cap: {msg}");
    }

    #[tokio::test]
    async fn over_cap_stream_without_content_length_rejected_mid_stream() {
        // A chunked body with no Content-Length: the pre-check cannot fire,
        // so the chunk loop must abort part-way instead of buffering it all.
        let chunks: Vec<Result<Vec<u8>, std::io::Error>> =
            (0..10).map(|_| Ok(vec![0u8; 1024])).collect();
        let body = reqwest::Body::wrap_stream(futures::stream::iter(chunks));
        let resp = reqwest::Response::from(
            http::Response::builder().status(200).body(body).expect("response"),
        );
        assert!(resp.content_length().is_none(), "test premise: no declared length");
        let err = read_body_capped(resp, 4096, "streamed")
            .await
            .expect_err("over cap must error");
        assert!(matches!(err, BodyCapError::TooLarge { cap: 4096, .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn text_decodes_lossy_like_reqwest_without_charset_feature() {
        let server = serve(vec![0x68, 0x69, 0xFF]).await; // "hi" + invalid UTF-8
        let got = read_text_capped(get(&server).await, 100, "test").await.expect("under cap");
        assert_eq!(got, "hi\u{FFFD}");
    }
}
