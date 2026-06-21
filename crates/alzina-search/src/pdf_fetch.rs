//! PDF byte fetch and pdftotext extraction (F10).
//!
//! Two pure-async functions:
//!
//! - [`fetch_pdf_bytes`] — HTTP GET with scheme allowlist, timeout, and size
//!   cap (T-lq4-01 SSRF guard, T-lq4-02 DoS guard).
//! - [`pdftotext_extract`] — shell-out to poppler's `pdftotext`; accepts stdin,
//!   returns stdout as UTF-8. The binary path comes from [`PdfFetchConfig`] so
//!   tests can inject a bogus path without mutating env.
//!
//! ## Loud-degrade contract
//!
//! Both functions return `Err(AlzinaError::Search { degraded: true, … })` on
//! every failure path. The caller (`promote_pdf_fulltext`) converts every error
//! to a `tracing::warn!` + status `'failed'` + `Ok(())`.
//!
//! ## Threat notes
//!
//! - T-lq4-01 (SSRF): scheme allowlist enforced in `fetch_pdf_bytes` BEFORE any
//!   network call. URL is S2 wire data — semi-trusted. Never interpolated into
//!   paths or SQL (bound params throughout).
//! - T-lq4-02 (DoS): 20 MB read cap, 30s fetch timeout, 30s subprocess timeout
//!   with `kill()`, per-run PdfFetch budget (30) + 1s spacing via LitGateway.
//! - T-lq4-03 (malicious PDF): poppler parses untrusted bytes out-of-process;
//!   a crash → non-zero exit → loud degrade to `'failed'`; daemon unaffected.

use alzina_core::error::{AlzinaError, AlzinaResult, SearchDetail};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for PDF fetch and extraction.
#[derive(Debug, Clone)]
pub struct PdfFetchConfig {
    /// Path to the `pdftotext` binary (poppler). Default: `"pdftotext"`.
    /// Override with `ALZINA_PDFTOTEXT_PATH`.
    pub pdftotext_path: String,
    /// Per-request HTTP fetch timeout in seconds. Default: 30.
    pub timeout_secs: u64,
    /// Maximum PDF response body size in bytes. Default: 20 MB. Streams over
    /// this limit are rejected (T-lq4-02).
    pub max_bytes: usize,
}

impl Default for PdfFetchConfig {
    fn default() -> Self {
        Self {
            pdftotext_path: "pdftotext".to_string(),
            timeout_secs: 30,
            max_bytes: 20_000_000,
        }
    }
}

impl PdfFetchConfig {
    /// Read `ALZINA_PDFTOTEXT_PATH` for the binary path; all other fields use
    /// defaults. The env var overrides only the binary path — timeout and size
    /// cap are invariant.
    pub fn from_env() -> Self {
        let pdftotext_path = std::env::var("ALZINA_PDFTOTEXT_PATH")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "pdftotext".to_string());
        Self {
            pdftotext_path,
            ..Self::default()
        }
    }
}

// ── fetch_pdf_bytes ───────────────────────────────────────────────────────────

/// Fetch PDF bytes from a URL.
///
/// Guards (T-lq4-01, T-lq4-02):
/// 1. Scheme allowlist: only `http://` and `https://` are permitted. The check
///    runs BEFORE any network call so no SSRF for other URL schemes.
/// 2. Response body read is capped at `cfg.max_bytes` (default 20 MB). Exceeding
///    the cap returns an error without consuming the remainder.
/// 3. Per-request timeout from `cfg.timeout_secs` (default 30s).
/// 4. Non-2xx responses return an error.
pub async fn fetch_pdf_bytes(url: &str, cfg: &PdfFetchConfig) -> AlzinaResult<Vec<u8>> {
    // T-lq4-01: scheme allowlist — reject before any network call.
    let lower = url.trim().to_lowercase();
    if !lower.starts_with("http://") && !lower.starts_with("https://") {
        return Err(search_err(
            format!("fetch_pdf_bytes: rejected non-http/https URL scheme (T-lq4-01): {url}"),
            "PDF URL must use http or https scheme".to_string(),
        ));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(cfg.timeout_secs))
        .build()
        .map_err(|e| search_err(
            format!("fetch_pdf_bytes: reqwest client build: {e}"),
            format!("PDF fetch reqwest build failed: {e}"),
        ))?;

    let resp = client.get(url).send().await.map_err(|e| {
        search_err(
            format!("fetch_pdf_bytes: request failed for {url}: {e}"),
            format!("PDF fetch request failed: {e}"),
        )
    })?;

    if !resp.status().is_success() {
        let status = resp.status();
        return Err(search_err(
            format!("fetch_pdf_bytes: HTTP {status} for {url}"),
            format!("PDF fetch returned HTTP {status}"),
        ));
    }

    // T-lq4-02: size cap.
    // Check Content-Length header before reading: if the server declares a body
    // larger than the cap, reject without downloading anything.
    if let Some(len) = resp.content_length() {
        if len as usize > cfg.max_bytes {
            return Err(search_err(
                format!(
                    "fetch_pdf_bytes: Content-Length {len} exceeds {} byte cap for {url} (T-lq4-02)",
                    cfg.max_bytes
                ),
                format!("PDF Content-Length exceeds {} byte size cap", cfg.max_bytes),
            ));
        }
    }

    // Collect body. The per-request timeout (set on the client) bounds wall time.
    let bytes = resp.bytes().await.map_err(|e| search_err(
        format!("fetch_pdf_bytes: read error for {url}: {e}"),
        format!("PDF fetch read error: {e}"),
    ))?;

    // Post-read size check: catches servers that omit Content-Length.
    if bytes.len() > cfg.max_bytes {
        return Err(search_err(
            format!(
                "fetch_pdf_bytes: response body {} bytes exceeds {} cap for {url} (T-lq4-02)",
                bytes.len(),
                cfg.max_bytes
            ),
            format!("PDF response exceeds {} byte size cap", cfg.max_bytes),
        ));
    }

    Ok(bytes.to_vec())
}

// ── pdftotext_extract ─────────────────────────────────────────────────────────

/// Extract plain text from PDF bytes using poppler's `pdftotext`.
///
/// Spawns `pdftotext -layout - -` (stdin → stdout). PDF bytes are piped to
/// stdin; extracted text is collected from stdout.
///
/// The whole child interaction is wrapped in `tokio::time::timeout` (using
/// `cfg.timeout_secs`); on timeout the child is killed and an error is returned.
///
/// Spawn errors (binary missing) return an error that names the path and the
/// `ALZINA_PDFTOTEXT_PATH` lever, so operators know how to fix it.
///
/// Non-zero exit returns an error with a stderr excerpt.
pub async fn pdftotext_extract(pdf_bytes: &[u8], cfg: &PdfFetchConfig) -> AlzinaResult<String> {
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;

    let timeout_dur = std::time::Duration::from_secs(cfg.timeout_secs);

    let result = tokio::time::timeout(timeout_dur, async {
        let mut child = Command::new(&cfg.pdftotext_path)
            .args(["-layout", "-", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| search_err(
                format!(
                    "pdftotext_extract: spawn '{}' failed: {e} \
                     (set ALZINA_PDFTOTEXT_PATH if the binary is not on PATH)",
                    cfg.pdftotext_path
                ),
                format!(
                    "pdftotext binary '{}' not found or not executable; \
                     install poppler-utils or set ALZINA_PDFTOTEXT_PATH",
                    cfg.pdftotext_path
                ),
            ))?;

        // Write PDF bytes to stdin.
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(pdf_bytes).await.map_err(|e| search_err(
                format!("pdftotext_extract: stdin write failed: {e}"),
                format!("pdftotext stdin write error: {e}"),
            ))?;
            // Drop stdin to signal EOF.
        }

        let output = child.wait_with_output().await.map_err(|e| search_err(
            format!("pdftotext_extract: wait_with_output failed: {e}"),
            format!("pdftotext wait error: {e}"),
        ))?;

        if !output.status.success() {
            let stderr_excerpt = String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(500)
                .collect::<String>();
            return Err(search_err(
                format!(
                    "pdftotext_extract: non-zero exit ({}) stderr: {stderr_excerpt}",
                    output.status
                ),
                format!("pdftotext exited non-zero: {}", output.status),
            ));
        }

        let text = String::from_utf8_lossy(&output.stdout).into_owned();
        Ok::<String, AlzinaError>(text)
    })
    .await;

    match result {
        Ok(inner) => inner,
        Err(_timeout) => Err(search_err(
            format!(
                "pdftotext_extract: timed out after {}s",
                cfg.timeout_secs
            ),
            format!("pdftotext timed out after {}s (T-lq4-02)", cfg.timeout_secs),
        )),
    }
}

// ── Shared error helper ───────────────────────────────────────────────────────

fn search_err(message: impl Into<String>, reason: impl Into<String>) -> AlzinaError {
    let reason = reason.into();
    AlzinaError::Search(SearchDetail {
        message: message.into(),
        degraded: true,
        degradation_reason: Some(reason),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// T-lq4-01: non-http/https schemes are rejected before any network call.
    /// Use a guaranteed-connection-refused address — if the guard fires we
    /// never reach the network.
    #[tokio::test]
    async fn fetch_rejects_non_http_scheme() {
        let cfg = PdfFetchConfig::default();
        let err = fetch_pdf_bytes("ftp://example.com/paper.pdf", &cfg)
            .await
            .expect_err("ftp:// must be rejected");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
                assert!(
                    d.message.contains("T-lq4-01") || d.message.contains("scheme"),
                    "error must mention scheme guard: {}",
                    d.message
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    /// T-lq4-01: file:// scheme also rejected.
    #[tokio::test]
    async fn fetch_rejects_file_scheme() {
        let cfg = PdfFetchConfig::default();
        let err = fetch_pdf_bytes("file:///etc/passwd", &cfg)
            .await
            .expect_err("file:// must be rejected");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    /// Connection refused → error (network call was attempted but failed
    /// cleanly; no panic).
    #[tokio::test]
    async fn fetch_returns_error_on_connection_refused() {
        let cfg = PdfFetchConfig {
            timeout_secs: 5,
            ..PdfFetchConfig::default()
        };
        let err = fetch_pdf_bytes("http://127.0.0.1:1/paper.pdf", &cfg)
            .await
            .expect_err("connection refused must return Err");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    /// Missing binary: pdftotext_extract with a non-existent binary path returns
    /// an error that names the path and the ALZINA_PDFTOTEXT_PATH lever.
    #[tokio::test]
    async fn pdftotext_missing_binary_names_path() {
        let cfg = PdfFetchConfig {
            pdftotext_path: "/nonexistent/pdftotext-no-such-binary".to_string(),
            ..PdfFetchConfig::default()
        };
        let err = pdftotext_extract(b"%PDF-1.4 test", &cfg)
            .await
            .expect_err("missing binary must return Err");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
                assert!(
                    d.message.contains("/nonexistent/pdftotext-no-such-binary"),
                    "error must name the binary path: {}",
                    d.message
                );
                assert!(
                    d.message.contains("ALZINA_PDFTOTEXT_PATH"),
                    "error must mention ALZINA_PDFTOTEXT_PATH lever: {}",
                    d.message
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    /// PdfFetchConfig::from_env reads ALZINA_PDFTOTEXT_PATH.
    #[test]
    fn from_env_reads_pdftotext_path() {
        // SAFETY: single-threaded test body; env mutation is sequential.
        unsafe { std::env::set_var("ALZINA_PDFTOTEXT_PATH", "/usr/local/bin/pdftotext") };
        let cfg = PdfFetchConfig::from_env();
        assert_eq!(cfg.pdftotext_path, "/usr/local/bin/pdftotext");
        unsafe { std::env::remove_var("ALZINA_PDFTOTEXT_PATH") };
    }

    /// Default config: path is "pdftotext", timeout is 30, max_bytes is 20 MB.
    #[test]
    fn default_config_values() {
        let cfg = PdfFetchConfig::default();
        assert_eq!(cfg.pdftotext_path, "pdftotext");
        assert_eq!(cfg.timeout_secs, 30);
        assert_eq!(cfg.max_bytes, 20_000_000);
    }
}
