//! arxiv Atom search + ar5iv full-text fetch client.
//!
//! Phase 21 Plan 02. Modelled exactly on `s2_enrichment.rs`.
//!
//! ## Endpoints
//!
//! - Atom search: `GET {base_url}/query?search_query=<q>&max_results=10`
//!   Returns Atom XML; each `<entry>` has `<id>`, `<title>`, `<summary>`
//!   (abstract), `<author>`, `<published>`.
//! - ar5iv full-text: `GET {ar5iv_base_url}/abs/{arxiv_id}`
//!   Returns HTML with `section.ltx_section` elements. Use the labs URL
//!   directly (`ar5iv.labs.arxiv.org`) to avoid a 301 redirect hop.
//!
//! ## Fallback
//!
//! When a paper has no ar5iv render (404, not LaTeX-sourced), the client
//! returns an `ArxivFullText` carrying the Atom abstract as the body with
//! `had_ar5iv: false`. The paper is still indexable; provenance is intact.
//!
//! ## Rate limiting + etiquette
//!
//! arxiv asks for ≤3 req/sec with a `User-Agent` identifying the client.
//! We self-throttle at 500 ms between requests (~2 req/sec) via the same
//! `tokio::sync::Mutex<Option<Instant>>` pattern as `s2_enrichment.rs`.
//!
//! ## SSRF mitigation (T-21-03)
//!
//! Base URLs are hardcoded constants in `ArxivConfig::default()`. Only the
//! `arxiv_id` path segment derives from upstream Atom data; caller-supplied
//! full URLs are never accepted.

use alzina_core::error::{AlzinaError, AlzinaResult, SearchDetail};
use quick_xml::events::Event;
use quick_xml::reader::Reader;

// ── Public output types ───────────────────────────────────────────────────

/// One arxiv search hit, parsed from the Atom feed.
#[derive(Debug, Clone)]
pub struct ArxivResult {
    /// arxiv ID without version suffix, e.g. `"2105.14103"`.
    pub arxiv_id: String,
    pub title: String,
    /// The Atom `<summary>` field (abstract).
    pub abstract_text: String,
    pub authors: Vec<String>,
    pub published: String,
}

/// Full-text content for one arxiv paper.
/// When `had_ar5iv=false`, `body` holds the Atom abstract.
#[derive(Debug, Clone)]
pub struct ArxivFullText {
    pub arxiv_id: String,
    /// Raw HTML string when `had_ar5iv=true`, plain abstract text otherwise.
    pub body: String,
    /// `true` if the body came from an ar5iv HTML render.
    pub had_ar5iv: bool,
}

// ── Config + client ────────────────────────────────────────────────────────

/// Configuration knobs for the arxiv/ar5iv client.
#[derive(Debug, Clone)]
pub struct ArxivConfig {
    /// Base URL for the arxiv Atom API (no trailing slash).
    pub base_url: String,
    /// Base URL for ar5iv HTML renders (no trailing slash).
    pub ar5iv_base_url: String,
    /// Min duration between requests. Default 500ms (~2 req/sec).
    pub min_interval_ms: u64,
    /// Per-request timeout in seconds. Default 15.
    pub timeout_secs: u64,
    /// Max search results to return. Default 10.
    pub limit: usize,
    /// User-Agent header. arxiv etiquette requires identification.
    pub user_agent: String,
}

impl Default for ArxivConfig {
    fn default() -> Self {
        Self {
            base_url: "https://export.arxiv.org/api".into(),
            ar5iv_base_url: "https://ar5iv.labs.arxiv.org".into(),
            min_interval_ms: 500,
            timeout_secs: 15,
            limit: 10,
            user_agent: "alzina/0.1 (mailto:sam@axiotic.ai)".into(),
        }
    }
}

/// arxiv/ar5iv client. Cheap to construct; share across queries.
/// Owns a `reqwest::Client` + rate-limit state.
pub struct ArxivClient {
    client: reqwest::Client,
    config: ArxivConfig,
    /// Wall-clock instant of the last outbound request. `tokio::sync::Mutex`
    /// so it is held safely across await points.
    last_request_at: tokio::sync::Mutex<Option<std::time::Instant>>,
}

// ── Private helper ─────────────────────────────────────────────────────────

fn search_err(message: impl Into<String>, reason: impl Into<String>) -> AlzinaError {
    let reason = reason.into();
    AlzinaError::Search(SearchDetail {
        message: message.into(),
        degraded: true,
        degradation_reason: Some(reason),
    })
}

impl ArxivClient {
    /// Construct with explicit config.
    pub fn new(config: ArxivConfig) -> AlzinaResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| {
                search_err(
                    format!("reqwest client build: {e}"),
                    format!("reqwest client build failed: {e}"),
                )
            })?;
        Ok(Self {
            client,
            config,
            last_request_at: tokio::sync::Mutex::new(None),
        })
    }

    // ── Rate-limit gate (verbatim from s2_enrichment.rs:174-185) ─────────

    async fn rate_limit_gate(&self) {
        let mut last = self.last_request_at.lock().await;
        if let Some(prev) = *last {
            let min = std::time::Duration::from_millis(self.config.min_interval_ms);
            let elapsed = prev.elapsed();
            if elapsed < min {
                let sleep_for = min - elapsed;
                tokio::time::sleep(sleep_for).await;
            }
        }
        *last = Some(std::time::Instant::now());
    }

    // ── Atom search ────────────────────────────────────────────────────────

    /// Search arxiv via the Atom API. Returns up to `config.limit` results.
    ///
    /// On HTTP non-2xx returns `Err(AlzinaError::Search { degraded: true, ... })`.
    pub async fn search(&self, query: &str) -> AlzinaResult<Vec<ArxivResult>> {
        self.rate_limit_gate().await;

        let limit_str = self.config.limit.to_string();
        let url = format!("{}/query", self.config.base_url.trim_end_matches('/'));

        let resp = self
            .client
            .get(&url)
            .query(&[("search_query", query), ("max_results", &limit_str)])
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "arxiv Atom send failed");
                search_err(
                    format!("arxiv send failed: {e}"),
                    format!("arxiv unreachable: {e}"),
                )
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            tracing::warn!(status = %status, body = %body_text, "arxiv API non-2xx response");
            let reason = match status.as_u16() {
                429 => "arxiv rate-limited; retry later".to_string(),
                404 => "arxiv search endpoint not found".to_string(),
                _ => format!("arxiv returned {status}"),
            };
            return Err(search_err(format!("arxiv HTTP {status}: {body_text}"), reason));
        }

        let xml = resp.text().await.map_err(|e| {
            search_err(
                format!("arxiv Atom body read failed: {e}"),
                format!("arxiv Atom response unreadable: {e}"),
            )
        })?;

        parse_atom_xml(&xml)
    }

    /// Fetch metadata for specific arxiv ids in one batched Atom call.
    ///
    /// Uses the arxiv API `id_list` parameter (comma-separated). Returns one
    /// `ArxivResult` per resolvable id (authors + published populated). Used by
    /// the author/year backfill — keep batches modest (≤ ~100 ids) so the URL
    /// and `max_results` stay within arxiv's limits.
    ///
    /// SSRF: ids are sent only as the `id_list` query value against the
    /// hardcoded base url; no caller-supplied URLs are used.
    pub async fn fetch_by_ids(&self, ids: &[String]) -> AlzinaResult<Vec<ArxivResult>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        self.rate_limit_gate().await;

        let id_list = ids.join(",");
        let max_results = ids.len().to_string();
        let url = format!("{}/query", self.config.base_url.trim_end_matches('/'));

        let resp = self
            .client
            .get(&url)
            .query(&[("id_list", id_list.as_str()), ("max_results", &max_results)])
            .send()
            .await
            .map_err(|e| {
                search_err(
                    format!("arxiv id_list send failed: {e}"),
                    format!("arxiv unreachable: {e}"),
                )
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            let reason = match status.as_u16() {
                429 => "arxiv rate-limited; retry later".to_string(),
                _ => format!("arxiv returned {status}"),
            };
            return Err(search_err(
                format!("arxiv id_list HTTP {status}: {body_text}"),
                reason,
            ));
        }

        let xml = resp.text().await.map_err(|e| {
            search_err(
                format!("arxiv id_list body read failed: {e}"),
                format!("arxiv Atom response unreadable: {e}"),
            )
        })?;

        parse_atom_xml(&xml)
    }

    // ── ar5iv full-text fetch ─────────────────────────────────────────────

    /// Fetch the ar5iv HTML full-text for a given `arxiv_id`.
    ///
    /// On 404 (no LaTeX render) returns `ArxivFullText { had_ar5iv: false,
    /// body: abstract_fallback }` — still indexable, provenance intact.
    ///
    /// SSRF: `arxiv_id` is used only as a path segment appended to the
    /// hardcoded `ar5iv_base_url`. Caller-supplied full URLs are never used.
    pub async fn fetch_fulltext(
        &self,
        arxiv_id: &str,
        abstract_fallback: &str,
    ) -> AlzinaResult<ArxivFullText> {
        self.rate_limit_gate().await;

        // SSRF guard: only the path segment is caller-supplied.
        let url = format!(
            "{}/abs/{}",
            self.config.ar5iv_base_url.trim_end_matches('/'),
            arxiv_id
        );

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "ar5iv fetch send failed");
                search_err(
                    format!("ar5iv send failed: {e}"),
                    format!("ar5iv unreachable: {e}"),
                )
            })?;

        let status = resp.status();

        // 404 → no LaTeX render; fall back to abstract. Not degraded.
        if status.as_u16() == 404 {
            tracing::info!(arxiv_id = %arxiv_id, "ar5iv: 404 — no LaTeX render; using Atom abstract");
            return Ok(ArxivFullText {
                arxiv_id: arxiv_id.to_string(),
                body: abstract_fallback.to_string(),
                had_ar5iv: false,
            });
        }

        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            tracing::warn!(arxiv_id = %arxiv_id, status = %status, body = %body_text, "ar5iv non-2xx response");
            let reason = format!("ar5iv returned {status} for {arxiv_id}");
            return Err(search_err(
                format!("ar5iv HTTP {status}: {body_text}"),
                reason,
            ));
        }

        let html = resp.text().await.map_err(|e| {
            search_err(
                format!("ar5iv body read failed: {e}"),
                format!("ar5iv HTML unreadable: {e}"),
            )
        })?;

        Ok(ArxivFullText {
            arxiv_id: arxiv_id.to_string(),
            body: html,
            had_ar5iv: true,
        })
    }
}

// ── Atom XML parser ────────────────────────────────────────────────────────

/// Parse an arxiv Atom XML feed into `Vec<ArxivResult>`.
/// Uses `quick_xml::Reader` — not regex (per Don't-Hand-Roll).
fn parse_atom_xml(xml: &str) -> AlzinaResult<Vec<ArxivResult>> {
    let mut reader = Reader::from_str(xml);
    reader.trim_text(true);

    let mut results = Vec::new();
    let mut in_entry = false;
    let mut current: Option<AtomEntry> = None;
    let mut current_tag: Option<String> = None;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_string();
                // Strip namespace prefix (e.g. "feed" not "ns:feed")
                let local = name.split(':').last().unwrap_or(&name).to_string();
                if local == "entry" {
                    in_entry = true;
                    current = Some(AtomEntry::default());
                }
                if in_entry {
                    current_tag = Some(local);
                }
            }
            Ok(Event::End(ref e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .unwrap_or("")
                    .to_string();
                let local = name.split(':').last().unwrap_or(&name).to_string();
                if local == "entry" {
                    if let Some(entry) = current.take() {
                        if let Some(result) = entry.into_arxiv_result() {
                            results.push(result);
                        }
                    }
                    in_entry = false;
                }
                current_tag = None;
            }
            Ok(Event::Text(ref e)) => {
                if !in_entry {
                    buf.clear();
                    continue;
                }
                let text = e.unescape().unwrap_or_default().trim().to_string();
                if text.is_empty() {
                    buf.clear();
                    continue;
                }
                if let (Some(tag), Some(ref mut entry)) =
                    (current_tag.as_deref(), current.as_mut())
                {
                    match tag {
                        "id" => {
                            entry.id = strip_arxiv_id(&text);
                        }
                        "title" => {
                            if entry.title.is_empty() {
                                entry.title = text;
                            }
                        }
                        "summary" => {
                            entry.summary = text;
                        }
                        "published" => {
                            entry.published = text;
                        }
                        "name" => {
                            // Inside <author><name>
                            entry.authors.push(text);
                        }
                        _ => {}
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(search_err(
                    format!("arxiv Atom XML parse error: {e}"),
                    format!("arxiv returned malformed Atom XML: {e}"),
                ));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(results)
}

/// Strip the arxiv ID URL prefix and version suffix.
/// `"http://arxiv.org/abs/2105.14103v2"` → `"2105.14103"`
fn strip_arxiv_id(raw: &str) -> String {
    let stripped = raw
        .trim_start_matches("http://arxiv.org/abs/")
        .trim_start_matches("https://arxiv.org/abs/");
    // Remove version suffix vN
    if let Some(pos) = stripped.rfind('v') {
        let after = &stripped[pos + 1..];
        if after.chars().all(|c| c.is_ascii_digit()) && !after.is_empty() {
            return stripped[..pos].to_string();
        }
    }
    stripped.to_string()
}

// ── Entry accumulator ──────────────────────────────────────────────────────

#[derive(Default)]
struct AtomEntry {
    id: String,
    title: String,
    summary: String,
    authors: Vec<String>,
    published: String,
}

impl AtomEntry {
    fn into_arxiv_result(self) -> Option<ArxivResult> {
        if self.id.is_empty() {
            return None;
        }
        Some(ArxivResult {
            arxiv_id: self.id,
            title: self.title,
            abstract_text: self.summary,
            authors: self.authors,
            published: self.published,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg_for(server: &MockServer) -> ArxivConfig {
        ArxivConfig {
            base_url: server.uri(),
            ar5iv_base_url: server.uri(),
            min_interval_ms: 100,
            timeout_secs: 10,
            limit: 10,
            user_agent: "test-agent/0.1".into(),
        }
    }

    fn atom_xml_one_entry() -> &'static str {
        r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2105.14103v2</id>
    <title>An Attention Free Transformer</title>
    <summary>We introduce a simple yet effective attention-free architecture.</summary>
    <published>2021-05-28T20:45:30Z</published>
    <author><name>Shuangfei Zhai</name></author>
    <author><name>Walter Talbott</name></author>
  </entry>
</feed>"#
    }

    // ── Test 1: Atom parse + ID stripping ─────────────────────────────────

    #[tokio::test]
    async fn enabled_client_calls_endpoint_and_parses() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/query"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(atom_xml_one_entry())
                    .insert_header("content-type", "application/atom+xml"),
            )
            .mount(&server)
            .await;

        let client = ArxivClient::new(cfg_for(&server)).expect("builds");
        let results = client.search("attention transformer").await.expect("ok");

        assert_eq!(results.len(), 1);
        let r = &results[0];
        // ID must be stripped of URL prefix and version suffix
        assert_eq!(r.arxiv_id, "2105.14103");
        assert_eq!(r.title, "An Attention Free Transformer");
        assert!(
            r.abstract_text.contains("attention-free"),
            "abstract: {}",
            r.abstract_text
        );
        assert_eq!(r.authors, vec!["Shuangfei Zhai", "Walter Talbott"]);
        assert!(!r.published.is_empty());
    }

    // ── Test 1b: fetch_by_ids (id_list batch) ──────────────────────────────

    #[tokio::test]
    async fn fetch_by_ids_parses_authors_and_year() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/query"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(atom_xml_one_entry())
                    .insert_header("content-type", "application/atom+xml"),
            )
            .mount(&server)
            .await;

        let client = ArxivClient::new(cfg_for(&server)).expect("builds");
        let results = client
            .fetch_by_ids(&["2105.14103".to_string()])
            .await
            .expect("ok");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].arxiv_id, "2105.14103");
        assert_eq!(results[0].authors, vec!["Shuangfei Zhai", "Walter Talbott"]);
        assert!(results[0].published.starts_with("2021"));

        // Empty input short-circuits without a request.
        assert!(client.fetch_by_ids(&[]).await.unwrap().is_empty());
    }

    // ── Test 2: ar5iv 404 fallback ─────────────────────────────────────────

    #[tokio::test]
    async fn fallback_on_404_returns_abstract_not_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/abs/2105.14103"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Not Found"))
            .mount(&server)
            .await;

        let client = ArxivClient::new(cfg_for(&server)).expect("builds");
        let result = client
            .fetch_fulltext("2105.14103", "The abstract text from Atom feed.")
            .await
            .expect("should not error on 404");

        assert!(!result.had_ar5iv, "had_ar5iv should be false on 404");
        assert_eq!(result.arxiv_id, "2105.14103");
        assert_eq!(result.body, "The abstract text from Atom feed.");
    }

    // ── Test 3: ar5iv 200 returns HTML ─────────────────────────────────────

    #[tokio::test]
    async fn fulltext_fetch_returns_html_on_200() {
        let html = r#"<html><body><section id="S1" class="ltx_section"><h2>Introduction</h2><p>Hello</p></section></body></html>"#;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/abs/2105.14103"))
            .respond_with(ResponseTemplate::new(200).set_body_string(html))
            .mount(&server)
            .await;

        let client = ArxivClient::new(cfg_for(&server)).expect("builds");
        let result = client
            .fetch_fulltext("2105.14103", "fallback abstract")
            .await
            .expect("ok");

        assert!(result.had_ar5iv, "had_ar5iv should be true on 200");
        assert_eq!(result.arxiv_id, "2105.14103");
        assert!(result.body.contains("ltx_section"));
    }

    // ── Test 4: rate-limit enforced between calls ──────────────────────────

    #[tokio::test]
    async fn rate_limit_enforced_between_calls() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/query"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(atom_xml_one_entry())
                    .insert_header("content-type", "application/atom+xml"),
            )
            .mount(&server)
            .await;

        let mut cfg = cfg_for(&server);
        cfg.min_interval_ms = 200;
        let client = ArxivClient::new(cfg).expect("builds");

        let start = std::time::Instant::now();
        client.search("a").await.expect("ok");
        client.search("b").await.expect("ok");
        let elapsed = start.elapsed();

        assert!(
            elapsed >= std::time::Duration::from_millis(200),
            "expected >= 200ms between two calls, got {elapsed:?}"
        );
    }

    // ── Test 5: HTTP 500 returns degraded error ────────────────────────────

    #[tokio::test]
    async fn search_returns_degraded_on_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/query"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Error"))
            .mount(&server)
            .await;

        let client = ArxivClient::new(cfg_for(&server)).expect("builds");
        let err = client.search("foo").await.expect_err("should fail");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
                assert!(d.degradation_reason.is_some());
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    // ── Test 6: arxiv ID stripping edge cases ─────────────────────────────

    #[test]
    fn strip_arxiv_id_strips_url_and_version() {
        assert_eq!(strip_arxiv_id("http://arxiv.org/abs/2105.14103v2"), "2105.14103");
        assert_eq!(strip_arxiv_id("http://arxiv.org/abs/2105.14103"), "2105.14103");
        assert_eq!(strip_arxiv_id("http://arxiv.org/abs/1706.03762v5"), "1706.03762");
        assert_eq!(strip_arxiv_id("2105.14103"), "2105.14103");
    }
}
