//! Semantic Scholar live enrichment.
//!
//! Phase 3 Task 3.8. Optional, opt-in (env `S2_LIVE_ENABLED=true` to activate).
//! Returns external paper metadata as a separate `s2_results` field on
//! search responses — NOT fused into the local RRF ranking.
//!
//! AC-1: API failures (rate limits, timeouts, HTTP errors) degrade loudly
//! with a reason. Disabled state is announced once at startup.
//!
//! ## Endpoint
//!
//! `GET {base_url}/paper/search?query=...&limit=...&fields=...` with optional
//! `x-api-key` header. The free tier works without a key but is more
//! aggressively rate-limited; we self-throttle to 1 req/sec by default.
//!
//! ## Disabled vs failing
//!
//! When `S2_LIVE_ENABLED=false` (default) `enrich()` returns `Ok(vec![])`
//! without any network I/O — S2 is opt-in extra signal, not a default
//! expectation, so disabled is silent. When enabled but the API fails (429,
//! 5xx, network error), `enrich()` returns
//! `Err(AlzinaError::Search(SearchDetail { degraded: true, ... }))`.

use serde::{Deserialize, Serialize};

use alzina_core::error::{AlzinaError, AlzinaResult, SearchDetail};

/// One S2 search hit. Snake-case for Rust ergonomics; the S2 wire shape uses
/// camelCase and is parsed via [`WireS2Paper`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2Result {
    pub paper_id: String,
    /// ArXiv id from S2 `externalIds.ArXiv`, when the paper is on arxiv.
    /// Drives canonical `arxiv:{id}` keying + the ar5iv full-text promotion
    /// path (probe-15 fix: S2-lane papers were locked out of full text).
    #[serde(default)]
    pub arxiv_id: Option<String>,
    pub title: String,
    /// Renamed because `abstract` is a Rust keyword. Kept wire-compatible.
    #[serde(rename = "abstract")]
    pub abstract_text: Option<String>,
    pub year: Option<i32>,
    pub authors: Vec<String>,
    pub citation_count: Option<i64>,
    /// S2's URL for the paper (e.g. `https://www.semanticscholar.org/paper/{paper_id}`).
    pub url: String,
    /// Open-access PDF URL from S2's `openAccessPdf.url` field, when present.
    /// `#[serde(default)]` is mandatory — old cached hits lack this field.
    #[serde(default)]
    pub open_access_pdf_url: Option<String>,
}

/// Configuration knobs for the S2 client.
#[derive(Debug, Clone)]
pub struct S2Config {
    pub base_url: String,
    /// Optional API key. When provided, attached as `x-api-key` header.
    /// S2's free tier works without a key but is more aggressively rate-limited.
    pub api_key: Option<String>,
    /// Min duration between requests. Default 1000ms.
    pub min_interval_ms: u64,
    /// Per-request timeout. Default 10s.
    pub timeout_secs: u64,
    /// Max results to return. Default 5.
    pub limit: usize,
}

impl Default for S2Config {
    fn default() -> Self {
        Self {
            base_url: "https://api.semanticscholar.org/graph/v1".into(),
            api_key: None,
            min_interval_ms: 1000,
            timeout_secs: 10,
            limit: 5,
        }
    }
}

/// S2 client. Cheap to construct; share across queries (it owns a
/// `reqwest::Client` + rate-limit state).
pub struct S2Client {
    client: reqwest::Client,
    config: S2Config,
    /// Wall-clock instant of the last outbound request. Held across awaits
    /// so we use `tokio::sync::Mutex` (not `std::sync::Mutex`).
    last_request_at: tokio::sync::Mutex<Option<std::time::Instant>>,
    enabled: bool,
}

// ── Private wire-shape helpers ────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct WireS2Response {
    #[serde(default)]
    data: Vec<WireS2Paper>,
}

/// Wire shape for S2's `openAccessPdf` object.
/// Only `url` is surfaced; status/license/disclaimer are ignored via
/// serde's default unknown-field tolerance.
#[derive(Debug, Deserialize)]
struct WireS2OpenAccessPdf {
    url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireS2Paper {
    #[serde(rename = "paperId")]
    paper_id: Option<String>,
    #[serde(rename = "externalIds", default)]
    external_ids: Option<WireS2ExternalIds>,
    title: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    year: Option<i32>,
    #[serde(default)]
    authors: Vec<WireS2Author>,
    #[serde(rename = "citationCount")]
    citation_count: Option<i64>,
    url: Option<String>,
    /// S2's `openAccessPdf` object — may be JSON null or absent.
    #[serde(rename = "openAccessPdf", default)]
    open_access_pdf: Option<WireS2OpenAccessPdf>,
}

#[derive(Debug, Deserialize)]
struct WireS2ExternalIds {
    #[serde(rename = "ArXiv")]
    arxiv: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireS2Author {
    name: Option<String>,
}

impl S2Client {
    /// Construct from env. Reads `S2_LIVE_ENABLED` (default false) and
    /// `S2_API_KEY` (optional). When `enabled=false`, calls to [`Self::enrich`]
    /// return `Ok(vec![])` without hitting the network.
    pub fn from_env() -> AlzinaResult<Self> {
        let enabled = std::env::var("S2_LIVE_ENABLED")
            .ok()
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);

        let api_key = std::env::var("S2_API_KEY").ok().filter(|s| !s.is_empty());

        let config = S2Config {
            api_key,
            ..S2Config::default()
        };

        if enabled {
            tracing::info!("S2 live enrichment ENABLED");
        } else {
            tracing::debug!("S2 live enrichment disabled (set S2_LIVE_ENABLED=true to opt in)");
        }

        Self::with_config(config, enabled)
    }

    /// Construct with explicit config. `enabled=true` activates the client;
    /// `enabled=false` still constructs the client but [`Self::enrich`] is a
    /// no-op.
    pub fn with_config(config: S2Config, enabled: bool) -> AlzinaResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
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
            config,
            last_request_at: tokio::sync::Mutex::new(None),
            enabled,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Run an S2 search. Returns up to `config.limit` results.
    ///
    /// AC-1: when disabled, returns `Ok(vec![])` — caller should NOT flag
    /// degraded for the disabled state (S2 is opt-in extra signal). When
    /// enabled but the API fails, returns
    /// `Err(AlzinaError::Search(SearchDetail { degraded: true, ... }))`.
    pub async fn enrich(&self, query: &str) -> AlzinaResult<Vec<S2Result>> {
        if !self.enabled {
            return Ok(Vec::new());
        }

        // ── Rate limit: serialize callers across the gate ─────────────
        {
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

        let url = format!(
            "{}/paper/search",
            self.config.base_url.trim_end_matches('/')
        );
        let limit_str = self.config.limit.to_string();
        let fields = "title,abstract,year,authors,citationCount,paperId,url,externalIds,openAccessPdf";

        let mut req = self.client.get(&url).query(&[
            ("query", query),
            ("limit", &limit_str),
            ("fields", fields),
        ]);
        if let Some(key) = &self.config.api_key {
            req = req.header("x-api-key", key);
        }

        let resp = req.send().await.map_err(|e| {
            tracing::warn!(error = %e, "S2 API send failed");
            AlzinaError::Search(SearchDetail {
                message: format!("S2 send failed: {e}"),
                degraded: true,
                degradation_reason: Some(format!("S2 unreachable: {e}")),
            })
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            tracing::warn!(status = %status, body = %body_text, "S2 API non-2xx response");
            let reason = match status.as_u16() {
                429 => "S2 rate-limited; retry later".to_string(),
                401 | 403 => "S2 auth failed".to_string(),
                _ => format!("S2 returned {status}"),
            };
            return Err(AlzinaError::Search(SearchDetail {
                message: format!("S2 HTTP {status}: {body_text}"),
                degraded: true,
                degradation_reason: Some(reason),
            }));
        }

        let parsed: WireS2Response = resp.json().await.map_err(|e| {
            tracing::warn!(error = %e, "S2 API response JSON decode failed");
            AlzinaError::Search(SearchDetail {
                message: format!("S2 response decode failed: {e}"),
                degraded: true,
                degradation_reason: Some(format!("S2 returned invalid JSON: {e}")),
            })
        })?;

        let results = parsed
            .data
            .into_iter()
            .map(|p| S2Result {
                paper_id: p.paper_id.unwrap_or_default(),
                arxiv_id: p.external_ids.and_then(|e| e.arxiv).filter(|a| !a.trim().is_empty()),
                title: p.title.unwrap_or_default(),
                abstract_text: p.abstract_text,
                year: p.year,
                authors: p.authors.into_iter().filter_map(|a| a.name).collect(),
                citation_count: p.citation_count,
                url: p.url.unwrap_or_default(),
                open_access_pdf_url: p.open_access_pdf
                    .and_then(|oa| oa.url)
                    .filter(|u| !u.trim().is_empty()),
            })
            .collect();

        Ok(results)
    }
}

// ── S2PaperFull ──────────────────────────────────────────────────────────────

/// Full paper record parsed from the Semantic Scholar graph API.
///
/// Port of clawd's `S2Paper` dataclass (semantic_scholar.py:42-78).
/// Used as the cache payload (Serialize + Deserialize) and as the return
/// type of the new graph endpoint methods.
///
/// T-iab-01: all fields populated via typed serde parsing; no raw wire
/// strings interpolated into SQL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2PaperFull {
    pub s2_id: String,
    pub arxiv_id: Option<String>,
    pub title: String,
    pub abstract_text: Option<String>,
    pub year: Option<i32>,
    pub citation_count: i64,
    /// S2's `influentialCitationCount` — citations S2 judges as substantively
    /// building on the paper, not just listing it. A sharper credibility signal
    /// than raw `citation_count`; feeds the per-source authenticity tier.
    /// `#[serde(default)]` MANDATORY — old s2_cache payloads lack this field.
    #[serde(default)]
    pub influential_citation_count: i64,
    pub reference_count: i64,
    pub authors: Vec<String>,
    pub venue: Option<String>,
    pub doi: Option<String>,
    /// Open-access PDF URL from S2's `openAccessPdf.url` field.
    /// `#[serde(default)]` is MANDATORY — old s2_cache payloads WITHOUT this
    /// field must still deserialize cleanly (back-compat invariant).
    #[serde(default)]
    pub open_access_pdf_url: Option<String>,
}

/// Field-set constants ported verbatim from semantic_scholar.py:28-29, extended
/// with `openAccessPdf` (F10). `S2_CITATION_FIELDS` is left verbatim — it is a
/// clawd port and citation/reference traversal never needs the URL.
pub const S2_DEFAULT_FIELDS: &str =
    "paperId,externalIds,title,abstract,year,citationCount,influentialCitationCount,referenceCount,authors,venue,openAccessPdf";
pub const S2_CITATION_FIELDS: &str =
    "paperId,externalIds,title,abstract,year,citationCount";

/// Wire shape for a single paper from the graph API — wider than the existing
/// `WireS2Paper` (adds externalIds and referenceCount).
#[derive(Debug, Deserialize)]
struct WireS2FullPaper {
    #[serde(rename = "paperId")]
    paper_id: Option<String>,
    title: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    year: Option<i32>,
    #[serde(rename = "citationCount")]
    citation_count: Option<i64>,
    #[serde(rename = "influentialCitationCount")]
    influential_citation_count: Option<i64>,
    #[serde(rename = "referenceCount")]
    reference_count: Option<i64>,
    #[serde(default)]
    authors: Vec<WireS2Author>,
    venue: Option<String>,
    #[serde(rename = "externalIds", default)]
    external_ids: std::collections::HashMap<String, serde_json::Value>,
    /// S2's `openAccessPdf` object — may be JSON null or absent.
    #[serde(rename = "openAccessPdf", default)]
    open_access_pdf: Option<WireS2OpenAccessPdf>,
}

impl WireS2FullPaper {
    fn into_full(self) -> Option<S2PaperFull> {
        let s2_id = self.paper_id?;

        // Normalise arxiv version suffix: "2105.14103v2" → "2105.14103".
        // Only strip when the suffix after 'v' is purely numeric (avoids
        // mangling old-style ids that contain 'v' for other reasons).
        let arxiv_id = self.external_ids.get("ArXiv").and_then(|v| v.as_str()).map(|raw| {
            if let Some(pos) = raw.rfind('v') {
                let suffix = &raw[pos + 1..];
                if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                    return raw[..pos].to_string();
                }
            }
            raw.to_string()
        });

        let doi = self.external_ids.get("DOI").and_then(|v| v.as_str()).map(String::from);
        let authors: Vec<String> = self.authors.into_iter().filter_map(|a| a.name).collect();

        let open_access_pdf_url = self.open_access_pdf
            .and_then(|oa| oa.url)
            .filter(|u| !u.trim().is_empty());

        Some(S2PaperFull {
            s2_id,
            arxiv_id,
            title: self.title.unwrap_or_default(),
            abstract_text: self.abstract_text,
            year: self.year,
            citation_count: self.citation_count.unwrap_or(0),
            influential_citation_count: self.influential_citation_count.unwrap_or(0),
            reference_count: self.reference_count.unwrap_or(0),
            authors,
            venue: self.venue,
            doi,
            open_access_pdf_url,
        })
    }
}

/// Error from a raw S2 graph-endpoint call.
///
/// Carries enough structure for the explorer to classify:
/// - 429 / 5xx → `RetryAdvice::Retry` (honour Retry-After when present)
/// - anything else → `RetryAdvice::Fatal`
#[derive(Debug)]
pub struct S2CallError {
    /// HTTP status if the error came from a server response.
    pub status: Option<u16>,
    /// Retry-After header parsed to a Duration (present on 429 responses that
    /// carry the header).
    pub retry_after: Option<std::time::Duration>,
    /// Human-readable message for logging.
    pub message: String,
}

// ── Resolve ID helper ──────────────────────────────────────────────────────

/// Port of `SemanticScholarClient._resolve_id` (semantic_scholar.py:207-218).
///
/// "1706.03762"  → "ARXIV:1706.03762"
/// "S2:abc..."   → "abc..."
/// opaque hex id → passes through
pub fn resolve_paper_id(paper_id: &str) -> String {
    if let Some(rest) = paper_id.strip_prefix("S2:") {
        return rest.to_string();
    }
    if paper_id.starts_with("ARXIV:") {
        return paper_id.to_string();
    }
    // Detect arxiv-style id: "digits.digits" optionally followed by "vN".
    // semantic_scholar.py:213 heuristic: contains '.' and stripping '.'/'v'/digits
    // leaves nothing (i.e. purely numeric + dot + optional version).
    let base = if let Some(pos) = paper_id.rfind('v') {
        let suffix = &paper_id[pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            &paper_id[..pos]
        } else {
            paper_id
        }
    } else {
        paper_id
    };
    if base.contains('.') {
        let stripped: String = base.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
        if stripped == base {
            return format!("ARXIV:{base}");
        }
    }
    paper_id.to_string()
}

impl S2Client {
    /// GET /paper/{resolved}?fields=DEFAULT_FIELDS — 404 → Ok(None).
    ///
    /// Raw endpoint call; pacing, budgets, and backoff are the
    /// LitGateway's job at the call site (A1 single chokepoint). Do NOT
    /// call the `enrich()` rate-limit mutex here.
    pub async fn get_paper(&self, paper_id: &str) -> Result<Option<S2PaperFull>, S2CallError> {
        if !self.enabled {
            return Ok(None);
        }
        let resolved = resolve_paper_id(paper_id);
        let url = format!(
            "{}/paper/{}?fields={S2_DEFAULT_FIELDS}",
            self.config.base_url.trim_end_matches('/'),
            resolved
        );

        let mut req = self.client.get(&url);
        if let Some(key) = &self.config.api_key {
            req = req.header("x-api-key", key);
        }

        let resp = req.send().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("get_paper send: {e}"),
        })?;

        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(None);
        }
        if !status.is_success() {
            let retry_after = if status.as_u16() == 429 {
                parse_retry_after(resp.headers())
            } else {
                None
            };
            let body = resp.text().await.unwrap_or_default();
            return Err(S2CallError {
                status: Some(status.as_u16()),
                retry_after,
                message: format!("get_paper HTTP {}: {body}", status.as_u16()),
            });
        }

        let wire: WireS2FullPaper = resp.json().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("get_paper decode: {e}"),
        })?;

        Ok(wire.into_full())
    }

    /// GET /paper/{resolved}/citations?fields=CITATION_FIELDS&limit={limit}
    ///
    /// Parses `data[].citingPaper`; drops entries without paperId (clawd
    /// :241-276). Raw call — no pacing/budget here.
    pub async fn get_citations(
        &self,
        paper_id: &str,
        limit: usize,
    ) -> Result<Vec<S2PaperFull>, S2CallError> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        let resolved = resolve_paper_id(paper_id);
        let url = format!(
            "{}/paper/{}/citations?fields={S2_CITATION_FIELDS}&limit={limit}",
            self.config.base_url.trim_end_matches('/'),
            resolved
        );

        let mut req = self.client.get(&url);
        if let Some(key) = &self.config.api_key {
            req = req.header("x-api-key", key);
        }

        let resp = req.send().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("get_citations send: {e}"),
        })?;

        let status = resp.status();
        if !status.is_success() {
            let retry_after = if status.as_u16() == 429 {
                parse_retry_after(resp.headers())
            } else {
                None
            };
            let body = resp.text().await.unwrap_or_default();
            return Err(S2CallError {
                status: Some(status.as_u16()),
                retry_after,
                message: format!("get_citations HTTP {}: {body}", status.as_u16()),
            });
        }

        #[derive(Deserialize)]
        struct CitationsResponse {
            #[serde(default)]
            data: Vec<CitationItem>,
        }
        #[derive(Deserialize)]
        struct CitationItem {
            #[serde(rename = "citingPaper")]
            citing_paper: Option<WireS2FullPaper>,
        }

        let body: CitationsResponse = resp.json().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("get_citations decode: {e}"),
        })?;

        let papers = body
            .data
            .into_iter()
            .filter_map(|item| item.citing_paper.and_then(|p| p.into_full()))
            .collect();

        Ok(papers)
    }

    /// GET /paper/{resolved}/references?fields=CITATION_FIELDS&limit={limit}
    ///
    /// Parses `data[].citedPaper` (clawd :278-313). Raw call.
    pub async fn get_references(
        &self,
        paper_id: &str,
        limit: usize,
    ) -> Result<Vec<S2PaperFull>, S2CallError> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        let resolved = resolve_paper_id(paper_id);
        let url = format!(
            "{}/paper/{}/references?fields={S2_CITATION_FIELDS}&limit={limit}",
            self.config.base_url.trim_end_matches('/'),
            resolved
        );

        let mut req = self.client.get(&url);
        if let Some(key) = &self.config.api_key {
            req = req.header("x-api-key", key);
        }

        let resp = req.send().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("get_references send: {e}"),
        })?;

        let status = resp.status();
        if !status.is_success() {
            let retry_after = if status.as_u16() == 429 {
                parse_retry_after(resp.headers())
            } else {
                None
            };
            let body = resp.text().await.unwrap_or_default();
            return Err(S2CallError {
                status: Some(status.as_u16()),
                retry_after,
                message: format!("get_references HTTP {}: {body}", status.as_u16()),
            });
        }

        #[derive(Deserialize)]
        struct ReferencesResponse {
            #[serde(default)]
            data: Vec<ReferenceItem>,
        }
        #[derive(Deserialize)]
        struct ReferenceItem {
            #[serde(rename = "citedPaper")]
            cited_paper: Option<WireS2FullPaper>,
        }

        let body: ReferencesResponse = resp.json().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("get_references decode: {e}"),
        })?;

        let papers = body
            .data
            .into_iter()
            .filter_map(|item| item.cited_paper.and_then(|p| p.into_full()))
            .collect();

        Ok(papers)
    }

    /// POST /paper/batch?fields=DEFAULT_FIELDS body `{"ids": [...]}`
    ///
    /// Results in input order; None for not-found (clawd :329-376).
    /// Raw call — no pacing/budget here.
    pub async fn get_papers_batch(
        &self,
        ids: &[String],
    ) -> Result<Vec<Option<S2PaperFull>>, S2CallError> {
        if !self.enabled {
            return Ok(vec![]);
        }
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let resolved: Vec<String> = ids.iter().map(|id| resolve_paper_id(id)).collect();
        let url = format!(
            "{}/paper/batch?fields={S2_DEFAULT_FIELDS}",
            self.config.base_url.trim_end_matches('/')
        );

        let body_json = serde_json::json!({ "ids": resolved });
        let mut req = self
            .client
            .post(&url)
            .json(&body_json);
        if let Some(key) = &self.config.api_key {
            req = req.header("x-api-key", key);
        }

        let resp = req.send().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("get_papers_batch send: {e}"),
        })?;

        let status = resp.status();
        if !status.is_success() {
            let retry_after = if status.as_u16() == 429 {
                parse_retry_after(resp.headers())
            } else {
                None
            };
            let body = resp.text().await.unwrap_or_default();
            return Err(S2CallError {
                status: Some(status.as_u16()),
                retry_after,
                message: format!("get_papers_batch HTTP {}: {body}", status.as_u16()),
            });
        }

        // Response is a JSON array, one entry per input id (may be null for not-found).
        let raw: Vec<Option<WireS2FullPaper>> = resp.json().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("get_papers_batch decode: {e}"),
        })?;

        let results = raw
            .into_iter()
            .map(|item| item.and_then(|p| p.into_full()))
            .collect();

        Ok(results)
    }

    /// GET /paper/search?query={}&fields=DEFAULT_FIELDS&limit={limit}
    ///
    /// Richer field set than `enrich()` (includes referenceCount + externalIds).
    /// `enrich()` is unchanged — it handles the fusion lane. This method is for
    /// Stage 0 seed discovery.
    pub async fn search_papers(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<S2PaperFull>, S2CallError> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        let url = format!(
            "{}/paper/search",
            self.config.base_url.trim_end_matches('/')
        );
        let limit_str = limit.to_string();

        let mut req = self.client.get(&url).query(&[
            ("query", query),
            ("limit", &limit_str),
            ("fields", S2_DEFAULT_FIELDS),
        ]);
        if let Some(key) = &self.config.api_key {
            req = req.header("x-api-key", key);
        }

        let resp = req.send().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("search_papers send: {e}"),
        })?;

        let status = resp.status();
        if !status.is_success() {
            let retry_after = if status.as_u16() == 429 {
                parse_retry_after(resp.headers())
            } else {
                None
            };
            let body = resp.text().await.unwrap_or_default();
            return Err(S2CallError {
                status: Some(status.as_u16()),
                retry_after,
                message: format!("search_papers HTTP {}: {body}", status.as_u16()),
            });
        }

        #[derive(Deserialize)]
        struct SearchResponse {
            #[serde(default)]
            data: Vec<WireS2FullPaper>,
        }

        let body: SearchResponse = resp.json().await.map_err(|e| S2CallError {
            status: None,
            retry_after: None,
            message: format!("search_papers decode: {e}"),
        })?;

        let papers = body.data.into_iter().filter_map(|p| p.into_full()).collect();
        Ok(papers)
    }
}

// ── Retry-After header parser ─────────────────────────────────────────────────

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    let val = headers.get("retry-after")?;
    let s = val.to_str().ok()?;
    // May be a delta-seconds value or an HTTP-date; we only handle seconds here.
    s.trim().parse::<u64>().ok().map(std::time::Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg_for(server: &MockServer) -> S2Config {
        S2Config {
            base_url: server.uri(),
            api_key: None,
            min_interval_ms: 1000,
            timeout_secs: 10,
            limit: 5,
        }
    }

    fn one_paper_response() -> serde_json::Value {
        json!({
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Attention Is All You Need",
                    "abstract": "We propose a new simple network architecture, the Transformer.",
                    "year": 2017,
                    "authors": [
                        {"name": "Ashish Vaswani"},
                        {"name": "Noam Shazeer"}
                    ],
                    "citationCount": 99999,
                    "url": "https://www.semanticscholar.org/paper/abc123"
                }
            ]
        })
    }

    #[tokio::test]
    async fn disabled_client_returns_empty_without_network() {
        // No mock server: any network call would fail.
        let client = S2Client::with_config(S2Config::default(), false).expect("builds");
        let out = client.enrich("anything").await.expect("ok");
        assert!(out.is_empty());
        assert!(!client.is_enabled());
    }

    #[tokio::test]
    async fn enabled_client_calls_endpoint_and_parses() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(one_paper_response()))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let out = client.enrich("transformers").await.expect("ok");
        assert_eq!(out.len(), 1);
        let p = &out[0];
        assert_eq!(p.paper_id, "abc123");
        assert_eq!(p.title, "Attention Is All You Need");
        assert_eq!(p.year, Some(2017));
        assert_eq!(p.citation_count, Some(99999));
        assert_eq!(p.authors, vec!["Ashish Vaswani", "Noam Shazeer"]);
        assert!(p.abstract_text.as_deref().unwrap().contains("Transformer"));
        assert_eq!(p.url, "https://www.semanticscholar.org/paper/abc123");
    }

    #[tokio::test]
    async fn disabled_client_makes_zero_live_calls() {
        // base_url points at a refused port (127.0.0.1:1). If any method made a
        // real network call it would Err; the disabled guards must short-circuit
        // to empty results WITHOUT touching the network. This is the structural
        // guarantee that an unkeyed daemon never hits S2 anonymously.
        let cfg = S2Config {
            base_url: "http://127.0.0.1:1".into(),
            api_key: None,
            min_interval_ms: 1000,
            timeout_secs: 5,
            limit: 5,
        };
        let client = S2Client::with_config(cfg, false).expect("builds");
        assert!(!client.is_enabled());

        assert!(client.enrich("q").await.expect("ok").is_empty());
        assert!(client.search_papers("q", 5).await.expect("ok").is_empty());
        assert!(client.get_paper("arxiv:1234.5678").await.expect("ok").is_none());
        assert!(client.get_citations("arxiv:1234.5678", 5).await.expect("ok").is_empty());
        assert!(client.get_references("arxiv:1234.5678", 5).await.expect("ok").is_empty());
        assert!(
            client
                .get_papers_batch(&["arxiv:1234.5678".to_string()])
                .await
                .expect("ok")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn enrich_with_api_key_attaches_header() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .and(header("x-api-key", "my_key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(one_paper_response()))
            .expect(1)
            .mount(&server)
            .await;

        let mut cfg = cfg_for(&server);
        cfg.api_key = Some("my_key".into());
        let client = S2Client::with_config(cfg, true).expect("builds");
        let out = client.enrich("foo").await.expect("ok");
        assert_eq!(out.len(), 1);
        // Mock's `.expect(1)` is verified on Drop of the MockServer, so
        // unmatched header would have caused a panic.
    }

    #[tokio::test]
    async fn enrich_returns_search_error_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(429).set_body_string("Too Many Requests"))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let err = client.enrich("foo").await.expect_err("should fail");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
                let reason = d.degradation_reason.unwrap_or_default();
                assert!(
                    reason.to_lowercase().contains("rate"),
                    "expected rate reason, got: {reason}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn enrich_returns_search_error_on_500() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Error"))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let err = client.enrich("foo").await.expect_err("should fail");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
                let reason = d.degradation_reason.unwrap_or_default();
                assert!(
                    reason.contains("500") || reason.contains("S2 returned"),
                    "expected 500/'S2 returned' reason, got: {reason}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn enrich_returns_search_error_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let err = client.enrich("foo").await.expect_err("should fail");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
                let reason = d.degradation_reason.unwrap_or_default();
                assert!(
                    reason.to_lowercase().contains("auth"),
                    "expected auth reason, got: {reason}"
                );
            }
            other => panic!("expected Search error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn enrich_returns_empty_on_200_with_no_data() {
        let server = MockServer::start().await;
        let body = json!({"total": 0, "offset": 0, "data": []});
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let out = client.enrich("nothing").await.expect("ok");
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn rate_limit_enforced_between_calls() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(one_paper_response()))
            .mount(&server)
            .await;

        let mut cfg = cfg_for(&server);
        cfg.min_interval_ms = 200;
        let client = S2Client::with_config(cfg, true).expect("builds");

        let start = std::time::Instant::now();
        client.enrich("a").await.expect("ok");
        client.enrich("b").await.expect("ok");
        let elapsed = start.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(200),
            "expected >= 200ms between two calls, got {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn enrich_handles_missing_optional_fields() {
        let server = MockServer::start().await;
        let body = json!({
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": "xyz",
                    "title": "Bare Paper",
                    "authors": [],
                    "url": "https://www.semanticscholar.org/paper/xyz"
                }
            ]
        });
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let out = client.enrich("bare").await.expect("ok");
        assert_eq!(out.len(), 1);
        let p = &out[0];
        assert_eq!(p.paper_id, "xyz");
        assert_eq!(p.title, "Bare Paper");
        assert!(p.abstract_text.is_none());
        assert!(p.year.is_none());
        assert!(p.citation_count.is_none());
        assert!(p.authors.is_empty());
    }

    #[tokio::test]
    async fn from_env_with_s2_live_enabled_true_creates_enabled_client() {
        // Use a unique env-var prefix per test? std::env::set_var is process-
        // global, so we just set + cleanup. Other tests in this file do not
        // depend on S2_LIVE_ENABLED state.
        // SAFETY: Edition 2024 marks env::set_var unsafe due to potential
        // races with concurrent getenv across threads. Tests in this file
        // do not depend on S2_LIVE_ENABLED state — they set + cleanup
        // serially within a single tokio runtime.
        unsafe { std::env::set_var("S2_LIVE_ENABLED", "true") };
        let client = S2Client::from_env().expect("builds");
        assert!(client.is_enabled());
        unsafe { std::env::remove_var("S2_LIVE_ENABLED") };
    }

    #[tokio::test]
    async fn from_env_default_disabled() {
        // SAFETY: see note above on edition-2024 env mutation safety.
        unsafe { std::env::remove_var("S2_LIVE_ENABLED") };
        let client = S2Client::from_env().expect("builds");
        assert!(!client.is_enabled());
    }

    // ── New graph endpoint tests ──────────────────────────────────────────────

    fn full_paper_fixture(id: &str, arxiv_id: Option<&str>) -> serde_json::Value {
        let mut external_ids = serde_json::json!({});
        if let Some(aid) = arxiv_id {
            external_ids["ArXiv"] = serde_json::Value::String(aid.to_string());
        }
        json!({
            "paperId": id,
            "title": format!("Paper {id}"),
            "abstract": format!("Abstract for {id}"),
            "year": 2022,
            "citationCount": 42,
            "referenceCount": 10,
            "authors": [{"name": "Author A"}],
            "venue": "NeurIPS",
            "externalIds": external_ids
        })
    }

    #[tokio::test]
    async fn get_paper_returns_full_paper() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/ARXIV:1706.03762"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                full_paper_fixture("abc123", Some("1706.03762"))
            ))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let paper = client.get_paper("1706.03762").await.expect("ok").expect("some");
        assert_eq!(paper.s2_id, "abc123");
        assert_eq!(paper.arxiv_id, Some("1706.03762".into()));
        assert_eq!(paper.citation_count, 42);
        assert_eq!(paper.reference_count, 10);
    }

    #[tokio::test]
    async fn get_paper_404_returns_none() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/ARXIV:9999.00001"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Not Found"))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let result = client.get_paper("9999.00001").await.expect("ok");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn arxiv_version_suffix_stripped() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper/ARXIV:2105.14103"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                full_paper_fixture("def456", Some("2105.14103v2"))
            ))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let paper = client.get_paper("2105.14103").await.expect("ok").expect("some");
        assert_eq!(paper.arxiv_id, Some("2105.14103".into()), "version suffix must be stripped");
    }

    #[tokio::test]
    async fn get_citations_parses_citing_paper() {
        let server = MockServer::start().await;
        let body = json!({
            "data": [
                {"citingPaper": full_paper_fixture("cite001", None)},
                {"citingPaper": {"paperId": null, "title": "No ID"}},
                {"citingPaper": full_paper_fixture("cite002", Some("2001.00001"))}
            ]
        });
        Mock::given(method("GET"))
            .and(path("/paper/ARXIV:1706.03762/citations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let papers = client.get_citations("1706.03762", 100).await.expect("ok");
        assert_eq!(papers.len(), 2, "entry without paperId must be dropped");
        assert_eq!(papers[0].s2_id, "cite001");
        assert_eq!(papers[1].s2_id, "cite002");
    }

    #[tokio::test]
    async fn get_references_parses_cited_paper() {
        let server = MockServer::start().await;
        let body = json!({
            "data": [
                {"citedPaper": full_paper_fixture("ref001", Some("1801.00001"))},
                {"citedPaper": {"paperId": null, "title": "No ID"}}
            ]
        });
        Mock::given(method("GET"))
            .and(path("/paper/ARXIV:2105.14103/references"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let papers = client.get_references("2105.14103", 100).await.expect("ok");
        assert_eq!(papers.len(), 1, "entry without paperId must be dropped");
        assert_eq!(papers[0].s2_id, "ref001");
        assert_eq!(papers[0].arxiv_id, Some("1801.00001".into()));
    }

    #[tokio::test]
    async fn get_papers_batch_preserves_order_with_none_gaps() {
        let server = MockServer::start().await;
        // S2 batch returns results in request order; nulls for not-found.
        let body = json!([
            full_paper_fixture("batch001", None),
            serde_json::Value::Null,
            full_paper_fixture("batch003", None)
        ]);
        Mock::given(method("POST"))
            .and(path("/paper/batch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let ids: Vec<String> = vec!["batch001".into(), "batch002".into(), "batch003".into()];
        let results = client.get_papers_batch(&ids).await.expect("ok");
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_none(), "null entry must map to None");
        assert!(results[2].is_some());
        assert_eq!(results[0].as_ref().unwrap().s2_id, "batch001");
        assert_eq!(results[2].as_ref().unwrap().s2_id, "batch003");
    }

    #[tokio::test]
    async fn resolve_paper_id_arxiv_format() {
        assert_eq!(resolve_paper_id("1706.03762"), "ARXIV:1706.03762");
        assert_eq!(resolve_paper_id("1706.03762v7"), "ARXIV:1706.03762");
        assert_eq!(resolve_paper_id("S2:abc123"), "abc123");
        assert_eq!(resolve_paper_id("ARXIV:2105.14103"), "ARXIV:2105.14103");
        // Opaque S2 hex id passes through.
        assert_eq!(resolve_paper_id("649def34f8be52c8b66281af98ae884c09aef38b"), "649def34f8be52c8b66281af98ae884c09aef38b");
    }

    /// Verify S2_API_KEY flows from process env through S2Client::from_env()
    /// into config.api_key.
    ///
    /// No network call is made — this is a configuration wiring check only.
    ///
    /// **Security note (T-25-03):** The assertion message intentionally does NOT
    /// include the key value — only its presence is checked. The key must never
    /// appear in test output, logs, or committed fixtures.
    ///
    /// **No dotenv loader:** There is NO dotenv loader in any crate under
    /// crates/ — a key sitting in `.env` does not reach the daemon unless the
    /// launching shell exports it. The key reaches S2Client only via
    /// `std::env::var("S2_API_KEY")` from the PROCESS env.
    ///
    /// Run with: `S2_API_KEY=<your-key> cargo test -p alzina-search --lib -- s2_api_key_flows_to_client --ignored`
    #[test]
    #[ignore = "requires S2_API_KEY exported in the shell env — run manually: cargo test -p alzina-search --lib -- s2_api_key_flows_to_client --ignored"]
    fn s2_api_key_flows_to_client() {
        let client = S2Client::from_env().expect("S2Client::from_env must succeed");
        // Assert presence only — never include the key value in the assertion message
        // (information-disclosure guard T-25-03).
        assert!(
            client.config.api_key.is_some(),
            "S2_API_KEY env var must be present and non-empty for this test to pass — \
             key value is intentionally omitted from this message"
        );
    }

    // ── F10: openAccessPdf wire deserialization tests ─────────────────────────

    /// Three wire shapes for openAccessPdf: object with url, JSON null, absent.
    /// All three must deserialize without error and map correctly to Option<String>.
    #[test]
    fn open_access_pdf_wire_shapes() {
        // Shape 1: object with url — must yield Some(url).
        let with_url: WireS2Paper = serde_json::from_value(json!({
            "paperId": "p1",
            "authors": [],
            "openAccessPdf": {"url": "https://example.com/paper.pdf", "status": "GOLD"}
        })).expect("deserializes");
        assert_eq!(
            with_url.open_access_pdf.as_ref().and_then(|o| o.url.as_deref()),
            Some("https://example.com/paper.pdf"),
            "object with url: must yield the url"
        );

        // Shape 2: JSON null — must yield None (Option<WireS2OpenAccessPdf>).
        let with_null: WireS2Paper = serde_json::from_value(json!({
            "paperId": "p2",
            "authors": [],
            "openAccessPdf": null
        })).expect("deserializes");
        assert!(
            with_null.open_access_pdf.is_none(),
            "JSON null: open_access_pdf must be None"
        );

        // Shape 3: field absent — must yield None.
        let absent: WireS2Paper = serde_json::from_value(json!({
            "paperId": "p3",
            "authors": []
        })).expect("deserializes");
        assert!(
            absent.open_access_pdf.is_none(),
            "absent field: open_access_pdf must be None"
        );
    }

    /// S2PaperFull serde round-trip back-compat: old JSON WITHOUT open_access_pdf_url
    /// must still deserialize (the s2_cache payload was written before F10).
    #[test]
    fn s2_paper_full_cache_back_compat() {
        let old_json = r#"{
            "s2_id": "abc123",
            "arxiv_id": null,
            "title": "Old Paper",
            "abstract_text": "Some abstract.",
            "year": 2020,
            "citation_count": 42,
            "reference_count": 10,
            "authors": ["Alice"],
            "venue": null,
            "doi": null
        }"#;
        let p: S2PaperFull = serde_json::from_str(old_json)
            .expect("old s2_cache payload must deserialize without open_access_pdf_url");
        assert_eq!(p.s2_id, "abc123");
        assert!(
            p.open_access_pdf_url.is_none(),
            "missing field must default to None"
        );
    }

    /// S2Result enrich() response: openAccessPdf object propagates to open_access_pdf_url.
    #[tokio::test]
    async fn enrich_propagates_open_access_pdf_url() {
        let server = MockServer::start().await;
        let body = json!({
            "total": 1,
            "offset": 0,
            "data": [{
                "paperId": "oa001",
                "title": "OA Paper",
                "authors": [],
                "url": "https://www.semanticscholar.org/paper/oa001",
                "openAccessPdf": {"url": "https://arxiv.org/pdf/2001.00001.pdf", "status": "GREEN"}
            }]
        });
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let results = client.enrich("open access").await.expect("ok");
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].open_access_pdf_url.as_deref(),
            Some("https://arxiv.org/pdf/2001.00001.pdf"),
            "enrich must propagate openAccessPdf url to S2Result"
        );
    }

    /// S2Result enrich() response: null openAccessPdf yields None.
    #[tokio::test]
    async fn enrich_null_open_access_pdf_yields_none() {
        let server = MockServer::start().await;
        let body = json!({
            "total": 1,
            "offset": 0,
            "data": [{
                "paperId": "closed001",
                "title": "Closed Access Paper",
                "authors": [],
                "url": "https://www.semanticscholar.org/paper/closed001",
                "openAccessPdf": null
            }]
        });
        Mock::given(method("GET"))
            .and(path("/paper/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let client = S2Client::with_config(cfg_for(&server), true).expect("builds");
        let results = client.enrich("closed").await.expect("ok");
        assert_eq!(results.len(), 1);
        assert!(
            results[0].open_access_pdf_url.is_none(),
            "null openAccessPdf must yield None"
        );
    }
}
