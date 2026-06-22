//! Source → panel adapter.
//!
//! Maps Phase 21 retrieval output (`Vec<FusedHit>`) into the `ExpertResponse`
//! shape the consensus extraction stage consumes.
//!
//! ## Key invariant
//!
//! One paper = one `ExpertResponse`. `expert_id` == `provenance.source_id` ==
//! `paper_id` verbatim. Any transform breaks the citation chain (graph_tasks.py
//! namespaces node IDs as `{expert_id}_{node_id}` and the quote-verification map
//! keys on `expert_id`).

use std::collections::{HashMap, HashSet};

use search::lit_fusion::FusedHit;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced by the source → panel adapter.
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    /// Attempted to construct a `SourceId` from an empty string.
    #[error("empty source_id")]
    EmptySourceId,

    /// Source-id conservation violation (loss): the adapter produced fewer
    /// distinct `expert_id` values than the input had distinct `paper_id`
    /// values — at least one paper never reached the output.
    ///
    /// Wired in Plan 22-02 via `conservation_assert`.
    #[error("source_id loss: expected {expected} distinct ids, got {actual}")]
    SourceIdLoss { expected: usize, actual: usize },

    /// Source-id conservation violation (duplication): the adapter emitted more
    /// `ExpertResponse` rows (`emitted`) than there are distinct `expert_id`
    /// values (`actual` == `expected` distinct papers). Two rows share one
    /// `expert_id`, inflating `panel_size` while the distinct-id count stays
    /// correct. `emitted` is the field that proves duplication.
    ///
    /// Wired in Plan 22-02 via `conservation_assert`.
    #[error(
        "source_id duplication: {emitted} responses for {actual} distinct ids (expected {expected})"
    )]
    SourceIdDuplication {
        expected: usize,
        actual: usize,
        emitted: usize,
    },

    /// SQLite error from a lit_chunks / papers query.
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),

    /// JSON decode error (authors column).
    #[error("json decode error: {0}")]
    Json(#[from] serde_json::Error),
}

// ── SourceId newtype ──────────────────────────────────────────────────────────

/// Non-empty paper identifier: `"arxiv:…"` or `"s2:…"`.
///
/// Invariant: the inner string is never empty. Validated at construction time.
/// Do not transform the value — `expert_id` must equal the `paper_id` verbatim
/// for quote verification to succeed in the downstream extraction stage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceId(String);

impl SourceId {
    /// Fallible constructor — rejects empty strings with `AdapterError::EmptySourceId`.
    pub fn try_new(s: impl Into<String>) -> Result<Self, AdapterError> {
        let s = s.into();
        if s.is_empty() {
            return Err(AdapterError::EmptySourceId);
        }
        Ok(Self(s))
    }

    /// Panicking convenience constructor.
    pub fn new(s: impl Into<String>) -> Self {
        Self::try_new(s).expect("invalid SourceId")
    }

    /// Returns the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── ResponseProvenance ────────────────────────────────────────────────────────

/// Provenance carried from retrieval to claim citation.
///
/// Populated from the `papers` table. If the row is absent (e.g. a paper
/// indexed via the S2 lane before the papers row lands), fields fall back
/// to `FusedHit` values / empty.
#[derive(Debug, Clone)]
pub struct ResponseProvenance {
    /// Paper identifier — equals `ExpertResponse.expert_id`.
    pub source_id: SourceId,
    /// Paper title from the `papers` table, or `FusedHit.title` on fallback.
    pub title: String,
    /// Publication year from the `papers` table; `None` if unknown.
    pub year: Option<i32>,
    /// Author list decoded from the JSON `papers.authors` column; empty on fallback.
    pub authors: Vec<String>,
    /// Mechanical per-source credibility tier, derived once here from the paper's
    /// S2 signals (citations / influential citations / venue). Rides the golden
    /// thread (`source_id`) forward so the tier is visible wherever the paper is
    /// rendered into a model's context — a soft filter, not a hard gate.
    /// `Unknown` when no signal was fetched (e.g. papers row absent).
    pub credibility_tier: search::CredibilityTier,
}

// ── ExpertResponse ────────────────────────────────────────────────────────────

/// One paper represented as a single expert voice.
///
/// Satisfies the consensus `ProseResponse` duck type:
/// - `expert_id` → `expert_id: str` (node-ID namespace key in graph_tasks.py)
/// - `prose`     → `response: str`  (the text `extraction_single.mustache` reads)
///
/// ## Identity invariant
///
/// `expert_id` MUST equal `provenance.source_id` MUST equal the `paper_id`
/// verbatim (`"arxiv:…"` / `"s2:…"`). Any transform causes quote verification
/// to stamp every quote `"absent"` and collapses groundedness scores.
///
/// ## Trust boundary
///
/// Retrieved paper text lands only in the `prose` data field. Do not splice
/// it into instruction position — the consensus mustache template treats
/// `response` as quoted source material.
#[derive(Debug, Clone)]
pub struct ExpertResponse {
    /// Paper identifier — verbatim `paper_id`; equals `provenance.source_id`.
    pub expert_id: SourceId,
    /// Full prose body: all chunks concatenated in `chunk_index` order with
    /// arxiv section headings as `## {section}` inline markers.
    ///
    /// Falls back to `FusedHit.content` when no `lit_chunks` rows exist
    /// (S2 abstract-only path).
    pub prose: String,
    /// Typed provenance — source_id, title, year, authors.
    pub provenance: ResponseProvenance,
}

// ── build_panel ───────────────────────────────────────────────────────────────

/// Map a list of fused retrieval hits into a panel of expert responses.
///
/// One `ExpertResponse` is produced per distinct `paper_id` in `hits`.
/// First-seen order is preserved.
///
/// ## Prose assembly
///
/// For each paper, all `lit_chunks` rows are fetched in `chunk_index` order
/// and concatenated with `"\n\n"`. Chunks with a non-empty `section` field are
/// prefixed with `"## {section}\n\n"`.
///
/// If no `lit_chunks` rows exist for a paper (S2 abstract-only), the adapter
/// falls back to `FusedHit.content` and logs at `tracing::debug!` level.
/// The paper is never dropped.
///
/// ## Panics
///
/// Does not panic. Returns `Err(AdapterError::EmptySourceId)` if any hit's
/// `source_id` is empty. Returns `Err(AdapterError::Db)` on SQLite error.
pub async fn build_panel(
    hits: Vec<FusedHit>,
    lit_pool: &SqlitePool,
) -> Result<Vec<ExpertResponse>, AdapterError> {
    // Collect distinct input paper_ids BEFORE consuming hits.
    // conservation_assert compares this set against output expert_ids.
    let input_ids: HashSet<String> = hits.iter().map(|h| h.source_id.clone()).collect();

    // Group hits by source_id, preserving first-seen order.
    let mut order: Vec<String> = Vec::new();
    let mut hit_map: HashMap<String, FusedHit> = HashMap::new();

    for hit in hits {
        let id = hit.source_id.clone();
        if !hit_map.contains_key(&id) {
            order.push(id.clone());
            hit_map.insert(id, hit);
        }
    }

    let mut responses = Vec::with_capacity(order.len());

    for paper_id in &order {
        let hit = &hit_map[paper_id];

        // Validate source_id is non-empty.
        let source_id = SourceId::try_new(paper_id.clone())?;

        // -- Step 1: fetch all chunks in chunk_index order --
        let rows: Vec<(i64, Option<String>, Option<String>)> = sqlx::query_as(
            "SELECT chunk_index, section, content \
             FROM lit_chunks \
             WHERE paper_id = ? \
             ORDER BY chunk_index ASC",
        )
        .bind(paper_id.as_str())
        .fetch_all(lit_pool)
        .await?;

        // -- Step 2: assemble prose body --
        let prose = if rows.is_empty() {
            // S2 / abstract-only fallback: no lit_chunks rows; use FusedHit.content.
            tracing::debug!(
                paper_id = paper_id.as_str(),
                "no lit_chunks rows; falling back to FusedHit.content (S2 abstract path)"
            );
            hit.content.clone()
        } else {
            let assembled = rows
                .iter()
                .map(|(_, section, content)| {
                    let text = content.as_deref().unwrap_or("");
                    match section.as_deref() {
                        Some(s) if !s.is_empty() => format!("## {s}\n\n{text}"),
                        _ => text.to_string(),
                    }
                })
                .collect::<Vec<_>>()
                .join("\n\n");

            // WR-04: rows can exist with NULL/empty `content` (the column is
            // nullable).  In that case `assembled` is heading-only or just blank
            // `"\n\n"` joins — a degenerate prose body that would stamp every
            // downstream quote absent.  Treat an assembled-but-empty body as a
            // fallback trigger, same as the no-rows path.
            if assembled.trim().is_empty() {
                tracing::debug!(
                    paper_id = paper_id.as_str(),
                    "lit_chunks present but content empty; falling back to FusedHit.content"
                );
                hit.content.clone()
            } else {
                assembled
            }
        };

        // -- Step 3: fetch provenance + credibility signals from papers table --
        let prov_row: Option<(String, Option<i32>, String, Option<i64>, Option<i64>, Option<String>)> =
            sqlx::query_as(
                "SELECT title, year, authors, citation_count, influential_citation_count, venue \
                 FROM papers WHERE paper_id = ?",
            )
            .bind(paper_id.as_str())
            .fetch_optional(lit_pool)
            .await?;

        let (title, year, authors, credibility_tier) = match prov_row {
            Some((title, year, authors_json, citation_count, influential, venue)) => {
                let authors: Vec<String> = serde_json::from_str(&authors_json)?;
                let tier = search::derive_tier(
                    citation_count,
                    influential,
                    venue.as_deref(),
                    &search::TierThresholds::default(),
                );
                (title, year, authors, tier)
            }
            None => {
                // Papers row absent — fall back to FusedHit.title; year/authors empty.
                // No signals → Unknown tier (honest: we never fetched credibility).
                tracing::debug!(
                    paper_id = paper_id.as_str(),
                    "no papers row found; using FusedHit.title as provenance fallback"
                );
                (hit.title.clone(), None, Vec::new(), search::CredibilityTier::Unknown)
            }
        };

        let provenance = ResponseProvenance {
            source_id: source_id.clone(),
            title,
            year,
            authors,
            credibility_tier,
        };

        responses.push(ExpertResponse {
            expert_id: source_id,
            prose,
            provenance,
        });
    }

    // Assert conservation before returning: distinct expert_ids leaving must
    // equal distinct paper_ids entering.  Fires loudly (tracing::error! + Err)
    // on both loss and duplication — either corrupts panel_size and downstream
    // agreement labels (Pitfall 3: assertion must be at this boundary, not Phase 25).
    conservation::conservation_assert(&input_ids, &responses)?;

    Ok(responses)
}

pub(crate) mod conservation;

#[cfg(test)]
mod tests;
