//! Bibliography accumulation store for the TTD synthesis engine.
//!
//! Phase 24 EXT-03: externalises the accumulating bibliography to the
//! literature KB across denoise iterations, deduped, observable in the store.
//! Write path is direct parameterised SQLite — NOT threaded through composition
//! channels (CONTEXT EXT-03, locked).
//!
//! ## Dedup layers
//!
//! Both consensus dedup layers are reproduced in `normalise_source_id` and
//! `normalise_quote`, and are enforced by the table's
//! `UNIQUE(run_id, source_id, expert_id, quote_normalised)` constraint with
//! `INSERT OR IGNORE` (accumulate-and-dedup-by-silence):
//!
//! - **Layer 1:** source_id compound suffix strip — "arxiv:2105_c001" →
//!   "arxiv:2105" (`domain/utils.py:28`, `runner.py:594-599`)
//! - **Layer 2:** expert_id + 200-char-truncated lowercase quote
//!   (`synthesis_tasks.py:945-948`, `deduplicate_sources`)

use async_trait::async_trait;
use sqlx::sqlite::SqlitePool;

use base::error::{AlzinaError, AlzinaResult, SearchDetail};

// ── BibEntry ──────────────────────────────────────────────────────────────────

/// One bibliography entry to record.
///
/// `quote_normalised` is computed at write time from `quote_raw`; it is not
/// stored on the struct to keep the caller interface clean.
#[derive(Debug, Clone)]
pub struct BibEntry {
    /// Raw source identifier, e.g. "arxiv:2105_c001" or "s2:abc123".
    /// Compound graph-node suffixes (_c001, _c002) are stripped at write time.
    pub source_id: String,
    /// Expert / agent identifier string.
    pub expert_id: String,
    /// Raw quote text, if any. May be None when the source is cited without
    /// a direct quote.
    pub quote_raw: Option<String>,
}

// ── BibliographyStore trait ───────────────────────────────────────────────────

/// Accumulate bibliography entries for one TTD run.
///
/// Injected into `TtdMachine<A>` alongside `self.retriever`.
/// The write path is direct SQLite — NOT composition channels (CONTEXT EXT-03).
///
/// `NoopBibliographyStore` is used for tests and stages without bibliography
/// tracking. `SqliteBibliographyStore` is used in production.
#[async_trait]
pub trait BibliographyStore: Send + Sync {
    /// Record a slice of bibliography entries for one denoise step.
    ///
    /// `run_id` uniquely identifies the TTD run.
    /// `stage` is the stage name ("graph", "synthesis", "narrative").
    /// `step` is the zero-based denoise step index.
    /// `sources` is the slice of entries to persist.
    ///
    /// Entries that would violate the
    /// `UNIQUE(run_id, source_id, expert_id, quote_normalised)` constraint are
    /// silently ignored (`INSERT OR IGNORE`) — this is the accumulate-and-
    /// dedup-by-silence contract.
    async fn record_sources(
        &self,
        run_id: &str,
        stage: &str,
        step: usize,
        sources: &[BibEntry],
    ) -> AlzinaResult<()>;
}

// ── NoopBibliographyStore ────────────────────────────────────────────────────

/// No-op implementation — for tests and stages without bibliography tracking.
///
/// Mirrors `NoopRetriever` in `retrieval.rs`. Enables Plan 24-04 tests and
/// non-bibliography stages to run without a DB pool.
pub struct NoopBibliographyStore;

#[async_trait]
impl BibliographyStore for NoopBibliographyStore {
    async fn record_sources(
        &self,
        _run_id: &str,
        _stage: &str,
        _step: usize,
        _sources: &[BibEntry],
    ) -> AlzinaResult<()> {
        Ok(())
    }
}

// ── SqliteBibliographyStore ───────────────────────────────────────────────────

/// SQLite-backed bibliography store.
///
/// Uses parameterised `.bind()` SQL with `INSERT OR IGNORE` — no value is
/// interpolated into the SQL string (T-24-02-I: no injection surface, same
/// pattern as `upsert_paper` in `lit_schema.rs:141-189`).
pub struct SqliteBibliographyStore {
    pool: SqlitePool,
}

impl SqliteBibliographyStore {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl BibliographyStore for SqliteBibliographyStore {
    async fn record_sources(
        &self,
        run_id: &str,
        stage: &str,
        step: usize,
        sources: &[BibEntry],
    ) -> AlzinaResult<()> {
        for entry in sources {
            // Layer 1: strip compound graph-node suffix (domain/utils.py:28)
            let base_id = normalise_source_id(&entry.source_id);
            // Layer 2: 200-char truncated lowercase quote (domain/utils.py:29)
            let quote_norm = normalise_quote(entry.quote_raw.as_deref().unwrap_or(""));
            let added_at = chrono::Utc::now().to_rfc3339();

            sqlx::query(
                "INSERT OR IGNORE INTO synthesis_bibliography \
                 (run_id, source_id, expert_id, quote_normalised, quote_raw, \
                  stage, denoise_step, added_at) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(run_id)
            .bind(&base_id)
            .bind(&entry.expert_id)
            .bind(&quote_norm)
            .bind(&entry.quote_raw)
            .bind(stage)
            .bind(step as i64)
            .bind(&added_at)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("bib_store record_sources: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!("bib insert: {e}")),
                })
            })?;
        }
        Ok(())
    }
}

// ── Normalisation helpers ─────────────────────────────────────────────────────

/// Strip graph-node compound suffixes from a source_id.
///
/// "arxiv:2105_c001" → "arxiv:2105"
/// "s2:abc_c002"     → "s2:abc"
/// "arxiv:2105"      → "arxiv:2105" (unchanged)
///
/// Faithful port of `consensus/domain/utils.py:28`:
///   `s.source_id.split("_c")[0]`
pub(crate) fn normalise_source_id(source_id: &str) -> String {
    if let Some(idx) = source_id.find("_c") {
        source_id[..idx].to_string()
    } else {
        source_id.to_string()
    }
}

/// Normalise a quote for dedup: trim, lowercase, first 200 chars.
///
/// Faithful port of `consensus/domain/utils.py:29`:
///   `(s.quote or "").strip().lower()[:200]`
pub(crate) fn normalise_quote(quote: &str) -> String {
    quote.trim().to_lowercase().chars().take(200).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lit_schema::in_memory_lit_pool;

    #[tokio::test]
    async fn compound_id_deduplicates_to_one_row() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteBibliographyStore::new(pool.clone());

        // Two entries with compound IDs for the same paper, same expert, same
        // quote. Layer 1 normalisation strips _c suffix → same base_id →
        // second INSERT OR IGNORE is silently dropped.
        store
            .record_sources(
                "run-1",
                "graph",
                0,
                &[
                    BibEntry {
                        source_id: "arxiv:2105_c001".into(),
                        expert_id: "e1".into(),
                        quote_raw: None,
                    },
                    BibEntry {
                        source_id: "arxiv:2105_c002".into(),
                        expert_id: "e1".into(),
                        quote_raw: None,
                    },
                ],
            )
            .await
            .unwrap();

        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM synthesis_bibliography WHERE run_id='run-1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(count.0, 1, "compound IDs for same paper must dedup to one row");
    }

    #[tokio::test]
    async fn same_expert_same_quote_deduplicates() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteBibliographyStore::new(pool.clone());

        // Same source_id, expert_id, and quote inserted twice. Layer 2
        // normalisation yields the same quote_normalised → second INSERT OR
        // IGNORE is silently dropped.
        let entry = BibEntry {
            source_id: "arxiv:2105".into(),
            expert_id: "e1".into(),
            quote_raw: Some("The result shows improvement.".into()),
        };
        store
            .record_sources("run-2", "graph", 0, &[entry.clone()])
            .await
            .unwrap();
        store
            .record_sources("run-2", "graph", 1, &[entry])
            .await
            .unwrap();

        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM synthesis_bibliography WHERE run_id='run-2'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(count.0, 1, "same expert + same quote must dedup to one row");
    }

    #[tokio::test]
    async fn accumulates_across_steps() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteBibliographyStore::new(pool.clone());

        // Same source_id and expert_id but DIFFERENT quotes at step 0 and 1.
        // quote_normalised differs → UNIQUE constraint does not collapse them
        // → two distinct rows accumulate (denoise-step accumulation).
        store
            .record_sources(
                "run-3",
                "graph",
                0,
                &[BibEntry {
                    source_id: "arxiv:2105".into(),
                    expert_id: "e1".into(),
                    quote_raw: Some("First quote from the paper.".into()),
                }],
            )
            .await
            .unwrap();
        store
            .record_sources(
                "run-3",
                "graph",
                1,
                &[BibEntry {
                    source_id: "arxiv:2105".into(),
                    expert_id: "e1".into(),
                    quote_raw: Some("Second quote from the paper.".into()),
                }],
            )
            .await
            .unwrap();

        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM synthesis_bibliography WHERE run_id='run-3'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(count.0, 2, "different quotes at different steps must accumulate");
    }

    // Unit tests for normalisation helpers (no DB needed)

    #[test]
    fn normalise_source_id_strips_compound_suffix() {
        assert_eq!(normalise_source_id("arxiv:2105_c001"), "arxiv:2105");
        assert_eq!(normalise_source_id("arxiv:2105_c002"), "arxiv:2105");
        assert_eq!(normalise_source_id("s2:abc_c007"), "s2:abc");
        assert_eq!(normalise_source_id("arxiv:2105"), "arxiv:2105");
        assert_eq!(normalise_source_id("s2:abc123"), "s2:abc123");
    }

    #[test]
    fn normalise_quote_trims_lowercases_and_truncates() {
        // Basic trim + lowercase
        assert_eq!(normalise_quote("  Hello World  "), "hello world");
        // 200-char truncation
        let long = "x".repeat(300);
        assert_eq!(normalise_quote(&long).len(), 200);
        // Empty string
        assert_eq!(normalise_quote(""), "");
    }
}
