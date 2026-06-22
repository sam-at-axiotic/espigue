//! Retrieval seam for the TTD denoise loop.
//!
//! The `Retriever` trait abstracts the per-gap query → dedup-by-source_id
//! result contract. Stage 3 (narrative) uses `NoopRetriever` — it has no
//! retrieval step (gaps are checked against the fixed synthesis, not the lit
//! store).
//!
//! ## Backend decision (23-BACKEND-DECISION.md)
//!
//! 10k-scale benchmark: p50=38ms, p95=43ms under N=5 fan-out (top_k=25).
//! Decision: KEEP sqlite-vec (canonical, no Qdrant infrastructure needed).
//! The lit-store entry point is `search::lit_fusion::gate_fused` /
//! `fuse_rrf_three_lane` from Phase 21 Plan 04.
//!
//! ## Dedup contract
//!
//! Retrieved results are deduped by `source_id` before returning, mirroring
//! `runner.py:492-699` (dedup by source_id across per-gap retrieval results).
//! The `LitRetriever` implementation does this before returning.

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use crate::ttd::mod_types::TtdError;
use crate::ttd::stages::RetrievedContext;

// ── RetrievalPolicy ───────────────────────────────────────────────────────────

/// Stage-scoped retrieval policy (sketch section D).
///
/// Controls which search lanes are active during a TTD gap-fill pass:
///
/// - `Live` — stage-1 (graph) gap filling only. Runs the full three-lane
///   fusion: arxiv search, S2 search, and the internal hybrid store.
///   Exploration stays live; the corpus grows as new papers are found.
///
/// - `LocalOnly` — stages 2-3. Runs the internal hybrid lane only; arxiv and
///   S2 search lanes are scoped out by policy (not by failure). Retrieval
///   itself still runs — the corpus built by stages 0-1, including any
///   full texts promoted by A2's background promotion, is queried.
///
/// Policy scoping is NOT reported as degradation — it is intentional budget
/// management. A loud `ttd_perf` log line is emitted per `LocalOnly` call so
/// the lane decision is always visible in the daemon log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalPolicy {
    /// Full three-lane fusion active (arxiv + S2 + internal). Used for
    /// Stage 1 (graph) gap filling and the initial corpus-building query.
    Live,
    /// Internal lane only; live arxiv/S2 search lanes scoped out by policy.
    /// Used for Stage 2 (synthesis) gap filling.
    LocalOnly,
}

// ── Type alias ────────────────────────────────────────────────────────────────

/// Boxed async search callback type.
///
/// The daemon (Plan 03) supplies a closure of this shape wrapping its
/// three-lane RRF fusion path. The closure converts `FusedHit → RetrievedContext`
/// before returning so the orchestration crate never touches daemon types.
type SearchCallback = Box<
    dyn Fn(
            String,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<RetrievedContext>, TtdError>> + Send>>
        + Send
        + Sync,
>;

// ── Retriever trait ───────────────────────────────────────────────────────────

/// Per-gap query → deduped results retrieval contract.
///
/// Injected into `TtdMachine<A>`. Stage 3 receives `NoopRetriever`.
/// Stages 1 and 2 receive the `LitRetriever` (lit-store wrapper).
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Issue one retrieval query for a gap and return deduped results.
    ///
    /// The `top_k` is taken from `TtdConfig.retrieval_top_k` (default: 25).
    ///
    /// Returns an empty vec when the lit store returns no results — the caller
    /// applies the empty-retrieved guard (return draft unchanged).
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievedContext>, TtdError>;
}

// ── NoopRetriever ─────────────────────────────────────────────────────────────

/// Always returns empty — used for Stage 3 (narrative) which has no retrieval.
///
/// Stage 3 gaps are checked against the fixed synthesis artifact, not the
/// lit store. The `NoopRetriever` keeps `TtdMachine<A>` interface uniform
/// across all three stages (CONTEXT Open Question 3, resolved).
pub struct NoopRetriever;

#[async_trait]
impl Retriever for NoopRetriever {
    async fn retrieve(
        &self,
        _query: &str,
        _top_k: usize,
    ) -> Result<Vec<RetrievedContext>, TtdError> {
        Ok(vec![])
    }
}

// ── LitRetriever ─────────────────────────────────────────────────────────────

/// Wraps the Phase 21 three-lane RRF lit-store retrieval surface.
///
/// Uses `search::lit_fusion::gate_fused` / `fuse_rrf_three_lane` as
/// the retrieval entry point. Results are deduped by `source_id` before
/// returning (mirrors runner.py:492-699 dedup-by-source_id contract).
///
/// ## Trust boundary (T-23-01)
///
/// Retrieved paper text arrives as `RetrievedContext.content` — data position
/// only. The TTD engine never interpolates retrieved text into instruction
/// prompts; it only passes it to gap_resolve as quoted source material.
///
/// This implementation lives here as a type stub for Wave 0. The full wiring
/// (actual embedding query + lit pool access) is completed in Wave 1
/// (Plan 23-02, GraphGapResolve) when the daemon's lit pool is injected.
///
/// Wave 0 tests use `NoopRetriever` exclusively — this struct compiles but
/// is not tested end-to-end in Wave 0.
pub struct LitRetriever {
    /// Shared reference to the internal hybrid lit store (sqlite-vec at 10k scale).
    /// Populated with the embed-then-search path from Phase 21 Plan 04.
    inner: Arc<dyn LitSearch>,
}

impl LitRetriever {
    pub fn new(inner: Arc<dyn LitSearch>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Retriever for LitRetriever {
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievedContext>, TtdError> {
        let hits = self.inner.search(query, top_k).await?;

        // Dedup by source_id — mirrors runner.py:492-699 dedup contract.
        let mut seen = HashSet::new();
        let deduped: Vec<RetrievedContext> = hits
            .into_iter()
            .filter(|h| seen.insert(h.source_id.clone()))
            .collect();

        Ok(deduped)
    }
}

// ── ArcRetriever newtype ──────────────────────────────────────────────────────

/// Thin newtype that allows sharing a single `Arc<dyn Retriever>` across the
/// two retrieval-stage builders (graph + synthesis) without changing
/// `TtdMachine.retriever`'s type from `Box<dyn Retriever>` (DISP-02, D-04).
///
/// Each stage builder constructs `Box::new(ArcRetriever(config.retriever.clone()))`,
/// giving each stage its own `Box` over the same shared inner `Arc`.  This is
/// strictly additive: `TtdMachine.retriever` stays `Box<dyn Retriever>`; no
/// field type changes in `mod.rs` or `run.rs`.
pub struct ArcRetriever(pub Arc<dyn Retriever>);

#[async_trait]
impl Retriever for ArcRetriever {
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievedContext>, TtdError> {
        self.0.retrieve(query, top_k).await
    }
}

// ── LitSearch trait (internal seam) ──────────────────────────────────────────

/// Internal seam for the lit-store search. Implemented by the daemon's Phase 21
/// three-lane RRF fusion path. Exposed here so `LitRetriever` can be tested
/// with a stub.
#[async_trait]
pub trait LitSearch: Send + Sync {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievedContext>, TtdError>;
}

// ── CallbackLitSearch ─────────────────────────────────────────────────────────

/// Production `LitSearch` impl backed by an async search callback.
///
/// This is the orchestration-side seam (DISP-01). The daemon (Plan 03) supplies
/// an async closure that wraps its three-lane RRF fusion path
/// (`fuse_rrf_three_lane` + `gate_fused`) and converts `FusedHit →
/// RetrievedContext` inside the closure.  The orchestration crate therefore
/// never imports daemon or `search` fusion types — the trust boundary
/// stays clean: retrieved paper text only ever lands in
/// `RetrievedContext.content` (data position, T-24.5-01).
///
/// ## Construction
///
/// Use `CallbackLitSearch::new(f)` where `f` is any async-returning `Fn`:
///
/// ```rust,ignore
/// let search = CallbackLitSearch::new(|query, top_k| async move {
///     // daemon-side fusion logic here
///     Ok(vec![])
/// });
/// let retriever = LitRetriever::new(Arc::new(search));
/// ```
pub struct CallbackLitSearch {
    cb: SearchCallback,
}

impl CallbackLitSearch {
    /// Construct a `CallbackLitSearch` from any async-returning `Fn`.
    ///
    /// The future is boxed internally so the stored field is a uniform
    /// `SearchCallback`. Callers do not need to box the future themselves.
    pub fn new<F, Fut>(f: F) -> Self
    where
        F: Fn(String, usize) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<RetrievedContext>, TtdError>> + Send + 'static,
    {
        Self {
            cb: Box::new(move |query, top_k| Box::pin(f(query, top_k))),
        }
    }
}

#[async_trait]
impl LitSearch for CallbackLitSearch {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievedContext>, TtdError> {
        (self.cb)(query.to_string(), top_k).await
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Task 1 failing tests (TDD RED) ───────────────────────────────────────
    // These tests exercise CallbackLitSearch which does not yet exist.

    /// CallbackLitSearch returns the two hits supplied by the async closure,
    /// deduplicated by LitRetriever (same source_id collapses to one).
    ///
    /// DISP-01: production LitSearch seam — callback-backed impl wrappable
    /// by LitRetriever.
    #[tokio::test]
    async fn callback_lit_search_returns_hits_and_deduplication_applies() {
        use std::sync::Arc;

        let search = CallbackLitSearch::new(|_query: String, _top_k: usize| async move {
            Ok(vec![
                RetrievedContext {
                    source_id: "arxiv:2105.14103".into(),
                    content: "first chunk".into(),
                    section: None,
                },
                RetrievedContext {
                    source_id: "arxiv:2105.14103".into(), // duplicate
                    content: "second chunk".into(),
                    section: Some("Introduction".into()),
                },
                RetrievedContext {
                    source_id: "s2:abc123".into(),
                    content: "other paper".into(),
                    section: None,
                },
            ])
        });

        let retriever = LitRetriever::new(Arc::new(search));
        let results = retriever
            .retrieve("test query", 25)
            .await
            .expect("retrieve must succeed");

        assert_eq!(
            results.len(),
            2,
            "LitRetriever must dedup the duplicate source_id to one result"
        );
        assert!(
            results.iter().any(|h| h.source_id == "arxiv:2105.14103"),
            "arxiv result must be present"
        );
        assert!(
            results.iter().any(|h| h.source_id == "s2:abc123"),
            "s2 result must be present"
        );
    }

    /// When the callback errors, LitRetriever propagates the error rather than
    /// swallowing it to empty.
    ///
    /// DISP-01: error propagation contract.
    #[tokio::test]
    async fn callback_lit_search_propagates_errors() {
        use std::sync::Arc;

        let search = CallbackLitSearch::new(|_query: String, _top_k: usize| async move {
            Err(TtdError::SpawnFailed("search backend unavailable".into()))
        });

        let retriever = LitRetriever::new(Arc::new(search));
        let result = retriever.retrieve("test query", 25).await;

        assert!(
            result.is_err(),
            "retrieve must propagate the callback error (not swallow to empty)"
        );
    }

    // ── Existing tests ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn noop_retriever_returns_empty() {
        let r = NoopRetriever;
        let results = r.retrieve("what is the main finding?", 25).await.unwrap();
        assert!(results.is_empty(), "NoopRetriever must always return empty");
    }

    #[tokio::test]
    async fn lit_retriever_deduplicates_by_source_id() {
        struct StubSearch;

        #[async_trait]
        impl LitSearch for StubSearch {
            async fn search(
                &self,
                _query: &str,
                _top_k: usize,
            ) -> Result<Vec<RetrievedContext>, TtdError> {
                Ok(vec![
                    RetrievedContext {
                        source_id: "arxiv:2105.14103".into(),
                        content: "first chunk".into(),
                        section: None,
                    },
                    RetrievedContext {
                        source_id: "arxiv:2105.14103".into(), // duplicate
                        content: "second chunk".into(),
                        section: Some("Introduction".into()),
                    },
                    RetrievedContext {
                        source_id: "s2:abc123".into(),
                        content: "other paper".into(),
                        section: None,
                    },
                ])
            }
        }

        let r = LitRetriever::new(Arc::new(StubSearch));
        let results = r.retrieve("test query", 25).await.unwrap();

        assert_eq!(results.len(), 2, "duplicate source_id must be deduped to one result");
        assert!(
            results.iter().any(|h| h.source_id == "arxiv:2105.14103"),
            "arxiv result must be present"
        );
        assert!(
            results.iter().any(|h| h.source_id == "s2:abc123"),
            "s2 result must be present"
        );
    }
}
