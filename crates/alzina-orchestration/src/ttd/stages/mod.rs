//! Stage-task trait definitions for the TTD engine.
//!
//! Each trait defines a single phase in the TTD denoise loop. Implementations
//! are supplied at construction time (injected into `TtdMachine<A>`), keeping
//! the machine generic over both artefact type and stage behaviour.
//!
//! ## Trait hierarchy
//!
//! ```text
//! DraftGen<A>    — produce N initial candidate drafts
//! GapIdentify<A> — find 3-5 gaps in one candidate
//! GapResolve<A>  — resolve gaps via patch → full-regen → heuristic fallback
//! FitnessEval<A> — evaluate one candidate, returning a FitnessEval
//! Merger<A>      — merge the best-selected candidate into the final artefact
//! ```
//!
//! No close analog exists in the codebase (the generic task-trait seam is
//! novel). The `AgentExecutor` trait (`runner/alzina_runner.rs:116`) is used
//! as the pattern for the single-method trait shape; this module extends it
//! to a multi-trait bundle.

pub mod graph;
pub mod narrative;
pub mod synthesis;

use std::sync::Arc;

use async_trait::async_trait;

use crate::adapter::ExpertResponse;
use crate::executor::AgentExecutor;
use crate::ttd::config::TtdConfig;
use crate::ttd::fitness::FitnessEval;
use crate::ttd::mod_types::TtdError;
use crate::ttd::state::IdentifiedGap;

// ── DraftGen ─────────────────────────────────────────────────────────────────

/// Produce N initial candidate drafts for one stage.
///
/// For Stage 1 (graph): map-reduce over expert responses.
/// For Stage 2 (synthesis): draft from the argumentation graph + expert panel.
/// For Stage 3 (narrative): draft from the synthesis.
#[async_trait]
pub trait DraftGen<A>: Send + Sync {
    async fn generate(
        &self,
        inputs: &[ExpertResponse],
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
        persona_prompt: Option<&str>,  // None → baked default template (EXT-01 Phase 24)
        sampling: Option<crate::executor::SamplingParams>,  // None → neutral defaults (EXT-01 Phase 24)
    ) -> Result<A, TtdError>;
}

// ── GapIdentify ───────────────────────────────────────────────────────────────

/// Identify 3-5 gaps in one candidate draft.
///
/// Returns an empty vec when gap_identify returns nothing and the heuristic
/// fallback also fires nothing (very unusual; the heuristic always generates
/// at least one gap from the draft content).
#[async_trait]
pub trait GapIdentify<A>: Send + Sync {
    async fn identify(
        &self,
        draft: &A,
        fitness: &FitnessEval,
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<Vec<IdentifiedGap>, TtdError>;
}

// ── GapResolve ───────────────────────────────────────────────────────────────

/// Resolve identified gaps in one candidate draft.
///
/// The full fallback chain (consensus graph_tasks.py / synthesis_tasks.py):
/// 1. `gap_resolve_patch` (patch-based incremental — preferred)
/// 2. On parse failure: `gap_resolve` (full regeneration)
/// 3. On that failure: heuristic (adds nodes / single-source claims)
///
/// If `retrieved` is empty the draft is returned UNCHANGED (empty-retrieved guard).
#[async_trait]
pub trait GapResolve<A>: Send + Sync {
    /// `fitness` is the current candidate's evaluation (C-N2): Stage-3 refine
    /// embeds it as feedback so the rewrite knows which dimensions scored low.
    /// Stages 1 and 2 ignore it — their resolve is retrieval-driven, not
    /// fitness-driven — so their prompts stay byte-identical.
    async fn resolve(
        &self,
        draft: &A,
        fitness: &FitnessEval,
        gaps: &[IdentifiedGap],
        retrieved: &[RetrievedContext],
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<A, TtdError>;
}

// ── FitnessEval trait ─────────────────────────────────────────────────────────

/// Run the 6-parallel fitness judge spawns for one candidate draft.
///
/// Returns a `FitnessEval` with one `Option<u8>` per fitness dimension.
/// A `None` dimension means the judge spawn produced unparseable output
/// (not 0 — the None-redistribution in `weighted_sum` handles it).
///
/// The two non-async methods (`validity_fn`, `weights`) let `TtdMachine::run()`
/// use the correct validity predicate and weight table for the current stage
/// without hard-coding stage-specific constants in the generic run() body.
#[async_trait]
pub trait EvalFitness<A>: Send + Sync {
    async fn evaluate(
        &self,
        draft: &A,
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<FitnessEval, TtdError>;

    /// Stage-specific validity predicate passed to `sort_candidates_best_first`.
    /// Graph: `is_valid_graph`; Synthesis/Narrative: `is_valid_synthesis`.
    fn validity_fn(&self) -> fn(&FitnessEval) -> bool;

    /// Stage-specific weight table used by `sort_candidates_best_first`.
    fn weights(&self) -> &'static [(&'static str, f32)];
}

// ── Merger ────────────────────────────────────────────────────────────────────

/// Merge the sorted candidates into a single final artefact for this stage.
///
/// Receives all N candidates in best-first order (best candidate first).
/// For Stages 1 and 2 this is a governed spawn through `AgentExecutor`.
/// For Stage 3 it is `narrative_final_merge` with `[Cx]` citation preservation.
#[async_trait]
pub trait Merger<A>: Send + Sync {
    async fn merge(
        &self,
        candidates: &[A],
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<A, TtdError>;
}

// ── RetrievedContext ─────────────────────────────────────────────────────────

/// One retrieved document returned by the `Retriever` for a gap query.
///
/// Carries enough provenance for the gap-resolve step to cite the source.
/// The `source_id` field preserves the paper ID through the retrieval boundary
/// (mirrors the adapter trust boundary — text is in data position, never
/// interpolated into instruction prompts).
#[derive(Debug, Clone)]
pub struct RetrievedContext {
    /// Paper identifier (same namespace as `ExpertResponse.expert_id`).
    pub source_id: String,
    /// Chunk text — raw retrieved content, data position only.
    pub content: String,
    /// Section heading (arxiv chunks); empty for S2 / internal.
    pub section: Option<String>,
}
