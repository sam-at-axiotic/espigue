//! TtdMachine — three-stage test-time diffusion engine.
//!
//! Wires the consensus TTD loop (FanOut → denoise Loop → fitness-select →
//! Synthesise) as a native alzina CompOp composition.
//!
//! # Architecture
//!
//! ```text
//! Vec<ExpertResponse>
//!   └─ Sequential(Stage1, Stage2, Stage3)
//!       └─ TtdMachine<A>::run() → FanOut(N) + Loop(S) + select + Synthesise
//! ```
//!
//! # Wave 0 status
//!
//! This plan (23-01) delivers the scaffold: `TtdConfig`, `TtdMachine<A>` struct
//! and trait impls stubs, `DiffusionState<A>`, stage-task traits, weights, fitness
//! selection, artifact serde, and the retrieval backend decision.
//!
//! `TtdMachine::run()` is a skeleton — the loop body is filled in Wave 1
//! (Plan 23-02, graph stage) and completed through Wave 3 (Plan 23-04).
//!
//! # Source fidelity
//!
//! Reproduces consensus's real defaults faithfully BEFORE enhancement:
//! - N=5 initial drafts, S=2 denoise steps, early_stopping=False (runner.py:67-86)
//! - Three stage-specific weight tables (fitness.py:326-351)
//! - None-redistribution weighted-sum (fitness.py:400-418)
//! - Hard validity gate sort order (is_valid, fitness.py:89-91, 161-163)
//!
//! # Phase 23 deliberate fidelity gap
//!
//! `randomize_sampling = false` — temperature diversity (temp_range 0.5–1.2
//! in consensus) is deferred to Phase 24. The static FanOut(N=5) produces N
//! structurally-identical prompts in Wave 0. Phase 25 must distinguish this
//! from Phase 24's per-trajectory sampling diversity.

pub mod artifact;
pub mod citations;
pub mod config;
pub mod personas;
pub mod sampling;
pub mod emit;
pub mod engine;
pub mod fitness;
pub mod guards;
mod mod_types;
pub mod plan;
pub mod post_process;
pub mod prompts;
pub mod retrieval;
pub mod run;
pub mod stages;
pub mod state;
pub mod term_sheet;
pub mod weights;

// Re-export the primary types needed by callers (Wave 1+ plan tasks).
pub use artifact::{ArgumentationGraph, SynthesisArtifact, SCHEMA_VERSION};
pub use term_sheet::{JudgeDim, PromptProfile, V2_JUDGE_DIMS};
pub use config::TtdConfig;
pub use fitness::{FitnessEval, ParsedFitnessScore, generate_feedback, is_valid_graph, is_valid_synthesis, is_valid_v2, traceability_veto_synthesis, traceability_veto_graph, parse_fitness_response, sort_candidates_best_first, weighted_sum};
pub use mod_types::TtdError;
pub use plan::{PlanMode, PlanTournamentOutcome, ReviewPlan, run_plan_tournament};
pub use retrieval::{LitRetriever, NoopRetriever, RetrievalPolicy, Retriever};
pub use stages::{DraftGen, EvalFitness, GapIdentify, GapResolve, Merger, RetrievedContext};
pub use state::{DiffusionState, IdentifiedGap, StepRecord};
pub use weights::{GRAPH_WEIGHTS, NARRATIVE_WEIGHTS, SYNTHESIS_WEIGHTS, V2_GRAPH_WEIGHTS, V2_SYNTHESIS_WEIGHTS, V2_NARRATIVE_WEIGHTS, V3_PLANNED_NARRATIVE_WEIGHTS};

use std::sync::Arc;

use crate::executor::AgentExecutor;
use alzina_search::bib_store::{BibliographyStore, NoopBibliographyStore};

// ── TtdMachine ────────────────────────────────────────────────────────────────

/// One TTD machine parameterised over artefact type `A`.
///
/// All three stages (graph/synthesis/narrative) share the same `run()` skeleton.
/// Stage-specific behaviour is injected via the task trait implementations.
///
/// ## Pattern source
///
/// Mirrors `TTDRunner<T>` / `TTDStageTasks<T>` from consensus runner.py:111-122.
/// The one-machine-per-stage design (CONTEXT decision) means the caller
/// constructs three `TtdMachine` instances (one per stage) rather than
/// dispatching on stage type inside the machine.
///
/// ## Wave 0 status
///
/// `run()` is a skeleton. Wave 1 implements `TtdMachine<ArgumentationGraph>::run()`,
/// Wave 2 implements `TtdMachine<SynthesisArtifact>::run()`, Wave 3 implements
/// `TtdMachine<String>::run()` (narrative).
pub struct TtdMachine<A> {
    /// Engine configuration (consensus defaults, resource guards).
    pub config: TtdConfig,
    /// Draft generation — produces the initial N candidates.
    pub draft_gen: Arc<dyn DraftGen<A>>,  // WR-04 Phase 24: Arc for concurrent fan-out
    /// Gap identification — finds 3-5 gaps per candidate per denoise step.
    pub gap_identify: Box<dyn GapIdentify<A>>,
    /// Gap resolution — patches each candidate with retrieved evidence.
    pub gap_resolve: Box<dyn GapResolve<A>>,
    /// Fitness evaluation — 6-parallel judge spawns per candidate.
    pub eval_fitness: Option<Box<dyn EvalFitness<A>>>,
    /// Candidate merger — synthesises the best-selected candidates.
    pub merger: Box<dyn Merger<A>>,
    /// Retrieval seam — per-gap lit-store queries. Stage 3 uses `NoopRetriever`.
    pub retriever: Box<dyn Retriever>,
    /// AgentExecutor for governed spawns (no new LLM client).
    pub executor: Arc<dyn AgentExecutor>,
    /// Bibliography accumulation store — records per-step sources directly to
    /// the literature KB (EXT-03, Phase 24). `NoopBibliographyStore` by default
    /// (engine.rs construction sites); `SqliteBibliographyStore` in production.
    /// Write path is direct — NOT threaded through composition channels (CONTEXT EXT-03).
    pub bib_store: Arc<dyn BibliographyStore>,
    /// TTD run identifier for bibliography records (weave_id or session_id).
    /// Empty string in test/default construction; set by the live dispatch path.
    pub run_id: String,
    /// Stage label for bibliography records ("graph", "synthesis", "narrative").
    /// Set explicitly at each engine.rs construction site — there is no implicit
    /// default; an unset field is an empty string.
    pub stage_label: String,
}

// TtdMachine::run() is implemented in run.rs (Wave 2, Plan 23-02).
// The impl block lives there to keep the loop logic separate from the struct
// definition and re-exports in this file.
