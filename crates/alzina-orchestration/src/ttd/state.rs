//! Diffusion state for the TTD engine.
//!
//! Tracks N candidate trajectories, S recorded denoise steps, and resource
//! budget counters (wall-clock + LLM call count).
//!
//! Generic over artefact type `A` — Stage 1 uses `ArgumentationGraph`,
//! Stage 2 uses `Synthesis` (a.k.a. `SynthesisArtifact`), Stage 3 uses `String`.

use crate::ttd::fitness::FitnessEval;

/// Per-stage TTD diffusion state.
///
/// `A` is the artefact type for this stage (one of `ArgumentationGraph`,
/// `SynthesisArtifact`, or `String` for the narrative stage).
#[derive(Debug, Clone)]
pub struct DiffusionState<A> {
    /// N current candidate drafts — parallel with the FanOut(N=5) fan-out.
    pub trajectories: Vec<A>,
    /// S recorded denoise steps; grows by one per Loop iteration.
    pub steps: Vec<StepRecord>,
    /// Index into the current step (0-based).
    pub current_step: usize,
    /// Total denoise steps planned (== `TtdConfig.n_denoise_steps`).
    pub total_steps: usize,
    /// Wall-clock elapsed in seconds — checked against `max_stage_seconds`.
    pub elapsed_secs: f64,
    /// LLM call count — checked against `max_llm_calls`.
    pub llm_calls: usize,
}

impl<A> DiffusionState<A> {
    /// Construct a fresh state for `n_trajectories` candidate drafts and
    /// `total_steps` denoise iterations.
    pub fn new(n_trajectories: usize, total_steps: usize, trajectories: Vec<A>) -> Self {
        Self {
            trajectories,
            steps: Vec::with_capacity(total_steps),
            current_step: 0,
            total_steps,
            elapsed_secs: 0.0,
            llm_calls: 0,
        }
    }

    /// Record the result of one denoise step and advance `current_step`.
    pub fn record_step(&mut self, record: StepRecord) {
        self.steps.push(record);
        self.current_step += 1;
    }

    /// True when all denoise steps have been run.
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.total_steps
    }
}

/// Record of one denoise step: fitness evaluations + identified gaps per trajectory.
#[derive(Debug, Clone)]
pub struct StepRecord {
    /// Step index (0-based).
    pub step: usize,
    /// One `FitnessEval` per trajectory — parallel with `DiffusionState.trajectories`.
    pub fitness_evaluations: Vec<FitnessEval>,
    /// Per-trajectory identified gaps — parallel with `DiffusionState.trajectories`.
    pub gaps: Vec<Vec<IdentifiedGap>>,
}

/// A single identified gap in a candidate draft.
///
/// Used by gap_identify and gap_resolve. `description` is human-readable;
/// `query` is the retrieval query issued to the lit store.
#[derive(Debug, Clone)]
pub struct IdentifiedGap {
    /// Human-readable gap description (from gap_identify XML output).
    pub description: String,
    /// Retrieval query issued to the lit store for this gap.
    /// Equals `description` when gap_identify's heuristic fallback fires
    /// (consensus synthesis_tasks.py:473 — description doubles as query).
    pub query: String,
}
