//! Shared error type for the TTD engine.
//!
//! Defined in a dedicated sub-module to allow `state.rs`, `fitness.rs`, and
//! `stages/mod.rs` to import it without circular module dependencies.

use alzina_core::error::AlzinaError;

/// Errors produced by the TTD engine.
///
/// Follows the thiserror pattern from `adapter/mod.rs` (`AdapterError`).
#[derive(Debug, thiserror::Error)]
pub enum TtdError {
    /// A weight table's sum is outside the 1.0 ± 0.001 tolerance.
    /// Port of `ValueError` from `weighted_select` (fitness.py:354).
    #[error("weight table sum {sum:.4} ≠ 1.0 (±0.001)")]
    InvalidWeightSum { sum: f32 },

    /// No candidates were provided to selection or merging.
    #[error("no candidates to select")]
    NoCandidates,

    /// Retrieved context for a gap query came back empty; caller should return
    /// the draft unchanged (empty-retrieved guard, graph_tasks.py:1105-1107).
    #[error("retrieval returned no results for gap query")]
    EmptyRetrieved,

    /// Propagated from `AgentExecutor::execute` or `execute_with_envelope`.
    #[error("executor error: {0}")]
    Executor(#[from] AlzinaError),

    /// serde_yaml serialisation or deserialisation error for artifact YAML.
    #[error("artifact YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// A governed spawn (AgentExecutor::execute) returned an error.
    /// String variant so callers can wrap without importing AlzinaError.
    #[error("spawn failed: {0}")]
    SpawnFailed(String),

    /// LLM output could not be parsed into the expected structure.
    #[error("parse failed: {0}")]
    ParseFailed(String),
}
