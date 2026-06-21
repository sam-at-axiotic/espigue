//! # alzina-orchestration
//!
//! ADK-Rust integration — the ONLY crate that depends on ADK-Rust.
//!
//! This crate provides ADK-backed implementations of the traits defined
//! in `alzina-core`. If ADK-Rust proves unsuitable, replace this crate's
//! internals — all other crates are ADK-agnostic.
//!
//! ## Public API
//!
//! The `OrchestratorEngine` is the primary interface. Everything else
//! (compiler, runner, session hierarchy, quality gates) is internal
//! machinery accessed through the engine.
//!
//! ```text
//! alzina_orchestration::OrchestratorEngine
//!   ├── execute(CompOp)                → run a composition
//!   ├── execute_pattern(name, context) → named pattern lookup + execute
//!   └── spawn_single(spec, weave_id)   → single-agent convenience
//! ```
//!
//! ## Design notes (from Kvasir red-team)
//!
//! - **GraphAgent is optional for fixed patterns.** For dynamic dispatch
//!   (where the orchestrator decides what to run based on the task), use
//!   a custom orchestrator. GraphAgent is best for known, fixed composition
//!   patterns like the Norn triad.
//!
//! - **Construction-time bootstrap** preferred over callback-based first-
//!   invocation detection. Bootstrap context is baked into the agent's
//!   instruction at build time.
//!
//! ## Phase 3: Full orchestration.

// Always compiled — the standalone literature-synthesis surface (TTD engine).
// These carry no governance / adk / workspace / memory dependencies.
pub mod adapter;
pub mod executor;
pub mod ttd;

// The full orchestration runtime (agent dispatch, composition, sessions,
// governance integration, the ADK sidecar). Gated behind the default-on
// `runner` feature so the TTD engine can be built standalone with
// `--no-default-features`.
#[cfg(feature = "runner")]
pub mod composition;
#[cfg(feature = "runner")]
pub mod engine;
#[cfg(feature = "runner")]
pub(crate) mod quality;
#[cfg(feature = "runner")]
pub mod runner;
#[cfg(feature = "runner")]
pub mod seam_tools;
#[cfg(feature = "runner")]
pub mod session;
#[cfg(feature = "runner")]
pub mod signals;
#[cfg(feature = "runner")]
pub mod tool_adapter;

#[cfg(all(feature = "runner", any(test, feature = "test-harness")))]
pub mod test_helpers;

#[cfg(feature = "test-harness")]
pub use test_helpers::SleepyExecutor;

// ── Primary re-exports ──────────────────────────────────────────────────────

// The executor seam is always available (no governance deps).
pub use executor::{AgentExecutor, ExecutorEventEmitter, SamplingParams};

// RT3-17: Public API surface — only listed types are fully public.
#[cfg(feature = "runner")]
pub use engine::{ExecutionResult, OrchestratorEngine, PatternRegistry};

#[cfg(feature = "runner")]
pub use composition::compiler::{CompOp, CompiledGraph, SpawnSpec};

// Re-export runner types needed to construct the engine
#[cfg(feature = "runner")]
pub use runner::alzina_runner::{AlzinaRunner, SpawnResult};
#[cfg(feature = "runner")]
pub use runner::assigned_dirs::{AssignedDirGuard, AssignedDirRegistry};
#[cfg(feature = "runner")]
pub use runner::claude_agent_sdk::ClaudeAgentSdkExecutor;
#[cfg(feature = "runner")]
pub use runner::sidecar_handle::{ChatEventEmitter, SidecarHandle};
#[cfg(feature = "runner")]
pub use runner::sidecar_protocol::CustomToolDefinition;
#[cfg(feature = "runner")]
pub use session::hierarchy::SessionHierarchy;

#[cfg(test)]
mod tests {
    #[test]
    fn orchestration_crate_compiles() {
        assert!(true);
    }
}
