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

// ── Primary re-exports ──────────────────────────────────────────────────────

// The executor seam is always available (no governance deps).
pub use executor::{AgentExecutor, ExecutorEventEmitter, SamplingParams};

#[cfg(test)]
mod tests {
    #[test]
    fn orchestration_crate_compiles() {
        assert!(true);
    }
}
