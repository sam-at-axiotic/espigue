//! Phase 3: Composition engine — compiling algebra to execution.

pub mod compiler;
pub mod edges;
pub mod leaf_hook;
pub mod nodes;
pub mod parser;

// Re-exports for compiler
pub use compiler::{
    CompOp, CompiledGraph, ConditionalLoopSpec, ExecutionResult, GateSpec, LoopSpec, Predicate,
    SpawnSpec, SynthesisSpec, compile,
};

// Re-exports for leaf hook (Phase 10 Plan 05)
pub use leaf_hook::{CompositionLeafHook, NoopLeafHook, noop_hook};

// Re-exports for parser (Phase 10)
pub use parser::{
    AncestorSummary, CompositionContext, ErrorCategory, GateFeedback, LeafIdent, PLAN_SIZE_CAP,
    ParseError, ParseErrorCode, ParseErrors, ParsedCompose, ReservedChannelState, SourceMap,
    parse_compose, render_task,
};
