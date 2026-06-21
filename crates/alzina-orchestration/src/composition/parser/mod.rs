//! Composition-grammar XML parser.
//!
//! Contract: round-trip against `docs/composition-grammar.md` §3 (operators)
//! and §4 (data flow). Strict-list rejection per §1.5. Error contract per
//! §6.2. The source-of-truth doc + this parser + `build_dispatch_tools` tool
//! description + `config/agents/vefr/narrative.md` move together per §8 —
//! any one without the others is a bug.
//!
//! Invariants:
//! - Parser NEVER constructs a CompOp that fails §4.4 happens-before scope
//!   validation. References to non-ancestors are rejected at parse time as
//!   Category C `REF_NON_ANCESTOR`.
//! - The strict reject list (§1.5: namespaces, DTDs, external entities,
//!   processing instructions, unknown tags, unknown attributes) is enforced
//!   before any AST construction (see `tokenizer.rs`).
//! - Rationale (XML comments per §1.4) is NOT stored on CompOp variants; it
//!   travels in the parallel `rationale_map: HashMap<String, String>` keyed
//!   by node_id.
//! - Error codes are public contract — adding new codes is additive; renaming
//!   is a breaking change requiring the §8 doc-round-trip update.

pub mod ast;
pub mod errors;
pub mod render;
pub mod scope;
pub mod tokenizer;

pub use ast::AstPath;
pub use errors::{ErrorCategory, ParseError, ParseErrorCode, ParseErrors};
pub use render::{
    AncestorSummary, CompositionContext, GateFeedback, ReservedChannelState, render_task,
    render_task_with_audit,
};
pub use tokenizer::SourceMap;

use crate::composition::compiler::CompOp;
use std::collections::HashMap;

/// Maximum plan size enforced at the parser entry point.
/// Plans exceeding this are rejected with Category B `PLAN_TOO_LARGE`.
pub const PLAN_SIZE_CAP: usize = 64 * 1024;

/// Result of parsing a `<Compose>` XML plan.
///
/// `op` is the typed AST root; `rationale_map` carries XML-comment rationale
/// per §1.4 keyed by node_id; `leaf_node_ids` lists every leaf Spawn / FanOut
/// prompt for pre-registration by the daemon (see D10-10 / Phase 10 RESEARCH
/// § Open Question Q1). `node_id_map` (per B-06) maps each AST node's
/// AstPath to its resolved id — explicit `id="..."` attributes take precedence
/// over auto-derived forms; scope.rs consumes this map directly.
pub struct ParsedCompose {
    pub op: CompOp,
    pub rationale_map: HashMap<String, String>,
    pub leaf_node_ids: Vec<LeafIdent>,
    /// Per B-06: AstPath → resolved node_id, built during AST construction.
    /// Lets downstream callers (Plan 05 prewalk, Plan 06 tests) look up any
    /// AST node's resolved id deterministically without re-deriving ids.
    pub node_id_map: HashMap<AstPath, String>,
}

impl std::fmt::Debug for ParsedCompose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParsedCompose")
            .field("op", &self.op)
            .field("rationale_map_len", &self.rationale_map.len())
            .field("leaf_node_ids_len", &self.leaf_node_ids.len())
            .field("node_id_map_len", &self.node_id_map.len())
            .finish()
    }
}

/// Identifies a leaf dispatch node in the parsed composition.
pub struct LeafIdent {
    pub node_id: String,
    pub agent: String,
    /// When `true`, this leaf is inside a `Sequential` chain at a position
    /// after the first child — meaning it will not begin execution immediately.
    /// The daemon should defer publishing `SessionSpawned` until the engine
    /// actually dispatches the leaf (via `on_leaf_dispatch`).
    pub deferred: bool,
    /// Structural parent chain from root (outermost) to immediate parent
    /// (innermost), exclusive of the leaf itself. Empty for top-level leaves.
    /// Each entry is a resolved node_id from the parser's `node_id_map`.
    /// FanOut per-prompt leaves include the fanout root id as their last
    /// ancestor (the fanout root is the structural parent of each `_p{i}`).
    pub ancestor_ids: Vec<String>,
}

impl std::fmt::Debug for LeafIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeafIdent")
            .field("node_id", &self.node_id)
            .field("agent", &self.agent)
            .field("deferred", &self.deferred)
            .field("ancestor_ids", &self.ancestor_ids)
            .finish()
    }
}

/// Parse a `<Compose>` XML plan into a typed AST.
///
/// Errors are Category-A/B/C per `docs/composition-grammar.md` §6.2.
/// Returns `Err(PlanTooLarge)` for plans exceeding 64 KiB; otherwise
/// runs the full tokenize → AST → scope analysis pipeline.
pub fn parse_compose(xml: &str) -> Result<ParsedCompose, ParseErrors> {
    if xml.len() > PLAN_SIZE_CAP {
        return Err(ParseErrors::single(ParseError {
            category: ErrorCategory::B,
            code: ParseErrorCode::PlanTooLarge,
            line: 1,
            column: 1,
            message: format!(
                "plan size {} bytes exceeds cap of {} bytes",
                xml.len(),
                PLAN_SIZE_CAP
            ),
            hint: "split the plan into smaller phases or trim verbose tasks".into(),
        }));
    }
    let sm = tokenizer::SourceMap::new(xml);
    let (op, rationale_map, leaf_node_ids, node_id_map) = ast::parse(xml, &sm)?;
    // B-06: pass node_id_map into scope.rs; the monotonic-counter fallback
    // path inside scope.rs is EXCISED, so this argument is load-bearing.
    scope::analyze(&op, &node_id_map)?;
    Ok(ParsedCompose {
        op,
        rationale_map,
        leaf_node_ids,
        node_id_map,
    })
}
