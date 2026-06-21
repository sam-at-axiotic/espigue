//! Bootstrap types — construction-time context injection.
//!
//! Bootstrap is assembled at agent construction time, NOT via callbacks.
//! This is a deliberate design correction from Kvasir's red-team:
//! construction-time injection is strictly more reliable than runtime
//! callback-based detection of "first invocation."
//!
//! The BootstrapPipeline assembles the full context by reading governance
//! state from the filesystem, then the assembled context is baked into
//! the agent's instruction before it ever runs.

use crate::error::AlzinaResult;
use crate::identity::{AgentId, WeaveId, WriteTier};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Distinguishes the two rendering paths for bootstrap assembly.
///
/// - `Root` — orchestrator bootstrap (system preamble expanded, no dispatch envelope)
/// - `SubAgent` — sub-agent bootstrap (system preamble compressed, dispatch envelope included)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionType {
    Root,
    SubAgent,
}

/// The assembled context injected into a sub-agent session at construction time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapContext {
    /// Raw TOML identity content, for debugging and audit. Not rendered in
    /// prompts — the template uses `spawn_essence` and structured fields instead.
    /// Capped at 8KB by the identity stage via `safe_truncate`.
    pub identity: Option<String>,
    /// Condensed identity for prompt injection.
    pub spawn_essence: Option<String>,
    /// Weave state from orlog.md.
    pub weave_state: Option<String>,
    /// Domain learnings from the learnings/ directory.
    pub learnings: Vec<String>,
    /// Active governance constraints.
    pub governance_gates: Vec<String>,
    /// Envelope format template for the agent's return.
    pub dispatch_template: Option<String>,
    /// Permitted tools for this agent.
    pub tool_allowlist: Vec<String>,
    /// Enforced write tier level.
    pub write_tier: WriteTier,
    /// Custom fragments injected by hooks.
    pub operator_fragments: Vec<BootstrapFragment>,
    /// Session type: Root (orchestrator) or SubAgent.
    pub session_type: SessionType,
    /// Workspace governance documents (SOUL.md, GLOSSARY.md, etc.).
    pub system_preamble: Option<String>,
    /// Narrative companion file content (narrative.md).
    pub agent_profile: Option<String>,
    /// Curated workspace state (populated by Phase 4).
    pub curated_memory: Option<String>,
    /// Model the agent's identity pins itself to (the `model` field in
    /// `identity.toml`), if any. The dispatch runner feeds this into model
    /// resolution so a per-agent pin (e.g. a haiku reader) takes effect.
    /// `None` = fall through to the per-dispatch override or workspace default.
    pub agent_model: Option<String>,
}

impl Default for BootstrapContext {
    fn default() -> Self {
        Self {
            identity: None,
            spawn_essence: None,
            weave_state: None,
            learnings: Vec::new(),
            governance_gates: Vec::new(),
            dispatch_template: None,
            tool_allowlist: Vec::new(),
            write_tier: WriteTier::FreeWrite,
            operator_fragments: Vec::new(),
            session_type: SessionType::SubAgent,
            system_preamble: None,
            agent_profile: None,
            curated_memory: None,
            agent_model: None,
        }
    }
}

/// A fragment of content to inject into the bootstrap context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapFragment {
    /// Label for ordering and logging.
    pub label: String,
    /// The content to inject.
    pub content: String,
    /// Priority for ordering within custom fragments (lower = earlier).
    pub priority: u32,
}

/// Assembles the full bootstrap context for an agent spawn.
///
/// This is a construction-time pipeline: it runs BEFORE the agent is created,
/// and the resulting context is baked into the agent's instruction.
/// This design avoids the fragile pattern of detecting "first invocation"
/// at callback time (see Kvasir red-team §SP1).
#[async_trait]
pub trait BootstrapPipeline: Send + Sync {
    /// Assemble the full bootstrap context for an agent spawn.
    ///
    /// Pipeline order (8+1 stages):
    /// Identity → AgentProfile → SystemPreamble → CuratedMemory(placeholder)
    /// → Governance → WeaveContext → Learnings → Operator → Template.
    async fn assemble(
        &self,
        agent_id: &AgentId,
        task: &str,
        weave_id: Option<&WeaveId>,
        session_type: SessionType,
    ) -> AlzinaResult<BootstrapContext>;
}
