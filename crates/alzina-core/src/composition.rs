//! Composition algebra — the operators for multi-agent orchestration.
//!
//! These types define WHAT to compose. HOW to execute them (e.g. mapping
//! to ADK GraphAgent) lives in alzina-orchestration. The algebra itself
//! is framework-agnostic and survives a backend swap.

use crate::envelope::EnvelopeStatus;
use crate::identity::AgentId;
use serde::{Deserialize, Serialize};

/// The composition algebra operators.
///
/// # Examples
///
/// ```
/// use alzina_core::{CompOp, CompNode, AgentId};
///
/// let pipeline = CompOp::Seq(vec![
///     CompNode {
///         agent_id: AgentId::new("urdhr"),
///         task: "read context".into(),
///         model_override: None,
///         tool_overrides: None,
///     },
///     CompNode {
///         agent_id: AgentId::new("skuld"),
///         task: "plan next step".into(),
///         model_override: None,
///         tool_overrides: None,
///     },
/// ]);
/// assert!(matches!(pipeline, CompOp::Seq(nodes) if nodes.len() == 2));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompOp {
    /// → Sequential: A then B, output flows forward.
    Seq(Vec<CompNode>),

    /// ∥ Parallel: A and B concurrently.
    Par(Vec<CompNode>),

    /// ⊕ Synthesis: Run nodes in parallel, then synthesise results.
    Synthesis {
        branches: Vec<CompNode>,
        synthesiser: CompNode,
    },

    /// ⊘ Quality gate: Run node, evaluate output, pass or reject.
    Gate {
        node: Box<CompNode>,
        criteria: GateCriteria,
        on_fail: GateFailAction,
    },

    /// ↻? Conditional iteration: Run, gate, retry on failure.
    ConditionalLoop {
        node: Box<CompNode>,
        gate: GateCriteria,
        max_iterations: usize,
        on_exhaust: ExhaustAction,
    },

    /// [n] Fan-out: Same agent, N varied prompts.
    FanOut {
        agent_id: AgentId,
        prompts: Vec<String>,
        combine: CombineStrategy,
    },
}

/// A node in a composition — identifies the agent and its task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompNode {
    pub agent_id: AgentId,
    pub task: String,
    pub model_override: Option<String>,
    pub tool_overrides: Option<Vec<String>>,
}

/// Criteria for a quality gate evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCriteria {
    /// Required fields in the envelope (e.g. ["status", "signal"]).
    pub envelope_required_fields: Vec<String>,
    /// If set, the envelope status must match this value.
    pub status_must_be: Option<EnvelopeStatus>,
    /// Maximum number of tensions allowed before failing.
    pub max_tensions: Option<usize>,
}

/// Verdict from a quality gate evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateVerdict {
    Pass,
    Fail {
        issues: Vec<crate::envelope::QualityIssue>,
        recommendation: GateFailAction,
    },
    /// Needs human decision.
    Deferred {
        reason: String,
    },
}

/// What to do when a quality gate fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateFailAction {
    /// ↻ Retry the node with gate feedback.
    RetryWithFeedback,
    /// Send to operator for decision.
    Escalate,
    /// Accept with a degradation warning.
    Degrade(String),
}

/// What to do when a conditional loop exhausts its iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExhaustAction {
    Escalate,
    AcceptLast,
    Fail,
}

/// How to combine results from fan-out operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombineStrategy {
    /// ⊕ Synthesise through a dedicated agent.
    Synthesise(AgentId),
    /// Take the first result.
    TakeFirst,
    /// Take all results.
    TakeAll,
}
