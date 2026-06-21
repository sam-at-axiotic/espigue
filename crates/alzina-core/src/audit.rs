//! Audit logging types — append-only governance event trail.

use crate::error::AlzinaResult;
use crate::hooks::HookAction;
use crate::identity::{AgentId, SessionId, WeaveId};
use crate::tiers::TierDecision;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

/// A single audit trail entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub session_id: SessionId,
    pub agent_id: AgentId,
    pub event_type: AuditEventType,
    pub detail: Value,
    pub weave_id: Option<WeaveId>,
}

/// Classification outcome for incoming user messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageClassification {
    /// Message relates to an open weave — route to it.
    ExistingWeave,
    /// Message requires structured multi-agent work — open a new weave.
    NewWeave,
    /// Message can be answered directly without agent dispatch.
    Direct,
}

/// Types of auditable events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    AgentSpawned,
    AgentCompleted,
    EnvelopeProcessed,
    WriteAttempt {
        path: PathBuf,
        decision: TierDecision,
    },
    HookExecuted {
        hook: String,
        action: HookAction,
    },
    QualityGateResult {
        passed: bool,
    },
    CronTriggered {
        job: String,
    },
    SignalRouted {
        signal_type: String,
    },
    /// A system process (reflection, compaction, etc.) performed a governed write.
    SystemProcessWrite {
        /// System process identifier (e.g. "_system-reflection").
        process_id: String,
        /// Path written to.
        path: std::path::PathBuf,
        /// Operation type (e.g. "Modify", "Append").
        operation: String,
    },
    /// Vefr classified an incoming user message for routing.
    MessageClassified {
        /// The classification outcome.
        classification: MessageClassification,
        /// Weave ID if routed to an existing weave or a new weave was opened.
        weave_id: Option<WeaveId>,
        /// Brief reasoning for the classification decision.
        reason: String,
    },
}

/// Filter criteria for querying audit entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditFilter {
    pub agent_id: Option<AgentId>,
    pub session_id: Option<SessionId>,
    pub weave_id: Option<WeaveId>,
    pub event_type: Option<String>,
    pub after: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
}

/// Result of an audit query, including metadata about skipped lines.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub entries: Vec<AuditEntry>,
    pub skipped_count: usize,
}

/// Append-only audit logger.
#[async_trait]
pub trait AuditLogger: Send + Sync {
    /// Log a governance event.
    async fn log(&self, entry: AuditEntry) -> AlzinaResult<()>;

    /// Query recent entries for debugging / dashboard.
    async fn query(&self, filter: AuditFilter, limit: usize) -> AlzinaResult<QueryResult>;
}
