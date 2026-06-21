//! Session types — tracking agent session lifecycle.

use crate::envelope::EnvelopeStatus;
use crate::identity::{AgentId, SessionId, WeaveId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A node in the session hierarchy tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionNode {
    pub session_id: SessionId,
    pub agent_id: AgentId,
    pub parent: Option<SessionId>,
    pub children: Vec<SessionId>,
    pub status: SessionStatus,
    pub weave_id: Option<WeaveId>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Status of a session through its lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Pending,
    Bootstrapping,
    Running,
    AwaitingChildren,
    Completing,
    Complete(EnvelopeStatus),
    Failed(String),
}
