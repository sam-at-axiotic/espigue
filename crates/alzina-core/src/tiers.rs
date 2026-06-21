//! Write tier enforcement types.

use crate::error::AlzinaResult;
use crate::identity::{AgentId, SessionId, WriteTier};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The type of filesystem write operation being attempted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WriteOp {
    Create,
    Append,
    Modify,
    Delete,
}

/// Decision from write tier enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TierDecision {
    /// Write is allowed.
    Allowed,
    /// Write is blocked.
    Blocked { tier: WriteTier, reason: String },
    /// Write requires operator approval (Tier 1: rune pipeline).
    RequiresApproval { tier: WriteTier },
}

/// Enforces write tier restrictions on filesystem operations.
#[async_trait]
pub trait WriteTierEnforcer: Send + Sync {
    /// Check whether an agent has write permission for a path.
    fn check(&self, agent_id: &AgentId, path: &Path, operation: WriteOp) -> TierDecision;

    /// Wrap a filesystem write with tier enforcement.
    ///
    /// The caller specifies the `operation` (e.g., `Modify` vs `Append`) and
    /// provides their `session_id` for audit attribution.
    async fn guarded_write(
        &self,
        agent_id: &AgentId,
        session_id: &SessionId,
        path: &Path,
        content: &[u8],
        operation: WriteOp,
    ) -> AlzinaResult<()>;
}
