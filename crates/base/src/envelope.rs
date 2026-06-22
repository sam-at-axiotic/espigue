//! Envelope types — structured return format from sub-agents.
//!
//! The envelope is the contract between agents and the governance layer.
//! Every sub-agent returns a structured envelope; the governance layer
//! parses, validates, and routes signals extracted from it.

use crate::identity::{SessionId, WeaveId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Raw, unparsed envelope text from agent output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEnvelope {
    pub text: String,
    pub session_id: SessionId,
}

/// Parsed return envelope from a sub-agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub status: EnvelopeStatus,
    pub artifacts: Vec<PathBuf>,
    pub signal: Option<String>,
    pub tensions: Option<String>,
    pub emergent: Option<String>,
    pub next: Option<String>,
    pub context_update: Option<String>,
}

/// Status field from the return envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvelopeStatus {
    Complete,
    Partial,
    Error,
}

/// Signals extracted from envelope processing — governance events
/// that the system acts upon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Signal {
    EmergenceDetected {
        content: String,
    },
    CrossWeaveReference {
        target_weave: WeaveId,
        content: String,
    },
    ContextUpdate {
        learning: String,
    },
    TensionFlagged {
        location: String,
        content: String,
    },
    NextStepRecommended {
        action: String,
    },
}

/// A quality issue found during envelope validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub severity: IssueSeverity,
    pub field: String,
    pub message: String,
}

/// Severity levels for quality issues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}
