//! Error taxonomy for the Alzina runtime.
//!
//! String-wrapping variants are preserved for backwards compatibility.
//! Structured variants (e.g. `GovernanceStructured`, `TierViolationDetail`)
//! carry typed payloads for programmatic error handling.

use crate::identity::WriteTier;
use std::path::PathBuf;
use thiserror::Error;

/// Structured governance error detail.
#[derive(Debug, Clone)]
pub struct GovernanceDetail {
    /// The write tier involved.
    pub tier: WriteTier,
    /// The path that triggered the error.
    pub path: PathBuf,
    /// Human-readable description.
    pub message: String,
}

impl std::fmt::Display for GovernanceDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:?}] {}: {}",
            self.tier,
            self.path.display(),
            self.message
        )
    }
}

/// Structured search error detail.
///
/// Carries an explicit `degraded` flag and `degradation_reason` so callers can
/// surface loud-degradation signals (AC-1) to downstream agents and tools rather
/// than silently returning empty or partial results.
#[derive(Debug, Clone)]
pub struct SearchDetail {
    /// Human-readable description.
    pub message: String,
    /// Whether the search ran in a degraded mode (e.g. embedding fallback,
    /// partial index availability).
    pub degraded: bool,
    /// Optional human-readable reason for the degradation.
    pub degradation_reason: Option<String>,
}

impl std::fmt::Display for SearchDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if self.degraded {
            write!(f, " (degraded")?;
            if let Some(reason) = &self.degradation_reason {
                write!(f, ": {}", reason)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// Structured tier violation detail.
#[derive(Debug, Clone)]
pub struct TierViolationDetail {
    /// The path being written to.
    pub path: PathBuf,
    /// The tier the path belongs to.
    pub expected_tier: WriteTier,
    /// The tier the caller was authorised for (if applicable).
    pub actual_tier: Option<WriteTier>,
    /// Human-readable description.
    pub message: String,
}

impl std::fmt::Display for TierViolationDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tier violation on {}: expected {:?}",
            self.path.display(),
            self.expected_tier
        )?;
        if let Some(actual) = self.actual_tier {
            write!(f, ", got {:?}", actual)?;
        }
        write!(f, " — {}", self.message)
    }
}

/// Unified error type for the Alzina runtime.
///
/// # Examples
///
/// ```
/// use alzina_core::AlzinaError;
///
/// let err = AlzinaError::Config("missing field".into());
/// assert!(err.to_string().contains("missing field"));
/// ```
#[derive(Debug, Error)]
pub enum AlzinaError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("workspace error: {0}")]
    Workspace(String),

    #[error("channel error: {0}")]
    Channel(String),

    /// Legacy string-based governance error (preserved for backwards compat).
    #[error("governance error: {0}")]
    Governance(String),

    /// Structured governance error with tier and path detail.
    #[error("governance error: {0}")]
    GovernanceStructured(GovernanceDetail),

    #[error("hook error: {0}")]
    Hook(String),

    #[error("envelope parse error: {0}")]
    EnvelopeParse(String),

    /// Legacy string-based tier violation (preserved for backwards compat).
    #[error("tier violation: {0}")]
    TierViolation(String),

    /// Structured tier violation with path/tier detail.
    #[error("tier violation: {0}")]
    TierViolationDetail(TierViolationDetail),

    #[error("orchestration error: {0}")]
    Orchestration(String),

    #[error("session error: {0}")]
    Session(String),

    #[error("capacity exceeded: {0}")]
    CapacityExceeded(String),

    #[error("audit error: {0}")]
    Audit(String),

    #[error("bootstrap error: {0}")]
    Bootstrap(String),

    #[error("template error: {0}")]
    Template(String),

    #[error("quality gate error: {0}")]
    QualityGate(String),

    #[error("memory error: {0}")]
    Memory(String),

    /// Structured search error with degradation detail.
    #[error("search error: {0}")]
    Search(SearchDetail),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Convenience type alias for Alzina results.
pub type AlzinaResult<T> = Result<T, AlzinaError>;
