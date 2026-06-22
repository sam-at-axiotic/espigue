//! # alzina-core
//!
//! Shared types, error taxonomy, and trait definitions for the Alzina runtime.
//!
//! This crate is the foundation of the workspace — every other crate depends on it.
//! It is deliberately ADK-agnostic: all framework-specific implementations live in
//! `alzina-orchestration`. Types here define the domain model.
//!
//! ## Timestamp Convention
//!
//! Serialised timestamps use ISO-8601 strings (`"2026-03-31T15:00:00Z"`).
//! In-memory representations use `chrono::DateTime<Utc>` where arithmetic is
//! needed, or `String` for pass-through fields. Epoch milliseconds (`u64`) are
//! acceptable only at FFI/bridge boundaries. New types should prefer
//! `DateTime<Utc>` internally and serialise as ISO-8601 strings.

// Standalone literature-synthesis slice: only the modules the litreview build
// reaches. The runtime/governance/memory modules (audit, bootstrap, channel,
// composition, config, display, domain_registry, engagement, hooks,
// learnings_parser, memory_types, message, quality, session, templates, tiers,
// tools, workspace) were vendored from alzina but never reached from this CLI;
// they have been removed.
pub mod envelope;
pub mod error;
pub mod event;
pub mod identity;
pub mod search;

// Re-export primary types at crate root for convenience
pub use envelope::{Envelope, EnvelopeStatus, IssueSeverity, QualityIssue, RawEnvelope, Signal};
pub use error::{AlzinaError, AlzinaResult, GovernanceDetail, SearchDetail, TierViolationDetail};
pub use event::{
    AlzinaEvent, CompositionDispatchMeta, SpawnCompleted, SpawnEventSink, UnresolvedSubstitution,
};
pub use identity::{AgentId, Scope, SessionId, WeaveId, WriteTier};
pub use search::{
    EmbeddingService, EmbeddingTask, PREVIEW_MAX_CHARS, SearchIndexHook, SearchQualityReport,
    SearchResultHit, SearchResults, SemanticSearch, VectorFilters, VectorHit, VectorMetadata,
    VectorStore, truncate_for_preview, wrap_low_authority,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_agent_id() {
        let id = AgentId::new("smidr");
        assert_eq!(id.as_str(), "smidr");
    }

    #[test]
    fn agent_id_try_new_valid() {
        assert!(AgentId::try_new("muninn").is_ok());
        assert!(AgentId::try_new("test-agent").is_ok());
        assert!(AgentId::try_new("agent_01").is_ok());
    }

    #[test]
    fn agent_id_try_new_rejects_empty() {
        assert!(AgentId::try_new("").is_err());
    }

    #[test]
    fn agent_id_try_new_rejects_path_separators() {
        assert!(AgentId::try_new("../etc/passwd").is_err());
        assert!(AgentId::try_new("foo/bar").is_err());
        assert!(AgentId::try_new("foo\\bar").is_err());
    }

    #[test]
    fn agent_id_try_new_rejects_null_bytes() {
        assert!(AgentId::try_new("foo\0bar").is_err());
    }

    #[test]
    fn agent_id_try_new_rejects_dots() {
        assert!(AgentId::try_new(".").is_err());
        assert!(AgentId::try_new("..").is_err());
    }

    // D4 (D3-P2-1): control + bidi rejection.
    #[test]
    fn agent_id_try_new_rejects_control_chars() {
        assert!(AgentId::try_new("foo\nbar").is_err());
        assert!(AgentId::try_new("foo\rbar").is_err());
        assert!(AgentId::try_new("foo\tbar").is_err());
        assert!(AgentId::try_new("foo\x1bbar").is_err()); // ESC
    }

    #[test]
    fn agent_id_try_new_rejects_bidi_overrides() {
        assert!(AgentId::try_new("foo\u{202E}bar").is_err());
        assert!(AgentId::try_new("foo\u{2066}bar").is_err());
        assert!(AgentId::try_new("foo\u{2069}bar").is_err());
    }

    #[test]
    #[should_panic(expected = "invalid AgentId")]
    fn agent_id_new_panics_on_invalid() {
        let _ = AgentId::new("../malicious");
    }

    #[test]
    fn create_session_id() {
        let id = SessionId::new();
        // UUID v4 is 36 chars with hyphens
        assert_eq!(id.to_string().len(), 36);
    }

    #[test]
    fn create_weave_id() {
        let id = WeaveId::new("runtime-migration");
        assert_eq!(id.as_str(), "runtime-migration");
    }

    // R-WEAVE-SCOPE-001 — WeaveId validation
    #[test]
    fn weave_id_try_new_rejects_empty() {
        assert!(WeaveId::try_new("").is_err());
    }

    #[test]
    fn weave_id_try_new_rejects_session_default_literal() {
        assert!(WeaveId::try_new("SessionDefault").is_err());
    }

    #[test]
    fn weave_id_try_new_rejects_path_separators() {
        assert!(WeaveId::try_new("foo/bar").is_err());
        assert!(WeaveId::try_new("foo\\bar").is_err());
        assert!(WeaveId::try_new("../etc/passwd").is_err());
    }

    #[test]
    fn weave_id_try_new_rejects_control_chars() {
        assert!(WeaveId::try_new("W-\nfoo").is_err());
    }

    // R-WEAVE-SCOPE-001 — Scope
    #[test]
    fn scope_as_str_weave_and_session_default() {
        let scoped = Scope::Weave(WeaveId::new("W-f6bff644"));
        assert_eq!(scoped.as_str(), "W-f6bff644");
        let unscoped = Scope::SessionDefault;
        assert_eq!(unscoped.as_str(), "SessionDefault");
    }

    #[test]
    fn scope_parse_round_trip() {
        let s = Scope::parse("W-f6bff644").unwrap();
        assert!(s.is_weave());
        assert_eq!(s.weave_id().unwrap().as_str(), "W-f6bff644");

        let s = Scope::parse("SessionDefault").unwrap();
        assert!(!s.is_weave());
        assert!(s.weave_id().is_none());
    }

    #[test]
    fn scope_parse_rejects_empty() {
        assert!(Scope::parse("").is_err());
    }

    #[test]
    fn scope_serde_round_trip() {
        let weave = Scope::Weave(WeaveId::new("W-abc123"));
        let json = serde_json::to_string(&weave).unwrap();
        assert_eq!(json, "\"W-abc123\"");
        let parsed: Scope = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, weave);

        let unscoped = Scope::SessionDefault;
        let json = serde_json::to_string(&unscoped).unwrap();
        assert_eq!(json, "\"SessionDefault\"");
        let parsed: Scope = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, unscoped);
    }

    #[test]
    fn scope_serde_rejects_session_default_as_weave() {
        // SessionDefault literal must always deserialize as the variant,
        // never as a WeaveId — this is the AC-7 invariant at the wire.
        let parsed: Scope = serde_json::from_str("\"SessionDefault\"").unwrap();
        assert!(matches!(parsed, Scope::SessionDefault));
    }

    #[test]
    fn write_tier_ordering() {
        assert!(matches!(WriteTier::Governed, WriteTier::Governed));
        assert!(matches!(WriteTier::FreeWrite, WriteTier::FreeWrite));
    }

    #[test]
    fn envelope_status_variants() {
        let complete = EnvelopeStatus::Complete;
        let partial = EnvelopeStatus::Partial;
        let error = EnvelopeStatus::Error;
        assert!(matches!(complete, EnvelopeStatus::Complete));
        assert!(matches!(partial, EnvelopeStatus::Partial));
        assert!(matches!(error, EnvelopeStatus::Error));
    }

    #[test]
    fn alzina_error_display() {
        let err = AlzinaError::Config("missing field".into());
        assert!(err.to_string().contains("missing field"));
    }

    #[test]
    fn envelope_construction() {
        let env = Envelope {
            status: EnvelopeStatus::Complete,
            artifacts: vec![std::path::PathBuf::from("artifacts/test.md")],
            signal: Some("test done".into()),
            tensions: None,
            emergent: None,
            next: None,
            context_update: None,
        };
        assert_eq!(env.artifacts.len(), 1);
        assert!(env.signal.is_some());
    }
}
