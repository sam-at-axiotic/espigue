//! Envelope quality gate — evaluates agent output envelopes against gate criteria.
//!
//! Bridges the `QualityGate` trait (alzina-core) to the `GovernanceLayer` facade
//! (alzina-governance). The gate evaluates structural quality via
//! `GovernanceLayer::validate_envelope()` and maps the results, together with
//! criteria-specific checks, into a `GateVerdict`.

use std::sync::Arc;

use alzina_core::composition::{GateCriteria, GateFailAction, GateVerdict};
use alzina_core::envelope::{Envelope, IssueSeverity, QualityIssue};
use alzina_core::error::AlzinaResult;
use alzina_core::quality::QualityGate;
use alzina_governance::GovernanceLayer;
use async_trait::async_trait;
use tracing::debug;

/// Quality gate implementation backed by `GovernanceLayer::validate_envelope()`.
///
/// Evaluates envelopes in two passes:
/// 1. Structural validation via GovernanceLayer (required fields, artifact paths, etc.)
/// 2. Criteria-specific checks (status match, tension count)
///
/// Maps the combined issues into a `GateVerdict`.
pub struct EnvelopeQualityGate {
    governance: Arc<GovernanceLayer>,
}

impl EnvelopeQualityGate {
    /// Construct with a reference to the governance layer.
    pub fn new(governance: Arc<GovernanceLayer>) -> Self {
        Self { governance }
    }

    /// Criteria-specific checks beyond structural validation.
    fn check_criteria(envelope: &Envelope, criteria: &GateCriteria) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        // Status must match (if specified)
        if let Some(ref required_status) = criteria.status_must_be
            && envelope.status != *required_status
        {
            issues.push(QualityIssue {
                severity: IssueSeverity::Error,
                field: "status".into(),
                message: format!(
                    "Expected status '{:?}' but got '{:?}'",
                    required_status, envelope.status
                ),
            });
        }

        // Required envelope fields (criteria-level, distinct from config-level)
        for field in &criteria.envelope_required_fields {
            let present = match field.to_lowercase().as_str() {
                "signal" => envelope.signal.is_some(),
                "artifacts" => !envelope.artifacts.is_empty(),
                "tensions" => envelope.tensions.is_some(),
                "emergent" => envelope.emergent.is_some(),
                "next" => envelope.next.is_some(),
                "context_update" => envelope.context_update.is_some(),
                _ => {
                    issues.push(QualityIssue {
                        severity: IssueSeverity::Warning,
                        field: field.clone(),
                        message: format!("Unknown criteria field: '{field}'"),
                    });
                    continue;
                }
            };

            if !present {
                issues.push(QualityIssue {
                    severity: IssueSeverity::Error,
                    field: field.clone(),
                    message: format!("Criteria requires field '{field}' but it is missing"),
                });
            }
        }

        // Tension count limit from criteria
        if let Some(max) = criteria.max_tensions
            && let Some(ref tensions) = envelope.tensions
        {
            // RT3-14: Structured tension check — "none" or variants mean zero tensions
            let trimmed = tensions.trim().to_lowercase();
            let has_real_tensions =
                !trimmed.is_empty() && trimmed != "none" && !trimmed.starts_with("no tensions");
            if has_real_tensions {
                // Count structured entries (lines starting with `- ` or `* `)
                let tension_count = tensions
                    .lines()
                    .filter(|l| {
                        let t = l.trim();
                        t.starts_with("- ") || t.starts_with("* ")
                    })
                    .count()
                    .max(1); // At least 1 if field has real content
                if tension_count > max {
                    issues.push(QualityIssue {
                        severity: IssueSeverity::Error,
                        field: "tensions".into(),
                        message: format!(
                            "Tension count ({tension_count}) exceeds criteria max ({max})"
                        ),
                    });
                }
            }
        }

        issues
    }
}

#[async_trait]
impl QualityGate for EnvelopeQualityGate {
    /// Evaluate an envelope against gate criteria.
    ///
    /// Pass 1: GovernanceLayer structural validation.
    /// Pass 2: Criteria-specific checks (status, required fields, tension count).
    ///
    /// Verdict mapping:
    /// - No errors → `Pass`
    /// - Errors present → `Fail` with `RetryWithFeedback` recommendation
    /// - Deferred if the envelope status is `Partial` (agent signalled incomplete work)
    async fn evaluate(
        &self,
        envelope: &Envelope,
        criteria: &GateCriteria,
    ) -> AlzinaResult<GateVerdict> {
        // Pass 1: structural validation
        let structural_issues = self.governance.validate_envelope(envelope);

        // Pass 2: criteria-specific
        let criteria_issues = Self::check_criteria(envelope, criteria);

        // Combine
        let all_issues: Vec<QualityIssue> = structural_issues
            .into_iter()
            .chain(criteria_issues)
            .collect();

        let error_count = all_issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count();

        debug!(
            total_issues = all_issues.len(),
            errors = error_count,
            "envelope quality gate evaluated"
        );

        // Verdict mapping
        //
        // RT3-06: Partial envelopes are evaluated against criteria rather than
        // short-circuiting to Deferred. Fields may be incomplete, so errors on
        // Partial envelopes yield Fail with actionable feedback.
        let is_partial = envelope.status == alzina_core::EnvelopeStatus::Partial;

        if error_count > 0 {
            Ok(GateVerdict::Fail {
                issues: all_issues,
                recommendation: GateFailAction::RetryWithFeedback,
            })
        } else if is_partial {
            // No errors against criteria, but envelope is incomplete.
            // Note: evaluated fields may be incomplete.
            Ok(GateVerdict::Deferred {
                reason: "Partial envelope passed criteria checks, but fields may be incomplete"
                    .into(),
            })
        } else {
            Ok(GateVerdict::Pass)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_core::composition::GateCriteria;
    use alzina_core::envelope::{Envelope, EnvelopeStatus};
    use alzina_governance::{GovernanceConfig, GovernanceLayer};
    use alzina_workspace::WorkspaceHandle;
    use std::path::PathBuf;

    fn test_gate() -> (tempfile::TempDir, EnvelopeQualityGate) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(WorkspaceHandle::open(dir.path().to_path_buf()).unwrap());
        let config = GovernanceConfig::default();
        let governance = Arc::new(GovernanceLayer::new(config, ws).unwrap());
        (dir, EnvelopeQualityGate::new(governance))
    }

    fn base_envelope() -> Envelope {
        Envelope {
            status: EnvelopeStatus::Complete,
            artifacts: Vec::new(),
            signal: Some("done".into()),
            tensions: None,
            emergent: None,
            next: None,
            context_update: None,
        }
    }

    fn permissive_criteria() -> GateCriteria {
        GateCriteria {
            envelope_required_fields: Vec::new(),
            status_must_be: None,
            max_tensions: None,
        }
    }

    // === Pass verdict ===

    #[tokio::test]
    async fn pass_with_permissive_criteria() {
        let (_dir, gate) = test_gate();
        let envelope = base_envelope();
        let criteria = permissive_criteria();
        let verdict = gate.evaluate(&envelope, &criteria).await.unwrap();
        assert!(matches!(verdict, GateVerdict::Pass));
    }

    // === Fail: status mismatch ===

    #[tokio::test]
    async fn fail_on_status_mismatch() {
        let (_dir, gate) = test_gate();
        let mut envelope = base_envelope();
        envelope.status = EnvelopeStatus::Error;
        let criteria = GateCriteria {
            status_must_be: Some(EnvelopeStatus::Complete),
            ..permissive_criteria()
        };
        let verdict = gate.evaluate(&envelope, &criteria).await.unwrap();
        assert!(matches!(verdict, GateVerdict::Fail { .. }));
    }

    // === Fail: missing required field from criteria ===

    #[tokio::test]
    async fn fail_on_missing_required_field() {
        let (_dir, gate) = test_gate();
        let mut envelope = base_envelope();
        envelope.emergent = None;
        let criteria = GateCriteria {
            envelope_required_fields: vec!["emergent".into()],
            ..permissive_criteria()
        };
        let verdict = gate.evaluate(&envelope, &criteria).await.unwrap();
        assert!(matches!(verdict, GateVerdict::Fail { .. }));
    }

    // === Fail: too many tensions ===

    #[tokio::test]
    async fn fail_on_excessive_tensions() {
        let (_dir, gate) = test_gate();
        let mut envelope = base_envelope();
        envelope.tensions = Some("- tension one\n- tension two\n- tension three".into());
        let criteria = GateCriteria {
            max_tensions: Some(1),
            ..permissive_criteria()
        };
        let verdict = gate.evaluate(&envelope, &criteria).await.unwrap();
        assert!(matches!(verdict, GateVerdict::Fail { .. }));
    }

    // === Deferred: partial status ===

    #[tokio::test]
    async fn deferred_on_partial_status() {
        let (_dir, gate) = test_gate();
        let mut envelope = base_envelope();
        envelope.status = EnvelopeStatus::Partial;
        let criteria = permissive_criteria();
        let verdict = gate.evaluate(&envelope, &criteria).await.unwrap();
        assert!(matches!(verdict, GateVerdict::Deferred { .. }));
    }

    // === Pass with satisfied criteria ===

    #[tokio::test]
    async fn pass_when_all_criteria_met() {
        let (_dir, gate) = test_gate();
        let mut envelope = base_envelope();
        envelope.emergent = Some("finding".into());
        envelope.context_update = Some("learning".into());
        let criteria = GateCriteria {
            envelope_required_fields: vec!["signal".into(), "emergent".into()],
            status_must_be: Some(EnvelopeStatus::Complete),
            max_tensions: None,
        };
        let verdict = gate.evaluate(&envelope, &criteria).await.unwrap();
        assert!(matches!(verdict, GateVerdict::Pass));
    }

    // === Structural validation (from GovernanceLayer) ===

    #[tokio::test]
    async fn structural_missing_artifact_fails() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(WorkspaceHandle::open(dir.path().to_path_buf()).unwrap());
        let mut config = GovernanceConfig::default();
        config.envelope.validate_artifact_paths = true;
        let governance = Arc::new(GovernanceLayer::new(config, ws).unwrap());
        let gate = EnvelopeQualityGate::new(governance);

        let mut envelope = base_envelope();
        envelope.artifacts = vec![PathBuf::from("nonexistent.md")];
        let criteria = permissive_criteria();
        let verdict = gate.evaluate(&envelope, &criteria).await.unwrap();
        assert!(matches!(verdict, GateVerdict::Fail { .. }));
    }

    // === Criteria check unit tests (sync, no governance layer needed) ===

    #[test]
    fn check_criteria_status_match() {
        let envelope = base_envelope();
        let criteria = GateCriteria {
            status_must_be: Some(EnvelopeStatus::Complete),
            ..permissive_criteria()
        };
        let issues = EnvelopeQualityGate::check_criteria(&envelope, &criteria);
        assert!(issues.is_empty());
    }

    #[test]
    fn check_criteria_unknown_field_warns() {
        let envelope = base_envelope();
        let criteria = GateCriteria {
            envelope_required_fields: vec!["nonexistent".into()],
            ..permissive_criteria()
        };
        let issues = EnvelopeQualityGate::check_criteria(&envelope, &criteria);
        assert!(issues.iter().any(|i| i.severity == IssueSeverity::Warning));
    }

    #[test]
    fn check_criteria_tension_count_within_limit() {
        let mut envelope = base_envelope();
        envelope.tensions = Some("- one tension".into());
        let criteria = GateCriteria {
            max_tensions: Some(2),
            ..permissive_criteria()
        };
        let issues = EnvelopeQualityGate::check_criteria(&envelope, &criteria);
        assert!(issues.iter().all(|i| i.severity != IssueSeverity::Error));
    }
}
