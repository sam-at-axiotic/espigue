//! GateNode — evaluates upstream envelope against quality criteria.
//!
//! Implements `adk_rust::graph::Node`. Reads an upstream envelope from
//! state, evaluates it via `QualityGate::evaluate()`, and writes the
//! verdict to the `_gate:verdict` channel.

use std::sync::Arc;

use adk_rust::graph::prelude::*;
use async_trait::async_trait;
use serde_json::json;
use tracing::{debug, instrument, warn};

use alzina_core::QualityGate;
use alzina_core::composition::{GateCriteria, GateVerdict};
use alzina_core::envelope::Envelope;

/// GateNode: evaluates an upstream envelope against quality criteria.
pub struct GateNode {
    name: String,
    upstream_channel: String,
    criteria: GateCriteria,
    quality_gate: Arc<dyn QualityGate>,
}

impl GateNode {
    pub fn new(
        name: impl Into<String>,
        upstream_channel: impl Into<String>,
        criteria: GateCriteria,
        quality_gate: Arc<dyn QualityGate>,
    ) -> Self {
        Self {
            name: name.into(),
            upstream_channel: upstream_channel.into(),
            criteria,
            quality_gate,
        }
    }

    /// Convenience: reads from `{upstream_name}:envelope`.
    pub fn for_upstream(
        name: impl Into<String>,
        upstream_name: &str,
        criteria: GateCriteria,
        quality_gate: Arc<dyn QualityGate>,
    ) -> Self {
        Self::new(
            name,
            format!("{upstream_name}:envelope"),
            criteria,
            quality_gate,
        )
    }
}

#[async_trait]
impl Node for GateNode {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, ctx), fields(node = %self.name, upstream = %self.upstream_channel))]
    async fn execute(&self, ctx: &NodeContext) -> adk_rust::graph::error::Result<NodeOutput> {
        let envelope: Envelope =
            ctx.get_as::<Envelope>(&self.upstream_channel)
                .ok_or_else(|| adk_rust::graph::error::GraphError::NodeExecutionFailed {
                    node: self.name.clone(),
                    message: format!(
                        "upstream envelope not found in state channel '{}'",
                        self.upstream_channel
                    ),
                })?;

        debug!(node = %self.name, status = ?envelope.status, "evaluating gate");

        let verdict = self
            .quality_gate
            .evaluate(&envelope, &self.criteria)
            .await
            .map_err(
                |e| adk_rust::graph::error::GraphError::NodeExecutionFailed {
                    node: self.name.clone(),
                    message: format!("gate evaluation failed: {e}"),
                },
            )?;

        let route = match &verdict {
            GateVerdict::Pass => "pass",
            GateVerdict::Fail { .. } => "fail",
            GateVerdict::Deferred { .. } => {
                warn!(node = %self.name, "gate deferred — treating as fail");
                "fail"
            }
        };

        let verdict_json = serde_json::to_value(&verdict).unwrap_or(json!(null));
        debug!(node = %self.name, route = %route, "gate verdict");

        Ok(NodeOutput::new()
            .with_update("_gate:verdict", verdict_json)
            .with_update("_gate:route", json!(route)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_core::composition::GateFailAction;
    use alzina_core::envelope::{EnvelopeStatus, IssueSeverity, QualityIssue};

    struct MockQualityGate;

    #[async_trait]
    impl QualityGate for MockQualityGate {
        async fn evaluate(
            &self,
            envelope: &Envelope,
            criteria: &GateCriteria,
        ) -> alzina_core::AlzinaResult<GateVerdict> {
            if let Some(ref required_status) = criteria.status_must_be {
                if &envelope.status != required_status {
                    return Ok(GateVerdict::Fail {
                        issues: vec![QualityIssue {
                            severity: IssueSeverity::Error,
                            field: "status".to_string(),
                            message: format!(
                                "expected {:?}, got {:?}",
                                required_status, envelope.status
                            ),
                        }],
                        recommendation: GateFailAction::Escalate,
                    });
                }
            }
            Ok(GateVerdict::Pass)
        }
    }

    fn make_envelope(status: EnvelopeStatus) -> Envelope {
        Envelope {
            status,
            artifacts: vec![],
            signal: Some("test signal".to_string()),
            tensions: None,
            emergent: None,
            next: None,
            context_update: None,
        }
    }

    fn criteria_require_complete() -> GateCriteria {
        GateCriteria {
            envelope_required_fields: vec![],
            status_must_be: Some(EnvelopeStatus::Complete),
            max_tensions: None,
        }
    }

    #[tokio::test]
    async fn gate_passes_when_criteria_met() {
        let gate = Arc::new(MockQualityGate);
        let node = GateNode::new(
            "quality_gate",
            "agent:envelope",
            criteria_require_complete(),
            gate,
        );

        let mut state = State::new();
        state.insert(
            "agent:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete)).unwrap(),
        );

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(output.updates.get("_gate:route"), Some(&json!("pass")));
    }

    #[tokio::test]
    async fn gate_fails_when_status_wrong() {
        let gate = Arc::new(MockQualityGate);
        let node = GateNode::new(
            "quality_gate",
            "agent:envelope",
            criteria_require_complete(),
            gate,
        );

        let mut state = State::new();
        state.insert(
            "agent:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Error)).unwrap(),
        );

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(output.updates.get("_gate:route"), Some(&json!("fail")));
        assert!(output.updates.contains_key("_gate:verdict"));
    }

    #[tokio::test]
    async fn gate_errors_when_upstream_missing() {
        let gate = Arc::new(MockQualityGate);
        let node = GateNode::new(
            "quality_gate",
            "missing:envelope",
            criteria_require_complete(),
            gate,
        );

        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test"), 0);
        let result = node.execute(&ctx).await;
        assert!(result.is_err());
        let err = match result {
            Err(e) => format!("{:?}", e),
            Ok(_) => panic!("expected error"),
        };
        assert!(err.contains("not found"), "error was: {err}");
    }

    #[tokio::test]
    async fn gate_for_upstream_convenience() {
        let gate = Arc::new(MockQualityGate);
        let node = GateNode::for_upstream(
            "check_smidr",
            "smidr_analysis",
            criteria_require_complete(),
            gate,
        );
        assert_eq!(node.upstream_channel, "smidr_analysis:envelope");
    }
}
