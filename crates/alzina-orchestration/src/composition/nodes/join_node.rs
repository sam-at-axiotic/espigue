//! JoinNode — awaits multiple named state channels and merges results.
//!
//! Implements `adk_rust::graph::Node`. Reads envelopes from multiple
//! upstream state channels, collects them into a merged result,
//! and handles partial failures.
//!
//! # Compiler Duplication Notice
//!
//! This node is used directly for ADK integration (via `adk_rust::graph::Node` trait).
//! The compiler (`compiler.rs`) reimplements this logic for recursive composition.
//! **Security fixes must be applied to both paths.** See RT2-08 for details.

use adk_rust::graph::prelude::*;
use async_trait::async_trait;
use serde_json::json;
use tracing::{debug, instrument, warn};

use alzina_core::envelope::{Envelope, EnvelopeStatus};

/// Result of joining multiple branch envelopes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JoinResult {
    pub envelopes: std::collections::HashMap<String, Envelope>,
    pub failures: std::collections::HashMap<String, String>,
    pub status: EnvelopeStatus,
}

/// JoinNode: awaits multiple named state channels and merges results.
pub struct JoinNode {
    name: String,
    branches: Vec<String>,
    require_all: bool,
}

impl JoinNode {
    /// Create a new JoinNode. Defaults to `require_all: true` (fail-closed).
    ///
    /// Use `.allow_partial()` to explicitly opt into accepting partial results.
    pub fn new(name: impl Into<String>, branches: Vec<String>) -> Self {
        Self {
            name: name.into(),
            branches,
            require_all: true,
        }
    }

    /// Require all branches to succeed (default). Fail if any branch is missing.
    pub fn require_all(mut self) -> Self {
        self.require_all = true;
        self
    }

    /// Allow partial results — succeed even if some branches fail.
    /// Use only when partial results are explicitly acceptable.
    pub fn allow_partial(mut self) -> Self {
        self.require_all = false;
        self
    }

    pub fn input_channels(&self) -> Vec<String> {
        self.branches
            .iter()
            .map(|b| format!("{b}:envelope"))
            .collect()
    }
}

#[async_trait]
impl Node for JoinNode {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, ctx), fields(node = %self.name, branches = ?self.branches))]
    async fn execute(&self, ctx: &NodeContext) -> adk_rust::graph::error::Result<NodeOutput> {
        let mut envelopes = std::collections::HashMap::new();
        let mut failures = std::collections::HashMap::new();

        for branch in &self.branches {
            let channel = format!("{branch}:envelope");
            match ctx.get_as::<Envelope>(&channel) {
                Some(envelope) => {
                    debug!(node = %self.name, branch = %branch, "collected envelope");
                    envelopes.insert(branch.clone(), envelope);
                }
                None => {
                    let error_channel = format!("{branch}:error");
                    let error_msg = ctx
                        .get_as::<String>(&error_channel)
                        .unwrap_or_else(|| format!("no envelope in channel '{channel}'"));
                    warn!(node = %self.name, branch = %branch, error = %error_msg, "branch failed");
                    failures.insert(branch.clone(), error_msg);
                }
            }
        }

        let status = if failures.is_empty() {
            EnvelopeStatus::Complete
        } else if envelopes.is_empty() {
            EnvelopeStatus::Error
        } else {
            EnvelopeStatus::Partial
        };

        if self.require_all && !failures.is_empty() {
            return Err(adk_rust::graph::error::GraphError::NodeExecutionFailed {
                node: self.name.clone(),
                message: format!(
                    "required branches failed: {}",
                    failures.keys().cloned().collect::<Vec<_>>().join(", ")
                ),
            });
        }

        let join_result = JoinResult {
            envelopes,
            failures,
            status,
        };
        let result_json = serde_json::to_value(&join_result).unwrap_or(json!(null));

        debug!(
            node = %self.name,
            collected = join_result.envelopes.len(),
            failed = join_result.failures.len(),
            status = ?join_result.status,
            "join complete"
        );

        Ok(NodeOutput::new()
            .with_update(&format!("{}:result", self.name), result_json)
            .with_update(
                &format!("{}:status", self.name),
                json!(format!("{:?}", join_result.status)),
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_envelope(status: EnvelopeStatus, signal: &str) -> Envelope {
        Envelope {
            status,
            artifacts: vec![PathBuf::from("test.md")],
            signal: Some(signal.to_string()),
            tensions: None,
            emergent: None,
            next: None,
            context_update: None,
        }
    }

    #[tokio::test]
    async fn join_collects_all_branches() {
        let node = JoinNode::new("join", vec!["urdr".into(), "skuld".into()]);
        let mut state = State::new();
        state.insert(
            "urdr:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete, "past")).unwrap(),
        );
        state.insert(
            "skuld:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete, "future")).unwrap(),
        );

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        let result: JoinResult =
            serde_json::from_value(output.updates.get("join:result").unwrap().clone()).unwrap();

        assert_eq!(result.envelopes.len(), 2);
        assert!(result.failures.is_empty());
        assert_eq!(result.status, EnvelopeStatus::Complete);
    }

    #[tokio::test]
    async fn join_partial_failure_rejected_by_default() {
        // Default is require_all: true (fail-closed)
        let node = JoinNode::new("join", vec!["urdr".into(), "skuld".into()]);
        let mut state = State::new();
        state.insert(
            "urdr:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete, "past")).unwrap(),
        );

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let result = node.execute(&ctx).await;
        assert!(
            result.is_err(),
            "default require_all should reject partial results"
        );
    }

    #[tokio::test]
    async fn join_allow_partial_accepts_partial_failure() {
        let node = JoinNode::new("join", vec!["urdr".into(), "skuld".into()]).allow_partial();
        let mut state = State::new();
        state.insert(
            "urdr:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete, "past")).unwrap(),
        );

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        let result: JoinResult =
            serde_json::from_value(output.updates.get("join:result").unwrap().clone()).unwrap();

        assert_eq!(result.envelopes.len(), 1);
        assert_eq!(result.failures.len(), 1);
        assert!(result.failures.contains_key("skuld"));
        assert_eq!(result.status, EnvelopeStatus::Partial);
    }

    #[tokio::test]
    async fn join_errors_when_all_fail() {
        // With default require_all: true, all-fail returns error from require_all check
        let node = JoinNode::new("join", vec!["a".into(), "b".into()]);
        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test"), 0);
        let result = node.execute(&ctx).await;
        assert!(
            result.is_err(),
            "all branches failed with require_all should error"
        );
    }

    #[tokio::test]
    async fn join_allow_partial_errors_when_all_fail() {
        // With allow_partial, all-fail still reports Error status (not a hard error)
        let node = JoinNode::new("join", vec!["a".into(), "b".into()]).allow_partial();
        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        let result: JoinResult =
            serde_json::from_value(output.updates.get("join:result").unwrap().clone()).unwrap();

        assert!(result.envelopes.is_empty());
        assert_eq!(result.failures.len(), 2);
        assert_eq!(result.status, EnvelopeStatus::Error);
    }

    #[tokio::test]
    async fn join_require_all_rejects_partial() {
        let node = JoinNode::new("join", vec!["a".into(), "b".into()]).require_all();
        let mut state = State::new();
        state.insert(
            "a:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete, "done")).unwrap(),
        );

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let result = node.execute(&ctx).await;
        assert!(result.is_err());
        let err = match result {
            Err(e) => format!("{:?}", e),
            Ok(_) => panic!("expected error"),
        };
        assert!(err.contains("required branches failed"));
    }

    #[tokio::test]
    async fn join_reads_error_channel_on_failure() {
        let node = JoinNode::new("join", vec!["a".into()]).allow_partial();
        let mut state = State::new();
        state.insert("a:error".into(), json!("agent timed out"));

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        let result: JoinResult =
            serde_json::from_value(output.updates.get("join:result").unwrap().clone()).unwrap();
        assert!(result.failures.get("a").unwrap().contains("timed out"));
    }

    #[tokio::test]
    async fn join_input_channels_match_convention() {
        let node = JoinNode::new(
            "join",
            vec!["urdr".into(), "skuld".into(), "verdandi".into()],
        );
        assert_eq!(
            node.input_channels(),
            vec!["urdr:envelope", "skuld:envelope", "verdandi:envelope"]
        );
    }
}
