//! RouterNode — conditional dispatch based on state.
//!
//! Implements `adk_rust::graph::Node`. Reads state to evaluate a routing
//! predicate and returns a routing decision indicating which downstream
//! node to activate. Used for the `?` (conditional/select) operator.
//!
//! The router does not spawn agents — it reads state and decides.
//! The `_route:decision` channel is read by conditional edges in the graph.

use adk_rust::graph::prelude::*;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, instrument};

/// A routing predicate — determines which branch to activate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutePredicate {
    /// Route based on the value of a state channel matching a string.
    StateEquals {
        channel: String,
        value: serde_json::Value,
        /// Route name when predicate is true.
        on_match: String,
        /// Route name when predicate is false.
        on_miss: String,
    },
    /// Route based on whether a state channel exists (has a value).
    StateExists {
        channel: String,
        on_exists: String,
        on_missing: String,
    },
    /// Route based on the `_gate:route` channel (pass/fail).
    GateRoute { on_pass: String, on_fail: String },
}

/// RouterNode: reads state and writes routing decision.
pub struct RouterNode {
    name: String,
    predicate: RoutePredicate,
}

impl RouterNode {
    pub fn new(name: impl Into<String>, predicate: RoutePredicate) -> Self {
        Self {
            name: name.into(),
            predicate,
        }
    }
}

#[async_trait]
impl Node for RouterNode {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, ctx), fields(node = %self.name))]
    async fn execute(&self, ctx: &NodeContext) -> adk_rust::graph::error::Result<NodeOutput> {
        let decision = match &self.predicate {
            RoutePredicate::StateEquals {
                channel,
                value,
                on_match,
                on_miss,
            } => {
                let state_val = ctx.get(channel);
                let matched = state_val == Some(value);
                debug!(
                    node = %self.name,
                    channel = %channel,
                    matched,
                    "state-equals predicate"
                );
                if matched {
                    on_match.clone()
                } else {
                    on_miss.clone()
                }
            }
            RoutePredicate::StateExists {
                channel,
                on_exists,
                on_missing,
            } => {
                let exists = ctx.get(channel).is_some();
                debug!(
                    node = %self.name,
                    channel = %channel,
                    exists,
                    "state-exists predicate"
                );
                if exists {
                    on_exists.clone()
                } else {
                    on_missing.clone()
                }
            }
            RoutePredicate::GateRoute { on_pass, on_fail } => {
                let route = ctx
                    .get_as::<String>("_gate:route")
                    .unwrap_or_else(|| "fail".to_string());
                debug!(node = %self.name, route = %route, "gate-route predicate");
                if route == "pass" {
                    on_pass.clone()
                } else {
                    on_fail.clone()
                }
            }
        };

        debug!(node = %self.name, decision = %decision, "routing decision");

        Ok(NodeOutput::new().with_update("_route:decision", json!(decision)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn router_state_equals_matches() {
        let predicate = RoutePredicate::StateEquals {
            channel: "status".into(),
            value: json!("ready"),
            on_match: "proceed".into(),
            on_miss: "wait".into(),
        };
        let node = RouterNode::new("router", predicate);

        let mut state = State::new();
        state.insert("status".into(), json!("ready"));

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(
            output.updates.get("_route:decision"),
            Some(&json!("proceed"))
        );
    }

    #[tokio::test]
    async fn router_state_equals_misses() {
        let predicate = RoutePredicate::StateEquals {
            channel: "status".into(),
            value: json!("ready"),
            on_match: "proceed".into(),
            on_miss: "wait".into(),
        };
        let node = RouterNode::new("router", predicate);

        let mut state = State::new();
        state.insert("status".into(), json!("pending"));

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(output.updates.get("_route:decision"), Some(&json!("wait")));
    }

    #[tokio::test]
    async fn router_state_exists() {
        let predicate = RoutePredicate::StateExists {
            channel: "result:envelope".into(),
            on_exists: "has_result".into(),
            on_missing: "no_result".into(),
        };
        let node = RouterNode::new("router", predicate);

        // Channel exists
        let mut state = State::new();
        state.insert("result:envelope".into(), json!({"status": "Complete"}));
        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(
            output.updates.get("_route:decision"),
            Some(&json!("has_result"))
        );

        // Channel missing
        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(
            output.updates.get("_route:decision"),
            Some(&json!("no_result"))
        );
    }

    #[tokio::test]
    async fn router_gate_route_pass() {
        let predicate = RoutePredicate::GateRoute {
            on_pass: "done".into(),
            on_fail: "retry".into(),
        };
        let node = RouterNode::new("router", predicate);

        let mut state = State::new();
        state.insert("_gate:route".into(), json!("pass"));

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(output.updates.get("_route:decision"), Some(&json!("done")));
    }

    #[tokio::test]
    async fn router_gate_route_fail() {
        let predicate = RoutePredicate::GateRoute {
            on_pass: "done".into(),
            on_fail: "retry".into(),
        };
        let node = RouterNode::new("router", predicate);

        let mut state = State::new();
        state.insert("_gate:route".into(), json!("fail"));

        let ctx = NodeContext::new(state, ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(output.updates.get("_route:decision"), Some(&json!("retry")));
    }

    #[tokio::test]
    async fn router_defaults_to_fail_when_gate_missing() {
        let predicate = RoutePredicate::GateRoute {
            on_pass: "done".into(),
            on_fail: "retry".into(),
        };
        let node = RouterNode::new("router", predicate);

        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();
        assert_eq!(output.updates.get("_route:decision"), Some(&json!("retry")));
    }
}
