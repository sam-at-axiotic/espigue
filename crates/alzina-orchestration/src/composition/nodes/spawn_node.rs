//! SpawnNode — dispatches a single agent within a graph execution.
//!
//! Implements `adk_rust::graph::Node` to integrate with ADK's StateGraph.
//! Takes a `CompNode` + `Arc<AlzinaRunner>`, spawns the agent, writes
//! envelope to state channels.
//!
//! State channel convention:
//! - `{node_name}:envelope` — parsed Envelope from agent return
//! - `{node_name}:raw` — raw text output
//! - `{node_name}:status` — session status string

use std::sync::Arc;

use adk_rust::graph::prelude::*;
use async_trait::async_trait;
use serde_json::json;
use tracing::{debug, instrument};

use alzina_core::identity::{SessionId, WeaveId};

use crate::runner::alzina_runner::{AlzinaRunner, CompNode};

/// SpawnNode: dispatches a single agent within a graph execution.
pub struct SpawnNode {
    name: String,
    node: CompNode,
    runner: Arc<AlzinaRunner>,
    parent_session: Option<SessionId>,
    weave_id: Option<WeaveId>,
    /// P5-DEBUG-DISPATCH (Fix A): pre-allocated root session id from the
    /// outer dispatch path. When set, the runner uses this id rather
    /// than minting a fresh one, so mid-turn bus events
    /// (`TextDelta`, `TokenUsage`) carry the same id the streaming
    /// dispatch handler is filtering on.
    session_id_override: Option<SessionId>,
    /// D10-02: when set, the renderer is invoked at dispatch time to produce
    /// the preamble-prepended + channel-substituted task. When None, dispatch
    /// is byte-identical to the ad-hoc path (no preamble, no substitution).
    composition_context: Option<crate::composition::parser::CompositionContext>,
}

impl SpawnNode {
    pub fn new(name: impl Into<String>, node: CompNode, runner: Arc<AlzinaRunner>) -> Self {
        Self {
            name: name.into(),
            node,
            runner,
            parent_session: None,
            weave_id: None,
            session_id_override: None,
            composition_context: None,
        }
    }

    pub fn with_parent_session(mut self, session: SessionId) -> Self {
        self.parent_session = Some(session);
        self
    }

    pub fn with_weave_id(mut self, weave_id: WeaveId) -> Self {
        self.weave_id = Some(weave_id);
        self
    }

    /// P5-DEBUG-DISPATCH (Fix A): attach a pre-allocated session id from
    /// the caller. The runner consumes this id verbatim instead of
    /// minting its own; the caller MUST already have called
    /// `SessionHierarchy::create_root` with the same id.
    pub fn with_session_id_override(mut self, session_id: SessionId) -> Self {
        self.session_id_override = Some(session_id);
        self
    }

    /// D10-02: attach a CompositionContext so the runner invokes `render_task`
    /// before dispatch. When absent, dispatch is byte-identical to ad-hoc path.
    pub fn with_composition_context(
        mut self,
        ctx: crate::composition::parser::CompositionContext,
    ) -> Self {
        self.composition_context = Some(ctx);
        self
    }

    /// Override the task text on the inner `CompNode`. Used by the
    /// composition compiler to inject artifact-directory instructions
    /// returned by `CompositionLeafHook::on_leaf_dispatch`.
    pub fn with_task(mut self, task: String) -> Self {
        self.node.task = task;
        self
    }

    pub fn output_channels(&self) -> Vec<String> {
        vec![
            format!("{}:envelope", self.name),
            format!("{}:raw", self.name),
            format!("{}:status", self.name),
        ]
    }
}

#[async_trait]
impl Node for SpawnNode {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, _ctx), fields(node = %self.name, agent = %self.node.agent_id))]
    async fn execute(&self, _ctx: &NodeContext) -> adk_rust::graph::error::Result<NodeOutput> {
        debug!(node = %self.name, agent = %self.node.agent_id, "spawning agent");

        // P5-DEBUG-DISPATCH (Fix A): prefer `spawn_with_id` when the outer
        // dispatch path pre-allocated a session id; the runner then
        // stamps mid-turn bus events with that exact id so the streaming
        // dispatch handler's filter sees them.
        //
        // D10-02: when a composition_context is set AND a session_id_override
        // is set, pass the context to spawn_with_id so it renders the task.
        // When no session_id_override, the context is unused (ad-hoc path
        // through `spawn` doesn't support context — that path is unchanged).
        let result = match self.session_id_override.clone() {
            Some(session_id) => self
                .runner
                .spawn_with_id(
                    session_id,
                    &self.node,
                    self.parent_session.as_ref(),
                    self.weave_id.as_ref(),
                    // D10-02: clone the context (Arc fields are cheap; Clone now derived).
                    self.composition_context.clone(),
                )
                .await
                .map_err(
                    |e| adk_rust::graph::error::GraphError::NodeExecutionFailed {
                        node: self.name.clone(),
                        message: format!("spawn failed: {e}"),
                    },
                )?,
            None => self
                .runner
                .spawn(
                    &self.node,
                    self.parent_session.as_ref(),
                    self.weave_id.as_ref(),
                )
                .await
                .map_err(
                    |e| adk_rust::graph::error::GraphError::NodeExecutionFailed {
                        node: self.name.clone(),
                        message: format!("spawn failed: {e}"),
                    },
                )?,
        };

        let envelope_json = serde_json::to_value(&result.envelope).unwrap_or(json!(null));

        Ok(NodeOutput::new()
            .with_update(&format!("{}:envelope", self.name), envelope_json)
            .with_update(&format!("{}:raw", self.name), json!(result.raw))
            .with_update(
                &format!("{}:status", self.name),
                json!(format!("{:?}", result.envelope.status)),
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::alzina_runner::AgentExecutor;
    use crate::test_helpers::{build_test_runner, well_formed_envelope};
    use alzina_core::identity::AgentId;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::sync::Mutex;

    struct MockExecutor {
        response: Mutex<String>,
        executed: AtomicBool,
    }

    impl MockExecutor {
        fn new(response: &str) -> Self {
            Self {
                response: Mutex::new(response.to_string()),
                executed: AtomicBool::new(false),
            }
        }

        fn was_executed(&self) -> bool {
            self.executed.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl AgentExecutor for MockExecutor {
        async fn execute(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
        ) -> alzina_core::AlzinaResult<String> {
            self.executed.store(true, Ordering::SeqCst);
            Ok(self.response.lock().await.clone())
        }
    }

    fn test_comp_node(agent: &str, task: &str) -> CompNode {
        CompNode {
            agent_id: AgentId::new(agent),
            task: task.to_string(),
            model_override: None,
            timeout: None,
            weave_id: None,
            dispatch_id: None,
            sampling: None,
        }
    }

    #[tokio::test]
    async fn spawn_node_writes_state_channels() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&well_formed_envelope()));
        let runner = Arc::new(
            build_test_runner(executor.clone(), None, dir.path())
                .await
                .unwrap(),
        );

        let comp = test_comp_node("smidr", "analyse workspace");
        let spawn_node = SpawnNode::new("smidr_analysis", comp, runner);

        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test-thread"), 0);
        let output = spawn_node.execute(&ctx).await.unwrap();

        assert!(output.updates.contains_key("smidr_analysis:envelope"));
        assert!(output.updates.contains_key("smidr_analysis:raw"));
        assert!(output.updates.contains_key("smidr_analysis:status"));
        assert!(executor.was_executed());

        let status = output.updates.get("smidr_analysis:status").unwrap();
        assert_eq!(status, &json!("Complete"));
    }

    #[tokio::test]
    async fn spawn_node_propagates_error_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new("garbage with no structure"));
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());

        let comp = test_comp_node("confused", "produce garbage");
        // Use a parent session so the spawn is treated as sub-agent (strict envelope parsing)
        let spawn_node = SpawnNode::new("confused_node", comp, runner)
            .with_parent_session(alzina_core::SessionId::new());

        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test-thread"), 0);
        let result = spawn_node.execute(&ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn spawn_node_output_channels_match_convention() {
        let comp = test_comp_node("galdr", "build");
        let runner = Arc::new({
            let dir = tempfile::tempdir().unwrap();
            let executor = Arc::new(MockExecutor::new(&well_formed_envelope()));
            build_test_runner(executor, None, dir.path()).await.unwrap()
        });

        let node = SpawnNode::new("galdr_build", comp, runner);
        let channels = node.output_channels();
        assert_eq!(
            channels,
            vec![
                "galdr_build:envelope",
                "galdr_build:raw",
                "galdr_build:status",
            ]
        );
    }
}
