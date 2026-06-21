//! FanOutNode — spawns one agent with N prompts concurrently.
//!
//! Implements `adk_rust::graph::Node`. Takes a single agent and N prompts,
//! spawns N concurrent instances via `tokio::JoinSet`, collects all results
//! into state channels (one per prompt). Configurable: fail-fast or collect-all.
//!
//! Used for the `[n]` (fan-out) operator in the composition algebra.

use std::sync::Arc;

use adk_rust::graph::prelude::*;
use async_trait::async_trait;
use serde_json::json;
use tracing::{debug, error, instrument, warn};

use alzina_core::identity::{AgentId, SessionId, WeaveId};

use crate::runner::alzina_runner::{AlzinaRunner, CompNode};

/// Strategy for handling failures in fan-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanOutStrategy {
    /// Abort all remaining on first failure.
    FailFast,
    /// Collect all results, including errors.
    CollectAll,
}

/// FanOutNode: spawns same agent with N different prompts concurrently.
///
/// # Naming Convention (RT3-13)
///
/// Output channels use `{name}:{index}:{field}` convention. The `name` prefix
/// is assigned by the compiler via `next_name("fanout")`, which includes a
/// monotonic counter to prevent channel-name collisions across fan-out nodes
/// in the same composition graph.
pub struct FanOutNode {
    name: String,
    agent_id: AgentId,
    prompts: Vec<String>,
    runner: Arc<AlzinaRunner>,
    strategy: FanOutStrategy,
    parent_session: Option<SessionId>,
    weave_id: Option<WeaveId>,
}

impl FanOutNode {
    pub fn new(
        name: impl Into<String>,
        agent_id: AgentId,
        prompts: Vec<String>,
        runner: Arc<AlzinaRunner>,
    ) -> Self {
        Self {
            name: name.into(),
            agent_id,
            prompts,
            runner,
            strategy: FanOutStrategy::CollectAll,
            parent_session: None,
            weave_id: None,
        }
    }

    pub fn with_strategy(mut self, strategy: FanOutStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_parent_session(mut self, session: SessionId) -> Self {
        self.parent_session = Some(session);
        self
    }

    pub fn with_weave_id(mut self, weave_id: WeaveId) -> Self {
        self.weave_id = Some(weave_id);
        self
    }

    pub fn output_channels(&self) -> Vec<String> {
        let mut channels = Vec::new();
        for i in 0..self.prompts.len() {
            channels.push(format!("{}:{i}:envelope", self.name));
            channels.push(format!("{}:{i}:raw", self.name));
            channels.push(format!("{}:{i}:status", self.name));
        }
        channels.push(format!("{}:count", self.name));
        channels
    }
}

#[async_trait]
impl Node for FanOutNode {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, _ctx), fields(node = %self.name, agent = %self.agent_id, prompts = self.prompts.len()))]
    async fn execute(&self, _ctx: &NodeContext) -> adk_rust::graph::error::Result<NodeOutput> {
        debug!(node = %self.name, count = self.prompts.len(), "fanning out");

        let mut join_set = tokio::task::JoinSet::new();

        // Spawn all prompts concurrently
        for (idx, prompt) in self.prompts.iter().enumerate() {
            let runner = self.runner.clone();
            let parent = self.parent_session.clone();
            let weave = self.weave_id.clone();
            let comp_node = CompNode {
                agent_id: self.agent_id.clone(),
                task: prompt.clone(),
                model_override: None,
                timeout: None,
                weave_id: self.weave_id.clone(),
                // Phase 1B substrate cascade: FanoutNode (the simpler
                // builder-style node, distinct from the compiler's
                // execute_fanout path) has no dispatch_id field today.
                // None is correct — this node type is not used on the
                // chat-tool path that allocates dispatch_ids.
                dispatch_id: None,
                sampling: None,
            };

            join_set.spawn(async move {
                let result = runner
                    .spawn(&comp_node, parent.as_ref(), weave.as_ref())
                    .await;
                (idx, result)
            });
        }

        let mut output = NodeOutput::new();
        let mut errors = Vec::new();

        // Collect results
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok((idx, Ok(spawn_result))) => {
                    debug!(node = %self.name, idx, "fan-out prompt completed");
                    let envelope_json =
                        serde_json::to_value(&spawn_result.envelope).unwrap_or(json!(null));
                    output = output
                        .with_update(&format!("{}:{idx}:envelope", self.name), envelope_json)
                        .with_update(&format!("{}:{idx}:raw", self.name), json!(spawn_result.raw))
                        .with_update(
                            &format!("{}:{idx}:status", self.name),
                            json!(format!("{:?}", spawn_result.envelope.status)),
                        );
                }
                Ok((idx, Err(e))) => {
                    let msg = format!("fan-out prompt {idx} failed: {e}");
                    warn!(node = %self.name, idx, error = %e, "fan-out prompt failed");
                    output = output.with_update(&format!("{}:{idx}:error", self.name), json!(msg));
                    errors.push((idx, msg));

                    if self.strategy == FanOutStrategy::FailFast {
                        error!(node = %self.name, "fail-fast: aborting remaining prompts");
                        join_set.abort_all();
                        return Err(adk_rust::graph::error::GraphError::NodeExecutionFailed {
                            node: self.name.clone(),
                            message: format!(
                                "fan-out aborted (fail-fast): prompt {idx} failed: {e}"
                            ),
                        });
                    }
                }
                Err(join_err) => {
                    // tokio JoinError — task panicked or was cancelled
                    warn!(node = %self.name, error = %join_err, "fan-out task join error");
                    errors.push((usize::MAX, format!("join error: {join_err}")));
                }
            }
        }

        output = output.with_update(&format!("{}:count", self.name), json!(self.prompts.len()));

        if !errors.is_empty() {
            output = output.with_update(
                &format!("{}:errors", self.name),
                json!(
                    errors
                        .iter()
                        .map(|(i, e)| format!("[{i}] {e}"))
                        .collect::<Vec<_>>()
                ),
            );
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::alzina_runner::AgentExecutor;
    use crate::test_helpers::{build_test_runner, well_formed_envelope};
    use alzina_core::identity::AgentId;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    struct MockFanOutExecutor {
        responses: Mutex<Vec<std::result::Result<String, String>>>,
        call_count: AtomicUsize,
    }

    impl MockFanOutExecutor {
        fn all_succeed(count: usize) -> Self {
            let responses = (0..count).map(|_| Ok(well_formed_envelope())).collect();
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicUsize::new(0),
            }
        }

        fn with_failure_at(count: usize, fail_idx: usize) -> Self {
            let responses = (0..count)
                .map(|i| {
                    if i == fail_idx {
                        Err("simulated failure".to_string())
                    } else {
                        Ok(well_formed_envelope())
                    }
                })
                .collect();
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl AgentExecutor for MockFanOutExecutor {
        async fn execute(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
        ) -> alzina_core::AlzinaResult<String> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let responses: tokio::sync::MutexGuard<'_, Vec<std::result::Result<String, String>>> =
                self.responses.lock().await;
            match responses.get(idx) {
                Some(Ok(resp)) => Ok(resp.clone()),
                Some(Err(e)) => Err(alzina_core::AlzinaError::Orchestration(e.clone())),
                None => Ok(well_formed_envelope()),
            }
        }
    }

    #[tokio::test]
    async fn fanout_three_prompts_all_succeed() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockFanOutExecutor::all_succeed(3));
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());

        let node = FanOutNode::new(
            "fan",
            AgentId::new("kvasir"),
            vec!["prompt A".into(), "prompt B".into(), "prompt C".into()],
            runner,
        );

        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();

        // All three prompts should have envelope channels
        assert!(output.updates.contains_key("fan:0:envelope"));
        assert!(output.updates.contains_key("fan:1:envelope"));
        assert!(output.updates.contains_key("fan:2:envelope"));
        assert_eq!(output.updates.get("fan:count"), Some(&json!(3)));
    }

    #[tokio::test]
    async fn fanout_collect_all_with_failure() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockFanOutExecutor::with_failure_at(3, 1));
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());

        let node = FanOutNode::new(
            "fan",
            AgentId::new("kvasir"),
            vec!["prompt A".into(), "prompt B".into(), "prompt C".into()],
            runner,
        )
        .with_strategy(FanOutStrategy::CollectAll);

        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test"), 0);
        let output = node.execute(&ctx).await.unwrap();

        // Should complete despite one failure
        assert!(output.updates.contains_key("fan:count"));
        // Error channel should exist for exactly one failed prompt
        // (index may vary — JoinSet execution order is non-deterministic)
        let error_keys: Vec<_> = output
            .updates
            .keys()
            .filter(|k| k.starts_with("fan:") && k.ends_with(":error"))
            .collect();
        assert_eq!(
            error_keys.len(),
            1,
            "expected exactly one error channel, got: {error_keys:?}"
        );
    }

    #[tokio::test]
    async fn fanout_fail_fast_aborts_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        // Make the first call fail immediately
        let executor = Arc::new(MockFanOutExecutor::with_failure_at(3, 0));
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());

        let node = FanOutNode::new(
            "fan",
            AgentId::new("kvasir"),
            vec!["prompt A".into(), "prompt B".into(), "prompt C".into()],
            runner,
        )
        .with_strategy(FanOutStrategy::FailFast);

        let ctx = NodeContext::new(State::new(), ExecutionConfig::new("test"), 0);
        let result = node.execute(&ctx).await;
        assert!(result.is_err());
        let err = match result {
            Err(e) => format!("{:?}", e),
            Ok(_) => panic!("expected error"),
        };
        assert!(
            err.contains("fail-fast"),
            "error should mention fail-fast: {err}"
        );
    }

    #[tokio::test]
    async fn fanout_output_channels_match_convention() {
        let runner = Arc::new({
            let dir = tempfile::tempdir().unwrap();
            let executor = Arc::new(MockFanOutExecutor::all_succeed(2));
            build_test_runner(executor, None, dir.path()).await.unwrap()
        });

        let node = FanOutNode::new(
            "fan",
            AgentId::new("kvasir"),
            vec!["a".into(), "b".into()],
            runner,
        );
        let channels = node.output_channels();
        assert!(channels.contains(&"fan:0:envelope".to_string()));
        assert!(channels.contains(&"fan:1:envelope".to_string()));
        assert!(channels.contains(&"fan:count".to_string()));
    }
}
