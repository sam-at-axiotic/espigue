//! SynthesisNode — merges branch results and spawns a synthesiser agent.
//!
//! Implements `adk_rust::graph::Node`. Reads envelopes from multiple
//! upstream state channels (like JoinNode), builds a synthesis task
//! combining all branch signals/envelopes, spawns a synthesiser agent
//! via AlzinaRunner, and writes the synthesiser's envelope to its own
//! state channel.
//!
//! Used for the `⊕` (synthesis) operator in the composition algebra.
//!
//! # Compiler Duplication Notice
//!
//! This node is used directly for ADK integration (via `adk_rust::graph::Node` trait).
//! The compiler (`compiler.rs`) reimplements this logic for recursive composition.
//! **Security fixes must be applied to both paths.** See RT2-08 for details.

use std::collections::HashMap;
use std::sync::Arc;

use adk_rust::graph::prelude::*;
use async_trait::async_trait;
use serde_json::json;
use tracing::{debug, instrument, warn};

use alzina_core::envelope::Envelope;
use alzina_core::identity::{SessionId, WeaveId};

use crate::composition::parser::{AncestorSummary, CompositionContext};
use crate::runner::alzina_runner::{AlzinaRunner, CompNode};

/// SynthesisNode: reads branch results from state, spawns synthesiser.
pub struct SynthesisNode {
    name: String,
    synthesiser: CompNode,
    branch_channels: Vec<String>,
    runner: Arc<AlzinaRunner>,
    parent_session: Option<SessionId>,
    weave_id: Option<WeaveId>,
    /// E1 / D11-04: optional pre-allocated SessionId so the composition
    /// path can publish SessionCompleted/SessionFailed against the SAME
    /// id register_dispatch is watching.
    session_id_override: Option<SessionId>,
    /// Bug 2: parent ancestors ∪ inner-op descendants (per §4.4) so the
    /// synthesiser sees both upstream context AND the branch summaries
    /// it is meant to merge. Consumed by execute() when building the
    /// synth spawn's CompositionContext (Bug 4).
    ancestors: Vec<AncestorSummary>,
    /// Bug 4: when set, this CompositionContext is forwarded to
    /// `AlzinaRunner::spawn_with_id` so the renderer applies §4.3
    /// preamble + §4.2 channel substitutions to the synth's task text.
    /// Pre-fix the synth was always dispatched with `composition_context:
    /// None`, leaving `{channel:envelope}` references in the synth's
    /// task literal and skipping the upstream preamble entirely.
    composition_context: Option<CompositionContext>,
}

impl SynthesisNode {
    pub fn new(
        name: impl Into<String>,
        synthesiser: CompNode,
        branch_channels: Vec<String>,
        runner: Arc<AlzinaRunner>,
    ) -> Self {
        Self {
            name: name.into(),
            synthesiser,
            branch_channels,
            runner,
            parent_session: None,
            weave_id: None,
            session_id_override: None,
            ancestors: Vec::new(),
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

    /// E1 / D11-04: bind a pre-allocated SessionId so the synthesiser leaf
    /// completes against the same id `register_dispatch` is watching.
    pub fn with_session_id_override(mut self, session_id: SessionId) -> Self {
        self.session_id_override = Some(session_id);
        self
    }

    /// Bug 2: thread parent ancestors ∪ inner-op descendants into the
    /// synthesiser so its dispatch carries §4.3 preamble context AND
    /// the branch summaries it is meant to merge.
    pub fn with_ancestors(mut self, ancestors: Vec<AncestorSummary>) -> Self {
        self.ancestors = ancestors;
        self
    }

    /// Bug 4: bind the CompositionContext used for renderer dispatch.
    /// When present AND `session_id_override` is set, `execute()` forwards
    /// it through `spawn_with_id` so the renderer applies the §4.3
    /// preamble + §4.2 channel substitutions to the synth's task.
    pub fn with_composition_context(mut self, ctx: CompositionContext) -> Self {
        self.composition_context = Some(ctx);
        self
    }

    /// Read-only accessor for tests verifying ancestors propagation.
    #[cfg(test)]
    pub fn ancestors(&self) -> &[AncestorSummary] {
        &self.ancestors
    }

    /// Read-only accessor for tests verifying composition_context wiring.
    #[cfg(test)]
    pub fn composition_context(&self) -> Option<&CompositionContext> {
        self.composition_context.as_ref()
    }

    /// Override the synthesiser's task text. Used by the composition
    /// compiler to inject artifact-directory instructions returned by
    /// `CompositionLeafHook::on_leaf_dispatch`.
    pub fn with_task(mut self, task: String) -> Self {
        self.synthesiser.task = task;
        self
    }

    pub fn output_channels(&self) -> Vec<String> {
        vec![
            format!("{}:envelope", self.name),
            format!("{}:raw", self.name),
            format!("{}:status", self.name),
        ]
    }

    /// Sanitise error text before including in LLM context.
    ///
    /// Strips file paths, stack traces, and internal module names that could
    /// leak system internals to the LLM (and potentially into agent output).
    fn sanitise_error_text(error: &str) -> String {
        let mut lines_out: Vec<String> = Vec::new();
        for line in error.lines() {
            let trimmed = line.trim();
            // Skip lines that look like stack traces
            if trimmed.starts_with("at ") || trimmed.starts_with("stack backtrace:") {
                continue;
            }
            // Skip stack frame lines (e.g. "  0: alzina_core::...")
            if let Some(colon_pos) = trimmed.find(": ") {
                let prefix = &trimmed[..colon_pos];
                if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
                    continue;
                }
            }
            let mut cleaned = line.to_string();
            // Redact Rust module paths (word::word::word+)
            while let Some(pos) = cleaned.find("::") {
                // Walk backwards to find start of identifier
                let start = cleaned[..pos]
                    .rfind(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .map(|i| i + 1)
                    .unwrap_or(0);
                // Walk forwards past all ::segments
                let mut end = pos + 2;
                loop {
                    // Read identifier segment
                    while end < cleaned.len() {
                        let ch = cleaned.as_bytes()[end] as char;
                        if ch.is_ascii_alphanumeric() || ch == '_' {
                            end += 1;
                        } else {
                            break;
                        }
                    }
                    // Check for another ::
                    if end + 1 < cleaned.len() && &cleaned[end..end + 2] == "::" {
                        end += 2;
                    } else {
                        break;
                    }
                }
                let module_path = &cleaned[start..end];
                if module_path.matches("::").count() >= 2 {
                    cleaned = format!("{}[module-redacted]{}", &cleaned[..start], &cleaned[end..]);
                } else {
                    // Not enough segments — skip this occurrence
                    // Replace just the :: to avoid infinite loop
                    cleaned = format!("{}%%COLON%%{}", &cleaned[..pos], &cleaned[pos + 2..]);
                }
            }
            cleaned = cleaned.replace("%%COLON%%", "::");
            lines_out.push(cleaned);
        }
        lines_out.join("\n")
    }

    /// Collect branch envelopes from state, returning (found, failed).
    fn collect_branches(
        &self,
        ctx: &NodeContext,
    ) -> (HashMap<String, Envelope>, HashMap<String, String>) {
        let mut envelopes = HashMap::new();
        let mut failures = HashMap::new();

        for channel in &self.branch_channels {
            let envelope_key = format!("{channel}:envelope");
            match ctx.get_as::<Envelope>(&envelope_key) {
                Some(envelope) => {
                    debug!(node = %self.name, branch = %channel, "collected branch envelope");
                    envelopes.insert(channel.clone(), envelope);
                }
                None => {
                    let error_key = format!("{channel}:error");
                    let error_msg = ctx
                        .get_as::<String>(&error_key)
                        .unwrap_or_else(|| format!("no envelope in channel '{envelope_key}'"));
                    warn!(node = %self.name, branch = %channel, error = %error_msg, "branch missing");
                    failures.insert(channel.clone(), error_msg);
                }
            }
        }

        (envelopes, failures)
    }

    /// Build the synthesis task text from collected branch envelopes.
    ///
    /// When `self.synthesiser.task` is empty (set by the compiler when
    /// SynthesisSpec.task drives resolution — see D10-13), falls back to
    /// the §3.4 default prompt so the ADK-integration path (this node)
    /// produces the same anti-averaging discipline cue as the
    /// recursive-executor path.
    fn build_synthesis_task(
        &self,
        envelopes: &HashMap<String, Envelope>,
        failures: &HashMap<String, String>,
    ) -> String {
        // Resolve synthesis task: use the set task, or fall back to §3.4 default.
        let resolved_task = Some(self.synthesiser.task.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| crate::composition::compiler::DEFAULT_SYNTHESIS_PROMPT);
        let mut task = format!("{resolved_task}\n\n## Branch Results for Synthesis\n\n");

        for (branch, envelope) in envelopes {
            task.push_str(&format!("### Branch: {branch}\n"));
            task.push_str(&format!("- Status: {:?}\n", envelope.status));
            if let Some(ref signal) = envelope.signal {
                task.push_str(&format!("- Signal: {signal}\n"));
            }
            if let Some(ref tensions) = envelope.tensions {
                task.push_str(&format!("- Tensions: {tensions}\n"));
            }
            if let Some(ref emergent) = envelope.emergent {
                task.push_str(&format!("- Emergent: {emergent}\n"));
            }
            // Bug 3: pre-fix this method emitted only status/signal/
            // tensions/emergent, so file paths the children wrote and the
            // next-step / context-update hints they surfaced were dropped
            // before reaching the synthesiser. Emit the remaining
            // Envelope fields (see alzina-core/src/envelope.rs:20-28).
            if !envelope.artifacts.is_empty() {
                task.push_str("- Artifacts:\n");
                for path in &envelope.artifacts {
                    task.push_str(&format!("    {}\n", path.display()));
                }
            }
            if let Some(ref next) = envelope.next {
                task.push_str(&format!("- Next: {next}\n"));
            }
            if let Some(ref context_update) = envelope.context_update {
                task.push_str(&format!("- Context update: {context_update}\n"));
            }
            task.push('\n');
        }

        if !failures.is_empty() {
            task.push_str("### Failed Branches\n\n");
            for (branch, error) in failures {
                let sanitised = Self::sanitise_error_text(error);
                task.push_str(&format!("- {branch}: {sanitised}\n"));
            }
            task.push('\n');
        }

        task
    }
}

#[async_trait]
impl Node for SynthesisNode {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, ctx), fields(node = %self.name, agent = %self.synthesiser.agent_id, branches = ?self.branch_channels))]
    async fn execute(&self, ctx: &NodeContext) -> adk_rust::graph::error::Result<NodeOutput> {
        let (envelopes, failures) = self.collect_branches(ctx);

        debug!(
            node = %self.name,
            collected = envelopes.len(),
            failed = failures.len(),
            "collected branch results for synthesis"
        );

        // Build the enriched synthesis task
        let synthesis_task = self.build_synthesis_task(&envelopes, &failures);

        // Create a CompNode with the enriched task
        let synth_node = CompNode {
            agent_id: self.synthesiser.agent_id.clone(),
            task: synthesis_task,
            model_override: self.synthesiser.model_override.clone(),
            timeout: self.synthesiser.timeout,
            weave_id: self.weave_id.clone(),
            // Phase 1B substrate cascade: forward the synthesiser spec's
            // dispatch_id so a synthesis stitch fired by this node carries
            // the enclosing chat-tool dispatch attribution.
            dispatch_id: self.synthesiser.dispatch_id.clone(),
            sampling: None,
        };

        // Spawn the synthesiser agent. E1 / D11-04: when session_id_override
        // is set, use spawn_with_id so the daemon's register_dispatch
        // watcher sees the SessionCompleted on the SAME id it's watching.
        //
        // Bug 4: forward the bound CompositionContext (set by
        // execute_synthesis in compiler.rs) so AlzinaRunner::spawn_with_id
        // at alzina_runner.rs:427-429 renders the §4.3 preamble and
        // substitutes `{channel:envelope}` references in the synth's task.
        // Pre-fix this argument was always None.
        let result = match self.session_id_override.clone() {
            Some(sid) => self
                .runner
                .spawn_with_id(
                    sid,
                    &synth_node,
                    self.parent_session.as_ref(),
                    self.weave_id.as_ref(),
                    self.composition_context.clone(),
                )
                .await
                .map_err(
                    |e| adk_rust::graph::error::GraphError::NodeExecutionFailed {
                        node: self.name.clone(),
                        message: format!("synthesis spawn failed: {e}"),
                    },
                )?,
            None => self
                .runner
                .spawn(
                    &synth_node,
                    self.parent_session.as_ref(),
                    self.weave_id.as_ref(),
                )
                .await
                .map_err(
                    |e| adk_rust::graph::error::GraphError::NodeExecutionFailed {
                        node: self.name.clone(),
                        message: format!("synthesis spawn failed: {e}"),
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
    use alzina_core::envelope::EnvelopeStatus;
    use alzina_core::identity::AgentId;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    struct MockSynthExecutor {
        response: Mutex<String>,
        call_count: AtomicUsize,
        last_task: Mutex<Option<String>>,
    }

    impl MockSynthExecutor {
        fn new(response: &str) -> Self {
            Self {
                response: Mutex::new(response.to_string()),
                call_count: AtomicUsize::new(0),
                last_task: Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl AgentExecutor for MockSynthExecutor {
        async fn execute(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            task: &str,
        ) -> alzina_core::AlzinaResult<String> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            *self.last_task.lock().await = Some(task.to_string());
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

    fn make_envelope(status: EnvelopeStatus, signal: &str) -> Envelope {
        Envelope {
            status,
            artifacts: vec![],
            signal: Some(signal.to_string()),
            tensions: None,
            emergent: None,
            next: None,
            context_update: None,
        }
    }

    #[tokio::test]
    async fn synthesis_two_branches_produces_combined_result() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockSynthExecutor::new(&well_formed_envelope()));
        let runner = Arc::new(
            build_test_runner(executor.clone(), None, dir.path())
                .await
                .unwrap(),
        );

        let synth = test_comp_node("vefr", "synthesise branch results");
        let node = SynthesisNode::new(
            "synthesis",
            synth,
            vec!["urdr".into(), "skuld".into()],
            runner,
        );

        let mut state = State::new();
        state.insert(
            "urdr:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete, "past context")).unwrap(),
        );
        state.insert(
            "skuld:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete, "future plan")).unwrap(),
        );

        let ctx = NodeContext::new(state, ExecutionConfig::new("test-thread"), 0);
        let output = node.execute(&ctx).await.unwrap();

        // Verify outputs written
        assert!(output.updates.contains_key("synthesis:envelope"));
        assert!(output.updates.contains_key("synthesis:raw"));
        assert!(output.updates.contains_key("synthesis:status"));

        // Verify synthesiser was called with branch context
        let last_task = executor.last_task.lock().await;
        let task_text = last_task.as_ref().unwrap();
        assert!(
            task_text.contains("past context"),
            "task should contain urdr signal"
        );
        assert!(
            task_text.contains("future plan"),
            "task should contain skuld signal"
        );
        assert!(task_text.contains("Branch: urdr"));
        assert!(task_text.contains("Branch: skuld"));
    }

    #[tokio::test]
    async fn synthesis_handles_partial_branch_failure() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockSynthExecutor::new(&well_formed_envelope()));
        let runner = Arc::new(
            build_test_runner(executor.clone(), None, dir.path())
                .await
                .unwrap(),
        );

        let synth = test_comp_node("vefr", "synthesise with partial results");
        let node = SynthesisNode::new(
            "synthesis",
            synth,
            vec!["urdr".into(), "skuld".into()],
            runner,
        );

        // Only urdr present — skuld failed
        let mut state = State::new();
        state.insert(
            "urdr:envelope".into(),
            serde_json::to_value(make_envelope(EnvelopeStatus::Complete, "past context")).unwrap(),
        );
        state.insert("skuld:error".into(), json!("agent timed out"));

        let ctx = NodeContext::new(state, ExecutionConfig::new("test-thread"), 0);
        let output = node.execute(&ctx).await.unwrap();

        // Synthesiser still runs with partial results
        assert!(output.updates.contains_key("synthesis:envelope"));

        let last_task = executor.last_task.lock().await;
        let task_text = last_task.as_ref().unwrap();
        assert!(
            task_text.contains("past context"),
            "should include successful branch"
        );
        assert!(
            task_text.contains("Failed Branches"),
            "should report failures"
        );
        assert!(
            task_text.contains("timed out"),
            "should include error detail"
        );
    }

    #[tokio::test]
    async fn synthesis_output_channels_match_convention() {
        let synth = test_comp_node("vefr", "synth");
        let runner = Arc::new({
            let dir = tempfile::tempdir().unwrap();
            let executor = Arc::new(MockSynthExecutor::new(&well_formed_envelope()));
            build_test_runner(executor, None, dir.path()).await.unwrap()
        });

        let node = SynthesisNode::new("synth_node", synth, vec!["a".into()], runner);
        let channels = node.output_channels();
        assert_eq!(
            channels,
            vec!["synth_node:envelope", "synth_node:raw", "synth_node:status",]
        );
    }

    /// Bug 3 regression: `build_synthesis_task` MUST emit artifacts,
    /// next, and context_update per branch, not only status/signal/
    /// tensions/emergent. Without these fields the synthesiser cannot
    /// see file paths the children wrote, their next-step hints, or
    /// their context-update learnings.
    #[tokio::test]
    async fn build_synthesis_task_emits_artifacts_next_context_update() {
        use std::path::PathBuf;

        let synth = test_comp_node("vefr", "");
        let runner = Arc::new({
            let dir = tempfile::tempdir().unwrap();
            let executor = Arc::new(MockSynthExecutor::new(&well_formed_envelope()));
            build_test_runner(executor, None, dir.path()).await.unwrap()
        });

        let node = SynthesisNode::new("synth", synth, vec!["urdr".into()], runner);

        let mut envelopes = HashMap::new();
        envelopes.insert(
            "urdr".to_string(),
            Envelope {
                status: EnvelopeStatus::Complete,
                artifacts: vec![PathBuf::from("artifacts/probe_u_output.md")],
                signal: Some("ALPHA".into()),
                tensions: Some("none".into()),
                emergent: Some("none".into()),
                next: Some("review skuld branch".into()),
                context_update: Some("urdr learning: timestamps matter".into()),
            },
        );
        let failures = HashMap::new();

        let task = node.build_synthesis_task(&envelopes, &failures);

        assert!(
            task.contains("artifacts/probe_u_output.md"),
            "task must list per-branch artifact paths; task was:\n{task}"
        );
        assert!(
            task.contains("review skuld branch"),
            "task must include per-branch `next` hint; task was:\n{task}"
        );
        assert!(
            task.contains("urdr learning: timestamps matter"),
            "task must include per-branch context_update; task was:\n{task}"
        );
    }

    /// Bug 2 regression: ancestors threaded via `with_ancestors` MUST be
    /// stored on the SynthesisNode so downstream code (Bug 4's
    /// composition_context wiring) can read them when dispatching the
    /// synthesiser. Pre-fix, `execute_synthesis` discarded the ancestors
    /// list via `let _ = ancestors;` and the synth had no §4.3 preamble.
    #[tokio::test]
    async fn with_ancestors_stores_descendant_summaries() {
        let synth = test_comp_node("vefr", "synth");
        let runner = Arc::new({
            let dir = tempfile::tempdir().unwrap();
            let executor = Arc::new(MockSynthExecutor::new(&well_formed_envelope()));
            build_test_runner(executor, None, dir.path()).await.unwrap()
        });

        let ancestors = vec![
            AncestorSummary {
                node_id: "probe_u".into(),
                agent: "urdr".into(),
                status: EnvelopeStatus::Complete,
                signal: Some("ALPHA".into()),
                artifact_paths: vec![],
                emergent: None,
                next: None,
            },
            AncestorSummary {
                node_id: "probe_s".into(),
                agent: "skuld".into(),
                status: EnvelopeStatus::Complete,
                signal: Some("BETA".into()),
                artifact_paths: vec![],
                emergent: None,
                next: None,
            },
        ];

        let node = SynthesisNode::new("sjofn_synth", synth, vec![], runner)
            .with_ancestors(ancestors.clone());

        let stored = node.ancestors();
        assert_eq!(stored.len(), 2, "both ancestors must be stored");
        assert_eq!(stored[0].node_id, "probe_u");
        assert_eq!(stored[0].signal.as_deref(), Some("ALPHA"));
        assert_eq!(stored[1].node_id, "probe_s");
        assert_eq!(stored[1].signal.as_deref(), Some("BETA"));
    }
}
