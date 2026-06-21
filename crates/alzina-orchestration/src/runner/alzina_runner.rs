//! AlzinaRunner — core agent dispatch lifecycle.
//!
//! Orchestrates the full spawn lifecycle: governance check → session creation →
//! bootstrap assembly → model resolution → agent execution → envelope processing
//! → learnings merge → session completion.
//!
//! The actual LLM execution is abstracted behind the `AgentExecutor` trait,
//! enabling full lifecycle testing without LLM calls.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use tracing::{debug, error, info, instrument, warn};

use alzina_core::bootstrap::{BootstrapPipeline, SessionType};
use alzina_core::envelope::{Envelope, QualityIssue, Signal};
use alzina_core::event::{SpawnCompleted, SpawnEventSink};
use alzina_core::hooks::{LifecycleEvent, OrlogSummary};
use alzina_core::identity::{AgentId, SessionId, WeaveId};
use alzina_core::{AlzinaError, AlzinaEvent, AlzinaResult};

use alzina_governance::{AgentIdentity, BootstrapEngine, GovernanceLayer, LearningsMerger};
use alzina_memory::SignalRecordsStore;
use alzina_workspace::WorkspaceHandle;

use crate::runner::model_resolver;
use crate::runner::stop_conditions::StopConditionEvaluator;
use crate::session::hierarchy::SessionHierarchy;

// `AgentExecutor`, `SamplingParams`, and `ExecutorEventEmitter` now live in
// `crate::executor` (always compiled, no governance deps). Re-exported here so
// the original `crate::runner::alzina_runner::*` paths keep resolving.
pub use crate::executor::{AgentExecutor, ExecutorEventEmitter, SamplingParams};

// ── Public types ────────────────────────────────────────────────────────────

/// Controls behaviour when the Complete governance hook fails.
///
/// Default is `FailClosed`: a hook failure causes the spawn to fail.
/// Use `WarnAndContinue` only when the hook is purely observational.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HookFailureMode {
    /// Hook failure causes the spawn to return an error (default).
    #[default]
    FailClosed,
    /// Hook failure is logged as a warning; spawn proceeds.
    WarnAndContinue,
}

/// Computation node — describes what to spawn.
#[derive(Debug, Clone)]
pub struct CompNode {
    pub agent_id: AgentId,
    pub task: String,
    pub model_override: Option<String>,
    pub timeout: Option<Duration>,
    /// Phase 9 R2: weave the node belongs to. Forwarded from the
    /// CompiledGraph executor (which threads it from the dispatch request)
    /// to `runner.spawn(node, parent, node.weave_id.as_ref())`.
    pub weave_id: Option<WeaveId>,
    /// Phase 1B substrate cascade: daemon-allocated `dispatch_id` of the
    /// chat-tool dispatch this node belongs to. Threaded from
    /// `SpawnSpec.dispatch_id` by `CompiledGraph::execute_spawn` (and the
    /// other CompNode constructors) so the runner can stamp it onto the
    /// outgoing `SpawnCompleted` event at the emit site. `None` for
    /// non-chat-tool paths.
    pub dispatch_id: Option<String>,
    /// Per-trajectory sampling params (EXT-01 Phase 24). None → neutral defaults
    /// (temperature=1.0, top_p=1.0, top_k=0, matching Phase 23 behaviour).
    pub sampling: Option<SamplingParams>,
}

/// Result of a successful spawn.
#[derive(Debug, Clone)]
pub struct SpawnResult {
    pub session_id: SessionId,
    pub envelope: Envelope,
    pub raw: String,
    pub signals: Vec<Signal>,
    pub quality_issues: Vec<QualityIssue>,
}

// ── AlzinaRunner ────────────────────────────────────────────────────────────

/// Core agent dispatch lifecycle manager.
///
/// Owns governance, bootstrap, sessions, and learnings. Delegates actual
/// LLM execution to an `AgentExecutor`, keeping the runner testable
/// and the execution seam replaceable (§7.3).
pub struct AlzinaRunner {
    governance: Arc<GovernanceLayer>,
    bootstrap: Arc<BootstrapEngine>,
    sessions: Arc<SessionHierarchy>,
    learnings: Arc<LearningsMerger>,
    executor: Arc<dyn AgentExecutor>,
    default_model: String,
    default_timeout: Duration,
    hook_failure_mode: HookFailureMode,
    learnings_merge_failures: AtomicU64,
    memory_sink: Option<Arc<dyn SpawnEventSink>>,
    /// Optional emitter for sub-agent progress events (P5-LIVENESS-INNER).
    /// When set, the runner constructs a session-tagged closure and forwards
    /// it to `AgentExecutor::execute_with_emitter` so streaming `TextDelta` /
    /// `TokenUsage` events are republished onto the daemon's event bus.
    progress_emitter: Option<ExecutorEventEmitter>,
    /// Workspace handle for signal triage file writes (Phase 9 R3, LANDMINE 6 fix).
    /// Required field — `process_signals` uses this to write `well/{class}/` files.
    workspace: Arc<WorkspaceHandle>,
    /// Optional signal records store for persisting process_signals output to SQLite.
    /// When `Some`, each processed signal produces a `signal_records` row.
    /// `None` keeps the runner functional without a DB pool (e.g. standalone tests).
    signal_store: Option<Arc<SignalRecordsStore>>,
    /// A3 / D11-13: optional stop-conditions evaluator. When set, the runner
    /// notes failures/successes per weave and evaluates the three LIVE arms
    /// (tool_failure_threshold, redo_count, wall_time) after each spawn. When
    /// None, A3 stop-conditions never fire (preserves pre-P11 behaviour for
    /// callers that don't wire the evaluator). The wire site is
    /// `evaluate_stop_conditions` called immediately after extract_signals.
    stop_conditions: Option<Arc<StopConditionEvaluator>>,
}

impl AlzinaRunner {
    /// Construct a new runner with all dependencies.
    ///
    /// `workspace` is required for signal triage file writes (Phase 9 R3, LANDMINE 6 fix).
    /// It is the 6th positional parameter, before `default_model`.
    pub fn new(
        governance: Arc<GovernanceLayer>,
        bootstrap: Arc<BootstrapEngine>,
        sessions: Arc<SessionHierarchy>,
        learnings: Arc<LearningsMerger>,
        executor: Arc<dyn AgentExecutor>,
        workspace: Arc<WorkspaceHandle>,
        default_model: String,
        default_timeout: Duration,
    ) -> Self {
        Self {
            governance,
            bootstrap,
            sessions,
            learnings,
            executor,
            workspace,
            default_model,
            default_timeout,
            hook_failure_mode: HookFailureMode::default(),
            learnings_merge_failures: AtomicU64::new(0),
            memory_sink: None,
            progress_emitter: None,
            signal_store: None,
            stop_conditions: None,
        }
    }

    /// A3 / D11-13: attach the stop-conditions evaluator. When set, the
    /// runner notes spawn outcomes per weave and fires
    /// `LifecycleEvent::StopConditionTripped` via the governance layer when
    /// one of the three LIVE arms is met (tool_failure_threshold N=5,
    /// redo_count N=10, wall_time 600s).
    pub fn with_stop_conditions(mut self, evaluator: Arc<StopConditionEvaluator>) -> Self {
        self.stop_conditions = Some(evaluator);
        self
    }

    /// Set the failure mode for the Complete governance hook.
    pub fn with_hook_failure_mode(mut self, mode: HookFailureMode) -> Self {
        self.hook_failure_mode = mode;
        self
    }

    /// Attach a progress emitter (P5-LIVENESS-INNER).
    ///
    /// When set, every spawn forwards mid-turn sidecar events
    /// (`TextDelta`, `TokenUsage`, ...) into this closure tagged with the
    /// spawn's `session_id`. The daemon wires this to
    /// `EventBus::publish` so the streaming `dispatch_agent` handler's
    /// `event_belongs_to_dispatch_tree` filter sees traffic during sub-agent
    /// execution and resets its idle timer.
    pub fn with_progress_emitter(mut self, emitter: ExecutorEventEmitter) -> Self {
        self.progress_emitter = Some(emitter);
        self
    }

    /// Attach a memory event sink for recording spawn lifecycle events.
    ///
    /// When set, the runner emits `SpawnCompleted` events after each
    /// successful envelope processing, recording them into daily memory.
    pub fn with_memory_sink(mut self, sink: Arc<dyn SpawnEventSink>) -> Self {
        self.memory_sink = Some(sink);
        self
    }

    /// Attach a signal records store for persisting process_signals output.
    ///
    /// When set, each signal processed by `process_signals` inserts a row into the
    /// `signal_records` SQLite table (Phase 9 R3). `None` leaves signal persistence
    /// disabled (signals still write triage files; just no DB index row).
    pub fn with_signal_store(mut self, store: Arc<SignalRecordsStore>) -> Self {
        self.signal_store = Some(store);
        self
    }

    /// Access the session hierarchy (for inspection in composition nodes).
    pub fn sessions(&self) -> &Arc<SessionHierarchy> {
        &self.sessions
    }

    /// Access the governance layer.
    pub fn governance(&self) -> &Arc<GovernanceLayer> {
        &self.governance
    }

    /// Spawn an agent — the full dispatch lifecycle.
    ///
    /// Allocates a fresh `SessionId` and forwards to
    /// [`spawn_with_id`](Self::spawn_with_id). The runner registers the
    /// session in the hierarchy itself.
    ///
    /// Steps (per §4.1 dispatch sequence diagram):
    /// 1. Validate agent_id
    /// 2. PreSpawn governance hook — if blocked, return error
    /// 3. Register session in hierarchy
    /// 4. Bootstrap assembly — build context
    /// 5. Resolve model
    /// 6-7. Execute agent with timeout
    /// 8. Parse envelope from output
    /// 9. Validate envelope
    /// 10. Extract signals
    /// 11. Process CONTEXT_UPDATE via LearningsMerger
    /// 12. Complete governance hook
    /// 13. Mark session done
    /// 14. Return SpawnResult
    #[instrument(skip(self, node, parent_session, weave_id), fields(
        agent = %node.agent_id,
        parent = ?parent_session.map(|s| s.to_string()),
        weave = ?weave_id.map(|w| w.to_string()),
    ))]
    pub async fn spawn(
        &self,
        node: &CompNode,
        parent_session: Option<&SessionId>,
        weave_id: Option<&WeaveId>,
    ) -> AlzinaResult<SpawnResult> {
        let session_id = SessionId::new();
        self.spawn_inner(session_id, node, parent_session, weave_id, false, false)
            .await
    }

    /// Spawn an agent using a pre-allocated session id.
    ///
    /// P5-DEBUG-DISPATCH (Fix A): the streaming dispatch handler in
    /// `alzina-daemon::api::dispatch` filters bus events by
    /// `session_id == root_session_id`, where `root_session_id` is
    /// allocated by `SessionManager::dispatch_composition_with_id_tx`.
    /// Before this method existed, the runner allocated its *own* id
    /// (`SessionId::new()` in `spawn`), so every TextDelta the executor
    /// emitted carried the wrong id and was silently filtered out.
    ///
    /// When called via this entrypoint, the caller is expected to have
    /// already inserted `session_id` into the session hierarchy via
    /// `SessionHierarchy::create_root` (or equivalent for sub-agents),
    /// so the runner skips its own registration.
    ///
    /// See `docs/p5-debug-dispatch-synth.md` for the full diagnosis.
    ///
    /// `composition_context`: when `Some(ctx)`, the renderer is invoked to
    /// produce the preamble-prepended + channel-substituted task before
    /// dispatch. When `None`, dispatch is byte-identical to today (D10-02).
    #[instrument(skip(self, node, parent_session, weave_id, composition_context), fields(
        session = %session_id,
        agent = %node.agent_id,
        parent = ?parent_session.map(|s| s.to_string()),
        weave = ?weave_id.map(|w| w.to_string()),
    ))]
    pub async fn spawn_with_id(
        &self,
        session_id: SessionId,
        node: &CompNode,
        parent_session: Option<&SessionId>,
        weave_id: Option<&WeaveId>,
        composition_context: Option<crate::composition::parser::CompositionContext>,
    ) -> AlzinaResult<SpawnResult> {
        self.spawn_with_id_and_cancel(
            session_id,
            node,
            parent_session,
            weave_id,
            composition_context,
            None,
        )
        .await
    }

    /// E3 / D11-16: spawn variant that races the executor against a
    /// `CancellationToken`. When the token fires mid-spawn, the executor
    /// short-circuits and the spawn returns
    /// `Err(AlzinaError::Orchestration("cancelled"))`. The error string
    /// is `cancelled` (not a generic error) so downstream consumers can
    /// distinguish from natural failure — E1's `on_leaf_failed` callback
    /// re-publishes with the same string verbatim.
    #[instrument(skip(self, node, parent_session, weave_id, composition_context, cancel_token), fields(
        session = %session_id,
        agent = %node.agent_id,
        parent = ?parent_session.map(|s| s.to_string()),
        weave = ?weave_id.map(|w| w.to_string()),
    ))]
    pub async fn spawn_with_id_and_cancel(
        &self,
        session_id: SessionId,
        node: &CompNode,
        parent_session: Option<&SessionId>,
        weave_id: Option<&WeaveId>,
        composition_context: Option<crate::composition::parser::CompositionContext>,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
    ) -> AlzinaResult<SpawnResult> {
        // D10-02: when a composition context is present, render the task
        // (preamble + channel substitutions) before dispatch. When absent,
        // forward the raw task_template — byte-identical to the ad-hoc path.
        //
        // Commit B: capture substitution misses (envelope-backed refs whose
        // referenced leaf has no envelope — typically failed siblings from
        // the f5e9d17 sibling-survival path). Publish a
        // SubstitutionsUnresolved event so the partial render is auditable
        // rather than silent. The leaf is still dispatched with empty
        // substitutions for the missing refs (preserves existing render
        // semantics); the event records exactly what did not resolve.
        let rendered_task: Option<String> = composition_context.as_ref().map(|ctx| {
            let (rendered, unresolved) =
                crate::composition::parser::render_task_with_audit(&node.task, ctx);
            if !unresolved.is_empty() {
                if let Some(ref emit) = self.progress_emitter {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0);
                    emit(AlzinaEvent::SubstitutionsUnresolved {
                        session_id: session_id.to_string(),
                        compose_id: ctx.compose_id.clone(),
                        consumer_node_id: ctx.node_id.clone(),
                        consumer_agent: node.agent_id.to_string(),
                        unresolved,
                        timestamp: ts,
                    });
                }
            }
            rendered
        });

        // Build a potentially-overridden node for dispatch.
        let dispatch_node: std::borrow::Cow<'_, CompNode> = match rendered_task {
            Some(ref rt) => {
                let mut n = node.clone();
                n.task = rt.clone();
                std::borrow::Cow::Owned(n)
            }
            None => std::borrow::Cow::Borrowed(node),
        };

        // E3: if the cancel token is already fired at entry, short-circuit
        // before ANY work happens — preserves cancel intent for races where
        // the cancel arrives before the spawn even starts.
        if let Some(ref tok) = cancel_token
            && tok.is_cancelled()
        {
            tracing::info!(session = %session_id, "spawn cancelled before execution");
            return Err(AlzinaError::Orchestration("cancelled".into()));
        }

        // Race the spawn against the cancel token. The spawn itself drives
        // through the existing execute_lifecycle path; the cancel arm
        // short-circuits with a synthetic Err. Note: this is a coarse-grain
        // cancel — the executor's in-flight call isn't aborted mid-rpc.
        // The next observable spawn returns at this boundary will surface
        // the cancel. Wave 2/3 may push the token deeper into the executor.
        // C-01 (compose-leaf dispatch template fix): compose leaves are
        // dispatched via spawn_with_id (not spawn), so they arrive here.
        // They have no parent session → session_type=Root → bootstrap Stage 8
        // skips the dispatch template. We pass inject_dispatch_template=true
        // so execute_lifecycle patches the template in after assembly.
        let spawn_fut = self.spawn_inner(
            session_id.clone(),
            &dispatch_node,
            parent_session,
            weave_id,
            true,
            true, // inject_dispatch_template: compose leaves need envelope instructions
        );
        match cancel_token {
            Some(tok) => {
                tokio::select! {
                    biased;
                    result = spawn_fut => result,
                    _ = tok.cancelled() => {
                        tracing::info!(session = %session_id, "spawn cancelled via cancel_token");
                        Err(AlzinaError::Orchestration("cancelled".into()))
                    }
                }
            }
            None => spawn_fut.await,
        }
    }

    /// Shared lifecycle for both `spawn` and `spawn_with_id`.
    ///
    /// `caller_registered_session` controls whether the runner inserts
    /// the session into the hierarchy itself. When `true` (the
    /// `spawn_with_id` path), the caller has already called
    /// `create_root` (or `spawn_child`) and we skip Step 3 to avoid
    /// double-insertion errors.
    async fn spawn_inner(
        &self,
        session_id: SessionId,
        node: &CompNode,
        parent_session: Option<&SessionId>,
        weave_id: Option<&WeaveId>,
        caller_registered_session: bool,
        // When true, the dispatch template is injected into the bootstrap context
        // after assembly even if session_type is Root. Used for compose leaves,
        // which run as Root sessions but must return a structured envelope.
        inject_dispatch_template: bool,
    ) -> AlzinaResult<SpawnResult> {
        // Step 1: Validate agent_id
        alzina_governance::validate_agent_id(node.agent_id.as_str())?;

        // Step 2: PreSpawn governance hook
        // D7 (D5-P1-3): chat_root currently None at this layer; threading
        // chat_root through SpawnSpec/CompNode is a follow-up. The variant
        // is wired so hooks observe the field shape today and pre-existing
        // chat-aware hooks can be ready when threading lands.
        let pre_spawn_event = LifecycleEvent::PreSpawn {
            agent_id: node.agent_id.clone(),
            task: node.task.clone(),
            parent_session: parent_session.cloned(),
            chat_root: None,
        };
        let outcome = self.governance.process_event(&pre_spawn_event).await?;
        if outcome.blocked {
            let reason = outcome
                .block_reason
                .unwrap_or_else(|| "blocked by governance hook".into());
            return Err(AlzinaError::Governance(format!(
                "PreSpawn blocked for agent {}: {reason}",
                node.agent_id
            )));
        }

        // Step 3: Register session in hierarchy (unless the caller did).
        //
        // P5-DEBUG-DISPATCH (Fix A): when `caller_registered_session` is
        // true, the outer dispatch path already inserted this id via
        // `SessionHierarchy::create_root`. Inserting again would either
        // fail or double-allocate, so skip.
        if !caller_registered_session {
            match parent_session {
                Some(parent) => {
                    self.sessions
                        .spawn_child(parent, &session_id, &node.agent_id, weave_id)
                        .await?;
                }
                None => {
                    self.sessions
                        .create_root(&session_id, &node.agent_id, weave_id)
                        .await?;
                }
            }
        }

        // Derive session type from parent_session presence
        let session_type = match parent_session {
            Some(_) => SessionType::SubAgent,
            None => SessionType::Root,
        };

        // A3 / D11-13: anchor the wall-time clock for this weave (idempotent
        // per `note_session_start`; subsequent calls on the same weave_id
        // are no-ops). Required for the wall_time arm to fire.
        if let Some(ref evaluator) = self.stop_conditions {
            evaluator.note_session_start(weave_id);
        }

        // From here, if anything fails we mark the session as failed.
        match self
            .execute_lifecycle(
                node,
                &session_id,
                weave_id,
                session_type,
                inject_dispatch_template,
            )
            .await
        {
            Ok(result) => {
                // A3 / D11-13: success arm — reset per-weave consecutive
                // failure counter so a burst→recovery doesn't trip the arm.
                if let Some(ref evaluator) = self.stop_conditions {
                    evaluator.note_success(weave_id);
                    self.evaluate_stop_conditions(&session_id, &node.agent_id, weave_id)
                        .await;
                }
                Ok(result)
            }
            Err(e) => {
                // A3 / D11-13: failure arm — increment per-weave counter,
                // then evaluate (may fire tool_failure_threshold).
                if let Some(ref evaluator) = self.stop_conditions {
                    evaluator.note_failure(weave_id);
                    self.evaluate_stop_conditions(&session_id, &node.agent_id, weave_id)
                        .await;
                }
                warn!(
                    session = %session_id,
                    agent = %node.agent_id,
                    error = %e,
                    "spawn failed, marking session as failed"
                );
                if let Err(complete_err) = self.sessions.fail(&session_id, &e.to_string()).await {
                    error!(
                        session = %session_id,
                        agent = %node.agent_id,
                        error = %complete_err,
                        "failed to mark session as failed"
                    );
                }
                Err(e)
            }
        }
    }

    /// A3 / D11-13: evaluate all live stop-condition arms. Each `Some`
    /// reason returned by an arm fires exactly one
    /// `LifecycleEvent::StopConditionTripped` via the governance layer;
    /// `StopConditionHook` (alzina-governance::hooks::builtins) is the
    /// consumer.
    ///
    /// AC-1 loud-degradation: every trip also emits `tracing::warn!` so ops
    /// sees the engagement even before the hook handler runs.
    async fn evaluate_stop_conditions(
        &self,
        session_id: &SessionId,
        agent_id: &AgentId,
        weave_id: Option<&WeaveId>,
    ) {
        let evaluator = match self.stop_conditions.as_ref() {
            Some(e) => e,
            None => return,
        };

        // Collect trip reasons (each arm returns Option<&'static str>).
        let mut tripped: Vec<&'static str> = Vec::new();
        if let Some(r) = evaluator.tool_failure_arm(weave_id) {
            tripped.push(r);
        }
        if let Some(r) = evaluator.redo_arm(session_id) {
            tripped.push(r);
        }
        if let Some(r) = evaluator.wall_time_arm(weave_id) {
            tripped.push(r);
        }

        for reason in tripped {
            warn!(
                session = %session_id,
                agent = %agent_id,
                weave = ?weave_id,
                reason = %reason,
                "stop condition tripped"
            );
            // Build a minimal OrlogSummary carrying the weave_id. Hooks that
            // need the full blueprint re-read it from disk; this is the
            // thin-snapshot pattern from `OrlogSummary` docs.
            let weave_str = weave_id.map(|w| w.to_string()).unwrap_or_default();
            let tripped_by = format!("_system-runner:{agent_id}");
            let summary = OrlogSummary {
                weave_id: weave_str.clone(),
                classification: String::new(),
                hitl_mode: String::new(),
                goal: String::new(),
                study_hook: String::new(),
                phases_count: 0,
                stop_conditions: vec![reason.to_string()],
            };
            let event = LifecycleEvent::StopConditionTripped {
                weave_id: weave_str.clone(),
                summary: summary.clone(),
                condition: reason.to_string(),
                tripped_by: tripped_by.clone(),
            };
            let mut outcome = match self.governance.process_event(&event).await {
                Ok(o) => o,
                Err(err) => {
                    warn!(
                        session = %session_id,
                        agent = %agent_id,
                        error = %err,
                        reason = %reason,
                        "StopConditionTripped hook delivery failed (non-fatal)"
                    );
                    continue;
                }
            };

            // Phase 11 (C8.1): if the StopConditionHook produced an
            // `Engage(Choice)`, route the request through the broker so
            // the operator's choice resolves. When the choice is
            // `"continue"`, fire a follow-up `StopConditionOverridden`
            // event to drive `StopConditionJustificationHook` —
            // EngagementMode::FreeForm's first production trigger.
            //
            // Non-`continue` resolutions (`"halt"`, `"amend-orlog"`),
            // fallbacks, and abandonments still emit the override event
            // so future observer hooks can react; the `choice` field
            // discriminates. The override pipeline is best-effort —
            // a delivery failure does not propagate up since the
            // operator already made the call.
            if outcome.engaged.is_some() {
                let resolution = match self.governance.resolve_engagement(&mut outcome).await {
                    Ok(r) => r,
                    Err(err) => {
                        warn!(
                            session = %session_id,
                            agent = %agent_id,
                            error = %err,
                            reason = %reason,
                            "stop-condition engagement resolution failed (non-fatal)"
                        );
                        continue;
                    }
                };
                let choice = match resolution.as_ref().map(|r| &r.outcome) {
                    Some(alzina_core::ResolutionOutcome::Resolved { value }) => value
                        .as_str()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| value.to_string()),
                    Some(alzina_core::ResolutionOutcome::FellBack { .. }) => {
                        // Fallback fired (timeout, no surface). Skip the
                        // override event — operator never made a choice.
                        continue;
                    }
                    Some(alzina_core::ResolutionOutcome::Abandoned) => {
                        continue;
                    }
                    None => continue,
                };
                let override_event = LifecycleEvent::StopConditionOverridden {
                    weave_id: weave_str,
                    summary,
                    condition: reason.to_string(),
                    tripped_by,
                    choice,
                };
                if let Err(err) = self.governance.process_event(&override_event).await {
                    warn!(
                        session = %session_id,
                        agent = %agent_id,
                        error = %err,
                        reason = %reason,
                        "StopConditionOverridden hook delivery failed (non-fatal)"
                    );
                }
            }
        }
    }

    /// Run the executor with a progress-driven idle watchdog.
    ///
    /// Wraps the runner's `progress_emitter` so that every published
    /// `AlzinaEvent` (TextDelta, TokenUsage, sub-agent SessionSpawned,
    /// …) bumps a shared `last_event` instant. A select! races the
    /// executor against `sleep_until(last_event + idle_timeout)`. The
    /// sleep arm re-checks elapsed time on wake (events arriving
    /// during the sleep extend the deadline transparently in the next
    /// loop iteration).
    ///
    /// Only invoked when `progress_emitter` is set. The wall-clock
    /// branch in `execute_lifecycle` covers test executors that don't
    /// publish anything.
    async fn execute_with_idle_watchdog(
        &self,
        node: &CompNode,
        instruction: &str,
        model: &str,
        session_id: &SessionId,
        idle_timeout: Duration,
    ) -> AlzinaResult<(String, Option<Envelope>)> {
        let base = self
            .progress_emitter
            .clone()
            .expect("execute_with_idle_watchdog requires a progress_emitter");

        let last_event: Arc<StdMutex<tokio::time::Instant>> =
            Arc::new(StdMutex::new(tokio::time::Instant::now()));
        let watchdog_clock = Arc::clone(&last_event);
        let wrapped: ExecutorEventEmitter = Arc::new(move |ev: AlzinaEvent| {
            if let Ok(mut guard) = watchdog_clock.lock() {
                *guard = tokio::time::Instant::now();
            }
            base(ev);
        });

        // 260515-ndk Task 2: swap to execute_with_envelope so the typed
        // envelope (when the executor captured one via the return_envelope
        // tool) flows through to the runner's parse step. Default impl on
        // the trait returns (raw, None) so legacy executors keep working.
        let exec_fut = self.executor.execute_with_envelope(
            &node.agent_id,
            instruction,
            model,
            &node.task,
            session_id,
            Some(wrapped),
        );
        tokio::pin!(exec_fut);

        loop {
            let deadline = {
                let guard = last_event.lock().expect("watchdog mutex poisoned");
                *guard + idle_timeout
            };
            tokio::select! {
                biased;
                result = &mut exec_fut => return result,
                _ = tokio::time::sleep_until(deadline) => {
                    let elapsed = {
                        let guard = last_event.lock().expect("watchdog mutex poisoned");
                        guard.elapsed()
                    };
                    if elapsed >= idle_timeout {
                        warn!(
                            session = %session_id,
                            agent = %node.agent_id,
                            idle_timeout = ?idle_timeout,
                            "agent produced no progress events for idle_timeout — aborting"
                        );
                        return Err(AlzinaError::Orchestration(format!(
                            "agent {} timed out after {idle_timeout:?} (no progress events)",
                            node.agent_id
                        )));
                    }
                    // Otherwise: an event arrived during the sleep window.
                    // Loop and recompute the deadline against the latest event.
                }
            }
        }
    }

    /// Inner lifecycle after session registration — separates error handling.
    ///
    /// `inject_dispatch_template`: when true and the assembled context has no
    /// dispatch template, the bootstrap engine's post-assembly injection method
    /// is called to patch it in. Used for compose leaves (C-01).
    async fn execute_lifecycle(
        &self,
        node: &CompNode,
        session_id: &SessionId,
        weave_id: Option<&WeaveId>,
        session_type: SessionType,
        inject_dispatch_template: bool,
    ) -> AlzinaResult<SpawnResult> {
        // Step 4: Bootstrap assembly
        let mut bootstrap_context = self
            .bootstrap
            .assemble(&node.agent_id, &node.task, weave_id, session_type)
            .await?;

        // C-01: compose leaves run as Root sessions (no parent) but must
        // return a structured envelope. Inject the dispatch template if the
        // caller requested it and assembly skipped Stage 8.
        if inject_dispatch_template {
            self.bootstrap
                .inject_dispatch_template_if_absent(&mut bootstrap_context, &node.agent_id)?;
        }

        // Step 5: Resolve model. Carry the agent's identity-pinned model
        // (from identity.toml, surfaced on the bootstrap context) into the
        // resolver so a per-agent pin — e.g. a haiku reader — takes effect.
        // Priority is preserved by resolve_model: per-dispatch override >
        // identity `model` > workspace default. Before this, model resolution
        // ran against an empty identity, so identity.toml `model` was a no-op
        // and every agent used the workspace default.
        let mut identity_fields = BTreeMap::new();
        if let Some(m) = bootstrap_context.agent_model.clone() {
            identity_fields.insert("model".to_string(), m);
        }
        let resolved_identity = AgentIdentity {
            id: node.agent_id.clone(),
            archetype: None,
            fields: identity_fields,
            typed_fields: BTreeMap::new(),
            sections: BTreeMap::new(),
            raw: String::new(),
            denied_tools: Vec::new(),
            // Synthetic identity used only for model resolution; the
            // enforcement-time allowlist is read from governance, not here.
            // Default is fail-closed, which is the safe value for a stub.
            shell_allow: Default::default(),
        };

        let model = model_resolver::resolve_model(
            &resolved_identity,
            node.model_override.as_deref(),
            &self.default_model,
        );
        debug!(
            agent = %node.agent_id,
            model = %model,
            identity_pinned = bootstrap_context.agent_model.is_some(),
            "resolved dispatch model"
        );

        // Render the bootstrap context into an instruction string.
        let instruction = self.bootstrap.render(&bootstrap_context)?;

        debug!(
            session = %session_id,
            agent = %node.agent_id,
            model = %model,
            instruction_len = instruction.len(),
            "executing agent"
        );

        // Steps 6-7: Execute with progress-driven liveness.
        //
        // P5-DEBUG-DISPATCH (P2 cleanup): a wall-clock
        // `tokio::time::timeout(timeout, …)` used to wrap the executor
        // call. Once Fix A+B made the streaming pipeline reliably emit
        // TextDelta during thinking and tool-input accumulation, the
        // wall-clock became the binding constraint — long but actively
        // streaming dispatches were killed at exactly 120s. Replace
        // with an idle watchdog: each progress event resets the
        // deadline; the spawn fails only if no events flow for
        // `timeout`.
        //
        // Tests / executors without a `progress_emitter` keep the
        // wall-clock semantics so `timeout_returns_error_and_marks_session_failed`
        // and other lifecycle invariants hold.
        let timeout = node.timeout.unwrap_or(self.default_timeout);
        // 260515-ndk Task 2: bind both raw output AND optional typed
        // envelope captured by the executor. Default trait impl returns
        // (raw, None) so legacy executors keep working — the parse branch
        // below falls through to the unchanged strict-then-lenient prose
        // parser when typed_envelope is None.
        let (raw, typed_envelope) = if self.progress_emitter.is_some() {
            self.execute_with_idle_watchdog(node, &instruction, &model, session_id, timeout)
                .await?
        } else {
            match tokio::time::timeout(
                timeout,
                self.executor.execute_with_envelope(
                    &node.agent_id,
                    &instruction,
                    &model,
                    &node.task,
                    session_id,
                    None,
                ),
            )
            .await
            {
                Ok(result) => result?,
                Err(_elapsed) => {
                    return Err(AlzinaError::Orchestration(format!(
                        "agent {} timed out after {:?}",
                        node.agent_id, timeout
                    )));
                }
            }
        };

        // Step 8: Parse envelope
        // All session types use strict parse. The lenient fallback
        // (root-session prose fabrication) was removed — agents must call
        // the return_envelope tool. If no typed envelope was captured and
        // the prose parse fails, that is a hard error.
        //
        // 260515-ndk Task 2: when the executor captured a typed envelope
        // via the return_envelope tool, use it directly and skip the prose
        // parse. The `raw` string is then re-rendered from the typed envelope
        // via render_envelope_as_prose so downstream consumers (audit
        // SessionCompleted arm, prepare_envelope in chat.rs, SSE
        // [low-authority source=...] wrapper) see byte-identical output
        // regardless of which path produced the envelope.
        let (envelope, raw) = match typed_envelope {
            Some(env) => {
                let canonical_raw = alzina_governance::envelope::render_envelope_as_prose(&env);
                debug!(
                    agent = %node.agent_id,
                    "envelope captured via typed return_envelope tool — skipping prose parse"
                );
                (env, canonical_raw)
            }
            None => {
                match self.governance.parse_envelope(&raw) {
                    Ok(env) => (env, raw),
                    Err(e) => {
                        // Publish EnvelopeParseFailure before propagating so the
                        // audit JSONL and ObservationService can detect parse-failure
                        // seams without requiring operators to grep tracing output.
                        if let Some(ref emit) = self.progress_emitter {
                            let raw_preview = if raw.len() > 200 {
                                Some(raw[..200].to_string())
                            } else if raw.is_empty() {
                                None
                            } else {
                                Some(raw.clone())
                            };
                            let ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as i64)
                                .unwrap_or(0);
                            emit(AlzinaEvent::EnvelopeParseFailure {
                                session_id: session_id.to_string(),
                                agent_id: node.agent_id.to_string(),
                                raw_preview,
                                error: e.to_string(),
                                timestamp: ts,
                            });
                        }
                        return Err(e);
                    }
                }
            }
        };

        // Step 9: Validate envelope
        let quality_issues = self.governance.validate_envelope(&envelope);

        // Step 10: Extract signals
        let signals = self.governance.extract_signals(&envelope);

        // Step 11: Process all 4 signal classes uniformly via process_signals
        // (Phase 9 R3 wiring; LANDMINE 2 fixed: ContextUpdate is processed here,
        // NOT in a separate inline branch — would otherwise double-merge).
        let envelope_id = session_id.to_string();
        match crate::signals::process_signals(
            &signals,
            &self.governance,
            &self.workspace,
            &node.agent_id,
            &self.learnings,
            self.signal_store.as_ref(),
            &envelope_id,
            weave_id,
        )
        .await
        {
            Ok(signal_outcome) => {
                if !signal_outcome.failures.is_empty() {
                    let n = signal_outcome.failures.len() as u64;
                    self.learnings_merge_failures
                        .fetch_add(n, AtomicOrdering::Relaxed);
                    warn!(
                        agent = %node.agent_id,
                        failures = signal_outcome.failures.len(),
                        "signal processing had partial failures"
                    );
                }
                debug!(
                    agent = %node.agent_id,
                    processed = signal_outcome.total_processed(),
                    "signal processing complete"
                );
            }
            Err(e) => {
                warn!(
                    agent = %node.agent_id,
                    error = %e,
                    "process_signals returned error (non-fatal)"
                );
            }
        }

        // Step 12: Complete governance hook
        let complete_event = LifecycleEvent::Complete {
            agent_id: node.agent_id.clone(),
            session_id: session_id.clone(),
            envelope: alzina_core::envelope::RawEnvelope {
                text: raw.clone(),
                session_id: session_id.clone(),
            },
        };
        if let Err(e) = self.governance.process_event(&complete_event).await {
            match self.hook_failure_mode {
                HookFailureMode::FailClosed => {
                    error!(
                        agent = %node.agent_id,
                        error = %e,
                        "Complete hook failed (fail-closed mode)"
                    );
                    return Err(AlzinaError::Governance(format!(
                        "Complete hook failed for agent {}: {e}",
                        node.agent_id
                    )));
                }
                HookFailureMode::WarnAndContinue => {
                    warn!(
                        agent = %node.agent_id,
                        error = %e,
                        "Complete hook failed (warn-and-continue mode)"
                    );
                }
            }
        }

        // Step 13: Mark session done
        self.sessions
            .complete(session_id, envelope.status.clone())
            .await?;

        // Step 13b: Record envelope-processed event in daily memory (non-fatal)
        if let Some(ref sink) = self.memory_sink {
            let summary = match &envelope.signal {
                Some(sig) => format!("{:?}: {sig}", envelope.status),
                None => format!("{:?}", envelope.status),
            };
            let spawn_event = SpawnCompleted {
                agent_id: node.agent_id.to_string(),
                session_id: session_id.to_string(),
                weave_id: weave_id.map(|w| w.to_string()),
                // Phase 1B substrate cascade: carry the chat-tool dispatch_id
                // off the node so the memory event sink can stamp it onto
                // `stitch_records.dispatch_id` (closes G4 on the chat path).
                dispatch_id: node.dispatch_id.clone(),
                summary,
            };
            if let Err(e) = sink.on_spawn_completed(spawn_event).await {
                warn!(
                    agent = %node.agent_id,
                    session = %session_id,
                    error = %e,
                    "memory sink record failed (non-fatal)"
                );
            }
        }

        info!(
            session = %session_id,
            agent = %node.agent_id,
            status = ?envelope.status,
            signals = signals.len(),
            quality_issues = quality_issues.len(),
            "spawn complete"
        );

        // Step 14: Return SpawnResult
        Ok(SpawnResult {
            session_id: session_id.clone(),
            envelope,
            raw,
            signals,
            quality_issues,
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_core::TemplateEngine;
    use alzina_core::envelope::EnvelopeStatus;
    use alzina_core::hooks::{HookAction, HookHandler, SagaState};
    use alzina_governance::config::{GovernanceConfig, LearningsConfig};
    use alzina_governance::facade::HookSet;
    use alzina_governance::hooks::EventFilter;
    use alzina_workspace::WorkspaceHandle;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::sync::Mutex;

    // ── Mock AgentExecutor ──────────────────────────────────────────────────

    struct MockExecutor {
        response: Mutex<String>,
        executed: AtomicBool,
        delay: Option<Duration>,
    }

    impl MockExecutor {
        fn new(response: &str) -> Self {
            Self {
                response: Mutex::new(response.to_string()),
                executed: AtomicBool::new(false),
                delay: None,
            }
        }

        fn with_delay(response: &str, delay: Duration) -> Self {
            Self {
                response: Mutex::new(response.to_string()),
                executed: AtomicBool::new(false),
                delay: Some(delay),
            }
        }

        fn was_executed(&self) -> bool {
            self.executed.load(Ordering::SeqCst)
        }
    }

    /// Executor that emits a configurable burst of `TextDelta` events
    /// through the supplied emitter, then returns the response. Used to
    /// exercise the idle-watchdog reset path.
    struct StreamingMockExecutor {
        response: String,
        interval: Duration,
        iterations: u32,
    }

    #[async_trait]
    impl AgentExecutor for StreamingMockExecutor {
        async fn execute(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
        ) -> AlzinaResult<String> {
            tokio::time::sleep(self.interval * self.iterations).await;
            Ok(self.response.clone())
        }

        async fn execute_with_emitter(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
            session_id: &SessionId,
            emitter: Option<ExecutorEventEmitter>,
        ) -> AlzinaResult<String> {
            for i in 0..self.iterations {
                tokio::time::sleep(self.interval).await;
                if let Some(ref e) = emitter {
                    e(AlzinaEvent::TextDelta {
                        session_id: session_id.to_string(),
                        turn_id: format!("t-{i}"),
                        content: format!("chunk {i}"),
                        timestamp: 0,
                    });
                }
            }
            Ok(self.response.clone())
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
        ) -> AlzinaResult<String> {
            self.executed.store(true, Ordering::SeqCst);
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }
            Ok(self.response.lock().await.clone())
        }
    }

    // ── Blocking hook ───────────────────────────────────────────────────────

    struct BlockingHook {
        reason: String,
    }

    #[async_trait]
    impl HookHandler for BlockingHook {
        fn name(&self) -> &str {
            "blocking-hook"
        }
        fn priority(&self) -> u32 {
            0
        }
        fn blocking(&self) -> bool {
            true
        }
        async fn execute(
            &self,
            _event: &LifecycleEvent,
            _saga: &mut SagaState,
        ) -> AlzinaResult<HookAction> {
            Ok(HookAction::Block(self.reason.clone()))
        }
    }

    // ── Test helpers ────────────────────────────────────────────────────────

    fn well_formed_envelope() -> String {
        r#"Analysis complete.

STATUS: complete
ARTIFACTS: artifacts/output.md
SIGNAL: analysis done
TENSIONS: none
EMERGENT: hidden coupling detected
CONTEXT_UPDATE: always validate config at load time"#
            .to_string()
    }

    fn envelope_no_context_update() -> String {
        r#"STATUS: complete
ARTIFACTS: artifacts/output.md
SIGNAL: done
TENSIONS: none
EMERGENT: none"#
            .to_string()
    }

    fn malformed_envelope() -> String {
        "This is just plain text with no envelope structure.".to_string()
    }

    const TEST_AGENTS: &[&str] = &[
        "smidr",
        "galdr",
        "rogue",
        "confused",
        "slowagent",
        "muninn",
        "urdr",
        "skuld",
        "vefr",
        "test",
        "kvasir",
        "test-agent",
        "default-agent",
        "huginn",
        "sjofn",
        "verdandi",
        "a",
        "b",
        "fast_agent",
        "slow_agent",
    ];

    /// Set up a test workspace with required directories and template files.
    fn setup_test_workspace(dir: &std::path::Path) {
        // Create template directory with the bootstrap system prompt template.
        let tmpl_dir = dir.join("templates/bootstrap");
        std::fs::create_dir_all(&tmpl_dir).unwrap();
        std::fs::write(
            tmpl_dir.join("system-prompt.jinja"),
            "{% if spawn_essence %}{{ spawn_essence }}{% endif %}\n\
             {% if identity %}{{ identity }}{% endif %}\n\
             {% for learning in learnings %}{{ learning }}\n{% endfor %}\n\
             {% for gate in governance_gates %}{{ gate }}\n{% endfor %}",
        )
        .unwrap();

        // Create artifacts directory for envelope validation.
        std::fs::create_dir_all(dir.join("artifacts")).unwrap();
        std::fs::write(dir.join("artifacts/output.md"), "test content").unwrap();

        // Create identity configs for all test agents so PreSpawn hooks pass.
        for agent in TEST_AGENTS {
            let agent_dir = dir.join(format!("config/agents/{}", agent));
            std::fs::create_dir_all(&agent_dir).unwrap();
            std::fs::write(
                agent_dir.join("identity.toml"),
                "[identity]\nname = \"test\"\n",
            )
            .unwrap();
        }
    }

    fn test_governance_config(dir: &std::path::Path) -> GovernanceConfig {
        let mut config = GovernanceConfig::default();
        for agent in TEST_AGENTS {
            config
                .archetype_profiles
                .insert((*agent).to_string(), "builder".into());
        }
        config.bootstrap.agent_config_dir =
            dir.join("config/agents").to_string_lossy().into_owned();
        config
    }

    /// Build a test runner with the given executor and optional hook set.
    async fn build_runner(
        executor: Arc<dyn AgentExecutor>,
        hook_set: Option<HookSet>,
        workspace_dir: &std::path::Path,
    ) -> AlzinaResult<AlzinaRunner> {
        setup_test_workspace(workspace_dir);

        let workspace = Arc::new(WorkspaceHandle::open(workspace_dir.to_path_buf())?);
        let config = test_governance_config(workspace_dir);
        let learnings_config = LearningsConfig::default();

        let governance = match hook_set {
            Some(hs) => Arc::new(GovernanceLayer::with_hooks(
                config.clone(),
                workspace.clone(),
                hs,
            )?),
            None => Arc::new(GovernanceLayer::new(config.clone(), workspace.clone())?),
        };

        let sessions = Arc::new(SessionHierarchy::in_memory().await?);
        let store = Arc::new(alzina_governance::FileLearningsStore::new(
            workspace
                .root()
                .join(&learnings_config.learnings_dir)
                .to_string_lossy()
                .to_string(),
            learnings_config.max_entries_per_agent,
        ));
        let learnings = Arc::new(LearningsMerger::with_store(
            workspace.clone(),
            learnings_config,
            alzina_core::canonical_domain_mapping(),
            store,
        ));

        let tmpl_dir = workspace_dir.join("templates");
        let template_engine = Arc::new(TemplateEngine::new(&tmpl_dir)?);
        let bootstrap = Arc::new(BootstrapEngine::new(
            workspace.clone(),
            template_engine,
            alzina_governance::BootstrapConfig::default(),
            config,
            None,
            None,
        ));

        Ok(AlzinaRunner::new(
            governance,
            bootstrap,
            sessions,
            learnings,
            executor,
            workspace,
            "test-model".to_string(),
            Duration::from_secs(30),
        ))
    }

    /// Build a runner with a no-op progress emitter attached. Selects
    /// the watchdog code path in `execute_lifecycle` for tests that
    /// need to exercise idle-timer behaviour.
    async fn build_runner_with_emitter(
        executor: Arc<dyn AgentExecutor>,
        workspace_dir: &std::path::Path,
    ) -> AlzinaResult<AlzinaRunner> {
        let runner = build_runner(executor, None, workspace_dir).await?;
        let noop: ExecutorEventEmitter = Arc::new(|_ev: AlzinaEvent| {});
        Ok(runner.with_progress_emitter(noop))
    }

    fn test_node(agent: &str, task: &str) -> CompNode {
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

    // ── AC-O1: Full lifecycle with mock executor ────────────────────────────

    #[tokio::test]
    async fn full_lifecycle_returns_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&well_formed_envelope()));
        let runner = build_runner(executor.clone(), None, dir.path())
            .await
            .unwrap();

        let node = test_node("smidr", "analyse workspace");
        let result = runner.spawn(&node, None, None).await.unwrap();

        assert_eq!(result.envelope.status, EnvelopeStatus::Complete);
        assert!(executor.was_executed());
        assert!(!result.raw.is_empty());
        assert!(result.envelope.signal.is_some());
    }

    // ── AC-O2: Lifecycle fires governance hooks in order ────────────────────

    #[tokio::test]
    async fn lifecycle_fires_prespawn_and_complete_hooks() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&well_formed_envelope()));

        let pre_spawn_fired = Arc::new(AtomicBool::new(false));
        let complete_fired = Arc::new(AtomicBool::new(false));

        struct TrackingHook {
            name: String,
            flag: Arc<AtomicBool>,
        }

        #[async_trait]
        impl HookHandler for TrackingHook {
            fn name(&self) -> &str {
                &self.name
            }
            fn priority(&self) -> u32 {
                10
            }
            fn blocking(&self) -> bool {
                false
            }
            async fn execute(
                &self,
                _event: &LifecycleEvent,
                _saga: &mut SagaState,
            ) -> AlzinaResult<HookAction> {
                self.flag.store(true, Ordering::SeqCst);
                Ok(HookAction::Continue)
            }
        }

        let hook_set = HookSet::new()
            .add(
                EventFilter::PreSpawn,
                Box::new(TrackingHook {
                    name: "pre-spawn-tracker".into(),
                    flag: pre_spawn_fired.clone(),
                }),
            )
            .add(
                EventFilter::Complete,
                Box::new(TrackingHook {
                    name: "complete-tracker".into(),
                    flag: complete_fired.clone(),
                }),
            );

        let runner = build_runner(executor, Some(hook_set), dir.path())
            .await
            .unwrap();

        let node = test_node("galdr", "build something");
        runner.spawn(&node, None, None).await.unwrap();

        assert!(
            pre_spawn_fired.load(Ordering::SeqCst),
            "PreSpawn hook should have fired"
        );
        assert!(
            complete_fired.load(Ordering::SeqCst),
            "Complete hook should have fired"
        );
    }

    // ── AC-O5: PreSpawn block prevents execution ────────────────────────────

    #[tokio::test]
    async fn prespawn_block_prevents_execution() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&well_formed_envelope()));

        let hook_set = HookSet::new().add(
            EventFilter::PreSpawn,
            Box::new(BlockingHook {
                reason: "agent not authorised".into(),
            }),
        );

        let runner = build_runner(executor.clone(), Some(hook_set), dir.path())
            .await
            .unwrap();

        let node = test_node("rogue", "do bad things");
        let result = runner.spawn(&node, None, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("blocked") || err.contains("Blocked"),
            "error was: {err}"
        );
        assert!(
            !executor.was_executed(),
            "executor should NOT have been called"
        );
    }

    // ── Timeout → error + session marked failed ─────────────────────────────

    #[tokio::test]
    async fn timeout_returns_error_and_marks_session_failed() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::with_delay(
            &well_formed_envelope(),
            Duration::from_secs(5),
        ));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        let node = CompNode {
            agent_id: AgentId::new("slowagent"),
            task: "take forever".to_string(),
            model_override: None,
            timeout: Some(Duration::from_millis(50)),
            weave_id: None,
            dispatch_id: None,
            sampling: None,
        };

        let result = runner.spawn(&node, None, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("timed out"),
            "error should mention timeout, was: {err}"
        );
    }

    // ── Envelope parse failure → error handling ─────────────────────────────

    #[tokio::test]
    async fn envelope_parse_failure_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&malformed_envelope()));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        // Spawn as sub-agent (strict envelope parsing)
        let root_id = alzina_core::SessionId::new();
        runner
            .sessions
            .create_root(&root_id, &alzina_core::AgentId::new("vefr"), None)
            .await
            .unwrap();
        let node = test_node("confused", "produce garbage");
        let result = runner.spawn(&node, Some(&root_id), None).await;

        assert!(result.is_err());
    }

    // ── CONTEXT_UPDATE → learnings merged ───────────────────────────────────

    #[tokio::test]
    async fn context_update_merges_learnings() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&well_formed_envelope()));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        let node = test_node("smidr", "workspace analysis");
        let result = runner.spawn(&node, None, None).await.unwrap();

        assert!(result.envelope.context_update.is_some());

        // With store-backed merger, learnings go to domain-mapped path
        let domain_path = dir.path().join("learnings/implementation/_index.md");
        let learnings_content = std::fs::read_to_string(&domain_path).unwrap();
        assert!(
            learnings_content.contains("always validate config at load time"),
            "learnings should contain the CONTEXT_UPDATE, got: {learnings_content}"
        );
    }

    // ── Session hierarchy parent links correct ──────────────────────────────

    #[tokio::test]
    async fn session_hierarchy_parent_links_correct() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&envelope_no_context_update()));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        // Create a root session.
        let root_session = SessionId::new();
        let root_agent = AgentId::new("vefr");
        let weave = WeaveId::new("test-weave");
        runner
            .sessions
            .create_root(&root_session, &root_agent, Some(&weave))
            .await
            .unwrap();

        // Spawn a child.
        let node = test_node("muninn", "research something");
        let result = runner
            .spawn(&node, Some(&root_session), Some(&weave))
            .await
            .unwrap();

        let child_node = runner
            .sessions
            .get(&result.session_id)
            .await
            .unwrap()
            .expect("child session should exist");

        assert_eq!(child_node.parent.as_ref().unwrap(), &root_session);
        assert_eq!(child_node.agent_id.as_str(), "muninn");
        assert_eq!(child_node.weave_id.as_ref().unwrap().as_str(), "test-weave");
    }

    // ── Root session (no parent) ────────────────────────────────────────────

    #[tokio::test]
    async fn spawn_without_parent_creates_root() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&envelope_no_context_update()));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        let node = test_node("galdr", "create something");
        let result = runner.spawn(&node, None, None).await.unwrap();

        let session = runner
            .sessions
            .get(&result.session_id)
            .await
            .unwrap()
            .expect("session should exist");

        assert!(session.parent.is_none());
        assert!(matches!(
            session.status,
            alzina_core::session::SessionStatus::Complete(EnvelopeStatus::Complete)
        ));
    }

    // ── Invalid agent_id ────────────────────────────────────────────────────

    #[test]
    fn invalid_agent_id_rejected_at_construction() {
        // AgentId::new() now validates at construction time.
        let result = AgentId::try_new("../../etc/passwd");
        assert!(result.is_err());
    }

    // ── Signals extracted correctly ─────────────────────────────────────────

    #[tokio::test]
    async fn signals_extracted_from_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&well_formed_envelope()));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        let node = test_node("urdr", "look at history");
        let result = runner.spawn(&node, None, None).await.unwrap();

        assert!(
            !result.signals.is_empty(),
            "should extract signals from envelope"
        );
    }

    // ── Idle watchdog: progress events reset the timer ──────────────────────

    /// With a progress emitter attached, the watchdog must keep the
    /// spawn alive as long as events flow at intervals shorter than
    /// `timeout`, even when total runtime exceeds `timeout`.
    #[tokio::test]
    async fn idle_watchdog_resets_on_progress_events() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(StreamingMockExecutor {
            response: well_formed_envelope(),
            interval: Duration::from_millis(40),
            iterations: 8, // ~320ms total — well past the 150ms idle timeout
        });
        let runner = build_runner_with_emitter(executor, dir.path())
            .await
            .unwrap();

        let node = CompNode {
            agent_id: AgentId::new("smidr"),
            task: "stream events".to_string(),
            model_override: None,
            timeout: Some(Duration::from_millis(150)),
            weave_id: None,
            dispatch_id: None,
            sampling: None,
        };

        let result = runner.spawn(&node, None, None).await;
        assert!(
            result.is_ok(),
            "watchdog should reset on each progress event, got: {:?}",
            result.err()
        );
    }

    /// With a progress emitter attached but no events flowing, the
    /// watchdog must abort with an idle-timeout error.
    #[tokio::test]
    async fn idle_watchdog_aborts_when_no_events_flow() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::with_delay(
            &well_formed_envelope(),
            Duration::from_secs(5),
        ));
        let runner = build_runner_with_emitter(executor, dir.path())
            .await
            .unwrap();

        let node = CompNode {
            agent_id: AgentId::new("slowagent"),
            task: "be silent".to_string(),
            model_override: None,
            timeout: Some(Duration::from_millis(100)),
            weave_id: None,
            dispatch_id: None,
            sampling: None,
        };

        let result = runner.spawn(&node, None, None).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no progress events"),
            "expected idle-timeout message, got: {err}"
        );
    }

    // ── 260515-ndk Task 2: AgentExecutor::execute_with_envelope branch ──
    //
    // Test 2 from the plan: when execute_with_envelope returns
    // (raw, Some(env)), the runner uses `env` directly. We can't easily
    // assert "governance.parse_envelope was NOT called" without a custom
    // governance mock, so we drive the proof via observable side-effects:
    //
    //   (a) The typed envelope's fields appear verbatim in the
    //       SpawnResult.envelope (a value the prose parse path could not
    //       produce because the raw string is intentionally non-canonical).
    //   (b) The SpawnResult.raw matches render_envelope_as_prose(&env),
    //       proving Task 2's re-render-from-typed step fired.
    //
    // Test 3 from the plan (lenient prose fallback, typed=None) is
    // already covered by the existing `full_lifecycle_returns_envelope`
    // test above — MockExecutor uses the default trait impl which
    // returns (raw, None), and the runner falls through to the
    // strict-then-lenient prose parser unchanged.

    /// Mock executor that overrides `execute_with_envelope` to return a
    /// typed Envelope alongside non-canonical raw text. Proves the runner
    /// picks the typed envelope (no prose parse) and re-renders raw via
    /// `render_envelope_as_prose` for downstream consumers.
    struct TypedEnvelopeExecutor {
        envelope: Envelope,
    }

    #[async_trait]
    impl AgentExecutor for TypedEnvelopeExecutor {
        async fn execute(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
        ) -> AlzinaResult<String> {
            // Default impl on the trait routes through here when
            // execute_with_envelope is NOT overridden, but we DO override
            // it below, so this returns a clearly-broken raw to make any
            // accidental fallthrough trivially detectable.
            Ok("BROKEN: prose parse path should not run when typed envelope captured".into())
        }

        async fn execute_with_envelope(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
            _session_id: &SessionId,
            _emitter: Option<ExecutorEventEmitter>,
        ) -> AlzinaResult<(String, Option<Envelope>)> {
            // Return non-canonical garbage as `raw` to prove the runner
            // does NOT fall back to prose parsing when typed envelope is Some.
            // If the runner tried, the test would fail at envelope_parse_failure.
            Ok((
                "this is intentionally-non-canonical prose with no STATUS marker".to_string(),
                Some(self.envelope.clone()),
            ))
        }
    }

    #[tokio::test]
    async fn typed_envelope_path_skips_prose_parse() {
        let dir = tempfile::tempdir().unwrap();
        let typed_env = Envelope {
            status: EnvelopeStatus::Partial,
            artifacts: vec![std::path::PathBuf::from("artifacts/x.md")],
            signal: Some("typed path proof".into()),
            tensions: None,
            emergent: None,
            next: None,
            context_update: None,
        };
        let executor = Arc::new(TypedEnvelopeExecutor {
            envelope: typed_env.clone(),
        });
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        // Create artifacts/x.md so the validator sees the path exists.
        std::fs::write(dir.path().join("artifacts/x.md"), "x").unwrap();

        let node = test_node("kvasir", "do stuff");
        let result = runner.spawn(&node, None, None).await.unwrap();

        // (a) Typed envelope reaches the SpawnResult.
        assert_eq!(result.envelope.status, EnvelopeStatus::Partial);
        assert_eq!(result.envelope.signal.as_deref(), Some("typed path proof"));
        assert_eq!(result.envelope.artifacts.len(), 1);

        // (b) raw was re-rendered from the typed envelope (proves Task 2
        // "synthesise canonical raw" step fired — the original raw was
        // garbage with no STATUS line).
        let expected_raw = alzina_governance::envelope::render_envelope_as_prose(&typed_env);
        assert_eq!(
            result.raw, expected_raw,
            "raw must be re-rendered from typed envelope via render_envelope_as_prose"
        );
    }

    /// Root session + missing tool call → hard error (lenient fallback removed).
    /// When executor returns (malformed_prose, None) for a root session,
    /// the prose parse fails and the runner returns Err(EnvelopeParse).
    #[tokio::test]
    async fn root_session_missing_tool_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&malformed_envelope()));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        // No parent → root session
        let node = test_node("sjofn", "produce garbage without tool");
        let result = runner.spawn(&node, None, None).await;

        assert!(
            result.is_err(),
            "root session with missing return_envelope tool must return Err"
        );
    }

    /// Sub-agent + missing tool call → hard error (unchanged behaviour,
    /// regression lock after lenient path removal).
    /// When executor returns (malformed_prose, None) for a sub-agent session,
    /// the prose parse fails and the runner returns Err(EnvelopeParse).
    #[tokio::test]
    async fn sub_agent_missing_tool_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&malformed_envelope()));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        // Spawn as sub-agent (strict envelope parsing)
        let root_id = alzina_core::SessionId::new();
        runner
            .sessions
            .create_root(&root_id, &alzina_core::AgentId::new("vefr"), None)
            .await
            .unwrap();
        let node = test_node("confused", "produce garbage without tool");
        let result = runner.spawn(&node, Some(&root_id), None).await;

        assert!(
            result.is_err(),
            "sub-agent with missing return_envelope tool must return Err"
        );
    }

    // ── No CONTEXT_UPDATE → no learnings merge ──────────────────────────────

    #[tokio::test]
    async fn no_context_update_skips_learnings() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::new(&envelope_no_context_update()));
        let runner = build_runner(executor, None, dir.path()).await.unwrap();

        let node = test_node("skuld", "plan future");
        runner.spawn(&node, None, None).await.unwrap();

        let ws = WorkspaceHandle::open(dir.path().to_path_buf()).unwrap();
        assert!(
            !ws.exists("learnings/skuld.md").unwrap_or(false),
            "no learnings file should be created without CONTEXT_UPDATE"
        );
    }
}
