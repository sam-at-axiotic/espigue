//! CompositionCompiler — transforms the recursive composition algebra into
//! an executable `CompiledGraph`.
//!
//! The composition algebra (`CompOp`) is a recursive AST where operators
//! compose other operators. ADK's `StateGraph` is too flat for nested
//! composition (no recursive subgraphs), so `CompiledGraph` is a custom
//! executor that calls nodes directly with `tokio::JoinSet` for parallelism.
//!
//! # Architecture
//!
//! ```text
//! CompOp (recursive AST)
//!   └─ compile() → CompiledGraph (execution plan + runner)
//!       └─ execute() → walks AST, dispatches nodes, wires state
//! ```
//!
//! # State Channel Conventions
//!
//! Follows existing codebase grain:
//! - `{name}:envelope` — parsed Envelope from agent return
//! - `{name}:raw` — raw text output
//! - `{name}:status` — session status string
//! - `{gate_name}:_gate:verdict` / `{gate_name}:_gate:route` — namespaced gate evaluation results
//! - `_meta:iteration` — loop counter

use indexmap::IndexMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

use adk_rust::graph::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info, warn};

use alzina_core::composition::{ExhaustAction, GateCriteria, GateFailAction, GateVerdict};
use alzina_core::envelope::{Envelope, Signal};
#[cfg(test)]
use alzina_core::event::ScopedEnvelope;
use alzina_core::identity::{AgentId, SessionId, WeaveId};
use alzina_core::{AlzinaError, AlzinaResult, QualityGate};
#[cfg(test)]
use alzina_core::AlzinaEvent;

use crate::composition::edges::{LoopDecision, LoopEdge, LoopEdgeConfig};
use crate::composition::leaf_hook::{CompositionLeafHook, noop_hook};
use crate::composition::nodes::{SpawnNode, SynthesisNode};
use crate::composition::parser::{
    AncestorSummary, CompositionContext, GateFeedback, ReservedChannelState,
};
use crate::runner::alzina_runner::{AlzinaRunner, CompNode};

// ── Limits ────────────────────────────────────────────────────────────────────

/// Maximum recursion depth for execute_op. Prevents stack overflow from
/// deeply nested or recursive compositions at runtime.
const MAX_EXECUTE_DEPTH: usize = 50;

/// Default synthesis prompt used when `<Synthesise>` omits the `task` attribute.
/// Source of truth: `docs/composition-grammar.md` §3.4.
/// Anti-averaging cue lives in the prompt (not the parser — SPEC out-of-scope).
pub const DEFAULT_SYNTHESIS_PROMPT: &str =
    "Synthesise the upstream branches without averaging. Name tensions explicitly.";

/// Maximum number of fan-out prompts allowed in a single FanOut operation.
const MAX_FANOUT_PROMPTS: usize = 100;

// ── Recursive Algebra AST ───────────────────────────────────────────────────

/// Recursive composition algebra. Unlike `alzina_core::CompOp` (flat,
/// operates on `CompNode` directly), this AST allows operators to nest
/// arbitrarily: `Gate(Sequential([A, B]), spec)` etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompOp {
    /// Single agent dispatch.
    Spawn(SpawnSpec),

    /// → Sequential: execute in order, output flows forward.
    Sequential(Vec<CompOp>),

    /// ∥ Parallel: execute concurrently.
    Parallel(Vec<CompOp>),

    /// ⊕ Synthesis: wraps parallel branches, adds synthesis step.
    Synthesise(Box<CompOp>, SynthesisSpec),

    /// ⊘ Quality gate: run inner, evaluate, pass or reject.
    Gate(Box<CompOp>, GateSpec),

    /// ↻ Bounded loop: repeat inner up to max iterations.
    Loop(Box<CompOp>, LoopSpec),

    /// ↻? Conditional loop: repeat with gate until pass or exhaust.
    ConditionalLoop(Box<CompOp>, ConditionalLoopSpec),

    /// ? Conditional: evaluate predicates, run first match.
    Conditional(Vec<(Predicate, CompOp)>),

    /// [n] Fan-out: same agent, N varied prompts.
    FanOut(SpawnSpec, Vec<String>),
}

/// Specification for spawning a single agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnSpec {
    pub agent_id: AgentId,
    pub task_template: String,
    pub model_override: Option<String>,

    pub timeout: Option<Duration>,

    /// P5-DEBUG-DISPATCH (Fix A): pre-allocated root session id from the
    /// caller. When `Some(id)`, the runner uses this id verbatim — both
    /// for hierarchy registration AND as the session_id stamped on bus
    /// events emitted via `execute_with_emitter`. The caller (e.g.
    /// `SessionManager::dispatch_composition_with_id_tx`) MUST also have
    /// already called `SessionHierarchy::create_root` with this same id;
    /// the runner skips its own `create_root` to avoid double-insertion.
    ///
    /// When `None`, the runner allocates a fresh id and registers the
    /// session itself, preserving the original spawn-from-test path.
    ///
    /// See `docs/p5-debug-dispatch-synth.md` for the full diagnosis.
    #[serde(skip)]
    pub session_id_override: Option<alzina_core::identity::SessionId>,

    /// Phase 9 R2: weave the spawn belongs to. Inherited from the parent
    /// dispatch's weave_id during composition execution. `#[serde(default)]`
    /// keeps existing JSON payloads (which omit this field entirely)
    /// deserialising to `None` — backwards-compat per LANDMINE 4 + 8.
    #[serde(default)]
    pub weave_id: Option<WeaveId>,

    /// Plan 10-05 (followup): parser-assigned static node identifier
    /// (e.g. `"skuld_0"`). Threaded by `parse_spawn` so `execute_spawn`
    /// invokes the leaf hook with the SAME key that `dispatch_compose`
    /// used when populating `static_leaf_map`. Without this, the compiler
    /// generates a runtime name via `next_name` whose per-`CompiledGraph`
    /// counter (`idx * 1000` offset in parallel branches) diverges from
    /// the parser's globally-incrementing `state.next_index`, the hook
    /// lookup falls through to the dynamic-leaf path, and the actual
    /// child runs under a SessionId that no watcher in `register_dispatch`
    /// is listening for — wedging the dispatch registry. `None` means
    /// "no parser-assigned id, generate one" (preserves the ad-hoc path).
    #[serde(default)]
    pub node_id: Option<String>,

    /// Phase 1B substrate cascade: daemon-allocated `dispatch_id` of the
    /// chat-tool dispatch this spec belongs to. Threaded from
    /// `alzina-daemon::DispatchRequest.dispatch_id` through
    /// `SessionManager::dispatch_composition_with_id_tx` into the spec, then
    /// forwarded onto `CompNode.dispatch_id` by `spec_to_comp_node` /
    /// `execute_spawn`. The runner stamps it onto `SpawnCompleted`, which
    /// the memory event sink writes to `stitch_records.dispatch_id`.
    ///
    /// `#[serde(default)]` preserves wire-format BC: existing JSON
    /// payloads omitting this field deserialise to `None` (LANDMINE 4 + 8
    /// pattern, same as `weave_id` / `node_id`).
    ///
    /// `None` for non-chat-tool dispatch paths (CLI, internal triggers,
    /// composition-internal sub-spawns).
    #[serde(default)]
    pub dispatch_id: Option<String>,
}

impl SpawnSpec {
    /// R-WEAVE-SCOPE-001: typed scope key derived from `weave_id`.
    ///
    /// `None` weave_id maps to `Scope::SessionDefault` — the runner's
    /// canonical handle for SessionDefault dispatches (lightweight
    /// chat, health, observation). `Some(w)` maps to `Scope::Weave(w)`.
    ///
    /// The internal field stays `Option<WeaveId>` during the §5.4 BC
    /// window so existing JSON payloads keep deserialising (per
    /// LANDMINE 4 + 8). This getter is the structural handle runner
    /// code should consume — emitting `ScopeViolation` and gating
    /// per-weave records key on the typed value.
    pub fn scope(&self) -> alzina_core::Scope {
        match &self.weave_id {
            Some(w) => alzina_core::Scope::Weave(w.clone()),
            None => alzina_core::Scope::SessionDefault,
        }
    }
}

/// Specification for the synthesis step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisSpec {
    pub synthesiser: SpawnSpec,
    /// Explicit synthesis instruction from the XML's `<Synthesise task="…">`.
    /// When `None`, the runtime falls back to the §3.4 default prompt
    /// (`DEFAULT_SYNTHESIS_PROMPT`).
    ///
    /// Serde-skipped when `None` so existing JSON wire format for ad-hoc
    /// constructions (test fixtures, internal callers) stays byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
}

/// Specification for a quality gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSpec {
    pub criteria: GateCriteria,
    pub on_fail: GateFailAction,
}

/// Specification for a bounded loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopSpec {
    pub max_iterations: usize,
    pub on_exhaust: ExhaustAction,
}

/// Specification for a conditional loop (↻? — loop + gate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalLoopSpec {
    pub gate: GateCriteria,
    pub max_iterations: usize,
    pub on_exhaust: ExhaustAction,
}

/// Predicate for conditional routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Predicate {
    /// Route based on a state channel value.
    StateEquals {
        channel: String,
        value: serde_json::Value,
    },
    /// Route based on whether a channel exists.
    StateExists { channel: String },
    /// Default/fallback branch (always true).
    Default,
}

// ── Compiler Configuration ────────────────────────────────────────────────────

/// Configuration for the composition compiler.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Maximum number of concurrent parallel/fan-out tasks.
    /// Default: 10. Controls the tokio Semaphore permit count.
    pub max_concurrent: usize,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self { max_concurrent: 10 }
    }
}

// ── CompiledGraph ───────────────────────────────────────────────────────────

/// A compiled composition ready for execution.
///
/// Not a wrapper around ADK's `StateGraph` — it's a custom executor that
/// walks the `CompOp` AST directly. This gives us recursive composition,
/// nested parallelism, and loop control that `StateGraph` can't express.
pub struct CompiledGraph {
    op: CompOp,
    runner: Arc<AlzinaRunner>,
    quality_gate: Arc<dyn QualityGate>,
    counter: std::sync::atomic::AtomicUsize,
    config: CompilerConfig,
    semaphore: Arc<Semaphore>,
    /// D10-02: when set, the executor builds `CompositionContext` per spawn
    /// and passes it through `spawn_with_id`. When `None`, all dispatches
    /// are byte-identical to the ad-hoc path (no preamble, no substitution).
    compose_id: Option<String>,
    /// Plan 10-05: per-leaf dispatch hook. Called in `execute_spawn` before
    /// the runner dispatches the spawn, when `compose_id` is set. Lets
    /// `alzina-daemon` inject `register_dispatch` logic via `DaemonLeafHook`
    /// without coupling this crate to the daemon. Defaults to `NoopLeafHook`.
    leaf_hook: Arc<dyn CompositionLeafHook>,
}

/// Result of executing a compiled graph.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Final state after execution.
    pub state: State,
    /// All envelopes produced, keyed by node name.
    pub envelopes: IndexMap<String, Envelope>,
    /// Raw text outputs from each spawn, keyed by node name.
    pub raw_outputs: HashMap<String, String>,
    /// All signals extracted from all spawns.
    pub all_signals: Vec<Signal>,
    /// Emergent observations collected from all spawns: (node_id, text).
    pub emergent_observations: Vec<(String, String)>,
    /// Next steps collected from all spawns: (node_id, text).
    pub next_steps: Vec<(String, String)>,
}

impl CompiledGraph {
    /// Generate a unique node name for this execution.
    fn next_name(&self, prefix: &str) -> String {
        let n = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("{prefix}_{n}")
    }

    /// Execute the compiled graph (ad-hoc path — no composition context).
    ///
    /// `weave_id` is the dispatch-level weave identity; it is threaded through
    /// every recursive `execute_op` call so all spawned agents inherit it.
    pub async fn execute(&self, weave_id: Option<WeaveId>) -> AlzinaResult<ExecutionResult> {
        self.execute_inner(weave_id, &[]).await
    }

    /// Execute with a composition id, enabling CompositionContext building.
    ///
    /// When `compose_id` is set on the graph (via `execute_with_compose_id`),
    /// the Spawn arm builds a `CompositionContext` carrying the accumulated
    /// ancestors and passes it through `spawn_with_id` so the renderer injects
    /// the §4.3 preamble + §4.2 channel substitutions at dispatch time.
    ///
    /// This is the entrypoint for the daemon's `dispatch_compose` handler
    /// (Plan 05 / 06). The existing `execute()` path is unchanged.
    pub async fn execute_with_compose_id(
        &self,
        weave_id: Option<WeaveId>,
        compose_id: String,
    ) -> AlzinaResult<ExecutionResult> {
        // Temporarily set compose_id on the graph via a shadow struct — since
        // CompiledGraph.compose_id is set at construction, we clone just the
        // functional bits. For v1 this is a one-off execution context.
        let shadow = CompiledGraph {
            op: self.op.clone(),
            runner: self.runner.clone(),
            quality_gate: self.quality_gate.clone(),
            counter: std::sync::atomic::AtomicUsize::new(
                self.counter.load(std::sync::atomic::Ordering::SeqCst),
            ),
            config: self.config.clone(),
            semaphore: self.semaphore.clone(),
            compose_id: Some(compose_id),
            leaf_hook: Arc::clone(&self.leaf_hook),
        };
        shadow.execute_inner(weave_id, &[]).await
    }

    /// Execute with an explicit compose_id and leaf hook (Plan 10-05 daemon seam).
    ///
    /// The daemon's `dispatch_compose` handler uses this to inject a
    /// `DaemonLeafHook` that calls `register_dispatch` per leaf before
    /// the runner dispatches it.
    pub async fn execute_with_hook(
        &self,
        weave_id: Option<WeaveId>,
        compose_id: String,
        hook: Arc<dyn CompositionLeafHook>,
    ) -> AlzinaResult<ExecutionResult> {
        let shadow = CompiledGraph {
            op: self.op.clone(),
            runner: self.runner.clone(),
            quality_gate: self.quality_gate.clone(),
            counter: std::sync::atomic::AtomicUsize::new(
                self.counter.load(std::sync::atomic::Ordering::SeqCst),
            ),
            config: self.config.clone(),
            semaphore: self.semaphore.clone(),
            compose_id: Some(compose_id.clone()),
            leaf_hook: Arc::clone(&hook),
        };
        let result = shadow.execute_inner(weave_id, &[]).await;
        // Plan 11-01.1 gap-closure: drain any pre-registered leaves that never
        // received a terminal callback (e.g. Parallel branches aborted mid-flight
        // by JoinSet drop, Sequential children short-circuited by a preceding op's
        // failure, Conditional branches pruned by routing). Without this, those
        // orphan leaves stay in `DispatchRegistry.in_flight` forever. Fires once
        // regardless of whether execution succeeded or failed.
        hook.on_composition_terminal(&compose_id).await;
        result
    }

    /// Shared execution path used by both `execute` and `execute_with_compose_id`.
    ///
    /// `initial_ancestors` is the ancestor list inherited from the caller (used
    /// when the executor recurses into sub-graphs from the parallel branch path
    /// where sub-CompiledGraph instances don't carry composition state).
    async fn execute_inner(
        &self,
        weave_id: Option<WeaveId>,
        initial_ancestors: &[AncestorSummary],
    ) -> AlzinaResult<ExecutionResult> {
        let mut state = State::default();
        let mut envelopes = IndexMap::new();
        let mut raw_outputs = HashMap::new();
        let mut all_signals = Vec::new();

        self.execute_op(
            &self.op,
            &mut state,
            &mut envelopes,
            0,
            &mut raw_outputs,
            &mut all_signals,
            weave_id.as_ref(),
            initial_ancestors.to_vec(),
            ReservedChannelState::default(),
        )
        .await?;

        // Collect emergent observations and next steps from all envelopes.
        let emergent_observations: Vec<(String, String)> = envelopes
            .iter()
            .filter_map(|(node_id, env)| {
                env.emergent.as_ref().map(|e| (node_id.clone(), e.clone()))
            })
            .collect();
        let next_steps: Vec<(String, String)> = envelopes
            .iter()
            .filter_map(|(node_id, env)| env.next.as_ref().map(|n| (node_id.clone(), n.clone())))
            .collect();

        Ok(ExecutionResult {
            state,
            envelopes,
            raw_outputs,
            all_signals,
            emergent_observations,
            next_steps,
        })
    }

    /// Recursively execute a CompOp, mutating shared state.
    ///
    /// RT3-01: `depth` tracks recursion depth. Returns error if > MAX_EXECUTE_DEPTH.
    /// `weave_id` is threaded unchanged into every recursive call so all
    /// descendant spawns inherit the dispatch-level weave identity.
    ///
    /// `ancestors` is the accumulated list of ancestor summaries for the current
    /// execution context. Empty at the root; grown by the Sequential arm as
    /// siblings complete; passed unchanged to Parallel children (siblings excluded
    /// per §4.4 happens-before).
    ///
    /// `reserved` carries per-iteration reserved channel state: `prev_iteration`
    /// is snapshotted by Loop/ConditionalLoop arms from the prior iteration's
    /// envelope; `gate_feedback` is built by ConditionalLoop from the prior
    /// gate failure. Both resolve `{this:prev_iteration.*}` and
    /// `{this:gate.feedback}` substitutions at render time (§4.5).
    #[async_recursion::async_recursion]
    async fn execute_op(
        &self,
        op: &CompOp,
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        depth: usize,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&'async_recursion WeaveId>,
        ancestors: Vec<AncestorSummary>,
        reserved: ReservedChannelState,
    ) -> AlzinaResult<()> {
        if depth > MAX_EXECUTE_DEPTH {
            return Err(AlzinaError::Orchestration(
                "recursion depth exceeded".to_string(),
            ));
        }

        match op {
            CompOp::Spawn(spec) => {
                self.execute_spawn(
                    spec,
                    state,
                    envelopes,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    ancestors,
                    reserved,
                )
                .await
            }

            CompOp::Sequential(ops) => {
                // Sequential arm: each child sees prior siblings as ancestors.
                // Four-step ordering per W-09: (1) inner completes, (2) envelope
                // published into envelopes map, (3) last envelope → AncestorSummary,
                // (4) append BEFORE next sibling dispatches.
                let mut acc = ancestors.clone();
                let env_count_before = envelopes.len();
                for inner_op in ops {
                    self.execute_op(
                        inner_op,
                        state,
                        envelopes,
                        depth + 1,
                        raw_outputs,
                        all_signals,
                        weave_id,
                        acc.clone(),
                        reserved.clone(),
                    )
                    .await?;
                    // Pull the most-recent envelope inserted since the last
                    // child completed and append it as ancestor for the next.
                    let env_count_after = envelopes.len();
                    if env_count_after > env_count_before || !acc.is_empty() {
                        // Walk the newly inserted entries (diff since last iteration)
                        for (node_id, env) in envelopes
                            .iter()
                            .skip(acc.len() + ancestors.len().saturating_sub(acc.len()))
                        {
                            let agent_name = node_id
                                .split('_')
                                .take(node_id.split('_').count().saturating_sub(1))
                                .collect::<Vec<_>>()
                                .join("_");
                            let agent_name = if agent_name.is_empty() {
                                node_id.clone()
                            } else {
                                agent_name
                            };
                            acc.push(AncestorSummary {
                                node_id: node_id.clone(),
                                agent: agent_name,
                                status: env.status.clone(),
                                signal: env.signal.clone(),
                                artifact_paths: env
                                    .artifacts
                                    .iter()
                                    .map(|p| p.display().to_string())
                                    .collect(),
                                emergent: env.emergent.clone(),
                                next: env.next.clone(),
                            });
                        }
                    }
                }
                Ok(())
            }

            CompOp::Parallel(ops) => {
                // Parallel arm: each branch sees parent ancestors ONLY —
                // siblings excluded per §4.4 happens-before.
                self.execute_parallel(
                    ops,
                    state,
                    envelopes,
                    depth,
                    raw_outputs,
                    all_signals,
                    weave_id,
                )
                .await
            }

            CompOp::Synthesise(inner, synth_spec) => {
                // Execute inner (typically Parallel), then synthesise.
                // Synthesise arm: synthesiser sees parent ancestors ∪ ALL
                // descendants of inner (Pitfall 5 fix per §4.4).
                // reserved passes through to the inner op as-is.
                //
                // Plan 11-01.1 gap-closure: if `inner` fails, `?` short-circuits
                // before `execute_synthesis` runs. The synthesiser leaf's
                // pre-registered DispatchRegistry slot is drained by
                // `on_composition_terminal` at the end of `execute_with_hook`.
                let inner_env_start = envelopes.len();
                self.execute_op(
                    inner,
                    state,
                    envelopes,
                    depth + 1,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    ancestors.clone(),
                    reserved,
                )
                .await?;
                // Collect all envelopes produced by the inner op.
                let inner_descendant_summaries: Vec<AncestorSummary> = envelopes
                    .iter()
                    .skip(inner_env_start)
                    .map(|(node_id, env)| {
                        let agent = node_id
                            .split('_')
                            .take(node_id.split('_').count().saturating_sub(1))
                            .collect::<Vec<_>>()
                            .join("_");
                        let agent = if agent.is_empty() {
                            node_id.clone()
                        } else {
                            agent
                        };
                        AncestorSummary {
                            node_id: node_id.clone(),
                            agent,
                            status: env.status.clone(),
                            signal: env.signal.clone(),
                            artifact_paths: env
                                .artifacts
                                .iter()
                                .map(|p| p.display().to_string())
                                .collect(),
                            emergent: env.emergent.clone(),
                            next: env.next.clone(),
                        }
                    })
                    .collect();
                // Synthesiser's ancestors = parent ancestors ∪ inner descendants.
                let mut synth_ancestors = ancestors;
                synth_ancestors.extend(inner_descendant_summaries);
                self.execute_synthesis(
                    synth_spec,
                    state,
                    envelopes,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    synth_ancestors,
                )
                .await
            }

            CompOp::Gate(inner, gate_spec) => {
                self.execute_gate(
                    inner,
                    gate_spec,
                    state,
                    envelopes,
                    depth,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    ancestors,
                    reserved,
                )
                .await
            }

            CompOp::Loop(inner, loop_spec) => {
                self.execute_loop(
                    inner,
                    loop_spec,
                    state,
                    envelopes,
                    depth,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    ancestors,
                )
                .await
            }

            CompOp::ConditionalLoop(inner, cl_spec) => {
                self.execute_conditional_loop(
                    inner,
                    cl_spec,
                    state,
                    envelopes,
                    depth,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    ancestors,
                )
                .await
            }

            CompOp::Conditional(branches) => {
                self.execute_conditional(
                    branches,
                    state,
                    envelopes,
                    depth,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    ancestors,
                    reserved,
                )
                .await
            }

            CompOp::FanOut(spec, prompts) => {
                self.execute_fanout(
                    spec,
                    prompts,
                    state,
                    envelopes,
                    depth,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    ancestors,
                    reserved,
                )
                .await
            }
        }
    }

    /// Execute a single agent spawn.
    ///
    /// When `compose_id` is set on the graph and the spawn is inside a
    /// composition (derive from `ancestors`), builds a `CompositionContext`
    /// and passes it through `spawn_with_id` so the renderer injects the
    /// §4.3 preamble + §4.2 channel substitutions at dispatch time (D10-02).
    ///
    /// `reserved` is threaded from `execute_op` and carries the current
    /// reserved channel state (prev_iteration, gate_feedback) for §4.5
    /// substitution at render time. Loop/ConditionalLoop arms populate this
    /// before each iteration so the body spawn sees the correct prior-iteration
    /// envelope and gate feedback.
    async fn execute_spawn(
        &self,
        spec: &SpawnSpec,
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&WeaveId>,
        ancestors: Vec<AncestorSummary>,
        reserved: ReservedChannelState,
    ) -> AlzinaResult<()> {
        // Plan 10-05 (followup): prefer the parser-assigned `node_id` when it
        // was threaded through (`parse_spawn` sets `Some(id)`). This makes the
        // hook-lookup key in `DaemonLeafHook::static_leaf_map` match the key
        // `dispatch_compose` pre-registered. Falls back to `next_name` for
        // ad-hoc paths (`engine.spawn_single`, core→compiler translation in
        // `core_node_to_spec`) where no static plan exists.
        let name = spec
            .node_id
            .clone()
            .unwrap_or_else(|| self.next_name(&spec.agent_id.as_str().replace('/', "_")));

        // Phase 9 R2: derive effective weave_id — spec-level override wins over
        // dispatch-level (spec.weave_id is usually None; the dispatch arg is the
        // primary source). Falls back to None for un-weaved dispatches.
        let effective_weave: Option<WeaveId> = spec.weave_id.clone().or_else(|| weave_id.cloned());

        // spec.scope() is non-Option — the real compile gate at the leaf
        // boundary. See 15-08-SUMMARY.md for the BL-04 PATH A retrospective.

        // Build CompNode with effective weave_id so runner.spawn propagates it
        // onto the SpawnCompleted event (R1 wiring from Plan 02). Phase 1B
        // substrate cascade: also forward the chat-tool dispatch_id so the
        // memory event sink can stamp it onto `stitch_records.dispatch_id`.
        let comp_node = CompNode {
            agent_id: spec.agent_id.clone(),
            task: spec.task_template.clone(),
            model_override: spec.model_override.clone(),
            timeout: spec.timeout,
            weave_id: effective_weave.clone(),
            dispatch_id: spec.dispatch_id.clone(),
            sampling: None,
        };

        debug!(name = %name, agent = %spec.agent_id, weave = ?effective_weave, "executing spawn");

        // P5-DEBUG-DISPATCH (Fix A): if the outer dispatch path
        // pre-allocated a root session id, propagate it to the
        // `SpawnNode` so the runner stamps mid-turn bus events with the
        // exact id the streaming handler is filtering on. See
        // `docs/p5-debug-dispatch-synth.md`.
        let mut spawn_node = SpawnNode::new(&name, comp_node, self.runner.clone());
        if let Some(ref wid) = effective_weave {
            spawn_node = spawn_node.with_weave_id(wid.clone());
        }
        if let Some(session_id) = spec_session_id_override(spec) {
            spawn_node = spawn_node.with_session_id_override(session_id);
        }

        // D10-02 / Plan 10-05: when composition mode is active (compose_id set),
        // build a CompositionContext and invoke the leaf hook before dispatch.
        // The hook is called for side effects (daemon's DaemonLeafHook calls
        // `register_dispatch` so the DispatchRegistry counter + watcher fires).
        // The hook's returned SessionId is used as `session_id_override`, and
        // the session is pre-registered in the hierarchy so `spawn_with_id`
        // can use `caller_registered_session=true` correctly.
        //
        // E1 / D11-04: capture the hook-allocated SessionId so we can pass it
        // to the terminal on_leaf_completed / on_leaf_failed callbacks below.
        // This is the SAME id `register_dispatch` is watching, ensuring the
        // watcher fires on publish.
        let mut hook_sid_for_terminal: Option<SessionId> = None;
        if let Some(ref cid) = self.compose_id {
            // Invoke the leaf hook. DaemonLeafHook registers in DispatchRegistry
            // and returns a pre-allocated session id. NoopLeafHook allocates a
            // fresh id with no side effects.
            let (hook_sid, wrapped_task) = self
                .leaf_hook
                .on_leaf_dispatch(cid, &name, spec.agent_id.as_str(), &spec.task_template, None)
                .await?;

            // Pre-register the hook-allocated session in SessionHierarchy so
            // `spawn_with_id` (called by SpawnNode::execute) can use
            // caller_registered_session=true without failing on hierarchy lookup.
            // Composition spawns have no parent (root-level within the composition).
            self.runner
                .sessions()
                .create_root(&hook_sid, &spec.agent_id, weave_id)
                .await
                .map_err(|e| {
                    AlzinaError::Orchestration(format!(
                        "leaf hook: failed to register session for {}: {e}",
                        name
                    ))
                })?;

            let ctx = CompositionContext {
                compose_id: cid.clone(),
                node_id: name.clone(),
                rationale: None, // rationale_map lookup deferred to Plan 06
                ancestors,
                envelopes: std::sync::Arc::new(envelopes.clone()),
                raw_outputs: std::sync::Arc::new(raw_outputs.clone()),
                reserved,
            };
            hook_sid_for_terminal = Some(hook_sid.clone());
            spawn_node = spawn_node
                .with_task(wrapped_task)
                .with_composition_context(ctx)
                .with_session_id_override(hook_sid);
        }

        let ctx = NodeContext::new(state.clone(), ExecutionConfig::new("compiler"), 0);

        // E1 / D11-04: run the spawn, then invoke the leaf hook's terminal
        // callback in the Ok/Err arm BEFORE bubbling the error. This keeps
        // the daemon's DispatchRegistry watcher fed even when a leaf fails.
        let result = spawn_node.execute(&ctx).await;
        if let (Some(cid), Some(hook_sid)) =
            (self.compose_id.as_deref(), hook_sid_for_terminal.as_ref())
        {
            match &result {
                Ok(output) => {
                    // Materialise the envelope from the SpawnNode's state-update
                    // payload (the canonical site that records `{name}:envelope`).
                    let env_key = format!("{name}:envelope");
                    if let Some(env_val) = output.updates.get(&env_key)
                        && let Ok(env) = serde_json::from_value::<Envelope>(env_val.clone())
                    {
                        self.leaf_hook
                            .on_leaf_completed(cid, &name, spec.agent_id.as_str(), hook_sid, &env)
                            .await;
                    } else {
                        // No envelope materialised — still surface completion so
                        // the watcher decrements in_flight. Use a synthetic
                        // Complete-status envelope (matches the ad-hoc path's
                        // SessionCompleted semantics when no parse data exists).
                        let env = Envelope {
                            status: alzina_core::EnvelopeStatus::Complete,
                            artifacts: Vec::new(),
                            signal: None,
                            tensions: None,
                            emergent: None,
                            next: None,
                            context_update: None,
                        };
                        self.leaf_hook
                            .on_leaf_completed(cid, &name, spec.agent_id.as_str(), hook_sid, &env)
                            .await;
                    }
                }
                Err(e) => {
                    self.leaf_hook
                        .on_leaf_failed(
                            cid,
                            &name,
                            spec.agent_id.as_str(),
                            hook_sid,
                            &e.to_string(),
                        )
                        .await;
                }
            }
        }
        let output =
            result.map_err(|e| AlzinaError::Orchestration(format!("spawn {name} failed: {e}")))?;

        // Merge output into state
        for (k, v) in output.updates {
            state.insert(k, v);
        }

        // Extract envelope for result tracking
        let envelope_key = format!("{name}:envelope");
        if let Some(env_val) = state.get(&envelope_key)
            && let Ok(env) = serde_json::from_value::<Envelope>(env_val.clone())
        {
            envelopes.insert(name.clone(), env);
        }

        // Extract raw output
        let raw_key = format!("{name}:raw");
        if let Some(raw_val) = state.get(&raw_key)
            && let Some(raw_str) = raw_val.as_str()
        {
            raw_outputs.insert(name.clone(), raw_str.to_owned());
        }

        // Extract signals
        let signals_key = format!("{name}:signals");
        if let Some(sig_val) = state.get(&signals_key)
            && let Ok(sigs) = serde_json::from_value::<Vec<Signal>>(sig_val.clone())
        {
            all_signals.extend(sigs);
        }

        Ok(())
    }

    /// Execute parallel branches using tokio::JoinSet.
    ///
    /// M-01: Branch state keys are namespaced with `branch_{idx}:` to prevent
    /// last-writer-wins collisions when merging back into parent state.
    /// M-04: Concurrency is bounded by `self.semaphore`.
    ///
    /// D10-02: parallel branches each get ONLY the parent's ancestors (siblings
    /// excluded per §4.4). compose_id is propagated into each sub-graph.
    async fn execute_parallel(
        &self,
        ops: &[CompOp],
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        depth: usize,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&WeaveId>,
    ) -> AlzinaResult<()> {
        if ops.is_empty() {
            return Ok(());
        }
        if ops.len() == 1 {
            return self
                .execute_op(
                    &ops[0],
                    state,
                    envelopes,
                    depth + 1,
                    raw_outputs,
                    all_signals,
                    weave_id,
                    vec![], // single parallel branch has no ancestors
                    ReservedChannelState::default(),
                )
                .await;
        }

        // M-04: Spawn each branch in a JoinSet, bounded by semaphore.
        // Each branch executes a fresh CompiledGraph rooted at its own sub-op;
        // branch envelopes are merged back into the parent `state` below
        // (last-writer-wins per §4.4). State is NOT threaded into branches —
        // siblings cannot read one another's intermediate state by design.
        let weave_id_owned: Option<WeaveId> = weave_id.cloned();
        let compose_id_owned = self.compose_id.clone();
        let leaf_hook_owned = Arc::clone(&self.leaf_hook);
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, op) in ops.iter().enumerate() {
            let op = op.clone();
            let runner = self.runner.clone();
            let quality_gate = self.quality_gate.clone();
            let sem = self.semaphore.clone();
            let config = self.config.clone();
            let branch_weave = weave_id_owned.clone();
            let branch_compose_id = compose_id_owned.clone();
            let branch_hook = Arc::clone(&leaf_hook_owned);

            join_set.spawn(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|_| AlzinaError::Orchestration("semaphore closed".to_string()));
                if let Err(e) = _permit {
                    return (idx, Err(e));
                }
                let compiled = CompiledGraph {
                    op: op.clone(),
                    runner,
                    quality_gate,
                    counter: std::sync::atomic::AtomicUsize::new(idx * 1000),
                    semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
                    config,
                    // D10-02: propagate compose_id — each branch is spawned with
                    // parent's ancestors only (siblings excluded per §4.4).
                    compose_id: branch_compose_id,
                    // Plan 10-05: propagate leaf hook into parallel branches.
                    leaf_hook: branch_hook,
                };
                let result = compiled.execute(branch_weave).await;
                (idx, result)
            });
        }

        // Collect all branch results. A branch failure does NOT short-circuit
        // sibling branches — siblings drain to completion and contribute their
        // envelopes to the merged state. Per-leaf SessionFailed events have
        // already been published by the leaf hook for any leaves that errored;
        // the only thing the prior `return Err(e)` did was kill in-flight
        // siblings via JoinSet drop, which is the cascade we are removing.
        //
        // A tokio task panic (`Err(join_err)`) remains a hard abort — that is
        // a programming bug, not a partial-result condition.
        //
        // If EVERY branch errored, return Err so the parent op (Sequential /
        // Synthesise / Gate) sees the composition genuinely failed. Otherwise
        // return Ok and let downstream consumers react to the mixed envelope
        // set — synthesisers substitute with what resolved, gates apply their
        // status criteria to the partial bag.
        let mut branch_results: Vec<(usize, ExecutionResult)> = Vec::new();
        let mut branch_failures: Vec<(usize, AlzinaError)> = Vec::new();
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok((idx, Ok(exec_result))) => {
                    branch_results.push((idx, exec_result));
                }
                Ok((idx, Err(e))) => {
                    warn!(
                        branch = idx,
                        error = %e,
                        "parallel branch failed; siblings will drain to completion"
                    );
                    branch_failures.push((idx, e));
                }
                Err(join_err) => {
                    return Err(AlzinaError::Orchestration(format!(
                        "parallel branch join panic: {join_err}"
                    )));
                }
            }
        }

        if branch_results.is_empty() && !branch_failures.is_empty() {
            // Every branch errored — surface the first failure as the
            // composition-level error so the parent op treats this Parallel
            // as failed. Subsequent failures are already logged.
            let (_, first) = branch_failures.into_iter().next().unwrap();
            return Err(first);
        }

        // M-01: Merge branch states into parent with namespace prefixes.
        // Keys are prefixed with `branch_{idx}:` to prevent collisions.
        //
        // Bug 1 fix (compose_id 5e0d5858 — sjofn synthesis blocked):
        // The envelope IndexMap is keyed by leaf node_id (no prefix — see
        // execute_op Spawn arm at ~line 793). Downstream consumers
        // (SynthesisNode::collect_branches at synthesis_node.rs:156) look
        // up envelopes from state via `{leaf_node_id}:envelope`. With only
        // prefixed state keys, that lookup misses every branch. We
        // additionally mirror each branch's envelope into the parent state
        // under the unprefixed `{leaf_node_id}:envelope` key so the
        // synthesiser can resolve them. Leaf node_ids are unique across
        // the composition by parser construction, so collisions are
        // impossible. The prefixed `branch_{idx}:{k}` keys remain for
        // traceability and for non-envelope state entries.
        branch_results.sort_by_key(|(idx, _)| *idx);
        for (idx, result) in branch_results {
            for (k, v) in result.state {
                state.insert(format!("branch_{idx}:{k}"), v);
            }
            for (k, v) in result.envelopes {
                // Restore unprefixed `{leaf}:envelope` state lookup path
                // for synthesiser consumers. Serialisation here is
                // infallible for Envelope (all fields serde-derived).
                if let Ok(env_val) = serde_json::to_value(&v) {
                    state.insert(format!("{k}:envelope"), env_val);
                }
                envelopes.insert(k, v);
            }
            for (k, v) in result.raw_outputs {
                raw_outputs.insert(k, v);
            }
            all_signals.extend(result.all_signals);
        }

        Ok(())
    }

    /// Execute synthesis: spawn a synthesiser with collected branch envelopes.
    ///
    /// `ancestors` carries the parent ancestors PLUS all descendants of the
    /// inner op (Pitfall 5 fix per §4.4 — synthesiser sees inner descendants).
    async fn execute_synthesis(
        &self,
        synth_spec: &SynthesisSpec,
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&WeaveId>,
        ancestors: Vec<AncestorSummary>,
    ) -> AlzinaResult<()> {
        // E2 / D11-05: prefer the parser-stamped synthesiser node_id so the
        // daemon's DaemonLeafHook lookup (keyed on node_id) succeeds and the
        // <Synthesise> path stops wedging.
        //
        // Bug 2 fix: `ancestors` (parent ancestors ∪ inner-op descendants,
        // built at compiler.rs:511-513) is now threaded into SynthesisNode
        // via with_ancestors below. Pre-fix it was discarded as a Phase 10
        // carryover, which left the synthesiser with no §4.3 preamble
        // context and (combined with Bug 4) no second channel for the
        // children's data. Bug 4's composition_context wiring consumes
        // this list when building the synth spawn's CompositionContext.
        let name = synth_spec
            .synthesiser
            .node_id
            .clone()
            .unwrap_or_else(|| self.next_name("synthesis"));

        // Synthesiser belongs to the same weave as the parallel branches.
        let effective_weave = synth_spec.synthesiser.weave_id.as_ref().or(weave_id);
        let mut synth_comp = spec_to_comp_node(&synth_spec.synthesiser);
        synth_comp.weave_id = effective_weave.cloned();

        // Resolve the synthesis task: use the explicit task from the spec when
        // set, falling back to the §3.4 default prompt when absent.
        // This is the canonical resolution site per D10-13.
        synth_comp.task = synth_spec
            .task
            .clone()
            .unwrap_or_else(|| DEFAULT_SYNTHESIS_PROMPT.to_string());

        // Build branch channels from currently collected envelopes
        let branch_channels: Vec<String> = envelopes.keys().cloned().collect();

        debug!(
            name = %name,
            branches = ?branch_channels,
            "executing synthesis"
        );

        let mut synth_node =
            SynthesisNode::new(&name, synth_comp, branch_channels, self.runner.clone())
                .with_ancestors(ancestors.clone());
        if let Some(wid) = effective_weave {
            synth_node = synth_node.with_weave_id(wid.clone());
        }

        // E1 / D11-04: when composition mode is active, invoke the leaf hook
        // for the synthesiser too so DaemonLeafHook::on_leaf_dispatch
        // looks up its pre-registered dispatch_id. Pre-register the
        // hook-allocated session in SessionHierarchy so spawn_with_id can
        // use caller_registered_session=true.
        //
        // Bug 4: while we're here, build a CompositionContext mirroring
        // the leaf-Spawn site (compiler.rs:703-711) so the renderer
        // applies §4.3 preamble + §4.2 channel substitutions to the
        // synth's task before dispatch.
        let mut hook_sid_for_terminal: Option<SessionId> = None;
        if let Some(ref cid) = self.compose_id {
            let (hook_sid, wrapped_task) = self
                .leaf_hook
                .on_leaf_dispatch(cid, &name, synth_spec.synthesiser.agent_id.as_str(), &synth_spec.synthesiser.task_template, None)
                .await?;
            self.runner
                .sessions()
                .create_root(&hook_sid, &synth_spec.synthesiser.agent_id, weave_id)
                .await
                .map_err(|e| {
                    AlzinaError::Orchestration(format!(
                        "leaf hook: failed to register session for {}: {e}",
                        name
                    ))
                })?;
            hook_sid_for_terminal = Some(hook_sid.clone());

            let synth_ctx = CompositionContext {
                compose_id: cid.clone(),
                node_id: name.clone(),
                rationale: None,
                ancestors: ancestors.clone(),
                envelopes: std::sync::Arc::new(envelopes.clone()),
                raw_outputs: std::sync::Arc::new(raw_outputs.clone()),
                reserved: ReservedChannelState::default(),
            };
            synth_node = synth_node
                .with_task(wrapped_task)
                .with_session_id_override(hook_sid)
                .with_composition_context(synth_ctx);
        }

        let ctx = NodeContext::new(state.clone(), ExecutionConfig::new("compiler"), 0);

        let result = synth_node.execute(&ctx).await;
        // E1 / D11-04: terminal callbacks for the synthesiser leaf — fires
        // exactly one on_leaf_completed/on_leaf_failed so the watcher
        // decrements in_flight.
        if let (Some(cid), Some(hook_sid)) =
            (self.compose_id.as_deref(), hook_sid_for_terminal.as_ref())
        {
            match &result {
                Ok(output) => {
                    let env_key = format!("{name}:envelope");
                    let env = output
                        .updates
                        .get(&env_key)
                        .and_then(|v| serde_json::from_value::<Envelope>(v.clone()).ok())
                        .unwrap_or_else(|| Envelope {
                            status: alzina_core::EnvelopeStatus::Complete,
                            artifacts: Vec::new(),
                            signal: None,
                            tensions: None,
                            emergent: None,
                            next: None,
                            context_update: None,
                        });
                    self.leaf_hook
                        .on_leaf_completed(
                            cid,
                            &name,
                            synth_spec.synthesiser.agent_id.as_str(),
                            hook_sid,
                            &env,
                        )
                        .await;
                }
                Err(e) => {
                    self.leaf_hook
                        .on_leaf_failed(
                            cid,
                            &name,
                            synth_spec.synthesiser.agent_id.as_str(),
                            hook_sid,
                            &e.to_string(),
                        )
                        .await;
                }
            }
        }
        let output = result
            .map_err(|e| AlzinaError::Orchestration(format!("synthesis {name} failed: {e}")))?;

        for (k, v) in output.updates {
            state.insert(k, v);
        }

        let envelope_key = format!("{name}:envelope");
        if let Some(env_val) = state.get(&envelope_key)
            && let Ok(env) = serde_json::from_value::<Envelope>(env_val.clone())
        {
            envelopes.insert(name.clone(), env);
        }

        // Extract raw output and signals from synthesis
        let raw_key = format!("{name}:raw");
        if let Some(raw_val) = state.get(&raw_key)
            && let Some(raw_str) = raw_val.as_str()
        {
            raw_outputs.insert(name.clone(), raw_str.to_owned());
        }
        let signals_key = format!("{name}:signals");
        if let Some(sig_val) = state.get(&signals_key)
            && let Ok(sigs) = serde_json::from_value::<Vec<Signal>>(sig_val.clone())
        {
            all_signals.extend(sigs);
        }

        Ok(())
    }

    /// Execute a quality gate: run inner, evaluate, apply fail action.
    async fn execute_gate(
        &self,
        inner: &CompOp,
        gate_spec: &GateSpec,
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        depth: usize,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&WeaveId>,
        ancestors: Vec<AncestorSummary>,
        reserved: ReservedChannelState,
    ) -> AlzinaResult<()> {
        // RT2-07: Namespace gate channels so parallel gates don't clobber each other.
        let gate_name = self.next_name("gate");

        // Execute the inner operation, threading reserved state through.
        self.execute_op(
            inner,
            state,
            envelopes,
            depth + 1,
            raw_outputs,
            all_signals,
            weave_id,
            ancestors,
            reserved,
        )
        .await?;

        // Find the most recent envelope to evaluate
        let last_envelope = envelopes.values().last().cloned().ok_or_else(|| {
            AlzinaError::Orchestration("gate: no envelope produced by inner operation".to_string())
        })?;

        // Evaluate gate
        let verdict = self
            .quality_gate
            .evaluate(&last_envelope, &gate_spec.criteria)
            .await?;

        let route = match &verdict {
            GateVerdict::Pass => "pass",
            GateVerdict::Fail { .. } => "fail",
            GateVerdict::Deferred { .. } => "fail",
        };

        debug!(gate = %gate_name, route = %route, "gate verdict");

        // RT2-07: Namespace gate state channels: {gate_name}:_gate:*
        let verdict_key = format!("{gate_name}:_gate:verdict");
        let route_key = format!("{gate_name}:_gate:route");

        state.insert(
            verdict_key,
            serde_json::to_value(&verdict).unwrap_or(json!(null)),
        );
        state.insert(route_key.clone(), json!(route));

        if route == "fail" {
            match &gate_spec.on_fail {
                GateFailAction::Escalate => {
                    warn!("gate failed — escalating");
                    return Err(AlzinaError::Orchestration(
                        "gate failed: escalated to operator".to_string(),
                    ));
                }
                GateFailAction::Degrade(msg) => {
                    warn!(msg = %msg, "gate failed — degrading");
                    // RT2-03: Write route as "degraded" so downstream can
                    // distinguish degraded-pass from clean-pass.
                    state.insert(route_key.clone(), json!("degraded"));
                    state.insert(format!("{gate_name}:_gate:degraded"), json!(msg));
                }
                GateFailAction::RetryWithFeedback => {
                    debug!("gate failed — retry requested (no loop context)");
                }
            }
        }

        Ok(())
    }

    /// Execute a bounded loop.
    ///
    /// REQ-10-04 (Pitfall 3): snapshots the most-recent envelope produced by
    /// each iteration into `ReservedChannelState.prev_iteration` BEFORE the
    /// next iteration dispatches. This makes `{this:prev_iteration.*}` resolve
    /// correctly in the loop body: empty on iteration 1, prior-iteration value
    /// on iteration 2+. The inherited `ancestors` are passed unchanged into each
    /// iteration (loop body siblings are not visible to each other per §4.4).
    async fn execute_loop(
        &self,
        inner: &CompOp,
        loop_spec: &LoopSpec,
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        depth: usize,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&WeaveId>,
        ancestors: Vec<AncestorSummary>,
    ) -> AlzinaResult<()> {
        let loop_edge = LoopEdge::new(LoopEdgeConfig::new(
            loop_spec.max_iterations,
            loop_spec.on_exhaust.clone(),
        ));

        let mut iteration = 0usize;

        // REQ-10-04: track the most-recent envelope for prev_iteration snapshot.
        // Starts as None; set to the last envelope after each iteration completes.
        let mut prev_iteration: Option<Arc<Envelope>> = None;

        // RT2-04: Clear loop metadata at entry so nested/re-entered loops
        // don't carry stale state from prior executions.
        state.remove("_meta:iteration");
        state.remove("_meta:loop_exhausted");

        loop {
            let (decision, updates) = loop_edge.evaluate(if iteration == 0 {
                None
            } else {
                Some(iteration)
            });

            // Apply loop edge state updates
            for (k, v) in updates {
                state.insert(k, v);
            }

            match decision {
                LoopDecision::Continue => {
                    debug!(iteration = iteration + 1, "loop iteration");
                    // REQ-10-04: build reserved with the prior iteration's envelope.
                    // On iteration 0 this is None (§4.5: empty on first iteration).
                    let reserved = ReservedChannelState {
                        prev_iteration: prev_iteration.clone(),
                        gate_feedback: None,
                    };
                    let env_len_before = envelopes.len();
                    self.execute_op(
                        inner,
                        state,
                        envelopes,
                        depth + 1,
                        raw_outputs,
                        all_signals,
                        weave_id,
                        ancestors.clone(),
                        reserved,
                    )
                    .await?;
                    iteration += 1;
                    // Snapshot the most-recent envelope produced by this iteration
                    // (the new entry appended since we started this iteration).
                    // §4.5 invariant: only the most-recent iteration is retained.
                    if envelopes.len() > env_len_before {
                        prev_iteration = envelopes.values().last().cloned().map(Arc::new);
                    }
                }
                LoopDecision::Exhausted => {
                    info!(
                        iterations = iteration,
                        action = ?loop_spec.on_exhaust,
                        "loop exhausted"
                    );
                    match &loop_spec.on_exhaust {
                        ExhaustAction::Fail => {
                            return Err(AlzinaError::Orchestration(format!(
                                "loop exhausted after {iteration} iterations"
                            )));
                        }
                        ExhaustAction::Escalate => {
                            return Err(AlzinaError::Orchestration(
                                "loop exhausted: escalated".to_string(),
                            ));
                        }
                        ExhaustAction::AcceptLast => {
                            debug!("loop exhausted — accepting last result");
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    /// Execute a conditional loop (↻? — gate-driven retry).
    ///
    /// REQ-10-04: threads `prev_iteration` (like execute_loop) AND builds a
    /// `GateFeedback` from the prior iteration's gate failure verdict. This
    /// makes `{this:gate.feedback}` resolve to the actual failure message on
    /// retry iterations so the body spawn can incorporate the feedback.
    async fn execute_conditional_loop(
        &self,
        inner: &CompOp,
        cl_spec: &ConditionalLoopSpec,
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        depth: usize,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&WeaveId>,
        ancestors: Vec<AncestorSummary>,
    ) -> AlzinaResult<()> {
        let loop_edge = LoopEdge::new(LoopEdgeConfig::new(
            cl_spec.max_iterations,
            cl_spec.on_exhaust.clone(),
        ));

        let mut iteration = 0usize;

        // REQ-10-04: track prior-iteration envelope for prev_iteration snapshot.
        let mut prev_iteration: Option<Arc<Envelope>> = None;
        // REQ-10-04: track gate feedback from the prior failed iteration.
        let mut last_gate_feedback: Option<GateFeedback> = None;

        // RT2-04: Clear loop metadata at entry so nested/re-entered loops
        // don't carry stale state from prior executions.
        state.remove("_meta:iteration");
        state.remove("_meta:loop_exhausted");

        loop {
            let (decision, updates) = loop_edge.evaluate(if iteration == 0 {
                None
            } else {
                Some(iteration)
            });

            for (k, v) in updates {
                state.insert(k, v);
            }

            match decision {
                LoopDecision::Continue => {
                    debug!(iteration = iteration + 1, "conditional loop iteration");

                    // REQ-10-04: build reserved state carrying both prev_iteration
                    // and the gate feedback from the prior failed iteration.
                    // On iteration 0 both are None (§4.5).
                    let reserved = ReservedChannelState {
                        prev_iteration: prev_iteration.clone(),
                        gate_feedback: last_gate_feedback.clone(),
                    };

                    let env_len_before = envelopes.len();
                    // Execute inner
                    self.execute_op(
                        inner,
                        state,
                        envelopes,
                        depth + 1,
                        raw_outputs,
                        all_signals,
                        weave_id,
                        ancestors.clone(),
                        reserved,
                    )
                    .await?;
                    iteration += 1;

                    // REQ-10-04: snapshot most-recent envelope for next iteration.
                    if envelopes.len() > env_len_before {
                        prev_iteration = envelopes.values().last().cloned().map(Arc::new);
                    }

                    // Evaluate gate on most recent envelope
                    let last_envelope = envelopes.values().last().cloned().ok_or_else(|| {
                        AlzinaError::Orchestration(
                            "conditional loop: no envelope for gate evaluation".to_string(),
                        )
                    })?;

                    let verdict = self
                        .quality_gate
                        .evaluate(&last_envelope, &cl_spec.gate)
                        .await?;

                    match verdict {
                        GateVerdict::Pass => {
                            debug!(iteration, "conditional loop: gate passed");
                            state.insert("_gate:route".to_string(), json!("pass"));
                            return Ok(());
                        }
                        GateVerdict::Fail {
                            issues,
                            recommendation: _,
                        } => {
                            debug!(
                                iteration,
                                issues = issues.len(),
                                "conditional loop: gate failed, retrying"
                            );
                            state.insert("_gate:route".to_string(), json!("fail"));

                            // Inject feedback into state for next iteration (existing path)
                            let feedback_msgs: Vec<String> =
                                issues.iter().map(|i| i.message.clone()).collect();
                            state.insert("_gate:feedback".to_string(), json!(feedback_msgs));

                            // REQ-10-04: build GateFeedback for reserved channel
                            // so {this:gate.feedback} resolves on the next iteration.
                            last_gate_feedback = Some(GateFeedback {
                                signal: last_envelope.signal.clone(),
                                tensions: last_envelope.tensions.clone(),
                                reason: issues
                                    .iter()
                                    .map(|i| i.message.as_str())
                                    .collect::<Vec<_>>()
                                    .join("; "),
                                next: last_envelope.next.clone(),
                            });
                        }
                        GateVerdict::Deferred { reason } => {
                            warn!(reason = %reason, "conditional loop: gate deferred");
                            state.insert("_gate:route".to_string(), json!("fail"));
                            // No structured feedback for Deferred; clear any prior feedback.
                            last_gate_feedback = None;
                        }
                    }
                }
                LoopDecision::Exhausted => {
                    info!(
                        iterations = iteration,
                        action = ?cl_spec.on_exhaust,
                        "conditional loop exhausted"
                    );
                    match &cl_spec.on_exhaust {
                        ExhaustAction::Fail => {
                            return Err(AlzinaError::Orchestration(format!(
                                "conditional loop exhausted after {iteration} iterations"
                            )));
                        }
                        ExhaustAction::Escalate => {
                            return Err(AlzinaError::Orchestration(
                                "conditional loop exhausted: escalated".to_string(),
                            ));
                        }
                        ExhaustAction::AcceptLast => {
                            debug!("conditional loop exhausted — accepting last result");
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    /// Execute conditional routing: evaluate predicates, run first match.
    async fn execute_conditional(
        &self,
        branches: &[(Predicate, CompOp)],
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        depth: usize,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&WeaveId>,
        ancestors: Vec<AncestorSummary>,
        reserved: ReservedChannelState,
    ) -> AlzinaResult<()> {
        for (predicate, op) in branches {
            if evaluate_predicate(predicate, state) {
                debug!(predicate = ?predicate, "conditional: matched");
                return self
                    .execute_op(
                        op,
                        state,
                        envelopes,
                        depth + 1,
                        raw_outputs,
                        all_signals,
                        weave_id,
                        ancestors,
                        reserved,
                    )
                    .await;
            }
        }

        // M-03: No predicate matched and no Default branch — this is a
        // routing error, not a silent pass-through.
        Err(AlzinaError::Orchestration(
            "conditional: no predicate matched and no Default branch".to_string(),
        ))
    }

    /// Execute fan-out: same agent, N prompts concurrently.
    ///
    /// M-04: Prompt count validated ≤ MAX_FANOUT_PROMPTS. Concurrency bounded
    /// by semaphore (FanOutNode handles its own spawning, so we acquire one
    /// permit for the whole operation).
    ///
    /// `ancestors` and `reserved` are accepted for API consistency but FanOutNode
    /// handles rendering internally. The reserved state is not threaded through
    /// fanout legs in v1 (deferred to Plan 11 when FanOut is used inside Loop).
    #[allow(unused_variables)]
    async fn execute_fanout(
        &self,
        spec: &SpawnSpec,
        prompts: &[String],
        state: &mut State,
        envelopes: &mut IndexMap<String, Envelope>,
        depth: usize,
        raw_outputs: &mut HashMap<String, String>,
        all_signals: &mut Vec<Signal>,
        weave_id: Option<&WeaveId>,
        ancestors: Vec<AncestorSummary>,
        reserved: ReservedChannelState,
    ) -> AlzinaResult<()> {
        // M-04: Validate prompt count
        if prompts.len() > MAX_FANOUT_PROMPTS {
            return Err(AlzinaError::Orchestration(format!(
                "fan-out: prompt count ({}) exceeds limit ({MAX_FANOUT_PROMPTS})",
                prompts.len(),
            )));
        }

        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| AlzinaError::Orchestration("semaphore closed".to_string()))?;

        // E2 / D11-05: prefer the parser-stamped fanout id so per-prompt
        // channel keys (`{name}:{i}:*`) align with the daemon's per-prompt
        // LeafIdent format (`{fanout_id}_p{i}`). The per-prompt leaves are
        // registered in the daemon's static_leaf_map under `{name}_p{i}`,
        // which is exactly what `on_leaf_dispatch` looks up below.
        let name = spec
            .node_id
            .clone()
            .unwrap_or_else(|| self.next_name("fanout"));

        debug!(name = %name, count = prompts.len(), "executing fan-out");

        // All fan-out legs share the same weave_id (spec-level override wins
        // over dispatch-level, matching the Spawn case).
        let effective_weave: Option<WeaveId> = spec.weave_id.clone().or_else(|| weave_id.cloned());

        // Plan 11-01.1 gap-closure: spawn each prompt as its own leaf,
        // mirroring `execute_spawn`'s hook treatment. Each per-prompt leaf
        // pre-allocates a SessionId via `on_leaf_dispatch` (whose key is
        // `{name}_p{i}` — the daemon registers them under this node_id), and
        // fires `on_leaf_completed` / `on_leaf_failed` after the spawn
        // resolves so the daemon's `register_dispatch` watcher decrements
        // `in_flight` for every per-prompt leaf.
        //
        // Concurrency mirrors the previous `FanOutNode` behaviour: all
        // prompts run concurrently in a `JoinSet`. Per-prompt state is
        // collected then merged back into the parent `state` /
        // `envelopes` / `raw_outputs` maps on the parent task.

        // Per-prompt spawn future returns (idx, hook_sid, agent_id, node_id, result).
        // We pre-allocate hook_sid + create_root on the parent task (sequential)
        // because `runner.sessions()` is shared mutable state and the trait
        // hook itself is non-Send across `&mut state` boundaries we don't need.
        // The actual `spawn_node.execute` runs concurrently in the JoinSet.
        let mut prompt_specs: Vec<(usize, String, SessionId, CompNode)> =
            Vec::with_capacity(prompts.len());
        for (i, prompt) in prompts.iter().enumerate() {
            let leaf_name = format!("{name}_p{i}");
            let comp_node = CompNode {
                agent_id: spec.agent_id.clone(),
                task: prompt.clone(),
                model_override: spec.model_override.clone(),
                timeout: spec.timeout,
                weave_id: effective_weave.clone(),
                // Phase 1B substrate cascade: all per-prompt fan-out legs
                // inherit the parent spec's dispatch_id.
                dispatch_id: spec.dispatch_id.clone(),
                sampling: None,
            };

            // Pre-allocate hook_sid for this per-prompt leaf so the daemon's
            // `DaemonLeafHook::on_leaf_completed` publishes against the SAME
            // SessionId that `register_dispatch` is watching (matched by the
            // daemon's `static_leaf_map[leaf_name]` lookup).
            let (hook_sid, wrapped_task) = if self.compose_id.is_some() {
                let cid = self.compose_id.as_deref().unwrap();
                let (sid, task) = self
                    .leaf_hook
                    .on_leaf_dispatch(cid, &leaf_name, spec.agent_id.as_str(), &prompt, None)
                    .await?;
                // Pre-register the hook-allocated session in SessionHierarchy
                // so `spawn_with_id` can use caller_registered_session=true.
                self.runner
                    .sessions()
                    .create_root(&sid, &spec.agent_id, weave_id)
                    .await
                    .map_err(|e| {
                        AlzinaError::Orchestration(format!(
                            "fanout leaf hook: failed to register session for {leaf_name}: {e}"
                        ))
                    })?;
                (sid, task)
            } else {
                (SessionId::new(), prompt.clone())
            };

            // Apply the wrapped task (which includes artifact-dir
            // instructions when dispatched via the daemon hook).
            let comp_node = CompNode {
                task: wrapped_task,
                ..comp_node
            };

            prompt_specs.push((i, leaf_name, hook_sid, comp_node));
        }

        let mut join_set = tokio::task::JoinSet::new();
        for (idx, leaf_name, hook_sid, comp_node) in prompt_specs.into_iter() {
            let runner = self.runner.clone();
            let effective_weave_b = effective_weave.clone();
            let compose_id = self.compose_id.clone();
            let leaf_hook = Arc::clone(&self.leaf_hook);
            let agent_id_str = comp_node.agent_id.to_string();
            join_set.spawn(async move {
                let mut spawn_node = SpawnNode::new(&leaf_name, comp_node, runner);
                if let Some(ref wid) = effective_weave_b {
                    spawn_node = spawn_node.with_weave_id(wid.clone());
                }
                if compose_id.is_some() {
                    spawn_node = spawn_node.with_session_id_override(hook_sid.clone());
                }
                let ctx = NodeContext::new(State::new(), ExecutionConfig::new("compiler"), 0);
                let result = spawn_node.execute(&ctx).await;

                // E1 / D11-04: terminal callback per per-prompt leaf so the
                // daemon's `register_dispatch` watcher fires for every leaf.
                if let Some(ref cid) = compose_id {
                    match &result {
                        Ok(output) => {
                            let env_key = format!("{leaf_name}:envelope");
                            let env = output
                                .updates
                                .get(&env_key)
                                .and_then(|v| serde_json::from_value::<Envelope>(v.clone()).ok())
                                .unwrap_or_else(|| Envelope {
                                    status: alzina_core::EnvelopeStatus::Complete,
                                    artifacts: Vec::new(),
                                    signal: None,
                                    tensions: None,
                                    emergent: None,
                                    next: None,
                                    context_update: None,
                                });
                            leaf_hook
                                .on_leaf_completed(cid, &leaf_name, &agent_id_str, &hook_sid, &env)
                                .await;
                        }
                        Err(e) => {
                            leaf_hook
                                .on_leaf_failed(
                                    cid,
                                    &leaf_name,
                                    &agent_id_str,
                                    &hook_sid,
                                    &e.to_string(),
                                )
                                .await;
                        }
                    }
                }

                (idx, leaf_name, result)
            });
        }

        // Collect per-prompt results. Each prompt's envelope/raw/status keys
        // are rewritten from the leaf-name-based form (`{leaf_name}:envelope`)
        // produced by SpawnNode into the FanOut-canonical form
        // (`{name}:{i}:envelope`) so downstream channel substitutions
        // (`{fanout:0:signal}`, etc.) keep working.
        let mut errors: Vec<(usize, String)> = Vec::new();
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok((idx, leaf_name, Ok(output))) => {
                    let entry_name = format!("{name}:{idx}");
                    // Rewrite SpawnNode's `{leaf_name}:*` keys → `{name}:{idx}:*`.
                    for (k, v) in output.updates {
                        if let Some(suffix) = k.strip_prefix(&format!("{leaf_name}:")) {
                            state.insert(format!("{name}:{idx}:{suffix}"), v);
                        } else {
                            state.insert(k, v);
                        }
                    }

                    let env_key = format!("{name}:{idx}:envelope");
                    if let Some(env_val) = state.get(&env_key)
                        && let Ok(env) = serde_json::from_value::<Envelope>(env_val.clone())
                    {
                        envelopes.insert(entry_name.clone(), env);
                    }
                    let raw_key = format!("{name}:{idx}:raw");
                    if let Some(raw_val) = state.get(&raw_key)
                        && let Some(raw_str) = raw_val.as_str()
                    {
                        raw_outputs.insert(entry_name.clone(), raw_str.to_owned());
                    }
                    let sig_key = format!("{name}:{idx}:signals");
                    if let Some(sig_val) = state.get(&sig_key)
                        && let Ok(sigs) = serde_json::from_value::<Vec<Signal>>(sig_val.clone())
                    {
                        all_signals.extend(sigs);
                    }
                }
                Ok((idx, _leaf_name, Err(e))) => {
                    let msg = format!("fanout prompt {idx} failed: {e}");
                    warn!(name = %name, idx, error = %e, "fan-out prompt failed");
                    state.insert(format!("{name}:{idx}:error"), serde_json::json!(msg));
                    errors.push((idx, msg));
                }
                Err(join_err) => {
                    warn!(name = %name, error = %join_err, "fan-out task join error");
                    errors.push((usize::MAX, format!("join error: {join_err}")));
                }
            }
        }

        state.insert(format!("{name}:count"), serde_json::json!(prompts.len()));

        if !errors.is_empty() {
            state.insert(
                format!("{name}:errors"),
                serde_json::json!(
                    errors
                        .iter()
                        .map(|(i, e)| format!("[{i}] {e}"))
                        .collect::<Vec<_>>()
                ),
            );
        }

        Ok(())
    }
}

// ── Compiler Entry Point ────────────────────────────────────────────────────

/// Compile a composition algebra AST into an executable graph.
///
/// # Arguments
///
/// * `op` — The recursive composition AST to compile.
/// * `runner` — The agent runner for dispatching spawns.
/// * `quality_gate` — Gate evaluator for quality checks.
///
/// # Returns
///
/// A `CompiledGraph` ready for execution via `execute()`.
pub fn compile(
    op: CompOp,
    runner: Arc<AlzinaRunner>,
    quality_gate: Arc<dyn QualityGate>,
) -> AlzinaResult<CompiledGraph> {
    compile_with_config(op, runner, quality_gate, CompilerConfig::default())
}

/// Compile with explicit configuration.
pub fn compile_with_config(
    op: CompOp,
    runner: Arc<AlzinaRunner>,
    quality_gate: Arc<dyn QualityGate>,
    config: CompilerConfig,
) -> AlzinaResult<CompiledGraph> {
    // Validate the AST before compiling
    validate_op(&op)?;

    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

    Ok(CompiledGraph {
        op,
        runner,
        quality_gate,
        counter: std::sync::atomic::AtomicUsize::new(0),
        config,
        semaphore,
        compose_id: None, // Set via execute_with_compose_id when composition mode is needed
        leaf_hook: noop_hook(), // Plan 10-05: no-op default; daemon injects DaemonLeafHook
    })
}

/// Compile with a custom leaf hook (Plan 10-05 daemon seam).
///
/// Like `compile`, but installs a custom hook that fires per leaf before
/// the runner dispatches the spawn. The daemon uses this to inject
/// `DaemonLeafHook`, which calls `register_dispatch` for each composition
/// leaf so Phase 6/7's DispatchRegistry counter + watcher fires correctly.
pub fn compile_with_hook(
    op: CompOp,
    runner: Arc<AlzinaRunner>,
    quality_gate: Arc<dyn QualityGate>,
    hook: Arc<dyn CompositionLeafHook>,
) -> AlzinaResult<CompiledGraph> {
    let config = CompilerConfig::default();
    validate_op(&op)?;
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
    Ok(CompiledGraph {
        op,
        runner,
        quality_gate,
        counter: std::sync::atomic::AtomicUsize::new(0),
        config,
        semaphore,
        compose_id: None,
        leaf_hook: hook,
    })
}

// ── Validation ──────────────────────────────────────────────────────────────

/// Validate a CompOp AST for structural correctness.
pub fn validate_op(op: &CompOp) -> AlzinaResult<()> {
    match op {
        CompOp::Spawn(spec) => {
            if spec.task_template.is_empty() {
                return Err(AlzinaError::Orchestration(
                    "spawn: task_template cannot be empty".to_string(),
                ));
            }
            Ok(())
        }
        CompOp::Sequential(ops) => {
            if ops.is_empty() {
                return Err(AlzinaError::Orchestration(
                    "sequential: must have at least one operation".to_string(),
                ));
            }
            for inner in ops {
                validate_op(inner)?;
            }
            Ok(())
        }
        CompOp::Parallel(ops) => {
            if ops.is_empty() {
                return Err(AlzinaError::Orchestration(
                    "parallel: must have at least one operation".to_string(),
                ));
            }
            for inner in ops {
                validate_op(inner)?;
            }
            Ok(())
        }
        CompOp::Synthesise(inner, _) => validate_op(inner),
        CompOp::Gate(inner, _) => validate_op(inner),
        CompOp::Loop(inner, spec) => {
            if spec.max_iterations == 0 {
                return Err(AlzinaError::Orchestration(
                    "loop: max_iterations must be > 0".to_string(),
                ));
            }
            // RT3-10: Reject unbounded loops — 100 is a sane ceiling for
            // composition-level iteration.
            if spec.max_iterations > 100 {
                return Err(AlzinaError::Orchestration(format!(
                    "loop: max_iterations {} exceeds limit of 100",
                    spec.max_iterations
                )));
            }
            validate_op(inner)
        }
        CompOp::ConditionalLoop(inner, spec) => {
            if spec.max_iterations == 0 {
                return Err(AlzinaError::Orchestration(
                    "conditional loop: max_iterations must be > 0".to_string(),
                ));
            }
            // RT3-10: Same ceiling as Loop — reject > 100.
            if spec.max_iterations > 100 {
                return Err(AlzinaError::Orchestration(format!(
                    "conditional loop: max_iterations {} exceeds limit of 100",
                    spec.max_iterations
                )));
            }
            validate_op(inner)
        }
        CompOp::Conditional(branches) => {
            if branches.is_empty() {
                return Err(AlzinaError::Orchestration(
                    "conditional: must have at least one branch".to_string(),
                ));
            }
            for (_, inner) in branches {
                validate_op(inner)?;
            }
            Ok(())
        }
        CompOp::FanOut(spec, prompts) => {
            if prompts.is_empty() {
                return Err(AlzinaError::Orchestration(
                    "fan-out: must have at least one prompt".to_string(),
                ));
            }
            // Plan 11-01.1: FanOut's per-prompt tasks live in `prompts`, not
            // `spec.task_template` (parser leaves that empty by design — see
            // `parse_fanout`). The previous `task_template.is_empty()` check
            // unconditionally rejected every parsed FanOut composition.
            let _ = spec;
            Ok(())
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a SpawnSpec to the runner's CompNode.
/// NOTE: weave_id is NOT copied here — the execute_spawn path derives the
/// effective weave_id from (spec.weave_id OR the dispatch-level weave_id) and
/// sets it directly on the CompNode. Leaving it None here keeps this helper
/// usable for pre-weave code paths (FanOut, SynthesisNode) that manage weave_id
/// separately via the node builder API.
fn spec_to_comp_node(spec: &SpawnSpec) -> CompNode {
    CompNode {
        agent_id: spec.agent_id.clone(),
        task: spec.task_template.clone(),
        model_override: spec.model_override.clone(),
        timeout: spec.timeout,
        weave_id: None, // set by execute_spawn using effective weave_id
        // Phase 1B substrate cascade: forward the chat-tool dispatch_id
        // off the spec. `None` for paths that never set it.
        dispatch_id: spec.dispatch_id.clone(),
        sampling: None,
    }
}

/// Extract the optional session-id override from a `SpawnSpec` for the
/// dispatch path (P5-DEBUG-DISPATCH Fix A). The compiler keeps the
/// override outside `CompNode` so that nested composition nodes
/// (Sequential / Parallel / FanOut) don't accidentally inherit a single
/// pre-allocated root id across many sibling spawns.
fn spec_session_id_override(spec: &SpawnSpec) -> Option<alzina_core::identity::SessionId> {
    spec.session_id_override.clone()
}

/// Evaluate a routing predicate against current state.
fn evaluate_predicate(predicate: &Predicate, state: &State) -> bool {
    match predicate {
        Predicate::StateEquals { channel, value } => state.get(channel) == Some(value),
        Predicate::StateExists { channel } => state.contains_key(channel),
        Predicate::Default => true,
    }
}

/// Convenience: convert a flat `alzina_core::CompOp` into the recursive form.
impl From<alzina_core::CompOp> for CompOp {
    fn from(core_op: alzina_core::CompOp) -> Self {
        match core_op {
            alzina_core::CompOp::Seq(nodes) => CompOp::Sequential(
                nodes
                    .into_iter()
                    .map(|n| CompOp::Spawn(core_node_to_spec(n)))
                    .collect(),
            ),
            alzina_core::CompOp::Par(nodes) => CompOp::Parallel(
                nodes
                    .into_iter()
                    .map(|n| CompOp::Spawn(core_node_to_spec(n)))
                    .collect(),
            ),
            alzina_core::CompOp::Synthesis {
                branches,
                synthesiser,
            } => {
                let par = CompOp::Parallel(
                    branches
                        .into_iter()
                        .map(|n| CompOp::Spawn(core_node_to_spec(n)))
                        .collect(),
                );
                CompOp::Synthesise(
                    Box::new(par),
                    SynthesisSpec {
                        synthesiser: core_node_to_spec(synthesiser),
                        task: None,
                    },
                )
            }
            alzina_core::CompOp::Gate {
                node,
                criteria,
                on_fail,
            } => CompOp::Gate(
                Box::new(CompOp::Spawn(core_node_to_spec(*node))),
                GateSpec { criteria, on_fail },
            ),
            alzina_core::CompOp::ConditionalLoop {
                node,
                gate,
                max_iterations,
                on_exhaust,
            } => CompOp::ConditionalLoop(
                Box::new(CompOp::Spawn(core_node_to_spec(*node))),
                ConditionalLoopSpec {
                    gate,
                    max_iterations,
                    on_exhaust,
                },
            ),
            alzina_core::CompOp::FanOut {
                agent_id,
                prompts,
                combine: _,
            } => CompOp::FanOut(
                SpawnSpec {
                    agent_id,
                    task_template: "fan-out".to_string(),
                    model_override: None,
                    timeout: None,
                    session_id_override: None,
                    weave_id: None,
                    node_id: None,
                    dispatch_id: None,
                },
                prompts,
            ),
        }
    }
}

/// Convert a core CompNode to a SpawnSpec.
fn core_node_to_spec(node: alzina_core::CompNode) -> SpawnSpec {
    SpawnSpec {
        agent_id: node.agent_id,
        task_template: node.task,
        model_override: node.model_override,
        timeout: None,
        session_id_override: None,
        weave_id: None,
        node_id: None,
        // Core-side CompNode does not carry dispatch_id (wire-format type
        // separate from the orchestration-side carrier); paths that go
        // through this converter are non-chat-tool by construction.
        dispatch_id: None,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::alzina_runner::AgentExecutor;
    use crate::test_helpers::{build_test_runner, well_formed_envelope};
    use alzina_core::composition::{GateCriteria, GateFailAction, GateVerdict};
    use alzina_core::envelope::{EnvelopeStatus, IssueSeverity, QualityIssue};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    // ── Mock Executor ───────────────────────────────────────────────────

    struct MockExecutor {
        responses: Mutex<Vec<String>>,
        call_count: AtomicUsize,
    }

    impl MockExecutor {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicUsize::new(0),
            }
        }

        fn single() -> Self {
            Self::new(vec![well_formed_envelope()])
        }

        fn repeating(count: usize) -> Self {
            Self::new(vec![well_formed_envelope(); count])
        }
    }

    #[async_trait::async_trait]
    impl AgentExecutor for MockExecutor {
        async fn execute(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
        ) -> AlzinaResult<String> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let responses = self.responses.lock().await;
            Ok(responses
                .get(idx)
                .cloned()
                .unwrap_or_else(well_formed_envelope))
        }
    }

    // ── Mock Quality Gate ───────────────────────────────────────────────

    struct AlwaysPassGate;

    #[async_trait::async_trait]
    impl QualityGate for AlwaysPassGate {
        async fn evaluate(
            &self,
            _envelope: &Envelope,
            _criteria: &GateCriteria,
        ) -> AlzinaResult<GateVerdict> {
            Ok(GateVerdict::Pass)
        }
    }

    struct AlwaysFailGate;

    #[async_trait::async_trait]
    impl QualityGate for AlwaysFailGate {
        async fn evaluate(
            &self,
            _envelope: &Envelope,
            _criteria: &GateCriteria,
        ) -> AlzinaResult<GateVerdict> {
            Ok(GateVerdict::Fail {
                issues: vec![QualityIssue {
                    severity: IssueSeverity::Error,
                    field: "test".to_string(),
                    message: "always fails".to_string(),
                }],
                recommendation: GateFailAction::RetryWithFeedback,
            })
        }
    }

    /// Gate that fails N times then passes.
    struct FailThenPassGate {
        fail_count: AtomicUsize,
        max_fails: usize,
    }

    impl FailThenPassGate {
        fn new(max_fails: usize) -> Self {
            Self {
                fail_count: AtomicUsize::new(0),
                max_fails,
            }
        }
    }

    #[async_trait::async_trait]
    impl QualityGate for FailThenPassGate {
        async fn evaluate(
            &self,
            _envelope: &Envelope,
            _criteria: &GateCriteria,
        ) -> AlzinaResult<GateVerdict> {
            let n = self.fail_count.fetch_add(1, Ordering::SeqCst);
            if n < self.max_fails {
                Ok(GateVerdict::Fail {
                    issues: vec![QualityIssue {
                        severity: IssueSeverity::Error,
                        field: "test".to_string(),
                        message: format!("fail #{}", n + 1),
                    }],
                    recommendation: GateFailAction::RetryWithFeedback,
                })
            } else {
                Ok(GateVerdict::Pass)
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn test_spawn_spec(agent: &str, task: &str) -> SpawnSpec {
        SpawnSpec {
            agent_id: AgentId::new(agent),
            task_template: task.to_string(),
            model_override: None,
            timeout: None,
            session_id_override: None,
            weave_id: None,
            node_id: None,
            dispatch_id: None,
        }
    }

    // R-WEAVE-SCOPE-001 — SpawnSpec scope derivation
    #[test]
    fn spawn_spec_scope_none_weave_id_maps_to_session_default() {
        let spec = test_spawn_spec("muninn", "task");
        assert!(matches!(spec.scope(), alzina_core::Scope::SessionDefault));
    }

    #[test]
    fn spawn_spec_scope_some_weave_id_maps_to_weave() {
        let mut spec = test_spawn_spec("muninn", "task");
        spec.weave_id = Some(alzina_core::WeaveId::new("W-f6bff644"));
        let s = spec.scope();
        assert!(s.is_weave());
        assert_eq!(s.as_str(), "W-f6bff644");
    }

    fn permissive_criteria() -> GateCriteria {
        GateCriteria {
            envelope_required_fields: vec![],
            status_must_be: None,
            max_tensions: None,
        }
    }

    async fn test_compiled_graph(
        op: CompOp,
        executor: Arc<dyn AgentExecutor>,
        quality_gate: Arc<dyn QualityGate>,
    ) -> AlzinaResult<ExecutionResult> {
        let dir = tempfile::tempdir().unwrap();
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());
        let graph = compile(op, runner, quality_gate)?;
        graph.execute(None).await
    }

    // ── Test: Spawn → single node ───────────────────────────────────────

    #[tokio::test]
    async fn compile_spawn_produces_single_node() {
        let op = CompOp::Spawn(test_spawn_spec("smidr", "analyse workspace"));
        let executor = Arc::new(MockExecutor::single());
        let gate = Arc::new(AlwaysPassGate);

        let result = test_compiled_graph(op, executor, gate).await.unwrap();

        assert_eq!(result.envelopes.len(), 1);
        let env = result.envelopes.values().next().unwrap();
        assert_eq!(env.status, EnvelopeStatus::Complete);
    }

    // ── Test: Sequential([A, B]) → two nodes in order ───────────────────

    #[tokio::test]
    async fn compile_sequential_two_nodes() {
        let op = CompOp::Sequential(vec![
            CompOp::Spawn(test_spawn_spec("urdr", "read context")),
            CompOp::Spawn(test_spawn_spec("skuld", "plan future")),
        ]);
        let executor = Arc::new(MockExecutor::repeating(2));
        let gate = Arc::new(AlwaysPassGate);

        let result = test_compiled_graph(op, executor.clone(), gate)
            .await
            .unwrap();

        assert_eq!(result.envelopes.len(), 2);
        // Both executed
        assert_eq!(executor.call_count.load(Ordering::SeqCst), 2);
    }

    // ── Test: Parallel([A, B]) + Synthesise ─────────────────────────────

    #[tokio::test]
    async fn compile_parallel_with_synthesis() {
        let op = CompOp::Synthesise(
            Box::new(CompOp::Parallel(vec![
                CompOp::Spawn(test_spawn_spec("urdr", "read context")),
                CompOp::Spawn(test_spawn_spec("skuld", "plan future")),
            ])),
            SynthesisSpec {
                synthesiser: test_spawn_spec("vefr", "synthesise results"),
                task: None,
            },
        );
        // 2 parallel branches + 1 synthesiser = 3 total
        let executor = Arc::new(MockExecutor::repeating(3));
        let gate = Arc::new(AlwaysPassGate);

        let result = test_compiled_graph(op, executor.clone(), gate)
            .await
            .unwrap();

        // 2 parallel + 1 synthesis
        assert!(result.envelopes.len() >= 3);
        assert_eq!(executor.call_count.load(Ordering::SeqCst), 3);
    }

    /// Executor that returns a well-formed envelope for every agent except
    /// `fail_agent`, which gets a STATUS-less response that fails envelope
    /// parsing downstream. Used to test partial-failure semantics in Parallel.
    struct FailForAgentMock {
        fail_agent: String,
        call_count: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl AgentExecutor for FailForAgentMock {
        async fn execute(
            &self,
            agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
        ) -> AlzinaResult<String> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            if agent_id.as_str() == self.fail_agent {
                // STATUS-less response — envelope parser will reject with
                // EnvelopeParse, which bubbles up as a branch-level Err in
                // execute_parallel. Mirrors the huginn audit-leaf failure
                // pattern from 2026-05-17 09:16:50 cascade.
                Ok("Just a prose answer with no envelope structure.".to_string())
            } else {
                Ok(well_formed_envelope())
            }
        }
    }

    /// Cascade-survival regression: when one branch in a Parallel errors
    /// (e.g. envelope parse failure on huginn), sibling branches MUST drain
    /// to completion instead of being killed by JoinSet drop. The surviving
    /// branches contribute their envelopes; the failed branch contributes
    /// none. Mirrors the 2026-05-17 09:16:50 production cascade where huginn
    /// (audit) failed and kvasir (redteam) was killed mid-flight.
    #[tokio::test]
    async fn parallel_branch_failure_does_not_cancel_siblings() {
        let op = CompOp::Parallel(vec![
            CompOp::Spawn(test_spawn_spec("huginn", "audit task")),
            CompOp::Spawn(test_spawn_spec("kvasir", "redteam task")),
        ]);
        let executor = Arc::new(FailForAgentMock {
            fail_agent: "huginn".to_string(),
            call_count: AtomicUsize::new(0),
        });
        let gate = Arc::new(AlwaysPassGate);

        let result = test_compiled_graph(op, executor.clone(), gate)
            .await
            .expect(
                "Parallel must return Ok when at least one branch survives — \
                 partial failure is not a composition failure",
            );

        // Both branches executed (the failing one was not aborted by sibling
        // failure; the surviving one was not aborted by JoinSet drop).
        assert_eq!(
            executor.call_count.load(Ordering::SeqCst),
            2,
            "both branches must execute even when one fails"
        );

        // The surviving branch (kvasir) contributed its envelope.
        // The failed branch (huginn) contributed none — downstream consumers
        // (synthesisers, gates) react to the missing envelope per option 3.
        let envelope_keys: Vec<&str> = result.envelopes.keys().map(String::as_str).collect();
        assert!(
            envelope_keys.iter().any(|k| k.contains("kvasir")),
            "surviving kvasir branch must contribute an envelope; got keys: {envelope_keys:?}"
        );
    }

    /// Inverse condition: when ALL branches fail, Parallel returns Err so the
    /// parent op (Sequential / Synthesise / Gate) sees the composition
    /// genuinely failed. Single-variable guard against the partial-success
    /// path masking total failure.
    #[tokio::test]
    async fn parallel_all_branches_fail_returns_err() {
        // Executor fails for every agent — sentinel "*" matches all via the
        // FailForAgentMock contract (no agent_id will equal "*" so the
        // explicit fail-list is the only way; use a per-call always-fail).
        struct AllFailMock {
            call_count: AtomicUsize,
        }
        #[async_trait::async_trait]
        impl AgentExecutor for AllFailMock {
            async fn execute(
                &self,
                _agent_id: &AgentId,
                _instruction: &str,
                _model: &str,
                _task: &str,
            ) -> AlzinaResult<String> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                Ok("no envelope here".to_string())
            }
        }
        let op = CompOp::Parallel(vec![
            CompOp::Spawn(test_spawn_spec("huginn", "t1")),
            CompOp::Spawn(test_spawn_spec("kvasir", "t2")),
        ]);
        let executor = Arc::new(AllFailMock {
            call_count: AtomicUsize::new(0),
        });
        let gate = Arc::new(AlwaysPassGate);

        let err = test_compiled_graph(op, executor.clone(), gate)
            .await
            .expect_err("Parallel with all branches failed must return Err");

        assert_eq!(
            executor.call_count.load(Ordering::SeqCst),
            2,
            "both branches must still execute even though both fail"
        );
        let msg = format!("{err}");
        assert!(
            !msg.is_empty(),
            "the surfaced error should be non-empty (first branch's error)"
        );
    }

    // ── Test: leaf-hook key matches SpawnSpec.node_id (Plan 10-05 followup) ──

    /// Regression test for the dispatch_compose wedge: when a composition has
    /// parallel branches, the per-branch `CompiledGraph.counter` starts at
    /// `idx * 1000`, so `next_name` produces ids like `urdr_1000`, `verdandi_2000`
    /// — diverging from the parser's globally-incrementing `state.next_index`
    /// (`urdr_1`, `verdandi_2`). The `DaemonLeafHook::static_leaf_map` is keyed
    /// on parser ids; without threading the parser id through `SpawnSpec.node_id`,
    /// every lookup misses and the dispatch registry wedges.
    ///
    /// This test injects a recording hook and asserts the keys it observes are
    /// the ids the parser would have emitted (i.e. taken from `SpawnSpec.node_id`),
    /// NOT the runtime-counter names from `next_name`.
    #[tokio::test]
    async fn leaf_hook_called_with_spawnspec_node_id_not_runtime_name() {
        use crate::composition::leaf_hook::CompositionLeafHook;
        use alzina_core::AlzinaResult;
        use alzina_core::identity::SessionId;
        use async_trait::async_trait;
        use std::sync::Mutex;

        struct RecordingHook {
            keys: Mutex<Vec<String>>,
        }

        #[async_trait]
        impl CompositionLeafHook for RecordingHook {
            async fn on_leaf_dispatch(
                &self,
                _compose_id: &str,
                node_id: &str,
                _agent: &str,
                task: &str,
                _parent_session_id: Option<&SessionId>,
            ) -> AlzinaResult<(SessionId, String)> {
                self.keys.lock().unwrap().push(node_id.to_string());
                Ok((SessionId::new(), task.to_string()))
            }

            async fn on_leaf_completed(
                &self,
                _compose_id: &str,
                _node_id: &str,
                _agent: &str,
                _session_id: &SessionId,
                _envelope: &alzina_core::envelope::Envelope,
            ) {
                // Test hook: not asserted in this test.
            }

            async fn on_leaf_failed(
                &self,
                _compose_id: &str,
                _node_id: &str,
                _agent: &str,
                _session_id: &SessionId,
                _error: &str,
            ) {
                // Test hook: not asserted in this test.
            }

            async fn on_composition_terminal(&self, _compose_id: &str) {
                // Test hook: not asserted in this test.
            }
        }

        // Construct SpawnSpecs with parser-style ids. The plan shape
        // `Sequential(Parallel(skuld_0, urdr_1, verdandi_2), kvasir_3)` is the
        // canonical wedge case: pre-fix, the parallel branch counter offsets
        // produce `urdr_1000` / `verdandi_2000` at runtime, masking the hook
        // map populated under `urdr_1` / `verdandi_2`.
        let mut s0 = test_spawn_spec("skuld", "past");
        s0.node_id = Some("skuld_0".into());
        let mut s1 = test_spawn_spec("urdr", "present");
        s1.node_id = Some("urdr_1".into());
        let mut s2 = test_spawn_spec("verdandi", "future");
        s2.node_id = Some("verdandi_2".into());
        let mut s3 = test_spawn_spec("kvasir", "synthesis");
        s3.node_id = Some("kvasir_3".into());

        let op = CompOp::Sequential(vec![
            CompOp::Parallel(vec![
                CompOp::Spawn(s0),
                CompOp::Spawn(s1),
                CompOp::Spawn(s2),
            ]),
            CompOp::Spawn(s3),
        ]);

        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::repeating(4));
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());
        let gate: Arc<dyn QualityGate> = Arc::new(AlwaysPassGate);
        let graph = compile(op, runner, gate).unwrap();
        let hook = Arc::new(RecordingHook {
            keys: Mutex::new(Vec::new()),
        });
        let _result = graph
            .execute_with_hook(None, "compose-test".into(), hook.clone())
            .await
            .unwrap();

        let mut keys = hook.keys.lock().unwrap().clone();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "kvasir_3".to_string(),
                "skuld_0".to_string(),
                "urdr_1".to_string(),
                "verdandi_2".to_string(),
            ],
            "leaf hook must see parser-style node_ids from SpawnSpec.node_id; \
             if this fails, the daemon's static_leaf_map lookup will miss and \
             every composition wedges on stuck in_flight counts"
        );
    }

    // ── E1 / D11-04 callback invocation tests ────────────────────────────

    /// Hook that records all three callback kinds for inspection.
    struct ThreeWayRecordingHook {
        dispatched: std::sync::Mutex<Vec<String>>,
        completed: std::sync::Mutex<Vec<String>>,
        failed: std::sync::Mutex<Vec<(String, String)>>,
    }

    impl ThreeWayRecordingHook {
        fn new() -> Self {
            Self {
                dispatched: std::sync::Mutex::new(Vec::new()),
                completed: std::sync::Mutex::new(Vec::new()),
                failed: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl CompositionLeafHook for ThreeWayRecordingHook {
        async fn on_leaf_dispatch(
            &self,
            _compose_id: &str,
            node_id: &str,
            _agent: &str,
            task: &str,
            _parent_session_id: Option<&SessionId>,
        ) -> AlzinaResult<(SessionId, String)> {
            self.dispatched.lock().unwrap().push(node_id.to_string());
            Ok((SessionId::new(), task.to_string()))
        }

        async fn on_leaf_completed(
            &self,
            _compose_id: &str,
            node_id: &str,
            _agent: &str,
            _session_id: &SessionId,
            _envelope: &Envelope,
        ) {
            self.completed.lock().unwrap().push(node_id.to_string());
        }

        async fn on_leaf_failed(
            &self,
            _compose_id: &str,
            node_id: &str,
            _agent: &str,
            _session_id: &SessionId,
            error: &str,
        ) {
            self.failed
                .lock()
                .unwrap()
                .push((node_id.to_string(), error.to_string()));
        }

        async fn on_composition_terminal(&self, _compose_id: &str) {
            // Test hook: drain not asserted in execute_spawn-level tests.
        }
    }

    /// E1: when execute_spawn succeeds, on_leaf_completed fires exactly once
    /// per leaf with the same node_id used at dispatch time.
    #[tokio::test]
    async fn execute_spawn_invokes_on_leaf_completed_on_ok() {
        let mut s0 = test_spawn_spec("skuld", "past");
        s0.node_id = Some("skuld_0".into());

        let op = CompOp::Spawn(s0);

        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(MockExecutor::repeating(1));
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());
        let gate: Arc<dyn QualityGate> = Arc::new(AlwaysPassGate);
        let graph = compile(op, runner, gate).unwrap();
        let hook = Arc::new(ThreeWayRecordingHook::new());

        let result = graph
            .execute_with_hook(None, "compose-test".into(), hook.clone())
            .await;
        assert!(result.is_ok(), "spawn must succeed for this test");

        let completed = hook.completed.lock().unwrap();
        assert_eq!(
            completed.len(),
            1,
            "exactly one on_leaf_completed invocation expected"
        );
        assert_eq!(completed[0], "skuld_0");
        let failed = hook.failed.lock().unwrap();
        assert!(failed.is_empty(), "no failures expected, got: {failed:?}");
    }

    /// E1: when execute_spawn fails, on_leaf_failed fires exactly once with
    /// the error message. The spawn's Err is still bubbled to the caller.
    #[tokio::test]
    async fn execute_spawn_invokes_on_leaf_failed_on_err() {
        struct FailingExecutor;
        #[async_trait::async_trait]
        impl AgentExecutor for FailingExecutor {
            async fn execute(
                &self,
                _agent_id: &AgentId,
                _instruction: &str,
                _model: &str,
                _task: &str,
            ) -> AlzinaResult<String> {
                Err(AlzinaError::Orchestration("forced failure".into()))
            }
        }

        let mut s0 = test_spawn_spec("skuld", "past");
        s0.node_id = Some("skuld_0".into());

        let op = CompOp::Spawn(s0);

        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(FailingExecutor);
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());
        let gate: Arc<dyn QualityGate> = Arc::new(AlwaysPassGate);
        let graph = compile(op, runner, gate).unwrap();
        let hook = Arc::new(ThreeWayRecordingHook::new());

        let result = graph
            .execute_with_hook(None, "compose-test".into(), hook.clone())
            .await;
        assert!(result.is_err(), "spawn must fail for this test");

        let failed = hook.failed.lock().unwrap();
        assert_eq!(
            failed.len(),
            1,
            "exactly one on_leaf_failed invocation expected"
        );
        assert_eq!(failed[0].0, "skuld_0");
        assert!(!failed[0].1.is_empty(), "error message must be non-empty");
        let completed = hook.completed.lock().unwrap();
        assert!(
            completed.is_empty(),
            "no completions expected on failed spawn"
        );
    }

    // ── Test: Gate(A, spec) → node + gate + conditional ─────────────────

    #[tokio::test]
    async fn compile_gate_passes() {
        let op = CompOp::Gate(
            Box::new(CompOp::Spawn(test_spawn_spec("smidr", "build"))),
            GateSpec {
                criteria: permissive_criteria(),
                on_fail: GateFailAction::Escalate,
            },
        );
        let executor = Arc::new(MockExecutor::single());
        let gate = Arc::new(AlwaysPassGate);

        let result = test_compiled_graph(op, executor, gate).await.unwrap();

        // RT2-07: Gate channels are now namespaced as {gate_name}:_gate:route
        let has_pass = result
            .state
            .iter()
            .any(|(k, v)| k.ends_with(":_gate:route") && v == &json!("pass"));
        assert!(
            has_pass,
            "expected a namespaced gate route=pass in state: {:?}",
            result.state.keys().collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn compile_gate_fails_escalate() {
        let op = CompOp::Gate(
            Box::new(CompOp::Spawn(test_spawn_spec("smidr", "build"))),
            GateSpec {
                criteria: permissive_criteria(),
                on_fail: GateFailAction::Escalate,
            },
        );
        let executor = Arc::new(MockExecutor::single());
        let gate = Arc::new(AlwaysFailGate);

        let result = test_compiled_graph(op, executor, gate).await;

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("escalated"), "error was: {err}");
    }

    // ── Test: ConditionalLoop(A, gate, max=3) → gate-driven retry ───────

    #[tokio::test]
    async fn compile_conditional_loop_passes_on_second() {
        let op = CompOp::ConditionalLoop(
            Box::new(CompOp::Spawn(test_spawn_spec("smidr", "iterative fix"))),
            ConditionalLoopSpec {
                gate: permissive_criteria(),
                max_iterations: 3,
                on_exhaust: ExhaustAction::Fail,
            },
        );
        // Fail once, then pass
        let executor = Arc::new(MockExecutor::repeating(2));
        let gate = Arc::new(FailThenPassGate::new(1));

        let result = test_compiled_graph(op, executor.clone(), gate)
            .await
            .unwrap();

        // Should have executed twice (first fail, second pass)
        assert_eq!(executor.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(result.state.get("_gate:route"), Some(&json!("pass")));
    }

    #[tokio::test]
    async fn compile_conditional_loop_exhausts() {
        let op = CompOp::ConditionalLoop(
            Box::new(CompOp::Spawn(test_spawn_spec("smidr", "never good enough"))),
            ConditionalLoopSpec {
                gate: permissive_criteria(),
                max_iterations: 3,
                on_exhaust: ExhaustAction::Fail,
            },
        );
        let executor = Arc::new(MockExecutor::repeating(3));
        let gate = Arc::new(AlwaysFailGate); // Always fails

        let result = test_compiled_graph(op, executor.clone(), gate).await;

        // Should exhaust after 2 iterations
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("exhausted"), "error was: {err}");
        assert_eq!(executor.call_count.load(Ordering::SeqCst), 2);
    }

    // ── Test: Validation ────────────────────────────────────────────────

    #[test]
    fn validate_empty_sequential_fails() {
        let result = validate_op(&CompOp::Sequential(vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn validate_empty_task_fails() {
        let result = validate_op(&CompOp::Spawn(SpawnSpec {
            agent_id: AgentId::new("test"),
            task_template: "".to_string(),
            model_override: None,
            timeout: None,
            session_id_override: None,
            weave_id: None,
            node_id: None,
            dispatch_id: None,
        }));
        assert!(result.is_err());
    }

    #[test]
    fn validate_zero_max_iterations_fails() {
        let result = validate_op(&CompOp::Loop(
            Box::new(CompOp::Spawn(test_spawn_spec("a", "b"))),
            LoopSpec {
                max_iterations: 0,
                on_exhaust: ExhaustAction::Fail,
            },
        ));
        assert!(result.is_err());
    }

    // ── Test: From<alzina_core::CompOp> conversion ──────────────────────

    #[test]
    fn convert_core_seq_to_recursive() {
        let core_op = alzina_core::CompOp::Seq(vec![
            alzina_core::CompNode {
                agent_id: AgentId::new("urdr"),
                task: "read".to_string(),
                model_override: None,
                tool_overrides: None,
            },
            alzina_core::CompNode {
                agent_id: AgentId::new("skuld"),
                task: "plan".to_string(),
                model_override: None,
                tool_overrides: None,
            },
        ]);

        let recursive: CompOp = core_op.into();
        assert!(matches!(recursive, CompOp::Sequential(ops) if ops.len() == 2));
    }

    // ── Test: Conditional routing ───────────────────────────────────────

    #[tokio::test]
    async fn compile_conditional_routes_correctly() {
        let op = CompOp::Conditional(vec![
            (
                Predicate::StateEquals {
                    channel: "mode".to_string(),
                    value: json!("fast"),
                },
                CompOp::Spawn(test_spawn_spec("fast_agent", "quick run")),
            ),
            (
                Predicate::Default,
                CompOp::Spawn(test_spawn_spec("slow_agent", "thorough run")),
            ),
        ]);
        let executor = Arc::new(MockExecutor::single());
        let gate = Arc::new(AlwaysPassGate);

        let dir = tempfile::tempdir().unwrap();
        let runner = Arc::new(
            build_test_runner(executor.clone(), None, dir.path())
                .await
                .unwrap(),
        );
        let graph = compile(op, runner, gate).unwrap();

        // Execute with no "mode" in state → default branch
        let result = graph.execute(None).await.unwrap();
        assert_eq!(result.envelopes.len(), 1);
        // The slow_agent should have been dispatched (default branch)
        assert_eq!(executor.call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn predicate_evaluation() {
        let mut state = State::new();
        state.insert("mode".to_string(), json!("fast"));

        assert!(evaluate_predicate(
            &Predicate::StateEquals {
                channel: "mode".to_string(),
                value: json!("fast"),
            },
            &state,
        ));

        assert!(!evaluate_predicate(
            &Predicate::StateEquals {
                channel: "mode".to_string(),
                value: json!("slow"),
            },
            &state,
        ));

        assert!(evaluate_predicate(
            &Predicate::StateExists {
                channel: "mode".to_string(),
            },
            &state,
        ));

        assert!(!evaluate_predicate(
            &Predicate::StateExists {
                channel: "missing".to_string(),
            },
            &state,
        ));

        assert!(evaluate_predicate(&Predicate::Default, &state));
    }

    // ── Test: M-03 — Conditional with no Default branch returns error ───

    #[tokio::test]
    async fn compile_conditional_no_default_errors() {
        let op = CompOp::Conditional(vec![
            (
                Predicate::StateEquals {
                    channel: "mode".to_string(),
                    value: json!("fast"),
                },
                CompOp::Spawn(test_spawn_spec("fast_agent", "quick run")),
            ),
            // No Default branch
        ]);
        let executor = Arc::new(MockExecutor::single());
        let gate = Arc::new(AlwaysPassGate);

        let dir = tempfile::tempdir().unwrap();
        let runner = Arc::new(
            build_test_runner(executor.clone(), None, dir.path())
                .await
                .unwrap(),
        );
        let graph = compile(op, runner, gate).unwrap();

        // Execute with no "mode" in state -> no branch matches, no Default
        let result = graph.execute(None).await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("no predicate matched"), "error was: {err}");
    }

    // ── Test: RT3-10 — Loop with max_iterations > 100 rejected ──────────

    #[test]
    fn validate_loop_max_iterations_exceeds_limit() {
        let result = validate_op(&CompOp::Loop(
            Box::new(CompOp::Spawn(test_spawn_spec("a", "b"))),
            LoopSpec {
                max_iterations: 101,
                on_exhaust: ExhaustAction::Fail,
            },
        ));
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("exceeds limit of 100"), "error was: {err}");
    }

    #[test]
    fn validate_conditional_loop_max_iterations_exceeds_limit() {
        let result = validate_op(&CompOp::ConditionalLoop(
            Box::new(CompOp::Spawn(test_spawn_spec("a", "b"))),
            ConditionalLoopSpec {
                gate: permissive_criteria(),
                max_iterations: 200,
                on_exhaust: ExhaustAction::Fail,
            },
        ));
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("exceeds limit of 100"), "error was: {err}");
    }

    #[test]
    fn validate_loop_at_100_passes() {
        let result = validate_op(&CompOp::Loop(
            Box::new(CompOp::Spawn(test_spawn_spec("a", "b"))),
            LoopSpec {
                max_iterations: 100,
                on_exhaust: ExhaustAction::Fail,
            },
        ));
        assert!(result.is_ok());
    }
}

// ── Task 1 tests: SynthesisSpec.task + DEFAULT_SYNTHESIS_PROMPT ─────────────

#[cfg(test)]
mod synthesis_task_tests {
    use super::*;
    use crate::composition::parser::parse_compose;

    #[test]
    fn synthesis_spec_with_none_task_compiles() {
        let spec = SynthesisSpec {
            synthesiser: SpawnSpec {
                agent_id: AgentId::new("sjofn"),
                task_template: String::new(),
                model_override: None,
                timeout: None,
                session_id_override: None,
                weave_id: None,
                node_id: None,
                dispatch_id: None,
            },
            task: None,
        };
        assert!(spec.task.is_none());
    }

    #[test]
    fn synthesis_spec_serialises_without_task_field_when_none() {
        let spec = SynthesisSpec {
            synthesiser: SpawnSpec {
                agent_id: AgentId::new("sjofn"),
                task_template: String::new(),
                model_override: None,
                timeout: None,
                session_id_override: None,
                weave_id: None,
                node_id: None,
                dispatch_id: None,
            },
            task: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(
            !json.contains("\"task\""),
            "task=None must be skipped: {json}"
        );
    }

    #[test]
    fn synthesise_task_attribute_parses_into_some() {
        let xml = r#"<Compose><Synthesise synthesiser="sjofn" task="Custom prompt"><Spawn agent="a" task="t"/></Synthesise></Compose>"#;
        let parsed = parse_compose(xml).unwrap();
        if let CompOp::Synthesise(_, spec) = parsed.op {
            assert_eq!(spec.task.as_deref(), Some("Custom prompt"));
        } else {
            panic!("expected Synthesise");
        }
    }

    #[test]
    fn synthesise_without_task_attribute_parses_into_none() {
        let xml = r#"<Compose><Synthesise synthesiser="sjofn"><Spawn agent="a" task="t"/></Synthesise></Compose>"#;
        let parsed = parse_compose(xml).unwrap();
        if let CompOp::Synthesise(_, spec) = parsed.op {
            assert!(spec.task.is_none());
        } else {
            panic!("expected Synthesise");
        }
    }

    #[test]
    fn default_synthesis_prompt_matches_doc_section_3_4() {
        assert_eq!(
            DEFAULT_SYNTHESIS_PROMPT,
            "Synthesise the upstream branches without averaging. Name tensions explicitly."
        );
    }
}

// ── Task 2 tests: execute_with_compose_id + CompositionContext per arm ────────

#[cfg(test)]
mod composition_context_tests {
    use super::*;
    use crate::runner::alzina_runner::AgentExecutor;
    use crate::test_helpers::{build_test_runner, well_formed_envelope};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    struct MockExecutor {
        responses: Mutex<Vec<String>>,
        call_count: AtomicUsize,
        /// Records the task string for each call (in order).
        tasks: Mutex<Vec<String>>,
    }

    impl MockExecutor {
        fn new(count: usize) -> Self {
            Self {
                responses: Mutex::new(vec![well_formed_envelope(); count]),
                call_count: AtomicUsize::new(0),
                tasks: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl AgentExecutor for MockExecutor {
        async fn execute(
            &self,
            _agent_id: &AgentId,
            _instruction: &str,
            _model: &str,
            task: &str,
        ) -> AlzinaResult<String> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            self.tasks.lock().await.push(task.to_string());
            let responses = self.responses.lock().await;
            Ok(responses
                .get(idx)
                .cloned()
                .unwrap_or_else(well_formed_envelope))
        }
    }

    fn make_spawn(agent: &str, task: &str) -> SpawnSpec {
        SpawnSpec {
            agent_id: AgentId::new(agent),
            task_template: task.to_string(),
            model_override: None,
            timeout: None,
            session_id_override: None,
            weave_id: None,
            node_id: None,
            dispatch_id: None,
        }
    }

    /// Build a runner and keep the temp dir alive for the duration of the test.
    /// Returns `(runner, _dir)` — caller must bind both to avoid the dir being
    /// dropped before the governance hooks run (which read agent identity configs).
    async fn make_runner_and_dir(
        executor: Arc<dyn AgentExecutor>,
    ) -> (Arc<AlzinaRunner>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());
        (runner, dir)
    }

    struct MockGate;

    #[async_trait::async_trait]
    impl QualityGate for MockGate {
        async fn evaluate(
            &self,
            _envelope: &Envelope,
            _criteria: &GateCriteria,
        ) -> AlzinaResult<GateVerdict> {
            Ok(GateVerdict::Pass)
        }
    }

    fn gate() -> Arc<dyn QualityGate> {
        Arc::new(MockGate)
    }

    /// Ad-hoc path: compose_id=None → no preamble, task unchanged.
    #[tokio::test]
    async fn ad_hoc_spawn_path_produces_no_preamble() {
        let executor = Arc::new(MockExecutor::new(1));
        let (runner, _dir) = make_runner_and_dir(executor.clone()).await;
        let op = CompOp::Spawn(make_spawn("smidr", "do the work"));
        let graph = compile(op, runner, gate()).unwrap();
        graph.execute(None).await.unwrap();

        let tasks = executor.tasks.lock().await;
        // No composition context → task is the raw template
        assert_eq!(
            tasks[0], "do the work",
            "ad-hoc task must be byte-identical"
        );
        assert!(
            !tasks[0].contains("## Upstream context"),
            "no preamble on ad-hoc"
        );
    }

    /// execute_with_compose_id exists and returns Ok.
    #[tokio::test]
    async fn execute_with_compose_id_exists_and_runs() {
        let executor = Arc::new(MockExecutor::new(1));
        let (runner, _dir) = make_runner_and_dir(executor.clone()).await;
        let op = CompOp::Spawn(make_spawn("smidr", "do the work"));
        let graph = compile(op, runner, gate()).unwrap();
        // The method must exist; it must succeed.
        graph
            .execute_with_compose_id(None, "compose-abc-123".to_string())
            .await
            .unwrap();
    }

    /// Sequential: child 2 rendered task includes "## Upstream context" preamble.
    #[tokio::test]
    async fn sequential_child_sees_prior_sibling_as_ancestor_in_preamble() {
        let executor = Arc::new(MockExecutor::new(2));
        let (runner, _dir) = make_runner_and_dir(executor.clone()).await;
        let op = CompOp::Sequential(vec![
            CompOp::Spawn(make_spawn("smidr", "first task")),
            CompOp::Spawn(make_spawn("skuld", "second task")),
        ]);
        let graph = compile(op, runner, gate()).unwrap();
        graph
            .execute_with_compose_id(None, "seq-compose-test".to_string())
            .await
            .unwrap();

        let tasks = executor.tasks.lock().await;
        // First spawn has no ancestors → no preamble.
        assert!(
            !tasks[0].contains("## Upstream context"),
            "first spawn has no preamble"
        );
        // Second spawn sees first as ancestor → preamble injected.
        assert!(
            tasks[1].contains("## Upstream context"),
            "second spawn must have upstream context preamble; got: {}",
            tasks[1]
        );
    }

    /// Parallel: neither sibling sees the other in their preamble.
    #[tokio::test]
    async fn parallel_siblings_do_not_see_each_other_in_preamble() {
        let executor = Arc::new(MockExecutor::new(2));
        let (runner, _dir) = make_runner_and_dir(executor.clone()).await;
        let op = CompOp::Parallel(vec![
            CompOp::Spawn(make_spawn("smidr", "branch a")),
            CompOp::Spawn(make_spawn("skuld", "branch b")),
        ]);
        let graph = compile(op, runner, gate()).unwrap();
        graph
            .execute_with_compose_id(None, "par-compose-test".to_string())
            .await
            .unwrap();

        let tasks = executor.tasks.lock().await;
        // Parallel spawns have no ancestors by §4.4 — no preamble.
        for (i, t) in tasks.iter().enumerate() {
            assert!(
                !t.contains("## Upstream context"),
                "parallel branch {i} must not have upstream preamble; got: {t}"
            );
        }
    }
}

// ── D15-05: ScopedEnvelope compile-gate tests (composition boundary) ─────────

#[cfg(test)]
mod scope_gate_tests {
    use super::*;
    use alzina_core::identity::{Scope, WeaveId};

    /// Verify that the composition SpawnSpec emission boundary constructs a
    /// `ScopedEnvelope` with the scope derived from `SpawnSpec::scope()`,
    /// and that the inner event is an AlzinaEvent that can be published to
    /// the bus (bus stays `Sender<AlzinaEvent>` per D15-05).
    #[test]
    fn composition_spawn_wraps_emission_with_spawn_spec_scope() {
        // SpawnSpec with a weave-bound scope.
        let spec = SpawnSpec {
            agent_id: AgentId::new("huginn"),
            task_template: "test task".into(),
            model_override: None,
            timeout: None,
            session_id_override: None,
            weave_id: Some(WeaveId::new("W-comp-001")),
            node_id: Some("huginn_0".into()),
            dispatch_id: None,
        };

        // spec.scope() returns Scope::Weave("W-comp-001") — the typed accessor.
        assert_eq!(spec.scope().as_str(), "W-comp-001");

        // Construct a ScopedEnvelope at the SpawnSpec emission boundary.
        // This is the compile gate: the writer MUST have a scope from SpawnSpec.
        let event = AlzinaEvent::session_spawned_ad_hoc(
            String::new(), // session_id allocated by runner
            spec.agent_id.as_str().to_owned(),
            None,
            spec.task_template.chars().take(120).collect(),
            0,
            spec.scope(),
        );
        let envelope = ScopedEnvelope::new(spec.scope(), event);

        // Scope "W-comp-001" is preserved on the envelope.
        assert_eq!(envelope.scope.as_str(), "W-comp-001");

        // The inner event is AlzinaEvent::SessionSpawned — the bus receives
        // `envelope.event`, not the ScopedEnvelope (channel stays AlzinaEvent).
        assert!(matches!(envelope.event, AlzinaEvent::SessionSpawned { .. }));
    }

    /// Verify SpawnSpec with no weave_id gives Scope::SessionDefault.
    #[test]
    fn composition_spawn_wraps_emission_with_session_default_scope_when_unweaved() {
        let spec = SpawnSpec {
            agent_id: AgentId::new("vefr"),
            task_template: "unweaved task".into(),
            model_override: None,
            timeout: None,
            session_id_override: None,
            weave_id: None, // no weave
            node_id: None,
            dispatch_id: None,
        };

        assert_eq!(spec.scope(), Scope::SessionDefault);

        let event = AlzinaEvent::session_spawned_ad_hoc(
            String::new(),
            "vefr".into(),
            None,
            "unweaved".into(),
            0,
            spec.scope(),
        );
        let envelope = ScopedEnvelope::new(spec.scope(), event);
        assert_eq!(envelope.scope, Scope::SessionDefault);
    }
}
