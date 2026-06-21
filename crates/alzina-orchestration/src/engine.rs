//! OrchestratorEngine — the top-level public API for Alzina's composition system.
//!
//! Everything else in this crate is internal machinery. The engine is the
//! sole entry point for dispatching agent work — whether a single spawn,
//! a multi-step composition, or a named pattern lookup.
//!
//! # Architecture
//!
//! ```text
//! OrchestratorEngine
//!   ├── execute(CompOp)           → compile + run composition
//!   ├── execute_pattern(name)     → lookup pattern → build CompOp → execute
//!   └── spawn_single(SpawnSpec)   → convenience for single-agent dispatch
//!
//! Internally:
//!   CompOp → CompositionCompiler → CompiledGraph → execute → ExecutionResult
//! ```
//!
//! # Governance Integration
//!
//! Every path through the engine passes through GovernanceLayer:
//! - PreSpawn hooks fire before any agent execution
//! - Envelope parsing + validation on every return
//! - CONTEXT_UPDATE → learnings merge
//! - Complete hooks fire after every spawn

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{debug, info, instrument};

use alzina_core::composition::GateVerdict;
use alzina_core::identity::WeaveId;
use alzina_core::{AlzinaError, AlzinaResult, QualityGate};

use alzina_governance::GovernanceLayer;

use alzina_core::envelope::Signal;

use crate::composition::compiler::{self, CompOp, GateSpec, SpawnSpec};
use crate::composition::leaf_hook::{CompositionLeafHook, noop_hook};
use crate::quality::envelope_gate::EnvelopeQualityGate;
use crate::runner::alzina_runner::{AlzinaRunner, SpawnResult};
use crate::session::hierarchy::SessionHierarchy;

// ── Pattern Registry ────────────────────────────────────────────────────────

/// A named composition pattern — maps pattern names (e.g. "norn-triad",
/// "build-and-gate") to composition constructors.
///
/// Patterns are the DISPATCH.md-style lookup table: given a name and a
/// `PatternContext`, produce a `CompOp` ready for execution.
pub trait PatternRegistry: Send + Sync {
    /// Look up a named pattern and instantiate it with the given context.
    fn resolve(&self, pattern: &str, context: &PatternContext) -> AlzinaResult<CompOp>;

    /// List available pattern names.
    fn list_patterns(&self) -> Vec<String>;
}

/// In-memory pattern registry backed by a closure map.
///
/// For production use, patterns are registered at engine construction time
/// from workspace configuration. For testing, patterns are registered inline.
pub struct InMemoryPatternRegistry {
    patterns: HashMap<String, Box<dyn Fn(&PatternContext) -> AlzinaResult<CompOp> + Send + Sync>>,
}

impl Default for InMemoryPatternRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryPatternRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Register a named pattern.
    pub fn register(
        mut self,
        name: impl Into<String>,
        builder: impl Fn(&PatternContext) -> AlzinaResult<CompOp> + Send + Sync + 'static,
    ) -> Self {
        self.patterns.insert(name.into(), Box::new(builder));
        self
    }
}

impl PatternRegistry for InMemoryPatternRegistry {
    fn resolve(&self, pattern: &str, context: &PatternContext) -> AlzinaResult<CompOp> {
        let builder = self.patterns.get(pattern).ok_or_else(|| {
            AlzinaError::Orchestration(format!(
                "unknown pattern '{pattern}'. Available: {:?}",
                self.list_patterns()
            ))
        })?;
        builder(context)
    }

    fn list_patterns(&self) -> Vec<String> {
        self.patterns.keys().cloned().collect()
    }
}

// ── Engine Types ────────────────────────────────────────────────────────────

/// Result of executing a composition through the engine.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// All spawn outputs, keyed by agent/node name.
    pub outputs: HashMap<String, SpawnResult>,
    /// Synthesis output, if a synthesis step was part of the composition.
    pub synthesis: Option<SpawnResult>,
    /// Quality gate verdict, if a gate was evaluated.
    pub quality_verdict: Option<GateVerdict>,
    /// Number of iterations (for looped compositions).
    pub iterations: u32,
    /// Wall-clock duration of the entire execution.
    pub duration: Duration,
    /// Raw text outputs from all spawns, keyed by node name.
    pub raw_outputs: HashMap<String, String>,
    /// All signals extracted from all spawns during execution.
    pub all_signals: Vec<Signal>,
    /// 260515-lmg/3: aggregated artifact paths from all per-agent envelopes
    /// and synthesis. Populated by `bridge_result` as the deduplicated union
    /// of `outputs.values().envelope.artifacts` and
    /// `synthesis.envelope.artifacts`. Forwarded by the daemon's
    /// pattern-dispatch SessionCompleted event so the audit log shows what
    /// the composition actually wrote. Pre-fix this aggregation did not
    /// exist and `session_manager` hardcoded `artifacts: vec![]`,
    /// producing `artifacts:[]` events even when sub-agents returned
    /// populated envelopes.
    pub artifacts: Vec<PathBuf>,
    /// Emergent observations aggregated from all spawns: (node_id, text).
    pub emergent_observations: Vec<(String, String)>,
    /// Next steps aggregated from all spawns: (node_id, text).
    pub next_steps: Vec<(String, String)>,
}

/// Context for instantiating a named pattern.
#[derive(Debug, Clone)]
pub struct PatternContext {
    /// Available agents, keyed by role name (e.g. "gna", "smidr").
    pub agents: HashMap<String, SpawnSpec>,
    /// Optional synthesiser for patterns that include synthesis.
    pub synthesiser: Option<SpawnSpec>,
    /// Optional quality gate specification.
    pub gate: Option<GateSpec>,
}

// ── OrchestratorEngine ──────────────────────────────────────────────────────

/// Top-level API for Alzina's composition system.
///
/// The engine owns the runner, governance layer, session hierarchy, and
/// pattern registry. All composition work flows through one of three methods:
///
/// - `execute` — run a pre-built `CompOp`
/// - `execute_pattern` — look up a named pattern, instantiate, execute
/// - `spawn_single` — convenience for single-agent dispatch
pub struct OrchestratorEngine {
    runner: Arc<AlzinaRunner>,
    governance: Arc<GovernanceLayer>,
    sessions: Arc<SessionHierarchy>,
    quality_gate: Arc<dyn QualityGate>,
    patterns: Arc<dyn PatternRegistry>,
    /// Plan 10-05: per-leaf dispatch hook. Defaults to `NoopLeafHook`.
    /// Daemon injects `DaemonLeafHook` via `with_leaf_hook` so each
    /// composition leaf registers in `DispatchRegistry` before the runner
    /// dispatches it, wiring Phase 6/7's announcement + auto-continuation chain.
    leaf_hook: Arc<dyn CompositionLeafHook>,
}

impl OrchestratorEngine {
    /// Construct a new engine with all dependencies.
    pub fn new(
        runner: Arc<AlzinaRunner>,
        governance: Arc<GovernanceLayer>,
        sessions: Arc<SessionHierarchy>,
    ) -> Self {
        let quality_gate: Arc<dyn QualityGate> =
            Arc::new(EnvelopeQualityGate::new(governance.clone()));
        let patterns: Arc<dyn PatternRegistry> = Arc::new(InMemoryPatternRegistry::new());

        Self {
            runner,
            governance,
            sessions,
            quality_gate,
            patterns,
            leaf_hook: noop_hook(),
        }
    }

    /// Construct with a custom pattern registry.
    pub fn with_patterns(
        runner: Arc<AlzinaRunner>,
        governance: Arc<GovernanceLayer>,
        sessions: Arc<SessionHierarchy>,
        patterns: Arc<dyn PatternRegistry>,
    ) -> Self {
        let quality_gate: Arc<dyn QualityGate> =
            Arc::new(EnvelopeQualityGate::new(governance.clone()));

        Self {
            runner,
            governance,
            sessions,
            quality_gate,
            patterns,
            leaf_hook: noop_hook(),
        }
    }

    /// Construct with both custom pattern registry and quality gate.
    pub fn with_patterns_and_gate(
        runner: Arc<AlzinaRunner>,
        governance: Arc<GovernanceLayer>,
        sessions: Arc<SessionHierarchy>,
        patterns: Arc<dyn PatternRegistry>,
        quality_gate: Arc<dyn QualityGate>,
    ) -> Self {
        Self {
            runner,
            governance,
            sessions,
            quality_gate,
            patterns,
            leaf_hook: noop_hook(),
        }
    }

    /// Inject a custom leaf hook (used by the daemon to wire register_dispatch).
    ///
    /// The daemon calls this to replace the default `NoopLeafHook` with a
    /// `DaemonLeafHook` that calls `register_dispatch` for each composition leaf.
    pub fn with_leaf_hook(mut self, hook: Arc<dyn CompositionLeafHook>) -> Self {
        self.leaf_hook = hook;
        self
    }

    /// Execute with an explicit compose_id and leaf hook (Plan 10-05 daemon seam).
    ///
    /// Used by the daemon's `dispatch_compose` handler to inject a
    /// `DaemonLeafHook` that calls `register_dispatch` per leaf. All other
    /// callers use `execute()` which installs `NoopLeafHook` (backwards-compat).
    #[instrument(skip(self, op, hook), fields(compose_id = %compose_id, weave = ?weave_id))]
    pub async fn execute_with_hook(
        &self,
        op: CompOp,
        weave_id: Option<WeaveId>,
        compose_id: String,
        hook: Arc<dyn CompositionLeafHook>,
    ) -> AlzinaResult<crate::engine::ExecutionResult> {
        let start = std::time::Instant::now();

        compiler::validate_op(&op)?;

        debug!("compiling composition with hook");
        let compiled = compiler::compile(op, self.runner.clone(), self.quality_gate.clone())?;

        debug!("executing compiled graph with hook");
        let graph_result = compiled
            .execute_with_hook(weave_id, compose_id, hook)
            .await?;

        let duration = start.elapsed();
        let result = self.bridge_result(graph_result, duration);

        info!(
            outputs = result.outputs.len(),
            has_synthesis = result.synthesis.is_some(),
            has_verdict = result.quality_verdict.is_some(),
            iterations = result.iterations,
            duration_ms = duration.as_millis(),
            "composition execution with hook complete"
        );

        Ok(result)
    }

    /// Execute a composition operation.
    ///
    /// Compiles the `CompOp` into a `CompiledGraph`, executes it, and
    /// bridges the internal result into the engine's `ExecutionResult`.
    #[instrument(skip(self, op), fields(weave = ?weave_id))]
    pub async fn execute(
        &self,
        op: CompOp,
        weave_id: Option<WeaveId>,
    ) -> AlzinaResult<ExecutionResult> {
        let start = Instant::now();

        // RT3-05: Validate CompOp at engine boundary before compilation.
        // compile() also validates internally, but patterns from PatternRegistry
        // should be validated at the public API surface too.
        compiler::validate_op(&op)?;

        debug!("compiling composition");
        let compiled = compiler::compile(op, self.runner.clone(), self.quality_gate.clone())?;

        debug!("executing compiled graph");
        let graph_result = compiled.execute(weave_id).await?;

        let duration = start.elapsed();

        // Bridge compiler::ExecutionResult → engine::ExecutionResult
        let result = self.bridge_result(graph_result, duration);

        info!(
            outputs = result.outputs.len(),
            has_synthesis = result.synthesis.is_some(),
            has_verdict = result.quality_verdict.is_some(),
            iterations = result.iterations,
            duration_ms = duration.as_millis(),
            "composition execution complete"
        );

        Ok(result)
    }

    /// Look up a named pattern, instantiate it with context, and execute.
    ///
    /// Pattern names map to DISPATCH.md-style composition templates:
    /// - `"norn-triad"` → Parallel(Urðr, Skuld) + Synthesis(Verðandi)
    /// - `"build-and-gate"` → Gate(Spawn(builder), criteria)
    /// - etc.
    ///
    /// `weave_id` is forwarded to `execute()` so pattern-dispatched ops can be
    /// weaved (LANDMINE 9 fix — previously hardcoded to `None`). Phase 10 will
    /// wire the weave_id from the HTTP gate through pattern dispatch.
    #[instrument(skip(self, context), fields(pattern = %pattern, weave = ?weave_id))]
    pub async fn execute_pattern(
        &self,
        pattern: &str,
        context: PatternContext,
        weave_id: Option<WeaveId>,
    ) -> AlzinaResult<ExecutionResult> {
        debug!("resolving pattern");
        let op = self.patterns.resolve(pattern, &context)?;
        self.execute(op, weave_id).await
    }

    /// Convenience: dispatch a single agent.
    ///
    /// Wraps the spec in `CompOp::Spawn` and runs through the full engine
    /// pipeline, returning the `SpawnResult` directly.
    #[instrument(skip(self, spec), fields(
        agent = %spec.agent_id,
        weave = ?weave_id,
    ))]
    pub async fn spawn_single(
        &self,
        spec: SpawnSpec,
        weave_id: Option<WeaveId>,
    ) -> AlzinaResult<SpawnResult> {
        let mut result = self.execute(CompOp::Spawn(spec), weave_id).await?;

        // Extract the single spawn result. The compiler's bridge_result
        // populates SpawnResult.envelope and SpawnResult.raw_outputs but
        // leaves SpawnResult.raw empty (envelope/raw are tracked in
        // separate maps). Phase 4 surfaces the full sub-agent return
        // text on SessionCompleted.envelope, so populate `raw` from the
        // matching `raw_outputs` entry before returning.
        let (name, mut spawn_result) = result
            .outputs
            .drain()
            .next()
            .ok_or_else(|| AlzinaError::Orchestration("spawn produced no output".into()))?;
        if spawn_result.raw.is_empty() {
            if let Some(raw) = result.raw_outputs.remove(&name) {
                spawn_result.raw = raw;
            }
        }
        Ok(spawn_result)
    }

    /// Bridge the compiler's internal result to the engine's public result.
    fn bridge_result(
        &self,
        graph_result: compiler::ExecutionResult,
        duration: Duration,
    ) -> ExecutionResult {
        let mut outputs = HashMap::new();
        let mut synthesis = None;
        let mut quality_verdict = None;
        let mut iterations = 0u32;

        // Convert envelopes to SpawnResults. The compiler's ExecutionResult
        // carries envelopes keyed by node name.
        //
        // WARN-01 fix: populate `raw` from render_envelope_as_prose so that
        // downstream consumers (audit logs, session_manager.envelope) see
        // consistent text on this compiler-internal (trusted) path, matching
        // the runner's typed-envelope branch at alzina_runner.rs:916.
        for (name, envelope) in &graph_result.envelopes {
            let raw = alzina_governance::envelope::render_envelope_as_prose(envelope);
            let spawn_result = SpawnResult {
                session_id: alzina_core::identity::SessionId::new(),
                envelope: envelope.clone(),
                raw,
                signals: Vec::new(),
                quality_issues: Vec::new(),
            };

            if name.starts_with("synthesis") {
                synthesis = Some(spawn_result);
            } else {
                outputs.insert(name.clone(), spawn_result);
            }
        }

        // Extract gate verdict from state if present
        if let Some(verdict_val) = graph_result.state.get("_gate:verdict") {
            if let Ok(verdict) = serde_json::from_value::<GateVerdict>(verdict_val.clone()) {
                quality_verdict = Some(verdict);
            }
        }

        // Extract iteration count from state if present
        if let Some(iter_val) = graph_result.state.get("_meta:iteration") {
            if let Some(n) = iter_val.as_u64() {
                iterations = n as u32;
            }
        }

        // 260515-lmg/3: aggregate per-agent envelope artifacts (plus
        // synthesis envelope artifacts, when present) into a deduplicated
        // Vec<PathBuf> on the public ExecutionResult. The daemon's
        // pattern-dispatch SessionCompleted handler forwards this so the
        // audit log surfaces what the composition actually wrote.
        // HashSet drives dedupe; Vec preserves first-seen insertion order
        // (outputs in HashMap iteration order, then synthesis).
        let mut seen: HashSet<PathBuf> = HashSet::new();
        let mut artifacts: Vec<PathBuf> = Vec::new();
        for sr in outputs.values() {
            for p in &sr.envelope.artifacts {
                if seen.insert(p.clone()) {
                    artifacts.push(p.clone());
                }
            }
        }
        if let Some(s) = synthesis.as_ref() {
            for p in &s.envelope.artifacts {
                if seen.insert(p.clone()) {
                    artifacts.push(p.clone());
                }
            }
        }

        ExecutionResult {
            outputs,
            synthesis,
            quality_verdict,
            iterations,
            duration,
            raw_outputs: graph_result.raw_outputs,
            all_signals: graph_result.all_signals,
            artifacts,
            emergent_observations: graph_result.emergent_observations,
            next_steps: graph_result.next_steps,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::compiler::{CompOp, SpawnSpec, SynthesisSpec};
    use crate::runner::alzina_runner::AgentExecutor;
    use crate::test_helpers::{build_test_runner, well_formed_envelope};

    use alzina_core::envelope::EnvelopeStatus;
    use alzina_core::identity::AgentId;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    // ── Mock executor ───────────────────────────────────────────────────

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

    // ── Helpers ─────────────────────────────────────────────────────────

    fn test_spec(agent: &str, task: &str) -> SpawnSpec {
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

    async fn build_engine(
        executor: Arc<dyn AgentExecutor>,
        patterns: Option<Arc<dyn PatternRegistry>>,
    ) -> (tempfile::TempDir, OrchestratorEngine) {
        let dir = tempfile::tempdir().unwrap();
        let runner = Arc::new(build_test_runner(executor, None, dir.path()).await.unwrap());
        let governance = runner.governance().clone();
        let sessions = runner.sessions().clone();

        let engine = match patterns {
            Some(p) => OrchestratorEngine::with_patterns(runner, governance, sessions, p),
            None => OrchestratorEngine::new(runner, governance, sessions),
        };

        (dir, engine)
    }

    // ── Test: Single spawn via engine ───────────────────────────────────

    #[tokio::test]
    async fn single_spawn_via_engine() {
        let executor = Arc::new(MockExecutor::single());
        let (_dir, engine) = build_engine(executor, None).await;

        let spec = test_spec("smidr", "analyse workspace");
        let result = engine.spawn_single(spec, None).await.unwrap();

        assert_eq!(result.envelope.status, EnvelopeStatus::Complete);
    }

    // ── Test: Sequential composition via engine ─────────────────────────

    #[tokio::test]
    async fn sequential_composition_via_engine() {
        let executor = Arc::new(MockExecutor::repeating(2));
        let (_dir, engine) = build_engine(executor, None).await;

        let op = CompOp::Sequential(vec![
            CompOp::Spawn(test_spec("urdr", "read context")),
            CompOp::Spawn(test_spec("skuld", "plan future")),
        ]);
        let result = engine.execute(op, None).await.unwrap();

        assert_eq!(result.outputs.len(), 2);
        assert!(result.synthesis.is_none());
    }

    // ── Test: Pattern lookup + execution (mock pattern registry) ────────

    #[tokio::test]
    async fn pattern_lookup_and_execution() {
        let executor = Arc::new(MockExecutor::repeating(3));

        let registry = Arc::new(
            InMemoryPatternRegistry::new().register("norn-triad", |ctx| {
                let agents: Vec<CompOp> = ctx
                    .agents
                    .values()
                    .map(|spec| CompOp::Spawn(spec.clone()))
                    .collect();

                match &ctx.synthesiser {
                    Some(synth) => Ok(CompOp::Synthesise(
                        Box::new(CompOp::Parallel(agents)),
                        SynthesisSpec {
                            synthesiser: synth.clone(),
                            task: None,
                        },
                    )),
                    None => Ok(CompOp::Parallel(agents)),
                }
            }),
        );

        let (_dir, engine) = build_engine(executor, Some(registry)).await;

        let mut agents = HashMap::new();
        agents.insert("past".into(), test_spec("urdr", "read history"));
        agents.insert("future".into(), test_spec("skuld", "plan ahead"));

        let context = PatternContext {
            agents,
            synthesiser: Some(test_spec("verdandi", "synthesise threads")),
            gate: None,
        };

        let result = engine
            .execute_pattern("norn-triad", context, None)
            .await
            .unwrap();

        // 2 parallel branches + 1 synthesis
        assert!(result.outputs.len() >= 2);
        assert!(result.synthesis.is_some());
    }

    // ── Test: Unknown pattern → error ───────────────────────────────────

    #[tokio::test]
    async fn unknown_pattern_returns_error() {
        let executor = Arc::new(MockExecutor::single());
        let (_dir, engine) = build_engine(executor, None).await;

        let context = PatternContext {
            agents: HashMap::new(),
            synthesiser: None,
            gate: None,
        };

        let result = engine.execute_pattern("nonexistent", context, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown pattern"), "error was: {err}");
    }

    // ── Test: Error propagation from compiler ───────────────────────────

    #[tokio::test]
    async fn error_propagation_from_compiler_validation() {
        let executor = Arc::new(MockExecutor::single());
        let (_dir, engine) = build_engine(executor, None).await;

        let op = CompOp::Sequential(vec![]);
        let result = engine.execute(op, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("at least one"),
            "should be compiler validation error, was: {err}"
        );
    }

    // ── Test: Error propagation from runner ─────────────────────────────

    #[tokio::test]
    async fn error_propagation_from_runner() {
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
                Err(AlzinaError::Orchestration("LLM API unavailable".into()))
            }
        }

        let executor: Arc<dyn AgentExecutor> = Arc::new(FailingExecutor);
        let (_dir, engine) = build_engine(executor, None).await;

        let op = CompOp::Spawn(test_spec("smidr", "do work"));
        let result = engine.execute(op, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("LLM API unavailable") || err.contains("failed"),
            "error should propagate from runner, was: {err}"
        );
    }

    // ── Test: Duration is tracked ───────────────────────────────────────

    #[tokio::test]
    async fn duration_is_tracked() {
        let executor = Arc::new(MockExecutor::single());
        let (_dir, engine) = build_engine(executor, None).await;

        let op = CompOp::Spawn(test_spec("test", "quick task"));
        let result = engine.execute(op, None).await.unwrap();

        assert!(
            result.duration.as_nanos() > 0,
            "duration should be non-zero"
        );
    }

    // ── Test: 260515-lmg/3 — ExecutionResult.artifacts aggregation ──────
    //
    // bridge_result must fold per-agent envelope.artifacts (plus synthesis
    // envelope artifacts, when present) into a deduplicated Vec<PathBuf>
    // on ExecutionResult so the daemon's pattern-dispatch SessionCompleted
    // event can surface what was actually written. Pre-fix the field did
    // not exist and the daemon hardcoded `artifacts: vec![]`.

    #[tokio::test]
    async fn execution_result_artifacts_aggregates_per_agent_and_dedupes() {
        use indexmap::IndexMap;
        use std::path::PathBuf;
        use std::time::Duration;

        // Build engine purely to access bridge_result; the executor is
        // never invoked because we feed a synthetic graph_result.
        let executor = Arc::new(MockExecutor::single());
        let (_dir, engine) = build_engine(executor, None).await;

        // Two per-agent envelopes share `studies/x/findings.md`; the
        // aggregated artifacts list must contain three distinct paths.
        let mut envelopes: IndexMap<String, alzina_core::Envelope> = IndexMap::new();
        envelopes.insert(
            "urdr".into(),
            alzina_core::Envelope {
                status: EnvelopeStatus::Complete,
                artifacts: vec![
                    PathBuf::from("studies/x/findings.md"),
                    PathBuf::from("studies/x/synthesis.md"),
                ],
                signal: None,
                tensions: None,
                emergent: None,
                next: None,
                context_update: None,
            },
        );
        envelopes.insert(
            "skuld".into(),
            alzina_core::Envelope {
                status: EnvelopeStatus::Complete,
                artifacts: vec![
                    PathBuf::from("studies/x/findings.md"), // duplicate
                    PathBuf::from("studies/x/design.md"),
                ],
                signal: None,
                tensions: None,
                emergent: None,
                next: None,
                context_update: None,
            },
        );
        // Synthesis envelope contributes one more unique path.
        envelopes.insert(
            "synthesis_verdandi".into(),
            alzina_core::Envelope {
                status: EnvelopeStatus::Complete,
                artifacts: vec![PathBuf::from("studies/x/synth-summary.md")],
                signal: None,
                tensions: None,
                emergent: None,
                next: None,
                context_update: None,
            },
        );

        let graph_result = compiler::ExecutionResult {
            state: Default::default(),
            envelopes,
            raw_outputs: HashMap::new(),
            all_signals: Vec::new(),
            emergent_observations: Vec::new(),
            next_steps: Vec::new(),
        };

        let result = engine.bridge_result(graph_result, Duration::from_millis(1));

        assert_eq!(
            result.artifacts.len(),
            4,
            "deduplicated union of (findings, synthesis, design, synth-summary) must be 4 distinct paths, got: {:?}",
            result.artifacts
        );
        for expected in [
            "studies/x/findings.md",
            "studies/x/synthesis.md",
            "studies/x/design.md",
            "studies/x/synth-summary.md",
        ] {
            assert!(
                result
                    .artifacts
                    .iter()
                    .any(|p| p == &PathBuf::from(expected)),
                "aggregate must contain {expected}: {:?}",
                result.artifacts
            );
        }
    }

    // ── Test: Pattern registry lists patterns ───────────────────────────

    #[test]
    fn pattern_registry_lists_patterns() {
        let registry = InMemoryPatternRegistry::new()
            .register("alpha", |_| Ok(CompOp::Spawn(test_spec("a", "t"))))
            .register("beta", |_| Ok(CompOp::Spawn(test_spec("b", "t"))));

        let patterns = registry.list_patterns();
        assert_eq!(patterns.len(), 2);
        assert!(patterns.contains(&"alpha".to_string()));
        assert!(patterns.contains(&"beta".to_string()));
    }
}
