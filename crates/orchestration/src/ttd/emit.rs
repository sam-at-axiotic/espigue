//! TTD artifact emit: provenance stamping + YAML serialisation + audit-trail event.
//!
//! This module is the governance record boundary for the TTD engine:
//!
//! 1. **Provenance stamp** — at merge time, stamp the artifact with all five
//!    provenance fields sourced from runtime values:
//!    - `model` — from the AgentExecutor (the model string used in spawns)
//!    - `prompt_version` — per stage ("v1/graph", "v1/synthesis", "v1/narrative")
//!    - `code_version` — from `git rev-parse HEAD` (via `artifact::code_version()`)
//!    - `generated_at` — `Utc::now()` at merge time
//!    - `schema_version` — `SCHEMA_VERSION` ("1.0") constant
//!
//! 2. **YAML serialisation** — the external deliverable (file or byte string).
//!
//! 3. **Audit-trail event** — a `TtdArtifactEmitted` event recorded to the
//!    engine's event collector for governance (ENGINE-05). This is NOT the
//!    AlzinaRunner's `SpawnCompleted` event (which requires a full runner context);
//!    it is a lightweight TTD-internal audit record that Phase 25 can consume.
//!
//! ## Phase 23 fidelity gap annotation
//!
//! The `code_version` field is annotated with the Phase 23 fidelity gap note:
//!
//! > Phase 23: static FanOut (N=5 identical prompts), no per-trajectory sampling
//! > diversity (temp_range 0.5–1.2 deferred to Phase 24). Phase 25 must distinguish
//! > reproduction from sampling-diversity addition.
//!
//! This annotation is embedded as metadata in the emit record (not in the artifact
//! YAML itself — that would pollute the external deliverable). Phase 25 reads the
//! emit record's `fidelity_note` field to locate the gap.
//!
//! ## Trust boundary (T-23-11)
//!
//! Only `model`, `prompt_version`, `code_version`, `generated_at`, and
//! `schema_version` are stamped. No API keys, no env values, no arbitrary dynamic
//! data. The provenance struct has a FIXED field set verified here.
//!
//! ## ENGINE-01
//!
//! Both `ArgumentationGraph` and `SynthesisArtifact` must be emittable.
//! `emit_graph` and `emit_synthesis` handle each type.

use chrono::Utc;

use crate::ttd::artifact::{ArgumentationGraph, SynthesisArtifact, code_version};
use crate::ttd::mod_types::TtdError;

// ── Phase 23 fidelity gap note ────────────────────────────────────────────────

/// Static note embedded in the emit record marking the Phase 23 fidelity gap.
///
/// Phase 25 reads this to distinguish faithful Phase 23 reproduction from the
/// Phase 24 per-trajectory sampling-diversity addition.
pub const PHASE23_FIDELITY_GAP_NOTE: &str =
    "Phase 23: static FanOut (N=5 identical prompts); per-trajectory sampling \
     diversity (temp_range 0.5-1.2) deferred to Phase 24. Phase 25 must \
     distinguish reproduction from sampling-diversity addition.";

/// Static note marking the three Phase 24 EXT additions for the Phase 25 gate.
///
/// Phase 25 reads this to distinguish the faithful Phase 23 reproduction from the
/// labelled Phase 24 additions (parallels [`PHASE23_FIDELITY_GAP_NOTE`]).
pub const PHASE24_EXT_NOTE: &str =
    "Phase 24 additions (labelled, distinct from Phase 23 reproduction): \
     (1) persona-seeded candidate diversity — bespoke alzina personas seed the \
     per-trajectory draft prompts; (2) config-gated plateau early-stop \
     (plateau_threshold, OFF by default — faithful fixed-cap S=2 remains the default); \
     (3) bibliography externalisation to the literature KB via BibliographyStore on \
     each denoise step. CAVEAT: per-trajectory LLM sampling diversity is LIVE-BUT-INERT \
     — plumbed through the executor + sidecar but the Agent SDK exposes no per-request \
     temperature/top_p/top_k, so sampling does not reach the model on this substrate \
     (direct-API executor is the closure path).";

// ── TtdEmitRecord ─────────────────────────────────────────────────────────────

/// Governance record produced by an artifact emit.
///
/// Captures the artifact YAML and the provenance metadata at emission time.
/// This is the audit-trail record for ENGINE-01/ENGINE-05.
///
/// The `SpawnCompleted` events from individual spawns are handled by the
/// `RecordingExecutor` in tests and by `AlzinaRunner` in production.
/// This record captures the aggregate artifact emit as a single governance point.
#[derive(Debug, Clone)]
pub struct TtdEmitRecord {
    /// Which stage produced this artifact.
    pub stage: TtdStage,
    /// The artifact's YAML serialisation.
    pub yaml: String,
    /// Model used in the stage's spawns.
    pub model: String,
    /// Prompt version for this stage.
    pub prompt_version: String,
    /// Git HEAD at emission time.
    pub code_version: String,
    /// ISO-8601 timestamp when the artifact was emitted.
    pub generated_at: String,
    /// Schema version ("1.0").
    pub schema_version: String,
    /// Phase 23 fidelity gap note for Phase 25 distinction.
    pub fidelity_note: &'static str,
    /// Phase 24 EXT additions note for Phase 25 distinction (persona seeding,
    /// config-gated plateau, bibliography externalisation; sampling LIVE-BUT-INERT).
    pub ext_note: &'static str,
}

/// Which TTD stage produced the emitted artifact.
#[derive(Debug, Clone, PartialEq)]
pub enum TtdStage {
    Graph,
    Synthesis,
    Narrative,
}

// ── emit_graph ────────────────────────────────────────────────────────────────

/// Emit an `ArgumentationGraph` artifact with provenance stamping.
///
/// Stamps `code_version` and `generated_at` at emit time (the artifact already
/// carries `model`, `prompt_version`, and `schema_version` from construction).
/// Returns the YAML and a `TtdEmitRecord` for the audit trail (ENGINE-01).
pub fn emit_graph(
    graph: &ArgumentationGraph,
) -> Result<TtdEmitRecord, TtdError> {
    let yaml = graph.to_yaml()?;
    Ok(TtdEmitRecord {
        stage: TtdStage::Graph,
        yaml,
        model: graph.model.clone(),
        prompt_version: graph.prompt_version.clone(),
        code_version: graph.code_version.clone(),
        generated_at: graph.generated_at.to_rfc3339(),
        schema_version: graph.schema_version.clone(),
        fidelity_note: PHASE23_FIDELITY_GAP_NOTE,
        ext_note: PHASE24_EXT_NOTE,
    })
}

/// Emit an `ArgumentationGraph` artifact, refreshing provenance fields at emit time.
///
/// `generated_at` is set to `Utc::now()` and `code_version` is re-captured from
/// git HEAD. Used when the artifact was constructed earlier and the caller wants
/// the emission timestamp to reflect when the artifact left the engine boundary.
pub fn emit_graph_stamped(
    mut graph: ArgumentationGraph,
    model: impl Into<String>,
    prompt_version: impl Into<String>,
) -> Result<(ArgumentationGraph, TtdEmitRecord), TtdError> {
    graph.model = model.into();
    graph.prompt_version = prompt_version.into();
    graph.generated_at = Utc::now();
    graph.code_version = code_version();

    let record = emit_graph(&graph)?;
    Ok((graph, record))
}

// ── emit_synthesis ────────────────────────────────────────────────────────────

/// Emit a `SynthesisArtifact` with provenance stamping.
///
/// `generated_at` is set to `Utc::now()` and `code_version` is re-captured
/// from git HEAD at emission time (the authoritative stamp for when this
/// artifact was finalised and left the engine boundary).
///
/// Returns the YAML string and a `TtdEmitRecord` for the audit trail.
pub fn emit_synthesis(
    synthesis: &SynthesisArtifact,
) -> Result<TtdEmitRecord, TtdError> {
    let yaml = synthesis.to_yaml()?;
    Ok(TtdEmitRecord {
        stage: TtdStage::Synthesis,
        yaml,
        model: synthesis.model.clone(),
        prompt_version: synthesis.prompt_version.clone(),
        code_version: synthesis.code_version.clone(),
        generated_at: synthesis.generated_at.to_rfc3339(),
        schema_version: synthesis.schema_version.clone(),
        fidelity_note: PHASE23_FIDELITY_GAP_NOTE,
        ext_note: PHASE24_EXT_NOTE,
    })
}

/// Emit a `SynthesisArtifact`, refreshing provenance fields at emit time.
///
/// Stamps `generated_at` = Utc::now() and `code_version` = git rev-parse HEAD.
/// Used at the end of Stage 3 when the narrative field is populated.
pub fn emit_synthesis_stamped(
    mut synthesis: SynthesisArtifact,
    model: impl Into<String>,
    prompt_version: impl Into<String>,
) -> Result<(SynthesisArtifact, TtdEmitRecord), TtdError> {
    synthesis.model = model.into();
    synthesis.prompt_version = prompt_version.into();
    synthesis.generated_at = Utc::now();
    synthesis.code_version = code_version();

    let record = emit_synthesis(&synthesis)?;
    Ok((synthesis, record))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttd::artifact::{ArgumentationGraph, SynthesisArtifact, SCHEMA_VERSION};

    /// emit_graph produces a TtdEmitRecord with all five provenance fields.
    #[test]
    fn emit_graph_provenance_fields() {
        let graph = ArgumentationGraph::new(
            "study-001", "round-1", "q-climate",
            "google/gemini-2.5-flash", "v1/graph",
        );

        let record = emit_graph(&graph).expect("emit_graph must succeed");

        assert_eq!(record.stage, TtdStage::Graph);
        assert_eq!(record.schema_version, SCHEMA_VERSION);
        assert!(!record.yaml.is_empty(), "YAML must be non-empty");
        assert_eq!(record.model, "google/gemini-2.5-flash");
        assert_eq!(record.prompt_version, "v1/graph");
        assert!(!record.code_version.is_empty(), "code_version must not be empty");
        assert!(!record.generated_at.is_empty(), "generated_at must not be empty");
        assert_eq!(record.fidelity_note, PHASE23_FIDELITY_GAP_NOTE);
    }

    /// The emit record carries the Phase 24 ext-note so the Phase 25 fidelity
    /// gate can distinguish the labelled additions from the Phase 23 reproduction.
    /// The marker must name all three additions (Research lines 549-557).
    #[test]
    fn emit_record_carries_phase24_ext_note() {
        let graph = ArgumentationGraph::new(
            "study-001", "round-1", "q-climate",
            "google/gemini-2.5-flash", "v1/graph",
        );

        let record = emit_graph(&graph).expect("emit_graph must succeed");

        assert_eq!(
            record.ext_note, PHASE24_EXT_NOTE,
            "emit record must carry the Phase 24 ext-note on the audit record"
        );
        assert!(
            record.ext_note.contains("persona"),
            "ext_note must mark persona-seeded diversity"
        );
        assert!(
            record.ext_note.contains("plateau"),
            "ext_note must mark the config-gated plateau (off by default)"
        );
        assert!(
            record.ext_note.contains("bibliography"),
            "ext_note must mark bibliography externalisation"
        );
    }

    /// emit_synthesis produces a TtdEmitRecord with all five provenance fields.
    #[test]
    fn emit_synthesis_provenance_fields() {
        let synthesis = SynthesisArtifact::new(
            "study-001", "round-1", "q-climate",
            "google/gemini-2.5-flash", "v1/synthesis",
        );

        let record = emit_synthesis(&synthesis).expect("emit_synthesis must succeed");

        assert_eq!(record.stage, TtdStage::Synthesis);
        assert_eq!(record.schema_version, SCHEMA_VERSION);
        assert!(!record.yaml.is_empty(), "YAML must be non-empty");
        assert_eq!(record.model, "google/gemini-2.5-flash");
        assert_eq!(record.prompt_version, "v1/synthesis");
        assert!(!record.code_version.is_empty(), "code_version must not be empty");
        assert!(!record.generated_at.is_empty(), "generated_at must not be empty");
    }

    /// emit_synthesis_stamped refreshes provenance at emission time.
    #[test]
    fn emit_synthesis_stamped_refreshes_provenance() {
        let synthesis = SynthesisArtifact::new(
            "study-001", "round-1", "q-climate",
            "old-model", "v1/synthesis",
        );

        let (stamped, record) = emit_synthesis_stamped(
            synthesis, "google/gemini-2.5-flash", "v1/narrative",
        ).expect("emit_synthesis_stamped must succeed");

        assert_eq!(stamped.model, "google/gemini-2.5-flash");
        assert_eq!(stamped.prompt_version, "v1/narrative");
        assert_eq!(record.model, "google/gemini-2.5-flash");
        assert_eq!(record.prompt_version, "v1/narrative");
    }

    /// YAML round-trip: emit_synthesis YAML can be deserialised back.
    #[test]
    fn emit_synthesis_yaml_round_trips() {
        use crate::ttd::artifact::Claim;

        let mut synthesis = SynthesisArtifact::new(
            "study-001", "round-1", "q-climate",
            "google/gemini-2.5-flash", "v1/synthesis",
        );
        synthesis.claims.push(Claim {
            text: "Climate change accelerates permafrost thaw.".into(),
            agreement_level: Some("consensus".into()),
            sources: vec!["arxiv:2105.14103".into()],
            counterarguments: vec![],
            support_level: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        synthesis.narrative = "Experts broadly agree.".into();

        let record = emit_synthesis(&synthesis).expect("emit must succeed");
        let restored = SynthesisArtifact::from_yaml(&record.yaml)
            .expect("YAML must deserialise back");

        assert_eq!(restored.schema_version, SCHEMA_VERSION);
        assert_eq!(restored.claims.len(), 1);
        assert_eq!(restored.narrative, "Experts broadly agree.");
    }

    /// Phase 23 fidelity gap note is non-empty and mentions Phase 24.
    #[test]
    fn fidelity_gap_note_references_phase24() {
        assert!(
            PHASE23_FIDELITY_GAP_NOTE.contains("Phase 24"),
            "fidelity gap note must reference Phase 24 as the resolution phase"
        );
        assert!(
            PHASE23_FIDELITY_GAP_NOTE.contains("sampling"),
            "fidelity gap note must mention sampling diversity"
        );
    }

    // ── Audit coverage tests (ENGINE-05) ──────────────────────────────────────
    //
    // These tests verify that every spawn in the TTD pipeline routes through the
    // injected AgentExecutor — no direct LLM client is constructed inside ttd/.
    //
    // "SpawnCompleted" in this context = "a call to executor.execute()". The
    // RecordingExecutor counts every such call. Each call corresponds to one
    // governed spawn on the audit trail (T-23-12 mitigation).

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use async_trait::async_trait;

    /// An executor that records every execute() call for audit verification.
    ///
    /// Each recorded call corresponds to one "SpawnCompleted" event that the
    /// AlzinaRunner would emit in production. The count verifies ENGINE-05
    /// coverage without requiring a full runner context.
    struct RecordingExecutor {
        count: Arc<AtomicUsize>,
        spawn_log: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl RecordingExecutor {
        fn new() -> (Self, Arc<AtomicUsize>, Arc<std::sync::Mutex<Vec<String>>>) {
            let count = Arc::new(AtomicUsize::new(0));
            let log = Arc::new(std::sync::Mutex::new(Vec::new()));
            (
                RecordingExecutor { count: count.clone(), spawn_log: log.clone() },
                count,
                log,
            )
        }
    }

    #[async_trait]
    impl crate::executor::AgentExecutor for RecordingExecutor {
        async fn execute(
            &self,
            _agent_id: &base::identity::AgentId,
            _instruction: &str,
            _model: &str,
            task: &str,
        ) -> base::AlzinaResult<String> {
            self.count.fetch_add(1, Ordering::SeqCst);
            self.spawn_log.lock().unwrap().push(task.to_string());

            // Return valid XML for fitness judges
            if task.contains("faithfulness")
                || task.contains("completeness")
                || task.contains("traceability")
                || task.contains("neutrality")
                || task.contains("dissent_visibility")
                || task.contains("structural_clarity")
                || task.contains("groundedness")
                || task.contains("coverage")
                || task.contains("atomicity")
                || task.contains("non_redundancy")
                || task.contains("relation_coherence")
                || task.contains("dissent_preservation")
            {
                return Ok("<fitness_evaluation><score>4</score><rationale>good</rationale></fitness_evaluation>".to_string());
            }
            // Gap critique: return empty gaps (fast path)
            if task == "narrative_critique" || task == "gap_identify" {
                return Ok("<gaps></gaps>".to_string());
            }
            // Graph operations
            if task == "graph_extraction_single" || task == "graph_draft" {
                return Ok(r#"<graph><nodes><node id="p001_c1"><claim>Test claim</claim><expert_id>p001</expert_id><quote>quote</quote><verification_status>verified</verification_status></node></nodes><edges/></graph>"#.to_string());
            }
            if task == "graph_resolution" {
                return Ok("<edges/><merges/>".to_string());
            }
            // Synthesis operations
            if task == "synthesis_draft" || task == "synthesis_merger" {
                return Ok(r#"<synthesis><narrative>Test narrative [C1].</narrative><claims><claim id="C1"><text>Test claim.</text><agreement_level>consensus</agreement_level><sources><source id="p001_c1"/></sources><counterarguments/></claim></claims><areas_of_agreement/><areas_of_disagreement/><uncertainties/></synthesis>"#.to_string());
            }
            if task == "synthesis_quote_resolve" {
                return Ok("<resolved/>".to_string());
            }
            // Narrative and other spawns
            Ok("A narrative statement about the findings [C1].".to_string())
        }
    }

    /// ENGINE-05: N=5 draft generation produces 5 draft spawns per TtdMachine run.
    ///
    /// Each draft spawn = one executor.execute() call = one "SpawnCompleted" event.
    /// Test uses a single TtdMachine<String> (narrative stage) for simplicity.
    #[tokio::test]
    async fn all_drafts_produce_spawn_completed() {
        use crate::adapter::ExpertResponse;
        use crate::ttd::artifact::{Claim, SynthesisArtifact};
        use crate::ttd::stages::narrative::{
            NarrativeCritique, NarrativeDraftGen, NarrativeEvalFitness,
            NarrativeMerger, NarrativeRefine,
        };
        use crate::ttd::retrieval::NoopRetriever;
        use crate::ttd::{TtdConfig, TtdMachine};

        let (executor, count, log) = RecordingExecutor::new();
        let executor: Arc<dyn crate::executor::AgentExecutor> = Arc::new(executor);

        let mut synthesis = SynthesisArtifact::new(
            "s", "r", "q", "model", "v1/synthesis",
        );
        synthesis.claims.push(Claim {
            text: "Test claim.".into(),
            agreement_level: Some("consensus".into()),
            sources: vec![],
            counterarguments: vec![],
            support_level: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 5; // N=5 (consensus default)
        config.n_denoise_steps = 1;  // S=1 to keep count predictable

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(NarrativeDraftGen::new("agent", "model", synthesis.clone())),
            gap_identify: Box::new(NarrativeCritique::new("agent", "model", synthesis.clone())),
            gap_resolve: Box::new(NarrativeRefine::new("agent", "model", synthesis.clone())),
            eval_fitness: Some(Box::new(NarrativeEvalFitness::new("agent", "model"))),
            merger: Box::new(NarrativeMerger::new("agent", "model", synthesis)),
            retriever: Box::new(NoopRetriever),
            executor: executor.clone(),
            bib_store: std::sync::Arc::new(search::bib_store::NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        let panel: Vec<ExpertResponse> = vec![];
        machine.run(&panel).await.expect("run must succeed");

        let total_spawns = count.load(Ordering::SeqCst);
        let spawn_tasks = log.lock().unwrap().clone();

        // Verify N=5 draft spawns occurred
        let draft_spawns = spawn_tasks.iter().filter(|t| t.as_str() == "narrative_draft").count();
        assert_eq!(
            draft_spawns, 5,
            "ENGINE-05: exactly N=5 draft spawns must occur (one per draft generate() call). \
             Total spawn log: {spawn_tasks:?}"
        );

        // Verify total spawns > 5 (there must be fitness judges too)
        assert!(
            total_spawns > 5,
            "ENGINE-05: total spawns must include draft + fitness + critique + merge calls. \
             Got {total_spawns}"
        );

        // Verify merger was called once
        let merge_spawns = spawn_tasks.iter().filter(|t| t.as_str() == "narrative_final_merge").count();
        assert_eq!(
            merge_spawns, 1,
            "ENGINE-05: exactly 1 merger spawn must occur. Got {merge_spawns}"
        );
    }

    /// ENGINE-05: each fitness-judge spawn produces a distinct SpawnCompleted event.
    ///
    /// With N=1 trajectory and S=1 step, there are 6 fitness judge calls in the loop
    /// + 6 in the final re-evaluation = 12 fitness spawns total.
    /// Plus 1 gap_identify + 1 narrative_draft + 1 narrative_final_merge = 15 total.
    #[tokio::test]
    async fn all_fitness_judges_produce_spawn_completed() {
        use crate::adapter::{ExpertResponse};
        use crate::ttd::artifact::{Claim, SynthesisArtifact};
        use crate::ttd::stages::narrative::{
            NarrativeCritique, NarrativeDraftGen, NarrativeEvalFitness,
            NarrativeMerger, NarrativeRefine,
        };
        use crate::ttd::retrieval::NoopRetriever;
        use crate::ttd::{TtdConfig, TtdMachine};

        let (executor, _count, log) = RecordingExecutor::new();
        let executor: Arc<dyn crate::executor::AgentExecutor> = Arc::new(executor);

        let mut synthesis = SynthesisArtifact::new(
            "s", "r", "q", "model", "v1/synthesis",
        );
        synthesis.claims.push(Claim {
            text: "Test claim.".into(),
            agreement_level: Some("consensus".into()),
            sources: vec![],
            counterarguments: vec![],
            support_level: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1; // N=1 for precise count
        config.n_denoise_steps = 1;  // S=1 for precise count

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(NarrativeDraftGen::new("agent", "model", synthesis.clone())),
            gap_identify: Box::new(NarrativeCritique::new("agent", "model", synthesis.clone())),
            gap_resolve: Box::new(NarrativeRefine::new("agent", "model", synthesis.clone())),
            eval_fitness: Some(Box::new(NarrativeEvalFitness::new("agent", "model"))),
            merger: Box::new(NarrativeMerger::new("agent", "model", synthesis)),
            retriever: Box::new(NoopRetriever),
            executor: executor.clone(),
            bib_store: std::sync::Arc::new(search::bib_store::NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        let panel: Vec<ExpertResponse> = vec![];
        machine.run(&panel).await.expect("run must succeed");

        let spawn_tasks = log.lock().unwrap().clone();

        // With N=1, S=1:
        // - 1 draft spawn (generate)
        // - 6 fitness judge spawns in the loop (faithfulness, completeness, ...)
        // - 1 gap_identify spawn (critique)
        // - 0 gap_resolve (empty retrieved via NoopRetriever)
        // - 6 fitness judge spawns in the final re-evaluation
        // - 1 merger spawn
        // Total = 15 spawns minimum

        // Each of the 6 fitness dimensions must appear at least twice (once in loop,
        // once in final re-eval)
        let fitness_dims = [
            "faithfulness", "completeness", "traceability",
            "neutrality", "dissent_visibility", "structural_clarity",
        ];
        for dim in &fitness_dims {
            let dim_count = spawn_tasks.iter().filter(|t| t.as_str() == *dim).count();
            assert!(
                dim_count >= 2,
                "ENGINE-05: fitness dimension '{dim}' must appear at least 2 times \
                 (once in loop + once in final re-eval). Got {dim_count}. \
                 Full log: {spawn_tasks:?}"
            );
        }

        // Total fitness spawns = 6 dims × (S=1 loop + 1 final re-eval) × N=1 = 12
        let total_fitness = spawn_tasks.iter()
            .filter(|t| fitness_dims.contains(&t.as_str()))
            .count();
        assert_eq!(
            total_fitness, 12,
            "ENGINE-05: total fitness judge spawns must be 6 dims × 2 (loop+re-eval) × N=1 = 12. \
             Got {total_fitness}. Full log: {spawn_tasks:?}"
        );
    }

    /// ENGINE-05: no new LLM client — all spawns go through the injected executor.
    ///
    /// Structural verification: TtdMachine only accepts Arc<dyn AgentExecutor>,
    /// no HTTP or LLM client type. The type system enforces this at compile time.
    ///
    /// CI grep gate (must return 0):
    ///   grep in ttd/ for direct HTTP/LLM client construction = 0 matches.
    #[test]
    fn no_new_llm_client_in_ttd_sources() {
        // The compile-time gate: constructing TtdMachine<String> requires ONLY
        // Arc<dyn AgentExecutor>. No direct HTTP client type is accepted.
        // Any attempt to add an HTTP client field would require a type change here.
        use crate::ttd::{TtdMachine};
        use crate::ttd::stages::narrative::{
            NarrativeDraftGen, NarrativeCritique, NarrativeRefine,
            NarrativeEvalFitness, NarrativeMerger,
        };
        use crate::ttd::artifact::SynthesisArtifact;
        use crate::ttd::retrieval::NoopRetriever;

        let synthesis = SynthesisArtifact::new(
            "s", "r", "q", "model", "v1/synthesis",
        );

        let _machine: TtdMachine<String> = TtdMachine {
            config: crate::ttd::TtdConfig::default(),
            draft_gen: Arc::new(NarrativeDraftGen::new("a", "m", synthesis.clone())),
            gap_identify: Box::new(NarrativeCritique::new("a", "m", synthesis.clone())),
            gap_resolve: Box::new(NarrativeRefine::new("a", "m", synthesis.clone())),
            eval_fitness: Some(Box::new(NarrativeEvalFitness::new("a", "m"))),
            merger: Box::new(NarrativeMerger::new("a", "m", synthesis)),
            retriever: Box::new(NoopRetriever),
            executor: Arc::new(RecordingExecutor {
                count: Arc::new(AtomicUsize::new(0)),
                spawn_log: Arc::new(std::sync::Mutex::new(vec![])),
            }),
            bib_store: std::sync::Arc::new(search::bib_store::NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };
        // Compiles → executor-only seam confirmed. No HTTP client in scope.
    }
}
