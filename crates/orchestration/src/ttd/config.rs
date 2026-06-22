//! `TtdConfig` — consensus TTD engine configuration.
//!
//! Source: `consensus/src/consensus/diffusion/runner.py:67-86` [VERIFIED]
//! Reproduces the defaults exactly. Phase 23 deliberate gap: `randomize_sampling`
//! defaults to `false` (temperature diversity is deferred to Phase 24).

/// Wall-clock guard for the v3 long-form profile (Decision 0 / Phase 0).
///
/// Kvasir's arithmetic (W-e714abb4 gate re-review §3): 55-75 Stage-3 spawns at
/// 90-150s each under long-form output trips the consensus 1800s guard
/// mid-loop, and `run.rs` break-and-keep-best semantics make that a SILENT
/// truncation. 7200s covers the median long-form run; operators may still set
/// `max_stage_seconds` explicitly (an explicit value is never overridden —
/// see `EngineConfig::with_profile`).
pub const V3_MAX_STAGE_SECONDS: u64 = 7200;

/// Configuration for the native TTD engine.
///
/// All fields carry the consensus `TTDConfig` defaults from runner.py:67-86.
/// Field semantics are annotated with the Python field name for traceability.
#[derive(Debug, Clone)]
pub struct TtdConfig {
    /// Prompt/schema dialect. Defaults to V1Delphi (byte-identical to pre-B2 behaviour).
    /// `run.rs` reads this to select the persona set; stage structs read it at
    /// render time to fork v1/v2 prompt paths.
    pub profile: crate::ttd::term_sheet::PromptProfile,
    /// FanOut width — N initial drafts per stage (n_initial_drafts=5).
    pub n_initial_drafts: usize,
    /// Denoise loop depth — S steps per trajectory (n_denoise_steps=2).
    pub n_denoise_steps: usize,
    /// Retrieval top-k per gap query (retrieval_top_k=25).
    pub retrieval_top_k: usize,
    /// Feed prior-step fitness document into gap_identify (use_fitness_feedback=True).
    pub use_fitness_feedback: bool,
    /// Dimensions scoring ≤ this threshold appear in the feedback document (fitness_threshold=3).
    pub fitness_threshold: u8,
    /// Phase 23: false — temperature diversity deferred to Phase 24 (randomize_sampling=True in consensus).
    pub randomize_sampling: bool,
    /// Fixed-cap loop; NOT a score-plateau gate (early_stopping=False).
    pub early_stopping: bool,
    /// Stage 2 uses graph template when `ArgumentationGraph` is present (use_graph_draft=True).
    pub use_graph_draft: bool,
    /// Patch → full-regen → heuristic gap-resolve chain (incremental_resolve=True).
    pub incremental_resolve: bool,
    /// Wall-clock resource guard per stage in seconds (max_stage_seconds=1800).
    pub max_stage_seconds: u64,
    /// LLM call-count budget guard per stage (max_llm_calls=1000).
    pub max_llm_calls: usize,

    // --- Phase 24 EXT-01 additions ---
    /// Master seed for per-trajectory RNG (seed=42 in runner.py:75).
    pub seed: Option<u64>,
    /// Temperature range for per-trajectory sampling (temp_range=(0.5,1.2) runner.py:75).
    pub temp_range: (f32, f32),
    /// top_p range for per-trajectory sampling (top_p_range=(0.8,1.0) runner.py:75).
    pub top_p_range: (f32, f32),
    // --- Phase 24 EXT-02 addition ---
    /// Consensus early-stopping delta threshold (early_stopping_threshold=0.01 runner.py:83).
    /// Only active when early_stopping=true. Missing from Phase 23 (moot since off by default).
    pub early_stopping_threshold: f32,
    /// Clawd-style plateau threshold — OFF by default (None).
    /// Score ≥ threshold OR Δ < 0.15 fires early stop (Phase 25 validates separately).
    pub plateau_threshold: Option<f32>,
    // --- Phase 24 A5 (rung 4) addition ---
    /// Max concurrent eval_fitness.evaluate() calls across all in-flight trajectories.
    ///
    /// Judges are SEQUENTIAL within one evaluate() call (graph.rs:1133,
    /// synthesis.rs:523) — so capping concurrent evaluate() calls caps
    /// concurrent judge sidecar spawns 1:1.
    ///
    /// Default 5: probe-10's draft fan-out already ran N=5 concurrent sidecar
    /// spawns without failure (4.2 min, finding in baseline); per-spawn p95
    /// was 49.6s across 277 spawns — cap 5 bounds peak concurrent fitness
    /// sidecars at the proven fan-out width.
    pub max_concurrent_fitness_evals: usize,

    // --- Rubric-encoding Phase 1 addition (W-e714abb4) ---
    /// Plan-stage operating mode (spec §4 shape (b) + reversal chain).
    ///
    /// `PlanMode::Disabled` (default) makes ZERO plan calls and leaves every
    /// existing prompt byte-identical. `Tournament` runs the 5-draft plan
    /// tournament between Stage 2 and Stage 3 and injects the winning
    /// `ReviewPlan` into Stage-3 drafting/refine/merge/judging.
    /// `SinglePlanner` is the round-1 config fallback (1 draft, same judging).
    ///
    /// NOTE — named deviation from muninn §3: the spec proposed a
    /// `v2.1/lit-review-planned` profile clone; this field achieves the same
    /// byte-stability guarantee (default-off, plan-presence-gated prompt
    /// forks) without cloning the profile surface. Recorded in galdr findings.
    pub plan_mode: crate::ttd::plan::PlanMode,

    // --- Rubric-encoding Phase P (W-522022c5) addition ---
    /// Allow Stage-3 gap resolution (narrative refine) to run even when
    /// retrieval is structurally empty — the C-N2 refine-feedback path.
    ///
    /// Stage 3 uses `NoopRetriever`, so `unique_retrieved` is always empty and
    /// the consensus empty-retrieved guard otherwise makes `narrative_refine`
    /// unreachable (the critique→refine loop that was live in consensus Python).
    /// `false` (default) keeps that guard byte-identical and lets the critique
    /// skip its wasted spawn. `true` re-opens the loop: critique runs, fitness
    /// feedback flows into refine.
    ///
    /// Scoped to Stage 3 — the graph/synthesis builders force this off, because
    /// their retrieval is real and their empty-retrieved guard is genuine
    /// consensus reproduction (Smidr stress map, seam 3).
    pub resolve_without_retrieval: bool,
}

impl Default for TtdConfig {
    fn default() -> Self {
        Self {
            profile: crate::ttd::term_sheet::PromptProfile::V1Delphi, // default: byte-identical
            n_initial_drafts: 5,
            n_denoise_steps: 2,
            retrieval_top_k: 25,
            use_fitness_feedback: true,
            fitness_threshold: 3,
            randomize_sampling: false, // Phase 23 deliberate gap — Phase 24 adds temperature diversity
            early_stopping: false,
            use_graph_draft: true,
            incremental_resolve: true,
            max_stage_seconds: 1800,
            max_llm_calls: 1000,
            // Phase 24 EXT-01/EXT-02 fields — consensus-faithful defaults
            seed: Some(42),
            temp_range: (0.5, 1.2),         // runner.py:75 — NOT config.py SamplingConfig.random() default
            top_p_range: (0.8, 1.0),
            early_stopping_threshold: 0.01, // consensus default runner.py:83
            plateau_threshold: None,        // off by default — Phase 25 validates
            max_concurrent_fitness_evals: 5, // proven fan-out width from probe-10
            plan_mode: crate::ttd::plan::PlanMode::Disabled, // Phase 1 opt-in only
            resolve_without_retrieval: false, // Phase P opt-in only — default byte-stable
        }
    }
}

impl TtdConfig {
    /// Return a config with sampling diversity enabled (EXT-01).
    ///
    /// Use this for Phase 24 EXT-01 tests and production diversity runs.
    /// `TtdConfig::default()` remains the Phase 23 reproduction config
    /// (`randomize_sampling=false`).
    pub fn with_diversity(mut self) -> Self {
        self.randomize_sampling = true;
        self
    }

    /// Return a config with the v2 lit-review profile (B2).
    ///
    /// Selects V2_PERSONAS, v2 render functions, and the v2 schema dialect.
    /// `TtdConfig::default()` remains V1Delphi (byte-identical to pre-B2).
    pub fn with_v2_profile(mut self) -> Self {
        self.profile = crate::ttd::term_sheet::PromptProfile::V2LitReview;
        self
    }

    /// Return a config with the v3 long-form lit-review profile (Decision 0 / Phase 0).
    ///
    /// Selects the v3 profile (v2 prompts/judges/personas with the Stage-3
    /// 300-500-word headerless constraint lifted to sectioned long-form) and
    /// raises `max_stage_seconds` to `V3_MAX_STAGE_SECONDS` — long-form
    /// regeneration trips the consensus 1800s guard silently (kvasir §3).
    /// `TtdConfig::default()` remains V1Delphi; v3 is explicit opt-in only.
    pub fn with_v3_profile(mut self) -> Self {
        self.profile = crate::ttd::term_sheet::PromptProfile::V3LitReviewLong;
        self.max_stage_seconds = V3_MAX_STAGE_SECONDS;
        self
    }

    /// Return a config with the plan tournament enabled (rubric-encoding
    /// Phase 1, W-e714abb4 §4 shape (b)).
    ///
    /// Opt-in only — `TtdConfig::default()` keeps `PlanMode::Disabled` so
    /// every pre-Phase-1 path stays byte-identical.
    pub fn with_plan_tournament(mut self) -> Self {
        self.plan_mode = crate::ttd::plan::PlanMode::Tournament;
        self
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_reproduce_consensus_runner_py_67_86() {
        let cfg = TtdConfig::default();
        assert_eq!(cfg.n_initial_drafts, 5);
        assert_eq!(cfg.n_denoise_steps, 2);
        assert_eq!(cfg.retrieval_top_k, 25);
        assert_eq!(cfg.fitness_threshold, 3);
        assert_eq!(cfg.max_stage_seconds, 1800);
        assert_eq!(cfg.max_llm_calls, 1000);
    }

    #[test]
    fn early_stopping_off_by_default() {
        assert!(!TtdConfig::default().early_stopping);
    }

    #[test]
    fn randomize_sampling_off_by_default_phase23_deliberate_gap() {
        // Phase 23 deliberate fidelity gap: temperature diversity deferred to Phase 24.
        assert!(!TtdConfig::default().randomize_sampling);
    }

    #[test]
    fn use_graph_draft_and_incremental_resolve_on_by_default() {
        let cfg = TtdConfig::default();
        assert!(cfg.use_graph_draft);
        assert!(cfg.incremental_resolve);
    }

    #[test]
    fn use_fitness_feedback_on_by_default() {
        assert!(TtdConfig::default().use_fitness_feedback);
    }

    // ── Phase 24 EXT-01 / EXT-02 tests ────────────────────────────────────────

    #[test]
    fn plateau_off_by_default() {
        assert!(TtdConfig::default().plateau_threshold.is_none());
    }

    #[test]
    fn temp_range_matches_runner_py_75_not_config_py_default() {
        let cfg = TtdConfig::default();
        assert_eq!(cfg.temp_range, (0.5_f32, 1.2_f32)); // runner.py:75, NOT (0.8,1.6)
    }

    #[test]
    fn default_keeps_randomize_sampling_false() {
        // Phase 23 reproduction preserved — diversity is opt-in via with_diversity().
        assert!(!TtdConfig::default().randomize_sampling);
    }

    #[test]
    fn with_diversity_enables_sampling() {
        assert!(TtdConfig::default().with_diversity().randomize_sampling);
    }

    // ── Decision 0 / Phase 0 tests ────────────────────────────────────────────

    /// v3 is opt-in: the default config must never carry it, and selecting it
    /// must raise the wall-clock guard (kvasir §3 silent-truncation arithmetic).
    #[test]
    fn with_v3_profile_opt_in_and_wall_clock_raise() {
        use crate::ttd::term_sheet::PromptProfile;

        // Default untouched: profile AND wall-clock stay consensus-faithful.
        let default_cfg = TtdConfig::default();
        assert_eq!(default_cfg.profile, PromptProfile::V1Delphi);
        assert_eq!(default_cfg.max_stage_seconds, 1800);

        // v2 selection does NOT raise the guard (byte-stable v2 behaviour).
        let v2_cfg = TtdConfig::default().with_v2_profile();
        assert_eq!(v2_cfg.max_stage_seconds, 1800);

        // v3 selection sets profile and raises the guard together.
        let v3_cfg = TtdConfig::default().with_v3_profile();
        assert_eq!(v3_cfg.profile, PromptProfile::V3LitReviewLong);
        assert_eq!(v3_cfg.max_stage_seconds, V3_MAX_STAGE_SECONDS);
        assert!(v3_cfg.max_stage_seconds > 1800, "v3 guard must exceed the consensus default");
    }

    // ── Rubric-encoding Phase 1 tests (W-e714abb4) ────────────────────────────

    /// Byte-stability gate: the plan stage must be OFF unless explicitly
    /// selected; the builder enables exactly the tournament mode.
    #[test]
    fn plan_mode_disabled_by_default_and_tournament_opt_in() {
        use crate::ttd::plan::PlanMode;

        assert_eq!(TtdConfig::default().plan_mode, PlanMode::Disabled);
        // Profile builders never smuggle the plan stage in.
        assert_eq!(TtdConfig::default().with_v2_profile().plan_mode, PlanMode::Disabled);
        assert_eq!(TtdConfig::default().with_v3_profile().plan_mode, PlanMode::Disabled);
        // Explicit opt-in.
        assert_eq!(
            TtdConfig::default().with_plan_tournament().plan_mode,
            PlanMode::Tournament
        );
    }
}
