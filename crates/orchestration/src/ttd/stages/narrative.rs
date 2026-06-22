//! Stage-3 narrative task implementations.
//!
//! Implements the stage-task traits over `String` (the narrative text) for the
//! TTD Stage-3 narrative pipeline. Mirrors
//! `consensus/src/consensus/diffusion/` narrative stage behaviour documented in
//! `ARCHITECTURE.md:102-104` and the `23-RESEARCH.md` Pattern 5.
//!
//! ## How Stage 3 differs from Stages 1 and 2
//!
//! - Artefact type = `String` (the narrative text)
//! - Input = `[synthesis]` in the `responses` slot (NOT the original expert responses)
//! - Gap identification uses `narrative_critique` (hallucinations/omissions/bias vs
//!   the fixed synthesis) — NOT `gap_identify`
//! - Gap resolution uses `narrative_refine` (rewrite the narrative) — NOT `gap_resolve_patch`
//! - **No retrieval** — `NoopRetriever` is injected; gaps are checked against the
//!   fixed synthesis, not the lit store
//! - Merge = `narrative_final_merge` preserving inline `[Cx]` citation markers
//! - Fitness reuses the 6 synthesis fitness-evaluation judges (Assumption A5)
//!   scored with `NARRATIVE_WEIGHTS` and `is_valid_synthesis` (faithfulness ≥ 4)
//!
//! ## Trust boundary (T-23-10)
//!
//! Stage-3 post-processing sanitises hallucinated `[Cx]` citations: inline
//! citation markers are stripped if the claim index does not appear in the
//! synthesis claim set. This prevents a fabricated `[C99]` from being emitted
//! in the final artifact.

use std::sync::Arc;

use async_trait::async_trait;

use crate::adapter::ExpertResponse;
use crate::executor::AgentExecutor;
use crate::ttd::artifact::SynthesisArtifact;
use crate::ttd::config::TtdConfig;
use crate::ttd::fitness::{is_valid_synthesis, is_valid_v2, FitnessEval};
use crate::ttd::mod_types::TtdError;
use crate::ttd::plan::ReviewPlan;
use crate::ttd::stages::{DraftGen, EvalFitness, GapIdentify, GapResolve, Merger, RetrievedContext};
use crate::ttd::state::IdentifiedGap;
use crate::ttd::term_sheet::PromptProfile;
use crate::ttd::weights::{NARRATIVE_WEIGHTS, V2_NARRATIVE_WEIGHTS, V3_PLANNED_NARRATIVE_WEIGHTS};

// ── NarrativeDraftGen ─────────────────────────────────────────────────────────

/// Stage-3 draft generation: produces a 300-500 word narrative from the synthesis.
///
/// Input is the synthesis artifact (passed as the single `ExpertResponse`-shaped
/// slot). Prompt: `narrative/draft.mustache` ported as `render_narrative_draft`.
pub struct NarrativeDraftGen {
    /// Agent ID for narrative draft spawns.
    pub agent_id: String,
    /// Model to use for narrative spawns.
    pub model: String,
    /// The Stage-2 synthesis that seeds the narrative draft.
    /// Stage 3 input = `[synthesis]` (NOT the original expert responses).
    pub synthesis: SynthesisArtifact,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
    /// Winning `ReviewPlan` from the plan tournament (rubric-encoding Phase 1,
    /// W-e714abb4). `None` (default) keeps every prompt byte-identical; `Some`
    /// switches the v2/v3 arm to the `_planned` renderer. v1 never consumes a
    /// plan (the engine only runs the tournament for v2/v3 profiles).
    pub plan: Option<Arc<ReviewPlan>>,
}

impl NarrativeDraftGen {
    pub fn new(
        agent_id: impl Into<String>,
        model: impl Into<String>,
        synthesis: SynthesisArtifact,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            synthesis,
            profile: PromptProfile::V1Delphi,
            plan: None,
        }
    }

    /// Set the prompt/schema profile (consuming builder).
    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Inject the winning plan (consuming builder). `None` is the no-plan
    /// identity — call sites can pass the engine's `Option` straight through.
    pub fn with_plan(mut self, plan: Option<Arc<ReviewPlan>>) -> Self {
        self.plan = plan;
        self
    }
}

#[async_trait]
impl DraftGen<String> for NarrativeDraftGen {
    /// Generate one narrative draft from the synthesis.
    ///
    /// Stage-3 input is `[synthesis]` — the `inputs` slice is ignored.
    /// The synthesis was injected at construction time.
    async fn generate(
        &self,
        _inputs: &[ExpertResponse],
        executor: &Arc<dyn AgentExecutor>,
        _config: &TtdConfig,
        persona_prompt: Option<&str>,
        sampling: Option<crate::executor::SamplingParams>,
    ) -> Result<String, TtdError> {
        use crate::ttd::prompts::narrative::render_narrative_draft;
        use base::identity::AgentId;

        // B2: fork on profile — V2LitReview uses lit-review-framed narrative draft.
        // Decision 0 / Phase 0: v3 shares the v2 renderer; shape is derived from the
        // profile via narrative_shape() (one run yields one shape — total function).
        let base_prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                // Rubric-encoding Phase 1: when a winning plan exists, draft
                // UNDER it. The planned renderer composes the base prompt as
                // an exact prefix — plan-absent output is byte-identical.
                match self.plan.as_deref() {
                    Some(plan) => crate::ttd::prompts::lit_review::render_narrative_draft_v2_planned(
                        &self.synthesis,
                        self.profile.narrative_shape(),
                        plan,
                    ),
                    None => crate::ttd::prompts::lit_review::render_narrative_draft_v2(
                        &self.synthesis,
                        self.profile.narrative_shape(),
                    ),
                }
            }
            PromptProfile::V1Delphi => render_narrative_draft(&self.synthesis),
        };

        // EXT-01 Phase 24: when a persona prompt is supplied, prefix it.
        // None → existing template behaviour (Phase 23 reproduction semantics).
        let prompt = if let Some(persona) = persona_prompt {
            format!("{}\n\n---\n\n{}", persona, base_prompt)
        } else {
            base_prompt
        };

        let agent_id = AgentId::new(self.agent_id.as_str());
        // EXT-01 Phase 24: call execute_with_sampling to thread per-trajectory
        // sampling params. Default impl falls through to execute() when sampling=None.
        executor
            .execute_with_sampling(&agent_id, &prompt, &self.model, "narrative_draft", sampling)
            .await
            .map_err(|e| TtdError::SpawnFailed(e.to_string()))
    }
}

// ── NarrativeCritique (GapIdentify<String>) ───────────────────────────────────

/// Stage-3 gap identification: identifies hallucinations/omissions/bias in one
/// narrative draft by comparing against the fixed synthesis.
///
/// Uses `narrative_critique.mustache` (NOT `gap_identify.mustache` from Stages 1/2).
/// Output format is the same `<gaps>` XML as other stages, so the shared parse
/// logic applies.
///
/// No retrieval is issued from this step — gaps are checked against the fixed
/// synthesis, not the lit store (Stage 3 invariant, RESEARCH Pattern 5).
pub struct NarrativeCritique {
    /// Agent ID for narrative critique spawns.
    pub agent_id: String,
    /// Model to use for critique spawns.
    pub model: String,
    /// The fixed synthesis (reference for gap checking — NOT the lit store).
    pub synthesis: SynthesisArtifact,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
}

impl NarrativeCritique {
    pub fn new(
        agent_id: impl Into<String>,
        model: impl Into<String>,
        synthesis: SynthesisArtifact,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            synthesis,
            profile: PromptProfile::V1Delphi,
        }
    }

    /// Set the prompt/schema profile (consuming builder).
    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }
}

#[async_trait]
impl GapIdentify<String> for NarrativeCritique {
    async fn identify(
        &self,
        draft: &String,
        fitness: &FitnessEval,
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<Vec<IdentifiedGap>, TtdError> {
        use crate::ttd::fitness::generate_feedback;
        use crate::ttd::prompts::narrative::{render_narrative_critique, NarrativeCritiqueInput};
        use base::identity::AgentId;

        // Phase P (Smidr degradation type 2): when refine is unreachable
        // (`resolve_without_retrieval` off → Stage-3 empty-retrieved guard
        // returns the draft unchanged), the critique output is discarded. Skip
        // the spawn — the emitted document is unchanged, and ~1 wasted LLM call
        // per denoise step is removed. (`run.rs` still books +1 for gap_identify
        // unconditionally: a conservative overcount that stops the budget guard
        // earlier, never later.)
        if !config.resolve_without_retrieval {
            return Ok(Vec::new());
        }

        let fitness_feedback = if config.use_fitness_feedback && !fitness.all_none() {
            Some(generate_feedback(fitness, config.fitness_threshold))
        } else {
            None
        };

        // B2: fork on profile — V2LitReview uses extended lit-review critique.
        // v3 = v2 critique unchanged (shape-neutral: critique dims judge content,
        // not length — Decision 0 stress map, grain decision 3).
        let prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                // C-N2: feed fitness into the critique too (was dropped — Kvasir
                // F1). Only reachable flag-on, so the default path is unaffected.
                crate::ttd::prompts::lit_review::render_narrative_critique_v2(
                    draft,
                    &self.synthesis,
                    fitness_feedback.as_deref(),
                )
            }
            PromptProfile::V1Delphi => render_narrative_critique(&NarrativeCritiqueInput {
                synthesis: &self.synthesis,
                narrative: draft,
                fitness_feedback: fitness_feedback.as_deref(),
            }),
        };

        let agent_id = AgentId::new(self.agent_id.as_str());
        let output = executor
            .execute(&agent_id, &prompt, &self.model, "narrative_critique")
            .await
            .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

        parse_gaps_xml(&output)
    }
}

// ── NarrativeRefine (GapResolve<String>) ────────────────────────────────────

/// Stage-3 gap resolution: rewrites the narrative to address critique findings.
///
/// Uses `narrative_refine.mustache` (NOT `gap_resolve_patch.mustache`).
/// Skips retrieval entirely — Stage 3 gaps are checked against the fixed
/// synthesis, not the lit store. The `retrieved` slice is IGNORED (the
/// `NoopRetriever` always passes an empty slice anyway, and this resolver
/// does not use it even if it were non-empty).
///
/// Contra Stages 1 and 2, there is no patch/full-regen/heuristic fallback
/// chain — the rewrite is always a full regeneration against the critique.
pub struct NarrativeRefine {
    /// Agent ID for narrative refine spawns.
    pub agent_id: String,
    /// Model to use for refine spawns.
    pub model: String,
    /// The fixed synthesis (reference for the rewrite).
    pub synthesis: SynthesisArtifact,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
    /// Winning `ReviewPlan` (rubric-encoding Phase 1). `None` (default) keeps
    /// the refine prompt byte-identical; `Some` re-asserts the plan so the
    /// critique never licenses plan departure.
    pub plan: Option<Arc<ReviewPlan>>,
}

impl NarrativeRefine {
    pub fn new(
        agent_id: impl Into<String>,
        model: impl Into<String>,
        synthesis: SynthesisArtifact,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            synthesis,
            profile: PromptProfile::V1Delphi,
            plan: None,
        }
    }

    /// Set the prompt/schema profile (consuming builder).
    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Inject the winning plan (consuming builder). `None` is the no-plan identity.
    pub fn with_plan(mut self, plan: Option<Arc<ReviewPlan>>) -> Self {
        self.plan = plan;
        self
    }
}

#[async_trait]
impl GapResolve<String> for NarrativeRefine {
    /// Rewrite the narrative to address critique findings.
    ///
    /// `retrieved` is ignored — Stage 3 has no retrieval step.
    async fn resolve(
        &self,
        draft: &String,
        fitness: &FitnessEval,
        gaps: &[IdentifiedGap],
        _retrieved: &[RetrievedContext],
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<String, TtdError> {
        use crate::ttd::fitness::generate_feedback;
        use crate::ttd::prompts::narrative::{render_narrative_refine, NarrativeRefineInput};
        use base::identity::AgentId;

        // C-N2: embed the candidate's low-scoring fitness dimensions as feedback
        // so the rewrite targets them. Mirrors the critique feedback grain.
        // Only reachable when `resolve_without_retrieval` re-opens the refine
        // loop (flag off → refine unreachable → this path never runs).
        let fitness_feedback = if config.use_fitness_feedback && !fitness.all_none() {
            Some(generate_feedback(fitness, config.fitness_threshold))
        } else {
            None
        };

        // Combine gap descriptions as the critique text.
        let critique: String = gaps
            .iter()
            .map(|g| format!("- {}", g.description))
            .collect::<Vec<_>>()
            .join("\n");

        // B2: fork on profile — V2LitReview uses re-weave framed refine prompt.
        // Decision 0: refine is constraint site 2 of 3 — shape travels with the
        // profile, or refine silently squeezes long-form drafts back to 500 words
        // (muninn S2b warning).
        let prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                // Rubric-encoding Phase 1: plan presence forks to the planned
                // renderer (base prompt is its exact prefix — byte-stable).
                match self.plan.as_deref() {
                    Some(plan) => crate::ttd::prompts::lit_review::render_narrative_refine_v2_planned(
                        draft,
                        &critique,
                        self.profile.narrative_shape(),
                        plan,
                        fitness_feedback.as_deref(),
                    ),
                    None => crate::ttd::prompts::lit_review::render_narrative_refine_v2(
                        draft,
                        &critique,
                        self.profile.narrative_shape(),
                        fitness_feedback.as_deref(),
                    ),
                }
            }
            PromptProfile::V1Delphi => render_narrative_refine(&NarrativeRefineInput {
                synthesis: &self.synthesis,
                narrative: draft,
                critique: &critique,
                // No retrieved content for Stage 3 (NoopRetriever invariant).
                retrieved: &[],
                fitness_feedback: fitness_feedback.as_deref(),
            }),
        };

        let agent_id = AgentId::new(self.agent_id.as_str());
        executor
            .execute(&agent_id, &prompt, &self.model, "narrative_refine")
            .await
            .map_err(|e| TtdError::SpawnFailed(e.to_string()))
    }
}

// ── NarrativeMerger ───────────────────────────────────────────────────────────

/// Stage-3 merge: fuses candidate narratives preserving inline `[Cx]` citation
/// markers.
///
/// Uses `narrative_final_merge.mustache`.
///
/// ## Citation preservation (T-23-10 mitigation)
///
/// The merge prompt EXPLICITLY instructs the model to preserve all `[Cx]` markers.
/// After merge, `sanitise_cx_citations` strips hallucinated `[Cx]` markers whose
/// index does not appear in the synthesis claim set. Valid markers are kept.
pub struct NarrativeMerger {
    /// Agent ID for narrative merge spawns.
    pub agent_id: String,
    /// Model to use for merge spawns.
    pub model: String,
    /// The fixed synthesis (used for citation sanitisation post-merge).
    pub synthesis: SynthesisArtifact,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
    /// Winning `ReviewPlan` from the plan tournament (rubric-encoding Phase 1,
    /// W-e714abb4). `None` (default) keeps every existing path byte-identical.
    /// When set with a sectioned plan under the long-form shape, merging runs
    /// section-by-section per C-N3; otherwise a whole-document planned merge.
    pub plan: Option<Arc<ReviewPlan>>,
}

impl NarrativeMerger {
    pub fn new(
        agent_id: impl Into<String>,
        model: impl Into<String>,
        synthesis: SynthesisArtifact,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            synthesis,
            profile: PromptProfile::V1Delphi,
            plan: None,
        }
    }

    /// Set the prompt/schema profile (consuming builder).
    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set the winning review plan (consuming builder). `None` is a no-op.
    pub fn with_plan(mut self, plan: Option<Arc<ReviewPlan>>) -> Self {
        self.plan = plan;
        self
    }

    /// Planned merge (rubric-encoding Phase 1, C-N3).
    ///
    /// Section-by-section sequential merge when the plan declares sections,
    /// the profile shape is `SectionedLongForm`, and ≥2 candidates carry each
    /// plan heading (a single carrier leaves nothing to merge). Otherwise a
    /// single whole-document planned merge (`render_narrative_final_merge_v2_planned`).
    ///
    /// ## Vetoed-trajectory policy (C-N3)
    ///
    /// Candidates arrive pre-sorted best-first (run.rs sorts via
    /// `sort_candidates_best_first` before the `Merger` seam; validity flags
    /// are not visible here). ALL candidates enter fan-in with rank
    /// annotations (rank = index + 1); the section-merge prompt instructs the
    /// model to anchor each section's STRUCTURE on the rank-1 candidate.
    ///
    /// ## Budget note
    ///
    /// Section-merge spawns `plan.sections.len()` calls instead of 1. Merge
    /// calls are NOT booked in `state.llm_calls` (pre-existing run.rs
    /// pattern — only fitness/gap calls are booked), so accounting is
    /// unchanged either way.
    async fn merge_planned(
        &self,
        plan: &ReviewPlan,
        candidates: &[&str],
        executor: &Arc<dyn AgentExecutor>,
        agent_id: &base::identity::AgentId,
    ) -> Result<String, TtdError> {
        use crate::ttd::plan::split_sections;
        use crate::ttd::prompts::lit_review::{
            render_narrative_final_merge_v2_planned, render_section_merge_v3,
        };
        use crate::ttd::term_sheet::NarrativeShape;

        // C-N3 sectionability gate.
        let candidate_sections: Vec<Vec<(String, String)>> =
            candidates.iter().map(|c| split_sections(c)).collect();
        let sectionable = self.profile.narrative_shape() == NarrativeShape::SectionedLongForm
            && !plan.sections.is_empty()
            && plan.sections.iter().all(|section| {
                candidate_sections
                    .iter()
                    .filter(|secs| secs.iter().any(|(h, _)| h == &section.heading))
                    .count()
                    >= 2
            });

        if !sectionable {
            tracing::warn!(
                target: "ttd_plan",
                "planned merge: candidates not sectionable against the plan — \
                 falling back to whole-document planned merge"
            );
            let prompt = render_narrative_final_merge_v2_planned(
                candidates,
                self.profile.narrative_shape(),
                plan,
            );
            return executor
                .execute(agent_id, &prompt, &self.model, "narrative_final_merge")
                .await
                .map_err(|e| TtdError::SpawnFailed(e.to_string()));
        }

        // Sequential section merge: each section sees the previous section's
        // tail paragraph (C-N3 continuity requirement).
        let mut merged_sections: Vec<String> = Vec::with_capacity(plan.sections.len());
        let mut prev_tail: Option<String> = None;

        for section in &plan.sections {
            let ranked: Vec<(usize, &str)> = candidate_sections
                .iter()
                .enumerate()
                .filter_map(|(i, secs)| {
                    secs.iter()
                        .find(|(h, _)| h == &section.heading)
                        .map(|(_, body)| (i + 1, body.as_str()))
                })
                .collect();

            let prompt = render_section_merge_v3(plan, section, &ranked, prev_tail.as_deref());
            let response = executor
                .execute(agent_id, &prompt, &self.model, "narrative_section_merge")
                .await
                .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

            // Guarantee the heading line survives even if the model dropped it.
            let section_text = if response.trim_start().starts_with("## ") {
                response.trim().to_string()
            } else {
                format!("## {}\n\n{}", section.heading, response.trim())
            };

            prev_tail = last_paragraph(&section_text);
            merged_sections.push(section_text);
        }

        Ok(merged_sections.join("\n\n"))
    }
}

#[async_trait]
impl Merger<String> for NarrativeMerger {
    async fn merge(
        &self,
        candidates: &[String],
        executor: &Arc<dyn AgentExecutor>,
        _config: &TtdConfig,
    ) -> Result<String, TtdError> {
        use crate::ttd::prompts::narrative::{render_narrative_final_merge, NarrativeFinalMergeInput};
        use base::identity::AgentId;

        let agent_id = AgentId::new(self.agent_id.as_str());

        // B2: fork on profile — V2LitReview uses anti-formulaic merge prompt.
        // Decision 0: merge is constraint site 3 of 3 — same S2b shape-travel rule.
        // Rubric-encoding Phase 1: plan-presence fork — `plan: None` keeps the
        // v2/v3 path byte-identical to the pre-plan behaviour.
        let merged = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                let candidate_strs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
                match self.plan.clone() {
                    Some(plan) => {
                        self.merge_planned(&plan, &candidate_strs, executor, &agent_id)
                            .await?
                    }
                    None => {
                        let prompt = crate::ttd::prompts::lit_review::render_narrative_final_merge_v2(
                            &candidate_strs,
                            self.profile.narrative_shape(),
                        );
                        executor
                            .execute(&agent_id, &prompt, &self.model, "narrative_final_merge")
                            .await
                            .map_err(|e| TtdError::SpawnFailed(e.to_string()))?
                    }
                }
            }
            PromptProfile::V1Delphi => {
                let prompt = render_narrative_final_merge(&NarrativeFinalMergeInput {
                    synthesis: &self.synthesis,
                    narratives: candidates,
                });
                executor
                    .execute(&agent_id, &prompt, &self.model, "narrative_final_merge")
                    .await
                    .map_err(|e| TtdError::SpawnFailed(e.to_string()))?
            }
        };

        // Post-merge mechanical scans (rubric-encoding Phase 1): term-registry
        // drift, banned phrases, section-budget ratios, planted-thread
        // callbacks — the deterministic pair of the prompt's merge rules.
        // Feedback tier ONLY: findings are logged, never written into the
        // text. The single mutating post-step remains sanitise_cx_citations.
        if let Some(plan) = self.plan.as_deref() {
            for finding in crate::ttd::plan::run_plan_document_lints(&merged, plan) {
                tracing::warn!(target: "ttd_plan", finding = %finding, "post-merge plan lint");
            }
        }

        // Post-merge: sanitise hallucinated [Cx] citation markers.
        // Valid claim IDs come from the synthesis's claim set.
        // T-23-10: injected fake citation markers are stripped, not emitted.
        let valid_claim_ids: Vec<String> = self
            .synthesis
            .claims
            .iter()
            .enumerate()
            .map(|(i, _)| format!("C{}", i + 1))
            .collect();

        Ok(sanitise_cx_citations(&merged, &valid_claim_ids))
    }
}

// ── NarrativeEvalFitness ─────────────────────────────────────────────────────

/// Stage-3 fitness evaluation.
///
/// Profile fork:
/// - `V1Delphi` (default): reuses the 6 synthesis fitness judges with `NARRATIVE_WEIGHTS`
///   and `is_valid_synthesis`. Source: Assumption A5 (no narrative-specific judge templates).
/// - `V2LitReview`: 5 v2 lit-review dims via `render_fitness_judge_v2_narrative`; `is_valid_v2`;
///   `V2_NARRATIVE_WEIGHTS`. NEVER sets a veto — the narrative pseudo-artifact wrapper has
///   `sources: vec![]` BY CONSTRUCTION (narrative.rs:455), so a synthesis veto would fail
///   every narrative candidate. Narrative is explicitly exempt from the traceability veto.
pub struct NarrativeEvalFitness {
    /// Agent ID for fitness judge spawns.
    pub agent_id: String,
    /// Model to use for fitness judges.
    pub model: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
    /// Winning `ReviewPlan` from the plan tournament (rubric-encoding Phase 1,
    /// W-e714abb4). When set under v2/v3 profiles, a sixth judge dimension
    /// (`plan_conformance` — draft-against-plan, NOT draft-against-taste) is
    /// appended and `weights()` switches to `V3_PLANNED_NARRATIVE_WEIGHTS`,
    /// which keeps run.rs's `weights().len()`-derived call accounting correct.
    /// `None` (default) keeps every existing path byte-identical.
    pub plan: Option<Arc<ReviewPlan>>,
}

impl NarrativeEvalFitness {
    pub fn new(agent_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            profile: PromptProfile::V1Delphi, // default: backward compat
            plan: None,
        }
    }

    /// Set the prompt/schema profile (consuming builder).
    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set the winning review plan (consuming builder). `None` is a no-op.
    pub fn with_plan(mut self, plan: Option<Arc<ReviewPlan>>) -> Self {
        self.plan = plan;
        self
    }
}

#[async_trait]
impl EvalFitness<String> for NarrativeEvalFitness {
    /// Evaluate the narrative draft.
    ///
    /// **V1Delphi (default):** reuses the 6 synthesis fitness dimensions via
    /// NARRATIVE_WEIGHTS and `is_valid_synthesis`. Source: Assumption A5.
    ///
    /// **V2LitReview:** iterates V2_NARRATIVE_JUDGE_DIMS (5 dims — shared
    /// names/definitions, narrative-scoped anchors) using
    /// `render_fitness_judge_v2_narrative` against the raw narrative text.
    /// NEVER sets a veto — the pseudo-artifact has `sources: vec![]` by
    /// construction (see Claim push below), so traceability_veto_synthesis
    /// would fail every narrative candidate. Narrative is explicitly exempt
    /// from the traceability veto gate; only graph/synthesis bear that gate.
    ///
    /// Each judge spawn is routed through `executor` (ENGINE-05).
    async fn evaluate(
        &self,
        draft: &String,
        executor: &Arc<dyn AgentExecutor>,
        _config: &TtdConfig,
    ) -> Result<FitnessEval, TtdError> {
        use base::identity::AgentId;

        let agent_id = AgentId::new(self.agent_id.as_str());

        match self.profile {
            // Decision 0: v3 scores with the v2 judges unchanged (task exclusion —
            // "no judge changes"; v2 narrative dims judge content properties, not length).
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                use crate::ttd::prompts::lit_review::render_fitness_judge_v2_narrative;
                use crate::ttd::term_sheet::V2_NARRATIVE_JUDGE_DIMS;

                let mut scores: Vec<(String, Option<u8>)> =
                    Vec::with_capacity(V2_NARRATIVE_JUDGE_DIMS.len());

                // WR-05: degrade a failed judge spawn to `None`.
                // Narrative is exempt from traceability veto — see doc comment above.
                // Dims carry narrative-scoped anchors (shared names/definitions); see
                // `.planning/JUDGE-CALIBRATION-PLAN.md` (2026-06-19).
                for dim in V2_NARRATIVE_JUDGE_DIMS.iter() {
                    let prompt = render_fitness_judge_v2_narrative(dim, draft);
                    let score =
                        match executor.execute(&agent_id, &prompt, &self.model, dim.name).await {
                            Ok(output) => parse_fitness_score(&output),
                            Err(e) => {
                                tracing::debug!(
                                    dimension = dim.name,
                                    error = %e,
                                    "NarrativeEvalFitness(V2): judge spawn failed — score=None"
                                );
                                None
                            }
                        };
                    scores.push((dim.name.to_string(), score));
                }

                // Rubric-encoding Phase 1: sixth judge — plan conformance.
                // Verifies draft-against-plan (archetype, sections, registry,
                // threads), NOT draft-against-taste. Only spawned when a plan
                // is present; weights() switches to the 6-dim planned table in
                // lockstep, so run.rs's weights().len()-derived call booking
                // stays correct.
                if let Some(plan) = self.plan.as_deref() {
                    use crate::ttd::plan::render_plan_conformance_judge;

                    let prompt = render_plan_conformance_judge(plan, draft);
                    let score = match executor
                        .execute(&agent_id, &prompt, &self.model, "plan_conformance")
                        .await
                    {
                        Ok(output) => parse_fitness_score(&output),
                        Err(e) => {
                            tracing::debug!(
                                dimension = "plan_conformance",
                                error = %e,
                                "NarrativeEvalFitness(planned): judge spawn failed — score=None"
                            );
                            None
                        }
                    };
                    scores.push(("plan_conformance".to_string(), score));
                }

                // No veto — narrative pseudo-artifact has empty sources by construction.
                Ok(FitnessEval::new(scores))
            }

            PromptProfile::V1Delphi => {
                use crate::ttd::artifact::Claim;
                use crate::ttd::prompts::synthesis::{
                    render_synthesis_fitness_completeness,
                    render_synthesis_fitness_dissent_visibility,
                    render_synthesis_fitness_faithfulness,
                    render_synthesis_fitness_neutrality,
                    render_synthesis_fitness_structural_clarity,
                    render_synthesis_fitness_traceability,
                    SynthesisFitnessInput,
                };

                // Build a minimal SynthesisArtifact wrapping the narrative text so the
                // synthesis fitness judges (which take &SynthesisArtifact) can evaluate
                // it.  The narrative text becomes a single claim — the judges evaluate
                // alignment with synthesis dimensions on this text.
                //
                // This reuses the 6 synthesis judge templates with NARRATIVE_WEIGHTS per
                // Assumption A5: no narrative-specific judge templates exist.
                //
                // Each of the 6 dimensions is a separate governed spawn (ENGINE-05).
                let mut pseudo_synthesis = SynthesisArtifact::new(
                    "narrative-eval",
                    "r1",
                    "q1",
                    &self.model,
                    "v1/narrative",
                );
                pseudo_synthesis.claims.push(Claim {
                    text: draft.clone(),
                    agreement_level: Some("majority".into()),
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

                let fitness_input = SynthesisFitnessInput {
                    draft: &pseudo_synthesis,
                };

                let dims = [
                    (
                        "faithfulness",
                        render_synthesis_fitness_faithfulness(&fitness_input),
                    ),
                    (
                        "completeness",
                        render_synthesis_fitness_completeness(&fitness_input),
                    ),
                    (
                        "traceability",
                        render_synthesis_fitness_traceability(&fitness_input),
                    ),
                    (
                        "neutrality",
                        render_synthesis_fitness_neutrality(&fitness_input),
                    ),
                    (
                        "dissent_visibility",
                        render_synthesis_fitness_dissent_visibility(&fitness_input),
                    ),
                    (
                        "structural_clarity",
                        render_synthesis_fitness_structural_clarity(&fitness_input),
                    ),
                ];

                let mut scores: Vec<(String, Option<u8>)> = Vec::with_capacity(6);

                // WR-05: degrade a failed judge spawn to `None` rather than `?`-aborting
                // the run (matches consensus's graceful per-dimension degradation and the
                // synthesis/graph evaluators). Every dimension issues exactly one spawn,
                // so the run loop's fixed `+6` reflects the real spawn count.
                for (dim_name, prompt) in dims {
                    let score =
                        match executor.execute(&agent_id, &prompt, &self.model, dim_name).await {
                            Ok(output) => parse_fitness_score(&output),
                            Err(e) => {
                                tracing::debug!(
                                    dimension = dim_name,
                                    error = %e,
                                    "NarrativeEvalFitness: judge spawn failed — score=None (parse failure)"
                                );
                                None
                            }
                        };
                    scores.push((dim_name.to_string(), score));
                }

                Ok(FitnessEval::new(scores))
            }
        }
    }

    fn validity_fn(&self) -> fn(&FitnessEval) -> bool {
        match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => is_valid_v2,
            PromptProfile::V1Delphi => is_valid_synthesis, // faithfulness ≥ 4
        }
    }

    fn weights(&self) -> &'static [(&'static str, f32)] {
        match self.profile {
            // Rubric-encoding Phase 1: plan present ⇒ 6-dim planned table
            // (plan_conformance leads). Must stay in lockstep with the
            // conditional sixth judge in evaluate() — run.rs derives the
            // fitness call count from weights().len().
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                if self.plan.is_some() {
                    V3_PLANNED_NARRATIVE_WEIGHTS
                } else {
                    V2_NARRATIVE_WEIGHTS
                }
            }
            PromptProfile::V1Delphi => NARRATIVE_WEIGHTS,
        }
    }
}

// ── Parse helpers ─────────────────────────────────────────────────────────────

/// Parse `<gaps>` XML output from narrative_critique into `IdentifiedGap` list.
///
/// T1 ruled contract (shared with synthesis.rs / graph.rs): missing or empty
/// `<gaps>` block → `Ok(vec![])` (never `Err`, never a bare `Vec`). A gap is
/// valid iff it has a non-empty `<description>`; `<query>` defaults to the
/// description when absent. Uses the canonical quick_xml event parser so the
/// three stages share one parse behaviour.
fn parse_gaps_xml(xml: &str) -> Result<Vec<IdentifiedGap>, TtdError> {
    let xml_block = match extract_xml_block(xml, "gaps") {
        Some(block) => block,
        None => return Ok(Vec::new()),
    };

    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(&xml_block);
    reader.trim_text(true);

    let mut gaps = Vec::new();
    let mut buf = Vec::new();
    let mut in_description = false;
    let mut in_query = false;
    let mut description = String::new();
    let mut query = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "gap" => {
                        description.clear();
                        query.clear();
                    }
                    "description" => { in_description = true; }
                    "query" => { in_query = true; }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                let text = e.unescape().unwrap_or_default().to_string();
                if in_description {
                    description.push_str(&text);
                } else if in_query {
                    query.push_str(&text);
                }
            }
            Ok(Event::End(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "description" => { in_description = false; }
                    "query" => { in_query = false; }
                    "gap" => {
                        if !description.is_empty() {
                            gaps.push(IdentifiedGap {
                                description: description.trim().to_string(),
                                query: if query.is_empty() {
                                    description.trim().to_string()
                                } else {
                                    query.trim().to_string()
                                },
                            });
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    Ok(gaps)
}

/// Parse a fitness score (1–5) from a fitness-judge LLM response.
///
/// WR-08: delegates to the single canonical parser
/// `fitness::parse_fitness_response`. This gives the narrative parser the
/// integer CLAMP to `[1,5]` (CR-03) and the empty/whitespace → `None` abstain
/// (WR-03) it was missing. IN-01: attributed/garbled `<score unit="x">` tags are
/// not matched by the exact-tag path; they fall back to the regex tier. One
/// source of truth across all three stages.
fn parse_fitness_score(output: &str) -> Option<u8> {
    crate::ttd::fitness::parse_fitness_response(output).score
}

/// Extract the whole `<{tag}...>...</{tag}>` block from the given string.
///
/// T1 ruled contract (shared with synthesis.rs / graph.rs): returns the WHOLE
/// tagged block (open + body + close) using prefix-open (`<{tag}` — matches
/// `<tag>` and attributed `<tag attr=..>`) / first-close matching. Returns
/// `None` if the open or its first close is absent.
fn extract_xml_block(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let start = text.find(&open)?;
    let end = text[start..].find(&close).map(|i| start + i + close.len())?;
    Some(text[start..end].to_string())
}

/// Last non-empty paragraph of a text block (split on blank lines).
///
/// Used by the sequential section merge (C-N3) to hand the previous section's
/// tail to the next section's prompt for continuity. Returns `None` for
/// effectively-empty input.
fn last_paragraph(text: &str) -> Option<String> {
    text.split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .last()
        .map(str::to_string)
}

// ── Citation sanitisation ──────────────────────────────────────────────────────

/// Strip hallucinated `[Cx]` citation markers from narrative text.
///
/// Keeps markers whose index exists in `valid_claim_ids` (e.g. `["C1", "C2"]`).
/// Removes markers that reference claim indices not present in the synthesis.
///
/// T-23-10 mitigation: a fake injected `[C99]` is stripped rather than emitted.
///
/// ## Example
///
/// ```ignore
/// let valid = vec!["C1".to_string(), "C2".to_string()];
/// let text = "Some statement [C1, C3] and another [C2].";
/// let result = sanitise_cx_citations(text, &valid);
/// // "[C3]" is stripped; "[C1]" and "[C2]" are kept.
/// assert!(result.contains("[C1]") || result.contains("[C2]") || !result.contains("[C3]"));
/// ```
pub fn sanitise_cx_citations(text: &str, valid_claim_ids: &[String]) -> String {
    // Pattern: `[C1]`, `[C1, C2]`, `[C1, C3, C5]` etc.
    // Strategy: find all [Cx...] groups and filter out invalid IDs.
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' {
            // Collect until ']' or end of string
            let mut bracket_content = String::new();
            let mut found_close = false;

            while let Some(&next) = chars.peek() {
                chars.next();
                if next == ']' {
                    found_close = true;
                    break;
                }
                bracket_content.push(next);
            }

            if found_close {
                // Parse as comma-separated list of claim IDs
                let ids: Vec<&str> = bracket_content
                    .split(',')
                    .map(|s| s.trim())
                    .collect();

                // Determine which look like Cx markers
                let all_cx = ids.iter().all(|id| {
                    id.starts_with('C')
                        && id[1..].parse::<usize>().is_ok()
                });

                if all_cx {
                    // Keep only valid IDs
                    let valid_ids: Vec<&str> = ids
                        .iter()
                        .filter(|id| valid_claim_ids.iter().any(|v| v == *id))
                        .copied()
                        .collect();

                    if !valid_ids.is_empty() {
                        result.push('[');
                        result.push_str(&valid_ids.join(", "));
                        result.push(']');
                    }
                    // else: all IDs were hallucinated — drop the entire bracket
                } else {
                    // Not a Cx citation marker — preserve as-is
                    result.push('[');
                    result.push_str(&bracket_content);
                    result.push(']');
                }
            } else {
                // Unclosed bracket — preserve as-is
                result.push('[');
                result.push_str(&bracket_content);
            }
        } else {
            result.push(c);
        }
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;

    use super::*;
    use crate::adapter::{ExpertResponse, ResponseProvenance, SourceId};
    use crate::ttd::artifact::{Claim, SynthesisArtifact};
    use crate::ttd::fitness::FitnessEval;
    use crate::ttd::mod_types::TtdError;
    use crate::ttd::state::IdentifiedGap;
    use crate::ttd::weights::NARRATIVE_WEIGHTS;
    use crate::ttd::{TtdConfig, TtdMachine};

    // ── Mock helpers ──────────────────────────────────────────────────────────

    struct EchoExecutor;

    #[async_trait]
    impl crate::executor::AgentExecutor for EchoExecutor {
        async fn execute(
            &self,
            _agent_id: &base::identity::AgentId,
            _instruction: &str,
            _model: &str,
            task: &str,
        ) -> base::AlzinaResult<String> {
            // For fitness judges return a valid score XML
            if task.contains("faithfulness")
                || task.contains("completeness")
                || task.contains("traceability")
                || task.contains("neutrality")
                || task.contains("dissent_visibility")
                || task.contains("structural_clarity")
            {
                return Ok("<fitness_evaluation><score>4</score><rationale>good</rationale></fitness_evaluation>".to_string());
            }
            // For gap critique return empty gaps (fast path — no denoise changes)
            if task == "narrative_critique" {
                return Ok("<gaps></gaps>".to_string());
            }
            // For draft, refine, and merge — return a simple narrative
            Ok("Experts broadly agree that climate change accelerates permafrost thaw [C1]. There is ongoing debate about the rate of methane release [C2].".to_string())
        }
    }

    /// Build a minimal SynthesisArtifact for testing.
    fn stub_synthesis() -> SynthesisArtifact {
        let mut s = SynthesisArtifact::new(
            "study-test",
            "round-1",
            "q-climate",
            "google/gemini-2.5-flash",
            "v1/synthesis",
        );
        s.claims.push(Claim {
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
        s.claims.push(Claim {
            text: "The rate of methane release is uncertain.".into(),
            agreement_level: Some("divided".into()),
            sources: vec!["arxiv:2105.14103".into(), "s2:abc123".into()],
            counterarguments: vec!["Some models predict low rates.".into()],
            support_level: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        s
    }

    fn stub_panel() -> Vec<ExpertResponse> {
        vec![ExpertResponse {
            expert_id: SourceId::new("arxiv:2105.14103"),
            prose: "Permafrost thaw is accelerating under warming conditions.".into(),
            provenance: ResponseProvenance {
                source_id: SourceId::new("arxiv:2105.14103"),
                title: "Permafrost Study".into(),
                year: Some(2021),
                authors: vec![],
                credibility_tier: search::CredibilityTier::Unknown,
            },
        }]
    }

    // ── Test: narrative_uses_noop_retriever ───────────────────────────────────

    /// Stage 3 issues no retrieval queries (NoopRetriever; gaps checked vs the
    /// fixed synthesis).
    ///
    /// We verify by using a RecordingRetriever that panics if retrieve() is called.
    #[tokio::test]
    async fn narrative_uses_noop_retriever() {
        use crate::ttd::stages::RetrievedContext;

        struct PanicRetriever;

        #[async_trait]
        impl crate::ttd::retrieval::Retriever for PanicRetriever {
            async fn retrieve(
                &self,
                _query: &str,
                _top_k: usize,
            ) -> Result<Vec<RetrievedContext>, TtdError> {
                panic!("Stage 3 must not call retrieve() — NoopRetriever invariant violated")
            }
        }

        let synthesis = stub_synthesis();
        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(EchoExecutor);

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1;
        config.n_denoise_steps = 1;

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(NarrativeDraftGen::new(
                "narrative-agent", "google/gemini-2.5-flash", synthesis.clone(),
            )),
            gap_identify: Box::new(NarrativeCritique::new(
                "critique-agent", "google/gemini-2.5-flash", synthesis.clone(),
            )),
            gap_resolve: Box::new(NarrativeRefine::new(
                "refine-agent", "google/gemini-2.5-flash", synthesis.clone(),
            )),
            eval_fitness: Some(Box::new(NarrativeEvalFitness::new(
                "fitness-agent", "google/gemini-2.5-flash",
            ))),
            merger: Box::new(NarrativeMerger::new(
                "merge-agent", "google/gemini-2.5-flash", synthesis,
            )),
            retriever: Box::new(PanicRetriever), // must NEVER be called for Stage 3
            executor,
            bib_store: std::sync::Arc::new(search::bib_store::NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        // The test passes only if PanicRetriever is never called.
        // The EchoExecutor returns empty gaps, so gap_resolve is never invoked
        // (empty-retrieved guard). But even if gaps were returned, NarrativeRefine
        // ignores the retrieved slice — it does not call retriever.retrieve().
        let result = machine.run(&stub_panel()).await;
        assert!(
            result.is_ok(),
            "Stage 3 TtdMachine run must succeed without retrieval: {result:?}"
        );
    }

    // ── Test: narrative_input_is_synthesis ───────────────────────────────────

    /// Stage-3 input slot is [synthesis], not the original expert responses.
    ///
    /// We verify that NarrativeDraftGen uses `self.synthesis` and ignores `inputs`.
    /// The draft prompt must reference synthesis claims, not expert response prose.
    #[tokio::test]
    async fn narrative_input_is_synthesis() {
        use crate::ttd::prompts::narrative::render_narrative_draft;

        let synthesis = stub_synthesis();
        let prompt = render_narrative_draft(&synthesis);

        // The prompt must reference the synthesis claims.
        assert!(
            prompt.contains("Climate change accelerates permafrost thaw"),
            "narrative draft prompt must reference synthesis claims, not expert responses. \
             Got prompt start: {}",
            &prompt[..200.min(prompt.len())]
        );

        // The prompt must NOT reference raw expert response prose as the primary source.
        // (The expert prose is in the synthesis indirectly, but the direct input is
        // the synthesis structure.)
        assert!(
            !prompt.contains("expert_response"),
            "narrative draft prompt must NOT include raw expert_response XML tags"
        );
    }

    // ── Test: final_merge_preserves_cx_markers ───────────────────────────────

    /// `narrative_final_merge` keeps inline `[Cx]` citation markers intact.
    ///
    /// We verify via the merge prompt text and the sanitise_cx_citations function:
    /// - Valid markers (claim IDs present in synthesis) are preserved.
    /// - Hallucinated markers (claim IDs not in synthesis) are stripped.
    #[test]
    fn final_merge_preserves_cx_markers() {
        let valid_ids = vec!["C1".to_string(), "C2".to_string()];

        // Valid citation: must be preserved
        let text = "Climate change is accelerating [C1] with uncertain feedback [C2].";
        let result = sanitise_cx_citations(text, &valid_ids);
        assert!(
            result.contains("[C1]"),
            "valid citation [C1] must be preserved: {result}"
        );
        assert!(
            result.contains("[C2]"),
            "valid citation [C2] must be preserved: {result}"
        );

        // Hallucinated citation: must be stripped (T-23-10)
        let text_with_hallucination = "Some claim [C1, C99] with invented reference.";
        let result = sanitise_cx_citations(text_with_hallucination, &valid_ids);
        assert!(
            !result.contains("C99"),
            "hallucinated citation [C99] must be stripped: {result}"
        );
        assert!(
            result.contains("C1"),
            "valid citation [C1] must be preserved when mixed with hallucinated: {result}"
        );

        // All-hallucinated citation: entire bracket must be dropped
        let all_fake = "A statement [C50, C99] follows.";
        let result = sanitise_cx_citations(all_fake, &valid_ids);
        assert!(
            !result.contains('['),
            "all-hallucinated citation bracket must be fully removed: {result}"
        );

        // Non-Cx bracket: must be preserved as-is
        let non_cx = "See [Table 1] and [Figure 2].";
        let result = sanitise_cx_citations(non_cx, &valid_ids);
        assert!(
            result.contains("[Table 1]"),
            "non-Cx bracket [Table 1] must be preserved: {result}"
        );
    }

    // ── Task 3 RED tests ──────────────────────────────────────────────────────

    /// TtdConfig default profile is V1Delphi (reproduction intact).
    #[test]
    fn ttd_config_default_profile_is_v1_delphi() {
        use crate::ttd::term_sheet::PromptProfile;
        let cfg = TtdConfig::default();
        assert_eq!(
            cfg.profile,
            PromptProfile::V1Delphi,
            "TtdConfig::default().profile must be V1Delphi"
        );
    }

    /// Narrative machine under V2LitReview produces prompts with v2 marker;
    /// under V1Delphi produces prompts with v1 marker.
    ///
    /// Checks NarrativeDraftGen::generate routing — v2 prompt contains
    /// "critical review of the literature"; v1 contains "multi-expert consultation".
    #[tokio::test]
    async fn narrative_draft_routes_on_profile() {
        use crate::ttd::term_sheet::PromptProfile;
        use std::sync::Mutex;

        // Recording executor that captures prompts
        struct CapturingExecutor {
            prompts: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl crate::executor::AgentExecutor for CapturingExecutor {
            async fn execute(
                &self,
                _agent_id: &base::identity::AgentId,
                instruction: &str,
                _model: &str,
                _task: &str,
            ) -> base::AlzinaResult<String> {
                self.prompts.lock().unwrap().push(instruction.to_string());
                Ok("Narrative text for testing.".to_string())
            }
        }

        let synthesis = stub_synthesis();

        // V2LitReview: must produce lit-review framed prompt
        let v2_prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let v2_executor: Arc<dyn crate::executor::AgentExecutor> = Arc::new(CapturingExecutor {
            prompts: v2_prompts.clone(),
        });

        let mut v2_draft_gen = NarrativeDraftGen::new(
            "agent", "model", synthesis.clone(),
        );
        v2_draft_gen = v2_draft_gen.with_profile(PromptProfile::V2LitReview);

        let _ = v2_draft_gen.generate(
            &[],
            &v2_executor,
            &TtdConfig::default(),
            None,
            None,
        ).await;

        let v2_captured = v2_prompts.lock().unwrap();
        assert!(
            !v2_captured.is_empty(),
            "v2 executor must have been called"
        );
        assert!(
            v2_captured[0].contains("critical literature review"),
            "v2 narrative draft must contain 'critical literature review'; got start: {}",
            &v2_captured[0][..200.min(v2_captured[0].len())]
        );
        assert!(
            !v2_captured[0].contains("multi-expert consultation"),
            "v2 narrative draft must NOT contain 'multi-expert consultation'"
        );
        drop(v2_captured);

        // V1Delphi: must produce v1-framed prompt (default)
        let v1_prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let v1_executor: Arc<dyn crate::executor::AgentExecutor> = Arc::new(CapturingExecutor {
            prompts: v1_prompts.clone(),
        });

        let v1_draft_gen = NarrativeDraftGen::new(
            "agent", "model", synthesis.clone(),
        );
        // default profile = V1Delphi — no .with_profile() call

        let _ = v1_draft_gen.generate(
            &[],
            &v1_executor,
            &TtdConfig::default(),
            None,
            None,
        ).await;

        let v1_captured = v1_prompts.lock().unwrap();
        assert!(
            !v1_captured.is_empty(),
            "v1 executor must have been called"
        );
        assert!(
            v1_captured[0].contains("multi-expert consultation"),
            "v1 narrative draft must contain 'multi-expert consultation'"
        );
    }

    /// Decision 0 / Phase 0: V3LitReviewLong routes through the v2 renderer with
    /// the long-form shape — the draft prompt carries section headings and no
    /// 300-500 cap — while V2LitReview remains byte-stable (cap present).
    #[tokio::test]
    async fn narrative_draft_v3_long_form_routing() {
        use crate::ttd::term_sheet::PromptProfile;
        use std::sync::Mutex;

        struct CapturingExecutor {
            prompts: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl crate::executor::AgentExecutor for CapturingExecutor {
            async fn execute(
                &self,
                _agent_id: &base::identity::AgentId,
                instruction: &str,
                _model: &str,
                _task: &str,
            ) -> base::AlzinaResult<String> {
                self.prompts.lock().unwrap().push(instruction.to_string());
                Ok("Narrative text for testing.".to_string())
            }
        }

        let synthesis = stub_synthesis();

        // V3: long-form shape — headings invited, cap lifted.
        let v3_prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let v3_executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(CapturingExecutor { prompts: v3_prompts.clone() });
        let v3_draft_gen = NarrativeDraftGen::new("agent", "model", synthesis.clone())
            .with_profile(PromptProfile::V3LitReviewLong);
        let _ = v3_draft_gen
            .generate(&[], &v3_executor, &TtdConfig::default(), None, None)
            .await;
        let v3_captured = v3_prompts.lock().unwrap();
        assert!(!v3_captured.is_empty(), "v3 executor must have been called");
        assert!(
            v3_captured[0].contains("critical literature review"),
            "v3 draft must keep the v2 lit-review framing"
        );
        assert!(
            v3_captured[0].contains("## section headings"),
            "v3 draft must invite markdown section headings"
        );
        assert!(
            !v3_captured[0].contains("300-500"),
            "v3 draft must NOT carry the 300-500 word cap"
        );
        drop(v3_captured);

        // V2 remains byte-stable: cap present, no long-form text.
        let v2_prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let v2_executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(CapturingExecutor { prompts: v2_prompts.clone() });
        let v2_draft_gen = NarrativeDraftGen::new("agent", "model", synthesis.clone())
            .with_profile(PromptProfile::V2LitReview);
        let _ = v2_draft_gen
            .generate(&[], &v2_executor, &TtdConfig::default(), None, None)
            .await;
        let v2_captured = v2_prompts.lock().unwrap();
        assert!(
            v2_captured[0].contains("300-500"),
            "v2 draft must still carry the 300-500 word cap (byte-stability)"
        );
        assert!(
            !v2_captured[0].contains("## section headings"),
            "v2 draft must NOT carry long-form section text"
        );
    }

    // ── Test: narrative_reuses_synthesis_fitness_judges ──────────────────────

    /// Narrative fitness uses the 6 synthesis judge templates with NARRATIVE_WEIGHTS.
    ///
    /// Verify that NarrativeEvalFitness returns NARRATIVE_WEIGHTS and the validity
    /// predicate is is_valid_synthesis (faithfulness ≥ 4).
    #[test]
    fn narrative_reuses_synthesis_fitness_judges() {
        let eval = NarrativeEvalFitness::new("agent", "model");

        // Weights must be NARRATIVE_WEIGHTS (not SYNTHESIS_WEIGHTS or GRAPH_WEIGHTS)
        let weights = eval.weights();
        assert_eq!(
            weights, NARRATIVE_WEIGHTS,
            "NarrativeEvalFitness must return NARRATIVE_WEIGHTS"
        );

        // Validity function must be is_valid_synthesis (faithfulness ≥ 4)
        let validity_fn = eval.validity_fn();

        // faithfulness=4 → valid
        let valid = FitnessEval::new(vec![("faithfulness".to_string(), Some(4))]);
        assert!(
            validity_fn(&valid),
            "faithfulness=4 must be valid for narrative stage"
        );

        // faithfulness=3 → invalid (sorts last)
        let invalid = FitnessEval::new(vec![("faithfulness".to_string(), Some(3))]);
        assert!(
            !validity_fn(&invalid),
            "faithfulness=3 must be invalid for narrative stage (sorts last)"
        );
    }

    // ── V2 profile selection tests ────────────────────────────────────────────

    /// V2LitReview profile returns V2_NARRATIVE_WEIGHTS (5 dims) and is_valid_v2.
    #[test]
    fn v2_narrative_fitness_returns_v2_weights_and_validity_fn() {
        use crate::ttd::fitness::FitnessEval;
        use crate::ttd::term_sheet::PromptProfile;
        use crate::ttd::weights::V2_NARRATIVE_WEIGHTS;

        let v2 =
            NarrativeEvalFitness::new("agent", "model").with_profile(PromptProfile::V2LitReview);
        assert_eq!(
            v2.weights(),
            V2_NARRATIVE_WEIGHTS,
            "V2LitReview must return V2_NARRATIVE_WEIGHTS"
        );
        assert_eq!(
            v2.weights().len(),
            5,
            "V2 narrative weight table must have 5 dims"
        );
        let vfn = v2.validity_fn();
        let ok = FitnessEval::new(vec![("faithfulness".into(), Some(4))]);
        assert!(vfn(&ok), "faithfulness=4 must be valid for V2 narrative");
        let fail = FitnessEval::new(vec![("faithfulness".into(), Some(3))]);
        assert!(!vfn(&fail), "faithfulness=3 must be invalid for V2 narrative");

        // V1 path unchanged
        let v1 = NarrativeEvalFitness::new("agent", "model");
        assert_eq!(
            v1.weights(),
            NARRATIVE_WEIGHTS,
            "V1Delphi must return NARRATIVE_WEIGHTS"
        );
        assert_eq!(
            v1.weights().len(),
            6,
            "V1 narrative weight table must have 6 dims"
        );
    }

    /// V2 narrative eval never sets a veto — the pseudo-artifact has empty sources
    /// by construction, so traceability_veto_synthesis would fail every candidate.
    /// Verify by checking that FitnessEval returned by the v2 path has veto=None.
    #[tokio::test]
    async fn v2_narrative_veto_is_none() {
        use std::sync::Arc;

        use async_trait::async_trait;

        use crate::executor::AgentExecutor;
        use crate::ttd::term_sheet::PromptProfile;
        use crate::ttd::TtdConfig;

        // Executor returns a score-5 response for every judge call.
        struct Score5Executor;
        #[async_trait]
        impl AgentExecutor for Score5Executor {
            async fn execute(
                &self,
                _agent_id: &base::identity::AgentId,
                _instruction: &str,
                _model: &str,
                _task: &str,
            ) -> base::AlzinaResult<String> {
                Ok("<fitness_evaluation><score>5</score><evidence>all good</evidence></fitness_evaluation>".into())
            }
        }

        let eval =
            NarrativeEvalFitness::new("agent", "model").with_profile(PromptProfile::V2LitReview);
        let executor: Arc<dyn AgentExecutor> = Arc::new(Score5Executor);
        let draft = "The permafrost thaw narrative summary.".to_string();
        let config = TtdConfig::default();

        let result = eval.evaluate(&draft, &executor, &config).await.unwrap();

        assert!(
            result.veto.is_none(),
            "V2 narrative eval must never set a traceability veto (pseudo-artifact has empty \
             sources by construction); got: {:?}",
            result.veto
        );
    }

    // ── Rubric-encoding Phase 1: planned-path tests (W-e714abb4) ─────────────

    /// Build a sectioned fixture plan for the planned-path tests.
    fn fixture_plan() -> crate::ttd::plan::ReviewPlan {
        use crate::ttd::plan::{
            PlanArchetype, PlanSection, PlantedThread, ReviewPlan, TermRegistryEntry,
        };

        ReviewPlan {
            archetype: PlanArchetype::ThesisAndConvergence,
            archetype_rationale: "evidence lines converge on one thesis".into(),
            focal_question: "Does abrupt thaw dominate the methane budget?".into(),
            scope_exclusions: vec!["ocean clathrates".into()],
            term_registry: vec![TermRegistryEntry {
                term: "abrupt thaw".into(),
                definition: "thermokarst-driven rapid permafrost collapse".into(),
                banned_synonyms: vec!["sudden melting".into()],
            }],
            planted_threads: vec![PlantedThread {
                id: "T1".into(),
                description: "instrument coverage limits the budget estimate".into(),
                marker: "the measurement gap".into(),
                setup_section: "Background".into(),
                payoff_section: "Convergence".into(),
            }],
            sections: vec![
                PlanSection {
                    heading: "Background".into(),
                    purpose: "establish the thaw-rate evidence base".into(),
                    budget_words: Some(450),
                    claim_ids: vec!["C1".into(), "C2".into()],
                },
                PlanSection {
                    heading: "Convergence".into(),
                    purpose: "argue the thesis from converging lines".into(),
                    budget_words: Some(600),
                    claim_ids: vec!["C3".into()],
                },
            ],
        }
    }

    /// Executor that records every (task, instruction) pair and echoes a fixed body.
    struct RecordingExecutor {
        calls: std::sync::Mutex<Vec<(String, String)>>,
        response: String,
    }

    impl RecordingExecutor {
        fn new(response: impl Into<String>) -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
                response: response.into(),
            }
        }
    }

    #[async_trait]
    impl crate::executor::AgentExecutor for RecordingExecutor {
        async fn execute(
            &self,
            _agent_id: &base::identity::AgentId,
            instruction: &str,
            _model: &str,
            task: &str,
        ) -> base::AlzinaResult<String> {
            self.calls
                .lock()
                .unwrap()
                .push((task.to_string(), instruction.to_string()));
            Ok(self.response.clone())
        }
    }

    /// Phase P: NarrativeCritique skips its spawn when refine is unreachable
    /// (`resolve_without_retrieval` off) — the wasted-spend fix. With the flag
    /// on it spawns once, and the v2 prompt carries the prior fitness
    /// evaluation (C-N2 — the critique wire Kvasir F1 found dropped).
    #[tokio::test]
    async fn narrative_critique_skips_spawn_when_refine_unreachable() {
        use crate::ttd::term_sheet::PromptProfile;

        let synthesis = stub_synthesis();
        let recorder = Arc::new(RecordingExecutor::new("<gaps></gaps>"));
        let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();
        let critique = NarrativeCritique::new("critique-agent", "model", synthesis)
            .with_profile(PromptProfile::V2LitReview);
        let fitness = FitnessEval::new(vec![("faithfulness".to_string(), Some(2))]);

        // Flag off (default): skipped — no spawn, empty gaps.
        let mut config = TtdConfig::default();
        let gaps = critique
            .identify(&"draft".to_string(), &fitness, &executor, &config)
            .await
            .unwrap();
        assert!(gaps.is_empty(), "critique must return no gaps when refine is unreachable");
        assert_eq!(recorder.calls.lock().unwrap().len(), 0, "no spawn when skipped");

        // Flag on: spawns once, and the prompt carries the fitness feedback.
        config.resolve_without_retrieval = true;
        critique
            .identify(&"draft".to_string(), &fitness, &executor, &config)
            .await
            .unwrap();
        let calls = recorder.calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "critique spawns once when refine is reachable");
        assert!(
            calls[0].1.contains("Prior fitness evaluation"),
            "flag-on v2 critique must carry the fitness feedback section"
        );
    }

    /// Draft prompt carries the plan block iff a plan is injected; the no-plan
    /// path stays byte-free of plan content (byte-stability invariant).
    #[tokio::test]
    async fn planned_draft_prompt_carries_plan_iff_plan_set() {
        use crate::ttd::term_sheet::PromptProfile;

        let synthesis = stub_synthesis();
        let config = TtdConfig::default();

        // Without plan: no plan content in the prompt.
        let recorder = Arc::new(RecordingExecutor::new("draft text"));
        let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();
        let draft_gen = NarrativeDraftGen::new("agent", "model", synthesis.clone())
            .with_profile(PromptProfile::V3LitReviewLong);
        draft_gen.generate(&[], &executor, &config, None, None).await.unwrap();
        let (_, prompt_no_plan) = recorder.calls.lock().unwrap()[0].clone();
        assert!(
            !prompt_no_plan.contains("Winning plan"),
            "plan-absent draft prompt must carry no plan content"
        );

        // With plan: plan block appended, base prompt is an exact prefix.
        let recorder = Arc::new(RecordingExecutor::new("draft text"));
        let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();
        let draft_gen = NarrativeDraftGen::new("agent", "model", synthesis)
            .with_profile(PromptProfile::V3LitReviewLong)
            .with_plan(Some(Arc::new(fixture_plan())));
        draft_gen.generate(&[], &executor, &config, None, None).await.unwrap();
        let (_, prompt_planned) = recorder.calls.lock().unwrap()[0].clone();
        assert!(
            prompt_planned.contains("Winning plan"),
            "planned draft prompt must carry the plan block"
        );
        assert!(
            prompt_planned.starts_with(&prompt_no_plan),
            "planned draft prompt must compose the base prompt as an exact prefix"
        );
    }

    /// C-N3: with a sectioned plan and sectionable candidates the merger runs
    /// one `narrative_section_merge` call per plan section, threads the
    /// previous section's tail into the next prompt, and reassembles headings.
    #[tokio::test]
    async fn planned_merge_runs_section_by_section_with_prev_tail() {
        use crate::ttd::term_sheet::PromptProfile;

        let candidate_a = "## Background\nThaw rates rise [C1].\n\n## Convergence\nLines converge [C2].".to_string();
        let candidate_b = "## Background\nField data agree [C1].\n\n## Convergence\nThe thesis holds [C2].".to_string();

        let recorder = Arc::new(RecordingExecutor::new("Merged section body."));
        let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();

        let merger = NarrativeMerger::new("agent", "model", stub_synthesis())
            .with_profile(PromptProfile::V3LitReviewLong)
            .with_plan(Some(Arc::new(fixture_plan())));

        let merged = merger
            .merge(&[candidate_a, candidate_b], &executor, &TtdConfig::default())
            .await
            .unwrap();

        let calls = recorder.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 2, "one merge call per plan section, got {}", calls.len());
        assert!(
            calls.iter().all(|(task, _)| task == "narrative_section_merge"),
            "sectionable planned merge must use the narrative_section_merge task"
        );
        // First section sees no prior text; second sees the first's tail.
        assert!(
            calls[0].1.contains("first section"),
            "first section prompt must declare no prior text"
        );
        assert!(
            calls[1].1.contains("Merged section body."),
            "second section prompt must carry the previous section's tail paragraph"
        );
        // Both plan headings survive reassembly (prepended when the model drops them).
        assert!(merged.contains("## Background"), "merged doc must carry section 1 heading");
        assert!(merged.contains("## Convergence"), "merged doc must carry section 2 heading");
    }

    /// C-N3 fallback: candidates that do not carry the plan's headings degrade
    /// to ONE whole-document planned merge (never a panic, never a plain merge).
    #[tokio::test]
    async fn planned_merge_falls_back_to_whole_document_when_not_sectionable() {
        use crate::ttd::term_sheet::PromptProfile;

        let recorder = Arc::new(RecordingExecutor::new("Merged narrative."));
        let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();

        let merger = NarrativeMerger::new("agent", "model", stub_synthesis())
            .with_profile(PromptProfile::V3LitReviewLong)
            .with_plan(Some(Arc::new(fixture_plan())));

        // No "## " headings anywhere — not sectionable.
        merger
            .merge(
                &["Plain narrative one.".to_string(), "Plain narrative two.".to_string()],
                &executor,
                &TtdConfig::default(),
            )
            .await
            .unwrap();

        let calls = recorder.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 1, "fallback must issue exactly one whole-document merge");
        assert_eq!(calls[0].0, "narrative_final_merge");
        assert!(
            calls[0].1.contains("Winning plan"),
            "fallback prompt must still carry the plan block"
        );
        assert!(
            calls[0].1.contains("never average"),
            "fallback prompt must carry the merge rules (verdict-fusion rule)"
        );
    }

    /// Plan presence adds the sixth judge (`plan_conformance`) and switches the
    /// weight table to V3_PLANNED_NARRATIVE_WEIGHTS in lockstep — run.rs books
    /// fitness calls from weights().len(), so these MUST move together.
    #[tokio::test]
    async fn planned_eval_fitness_adds_sixth_judge_and_switches_weights() {
        use crate::ttd::term_sheet::PromptProfile;
        use crate::ttd::weights::{V2_NARRATIVE_WEIGHTS, V3_PLANNED_NARRATIVE_WEIGHTS};

        let recorder = Arc::new(RecordingExecutor::new(
            "<fitness_evaluation><score>4</score></fitness_evaluation>",
        ));
        let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();

        let planned = NarrativeEvalFitness::new("agent", "model")
            .with_profile(PromptProfile::V3LitReviewLong)
            .with_plan(Some(Arc::new(fixture_plan())));
        assert_eq!(
            planned.weights(),
            V3_PLANNED_NARRATIVE_WEIGHTS,
            "plan present must switch to the 6-dim planned weight table"
        );

        let eval = planned
            .evaluate(&"Draft text.".to_string(), &executor, &TtdConfig::default())
            .await
            .unwrap();
        assert_eq!(
            eval.scores.len(),
            planned.weights().len(),
            "judge count must equal weights().len() (run.rs call accounting)"
        );
        assert_eq!(
            eval.scores.last().map(|(name, _)| name.as_str()),
            Some("plan_conformance"),
            "sixth dimension must be plan_conformance"
        );
        assert!(eval.veto.is_none(), "planned narrative eval must never veto");
        let tasks: Vec<String> =
            recorder.calls.lock().unwrap().iter().map(|(t, _)| t.clone()).collect();
        assert_eq!(tasks.last().map(String::as_str), Some("plan_conformance"));

        // Plan absent: unchanged v2 table, no sixth judge.
        let unplanned =
            NarrativeEvalFitness::new("agent", "model").with_profile(PromptProfile::V3LitReviewLong);
        assert_eq!(
            unplanned.weights(),
            V2_NARRATIVE_WEIGHTS,
            "plan absent must keep the v2 narrative weight table"
        );
    }

    /// `last_paragraph` returns the final non-empty block (C-N3 prev-tail seed).
    #[test]
    fn last_paragraph_returns_final_nonempty_block() {
        assert_eq!(
            last_paragraph("## H\n\nfirst para.\n\nsecond para.\n\n"),
            Some("second para.".to_string())
        );
        assert_eq!(last_paragraph("   \n\n  "), None, "blank text yields None");
    }

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║ SEAM F1 — DEAD-NARRATIVE REFINE PATH  (characterisation net, W-522022c5) ║
    // ║ Pins the OFF/ON critique-spawn boundary + the refine "retrieved ignored" ║
    // ║ contract. CHARACTERISATION (not spec): a red test here = a behavioural   ║
    // ║ change occurred; update in the same commit with rationale.               ║
    // ╚═══════════════════════════════════════════════════════════════════════╝
    mod f1_dead_narrative_refine {
        use super::*;

        /// PINS: flag-OFF default → critique returns empty AND issues zero spawns.
        /// If the fix sprint changes the default or removes the guard, this fails.
        #[tokio::test]
        async fn f1_critique_is_dead_when_flag_off_default() {
            use crate::ttd::term_sheet::PromptProfile;
            use std::sync::Arc;

            let synthesis = stub_synthesis();
            let recorder = Arc::new(RecordingExecutor::new("<gaps><gap><description>d</description></gap></gaps>"));
            let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();
            let critique = NarrativeCritique::new("critique-agent", "model", synthesis)
                .with_profile(PromptProfile::V2LitReview);
            let fitness = FitnessEval::new(vec![("faithfulness".to_string(), Some(2))]);

            let config = TtdConfig::default(); // resolve_without_retrieval == false
            assert!(
                !config.resolve_without_retrieval,
                "PRECONDITION: default config must keep refine unreachable"
            );

            let gaps = critique
                .identify(&"draft".to_string(), &fitness, &executor, &config)
                .await
                .unwrap();

            // Even though the executor WOULD have returned a parseable gap, the guard
            // fires first: empty result, zero spawns.
            assert!(gaps.is_empty(), "F1: flag-off critique must yield zero gaps");
            assert_eq!(
                recorder.calls.lock().unwrap().len(),
                0,
                "F1: flag-off critique must issue ZERO executor spawns (dead path)"
            );
        }

        /// PINS: flag-ON → critique spawns exactly once. The refine path becomes
        /// reachable. This is the boundary the fix sprint will move; pin it so the
        /// move is loud.
        #[tokio::test]
        async fn f1_critique_spawns_once_when_flag_on() {
            use crate::ttd::term_sheet::PromptProfile;
            use std::sync::Arc;

            let synthesis = stub_synthesis();
            let recorder = Arc::new(RecordingExecutor::new("<gaps></gaps>"));
            let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();
            let critique = NarrativeCritique::new("critique-agent", "model", synthesis)
                .with_profile(PromptProfile::V2LitReview);
            let fitness = FitnessEval::new(vec![("faithfulness".to_string(), Some(2))]);

            let mut config = TtdConfig::default();
            config.resolve_without_retrieval = true;

            critique
                .identify(&"draft".to_string(), &fitness, &executor, &config)
                .await
                .unwrap();

            let calls = recorder.calls.lock().unwrap();
            assert_eq!(calls.len(), 1, "F1: flag-on critique must spawn exactly once");
            assert_eq!(calls[0].0, "narrative_critique", "F1: spawn task label pinned");
        }

        /// PINS A LATENT QUIRK (report, do not fix): NarrativeRefine::resolve, when
        /// invoked directly (bypassing the dead guard), IGNORES the `retrieved`
        /// slice entirely. We pin that a non-empty retrieved slice does NOT reach
        /// the prompt. If a fix wires retrieval into refine, this test fails and
        /// forces a conscious decision.
        #[tokio::test]
        async fn f1_refine_ignores_retrieved_slice() {
            use crate::ttd::term_sheet::PromptProfile;
            use crate::ttd::stages::RetrievedContext;
            use std::sync::Arc;

            let synthesis = stub_synthesis();
            let recorder = Arc::new(RecordingExecutor::new("rewritten narrative"));
            let executor: Arc<dyn crate::executor::AgentExecutor> = recorder.clone();
            let refine = NarrativeRefine::new("refine-agent", "model", synthesis)
                .with_profile(PromptProfile::V1Delphi);
            let fitness = FitnessEval::new(vec![]);
            let gaps = vec![IdentifiedGap {
                description: "missing methods section".into(),
                query: "methods".into(),
            }];
            // A NON-EMPTY retrieved slice — the contract says it must be ignored.
            let retrieved = vec![RetrievedContext {
                source_id: "src:should-be-ignored".into(),
                content: "this content must never reach the refine prompt".into(),
                section: None,
            }];
            let config = TtdConfig::default();

            let out = refine
                .resolve(&"draft".to_string(), &fitness, &gaps, &retrieved, &executor, &config)
                .await
                .unwrap();

            assert_eq!(out, "rewritten narrative", "F1: refine returns the spawn body");
            let calls = recorder.calls.lock().unwrap();
            assert_eq!(calls.len(), 1, "F1: refine spawns exactly once");
            assert_eq!(calls[0].0, "narrative_refine", "F1: refine task label pinned");
            assert!(
                !calls[0].1.contains("src:should-be-ignored"),
                "F1: refine prompt must NOT contain retrieved content (retrieved is ignored)"
            );
        }
    }

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║ SEAM F4c — NARRATIVE parser (characterisation net, W-522022c5)          ║
    // ║ Reaches the private file-level `parse_gaps_xml` via super::super.        ║
    // ║ PINS THE T1 RULED CONTRACT: desc-only→Ok(1) (query defaults to desc);    ║
    // ║ missing block→Ok(vec![]); panic-freedom on adversarial multibyte input.  ║
    // ║ (Narrative already matched the missing→Ok([]) contract; the prior        ║
    // ║  Err-vs-Ok asymmetry with synthesis is now closed — all three agree.)    ║
    // ╚═══════════════════════════════════════════════════════════════════════╝
    mod f4c_narrative_parser {
        #[test]
        fn f4_narrative_desc_only_yields_one_gap() {
            let xml = "<gaps><gap><description>a gap</description></gap></gaps>";
            let out = super::super::parse_gaps_xml(xml).expect("narrative: must be Ok");
            assert_eq!(out.len(), 1, "F4: narrative emits a gap on desc-only (query defaults to desc)");
            assert_eq!(out[0].query, out[0].description, "F4: narrative defaults query→description");
        }

        #[test]
        fn f4_narrative_missing_block_is_ok_empty() {
            // T1 RULED CONTRACT: missing block → Ok(vec![]) (never Err) — now shared
            // by all three parsers (synthesis no longer Errs).
            let out = super::super::parse_gaps_xml("no gaps block").expect("narrative returns Ok even when empty");
            assert!(out.is_empty(), "F4: narrative returns Ok(vec![]) on missing block (T1 contract)");
        }

        /// PINS PANIC-FREEDOM on the parser — the literal probe claim.
        /// Five adversarial multibyte fixtures (A1–A5) engineered to break slice
        /// arithmetic. Trip-wire against a future non-ASCII-anchored slice edit.
        #[test]
        fn f4_narrative_byte_slicing_never_panics_on_multibyte() {
            let fixtures = [
                "<gaps><gap><description>café</description></gap></gaps>",        // boundary
                "<gaps>—<gap>—<description>naïve</description>—</gap>—</gaps>",   // interleave
                "<gaps><gap><description>x</description></gap>—",                 // straddle end
                "<gaps><gap><description>日本語のギャップ</description></gap></gaps>", // wide multibyte
                "<gaps><gap><description>café",                                   // unterminated multibyte tail
            ];
            for (i, fx) in fixtures.iter().enumerate() {
                let r = std::panic::catch_unwind(|| super::super::parse_gaps_xml(fx));
                assert!(
                    r.is_ok(),
                    "F4: narrative byte-slicing MUST NOT panic on multibyte fixture #{i}: {fx:?}"
                );
            }
        }
    }
}
