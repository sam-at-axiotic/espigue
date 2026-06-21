//! Stage-2 synthesis task implementations.
//!
//! Implements the stage-task traits over `SynthesisArtifact` for the TTD
//! Stage-2 synthesis pipeline. Mirrors `consensus/src/consensus/diffusion/
//! synthesis_tasks.py`.
//!
//! ## use_graph_draft branch (synthesis_tasks.py:131-133)
//!
//! When an `ArgumentationGraph` is present and `use_graph_draft=true` (the
//! consensus default), `SynthesisDraftGen::generate` picks the `draft_graph`
//! prompt that seeds the synthesis from the graph structure rather than the
//! raw expert prose. When `graph=None`, it falls back to `draft`.
//!
//! ## Heuristic fallbacks
//!
//! Both gap-identify and gap-resolve have heuristic fallbacks:
//! - `_heuristic_identify` fires when `gap_identify` returns no gaps or parse
//!   fails (synthesis_tasks.py:424-430).
//! - `_heuristic_resolve` finds single-source claims and marks them for
//!   additional evidence (synthesis_tasks.py:432-614).
//!
//! ## Trust boundary (T-23-07)
//!
//! Retrieved text and `ExpertResponse.prose` stay in the data section of
//! rendered prompts — never the instruction position (mirrors Phase 22 adapter).

use std::sync::Arc;

use async_trait::async_trait;

use crate::adapter::ExpertResponse;
use crate::executor::AgentExecutor;
use crate::ttd::artifact::{ArgumentationGraph, Gap, SynthesisArtifact};
use crate::ttd::config::TtdConfig;
use crate::ttd::fitness::{is_valid_synthesis, is_valid_v2, traceability_veto_synthesis, FitnessEval};
use crate::ttd::mod_types::TtdError;
use crate::ttd::stages::{DraftGen, EvalFitness, GapIdentify, GapResolve, Merger, RetrievedContext};
use crate::ttd::state::IdentifiedGap;
use crate::ttd::term_sheet::{normalise_support_level, PromptProfile};
use crate::ttd::weights::{SYNTHESIS_WEIGHTS, V2_SYNTHESIS_WEIGHTS};

// ── SynthesisDraftGen ──────────────────────────────────────────────────────────

/// Stage-2 draft generation: seeded from the Stage-1 argumentation graph.
///
/// When `graph` is `Some`, the draft uses `draft_graph.mustache` (the consensus
/// default: `use_graph_draft=true`, synthesis_tasks.py:131-133). When `graph`
/// is `None`, falls back to `draft.mustache`.
///
/// The Stage-1 `ArgumentationGraph` must be threaded in here — NOT omitted
/// (Anti-Pattern "Stage threading omission").
pub struct SynthesisDraftGen {
    /// Agent ID for synthesis draft spawns.
    pub agent_id: String,
    /// Model to use for synthesis spawns.
    pub model: String,
    /// Prompt version string for provenance stamping.
    pub prompt_version: String,
    /// Stage-1 argumentation graph (None → fall back to plain draft).
    /// The consensus default is `use_graph_draft=True` — this field should
    /// normally be `Some`.
    pub graph: Option<ArgumentationGraph>,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
}

impl SynthesisDraftGen {
    pub fn new(
        agent_id: impl Into<String>,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
        graph: Option<ArgumentationGraph>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            prompt_version: prompt_version.into(),
            graph,
            profile: PromptProfile::V1Delphi, // default: backward compat
        }
    }

    /// Set the prompt/schema profile (consuming builder, matches EngineConfig pattern).
    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }
}

#[async_trait]
impl DraftGen<SynthesisArtifact> for SynthesisDraftGen {
    /// Generate one synthesis draft.
    ///
    /// Picks `draft_graph` when `self.graph.is_some()` (use_graph_draft=true,
    /// the consensus default). Falls back to `draft` when graph is None.
    async fn generate(
        &self,
        inputs: &[ExpertResponse],
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
        persona_prompt: Option<&str>,
        sampling: Option<crate::executor::SamplingParams>,
    ) -> Result<SynthesisArtifact, TtdError> {
        use crate::ttd::prompts::synthesis::{
            render_synthesis_draft, render_synthesis_draft_graph,
            SynthesisDraftGraphInput, SynthesisDraftInput,
        };
        use alzina_core::identity::AgentId;

        // use_graph_draft branch (synthesis_tasks.py:131-133):
        // Pick draft_graph when self._graph is not None.
        // B2: fork on profile — V2LitReview uses lit_review:: render functions.
        // Decision 0: v3 = v2 at Stage 2 (long-form changes Stage-3 shape only).
        let prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                // v2 path: paper-framing, no Delphi dressing, guards, quality gate.
                if let Some(ref graph) = self.graph {
                    tracing::debug!(
                        n_nodes = graph.nodes.len(),
                        "SynthesisDraftGen: v2 graph-based draft"
                    );
                    crate::ttd::prompts::lit_review::render_synthesis_draft_graph_v2(
                        &format!("Synthesise the argumentation graph from {} papers", inputs.len()),
                        graph,
                        inputs,
                        "500-800",
                    )
                } else {
                    tracing::debug!("SynthesisDraftGen: v2 plain draft");
                    crate::ttd::prompts::lit_review::render_synthesis_draft_v2(
                        "Synthesise the papers",
                        inputs,
                        "500-800",
                    )
                }
            }
            PromptProfile::V1Delphi => {
                // v1 path (byte-identical to pre-B2).
                if let Some(ref graph) = self.graph {
                    tracing::debug!(
                        n_nodes = graph.nodes.len(),
                        n_edges = graph.edges.len(),
                        "SynthesisDraftGen: using graph-based draft template (use_graph_draft=true)"
                    );

                    let unverified_nodes: Vec<(String, String)> = graph
                        .nodes
                        .iter()
                        .filter(|n| {
                            n.verification_status
                                .as_deref()
                                .map_or(false, |s| s == "failed" || s == "unverified")
                        })
                        .map(|n| (n.id.clone(), n.claim.chars().take(100).collect()))
                        .collect();

                    render_synthesis_draft_graph(&SynthesisDraftGraphInput {
                        question: format!(
                            "Synthesise the argumentation graph from {} expert responses",
                            inputs.len()
                        ),
                        expert_count: inputs.len(),
                        graph,
                        has_unverified_nodes: !unverified_nodes.is_empty(),
                        unverified_nodes: &unverified_nodes,
                        target_length: "500-800",
                    })
                } else {
                    tracing::debug!(
                        "SynthesisDraftGen: graph is None — using plain draft template (use_graph_draft=false)"
                    );

                    let responses: Vec<(String, Vec<String>)> = inputs
                        .iter()
                        .map(|r| {
                            (
                                r.expert_id.as_str().to_string(),
                                vec![r.prose.clone()],
                            )
                        })
                        .collect();

                    render_synthesis_draft(&SynthesisDraftInput {
                        question: "Synthesise the expert responses".into(),
                        responses: &responses,
                    })
                }
            }
        };

        // EXT-01 Phase 24: when a persona prompt is supplied, prefix it so the
        // trajectory adopts the persona's analytical lens. None → existing template
        // behaviour preserved (Phase 23 reproduction semantics).
        let effective_prompt = if let Some(persona) = persona_prompt {
            format!("{}\n\n---\n\n{}", persona, prompt)
        } else {
            prompt
        };

        // Execute through governed AgentExecutor (ENGINE-05).
        // EXT-01 Phase 24: call execute_with_sampling to thread per-trajectory
        // sampling params to the sidecar. Default impl falls through to execute()
        // when sampling=None (Pitfall 3 — backward compatible).
        let agent_id = AgentId::new(self.agent_id.as_str());
        let output = executor
            .execute_with_sampling(&agent_id, &effective_prompt, &self.model, "synthesis_draft", sampling)
            .await
            .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

        // Parse the XML output into a SynthesisArtifact.
        parse_synthesis_xml(&output, &self.model, &self.prompt_version, self.profile)
    }
}

// ── SynthesisGapIdentify ───────────────────────────────────────────────────────

/// Stage-2 gap identification: identifies 3-5 gaps in one synthesis draft.
///
/// Fires `_heuristic_identify` when the LLM returns no gaps or parse fails
/// (synthesis_tasks.py:424-430). Heuristic rules:
/// - Single-source claims → "Find additional evidence for…"
/// - majority/consensus without counterarguments → "Find dissenting views on…"
/// - agreement > 2× disagreement → "Check for additional areas of disagreement"
/// - No uncertainties → "Identify uncertainties"
pub struct SynthesisGapIdentify {
    pub agent_id: String,
    pub model: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
}

impl SynthesisGapIdentify {
    pub fn new(agent_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            profile: PromptProfile::V1Delphi,
        }
    }

    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Heuristic gap identification fallback (synthesis_tasks.py:432-464).
    ///
    /// Returns `IdentifiedGap` where `description == query` (description doubles
    /// as the retrieval query for heuristic gaps).
    fn heuristic_identify(&self, draft: &SynthesisArtifact) -> Vec<IdentifiedGap> {
        let mut gaps: Vec<String> = Vec::new();

        // Single-source claims → request additional evidence
        for claim in &draft.claims {
            if claim.sources.len() == 1 {
                let text_snippet: String = claim.text.chars().take(50).collect();
                gaps.push(format!("Find additional evidence for: {text_snippet}..."));
            }
            // majority/consensus without counterarguments → dissenting views
            if matches!(
                claim.agreement_level.as_deref(),
                Some("consensus") | Some("majority")
            ) && claim.counterarguments.is_empty()
            {
                let text_snippet: String = claim.text.chars().take(50).collect();
                gaps.push(format!("Find dissenting views on: {text_snippet}..."));
            }
        }

        // Agreement vs disagreement imbalance
        if draft.areas_of_agreement.len() > 2 * draft.areas_of_disagreement.len() {
            gaps.push("Check for additional areas of disagreement".into());
        }

        // Missing uncertainties
        if draft.uncertainties.is_empty() {
            gaps.push("Identify uncertainties in the expert responses".into());
        }

        // description doubles as query for heuristic gaps (synthesis_tasks.py:473)
        gaps.into_iter()
            .map(|g| IdentifiedGap {
                description: g.clone(),
                query: g,
            })
            .collect()
    }
}

#[async_trait]
impl GapIdentify<SynthesisArtifact> for SynthesisGapIdentify {
    async fn identify(
        &self,
        draft: &SynthesisArtifact,
        fitness: &FitnessEval,
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<Vec<IdentifiedGap>, TtdError> {
        use crate::ttd::fitness::generate_feedback;
        use crate::ttd::prompts::synthesis::{render_synthesis_gap_identify, SynthesisGapIdentifyInput};
        use alzina_core::identity::AgentId;

        // Build fitness feedback for injection into prompt.
        let fitness_feedback = if config.use_fitness_feedback && !fitness.all_none() {
            Some(generate_feedback(fitness, config.fitness_threshold))
        } else {
            None
        };

        // Decision 0: v3 = v2.
        let prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                // Serialize the synthesis to XML for the gap identify prompt context.
                let synthesis_xml = format!("<synthesis><narrative>{}</narrative></synthesis>", draft.narrative);
                crate::ttd::prompts::lit_review::render_synthesis_gap_identify_v2(
                    &synthesis_xml,
                    "Identify coverage gaps",
                    fitness_feedback.as_deref(),
                )
            }
            PromptProfile::V1Delphi => {
                render_synthesis_gap_identify(&SynthesisGapIdentifyInput {
                    draft,
                    fitness_feedback: fitness_feedback.as_deref(),
                })
            }
        };

        let agent_id = AgentId::new(self.agent_id.as_str());
        let output = executor
            .execute(&agent_id, &prompt, &self.model, "synthesis_gap_identify")
            .await
            .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

        // Parse XML gaps response.
        match parse_gaps_xml(&output) {
            Ok(gaps) if !gaps.is_empty() => Ok(gaps),
            Ok(_) => {
                tracing::debug!(
                    "SynthesisGapIdentify: returned no gaps — firing heuristic fallback"
                );
                Ok(self.heuristic_identify(draft))
            }
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "SynthesisGapIdentify: parse failed — firing heuristic fallback"
                );
                Ok(self.heuristic_identify(draft))
            }
        }
    }
}

// ── SynthesisGapResolve ───────────────────────────────────────────────────────

/// Stage-2 gap resolution: three-tier fallback chain.
///
/// 1. `gap_resolve_patch` (patch-based incremental — default).
/// 2. `gap_resolve` (full regeneration) on parse failure.
/// 3. `_heuristic_resolve` (finds single-source claims) on that failure.
///
/// Empty-retrieved guard lives in `run.rs` — by the time `resolve()` is called,
/// retrieved is guaranteed non-empty.
pub struct SynthesisGapResolve {
    pub agent_id: String,
    pub model: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
}

impl SynthesisGapResolve {
    pub fn new(agent_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            profile: PromptProfile::V1Delphi,
        }
    }

    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Heuristic resolve: marks single-source claims with additional sourcing notes.
    ///
    /// Returns the draft with a note added to single-source claims.
    fn heuristic_resolve(
        &self,
        draft: &SynthesisArtifact,
        retrieved: &[RetrievedContext],
    ) -> SynthesisArtifact {
        tracing::debug!(
            n_retrieved = retrieved.len(),
            "SynthesisGapResolve: heuristic_resolve — adding retrieved source IDs to single-source claims"
        );
        let mut refined = draft.clone();
        // Add retrieved source IDs to single-source claims (heuristic resolution).
        let new_source_ids: Vec<String> = retrieved.iter().map(|r| r.source_id.clone()).collect();
        for claim in &mut refined.claims {
            if claim.sources.len() == 1 {
                for sid in &new_source_ids {
                    if !claim.sources.contains(sid) {
                        claim.sources.push(sid.clone());
                    }
                }
            }
        }
        refined
    }
}

#[async_trait]
impl GapResolve<SynthesisArtifact> for SynthesisGapResolve {
    async fn resolve(
        &self,
        draft: &SynthesisArtifact,
        _fitness: &FitnessEval, // Stage 2 resolve is retrieval-driven; fitness ignored (byte-stable)
        gaps: &[IdentifiedGap],
        retrieved: &[RetrievedContext],
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<SynthesisArtifact, TtdError> {
        use crate::ttd::prompts::synthesis::{
            render_synthesis_gap_resolve, render_synthesis_gap_resolve_patch,
            SynthesisGapResolveInput,
        };
        use alzina_core::identity::AgentId;

        let agent_id = AgentId::new(self.agent_id.as_str());

        // Tier 1: patch-based incremental resolve.
        // B2: fork prompt — v2 uses lit_review:: render; v1 path unchanged.
        let retrieved_as_expert: Vec<crate::adapter::ExpertResponse> = retrieved
            .iter()
            .filter_map(|r| {
                // Only build ExpertResponse from retrieved context when provenance is available.
                // Fall back: skip items without a valid source_id for the v2 render path.
                crate::adapter::SourceId::try_new(r.source_id.clone()).ok().map(|sid| {
                    use crate::adapter::{ResponseProvenance, SourceId as SId};
                    crate::adapter::ExpertResponse {
                        expert_id: sid.clone(),
                        prose: r.content.clone(),
                        provenance: ResponseProvenance {
                            source_id: sid,
                            title: r.source_id.clone(), // no title from RetrievedContext
                            year: None,
                            authors: vec![],
                            credibility_tier: alzina_search::CredibilityTier::Unknown,
                        },
                    }
                })
            })
            .collect();

        // Decision 0: v3 = v2.
        let patch_prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                let synthesis_xml = format!("<synthesis><narrative>{}</narrative></synthesis>", draft.narrative);
                let gap_desc = gaps.iter().map(|g| g.description.as_str()).collect::<Vec<_>>().join("; ");
                crate::ttd::prompts::lit_review::render_synthesis_gap_resolve_patch_v2(
                    &synthesis_xml,
                    &gap_desc,
                    &retrieved_as_expert,
                    "500-800",
                )
            }
            PromptProfile::V1Delphi => {
                render_synthesis_gap_resolve_patch(&SynthesisGapResolveInput {
                    draft,
                    gaps,
                    retrieved,
                    fitness_feedback: None,
                })
            }
        };

        let patch_output = executor
            .execute(&agent_id, &patch_prompt, &self.model, "synthesis_gap_resolve_patch")
            .await
            .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

        match apply_synthesis_patch(draft, &patch_output) {
            Ok(refined) => return Ok(refined),
            Err(patch_err) => {
                tracing::debug!(
                    error = %patch_err,
                    "SynthesisGapResolve: patch failed — trying full-regen fallback"
                );
            }
        }

        // Tier 2: full regeneration.
        // Decision 0: v3 = v2.
        let full_prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                let synthesis_xml = format!("<synthesis><narrative>{}</narrative></synthesis>", draft.narrative);
                let gap_desc = gaps.iter().map(|g| g.description.as_str()).collect::<Vec<_>>().join("; ");
                crate::ttd::prompts::lit_review::render_synthesis_gap_resolve_full_v2(
                    &synthesis_xml,
                    &gap_desc,
                    &retrieved_as_expert,
                    "500-800",
                )
            }
            PromptProfile::V1Delphi => {
                render_synthesis_gap_resolve(&SynthesisGapResolveInput {
                    draft,
                    gaps,
                    retrieved,
                    fitness_feedback: None,
                })
            }
        };

        let full_output = executor
            .execute(&agent_id, &full_prompt, &self.model, "synthesis_gap_resolve")
            .await
            .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

        match parse_synthesis_xml(&full_output, &self.model, &draft.prompt_version, self.profile) {
            Ok(refined) => Ok(refined),
            Err(full_err) => {
                tracing::debug!(
                    error = %full_err,
                    "SynthesisGapResolve: full-regen failed — using heuristic_resolve"
                );
                // Tier 3: heuristic fallback
                Ok(self.heuristic_resolve(draft, retrieved))
            }
        }
    }
}

// ── SynthesisMerger ───────────────────────────────────────────────────────────

/// Stage-2 merger: folds sorted candidates into one synthesis via a governed spawn.
///
/// Receives candidates in best-first order from `sort_candidates_best_first`.
/// Calls the `Synthesise` merger prompt to fold them into the final synthesis.
pub struct SynthesisMerger {
    pub agent_id: String,
    pub model: String,
    pub prompt_version: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
    /// F14: Stage-1 graph, used to resolve each candidate's cited `node_refs`
    /// into the verified-quote evidence the merger copies from. `None` keeps the
    /// pre-F14 behaviour (merger folds candidates with no node evidence).
    pub graph: Option<crate::ttd::artifact::ArgumentationGraph>,
    /// Depth-probe B: stage-2 panel prose as `(expert_id, prose)`. Lets the
    /// merger widen each node's one-sentence quote to its enclosing `## section`
    /// (mechanism context). Empty keeps the pre-B node-quote-only evidence.
    pub panel_prose: Vec<(String, String)>,
    /// Stage-2 credibility soft-filter: `source_id -> tier`, derived once from the
    /// panel provenance. Tags each source header in the merger evidence so Opus
    /// down-weights weak sources on its own. Empty keeps the untagged baseline.
    pub tier_map: std::collections::BTreeMap<String, alzina_search::CredibilityTier>,
}

impl SynthesisMerger {
    pub fn new(
        agent_id: impl Into<String>,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            prompt_version: prompt_version.into(),
            profile: PromptProfile::V1Delphi,
            graph: None,
            panel_prose: Vec::new(),
            tier_map: std::collections::BTreeMap::new(),
        }
    }

    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// F14: inject the Stage-1 graph so the merger can resolve cited node ids
    /// into verified-quote evidence (the relevant sections it copies from).
    pub fn with_graph(mut self, graph: Option<crate::ttd::artifact::ArgumentationGraph>) -> Self {
        self.graph = graph;
        self
    }

    /// Depth-probe B: inject the stage-2 panel so the merger can widen each
    /// node's one-sentence quote to its enclosing `## section` (mechanism
    /// context). Stores only `(expert_id, prose)`; an empty panel keeps the
    /// pre-B node-quote-only evidence.
    pub fn with_panel(mut self, panel: &[crate::adapter::ExpertResponse]) -> Self {
        self.panel_prose = panel
            .iter()
            .map(|r| (r.expert_id.as_str().to_string(), r.prose.clone()))
            .collect();
        // Stage-2 soft-filter: derive the tier per source once from provenance.
        // The renderer skips `Unknown`, so carrying every entry here is harmless.
        self.tier_map = panel
            .iter()
            .map(|r| {
                (
                    r.expert_id.as_str().to_string(),
                    r.provenance.credibility_tier,
                )
            })
            .collect();
        self
    }
}

/// F14 (option B, probe-23 follow-up): build the merger's quote evidence from
/// the FULL Stage-1 graph, not from draft-emitted `node_refs`.
///
/// Probe 23 proved the cheap haiku drafts emit zero `node_refs`, so evidence
/// built from cited ids was always empty. Instead, render every DB-verified,
/// quote-bearing graph node — grouped by source paper — as the verbatim
/// evidence base. The Opus merger (capable of the attribution haiku is not)
/// selects, per claim, the node(s) under the claim's cited sources and copies a
/// verbatim quote, tagging the node id.
///
/// Only `verification_status == "verified"` nodes with a non-empty quote are
/// included: their quotes are guaranteed verbatim-in-source, so a copied quote
/// survives the post-process re-verify. Returns empty when no such node exists.
///
/// Depth-probe B superseded both production call sites with
/// [`build_graph_evidence_with_sections`] (which equals this output when its
/// panel is empty). Retained as the node-anchors-only byte-identity baseline the
/// section builder is tested against.
#[allow(dead_code)]
pub(crate) fn build_graph_evidence(graph: &crate::ttd::artifact::ArgumentationGraph) -> String {
    use std::collections::BTreeMap;
    let mut by_source: BTreeMap<&str, Vec<&crate::ttd::artifact::GraphNode>> = BTreeMap::new();
    for node in &graph.nodes {
        let has_quote = node.quote.as_deref().map_or(false, |q| !q.trim().is_empty());
        let verified = node.verification_status.as_deref() == Some("verified");
        if has_quote && verified {
            by_source.entry(node.expert_id.as_str()).or_default().push(node);
        }
    }
    let mut out = String::new();
    for (src, nodes) in &by_source {
        out.push_str(&format!("### {src}\n"));
        for node in nodes {
            out.push_str(&format!(
                "- node `{id}`\n  claim: {claim}\n  quote: \"{quote}\"\n",
                id = node.id,
                claim = node.claim.trim(),
                quote = node.quote.as_deref().unwrap_or("").trim(),
            ));
        }
    }
    out
}

/// Depth-probe B (2026-06-16): widen the merger/revision evidence from a single
/// one-sentence node quote to the `## section` each quote sits in.
///
/// The one-sentence quote is verbatim-guaranteed but field-level — it tells the
/// merger WHAT a paper claims, never HOW the approach works. The enclosing
/// section carries the mechanism: the architecture, the ablation that isolates
/// the contribution, the assumption. Feeding it lets the merger author
/// idea-level claims a PhD reader can grep, while the node quote stays the tagged
/// verbatim anchor (post-process still verifies it against the same panel prose,
/// so the fidelity invariant is untouched).
///
/// `panel_prose` is `(expert_id, prose)` from the stage-2 panel — the same prose
/// post-process verifies against. Sections are located by exact quote-substring
/// match, deduped per paper, and bounded ([`MAX_SECTIONS_PER_SOURCE`] sections,
/// [`MAX_SECTION_CHARS`] chars each) to keep the Opus merger input in budget.
/// With an empty `panel_prose` the output is byte-identical to
/// [`build_graph_evidence`] — the section loop simply adds nothing.
pub(crate) fn build_graph_evidence_with_sections(
    graph: &crate::ttd::artifact::ArgumentationGraph,
    panel_prose: &[(String, String)],
    tier_map: &std::collections::BTreeMap<String, alzina_search::CredibilityTier>,
) -> String {
    use std::collections::BTreeMap;
    /// Per-section char cap — generous enough to hold a method or ablation
    /// paragraph, bounded enough that ~10 papers stay within the merger budget.
    const MAX_SECTION_CHARS: usize = 3000;
    /// Sections per source. Verified quotes usually land in 1-2 sections; the cap
    /// guards against a pathological paper dumping its whole body.
    const MAX_SECTIONS_PER_SOURCE: usize = 3;

    let mut by_source: BTreeMap<&str, Vec<&crate::ttd::artifact::GraphNode>> = BTreeMap::new();
    for node in &graph.nodes {
        let has_quote = node.quote.as_deref().map_or(false, |q| !q.trim().is_empty());
        let verified = node.verification_status.as_deref() == Some("verified");
        if has_quote && verified {
            by_source.entry(node.expert_id.as_str()).or_default().push(node);
        }
    }
    let mut out = String::new();
    for (src, nodes) in &by_source {
        // Soft-filter (visibility only): tag the source header with its mechanical
        // credibility tier so the merger down-weights weak sources on its own.
        // `Unknown` renders no tag — absence of a signal is not a judgment, and an
        // empty `tier_map` keeps the output byte-identical to the untagged baseline.
        let tier = tier_map
            .get(*src)
            .filter(|t| **t != alzina_search::CredibilityTier::Unknown);
        match tier {
            Some(t) => out.push_str(&format!("### {src} — {tier}\n", tier = t.label())),
            None => out.push_str(&format!("### {src}\n")),
        }
        // Node anchors: verbatim quote + id, exactly as build_graph_evidence.
        for node in nodes {
            out.push_str(&format!(
                "- node `{id}`\n  claim: {claim}\n  quote: \"{quote}\"\n",
                id = node.id,
                claim = node.claim.trim(),
                quote = node.quote.as_deref().unwrap_or("").trim(),
            ));
        }
        // Section context: the prose sections that contain this source's quotes.
        let Some((_, prose)) = panel_prose.iter().find(|(id, _)| id.as_str() == *src) else {
            continue;
        };
        let sections = crate::ttd::plan::split_sections(prose);
        let mut included: Vec<String> = Vec::new();
        for (heading, body) in &sections {
            if included.len() >= MAX_SECTIONS_PER_SOURCE {
                break;
            }
            if included.contains(heading) {
                continue;
            }
            let contains_quote = nodes.iter().any(|n| {
                n.quote.as_deref().map_or(false, |q| {
                    let q = q.trim();
                    !q.is_empty() && body.contains(q)
                })
            });
            if contains_quote {
                let ctx: String = body.trim().chars().take(MAX_SECTION_CHARS).collect();
                out.push_str(&format!(
                    "\n  context section \"{heading}\":\n  {ctx}\n",
                    heading = heading.trim(),
                    ctx = ctx,
                ));
                included.push(heading.clone());
            }
        }
    }
    out
}

#[async_trait]
impl Merger<SynthesisArtifact> for SynthesisMerger {
    async fn merge(
        &self,
        candidates: &[SynthesisArtifact],
        executor: &Arc<dyn AgentExecutor>,
        _config: &TtdConfig,
    ) -> Result<SynthesisArtifact, TtdError> {
        use crate::ttd::prompts::synthesis::{render_synthesis_merger, SynthesisMergerInput};
        use alzina_core::identity::AgentId;

        if candidates.is_empty() {
            return Err(TtdError::NoCandidates);
        }

        // B2: fork prompt on profile — v2 uses lit_review::render_synthesis_merger_v2.
        // Decision 0: v3 = v2 (Stage-2 merger; the 500-800 target is the synthesis
        // narrative length, NOT the Stage-3 constraint — untouched by Phase 0).
        let prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                let candidate_refs: Vec<&SynthesisArtifact> = candidates.iter().collect();
                // F14 (option B): give the Opus merger the FULL graph's verified
                // quotes — it selects the relevant node per claim and authors the
                // verbatim quote. Drafts no longer need to cite node ids (haiku
                // does not — probe 23).
                // Depth-probe B: section-widened evidence. With an empty panel
                // this is byte-identical to build_graph_evidence (node anchors
                // only) — the F14 fidelity path is unchanged.
                let node_evidence = self
                    .graph
                    .as_ref()
                    .map(|g| {
                        build_graph_evidence_with_sections(g, &self.panel_prose, &self.tier_map)
                    })
                    .unwrap_or_default();
                // Diagnostic: how much verified-quote evidence reached the merger,
                // and how many draft node_refs survived (now informational only —
                // the merger no longer depends on them).
                {
                    let cited_node_refs: usize = candidates
                        .iter()
                        .flat_map(|c| c.claims.iter())
                        .map(|cl| cl.node_refs.len())
                        .sum();
                    let (graph_nodes, verified_quote_nodes) = match self.graph.as_ref() {
                        Some(g) => (
                            g.nodes.len(),
                            g.nodes
                                .iter()
                                .filter(|n| {
                                    n.verification_status.as_deref() == Some("verified")
                                        && n.quote.as_deref().map_or(false, |q| !q.trim().is_empty())
                                })
                                .count(),
                        ),
                        None => (0, 0),
                    };
                    tracing::info!(
                        target: "ttd_perf",
                        n_candidates = candidates.len(),
                        graph_nodes,
                        verified_quote_nodes,
                        node_evidence_chars = node_evidence.len(),
                        draft_cited_node_refs = cited_node_refs,
                        "ttd_perf: F14 merger graph-evidence (option B — full graph to Opus)"
                    );
                }
                crate::ttd::prompts::lit_review::render_synthesis_merger_v2(
                    &candidate_refs,
                    "Merge synthesis candidates",
                    "500-800",
                    &node_evidence,
                    &self.tier_map,
                )
            }
            PromptProfile::V1Delphi => {
                render_synthesis_merger(&SynthesisMergerInput { candidates })
            }
        };
        let agent_id = AgentId::new(self.agent_id.as_str());

        let output = executor
            .execute(&agent_id, &prompt, &self.model, "synthesis_merger")
            .await
            .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

        parse_synthesis_xml(&output, &self.model, &self.prompt_version, self.profile)
    }
}

// ── SynthesisEvalFitness ───────────────────────────────────────────────────────

/// Stage-2 fitness evaluation: sequential judge spawns per evaluate() call.
///
/// Profile fork:
/// - `V1Delphi` (default): 6 v1 dims; `is_valid_synthesis`; `SYNTHESIS_WEIGHTS`. Byte-identical to pre-B3.
/// - `V2LitReview`: 5 v2 lit-review dims; `is_valid_v2`; `V2_SYNTHESIS_WEIGHTS`; synthesis traceability veto.
///
/// Cross-trajectory concurrency is capped externally by the `max_concurrent_fitness_evals`
/// semaphore in TtdMachine::run() (A5 rung 4). Returns a `FitnessEval` with `Option<u8>` per dim.
pub struct SynthesisEvalFitness {
    pub agent_id: String,
    pub model: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
    /// Known panel expert-id set. Threaded at machine build time (stage-2 panel).
    /// Used by `traceability_veto_synthesis` on the v2 arm only. Empty default is
    /// safe because the shape lane (arxiv:/s2:) covers non-panel paper sources.
    pub panel_ids: std::collections::HashSet<String>,
}

impl SynthesisEvalFitness {
    pub fn new(agent_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            profile: PromptProfile::V1Delphi, // default: backward compat
            panel_ids: std::collections::HashSet::new(),
        }
    }

    /// Set the prompt/schema profile (consuming builder).
    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set the panel expert-id set (consuming builder).
    /// Call at machine build time with the stage-2 panel ids (F13 allowlist).
    pub fn with_panel_ids(mut self, panel_ids: std::collections::HashSet<String>) -> Self {
        self.panel_ids = panel_ids;
        self
    }
}

#[async_trait]
impl EvalFitness<SynthesisArtifact> for SynthesisEvalFitness {
    async fn evaluate(
        &self,
        draft: &SynthesisArtifact,
        executor: &Arc<dyn AgentExecutor>,
        _config: &TtdConfig,
    ) -> Result<FitnessEval, TtdError> {
        use alzina_core::identity::AgentId;

        let agent_id = AgentId::new(self.agent_id.as_str());

        match self.profile {
            // Decision 0: v3 scores with the v2 judges unchanged (no judge changes).
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                // v2 path: 5 dims from V2_JUDGE_DIMS, anchored lit-review prompts.
                use crate::ttd::prompts::lit_review::render_fitness_judge_v2_synthesis;
                use crate::ttd::term_sheet::V2_JUDGE_DIMS;

                let mut scores: Vec<(String, Option<u8>)> = Vec::with_capacity(5);

                // WR-05: degrade a failed spawn to None, do NOT abort.
                for dim in &V2_JUDGE_DIMS {
                    let prompt = render_fitness_judge_v2_synthesis(dim, draft);
                    let score = match executor.execute(&agent_id, &prompt, &self.model, dim.name).await {
                        Ok(output) => parse_fitness_score(&output),
                        Err(e) => {
                            tracing::debug!(
                                dimension = dim.name,
                                error = %e,
                                "SynthesisEvalFitness (v2): judge spawn failed — score=None"
                            );
                            None
                        }
                    };
                    scores.push((dim.name.to_string(), score));
                }

                // Deterministic traceability veto — computed on the draft structure,
                // NOT via an LLM judge. Attached before returning (T-B3-01 closure).
                // Vetoed candidates still get judge scores and feedback — the denoise
                // loop can repair them (veto bites at selection, matching "sort only,
                // never terminate" semantics).
                // F13: pass panel_ids so the allowlist covers panel-member expert ids
                // (shape lane handles non-panel arxiv:/s2: ids without panel data).
                let veto = traceability_veto_synthesis(draft, &self.panel_ids);
                let eval = FitnessEval::new(scores);
                Ok(if let Some(reason) = veto { eval.with_veto(reason) } else { eval })
            }

            PromptProfile::V1Delphi => {
                // v1 path: 6 dims, existing v1 prompts (byte-identical to pre-B3).
                use crate::ttd::prompts::synthesis::{
                    render_synthesis_fitness_completeness,
                    render_synthesis_fitness_dissent_visibility,
                    render_synthesis_fitness_faithfulness,
                    render_synthesis_fitness_neutrality,
                    render_synthesis_fitness_structural_clarity,
                    render_synthesis_fitness_traceability,
                    SynthesisFitnessInput,
                };

                let input = SynthesisFitnessInput { draft };

                let dimensions: &[(&str, fn(&SynthesisFitnessInput) -> String)] = &[
                    ("faithfulness", |i| render_synthesis_fitness_faithfulness(i)),
                    ("completeness", |i| render_synthesis_fitness_completeness(i)),
                    ("traceability", |i| render_synthesis_fitness_traceability(i)),
                    ("neutrality", |i| render_synthesis_fitness_neutrality(i)),
                    ("dissent_visibility", |i| render_synthesis_fitness_dissent_visibility(i)),
                    ("structural_clarity", |i| render_synthesis_fitness_structural_clarity(i)),
                ];

                let mut scores: Vec<(String, Option<u8>)> = Vec::with_capacity(6);

                for (dim_name, render_fn) in dimensions {
                    let prompt = render_fn(&input);
                    let result = executor
                        .execute(&agent_id, &prompt, &self.model, dim_name)
                        .await;

                    let score = match result {
                        Ok(output) => parse_fitness_score(&output),
                        Err(e) => {
                            tracing::debug!(
                                dimension = *dim_name,
                                error = %e,
                                "SynthesisEvalFitness: judge spawn failed — score=None (parse failure)"
                            );
                            None
                        }
                    };
                    scores.push((dim_name.to_string(), score));
                }

                // v1 path: no veto (None by default in FitnessEval::new).
                Ok(FitnessEval::new(scores))
            }
        }
    }

    fn validity_fn(&self) -> fn(&FitnessEval) -> bool {
        match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => is_valid_v2,
            PromptProfile::V1Delphi => is_valid_synthesis,
        }
    }

    fn weights(&self) -> &'static [(&'static str, f32)] {
        match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => V2_SYNTHESIS_WEIGHTS,
            PromptProfile::V1Delphi => SYNTHESIS_WEIGHTS,
        }
    }
}

// ── XML parsing helpers ───────────────────────────────────────────────────────

/// Parse a `<synthesis>` XML block from LLM output into a `SynthesisArtifact`.
///
/// The `profile` parameter selects the active dialect:
/// - `V1Delphi`: byte-for-byte existing behaviour; `claim_agreement` pre-initialised
///   to "divided" and normalised to "divided" for out-of-vocab (the all-divided
///   mechanism — kept as-is for v1).
/// - `V2LitReview`: no fabricated "divided"; out-of-vocab `support_level` → None
///   + warn; `agreement_level` set ONLY when explicit child/attribute is in v1 vocab.
///   Also accepts v2 child elements (`<support_level>`, `<evidence_grade>`,
///   `<method>`, `<year>`, `<lineage>`) and a `<gaps>` section.
///
/// Both dialects accept the new v2 child elements — v1 models never emit them so
/// acceptance costs nothing; the values are stored on v2 fields if present.
///
/// # Security note (T-B1-01 / T-B1-02)
///
/// `support_level` passes closed-vocabulary normalisation; all free-string fields
/// are stored verbatim and never interpolated into SQL or shell. T-B1-02
/// acceptance: v2 removes the v1 deterministic-override design (T-23-08); the
/// named closure path is B3's traceability VETO.
pub(crate) fn parse_synthesis_xml(
    output: &str,
    model: &str,
    prompt_version: &str,
    profile: PromptProfile,
) -> Result<SynthesisArtifact, TtdError> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    // Extract the <synthesis> block from noisy LLM output.
    let xml_block = extract_xml_block(output, "synthesis").ok_or_else(|| {
        TtdError::ParseFailed("no <synthesis> block in LLM output".into())
    })?;

    let mut reader = Reader::from_str(&xml_block);
    reader.trim_text(true);

    let mut artifact = SynthesisArtifact {
        schema_version: "1.0".into(),
        study_id: String::new(),
        round_id: String::new(),
        question_id: String::new(),
        generated_at: chrono::Utc::now(),
        model: model.to_string(),
        prompt_version: prompt_version.to_string(),
        code_version: crate::ttd::artifact::code_version(),
        claims: vec![],
        areas_of_agreement: vec![],
        areas_of_disagreement: vec![],
        uncertainties: vec![],
        minority_reports: vec![],
        narrative: String::new(),
        narrative_statements: vec![],
        gaps: vec![],
    };

    let mut buf = Vec::new();
    let mut current_section = String::new();
    let mut current_text = String::new();
    let mut in_narrative = false;
    let mut in_claim = false;
    // V1: pre-initialised to "divided" (existing mechanism preserved).
    // V2: None initially; set only on explicit child/attribute.
    let mut claim_agreement: Option<String> = None;
    let mut claim_sources: Vec<String> = vec![];
    let mut claim_counterarguments: Vec<String> = vec![];
    let mut in_claim_text = false;
    let mut claim_text = String::new();
    let mut in_claim_agreement = false;
    let mut in_counterargument = false;
    let mut counter_text = String::new();
    // v2 per-claim fields
    let mut claim_support_level: Option<String> = None;
    let mut claim_evidence_grade: Option<String> = None;
    let mut claim_method: Option<String> = None;
    let mut claim_year: Option<String> = None;
    let mut claim_lineage: Option<String> = None;
    let mut in_v2_field: Option<&'static str> = None; // tag name for active v2 field
    let mut v2_field_text = String::new();
    // v2 per-claim quotes (worklist item 4): <quotes><quote source="...">text</quote></quotes>
    let mut claim_quotes: Vec<crate::ttd::artifact::ClaimQuote> = vec![];
    let mut in_claim_quote = false;
    let mut claim_quote_source = String::new();
    let mut claim_quote_text = String::new();
    let mut claim_quote_node: Option<String> = None;
    // F14 node-cited drafts: <node_refs><ref>node_id</ref></node_refs>
    let mut claim_node_refs: Vec<String> = vec![];
    let mut in_claim_ref = false;
    let mut claim_ref_text = String::new();
    // v2 gaps section
    let mut in_gaps = false;
    let mut in_gap = false;
    let mut gap_type: Option<String> = None;
    let mut gap_text = String::new();
    let mut in_gap_text = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "narrative" => { in_narrative = true; current_text.clear(); }
                    "claim" => {
                        in_claim = true;
                        claim_text.clear();
                        // V1: pre-initialise to "divided" (the existing all-divided mechanism).
                        // V2: None — set only if the model provides an explicit value.
                        claim_agreement = match profile {
                            PromptProfile::V1Delphi => Some("divided".into()),
                            // Decision 0: v3 parses as v2 (schema_version stays 2.0).
                            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => None,
                        };
                        claim_sources.clear();
                        claim_counterarguments.clear();
                        claim_quotes.clear();
                        claim_node_refs.clear();
                        claim_support_level = None;
                        claim_evidence_grade = None;
                        claim_method = None;
                        claim_year = None;
                        claim_lineage = None;
                        // The aggregator_revision format carries agreement as a
                        // `<claim agreement="...">` ATTRIBUTE (not the
                        // `<agreement_level>` child the diffusion format uses).
                        // Read it as the initial value; a child element still
                        // overrides it below.
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"agreement" {
                                if let Ok(v) = std::str::from_utf8(&attr.value) {
                                    claim_agreement = Some(v.trim().to_string());
                                }
                            }
                        }
                    }
                    "text" if in_claim => { in_claim_text = true; claim_text.clear(); }
                    "agreement_level" if in_claim => { in_claim_agreement = true; }
                    "source" if in_claim => {
                        // Two source dialects exist in the prompt set:
                        // diffusion format `<source id="...">`, revision format
                        // `<source expert_id="...">`. Accept both.
                        for attr in e.attributes().flatten() {
                            if matches!(attr.key.as_ref(), b"id" | b"expert_id") {
                                if let Ok(v) = std::str::from_utf8(&attr.value) {
                                    claim_sources.push(v.to_string());
                                }
                            }
                        }
                    }
                    "counterargument" if in_claim => {
                        in_counterargument = true;
                        counter_text.clear();
                    }
                    // F14 node-cited drafts: <node_refs><ref>node_id</ref></node_refs>
                    "ref" if in_claim => {
                        in_claim_ref = true;
                        claim_ref_text.clear();
                    }
                    // v2 claim quotes: <quote source="...">verbatim</quote>
                    "quote" if in_claim => {
                        in_claim_quote = true;
                        claim_quote_text.clear();
                        claim_quote_source.clear();
                        claim_quote_node = None;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"source" {
                                if let Ok(v) = std::str::from_utf8(&attr.value) {
                                    claim_quote_source = v.trim().to_string();
                                }
                            } else if attr.key.as_ref() == b"node" {
                                // F14: the merger tags each authored quote with
                                // the graph node id it was copied from.
                                if let Ok(v) = std::str::from_utf8(&attr.value) {
                                    let n = v.trim();
                                    if !n.is_empty() {
                                        claim_quote_node = Some(n.to_string());
                                    }
                                }
                            }
                        }
                    }
                    // v2 per-claim fields — accepted in both dialects (v1 models never emit them).
                    "support_level" if in_claim => {
                        in_v2_field = Some("support_level");
                        v2_field_text.clear();
                    }
                    "evidence_grade" if in_claim => {
                        in_v2_field = Some("evidence_grade");
                        v2_field_text.clear();
                    }
                    "method" if in_claim => {
                        in_v2_field = Some("method");
                        v2_field_text.clear();
                    }
                    "year" if in_claim => {
                        in_v2_field = Some("year");
                        v2_field_text.clear();
                    }
                    "lineage" if in_claim => {
                        in_v2_field = Some("lineage");
                        v2_field_text.clear();
                    }
                    // v2 gaps section
                    "gaps" => { in_gaps = true; }
                    "gap" if in_gaps => {
                        in_gap = true;
                        gap_text.clear();
                        in_gap_text = false;
                        // Read gap_type from the `type` attribute.
                        gap_type = None;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"type" {
                                if let Ok(v) = std::str::from_utf8(&attr.value) {
                                    let v = v.trim().to_string();
                                    if !v.is_empty() { gap_type = Some(v); }
                                }
                            }
                        }
                        in_gap_text = true; // gap body is free text (inline, not child element)
                    }
                    "areas_of_agreement" | "areas_of_disagreement" | "uncertainties" => {
                        current_section = tag.clone();
                    }
                    "area" | "item" | "uncertainty" => {
                        current_text.clear();
                    }
                    _ => {}
                }
            }
            // Self-closing tags (`<source id="x"/>`, `<gap/>`) arrive as Event::Empty,
            // NOT Event::Start — the shape the prompt format examples show.
            // Missing this arm silently dropped every spec-compliant source
            // (probe 10: all claims emitted with sources: []).
            Ok(Event::Empty(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "source" if in_claim => {
                        for attr in e.attributes().flatten() {
                            if matches!(attr.key.as_ref(), b"id" | b"expert_id") {
                                if let Ok(v) = std::str::from_utf8(&attr.value) {
                                    claim_sources.push(v.to_string());
                                }
                            }
                        }
                    }
                    // Self-closing `<gap type="..."/>` — no body text.
                    "gap" if in_gaps => {
                        let mut gt: Option<String> = None;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"type" {
                                if let Ok(v) = std::str::from_utf8(&attr.value) {
                                    let v = v.trim().to_string();
                                    if !v.is_empty() { gt = Some(v); }
                                }
                            }
                        }
                        // Empty gap with no description — skip (no meaningful content).
                        let _ = gt; // gap_type without description is dropped
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                let text = e.unescape().unwrap_or_default().to_string();
                if in_narrative {
                    current_text.push_str(&text);
                } else if in_gap_text && in_gap {
                    gap_text.push_str(&text);
                } else if in_claim_text {
                    claim_text.push_str(&text);
                } else if in_claim_agreement {
                    // claim_agreement receives the text of <agreement_level> child.
                    claim_agreement = Some(text.trim().to_string());
                } else if in_claim_quote {
                    claim_quote_text.push_str(&text);
                } else if in_claim_ref {
                    claim_ref_text.push_str(&text);
                } else if in_v2_field.is_some() {
                    v2_field_text.push_str(&text);
                } else if in_counterargument {
                    counter_text.push_str(&text);
                } else if matches!(current_section.as_str(), "areas_of_agreement" | "areas_of_disagreement" | "uncertainties") {
                    current_text.push_str(&text);
                }
            }
            Ok(Event::End(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "narrative" => {
                        artifact.narrative = current_text.trim().to_string();
                        in_narrative = false;
                        current_text.clear();
                    }
                    "gap" if in_gaps => {
                        let desc = gap_text.trim().to_string();
                        if !desc.is_empty() {
                            artifact.gaps.push(Gap { description: desc, gap_type: gap_type.take() });
                        }
                        in_gap = false;
                        in_gap_text = false;
                        gap_text.clear();
                        gap_type = None;
                    }
                    "gaps" => {
                        in_gaps = false;
                    }
                    "claim" => {
                        if !claim_text.is_empty() {
                            use crate::ttd::artifact::Claim;
                            // Normalise agreement level — dialect-specific.
                            let agreement_level = match profile {
                                PromptProfile::V1Delphi => {
                                    // v1: normalise out-of-vocab to "divided" (existing mechanism).
                                    let raw = claim_agreement.as_deref().unwrap_or("divided");
                                    let level = if matches!(
                                        raw,
                                        "consensus" | "majority" | "divided" | "minority"
                                    ) {
                                        raw.to_string()
                                    } else {
                                        "divided".to_string()
                                    };
                                    Some(level)
                                }
                                PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                                    // v2: preserve explicit in-vocab label if present; otherwise None.
                                    // Never fabricate "divided". Decision 0: v3 parses as v2.
                                    claim_agreement.as_deref().and_then(|raw| {
                                        if matches!(raw, "consensus" | "majority" | "divided" | "minority") {
                                            Some(raw.to_string())
                                        } else {
                                            None
                                        }
                                    })
                                }
                            };
                            // Normalise support_level via closed vocabulary (T-B1-01).
                            let support_level = if let Some(ref raw) = claim_support_level {
                                match normalise_support_level(raw) {
                                    Some(canonical) => Some(canonical.to_string()),
                                    None => {
                                        tracing::warn!(
                                            raw = %raw,
                                            "parse_synthesis_xml: out-of-vocabulary support_level rejected \
                                             (storing None — claim retained)"
                                        );
                                        None
                                    }
                                }
                            } else {
                                None
                            };
                            artifact.claims.push(Claim {
                                text: claim_text.trim().to_string(),
                                agreement_level,
                                sources: claim_sources.clone(),
                                counterarguments: claim_counterarguments.clone(),
                                support_level,
                                evidence_grade: claim_evidence_grade.take(),
                                method: claim_method.take(),
                                year: claim_year.take(),
                                lineage: claim_lineage.take(),
                                quotes: std::mem::take(&mut claim_quotes),
                                node_refs: std::mem::take(&mut claim_node_refs),
                                citation: None,
                            });
                        }
                        in_claim = false;
                        in_claim_text = false;
                        in_claim_agreement = false;
                        in_v2_field = None;
                        claim_support_level = None;
                        v2_field_text.clear();
                    }
                    "text" if in_claim => { in_claim_text = false; }
                    "agreement_level" if in_claim => { in_claim_agreement = false; }
                    "ref" if in_claim => {
                        let r = claim_ref_text.trim().to_string();
                        if !r.is_empty() {
                            claim_node_refs.push(r);
                        }
                        in_claim_ref = false;
                    }
                    "quote" if in_claim => {
                        let q = claim_quote_text.trim().to_string();
                        let src = claim_quote_source.trim().to_string();
                        // Source-less or empty quotes are dropped — a quote
                        // that names no source cannot be verified.
                        if !q.is_empty() && !src.is_empty() {
                            claim_quotes.push(crate::ttd::artifact::ClaimQuote {
                                source: src,
                                text: q,
                                status: None,
                                snapped: false,
                                inherited: false,
                                node_id: claim_quote_node.take(),
                            });
                        }
                        in_claim_quote = false;
                        claim_quote_text.clear();
                        claim_quote_source.clear();
                    }
                    "support_level" if in_claim => {
                        claim_support_level = Some(v2_field_text.trim().to_string());
                        in_v2_field = None;
                        v2_field_text.clear();
                    }
                    "evidence_grade" if in_claim => {
                        let v = v2_field_text.trim().to_string();
                        if !v.is_empty() { claim_evidence_grade = Some(v); }
                        in_v2_field = None;
                        v2_field_text.clear();
                    }
                    "method" if in_claim => {
                        let v = v2_field_text.trim().to_string();
                        if !v.is_empty() { claim_method = Some(v); }
                        in_v2_field = None;
                        v2_field_text.clear();
                    }
                    "year" if in_claim => {
                        let v = v2_field_text.trim().to_string();
                        if !v.is_empty() { claim_year = Some(v); }
                        in_v2_field = None;
                        v2_field_text.clear();
                    }
                    "lineage" if in_claim => {
                        let v = v2_field_text.trim().to_string();
                        if !v.is_empty() { claim_lineage = Some(v); }
                        in_v2_field = None;
                        v2_field_text.clear();
                    }
                    "counterargument" => {
                        if !counter_text.is_empty() {
                            claim_counterarguments.push(counter_text.trim().to_string());
                        }
                        in_counterargument = false;
                        counter_text.clear();
                    }
                    "area" | "item" => {
                        let text = current_text.trim().to_string();
                        if !text.is_empty() {
                            match current_section.as_str() {
                                "areas_of_agreement" => artifact.areas_of_agreement.push(text),
                                "areas_of_disagreement" => artifact.areas_of_disagreement.push(text),
                                "uncertainties" => artifact.uncertainties.push(text),
                                _ => {}
                            }
                        }
                        current_text.clear();
                    }
                    "uncertainty" => {
                        let text = current_text.trim().to_string();
                        if !text.is_empty() {
                            artifact.uncertainties.push(text);
                        }
                        current_text.clear();
                    }
                    "areas_of_agreement" | "areas_of_disagreement" | "uncertainties" => {
                        current_section.clear();
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

    // Provenance drift guard: claims without ANY sources mean the model's
    // XML shape diverged from what this parser captures (or the model omitted
    // sources). Downstream this silently degrades agreement levels and kills
    // traceability — make it loud.
    if !artifact.claims.is_empty()
        && artifact.claims.iter().all(|c| c.sources.is_empty())
    {
        tracing::warn!(
            n_claims = artifact.claims.len(),
            "parse_synthesis_xml: ALL claims parsed with empty sources — \
             provenance lost (model output shape drift or sources omitted)"
        );
    }

    Ok(artifact)
}

/// Extract the first occurrence of `<tag>...</tag>` from noisy LLM output.
pub(crate) fn extract_xml_block(output: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");

    let start = output.find(&open)?;
    let end = output[start..].find(&close).map(|i| start + i + close.len())?;
    Some(output[start..end].to_string())
}

/// Parse `<gaps>` XML into `Vec<IdentifiedGap>`.
///
/// T1 ruled contract: missing or empty `<gaps>` block → `Ok(vec![])` (never
/// `Err`, never a bare `Vec`). A gap is valid iff it has a non-empty
/// `<description>`; `<query>` defaults to the description when absent.
pub(crate) fn parse_gaps_xml(output: &str) -> Result<Vec<IdentifiedGap>, TtdError> {
    let xml_block = match extract_xml_block(output, "gaps") {
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

/// Parse a fitness score from a fitness-judge LLM response.
///
/// WR-08: delegates to the single canonical parser
/// `fitness::parse_fitness_response`. This carries the empty/whitespace → `None`
/// abstain (WR-03) and the integer CLAMP to `[1,5]` (CR-03). One source of truth
/// across all three stages.
pub(crate) fn parse_fitness_score(output: &str) -> Option<u8> {
    crate::ttd::fitness::parse_fitness_response(output).score
}

/// Apply a `<patch>` XML document to a synthesis draft.
///
/// Mirrors `gap_resolve_patch` semantics: add/modify/remove claims.
pub(crate) fn apply_synthesis_patch(
    draft: &SynthesisArtifact,
    patch_output: &str,
) -> Result<SynthesisArtifact, TtdError> {
    // For Phase 23, a patch that can't be parsed returns an error so the
    // full-regen fallback fires. A "<patch/>" empty patch is a no-op success.
    let xml_block = extract_xml_block(patch_output, "patch").ok_or_else(|| {
        TtdError::ParseFailed("no <patch> block in LLM output".into())
    })?;

    // Empty patch (<patch/> or <patch></patch>) is a no-op — return draft unchanged.
    let trimmed = xml_block
        .trim()
        .replace("<patch/>", "")
        .replace("<patch></patch>", "");
    if trimmed.trim().is_empty() {
        return Ok(draft.clone());
    }

    // For non-empty patches, the approach is best-effort: parse <add> blocks
    // and append new claims. Full patch semantics (modify/remove) are complex
    // XML operations; the fallback to full-regen handles the complex cases.
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut refined = draft.clone();
    let mut reader = Reader::from_str(&xml_block);
    reader.trim_text(true);
    let mut buf = Vec::new();
    let mut in_add = false;
    let mut in_add_claim = false;
    let mut claim_text = String::new();
    let mut in_text = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "add" => { in_add = true; }
                    "claim" if in_add => { in_add_claim = true; claim_text.clear(); }
                    "text" if in_add_claim => { in_text = true; }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_text {
                    claim_text.push_str(&e.unescape().unwrap_or_default());
                }
            }
            Ok(Event::End(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "text" if in_add_claim => { in_text = false; }
                    "claim" if in_add => {
                        if !claim_text.is_empty() {
                            use crate::ttd::artifact::Claim;
                            refined.claims.push(Claim {
                                text: claim_text.trim().to_string(),
                                agreement_level: Some("divided".into()),
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
                        }
                        in_add_claim = false;
                        claim_text.clear();
                    }
                    "add" => { in_add = false; }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    Ok(refined)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use crate::adapter::ExpertResponse;
    use crate::adapter::SourceId;
    use crate::executor::AgentExecutor;
    use crate::ttd::artifact::{ArgumentationGraph, GraphNode};
    use crate::ttd::config::TtdConfig;
    use crate::ttd::fitness::FitnessEval;
    use crate::ttd::mod_types::TtdError;
    use crate::ttd::stages::RetrievedContext;
    use crate::ttd::state::IdentifiedGap;

    // ── Mock executor ─────────────────────────────────────────────────────────

    /// Records which tasks were invoked and returns a canned response.
    struct RecordingExecutor {
        response: String,
        invocations: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl AgentExecutor for RecordingExecutor {
        async fn execute(
            &self,
            _agent_id: &alzina_core::identity::AgentId,
            _instruction: &str,
            _model: &str,
            task: &str,
        ) -> alzina_core::AlzinaResult<String> {
            self.invocations.lock().unwrap().push(task.to_string());
            Ok(self.response.clone())
        }
    }

    fn make_synthesis_xml_response() -> String {
        r#"<synthesis>
  <narrative>Test narrative.</narrative>
  <claims>
    <claim id="C1">
      <text>Climate change accelerates permafrost thaw.</text>
      <agreement_level>consensus</agreement_level>
      <sources>
        <source id="arxiv:2105.14103"/>
        <source id="arxiv:2105.14104"/>
      </sources>
      <counterarguments>
        <counterargument>Rate is uncertain.</counterargument>
      </counterarguments>
    </claim>
  </claims>
  <areas_of_agreement>
    <area>Experts agree on the mechanism</area>
  </areas_of_agreement>
  <areas_of_disagreement>
    <area>Rate of change is contested</area>
  </areas_of_disagreement>
  <uncertainties>
    <uncertainty>Long-term feedback loops remain unclear</uncertainty>
  </uncertainties>
</synthesis>"#
            .to_string()
    }

    fn make_graph() -> ArgumentationGraph {
        let mut g = ArgumentationGraph::new("study-1", "round-1", "q-1", "model", "v1/graph");
        g.nodes.push(GraphNode {
            id: "arxiv:2105.14103_c001".into(),
            claim: "Permafrost thaw releases methane.".into(),
            expert_id: "arxiv:2105.14103".into(),
            quote: Some("permafrost thaw".into()),
            verification_status: Some("verified".into()),
        });
        g
    }

    fn make_config() -> TtdConfig {
        TtdConfig::default()
    }

    fn make_expert_response() -> ExpertResponse {
        use crate::adapter::ResponseProvenance;
        ExpertResponse {
            expert_id: SourceId::new("arxiv:2105.14103"),
            prose: "Permafrost thaw accelerates under warming.".into(),
            provenance: ResponseProvenance {
                source_id: SourceId::new("arxiv:2105.14103"),
                title: "Permafrost Thaw Study".into(),
                year: Some(2021),
                authors: vec![],
                credibility_tier: alzina_search::CredibilityTier::Unknown,
            },
        }
    }

    // ── Test 1: use_graph_draft selects draft_graph ───────────────────────────

    /// When graph is Some, the executor is invoked with task="synthesis_draft"
    /// and the prompt contains graph-seeded content. When graph is None, it uses
    /// the plain draft path.
    ///
    /// Both branches must produce a SynthesisArtifact (from mock XML output).
    #[tokio::test]
    async fn use_graph_draft_selects_draft_graph() {
        let invocations = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let executor: Arc<dyn AgentExecutor> = Arc::new(RecordingExecutor {
            response: make_synthesis_xml_response(),
            invocations: invocations.clone(),
        });

        // Branch 1: graph present → use_graph_draft path
        let draft_gen_with_graph = SynthesisDraftGen::new(
            "synth-agent",
            "test-model",
            "v1/synthesis",
            Some(make_graph()),
        );

        let inputs = vec![make_expert_response()];
        let config = make_config();

        let result = draft_gen_with_graph
            .generate(&inputs, &executor, &config, None, None)
            .await
            .expect("generate with graph must succeed");

        assert!(
            !result.claims.is_empty(),
            "parsed synthesis must contain claims"
        );

        let invoked = invocations.lock().unwrap().clone();
        assert!(
            invoked.contains(&"synthesis_draft".to_string()),
            "executor must be called with task=synthesis_draft"
        );

        // Branch 2: graph = None → plain draft path
        invocations.lock().unwrap().clear();
        let draft_gen_no_graph = SynthesisDraftGen::new(
            "synth-agent",
            "test-model",
            "v1/synthesis",
            None, // no graph
        );

        let result_no_graph = draft_gen_no_graph
            .generate(&inputs, &executor, &config, None, None)
            .await
            .expect("generate without graph must succeed");

        assert!(
            !result_no_graph.claims.is_empty(),
            "plain draft must also produce claims"
        );
    }

    /// F14 option B: build_graph_evidence renders ALL verified-quote nodes
    /// grouped by source, independent of any draft node_refs, and skips
    /// unverified / quoteless nodes.
    #[test]
    fn build_graph_evidence_renders_verified_quote_nodes_only() {
        use crate::ttd::artifact::{ArgumentationGraph, GraphNode};
        let mut g = ArgumentationGraph::new("s", "r", "q", "m", "v2/lit-review");
        g.nodes.push(GraphNode {
            id: "arxiv:1_C1".into(),
            claim: "verified claim".into(),
            expert_id: "arxiv:1".into(),
            quote: Some("a verbatim verified passage".into()),
            verification_status: Some("verified".into()),
        });
        g.nodes.push(GraphNode {
            id: "arxiv:1_C2".into(),
            claim: "absent claim".into(),
            expert_id: "arxiv:1".into(),
            quote: Some("this one is not verified".into()),
            verification_status: Some("absent".into()),
        });
        g.nodes.push(GraphNode {
            id: "arxiv:2_C1".into(),
            claim: "quoteless".into(),
            expert_id: "arxiv:2".into(),
            quote: None,
            verification_status: Some("verified".into()),
        });

        let ev = build_graph_evidence(&g);
        assert!(ev.contains("### arxiv:1"), "verified node's source grouped: {ev}");
        assert!(ev.contains("arxiv:1_C1"), "verified node id present: {ev}");
        assert!(ev.contains("a verbatim verified passage"), "verified quote present: {ev}");
        assert!(!ev.contains("not verified"), "absent-status node excluded: {ev}");
        assert!(!ev.contains("arxiv:2"), "quoteless node excluded entirely: {ev}");
    }

    /// Depth-probe B: with panel prose, the evidence widens to the `## section`
    /// holding the verified quote; with an empty panel it is byte-identical to
    /// the node-anchors-only baseline (the F14 fidelity path is unchanged).
    #[test]
    fn build_graph_evidence_with_sections_adds_enclosing_section() {
        use crate::ttd::artifact::{ArgumentationGraph, GraphNode};
        let mut g = ArgumentationGraph::new("s", "r", "q", "m", "v2/lit-review");
        g.nodes.push(GraphNode {
            id: "arxiv:1_C1".into(),
            claim: "the LG/ME ablation".into(),
            expert_id: "arxiv:1".into(),
            quote: Some("removing the ME module degrades multi-hop reasoning".into()),
            verification_status: Some("verified".into()),
        });

        // Panel prose: a labelled method section whose body holds the quote plus
        // the surrounding mechanism the one-sentence quote omits.
        let prose = "## Introduction\n\nWe study memory modules.\n\n## Method\n\n\
            The architecture pairs a latent-graph (LG) module with a \
            memory-encoder (ME) module. In ablation, removing the ME module \
            degrades multi-hop reasoning by 23 points, isolating its contribution."
            .to_string();
        let panel = vec![("arxiv:1".to_string(), prose)];

        let widened = build_graph_evidence_with_sections(&g, &panel, &std::collections::BTreeMap::new());
        assert!(widened.contains("arxiv:1_C1"), "node anchor still present: {widened}");
        assert!(
            widened.contains("removing the ME module degrades multi-hop reasoning"),
            "verbatim quote anchor still present: {widened}"
        );
        assert!(
            widened.contains("context section \"Method\""),
            "enclosing section labelled: {widened}"
        );
        assert!(
            widened.contains("degrades multi-hop reasoning by 23 points"),
            "mechanism prose from the section reaches the merger: {widened}"
        );
        assert!(
            !widened.contains("We study memory modules"),
            "non-matching section excluded: {widened}"
        );

        // Empty panel ⇒ byte-identical to the node-anchors-only baseline.
        assert_eq!(
            build_graph_evidence_with_sections(&g, &[], &std::collections::BTreeMap::new()),
            build_graph_evidence(&g),
            "empty panel must equal the pre-B baseline"
        );
    }

    /// Stage-2 soft-filter: a rated source tags its `### header`; `Unknown` does
    /// not; an absent source does not. Tagging is independent of the section path.
    #[test]
    fn build_graph_evidence_tags_source_with_credibility_tier() {
        use crate::ttd::artifact::{ArgumentationGraph, GraphNode};
        use alzina_search::CredibilityTier;
        use std::collections::BTreeMap;

        let mut g = ArgumentationGraph::new("s", "r", "q", "m", "v2/lit-review");
        for src in ["arxiv:high", "arxiv:unknown", "arxiv:absent"] {
            g.nodes.push(GraphNode {
                id: format!("{src}_C1"),
                claim: "a claim".into(),
                expert_id: src.into(),
                quote: Some(format!("verbatim from {src}")),
                verification_status: Some("verified".into()),
            });
        }

        let mut tier_map = BTreeMap::new();
        tier_map.insert("arxiv:high".to_string(), CredibilityTier::High);
        tier_map.insert("arxiv:unknown".to_string(), CredibilityTier::Unknown);
        // "arxiv:absent" intentionally not in the map.

        let out = build_graph_evidence_with_sections(&g, &[], &tier_map);
        assert!(
            out.contains("### arxiv:high — high credibility"),
            "rated source tagged: {out}"
        );
        assert!(
            out.contains("### arxiv:unknown\n"),
            "Unknown renders no tag: {out}"
        );
        assert!(
            out.contains("### arxiv:absent\n"),
            "absent source renders no tag: {out}"
        );
    }

    // ── Test 2: heuristic_identify fires on no gaps ───────────────────────────

    /// When gap_identify returns no gaps (empty XML or no <gap> elements), the
    /// heuristic fires and produces IdentifiedGap items where description == query.
    #[tokio::test]
    async fn heuristic_identify_fires_on_no_gaps() {
        // Empty <gaps/> response → heuristic fires
        let invocations = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let executor: Arc<dyn AgentExecutor> = Arc::new(RecordingExecutor {
            response: "<gaps/>".to_string(), // no gaps
            invocations: invocations.clone(),
        });

        let gap_identify = SynthesisGapIdentify::new("gap-agent", "test-model");

        // Build a draft with one single-source claim (heuristic should catch it)
        use crate::ttd::artifact::{Claim, SynthesisArtifact};
        let mut draft = SynthesisArtifact::new(
            "study-1", "round-1", "q-1", "model", "v1/synthesis",
        );
        draft.claims.push(Claim {
            text: "Climate change is accelerating".into(),
            agreement_level: Some("consensus".into()),
            sources: vec!["arxiv:2105.14103".into()], // single source
            counterarguments: vec![],                  // no counterargs
            support_level: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        // no uncertainties (another heuristic trigger)

        let config = make_config();
        let fitness = FitnessEval::new(vec![]);

        let gaps = gap_identify
            .identify(&draft, &fitness, &executor, &config)
            .await
            .expect("identify must succeed");

        // Heuristic should have produced gaps
        assert!(
            !gaps.is_empty(),
            "heuristic_identify must produce at least one gap"
        );

        // description == query for heuristic gaps
        for gap in &gaps {
            assert_eq!(
                gap.description, gap.query,
                "heuristic gap: description must equal query (synthesis_tasks.py:473)"
            );
        }
    }

    // ── Test 3: synthesis_gap_resolve_fallback ────────────────────────────────

    /// patch → full-regen → heuristic three-tier fallback.
    /// Tier 1 (patch): returns invalid XML → falls through.
    /// Tier 2 (full-regen): also returns invalid XML → falls through.
    /// Tier 3 (heuristic): returns draft with retrieved IDs added.
    #[tokio::test]
    async fn synthesis_gap_resolve_fallback() {
        let invocations = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        // Both spawns return invalid XML so both tiers 1 and 2 fail
        let executor: Arc<dyn AgentExecutor> = Arc::new(RecordingExecutor {
            response: "not xml".to_string(),
            invocations: invocations.clone(),
        });

        let gap_resolve = SynthesisGapResolve::new("resolve-agent", "test-model");

        use crate::ttd::artifact::{Claim, SynthesisArtifact};
        let mut draft = SynthesisArtifact::new(
            "study-1", "round-1", "q-1", "model", "v1/synthesis",
        );
        draft.claims.push(Claim {
            text: "Permafrost thaw is accelerating".into(),
            agreement_level: Some("majority".into()),
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

        let gaps = vec![IdentifiedGap {
            description: "Find more evidence".into(),
            query: "permafrost thaw evidence".into(),
        }];

        let retrieved = vec![RetrievedContext {
            source_id: "arxiv:9999.00001".into(),
            content: "Additional evidence on permafrost thaw.".into(),
            section: None,
        }];

        let config = make_config();

        let result = gap_resolve
            .resolve(&draft, &FitnessEval::new(vec![]), &gaps, &retrieved, &executor, &config)
            .await
            .expect("resolve with all-fail tiers must still succeed via heuristic");

        // Heuristic resolve adds the retrieved source ID to single-source claims
        let claim = &result.claims[0];
        assert!(
            claim.sources.contains(&"arxiv:9999.00001".to_string()),
            "heuristic_resolve must add retrieved source IDs to single-source claims"
        );

        // Two spawns must have been invoked (patch + full-regen), then heuristic
        let invoked = invocations.lock().unwrap().clone();
        assert!(
            invoked.contains(&"synthesis_gap_resolve_patch".to_string()),
            "tier 1: patch spawn must be attempted"
        );
        assert!(
            invoked.contains(&"synthesis_gap_resolve".to_string()),
            "tier 2: full-regen spawn must be attempted"
        );
    }

    // ── Test 4: heuristic_gap_description_doubles_as_query ───────────────────

    /// The heuristic_identify method's gaps have description == query.
    #[test]
    fn heuristic_gap_description_doubles_as_query() {
        let gap_identify = SynthesisGapIdentify::new("gap-agent", "test-model");

        use crate::ttd::artifact::{Claim, SynthesisArtifact};
        let mut draft = SynthesisArtifact::new(
            "study-1", "round-1", "q-1", "model", "v1/synthesis",
        );
        draft.claims.push(Claim {
            text: "Test claim with single source".into(),
            agreement_level: Some("consensus".into()),
            sources: vec!["s001".into()], // single source → triggers heuristic
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

        let gaps = gap_identify.heuristic_identify(&draft);
        assert!(!gaps.is_empty(), "heuristic must produce gaps for single-source claim");
        for gap in &gaps {
            assert_eq!(
                gap.description, gap.query,
                "description must equal query for heuristic gaps"
            );
        }
    }

    // ── Test 5: parse_synthesis_xml round-trip ────────────────────────────────

    #[test]
    fn parse_synthesis_xml_extracts_claims() {
        let xml = make_synthesis_xml_response();
        let artifact = parse_synthesis_xml(&xml, "test-model", "v1/synthesis", PromptProfile::V1Delphi)
            .expect("parse must succeed on valid XML");

        assert_eq!(artifact.claims.len(), 1);
        assert_eq!(artifact.claims[0].text, "Climate change accelerates permafrost thaw.");
        assert_eq!(artifact.claims[0].agreement_level.as_deref(), Some("consensus"));
        assert_eq!(artifact.areas_of_agreement.len(), 1);
        assert_eq!(artifact.areas_of_disagreement.len(), 1);
        assert_eq!(artifact.uncertainties.len(), 1);
    }

    // ── Probe-10 provenance regression: source tag dialects ──────────────────

    /// Self-closing `<source id="x"/>` (the diffusion prompt format example)
    /// arrives as Event::Empty and MUST be captured. Probe 10 shipped every
    /// claim with sources: [] because this arm was missing.
    #[test]
    fn parse_self_closing_source_ids() {
        let xml = r#"<synthesis>
  <claims>
    <claim id="C1">
      <text>Permafrost thaw releases methane.</text>
      <agreement_level>consensus</agreement_level>
      <sources>
        <source id="arxiv:2105.14103"/>
        <source id="s2:abc123"/>
      </sources>
    </claim>
  </claims>
</synthesis>"#;
        let artifact = parse_synthesis_xml(xml, "test-model", "v1/synthesis", PromptProfile::V1Delphi)
            .expect("parse must succeed");
        assert_eq!(artifact.claims.len(), 1);
        assert_eq!(
            artifact.claims[0].sources,
            vec!["arxiv:2105.14103".to_string(), "s2:abc123".to_string()],
            "self-closing <source id=.../> must be captured (Event::Empty arm)"
        );
    }

    /// Worklist item 4 (quote-grounded synthesis): v2 claim quotes
    /// `<quotes><quote source="...">text</quote></quotes>` parse into
    /// `Claim.quotes` with status None (verification stamps later).
    /// Source-less quotes are dropped — unverifiable.
    #[test]
    fn parse_v2_claim_quotes() {
        let xml = r#"<synthesis>
  <claims>
    <claim id="C1">
      <text>Permafrost thaw releases methane.</text>
      <sources><source id="arxiv:2105.14103"/></sources>
      <quotes>
        <quote source="arxiv:2105.14103">observed methane flux increased by 38%</quote>
        <quote>source-less quote must be dropped</quote>
      </quotes>
    </claim>
  </claims>
</synthesis>"#;
        let artifact =
            parse_synthesis_xml(xml, "test-model", "v2/lit-review", PromptProfile::V2LitReview)
                .expect("parse must succeed");
        assert_eq!(artifact.claims.len(), 1);
        let quotes = &artifact.claims[0].quotes;
        assert_eq!(quotes.len(), 1, "source-less quote must be dropped");
        assert_eq!(quotes[0].source, "arxiv:2105.14103");
        assert_eq!(quotes[0].text, "observed methane flux increased by 38%");
        assert_eq!(quotes[0].status, None, "status stamps only at verification");
    }

    /// F14: node-cited drafts emit `<node_refs><ref>node_id</ref></node_refs>`
    /// per claim instead of authored quotes. The parser must capture them into
    /// `Claim.node_refs` so the Opus merger and the deterministic floor can
    /// resolve each claim's supporting graph nodes by exact id.
    #[test]
    fn parse_v2_claim_node_refs() {
        let xml = r#"<synthesis>
  <claims>
    <claim id="C1">
      <text>Abrupt thaw outpaces gradual thaw.</text>
      <sources><source id="arxiv:2105.14103"/></sources>
      <node_refs><ref>p001_c1</ref><ref>p004_c2</ref><ref>  </ref></node_refs>
    </claim>
  </claims>
</synthesis>"#;
        let artifact =
            parse_synthesis_xml(xml, "test-model", "v2/lit-review", PromptProfile::V2LitReview)
                .expect("parse must succeed");
        assert_eq!(artifact.claims.len(), 1);
        assert_eq!(
            artifact.claims[0].node_refs,
            vec!["p001_c1".to_string(), "p004_c2".to_string()],
            "non-empty <ref> node ids captured in order; blank refs dropped"
        );
        assert!(
            artifact.claims[0].quotes.is_empty(),
            "node-cited drafts author no quotes"
        );
    }

    /// The aggregator_revision output format uses `<claim agreement="...">`
    /// (attribute) and `<source expert_id="...">` (different attribute name).
    /// The parser must speak this dialect too — the revision output REPLACES
    /// the synthesis, so missing it launders provenance and agreement away.
    #[test]
    fn parse_revision_dialect_sources_and_agreement() {
        let xml = r#"<synthesis>
  <claims>
    <claim id="C1" agreement="majority" corroboration_count="3">
      <text>Permafrost thaw releases methane.</text>
      <sources>
        <source expert_id="arxiv:2105.14103">
          <quote>permafrost thaw releases significant methane</quote>
        </source>
      </sources>
    </claim>
  </claims>
</synthesis>"#;
        let artifact = parse_synthesis_xml(xml, "test-model", "v1/synthesis", PromptProfile::V1Delphi)
            .expect("parse must succeed");
        assert_eq!(artifact.claims.len(), 1);
        assert_eq!(
            artifact.claims[0].sources,
            vec!["arxiv:2105.14103".to_string()],
            "revision-format <source expert_id=...> must be captured"
        );
        assert_eq!(
            artifact.claims[0].agreement_level.as_deref(),
            Some("majority"),
            "revision-format <claim agreement=...> attribute must be read"
        );
    }

    // ── V2 profile selection tests ────────────────────────────────────────────

    /// V2LitReview profile returns V2_SYNTHESIS_WEIGHTS (5 dims) and is_valid_v2.
    #[test]
    fn v2_synthesis_fitness_returns_v2_weights_and_validity_fn() {
        use crate::ttd::fitness::FitnessEval;
        use crate::ttd::term_sheet::PromptProfile;
        use crate::ttd::weights::{SYNTHESIS_WEIGHTS, V2_SYNTHESIS_WEIGHTS};

        let v2 = SynthesisEvalFitness::new("agent", "model")
            .with_profile(PromptProfile::V2LitReview);
        assert_eq!(
            v2.weights(),
            V2_SYNTHESIS_WEIGHTS,
            "V2LitReview must return V2_SYNTHESIS_WEIGHTS"
        );
        assert_eq!(
            v2.weights().len(),
            5,
            "V2 synthesis weight table must have 5 dims"
        );
        let vfn = v2.validity_fn();
        let ok = FitnessEval::new(vec![("faithfulness".into(), Some(4))]);
        assert!(vfn(&ok), "faithfulness=4 must be valid for V2 synthesis");
        let fail = FitnessEval::new(vec![("faithfulness".into(), Some(3))]);
        assert!(!vfn(&fail), "faithfulness=3 must be invalid for V2 synthesis");

        // V1 path unchanged
        let v1 = SynthesisEvalFitness::new("agent", "model");
        assert_eq!(
            v1.weights(),
            SYNTHESIS_WEIGHTS,
            "V1Delphi must return SYNTHESIS_WEIGHTS"
        );
        assert_eq!(
            v1.weights().len(),
            6,
            "V1 synthesis weight table must have 6 dims"
        );
    }

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║ SEAM F4a — SYNTHESIS parser (characterisation net, W-522022c5)          ║
    // ║ Reaches the file-level `parse_gaps_xml` (pub(crate)) via super::super.   ║
    // ║ PINS THE T1 RULED CONTRACT: missing/empty <gaps> → Ok(vec![]); a gap is  ║
    // ║ valid iff non-empty <description>; <query> defaults to description.      ║
    // ║ (Re-baselined from the prior Err-on-missing split per the trip-wire      ║
    // ║  protocol — the contract change is INTENDED under Skuld's T1 ruling.)    ║
    // ╚═══════════════════════════════════════════════════════════════════════╝
    mod f4a_synthesis_parser {
        #[test]
        fn f4_synthesis_desc_only_yields_one_gap() {
            let xml = "<gaps><gap><description>a gap</description></gap></gaps>";
            let out = super::super::parse_gaps_xml(xml).expect("synthesis: desc-only must be Ok");
            assert_eq!(out.len(), 1, "F4: synthesis emits a gap on desc-only (query defaults to desc)");
            assert_eq!(out[0].query, out[0].description, "F4: synthesis defaults query→description");
        }

        #[test]
        fn f4_synthesis_missing_block_is_ok_empty() {
            // PINS THE T1 RULED CONTRACT: missing block → Ok(vec![]) (never Err).
            let out = super::super::parse_gaps_xml("no gaps block here at all");
            assert!(
                matches!(out, Ok(ref v) if v.is_empty()),
                "F4: synthesis MUST return Ok(vec![]) on missing block (T1 contract: never Err)"
            );
        }

        #[test]
        fn f4_synthesis_does_not_panic_on_multibyte_unterminated() {
            // Probe fixtures I2/A3 shapes — must not panic, returns Err for unterminated.
            let multibyte_unterminated = "<gaps><gap><description>café — naïve</description>";
            let r = std::panic::catch_unwind(|| super::super::parse_gaps_xml(multibyte_unterminated));
            assert!(r.is_ok(), "F4: synthesis must not PANIC on multibyte/unterminated input");
        }
    }
}
