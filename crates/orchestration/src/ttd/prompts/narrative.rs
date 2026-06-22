//! Native port of the 4 consensus narrative mustache templates.
//!
//! Sources:
//! - `consensus/prompts/diffusion/v1/narrative/draft.mustache`
//! - `consensus/prompts/diffusion/v1/narrative/narrative_critique.mustache`
//! - `consensus/prompts/diffusion/v1/narrative/narrative_refine.mustache`
//! - `consensus/prompts/diffusion/v1/narrative/narrative_final_merge.mustache`
//!
//! All templates are versioned `v1/narrative`.
//! Mustache semantics are hand-translated per the 5 rules in `prompts/render.rs`.
//!
//! ## Stage-3 contract
//!
//! - Draft: input = synthesis claims/areas/uncertainties; output = plain text narrative
//! - Critique: input = synthesis claims + draft narrative; output = XML `<gaps>`
//! - Refine: input = critique + synthesis + current narrative; output = plain text
//! - Final merge: input = synthesis claims + N candidate narratives; output = plain text
//!
//! ## Citation preservation (T-23-10)
//!
//! The merge prompt EXPLICITLY instructs the model to preserve all `[Cx]` markers.
//! The `narrative_final_merge` renderer includes the critical citation preservation
//! block from the original mustache template verbatim.
//!
//! ## Trust boundary
//!
//! Synthesis claims and narrative text stay in data sections — the instruction
//! positions contain only engine-authored directives (T-23-10 mitigation).

use crate::ttd::artifact::SynthesisArtifact;

/// Prompt version for all narrative-stage templates.
pub const NARRATIVE_PROMPT_VERSION: &str = "v1/narrative";

// ── Template 1: narrative/draft ───────────────────────────────────────────────

/// Render `narrative/draft.mustache` — initial narrative draft from synthesis.
///
/// Source: `consensus/prompts/diffusion/v1/narrative/draft.mustache` [VERIFIED]
///
/// Input = the Stage-2 synthesis artifact. Output = 300-500 word narrative text.
/// The `{{#synthesis}}` section iteration is preserved: all claims, areas of
/// agreement/disagreement, and uncertainties are rendered.
pub fn render_narrative_draft(synthesis: &SynthesisArtifact) -> String {
    // Render claims with IDs (matches {{#claims_with_ids}} section in mustache)
    let claims_section = render_claims_with_ids(&synthesis.claims_with_ids());

    // Render areas of agreement ({{#areas_of_agreement}})
    let agreement_section = render_list_items(&synthesis.areas_of_agreement);

    // Render areas of disagreement ({{#areas_of_disagreement}})
    let disagreement_section = render_list_items(&synthesis.areas_of_disagreement);

    // Render uncertainties ({{#uncertainties}})
    let uncertainties_section = render_list_items(&synthesis.uncertainties);

    format!(
        r#"You are summarizing the results of a multi-expert consultation.

## Task
Write a cohesive narrative synthesis that weaves together the agreed-upon claims, areas of debate, and identified uncertainties.

## Input Context

### Synthesized Claims
{claims_section}
### Areas of Agreement
{agreement_section}
### Areas of Disagreement
{disagreement_section}
### Uncertainties
{uncertainties_section}
## Instructions
1. Write a fluent, professional summary (approx. 300-500 words).
2. Do not just list the claims; weave them into a coherent story about the expert consensus.
3. Highlight the tension between agreement and disagreement.
4. Maintain a neutral, facilitator tone.
5. Do not introduce new information not present in the claims.
6. Include inline citations using claim IDs (e.g., [C1, C3]) after each substantive statement. Every factual assertion should reference at least one claim ID from the list above.

## Output Format
Return ONLY the narrative text with inline citation markers. Do not use Markdown formatting or headings."#,
        claims_section = claims_section,
        agreement_section = agreement_section,
        disagreement_section = disagreement_section,
        uncertainties_section = uncertainties_section,
    )
}

// ── Template 2: narrative/narrative_critique ─────────────────────────────────

/// Input data for `narrative/narrative_critique` rendering.
pub struct NarrativeCritiqueInput<'a> {
    /// The fixed synthesis (reference for gap checking — NOT the lit store).
    pub synthesis: &'a SynthesisArtifact,
    /// The current narrative draft being critiqued.
    pub narrative: &'a str,
    /// Optional fitness feedback document (unescaped raw markdown).
    pub fitness_feedback: Option<&'a str>,
}

/// Render `narrative/narrative_critique.mustache`.
///
/// Source: `consensus/prompts/diffusion/v1/narrative/narrative_critique.mustache` [VERIFIED]
///
/// Identifies hallucinations/omissions/bias in the narrative vs the fixed synthesis.
/// Output format: XML `<gaps>` block (same as Stage-1/2 gap_identify).
///
/// ## No retrieval
///
/// Gaps identified here are checked against the SYNTHESIS, not the lit store.
/// The `NarrativeRefine` step that follows does NOT issue retrieval queries.
pub fn render_narrative_critique(input: &NarrativeCritiqueInput) -> String {
    let claims_section = render_claims_with_ids(&input.synthesis.claims_with_ids());

    // Fitness feedback block (unescaped, triple-mustache semantics):
    // `{{#fitness_feedback}}...{{{fitness_feedback}}}...{{/fitness_feedback}}`
    let feedback_block = match input.fitness_feedback {
        Some(feedback) if !feedback.is_empty() => format!(
            "\n### Fitness Evaluation Feedback\n{feedback}\n",
            feedback = feedback
        ),
        _ => String::new(),
    };

    format!(
        r#"You are a strict editor reviewing a synthesis narrative against the source data.

## Task
Identify any weaknesses, inaccuracies, or hallucinations in the narrative.

## Context
### Source Data
Claims:
{claims_section}
### Draft Narrative
{narrative}
{feedback_block}
## Criteria
1. **Faithfulness**: Does the narrative make any claims not supported by the Source Data? (Hallucinations)
2. **Completeness**: Does it miss any MAJOR agreed-upon claims? (Minor omissions are fine)
3. **Neutrality**: Does it use biased language or take sides inappropriately?
4. **Clarity**: Is the flow logical?
5. **Citation accuracy**: Are inline citations [Cx] present and accurate? Do citation markers reference valid claim IDs from the source data? Are substantive statements missing citations?

## Output Format
Return a list of gaps in XML format. Each gap must include a description and a search query that could retrieve relevant source material to address it. If the narrative is perfect, return an empty <gaps> list.

Example:
<gaps>
  <gap>
    <description>The narrative claims X, but this is not in the source claims.</description>
    <query>claim X specific topic evidence</query>
  </gap>
</gaps>

Identify gaps if any exist (0 is acceptable if the narrative is comprehensive), prioritized by importance. Each query should be specific enough to retrieve targeted content from the synthesis claims and expert responses."#,
        claims_section = claims_section,
        narrative = input.narrative,
        feedback_block = feedback_block,
    )
}

// ── Template 3: narrative/narrative_refine ───────────────────────────────────

/// Input data for `narrative/narrative_refine` rendering.
pub struct NarrativeRefineInput<'a> {
    /// The fixed synthesis (reference for rewriting).
    pub synthesis: &'a SynthesisArtifact,
    /// The current narrative draft being refined.
    pub narrative: &'a str,
    /// Critique text (combined gap descriptions).
    pub critique: &'a str,
    /// Retrieved context (EMPTY for Stage 3 — NoopRetriever invariant).
    pub retrieved: &'a [(String, String)],
    /// Optional fitness feedback document.
    pub fitness_feedback: Option<&'a str>,
}

/// Render `narrative/narrative_refine.mustache`.
///
/// Source: `consensus/prompts/diffusion/v1/narrative/narrative_refine.mustache` [VERIFIED]
///
/// Rewrites the narrative to address critique findings. No retrieval is used
/// (Stage 3 invariant — `retrieved` is always empty here).
///
/// ## Citation preservation
///
/// The prompt instructs the model to "Preserve and correct all inline citation
/// markers [Cx]" — preserving valid markers and fixing incorrect ones.
pub fn render_narrative_refine(input: &NarrativeRefineInput) -> String {
    let claims_section = render_claims_with_ids(&input.synthesis.claims_with_ids());

    // {{#retrieved}} conditional block — renders ONLY when retrieved is non-empty.
    // For Stage 3, retrieved is always empty (NoopRetriever invariant), so this
    // block never renders.
    let retrieved_block = if !input.retrieved.is_empty() {
        let mut buf = String::from("\n### Retrieved Context\nThe following source material was retrieved to help address the gaps:\n");
        for (source_id, content) in input.retrieved {
            buf.push_str(&format!("- Source {source_id}: {content}\n"));
        }
        buf
    } else {
        String::new()
    };

    // Fitness feedback block (conditional — renders only when truthy).
    let feedback_block = match input.fitness_feedback {
        Some(feedback) if !feedback.is_empty() => format!(
            "\n### Fitness Evaluation Feedback\n{feedback}\n",
            feedback = feedback
        ),
        _ => String::new(),
    };

    format!(
        r#"You are an expert editor refining a consensus narrative.

## Task
Rewrite the narrative to address the provided critique.

## Context
### Critique (Issues to Fix)
{critique}
{feedback_block}
### Source Data
Claims:
{claims_section}
{retrieved_block}
### Current Draft
{narrative}

## Instructions
1. Rewrite the narrative to resolve the issues.
2. Maintain the parts that were already good.
3. Ensure the tone remains neutral and professional.
4. Do not exceed 500 words.
5. Preserve and correct all inline citation markers [Cx]. Every substantive statement should reference the relevant claim IDs.

## Output Format
Return ONLY the rewritten narrative text with inline citation markers."#,
        critique = input.critique,
        feedback_block = feedback_block,
        claims_section = claims_section,
        retrieved_block = retrieved_block,
        narrative = input.narrative,
    )
}

// ── Template 4: narrative/narrative_final_merge ───────────────────────────────

/// Input data for `narrative/narrative_final_merge` rendering.
pub struct NarrativeFinalMergeInput<'a> {
    /// The fixed synthesis (source claims for citation validation).
    pub synthesis: &'a SynthesisArtifact,
    /// Candidate narrative drafts to merge (best-first order from selection).
    pub narratives: &'a [String],
}

/// Render `narrative/narrative_final_merge.mustache`.
///
/// Source: `consensus/prompts/diffusion/v1/narrative/narrative_final_merge.mustache` [VERIFIED]
///
/// Merges N candidate narratives into one, preserving inline `[Cx]` citation markers.
///
/// ## CRITICAL: Citation preservation
///
/// The CRITICAL block from the mustache template is reproduced verbatim:
/// "Every substantive statement in the merged narrative MUST retain inline
/// citation markers [Cx] from the source drafts."
///
/// T-23-10 post-merge sanitisation (`sanitise_cx_citations`) removes hallucinated
/// markers after the model output is received.
pub fn render_narrative_final_merge(input: &NarrativeFinalMergeInput) -> String {
    let claims_section = render_claims_with_ids(&input.synthesis.claims_with_ids());

    // Render the {{#narratives}} section: each draft separated by ---
    let drafts_section: String = input
        .narratives
        .iter()
        .map(|n| format!("### Draft\n\n{n}\n\n---\n\n"))
        .collect();

    format!(
        r#"You are merging multiple narrative drafts into a final consensus summary.

## Task
Combine the strengths of the provided drafts into a single, high-quality narrative that is faithful to the source claims below.

## Source Claims
{claims_section}
## Drafts to Merge

{drafts_section}
## CRITICAL: Citation Preservation
Every substantive statement in the merged narrative MUST retain inline citation markers [Cx] from the source drafts. Do not remove, renumber, or omit any citation markers. If multiple drafts cite different claims for the same point, include all citation markers (e.g., [C1, C3, C5]). A merged narrative that drops citations is a failed merge.

## Instructions
1. Synthesize a single narrative that captures the best phrasing and structure from the drafts above.
2. Do not drop any claims from the source data. If drafts disagree about a claim, preserve both perspectives. Faithfulness to the source claims takes priority over narrative flow.
3. Ensure it covers all the key points raised across the drafts.
4. Resolve any inconsistencies between drafts by favoring the most neutral/supported phrasing.
5. Aim for clarity, flow, and professional tone.
6. The output should be approximately 300-500 words, but may be longer if needed to preserve all citations and claim coverage.
7. When merging a sentence from multiple drafts, always carry forward the citation markers from ALL drafts that made that point.

## Output Format
Return ONLY the merged narrative text with inline citation markers [Cx]. Do not include section headers or metadata. Every factual statement must have at least one citation marker."#,
        claims_section = claims_section,
        drafts_section = drafts_section,
    )
}

// ── Shared rendering helpers ──────────────────────────────────────────────────

/// Render claims with labelled text: `- C1. claim text (Agreement: consensus)`
///
/// Mirrors `{{#claims_with_ids}}` section iteration in the narrative templates.
/// This is the `labelled_text` form: `"{id}. {text}"`.
fn render_claims_with_ids(claims: &[(String, String, String)]) -> String {
    if claims.is_empty() {
        return String::new();
    }
    let mut buf = String::new();
    for (id, text, agreement) in claims {
        buf.push_str(&format!("- {id}. {text} (Agreement: {agreement})\n"));
    }
    buf
}

/// Render a flat list of strings as bullet points.
///
/// Empty list → empty string (section-iteration semantics).
fn render_list_items(items: &[String]) -> String {
    if items.is_empty() {
        return String::new();
    }
    items.iter().map(|s| format!("- {s}\n")).collect()
}

// ── SynthesisArtifact extension for narrative rendering ───────────────────────

/// Extension trait providing narrative-rendering helpers on `SynthesisArtifact`.
///
/// Provides `claims_with_ids()` returning `(id, text, agreement_level)` tuples
/// to mirror the `{{#claims_with_ids}}` mustache section from the narrative templates.
trait NarrativeRenderExt {
    /// Returns `(claim_id, text, agreement_level)` for each claim.
    ///
    /// Claim IDs are 1-indexed: C1, C2, ... matching the inline `[Cx]` markers
    /// in the narrative text.
    fn claims_with_ids(&self) -> Vec<(String, String, String)>;
}

impl NarrativeRenderExt for SynthesisArtifact {
    fn claims_with_ids(&self) -> Vec<(String, String, String)> {
        self.claims
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let id = format!("C{}", i + 1);
                let text = c.text.clone();
                let agreement = c.agreement_level.clone().unwrap_or_else(|| "unknown".into());
                (id, text, agreement)
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttd::artifact::{Claim, SynthesisArtifact};

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
        s.areas_of_agreement.push("Warming is accelerating permafrost thaw".into());
        s.areas_of_disagreement.push("Rate of methane release is disputed".into());
        s.uncertainties.push("Long-term feedback loops unclear".into());
        s
    }

    /// narrative draft prompt references synthesis claims (not expert response XML).
    #[test]
    fn draft_prompt_references_synthesis_claims() {
        let synthesis = stub_synthesis();
        let prompt = render_narrative_draft(&synthesis);

        assert!(
            prompt.contains("Climate change accelerates permafrost thaw"),
            "draft prompt must reference synthesis claims: {prompt}"
        );
        assert!(
            !prompt.contains("expert_response"),
            "draft prompt must NOT include raw expert_response XML tags"
        );
    }

    /// narrative critique prompt has fitness_feedback conditional block.
    #[test]
    fn critique_prompt_with_feedback() {
        let synthesis = stub_synthesis();
        let prompt = render_narrative_critique(&NarrativeCritiqueInput {
            synthesis: &synthesis,
            narrative: "A short narrative about climate [C1].",
            fitness_feedback: Some("## Priority Improvements\n- faithfulness: low"),
        });

        assert!(
            prompt.contains("Priority Improvements"),
            "critique prompt must include fitness feedback when provided"
        );
        assert!(
            prompt.contains("<gaps>"),
            "critique prompt must mention the <gaps> output format"
        );
    }

    /// narrative critique prompt without feedback has no feedback section.
    #[test]
    fn critique_prompt_without_feedback() {
        let synthesis = stub_synthesis();
        let prompt = render_narrative_critique(&NarrativeCritiqueInput {
            synthesis: &synthesis,
            narrative: "A short narrative.",
            fitness_feedback: None,
        });

        assert!(
            !prompt.contains("Fitness Evaluation Feedback"),
            "critique prompt must not include empty feedback section"
        );
    }

    /// narrative_refine retrieved block renders only when non-empty.
    #[test]
    fn refine_retrieved_block_conditional() {
        let synthesis = stub_synthesis();

        // Empty retrieved → no retrieved block
        let prompt_no_retrieved = render_narrative_refine(&NarrativeRefineInput {
            synthesis: &synthesis,
            narrative: "A draft narrative.",
            critique: "- Missing citation for claim 1.",
            retrieved: &[],
            fitness_feedback: None,
        });
        assert!(
            !prompt_no_retrieved.contains("Retrieved Context"),
            "refine prompt must NOT include retrieved block when retrieved is empty"
        );

        // Non-empty retrieved → retrieved block included
        let retrieved = vec![("arxiv:2105.14103".into(), "Some retrieved content".into())];
        let prompt_with_retrieved = render_narrative_refine(&NarrativeRefineInput {
            synthesis: &synthesis,
            narrative: "A draft narrative.",
            critique: "- Missing citation.",
            retrieved: &retrieved,
            fitness_feedback: None,
        });
        assert!(
            prompt_with_retrieved.contains("Retrieved Context"),
            "refine prompt must include retrieved block when retrieved is non-empty"
        );
    }

    /// narrative_final_merge prompt contains CRITICAL citation preservation block.
    #[test]
    fn final_merge_citation_preservation_block() {
        let synthesis = stub_synthesis();
        let prompt = render_narrative_final_merge(&NarrativeFinalMergeInput {
            synthesis: &synthesis,
            narratives: &[
                "First candidate [C1] with reference.".into(),
                "Second candidate [C2] with reference.".into(),
            ],
        });

        assert!(
            prompt.contains("Citation Preservation"),
            "merge prompt must contain the CRITICAL citation preservation block"
        );
        assert!(
            prompt.contains("[Cx]"),
            "merge prompt must mention [Cx] citation marker format"
        );
        assert!(
            prompt.contains("First candidate"),
            "merge prompt must include the candidate narratives"
        );
    }

    /// claims_with_ids produces 1-indexed claim IDs matching [Cx] marker convention.
    #[test]
    fn claims_with_ids_one_indexed() {
        let synthesis = stub_synthesis();
        let ids = synthesis.claims_with_ids();

        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].0, "C1");
        assert_eq!(ids[1].0, "C2");
        assert_eq!(ids[0].1, "Climate change accelerates permafrost thaw.");
        assert_eq!(ids[0].2, "consensus");
    }

    /// Narrative prompt version constant is "v1/narrative".
    #[test]
    fn narrative_prompt_version_constant() {
        assert_eq!(NARRATIVE_PROMPT_VERSION, "v1/narrative");
    }
}
