//! Native port of the 7 consensus synthesis mustache templates + aggregator_revision.
//!
//! Sources:
//! - `consensus/prompts/diffusion/v1/synthesis/` — 7 synthesis templates
//! - `consensus/prompts/committee/aggregator_revision.mustache` — 1 revision template
//!
//! All templates are versioned `v1/synthesis` (revision: `v1/committee`).
//! Mustache semantics are hand-translated per the 5 rules in `prompts/render.rs`.
//!
//! ## High-risk template: draft_graph
//!
//! `render_synthesis_draft_graph` includes a conditional `{{#has_unverified_nodes}}`
//! block. This block renders ONLY when `has_unverified_nodes=true`. Test:
//! `draft_graph_conditional_block`.
//!
//! ## Trust boundary (T-23-07)
//!
//! - ExpertResponse prose and retrieved content stay in data sections.
//! - Fitness judge prompts have no preamble (Pitfall 6 guard — empty ancestors).
//!
//! ## Fitness judges: empty ancestors (Pitfall 6)
//!
//! Synthesis fitness judge prompts are constructed without ancestral context so
//! the channel-substitution lexer does not inject a preamble. Test:
//! `synthesis_fitness_judge_no_preamble`.

/// Prompt version for all synthesis-stage templates.
pub const SYNTHESIS_PROMPT_VERSION: &str = "v1/synthesis";

/// Prompt version for the aggregator revision template.
pub const COMMITTEE_PROMPT_VERSION: &str = "v1/committee";

use crate::ttd::artifact::{ArgumentationGraph, SynthesisArtifact};

// ── Template 1: synthesis/draft ───────────────────────────────────────────────

/// Input data for `synthesis/draft` rendering.
pub struct SynthesisDraftInput<'a> {
    /// The research question.
    pub question: String,
    /// Expert responses as (response_id, vec_of_prose) tuples.
    pub responses: &'a [(String, Vec<String>)],
}

/// Render `synthesis/draft.mustache` — plain synthesis draft (no graph).
///
/// Source: `consensus/prompts/diffusion/v1/synthesis/draft.mustache` [VERIFIED]
///
/// Used when `use_graph_draft=false` or when no ArgumentationGraph is available.
pub fn render_synthesis_draft(input: &SynthesisDraftInput) -> String {
    let mut responses_section = String::new();
    for (response_id, prose_items) in input.responses {
        responses_section.push_str(&format!("### Expert {response_id}\n\n"));
        for prose in prose_items {
            responses_section.push_str(&format!("<expert_response id=\"{response_id}\">{prose}</expert_response>\n\n"));
        }
        responses_section.push_str("---\n");
    }

    format!(
        r#"You are a neutral facilitator synthesising expert responses for a Delphi-style consultation.

## Your Role

- Synthesise expert responses into a structured summary
- Preserve disagreement as a first-class artefact (do not average it away)
- Surface agreement, disagreement, and uncertainty without advocacy
- Do NOT introduce novel claims or make policy recommendations
- Every claim must be traceable to specific expert responses

## Question Being Addressed

{question}

## Expert Responses

{responses}

## Required Output Structure

Provide your synthesis in the following XML format:

```xml
<synthesis>
  <narrative>
    A coherent, high-quality prose summary of the expert consultation.
  </narrative>
  <claims>
    <claim id="C1">
      <text>The synthesised claim statement</text>
      <agreement_level>consensus|majority|divided|minority</agreement_level>
      <sources>
        <source id="response_id">
          <quote>Brief reference — exact quote resolved in a later step</quote>
        </source>
      </sources>
      <counterarguments>
        <counterargument>Any dissenting view on this claim</counterargument>
      </counterarguments>
    </claim>
  </claims>
  <areas_of_agreement>
    <area>Summary of area where experts agree</area>
  </areas_of_agreement>
  <areas_of_disagreement>
    <area>Summary of area where experts disagree</area>
  </areas_of_disagreement>
  <uncertainties>
    <uncertainty>Explicit uncertainty surfaced by experts</uncertainty>
  </uncertainties>
</synthesis>
```

### ⚠️ CRITICAL CONSTRAINTS

- **Output ONLY the `<synthesis>` XML block.** No preamble, no commentary.
- **Do NOT wrap the XML in a markdown code block** (no ```xml). Output raw XML directly.
- **Every claim MUST have at least one `<source>`** with the source_id.
- **Agreement levels must be exactly one of**: `consensus`, `majority`, `divided`, `minority`.
- **Do NOT introduce novel claims** not present in expert responses.
"#,
        question = input.question,
        responses = responses_section,
    )
}

// ── Template 2: synthesis/draft_graph ─────────────────────────────────────────

/// Input data for `synthesis/draft_graph` rendering.
pub struct SynthesisDraftGraphInput<'a> {
    /// The research question.
    pub question: String,
    /// Number of experts in the panel.
    pub expert_count: usize,
    /// The Stage-1 argumentation graph.
    pub graph: &'a ArgumentationGraph,
    /// Whether any nodes have unverified evidence (conditional block).
    pub has_unverified_nodes: bool,
    /// (node_id, text_snippet) pairs for unverified nodes.
    pub unverified_nodes: &'a [(String, String)],
    /// Target narrative length hint.
    pub target_length: &'a str,
}

/// Render `synthesis/draft_graph.mustache` — graph-seeded synthesis draft.
///
/// Source: `consensus/prompts/diffusion/v1/synthesis/draft_graph.mustache` [VERIFIED]
///
/// This is the DEFAULT path when `use_graph_draft=true` and an `ArgumentationGraph`
/// is present (synthesis_tasks.py:131-133).
///
/// ## High-risk conditional: {{#has_unverified_nodes}}
///
/// The block renders ONLY when `has_unverified_nodes=true`. This is the snapshot
/// validation point for this template (RESEARCH "Prompt Snapshot Mechanism").
pub fn render_synthesis_draft_graph(input: &SynthesisDraftGraphInput) -> String {
    use super::render::{render_edges_section, render_nodes_section};

    // Render nodes section
    let nodes: Vec<(String, String)> = input
        .graph
        .nodes
        .iter()
        .map(|n| (n.id.clone(), n.claim.clone()))
        .collect();
    let nodes_section = render_nodes_section(&nodes);

    // Render edges section
    let edges: Vec<(String, String, String)> = input
        .graph
        .edges
        .iter()
        .map(|e| (e.source.clone(), e.target.clone(), e.relation.clone()))
        .collect();
    let edges_section = render_edges_section(&edges);

    // Conditional {{#has_unverified_nodes}} block (high-risk — snapshot tested)
    let unverified_block = if input.has_unverified_nodes {
        let items: String = input
            .unverified_nodes
            .iter()
            .map(|(id, text)| format!("- [{id}]: {text}\n"))
            .collect();
        format!(
            "## Quality Gates\n\n\
             ⚠️ The following nodes have unverified or failed quote verification — \
             treat their evidence with appropriate scepticism:\n\
             {items}\n"
        )
    } else {
        String::new() // conditional: renders nothing when has_unverified_nodes=false
    };

    format!(
        r#"You are a neutral facilitator synthesising expert responses for a Delphi-style consultation.

## Your Role

You are receiving a **merged argumentation graph** extracted from {expert_count} expert responses in Stage 1 of a multi-stage pipeline. Your task is to synthesise this graph into a structured summary.

- Weave the graph's claims, evidence, and relationships into a coherent synthesis
- Preserve disagreement as a first-class artefact (do not average it away)
- Surface agreement, disagreement, and uncertainty without advocacy
- Do NOT introduce novel claims — synthesise only from the graph structure
- Every claim must be traceable to specific graph nodes and their source experts

## Provenance

This input is the merged argumentation graph from Stage 1 (graph extraction). The graph has already been verified and deduplicated. Your role is to synthesise, not to re-analyse the raw responses.

## Question Being Addressed

{question}

## Argumentation Graph

### Nodes (Claims, Evidence, Counterclaims)

{nodes_section}

### Relationships

{edges_section}

{unverified_block}

## Synthesis Instructions

1. **Dimensional confinement**: Synthesise from the graph structure above. Do not re-analyse or invent beyond what the graph contains.
2. **Completeness check**: If the graph appears incomplete, flag missing perspectives in the uncertainties section.
3. **Emergence space**: Surface tensions or connections between nodes not captured in graph edges.
4. **Agreement levels**: Derive agreement levels from the graph's support/attack edges and source counts.

## Required Output Structure

Provide your synthesis in the following XML format:

```xml
<synthesis>
  <narrative>
    A coherent, high-quality prose summary of the expert consultation.
    Aim for {target_length} words.
  </narrative>
  <claims>
    <claim id="C1">
      <text>The synthesised claim statement</text>
      <agreement_level>consensus|majority|divided|minority</agreement_level>
      <sources>
        <source id="response_id">
          <quote>Brief reference — exact quote resolved in a later step</quote>
        </source>
      </sources>
      <counterarguments>
        <counterargument>Any dissenting view on this claim</counterargument>
      </counterarguments>
    </claim>
  </claims>
  <areas_of_agreement>
    <area>Summary of area where experts agree</area>
  </areas_of_agreement>
  <areas_of_disagreement>
    <area>Summary of area where experts disagree</area>
  </areas_of_disagreement>
  <uncertainties>
    <uncertainty>Explicit uncertainty surfaced by experts</uncertainty>
  </uncertainties>
</synthesis>
```

### ⚠️ CRITICAL CONSTRAINTS

- **Output ONLY the `<synthesis>` XML block.** No preamble, no commentary.
- **Do NOT wrap the XML in a markdown code block** (no ```xml). Output raw XML directly.
- **Every claim MUST have at least one `<source>`** with the source_id from the graph nodes.
- **Agreement levels must be exactly one of**: `consensus`, `majority`, `divided`, `minority`.
- **Do NOT introduce novel claims** not present in the argumentation graph.
"#,
        expert_count = input.expert_count,
        question = input.question,
        nodes_section = nodes_section,
        edges_section = edges_section,
        unverified_block = unverified_block,
        target_length = input.target_length,
    )
}

// ── Template 3: synthesis/gap_identify ────────────────────────────────────────

/// Input data for `synthesis/gap_identify` rendering.
pub struct SynthesisGapIdentifyInput<'a> {
    /// The current synthesis draft.
    pub draft: &'a SynthesisArtifact,
    /// Optional fitness feedback for injection (triple-mustache unescaped).
    pub fitness_feedback: Option<&'a str>,
}

/// Render `synthesis/gap_identify.mustache` — identify synthesis gaps.
///
/// Source: `consensus/prompts/diffusion/v1/synthesis/gap_identify.mustache` [VERIFIED]
///
/// ## Mustache semantics preserved
///
/// - `{{#draft.claims}}` → section iteration over claims
/// - `{{^last}}, {{/last}}` → comma-join sources (inverted section)
/// - `{{{fitness_feedback}}}` → unescaped raw markdown inject (triple-mustache)
/// - `{{#fitness_feedback}}...{{/fitness_feedback}}` → conditional block
pub fn render_synthesis_gap_identify(input: &SynthesisGapIdentifyInput) -> String {
    use super::render::render_sources_comma_joined;

    // Claims section (section iteration)
    let mut claims_section = String::new();
    for claim in &input.draft.claims {
        let level = claim.agreement_level.as_deref().unwrap_or("divided");
        let sources_str = render_sources_comma_joined(&claim.sources);
        let ca_str = if claim.counterarguments.is_empty() {
            String::new()
        } else {
            format!(
                "  Counterarguments: {}\n",
                claim.counterarguments.join(", ")
            )
        };
        claims_section.push_str(&format!(
            "- **[{id}]** ({level}): \"{text}\"\n  Sources: {sources}\n{ca}",
            id = "C",
            level = level,
            text = claim.text,
            sources = sources_str,
            ca = ca_str,
        ));
        claims_section.push('\n');
    }

    // Areas of agreement
    let agreement_section: String = input
        .draft
        .areas_of_agreement
        .iter()
        .map(|a| format!("- \"{a}\"\n"))
        .collect();

    // Areas of disagreement
    let disagreement_section: String = input
        .draft
        .areas_of_disagreement
        .iter()
        .map(|a| format!("- \"{a}\"\n"))
        .collect();

    // Uncertainties
    let uncertainties_section: String = input
        .draft
        .uncertainties
        .iter()
        .map(|u| format!("- \"{u}\"\n"))
        .collect();

    // Fitness feedback block (conditional + unescaped triple-mustache)
    // The value is injected RAW — no HTML escaping (T-23-05).
    let feedback_block = match input.fitness_feedback {
        Some(feedback) if !feedback.is_empty() => {
            format!(
                "---\n\n## Fitness Evaluation Feedback\n\n\
                 The synthesis was evaluated and the following issues were identified:\n\n\
                 {feedback}\n\n\
                 Use this feedback to prioritize which gaps to address.\n\n---\n\n"
            )
        }
        _ => String::new(), // conditional: renders nothing when absent/empty
    };

    format!(
        r#"You are analyzing a synthesis artefact to identify gaps and generate retrieval queries.

## Your Role in the Agentic Pipeline

You are part of an iterative refinement process. Your task is to:
1. Analyze the current synthesis for what is missing or weak
2. Generate targeted search queries to find relevant content

Your output will guide retrieval from the source material, and a subsequent step will integrate the retrieved content.

---

## Current Synthesis

### Claims

{claims_section}
### Areas of Agreement

{agreement_section}
### Areas of Disagreement

{disagreement_section}
### Uncertainties

{uncertainties_section}
---

{feedback_block}## Your Task

Identify gaps in the synthesis that need to be filled. For each gap:
1. **Describe** what is missing or weak
2. **Generate a query** that will find relevant content

Focus on:
- Claims lacking evidence or citations
- Missing coverage of expert viewpoints
- Underrepresented disagreements or minority views
- Weak structural organization
- Missing counterarguments for divided claims
- Issues highlighted in the fitness feedback

---

## Required Output Format

```xml
<gaps>
  <gap>
    <description>What is missing or weak in the synthesis</description>
    <query>Search query to find relevant content</query>
  </gap>
</gaps>
```

Generate 3-5 gaps, prioritized by importance. Each query should be specific enough to retrieve targeted content.

Begin your analysis:
"#,
        claims_section = claims_section,
        agreement_section = agreement_section,
        disagreement_section = disagreement_section,
        uncertainties_section = uncertainties_section,
        feedback_block = feedback_block,
    )
}

// ── Template 4: synthesis/gap_resolve_patch ────────────────────────────────────

/// Input data for synthesis gap_resolve (both patch and full-regen).
pub struct SynthesisGapResolveInput<'a> {
    pub draft: &'a SynthesisArtifact,
    pub gaps: &'a [crate::ttd::state::IdentifiedGap],
    pub retrieved: &'a [crate::ttd::stages::RetrievedContext],
    pub fitness_feedback: Option<&'a str>,
}

/// Render `synthesis/gap_resolve_patch.mustache` — patch-based gap resolution.
///
/// Source: `consensus/prompts/diffusion/v1/synthesis/gap_resolve_patch.mustache` [VERIFIED]
pub fn render_synthesis_gap_resolve_patch(input: &SynthesisGapResolveInput) -> String {
    use super::render::{render_gaps_section, render_retrieved_section, render_sources_comma_joined};

    let claims_section = render_claims_section(&input.draft.claims);
    let agreement_section: String = input
        .draft
        .areas_of_agreement
        .iter()
        .map(|a| format!("- \"{a}\"\n"))
        .collect();
    let disagreement_section: String = input
        .draft
        .areas_of_disagreement
        .iter()
        .map(|a| format!("- \"{a}\"\n"))
        .collect();
    let uncertainties_section: String = input
        .draft
        .uncertainties
        .iter()
        .map(|u| format!("- \"{u}\"\n"))
        .collect();

    let gaps: Vec<(String, String)> = input
        .gaps
        .iter()
        .map(|g| (g.description.clone(), g.query.clone()))
        .collect();
    let gaps_section = render_gaps_section(&gaps);

    let retrieved: Vec<(String, String)> = input
        .retrieved
        .iter()
        .map(|r| (r.source_id.clone(), r.content.clone()))
        .collect();
    let retrieved_section = render_retrieved_section(&retrieved);

    let feedback_block = match input.fitness_feedback {
        Some(fb) if !fb.is_empty() => format!(
            "## Fitness Evaluation Feedback\n\n{fb}\n\nAddress these issues in your refinement.\n"
        ),
        _ => String::new(),
    };

    let _ = render_sources_comma_joined; // ensure import is used

    format!(
        r#"You are resolving gaps in a synthesis artefact by producing a PATCH — not a full synthesis.

## Your Role in the Agentic Pipeline

You are part of an iterative refinement process. Your task is to produce a **patch document** that adds, modifies, or removes specific claims. Do NOT reproduce the entire synthesis.

---

## Current Synthesis

### Claims

{claims_section}
### Areas of Agreement

{agreement_section}
### Areas of Disagreement

{disagreement_section}
### Uncertainties

{uncertainties_section}
---

## Identified Gaps

{gaps_section}
---

## Retrieved Content

{retrieved_section}
---

{feedback_block}

## Your Task

Produce a `<patch>` XML document that resolves the identified gaps.

### Allowed XML Schema

```xml
<patch>
  <add>
    <claim id="new_C1">
      <text>Claim text</text>
      <sources><source id="source_id">Brief reference — exact quote resolved later</source></sources>
      <counterarguments><counterargument>Dissenting view</counterargument></counterarguments>
    </claim>
  </add>
  <modify>
    <claim id="existing_claim_id">
      <update_text>Replacement text</update_text>
      <add_counterargument>New dissenting view</add_counterargument>
    </claim>
  </modify>
  <remove>
    <claim id="existing_claim_id"/>
  </remove>
</patch>
```

- Output ONLY a `<patch>` document. Do NOT output a full `<synthesis>`.
- If no changes are needed, output `<patch/>`.

Begin your patch:
"#,
        claims_section = claims_section,
        agreement_section = agreement_section,
        disagreement_section = disagreement_section,
        uncertainties_section = uncertainties_section,
        gaps_section = gaps_section,
        retrieved_section = retrieved_section,
        feedback_block = feedback_block,
    )
}

// ── Template 5: synthesis/gap_resolve ─────────────────────────────────────────

/// Render `synthesis/gap_resolve.mustache` — full regeneration gap resolve.
///
/// Source: `consensus/prompts/diffusion/v1/synthesis/gap_resolve.mustache` [VERIFIED]
pub fn render_synthesis_gap_resolve(input: &SynthesisGapResolveInput) -> String {
    use super::render::{render_gaps_section, render_retrieved_section};

    let claims_section = render_claims_section(&input.draft.claims);
    let agreement_section: String = input
        .draft
        .areas_of_agreement
        .iter()
        .map(|a| format!("- \"{a}\"\n"))
        .collect();
    let disagreement_section: String = input
        .draft
        .areas_of_disagreement
        .iter()
        .map(|a| format!("- \"{a}\"\n"))
        .collect();
    let uncertainties_section: String = input
        .draft
        .uncertainties
        .iter()
        .map(|u| format!("- \"{u}\"\n"))
        .collect();

    let gaps: Vec<(String, String)> = input
        .gaps
        .iter()
        .map(|g| (g.description.clone(), g.query.clone()))
        .collect();
    let gaps_section = render_gaps_section(&gaps);

    let retrieved: Vec<(String, String)> = input
        .retrieved
        .iter()
        .map(|r| (r.source_id.clone(), r.content.clone()))
        .collect();
    let retrieved_section = render_retrieved_section(&retrieved);

    let feedback_block = match input.fitness_feedback {
        Some(fb) if !fb.is_empty() => format!(
            "## Fitness Evaluation Feedback\n\n{fb}\n\nAddress these issues in your refinement.\n"
        ),
        _ => String::new(),
    };

    format!(
        r#"You are resolving gaps in a synthesis artefact by integrating retrieved content.

## Current Synthesis

### Claims

{claims_section}
### Areas of Agreement

{agreement_section}
### Areas of Disagreement

{disagreement_section}
### Uncertainties

{uncertainties_section}
---

## Identified Gaps

{gaps_section}
---

## Retrieved Content

{retrieved_section}
---

{feedback_block}

## Your Task

Produce a refined synthesis that fills the identified gaps using the retrieved content.

Output the complete refined synthesis in XML:

```xml
<synthesis>
  <claims>
    <claim id="unique_id">
      <text>Claim text</text>
      <agreement_level>consensus|majority|divided|minority</agreement_level>
      <sources>
        <source id="source_id">
          <quote>Brief reference — exact quote resolved later</quote>
        </source>
      </sources>
    </claim>
  </claims>
  <areas_of_agreement><area>Summary</area></areas_of_agreement>
  <areas_of_disagreement><area>Summary</area></areas_of_disagreement>
  <uncertainties><uncertainty>Uncertainty</uncertainty></uncertainties>
</synthesis>
```

Output ONLY the `<synthesis>` XML block. No preamble, no commentary.

Begin your refinement:
"#,
        claims_section = claims_section,
        agreement_section = agreement_section,
        disagreement_section = disagreement_section,
        uncertainties_section = uncertainties_section,
        gaps_section = gaps_section,
        retrieved_section = retrieved_section,
        feedback_block = feedback_block,
    )
}

// ── Template 6: synthesis/quote_resolve ───────────────────────────────────────

/// Input for quote_resolve rendering.
pub struct SynthesisQuoteResolveInput<'a> {
    /// Claims with sources that need quote resolution.
    pub draft: &'a SynthesisArtifact,
    /// Source texts keyed by source_id.
    pub source_texts: &'a [(String, String)],
}

/// Render `synthesis/quote_resolve.mustache` — dimensional-confinement quote fix.
///
/// Source: `consensus/prompts/diffusion/v1/synthesis/quote_resolve.mustache` [VERIFIED]
pub fn render_synthesis_quote_resolve(input: &SynthesisQuoteResolveInput) -> String {
    let mut claims_section = String::new();
    for claim in &input.draft.claims {
        claims_section.push_str(&format!("### Claim: \"{}\"\nSources to find quotes in:\n", claim.text));
        for src in &claim.sources {
            claims_section.push_str(&format!("- Source {src}\n"));
        }
        claims_section.push('\n');
    }

    let mut source_texts_section = String::new();
    for (source_id, text) in input.source_texts {
        source_texts_section.push_str(&format!("### Source {source_id}\n{text}\n\n---\n"));
    }

    format!(
        r#"You are a quote extraction tool. Your ONLY task is to find verbatim substrings in source texts that support given claims.

## Constraints (non-negotiable)
You CANNOT:
- Reason about, evaluate, or comment on claim content
- Modify, rephrase, or reorder claim text — claims are final
- Add or remove claims — the claim set is fixed
- Summarise or paraphrase source material — copy characters exactly
- Add commentary, explanation, or preamble

## Claims to Resolve

{claims_section}

## Source Texts

{source_texts_section}

## Task

For each claim, find the minimal contiguous substring (typically 1-3 sentences) from the referenced source that most directly supports the claim. Copy it character-for-character.

## Output Format

<quotes>
<claim id="C1">
<source id="source_id"><quote>CHARACTER-FOR-CHARACTER EXACT SUBSTRING FROM SOURCE TEXT</quote></source>
</claim>
</quotes>

If no exact supporting substring exists in the source, write: <quote>NO_EXACT_QUOTE</quote>
Output ONLY the <quotes> XML block. No preamble, no commentary.
"#,
        claims_section = claims_section,
        source_texts_section = source_texts_section,
    )
}

// ── Template 7+: synthesis/fitness_evaluation/* ────────────────────────────────

/// Input data for synthesis fitness judge rendering.
///
/// Fitness judges use empty ancestors (Pitfall 6 guard): no preamble injected.
pub struct SynthesisFitnessInput<'a> {
    pub draft: &'a SynthesisArtifact,
}

fn render_synthesis_claims_for_fitness(draft: &SynthesisArtifact) -> String {
    let mut buf = String::new();
    for claim in &draft.claims {
        let level = claim.agreement_level.as_deref().unwrap_or("divided");
        use super::render::render_sources_comma_joined;
        let sources = render_sources_comma_joined(&claim.sources);
        buf.push_str(&format!(
            "- **[C]** ({level}): \"{text}\"\n  Sources: {sources}\n",
            level = level,
            text = claim.text,
            sources = sources,
        ));
    }
    buf
}

fn synthesis_fitness_footer(dimension: &str) -> String {
    format!(
        r#"
---

## Required Output Format

Output ONLY the following XML block. Do NOT include preamble, commentary, or markdown code fences.

```xml
<fitness_evaluation>
  <score>3</score>
  <rationale>Multi-line analysis of {dimension}.</rationale>
  <suggestions>
    <suggestion>Specific actionable suggestion</suggestion>
  </suggestions>
</fitness_evaluation>
```

### ⚠️ CRITICAL CONSTRAINTS

- **Output ONLY the `<fitness_evaluation>` XML block.** No preamble, no markdown fences, no commentary.
- **`<score>` must be an integer from 1-5**.
- **`<rationale>` must reference specific claim IDs** when identifying issues.

Begin your evaluation (output raw XML only, starting with `<fitness_evaluation>`):
"#,
        dimension = dimension,
    )
}

/// Render `synthesis/fitness_evaluation/faithfulness.mustache`.
///
/// Faithfulness is the HARD CONSTRAINT validity gate (faithfulness ≥ 4 required).
/// No preamble injected — empty ancestors (Pitfall 6 guard).
pub fn render_synthesis_fitness_faithfulness(input: &SynthesisFitnessInput) -> String {
    let claims = render_synthesis_claims_for_fitness(input.draft);
    format!(
        r#"You are evaluating **Faithfulness** as part of an evolutionary fitness assessment.

## Your Role in the Evolutionary Process

You are one fitness evaluator in a multi-objective evolutionary system. Your singular focus is **faithfulness**.

Your role: ensure the synthesis accurately represents source material without introducing novel claims or distortions. This is a hard constraint for validity.

---

## Dimension: Faithfulness (HARD CONSTRAINT)

**Definition**: The degree to which the synthesis accurately represents source material without introducing novel claims, interpretations, or distortions.

---

## Synthesis Being Evaluated

{claims}

---

## Evaluation Rubric

| Band | Descriptor |
|------|------------|
| **5** | Fully Faithful: Every statement traces directly to sources. No novel claims. |
| **4** | Highly Faithful: All statements trace to sources. Minor rephrasing only. |
| **3** | Moderately Faithful: 80-90%% of statements trace to sources. |
| **2** | Poorly Faithful: 30-50%% of statements trace to sources. Significant distortion. |
| **1** | Unfaithful: <30%% of statements trace to sources. |

**Critical**: Scores below 4 mean the artefact is INVALID for Delphi controlled feedback.
{footer}
"#,
        claims = claims,
        footer = synthesis_fitness_footer("faithfulness"),
    )
}

/// Render `synthesis/fitness_evaluation/completeness.mustache`.
pub fn render_synthesis_fitness_completeness(input: &SynthesisFitnessInput) -> String {
    let claims = render_synthesis_claims_for_fitness(input.draft);
    format!(
        r#"You are evaluating **Completeness** as part of an evolutionary fitness assessment.

## Your Role in the Evolutionary Process

You are one fitness evaluator in a multi-objective evolutionary system. Your singular focus is **completeness**.

Your role: ensure the synthesis addresses all aspects of the question and represents all expert voices without omission.

---

## Dimension: Completeness

**Definition**: The degree to which the synthesis addresses all aspects of the question using all available expert input without omission.

---

## Synthesis Being Evaluated

{claims}

---

## Evaluation Rubric

| Band | Descriptor |
|------|------------|
| **5** | Fully Complete: Addresses all question aspects; all expert voices represented. |
| **4** | Highly Complete: Covers major aspects; minor omissions only. |
| **3** | Moderately Complete: Covers most aspects; several experts not represented. |
| **2** | Poorly Complete: Major question aspects missing; most experts not represented. |
| **1** | Incomplete: Addresses less than half the question; most expert voices absent. |
{footer}
"#,
        claims = claims,
        footer = synthesis_fitness_footer("completeness"),
    )
}

/// Render `synthesis/fitness_evaluation/traceability.mustache`.
pub fn render_synthesis_fitness_traceability(input: &SynthesisFitnessInput) -> String {
    let claims = render_synthesis_claims_for_fitness(input.draft);
    format!(
        r#"You are evaluating **Traceability** as part of an evolutionary fitness assessment.

## Your Role in the Evolutionary Process

You are one fitness evaluator in a multi-objective evolutionary system. Your singular focus is **traceability**.

Your role: ensure every statement can be traced to specific sources via explicit citations, enabling verification and audit.

---

## Dimension: Traceability

**Definition**: The degree to which every claim is explicitly linked to specific source responses via citations.

---

## Synthesis Being Evaluated

{claims}

---

## Evaluation Rubric

| Band | Descriptor |
|------|------------|
| **5** | Fully Traceable: Every claim has at least one specific source citation. |
| **4** | Highly Traceable: 90%+ claims cited; minor gaps only. |
| **3** | Moderately Traceable: 70-90%% claims cited. |
| **2** | Poorly Traceable: 50-70%% claims cited. |
| **1** | Untraceable: Less than 50%% claims have citations. |
{footer}
"#,
        claims = claims,
        footer = synthesis_fitness_footer("traceability"),
    )
}

/// Render `synthesis/fitness_evaluation/neutrality.mustache`.
pub fn render_synthesis_fitness_neutrality(input: &SynthesisFitnessInput) -> String {
    let claims = render_synthesis_claims_for_fitness(input.draft);
    format!(
        r#"You are evaluating **Neutrality** as part of an evolutionary fitness assessment.

## Your Role in the Evolutionary Process

You are one fitness evaluator in a multi-objective evolutionary system. Your singular focus is **neutrality**.

Your role: ensure the synthesis maintains strict neutrality — no evaluative language, advocacy, or implicit recommendations.

---

## Dimension: Neutrality

**Definition**: The degree to which the synthesis presents findings without evaluative language, advocacy, or implicit recommendations.

---

## Synthesis Being Evaluated

{claims}

---

## Evaluation Rubric

| Band | Descriptor |
|------|------------|
| **5** | Fully Neutral: No evaluative language; pure facilitation posture throughout. |
| **4** | Highly Neutral: Very minor instances of evaluative phrasing; easily corrected. |
| **3** | Moderately Neutral: Some evaluative language present; clear framing bias. |
| **2** | Poorly Neutral: Multiple advocacy statements or recommendations. |
| **1** | Not Neutral: Synthesis advocates for positions or makes policy recommendations. |
{footer}
"#,
        claims = claims,
        footer = synthesis_fitness_footer("neutrality"),
    )
}

/// Render `synthesis/fitness_evaluation/dissent_visibility.mustache`.
pub fn render_synthesis_fitness_dissent_visibility(input: &SynthesisFitnessInput) -> String {
    let claims = render_synthesis_claims_for_fitness(input.draft);
    format!(
        r#"You are evaluating **Dissent Visibility** as part of an evolutionary fitness assessment.

## Your Role in the Evolutionary Process

You are one fitness evaluator in a multi-objective evolutionary system. Your singular focus is **dissent visibility**.

Your role: ensure disagreement, minority views, and uncertainty are prominently and fairly represented, not collapsed or hidden.

---

## Dimension: Dissent Visibility

**Definition**: The degree to which disagreement, minority views, and uncertainty are clearly surfaced rather than averaged away.

---

## Synthesis Being Evaluated

{claims}

---

## Evaluation Rubric

| Band | Descriptor |
|------|------------|
| **5** | Fully Visible: All dissent, minority views, and uncertainties prominently represented. |
| **4** | Highly Visible: Most dissent visible; minor minority views may be condensed. |
| **3** | Moderately Visible: Some dissent visible; important minority views underrepresented. |
| **2** | Poorly Visible: Major disagreements collapsed; minority views marginalised. |
| **1** | Not Visible: Dissent averaged away; synthesis presents false consensus. |
{footer}
"#,
        claims = claims,
        footer = synthesis_fitness_footer("dissent_visibility"),
    )
}

/// Render `synthesis/fitness_evaluation/structural_clarity.mustache`.
pub fn render_synthesis_fitness_structural_clarity(input: &SynthesisFitnessInput) -> String {
    let claims = render_synthesis_claims_for_fitness(input.draft);
    format!(
        r#"You are evaluating **Structural Clarity** as part of an evolutionary fitness assessment.

## Your Role in the Evolutionary Process

You are one fitness evaluator in a multi-objective evolutionary system. Your singular focus is **structural clarity**.

Your role: ensure the synthesis is logically organized to facilitate expert comprehension and review in Delphi rounds.

---

## Dimension: Structural Clarity

**Definition**: The degree to which the synthesis is logically organized, clearly structured, and easy to navigate.

---

## Synthesis Being Evaluated

{claims}

---

## Evaluation Rubric

| Band | Descriptor |
|------|------------|
| **5** | Excellent Structure: Claims organized logically; clear agreement/disagreement distinction. |
| **4** | Good Structure: Well-organized; minor structural improvements possible. |
| **3** | Adequate Structure: Reasonably organized; some structural confusion. |
| **2** | Poor Structure: Disorganized; related claims not grouped; hard to navigate. |
| **1** | No Structure: Incoherent organization; claims scattered without logic. |
{footer}
"#,
        claims = claims,
        footer = synthesis_fitness_footer("structural_clarity"),
    )
}

// ── SynthesisMerger render ─────────────────────────────────────────────────────

/// Input for the synthesis merger render.
pub struct SynthesisMergerInput<'a> {
    pub candidates: &'a [SynthesisArtifact],
}

/// Render the synthesis merger prompt — fold sorted candidates into one.
pub fn render_synthesis_merger(input: &SynthesisMergerInput) -> String {
    let mut candidates_section = String::new();
    for (i, candidate) in input.candidates.iter().enumerate() {
        candidates_section.push_str(&format!(
            "## Candidate {} (rank {})\n\n### Claims\n\n",
            i + 1, i + 1
        ));
        candidates_section.push_str(&render_claims_section(&candidate.claims));
        candidates_section.push('\n');
    }

    format!(
        r#"You are merging multiple synthesis candidates into a single best synthesis.

The candidates are ranked best-first by fitness evaluation. Candidate 1 is the highest-ranked.

## Your Task

Produce a merged synthesis that:
1. Preserves the strongest elements from the top-ranked candidates
2. Resolves conflicts by favouring the higher-ranked candidate
3. Maintains all evidence citations from the source candidates

## Candidates (best-first)

{candidates_section}

## Required Output

Output ONLY the `<synthesis>` XML block. No preamble.

```xml
<synthesis>
  <narrative>Merged narrative</narrative>
  <claims>
    <claim id="C1">
      <text>Claim</text>
      <agreement_level>consensus|majority|divided|minority</agreement_level>
      <sources><source id="s1"><quote>quote</quote></source></sources>
      <counterarguments><counterargument>dissent</counterargument></counterarguments>
    </claim>
  </claims>
  <areas_of_agreement><area>area</area></areas_of_agreement>
  <areas_of_disagreement><area>area</area></areas_of_disagreement>
  <uncertainties><uncertainty>uncertainty</uncertainty></uncertainties>
</synthesis>
```

Begin your merge:
"#,
        candidates_section = candidates_section,
    )
}

// ── Template 8: committee/aggregator_revision ─────────────────────────────────

/// Input for the aggregator revision prompt.
pub struct AggregatorRevisionInput<'a> {
    /// IDs of experts not represented in the current synthesis.
    pub missing_expert_ids: &'a [String],
    /// The research question.
    pub question: &'a str,
    /// Expert responses (id, prose) tuples.
    pub expert_responses: &'a [(String, String)],
    /// Current synthesis claims as text.
    pub original_claims: &'a str,
    /// Current minority reports as text.
    pub original_minority_reports: &'a str,
    /// Current uncertainties as text.
    pub original_uncertainties: &'a str,
    /// Total number of experts.
    pub n_experts: usize,
    /// Optional graph context for missing experts.
    pub graph_context: Option<&'a str>,
}

/// Render `committee/aggregator_revision.mustache` — revision for missing experts.
///
/// Source: `consensus/prompts/committee/aggregator_revision.mustache` [VERIFIED]
/// Versioned: `v1/committee`.
///
/// Used in Stage-2 post-processing step 6 when `check_expert_coverage` finds
/// missing expert voices (strategies.py:519-534).
pub fn render_aggregator_revision(input: &AggregatorRevisionInput) -> String {
    use super::render::render_sources_comma_joined;

    // Missing experts as comma-joined list
    let missing_list = render_sources_comma_joined(input.missing_expert_ids);

    // Expert responses section
    let mut responses_section = String::new();
    for (expert_id, response) in input.expert_responses {
        responses_section.push_str(&format!(
            "--- Expert: {expert_id} ---\n{response}\n\n"
        ));
    }

    // Graph context block (conditional)
    let graph_block = match input.graph_context {
        Some(ctx) if !ctx.is_empty() => format!(
            "### Graph-Derived Positions for Missing Experts\n\n\
             The following positions were extracted from the argumentation graph \
             for the missing experts. Use these as structured evidence:\n\n\
             {ctx}\n\n"
        ),
        _ => String::new(),
    };

    format!(
        r#"## Task: Revision — Incorporate Missing Expert Voices

You are revising a committee synthesis. The original synthesis FAILED to represent all experts. Your job is to revise it so that every expert's distinctive position appears — as a claim, a counterargument, or a minority report.

**Missing experts:** {missing_list}

These experts' positions do not appear anywhere in the current synthesis. Read their responses below and revise the synthesis to include them.

### Rules
- Keep ALL existing claims — do not remove or weaken them.
- Add new claims or expand existing claims to incorporate the missing experts' positions.
- If a missing expert's position contradicts an existing claim, add it as a counterargument or as a new minority-agreement claim.
- If a missing expert's position is truly unique, add it as a minority report.
- Every missing expert MUST appear in at least one claim source, counterargument, or minority report.
- Preserve the same XML format as the original synthesis.

### Question
{question}

### Expert Responses (ALL — focus on the missing experts listed above)
{responses_section}
{graph_block}
### Current Synthesis (to revise)

**Claims:**
{original_claims}

**Minority Reports:**
{original_minority_reports}

**Uncertainties:**
{original_uncertainties}

### Output Format
Respond with the COMPLETE revised synthesis in the same XML format (inside a `<synthesis>` root element). Include ALL claims (existing + new/revised), areas, uncertainties, minority reports, and narrative.

For each claim, include `corroboration_count` attribute indicating how many of the {n_experts} experts support it.

```xml
<synthesis>
  <claims>
    <claim id="C1" agreement="consensus" corroboration_count="7">
      <text>Claim text</text>
      <sources>
        <source expert_id="expert_01">
          <quote>verbatim quote</quote>
        </source>
      </sources>
      <counterarguments>
        <counterargument>Challenge or alternative perspective</counterargument>
      </counterarguments>
    </claim>
  </claims>
  <areas_of_agreement><area>Area 1</area></areas_of_agreement>
  <areas_of_disagreement><area>Tension 1</area></areas_of_disagreement>
  <uncertainties><uncertainty>Uncertainty 1</uncertainty></uncertainties>
  <minority_reports>
    <report expert_id="expert_03" support_count="1">
      <position>Minority position</position>
      <evidence><quote>supporting quote</quote></evidence>
    </report>
  </minority_reports>
  <narrative>Revised narrative with inline citations [C1].</narrative>
</synthesis>
```
"#,
        missing_list = missing_list,
        question = input.question,
        responses_section = responses_section,
        graph_block = graph_block,
        original_claims = input.original_claims,
        original_minority_reports = input.original_minority_reports,
        original_uncertainties = input.original_uncertainties,
        n_experts = input.n_experts,
    )
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Render a claims section for use in synthesis prompts.
fn render_claims_section(claims: &[crate::ttd::artifact::Claim]) -> String {
    use super::render::render_sources_comma_joined;
    let mut buf = String::new();
    for claim in claims {
        let level = claim.agreement_level.as_deref().unwrap_or("divided");
        let sources = render_sources_comma_joined(&claim.sources);
        buf.push_str(&format!(
            "- **[C]** ({level}): \"{text}\"\n  Sources: {sources}\n",
            level = level,
            text = claim.text,
            sources = sources,
        ));
        if !claim.counterarguments.is_empty() {
            buf.push_str(&format!(
                "  Counterarguments: {}\n",
                claim.counterarguments.join(", ")
            ));
        }
        buf.push('\n');
    }
    buf
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttd::artifact::{ArgumentationGraph, Claim, GraphEdge, GraphNode, SynthesisArtifact};

    fn make_draft() -> SynthesisArtifact {
        let mut art = SynthesisArtifact::new(
            "study-1", "round-1", "q-1", "model", "v1/synthesis",
        );
        art.claims.push(Claim {
            text: "Climate change is accelerating.".into(),
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
        art.areas_of_agreement.push("Mechanism is clear".into());
        art.areas_of_disagreement.push("Rate is disputed".into());
        art.uncertainties.push("Long-term feedbacks unknown".into());
        art
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
        g.edges.push(GraphEdge {
            source: "arxiv:2105.14103_c001".into(),
            target: "arxiv:2105.14103_c002".into(),
            relation: "supports".into(),
        });
        g
    }

    // ── Test: all templates carry the version tag ────────────────────────────

    #[test]
    fn all_synthesis_templates_have_prompt_version() {
        assert_eq!(SYNTHESIS_PROMPT_VERSION, "v1/synthesis");
        assert_eq!(COMMITTEE_PROMPT_VERSION, "v1/committee");
    }

    // ── Test: draft_graph_snapshot — conditional block absent without unverified ─

    /// draft_graph renders without the unverified_nodes block when
    /// has_unverified_nodes=false (conditional {{#has_unverified_nodes}} = false).
    #[test]
    fn draft_graph_snapshot() {
        let graph = make_graph();
        let input = SynthesisDraftGraphInput {
            question: "What are the effects of permafrost thaw?".into(),
            expert_count: 3,
            graph: &graph,
            has_unverified_nodes: false,
            unverified_nodes: &[],
            target_length: "500-800",
        };

        let rendered = render_synthesis_draft_graph(&input);

        // Must contain the question
        assert!(
            rendered.contains("What are the effects of permafrost thaw?"),
            "rendered draft_graph must contain the question"
        );
        // Must contain the node claim text
        assert!(
            rendered.contains("Permafrost thaw releases methane"),
            "rendered draft_graph must contain graph node text"
        );
        // Must contain version indicator in template content
        assert!(
            rendered.contains("argumentation graph"),
            "rendered draft_graph must reference the argumentation graph"
        );
        // Without unverified nodes, the quality gates block must NOT appear
        assert!(
            !rendered.contains("Quality Gates"),
            "draft_graph with has_unverified_nodes=false must NOT render quality gates block"
        );
        // Target length hint must be present
        assert!(
            rendered.contains("500-800"),
            "target_length hint must appear in rendered output"
        );
    }

    // ── Test: draft_graph_conditional_block renders only when truthy ──────────

    /// {{#has_unverified_nodes}} block renders ONLY when has_unverified_nodes=true.
    #[test]
    fn draft_graph_conditional_block() {
        let graph = make_graph();

        // True: block renders
        let input_true = SynthesisDraftGraphInput {
            question: "Test question".into(),
            expert_count: 2,
            graph: &graph,
            has_unverified_nodes: true,
            unverified_nodes: &[("node_001".into(), "Unverified claim".into())],
            target_length: "500-800",
        };
        let rendered_true = render_synthesis_draft_graph(&input_true);
        assert!(
            rendered_true.contains("Quality Gates"),
            "has_unverified_nodes=true must render the quality gates block"
        );
        assert!(
            rendered_true.contains("Unverified claim"),
            "has_unverified_nodes=true must list unverified nodes"
        );

        // False: block does NOT render
        let input_false = SynthesisDraftGraphInput {
            question: "Test question".into(),
            expert_count: 2,
            graph: &graph,
            has_unverified_nodes: false,
            unverified_nodes: &[],
            target_length: "500-800",
        };
        let rendered_false = render_synthesis_draft_graph(&input_false);
        assert!(
            !rendered_false.contains("Quality Gates"),
            "has_unverified_nodes=false must NOT render the quality gates block"
        );
    }

    // ── Test: fitness judge no preamble (Pitfall 6 guard) ─────────────────────

    /// Synthesis fitness judge prompts must NOT contain "## Upstream context"
    /// (Pitfall 6 guard: no preamble injection).
    #[test]
    fn synthesis_fitness_judge_no_preamble() {
        let draft = make_draft();
        let input = SynthesisFitnessInput { draft: &draft };

        let judges = [
            render_synthesis_fitness_faithfulness(&input),
            render_synthesis_fitness_completeness(&input),
            render_synthesis_fitness_traceability(&input),
            render_synthesis_fitness_neutrality(&input),
            render_synthesis_fitness_dissent_visibility(&input),
            render_synthesis_fitness_structural_clarity(&input),
        ];

        for (i, judge) in judges.iter().enumerate() {
            assert!(
                !judge.contains("## Upstream context"),
                "synthesis fitness judge {i} must NOT contain '## Upstream context' preamble (Pitfall 6)"
            );
        }
    }

    // ── Test: gap_identify contains unescaped fitness feedback ───────────────

    /// The {{{fitness_feedback}}} block is injected raw (no HTML escaping).
    /// Raw markdown with & and < must NOT become &amp; or &lt;.
    #[test]
    fn synthesis_gap_identify_contains_unescaped_feedback() {
        let draft = make_draft();
        let feedback = "## Priority Improvements\n- **faithfulness**: score=2 (low & needs attention)\n- **completeness**: score < 3";
        let input = SynthesisGapIdentifyInput {
            draft: &draft,
            fitness_feedback: Some(feedback),
        };

        let rendered = render_synthesis_gap_identify(&input);

        // Feedback must be present
        assert!(
            rendered.contains("Priority Improvements"),
            "rendered gap_identify must contain the feedback heading"
        );
        // No HTML escaping — raw markdown must pass through unchanged
        assert!(
            !rendered.contains("&amp;"),
            "feedback must NOT be HTML-escaped (&amp; found — Pitfall 3 triple-mustache violation)"
        );
        assert!(
            !rendered.contains("&lt;"),
            "feedback must NOT be HTML-escaped (&lt; found)"
        );
        // The raw & and < must be present
        assert!(
            rendered.contains("& needs attention"),
            "raw & must pass through unescaped"
        );
    }

    // ── Test: aggregator_revision version tag ────────────────────────────────

    #[test]
    fn aggregator_revision_contains_version_constant() {
        assert_eq!(COMMITTEE_PROMPT_VERSION, "v1/committee");
        // Render a minimal revision and confirm it contains the missing expert ids
        let input = AggregatorRevisionInput {
            missing_expert_ids: &["expert_03".to_string()],
            question: "What is the effect of X?",
            expert_responses: &[("expert_03".into(), "Expert 03 says X is significant.".into())],
            original_claims: "C1: Climate change is real.",
            original_minority_reports: "None.",
            original_uncertainties: "Unknown feedback.",
            n_experts: 5,
            graph_context: None,
        };
        let rendered = render_aggregator_revision(&input);
        assert!(
            rendered.contains("expert_03"),
            "aggregator_revision must list the missing expert ids"
        );
        assert!(
            rendered.contains("Expert 03 says X is significant"),
            "aggregator_revision must include expert response text"
        );
    }
}
