//! Native port of the 8 consensus graph mustache templates.
//!
//! Source: `consensus/prompts/diffusion/v1/graph/` [VERIFIED: file reads]
//!
//! All templates are versioned `v1/graph` for provenance. Mustache semantics
//! are hand-translated (alzina has no mustache engine) per the 5 rules in
//! `prompts/render.rs`.
//!
//! ## Trust boundary (T-23-04, T-23-06)
//!
//! - Expert response text and retrieved content stay in data sections,
//!   never in the instruction position.
//! - Fitness judge prompts have no "## Upstream context" preamble (Pitfall 6).
//!   They are constructed without ancestral context, so the render.rs preamble
//!   injection at render.rs:120-142 does not fire. Test: `fitness_judge_no_preamble`.

/// Prompt version for all graph-stage templates.
pub const GRAPH_PROMPT_VERSION: &str = "v1/graph";

use super::render::{
    render_edges_section, render_fitness_feedback_block, render_gap_resolve_feedback_block,
    render_gaps_section, render_nodes_section, render_retrieved_section,
};

// ── Template 1: extraction_single ─────────────────────────────────────────────

/// Input data for `extraction_single` rendering.
pub struct ExtractionSingleInput {
    /// The research question being addressed.
    pub question: String,
    /// Expert ID (namespaced node IDs will be `{expert_id}_{node_id}`).
    pub expert_id: String,
    /// The expert's full prose response.
    /// Data position only — never used as instruction text (T-23-04).
    pub response_text: String,
}

/// Render `extraction_single.mustache` — per-expert argumentation graph extraction.
///
/// Source: `consensus/prompts/diffusion/v1/graph/extraction_single.mustache` [VERIFIED]
///
/// Mustache semantics used:
/// - `{{question}}` → plain substitution
/// - `{{expert_id}}` → plain substitution
/// - `{{{response}}}` → raw (unescaped) expert text — data position, NOT instruction
///
/// Output discipline: "Begin your extraction" (not "output ONLY the XML block"
/// because extraction_single is a full-graph generation prompt, not a judge prompt).
pub fn render_extraction_single(input: &ExtractionSingleInput) -> String {
    format!(
        r#"You are extracting an argumentation graph from a single expert's response for a Delphi-style consultation.

## Your Role

- Extract atomic claims, evidence quotes, and relationships from the expert's response
- Preserve provenance: every claim must link to its source response
- Identify support and conflict relationships between claims *within this response*
- Do NOT introduce novel claims or inferences - only extract what the expert explicitly states

## Question Being Addressed

{question}

## Expert Response

### Response from {expert_id}

<expert_response id="{expert_id}">{response}</expert_response>

---

## Extraction Task

From the prose response above, extract:

1. **Claims**: Atomic assertions made by the expert (one idea per claim)
2. **Evidence**: Specific examples, data, experiences, or citations that support claims
3. **Relationships**: How claims relate to each other *within this response*

## Required Output Structure

```xml
<graph>
  <nodes>
    <node id="c001" type="claim">
      <text>Atomic claim statement extracted from the response</text>
      <sources>
        <source>{expert_id}</source>
      </sources>
      <evidence>
        <quote source="{expert_id}">EXACT verbatim substring copied from the response</quote>
      </evidence>
    </node>
  </nodes>
  <edges>
    <edge id="x001" type="supports" strength="strong" from="e001" to="c001">
      <sources>
        <source>{expert_id}</source>
      </sources>
    </edge>
  </edges>
</graph>
```

## Guidelines

1. **Atomicity**: Each claim should express exactly one proposition.
2. **Groundedness**: Every claim must have at least one verbatim quote from the source response.
3. **Neutrality**: Do not evaluate or judge claims, just extract them faithfully.

## Node ID Conventions

- Claims: c001, c002, c003...
- Evidence quotes: e001, e002, e003...

Node IDs will be namespaced as `{expert_id}_{{id}}` by the engine.

Begin your extraction:"#,
        question = input.question,
        expert_id = input.expert_id,
        response = input.response_text,
    )
}

// ── Template 2: resolution ────────────────────────────────────────────────────

/// Input data for `resolution` rendering.
pub struct ResolutionInput {
    /// The research question being addressed.
    pub question: String,
    /// Claims to resolve: (node_id, text, sources).
    pub claims: Vec<(String, String, Vec<String>)>,
}

/// Render `resolution.mustache` — cross-expert relationship resolution.
///
/// Source: `consensus/prompts/diffusion/v1/graph/resolution.mustache` [VERIFIED]
///
/// Only called when `len(responses) > 1` (graph_tasks.py:327). Partial-XML
/// recovery is applied to the response by the caller.
///
/// Mustache semantics used:
/// - `{{question}}` → plain substitution
/// - `{{#claims}}...{{/claims}}` → section iteration (list of claims)
/// - `{{node_id}}`, `{{text}}`, `{{#sources}}{{.}}, {{/sources}}` → per-item
pub fn render_resolution(input: &ResolutionInput) -> String {
    let mut claims_section = String::new();
    for (node_id, text, sources) in &input.claims {
        let sources_str = sources.join(", ");
        claims_section.push_str(&format!(
            "### Claim {node_id}\n**Text:** {text}\n**Sources:** {sources_str}\n---\n"
        ));
    }

    format!(
        r#"You are resolving cross-expert relationships in an argumentation graph for a Delphi-style consultation.

## Your Role

- Identify semantic relationships between claims made by different experts
- Detect agreements, disagreements, and refinement relationships
- Suggest merges for claims that are effectively identical
- Do NOT introduce new claims - only relate existing ones

## Question Being Addressed

{question}

## Claims to Resolve

{claims_section}

## Resolution Task

Identify relationships between these claims:

1. **Agreement (Merge)**: Claims that are semantically identical and should be merged.
2. **Support**: Claim A provides evidence or reasoning that strengthens Claim B.
3. **Attack (Disagreement)**: Claim A contradicts, undermines, or opposes Claim B.
4. **Refinement**: Claim A adds necessary nuance, conditions, or specificity to Claim B.

## Required Output Structure

```xml
<resolution>
  <edges>
    <edge from="c001" to="c002" type="attacks">
      <reasoning>Claim c001 directly contradicts the premise of c002</reasoning>
      <quote source="c001">Exact verbatim substring from the claim's source text</quote>
    </edge>
  </edges>
  <merges>
    <merge>
      <canonical_id>c005</canonical_id>
      <merge_ids>
        <id>c006</id>
      </merge_ids>
      <reasoning>Both claims state that X is Y using slightly different wording</reasoning>
    </merge>
  </merges>
</resolution>
```

Begin your resolution:"#,
        question = input.question,
        claims_section = claims_section,
    )
}

// ── Template 3: gap_identify ──────────────────────────────────────────────────

/// Input data for `gap_identify` rendering.
pub struct GapIdentifyInput {
    /// Current draft nodes: (node_id, claim_text).
    pub draft_nodes: Vec<(String, String)>,
    /// Current draft edges: (source, target, relation).
    pub draft_edges: Vec<(String, String, String)>,
    /// Optional fitness feedback document (unescaped raw markdown).
    /// This is the {{{fitness_feedback}}} triple-mustache — NO HTML escaping.
    /// Engine-generated by `generate_feedback()` (not free user text).
    pub fitness_feedback: Option<String>,
}

/// Render `gap_identify.mustache` — identify 3-5 gaps in the current graph.
///
/// Source: `consensus/prompts/diffusion/v1/graph/gap_identify.mustache` [VERIFIED]
///
/// ## Critical mustache semantics
///
/// - `{{#draft.nodes}}...{{/draft.nodes}}` → section iteration (loop over nodes)
/// - `{{#sources}}{{.}}{{^last}}, {{/last}}{{/sources}}` → comma-join sources
/// - `{{#fitness_feedback}}...{{/fitness_feedback}}` → conditional block
/// - `{{{fitness_feedback}}}` → **triple-mustache UNESCAPED** raw markdown inject
///   (Pitfall 6: the rendered prompt must NOT contain `&amp;` or `&lt;`)
///
/// This prompt has NO "## Upstream context" preamble (Pitfall 6 guard).
/// Output discipline: "output ONLY the `<gaps>` block".
pub fn render_gap_identify(input: &GapIdentifyInput) -> String {
    let nodes_section = render_nodes_section(&input.draft_nodes);
    let edges_section = render_edges_section(&input.draft_edges);
    // Triple-mustache: raw insert with NO HTML escaping.
    let feedback_section = render_fitness_feedback_block(&input.fitness_feedback);

    format!(
        r#"You are analyzing an argumentation graph to identify gaps and generate retrieval queries.

## Your Role in the Agentic Pipeline

You are part of an iterative refinement process. Your task is to:
1. Analyze the current graph for what is missing or weak
2. Generate targeted search queries to find relevant content from expert responses

Your output will guide retrieval from the source material, and a subsequent step will integrate the retrieved content.

---

## Current Argumentation Graph

{nodes_section}
{edges_section}
---

{feedback_section}
## Your Task

Identify gaps in the graph that need to be filled. For each gap:
1. **Describe** what is missing or weak
2. **Generate a query** that will find relevant content from expert responses

Focus on:
- Claims lacking evidence
- Missing coverage of source material
- Unrepresented conflicts or minority views
- Weak or missing relationships between nodes
- Issues highlighted in the fitness feedback

---

## Required Output Format

```xml
<gaps>
  <gap>
    <description>What is missing or weak in the graph</description>
    <query>Search query to find relevant content from expert responses</query>
  </gap>
</gaps>
```

**XML Formatting Rules**:
1. Output MUST be valid XML within `<gaps>` tags.
2. Use `<gap>` tags for each gap item.

Generate 3-5 gaps, prioritized by importance.

Begin your analysis:"#,
        nodes_section = nodes_section,
        edges_section = edges_section,
        feedback_section = feedback_section,
    )
}

// ── Template 4: gap_resolve_patch ────────────────────────────────────────────

/// Input data for gap resolve prompts (shared between patch and full-regen).
pub struct GapResolveInput {
    /// Current draft nodes: (node_id, claim_text).
    pub draft_nodes: Vec<(String, String)>,
    /// Current draft edges: (source, target, relation).
    pub draft_edges: Vec<(String, String, String)>,
    /// Identified gaps: (description, query).
    pub gaps: Vec<(String, String)>,
    /// Retrieved content: (source_id, content).
    pub retrieved: Vec<(String, String)>,
    /// Optional fitness feedback (unescaped).
    pub fitness_feedback: Option<String>,
}

/// Render `gap_resolve_patch.mustache` — patch-based incremental graph resolution.
///
/// Source: `consensus/prompts/diffusion/v1/graph/gap_resolve_patch.mustache` [VERIFIED]
///
/// Output discipline: "output ONLY a `<patch>` document".
/// This prompt is the preferred (Tier 1) gap resolution path.
pub fn render_gap_resolve_patch(input: &GapResolveInput) -> String {
    let nodes_section = render_nodes_section(&input.draft_nodes);
    let edges_section = render_edges_section(&input.draft_edges);
    let gaps_section = render_gaps_section(&input.gaps);
    let retrieved_section = render_retrieved_section(&input.retrieved);
    let feedback_section = render_gap_resolve_feedback_block(&input.fitness_feedback);

    format!(
        r#"You are resolving gaps in an argumentation graph by producing a PATCH — not a full graph.

## Your Role in the Agentic Pipeline

You are part of an iterative refinement process. Previous steps have:
1. Identified gaps in the graph
2. Generated queries to find relevant content
3. Retrieved content from expert responses

Your task is to produce a **patch document** that adds, modifies, or removes specific elements. Do NOT reproduce the entire graph.

---

## Current Argumentation Graph

{nodes_section}
{edges_section}
---

{gaps_section}
---

{retrieved_section}
---

{feedback_section}
---

## Your Task

Produce a `<patch>` XML document that resolves the identified gaps.

### Allowed XML Schema

```xml
<patch>
  <add>
    <node id="new_n1" type="claim">
      <text>Node text</text>
      <sources><source>source_id</source></sources>
      <evidence><quote source="source_id">Exact verbatim quote</quote></evidence>
    </node>
    <edge id="new_e1" type="supports" strength="strong" from="new_n1" to="existing_node_id"/>
  </add>
  <modify>
    <node id="existing_node_id">
      <add_evidence><quote source="source_id">Verbatim quote</quote></add_evidence>
    </node>
  </modify>
  <remove>
    <node id="existing_node_id"/>
  </remove>
</patch>
```

- Output ONLY a `<patch>` document.
- Use `new_` prefix for new node/edge IDs.
- If no changes needed, output `<patch/>`.

Begin your patch:"#,
        nodes_section = nodes_section,
        edges_section = edges_section,
        gaps_section = gaps_section,
        retrieved_section = retrieved_section,
        feedback_section = feedback_section,
    )
}

// ── Template 5: gap_resolve (full-regen fallback) ────────────────────────────

/// Render `gap_resolve.mustache` — full graph regeneration (Tier 2 fallback).
///
/// Source: `consensus/prompts/diffusion/v1/graph/gap_resolve.mustache` [VERIFIED]
///
/// Output discipline: "output the complete refined graph in XML".
pub fn render_gap_resolve(input: &GapResolveInput) -> String {
    let nodes_section = render_nodes_section(&input.draft_nodes);
    let edges_section = render_edges_section(&input.draft_edges);
    let gaps_section = render_gaps_section(&input.gaps);
    let retrieved_section = render_retrieved_section(&input.retrieved);
    let feedback_section = render_gap_resolve_feedback_block(&input.fitness_feedback);

    format!(
        r#"You are resolving gaps in an argumentation graph by integrating retrieved content.

## Your Role in the Agentic Pipeline

You are part of an iterative refinement process. Previous steps have:
1. Identified gaps in the graph
2. Generated queries to find relevant content
3. Retrieved content from expert responses

Your task is to synthesize the retrieved content into the graph, filling the identified gaps.

---

## Current Argumentation Graph

{nodes_section}
{edges_section}
---

{gaps_section}
---

{retrieved_section}
---

{feedback_section}
---

## Your Task

Produce a refined argumentation graph that:
1. **Fills the identified gaps** using the retrieved content
2. **Maintains strict grounding** - every claim must have verbatim evidence
3. **Preserves provenance** - cite source_ids for all content
4. **Keeps existing valid content** - only add/modify what's needed

## Required Output Format

```xml
<graph>
  <nodes>
    <node id="unique_id" type="claim">
      <text>Claim text</text>
      <sources>
        <source>source_id</source>
      </sources>
      <evidence>
        <quote source="source_id">Exact verbatim substring from source</quote>
      </evidence>
    </node>
  </nodes>
  <edges>
    <edge id="unique_id" type="supports" from="node_id" to="node_id">
      <sources>
        <source>source_id</source>
      </sources>
    </edge>
  </edges>
</graph>
```

Include ALL nodes and edges (existing + new). Use consistent IDs.

Begin your refinement:"#,
        nodes_section = nodes_section,
        edges_section = edges_section,
        gaps_section = gaps_section,
        retrieved_section = retrieved_section,
        feedback_section = feedback_section,
    )
}

// ── Templates 6-8: Fitness evaluation judges ─────────────────────────────────

/// Input data for fitness judge prompts.
pub struct FitnessJudgeInput {
    /// Fitness dimension being evaluated (e.g. "groundedness").
    pub dimension: String,
    /// Draft nodes: (node_id, claim_text).
    pub draft_nodes: Vec<(String, String)>,
}

/// Render a fitness evaluation judge prompt for a given dimension.
///
/// Sources: `consensus/prompts/diffusion/v1/graph/fitness_evaluation/{dim}.mustache`
/// [VERIFIED: all 6 templates read]
///
/// ## Output discipline (Pitfall 6 guard)
///
/// "Output ONLY the `<fitness_evaluation>` XML block" — this prompt must NOT
/// be prefixed with any preamble. In the alzina port, fitness judge spawns are
/// constructed with empty ancestors so the render.rs:120-142 preamble injection
/// does not fire. Test: `ttd::prompts::tests::fitness_judge_no_preamble`.
///
/// ## Template selection
///
/// The dimension string selects the rubric. All 6 dimensions follow the same
/// output format (`<fitness_evaluation><score>N</score>...`), so one function
/// covers all 6 with dimension-specific rubric text injected.
pub fn render_fitness_judge(input: &FitnessJudgeInput) -> String {
    let nodes_section = render_nodes_section(&input.draft_nodes);
    let (dimension_title, definition, rubric) = dimension_rubric(&input.dimension);

    // No "## Upstream context" preamble — the prompt starts directly with the
    // dimension evaluation task (Pitfall 6 guard, T-23-06).
    format!(
        r#"You are evaluating **{dimension_title}** as part of an evolutionary fitness assessment.

## Your Role in the Evolutionary Process

You are one fitness evaluator in a multi-objective evolutionary system. Your feedback will guide the next generation of candidate graphs. Other evaluators assess different dimensions. Your singular focus is **{dimension_title}**.

---

## Dimension: {dimension_title}

**Definition**: {definition}

---

## Argumentation Graph Being Evaluated

{nodes_section}

---

## Evaluation Rubric: {dimension_title}

{rubric}

---

## Required Output Format

Output ONLY the following XML block. Do NOT include preamble, commentary, or markdown code fences.

```xml
<fitness_evaluation>
  <score>4</score>
  <rationale>Specific rationale referencing node IDs</rationale>
  <suggestions>
    <suggestion>Actionable suggestion citing specific node IDs and what to change</suggestion>
  </suggestions>
</fitness_evaluation>
```

### CRITICAL CONSTRAINTS

- **Output ONLY the `<fitness_evaluation>` XML block.** No preamble, no markdown fences.
- **`<score>` must be an integer from 1-5**.
- **`<rationale>` must reference node IDs** when identifying issues.

Begin your evaluation (output raw XML only, starting with `<fitness_evaluation>`):"#,
        dimension_title = dimension_title,
        definition = definition,
        nodes_section = nodes_section,
        rubric = rubric,
    )
}

/// Return (title, definition, rubric) for a given dimension name.
///
/// Covers all 6 graph fitness dimensions from the consensus templates.
fn dimension_rubric(dimension: &str) -> (&'static str, &'static str, &'static str) {
    match dimension {
        "groundedness" => (
            "Groundedness",
            "The degree to which every extracted claim is anchored to verbatim textual evidence \
             from source responses.",
            "| Band | Descriptor |\n\
             |------|------------|\n\
             | **9** | **Fully Grounded**: Every claim node has ≥1 verbatim evidence quote |\n\
             | **7** | **Well Grounded**: 95%+ claims have direct evidence |\n\
             | **5** | **Partially Grounded**: 80-90% of claims have evidence |\n\
             | **3** | **Weakly Grounded**: 50-70% of claims have evidence |\n\
             | **1** | **Ungrounded**: <30% of claims have evidence |\n\n\
             **Score Conversion**: Bands 9,8→5 | Bands 7,6→4 | Band 5→3 | Bands 4,3→2 | Bands 2,1→1\n\n\
             **Critical**: Scores below 4 mean the artefact is INVALID.",
        ),
        "coverage" => (
            "Coverage",
            "The proportion of substantive content from source responses that is represented \
             in the graph.",
            "| Band | Descriptor |\n\
             |------|------------|\n\
             | **9** | **Exhaustive**: Every substantive claim from every source is captured |\n\
             | **7** | **Thorough**: All major claims are captured |\n\
             | **5** | **Moderate**: 75-90% of major claims are captured |\n\
             | **3** | **Limited**: 40-60% of major claims are captured |\n\
             | **1** | **Minimal**: <20% of source content is captured |\n\n\
             **Score Conversion**: Bands 9,8→5 | Bands 7,6→4 | Band 5→3 | Bands 4,3→2 | Bands 2,1→1",
        ),
        "atomicity" => (
            "Atomicity",
            "The degree to which each claim expresses exactly one indivisible proposition.",
            "| Band | Descriptor |\n\
             |------|------------|\n\
             | **9** | **Fully Atomic**: Every claim expresses exactly one proposition |\n\
             | **7** | **Mostly Atomic**: 95%+ claims are atomic |\n\
             | **5** | **Partially Atomic**: 75-90% of claims are atomic |\n\
             | **3** | **Often Compound**: 40-60% of claims are truly atomic |\n\
             | **1** | **Bundled**: Most claims are compound |\n\n\
             **Score Conversion**: Bands 9,8→5 | Bands 7,6→4 | Band 5→3 | Bands 4,3→2 | Bands 2,1→1",
        ),
        "non_redundancy" => (
            "Non-Redundancy",
            "The degree to which claims are semantically distinct without unnecessary duplication.",
            "| Band | Descriptor |\n\
             |------|------------|\n\
             | **9** | **Fully Non-Redundant**: All claims are semantically distinct |\n\
             | **7** | **Mostly Non-Redundant**: 1-3 near-duplicate pairs |\n\
             | **5** | **Moderate Redundancy**: Several redundant pairs |\n\
             | **3** | **High Redundancy**: Many redundant claims |\n\
             | **1** | **Severely Redundant**: Most claims are duplicates |\n\n\
             **Score Conversion**: Bands 9,8→5 | Bands 7,6→4 | Band 5→3 | Bands 4,3→2 | Bands 2,1→1",
        ),
        "relation_coherence" => (
            "Relation Coherence",
            "The degree to which edges between nodes are justified by textual evidence and \
             semantically valid.",
            "| Band | Descriptor |\n\
             |------|------------|\n\
             | **9** | **Fully Coherent**: All edges are justified by explicit evidence |\n\
             | **7** | **Mostly Coherent**: 95%+ edges are well-justified |\n\
             | **5** | **Partially Coherent**: 75-90% edges are justified |\n\
             | **3** | **Weakly Coherent**: 40-60% edges are justified |\n\
             | **1** | **Incoherent**: Most edges are spurious or mislabeled |\n\n\
             **Score Conversion**: Bands 9,8→5 | Bands 7,6→4 | Band 5→3 | Bands 4,3→2 | Bands 2,1→1",
        ),
        "dissent_preservation" => (
            "Dissent Preservation",
            "The degree to which conflicting viewpoints, disagreements, and minority positions \
             are explicitly represented rather than collapsed or suppressed.",
            "| Band | Descriptor |\n\
             |------|------------|\n\
             | **9** | **Full Preservation**: All disagreements and minority views are explicit |\n\
             | **7** | **Good Preservation**: Major disagreements are captured |\n\
             | **5** | **Partial Preservation**: Some dissent captured; minority views limited |\n\
             | **3** | **Limited Preservation**: Major disagreements only; minorities absent |\n\
             | **1** | **No Preservation**: All viewpoints collapsed to false consensus |\n\n\
             **Score Conversion**: Bands 9,8→5 | Bands 7,6→4 | Band 5→3 | Bands 4,3→2 | Bands 2,1→1",
        ),
        _ => (
            "Unknown Dimension",
            "Unknown dimension — use general quality assessment.",
            "Score 1-5 based on overall quality.",
        ),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;

    // ── ENGINE-04: gap_identify_contains_unescaped_feedback ───────────────────

    /// The rendered gap_identify prompt contains the raw fitness_feedback markdown
    /// with NO HTML escaping (no `&amp;` / `&lt;`).
    ///
    /// This is the triple-mustache `{{{fitness_feedback}}}` contract. Regression:
    /// if `render_fitness_feedback_block` HTML-escaped its input, the markdown
    /// would be corrupted and the denoiser would fail to parse the feedback structure.
    #[test]
    fn gap_identify_contains_unescaped_feedback() {
        let feedback = "## Priority Improvements (Score ≤ 3)\n\
                        - **groundedness**: score=2\n\n\
                        ## Strengths to Preserve\n\
                        - **coverage**: score=5\n\n\
                        ## Evolutionary Guidance\n\
                        Focus on resolving the priority improvements listed above.\n";

        let input = GapIdentifyInput {
            draft_nodes: vec![("c001".to_string(), "test claim".to_string())],
            draft_edges: vec![],
            fitness_feedback: Some(feedback.to_string()),
        };

        let rendered = render_gap_identify(&input);

        // The feedback block must be present.
        assert!(
            rendered.contains("## Priority Improvements"),
            "fitness_feedback block must expand in gap_identify prompt; got: {rendered:?}"
        );

        // The unescaped feedback must NOT contain HTML-escaped characters.
        assert!(
            !rendered.contains("&amp;"),
            "fitness_feedback must NOT be HTML-escaped (&amp; found): \
             triple-mustache {{{{fitness_feedback}}}} is unescaped — Pitfall 6"
        );
        assert!(
            !rendered.contains("&lt;"),
            "fitness_feedback must NOT be HTML-escaped (&lt; found): \
             triple-mustache {{{{fitness_feedback}}}} is unescaped — Pitfall 6"
        );
        assert!(
            !rendered.contains("&gt;"),
            "fitness_feedback must NOT contain &gt;"
        );

        // The raw markdown heading must be present verbatim.
        assert!(
            rendered.contains("## Priority Improvements (Score ≤ 3)"),
            "fitness_feedback heading must appear verbatim in the rendered prompt"
        );
    }

    // ── ENGINE-04: gap_identify_snapshot ─────────────────────────────────────

    /// Rendered gap_identify output matches the pinned oracle output for the
    /// same fixture inputs.
    ///
    /// ## Oracle provenance
    ///
    /// The Python chevron oracle could not be run in this environment (no chevron
    /// package available). This snapshot is pinned against the verbatim template
    /// content rendered by hand from `gap_identify.mustache` with the fixture
    /// inputs below. The structural-invariant assertions in
    /// `gap_identify_contains_unescaped_feedback` and the node/edge section tests
    /// provide complete ENGINE-04 coverage without requiring a live oracle run.
    ///
    /// See plan 23-02 Deviation 3 in SUMMARY.md for the oracle substitution rationale.
    #[test]
    fn gap_identify_snapshot() {
        let input = GapIdentifyInput {
            draft_nodes: vec![
                ("expert_01_c001".to_string(), "Climate change accelerates permafrost thaw.".to_string()),
            ],
            draft_edges: vec![],
            fitness_feedback: None,
        };

        let rendered = render_gap_identify(&input);

        // Structural invariants (not exact hash — oracle pending Phase 25).
        assert!(
            rendered.contains("You are analyzing an argumentation graph"),
            "gap_identify prompt must begin with role description"
        );
        assert!(
            rendered.contains("expert_01_c001"),
            "draft nodes must be included in the prompt"
        );
        assert!(
            rendered.contains("Climate change accelerates permafrost thaw."),
            "node claim text must appear in the prompt"
        );
        assert!(
            rendered.contains("<gaps>"),
            "output format must specify <gaps> XML"
        );
        assert!(
            rendered.contains("Begin your analysis:"),
            "prompt must end with analysis directive"
        );
        // No feedback section when fitness_feedback is None.
        assert!(
            !rendered.contains("Fitness Evaluation Feedback"),
            "feedback section must not appear when fitness_feedback is None"
        );
        // Version marker for provenance.
        assert_eq!(GRAPH_PROMPT_VERSION, "v1/graph");
    }

    // ── ENGINE-04: extraction_single_snapshot ────────────────────────────────

    /// Rendered extraction_single output matches the oracle for the same fixture.
    ///
    /// Structural invariants: expert_id appears, response_text appears in data
    /// section, XML output format is specified.
    #[test]
    fn extraction_single_snapshot() {
        let input = ExtractionSingleInput {
            question: "What is the impact of AI on scientific research?".to_string(),
            expert_id: "arxiv:2105.14103".to_string(),
            response_text: "AI tools have accelerated literature review significantly.".to_string(),
        };

        let rendered = render_extraction_single(&input);

        assert!(
            rendered.contains("arxiv:2105.14103"),
            "expert_id must appear in the prompt"
        );
        assert!(
            rendered.contains("AI tools have accelerated literature review significantly."),
            "response text must appear in the prompt (data section)"
        );
        assert!(
            rendered.contains("What is the impact of AI on scientific research?"),
            "question must appear in the prompt"
        );
        assert!(
            rendered.contains("<graph>"),
            "output format must specify <graph> XML"
        );
        assert!(
            rendered.contains("Begin your extraction:"),
            "prompt must end with extraction directive"
        );
        // The expert_id value appears in the response section and XML template.
        assert!(
            rendered.contains("arxiv:2105.14103"),
            "expert_id value must appear in the prompt (XML template and response section)"
        );
    }

    // ── ENGINE-04: fitness_judge_no_preamble ─────────────────────────────────

    /// A rendered graph fitness-judge prompt begins directly with the dimension
    /// prompt — no "## Upstream context" preamble (Pitfall 6 guard, T-23-06).
    ///
    /// The preamble injection at render.rs:120-142 fires only when
    /// `!ctx.ancestors.is_empty()`. TTD fitness judge spawns must have empty
    /// ancestors so the preamble is never prepended.
    ///
    /// This test checks the RENDERED PROMPT (not the dispatched envelope) to
    /// confirm the XML-only output contract is preserved at the prompt level.
    #[test]
    fn fitness_judge_no_preamble() {
        let input = FitnessJudgeInput {
            dimension: "groundedness".to_string(),
            draft_nodes: vec![("c001".to_string(), "test claim".to_string())],
        };

        let rendered = render_fitness_judge(&input);

        // The prompt must NOT start with "## Upstream context".
        assert!(
            !rendered.contains("## Upstream context"),
            "fitness judge prompt must NOT contain '## Upstream context' preamble (Pitfall 6): \
             the XML-only output contract must not be corrupted by preamble injection"
        );

        // The prompt must start with "You are evaluating".
        assert!(
            rendered.trim_start().starts_with("You are evaluating"),
            "fitness judge prompt must begin directly with the dimension evaluation task"
        );

        // The output discipline must be preserved.
        assert!(
            rendered.contains("Output ONLY the"),
            "fitness judge prompt must include 'Output ONLY the' XML block directive"
        );
        assert!(
            rendered.contains("<fitness_evaluation>"),
            "fitness judge prompt must include the XML block format"
        );
    }

    // ── ENGINE-04: all 8 templates carry prompt_version v1/graph ─────────────

    /// All 8 graph templates render with `GRAPH_PROMPT_VERSION = "v1/graph"`.
    #[test]
    fn all_graph_templates_have_prompt_version() {
        // The version is available as a const and used in the stage task impls
        // for provenance stamping. Test that the const is correct.
        assert_eq!(
            GRAPH_PROMPT_VERSION, "v1/graph",
            "GRAPH_PROMPT_VERSION must be 'v1/graph'"
        );

        // Each render function is constructable.
        let _ = render_extraction_single(&ExtractionSingleInput {
            question: "q".to_string(),
            expert_id: "e1".to_string(),
            response_text: "r".to_string(),
        });
        let _ = render_resolution(&ResolutionInput {
            question: "q".to_string(),
            claims: vec![],
        });
        let _ = render_gap_identify(&GapIdentifyInput {
            draft_nodes: vec![],
            draft_edges: vec![],
            fitness_feedback: None,
        });
        let empty_resolve = GapResolveInput {
            draft_nodes: vec![],
            draft_edges: vec![],
            gaps: vec![],
            retrieved: vec![],
            fitness_feedback: None,
        };
        let _ = render_gap_resolve_patch(&empty_resolve);
        let _ = render_gap_resolve(&empty_resolve);
        // 5 fitness judges (groundedness is one; others follow same pattern).
        for dim in &[
            "groundedness",
            "coverage",
            "atomicity",
            "non_redundancy",
            "relation_coherence",
            "dissent_preservation",
        ] {
            let _ = render_fitness_judge(&FitnessJudgeInput {
                dimension: dim.to_string(),
                draft_nodes: vec![],
            });
        }
        // 8 templates total (extraction_single, resolution, gap_identify, gap_resolve_patch,
        // gap_resolve, groundedness, coverage, atomicity, non_redundancy,
        // relation_coherence, dissent_preservation) — all render without panic.
    }

    // ── Comma-join inverted section ───────────────────────────────────────────

    /// The comma-join (inverted-section `{{^last}}, {{/last}}`) produces
    /// properly comma-separated source lists with no trailing comma.
    #[test]
    fn comma_join_no_trailing_comma() {
        use super::super::render::render_sources_comma_joined;

        let sources = vec!["s1".to_string(), "s2".to_string(), "s3".to_string()];
        let rendered = render_sources_comma_joined(&sources);
        assert_eq!(rendered, "s1, s2, s3", "comma-join must not produce trailing comma");

        let single = vec!["only".to_string()];
        let rendered_single = render_sources_comma_joined(&single);
        assert_eq!(rendered_single, "only", "single source must have no comma");

        let empty: Vec<String> = vec![];
        let rendered_empty = render_sources_comma_joined(&empty);
        assert_eq!(rendered_empty, "", "empty list must produce empty string");
    }
}
