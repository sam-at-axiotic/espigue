//! v2 lit-review prompt rendering — shared building blocks for B2 prompts.
//!
//! MAS-PROMPT-CRAFT-V2 §1: all vocabulary terms are rendered verbatim from
//! term_sheet.rs constants — no paraphrasing. Every downstream v2 prompt
//! imports from this module to satisfy the inter-prompt term-coupling requirement.
//!
//! ## Module layout
//!
//! - **Vocabulary**: `render_vocabulary_block()` — iterates SUPPORT_LEVELS +
//!   EVIDENCE_GRADES + GAP_TYPES from term_sheet.rs at render time (§1).
//! - **Guards**: `GUARDS_BLOCK` const — five named anti-degeneration guards
//!   (D-3, §7). Guard names are stable; B3 judge anchors reference them verbatim.
//! - **Input quality gate**: `INPUT_QUALITY_GATE` + `CONFLICT_PATH` consts (D-6, §8).
//! - **Re-weave**: `REWEAVE_BLOCK` const (D-7, §5).
//! - **Output schema**: `V2_OUTPUT_SCHEMA` const — the D-5 XML contract as
//!   instruction text, pasted verbatim into generation prompts.
//! - **Paper framing**: `render_paper_header()` + `render_papers_section()` —
//!   paper provenance renderer (D-4). Replaces `<expert_response>` dressing.
//! - **Stage prompts**: all v2 render functions for Stages 1, 2, and 3.
//!
//! ## Security note (T-B2-01)
//!
//! Paper prose crosses the paper-text → prompt boundary inside a `<paper_text>`
//! wrapper — data position only, never instruction position (T-23-04 discipline
//! carried forward from v1). The wrapper tag is renamed from `expert_response`
//! so the model is never told these are survey respondents.

use crate::adapter::ExpertResponse;
use crate::ttd::artifact::{ArgumentationGraph, SynthesisArtifact};
use crate::ttd::plan::{render_plan_block, PlanSection, ReviewPlan, BANNED_PHRASES};
use crate::ttd::term_sheet::{
    EVIDENCE_GRADES, GAP_TYPES, JudgeDim, NarrativeShape, SUPPORT_LEVELS,
};
#[cfg(test)]
use crate::ttd::term_sheet::V2_JUDGE_DIMS;

// ── Persona version ───────────────────────────────────────────────────────────

/// Version identifier for the v2 deep-reviewer persona set.
pub const V2_PERSONA_SET_VERSION: &str = "v2/lit-review";

// ── Vocabulary block (§1 verbatim rule) ──────────────────────────────────────

/// Render the shared vocabulary block for v2 prompts.
///
/// Iterates SUPPORT_LEVELS, EVIDENCE_GRADES, and GAP_TYPES from term_sheet.rs
/// at render time — never paraphrases (MAS §1). B3 judge anchors use the same
/// consts, so a change to term_sheet propagates to both prompts and judges.
pub fn render_vocabulary_block() -> String {
    let mut out = String::from("## Vocabulary\n\
        The following terms have specific meanings in this pipeline. \
        Use them precisely. Do not paraphrase or substitute synonyms.\n\n");

    out.push_str("### Support levels (`support_level`)\n");
    for t in &SUPPORT_LEVELS {
        out.push_str(&format!("- **{}**: {}\n", t.name, t.definition));
    }

    out.push_str("\n### Evidence grades (`evidence_grade`)\n");
    for t in &EVIDENCE_GRADES {
        out.push_str(&format!("- **{}**: {}\n", t.name, t.definition));
    }

    out.push_str("\n### Gap types (`gap_type`)\n");
    for t in &GAP_TYPES {
        out.push_str(&format!("- **{}**: {}\n", t.name, t.definition));
    }

    out
}

// ── Anti-degeneration guards (D-3, §7) ───────────────────────────────────────

/// The five named anti-degeneration guards for v2 generation and refinement prompts.
///
/// Guard names are stable — B3 judge anchors reference them verbatim:
/// anti-vote-counting, anti-hedge, anti-list, anti-formulaic, anti-recency.
///
/// §0 complexity budget: this block is ~150 tokens; include in full on all
/// generation/refinement prompts and omit from the XML-output discipline itself.
pub const GUARDS_BLOCK: &str = "\
## Anti-degeneration guards

**anti-vote-counting**: Do not derive support_level by counting how many papers agree. \
Support levels are epistemics of the evidence: independence of methods, quality of corroboration, \
depth over time. Five papers re-analysing one dataset are weaker support than two independent replications.

**anti-hedge**: Take positions. Where the evidence supports a claim, state it plainly. \
Do not add \"may\", \"might\", \"could potentially\", or \"it is possible that\" unless the uncertainty \
is itself evidenced — and then say what the evidence of uncertainty is.

**anti-list**: Write flowing argument, not bullet lists. Claims connect through reasoning, not enumeration.

**anti-formulaic**: The following phrases are banned: \"it is important to note\", \
\"plays a crucial role\", \"in conclusion\", \"a growing body of literature\", \
\"further research is needed\" (the last is allowed only when naming a specific typed gap).

**anti-recency**: Weight evidence by quality and independence, not publication date. \
A 2019 replication outweighs a 2025 preprint. Use years to trace lineage, never to rank truth.";

// ── Input quality gate + conflict path (D-6, §8) ─────────────────────────────

/// Input quality gate instruction (D-6).
///
/// D-6 design decision: rejected input stays inside `<synthesis>` as the first
/// `<uncertainty>` — NOT a separate `<input_rejection>` root. A bare
/// `<input_rejection>` root would hit `ParseFailed("no <synthesis> block")`
/// and kill the run (fail-fast gather, WR-03). The gate stays loud but survivable.
pub const INPUT_QUALITY_GATE: &str = "\
## Input quality gate

Before synthesising, assess the papers. If they are irrelevant to the question, \
or too thin to support a review, do NOT manufacture claims — state the insufficiency \
as the FIRST <uncertainty>, label what little can be said honestly \
(`single-source` / `emerging`), and emit typed <gap> entries for what is missing.";

/// Irreconcilable-conflict path instruction (D-6).
pub const CONFLICT_PATH: &str = "\
## Conflict handling

When papers genuinely conflict, do NOT vote, do NOT average, do NOT manufacture a \
bridging narrative. Emit the claim with support_level `contested`, cite BOTH sides as \
sources, carry the dissenting position in <counterarguments> with its source named, \
and record the dispute's history in <lineage> (who claimed, who contradicted, when).";

// ── Re-weave instruction (D-7, §5) ───────────────────────────────────────────

/// Re-weave instruction for every v2 refinement/revision prompt.
///
/// Core wording from §5; synthesis verbs only — no "add", "include", "mention".
/// Anti-agreement guard from §4 appended (prompts that receive fitness feedback).
pub const REWEAVE_BLOCK: &str = "\
## Refinement instruction

You are rewriting, not appending. Produce a complete, self-contained replacement that \
integrates new material where it bears on existing claims; create new claims only for \
genuinely new propositions. Output must be the same length or shorter. \
A reader should not see the seams.

Synthesis verbs only: weave, surface, distinguish, integrate, recast. \
Do not use \"add\", \"include\", \"mention\" as instruction verbs.

If you believe a critique is wrong, argue back with evidence rather than complying.";

// ── v2 output schema contract (D-5) ──────────────────────────────────────────

/// The v2 synthesis output schema as instruction text.
///
/// This is the D-5 XML contract. Every v2 generation and merge prompt includes
/// this block verbatim so the instructed shape matches the parser contract in
/// stages/synthesis.rs:parse_synthesis_xml(V2LitReview).
///
/// Critical parser contracts (from the plan interfaces section):
/// - `<gap type="...">description as inline body text</gap>` — body NOT a child
/// - Self-closing `<source id="..."/>` is accepted
/// - NO `<agreement_level>` in v2 instructions
/// - support_level vocab: established | converging | contested | emerging | single-source
pub const V2_OUTPUT_SCHEMA: &str = r#"## Required output structure

Output ONLY the XML block below. No preamble, no commentary, no markdown code fences.

```xml
<synthesis>
  <narrative>Flowing review prose, {target_length} words. Established findings stated plainly; contested ones presented with both sides. Inline [Cx] citations where claims appear.</narrative>
  <claims>
    <claim id="C1">
      <text>One atomic claim</text>
      <support_level>established|converging|contested|emerging|single-source</support_level>
      <evidence_grade>strong|moderate|weak|anecdotal</evidence_grade>
      <method>brief note on how the evidence was produced</method>
      <year>2021</year>
      <lineage>first shown by {paper} ({year}); replicated by {paper} ({year}); challenged by {paper} ({year})</lineage>
      <sources><source id="arxiv:2304.07620"/></sources>
      <quotes><quote source="arxiv:2304.07620">EXACT verbatim substring copied from that source's graph quote or retrieved context</quote></quotes>
      <counterarguments><counterargument>published disagreement, with its source named</counterargument></counterarguments>
    </claim>
  </claims>
  <areas_of_agreement><area>convergent findings across independent lines of work</area></areas_of_agreement>
  <areas_of_disagreement><area>active disputes with named opposing positions</area></areas_of_disagreement>
  <uncertainties><uncertainty>open questions the field has not resolved</uncertainty></uncertainties>
  <gaps>
    <gap type="epistemic">what the field cannot currently know, and why</gap>
  </gaps>
</synthesis>
```

CRITICAL CONSTRAINTS:
- Every claim MUST have at least one `<source id="..."/>`.
- Every claim SHOULD carry at least one `<quote source="...">` — an EXACT verbatim \
substring copied character-for-character from the named source's quoted evidence \
(graph quotes or retrieved context). Quotes are checked mechanically against the \
stored source texts; paraphrases stamp the claim's quote absent.
- support_level MUST be exactly one of: established, converging, contested, emerging, single-source.
- evidence_grade MUST be exactly one of: strong, moderate, weak, anecdotal.
- gap type MUST be exactly one of: epistemic, empirical, methodological, theoretical.
- The gap description is the element body — do NOT use a `<description>` child.
- Do NOT add any vote-count or ballot element to claims.
- Do NOT wrap the XML in a markdown code block."#;

/// F14 merger-stage output schema. Like `V2_OUTPUT_SCHEMA` but each authored
/// `<quote>` carries a `node` attribute naming the graph node it was copied
/// from, and claims preserve their `<node_refs>`. The merger copies quotes
/// verbatim from the supplied node evidence — it never invents them.
pub const V2_MERGER_SCHEMA: &str = r#"## Required output structure

Output ONLY the XML block below. No preamble, no commentary, no markdown code fences.

```xml
<synthesis>
  <narrative>Flowing review prose, {target_length} words. Established findings stated plainly; contested ones presented with both sides. Inline [Cx] citations where claims appear.</narrative>
  <claims>
    <claim id="C1">
      <text>One atomic claim</text>
      <support_level>established|converging|contested|emerging|single-source</support_level>
      <evidence_grade>strong|moderate|weak|anecdotal</evidence_grade>
      <method>brief note on how the evidence was produced</method>
      <year>2021</year>
      <lineage>first shown by {paper} ({year}); replicated by {paper} ({year})</lineage>
      <sources><source id="arxiv:2304.07620"/></sources>
      <node_refs><ref>arxiv:2304.07620_C1</ref></node_refs>
      <quotes><quote source="arxiv:2304.07620" node="arxiv:2304.07620_C1">EXACT verbatim substring copied from that node's quote evidence</quote></quotes>
      <counterarguments><counterargument>published disagreement, with its source named</counterargument></counterarguments>
    </claim>
  </claims>
  <areas_of_agreement><area>convergent findings across independent lines of work</area></areas_of_agreement>
  <areas_of_disagreement><area>active disputes with named opposing positions</area></areas_of_disagreement>
  <uncertainties><uncertainty>open questions the field has not resolved</uncertainty></uncertainties>
  <gaps>
    <gap type="epistemic">what the field cannot currently know, and why</gap>
  </gaps>
</synthesis>
```

CRITICAL CONSTRAINTS:
- Every claim MUST have at least one `<source id="..."/>` and preserve its `<node_refs>`.
- Each `<quote>` MUST be copied character-for-character from the quote evidence of a \
node that claim cites, and MUST carry both `source` (the paper id) and `node` (the \
node id). Do NOT invent quotes; if no cited node has a usable passage, omit the quote.
- support_level MUST be exactly one of: established, converging, contested, emerging, single-source.
- evidence_grade MUST be exactly one of: strong, moderate, weak, anecdotal.
- gap type MUST be exactly one of: epistemic, empirical, methodological, theoretical.
- The gap description is the element body — do NOT use a `<description>` child.
- Do NOT add any vote-count or ballot element to claims.
- Do NOT wrap the XML in a markdown code block."#;

/// F14 draft-stage output schema. Identical to `V2_OUTPUT_SCHEMA` except claims
/// cite the supporting **graph node ids** (`<node_refs>`) instead of authoring
/// `<quotes>`. The draft model never copies quotes — quote authoring moves to
/// the Opus merger, which is given the cited nodes' relevant sections and copies
/// verbatim from them. Drafts still carry `<sources>` (the traceability veto
/// keys on sources, not quotes), so judge/tournament selection is unaffected.
pub const V2_DRAFT_SCHEMA: &str = r#"## Required output structure

Output ONLY the XML block below. No preamble, no commentary, no markdown code fences.

```xml
<synthesis>
  <narrative>Flowing review prose, {target_length} words. Established findings stated plainly; contested ones presented with both sides. Inline [Cx] citations where claims appear.</narrative>
  <claims>
    <claim id="C1">
      <text>One atomic claim</text>
      <support_level>established|converging|contested|emerging|single-source</support_level>
      <evidence_grade>strong|moderate|weak|anecdotal</evidence_grade>
      <method>brief note on how the evidence was produced</method>
      <year>2021</year>
      <lineage>first shown by {paper} ({year}); replicated by {paper} ({year}); challenged by {paper} ({year})</lineage>
      <sources><source id="arxiv:2304.07620"/></sources>
      <node_refs><ref>arxiv:2304.07620_C1</ref><ref>arxiv:2308.06046_C2</ref></node_refs>
      <counterarguments><counterargument>published disagreement, with its source named</counterargument></counterarguments>
    </claim>
  </claims>
  <areas_of_agreement><area>convergent findings across independent lines of work</area></areas_of_agreement>
  <areas_of_disagreement><area>active disputes with named opposing positions</area></areas_of_disagreement>
  <uncertainties><uncertainty>open questions the field has not resolved</uncertainty></uncertainties>
  <gaps>
    <gap type="epistemic">what the field cannot currently know, and why</gap>
  </gaps>
</synthesis>
```

CRITICAL CONSTRAINTS:
- Every claim MUST have at least one `<source id="..."/>`.
- Every claim MUST cite the graph node ids it draws from in `<node_refs>`. The \
argumentation graph above is markdown: each node appears as a bold id token, e.g. \
`- **arxiv:2304.07620_C1** [verified] — <claim text>`. Copy that bold id verbatim \
into one `<ref>` per supporting node (e.g. `<ref>arxiv:2304.07620_C1</ref>`), \
character-for-character — these ids are matched exactly to attach the verbatim \
quotes, so an altered or abbreviated id silently drops the evidence. Do NOT author \
quotes yourself — the quotes are attached later from the cited nodes' source text. \
Cite only nodes that genuinely support the claim.
- support_level MUST be exactly one of: established, converging, contested, emerging, single-source.
- evidence_grade MUST be exactly one of: strong, moderate, weak, anecdotal.
- gap type MUST be exactly one of: epistemic, empirical, methodological, theoretical.
- The gap description is the element body — do NOT use a `<description>` child.
- Do NOT add any vote-count or ballot element to claims.
- Do NOT wrap the XML in a markdown code block."#;

// ── Paper framing (D-4) ───────────────────────────────────────────────────────

/// Render a single paper header with provenance and `<paper_text>` wrapper.
///
/// D-4 design: paper prose stays in the data position (T-B2-01 / T-23-04).
/// The `<paper_text>` tag replaces `<expert_response>` — the model is never told
/// these are survey respondents.
///
/// Venue is NOT included (papers schema has no venue column — title/year/authors
/// delivered instead; venue deviation recorded in SUMMARY).
pub fn render_paper_header(response: &ExpertResponse) -> String {
    let title = &response.provenance.title;
    let year = match response.provenance.year {
        Some(y) => y.to_string(),
        None => "year unknown".to_string(),
    };
    let authors = if response.provenance.authors.is_empty() {
        "authors unknown".to_string()
    } else {
        response.provenance.authors.join(", ")
    };
    let paper_id = response.expert_id.as_str();

    format!(
        "### Paper {paper_id}: {title} ({year})\n\
         Authors: {authors}\n\n\
         <paper_text id=\"{paper_id}\">{prose}</paper_text>\n",
        prose = response.prose
    )
}

/// Render all papers in the panel as a section.
pub fn render_papers_section(inputs: &[ExpertResponse]) -> String {
    let mut out = String::from("## Papers\n\n");
    for resp in inputs {
        out.push_str(&render_paper_header(resp));
        out.push('\n');
    }
    out
}

/// One-line provenance headers, no bodies (quote-grounded synthesis, item 3).
///
/// Stage-2 prompts ground on the graph's quoted evidence plus gap-retrieval
/// context — paper bodies stay out of the prompt (a full-text dump at panel
/// size 27-30 approached ~150k tokens). This list exists only so the prose
/// can name papers with correct provenance.
pub fn render_paper_headers_section(inputs: &[ExpertResponse]) -> String {
    let mut out = String::from("## Sources (provenance only — evidence lives in the graph)\n\n");
    for r in inputs {
        let year = r
            .provenance
            .year
            .map(|y| y.to_string())
            .unwrap_or_else(|| "year unknown".to_string());
        let authors = if r.provenance.authors.is_empty() {
            "authors unknown".to_string()
        } else {
            r.provenance.authors.join(", ")
        };
        // Stage-2 soft-filter: tag the source header with its credibility tier so
        // the draft sees credibility from the first Stage-2 prompt it reads.
        // `Unknown` adds no tag (byte-identical to the pre-tier header).
        let tier_tag = match r.provenance.credibility_tier {
            alzina_search::CredibilityTier::Unknown => String::new(),
            t => format!(" [{}]", t.label()),
        };
        out.push_str(&format!(
            "- {}{}: {} ({}) — {}\n",
            r.expert_id.as_str(),
            tier_tag,
            r.provenance.title,
            year,
            authors,
        ));
    }
    out
}

// ── Stage 1: extraction + resolution + gap prompts ────────────────────────────

/// Input for the v2 per-paper extraction prompt.
pub struct ExtractionSingleV2Input {
    pub paper_id: String,
    pub title: String,
    pub year: Option<i32>,
    pub authors: Vec<String>,
    pub prose: String,
    /// Mechanical source-credibility tier, surfaced in the header so the tier is
    /// visible from the first stage a paper appears in. Pipeline-internal only.
    pub credibility_tier: alzina_search::CredibilityTier,
}

/// Render v2 Stage-1 extraction prompt for a single paper.
///
/// D-4: paper framing — no Delphi pseudo-question, no `<expert_response>` dressing.
/// The graph output format is dialect-neutral (graph parser is shared by v1/v2).
pub fn render_extraction_single_v2(input: &ExtractionSingleV2Input) -> String {
    let year_str = match input.year {
        Some(y) => y.to_string(),
        None => "year unknown".to_string(),
    };
    let authors_str = if input.authors.is_empty() {
        "authors unknown".to_string()
    } else {
        input.authors.join(", ")
    };

    format!(
        r#"You are extracting the argumentation structure of a published paper. \
Your task is to identify the paper's main claims, the evidence it offers, \
and the logical relationships between them.

## Paper

### {paper_id}: {title} ({year}) — {tier}
Authors: {authors}

<paper_text id="{paper_id}">{prose}</paper_text>

## Required output structure

Output ONLY the XML block. No preamble, no commentary.

```xml
<graph>
  <node id="{paper_id}_C1" type="claim">
    <text>The main claim</text>
    <quote>EXACT verbatim substring copied character-for-character from the paper text above</quote>
    <expert_id>{paper_id}</expert_id>
    <verification_status>verified</verification_status>
  </node>
  <edge source="{paper_id}_C1" target="{paper_id}_C2" type="supports"/>
</graph>
```

Node types: claim, evidence, method, finding. Edge types: supports, contradicts, qualifies, instantiates.
Every node id must be namespaced: `{paper_id}_Cn`.

Groundedness: every node MUST carry a `<quote>` that is an exact verbatim substring \
of the paper text. Copy it character-for-character — quotes are checked mechanically \
against the source text, and a paraphrased quote stamps the claim unverified.
"#,
        paper_id = input.paper_id,
        title = input.title,
        year = year_str,
        tier = input.credibility_tier.label(),
        authors = authors_str,
        prose = input.prose,
    )
}

/// Render v2 Stage-1 resolution prompt (cross-paper relationships).
///
/// Conflict instruction: contradictions become attack edges, never merges (D-9).
/// Anti-recency on edge weights: do not weight recent papers over older ones.
pub fn render_resolution_v2(
    claims_data: &[(String, String, Vec<String>)],
    graph_xml: &str,
) -> String {
    let claims_list: String = claims_data
        .iter()
        .map(|(id, claim, papers)| {
            format!("- {id}: \"{claim}\" [sources: {papers}]", papers = papers.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"You are mapping cross-paper relationships in an argumentation graph built from {n} papers.

## Claims

{claims}

## Current graph

<graph_context>{graph}</graph_context>

## Task

Identify relationships between claims from DIFFERENT papers. Add `supports`, `contradicts`, \
or `qualifies` edges. When papers genuinely contradict each other, add a `contradicts` edge — \
do NOT merge the claims or manufacture a bridging interpretation.

Anti-recency: do not weight relationships by publication date.

## Output

Output ONLY the XML. Add `<edge>` elements to the existing graph. \
If no cross-paper relationships exist, output the graph unchanged.

```xml
<graph>
  <!-- existing nodes unchanged; add cross-paper edges here -->
  <edge source="paper_a_C1" target="paper_b_C2" type="contradicts"/>
</graph>
```
"#,
        n = claims_data.len(),
        claims = claims_list,
        graph = graph_xml,
    )
}

/// Render v2 Stage-1 gap identify prompt (retrieval gap format — unchanged from v1).
///
/// D-9: retrieval gap output format (`<description>/<query>`) is dialect-neutral.
/// Anti-recency applied to query framing.
pub fn render_gap_identify_v2(graph_xml: &str, question: &str) -> String {
    format!(
        r#"You are identifying gaps in the literature coverage of an argumentation graph.

## Research question

{question}

## Current graph

<graph_context>{graph}</graph_context>

## Task

Identify 3-5 gaps in the literature coverage. A gap is a claim or relationship \
that would benefit from additional paper evidence. Weight gaps by importance to \
the research question, not by how recent the missing evidence might be.

## Output

Output ONLY the XML block.

```xml
<gaps>
  <gap>
    <description>What is missing from the current evidence base</description>
    <query>Search query to find relevant papers</query>
  </gap>
</gaps>
```
"#,
        question = question,
        graph = graph_xml,
    )
}

/// Render v2 Stage-1 gap resolve (patch) prompt.
pub fn render_gap_resolve_patch_v2(
    graph_xml: &str,
    gap_description: &str,
    retrieved_text: &str,
) -> String {
    format!(
        r#"You are weaving newly retrieved papers into an argumentation graph.

## Gap to address

{gap}

## Retrieved papers

<retrieved_context>{retrieved}</retrieved_context>

## Current graph

<graph_context>{graph}</graph_context>

## Task

{reweave}

Add nodes and edges to the graph to incorporate the retrieved evidence. \
Where retrieved papers contradict existing nodes, add `contradicts` edges — \
do NOT merge conflicting claims.

Every NEW node MUST carry a `<quote>` that is an exact verbatim substring of \
the retrieved context above (copy it character-for-character — quotes are \
checked mechanically against the stored source text). Keep existing nodes' \
content, including their `<quote>` elements, unchanged.

Output ONLY the updated graph XML.
"#,
        gap = gap_description,
        retrieved = retrieved_text,
        graph = graph_xml,
        reweave = REWEAVE_BLOCK,
    )
}

/// Render v2 Stage-1 gap resolve (full regen) prompt.
pub fn render_gap_resolve_v2(
    graph_xml: &str,
    gap_description: &str,
    retrieved_text: &str,
) -> String {
    // Full regen: same as patch for Stage 1 — graph structure is small enough
    // that full regen and patch are equivalent.
    render_gap_resolve_patch_v2(graph_xml, gap_description, retrieved_text)
}

// ── Stage 2: synthesis draft + gap prompts ────────────────────────────────────

/// Render v2 Stage-2 synthesis draft (graph-seeded) prompt.
///
/// This is the heart of the v2 redesign. Reviewer role (not facilitator),
/// full vocabulary block, all 5 guards, quality gate, conflict path, D-5 schema.
pub fn render_synthesis_draft_graph_v2(
    question: &str,
    graph: &ArgumentationGraph,
    inputs: &[ExpertResponse],
    target_length: &str,
) -> String {
    // Item 3: headers only — claims ground in the graph's quoted evidence,
    // not in paper bodies (which no longer enter stage-2 prompts).
    let papers_section = render_paper_headers_section(inputs);
    let vocab = render_vocabulary_block();
    let schema = V2_DRAFT_SCHEMA.replace("{target_length}", target_length);
    let node_count = graph.nodes.len();

    format!(
        r#"You are writing a critical review of the literature on this question, synthesising {n} papers.

Your task is to produce a structured synthesis that surfaces what the evidence shows, \
where it converges, where it conflicts, and what remains genuinely unknown.

## Research question

{question}

{vocab}

{guards}

{quality_gate}

{conflict_path}

## Argumentation graph (Stage 1 output — {nodes} nodes)

The graph below was extracted from the papers. Use it as a structural scaffold — \
the claims and relationships it identifies are starting points, not authoritative verdicts. \
Its quoted passages are your evidence base: when you quote, copy a graph quote verbatim \
and cite its source paper id. Do not invent quotes — they are checked mechanically \
against the stored source texts.

<graph_context>{graph_md}</graph_context>

{papers}

{schema}
"#,
        n = inputs.len(),
        question = question,
        vocab = vocab,
        guards = GUARDS_BLOCK,
        quality_gate = INPUT_QUALITY_GATE,
        conflict_path = CONFLICT_PATH,
        nodes = node_count,
        // Markdown over XML for the synthesis stage (Sam, 2026-06-12): the
        // markdown rendering carries quotes, verification status, and edge
        // previews grouped by source — richer grounding context than the
        // id/claim/source XML projection. Stage-1 graph prompts keep XML;
        // there the model reads AND emits the graph dialect.
        graph_md = graph.to_markdown(),
        papers = papers_section,
        schema = schema,
    )
}

/// Render v2 Stage-2 plain synthesis draft (no graph).
pub fn render_synthesis_draft_v2(
    question: &str,
    inputs: &[ExpertResponse],
    target_length: &str,
) -> String {
    let papers_section = render_papers_section(inputs);
    let vocab = render_vocabulary_block();
    let schema = V2_DRAFT_SCHEMA.replace("{target_length}", target_length);

    format!(
        r#"You are writing a critical review of the literature on this question, synthesising {n} papers.

## Research question

{question}

{vocab}

{guards}

{quality_gate}

{conflict_path}

{papers}

{schema}
"#,
        n = inputs.len(),
        question = question,
        vocab = vocab,
        guards = GUARDS_BLOCK,
        quality_gate = INPUT_QUALITY_GATE,
        conflict_path = CONFLICT_PATH,
        papers = papers_section,
        schema = schema,
    )
}

/// Render v2 Stage-2 synthesis gap-identify prompt.
///
/// Lit-coverage framing: gaps are in the reviewed corpus, not in expert coverage.
/// D-9: retrieval gap output format (`<description>/<query>`) unchanged.
pub fn render_synthesis_gap_identify_v2(
    synthesis_xml: &str,
    question: &str,
    fitness_feedback: Option<&str>,
) -> String {
    let feedback_block = if let Some(fb) = fitness_feedback {
        format!(
            "\n## Fitness feedback\n\n\
             The following weaknesses were identified in the current synthesis:\n\n{fb}\n"
        )
    } else {
        String::new()
    };

    format!(
        r#"You are identifying gaps in the literature coverage of a synthesis.

## Research question

{question}

## Current synthesis

<synthesis_context>{synthesis}</synthesis_context>
{feedback}

## Task

Identify 3-5 claims in the synthesis that need stronger or broader paper support. \
Weight gaps by epistemic importance, not recency of potential evidence.

## Output

Output ONLY the XML block.

```xml
<gaps>
  <gap>
    <description>Which claim needs more support and why</description>
    <query>Search query to find relevant papers</query>
  </gap>
</gaps>
```
"#,
        question = question,
        synthesis = synthesis_xml,
        feedback = feedback_block,
    )
}

/// Render v2 Stage-2 synthesis gap-resolve patch prompt.
pub fn render_synthesis_gap_resolve_patch_v2(
    synthesis_xml: &str,
    gap_description: &str,
    retrieved_papers: &[ExpertResponse],
    target_length: &str,
) -> String {
    let papers = render_papers_section(retrieved_papers);
    let vocab = render_vocabulary_block();
    let schema = V2_OUTPUT_SCHEMA.replace("{target_length}", target_length);

    format!(
        r#"You are weaving newly retrieved papers into a literature synthesis.

## Gap to address

{gap}

## Retrieved papers

{papers}

## Current synthesis

<synthesis_context>{synthesis}</synthesis_context>

{vocab}

{guards}

{reweave}

{conflict_path}

{schema}
"#,
        gap = gap_description,
        papers = papers,
        synthesis = synthesis_xml,
        vocab = vocab,
        guards = GUARDS_BLOCK,
        reweave = REWEAVE_BLOCK,
        conflict_path = CONFLICT_PATH,
        schema = schema,
    )
}

/// Render v2 Stage-2 synthesis gap-resolve full-regen prompt.
///
/// Field survival guarantee: instructs the full v2 claim schema so
/// support_level/lineage/gaps survive the wholesale replacement.
pub fn render_synthesis_gap_resolve_full_v2(
    synthesis_xml: &str,
    gap_description: &str,
    retrieved_papers: &[ExpertResponse],
    target_length: &str,
) -> String {
    // Full regen uses the same structure as patch for synthesis — the difference
    // is that the caller passes the full synthesis context (not just the gap area).
    // Both instruct the full v2 schema, satisfying the field-survival guarantee.
    render_synthesis_gap_resolve_patch_v2(synthesis_xml, gap_description, retrieved_papers, target_length)
}

/// Render v2 Stage-2 synthesis merger prompt.
///
/// Merge preserves v2 fields (support_level, lineage, etc.) and the UNION of
/// sources. Conflicts between candidates resolve by evidence_grade, not rank alone.
///
/// P-3 (contract/context alignment): every field the "## Merge rules" section
/// cites — sources, support_level, evidence_grade, method, year, lineage — is
/// rendered into each candidate's claim blocks, so the merger arbitrates on
/// evidence it can actually see rather than regenerating provenance from prior.
/// Field rendering mirrors `render_fitness_judge_v2_synthesis` (conditional
/// lines; absent v2 fields are omitted, never rendered as placeholders).
/// Depth-probe B (2026-06-16): tell the merger/revision to spend the `context
/// section` blocks on mechanism, not field position. The reader has domain
/// expertise and needs to grep the idea — how each approach works — not whether
/// the field agrees on it. Grounded in the sections; no speculation past them.
const MECHANISM_DEPTH_BLOCK: &str = r#"## Depth of each claim

The `context section` blocks above are the real prose around each quoted passage —
the paper's own method, results, and ablation text. Use them. A claim's text must
convey the IDEA itself, at a level a domain expert can act on, not just the
field's position on it. For each claim, where the sections support it, state:

- how the approach works — its mechanism or architecture, in concrete terms;
- the specific result or ablation that isolates its contribution (with the number
  when the section gives one);
- its key assumption, condition, or failure mode.

Ground every mechanism statement in the context sections — never speculate beyond
them. Prefer one claim that explains an approach in depth over three that only
assert where it lands. Do not collapse distinct mechanisms into a generic "method
X improves task Y" — name what is actually happening."#;

/// Render a comma-joined source list, each id tagged with its credibility tier
/// (Stage-2 soft-filter, visibility only). A source whose tier is `Unknown` or
/// is absent from the map renders untagged, so an empty map reproduces a plain
/// `sources.join(", ")` — the pre-tier baseline.
fn render_sources_with_tier(
    sources: &[String],
    tier_map: &std::collections::BTreeMap<String, alzina_search::CredibilityTier>,
) -> String {
    sources
        .iter()
        .map(|s| {
            match tier_map
                .get(s)
                .filter(|t| **t != alzina_search::CredibilityTier::Unknown)
            {
                Some(t) => format!("{s} [{}]", t.label()),
                None => s.clone(),
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn render_synthesis_merger_v2(
    candidates: &[&SynthesisArtifact],
    question: &str,
    target_length: &str,
    node_evidence: &str,
    tier_map: &std::collections::BTreeMap<String, alzina_search::CredibilityTier>,
) -> String {
    let vocab = render_vocabulary_block();
    // F14: the merger authors verbatim quotes by copying from node evidence and
    // tagging each with its node id; it also preserves the candidates' node_refs.
    let schema = V2_MERGER_SCHEMA.replace("{target_length}", target_length);

    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut block = format!("### Candidate {}\n", i + 1);
            for (j, cl) in c.claims.iter().enumerate() {
                block.push_str(&format!("\n#### Claim {}\n\n", j + 1));
                block.push_str(&format!("**Text**: {}\n", cl.text));
                if cl.sources.is_empty() {
                    block.push_str("**Sources**: (none)\n");
                } else {
                    block.push_str(&format!(
                        "**Sources**: {}\n",
                        render_sources_with_tier(&cl.sources, tier_map)
                    ));
                }
                if let Some(ref sl) = cl.support_level {
                    block.push_str(&format!("**support_level**: {}\n", sl));
                }
                if let Some(ref eg) = cl.evidence_grade {
                    block.push_str(&format!("**evidence_grade**: {}\n", eg));
                }
                if let Some(ref m) = cl.method {
                    block.push_str(&format!("**method**: {}\n", m));
                }
                if let Some(ref y) = cl.year {
                    block.push_str(&format!("**year**: {}\n", y));
                }
                if let Some(ref lin) = cl.lineage {
                    block.push_str(&format!("**lineage**: {}\n", lin));
                }
            }
            block
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"You are merging {n} synthesis candidates into one authoritative review.

## Research question

{question}

## Candidates

{candidates}

## Quote evidence (verified verbatim passages from the argumentation graph)

Below are the graph's DB-verified quotes, grouped by source paper, each tagged
with its node id. For every claim you author, look up the claim's cited
source(s), find the node(s) under those source(s) whose quote best supports the
claim, and copy ONE verbatim quote into the claim, tagging it with that node id:
`<quote source="PAPER_ID" node="NODE_ID">copied text</quote>`. Copy
character-for-character — quotes are checked mechanically. Set the claim's
`<node_refs>` to the node id(s) you quoted from. Do NOT invent quotes or copy
from anywhere else. If no node under the claim's sources has a passage that
genuinely supports it, omit the quote (it is attached deterministically
afterwards) — never fabricate one.

{node_evidence}

{depth}

{vocab}

{guards}

{reweave}

{conflict_path}

## Merge rules

- Preserve the UNION of sources across all candidates for each claim.
- When candidates disagree on support_level for the same claim, choose the level \
  best supported by the evidence_grade of the sourcing papers — not by candidate rank.
- Preserve all lineage, method, year, and evidence_grade fields from the best-evidenced candidate.
- Set each claim's `<node_refs>` to the graph node id(s) you quoted from above.
- Do not manufacture new claims not present in any candidate.
- Source ID rule: the "Candidate N" section headings above are presentation labels for working
  drafts. They are never valid source ids. Every `<source id>` in your output MUST be a
  paper id from the provenance headers (arxiv:/s2:/doi namespaces) — never "Candidate1",
  "Candidate2", or any other candidate label.

{schema}
"#,
        n = candidates.len(),
        question = question,
        candidates = candidates_text,
        node_evidence = if node_evidence.trim().is_empty() {
            "(no node evidence available — author no quotes; quotes are attached afterwards)".to_string()
        } else {
            node_evidence.to_string()
        },
        depth = MECHANISM_DEPTH_BLOCK,
        vocab = vocab,
        guards = GUARDS_BLOCK,
        reweave = REWEAVE_BLOCK,
        conflict_path = CONFLICT_PATH,
        schema = schema,
    )
}

/// Render v2 revision prompt (D-8) — replaces aggregator_revision on the v2 path.
///
/// Reframed from "incorporate missing expert voices" to "weave uncovered papers into
/// the review". Lists missing paper_ids with title/year/authors, instructs re-weave,
/// full v2 schema so support_level/lineage/gaps survive wholesale replacement.
/// No corroboration_count, no agreement attribute, no minority_reports (v1 ballot artifacts).
pub fn render_revision_v2(
    synthesis: &SynthesisArtifact,
    panel: &[ExpertResponse],
    missing_paper_ids: &[String],
    target_length: &str,
    node_evidence: &str,
) -> String {
    let vocab = render_vocabulary_block();
    // F14 (option B, probe-24): the revision REPLACES the synthesis wholesale, so
    // it must author node-grounded verbatim quotes like the merger — otherwise it
    // strips them (V2_OUTPUT_SCHEMA carries no node attribution). Use the merger
    // schema and feed it the same full-graph evidence.
    let schema = V2_MERGER_SCHEMA.replace("{target_length}", target_length);

    // Stage-2 soft-filter: source_id -> tier from panel provenance, so the
    // revision sees credibility next to each source it re-weaves over — same
    // visibility-only treatment as the merger. `Unknown` renders no tag.
    let tier_map: std::collections::BTreeMap<String, alzina_search::CredibilityTier> = panel
        .iter()
        .map(|r| (r.expert_id.as_str().to_string(), r.provenance.credibility_tier))
        .collect();

    // List missing papers with their provenance from the panel.
    let missing_papers: String = missing_paper_ids
        .iter()
        .filter_map(|id| panel.iter().find(|r| r.expert_id.as_str() == id.as_str()))
        .map(|r| {
            let year = r.provenance.year.map(|y| y.to_string()).unwrap_or_else(|| "year unknown".to_string());
            let authors = if r.provenance.authors.is_empty() {
                "authors unknown".to_string()
            } else {
                r.provenance.authors.join(", ")
            };
            // Tier tag from provenance; `Unknown` adds nothing (byte-identical).
            let tier_tag = match r.provenance.credibility_tier {
                alzina_search::CredibilityTier::Unknown => String::new(),
                t => format!(" [{}]", t.label()),
            };
            // Item 3: header only — no <paper_text> body. The revision grounds
            // in the synthesis's existing quoted claims; bodies stay out.
            format!(
                "- **{}**{}: {} ({}) — {}\n",
                r.expert_id.as_str(),
                tier_tag,
                r.provenance.title,
                year,
                authors,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let original_claims: String = synthesis
        .claims
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let sources = if c.sources.is_empty() {
                String::new()
            } else {
                format!(" [sources: {}]", render_sources_with_tier(&c.sources, &tier_map))
            };
            let support = c.support_level.as_deref().unwrap_or("unknown");
            format!("C{n}: ({support}) {text}{sources}", n = i + 1, text = c.text)
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Prong 1 (Fix B): source-vocabulary prohibition.
    // "Candidate N" labels are the engine's internal presentation labels for
    // working drafts. They are never valid source ids — using one as a
    // <source id> in the output launders a working label into provenance.
    let source_id_rule = "\n## Source ID rule\n\n\
        The paper ids in the provenance headers above are the ONLY valid source ids \
        (arxiv:/s2:/doi namespaces). \"Candidate 1\", \"Candidate 2\", and similar labels \
        are presentation labels for internal drafts and are never valid source ids. \
        Every <source id> in your output must be a paper id — not a candidate label.\n";

    format!(
        r#"You are weaving uncovered papers into an existing literature review.

The papers listed below were not incorporated in the current synthesis. \
Integrate them where they bear on existing claims — do NOT append a separate section.

## Uncovered papers

{missing}

## Current synthesis claims

{claims}

## Quote evidence (verified verbatim passages from the argumentation graph)

Below are the graph's DB-verified quotes, grouped by source paper, each tagged
with its node id. For every claim in the revised synthesis, look up the claim's
cited source(s), find the node(s) under those source(s) whose quote best
supports the claim, and copy ONE verbatim quote into the claim, tagging it with
that node id: `<quote source="PAPER_ID" node="NODE_ID">copied text</quote>`.
Copy character-for-character — quotes are checked mechanically. Set the claim's
`<node_refs>` to the node id(s) you quoted from. Do NOT invent quotes. If no
node under the claim's sources supports it, omit the quote — never fabricate one.

{node_evidence}

{depth}

{vocab}

{guards}

{reweave}

{conflict_path}

{source_id_rule}

{schema}
"#,
        missing = if missing_papers.is_empty() {
            "None — all papers covered.".to_string()
        } else {
            missing_papers
        },
        claims = original_claims,
        node_evidence = node_evidence,
        depth = MECHANISM_DEPTH_BLOCK,
        vocab = vocab,
        guards = GUARDS_BLOCK,
        reweave = REWEAVE_BLOCK,
        conflict_path = CONFLICT_PATH,
        source_id_rule = source_id_rule,
        schema = schema,
    )
}

// ── Stage 3: narrative prompts ────────────────────────────────────────────────

/// Render v2/v3 Stage-3 narrative draft prompt.
///
/// Voice: "you are writing the narrative of a critical literature review".
/// Established findings stated plainly; contested ones presented with both sides.
/// [Cx] citations kept. Guards block applied.
///
/// Output shape (Decision 0 / Phase 0, constraint site 1 of 3):
/// - `NarrativeShape::Concise` — 300-500 words, headerless, byte-identical to
///   the pre-Phase-0 prompt.
/// - `NarrativeShape::SectionedLongForm` — `##` headings, no fixed word cap.
///
/// `shape` must come from `PromptProfile::narrative_shape()` so draft, refine,
/// and merge always agree — one run yields one shape.
pub fn render_narrative_draft_v2(synthesis: &SynthesisArtifact, shape: NarrativeShape) -> String {
    let claims_summary: String = synthesis
        .claims
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let support = c.support_level.as_deref().unwrap_or("unknown");
            format!("[C{n}] ({support}) {text}", n = i + 1, text = c.text)
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Decision 0 / Phase 0: the ONLY shape-dependent text. The Concise arm is
    // byte-identical to the pre-Phase-0 template (including the literal
    // backslash-newline continuation inside the raw string).
    let (output_form, closing) = match shape {
        NarrativeShape::Concise => (
            "a 300-500 word flowing prose summary",
            "Write the narrative below. 300-500 words. No headers, no bullet lists — flowing prose only.",
        ),
        NarrativeShape::SectionedLongForm => (
            "a sectioned long-form critical review",
            "Write the review below. Organise it under markdown ## section headings of your choosing. \
No fixed word cap — develop each argument at the length the evidence demands. \
Flowing prose within sections — no bullet lists.",
        ),
    };

    format!(
        r#"You are writing the narrative of a critical literature review.

Your task is to produce {output_form} that communicates \
the state of the evidence clearly and honestly:
- State established findings plainly, without hedging.
- Present contested claims with BOTH sides and their sources.
- Weave support levels into the prose — do not list them as labels.
- Use inline [Cx] citation markers where claims appear.
- Do not introduce claims not present in the synthesis below.

## Claims to narrate

{claims}

## Areas of agreement

{agreement}

## Areas of disagreement

{disagreement}

## Uncertainties

{uncertainties}

{guards}

{closing}
"#,
        claims = claims_summary,
        agreement = synthesis.areas_of_agreement.join("; "),
        disagreement = synthesis.areas_of_disagreement.join("; "),
        uncertainties = synthesis.uncertainties.join("; "),
        guards = GUARDS_BLOCK,
    )
}

/// Render v2 Stage-3 narrative critique prompt.
///
/// Criteria extended with tension-visibility and lineage-clarity checks.
/// References v2 schema concepts, not judge dims (those are B3).
pub fn render_narrative_critique_v2(
    narrative: &str,
    synthesis: &SynthesisArtifact,
    fitness_feedback: Option<&str>,
) -> String {
    let synthesis_claims_count = synthesis.claims.len();
    // C-N2: prior-step fitness, embedded only when present. Default path passes
    // `None` — the block collapses to empty and the prompt is byte-identical.
    let feedback_block = match fitness_feedback {
        Some(fb) => format!("\n## Prior fitness evaluation\n\n{fb}\n"),
        None => String::new(),
    };

    format!(
        r#"You are evaluating the narrative of a critical literature review.

## Narrative to evaluate

{narrative}
{feedback_block}
## Synthesis it should reflect ({n} claims)

Evaluate the narrative against the synthesis. Identify:

1. **Faithfulness** — Does it accurately represent the synthesis claims? Any distortions?
2. **Tension visibility** — Does it present contested claims with both sides? Are disputes visible or smoothed over?
3. **Lineage clarity** — Are the origins and evolution of key claims traceable from the prose?
4. **Support level honesty** — Are established findings stated plainly? Are emerging/single-source claims appropriately qualified?
5. **Anti-degeneration** — Does it avoid hedging, listing, formulaic phrases, or recency bias?

Provide specific, actionable feedback on each criterion. Quote passages that need improvement.

Do NOT score — provide textual critique only.
"#,
        narrative = narrative,
        n = synthesis_claims_count,
    )
}

/// Render v2/v3 Stage-3 narrative refinement prompt.
///
/// Re-weave block + anti-agreement guard (§4, §5).
///
/// Output shape (Decision 0 / Phase 0, constraint site 2 of 3): the refine
/// prompt must carry the SAME shape as the draft prompt — otherwise a
/// resurrected refine pass silently compresses a long-form draft back to
/// 500 words (kvasir S2b). `shape` must come from
/// `PromptProfile::narrative_shape()`.
pub fn render_narrative_refine_v2(
    narrative: &str,
    critique: &str,
    shape: NarrativeShape,
    fitness_feedback: Option<&str>,
) -> String {
    // C-N2: the candidate's low-scoring fitness dimensions, embedded only when
    // present. The default refine path passes `None` — the block collapses to
    // empty and the prompt is byte-identical to the pre-Phase-P template.
    let feedback_block = match fitness_feedback {
        Some(fb) => format!("\n## Fitness evaluation feedback\n\n{fb}\n"),
        None => String::new(),
    };
    // Concise arm byte-identical to the pre-Phase-0 template.
    let rewrite_spec = match shape {
        NarrativeShape::Concise => "Rewrite the narrative (300-500 words) incorporating the critique.",
        NarrativeShape::SectionedLongForm => {
            "Rewrite the review incorporating the critique, preserving the sectioned \
long-form shape: markdown ## section headings. No fixed word cap — develop each \
argument at the length the evidence demands. Flowing prose within sections — no \
bullet lists."
        }
    };

    format!(
        r#"You are refining the narrative of a critical literature review.

## Current narrative

{narrative}

## Critique to address

{critique}
{feedback_block}
{reweave}

{guards}

{rewrite_spec} \
If you believe a critique is wrong, argue back with evidence rather than complying.
"#,
        narrative = narrative,
        critique = critique,
        feedback_block = feedback_block,
        reweave = REWEAVE_BLOCK,
        guards = GUARDS_BLOCK,
    )
}

/// Render v2/v3 Stage-3 narrative final merge prompt.
///
/// Citation-preservation block kept, plus anti-formulaic guard.
///
/// Output shape (Decision 0 / Phase 0, constraint site 3 of 3): the merge
/// prompt must carry the SAME shape as draft and refine — a concise-shaped
/// merge silently compresses long-form trajectories at the last step
/// (kvasir S2b). `shape` must come from `PromptProfile::narrative_shape()`.
pub fn render_narrative_final_merge_v2(candidates: &[&str], shape: NarrativeShape) -> String {
    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| format!("### Candidate {n}\n\n{c}\n", n = i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    // Concise arm byte-identical to the pre-Phase-0 template.
    let closing = match shape {
        NarrativeShape::Concise => {
            "Write the merged narrative (300-500 words). No headers, no bullet lists — flowing prose only."
        }
        NarrativeShape::SectionedLongForm => {
            "Write the merged review as sectioned long-form: organise it under markdown \
## section headings, merging the candidates' sections by topic and keeping every \
distinct argument. No fixed word cap. Flowing prose within sections — no bullet lists."
        }
    };

    format!(
        r#"You are merging {n} narrative candidates into one authoritative narrative.

## Candidates

{candidates}

{reweave}

## Citation preservation

Inline [Cx] citation markers MUST be preserved. A citation marker that appears in ANY \
candidate for a given sentence must appear in the merged output.

## Anti-formulaic

Banned phrases: "it is important to note", "plays a crucial role", "in conclusion", \
"a growing body of literature", "further research is needed" (allowed only when naming a specific typed gap).

{closing}
"#,
        n = candidates.len(),
        candidates = candidates_text,
        reweave = REWEAVE_BLOCK,
    )
}

// ── Rubric-encoding Phase 1: plan-aware Stage-3 renderers (W-e714abb4) ────────
//
// Byte-stability discipline: every `_planned` renderer COMPOSES the
// corresponding base renderer above and APPENDS plan context. The base
// templates are never edited, so all plan-absent paths (v1 / v2 / v3 with
// `PlanMode::Disabled`) remain byte-identical to pre-Phase-1 output — the
// planned prompt is provably `base + suffix` (pinned by the prefix tests).

/// Render the four enforceable merge rules (spec §8, kvasir C-N3).
///
/// Each rule is paired with its enforcement channel:
/// - Rule 1 (verdict fusion) — judged by tension_visibility + plan_conformance.
/// - Rule 2 (term registry) — mechanically scanned post-merge (`term_drift_scan`).
/// - Rule 3 (banned phrases) — mechanically scanned post-merge (`banned_phrase_scan`).
/// - Rule 4 (citation preservation) — enforced by `sanitise_cx_citations`.
pub fn render_merge_rules_block(plan: &ReviewPlan) -> String {
    let registry = if plan.term_registry.is_empty() {
        "   (no registry terms declared)".to_string()
    } else {
        plan.term_registry
            .iter()
            .map(|e| {
                let banned = if e.banned_synonyms.is_empty() {
                    "(none)".to_string()
                } else {
                    e.banned_synonyms.join(", ")
                };
                format!("   - \"{}\" — never: {}", e.term, banned)
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let phrases = BANNED_PHRASES
        .iter()
        .map(|p| format!("\"{p}\""))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "## Merge rules (enforced — rules 2 and 3 are mechanically scanned after the merge, \
rule 4 by the citation sanitiser)\n\n\
1. **Verdict fusion — never average into a hedge.** When candidates reach OPPOSING \
verdicts on the same question, do NOT blend them into a hedge (\"evidence is mixed\", \
\"results remain unclear\"). Pick the verdict with the strongest support span — the most \
specific [Cx]-cited evidence across the candidates — state it plainly, and carry the \
opposing position as explicit, attributed dissent.\n\
2. **Term-registry conformance.** Use ONLY the canonical registry terms for registered \
concepts:\n{registry}\n\
3. **Banned phrases.** {phrases} (the last allowed only when naming a specific typed gap).\n\
4. **Citation preservation.** An inline [Cx] marker that appears in ANY candidate for a \
given claim must appear in the merged output. Never emit a [Cx] id that no candidate carries.\n"
    )
}

/// Plan-aware Stage-3 draft prompt: the base v2/v3 draft prompt with the
/// winning `ReviewPlan` appended as the declared development target.
pub fn render_narrative_draft_v2_planned(
    synthesis: &SynthesisArtifact,
    shape: NarrativeShape,
    plan: &ReviewPlan,
) -> String {
    format!(
        "{base}\n{plan_block}\n\
Develop the narrative under the plan above: answer its focal question, keep to its \
declared archetype, use its term registry verbatim, plant its threads with their \
markers, and — when a section skeleton is present — use exactly its ## section headings.\n",
        base = render_narrative_draft_v2(synthesis, shape),
        plan_block = render_plan_block(plan),
    )
}

/// Plan-aware Stage-3 refine prompt: the base v2/v3 refine prompt with the
/// winning plan re-asserted — critique never licenses plan departure.
pub fn render_narrative_refine_v2_planned(
    narrative: &str,
    critique: &str,
    shape: NarrativeShape,
    plan: &ReviewPlan,
    fitness_feedback: Option<&str>,
) -> String {
    format!(
        "{base}\n{plan_block}\n\
The rewrite must STAY under the plan above. The critique never licenses departing \
from the declared archetype, focal question, term registry, planted threads, or \
section skeleton — if a critique point conflicts with the plan, satisfy it WITHIN \
the plan's structure.\n",
        base = render_narrative_refine_v2(narrative, critique, shape, fitness_feedback),
        plan_block = render_plan_block(plan),
    )
}

/// Plan-aware whole-document final merge: the base merge prompt with the
/// winning plan and the four merge rules appended.
///
/// Used for `PlanMode != Disabled` runs whose shape is `Concise` (no section
/// skeleton) or whose candidates carry too few matching sections for the
/// section-by-section path (C-N3 fallback in `NarrativeMerger`).
pub fn render_narrative_final_merge_v2_planned(
    candidates: &[&str],
    shape: NarrativeShape,
    plan: &ReviewPlan,
) -> String {
    format!(
        "{base}\n{plan_block}\n{rules}\n\
Merge under the plan above: the merged document keeps the plan's section skeleton \
(when present), its term registry, and its planted threads.\n",
        base = render_narrative_final_merge_v2(candidates, shape),
        plan_block = render_plan_block(plan),
        rules = render_merge_rules_block(plan),
    )
}

/// Section-by-section merge prompt (spec §8, kvasir C-N3).
///
/// One LLM call merges the candidates' versions of ONE plan section. The
/// context contents are ENUMERATED in the prompt itself (C-N3 condition 1),
/// merging is sequential with the previous merged section's tail carried
/// forward (condition 2), and the vetoed-trajectory fan-in policy is stated
/// explicitly (condition 3): every trajectory's section enters fan-in with
/// its whole-document fitness rank annotated; structure anchors on rank 1.
///
/// `ranked_candidates` — `(rank, section_body)` pairs, rank 1 = the
/// fitness-best whole document (the order `sort_candidates_best_first`
/// produced; validity flags are not available at the `Merger` seam, so rank
/// is the carried signal — see `NarrativeMerger` docs).
/// `prev_tail` — final paragraph of the previously merged section, `None`
/// for the first section.
pub fn render_section_merge_v3(
    plan: &ReviewPlan,
    section: &PlanSection,
    ranked_candidates: &[(usize, &str)],
    prev_tail: Option<&str>,
) -> String {
    let registry_terms = if plan.term_registry.is_empty() {
        "(none declared)".to_string()
    } else {
        plan.term_registry
            .iter()
            .map(|e| e.term.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let budget = section
        .budget_words
        .map(|w| format!("{w} words"))
        .unwrap_or_else(|| "(no budget declared)".to_string());
    let claim_ids = if section.claim_ids.is_empty() {
        "(none assigned)".to_string()
    } else {
        section.claim_ids.join(", ")
    };
    let prev_block = match prev_tail {
        Some(tail) if !tail.trim().is_empty() => format!(
            "## Tail of the previously merged section (continuity context — do NOT repeat it)\n\n{tail}\n"
        ),
        _ => "## Tail of the previously merged section\n\n(This is the first section — no prior text.)\n"
            .to_string(),
    };
    let candidates_text = ranked_candidates
        .iter()
        .map(|(rank, body)| format!("### Candidate (whole-document fitness rank {rank})\n\n{body}\n"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are merging the candidates' versions of ONE section of a literature review.\n\n\
Your context contains, in order, and nothing else:\n\
1. The winning plan header — archetype, focal question, term registry.\n\
2. This section's plan entry — heading, purpose, word budget, assigned claim IDs.\n\
3. The tail of the previously merged section (continuity).\n\
4. The candidates' versions of THIS section, each annotated with its whole-document \
fitness rank (rank 1 = best).\n\
5. The merge rules.\n\n\
## Vetoed-trajectory policy (fan-in)\n\n\
Every trajectory's section enters this fan-in — including sections from candidates \
that ranked low or failed validity at whole-document selection; the rank annotation \
carries that signal. Anchor this section's STRUCTURE on the rank-1 candidate. \
Lower-ranked candidates contribute only specific arguments, evidence, and [Cx] \
citations that survive the merge rules — never let a low-rank candidate's framing \
override rank 1's.\n\n\
## Winning plan header\n\n\
**Archetype**: {archetype}\n\
**Focal question**: {focal}\n\
**Term registry**: {registry_terms}\n\n\
## This section's plan entry\n\n\
**Heading**: {heading}\n\
**Purpose**: {purpose}\n\
**Word budget**: {budget}\n\
**Assigned claim IDs**: {claim_ids}\n\n\
{prev_block}\n\
## Candidate versions of this section\n\n\
{candidates_text}\n\
{rules}\n\
Write the merged section below. Begin with the heading line `## {heading}` exactly, \
then flowing prose — no bullet lists. Develop only this section's assigned material, \
aim at its word budget, and open in continuity with the previous section's tail.\n",
        archetype = plan.archetype.as_str(),
        focal = plan.focal_question,
        heading = section.heading,
        purpose = section.purpose,
        rules = render_merge_rules_block(plan),
    )
}

// ── Helper: ArgumentationGraph XML serialisation ──────────────────────────────

impl ArgumentationGraph {
    /// Produce a compact XML string for use in prompt context blocks.
    pub fn to_xml_string(&self) -> String {
        let mut out = String::from("<graph>\n");
        for node in &self.nodes {
            // Quote carried through the round-trip (worklist item 2): without
            // it, quotes extracted in stage 1 died at the first denoise
            // reserialisation (probe 14: 0/44 claims quoted).
            let quote_el = match node.quote.as_deref() {
                Some(q) if !q.trim().is_empty() => format!("<quote>{q}</quote>"),
                _ => String::new(),
            };
            out.push_str(&format!(
                "  <node id=\"{id}\" type=\"claim\"><text>{claim}</text>{quote_el}<source>{expert}</source></node>\n",
                id = node.id,
                claim = node.claim,
                expert = node.expert_id,
            ));
        }
        for edge in &self.edges {
            out.push_str(&format!(
                "  <edge source=\"{src}\" target=\"{tgt}\" type=\"{kind}\"/>\n",
                src = edge.source,
                tgt = edge.target,
                kind = edge.relation,
            ));
        }
        out.push_str("</graph>");
        out
    }
}

// ── v2 fitness judge prompt renderers (B3) ────────────────────────────────────

/// Render a v2 fitness judge prompt for evaluating a `SynthesisArtifact`.
///
/// Follows MAS §9 single-model pattern:
/// - Role line naming the artifact kind and the dimension being evaluated
/// - Definition + 1/3/5 anchors quoted verbatim from `V2_JUDGE_DIMS` (§1)
/// - Score-with-evidence instruction ("quote the specific passages")
/// - Anti-agreement guard verbatim (MAS §9)
/// - Notable field invitation (MAS §6)
/// - Output schema: `fitness_evaluation` wrapping `score` (1-5) and `rationale`
/// - NO "## Upstream context" preamble (MAS Pitfall 6)
///
/// Input rendering: each claim is listed with text, sources, and present v2 fields
/// (support_level, evidence_grade, method, year, lineage). Judges for recency_balance
/// and lineage_clarity are not scored blind.
pub fn render_fitness_judge_v2_synthesis(dim: &JudgeDim, draft: &SynthesisArtifact) -> String {
    let mut out = String::new();

    // Role line — names the artifact kind and the specific dimension
    out.push_str(&format!(
        "You are evaluating one candidate literature-review synthesis on a single dimension: {}.\n\n",
        dim.name
    ));

    // Definition + anchors verbatim from V2_JUDGE_DIMS (§1)
    out.push_str("## Dimension\n\n");
    out.push_str(&format!("**{}**: {}\n\n", dim.name, dim.definition));
    out.push_str("**Score anchors**:\n");
    out.push_str(&format!("- 1: {}\n", dim.anchor_1));
    out.push_str(&format!("- 3: {}\n", dim.anchor_3));
    out.push_str(&format!("- 5: {}\n\n", dim.anchor_5));

    // Score-with-evidence instruction
    out.push_str("Your rationale must quote the specific passages that drove the score.\n\n");

    // Anti-agreement guard verbatim (MAS §9)
    out.push_str(
        "Score the output on its merits, not on how well it matches your expectations.\n\n"
    );

    // Notable instruction (MAS §6)
    out.push_str(
        "If any aspect surprises you, positively or negatively, \
         note it under **Notable** inside your rationale.\n\n"
    );

    // Candidate input — claims with text, sources, and v2 fields
    out.push_str("## Candidate synthesis\n\n");
    for (i, claim) in draft.claims.iter().enumerate() {
        out.push_str(&format!("### Claim {}\n\n", i + 1));
        out.push_str(&format!("**Text**: {}\n\n", claim.text));
        if claim.sources.is_empty() {
            out.push_str("**Sources**: (none)\n");
        } else {
            out.push_str(&format!("**Sources**: {}\n", claim.sources.join(", ")));
        }
        if let Some(ref sl) = claim.support_level {
            out.push_str(&format!("**support_level**: {}\n", sl));
        }
        if let Some(ref eg) = claim.evidence_grade {
            out.push_str(&format!("**evidence_grade**: {}\n", eg));
        }
        if let Some(ref m) = claim.method {
            out.push_str(&format!("**method**: {}\n", m));
        }
        if let Some(ref y) = claim.year {
            out.push_str(&format!("**year**: {}\n", y));
        }
        if let Some(ref lin) = claim.lineage {
            out.push_str(&format!("**lineage**: {}\n", lin));
        }
        out.push('\n');
    }

    // Output schema instruction
    out.push_str("## Output\n\n");
    out.push_str("Respond ONLY with the following XML:\n\n");
    out.push_str(
        "<fitness_evaluation>\n  <score>N</score>\n  <rationale>Your rationale, quoting \
         specific passages. Include Notable if relevant.</rationale>\n</fitness_evaluation>\n"
    );

    out
}

/// Render a v2 fitness judge prompt for evaluating an `ArgumentationGraph`.
///
/// Input rendering: each node is listed with id, claim, expert_id, and quote when present.
/// Structure otherwise identical to the synthesis judge (MAS §9 single-model pattern).
pub fn render_fitness_judge_v2_graph(dim: &JudgeDim, graph: &ArgumentationGraph) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "You are evaluating one candidate literature-review argumentation graph on a single dimension: {}.\n\n",
        dim.name
    ));

    out.push_str("## Dimension\n\n");
    out.push_str(&format!("**{}**: {}\n\n", dim.name, dim.definition));
    out.push_str("**Score anchors**:\n");
    out.push_str(&format!("- 1: {}\n", dim.anchor_1));
    out.push_str(&format!("- 3: {}\n", dim.anchor_3));
    out.push_str(&format!("- 5: {}\n\n", dim.anchor_5));

    out.push_str("Your rationale must quote the specific passages that drove the score.\n\n");

    out.push_str(
        "Score the output on its merits, not on how well it matches your expectations.\n\n"
    );

    out.push_str(
        "If any aspect surprises you, positively or negatively, \
         note it under **Notable** inside your rationale.\n\n"
    );

    out.push_str("## Candidate argumentation graph\n\n");
    for node in &graph.nodes {
        out.push_str(&format!("- **{}** (source: {})\n", node.id, node.expert_id));
        out.push_str(&format!("  Claim: {}\n", node.claim));
        if let Some(ref q) = node.quote {
            if !q.trim().is_empty() {
                out.push_str(&format!("  Quote: \"{}\"\n", q.trim()));
            }
        }
        out.push('\n');
    }

    out.push_str("## Output\n\n");
    out.push_str("Respond ONLY with the following XML:\n\n");
    out.push_str(
        "<fitness_evaluation>\n  <score>N</score>\n  <rationale>Your rationale, quoting \
         specific passages. Include Notable if relevant.</rationale>\n</fitness_evaluation>\n"
    );

    out
}

/// Render a v2 fitness judge prompt for evaluating a narrative (Stage 3).
///
/// Narrative judges evaluate the narrative text directly — no pseudo-artifact wrapping.
/// Structure identical to synthesis/graph judges (MAS §9 single-model pattern).
pub fn render_fitness_judge_v2_narrative(dim: &JudgeDim, narrative: &str) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "You are evaluating one candidate literature-review narrative on a single dimension: {}.\n\n",
        dim.name
    ));

    out.push_str("## Dimension\n\n");
    out.push_str(&format!("**{}**: {}\n\n", dim.name, dim.definition));
    out.push_str("**Score anchors**:\n");
    out.push_str(&format!("- 1: {}\n", dim.anchor_1));
    out.push_str(&format!("- 3: {}\n", dim.anchor_3));
    out.push_str(&format!("- 5: {}\n\n", dim.anchor_5));

    out.push_str("Your rationale must quote the specific passages that drove the score.\n\n");

    out.push_str(
        "Score the output on its merits, not on how well it matches your expectations.\n\n"
    );

    out.push_str(
        "If any aspect surprises you, positively or negatively, \
         note it under **Notable** inside your rationale.\n\n"
    );

    out.push_str("## Candidate narrative\n\n");
    out.push_str(narrative);
    out.push_str("\n\n");

    out.push_str("## Output\n\n");
    out.push_str("Respond ONLY with the following XML:\n\n");
    out.push_str(
        "<fitness_evaluation>\n  <score>N</score>\n  <rationale>Your rationale, quoting \
         specific passages. Include Notable if relevant.</rationale>\n</fitness_evaluation>\n"
    );

    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{ResponseProvenance, SourceId};
    use crate::ttd::term_sheet::{EVIDENCE_GRADES, GAP_TYPES, SUPPORT_LEVELS};

    fn make_response(id: &str, title: &str, year: Option<i32>, authors: &[&str], prose: &str) -> ExpertResponse {
        ExpertResponse {
            expert_id: SourceId::new(id.to_string()),
            prose: prose.to_string(),
            provenance: ResponseProvenance {
                source_id: SourceId::new(id.to_string()),
                title: title.to_string(),
                year,
                authors: authors.iter().map(|s| s.to_string()).collect(),
                credibility_tier: alzina_search::CredibilityTier::Unknown,
            },
        }
    }

    // Test: V2_PERSONAS has exactly 5 entries ordered methodologist/theorist/empiricist/
    // field-historian/sceptic; each contains ≥2 CANNOT occurrences, all 5 §2 profile
    // fields, and the §12 escape clause "flag the gap".
    #[test]
    fn v2_personas_count_order_and_structure() {
        use crate::ttd::personas::V2_PERSONAS;

        assert_eq!(V2_PERSONAS.len(), 5, "must have exactly 5 v2 personas");

        let expected_names = ["methodologist", "theorist", "empiricist", "field-historian", "sceptic"];
        for (i, persona) in V2_PERSONAS.iter().enumerate() {
            let lower = persona.to_lowercase();
            // Each persona contains its own name (domain label)
            assert!(
                lower.contains(expected_names[i]),
                "persona {} must contain '{}'; got: {}",
                i, expected_names[i], &persona[..100.min(persona.len())]
            );
            // §2 profile fields: expertise, analytical_stance, prioritises, deprioritises, blind_spot
            assert!(lower.contains("expertise"), "persona {} missing <expertise>", i);
            assert!(lower.contains("analytical_stance"), "persona {} missing <analytical_stance>", i);
            assert!(lower.contains("prioritises"), "persona {} missing <prioritises>", i);
            assert!(lower.contains("deprioritises"), "persona {} missing <deprioritises>", i);
            assert!(lower.contains("blind_spot"), "persona {} missing <blind_spot>", i);
            // §12 escape clause
            assert!(lower.contains("flag the gap"), "persona {} missing §12 escape clause 'flag the gap'", i);
            // ≥2 "CANNOT" (case-insensitive)
            let cannot_count = persona.to_uppercase().matches("CANNOT").count();
            assert!(cannot_count >= 2, "persona {} has only {cannot_count} CANNOT constraints (need ≥2)", i);
        }
    }

    // Test: §12 separability — each persona's constraint block names at least one
    // other persona's domain.
    #[test]
    fn v2_personas_constraints_name_other_domains() {
        use crate::ttd::personas::V2_PERSONAS;

        let domains = ["methodologist", "theorist", "empiricist", "field-historian", "sceptic"];

        for (i, persona) in V2_PERSONAS.iter().enumerate() {
            let lower = persona.to_lowercase();
            let other_domains_mentioned = domains
                .iter()
                .enumerate()
                .filter(|(j, domain)| *j != i && lower.contains(*domain))
                .count();
            assert!(
                other_domains_mentioned >= 1,
                "persona {} ({}) constraint block must name at least one adjacent persona's domain",
                i, domains[i]
            );
        }
    }

    // Test: EVIDENCE_GRADES has 4 entries (strong/moderate/weak/anecdotal), non-empty definitions.
    #[test]
    fn evidence_grades_count_and_definitions() {
        assert_eq!(EVIDENCE_GRADES.len(), 4, "must have exactly 4 evidence grades");
        let expected = ["strong", "moderate", "weak", "anecdotal"];
        for (i, grade) in EVIDENCE_GRADES.iter().enumerate() {
            assert_eq!(grade.name, expected[i], "grade {} name mismatch", i);
            assert!(!grade.definition.is_empty(), "grade {} definition must be non-empty", i);
        }
    }

    // Test: render_vocabulary_block() output contains every SUPPORT_LEVELS,
    // EVIDENCE_GRADES, and GAP_TYPES definition string verbatim.
    #[test]
    fn vocabulary_block_contains_all_definitions_verbatim() {
        let block = render_vocabulary_block();

        for t in &SUPPORT_LEVELS {
            assert!(
                block.contains(t.definition),
                "vocabulary block missing SUPPORT_LEVELS definition for '{}'",
                t.name
            );
        }
        for t in &EVIDENCE_GRADES {
            assert!(
                block.contains(t.definition),
                "vocabulary block missing EVIDENCE_GRADES definition for '{}'",
                t.name
            );
        }
        for t in &GAP_TYPES {
            assert!(
                block.contains(t.definition),
                "vocabulary block missing GAP_TYPES definition for '{}'",
                t.name
            );
        }
    }

    // Test: GUARDS_BLOCK contains the five guard names verbatim and the
    // banned-phrase list.
    #[test]
    fn guards_block_contains_five_guard_names_and_banned_phrases() {
        let guard_names = [
            "anti-vote-counting",
            "anti-hedge",
            "anti-list",
            "anti-formulaic",
            "anti-recency",
        ];
        for name in &guard_names {
            assert!(
                GUARDS_BLOCK.contains(name),
                "GUARDS_BLOCK missing guard name '{}'",
                name
            );
        }

        // Banned phrases (anti-formulaic)
        let banned = [
            "it is important to note",
            "plays a crucial role",
            "in conclusion",
            "a growing body of literature",
            "further research is needed",
        ];
        for phrase in &banned {
            assert!(
                GUARDS_BLOCK.contains(phrase),
                "GUARDS_BLOCK missing banned phrase '{}'",
                phrase
            );
        }
    }

    // Test: render_paper_header() contains title, year, authors, paper_id,
    // and a <paper_text id=…> wrapper; contains NO "expert_response" substring;
    // year None renders "year unknown".
    #[test]
    fn render_paper_header_contains_provenance_and_no_expert_response() {
        let resp = make_response(
            "arxiv:2304.07620",
            "Permafrost thaw and methane",
            Some(2023),
            &["Smith, A.", "Jones, B."],
            "Paper body text here.",
        );

        let header = render_paper_header(&resp);

        assert!(header.contains("arxiv:2304.07620"), "missing paper_id");
        assert!(header.contains("Permafrost thaw and methane"), "missing title");
        assert!(header.contains("2023"), "missing year");
        assert!(header.contains("Smith, A."), "missing author");
        assert!(header.contains("<paper_text id=\"arxiv:2304.07620\">"), "missing <paper_text> wrapper");
        assert!(!header.contains("expert_response"), "must NOT contain 'expert_response'");
    }

    #[test]
    fn render_paper_header_year_unknown_when_none() {
        let resp = make_response("arxiv:test", "Title", None, &[], "prose");
        let header = render_paper_header(&resp);
        assert!(header.contains("year unknown"), "year None must render 'year unknown'");
    }

    // Test: existing term_sheet/personas tests still pass unchanged (additive-only proof).
    // This is verified by the cargo test run as a whole — no explicit test here,
    // but the behaviour test above proves V2_PERSONAS is additive (PERSONAS untouched).
    #[test]
    fn existing_personas_const_untouched() {
        use crate::ttd::personas::PERSONAS;
        // PERSONAS must still have 5 entries with the v1 content.
        assert_eq!(PERSONAS.len(), 5, "PERSONAS must still have 5 entries");
        // v1 personas contain "methodological" framing, not the v2 XML profile tags.
        assert!(
            PERSONAS[0].contains("methodologically rigorous"),
            "PERSONAS[0] must retain v1 text"
        );
    }

    // ── Task 2 tests ──────────────────────────────────────────────────────────

    fn sample_graph() -> crate::ttd::artifact::ArgumentationGraph {
        crate::ttd::artifact::ArgumentationGraph::new("s", "r", "q", "test-model", "v2/lit-review")
    }

    fn sample_responses() -> Vec<ExpertResponse> {
        vec![
            make_response("arxiv:2304.07620", "Methane from permafrost", Some(2023), &["Smith A"], "Paper body 1"),
            make_response("arxiv:2308.06046", "Arctic amplification", Some(2022), &["Jones B"], "Paper body 2"),
        ]
    }

    // Round-trip test (the load-bearing one): a sample model output following the
    // v2 schema parses under V2LitReview with v2 fields populated, agreement_level None.
    #[test]
    fn round_trip_v2_synthesis_schema() {
        use crate::ttd::stages::synthesis::parse_synthesis_xml;
        use crate::ttd::term_sheet::PromptProfile;

        // Hand-written sample output following render_synthesis_draft_graph_v2 schema exactly.
        // Two claims: one contested with counterargument+lineage, one single-source.
        // Self-closing sources. One <gap type="epistemic"> with inline body.
        let sample_output = r#"<synthesis>
  <narrative>Permafrost thaw releases methane [C1]. Single-source evidence for rapid acceleration [C2].</narrative>
  <claims>
    <claim id="C1">
      <text>Permafrost thaw is a net source of methane to the atmosphere</text>
      <support_level>contested</support_level>
      <evidence_grade>moderate</evidence_grade>
      <method>isotopic analysis and flux measurements</method>
      <year>2023</year>
      <lineage>first shown by arxiv:2304.07620 (2023); challenged by arxiv:2308.06046 (2022)</lineage>
      <sources><source id="arxiv:2304.07620"/><source id="arxiv:2308.06046"/></sources>
      <counterarguments><counterargument>Jones B (arxiv:2308.06046) argues oxidation offsets emissions</counterargument></counterarguments>
    </claim>
    <claim id="C2">
      <text>Methane release accelerated 40% post-2010</text>
      <support_level>single-source</support_level>
      <evidence_grade>weak</evidence_grade>
      <method>atmospheric inversion model</method>
      <year>2023</year>
      <lineage>first proposed by arxiv:2304.07620 (2023); no independent replication yet</lineage>
      <sources><source id="arxiv:2304.07620"/></sources>
    </claim>
  </claims>
  <areas_of_agreement><area>Permafrost thaw is occurring</area></areas_of_agreement>
  <areas_of_disagreement><area>Net emission sign</area></areas_of_disagreement>
  <uncertainties><uncertainty>Long-term oxidation dynamics</uncertainty></uncertainties>
  <gaps>
    <gap type="epistemic">Whether the field can bound total emission potential without resolving oxidation vs emission rates</gap>
  </gaps>
</synthesis>"#;

        let result = parse_synthesis_xml(sample_output, "test-model", "v2/lit-review", PromptProfile::V2LitReview)
            .expect("parse must succeed");

        assert_eq!(result.claims.len(), 2, "must parse 2 claims");

        let c1 = &result.claims[0];
        assert_eq!(c1.support_level.as_deref(), Some("contested"), "C1 support_level must be 'contested'");
        assert_eq!(c1.evidence_grade.as_deref(), Some("moderate"), "C1 evidence_grade must be 'moderate'");
        assert!(c1.lineage.is_some(), "C1 lineage must be populated");
        assert!(c1.method.is_some(), "C1 method must be populated");
        assert!(c1.year.is_some(), "C1 year must be populated");
        assert_eq!(c1.agreement_level, None, "V2LitReview must NOT fabricate agreement_level");
        assert!(!c1.sources.is_empty(), "C1 must have sources");

        let c2 = &result.claims[1];
        assert_eq!(c2.support_level.as_deref(), Some("single-source"), "C2 support_level must be 'single-source'");
        assert_eq!(c2.agreement_level, None, "V2LitReview must NOT fabricate agreement_level for C2");

        // Gap: one entry with type "epistemic"
        assert_eq!(result.gaps.len(), 1, "must parse 1 gap");
        assert_eq!(result.gaps[0].gap_type.as_deref(), Some("epistemic"), "gap_type must be 'epistemic'");
        assert!(!result.gaps[0].description.is_empty(), "gap description must be non-empty");
    }

    // Test: every v2 generation/refinement prompt contains the five guard names
    // and zero occurrences of "expert_response", "Delphi", consensus vocab,
    // and "agreement_level".
    #[test]
    fn v2_generation_prompts_contain_guards_no_v1_dressing() {
        let inputs = sample_responses();
        let graph = sample_graph();

        let empty_synth = crate::ttd::artifact::SynthesisArtifact::new("", "", "", "test", "v2");
        let prompts = vec![
            render_synthesis_draft_graph_v2("test question", &graph, &inputs, "400"),
            render_synthesis_draft_v2("test question", &inputs, "400"),
            render_synthesis_gap_resolve_patch_v2("<synthesis/>", "gap desc", &inputs, "400"),
            render_synthesis_gap_resolve_full_v2("<synthesis/>", "gap desc", &inputs, "400"),
            render_narrative_draft_v2(&empty_synth, NarrativeShape::Concise),
        ];

        let guard_names = ["anti-vote-counting", "anti-hedge", "anti-list", "anti-formulaic", "anti-recency"];
        let v1_dressing = ["expert_response", "Delphi", "agreement_level",
                           "consensus|majority|divided|minority"];

        for (i, prompt) in prompts.iter().enumerate() {
            for guard in &guard_names {
                assert!(
                    prompt.contains(guard),
                    "prompt {} missing guard '{}'", i, guard
                );
            }
            for forbidden in &["expert_response", "Delphi", "agreement_level"] {
                assert!(
                    !prompt.contains(forbidden),
                    "prompt {} must not contain '{}'", i, forbidden
                );
            }
        }
        let _ = v1_dressing; // used above
    }

    // ── Decision 0 / Phase 0: narrative shape tests (3 constraint sites) ──────

    /// Concise shape (v2 default path) pins the pre-Phase-0 constraint text at
    /// all three sites — falsifies any accidental drift of the v2 output shape.
    #[test]
    fn narrative_prompts_concise_shape_pins_v2_constraint_at_all_three_sites() {
        let empty_synth = crate::ttd::artifact::SynthesisArtifact::new("", "", "", "test", "v2");
        let prompts = [
            ("draft", render_narrative_draft_v2(&empty_synth, NarrativeShape::Concise)),
            ("refine", render_narrative_refine_v2("text", "critique", NarrativeShape::Concise, None)),
            ("merge", render_narrative_final_merge_v2(&["a", "b"], NarrativeShape::Concise)),
        ];
        for (site, prompt) in &prompts {
            assert!(
                prompt.contains("300-500 word"),
                "{site}: concise shape must keep the 300-500-word constraint"
            );
            assert!(
                !prompt.contains("No fixed word cap"),
                "{site}: concise shape must not carry long-form text"
            );
            assert!(
                !prompt.contains("## section headings"),
                "{site}: concise shape must not invite section headings"
            );
        }
        // The headerless instruction survives verbatim at draft and merge.
        assert!(prompts[0].1.contains("No headers, no bullet lists — flowing prose only."));
        assert!(prompts[2].1.contains("No headers, no bullet lists — flowing prose only."));
    }

    /// SectionedLongForm lifts the constraint at ALL three sites together —
    /// a single un-lifted site silently re-compresses long-form output
    /// (kvasir S2b). Falsified if any site still carries "300-500".
    #[test]
    fn narrative_prompts_long_form_shape_lifts_constraint_at_all_three_sites() {
        let empty_synth = crate::ttd::artifact::SynthesisArtifact::new("", "", "", "test", "v2");
        let prompts = [
            ("draft", render_narrative_draft_v2(&empty_synth, NarrativeShape::SectionedLongForm)),
            ("refine", render_narrative_refine_v2("text", "critique", NarrativeShape::SectionedLongForm, None)),
            ("merge", render_narrative_final_merge_v2(&["a", "b"], NarrativeShape::SectionedLongForm)),
        ];
        for (site, prompt) in &prompts {
            assert!(
                !prompt.contains("300-500"),
                "{site}: long-form shape must not carry the 300-500-word constraint"
            );
            assert!(
                prompt.contains("## section headings"),
                "{site}: long-form shape must invite markdown section headings"
            );
            assert!(
                prompt.contains("No fixed word cap"),
                "{site}: long-form shape must lift the word cap explicitly"
            );
            assert!(
                prompt.contains("no bullet lists"),
                "{site}: anti-list degeneration guard must survive the lift"
            );
        }
        // Guards and citation discipline survive the lift (D0-indep content rules).
        assert!(prompts[0].1.contains("anti-hedge"), "draft guards block must survive");
        assert!(
            prompts[2].1.contains("Citation preservation"),
            "merge citation-preservation block must survive"
        );
    }

    /// Phase P / C-N2: the v2 refine prompt embeds fitness feedback only when
    /// present. `None` (the default refine path) leaves the prompt byte-stable;
    /// `Some` carries the feedback section verbatim so the rewrite targets the
    /// low-scoring dimensions.
    #[test]
    fn narrative_refine_v2_embeds_fitness_feedback_only_when_some() {
        let none = render_narrative_refine_v2("draft", "crit", NarrativeShape::Concise, None);
        assert!(
            !none.contains("Fitness evaluation feedback"),
            "None feedback must leave the refine prompt byte-stable"
        );

        let some = render_narrative_refine_v2(
            "draft",
            "crit",
            NarrativeShape::Concise,
            Some("## Priority Improvements\n- faithfulness: low"),
        );
        assert!(
            some.contains("## Fitness evaluation feedback"),
            "Some feedback must embed the feedback section"
        );
        assert!(
            some.contains("faithfulness: low"),
            "feedback text must appear verbatim in the refine prompt"
        );
    }

    // Test: render_synthesis_draft_graph_v2 contains SUPPORT_LEVELS definitions
    // verbatim, the conflict-path block, and the input quality gate.
    #[test]
    fn synthesis_draft_graph_contains_schema_components() {
        let inputs = sample_responses();
        let graph = sample_graph();
        let prompt = render_synthesis_draft_graph_v2("test question", &graph, &inputs, "400");

        // SUPPORT_LEVELS definitions verbatim
        for t in &SUPPORT_LEVELS {
            assert!(prompt.contains(t.definition), "missing SUPPORT_LEVELS def for '{}'", t.name);
        }
        // Conflict-path block marker — uses "NOT" prohibition framing (D-6)
        assert!(prompt.contains("do NOT vote"), "conflict path must contain 'do NOT vote'");
        assert!(prompt.contains("contested"), "conflict path must reference 'contested'");
        // Quality gate marker
        assert!(prompt.contains("Before synthesising"), "missing quality gate");
    }

    // Test: render_gap_identify_v2 still instructs <description>/<query> format.
    #[test]
    fn gap_identify_v2_instructs_retrieval_format() {
        let graph_prompt = render_gap_identify_v2("<graph/>", "test question");
        let synth_prompt = render_synthesis_gap_identify_v2("<synthesis/>", "test question", None);

        assert!(graph_prompt.contains("<description>"), "graph gap_identify_v2 missing <description>");
        assert!(graph_prompt.contains("<query>"), "graph gap_identify_v2 missing <query>");
        assert!(synth_prompt.contains("<description>"), "synthesis gap_identify_v2 missing <description>");
        assert!(synth_prompt.contains("<query>"), "synthesis gap_identify_v2 missing <query>");
    }

    // Test: render_revision_v2 contains re-weave block, missing papers titles/years,
    // full v2 schema, and no "corroboration_count"/"minority_reports".
    #[test]
    fn revision_v2_contains_reweave_and_no_v1_ballot_artifacts() {
        use crate::ttd::artifact::SynthesisArtifact;
        let synthesis = SynthesisArtifact::new("", "", "", "test", "v2");
        let panel = sample_responses();
        let missing = vec!["arxiv:2304.07620".to_string()];

        let prompt = render_revision_v2(&synthesis, &panel, &missing, "400", "");

        // Re-weave block present
        assert!(prompt.contains("rewriting, not appending"), "missing re-weave instruction");
        // Missing paper title/year present
        assert!(prompt.contains("Methane from permafrost"), "missing missing paper title");
        assert!(prompt.contains("2023"), "missing missing paper year");
        // Full v2 schema present (support_level)
        assert!(prompt.contains("support_level"), "missing v2 schema in revision");
        // No v1 ballot artifacts
        assert!(!prompt.contains("corroboration_count"), "must not contain 'corroboration_count'");
        assert!(!prompt.contains("minority_reports"), "must not contain 'minority_reports'");
    }

    /// Point 3 (Stage-2 soft-filter): the revision tags both the missing-paper
    /// headers and the existing claim sources with each source's credibility
    /// tier, derived from panel provenance. `Unknown` stays untagged.
    #[test]
    fn revision_v2_tags_missing_papers_and_claim_sources_with_tier() {
        use crate::ttd::artifact::{Claim, SynthesisArtifact};
        use alzina_search::CredibilityTier;

        // Panel with one rated paper.
        let mut paper =
            make_response("arxiv:9999", "Title X", Some(2024), &["A. Author"], "## Method\nbody");
        paper.provenance.credibility_tier = CredibilityTier::Low;
        let panel = vec![paper];

        // Synthesis with one claim citing that source.
        let mut synthesis = SynthesisArtifact::new("", "", "", "test", "v2");
        synthesis.claims.push(Claim {
            text: "A claim.".into(),
            agreement_level: None,
            sources: vec!["arxiv:9999".into()],
            counterarguments: vec![],
            support_level: Some("converging".into()),
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        let missing = vec!["arxiv:9999".to_string()];

        let prompt = render_revision_v2(&synthesis, &panel, &missing, "400", "");

        assert!(
            prompt.contains("- **arxiv:9999** [low credibility]:"),
            "missing-paper header must carry its tier tag"
        );
        assert!(
            prompt.contains("[sources: arxiv:9999 [low credibility]]"),
            "claim source must carry its tier tag"
        );
    }

    /// Optional Stage-2 point: the provenance-only `## Sources` header (first
    /// Stage-2 prompt the draft reads) tags each paper with its tier; `Unknown`
    /// stays untagged.
    #[test]
    fn paper_headers_section_tags_sources_with_credibility_tier() {
        use alzina_search::CredibilityTier;

        let mut rated =
            make_response("arxiv:7777", "Rated Paper", Some(2024), &["B. Writer"], "body");
        rated.provenance.credibility_tier = CredibilityTier::High;
        let unrated =
            make_response("arxiv:0000", "Unrated Paper", Some(2020), &["C. Writer"], "body");

        let out = render_paper_headers_section(&[rated, unrated]);
        assert!(
            out.contains("- arxiv:7777 [high credibility]: Rated Paper"),
            "rated source header tagged: {out}"
        );
        assert!(
            out.contains("- arxiv:0000: Unrated Paper"),
            "Unknown source header untagged: {out}"
        );
    }

    // Test: gap_resolve full and merger instruct v2 claim fields — field survival guarantee.
    #[test]
    fn gap_resolve_full_and_merger_instruct_v2_fields() {
        use crate::ttd::artifact::SynthesisArtifact;
        let inputs = sample_responses();
        let resolve_prompt = render_synthesis_gap_resolve_full_v2(
            "<synthesis/>", "gap desc", &inputs, "400"
        );
        assert!(resolve_prompt.contains("support_level"), "gap_resolve_full missing support_level");
        assert!(resolve_prompt.contains("lineage"), "gap_resolve_full missing lineage");

        let synth = SynthesisArtifact::new("", "", "", "test", "v2");
        let merger_prompt = render_synthesis_merger_v2(
            &[&synth],
            "test question",
            "400",
            "",
            &std::collections::BTreeMap::new(),
        );
        assert!(merger_prompt.contains("support_level"), "merger_v2 missing support_level");
        assert!(merger_prompt.contains("evidence_grade"), "merger_v2 missing evidence_grade");
    }

    // ── v2 judge prompt tests (B3) ────────────────────────────────────────────

    /// Helper: minimal SynthesisArtifact with two sourced claims (one with all v2 fields).
    fn fixture_synthesis_for_judges() -> SynthesisArtifact {
        use crate::ttd::artifact::Claim;
        let mut a = SynthesisArtifact::new("study", "r1", "q1", "model", "v2/lit-review");
        a.claims.push(Claim {
            text: "Permafrost thaw accelerates methane release.".into(),
            agreement_level: None,
            sources: vec!["arxiv:2304.07620".into()],
            counterarguments: vec![],
            support_level: Some("converging".into()),
            evidence_grade: Some("moderate".into()),
            method: Some("observational".into()),
            year: Some("2023".into()),
            lineage: Some("Follows from Schaefer 2014 permafrost models.".into()),
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        a.claims.push(Claim {
            text: "Arctic amplification is contested.".into(),
            agreement_level: None,
            sources: vec!["s2:abc123".into(), "arxiv:2308.06046".into()],
            counterarguments: vec![],
            support_level: Some("contested".into()),
            evidence_grade: Some("weak".into()),
            method: None,
            year: Some("2022".into()),
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        a
    }

    fn fixture_graph_for_judges() -> ArgumentationGraph {
        use crate::ttd::artifact::GraphNode;
        let mut g = ArgumentationGraph::new("study", "r1", "q1", "model", "v2/graph");
        g.nodes.push(GraphNode {
            id: "arxiv:2304.07620_c1".into(),
            claim: "Permafrost thaw accelerates methane release.".into(),
            expert_id: "arxiv:2304.07620".into(),
            quote: Some("thaw rates increasing in Siberia 2014-2023".into()),
            verification_status: Some("verified".into()),
        });
        g.nodes.push(GraphNode {
            id: "s2:abc_c1".into(),
            claim: "Methane concentrations are rising.".into(),
            expert_id: "s2:abc".into(),
            quote: None,
            verification_status: None,
        });
        g
    }

    /// All 5 v2 synthesis judge prompts contain dim name, definition, all three anchors,
    /// anti-agreement guard, Notable instruction, and required output schema.
    #[test]
    fn synthesis_judge_prompts_contain_all_required_elements() {
        let synthesis = fixture_synthesis_for_judges();
        for dim in &V2_JUDGE_DIMS {
            let prompt = render_fitness_judge_v2_synthesis(dim, &synthesis);

            // Dim name and definition
            assert!(prompt.contains(dim.name), "synthesis judge must contain dim name '{}'", dim.name);
            assert!(prompt.contains(dim.definition), "synthesis judge must contain definition for '{}'", dim.name);

            // All three anchors
            assert!(prompt.contains(dim.anchor_1), "synthesis judge must contain anchor_1 for '{}'", dim.name);
            assert!(prompt.contains(dim.anchor_3), "synthesis judge must contain anchor_3 for '{}'", dim.name);
            assert!(prompt.contains(dim.anchor_5), "synthesis judge must contain anchor_5 for '{}'", dim.name);

            // Anti-agreement guard (MAS §9)
            assert!(
                prompt.contains("Score the output on its merits, not on how well it matches your expectations."),
                "synthesis judge for '{}' must contain anti-agreement guard", dim.name
            );

            // Notable instruction (MAS §6)
            assert!(
                prompt.contains("Notable"),
                "synthesis judge for '{}' must contain Notable instruction", dim.name
            );

            // Output schema (fitness_evaluation/score/rationale)
            assert!(
                prompt.contains("fitness_evaluation") && prompt.contains("score") && prompt.contains("rationale"),
                "synthesis judge for '{}' must instruct fitness_evaluation/score/rationale schema", dim.name
            );

            // No "## Upstream context" preamble (MAS Pitfall 6)
            assert!(
                !prompt.contains("## Upstream context"),
                "synthesis judge for '{}' must not contain '## Upstream context'", dim.name
            );
        }
    }

    /// Synthesis judge input rendering shows per-claim: text, sources, v2 fields.
    #[test]
    fn synthesis_judge_renders_claim_provenance() {
        let synthesis = fixture_synthesis_for_judges();
        // Use faithfulness dim (index 0)
        let prompt = render_fitness_judge_v2_synthesis(&V2_JUDGE_DIMS[0], &synthesis);

        // Claim text present
        assert!(prompt.contains("Permafrost thaw accelerates methane release."), "claim text must appear");
        // Sources present
        assert!(prompt.contains("arxiv:2304.07620"), "source must appear in synthesis judge input");
        // v2 fields present
        assert!(prompt.contains("converging") || prompt.contains("support_level"), "support_level or value must appear");
        assert!(prompt.contains("moderate") || prompt.contains("evidence_grade"), "evidence_grade or value must appear");
        assert!(prompt.contains("2023") || prompt.contains("year"), "year must appear");
    }

    /// All 5 v2 graph judge prompts contain the required elements.
    #[test]
    fn graph_judge_prompts_contain_all_required_elements() {
        let graph = fixture_graph_for_judges();
        for dim in &V2_JUDGE_DIMS {
            let prompt = render_fitness_judge_v2_graph(dim, &graph);

            assert!(prompt.contains(dim.name), "graph judge must contain dim name '{}'", dim.name);
            assert!(prompt.contains(dim.definition), "graph judge must contain definition for '{}'", dim.name);
            assert!(prompt.contains(dim.anchor_1), "graph judge must contain anchor_1 for '{}'", dim.name);
            assert!(prompt.contains(dim.anchor_3), "graph judge must contain anchor_3 for '{}'", dim.name);
            assert!(prompt.contains(dim.anchor_5), "graph judge must contain anchor_5 for '{}'", dim.name);
            assert!(
                prompt.contains("Score the output on its merits, not on how well it matches your expectations."),
                "graph judge for '{}' must contain anti-agreement guard", dim.name
            );
            assert!(prompt.contains("Notable"), "graph judge for '{}' must contain Notable", dim.name);
            assert!(
                prompt.contains("fitness_evaluation") && prompt.contains("score"),
                "graph judge for '{}' must contain output schema", dim.name
            );
            assert!(
                !prompt.contains("## Upstream context"),
                "graph judge for '{}' must not contain '## Upstream context'", dim.name
            );
        }
    }

    /// Graph judge input rendering shows per-node: id, claim, expert_id, quote when present.
    #[test]
    fn graph_judge_renders_node_provenance() {
        let graph = fixture_graph_for_judges();
        let prompt = render_fitness_judge_v2_graph(&V2_JUDGE_DIMS[0], &graph);

        assert!(prompt.contains("arxiv:2304.07620_c1") || prompt.contains("arxiv:2304.07620"), "node id/expert must appear");
        assert!(prompt.contains("Permafrost thaw accelerates methane release."), "node claim must appear");
        assert!(prompt.contains("thaw rates increasing in Siberia"), "quote must appear when present");
    }

    /// All 5 v2 narrative judge prompts contain the required elements.
    #[test]
    fn narrative_judge_prompts_contain_all_required_elements() {
        let narrative = "The permafrost system is releasing methane at increasing rates \
                         according to multiple independent lines of work (arxiv:2304.07620, s2:abc123). \
                         Contested mechanisms remain an active area of research.";
        for dim in &V2_JUDGE_DIMS {
            let prompt = render_fitness_judge_v2_narrative(dim, narrative);

            assert!(prompt.contains(dim.name), "narrative judge must contain dim name '{}'", dim.name);
            assert!(prompt.contains(dim.definition), "narrative judge must contain definition for '{}'", dim.name);
            assert!(prompt.contains(dim.anchor_1), "narrative judge must contain anchor_1 for '{}'", dim.name);
            assert!(prompt.contains(dim.anchor_3), "narrative judge must contain anchor_3 for '{}'", dim.name);
            assert!(prompt.contains(dim.anchor_5), "narrative judge must contain anchor_5 for '{}'", dim.name);
            assert!(
                prompt.contains("Score the output on its merits, not on how well it matches your expectations."),
                "narrative judge for '{}' must contain anti-agreement guard", dim.name
            );
            assert!(prompt.contains("Notable"), "narrative judge for '{}' must contain Notable", dim.name);
            assert!(
                prompt.contains("fitness_evaluation") && prompt.contains("score"),
                "narrative judge for '{}' must contain output schema", dim.name
            );
            assert!(
                !prompt.contains("## Upstream context"),
                "narrative judge for '{}' must not contain '## Upstream context'", dim.name
            );
        }
    }

    /// Sample response round-trips through parse_fitness_response for all 5 dims.
    #[test]
    fn v2_judge_schema_round_trips_through_parser() {
        use crate::ttd::fitness::parse_fitness_response;

        // A model response following the instructed schema
        let sample_response = r#"<fitness_evaluation>
<score>4</score>
<rationale>The synthesis accurately reflects the cited papers at the asserted evidence_grade.
Notable: The use of "converging" for the permafrost claim is appropriate given independent replications.</rationale>
</fitness_evaluation>"#;

        let parsed = parse_fitness_response(sample_response);
        assert_eq!(parsed.score, Some(4), "round-trip: schema response must parse to Some(4)");
        assert!(parsed.rationale.contains("permafrost"), "round-trip: rationale must be extracted");
    }

    // ── Fix B: source-vocabulary prohibition in v2 prompts ────────────────────

    /// Fix B (probe-17 cause 1, prong 1): render_synthesis_merger_v2 must
    /// contain the source-vocabulary prohibition phrase so the model knows
    /// Candidate N headings are presentation labels, not valid source ids.
    #[test]
    fn merger_v2_contains_source_vocab_prohibition() {
        use crate::ttd::artifact::SynthesisArtifact;

        // Build two minimal candidate artifacts (empty claims, just provenance).
        let c1 = SynthesisArtifact::new("s", "r", "q", "model", "v2/lit-review");
        let c2 = SynthesisArtifact::new("s", "r", "q", "model", "v2/lit-review");
        let candidates = vec![&c1, &c2];

        let prompt = render_synthesis_merger_v2(
            &candidates,
            "What are the effects?",
            "400-600",
            "",
            &std::collections::BTreeMap::new(),
        );

        assert!(
            prompt.contains("never valid source"),
            "merger_v2 prompt must contain 'never valid source' prohibition; got excerpt: {}",
            &prompt[..prompt.len().min(400)]
        );
    }

    // ── P-3: merger contract/context alignment ────────────────────────────────

    /// P-3 (kvasir C-N6): the merge rules arbitrate on sources, support_level,
    /// evidence_grade, method, year, and lineage — so each candidate block must
    /// render those fields with their VALUES, not merely cite the field names in
    /// the rules text. Asserts on the rendered `**field**: value` lines, which
    /// only the candidate blocks produce.
    #[test]
    fn merger_v2_renders_rule_cited_fields_into_candidate_blocks() {
        let c1 = fixture_synthesis_for_judges();
        let mut c2 = fixture_synthesis_for_judges();
        // Divergent provenance in candidate 2 — the conflict the rules arbitrate.
        c2.claims[0].evidence_grade = Some("strong".into());
        c2.claims[0].lineage = Some("Replicated by Hugelius 2020 field survey.".into());

        let prompt = render_synthesis_merger_v2(
            &[&c1, &c2],
            "test question",
            "400",
            "",
            &std::collections::BTreeMap::new(),
        );

        // Candidate headings preserved (the source-ID rule references them).
        assert!(prompt.contains("### Candidate 1"), "candidate 1 heading missing");
        assert!(prompt.contains("### Candidate 2"), "candidate 2 heading missing");

        // Claim text and sources rendered per claim.
        assert!(
            prompt.contains("**Text**: Permafrost thaw accelerates methane release."),
            "claim text must appear in candidate block"
        );
        assert!(
            prompt.contains("**Sources**: arxiv:2304.07620"),
            "claim sources must appear in candidate block"
        );

        // Every rule-cited v2 field rendered with its value.
        assert!(prompt.contains("**support_level**: converging"), "support_level value missing");
        assert!(prompt.contains("**evidence_grade**: moderate"), "candidate 1 evidence_grade missing");
        assert!(prompt.contains("**method**: observational"), "method value missing");
        assert!(prompt.contains("**year**: 2023"), "year value missing");
        assert!(
            prompt.contains("**lineage**: Follows from Schaefer 2014 permafrost models."),
            "candidate 1 lineage missing"
        );

        // Both sides of the divergence are visible — the merger can arbitrate.
        assert!(prompt.contains("**evidence_grade**: strong"), "candidate 2 evidence_grade missing");
        assert!(
            prompt.contains("**lineage**: Replicated by Hugelius 2020 field survey."),
            "candidate 2 lineage missing"
        );
    }

    /// Point 2 (Stage-2 soft-filter): a populated tier_map tags each merger
    /// candidate source with its credibility tier; `Unknown` and unmapped
    /// sources render untagged.
    #[test]
    fn merger_v2_tags_candidate_sources_with_credibility_tier() {
        use alzina_search::CredibilityTier;

        let c1 = fixture_synthesis_for_judges();
        let mut tier_map = std::collections::BTreeMap::new();
        tier_map.insert("arxiv:2304.07620".to_string(), CredibilityTier::Low);

        let tagged = render_synthesis_merger_v2(&[&c1], "q", "400", "", &tier_map);
        assert!(
            tagged.contains("**Sources**: arxiv:2304.07620 [low credibility]"),
            "mapped source must carry its tier tag: {}",
            &tagged[..tagged.len().min(600)]
        );

        // Empty map ⇒ untagged baseline (byte-stable with the pre-tier render).
        let untagged = render_synthesis_merger_v2(
            &[&c1],
            "q",
            "400",
            "",
            &std::collections::BTreeMap::new(),
        );
        assert!(
            untagged.contains("**Sources**: arxiv:2304.07620\n"),
            "empty map must render the source untagged"
        );
    }

    /// P-3 guard: v1-shaped claims (all v2 provenance fields None) render text
    /// and sources only — no empty placeholder lines for absent fields. Mirrors
    /// the conditional rendering of `render_fitness_judge_v2_synthesis`.
    #[test]
    fn merger_v2_omits_absent_provenance_fields_from_candidate_blocks() {
        use crate::ttd::artifact::{Claim, SynthesisArtifact};
        let mut c1 = SynthesisArtifact::new("s", "r", "q", "model", "v2/lit-review");
        c1.claims.push(Claim {
            text: "A v1-shaped claim.".into(),
            agreement_level: None,
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

        let prompt = render_synthesis_merger_v2(
            &[&c1],
            "test question",
            "400",
            "",
            &std::collections::BTreeMap::new(),
        );

        assert!(prompt.contains("**Text**: A v1-shaped claim."), "claim text must render");
        assert!(prompt.contains("**Sources**: (none)"), "empty sources must render '(none)'");
        // No rendered field lines for absent values (rule-text mentions like
        // "support_level" without the `**…**:` wrapper are expected and allowed).
        assert!(!prompt.contains("**support_level**:"), "absent support_level must be omitted");
        assert!(!prompt.contains("**evidence_grade**:"), "absent evidence_grade must be omitted");
        assert!(!prompt.contains("**method**:"), "absent method must be omitted");
        assert!(!prompt.contains("**year**:"), "absent year must be omitted");
        assert!(!prompt.contains("**lineage**:"), "absent lineage must be omitted");
    }

    /// Fix B (probe-17 cause 1, prong 1): render_revision_v2 must contain the
    /// source-vocabulary prohibition phrase.
    #[test]
    fn revision_v2_contains_source_vocab_prohibition() {
        use crate::adapter::{ExpertResponse, ResponseProvenance, SourceId};
        use crate::ttd::artifact::SynthesisArtifact;

        let synthesis = SynthesisArtifact::new("s", "r", "q", "model", "v2/lit-review");
        let panel = vec![ExpertResponse {
            expert_id: SourceId::new("arxiv:2105.14103".to_string()),
            prose: "Some prose.".to_string(),
            provenance: ResponseProvenance {
                source_id: SourceId::new("arxiv:2105.14103".to_string()),
                title: "Paper A".to_string(),
                year: Some(2023),
                authors: vec!["Author A".to_string()],
                credibility_tier: alzina_search::CredibilityTier::Unknown,
            },
        }];
        let missing = vec!["arxiv:2105.14103".to_string()];

        let prompt = render_revision_v2(&synthesis, &panel, &missing, "400-600", "");

        assert!(
            prompt.contains("never valid source"),
            "revision_v2 prompt must contain 'never valid source' prohibition; got excerpt: {}",
            &prompt[..prompt.len().min(400)]
        );
    }

    // ── Rubric-encoding Phase 1: plan-aware renderer tests (W-e714abb4) ───────

    /// Minimal plan with one registry term, one thread, two sections.
    fn fixture_plan() -> ReviewPlan {
        use crate::ttd::plan::{PlanArchetype, PlantedThread, TermRegistryEntry};
        ReviewPlan {
            archetype: PlanArchetype::ThesisAndConvergence,
            archetype_rationale: "Three independent lines converge on one thesis.".into(),
            focal_question: "Does permafrost thaw drive accelerating methane release?".into(),
            scope_exclusions: vec!["wetland methane outside permafrost zones".into()],
            term_registry: vec![TermRegistryEntry {
                term: "abrupt thaw".into(),
                definition: "Thermokarst-mediated rapid ground collapse.".into(),
                banned_synonyms: vec!["sudden melting".into(), "fast thaw".into()],
            }],
            planted_threads: vec![PlantedThread {
                id: "T1".into(),
                description: "Measurement-vs-model gap set up early, resolved late.".into(),
                marker: "the measurement gap".into(),
                setup_section: "Background".into(),
                payoff_section: "Convergence".into(),
            }],
            sections: vec![
                PlanSection {
                    heading: "Background".into(),
                    purpose: "Establish the thesis and the measurement landscape.".into(),
                    budget_words: Some(450),
                    claim_ids: vec!["C1".into(), "C2".into()],
                },
                PlanSection {
                    heading: "Convergence".into(),
                    purpose: "Cash in the planted threads against the thesis.".into(),
                    budget_words: Some(600),
                    claim_ids: vec!["C3".into()],
                },
            ],
        }
    }

    /// Byte-stability proof obligation: every planned renderer output is
    /// `base + suffix` — the base prompt is an EXACT PREFIX, and the base
    /// itself never mentions the plan. Falsifies any edit that mutates the
    /// shared template instead of composing it.
    #[test]
    fn planned_renderers_compose_base_as_exact_prefix() {
        let plan = fixture_plan();
        let synth = fixture_synthesis_for_judges();
        let shape = NarrativeShape::SectionedLongForm;

        let cases = [
            (
                "draft",
                render_narrative_draft_v2(&synth, shape),
                render_narrative_draft_v2_planned(&synth, shape, &plan),
            ),
            (
                "refine",
                render_narrative_refine_v2("text", "critique", shape, None),
                render_narrative_refine_v2_planned("text", "critique", shape, &plan, None),
            ),
            (
                "merge",
                render_narrative_final_merge_v2(&["a", "b"], shape),
                render_narrative_final_merge_v2_planned(&["a", "b"], shape, &plan),
            ),
        ];
        for (site, base, planned) in &cases {
            assert!(
                planned.starts_with(base.as_str()),
                "{site}: planned prompt must compose the base prompt as an exact prefix"
            );
            assert!(
                !base.contains("Winning plan"),
                "{site}: base prompt must never mention the plan (byte-stability)"
            );
            assert!(
                planned.contains("Winning plan"),
                "{site}: planned prompt must carry the plan block"
            );
        }
    }

    /// The four enforceable merge rules are all present, each with its
    /// enforcement mechanics: anti-hedge verdict fusion via strongest support
    /// span, term-registry conformance with the registry's banned synonyms,
    /// the banned-phrase list, and [Cx] citation preservation.
    #[test]
    fn merge_rules_block_carries_four_enforceable_rules() {
        let plan = fixture_plan();
        let rules = render_merge_rules_block(&plan);

        // Rule 1 — never average opposing verdicts into hedge.
        assert!(rules.contains("never average into a hedge"));
        assert!(rules.contains("strongest support span"));
        assert!(rules.contains("attributed dissent"));
        // Rule 2 — term registry with banned synonyms from THIS plan.
        assert!(rules.contains("Term-registry conformance"));
        assert!(rules.contains("\"abrupt thaw\""));
        assert!(rules.contains("sudden melting"));
        // Rule 3 — banned phrases from the shared const.
        for phrase in BANNED_PHRASES {
            assert!(rules.contains(phrase), "merge rules must list banned phrase '{phrase}'");
        }
        // Rule 4 — citation preservation.
        assert!(rules.contains("Citation preservation"));
        assert!(rules.contains("[Cx]"));
        // Mechanical enforcement is declared, not implied.
        assert!(rules.contains("mechanically scanned"));
    }

    /// `BANNED_PHRASES` (the post-merge scan referent) must stay in sync with
    /// the literal banned-phrase list inside the byte-stable base merge
    /// prompt — the scan must never check a phrase the prompt didn't ban.
    #[test]
    fn banned_phrases_const_in_sync_with_base_merge_prompt() {
        let base = render_narrative_final_merge_v2(&["a"], NarrativeShape::Concise);
        for phrase in BANNED_PHRASES {
            assert!(
                base.contains(&format!("\"{phrase}\"")),
                "base merge prompt must ban '{phrase}' (BANNED_PHRASES drifted from the template)"
            );
        }
    }

    /// C-N3: the section-merge prompt enumerates its context contents, carries
    /// the previous section's tail, rank-annotates every candidate, and states
    /// the vetoed-trajectory fan-in policy.
    #[test]
    fn section_merge_prompt_enumerates_context_and_states_fanin_policy() {
        let plan = fixture_plan();
        let section = &plan.sections[0];
        let prompt = render_section_merge_v3(
            &plan,
            section,
            &[(1, "Rank-one section body."), (3, "Rank-three section body.")],
            Some("The previous section closed on the measurement gap."),
        );

        // Condition 1 — enumerated context contents.
        assert!(prompt.contains("Your context contains, in order, and nothing else:"));
        assert!(prompt.contains("1. The winning plan header"));
        assert!(prompt.contains("5. The merge rules."));
        // Condition 2 — sequential merge with previous tail.
        assert!(prompt.contains("The previous section closed on the measurement gap."));
        assert!(prompt.contains("do NOT repeat it"));
        // Condition 3 — vetoed-trajectory policy stated at fan-in.
        assert!(prompt.contains("Vetoed-trajectory policy"));
        assert!(prompt.contains("failed validity"));
        assert!(prompt.contains("Anchor this section's STRUCTURE on the rank-1 candidate"));
        // Rank annotations on every candidate.
        assert!(prompt.contains("whole-document fitness rank 1"));
        assert!(prompt.contains("whole-document fitness rank 3"));
        assert!(prompt.contains("Rank-one section body."));
        assert!(prompt.contains("Rank-three section body."));
        // The section's plan entry, fully rendered.
        assert!(prompt.contains("**Heading**: Background"));
        assert!(prompt.contains("**Word budget**: 450 words"));
        assert!(prompt.contains("**Assigned claim IDs**: C1, C2"));
        // Exact heading instruction + merge rules embedded.
        assert!(prompt.contains("`## Background` exactly"));
        assert!(prompt.contains("never average into a hedge"));
    }

    /// First section: no previous tail — the prompt says so instead of
    /// rendering an empty block.
    #[test]
    fn section_merge_first_section_declares_no_prior_text() {
        let plan = fixture_plan();
        let prompt = render_section_merge_v3(&plan, &plan.sections[0], &[(1, "body")], None);
        assert!(prompt.contains("This is the first section — no prior text."));
    }
}
