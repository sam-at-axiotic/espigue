//! v2 lit-review term sheet — single source of vocabulary for B2 prompts and
//! B3 judges.
//!
//! MAS-PROMPT-CRAFT-V2 §1: define the terms ONCE here; every downstream consumer
//! (prompts, judges, schema) references these constants verbatim. No paraphrases.
//!
//! ## Scope (B1)
//!
//! B1 delivers the schema layer: vocabulary, profile enum, version consts,
//! additive artifact fields. Prompt text (B2) and judge dimensions (B3) hang off
//! these methods without re-plumbing.
//!
//! ## Security note (T-B1-02, recorded as design acceptance)
//!
//! v1's T-23-08 mitigation was "deterministic thresholds override LLM labels".
//! v2 removes that override by design — `support_level` is LLM-asserted, so a
//! prompt-injected claim could assert "established". Disposition: ACCEPTED at B1
//! with a named closure path — B3's traceability VETO (claim with empty sources
//! hard-fails the validity gate) is the enforcement layer.

// ── Version consts ────────────────────────────────────────────────────────────

/// Schema version string for v2 lit-review artifacts.
pub const SCHEMA_VERSION_V2: &str = "2.0";

/// Prompt version string for all v2 lit-review stages (graph, synthesis, narrative).
///
/// All three stage methods on `PromptProfile::V2LitReview` return this marker.
/// Per-stage prompt text differentiation is B2's responsibility; B1 pins one
/// version token per run so UAT probe #11 can gate on it without parsing YAML.
pub const PROMPT_VERSION_V2_LIT_REVIEW: &str = "v2/lit-review";

/// Prompt version string for all v3 long-form lit-review stages.
///
/// Decision 0 / Phase 0 (W-e714abb4): same one-token-per-run grain as v2 —
/// all three stage methods on `PromptProfile::V3LitReviewLong` return this
/// marker so probes can gate on the active profile without parsing YAML.
pub const PROMPT_VERSION_V3_LIT_REVIEW_LONG: &str = "v3/lit-review-long";

// ── PromptProfile ─────────────────────────────────────────────────────────────

/// Selects the active prompt/schema dialect for a synthesis run.
///
/// One profile per run — selected at the HTTP boundary via `prompt_profile` and
/// stamped onto `EngineConfig`. All three stages (graph, synthesis, narrative)
/// read their version strings from the profile's methods.
///
/// ## Default
///
/// `V1Delphi` is the default. Existing tests and callers that construct
/// `EngineConfig::new(...)` without `with_profile` get V1 behaviour, byte-identical
/// to pre-B1 runs.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum PromptProfile {
    /// v1 Delphi consensus schema — the default, byte-identical to pre-B1.
    #[default]
    V1Delphi,
    /// v2 lit-review schema — support_level taxonomy, typed gaps, schema 2.0.
    V2LitReview,
    /// v3 long-form lit-review — Decision 0 / Phase 0 (W-e714abb4): identical
    /// to `V2LitReview` everywhere (schema 2.0, v2 prompts, v2 judges, v2
    /// personas) EXCEPT the Stage-3 narrative output shape: the 300-500-word
    /// headerless constraint is lifted to sectioned long-form (`##` headings,
    /// no fixed word cap). Explicit opt-in only — never a default. One run
    /// yields one shape (see `narrative_shape`).
    V3LitReviewLong,
}

// ── NarrativeShape ────────────────────────────────────────────────────────────

/// Output shape contract for the Stage-3 narrative prompt family.
///
/// Derived from `PromptProfile::narrative_shape()` — the ONLY derivation point,
/// so all three constraint sites (draft, refine, merge) move together and one
/// run yields exactly one shape (no mixed-mode output). Decision 0 / Phase 0.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NarrativeShape {
    /// 300-500 words, no headers, no bullet lists — flowing prose only.
    /// Byte-identical to the pre-Phase-0 v2 prompt text.
    Concise,
    /// Sectioned long-form: markdown `##` headings, no fixed word cap,
    /// flowing prose within sections (bullet lists stay banned — the
    /// anti-list guard is a degeneration check, not a length constraint).
    SectionedLongForm,
}

impl PromptProfile {
    /// Schema version string stamped onto `SynthesisArtifact.schema_version`
    /// and `ArgumentationGraph.schema_version` at the engine boundary.
    pub fn schema_version(self) -> &'static str {
        match self {
            PromptProfile::V1Delphi => "1.0",
            // v3 changes document SHAPE, not artifact schema — stays "2.0".
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => SCHEMA_VERSION_V2,
        }
    }

    /// Prompt version string for the Stage-1 graph extraction stage.
    pub fn graph_prompt_version(self) -> &'static str {
        match self {
            PromptProfile::V1Delphi => "v1/graph",
            PromptProfile::V2LitReview => PROMPT_VERSION_V2_LIT_REVIEW,
            PromptProfile::V3LitReviewLong => PROMPT_VERSION_V3_LIT_REVIEW_LONG,
        }
    }

    /// Prompt version string for the Stage-2 synthesis stage.
    pub fn synthesis_prompt_version(self) -> &'static str {
        match self {
            PromptProfile::V1Delphi => "v1/synthesis",
            PromptProfile::V2LitReview => PROMPT_VERSION_V2_LIT_REVIEW,
            PromptProfile::V3LitReviewLong => PROMPT_VERSION_V3_LIT_REVIEW_LONG,
        }
    }

    /// Prompt version string for the Stage-3 narrative stage.
    pub fn narrative_prompt_version(self) -> &'static str {
        match self {
            PromptProfile::V1Delphi => "v1/narrative",
            PromptProfile::V2LitReview => PROMPT_VERSION_V2_LIT_REVIEW,
            PromptProfile::V3LitReviewLong => PROMPT_VERSION_V3_LIT_REVIEW_LONG,
        }
    }

    /// Output shape for the Stage-3 narrative prompt family (Decision 0 / Phase 0).
    ///
    /// The single derivation point for the draft/refine/merge constraint sites.
    /// Deliberately an exhaustive match (no wildcard): a future profile must
    /// declare its shape or fail to compile.
    pub fn narrative_shape(self) -> NarrativeShape {
        match self {
            PromptProfile::V1Delphi | PromptProfile::V2LitReview => NarrativeShape::Concise,
            PromptProfile::V3LitReviewLong => NarrativeShape::SectionedLongForm,
        }
    }
}

// ── TermDef ───────────────────────────────────────────────────────────────────

/// One entry in a controlled vocabulary table.
pub struct TermDef {
    /// Canonical term name (lowercase, hyphenated for multi-word terms).
    pub name: &'static str,
    /// Definition as it will appear verbatim in B2 prompts and B3 judge anchors.
    pub definition: &'static str,
}

// ── SUPPORT_LEVELS vocabulary ─────────────────────────────────────────────────

/// Closed vocabulary for `support_level` in v2 claims.
///
/// These five labels replace the v1 vote-based `agreement_level` taxonomy on the
/// v2 path. They are lit-review epistemics (what the evidence shows) NOT vote
/// shares (how many panellists agree).
///
/// B2 prompts and B3 judge anchors MUST reference these definitions verbatim —
/// no paraphrasing. This is the MAS-PROMPT-CRAFT-V2 §1 requirement.
pub const SUPPORT_LEVELS: [TermDef; 5] = [
    TermDef {
        name: "established",
        definition: "Corroborated across multiple independent lines of work over time; \
                     the claim is treated as settled in the reviewed literature.",
    },
    TermDef {
        name: "converging",
        definition: "Independent recent results pointing the same way but not yet settled; \
                     replication is ongoing or the mechanism is still debated.",
    },
    TermDef {
        name: "contested",
        definition: "Credible published disagreement on the claim itself; at least one \
                     peer-reviewed source contradicts or substantially qualifies it.",
    },
    TermDef {
        name: "emerging",
        definition: "Early-stage finding with limited replication; may appear in only one \
                     or two recent papers with no independent confirmation yet.",
    },
    TermDef {
        name: "single-source",
        definition: "Only one paper in the reviewed set speaks to this claim; the label \
                     is honest about corpus depth, not a judgment on that paper's quality.",
    },
];

// ── EVIDENCE_GRADES vocabulary ────────────────────────────────────────────────

/// Closed vocabulary for `evidence_grade` in v2 claims.
///
/// B1 stores `evidence_grade` as a free string — no normalisation at the parser
/// layer. Enforcement (rejecting out-of-vocabulary values, anchor scoring) is B3's
/// responsibility. B2 prompts embed these definitions verbatim via §1 rule.
pub const EVIDENCE_GRADES: [TermDef; 4] = [
    TermDef {
        name: "strong",
        definition: "Multiple independent studies with sound methods point the same way; \
                     direct measurement or successful replication.",
    },
    TermDef {
        name: "moderate",
        definition: "One methodologically sound study plus consistent indirect support from others.",
    },
    TermDef {
        name: "weak",
        definition: "Indirect, small-sample, or model-only support; no direct measurement.",
    },
    TermDef {
        name: "anecdotal",
        definition: "Illustrative or case-based only; not designed to test the claim.",
    },
];

// ── GAP_TYPES vocabulary ──────────────────────────────────────────────────────

/// Closed vocabulary for `gap_type` in v2 typed gaps.
///
/// Stored verbatim in `Gap.gap_type`; enforcement (rejecting out-of-vocabulary
/// values) is B3's responsibility. B1 stores the raw string.
pub const GAP_TYPES: [TermDef; 4] = [
    TermDef {
        name: "epistemic",
        definition: "A gap in what the field can currently know — the question cannot be \
                     answered with existing methods or data, not merely absent from this corpus.",
    },
    TermDef {
        name: "empirical",
        definition: "Missing observational or experimental data that could in principle be \
                     collected; the methods exist but the studies have not been done.",
    },
    TermDef {
        name: "methodological",
        definition: "The field lacks agreed methods or instruments to measure or test the \
                     relevant phenomenon reliably.",
    },
    TermDef {
        name: "theoretical",
        definition: "No adequate theoretical framework exists to explain or predict the \
                     phenomenon; mechanistic understanding is absent.",
    },
];

// ── Vocabulary normalisation ──────────────────────────────────────────────────

/// Case-insensitive, trimmed match of `raw` against `SUPPORT_LEVELS` names.
///
/// Returns `Some(canonical_name)` for in-vocabulary values; `None` for
/// out-of-vocabulary. The parser calls this and emits a `tracing::warn!` on
/// `None` — it NEVER substitutes a default. The v1 "divided" fabrication
/// mechanism has no v2 analogue.
pub fn normalise_support_level(raw: &str) -> Option<&'static str> {
    let lower = raw.trim().to_lowercase();
    for term in &SUPPORT_LEVELS {
        if term.name == lower.as_str() {
            return Some(term.name);
        }
    }
    None
}

// ── JudgeDim + V2_JUDGE_DIMS ─────────────────────────────────────────────────

/// One dimension in the v2 judge rubric.
///
/// Carries the dimension name, definition, and three concrete score anchors.
/// Anchors are concrete behaviours, not restated numbers. Where a dim touches
/// controlled vocabulary (faithfulness ↔ EVIDENCE_GRADES; tension_visibility ↔
/// "contested" in SUPPORT_LEVELS), the vocabulary name is used verbatim — never
/// paraphrased. MAS §1: define ONCE here; all consumers quote verbatim.
pub struct JudgeDim {
    /// Canonical dimension name (snake_case).
    pub name: &'static str,
    /// Definition as it will appear verbatim in B3 judge prompts.
    pub definition: &'static str,
    /// Anchor for score 1 — the worst concrete behaviour.
    pub anchor_1: &'static str,
    /// Anchor for score 3 — a mixed but recognisable behaviour.
    pub anchor_3: &'static str,
    /// Anchor for score 5 — the ideal concrete behaviour.
    pub anchor_5: &'static str,
}

/// The five v2 lit-review judge dimensions (B3).
///
/// Order: faithfulness is first because it is the validity anchor (is_valid_v2 gates on it,
/// mirroring is_valid_synthesis gating on the v1 faithfulness dim).
///
/// Each definition and its anchors are stable — B3 judge prompts quote them verbatim.
/// Where a dim references controlled vocabulary, the vocabulary term name is embedded
/// directly (faithfulness ↔ EVIDENCE_GRADES.strong/.weak; tension_visibility ↔ "contested").
pub const V2_JUDGE_DIMS: [JudgeDim; 5] = [
    JudgeDim {
        name: "faithfulness",
        definition: "Claims accurately reflect what the cited papers say, at no more than the \
                     asserted evidence_grade. An evidence_grade of \"strong\" requires convergent \
                     independent replication; an evidence_grade of \"weak\" means indirect or \
                     model-only support. Overstating the evidence_grade is a faithfulness failure \
                     even when the claim text is true.",
        anchor_1: "Claims contradict their cited papers, or the asserted evidence_grade (e.g. \
                   \"strong\") is contradicted by the paper's actual methods — the output invents \
                   beyond what the sources support.",
        anchor_3: "Most claims are accurate but isolated overstatement is present — one or two \
                   claims are assigned an evidence_grade one level above what the cited paper's \
                   methods support.",
        anchor_5: "Every claim checks out against its sources at the asserted evidence_grade; \
                   no claim overstates what the cited paper measured, replicated, or concluded.",
    },
    JudgeDim {
        name: "coverage",
        definition: "Spans the breadth of the reviewed corpus, not a convenient subset. \
                     A synthesis that builds its argument from one or two papers and ignores \
                     the rest fails coverage regardless of how well it handles those papers.",
        anchor_1: "Built from one or two papers; the majority of the reviewed corpus is absent \
                   from the synthesis without explanation.",
        anchor_3: "The majority of papers are reflected in the synthesis, but some relevant papers \
                   whose findings would change the picture are missing without explanation.",
        anchor_5: "Every paper in the reviewed corpus is reflected, or its exclusion is explicitly \
                   explained (e.g. out of scope, superseded by a later paper cited).",
    },
    JudgeDim {
        name: "tension_visibility",
        definition: "Published disagreement is surfaced and characterised, never smoothed into \
                     false agreement. A claim with support_level \"contested\" must be accompanied \
                     by an account of who disagrees and on what grounds. Omitting known contradiction \
                     is a tension_visibility failure.",
        anchor_1: "Real contradictions present in the input papers are absent from the synthesis; \
                   \"contested\" support_level claims appear without identifying the disagreement.",
        anchor_3: "Tensions are mentioned but not characterised — the synthesis notes disagreement \
                   exists without saying who disagrees, on what claim, or what divides them.",
        anchor_5: "Each contested claim names the disagreeing lines of work and articulates what \
                   divides them — method, measurement, interpretation, or scope — so the reader \
                   can evaluate the disagreement, not just note its existence.",
    },
    JudgeDim {
        name: "lineage_clarity",
        definition: "How findings build on prior work is traceable. Where the input papers \
                     carry lineage, year, or method provenance, the synthesis surfaces it so \
                     the reader can follow the intellectual chain from earlier findings to later ones.",
        anchor_1: "Claims float free of intellectual history — no lineage notes, years, or method \
                   attributions; the reader cannot tell which findings are foundational and which \
                   are recent extensions.",
        anchor_3: "Some lineage notes are present but inconsistent — a few claims trace their \
                   descent from prior work while others of similar nature do not.",
        anchor_5: "The reader can trace which findings descend from which; where the input papers \
                   support it, lineage notes, year markers, and method attributions are present \
                   and accurate.",
    },
    JudgeDim {
        name: "recency_balance",
        definition: "Recent and foundational work are in balance. The synthesis uses years to \
                     trace intellectual lineage, not to rank truth — a 2019 replication outweighs \
                     a 2025 preprint when the evidence quality supports it. Temporal spread \
                     is honestly represented.",
        anchor_1: "Reads as if the field stopped (or started) at one moment — either dominated \
                   by recent papers with no foundational grounding, or anchored in older work \
                   with no acknowledgement of more recent developments.",
        anchor_3: "A temporal skew is present and acknowledged — the synthesis notes it leans \
                   recent or foundational, but does not explain whether this reflects the evidence \
                   or a selection bias.",
        anchor_5: "Temporal spread is honestly represented; years are cited where available; the \
                   synthesis does not weight findings by recency when quality-based weighting \
                   would differ.",
    },
];

// ── V2_NARRATIVE_JUDGE_DIMS — narrative-scoped anchors ───────────────────────

/// The five v2 judge dims, re-anchored for the **narrative** stage.
///
/// Why this exists (`.planning/JUDGE-CALIBRATION-PLAN.md`, 2026-06-19): the
/// narrative judge reused `V2_JUDGE_DIMS` unchanged, whose anchors are written in
/// synthesis vocabulary — they key off per-claim fields (`support_level`,
/// `evidence_grade`) that a prose narrative does not carry. Across 10 corpora that
/// mis-fit pinned narrative `tension_visibility` at 5.0 and `faithfulness` low at
/// 2.2. This array re-states each dim's 1/3/5 anchors as concrete behaviours of a
/// *prose review* (sections, transitions, position-taking), per the lit-review
/// checklist §7.
///
/// MAS §1 (shared ontology) is preserved structurally: `name` and `definition` are
/// copied from `V2_JUDGE_DIMS` by index — they CANNOT drift from the shared
/// vocabulary. Only the anchors are narrative-native. Order matches
/// `V2_JUDGE_DIMS` so name→weight lookup against `V2_NARRATIVE_WEIGHTS` is
/// unaffected.
///
/// Residual (flagged in the plan): `faithfulness`'s shared definition still names
/// `evidence_grade`. Anchors alone may not fully un-pin it; the ≥2-corpus re-run
/// decides whether an artifact-neutral definition rephrase is also needed.
pub const V2_NARRATIVE_JUDGE_DIMS: [JudgeDim; 5] = [
    // [0] faithfulness — prose calibrates language to the evidence it cites.
    JudgeDim {
        name: V2_JUDGE_DIMS[0].name,
        definition: V2_JUDGE_DIMS[0].definition,
        anchor_1: "The prose asserts findings its cited papers do not support, or states as \
                   established what a cited paper only proposed or showed in a single model-only \
                   result — the review overstates the evidence.",
        anchor_3: "Most assertions match their citations, but the prose occasionally overstates — \
                   it presents a contested or model-only result as settled, or attaches a strong \
                   verb (\"proves\", \"establishes\") to support that is indirect or unreplicated.",
        anchor_5: "Every assertion matches what its cited papers report, and the prose calibrates \
                   its language to the evidence — hedged where support is indirect or model-only, \
                   confident only where replication warrants it.",
    },
    // [1] coverage — breadth of the corpus carried in the prose.
    JudgeDim {
        name: V2_JUDGE_DIMS[1].name,
        definition: V2_JUDGE_DIMS[1].definition,
        anchor_1: "The review builds its argument from a handful of papers; most of the reviewed \
                   corpus never appears in the prose, and the omission is not explained.",
        anchor_3: "The prose reflects most of the corpus, but relevant work that would change the \
                   picture is missing without a word on why — or whole sub-areas are named once \
                   and dropped.",
        anchor_5: "The prose engages the breadth of the corpus — representative works developed in \
                   depth, neighbours compressed to a line or citation — and any deliberate \
                   exclusion is stated (out of scope, superseded by a later cited paper).",
    },
    // [2] tension_visibility — disagreement surfaced in prose, not smoothed.
    JudgeDim {
        name: V2_JUDGE_DIMS[2].name,
        definition: V2_JUDGE_DIMS[2].definition,
        anchor_1: "The review reads as smooth consensus — real disagreements among the cited \
                   papers are absent, or named only as \"some debate exists\" with no sides.",
        anchor_3: "Disagreements are mentioned but not characterised — the prose notes that work \
                   conflicts without saying who takes which side or what divides them.",
        anchor_5: "The prose names the disagreeing lines of work, states what divides them (method, \
                   measurement, interpretation, or scope), and either resolves the objection with \
                   evidence or marks it open — the reader can weigh the disagreement, not just \
                   note it.",
    },
    // [3] lineage_clarity — the intellectual chain is followable in prose.
    JudgeDim {
        name: V2_JUDGE_DIMS[3].name,
        definition: V2_JUDGE_DIMS[3].definition,
        anchor_1: "Findings float free of intellectual history — the prose gives no sense of which \
                   work is foundational and which extends it; no years, no \"building on\", no \
                   descent from earlier results.",
        anchor_3: "Some lineage is drawn — a few passages trace how one finding grew from an \
                   earlier one — but it is inconsistent, and other transitions just announce the \
                   next topic without connecting it to what came before.",
        anchor_5: "The reader can follow the intellectual chain through the prose — transitions \
                   name the limitation that forces the next development, foundational work is \
                   distinguished from recent extension, and years or method provenance ground the \
                   descent where the sources support it.",
    },
    // [4] recency_balance — temporal spread honestly weighted in prose.
    JudgeDim {
        name: V2_JUDGE_DIMS[4].name,
        definition: V2_JUDGE_DIMS[4].definition,
        anchor_1: "The review reads as if the field began or ended at one moment — dominated by \
                   the newest preprints with no foundational grounding, or anchored in older work \
                   with no acknowledgement of recent developments.",
        anchor_3: "A temporal skew is present and the prose acknowledges it leans recent or \
                   foundational, but does not say whether this reflects the weight of evidence or \
                   just what was easiest to cite.",
        anchor_5: "Recent and foundational work are in balance; the prose cites years to trace \
                   lineage rather than to rank truth, and does not let a newer date outweigh a \
                   better-replicated older result.",
    },
];

// ── Source-id allowlist predicate (F13 — probe-18) ───────────────────────────

/// Return `true` when `id` is a valid claim/quote source identifier.
///
/// A source id is valid when it satisfies EITHER of two criteria:
///
/// 1. **Shape**: the id starts with `"arxiv:"` or `"s2:"` (ASCII-case-insensitive
///    prefix match). This covers any real paper cited by a v2 synthesis claim or
///    quote, including non-panel papers retrieved from arXiv/SemanticScholar — the
///    F11 lane. A casing wobble on the prefix (e.g. `"ArXiv:..."`) must not destroy
///    real provenance.
///
/// 2. **Panel membership**: the id is an exact member of `panel_ids`.  This covers
///    panel experts whose expert_id is opaque (e.g. `"pmc_case_study"`) and would
///    not pass the shape check.
///
/// Everything else is invalid — including probe-18 mutated labels
/// (`"s1_candidate1"` .. `"s5_candidate5"`) and probe-17 canonical forms
/// (`"Candidate1"`, `"candidate_3"`, etc.). Pattern-blacklisting those labels loses
/// to a generative model that mutates the label; the durable fix is validating that
/// the source LOOKS like a real id.
///
/// ## Why shape-plus-panel, not panel-only
///
/// F11 (probe-15): synthesis quotes may cite legitimate non-panel papers retrieved
/// from arXiv/S2 retrieval context. Those papers are not panel members; rejecting
/// them would silently drop real provenance. The shape check preserves them.
///
/// ## Why this supersedes `is_candidate_label`
///
/// `is_candidate_label` is a blacklist: it matches `candidate\d+` and variants.
/// Probe-18 proved haiku evades it by using `sN_candidateN`. Allowlisting valid
/// shapes is a positive check — a model cannot mint a label that starts with
/// `"arxiv:"` or `"s2:"` without it being a resolvable id shape.
///
/// ## Relationship to `is_candidate_label` (deleted in F13 Task 2)
///
/// `is_candidate_label` has been deleted; its callers (post_process.rs and
/// fitness.rs) now use `is_valid_source_id`. The canonical CandidateN rejection
/// test cases live on in the `is_valid_source_id_shape_and_panel` test below.
pub fn is_valid_source_id(id: &str, panel_ids: &std::collections::HashSet<String>) -> bool {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Shape lane: arxiv: or s2: prefix (case-insensitive on the prefix only).
    // "arxiv:" is 6 bytes; "s2:" is 3 bytes.
    let lower_prefix6 = trimmed.get(..6).map(|s| s.to_ascii_lowercase());
    let lower_prefix3 = trimmed.get(..3).map(|s| s.to_ascii_lowercase());
    if lower_prefix6.as_deref() == Some("arxiv:") || lower_prefix3.as_deref() == Some("s2:") {
        return true;
    }

    // Panel membership lane: exact match against the known panel expert-id set.
    panel_ids.contains(trimmed)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: PromptProfile::default() is V1Delphi; V1 version methods return
    /// the exact current strings; V2LitReview returns "2.0" / "v2/lit-review".
    #[test]
    fn prompt_profile_defaults_and_version_strings() {
        let default_profile = PromptProfile::default();
        assert_eq!(default_profile, PromptProfile::V1Delphi, "default must be V1Delphi");

        // V1 arms must return the exact existing literals (byte-identity).
        assert_eq!(PromptProfile::V1Delphi.schema_version(), "1.0");
        assert_eq!(PromptProfile::V1Delphi.graph_prompt_version(), "v1/graph");
        assert_eq!(PromptProfile::V1Delphi.synthesis_prompt_version(), "v1/synthesis");
        assert_eq!(PromptProfile::V1Delphi.narrative_prompt_version(), "v1/narrative");

        // V2 arms return the v2/lit-review marker for all stage methods.
        assert_eq!(PromptProfile::V2LitReview.schema_version(), "2.0");
        assert_eq!(PromptProfile::V2LitReview.graph_prompt_version(), "v2/lit-review");
        assert_eq!(PromptProfile::V2LitReview.synthesis_prompt_version(), "v2/lit-review");
        assert_eq!(PromptProfile::V2LitReview.narrative_prompt_version(), "v2/lit-review");
    }

    /// Decision 0 / Phase 0: V3LitReviewLong version strings and shape mapping.
    /// v3 keeps schema 2.0 (artifact schema unchanged) but carries its own
    /// prompt-version token; it is the ONLY profile with a long-form shape,
    /// and it is never the default.
    #[test]
    fn v3_long_form_profile_version_strings_and_shape() {
        assert_ne!(
            PromptProfile::default(),
            PromptProfile::V3LitReviewLong,
            "v3 must never be the default — explicit opt-in only"
        );

        assert_eq!(PromptProfile::V3LitReviewLong.schema_version(), "2.0");
        assert_eq!(PromptProfile::V3LitReviewLong.graph_prompt_version(), "v3/lit-review-long");
        assert_eq!(PromptProfile::V3LitReviewLong.synthesis_prompt_version(), "v3/lit-review-long");
        assert_eq!(PromptProfile::V3LitReviewLong.narrative_prompt_version(), "v3/lit-review-long");

        // One shape per profile — the single derivation point for all three
        // narrative constraint sites (draft, refine, merge).
        assert_eq!(PromptProfile::V1Delphi.narrative_shape(), NarrativeShape::Concise);
        assert_eq!(PromptProfile::V2LitReview.narrative_shape(), NarrativeShape::Concise);
        assert_eq!(
            PromptProfile::V3LitReviewLong.narrative_shape(),
            NarrativeShape::SectionedLongForm
        );
    }

    /// Test 2: SUPPORT_LEVELS has exactly 5 entries with non-empty definitions;
    /// GAP_TYPES has 4 entries including "epistemic".
    #[test]
    fn vocabulary_coverage() {
        assert_eq!(SUPPORT_LEVELS.len(), 5, "must have exactly 5 support levels");
        let expected_names = ["established", "converging", "contested", "emerging", "single-source"];
        for (i, term) in SUPPORT_LEVELS.iter().enumerate() {
            assert_eq!(term.name, expected_names[i], "name mismatch at index {i}");
            assert!(!term.definition.is_empty(), "definition must be non-empty for {}", term.name);
        }

        assert_eq!(GAP_TYPES.len(), 4, "must have exactly 4 gap types");
        let has_epistemic = GAP_TYPES.iter().any(|t| t.name == "epistemic");
        assert!(has_epistemic, "GAP_TYPES must contain 'epistemic'");
        for term in &GAP_TYPES {
            assert!(!term.definition.is_empty(), "gap_type definition must be non-empty for {}", term.name);
        }
    }

    /// V2_JUDGE_DIMS: exactly 5 dims, all with non-empty fields, "faithfulness" first.
    #[test]
    fn v2_judge_dims_coverage() {
        assert_eq!(V2_JUDGE_DIMS.len(), 5, "must have exactly 5 v2 judge dims");

        let expected_names = ["faithfulness", "coverage", "tension_visibility", "lineage_clarity", "recency_balance"];
        for (i, dim) in V2_JUDGE_DIMS.iter().enumerate() {
            assert_eq!(dim.name, expected_names[i], "name mismatch at index {i}");
            assert!(!dim.definition.is_empty(), "definition must be non-empty for {}", dim.name);
            assert!(!dim.anchor_1.is_empty(), "anchor_1 must be non-empty for {}", dim.name);
            assert!(!dim.anchor_3.is_empty(), "anchor_3 must be non-empty for {}", dim.name);
            assert!(!dim.anchor_5.is_empty(), "anchor_5 must be non-empty for {}", dim.name);
        }

        // faithfulness must be first (validity anchor)
        assert_eq!(V2_JUDGE_DIMS[0].name, "faithfulness", "faithfulness must be first");
    }

    /// V2_NARRATIVE_JUDGE_DIMS: §1 shared ontology held — names and definitions are
    /// byte-identical to V2_JUDGE_DIMS (same order), but every anchor is rewritten.
    #[test]
    fn v2_narrative_judge_dims_share_ontology_and_rewrite_anchors() {
        assert_eq!(
            V2_NARRATIVE_JUDGE_DIMS.len(),
            V2_JUDGE_DIMS.len(),
            "narrative dims must match shared dim count"
        );
        for (i, (nar, shared)) in V2_NARRATIVE_JUDGE_DIMS
            .iter()
            .zip(V2_JUDGE_DIMS.iter())
            .enumerate()
        {
            // §1: name + definition shared verbatim, same order.
            assert_eq!(nar.name, shared.name, "name must match shared at index {i}");
            assert_eq!(
                nar.definition, shared.definition,
                "definition must be shared verbatim for {}",
                nar.name
            );
            // Mechanism fix: anchors must actually be re-written, not copied.
            assert_ne!(nar.anchor_1, shared.anchor_1, "anchor_1 must be narrative-scoped for {}", nar.name);
            assert_ne!(nar.anchor_3, shared.anchor_3, "anchor_3 must be narrative-scoped for {}", nar.name);
            assert_ne!(nar.anchor_5, shared.anchor_5, "anchor_5 must be narrative-scoped for {}", nar.name);
            assert!(!nar.anchor_1.is_empty() && !nar.anchor_3.is_empty() && !nar.anchor_5.is_empty());
        }
    }

    /// V2_NARRATIVE_JUDGE_DIMS: the mechanism fix — narrative anchors must NOT grade
    /// against per-claim fields a prose narrative lacks (`support_level`,
    /// `evidence_grade`). Those references are what pinned the dims across 10 corpora.
    #[test]
    fn v2_narrative_anchors_drop_prose_absent_fields() {
        for dim in V2_NARRATIVE_JUDGE_DIMS.iter() {
            for (label, anchor) in [
                ("anchor_1", dim.anchor_1),
                ("anchor_3", dim.anchor_3),
                ("anchor_5", dim.anchor_5),
            ] {
                assert!(
                    !anchor.contains("support_level"),
                    "{} {} must not key off support_level (prose lacks it): {anchor}",
                    dim.name,
                    label
                );
                assert!(
                    !anchor.contains("evidence_grade"),
                    "{} {} must not key off evidence_grade (prose lacks it): {anchor}",
                    dim.name,
                    label
                );
            }
        }
    }

    /// V2_JUDGE_DIMS: "faithfulness" definition references evidence_grade vocabulary.
    #[test]
    fn faithfulness_dim_references_evidence_grade_vocab() {
        let faith = &V2_JUDGE_DIMS[0];
        assert_eq!(faith.name, "faithfulness");
        // Definition must reference evidence_grade (the controlled vocabulary link)
        assert!(
            faith.definition.contains("evidence_grade"),
            "faithfulness definition must reference evidence_grade: {}",
            faith.definition
        );
    }

    /// V2_JUDGE_DIMS: "tension_visibility" definition references "contested" (SUPPORT_LEVELS vocab).
    #[test]
    fn tension_visibility_dim_references_contested_vocab() {
        let tv = V2_JUDGE_DIMS.iter().find(|d| d.name == "tension_visibility").unwrap();
        assert!(
            tv.definition.contains("contested"),
            "tension_visibility definition must reference 'contested' (SUPPORT_LEVELS vocab): {}",
            tv.definition
        );
    }

    /// Test: normalise_support_level — in-vocab, case-insensitive; out-of-vocab → None.
    #[test]
    fn normalise_support_level_in_and_out_of_vocab() {
        assert_eq!(normalise_support_level("established"), Some("established"));
        assert_eq!(normalise_support_level("CONVERGING"), Some("converging"));
        assert_eq!(normalise_support_level("  Contested  "), Some("contested"));
        assert_eq!(normalise_support_level("single-source"), Some("single-source"));
        assert_eq!(normalise_support_level("strongly-agreed"), None);
        assert_eq!(normalise_support_level("consensus"), None);
        assert_eq!(normalise_support_level("divided"), None);
        assert_eq!(normalise_support_level(""), None);
    }

    // ── F13: is_valid_source_id ────────────────────────────────────────────────

    /// F13 (probe-18): is_valid_source_id must accept arxiv:/s2:-prefixed ids (shape
    /// lane, empty panel) and panel members by exact id, and reject everything else
    /// including probe-18 mutated labels (sN_candidateN), probe-17 canonical forms
    /// (CandidateN), synthetic system ids, and blank strings.
    #[test]
    fn is_valid_source_id_shape_and_panel() {
        use std::collections::HashSet;

        let empty: HashSet<String> = HashSet::new();
        let panel_with_pmc: HashSet<String> =
            ["pmc_case_study".to_string()].into_iter().collect();

        // Shape lane (empty panel): arxiv: and s2: prefixes accepted.
        assert!(is_valid_source_id("arxiv:2105.14103", &empty), "arxiv: id must be valid");
        assert!(is_valid_source_id("s2:abc123", &empty), "s2: id must be valid");

        // Case-insensitive prefix — a casing wobble must not destroy real provenance.
        assert!(is_valid_source_id("ArXiv:2502.12110", &empty), "ArXiv: (mixed case) must be valid");
        assert!(is_valid_source_id("S2:abc123", &empty), "S2: (uppercase) must be valid");

        // Panel membership — non-arxiv/s2 id accepted only when in the set.
        assert!(
            is_valid_source_id("pmc_case_study", &panel_with_pmc),
            "pmc_case_study must be valid via panel membership"
        );
        assert!(
            !is_valid_source_id("pmc_case_study", &empty),
            "pmc_case_study must be invalid when not in panel"
        );

        // Probe-18 mutated labels (sN_candidateN) must be rejected.
        assert!(!is_valid_source_id("s1_candidate1", &empty), "s1_candidate1 is invalid");
        assert!(!is_valid_source_id("s3_candidate3", &empty), "s3_candidate3 is invalid");
        assert!(!is_valid_source_id("s5_candidate5", &empty), "s5_candidate5 is invalid");

        // Probe-17 canonical forms still rejected (allowlist subsumes blacklist).
        assert!(!is_valid_source_id("Candidate1", &empty), "Candidate1 is invalid");
        assert!(!is_valid_source_id("candidate_3", &empty), "candidate_3 is invalid");
        assert!(!is_valid_source_id("CANDIDATE12", &empty), "CANDIDATE12 is invalid");

        // Synthetic system ids rejected.
        assert!(!is_valid_source_id("system", &empty), "system is invalid");
        assert!(!is_valid_source_id("system_resolution", &empty), "system_resolution is invalid");
        assert!(!is_valid_source_id("expert_consensus", &empty), "expert_consensus is invalid");

        // Empty / whitespace rejected.
        assert!(!is_valid_source_id("", &empty), "empty string is invalid");
        assert!(!is_valid_source_id("   ", &empty), "whitespace-only is invalid");
    }
}
