//! ReviewPlan + plan tournament — Phase 1 core of the rubric-encoding design.
//!
//! Spec: `artifacts/weaves/W-e714abb4/muninn-rubric-encoding-architecture-r2.md`
//! (§4 shape (b), §4.1 machinery reuse, §4.2 judge hardening, §7 lint tiers),
//! with Kvasir gate conditions folded in
//! (`artifacts/weaves/W-e714abb4/kvasir-gate-rereview-r2.md`): C-N1 (full claim
//! texts in the plan-judge corpus digest, never titles) and the corrected call
//! envelope (~26–32 calls).
//!
//! ## What this module is
//!
//! Consensus mush is manufactured at `narrative_final_merge` by averaging five
//! independently-chosen document architectures. The tournament removes that
//! averaging **by selection** before any prose exists: five rubric-informed
//! plan drafts compete, deterministic lints + three anchored judge dims rank
//! them with the EXISTING sort-only machinery (`sort_candidates_best_first`,
//! `FitnessEval` veto pattern, GraphMerger take-best precedent — §4.1: zero new
//! selection mechanisms), and all five Stage-3 trajectories then develop under
//! the single winning plan.
//!
//! ## D0 partition (spec §3)
//!
//! Core fields (archetype, focal question, scope exclusions, term registry,
//! planted threads) are **D0-indep** — they discipline even a 300–500-word
//! paragraph. Extension fields (section skeleton, per-section budgets,
//! section→claim-ID assignments) are **D0-dep** and only requested/enforced
//! under `NarrativeShape::SectionedLongForm` (the v3 profile, Decision 0).
//!
//! ## Byte-stability
//!
//! Nothing here runs unless `TtdConfig::plan_mode != PlanMode::Disabled`
//! (the default). v1/v2/v3-without-plan runs are byte-identical to pre-Phase-1.

use std::collections::HashMap;
use std::sync::Arc;

use crate::executor::AgentExecutor;
use crate::ttd::artifact::SynthesisArtifact;
use crate::ttd::fitness::{parse_fitness_response, sort_candidates_best_first, FitnessEval};
use crate::ttd::mod_types::TtdError;
use crate::ttd::term_sheet::{JudgeDim, NarrativeShape};

// ── Tunables (named constants so the envelope is auditable) ───────────────────

/// Tournament width — five rubric-informed plan drafts (spec §4: decorrelates
/// the plan-quality ceiling from 1 sample to 5).
pub const PLAN_TOURNAMENT_DRAFTS: usize = 5;

/// Bounded redrafts of the WINNER only (spec §4.1: the 2-step denoise loop is
/// deliberately NOT reused — refining losing plans pays refinement cost for
/// candidates selection discards).
pub const PLAN_WINNER_REDRAFT_CAP: usize = 2;

/// Claim-coverage floor for the claim-ID cross-check lint (spec §4.2 layer 2).
pub const PLAN_CLAIM_COVERAGE_MIN: f32 = 0.6;

/// No claim/tension ID may be assigned to more than this many sections —
/// defeats assign-everything-everywhere gaming (spec §4.2 layer 2).
pub const PLAN_MAX_SECTIONS_PER_ID: usize = 2;

/// Validity floor on the corpus_fit dim (spec §4.1: plan validity =
/// deterministic plan lints pass AND corpus-fit dim ≥ 4 — mirrors
/// `is_valid_v2`'s faithfulness ≥ 4 anchor).
pub const PLAN_CORPUS_FIT_VALIDITY_MIN: u8 = 4;

/// Hard call-budget ceiling for one tournament (C-N7 corrected arithmetic:
/// 5 drafts + 15 judges + ≤2 redrafts + ≤6 redraft re-evaluations = 28,
/// envelope 26–32). `run_plan_tournament` asserts it never exceeds this.
pub const PLAN_TOURNAMENT_MAX_CALLS: usize = 32;

// ── PlanMode ──────────────────────────────────────────────────────────────────

/// Plan-stage operating mode (spec §4 fallback chain — the reversal path).
///
/// `Disabled` is the default: no plan calls are made, no prompts change, and
/// every existing profile is byte-stable. `SinglePlanner` is round-1's shape
/// retained as the config fallback (1 draft + 3 judges). `Tournament` is the
/// recommended shape (b).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PlanMode {
    /// No plan stage — byte-identical to pre-Phase-1 behaviour.
    #[default]
    Disabled,
    /// Round-1 fallback: one plan draft, judged and lint-checked, no tournament.
    SinglePlanner,
    /// Shape (b): five rubric-informed drafts → sort-only take-best.
    Tournament,
}

// ── Plan artifact types ───────────────────────────────────────────────────────

/// The four document archetypes from the operator's quality rubric (rubric A).
///
/// Parsing is strict: exactly one archetype, or the plan fails to parse. This
/// is the mechanical half of `archetype_singularity` — a "hybrid with no
/// primary" cannot even be represented.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PlanArchetype {
    /// A single intellectual story unfolding from problem to resolution.
    NarrativeArc,
    /// A lattice of interlocking open problems, each cell a tension.
    ProblemLattice,
    /// A mutually-exclusive, collectively-exhaustive taxonomy of the field.
    MeceTaxonomy,
    /// One thesis argued throughout, with converging lines of evidence.
    ThesisAndConvergence,
}

impl PlanArchetype {
    /// Canonical lowercase-hyphenated name (the true name used in prompts,
    /// plan XML, and run metadata).
    pub fn as_str(self) -> &'static str {
        match self {
            PlanArchetype::NarrativeArc => "narrative-arc",
            PlanArchetype::ProblemLattice => "problem-lattice",
            PlanArchetype::MeceTaxonomy => "mece-taxonomy",
            PlanArchetype::ThesisAndConvergence => "thesis-and-convergence",
        }
    }

    /// Strict parse — trims, lowercases, normalises separators to hyphens,
    /// then requires an EXACT vocabulary match. "narrative-arc + problem-lattice"
    /// and any other hybrid fails: archetype singularity is enforced at the
    /// type level, not by judge goodwill.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let norm: String = raw
            .trim()
            .to_lowercase()
            .chars()
            .map(|c| if c == ' ' || c == '_' { '-' } else { c })
            .collect();
        match norm.as_str() {
            "narrative-arc" => Ok(PlanArchetype::NarrativeArc),
            "problem-lattice" => Ok(PlanArchetype::ProblemLattice),
            "mece-taxonomy" => Ok(PlanArchetype::MeceTaxonomy),
            "thesis-and-convergence" => Ok(PlanArchetype::ThesisAndConvergence),
            other => Err(format!(
                "archetype must be exactly one of narrative-arc | problem-lattice | \
                 mece-taxonomy | thesis-and-convergence; got '{other}'"
            )),
        }
    }
}

/// One canonical term in the plan's term registry.
///
/// `banned_synonyms` powers the deterministic TermDriftLint (spec §7, veto
/// tier, D0-indep): the registry names the drift it forbids, so the scan is
/// mechanical over declared referents — proxy IS intent.
#[derive(Clone, Debug, PartialEq)]
pub struct TermRegistryEntry {
    /// The canonical term — used verbatim everywhere downstream.
    pub term: String,
    /// One-line definition.
    pub definition: String,
    /// Synonyms the document must NOT substitute for the canonical term.
    pub banned_synonyms: Vec<String>,
}

/// A planted thread: a question or tension set up early and cashed in later.
///
/// `marker` is a short distinctive phrase repeated VERBATIM at both setup and
/// payoff, giving ThreadCashInLint (spec §7, feedback tier) a mechanical
/// referent. Necessary-not-sufficient by design — the sufficient check is
/// `argumentative_force` over the span (spec §6.2, Phase 2).
#[derive(Clone, Debug, PartialEq)]
pub struct PlantedThread {
    /// Stable thread id (e.g. "T1").
    pub id: String,
    /// What this thread sets up and must pay off.
    pub description: String,
    /// Distinctive phrase repeated verbatim at setup and payoff.
    pub marker: String,
    /// Section heading where the thread is planted.
    pub setup_section: String,
    /// Section heading where the thread is cashed in.
    pub payoff_section: String,
}

/// One section of the plan skeleton (D0-dep extension — only under
/// `NarrativeShape::SectionedLongForm`).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanSection {
    /// Markdown heading text (without the leading `## `).
    pub heading: String,
    /// What this section must accomplish — phrased against the focal question.
    pub purpose: String,
    /// Word budget for the section (ProportionalityLint denominator).
    pub budget_words: Option<usize>,
    /// Claim/tension IDs this section organises (e.g. "C1", "C7", "T2").
    pub claim_ids: Vec<String>,
}

/// The rubric-informed skeleton plan — the declared target every downstream
/// conformance check verifies against (draft-against-plan, never
/// draft-against-taste).
#[derive(Clone, Debug, PartialEq)]
pub struct ReviewPlan {
    /// Exactly one organising archetype (rubric A).
    pub archetype: PlanArchetype,
    /// Recorded reasoning for the archetype choice (from corpus shape).
    pub archetype_rationale: String,
    /// The one argued question the whole document answers.
    pub focal_question: String,
    /// Topics explicitly out of scope (scope-exclusion body-scan referent).
    pub scope_exclusions: Vec<String>,
    /// Canonical vocabulary + banned synonyms (TermDriftLint referent).
    pub term_registry: Vec<TermRegistryEntry>,
    /// Threads planted early and cashed in later (ThreadCashInLint referent).
    pub planted_threads: Vec<PlantedThread>,
    /// Section skeleton with budgets and claim assignments. EMPTY under
    /// `NarrativeShape::Concise` (D0-dep extension fields, spec §3).
    pub sections: Vec<PlanSection>,
}

// ── XML parsing ───────────────────────────────────────────────────────────────

/// Extract the inner content of the FIRST `<tag>...</tag>` block.
/// Attribute-free by design — plan drafts are instructed to emit element-only XML.
fn extract_block(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = s.find(&open)? + open.len();
    let end = s[start..].find(&close)? + start;
    Some(s[start..end].trim().to_string())
}

/// Extract the inner contents of ALL `<tag>...</tag>` blocks, in order.
fn extract_blocks(s: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(start_rel) = rest.find(&open) {
        let start = start_rel + open.len();
        let Some(end_rel) = rest[start..].find(&close) else { break };
        let end = start + end_rel;
        out.push(rest[start..end].trim().to_string());
        rest = &rest[end + close.len()..];
    }
    out
}

/// Parse a `<plan>` XML document into a `ReviewPlan`.
///
/// Hard requirements (parse failure, not lint failure): a `<plan>` block,
/// exactly one in-vocabulary `<archetype>`, and a non-empty `<focal_question>`.
/// Everything else degrades to empty collections — lints and judges grade the
/// gaps; the parser only refuses structurally unusable plans.
pub fn parse_review_plan(raw: &str) -> Result<ReviewPlan, TtdError> {
    let plan_block = extract_block(raw, "plan")
        .ok_or_else(|| TtdError::ParseFailed("no <plan> block in plan draft".into()))?;

    let archetype_raw = extract_block(&plan_block, "archetype")
        .ok_or_else(|| TtdError::ParseFailed("plan missing <archetype>".into()))?;
    let archetype = PlanArchetype::parse(&archetype_raw).map_err(TtdError::ParseFailed)?;

    let focal_question = extract_block(&plan_block, "focal_question").unwrap_or_default();
    if focal_question.is_empty() {
        return Err(TtdError::ParseFailed("plan missing <focal_question>".into()));
    }

    let archetype_rationale =
        extract_block(&plan_block, "archetype_rationale").unwrap_or_default();

    let scope_exclusions = extract_block(&plan_block, "scope_exclusions")
        .map(|b| extract_blocks(&b, "exclusion"))
        .unwrap_or_default();

    let term_registry = extract_block(&plan_block, "term_registry")
        .map(|b| {
            extract_blocks(&b, "term")
                .iter()
                .filter_map(|t| {
                    let term = extract_block(t, "name")?;
                    let definition = extract_block(t, "definition").unwrap_or_default();
                    let banned_synonyms = extract_block(t, "banned")
                        .map(|b| {
                            b.split('|')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect()
                        })
                        .unwrap_or_default();
                    Some(TermRegistryEntry { term, definition, banned_synonyms })
                })
                .collect()
        })
        .unwrap_or_default();

    let planted_threads = extract_block(&plan_block, "planted_threads")
        .map(|b| {
            extract_blocks(&b, "thread")
                .iter()
                .filter_map(|t| {
                    Some(PlantedThread {
                        id: extract_block(t, "id")?,
                        description: extract_block(t, "description").unwrap_or_default(),
                        marker: extract_block(t, "marker")?,
                        setup_section: extract_block(t, "setup").unwrap_or_default(),
                        payoff_section: extract_block(t, "payoff").unwrap_or_default(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let sections = extract_block(&plan_block, "sections")
        .map(|b| {
            extract_blocks(&b, "section")
                .iter()
                .filter_map(|s| {
                    Some(PlanSection {
                        heading: extract_block(s, "heading")?,
                        purpose: extract_block(s, "purpose").unwrap_or_default(),
                        budget_words: extract_block(s, "budget_words")
                            .and_then(|w| w.trim().parse::<usize>().ok()),
                        claim_ids: extract_block(s, "claim_ids")
                            .map(|ids| {
                                ids.split(',')
                                    .map(|i| i.trim().to_string())
                                    .filter(|i| !i.is_empty())
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(ReviewPlan {
        archetype,
        archetype_rationale,
        focal_question,
        scope_exclusions,
        term_registry,
        planted_threads,
        sections,
    })
}

// ── Corpus digest (C-N1) ──────────────────────────────────────────────────────

/// Render the corpus digest carried into every plan-draft and plan-judge
/// context.
///
/// **C-N1 (Kvasir gate re-review, BINDING):** the digest carries FULL claim
/// texts and full tension descriptions — never titles or one-liners. A
/// titles-only digest re-opens the digest-visible-half hole at the exact
/// check the tournament concentrates authority in: `corpus_fit` judged
/// against titles is satisfiable by a fluent-but-vacuous plan. Claims average
/// 1–2 sentences; a full-text digest is ~400 words for a typical corpus —
/// both halves of the plan ⟷ corpus relation still fit one context (spec §9
/// row 1, the structural payoff of selecting at plan level).
pub fn render_corpus_digest(synthesis: &SynthesisArtifact) -> String {
    let mut out = String::from("## Corpus digest (full texts — not titles)\n\n### Claims\n\n");
    for (i, claim) in synthesis.claims.iter().enumerate() {
        let support = claim.support_level.as_deref().unwrap_or("unknown");
        let grade = claim.evidence_grade.as_deref().unwrap_or("unknown");
        out.push_str(&format!(
            "[C{n}] (support: {support}; evidence: {grade}) {text}\n",
            n = i + 1,
            text = claim.text,
        ));
    }
    out.push_str("\n### Tensions\n\n");
    if synthesis.areas_of_disagreement.is_empty() {
        out.push_str("(none recorded)\n");
    } else {
        for (i, tension) in synthesis.areas_of_disagreement.iter().enumerate() {
            out.push_str(&format!("[T{n}] {tension}\n", n = i + 1));
        }
    }
    out
}

// ── Plan judge dims (spec §4.2 layer 3) ───────────────────────────────────────

/// The three plan-selection judge dimensions.
///
/// Order: `corpus_fit` first — it is the validity anchor (`is_valid_plan`
/// gates on it, mirroring `is_valid_v2` gating on faithfulness). Each anchor_1
/// names the known gaming path explicitly (spec §4.2: anchors that don't name
/// the gaming path select FOR it).
pub const PLAN_JUDGE_DIMS: [JudgeDim; 3] = [
    JudgeDim {
        name: "corpus_fit",
        definition: "The plan organises THIS corpus's claims and tensions — every section's \
                     purpose is derivable from the declared archetype applied to the actual \
                     claim texts in the digest, and the claim assignments group claims that \
                     genuinely belong together.",
        anchor_1: "Archetype declared but the sections are generic boilerplate not derivable \
                   from the declared framework; claim assignments are arbitrary — a \
                   self-consistent fiction that would fit any corpus equally well.",
        anchor_3: "Most sections follow from the archetype and most claim assignments are \
                   sensible, but at least one section is generic filler or one cluster of \
                   claims is split or grouped without rationale.",
        anchor_5: "Every section is a recognisable application of the declared archetype to \
                   the specific claims assigned to it; reading the digest alone, one could \
                   reconstruct why each claim landed where it did.",
    },
    JudgeDim {
        name: "archetype_singularity",
        definition: "One organising principle is primary (rubric A). The declared archetype \
                     governs the whole skeleton; any secondary structure is subordinate and \
                     explicitly so.",
        anchor_1: "A hybrid with no primary — sections alternate between organising logics \
                   (part taxonomy, part chronology, part thesis) with no declared hierarchy.",
        anchor_3: "The declared archetype governs most sections, but one or two sections \
                   follow a different logic without being flagged as deliberate exceptions.",
        anchor_5: "Every section's purpose statement is an instance of the single declared \
                   archetype; the skeleton would be visibly broken under any other archetype.",
    },
    JudgeDim {
        name: "focal_question_grip",
        definition: "The focal question does real organising work: each section's purpose \
                     statement depends on it, and the planned conclusion answers it.",
        anchor_1: "Focal question stated but inert — no section's purpose statement depends \
                   on it; deleting the question would change nothing in the skeleton.",
        anchor_3: "Some section purposes engage the focal question; others could belong to \
                   any review of this corpus.",
        anchor_5: "Every section purpose advances the focal question and the final section \
                   commits to answering it; the question is the skeleton's spine.",
    },
];

/// Plan-stage fitness weight table. Sums to 1.0 (tested below). `corpus_fit`
/// carries half the weight — it is the relational dim the tournament exists
/// to judge well (spec §4.2 layer 1).
pub const PLAN_WEIGHTS: &[(&str, f32)] = &[
    ("corpus_fit", 0.50),
    ("archetype_singularity", 0.25),
    ("focal_question_grip", 0.25),
];

/// Plan validity (spec §4.1): deterministic plan lints pass (no veto) AND
/// corpus_fit ≥ `PLAN_CORPUS_FIT_VALIDITY_MIN`. Exact `is_valid_v2` shape —
/// machinery reuse, not new selection mechanism.
pub fn is_valid_plan(eval: &FitnessEval) -> bool {
    eval.veto.is_none()
        && eval
            .score("corpus_fit")
            .map_or(false, |s| s >= PLAN_CORPUS_FIT_VALIDITY_MIN)
}

// ── Deterministic plan lints (spec §4.2 layer 2, §7 veto tier) ────────────────

/// Claim-ID cross-check (veto tier, D0-dep — inactive when `sections` is empty).
///
/// Verifies against validated referents (precedent: `sanitise_cx_citations`
/// validating `[Cx]` against the real claim set):
/// 1. every listed ID exists in the artifact (`C1..Cn` / `T1..Tm`),
/// 2. claim coverage ≥ `PLAN_CLAIM_COVERAGE_MIN`,
/// 3. no ID assigned to more than `PLAN_MAX_SECTIONS_PER_ID` sections
///    (defeats assign-everything-everywhere gaming).
///
/// Returns `Some(reason)` on failure — wired as a `FitnessEval` veto.
pub fn plan_lint_claim_id_cross_check(
    plan: &ReviewPlan,
    n_claims: usize,
    n_tensions: usize,
) -> Option<String> {
    if plan.sections.is_empty() {
        return None; // D0-indep degenerate case: no skeleton, lint inactive (spec §4.2).
    }

    let mut id_section_count: HashMap<&str, usize> = HashMap::new();
    let mut covered_claims: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for section in &plan.sections {
        for id in &section.claim_ids {
            let id = id.as_str();
            // Existence: C<i> with 1 ≤ i ≤ n_claims, or T<j> with 1 ≤ j ≤ n_tensions.
            // strip_prefix matches by char, not byte — a non-ASCII first char
            // (e.g. a Cyrillic "С1" lookalike from the LLM) yields None, not a panic.
            let valid = if let Some(idx) = id.strip_prefix('C') {
                idx.parse::<usize>()
                    .map_or(false, |i| i >= 1 && i <= n_claims)
            } else if let Some(idx) = id.strip_prefix('T') {
                idx.parse::<usize>()
                    .map_or(false, |i| i >= 1 && i <= n_tensions)
            } else {
                false
            };
            if !valid {
                return Some(format!(
                    "plan lint: section '{}' lists ID '{id}' which does not exist in the \
                     artifact ({n_claims} claims, {n_tensions} tensions)",
                    section.heading
                ));
            }
            if let Some(idx) = id.strip_prefix('C') {
                if let Ok(i) = idx.parse::<usize>() {
                    covered_claims.insert(i);
                }
            }
            *id_section_count.entry(id).or_insert(0) += 1;
        }
    }

    for (id, count) in &id_section_count {
        if *count > PLAN_MAX_SECTIONS_PER_ID {
            return Some(format!(
                "plan lint: ID '{id}' assigned to {count} sections \
                 (max {PLAN_MAX_SECTIONS_PER_ID} — assign-everything gaming guard)"
            ));
        }
    }

    if n_claims > 0 {
        let coverage = covered_claims.len() as f32 / n_claims as f32;
        if coverage < PLAN_CLAIM_COVERAGE_MIN {
            return Some(format!(
                "plan lint: claim coverage {:.2} below floor {PLAN_CLAIM_COVERAGE_MIN} \
                 ({}/{n_claims} claims assigned to sections)",
                coverage,
                covered_claims.len()
            ));
        }
    }

    None
}

/// Core-field lint (veto tier, D0-indep): the plan must carry at least one
/// term-registry entry and at least one planted thread — these are the
/// referents every downstream deterministic lint scans against; a plan
/// without them silently disables the enforcement layer it exists to feed.
pub fn plan_lint_core_fields(plan: &ReviewPlan) -> Option<String> {
    if plan.term_registry.is_empty() {
        return Some("plan lint: term_registry is empty — TermDriftLint has no referent".into());
    }
    if plan.planted_threads.is_empty() {
        return Some(
            "plan lint: planted_threads is empty — ThreadCashInLint has no referent".into(),
        );
    }
    None
}

/// Run all veto-tier plan lints; first failure wins.
pub fn run_plan_lints(plan: &ReviewPlan, n_claims: usize, n_tensions: usize) -> Option<String> {
    plan_lint_core_fields(plan)
        .or_else(|| plan_lint_claim_id_cross_check(plan, n_claims, n_tensions))
}

// ── Document-side deterministic lints (generated FROM the plan artifact) ──────

/// Banned phrases — single source of truth for both the merge-prompt rule and
/// the mechanical post-merge scan. A sync test asserts each phrase appears in
/// `render_narrative_final_merge_v2`'s output, so the prompt literal (kept
/// byte-stable) and this scan list cannot drift apart.
pub const BANNED_PHRASES: [&str; 5] = [
    "it is important to note",
    "plays a crucial role",
    "in conclusion",
    "a growing body of literature",
    "further research is needed",
];

/// One deterministic-lint finding (feedback or report-card line — these scans
/// never gate; tier assignments per spec §7).
pub type LintFinding = String;

/// TermDriftLint (spec §7, veto tier at plan-conformance, D0-indep): scan a
/// document for banned synonyms declared in the plan's term registry.
/// Case-insensitive substring scan — mechanical over declared referents.
pub fn term_drift_scan(text: &str, plan: &ReviewPlan) -> Vec<LintFinding> {
    let lower = text.to_lowercase();
    let mut findings = Vec::new();
    for entry in &plan.term_registry {
        for synonym in &entry.banned_synonyms {
            let syn = synonym.trim().to_lowercase();
            if !syn.is_empty() && lower.contains(&syn) {
                findings.push(format!(
                    "term-drift: banned synonym '{synonym}' present; canonical term is '{}'",
                    entry.term
                ));
            }
        }
    }
    findings
}

/// Banned-phrase scan (mechanical pair of the merge prompt's anti-formulaic
/// rule). "further research is needed" is reported but is allowed in-prose
/// when naming a specific typed gap — the prompt states the exception; the
/// scan reports unconditionally and the report-card reader adjudicates.
pub fn banned_phrase_scan(text: &str) -> Vec<LintFinding> {
    let lower = text.to_lowercase();
    BANNED_PHRASES
        .iter()
        .filter(|p| lower.contains(*p))
        .map(|p| format!("banned-phrase: '{p}' present in merged output"))
        .collect()
}

/// Split a sectioned long-form document into `(heading, body)` pairs on
/// markdown `## ` headings. Text before the first heading is ignored.
pub fn split_sections(text: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            sections.push((heading.trim().to_string(), String::new()));
        } else if let Some((_, body)) = sections.last_mut() {
            body.push_str(line);
            body.push('\n');
        }
    }
    sections
}

/// ProportionalityLint (spec §7, feedback tier, D0-dep): per-section word
/// counts vs the plan's budgets. Reports ratios outside [0.5, 2.0] and plan
/// sections missing from the document. Gaming path (pad with filler) is
/// known — hence feedback-only, never a gate.
pub fn section_budget_scan(text: &str, plan: &ReviewPlan) -> Vec<LintFinding> {
    if plan.sections.is_empty() {
        return vec![];
    }
    let doc_sections = split_sections(text);
    let mut findings = Vec::new();
    for plan_section in &plan.sections {
        let found = doc_sections
            .iter()
            .find(|(h, _)| h.eq_ignore_ascii_case(&plan_section.heading));
        match found {
            None => findings.push(format!(
                "proportionality: plan section '{}' missing from document",
                plan_section.heading
            )),
            Some((_, body)) => {
                if let Some(budget) = plan_section.budget_words {
                    if budget > 0 {
                        let words = body.split_whitespace().count();
                        let ratio = words as f32 / budget as f32;
                        if !(0.5..=2.0).contains(&ratio) {
                            findings.push(format!(
                                "proportionality: section '{}' is {words} words vs budget \
                                 {budget} (ratio {ratio:.2}, outside [0.5, 2.0])",
                                plan_section.heading
                            ));
                        }
                    }
                }
            }
        }
    }
    findings
}

/// ThreadCashInLint (spec §7, feedback tier): verify each planted thread's
/// marker appears at least twice (setup + payoff), and — when the document
/// parses into sections — that one occurrence lands in the declared payoff
/// section. Necessary-not-sufficient: the marker has no validated external
/// referent set (unlike `[Cx]`), so this NEVER gates; the sufficient check is
/// `argumentative_force` over the span (spec §6.2, Phase 2).
pub fn thread_callback_scan(text: &str, plan: &ReviewPlan) -> Vec<LintFinding> {
    let lower = text.to_lowercase();
    let doc_sections = split_sections(text);
    let mut findings = Vec::new();
    for thread in &plan.planted_threads {
        let marker = thread.marker.trim().to_lowercase();
        if marker.is_empty() {
            continue;
        }
        let occurrences = lower.matches(&marker).count();
        if occurrences < 2 {
            findings.push(format!(
                "planted-thread {}: marker '{}' appears {occurrences}x — needs setup AND payoff",
                thread.id, thread.marker
            ));
            continue;
        }
        if !doc_sections.is_empty() && !thread.payoff_section.is_empty() {
            let paid_off = doc_sections.iter().any(|(h, body)| {
                h.eq_ignore_ascii_case(&thread.payoff_section)
                    && body.to_lowercase().contains(&marker)
            });
            if !paid_off {
                findings.push(format!(
                    "planted-thread {}: marker '{}' never cashed in inside payoff section '{}'",
                    thread.id, thread.marker, thread.payoff_section
                ));
            }
        }
    }
    findings
}

/// Run every document-side deterministic lint generated from the plan
/// artifact. Feedback/report-card tier — callers log findings; nothing gates.
pub fn run_plan_document_lints(text: &str, plan: &ReviewPlan) -> Vec<LintFinding> {
    let mut findings = term_drift_scan(text, plan);
    findings.extend(banned_phrase_scan(text));
    findings.extend(section_budget_scan(text, plan));
    findings.extend(thread_callback_scan(text, plan));
    findings
}

// ── Prompt renderers ──────────────────────────────────────────────────────────

/// Rubric block shared by every plan-draft prompt — the four archetypes with
/// working definitions (the "rubric-informed" half of the tournament).
const ARCHETYPE_RUBRIC_BLOCK: &str = "\
## Document archetypes (choose EXACTLY ONE as primary)

- **narrative-arc**: a single intellectual story unfolding from problem to resolution; \
sections are stations on the arc.
- **problem-lattice**: a lattice of interlocking open problems; each section is a cell \
holding one tension and the evidence pulling at it.
- **mece-taxonomy**: a mutually-exclusive, collectively-exhaustive partition of the \
field; sections are the partition cells.
- **thesis-and-convergence**: one thesis argued throughout; sections are independent \
lines of evidence converging on (or resisting) it.

Declare ONE archetype and derive every section from it. A hybrid with no primary \
fails selection.";

/// Render the plan-draft prompt (one per tournament entrant).
///
/// Carries the full-text corpus digest (C-N1), the archetype rubric, and the
/// element categories the plan must declare. Under `Concise` shape the
/// skeleton/budget elements are omitted (D0-dep, spec §3).
pub fn render_plan_draft(synthesis: &SynthesisArtifact, shape: NarrativeShape) -> String {
    let digest = render_corpus_digest(synthesis);

    let skeleton_block = match shape {
        NarrativeShape::SectionedLongForm => {
            "\n  <sections>\n    <section>\n      <heading>Introduction</heading>\n      \
             <purpose>open with the tension or obstacle that makes this review necessary (not the topic); motivate from the reader's world before the field; state the one focal question; declare scope exclusions; give a map of the argument</purpose>\n      \
             <budget_words>400</budget_words>\n      \
             <claim_ids>C1</claim_ids>\n    </section>\n    \
             <section>\n      <heading>A descriptive title naming the actual tension or idea</heading>\n      \
             <purpose>What this section accomplishes, phrased against the focal question</purpose>\n      \
             <budget_words>600</budget_words>\n      \
             <claim_ids>C1, C3, T2</claim_ids>\n    </section>\n    \
             <section>\n      <heading>Open questions and future directions</heading>\n      \
             <purpose>convert the thin and unresolved literature into a research agenda — what the evidence cannot yet answer, and the concrete next steps</purpose>\n      \
             <budget_words>500</budget_words>\n      \
             <claim_ids>T2</claim_ids>\n    </section>\n    \
             <section>\n      <heading>Conclusion</heading>\n      \
             <purpose>replay the organising framework and re-sort the findings by it; render a verdict (take a position); map the gaps onto future work — a verdict of the framework, not a section summary</purpose>\n      \
             <budget_words>400</budget_words>\n      \
             <claim_ids>C3</claim_ids>\n    </section>\n    \
             <!-- one <section> per skeleton section; every claim ID must exist in the \
             digest; no ID in more than 2 sections; cover at least 60% of claims -->\n  \
             </sections>"
        }
        NarrativeShape::Concise => "",
    };

    // Bookend + heading rules — only meaningful when a section skeleton exists
    // (SectionedLongForm). Empty under Concise, so that prompt is byte-identical.
    let skeleton_rules = match shape {
        NarrativeShape::SectionedLongForm => {
            "\n6. A section skeleton that bookends the argument (great-review structure — see docs/synthesis/literature-review-checklist.md):\n\
             - The FIRST section MUST be an Introduction that opens with the TENSION or obstacle that makes this review necessary — not the topic — motivates from the reader's world before the field, states the one focal question, declares the scope exclusions, and gives a map of the argument.\n\
             - The SECOND-TO-LAST section MUST be an \"Open questions and future directions\" section that converts the thin and unresolved literature into a research agenda — what the evidence cannot yet answer, and the concrete next steps.\n\
             - The LAST section MUST be a Conclusion that REPLAYS the organising framework and re-sorts the findings by it, renders a verdict (takes a position), and maps the gaps onto future work — a verdict of the framework, not a summary of sections.\n\
             - Name every interior section for the substantive tension or idea it holds (e.g. \"External stores versus embedded memory operations\"). Do NOT use generic numbered labels like \"Problem 1\", \"Section 2\", or \"Part III\" — each heading must carry meaning on its own, so a reader could reconstruct the table of contents from the framing alone.\n"
        }
        NarrativeShape::Concise => "",
    };

    format!(
        r#"You are drafting the skeleton PLAN for a critical literature review — not the review itself.

The plan is the declared target every later draft will be checked against. It must \
organise THIS corpus (full claim and tension texts below), not a generic review shape.

{digest}

{rubric}

## Required plan elements

1. The archetype, with the corpus-shape reasoning behind the choice.
2. ONE focal question the whole document argues an answer to.
3. Scope exclusions: topics deliberately out of scope.
4. A term registry: the canonical term for each load-bearing concept, its one-line \
definition, and the banned synonyms the document must never substitute.
5. Planted threads: questions set up early and cashed in later. Each thread carries a \
short distinctive marker phrase repeated VERBATIM at setup and payoff.
{skeleton_rules}
Output EXACTLY this XML (element-only, no attributes):

<plan>
  <archetype>one of: narrative-arc | problem-lattice | mece-taxonomy | thesis-and-convergence</archetype>
  <archetype_rationale>why this archetype fits this corpus</archetype_rationale>
  <focal_question>the one argued question</focal_question>
  <scope_exclusions>
    <exclusion>excluded topic</exclusion>
  </scope_exclusions>
  <term_registry>
    <term>
      <name>canonical-term</name>
      <definition>one-line definition</definition>
      <banned>synonym1|synonym2</banned>
    </term>
  </term_registry>
  <planted_threads>
    <thread>
      <id>T1</id>
      <description>what is set up and must be paid off</description>
      <marker>distinctive verbatim phrase</marker>
      <setup>setup section heading</setup>
      <payoff>payoff section heading</payoff>
    </thread>
  </planted_threads>{skeleton}
</plan>
"#,
        digest = digest,
        rubric = ARCHETYPE_RUBRIC_BLOCK,
        skeleton_rules = skeleton_rules,
        skeleton = skeleton_block,
    )
}

/// Render the winner-redraft prompt — the draft prompt plus the failure
/// feedback (lint veto reasons and/or low-scoring dims). Bounded by
/// `PLAN_WINNER_REDRAFT_CAP`, applied post-selection to the winner ONLY.
pub fn render_plan_redraft(
    synthesis: &SynthesisArtifact,
    shape: NarrativeShape,
    previous_plan_raw: &str,
    feedback: &str,
) -> String {
    format!(
        "{base}\n## Your previous plan (selected as best of {n}, but failing validity)\n\n\
         {previous}\n\n## What must change\n\n{feedback}\n\n\
         Redraft the FULL plan XML, fixing only what the feedback names — keep what works.\n",
        base = render_plan_draft(synthesis, shape),
        n = PLAN_TOURNAMENT_DRAFTS,
        previous = previous_plan_raw,
        feedback = feedback,
    )
}

/// Render one plan-judge prompt: one dim, full plan text, full-text corpus
/// digest. BOTH halves of the plan ⟷ corpus relation are in this one context
/// (spec §9 row 1; C-N1 makes the corpus half full-text).
pub fn render_plan_judge(dim: &JudgeDim, plan_raw: &str, corpus_digest: &str) -> String {
    format!(
        "You are evaluating one candidate review PLAN on a single dimension: {name}.\n\n\
         ## Dimension\n\n**{name}**: {definition}\n\n\
         **Score anchors**:\n- 1: {a1}\n- 3: {a3}\n- 5: {a5}\n\n\
         Your rationale must quote the specific plan passages that drove the score.\n\n\
         Score the plan on its merits, not on how well it matches your expectations.\n\n\
         ## Candidate plan\n\n{plan}\n\n\
         {digest}\n\n\
         Output your score as <fitness_evaluation><score>N</score>\
         <rationale>why</rationale></fitness_evaluation>\n",
        name = dim.name,
        definition = dim.definition,
        a1 = dim.anchor_1,
        a3 = dim.anchor_3,
        a5 = dim.anchor_5,
        plan = plan_raw,
        digest = corpus_digest,
    )
}

/// Render the winning plan as a prompt-injection block for Stage-3 draft,
/// refine, and merge prompts ("develop under the single winning plan").
pub fn render_plan_block(plan: &ReviewPlan) -> String {
    let mut out = String::from("## Winning plan (develop under this plan — it is the declared target you will be checked against)\n\n");
    out.push_str(&format!(
        "**Archetype**: {} — {}\n",
        plan.archetype.as_str(),
        plan.archetype_rationale
    ));
    out.push_str(&format!("**Focal question**: {}\n", plan.focal_question));
    if !plan.scope_exclusions.is_empty() {
        out.push_str(&format!(
            "**Out of scope**: {}\n",
            plan.scope_exclusions.join("; ")
        ));
    }
    if !plan.term_registry.is_empty() {
        out.push_str("\n**Term registry** — use the canonical term, NEVER a banned synonym:\n");
        for entry in &plan.term_registry {
            out.push_str(&format!(
                "- \"{}\": {}{}\n",
                entry.term,
                entry.definition,
                if entry.banned_synonyms.is_empty() {
                    String::new()
                } else {
                    format!(" (never: {})", entry.banned_synonyms.join(", "))
                }
            ));
        }
    }
    if !plan.planted_threads.is_empty() {
        out.push_str(
            "\n**Planted threads** — set up, then cash in; repeat the marker phrase VERBATIM \
             at both sites:\n",
        );
        for thread in &plan.planted_threads {
            out.push_str(&format!(
                "- {} [marker: \"{}\"]: {} (setup: {}; payoff: {})\n",
                thread.id,
                thread.marker,
                thread.description,
                thread.setup_section,
                thread.payoff_section
            ));
        }
    }
    if !plan.sections.is_empty() {
        out.push_str("\n**Section skeleton** — use these exact `##` headings, in this order:\n");
        for (i, section) in plan.sections.iter().enumerate() {
            let budget = section
                .budget_words
                .map(|b| format!(" (~{b} words)"))
                .unwrap_or_default();
            let ids = if section.claim_ids.is_empty() {
                String::new()
            } else {
                format!(" [claims: {}]", section.claim_ids.join(", "))
            };
            out.push_str(&format!(
                "{n}. ## {heading} — {purpose}{budget}{ids}\n",
                n = i + 1,
                heading = section.heading,
                purpose = section.purpose,
            ));
        }
    }
    out
}

/// Render the plan-conformance judge prompt: verify the DRAFT against the
/// PLAN — never against taste. Both halves (plan + draft) in one context.
///
/// This is the judge half of draft-time enforcement; the mechanical half is
/// `run_plan_document_lints`. Anchors are phrased exclusively as
/// plan-conformance observations so a judge cannot substitute its own
/// architectural preference for the declared target.
pub fn render_plan_conformance_judge(plan: &ReviewPlan, draft: &str) -> String {
    format!(
        "You are evaluating one candidate narrative on a single dimension: plan_conformance.\n\n\
         ## Dimension\n\n**plan_conformance**: The draft develops under the declared plan \
         below — its archetype, focal question, term registry, planted threads, and (when \
         present) section skeleton. Judge ONLY conformance to this plan. Do NOT judge \
         whether the plan itself is good, and do NOT reward deviations you would have \
         preferred — the plan is the target, not your taste.\n\n\
         **Score anchors**:\n\
         - 1: The draft ignores the plan — different organising principle, focal question \
         unaddressed, registry terms replaced by synonyms, threads never planted or cashed.\n\
         - 3: The draft follows the plan's skeleton and vocabulary in most places but drops \
         at least one planted thread or drifts from the declared archetype in one section.\n\
         - 5: Every plan element is realised: the archetype governs the structure, each \
         section serves its declared purpose, canonical terms are used throughout, and \
         every planted thread is set up and cashed in at its declared sites.\n\n\
         Your rationale must quote the plan element and the draft passage for each \
         deviation found.\n\n\
         {plan_block}\n\n## Candidate narrative\n\n{draft}\n\n\
         Output your score as <fitness_evaluation><score>N</score>\
         <rationale>why</rationale></fitness_evaluation>\n",
        plan_block = render_plan_block(plan),
        draft = draft,
    )
}

// ── Tournament ────────────────────────────────────────────────────────────────

/// One scored tournament entrant.
#[derive(Clone, Debug)]
pub struct PlanCandidate {
    /// Parsed plan (None when the draft was unparseable — vetoed).
    pub plan: Option<ReviewPlan>,
    /// Raw LLM output (retained for run metadata / redraft context).
    pub raw: String,
    /// Fitness eval: 3 plan dims + lint veto.
    pub eval: FitnessEval,
}

/// Tournament outcome. `runners_up` are retained for run metadata — the
/// stated repair path for plan-induced correlated macro failures is "rerun
/// with amended plan (operator-supplied or runner-up)" (spec §8 C10).
#[derive(Clone, Debug)]
pub struct PlanTournamentOutcome {
    /// The winning plan all five trajectories develop under.
    pub winner: ReviewPlan,
    /// Raw text of the winning plan (run metadata).
    pub winner_raw: String,
    /// Whether the winner passed `is_valid_plan` (a best-of-invalid winner is
    /// surfaced, not hidden — the operator veto window needs to see it).
    pub winner_valid: bool,
    /// Losing candidates, best-first (run metadata; repair-path contrast).
    pub runners_up: Vec<PlanCandidate>,
    /// Total LLM calls spent (drafts + judges + redrafts + re-evaluations).
    pub calls_used: usize,
}

/// Evaluate one candidate: veto-tier lints, then (if not vetoed) the three
/// plan judge dims. Vetoed/unparseable candidates skip judge calls — paying
/// judge cost for a candidate that sorts last regardless wastes budget.
async fn evaluate_plan_candidate(
    raw: &str,
    synthesis: &SynthesisArtifact,
    digest: &str,
    executor: &Arc<dyn AgentExecutor>,
    agent_id: &alzina_core::identity::AgentId,
    model: &str,
    calls_used: &mut usize,
) -> PlanCandidate {
    let none_scores = || {
        PLAN_JUDGE_DIMS
            .iter()
            .map(|d| (d.name.to_string(), None))
            .collect::<Vec<(String, Option<u8>)>>()
    };

    let plan = match parse_review_plan(raw) {
        Ok(p) => p,
        Err(e) => {
            return PlanCandidate {
                plan: None,
                raw: raw.to_string(),
                eval: FitnessEval::new(none_scores()).with_veto(format!("unparseable plan: {e}")),
            };
        }
    };

    let n_claims = synthesis.claims.len();
    let n_tensions = synthesis.areas_of_disagreement.len();
    if let Some(reason) = run_plan_lints(&plan, n_claims, n_tensions) {
        return PlanCandidate {
            plan: Some(plan),
            raw: raw.to_string(),
            eval: FitnessEval::new(none_scores()).with_veto(reason),
        };
    }

    // 3 anchored dims, scored independently per plan (no pairwise ranking —
    // no position bias, exact machinery reuse; spec §4.2 layer 3).
    let mut scores: Vec<(String, Option<u8>)> = Vec::with_capacity(PLAN_JUDGE_DIMS.len());
    for dim in PLAN_JUDGE_DIMS.iter() {
        let prompt = render_plan_judge(dim, raw, digest);
        *calls_used += 1;
        let score = match executor
            .execute(agent_id, &prompt, model, &format!("plan_judge_{}", dim.name))
            .await
        {
            Ok(output) => parse_fitness_response(&output).score,
            Err(e) => {
                tracing::debug!(
                    dimension = dim.name,
                    error = %e,
                    "plan tournament: judge spawn failed — score=None"
                );
                None
            }
        };
        scores.push((dim.name.to_string(), score));
    }

    PlanCandidate { plan: Some(plan), raw: raw.to_string(), eval: FitnessEval::new(scores) }
}

/// Run the plan tournament (spec §4 shape (b)).
///
/// `n_drafts = PLAN_TOURNAMENT_DRAFTS` for `PlanMode::Tournament`; `1` for
/// `PlanMode::SinglePlanner` (round-1 fallback shape).
///
/// Flow (machinery reuse, §4.1 — zero new selection mechanisms):
/// 1. `n_drafts` single-shot rubric-informed plan drafts (NOT the 2-step
///    denoise loop — refining losers pays for discarded candidates),
/// 2. veto-tier lints + 3 judge dims per surviving candidate,
/// 3. `sort_candidates_best_first` with `PLAN_WEIGHTS` + `is_valid_plan`
///    (the `(valid, total)` sort key — sort-only, take-best, no merging;
///    GraphMerger precedent),
/// 4. ≤`PLAN_WINNER_REDRAFT_CAP` bounded redrafts of the WINNER ONLY when it
///    fails validity, each re-evaluated.
///
/// Call envelope: 5 + 15 + 2 + 6 = 28 worst-case (≤ `PLAN_TOURNAMENT_MAX_CALLS`,
/// the C-N7 corrected 26–32 budget). All calls carry small contexts (plans +
/// digest).
///
/// Errors only when every draft call fails or no candidate exists — a failed
/// tournament must not kill the run; the engine degrades to no-plan.
pub async fn run_plan_tournament(
    synthesis: &SynthesisArtifact,
    executor: &Arc<dyn AgentExecutor>,
    agent_id_str: &str,
    model: &str,
    shape: NarrativeShape,
    n_drafts: usize,
) -> Result<PlanTournamentOutcome, TtdError> {
    use alzina_core::identity::AgentId;

    let agent_id = AgentId::new(agent_id_str);
    let digest = render_corpus_digest(synthesis);
    let draft_prompt = render_plan_draft(synthesis, shape);
    let mut calls_used: usize = 0;

    // Step 1: n single-shot drafts (sequential — plan calls are small and the
    // stage budget is ~3% of max_llm_calls; concurrency is not worth the
    // machinery here).
    let mut candidates: Vec<PlanCandidate> = Vec::with_capacity(n_drafts);
    for i in 0..n_drafts {
        calls_used += 1;
        match executor
            .execute(&agent_id, &draft_prompt, model, "plan_draft")
            .await
        {
            Ok(raw) => {
                let candidate = evaluate_plan_candidate(
                    &raw, synthesis, &digest, executor, &agent_id, model, &mut calls_used,
                )
                .await;
                candidates.push(candidate);
            }
            Err(e) => {
                tracing::warn!(draft = i, error = %e, "plan tournament: draft spawn failed");
            }
        }
    }
    if candidates.is_empty() {
        return Err(TtdError::NoCandidates);
    }

    // Step 2: sort-only selection over indices (machinery reuse).
    let indices: Vec<usize> = (0..candidates.len()).collect();
    let evals: Vec<FitnessEval> = candidates.iter().map(|c| c.eval.clone()).collect();
    let sorted = sort_candidates_best_first(&indices, &evals, PLAN_WEIGHTS, is_valid_plan);
    let winner_idx = sorted[0];

    let mut winner = candidates[winner_idx].clone();
    let runners_up: Vec<PlanCandidate> = sorted[1..]
        .iter()
        .map(|&i| candidates[i].clone())
        .collect();

    // Step 3: ≤2 bounded redrafts of the winner ONLY (post-selection).
    let mut redrafts = 0;
    while !is_valid_plan(&winner.eval) && redrafts < PLAN_WINNER_REDRAFT_CAP {
        redrafts += 1;
        let feedback = match &winner.eval.veto {
            Some(reason) => format!("HARD FAIL: {reason}"),
            None => crate::ttd::fitness::generate_feedback(&winner.eval, 3),
        };
        let redraft_prompt = render_plan_redraft(synthesis, shape, &winner.raw, &feedback);
        calls_used += 1;
        let raw = match executor
            .execute(&agent_id, &redraft_prompt, model, "plan_redraft")
            .await
        {
            Ok(raw) => raw,
            Err(e) => {
                tracing::warn!(redraft = redrafts, error = %e, "plan tournament: redraft failed");
                continue;
            }
        };
        let candidate = evaluate_plan_candidate(
            &raw, synthesis, &digest, executor, &agent_id, model, &mut calls_used,
        )
        .await;
        // Keep the better of (winner, redraft) by the same (valid, total) key.
        let pair = [winner.clone(), candidate];
        let pair_evals: Vec<FitnessEval> = pair.iter().map(|c| c.eval.clone()).collect();
        let order =
            sort_candidates_best_first(&[0usize, 1usize], &pair_evals, PLAN_WEIGHTS, is_valid_plan);
        winner = pair[order[0]].clone();
    }

    debug_assert!(
        calls_used <= PLAN_TOURNAMENT_MAX_CALLS,
        "plan tournament exceeded its call envelope: {calls_used} > {PLAN_TOURNAMENT_MAX_CALLS}"
    );

    let winner_valid = is_valid_plan(&winner.eval);
    let plan = winner
        .plan
        .clone()
        .ok_or_else(|| TtdError::ParseFailed("tournament winner has no parseable plan".into()))?;

    // Run metadata: winner + runners-up surface for the operator veto window
    // (spec §4.2: five artifacts of evidence instead of one) and for the C10
    // repair path (rerun with runner-up plan).
    tracing::info!(
        target: "ttd_plan",
        archetype = plan.archetype.as_str(),
        focal_question = %plan.focal_question,
        winner_valid,
        n_candidates = n_drafts,
        n_runners_up = runners_up.len(),
        calls_used,
        "ttd_plan: tournament winner selected"
    );
    for (i, r) in runners_up.iter().enumerate() {
        tracing::info!(
            target: "ttd_plan",
            rank = i + 2,
            archetype = r.plan.as_ref().map(|p| p.archetype.as_str()).unwrap_or("unparseable"),
            veto = r.eval.veto.as_deref().unwrap_or(""),
            raw = %r.raw,
            "ttd_plan: runner-up retained in run metadata"
        );
    }

    Ok(PlanTournamentOutcome {
        winner: plan,
        winner_raw: winner.raw,
        winner_valid,
        runners_up,
        calls_used,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttd::artifact::Claim;

    fn sample_plan_xml(with_sections: bool) -> String {
        let sections = if with_sections {
            "\n  <sections>\n    <section>\n      <heading>Where the field agrees</heading>\n      \
             <purpose>Establish the settled core the focal question presses against</purpose>\n      \
             <budget_words>600</budget_words>\n      <claim_ids>C1, C2</claim_ids>\n    </section>\n    \
             <section>\n      <heading>The contested mechanism</heading>\n      \
             <purpose>Argue the focal question through the live dispute</purpose>\n      \
             <budget_words>800</budget_words>\n      <claim_ids>C3, T1</claim_ids>\n    </section>\n  \
             </sections>"
        } else {
            ""
        };
        format!(
            "<plan>\n  <archetype>problem-lattice</archetype>\n  \
             <archetype_rationale>The corpus splits into interlocking disputes.</archetype_rationale>\n  \
             <focal_question>Does mechanism X drive outcome Y?</focal_question>\n  \
             <scope_exclusions>\n    <exclusion>clinical applications</exclusion>\n  </scope_exclusions>\n  \
             <term_registry>\n    <term>\n      <name>mechanism-x</name>\n      \
             <definition>the causal pathway under dispute</definition>\n      \
             <banned>the X effect|x-process</banned>\n    </term>\n  </term_registry>\n  \
             <planted_threads>\n    <thread>\n      <id>T1</id>\n      \
             <description>whether replication failures are methodological</description>\n      \
             <marker>the replication ledger</marker>\n      <setup>Where the field agrees</setup>\n      \
             <payoff>The contested mechanism</payoff>\n    </thread>\n  </planted_threads>{sections}\n</plan>"
        )
    }

    fn sample_synthesis(n_claims: usize) -> SynthesisArtifact {
        let mut s = SynthesisArtifact::new("s", "r", "q", "m", "v2/lit-review");
        for i in 0..n_claims {
            s.claims.push(Claim {
                text: format!(
                    "Claim {i}: mechanism X raises outcome Y by a replicated margin across \
                     three independent cohorts, with the effect persisting under controls."
                ),
                agreement_level: None,
                sources: vec![format!("arxiv:200{i}.0000{i}")],
                counterarguments: vec![],
                support_level: Some("converging".into()),
                evidence_grade: Some("moderate".into()),
                method: None,
                year: None,
                lineage: None,
                quotes: vec![],
                node_refs: vec![],
                citation: None,
            });
        }
        s.areas_of_disagreement =
            vec!["Whether the X effect survives publication-bias correction is disputed \
                  between the 2021 meta-analysis and the 2023 registered replication."
                .to_string()];
        s
    }

    /// Phase 1: plan XML round-trips through the parser — core fields AND
    /// D0-dep extension fields (sections, budgets, claim assignments).
    #[test]
    fn parse_review_plan_round_trip() {
        let plan = parse_review_plan(&sample_plan_xml(true)).expect("must parse");
        assert_eq!(plan.archetype, PlanArchetype::ProblemLattice);
        assert_eq!(plan.focal_question, "Does mechanism X drive outcome Y?");
        assert_eq!(plan.scope_exclusions, vec!["clinical applications".to_string()]);
        assert_eq!(plan.term_registry.len(), 1);
        assert_eq!(plan.term_registry[0].term, "mechanism-x");
        assert_eq!(
            plan.term_registry[0].banned_synonyms,
            vec!["the X effect".to_string(), "x-process".to_string()]
        );
        assert_eq!(plan.planted_threads.len(), 1);
        assert_eq!(plan.planted_threads[0].marker, "the replication ledger");
        assert_eq!(plan.sections.len(), 2);
        assert_eq!(plan.sections[0].budget_words, Some(600));
        assert_eq!(plan.sections[1].claim_ids, vec!["C3".to_string(), "T1".to_string()]);
    }

    /// Archetype singularity is enforced at the type level: hybrids and
    /// unknown labels fail PARSE, not just judging (spec §4.2 mechanical layer).
    #[test]
    fn archetype_parse_rejects_hybrids_and_unknowns() {
        assert!(PlanArchetype::parse("narrative-arc").is_ok());
        assert!(PlanArchetype::parse("  Problem Lattice ").is_ok(), "normalised form accepted");
        assert!(PlanArchetype::parse("thesis_and_convergence").is_ok());
        assert!(PlanArchetype::parse("narrative-arc + problem-lattice").is_err(), "hybrid");
        assert!(PlanArchetype::parse("chronological").is_err(), "out of vocabulary");
        assert!(PlanArchetype::parse("").is_err());
    }

    /// C-N1 (BINDING): the corpus digest carries FULL claim texts and full
    /// tension descriptions — not titles. The digest must contain the entire
    /// long claim sentence and the entire disagreement sentence verbatim.
    #[test]
    fn corpus_digest_carries_full_texts_not_titles() {
        let synthesis = sample_synthesis(3);
        let digest = render_corpus_digest(&synthesis);
        // Full claim text present verbatim (not truncated to a title).
        assert!(
            digest.contains(
                "mechanism X raises outcome Y by a replicated margin across \
                 three independent cohorts, with the effect persisting under controls."
            ),
            "digest must carry the full claim text: {digest}"
        );
        // Provenance labels present.
        assert!(digest.contains("[C1] (support: converging; evidence: moderate)"));
        // Full tension text present verbatim.
        assert!(
            digest.contains("survives publication-bias correction"),
            "digest must carry full tension descriptions"
        );
        assert!(digest.contains("[T1]"));
    }

    /// Claim-ID cross-check lint: dangling IDs, >2-section assignment, and
    /// under-coverage each veto; a clean plan passes; the lint is INACTIVE
    /// when sections are empty (D0-indep degenerate case).
    #[test]
    fn claim_id_cross_check_vetoes_and_passes() {
        let mut plan = parse_review_plan(&sample_plan_xml(true)).unwrap();

        // Clean: 3 claims, C1..C3 covered (3/3 ≥ 0.6), 1 tension, no over-assignment.
        assert_eq!(plan_lint_claim_id_cross_check(&plan, 3, 1), None);

        // Dangling claim ID.
        plan.sections[0].claim_ids.push("C9".into());
        let veto = plan_lint_claim_id_cross_check(&plan, 3, 1);
        assert!(veto.is_some() && veto.unwrap().contains("C9"), "dangling ID must veto");
        plan.sections[0].claim_ids.pop();

        // Dangling tension ID.
        plan.sections[0].claim_ids.push("T5".into());
        assert!(plan_lint_claim_id_cross_check(&plan, 3, 1).is_some());
        plan.sections[0].claim_ids.pop();

        // Over-assignment: same ID in 3 sections (cap is 2).
        plan.sections.push(PlanSection {
            heading: "Extra A".into(),
            purpose: "p".into(),
            budget_words: None,
            claim_ids: vec!["C1".into()],
        });
        plan.sections.push(PlanSection {
            heading: "Extra B".into(),
            purpose: "p".into(),
            budget_words: None,
            claim_ids: vec!["C1".into()],
        });
        let veto = plan_lint_claim_id_cross_check(&plan, 3, 1);
        assert!(
            veto.is_some() && veto.unwrap().contains("assign-everything"),
            "ID in 3 sections must trip the over-assignment guard"
        );
        plan.sections.truncate(2);

        // Coverage floor: 10 claims, only C1-C3 assigned → 0.3 < 0.6.
        let veto = plan_lint_claim_id_cross_check(&plan, 10, 1);
        assert!(
            veto.is_some() && veto.unwrap().contains("coverage"),
            "under-coverage must veto"
        );

        // No skeleton → lint inactive (D0-indep degenerate case).
        plan.sections.clear();
        assert_eq!(plan_lint_claim_id_cross_check(&plan, 10, 1), None);
    }

    /// F5: a non-ASCII first char in a claim ID (e.g. an LLM emitting the
    /// Cyrillic "С1" lookalike) must veto, not panic. `split_at(1)` would
    /// split mid-codepoint; `strip_prefix(char)` matches by char and yields a
    /// clean "does not exist" veto.
    #[test]
    fn claim_id_cross_check_non_ascii_id_vetoes_without_panic() {
        let mut plan = parse_review_plan(&sample_plan_xml(true)).unwrap();
        plan.sections[0].claim_ids.push("\u{0421}1".into()); // Cyrillic Es + 1
        let veto = plan_lint_claim_id_cross_check(&plan, 3, 1);
        assert!(
            veto.is_some() && veto.unwrap().contains("does not exist"),
            "non-ASCII ID must veto as nonexistent, not panic"
        );
    }

    /// Plan judge dims: exactly 3, corpus_fit first (validity anchor), every
    /// anchor_1 names its gaming path (spec §4.2 layer 3).
    #[test]
    fn plan_judge_dims_name_the_gaming_paths() {
        assert_eq!(PLAN_JUDGE_DIMS.len(), 3);
        assert_eq!(PLAN_JUDGE_DIMS[0].name, "corpus_fit", "validity anchor first");
        assert!(
            PLAN_JUDGE_DIMS[0].anchor_1.contains("not derivable"),
            "corpus_fit anchor_1 must name the generic-sections gaming path"
        );
        assert!(
            PLAN_JUDGE_DIMS[1].anchor_1.contains("hybrid with no primary"),
            "archetype_singularity anchor_1 must name the hybrid failure"
        );
        assert!(
            PLAN_JUDGE_DIMS[2].anchor_1.contains("inert")
                || PLAN_JUDGE_DIMS[2].anchor_1.contains("no section"),
            "focal_question_grip anchor_1 must name the stated-but-inert failure"
        );
    }

    /// R7 / spec §9 row 1: the plan-judge prompt holds BOTH halves of the
    /// plan ⟷ corpus relation — the full plan text and the full-text digest.
    #[test]
    fn plan_judge_prompt_holds_both_halves() {
        let synthesis = sample_synthesis(2);
        let digest = render_corpus_digest(&synthesis);
        let plan_raw = sample_plan_xml(true);
        let prompt = render_plan_judge(&PLAN_JUDGE_DIMS[0], &plan_raw, &digest);
        assert!(prompt.contains("Does mechanism X drive outcome Y?"), "plan half in context");
        assert!(
            prompt.contains("three independent cohorts"),
            "full-text corpus half in context (C-N1)"
        );
        assert!(prompt.contains("corpus_fit"));
        assert!(prompt.contains("- 1: "), "anchors rendered");
    }

    /// is_valid_plan mirrors is_valid_v2: veto hard-fails; corpus_fit < 4
    /// fails; corpus_fit ≥ 4 with no veto passes (spec §4.1 validity row).
    #[test]
    fn is_valid_plan_gates_on_veto_and_corpus_fit() {
        let mk = |score: Option<u8>| {
            FitnessEval::new(vec![
                ("corpus_fit".into(), score),
                ("archetype_singularity".into(), Some(5)),
                ("focal_question_grip".into(), Some(5)),
            ])
        };
        assert!(is_valid_plan(&mk(Some(4))));
        assert!(is_valid_plan(&mk(Some(5))));
        assert!(!is_valid_plan(&mk(Some(3))), "corpus_fit 3 must fail validity");
        assert!(!is_valid_plan(&mk(None)), "unscored corpus_fit must fail validity");
        assert!(!is_valid_plan(&mk(Some(5)).with_veto("lint failed")), "veto hard-fails");
    }

    /// PLAN_WEIGHTS sums to 1.0 (±0.001) — same contract as every stage table.
    #[test]
    fn plan_weights_sum_to_one() {
        let s: f32 = PLAN_WEIGHTS.iter().map(|(_, w)| w).sum();
        assert!((s - 1.0).abs() < 0.001, "PLAN_WEIGHTS sum {s} ≠ 1.0");
        assert_eq!(PLAN_WEIGHTS[0].0, "corpus_fit");
    }

    /// TermDriftLint: banned synonyms are flagged; the canonical term is not.
    #[test]
    fn term_drift_scan_flags_banned_synonyms_only() {
        let plan = parse_review_plan(&sample_plan_xml(false)).unwrap();
        let clean = "The mechanism-x pathway is contested across cohorts.";
        assert!(term_drift_scan(clean, &plan).is_empty(), "canonical term must not flag");

        let drifted = "Recent work on the X effect suggests the x-process is robust.";
        let findings = term_drift_scan(drifted, &plan);
        assert_eq!(findings.len(), 2, "both banned synonyms must flag: {findings:?}");
        assert!(findings[0].contains("mechanism-x"), "finding names the canonical term");
    }

    /// ProportionalityLint: missing plan sections and out-of-budget ratios
    /// are reported; in-budget sections are silent.
    #[test]
    fn section_budget_scan_ratios_and_missing_sections() {
        let plan = parse_review_plan(&sample_plan_xml(true)).unwrap();

        // In-budget: ~600 words for budget 600 (ratio 1.0) — silent;
        // second plan section missing — reported.
        let body = (0..600).map(|_| "word").collect::<Vec<_>>().join(" ");
        let doc = format!("## Where the field agrees\n{body}\n");
        let findings = section_budget_scan(&doc, &plan);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("The contested mechanism"), "missing section reported");

        // Out of budget: 100 words vs budget 600 → ratio 0.17 < 0.5.
        let thin = (0..100).map(|_| "word").collect::<Vec<_>>().join(" ");
        let doc = format!(
            "## Where the field agrees\n{thin}\n## The contested mechanism\n{body}\n"
        );
        let findings = section_budget_scan(&doc, &plan);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("ratio"), "ratio breach reported: {findings:?}");
    }

    /// ThreadCashInLint: a marker appearing once (or paid off outside its
    /// declared payoff section) is reported; setup+payoff in place is silent.
    #[test]
    fn thread_callback_scan_verifies_setup_and_payoff() {
        let plan = parse_review_plan(&sample_plan_xml(true)).unwrap();

        // Marker appears once → finding.
        let doc = "## Where the field agrees\nWe open the replication ledger here.\n\
                   ## The contested mechanism\nNo callback.\n";
        let findings = thread_callback_scan(doc, &plan);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("the replication ledger"));

        // Marker twice but never inside the payoff section → finding.
        let doc = "## Where the field agrees\nThe replication ledger opens. \
                   The replication ledger again.\n## The contested mechanism\nNothing.\n";
        let findings = thread_callback_scan(doc, &plan);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("never cashed in"));

        // Setup + payoff in their sections → silent.
        let doc = "## Where the field agrees\nWe open the replication ledger.\n\
                   ## The contested mechanism\nClosing the replication ledger: resolved.\n";
        assert!(thread_callback_scan(doc, &plan).is_empty());
    }

    /// Plan block renders every element downstream prompts depend on,
    /// including the exact skeleton headings.
    #[test]
    fn plan_block_renders_all_elements() {
        let plan = parse_review_plan(&sample_plan_xml(true)).unwrap();
        let block = render_plan_block(&plan);
        assert!(block.contains("problem-lattice"));
        assert!(block.contains("Does mechanism X drive outcome Y?"));
        assert!(block.contains("\"mechanism-x\""));
        assert!(block.contains("never: the X effect, x-process"));
        assert!(block.contains("the replication ledger"));
        assert!(block.contains("## Where the field agrees"));
        assert!(block.contains("(~600 words)"));
        assert!(block.contains("[claims: C1, C2]"));
    }

    /// The conformance judge verifies draft-against-PLAN, not taste: prompt
    /// carries the plan block, the draft, and the anti-taste instruction.
    #[test]
    fn plan_conformance_judge_is_against_plan_not_taste() {
        let plan = parse_review_plan(&sample_plan_xml(true)).unwrap();
        let prompt = render_plan_conformance_judge(&plan, "Draft text here.");
        assert!(prompt.contains("plan_conformance"));
        assert!(prompt.contains("not your taste"), "anti-taste instruction present");
        assert!(prompt.contains("Does mechanism X drive outcome Y?"), "plan half in context");
        assert!(prompt.contains("Draft text here."), "draft half in context");
        assert!(prompt.contains("- 1: "), "anchored, not free-form");
    }

    /// Draft prompt is rubric-informed and shape-partitioned: archetypes +
    /// element categories always present; skeleton elements only under
    /// SectionedLongForm (D0 partition, spec §3).
    #[test]
    fn plan_draft_prompt_rubric_and_d0_partition() {
        let synthesis = sample_synthesis(2);
        let long = render_plan_draft(&synthesis, NarrativeShape::SectionedLongForm);
        assert!(long.contains("narrative-arc"));
        assert!(long.contains("problem-lattice"));
        assert!(long.contains("mece-taxonomy"));
        assert!(long.contains("thesis-and-convergence"));
        assert!(long.contains("<sections>"), "skeleton requested under long-form");
        assert!(long.contains("three independent cohorts"), "full-text digest (C-N1)");
        // Bookend structure (Sam, 2026-06-16): intro first, open-questions
        // second-to-last, conclusion last; no generic "Problem N" headings.
        assert!(long.contains("MUST be an Introduction"), "intro mandated under long-form");
        assert!(
            long.contains("Open questions and future directions"),
            "open-questions section mandated under long-form"
        );
        assert!(long.contains("MUST be a Conclusion"), "conclusion mandated under long-form");
        assert!(
            long.contains("Do NOT use generic numbered labels"),
            "descriptive-heading steer present (kills 'Problem N')"
        );

        let concise = render_plan_draft(&synthesis, NarrativeShape::Concise);
        assert!(!concise.contains("<sections>"), "no skeleton under Concise (D0-dep)");
        assert!(concise.contains("<term_registry>"), "core fields stay (D0-indep)");
        assert!(concise.contains("<planted_threads>"));
        // Bookend rules are skeleton-only — Concise stays byte-identical.
        assert!(!concise.contains("MUST be an Introduction"), "no bookend rules under Concise");
        assert!(
            !concise.contains("Open questions and future directions"),
            "no open-questions mandate under Concise"
        );
    }

    // ── Tournament integration (mock executor) ────────────────────────────────

    use alzina_core::error::AlzinaError;
    use alzina_core::identity::AgentId;
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Mock executor: plan drafts cycle through scripted plans; judges score
    /// from a scripted table keyed by marker strings in the plan text.
    struct ScriptedExecutor {
        calls: Mutex<Vec<String>>,
        /// (substring of prompt → response) for plan_draft calls, in order.
        drafts: Mutex<Vec<String>>,
        /// corpus_fit score keyed by a marker substring of the judged plan.
        corpus_fit: Vec<(&'static str, &'static str)>,
    }

    #[async_trait]
    impl AgentExecutor for ScriptedExecutor {
        async fn execute(
            &self,
            _agent_id: &AgentId,
            prompt: &str,
            _model: &str,
            task: &str,
        ) -> Result<String, AlzinaError> {
            self.calls.lock().unwrap().push(task.to_string());
            match task {
                "plan_draft" | "plan_redraft" => {
                    let mut drafts = self.drafts.lock().unwrap();
                    if drafts.is_empty() {
                        Ok("no plan here".to_string())
                    } else {
                        Ok(drafts.remove(0))
                    }
                }
                t if t.starts_with("plan_judge_corpus_fit") => {
                    for (marker, score) in &self.corpus_fit {
                        if prompt.contains(marker) {
                            return Ok(format!(
                                "<fitness_evaluation><score>{score}</score>\
                                 <rationale>scripted</rationale></fitness_evaluation>"
                            ));
                        }
                    }
                    Ok("<fitness_evaluation><score>3</score><rationale>default</rationale>\
                        </fitness_evaluation>"
                        .to_string())
                }
                _ => Ok("<fitness_evaluation><score>4</score><rationale>scripted</rationale>\
                         </fitness_evaluation>"
                    .to_string()),
            }
        }
    }

    fn plan_with_question(question: &str) -> String {
        sample_plan_xml(true).replace("Does mechanism X drive outcome Y?", question)
    }

    /// Tournament end-to-end: 5 drafts, judges score one plan above the rest,
    /// sort-only take-best selects it, no redrafts fire (winner valid), and
    /// the call count lands inside the 26–32 envelope (C-N7).
    #[tokio::test]
    async fn tournament_takes_best_within_call_envelope() {
        let executor: Arc<dyn AgentExecutor> = Arc::new(ScriptedExecutor {
            calls: Mutex::new(vec![]),
            drafts: Mutex::new(vec![
                plan_with_question("Q-mediocre-one?"),
                plan_with_question("Q-the-strong-plan?"),
                plan_with_question("Q-mediocre-two?"),
                "completely unparseable output".to_string(),
                plan_with_question("Q-mediocre-three?"),
            ]),
            corpus_fit: vec![("Q-the-strong-plan?", "5")],
        });
        let synthesis = sample_synthesis(3);

        let outcome = run_plan_tournament(
            &synthesis,
            &executor,
            "galdr-test",
            "test-model",
            NarrativeShape::SectionedLongForm,
            PLAN_TOURNAMENT_DRAFTS,
        )
        .await
        .expect("tournament must succeed");

        assert_eq!(outcome.winner.focal_question, "Q-the-strong-plan?", "take-best");
        assert!(outcome.winner_valid);
        assert_eq!(outcome.runners_up.len(), 4, "runners-up retained for run metadata");
        // Unparseable draft skipped judge calls: 5 drafts + 4×3 judges = 17.
        assert_eq!(outcome.calls_used, 17);
        assert!(outcome.calls_used <= PLAN_TOURNAMENT_MAX_CALLS, "C-N7 envelope");
        // The vetoed (unparseable) candidate sorts last.
        let last = outcome.runners_up.last().unwrap();
        assert!(last.eval.veto.as_deref().unwrap_or("").contains("unparseable"));
    }

    /// When every draft scores below the validity floor, the winner is
    /// redrafted (winner ONLY, ≤2 attempts) and a valid redraft replaces it.
    #[tokio::test]
    async fn tournament_redrafts_invalid_winner_bounded() {
        let executor: Arc<dyn AgentExecutor> = Arc::new(ScriptedExecutor {
            calls: Mutex::new(vec![]),
            drafts: Mutex::new(vec![
                // 5 initial drafts — all score corpus_fit 3 (invalid).
                plan_with_question("Q-weak-a?"),
                plan_with_question("Q-weak-b?"),
                plan_with_question("Q-weak-c?"),
                plan_with_question("Q-weak-d?"),
                plan_with_question("Q-weak-e?"),
                // Redraft 1 — strong.
                plan_with_question("Q-redrafted-strong?"),
            ]),
            corpus_fit: vec![("Q-redrafted-strong?", "5")],
        });
        let synthesis = sample_synthesis(3);

        let outcome = run_plan_tournament(
            &synthesis,
            &executor,
            "galdr-test",
            "test-model",
            NarrativeShape::SectionedLongForm,
            PLAN_TOURNAMENT_DRAFTS,
        )
        .await
        .expect("tournament must succeed");

        assert_eq!(outcome.winner.focal_question, "Q-redrafted-strong?");
        assert!(outcome.winner_valid);
        // 5 drafts + 15 judges + 1 redraft + 3 re-eval judges = 24.
        assert_eq!(outcome.calls_used, 24);
        assert!(outcome.calls_used <= PLAN_TOURNAMENT_MAX_CALLS);
        // Redraft count is asserted via the calls_used arithmetic above.
    }

    /// SinglePlanner fallback shape: n_drafts=1 still works (round-1 config
    /// fallback in the reversal chain).
    #[tokio::test]
    async fn single_planner_fallback_one_draft() {
        let executor: Arc<dyn AgentExecutor> = Arc::new(ScriptedExecutor {
            calls: Mutex::new(vec![]),
            drafts: Mutex::new(vec![plan_with_question("Q-solo?")]),
            corpus_fit: vec![("Q-solo?", "4")],
        });
        let synthesis = sample_synthesis(3);
        let outcome = run_plan_tournament(
            &synthesis,
            &executor,
            "galdr-test",
            "test-model",
            NarrativeShape::Concise,
            1,
        )
        .await
        .expect("single planner must succeed");
        assert_eq!(outcome.winner.focal_question, "Q-solo?");
        assert!(outcome.runners_up.is_empty());
        assert_eq!(outcome.calls_used, 4, "1 draft + 3 judges");
    }

    /// PlanMode default is Disabled — byte-stability of every existing path.
    #[test]
    fn plan_mode_default_is_disabled() {
        assert_eq!(PlanMode::default(), PlanMode::Disabled);
    }
}
