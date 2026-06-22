//! Fitness scoring and candidate selection for TTD stages.
//!
//! All functions are deterministic (zero LLM). This module is the Rust port of
//! `consensus/src/consensus/diffusion/fitness.py:354-428`.
//!
//! ## Key contracts
//!
//! - `FitnessEval` carries `Option<u8>` scores: `None` = parse failure, NOT 0.
//! - `weighted_sum` normalises over `scored_weight` (sum of weights for non-None
//!   dims only) — this is the None-redistribution that prevents score deflation.
//! - `sort_candidates_best_first` uses a compound sort key `(valid, total)` — valid
//!   candidates sort before invalid, then by score descending.
//! - Validity gate: `is_valid_graph` requires `groundedness ≥ 4`;
//!   `is_valid_synthesis` requires `faithfulness ≥ 4` (affects sort only, never
//!   loop termination — locked CONTEXT decision).
//! - `generate_feedback` ports `_generate_feedback` (fitness.py:213-311) preserving
//!   the exact heading structure the denoiser parses.
//!
//! ## Fitness parse ladder
//!
//! Reproduces `fitness.py:597-723` (four tiers):
//! 1. Try XML parse → extract `<score>`, `<rationale>`, `<suggestions>`.
//! 2. `_parse_score_text`: try integer parse (CLAMP to `[1,5]`), then regex
//!    `\b([1-5])\b`.
//! 3. On XML parse error: regex fallback (`_extract_via_regex`) —
//!    (a) full-text `<score>\s*(\d+)\s*</score>` with clamp, then (b) bare
//!    `\b([1-5])\b` bounded to the first 200 chars (WR-02).
//! 4. True failure (no score, including empty/whitespace): `None`.
//!
//! ## Empty vs absent (Pitfall 3 / WR-03)
//!
//! Consensus reserves the `score=3` sentinel for the literal "no response
//! object" case only (`fitness.py:731-732`: `if text is None`). An empty OR
//! whitespace-only STRING flows to `None` via the strip→parse-fail path
//! (`fitness.py:678`). This parser takes a `&str`, which always carries text,
//! so empty/whitespace → `None` here. The `text is None` sentinel belongs to
//! the caller layer (the executor knows when no response came back).
//! `None` redistributes weight in `weighted_sum`; a `3` would inflate a
//! silently-failed judge into a lukewarm pass — the less-safe direction.

use std::collections::HashMap;

use crate::ttd::artifact::{ArgumentationGraph, SynthesisArtifact};
use crate::ttd::mod_types::TtdError;
#[cfg(test)]
use crate::ttd::weights::{GRAPH_WEIGHTS, SYNTHESIS_WEIGHTS};

// ── Parse ladder result ───────────────────────────────────────────────────────

/// Result of parsing a fitness judge LLM response.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedFitnessScore {
    /// Parsed score, `1..=5`, or `None` on parse failure (not empty response).
    pub score: Option<u8>,
    /// Rationale text (empty if not parseable).
    pub rationale: String,
}

impl ParsedFitnessScore {
    fn scored(score: u8, rationale: impl Into<String>) -> Self {
        Self { score: Some(score), rationale: rationale.into() }
    }

    fn none() -> Self {
        Self { score: None, rationale: String::new() }
    }
}

/// Parse a fitness judge LLM response through the four-tier ladder.
///
/// Reproduces `fitness.py:597-723`.
///
/// ## Tier order
///
/// 1. Empty / whitespace-only response → `None` (WR-03; abstain, do not inflate)
/// 2. XML parse → extract `<score>` → `_parse_score_text` → `Some(score)`
/// 3. XML parse error → regex fallback: full-text `<score>(\d+)</score>` clamp,
///    then bare `\b([1-5])\b` bounded to the first 200 chars (WR-02)
/// 4. No score found → `None`
///
/// ## Distinction
///
/// Empty/whitespace → `None` (matches consensus strip→parse-fail). The
/// `score=3` sentinel is reserved for the literal absent-response-object case
/// and lives at the caller layer. `None` redistributes weight in `weighted_sum`.
pub fn parse_fitness_response(llm_response: &str) -> ParsedFitnessScore {
    let trimmed = llm_response.trim();

    // Tier 1: Empty / whitespace-only response → None (abstain).
    //
    // WR-03: match consensus. The `score=3` sentinel is reserved for the literal
    // "no response object" case (`fitness.py:731-732`: `if text is None`). An
    // empty OR whitespace-only STRING flows through `text.strip()` → ET parse
    // fail → regex fail → `None` (fitness.py:678 `if not text: return None`).
    // None redistributes weight in `weighted_sum`; a sentinel `3` would inflate
    // a silently-failed judge into a lukewarm pass — the less-safe direction.
    // The `3` sentinel for a genuinely absent response object is the caller
    // layer's responsibility (the executor knows when no response came back);
    // it is NOT reachable from a `&str`, which always carries some text.
    if trimmed.is_empty() {
        return ParsedFitnessScore::none();
    }

    // Tier 2: Try XML parse.
    if let Some(scored) = try_xml_parse(trimmed) {
        return scored;
    }

    // Tier 3: XML parse failed → regex fallback (`_extract_via_regex`,
    // fitness.py:692-723). Two ordered sub-tiers:
    //   (a) full-text `<score>\s*(\d+)\s*</score>` with clamp (fitness.py:695-697)
    //   (b) bare `\b([1-5])\b` bounded to the FIRST 200 chars (fitness.py:700)
    if let Some(score) = extract_score_from_garbled_tag(trimmed) {
        return ParsedFitnessScore::scored(score, "");
    }
    // WR-02: the bare-digit fallback must NOT scan the whole response — a stray
    // 1-5 digit deep in the rationale must not become the score. Bound to the
    // first 200 chars exactly as `raw[:200]` (fitness.py:700). Consensus indexes
    // a Python str by CHARACTER, so we take 200 chars (not bytes) to stay
    // UTF-8-boundary-safe.
    let window: String = trimmed.chars().take(200).collect();
    if let Some(score) = extract_score_via_regex(&window) {
        return ParsedFitnessScore::scored(score, "");
    }

    // Tier 4: No score found → None.
    ParsedFitnessScore::none()
}

/// Try to parse the LLM response as XML and extract `<score>`, `<rationale>`.
///
/// Mirrors `ET.fromstring` → `_parse_score_text` in fitness.py:597-630.
/// Returns `None` on XML parse error (falls through to regex).
fn try_xml_parse(text: &str) -> Option<ParsedFitnessScore> {
    // Find the XML block — LLM may wrap it in a code fence or preamble.
    // We look for the first occurrence of `<fitness_evaluation` or just
    // try parsing the whole response as XML.
    //
    // Simplified port: look for a <score> tag using string search, then
    // validate with a mini XML parse. Full quick-xml parse of the whole
    // response is done first; if that fails we fall through.

    // Attempt full XML parse of the response (may have preamble — try to find the block).
    let xml_block = extract_xml_block(text, "fitness_evaluation")
        .or_else(|| extract_xml_block(text, "score"))
        .unwrap_or(text);

    // Parse score from the XML block.
    let score = extract_tag_text(xml_block, "score")
        .and_then(|s| parse_score_text(s.trim()))?;

    let rationale = extract_tag_text(xml_block, "rationale")
        .unwrap_or_default()
        .to_string();

    Some(ParsedFitnessScore::scored(score, rationale))
}

/// Extract the text content of a named XML tag.
///
/// Simple string search — does not handle nested tags of the same name.
/// Sufficient for the flat structure of fitness judge XML output.
fn extract_tag_text<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)?;
    Some(&xml[start..start + end])
}

/// Extract an outer XML block by looking for `<tag_name>...</tag_name>`.
fn extract_xml_block<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = text.find(&open)?;
    let end_of_close = text[start..].find(&close)? + close.len();
    Some(&text[start..start + end_of_close])
}

/// Parse a score text string: try integer (CLAMPED to `[1,5]`), then regex
/// `\b([1-5])\b`.
///
/// Mirrors `_parse_score_text` from fitness.py:672-690. CR-03: an integer parse
/// CLAMPS via `max(1, min(5, val))` — so `<score>6</score>` → `5` and
/// `<score>0</score>` → `1`. `None` is reserved for a truly unparseable
/// (non-integer) text only, not for an out-of-range integer.
fn parse_score_text(text: &str) -> Option<u8> {
    // Try direct integer parse, then clamp to [1,5] (matches consensus int()).
    if let Ok(n) = text.trim().parse::<i64>() {
        return Some(n.clamp(1, 5) as u8);
    }
    // Not an integer at all — regex fallback: find a digit 1-5 with word boundaries.
    extract_score_via_regex(text)
}

/// Regex fallback sub-tier (a): find a `<score>\s*(\d+)\s*</score>` tag anywhere
/// in garbled text and CLAMP the integer to `[1,5]`.
///
/// Mirrors the first branch of `_extract_via_regex`
/// (`fitness.py:695-697`: `re.search(r'<score>\s*(\d+)\s*</score>', raw)` then
/// `max(1, min(5, int(...)))`). This catches a recognisable score tag even when
/// the surrounding XML is malformed enough that `try_xml_parse` returned `None`
/// (e.g. unclosed sibling tags). The whole-integer parse here is intentionally
/// NOT word-boundary bounded — a complete `<score>N</score>` tag is unambiguous.
fn extract_score_from_garbled_tag(text: &str) -> Option<u8> {
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find("<score>") {
        let after_open = search_from + rel + "<score>".len();
        if let Some(close_rel) = text[after_open..].find("</score>") {
            let inner = text[after_open..after_open + close_rel].trim();
            // Accept only a pure run of ASCII digits (matches `(\d+)`).
            if !inner.is_empty() && inner.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(n) = inner.parse::<i64>() {
                    return Some(n.clamp(1, 5) as u8);
                }
            }
            search_from = after_open + close_rel + "</score>".len();
        } else {
            break;
        }
    }
    None
}

/// Regex fallback sub-tier (b): search for `\b([1-5])\b` in the text.
///
/// Mirrors the bare-digit branch of `_extract_via_regex` (fitness.py:700).
/// Returns the first match as a score. WR-02: callers MUST bound the input to
/// the first 200 chars before calling this — a stray 1-5 digit deep in the
/// response must not be mistaken for the score.
fn extract_score_via_regex(text: &str) -> Option<u8> {
    // Simple manual implementation — no regex crate needed.
    // Find any digit 1-5 surrounded by word boundaries (non-word chars or string edges).
    for (i, ch) in text.char_indices() {
        if matches!(ch, '1'..='5') {
            let before_ok = i == 0 || !text[..i].ends_with(|c: char| c.is_alphanumeric() || c == '_');
            let after_pos = i + ch.len_utf8();
            let after_ok = after_pos >= text.len()
                || !text[after_pos..].starts_with(|c: char| c.is_alphanumeric() || c == '_');
            if before_ok && after_ok {
                return Some(ch as u8 - b'0');
            }
        }
    }
    None
}

// ── FitnessEval ───────────────────────────────────────────────────────────────

/// Scored output for one candidate from the fitness judge spawns.
///
/// Each dimension score is `Option<u8>`:
/// - `Some(1..=5)` — parsed successfully from the judge's XML output
/// - `None` — the judge returned unparseable XML or an empty response
///
/// `None` is NOT a score of 0. The `weighted_sum` function normalises over the
/// scored dimensions only (None-redistribution, fitness.py:400-418).
///
/// `veto`: when `Some`, the candidate is hard-failed regardless of scores.
/// On the v2 path, set by `traceability_veto_synthesis` / `traceability_veto_graph`.
/// Always `None` on the v1 path — all existing call sites unaffected.
#[derive(Debug, Clone)]
pub struct FitnessEval {
    /// Ordered (dimension_name, score_or_None) pairs.
    /// Order must match the weight table for the current stage.
    pub scores: Vec<(String, Option<u8>)>,
    /// Traceability veto (v2 only). When `Some(reason)`, the candidate is
    /// hard-failed: `is_valid_v2` returns false, `sort_candidates_best_first`
    /// sorts it last, and `generate_feedback` prepends a HARD FAIL section.
    /// Always `None` from `new(scores)` — v1 call sites unaffected.
    pub veto: Option<String>,
}

impl FitnessEval {
    /// Convenience constructor. `veto` defaults to `None` — v1 call sites unchanged.
    pub fn new(scores: Vec<(String, Option<u8>)>) -> Self {
        Self { scores, veto: None }
    }

    /// Attach a traceability veto reason. Consuming builder — call after `new`.
    pub fn with_veto(mut self, reason: impl Into<String>) -> Self {
        self.veto = Some(reason.into());
        self
    }

    /// True if all score dimensions are None (total parse failure).
    pub fn all_none(&self) -> bool {
        self.scores.iter().all(|(_, s)| s.is_none())
    }

    /// Score pairs as `(&str, Option<u8>)` slices for `weighted_sum`.
    pub fn score_pairs(&self) -> Vec<(&str, Option<u8>)> {
        self.scores.iter().map(|(k, v)| (k.as_str(), *v)).collect()
    }

    /// Named score lookup.
    pub fn score(&self, dim: &str) -> Option<u8> {
        self.scores
            .iter()
            .find(|(k, _)| k == dim)
            .and_then(|(_, v)| *v)
    }
}

// ── Validity gates ────────────────────────────────────────────────────────────

/// Graph stage validity: `groundedness ≥ 4` (fitness.py:89-91).
///
/// Validity affects sort order only — invalid candidates sort LAST.
/// The loop is NEVER terminated based on validity (CONTEXT locked decision).
pub fn is_valid_graph(eval: &FitnessEval) -> bool {
    eval.score("groundedness").map_or(false, |s| s >= 4)
}

/// Synthesis / narrative stage validity: `faithfulness ≥ 4` (fitness.py:161-163).
pub fn is_valid_synthesis(eval: &FitnessEval) -> bool {
    eval.score("faithfulness").map_or(false, |s| s >= 4)
}

/// v2 validity gate: no veto present AND faithfulness ≥ 4.
///
/// Mirrors the v1 gate semantics (validity affects sort only, never loop
/// termination — CONTEXT locked decision). Faithfulness is the v2 grounding
/// analog of the v1 `is_valid_synthesis` gate.
///
/// Both conditions must hold:
/// - `eval.veto.is_none()` — traceability VETO hard-fails; scores are irrelevant
/// - `eval.score("faithfulness") >= 4` — faithfulness is the anchor dim on the v2 path
pub fn is_valid_v2(eval: &FitnessEval) -> bool {
    eval.veto.is_none() && eval.score("faithfulness").map_or(false, |s| s >= 4)
}

/// Traceability veto for `SynthesisArtifact` (F13 — probe-18).
///
/// Deterministic check on artifact structure — NOT an LLM judgement.
/// Returns `Some(reason)` when any claim has empty `sources` (or sources that
/// are ALL invalid by the F13 allowlist), or when there are zero claims.
/// Returns `None` when all claims carry at least one valid source.
///
/// A source is valid iff `is_valid_source_id` passes: arxiv:/s2:-prefixed (shape
/// lane) or an exact panel member. This supersedes Fix B's `is_candidate_label`
/// blacklist — probe-18 proved haiku evades pattern-blacklisting by mutating the
/// label (s1_candidate1 .. s5_candidate5).
///
/// A claim with ONE valid source is not vetoed even if it also carries invalid ids
/// (the strip in post_process removes those before emit; this veto guards
/// selection-time, matching the existing semantics).
///
/// The reason names the first offending claim and the offending ids; it steers
/// repair toward valid shapes ("sources must be arxiv:/s2: paper ids") — this text
/// feeds the HARD FAIL section the denoiser reads.
pub fn traceability_veto_synthesis(
    artifact: &SynthesisArtifact,
    panel_ids: &std::collections::HashSet<String>,
) -> Option<String> {
    use crate::ttd::term_sheet::is_valid_source_id;

    if artifact.claims.is_empty() {
        return Some(
            "Traceability veto: synthesis has zero claims — \
             every claim must cite at least one source paper ID."
                .to_string(),
        );
    }
    for claim in &artifact.claims {
        if claim.sources.is_empty() {
            let preview: String = claim.text.chars().take(80).collect();
            return Some(format!(
                "Traceability veto: claim has no sources — \"{preview}\"… \
                 Every claim must cite at least one source paper ID."
            ));
        }
        // F13: a claim whose sources are ALL invalid (neither arxiv:/s2:-shaped
        // nor a panel member) has no real provenance — treat as unsourced.
        // A claim with at least one valid source is not vetoed (strip handles
        // the invalid ids at emit time; the veto guards selection-time).
        let valid_sources: Vec<&str> = claim
            .sources
            .iter()
            .map(|s| s.as_str())
            .filter(|s| is_valid_source_id(s, panel_ids))
            .collect();
        if valid_sources.is_empty() {
            let invalid_labels: String = claim.sources.join(", ");
            let preview: String = claim.text.chars().take(80).collect();
            return Some(format!(
                "Traceability veto: claim sources are all invalid ids \
                 ({invalid_labels}) — sources must be arxiv:… or s2:… paper ids \
                 of the provided papers (claim: \"{preview}\"…)."
            ));
        }
    }
    None
}

/// Traceability veto for `ArgumentationGraph` (F13 — probe-18).
///
/// Deterministic check on artifact structure — NOT an LLM judgement.
/// Returns `Some(reason)` when any node has an empty, whitespace-only, or
/// invalid `expert_id` (F13: an id not in the allowlist is not a real paper id),
/// or when there are zero nodes. Returns `None` when all nodes carry a valid
/// non-empty `expert_id`.
///
/// Validity is `is_valid_source_id(id, panel_ids)`: arxiv:/s2:-prefixed or
/// panel member. This supersedes Fix B's `is_candidate_label` blacklist —
/// probe-18 mutated labels (s2_candidate2 etc.) are invalid by the allowlist
/// without needing a pattern match.
pub fn traceability_veto_graph(
    graph: &ArgumentationGraph,
    panel_ids: &std::collections::HashSet<String>,
) -> Option<String> {
    use crate::ttd::term_sheet::is_valid_source_id;

    if graph.nodes.is_empty() {
        return Some(
            "Traceability veto: argumentation graph has zero nodes — \
             every node must have a non-empty expert_id."
                .to_string(),
        );
    }
    for node in &graph.nodes {
        if node.expert_id.trim().is_empty() {
            let preview: String = node.claim.chars().take(80).collect();
            return Some(format!(
                "Traceability veto: node has empty expert_id — \"{preview}\"… \
                 Every node must have a non-empty expert_id (the source paper ID)."
            ));
        }
        // F13: a node whose expert_id is not a valid source id has no real provenance.
        if !is_valid_source_id(&node.expert_id, panel_ids) {
            let preview: String = node.claim.chars().take(80).collect();
            return Some(format!(
                "Traceability veto: node expert_id is invalid ({label}) — \
                 expert_id must be an arxiv:… or s2:… paper id or a known panel id \
                 (claim: \"{preview}\"…).",
                label = node.expert_id
            ));
        }
    }
    None
}

// ── Weighted sum (None-redistribution) ────────────────────────────────────────

/// Compute the weighted sum for one candidate with None-redistribution.
///
/// Mirrors `weighted_select` in fitness.py:400-418.
///
/// # None-redistribution
///
/// Dimensions where `score == None` (parse failure) are skipped entirely.
/// The denominator is `scored_weight` (sum of weights for NON-None dims only),
/// NOT `total_weight` (sum of all weights). This prevents score deflation
/// when the LLM judge fails to parse one dimension.
///
/// # Panics
///
/// Does not panic. Returns `0.0` when `scored_weight == 0.0` (all None).
pub fn weighted_sum(scores: &[(&str, Option<u8>)], weights: &[(&str, f32)]) -> f32 {
    let weight_map: HashMap<&str, f32> = weights.iter().cloned().collect();

    let mut raw_total: f32 = 0.0;
    let mut scored_weight: f32 = 0.0;

    for (dim, score) in scores {
        if let Some(s) = score {
            let w = weight_map.get(*dim).copied().unwrap_or(0.0);
            raw_total += *s as f32 * w;
            scored_weight += w;
        }
    }

    if scored_weight > 0.0 {
        raw_total / scored_weight
    } else {
        0.0 // All scores are None — return 0 (degenerate case)
    }
}

// ── Sort candidates ────────────────────────────────────────────────────────────

/// Sort `candidates` by fitness, best first.
///
/// Sort key: `(valid, total)` descending — valid candidates sort before
/// invalid, then by score descending (fitness.py:400-428).
///
/// `is_valid` is the stage-specific validity predicate:
/// - Graph: `is_valid_graph`
/// - Synthesis / Narrative: `is_valid_synthesis`
pub fn sort_candidates_best_first<A: Clone>(
    candidates: &[A],
    evals: &[FitnessEval],
    weights: &[(&str, f32)],
    is_valid: fn(&FitnessEval) -> bool,
) -> Vec<A> {
    assert_eq!(
        candidates.len(),
        evals.len(),
        "candidates and evals must be parallel slices"
    );

    let mut scored: Vec<(bool, f32, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let score_pairs = evals[i].score_pairs();
            let total = weighted_sum(&score_pairs, weights);
            let valid = is_valid(&evals[i]);
            (valid, total, i)
        })
        .collect();

    // valid candidates first (true > false via b.0.cmp(&a.0)), then score descending
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    scored
        .iter()
        .map(|(_, _, i)| candidates[*i].clone())
        .collect()
}

// ── Weight sum enforcement ─────────────────────────────────────────────────────

/// Assert that a weight table sums to 1.0 ± 0.001.
///
/// Called at construction time for each stage's `TtdMachine`. Returns
/// `Err(TtdError::InvalidWeightSum)` when the sum is outside the tolerance.
pub fn check_weight_sum(weights: &[(&str, f32)]) -> Result<(), TtdError> {
    let sum: f32 = weights.iter().map(|(_, w)| w).sum();
    if (sum - 1.0).abs() > 0.001 {
        return Err(TtdError::InvalidWeightSum { sum });
    }
    Ok(())
}

// ── Feedback document generation ──────────────────────────────────────────────

/// Generate the fitness feedback markdown document injected into `gap_identify`.
///
/// Ports `_generate_feedback` from fitness.py:213-311. The exact heading
/// structure is load-bearing — the denoiser parses it to find the feedback block.
///
/// ## Output format (veto-free — v1 path, byte-identical)
///
/// ```text
/// ## Priority Improvements (Score ≤ 3)
/// - **dimension**: rationale
///
/// ## Strengths to Preserve
/// - **dimension**: rationale
///
/// ## Evolutionary Guidance
/// Focus on resolving the priority improvements ...
/// ```
///
/// ## Output format (vetoed — v2 path)
///
/// When `eval.veto` is `Some(reason)`, a HARD FAIL section is prepended BEFORE
/// the Priority Improvements heading. The three existing headings follow unchanged
/// (denoiser-load-bearing).
///
/// ```text
/// ## HARD FAIL — Traceability veto
/// {reason}
/// Every claim MUST cite at least one source paper ID. Fix this before anything else.
///
/// ## Priority Improvements (Score ≤ 3)
/// ...
/// ```
///
/// This function is called when `use_fitness_feedback = true` in `TtdConfig`.
/// Dims scoring ≤ `fitness_threshold` appear under "Priority Improvements".
pub fn generate_feedback(eval: &FitnessEval, threshold: u8) -> String {
    let mut priority: Vec<String> = Vec::new();
    let mut strengths: Vec<String> = Vec::new();

    for (dim, score) in &eval.scores {
        match score {
            Some(s) if *s <= threshold => {
                priority.push(format!("- **{}**: score={s}", dim));
            }
            Some(s) => {
                strengths.push(format!("- **{}**: score={s}", dim));
            }
            None => {
                // Parse failure — treat as a priority improvement (unknown is a gap)
                priority.push(format!("- **{}**: score unavailable (parse failure)", dim));
            }
        }
    }

    let mut doc = String::new();

    // v2: prepend HARD FAIL section when veto is Some.
    // Conditional on veto only — v1 feedback stays byte-identical.
    if let Some(ref reason) = eval.veto {
        doc.push_str("## HARD FAIL — Traceability veto\n");
        doc.push_str(reason);
        doc.push('\n');
        doc.push_str(
            "Every claim MUST cite at least one source paper ID. \
             Fix this before anything else.\n\n",
        );
    }

    doc.push_str(&format!("## Priority Improvements (Score ≤ {threshold})\n"));
    if priority.is_empty() {
        doc.push_str("No dimensions below threshold — candidate is strong.\n");
    } else {
        doc.push_str(&priority.join("\n"));
        doc.push('\n');
    }

    doc.push('\n');
    doc.push_str("## Strengths to Preserve\n");
    if strengths.is_empty() {
        doc.push_str("No dimensions above threshold.\n");
    } else {
        doc.push_str(&strengths.join("\n"));
        doc.push('\n');
    }

    doc.push('\n');
    doc.push_str("## Evolutionary Guidance\n");
    if priority.is_empty() {
        doc.push_str(
            "The candidate meets the threshold on all scored dimensions. \
             Focus on preserving identified strengths in the revised draft.\n",
        );
    } else {
        doc.push_str(
            "Focus on resolving the priority improvements listed above. \
             Preserve the strengths while addressing the low-scoring dimensions.\n",
        );
    }

    doc
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_eval(scores: &[(&str, Option<u8>)]) -> FitnessEval {
        FitnessEval::new(
            scores
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        )
    }

    // ── ENGINE-03: validity gate ──────────────────────────────────────────────

    /// groundedness=3 (is_valid=false) must sort LAST regardless of other scores.
    #[test]
    fn invalid_candidate_sorts_last_graph() {
        // Candidate 0: groundedness=5 (valid), all others=5 → best
        let valid_eval = graph_eval(&[
            ("groundedness", Some(5)),
            ("coverage", Some(5)),
            ("atomicity", Some(5)),
            ("non_redundancy", Some(5)),
            ("relation_coherence", Some(5)),
            ("dissent_preservation", Some(5)),
        ]);
        // Candidate 1: groundedness=3 (INVALID), all others=5 → must be LAST
        let invalid_eval = graph_eval(&[
            ("groundedness", Some(3)),
            ("coverage", Some(5)),
            ("atomicity", Some(5)),
            ("non_redundancy", Some(5)),
            ("relation_coherence", Some(5)),
            ("dissent_preservation", Some(5)),
        ]);

        let candidates = vec!["invalid", "valid"];
        let evals = vec![invalid_eval, valid_eval];
        let sorted = sort_candidates_best_first(&candidates, &evals, GRAPH_WEIGHTS, is_valid_graph);

        assert_eq!(
            sorted[0], "valid",
            "valid candidate (groundedness=5) must rank first"
        );
        assert_eq!(
            sorted[1], "invalid",
            "invalid candidate (groundedness=3) must rank last regardless of other scores"
        );
    }

    /// faithfulness=3 sorts LAST for synthesis/narrative.
    #[test]
    fn invalid_candidate_sorts_last_synthesis() {
        let valid_eval = graph_eval(&[
            ("faithfulness", Some(5)),
            ("completeness", Some(5)),
            ("traceability", Some(5)),
            ("neutrality", Some(5)),
            ("dissent_visibility", Some(5)),
            ("structural_clarity", Some(5)),
        ]);
        let invalid_eval = graph_eval(&[
            ("faithfulness", Some(3)),
            ("completeness", Some(5)),
            ("traceability", Some(5)),
            ("neutrality", Some(5)),
            ("dissent_visibility", Some(5)),
            ("structural_clarity", Some(5)),
        ]);

        let candidates = vec!["invalid", "valid"];
        let evals = vec![invalid_eval, valid_eval];
        let sorted =
            sort_candidates_best_first(&candidates, &evals, SYNTHESIS_WEIGHTS, is_valid_synthesis);

        assert_eq!(sorted[0], "valid");
        assert_eq!(sorted[1], "invalid");
    }

    // ── ENGINE-03: None redistribution ────────────────────────────────────────

    /// None score redistributes weight: raw_total / scored_weight, NOT raw_total / total_weight.
    #[test]
    fn none_score_redistributes_weight() {
        // 2-dim table: dim_a weight=0.6, dim_b weight=0.4
        let weights: &[(&str, f32)] = &[("dim_a", 0.6), ("dim_b", 0.4)];

        // Case 1: both scored → total = (5 * 0.6 + 3 * 0.4) / 1.0 = 4.2
        let both_scored = &[("dim_a", Some(5u8)), ("dim_b", Some(3u8))];
        let both_total = weighted_sum(both_scored, weights);
        assert!(
            (both_total - 4.2).abs() < 0.001,
            "both scored: expected 4.2, got {both_total}"
        );

        // Case 2: dim_b=None → scored_weight=0.6 → total = (5 * 0.6) / 0.6 = 5.0
        // NOT (5 * 0.6 + 0) / 1.0 = 3.0 (the deflation bug)
        let none_b = &[("dim_a", Some(5u8)), ("dim_b", None)];
        let none_total = weighted_sum(none_b, weights);
        assert!(
            (none_total - 5.0).abs() < 0.001,
            "None redistributes to scored_weight: expected 5.0, got {none_total}. \
             If you got 3.0, you are dividing by total_weight instead of scored_weight — \
             Pitfall 3 (None ≠ 0 in the denominator)."
        );

        // Case 3: dim_a=None → scored_weight=0.4 → total = (3 * 0.4) / 0.4 = 3.0
        let none_a = &[("dim_a", None), ("dim_b", Some(3u8))];
        let none_a_total = weighted_sum(none_a, weights);
        assert!(
            (none_a_total - 3.0).abs() < 0.001,
            "None on high-weight dim: expected 3.0, got {none_a_total}"
        );

        // Case 4: all None → 0.0
        let all_none = &[("dim_a", None), ("dim_b", None)];
        let all_none_total = weighted_sum(all_none, weights);
        assert!(
            all_none_total == 0.0,
            "All None → 0.0, got {all_none_total}"
        );
    }

    // ── Weight sum enforcement ────────────────────────────────────────────────

    #[test]
    fn invalid_weight_sum_returns_error() {
        let bad_weights: &[(&str, f32)] = &[("a", 0.5), ("b", 0.6)]; // sums to 1.1
        let result = check_weight_sum(bad_weights);
        assert!(result.is_err());
        match result.unwrap_err() {
            TtdError::InvalidWeightSum { sum } => {
                assert!((sum - 1.1).abs() < 0.001, "expected sum ~1.1, got {sum}");
            }
            other => panic!("expected InvalidWeightSum, got {other:?}"),
        }
    }

    #[test]
    fn valid_weight_sum_is_ok() {
        let ok_weights: &[(&str, f32)] = &[("a", 0.5), ("b", 0.5)];
        assert!(check_weight_sum(ok_weights).is_ok());
    }

    // ── Feedback generation ───────────────────────────────────────────────────

    #[test]
    fn generate_feedback_contains_required_headings() {
        let eval = graph_eval(&[
            ("groundedness", Some(2)),
            ("coverage", Some(5)),
            ("atomicity", None),
        ]);
        let doc = generate_feedback(&eval, 3);
        assert!(
            doc.contains("## Priority Improvements"),
            "must contain Priority Improvements heading"
        );
        assert!(
            doc.contains("## Strengths to Preserve"),
            "must contain Strengths heading"
        );
        assert!(
            doc.contains("## Evolutionary Guidance"),
            "must contain Evolutionary Guidance heading"
        );
    }

    #[test]
    fn generate_feedback_puts_low_scores_in_priority() {
        let eval = graph_eval(&[
            ("groundedness", Some(2)), // ≤3 → priority
            ("coverage", Some(5)),     // >3 → strength
        ]);
        let doc = generate_feedback(&eval, 3);
        let prio_section = doc.split("## Strengths").next().unwrap_or("");
        assert!(prio_section.contains("groundedness"), "groundedness must be in priority");
        let strength_section = doc.split("## Strengths to Preserve").nth(1).unwrap_or("");
        assert!(strength_section.contains("coverage"), "coverage must be in strengths");
    }

    // ── Parse ladder tests (ENGINE-03 fitness.py:597-723) ─────────────────────

    /// A well-formed `<fitness_evaluation>` block yields Some(score) + rationale.
    #[test]
    fn parse_well_formed_xml() {
        let response = r#"<fitness_evaluation>
  <score>4</score>
  <rationale>Strong groundedness with minor gaps.</rationale>
  <suggestions>Add more citations.</suggestions>
</fitness_evaluation>"#;
        let parsed = parse_fitness_response(response);
        assert_eq!(
            parsed.score,
            Some(4),
            "well-formed XML must yield Some(4)"
        );
        assert!(
            parsed.rationale.contains("groundedness"),
            "rationale must be extracted: {:?}", parsed.rationale
        );
    }

    /// Malformed XML with a recognisable score digit yields the regex-extracted score.
    #[test]
    fn parse_via_regex_fallback() {
        // Malformed — missing closing tag, but score 3 is readable.
        let response = "The overall score for this dimension is 3 out of 5. Great coverage.";
        let parsed = parse_fitness_response(response);
        assert_eq!(
            parsed.score,
            Some(3),
            "regex fallback must extract score=3 from non-XML text: {:?}", parsed
        );
    }

    /// WR-03: an empty response STRING yields None (abstain), NOT the sentinel 3.
    ///
    /// Consensus reserves `score=3` for the literal "no response object" case
    /// (`fitness.py:731-732`: `if text is None`). An empty string flows through
    /// `_parse_score_text` (`if not text: return None`, fitness.py:678) to None.
    /// None redistributes weight; a 3 would inflate a silently-failed judge.
    #[test]
    fn empty_response_string_abstains_with_none() {
        let parsed = parse_fitness_response("");
        assert_eq!(
            parsed.score,
            None,
            "empty response string must yield None, not sentinel 3 (WR-03)"
        );
    }

    /// WR-03: a whitespace-only response is the same as empty → None (abstain),
    /// not sentinel 3. Consensus `text.strip()` makes `"  \n"` empty → None.
    #[test]
    fn whitespace_only_response_abstains_with_none() {
        let parsed = parse_fitness_response("   \n\t  ");
        assert_eq!(
            parsed.score,
            None,
            "whitespace-only response must yield None, not sentinel 3 (WR-03)"
        );
    }

    /// A non-empty but completely unparseable response yields None (NOT 3).
    ///
    /// This is the load-bearing distinction from Pitfall 3:
    /// - Empty → sentinel 3 (contributes to weighted sum)
    /// - Non-empty unparseable → None (redistributes weight over scored dims)
    #[test]
    fn unparseable_nonempty_yields_none() {
        // Text has no digit 1-5 with word boundaries — e.g. all digits 6-9 or none.
        let response = "Error: the LLM backend returned a timeout message. Code: 9876.";
        let parsed = parse_fitness_response(response);
        assert_eq!(
            parsed.score,
            None,
            "non-empty but unparseable response must yield None (not 3, not 0): {:?}", parsed
        );
    }

    /// WR-02: the bare-digit regex fallback is bounded to the first 200 chars
    /// (`raw[:200]`, fitness.py:700). A stray `1-5` digit DEEP in the rationale
    /// of a malformed response (no `<score>` tag) must NOT become the score —
    /// consensus yields None there, and None redistributes weight.
    #[test]
    fn bare_digit_fallback_bounded_to_first_200_chars() {
        // 250 chars of digit-free preamble, then a stray "3" well past char 200.
        // No <score> tag, so try_xml_parse + garbled-tag tier both miss.
        let mut response = String::new();
        response.push_str(&"x".repeat(250));
        response.push_str(" covered 3 datasets");
        let parsed = parse_fitness_response(&response);
        assert_eq!(
            parsed.score, None,
            "a digit past char 200 must NOT be picked up (raw[:200] bound): {:?}",
            parsed
        );
    }

    /// WR-02: a bare digit WITHIN the first 200 chars is still picked up — the
    /// bound widens the None window, it does not disable the fallback entirely.
    #[test]
    fn bare_digit_fallback_within_200_chars_still_parses() {
        let response = "The score is 4 — strong coverage across the panel.";
        let parsed = parse_fitness_response(response);
        assert_eq!(
            parsed.score, Some(4),
            "a digit within the first 200 chars must still parse: {:?}",
            parsed
        );
    }

    /// WR-02: a recognisable `<score>N</score>` tag in OTHERWISE-garbled XML is
    /// caught by the full-text tag regex with clamp (fitness.py:695-697), even
    /// when it sits past char 200 — the tag tier is not 200-bounded.
    #[test]
    fn garbled_score_tag_parsed_with_clamp() {
        let mut response = String::new();
        response.push_str(&"noise <broken> ".repeat(20)); // > 200 chars of garbage
        response.push_str("<score>7</score>");
        let parsed = parse_fitness_response(&response);
        assert_eq!(
            parsed.score, Some(5),
            "a <score>7</score> tag must parse and clamp to 5 regardless of offset: {:?}",
            parsed
        );
    }

    /// Score text "5" as plain integer is parsed correctly.
    #[test]
    fn parse_score_text_integer_path() {
        let parsed = parse_fitness_response("<score>5</score>");
        assert_eq!(parsed.score, Some(5), "plain integer in <score> tag must parse");
    }

    /// Score text "  2  " with surrounding whitespace parses correctly.
    #[test]
    fn parse_score_text_trimmed() {
        let parsed = parse_fitness_response("<score>  2  </score>");
        assert_eq!(parsed.score, Some(2), "whitespace-padded score must trim and parse");
    }

    /// CR-03: an out-of-range integer score is CLAMPED to [1,5], not turned into
    /// None. Consensus `_parse_score_text` does `max(1, min(5, int(text)))`
    /// (fitness.py:672-690). `<score>6</score>` → 5, `<score>0</score>` → 1.
    /// None is reserved for parse failure (non-integer text) only.
    #[test]
    fn out_of_range_score_is_clamped() {
        let high = parse_fitness_response("<score>6</score>");
        assert_eq!(
            high.score,
            Some(5),
            "score=6 must clamp to 5 (fitness.py:683), not become None"
        );
        let low = parse_fitness_response("<score>0</score>");
        assert_eq!(
            low.score,
            Some(1),
            "score=0 must clamp to 1 (fitness.py:683)"
        );
        let very_high = parse_fitness_response("<score>99</score>");
        assert_eq!(very_high.score, Some(5), "score=99 must clamp to 5");
    }

    // ── v2 traceability veto ──────────────────────────────────────────────────

    fn empty_panel_ids() -> std::collections::HashSet<String> {
        std::collections::HashSet::new()
    }

    fn make_claim(text: &str, sources: Vec<&str>) -> crate::ttd::artifact::Claim {
        crate::ttd::artifact::Claim {
            text: text.to_string(),
            agreement_level: None,
            sources: sources.iter().map(|s| s.to_string()).collect(),
            counterarguments: vec![],
            support_level: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        }
    }

    fn make_synthesis_with_claims(claims: Vec<crate::ttd::artifact::Claim>) -> crate::ttd::artifact::SynthesisArtifact {
        let mut a = crate::ttd::artifact::SynthesisArtifact::new(
            "study", "r1", "q1", "model", "v2/lit-review",
        );
        a.claims = claims;
        a
    }

    fn make_graph_with_nodes(nodes: Vec<crate::ttd::artifact::GraphNode>) -> crate::ttd::artifact::ArgumentationGraph {
        let mut g = crate::ttd::artifact::ArgumentationGraph::new(
            "study", "r1", "q1", "model", "v2/graph",
        );
        g.nodes = nodes;
        g
    }

    fn make_node(id: &str, claim: &str, expert_id: &str) -> crate::ttd::artifact::GraphNode {
        crate::ttd::artifact::GraphNode {
            id: id.to_string(),
            claim: claim.to_string(),
            expert_id: expert_id.to_string(),
            quote: None,
            verification_status: None,
        }
    }

    /// Probe-10 reproduction: a SynthesisArtifact whose claims ALL have sources:[]
    /// is vetoed; with all-Some(5) judge scores, is_valid_v2 is false; and
    /// sort_candidates_best_first ranks it LAST behind a sourced candidate scoring all Some(4).
    #[test]
    fn probe_10_sources_empty_vetoed_sorts_last() {
        use crate::ttd::weights::V2_SYNTHESIS_WEIGHTS;

        // Vetoed candidate: all claims have sources:[] → veto fires
        let unsourced_artifact = make_synthesis_with_claims(vec![
            make_claim("Permafrost thaw accelerates.", vec![]),
            make_claim("Methane release is increasing.", vec![]),
        ]);
        let veto_reason = traceability_veto_synthesis(&unsourced_artifact, &empty_panel_ids());
        assert!(veto_reason.is_some(), "sources:[] claims must trigger veto");

        // Vetoed eval: all dims Some(5) but veto attached
        let vetoed_eval = FitnessEval::new(
            V2_SYNTHESIS_WEIGHTS.iter().map(|(k, _)| (k.to_string(), Some(5u8))).collect()
        ).with_veto(veto_reason.unwrap());

        // is_valid_v2 must be false even with all-Some(5) scores
        assert!(
            !is_valid_v2(&vetoed_eval),
            "vetoed eval must be invalid regardless of scores"
        );

        // Sourced candidate: all claims have sources → no veto; lower scores (all Some(4))
        let sourced_eval = FitnessEval::new(
            V2_SYNTHESIS_WEIGHTS.iter().map(|(k, _)| (k.to_string(), Some(4u8))).collect()
        );
        // no veto → is_valid_v2 checks faithfulness ≥ 4 → true
        assert!(
            is_valid_v2(&sourced_eval),
            "sourced eval (faithfulness=4, no veto) must be valid"
        );

        // sort_candidates_best_first must rank sourced (invalid=false → valid=true) first
        let candidates = vec!["vetoed", "sourced"];
        let evals = vec![vetoed_eval, sourced_eval];
        let sorted = sort_candidates_best_first(&candidates, &evals, V2_SYNTHESIS_WEIGHTS, is_valid_v2);

        assert_eq!(sorted[0], "sourced", "sourced (valid) must rank first");
        assert_eq!(sorted[1], "vetoed", "vetoed (invalid despite all-5 scores) must rank last");
    }

    /// traceability_veto_synthesis: all claims sourced → None.
    #[test]
    fn veto_synthesis_all_sourced_returns_none() {
        let artifact = make_synthesis_with_claims(vec![
            make_claim("Claim A", vec!["arxiv:2304.07620"]),
            make_claim("Claim B", vec!["s2:abc123"]),
        ]);
        assert_eq!(traceability_veto_synthesis(&artifact, &empty_panel_ids()), None);
    }

    /// traceability_veto_synthesis: zero claims → Some(reason).
    #[test]
    fn veto_synthesis_zero_claims_returns_some() {
        let artifact = make_synthesis_with_claims(vec![]);
        assert!(traceability_veto_synthesis(&artifact, &empty_panel_ids()).is_some());
    }

    /// traceability_veto_synthesis: one claim with empty sources → Some(reason naming claim).
    #[test]
    fn veto_synthesis_one_unsourced_claim() {
        let artifact = make_synthesis_with_claims(vec![
            make_claim("This claim has no sources.", vec![]),
        ]);
        let veto = traceability_veto_synthesis(&artifact, &empty_panel_ids());
        assert!(veto.is_some());
        // The reason must reference the claim text (first ~80 chars)
        let reason = veto.unwrap();
        assert!(reason.contains("This claim has no sources"), "reason must name the claim: {reason}");
    }

    /// traceability_veto_graph: all nodes have non-empty expert_id → None.
    #[test]
    fn veto_graph_all_sourced_returns_none() {
        let graph = make_graph_with_nodes(vec![
            make_node("arxiv:123_c1", "Claim A", "arxiv:123"),
            make_node("s2:abc_c1", "Claim B", "s2:abc"),
        ]);
        assert_eq!(traceability_veto_graph(&graph, &empty_panel_ids()), None);
    }

    /// traceability_veto_graph: zero nodes → Some(reason).
    #[test]
    fn veto_graph_zero_nodes_returns_some() {
        let graph = make_graph_with_nodes(vec![]);
        assert!(traceability_veto_graph(&graph, &empty_panel_ids()).is_some());
    }

    /// traceability_veto_graph: node with empty expert_id → Some(reason).
    #[test]
    fn veto_graph_empty_expert_id() {
        let graph = make_graph_with_nodes(vec![
            make_node("_c1", "A claim from unknown source.", ""),
        ]);
        let veto = traceability_veto_graph(&graph, &empty_panel_ids());
        assert!(veto.is_some());
        let reason = veto.unwrap();
        assert!(reason.contains("empty expert_id"), "reason must mention empty expert_id: {reason}");
    }

    /// traceability_veto_graph: node with whitespace expert_id → Some(reason).
    #[test]
    fn veto_graph_whitespace_expert_id() {
        let graph = make_graph_with_nodes(vec![
            make_node("_c1", "A claim.", "   "),
        ]);
        assert!(traceability_veto_graph(&graph, &empty_panel_ids()).is_some());
    }

    /// is_valid_v2: veto present → false regardless of scores.
    #[test]
    fn is_valid_v2_veto_overrides_scores() {
        use crate::ttd::weights::V2_SYNTHESIS_WEIGHTS;
        let eval = FitnessEval::new(
            V2_SYNTHESIS_WEIGHTS.iter().map(|(k, _)| (k.to_string(), Some(5u8))).collect()
        ).with_veto("some reason".to_string());
        assert!(!is_valid_v2(&eval), "veto present → always invalid");
    }

    /// is_valid_v2: no veto + faithfulness >= 4 → true.
    #[test]
    fn is_valid_v2_no_veto_faithfulness_high() {
        let eval = FitnessEval::new(vec![("faithfulness".to_string(), Some(4))]);
        assert!(is_valid_v2(&eval));
    }

    /// is_valid_v2: no veto + faithfulness 3 → false.
    #[test]
    fn is_valid_v2_no_veto_faithfulness_low() {
        let eval = FitnessEval::new(vec![("faithfulness".to_string(), Some(3))]);
        assert!(!is_valid_v2(&eval));
    }

    /// is_valid_v2: no veto + faithfulness None → false.
    #[test]
    fn is_valid_v2_no_veto_faithfulness_none() {
        let eval = FitnessEval::new(vec![("faithfulness".to_string(), None)]);
        assert!(!is_valid_v2(&eval));
    }

    /// FitnessEval::new keeps veto: None — all existing call sites unaffected.
    #[test]
    fn fitness_eval_new_veto_is_none() {
        let eval = FitnessEval::new(vec![("faithfulness".to_string(), Some(5))]);
        assert!(eval.veto.is_none(), "new() must default veto to None");
    }

    /// with_veto attaches the reason.
    #[test]
    fn fitness_eval_with_veto_attaches_reason() {
        let eval = FitnessEval::new(vec![])
            .with_veto("test reason".to_string());
        assert_eq!(eval.veto.as_deref(), Some("test reason"));
    }

    /// generate_feedback: vetoed eval prepends HARD FAIL section before Priority Improvements.
    #[test]
    fn generate_feedback_vetoed_prepends_hard_fail() {
        let eval = FitnessEval::new(vec![
            ("faithfulness".to_string(), Some(2)),
        ]).with_veto("Claim X has no sources.".to_string());
        let doc = generate_feedback(&eval, 3);
        // HARD FAIL section must come before Priority Improvements
        let hard_fail_pos = doc.find("## HARD FAIL — Traceability veto").expect("must contain HARD FAIL heading");
        let priority_pos = doc.find("## Priority Improvements").expect("must contain Priority Improvements");
        assert!(hard_fail_pos < priority_pos, "HARD FAIL must precede Priority Improvements");
        assert!(doc.contains("Claim X has no sources."), "reason must appear in doc");
        assert!(doc.contains("Every claim MUST cite"), "fix instruction must appear in doc");
    }

    /// generate_feedback: veto-free eval is byte-identical to pre-B3 (v1 path unchanged).
    #[test]
    fn generate_feedback_veto_free_byte_identical() {
        let eval = FitnessEval::new(vec![
            ("faithfulness".to_string(), Some(2)),
            ("coverage".to_string(), Some(5)),
        ]);
        let doc = generate_feedback(&eval, 3);
        // Must NOT contain the HARD FAIL heading on a veto-free eval
        assert!(!doc.contains("## HARD FAIL"), "veto-free feedback must not contain HARD FAIL");
        // Must still start with Priority Improvements (first heading)
        assert!(doc.starts_with("## Priority Improvements"), "veto-free feedback must start with Priority Improvements");
    }

    // ── Golden-vector ordering test ────────────────────────────────────────────

    /// ENGINE-03: golden-vector ordering matches the expected ordering.
    ///
    /// These vectors are STRUCTURAL placeholders — they test that the sort key
    /// (valid, total) works correctly for a 3-candidate set where:
    ///   - Candidate A: groundedness=5 (valid), all dims=5 → rank 1st
    ///   - Candidate B: groundedness=4 (valid), mixed dims → rank 2nd
    ///   - Candidate C: groundedness=3 (invalid), all dims=5 → rank LAST
    ///
    /// The real seed=42 oracle oracle vectors are pending a live consensus run
    /// (oracle dependency — see golden_vectors.md). This test pins the selection
    /// LOGIC without needing oracle numbers.
    ///
    /// TODO: replace with real oracle vectors from .planning/.../oracle/golden_vectors.md
    /// once a consensus oracle run with FIDELITY_SEED=42 is executed.
    #[test]
    fn golden_vector_ordering_matches_expected_logic() {
        // A: groundedness=5 (valid), coverage=5, others=5 → highest valid score
        let eval_a = graph_eval(&[
            ("groundedness", Some(5)),
            ("coverage", Some(5)),
            ("atomicity", Some(5)),
            ("non_redundancy", Some(5)),
            ("relation_coherence", Some(5)),
            ("dissent_preservation", Some(5)),
        ]);
        // B: groundedness=4 (valid), coverage=3, others=4 → valid but lower score
        let eval_b = graph_eval(&[
            ("groundedness", Some(4)),
            ("coverage", Some(3)),
            ("atomicity", Some(4)),
            ("non_redundancy", Some(4)),
            ("relation_coherence", Some(4)),
            ("dissent_preservation", Some(4)),
        ]);
        // C: groundedness=3 (INVALID), others=5 → invalid → LAST regardless
        let eval_c = graph_eval(&[
            ("groundedness", Some(3)),
            ("coverage", Some(5)),
            ("atomicity", Some(5)),
            ("non_redundancy", Some(5)),
            ("relation_coherence", Some(5)),
            ("dissent_preservation", Some(5)),
        ]);

        let candidates = vec!["C", "B", "A"]; // deliberately out of order
        let evals = vec![eval_c, eval_b, eval_a];
        let sorted = sort_candidates_best_first(&candidates, &evals, GRAPH_WEIGHTS, is_valid_graph);

        assert_eq!(sorted[0], "A", "A (groundedness=5, highest valid score) must rank first");
        assert_eq!(sorted[1], "B", "B (groundedness=4, lower valid score) must rank second");
        assert_eq!(sorted[2], "C", "C (groundedness=3, invalid) must rank last");
    }

    // ── Fix B migrated to F13 allowlist: candidate-aware traceability vetoes ──

    /// F13 (supersedes Fix B, probe-17 cause 1, prong 2b): traceability_veto_synthesis
    /// must veto a claim whose sources are ALL candidate-pattern labels (still invalid
    /// under the allowlist), and must NOT veto a claim with one real source plus a
    /// candidate label.
    #[test]
    fn fix_b_veto_synthesis_candidate_only_sources() {
        use crate::ttd::artifact::{Claim, SynthesisArtifact};

        // A claim with ONLY candidate-pattern sources — must veto (invalid by allowlist).
        let mut art_all_candidate = SynthesisArtifact::new("s", "r", "q", "model", "v2");
        art_all_candidate.claims.push(Claim {
            text: "A claim citing only candidates.".to_string(),
            agreement_level: None,
            support_level: None,
            sources: vec!["Candidate1".to_string(), "candidate_3".to_string()],
            quotes: vec![],
            node_refs: vec![],
            citation: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            counterarguments: vec![],
        });

        let veto = traceability_veto_synthesis(&art_all_candidate, &empty_panel_ids());
        assert!(
            veto.is_some(),
            "claim with only candidate-pattern sources must trigger veto"
        );
        let reason = veto.unwrap();
        assert!(
            reason.contains("Candidate1") || reason.contains("candidate_3"),
            "veto reason must name the invalid labels, got: {reason}"
        );

        // A claim with one real source plus a candidate label — must NOT veto.
        let mut art_mixed = SynthesisArtifact::new("s", "r", "q", "model", "v2");
        art_mixed.claims.push(Claim {
            text: "A mixed claim.".to_string(),
            agreement_level: None,
            support_level: None,
            sources: vec![
                "arxiv:2105.14103".to_string(),
                "Candidate1".to_string(),
            ],
            quotes: vec![],
            node_refs: vec![],
            citation: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            counterarguments: vec![],
        });

        assert!(
            traceability_veto_synthesis(&art_mixed, &empty_panel_ids()).is_none(),
            "claim with one real source must not veto even if a candidate label is also present"
        );
    }

    /// Fix B (probe-17 cause 1, prong 2b): traceability_veto_graph must veto a
    /// node whose expert_id is a candidate-pattern label.
    #[test]
    fn fix_b_veto_graph_candidate_expert_id() {
        use crate::ttd::artifact::{ArgumentationGraph, GraphNode};
        use std::collections::HashSet;

        let empty: HashSet<String> = HashSet::new();

        // A graph node with a candidate-pattern expert_id — must veto.
        let mut graph_candidate = ArgumentationGraph::new("s", "r", "q", "model", "v2/lit-review");
        graph_candidate.nodes.push(GraphNode {
            id: "n1".to_string(),
            claim: "Some claim.".to_string(),
            expert_id: "Candidate2".to_string(),
            quote: None,
            verification_status: None,
        });

        let veto = traceability_veto_graph(&graph_candidate, &empty);
        assert!(
            veto.is_some(),
            "node with candidate-pattern expert_id must trigger veto"
        );
        let reason = veto.unwrap();
        assert!(
            reason.contains("Candidate") || reason.contains("candidate"),
            "veto reason must name the candidate label, got: {reason}"
        );

        // A node with a real expert_id — must NOT veto.
        let mut graph_real = ArgumentationGraph::new("s", "r", "q", "model", "v2/lit-review");
        graph_real.nodes.push(GraphNode {
            id: "n1".to_string(),
            claim: "Some claim.".to_string(),
            expert_id: "arxiv:2105.14103".to_string(),
            quote: None,
            verification_status: None,
        });

        assert!(
            traceability_veto_graph(&graph_real, &empty).is_none(),
            "node with real expert_id must not veto"
        );
    }

    // ── F13: probe-18 regression — allowlist VETOs ────────────────────────────

    /// F13 (probe-18 regression, synthesis): a claim whose sources are ALL probe-18
    /// mutated labels (sN_candidateN) must trigger the veto; the reason must name
    /// the offending labels. Mixed real+invalid must NOT veto (Task-1 strip removes
    /// the junk at emit time). Non-panel arxiv:-shaped source must NOT veto (F11).
    #[test]
    fn f13_probe18_veto_synthesis_mutated_labels() {
        use crate::ttd::artifact::{Claim, SynthesisArtifact};
        use std::collections::HashSet;

        let empty: HashSet<String> = HashSet::new();

        // All-mutated-labels: must veto.
        let mut art_all_mutated = SynthesisArtifact::new("s", "r", "q", "model", "v2");
        art_all_mutated.claims.push(Claim {
            text: "Claim citing only probe-18 mutated labels.".to_string(),
            agreement_level: None,
            support_level: None,
            sources: vec!["s1_candidate1".to_string(), "s5_candidate5".to_string()],
            quotes: vec![],
            node_refs: vec![],
            citation: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            counterarguments: vec![],
        });

        let veto = traceability_veto_synthesis(&art_all_mutated, &empty);
        assert!(
            veto.is_some(),
            "all-mutated-label sources must trigger veto"
        );
        let reason = veto.unwrap();
        // Reason must name the offending labels.
        assert!(
            reason.contains("s1_candidate1") || reason.contains("s5_candidate5"),
            "veto reason must name the offending mutated labels, got: {reason}"
        );
        // Reason must steer toward valid shapes.
        assert!(
            reason.contains("arxiv:") || reason.contains("s2:"),
            "veto reason must mention valid id shapes, got: {reason}"
        );

        // Mixed real+mutated: must NOT veto (one real source saves the claim).
        let mut art_mixed = SynthesisArtifact::new("s", "r", "q", "model", "v2");
        art_mixed.claims.push(Claim {
            text: "Mixed claim.".to_string(),
            agreement_level: None,
            support_level: None,
            sources: vec!["arxiv:2105.14103".to_string(), "s3_candidate3".to_string()],
            quotes: vec![],
            node_refs: vec![],
            citation: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            counterarguments: vec![],
        });
        assert!(
            traceability_veto_synthesis(&art_mixed, &empty).is_none(),
            "mixed real+mutated sources must NOT veto"
        );

        // Non-panel arxiv: source — must NOT veto (F11 lane).
        let mut art_arxiv = SynthesisArtifact::new("s", "r", "q", "model", "v2");
        art_arxiv.claims.push(Claim {
            text: "Claim citing non-panel arxiv paper.".to_string(),
            agreement_level: None,
            support_level: None,
            sources: vec!["arxiv:2501.13956".to_string()],
            quotes: vec![],
            node_refs: vec![],
            citation: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            counterarguments: vec![],
        });
        assert!(
            traceability_veto_synthesis(&art_arxiv, &empty).is_none(),
            "non-panel arxiv: source must NOT veto (F11)"
        );
    }

    /// F13 (probe-18 regression, graph): a node with expert_id "s2_candidate2"
    /// (probe-18 mutated label) must veto; arxiv:-shaped expert_id must not.
    #[test]
    fn f13_probe18_veto_graph_mutated_label() {
        use crate::ttd::artifact::{ArgumentationGraph, GraphNode};
        use std::collections::HashSet;

        let empty: HashSet<String> = HashSet::new();

        // Node with probe-18 mutated label as expert_id — must veto.
        let mut graph_mutated = ArgumentationGraph::new("s", "r", "q", "model", "v2/lit-review");
        graph_mutated.nodes.push(GraphNode {
            id: "n1".to_string(),
            claim: "Some claim.".to_string(),
            expert_id: "s2_candidate2".to_string(),
            quote: None,
            verification_status: None,
        });

        let veto = traceability_veto_graph(&graph_mutated, &empty);
        assert!(
            veto.is_some(),
            "node with mutated-label expert_id must trigger veto"
        );
        let reason = veto.unwrap();
        assert!(
            reason.contains("s2_candidate2"),
            "veto reason must name the mutated label, got: {reason}"
        );

        // Node with arxiv:-shaped expert_id — must NOT veto (panel or not).
        let mut graph_arxiv = ArgumentationGraph::new("s", "r", "q", "model", "v2/lit-review");
        graph_arxiv.nodes.push(GraphNode {
            id: "n1".to_string(),
            claim: "Some claim.".to_string(),
            expert_id: "arxiv:2105.14103".to_string(),
            quote: None,
            verification_status: None,
        });

        assert!(
            traceability_veto_graph(&graph_arxiv, &empty).is_none(),
            "arxiv:-shaped expert_id must not veto (panel or not)"
        );
    }
}
