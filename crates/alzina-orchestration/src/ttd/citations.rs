//! F14 commit 5 — author-year citation rendering for v2/v3 lit-review output.
//!
//! The synthesis model cites sources by paper id (`Claim.sources`) and the
//! narrative carries inline `[Cx]` markers (claim-index references). This module
//! turns those machine ids into human author-year citations, e.g.
//! `(Smith et al., 2021; Jones, 2020)`, against paper metadata resolved from the
//! `papers` table at the daemon emit boundary.
//!
//! ## Design decisions (Sam-confirmed, 2026-06-14)
//!
//! - **Fork 1 — inline markers: REPLACE.** Each `[Cx]` in the narrative becomes
//!   the author-year citation for that claim's sources. `narrative_statements`
//!   are parsed from the original `[Cx]` markers inside the engine BEFORE this
//!   runs, so `claim_refs` (the machine link) survives; the rendered prose shows
//!   author-year only. Each statement's `.text` is transformed too, to keep it
//!   consistent with `narrative`.
//! - **Fork 2 — References: APPEND.** A `## References` markdown block is
//!   appended to the narrative string, one entry per cited source.
//! - **Fork 3 — per-claim citation: NEW FIELD.** `Claim.citation` is set to the
//!   rendered author-year string for the claim's sources.
//!
//! ## Purity
//!
//! This module is pure: it takes the artifact + a resolved metadata map and
//! mutates the artifact. No DB, no async. The DB fetch (source id → metadata)
//! lives in the daemon handler where `lit_pool` is in scope. v1 byte-identity is
//! the caller's responsibility — `apply_author_year_citations` must only run for
//! `V2LitReview`/`V3LitReviewLong`.

use std::collections::BTreeMap;

use crate::ttd::artifact::SynthesisArtifact;

/// Resolved metadata for one paper, parsed from the `papers` table row.
///
/// `authors` is the parsed author list (the table stores it as a JSON array
/// string; the daemon parses it before building the map). All fields are
/// best-effort — any may be missing for a sparsely-recorded source.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PaperMeta {
    /// Author full names, in author order (e.g. `["Alice Smith", "Bob Jones"]`).
    pub authors: Vec<String>,
    /// Publication year.
    pub year: Option<i64>,
    /// Paper title.
    pub title: Option<String>,
    /// Canonical URL (arxiv abstract link for arxiv sources).
    pub url: Option<String>,
}

/// Extract a citation surname from a full author name.
///
/// Handles two common stored forms:
/// - `"Smith, Alice"` (comma — last name first) → `"Smith"`
/// - `"Alice Smith"` (space — first name first) → `"Smith"`
///
/// Falls back to the whole trimmed string when neither separator is present.
fn surname(full: &str) -> String {
    let full = full.trim();
    if let Some((before, _)) = full.split_once(',') {
        let before = before.trim();
        if !before.is_empty() {
            return before.to_string();
        }
    }
    full.rsplit(char::is_whitespace)
        .find(|s| !s.is_empty())
        .unwrap_or(full)
        .to_string()
}

/// Render one source's author-year citation, e.g. `"Smith et al., 2021"`.
///
/// Returns `None` when there are no authors (the caller falls back to the raw
/// source id so the citation stays traceable). A missing year renders `"n.d."`.
fn render_author_year(meta: &PaperMeta) -> Option<String> {
    if meta.authors.is_empty() {
        return None;
    }
    let year = meta
        .year
        .map(|y| y.to_string())
        .unwrap_or_else(|| "n.d.".to_string());
    let author_part = match meta.authors.len() {
        1 => surname(&meta.authors[0]),
        2 => format!("{} & {}", surname(&meta.authors[0]), surname(&meta.authors[1])),
        _ => format!("{} et al.", surname(&meta.authors[0])),
    };
    Some(format!("{author_part}, {year}"))
}

/// Render the inner citation text for one source (no surrounding parentheses).
/// Falls back to the raw source id when no usable metadata exists.
fn cite_one(source_id: &str, meta: &BTreeMap<String, PaperMeta>) -> String {
    meta.get(source_id)
        .and_then(render_author_year)
        .unwrap_or_else(|| source_id.to_string())
}

/// Render a parenthesised citation group for a set of source ids, e.g.
/// `"(Smith et al., 2021; Jones, 2020)"`. Order follows `source_ids`; duplicate
/// ids are dropped. Returns `None` when `source_ids` is empty.
fn cite_group(source_ids: &[String], meta: &BTreeMap<String, PaperMeta>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for sid in source_ids {
        let rendered = cite_one(sid, meta);
        if !parts.contains(&rendered) {
            parts.push(rendered);
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(format!("({})", parts.join("; ")))
}

/// Resolve one `[Cx]` / `[C1, C3]` bracket to a citation group, or `None` when
/// the bracket is not a pure list of valid claim ids (then it is left untouched).
///
/// Claim ids are `C{n}` (1-based) referencing `claim_sources[n-1]`.
fn render_marker(
    bracket: &str,
    claim_sources: &[Vec<String>],
    meta: &BTreeMap<String, PaperMeta>,
) -> Option<String> {
    let mut sources: Vec<String> = Vec::new();
    for id in bracket.split(',') {
        let id = id.trim();
        let idx: usize = id.strip_prefix('C')?.parse().ok()?;
        if idx == 0 || idx > claim_sources.len() {
            return None;
        }
        for s in &claim_sources[idx - 1] {
            if !sources.contains(s) {
                sources.push(s.clone());
            }
        }
    }
    cite_group(&sources, meta)
}

/// Replace every `[Cx]`/`[C1, C3]` marker in `text` with its author-year group.
///
/// Brackets that are not pure claim-id lists (e.g. `[note]`), or that resolve to
/// no sources, are left verbatim — the transform never drops non-citation text.
fn replace_markers(
    text: &str,
    claim_sources: &[Vec<String>],
    meta: &BTreeMap<String, PaperMeta>,
) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '[' {
            // Find the matching ']' (no nested brackets in the marker grammar).
            let mut j = i + 1;
            while j < chars.len() && chars[j] != ']' {
                j += 1;
            }
            if j < chars.len() {
                let bracket: String = chars[i + 1..j].iter().collect();
                if let Some(citation) = render_marker(&bracket, claim_sources, meta) {
                    out.push_str(&citation);
                    i = j + 1;
                    continue;
                }
            }
            // Not a resolvable marker — emit '[' literally and move on.
            out.push('[');
            i += 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Render the `## References` block body (one markdown bullet per cited source),
/// or empty string when nothing is cited.
///
/// Entries follow the union of `claim.sources` across all claims, in first-seen
/// order. Each entry: `- {author-year}. {title}. {url}` with missing parts
/// gracefully omitted; sources without metadata render their raw id.
fn render_references(synthesis: &SynthesisArtifact, meta: &BTreeMap<String, PaperMeta>) -> String {
    let mut cited: Vec<String> = Vec::new();
    for claim in &synthesis.claims {
        for s in &claim.sources {
            if !cited.contains(s) {
                cited.push(s.clone());
            }
        }
    }
    if cited.is_empty() {
        return String::new();
    }

    let mut lines: Vec<String> = Vec::new();
    for sid in &cited {
        let line = match meta.get(sid) {
            Some(m) => {
                let head = render_author_year(m).unwrap_or_else(|| sid.clone());
                let mut entry = format!("- {head}");
                if let Some(title) = m.title.as_deref().filter(|t| !t.trim().is_empty()) {
                    entry.push_str(&format!(". {}", title.trim()));
                }
                if let Some(url) = m.url.as_deref().filter(|u| !u.trim().is_empty()) {
                    entry.push_str(&format!(". {}", url.trim()));
                }
                entry
            }
            None => format!("- {sid}"),
        };
        lines.push(line);
    }
    lines.join("\n")
}

/// Apply author-year citations to a v2/v3 synthesis artifact in place.
///
/// 1. Sets `Claim.citation` for every claim with sources (fork 3).
/// 2. Replaces `[Cx]` markers in `narrative` and each `narrative_statements[].text`
///    with author-year groups (fork 1).
/// 3. Appends a `## References` section to `narrative` (fork 2).
///
/// Caller MUST gate this on `V2LitReview`/`V3LitReviewLong` — running it on v1
/// would mutate byte-identical output.
pub fn apply_author_year_citations(
    synthesis: &mut SynthesisArtifact,
    meta: &BTreeMap<String, PaperMeta>,
) {
    // 1. Per-claim citation string (fork 3).
    for claim in &mut synthesis.claims {
        if let Some(citation) = cite_group(&claim.sources, meta) {
            claim.citation = Some(citation);
        }
    }

    // Snapshot each claim's sources by index for [Cx] resolution.
    let claim_sources: Vec<Vec<String>> =
        synthesis.claims.iter().map(|c| c.sources.clone()).collect();

    // 2. Replace inline markers in the narrative and its parsed statements (fork 1).
    synthesis.narrative = replace_markers(&synthesis.narrative, &claim_sources, meta);
    for stmt in &mut synthesis.narrative_statements {
        stmt.text = replace_markers(&stmt.text, &claim_sources, meta);
    }

    // 3. Append a References section (fork 2).
    let references = render_references(synthesis, meta);
    if !references.is_empty() {
        if !synthesis.narrative.is_empty() {
            synthesis.narrative.push_str("\n\n");
        }
        synthesis.narrative.push_str("## References\n\n");
        synthesis.narrative.push_str(&references);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttd::artifact::{Claim, NarrativeStatement, SynthesisArtifact};

    fn meta(authors: &[&str], year: Option<i64>) -> PaperMeta {
        PaperMeta {
            authors: authors.iter().map(|s| s.to_string()).collect(),
            year,
            title: Some("A Study of Things".to_string()),
            url: Some("https://arxiv.org/abs/2105.14103".to_string()),
        }
    }

    fn claim(text: &str, sources: &[&str]) -> Claim {
        Claim {
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

    fn artifact_with(claims: Vec<Claim>, narrative: &str) -> SynthesisArtifact {
        let mut a = SynthesisArtifact::new("s", "r", "q", "model", "v2/lit-review");
        a.claims = claims;
        a.narrative = narrative.to_string();
        a
    }

    #[test]
    fn surname_handles_both_name_orders() {
        assert_eq!(surname("Alice Smith"), "Smith");
        assert_eq!(surname("Smith, Alice"), "Smith");
        assert_eq!(surname("Plato"), "Plato");
        assert_eq!(surname("  Bob  Q.  Jones  "), "Jones");
    }

    #[test]
    fn author_year_pluralisation() {
        assert_eq!(
            render_author_year(&meta(&["Alice Smith"], Some(2021))).as_deref(),
            Some("Smith, 2021")
        );
        assert_eq!(
            render_author_year(&meta(&["Alice Smith", "Bob Jones"], Some(2021))).as_deref(),
            Some("Smith & Jones, 2021")
        );
        assert_eq!(
            render_author_year(&meta(&["A S", "B J", "C K"], Some(2021))).as_deref(),
            Some("S et al., 2021")
        );
        // Missing year renders n.d.
        assert_eq!(
            render_author_year(&meta(&["Alice Smith"], None)).as_deref(),
            Some("Smith, n.d.")
        );
        // No authors → None (caller falls back to source id).
        assert_eq!(render_author_year(&meta(&[], Some(2021))), None);
    }

    #[test]
    fn replaces_inline_markers_with_author_year() {
        let mut m = BTreeMap::new();
        m.insert("arxiv:s1".to_string(), meta(&["Alice Smith"], Some(2021)));
        m.insert("arxiv:s2".to_string(), meta(&["Bob Jones"], Some(2020)));

        let mut art = artifact_with(
            vec![claim("c1", &["arxiv:s1"]), claim("c2", &["arxiv:s2"])],
            "First finding [C1]. Second finding [C2]. Both agree [C1, C2].",
        );
        apply_author_year_citations(&mut art, &m);

        // The narrative body (before the appended References block) has markers
        // replaced and no raw [Cx] left.
        let body = art.narrative.split("\n\n## References").next().unwrap();
        assert!(body.contains("(Smith, 2021)"), "body: {body}");
        assert!(body.contains("(Jones, 2020)"), "body: {body}");
        assert!(body.contains("(Smith, 2021; Jones, 2020)"), "body: {body}");
        assert!(!body.contains("[C1]"), "raw markers must be replaced: {body}");
        assert!(!body.contains("[C2]"), "raw markers must be replaced: {body}");
    }

    #[test]
    fn sets_per_claim_citation_and_appends_references() {
        let mut m = BTreeMap::new();
        m.insert("arxiv:s1".to_string(), meta(&["Alice Smith"], Some(2021)));

        let mut art = artifact_with(vec![claim("c1", &["arxiv:s1"])], "A finding [C1].");
        apply_author_year_citations(&mut art, &m);

        assert_eq!(art.claims[0].citation.as_deref(), Some("(Smith, 2021)"));
        assert!(
            art.narrative.contains("## References"),
            "References section must be appended: {}",
            art.narrative
        );
        assert!(
            art.narrative.contains("Smith, 2021"),
            "reference entry must carry author-year: {}",
            art.narrative
        );
        assert!(
            art.narrative.contains("https://arxiv.org/abs/2105.14103"),
            "reference entry must carry the url/arxiv link: {}",
            art.narrative
        );
    }

    #[test]
    fn unknown_source_falls_back_to_raw_id_not_dropped() {
        let m: BTreeMap<String, PaperMeta> = BTreeMap::new(); // no metadata at all
        let mut art = artifact_with(vec![claim("c1", &["arxiv:s9"])], "Finding [C1].");
        apply_author_year_citations(&mut art, &m);

        // Citation still rendered with the raw id — traceable, never silently empty.
        assert_eq!(art.claims[0].citation.as_deref(), Some("(arxiv:s9)"));
        let body = art.narrative.split("\n\n## References").next().unwrap();
        assert!(body.contains("(arxiv:s9)"), "body: {body}");
    }

    #[test]
    fn non_citation_brackets_are_left_untouched() {
        let mut m = BTreeMap::new();
        m.insert("arxiv:s1".to_string(), meta(&["Alice Smith"], Some(2021)));
        let mut art = artifact_with(
            vec![claim("c1", &["arxiv:s1"])],
            "A finding [C1] with an aside [see note] and an array index [0].",
        );
        apply_author_year_citations(&mut art, &m);
        let body = art.narrative.split("\n\n## References").next().unwrap();
        assert!(body.contains("[see note]"), "non-marker bracket kept: {body}");
        assert!(body.contains("[0]"), "non-marker bracket kept: {body}");
        assert!(body.contains("(Smith, 2021)"), "real marker replaced: {body}");
    }

    #[test]
    fn narrative_statements_text_transformed_consistently() {
        let mut m = BTreeMap::new();
        m.insert("arxiv:s1".to_string(), meta(&["Alice Smith"], Some(2021)));
        let mut art = artifact_with(vec![claim("c1", &["arxiv:s1"])], "A finding [C1].");
        art.narrative_statements = vec![NarrativeStatement {
            text: "A finding [C1].".to_string(),
            claim_refs: vec!["C1".to_string()],
            expert_refs: vec![],
        }];
        apply_author_year_citations(&mut art, &m);

        assert!(
            art.narrative_statements[0].text.contains("(Smith, 2021)"),
            "statement text must be transformed: {}",
            art.narrative_statements[0].text
        );
        assert!(
            !art.narrative_statements[0].text.contains("[C1]"),
            "statement marker must be replaced"
        );
        // claim_refs (machine link) survives the transform.
        assert_eq!(art.narrative_statements[0].claim_refs, vec!["C1".to_string()]);
    }
}
