//! Provenance-conservation guard and `deduplicate_sources` mirror.
//!
//! Both functions are pure (no async, no I/O) and have no side effects
//! beyond `tracing::error!` on conservation violation.
//!
//! ## conservation_assert
//!
//! Fires `Err(AdapterError::SourceIdLoss)` when the set of distinct
//! `expert_id` values leaving `build_panel` does not EQUAL the set of
//! distinct `paper_id` values entering it.  Equality is required because
//! both loss and duplication corrupt `panel_size` and therefore every
//! downstream agreement label (consensus / majority / divided / minority).
//!
//! ## deduplicate_sources
//!
//! Mirrors `consensus/src/consensus/domain/utils.py:34-43`.
//! Strips the first `_c` graph-node suffix from a `source_id` and keys
//! on `(base_id, norm_quote)` where `norm_quote` is the first 200
//! lowercased characters of the quote.  Preserves first-seen order.
//!
//! The mirror is ASCII-faithful.  It diverges from the Python oracle only on
//! non-ASCII case folding (Rust `to_lowercase` full folding vs CPython
//! `str.lower` simple mapping — e.g. `ß`); see the inline note on `norm_quote`
//! (WR-03).

use std::collections::HashSet;

use super::{AdapterError, ExpertResponse};

// ── conservation_assert ───────────────────────────────────────────────────────

/// Assert that the panel preserves source-id provenance exactly: a panel of
/// `N` `ExpertResponse` rows must carry `N` distinct `expert_id` values, one
/// per distinct paper ID in `input_ids`.
///
/// Three quantities must agree:
/// - `expected` — distinct paper IDs entering (`input_ids.len()`)
/// - `actual`   — distinct `expert_id` values leaving (`HashSet` of output ids)
/// - `emitted`  — `ExpertResponse` row count (`output.len()` == `panel_size`)
///
/// Returns `Ok(())` only when `expected == actual == emitted`. On mismatch
/// emits `tracing::error!` with structured fields and returns a distinct
/// variant per broken invariant:
/// - loss (`actual != expected`) → `Err(AdapterError::SourceIdLoss { expected, actual })`
/// - duplication (`emitted != actual`) → `Err(AdapterError::SourceIdDuplication { expected, actual, emitted })`
///
/// Splitting the variants keeps the error legible at the boundary: a loss
/// carries the dropped count, a duplication carries `emitted` (the only
/// quantity that proves the inflation).
///
/// ## Why equality, not `actual < expected`
///
/// Both dropping a paper and emitting two responses for the same paper corrupt
/// `panel_size`.  `panel_size` drives agreement thresholds
/// (consensus ≥ 0.75, majority ≥ 0.50, divided ≥ 0.30); any shift is silent
/// and affects every downstream claim label.
pub fn conservation_assert(
    input_ids: &HashSet<String>,
    output: &[ExpertResponse],
) -> Result<(), AdapterError> {
    let output_ids: HashSet<&str> = output.iter().map(|r| r.expert_id.as_str()).collect();
    let expected = input_ids.len();
    let actual = output_ids.len(); // distinct expert_ids leaving
    let emitted = output.len(); // ExpertResponse rows = panel_size

    // Loss:        actual != expected (a distinct paper_id never reaches output).
    // Duplication: emitted != actual  (two rows share one expert_id, inflating
    //              panel_size while distinct-count stays equal to expected).
    // Both corrupt panel_size, so either condition must fire — but they carry
    // different proving fields, so they return distinct variants.

    // Loss branch: the dropped set is the proving field. It is only meaningful
    // here — on the duplication branch every input id is present in the output,
    // so the dropped set would always be empty (IN-01).
    if actual != expected {
        let dropped: Vec<&str> = input_ids
            .iter()
            .filter(|id| !output_ids.contains(id.as_str()))
            .map(String::as_str)
            .collect();
        tracing::error!(
            expected,
            actual,
            emitted,
            ?dropped,
            "CONSERVATION VIOLATION: source_id loss at adapter boundary"
        );
        return Err(AdapterError::SourceIdLoss { expected, actual });
    }

    // Duplication branch: `emitted` is the proving field — distinct-id count is
    // correct (`actual == expected`) but more rows were emitted.
    if emitted != actual {
        tracing::error!(
            expected,
            actual,
            emitted,
            "CONSERVATION VIOLATION: source_id duplication at adapter boundary"
        );
        return Err(AdapterError::SourceIdDuplication {
            expected,
            actual,
            emitted,
        });
    }

    Ok(())
}

// ── SourceRef (local test fixture type) ──────────────────────────────────────

/// Minimal local fixture type for `deduplicate_sources` testing.
///
/// The real consensus `SourceReference` arrives in Phase 23.  This struct
/// pins the dedup keying contract (`source_id` + `quote`) independently of
/// that import so the contract is proven before the integration wires up.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SourceRef {
    pub source_id: String,
    pub quote: String,
}

// ── deduplicate_sources ───────────────────────────────────────────────────────

/// Mirror of `consensus/src/consensus/domain/utils.py:34-43`.
///
/// Strips graph-node compound suffixes from `source_id` (`"arxiv:X_c003"` →
/// `"arxiv:X"`) and keys on `(base_id, norm_quote)` where `norm_quote` is
/// the first 200 lowercased characters of the quote.  Preserves first-seen
/// order.
///
/// ## Suffix-strip rule
///
/// `"arxiv:…"` / `"s2:…"` paper IDs contain no `"_c"` substring, so they
/// pass through unchanged.  The strip fires only on graph-node IDs such as
/// `"arxiv:2105.00001_c003"` produced during the Phase 23 extraction stage.
///
/// ## Why mirror utils.py exactly
///
/// Any deviation shifts agreement counts vs the consensus oracle (Pitfall 4).
/// The dedup keying is the identity contract between the adapter and the
/// consensus diffusion engine.  The casing mirror is faithful for ASCII and
/// all realistic abstract text; the only deviations are ~55 rare codepoints
/// that differ by Unicode runtime VERSION (not mapping policy) — see the
/// inline note on `norm_quote` (WR-03).
pub(crate) fn deduplicate_sources(sources: &[SourceRef]) -> Vec<SourceRef> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut result: Vec<SourceRef> = Vec::new();

    for s in sources {
        // Strip the FIRST "_c" graph-node suffix.
        // "arxiv:2105.00001_c003" → "arxiv:2105.00001"
        // "arxiv:2105.00001"      → "arxiv:2105.00001" (no-op, no "_c")
        let base_id = if s.source_id.contains("_c") {
            s.source_id
                .split("_c")
                .next()
                .unwrap_or(&s.source_id)
                .to_string()
        } else {
            s.source_id.clone()
        };

        // Normalise quote: strip surrounding whitespace, lowercase, first 200
        // chars — mirroring `(s.quote or "").strip().lower()[:200]` (utils.py:38)
        // in that exact order (strip → lower → truncate).
        //
        // ## Casing fidelity vs the Python oracle (WR-03)
        //
        // A direct codepoint-by-codepoint scan of U+0000..=U+10FFFF shows Rust
        // `str::to_lowercase` and CPython `str.lower()` agree for ALL ASCII and
        // for every character realistically present in arxiv / S2 abstract text.
        // Note `to_lowercase` is lowercase mapping, NOT case folding: German `ß`
        // stays `ß` (it does not expand to `ss`), and capital sharp-S U+1E9E
        // maps to `ß` in both runtimes — the two agree.
        //
        // The only divergences are ~55 rare codepoints (e.g. the Cyrillic
        // U+1C89/U+1C8A pair) where Rust's bundled Unicode tables and CPython's
        // differ by Unicode VERSION, not by simple-vs-full mapping policy. These
        // characters never appear in scientific abstracts; the divergence is a
        // runtime-version skew, accepted and pinned by
        // `dedup_lowercase_matches_python_oracle_for_ascii` below.
        let norm_quote = s.quote.trim().to_lowercase();
        let norm_quote: String = norm_quote.chars().take(200).collect();

        let key = (base_id, norm_quote);
        if seen.insert(key) {
            result.push(s.clone());
        }
    }

    result
}

// ── ADAPT-02 pure-function tests ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{AdapterError, ExpertResponse, ResponseProvenance, SourceId};

    fn make_expert_response(paper_id: &str) -> ExpertResponse {
        let sid = SourceId::new(paper_id);
        ExpertResponse {
            expert_id: sid.clone(),
            prose: "test prose".to_string(),
            provenance: ResponseProvenance {
                source_id: sid,
                title: "Test title".to_string(),
                year: None,
                authors: vec![],
                credibility_tier: alzina_search::CredibilityTier::Unknown,
            },
        }
    }

    /// conservation_assert returns Err(SourceIdLoss) when the output covers
    /// fewer distinct expert_ids than the input had distinct paper_ids.
    #[test]
    fn conservation_assert_fires_on_loss() {
        let input_ids: HashSet<String> = [
            "arxiv:2401.00001",
            "arxiv:2401.00002",
            "arxiv:2401.00003",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        // Output covers only 2 of the 3 papers — one is lost.
        let output = vec![
            make_expert_response("arxiv:2401.00001"),
            make_expert_response("arxiv:2401.00002"),
        ];

        let result = conservation_assert(&input_ids, &output);
        assert!(
            matches!(
                result,
                Err(AdapterError::SourceIdLoss { expected: 3, actual: 2 })
            ),
            "expected Err(SourceIdLoss {{ expected: 3, actual: 2 }}), got: {result:?}"
        );
    }

    /// conservation_assert returns Err(SourceIdDuplication) when the output
    /// contains two ExpertResponse rows for the same expert_id — duplication
    /// inflates panel_size (emitted=3) above the distinct-id count (actual=2).
    /// This pins the duplication half of the contract that WR-01 exposed as
    /// unguarded, and asserts `emitted` is carried on the error (WR-05) — the
    /// field that proves duplication, which the old SourceIdLoss return dropped.
    #[test]
    fn conservation_assert_fires_on_duplication() {
        let input_ids: HashSet<String> = ["arxiv:1", "arxiv:2"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Two responses for the same paper — panel_size inflated to 3 vs 2
        // distinct papers.  Distinct expert_ids still equals expected (2), so
        // a set-cardinality-only check would pass silently.
        let output = vec![
            make_expert_response("arxiv:1"),
            make_expert_response("arxiv:1"),
            make_expert_response("arxiv:2"),
        ];

        let result = conservation_assert(&input_ids, &output);
        assert!(
            matches!(
                result,
                Err(AdapterError::SourceIdDuplication {
                    expected: 2,
                    actual: 2,
                    emitted: 3,
                })
            ),
            "expected Err(SourceIdDuplication {{ expected: 2, actual: 2, emitted: 3 }}) \
             on duplicated expert_id (emitted is the proving field, WR-05), got: {result:?}"
        );
    }

    /// SourceId::try_new rejects an empty string.
    #[test]
    fn source_id_rejects_empty() {
        let result = SourceId::try_new("");
        assert!(
            matches!(result, Err(AdapterError::EmptySourceId)),
            "expected Err(EmptySourceId) for empty string, got: {result:?}"
        );
    }

    /// deduplicate_sources:
    ///   - two SourceRef with compound ids "arxiv:X_c001" and "arxiv:X_c002"
    ///     sharing the same first-200-lowercased quote collapse to ONE entry
    ///   - a bare "arxiv:X" with a different quote is kept separate
    ///   - whitespace/case differences in the first 200 quote chars do NOT
    ///     create a second entry
    #[test]
    fn dedup_strips_compound_suffix() {
        let shared_quote = "This is the same underlying sentence for dedup testing.";
        let compound_a = SourceRef {
            source_id: "arxiv:2105.00001_c001".to_string(),
            quote: shared_quote.to_string(),
        };
        let compound_b = SourceRef {
            source_id: "arxiv:2105.00001_c002".to_string(),
            // Same quote but with extra leading whitespace and different case
            // — must still dedup to ONE.
            quote: format!("  {}  ", shared_quote.to_uppercase()),
        };
        let bare = SourceRef {
            source_id: "arxiv:2105.00001".to_string(),
            // Different quote content — kept separate.
            quote: "A completely different sentence not present above.".to_string(),
        };
        // Also include a bare id with the same quote as compound_a/_b to confirm
        // the base_id key matters: bare with same quote should collapse with compound.
        let bare_same_quote = SourceRef {
            source_id: "arxiv:2105.00001".to_string(),
            quote: shared_quote.to_string(),
        };

        let sources = vec![
            compound_a.clone(),
            compound_b.clone(),
            bare.clone(),
            bare_same_quote,
        ];
        let deduped = deduplicate_sources(&sources);

        // compound_a and compound_b share (base_id="arxiv:2105.00001", norm_quote=shared)
        // → collapse to 1.
        // bare has a different quote → kept.
        // bare_same_quote shares (base_id="arxiv:2105.00001", norm_quote=shared) with
        // compound_a → also collapses.
        // Expected: 2 entries (compound_a first-seen, then bare with different quote).
        assert_eq!(
            deduped.len(),
            2,
            "two compound ids + bare with same quote all share the same (base_id, norm_quote) key \
             and should collapse to 1; bare with different quote is kept; total = 2; got: {deduped:?}"
        );

        // First-seen order: compound_a was inserted first.
        assert_eq!(deduped[0], compound_a, "first entry must be compound_a (first-seen)");
        assert_eq!(deduped[1], bare, "second entry must be bare (different quote)");

        // Confirm the _c split fires: "arxiv:2105.00001_c001" → base "arxiv:2105.00001"
        // (tested implicitly: if the split did NOT fire, compound_a and compound_b
        // would have different base_ids and would NOT collapse — the len==2 assertion
        // above would fail as len would be 3 or 4).
    }

    /// WR-03: pin the casing mirror against the CPython `str.lower()` oracle for
    /// ASCII and common non-ASCII characters that DO appear in abstract text.
    ///
    /// Each `(input, expected_python_lower)` pair was generated by CPython
    /// `str.lower()` (Unicode 15.1).  These are the realistic cases — accented
    /// Latin, German sharp-S, Greek including word-final sigma.  Rust
    /// `to_lowercase` must agree, proving the dedup key is oracle-faithful for
    /// real scientific text.  (The ~55 rare Unicode-version-skew divergences
    /// documented on `deduplicate_sources` are deliberately NOT exercised here:
    /// they never occur in abstracts.)
    #[test]
    fn dedup_lowercase_matches_python_oracle_for_ascii() {
        // (input, CPython str.lower() output)
        let oracle: &[(&str, &str)] = &[
            ("HELLO World", "hello world"),
            ("Café RÉSUMÉ", "café résumé"),
            // German sharp-S: lowercase mapping leaves it unchanged in BOTH
            // runtimes — it does NOT expand to "ss".
            ("STRAßE", "straße"),
            // Capital sharp-S U+1E9E lowercases to ß in both runtimes.
            ("STRA\u{1E9E}E", "straße"),
            // Greek with a word-final sigma; both map ΟΔΟΣ → οδος.
            ("\u{039F}\u{0394}\u{039F}\u{03A3}", "\u{03BF}\u{03B4}\u{03BF}\u{03C2}"),
            ("Naïve DÉJÀ vu", "naïve déjà vu"),
        ];

        for (input, expected) in oracle {
            let got = input.trim().to_lowercase();
            assert_eq!(
                got, *expected,
                "Rust to_lowercase({input:?}) = {got:?}, oracle str.lower() = {expected:?}; \
                 casing mirror diverged from the consensus oracle (WR-03)"
            );
        }

        // And prove the divergence is exercised through deduplicate_sources:
        // two quotes differing only by case collapse to one entry (same key).
        let sources = vec![
            SourceRef {
                source_id: "arxiv:1".to_string(),
                quote: "Café RÉSUMÉ".to_string(),
            },
            SourceRef {
                source_id: "arxiv:1".to_string(),
                quote: "café résumé".to_string(),
            },
        ];
        let deduped = deduplicate_sources(&sources);
        assert_eq!(
            deduped.len(),
            1,
            "case-only-different quotes must share a normalised key and collapse; got: {deduped:?}"
        );
    }
}
