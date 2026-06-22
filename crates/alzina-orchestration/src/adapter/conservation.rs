//! Provenance-conservation guard.
//!
//! `conservation_assert` is pure (no async, no I/O) and has no side effects
//! beyond `tracing::error!` on conservation violation.
//!
//! ## conservation_assert
//!
//! Fires `Err(AdapterError::SourceIdLoss)` when the set of distinct
//! `expert_id` values leaving `build_panel` does not EQUAL the set of
//! distinct `paper_id` values entering it.  Equality is required because
//! both loss and duplication corrupt `panel_size` and therefore every
//! downstream agreement label (consensus / majority / divided / minority).

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
}
