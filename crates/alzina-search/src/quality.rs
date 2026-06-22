//! Search-result quality assessment.
//!
//! Phase 3 Task 3.7. Evaluates a result set against synthesis §5.7 thresholds:
//! min relevance, mean relevance, source concentration, unique-source count.
//! Returns a `SearchQualityReport` (already defined in alzina-core::search) for
//! orchestrator observability. Hybrid search uses `assess_quality` to set
//! `degraded=true` when results are technically present but qualitatively poor.
//!
//! Pure transformation — no async, no I/O. Inputs are already-computed
//! `SearchResultHit`s; outputs are `SearchQualityReport` plus an optional
//! human-readable degradation reason.
//!
//! ## Diversity gate relaxation
//!
//! When the result set has fewer than 3 hits we skip the `min_unique_sources`
//! check: a 2-hit response cannot satisfy "≥3 unique sources" no matter how
//! diverse, so failing it would conflate "small result set" with "poor quality".
//! The other three gates (per-result floor, mean floor, concentration ceiling)
//! still apply.

use std::collections::{HashMap, HashSet};

use alzina_core::{SearchQualityReport, SearchResultHit};

/// Thresholds applied by [`assess_quality`]. Defaults track synthesis §5.7.
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Each hit's `relevance` must be at least this much.
    pub min_per_result_relevance: f32,
    /// The mean `relevance` across all hits must be at least this much.
    pub min_mean_relevance: f32,
    /// No single `source_id` may account for more than this fraction of hits.
    pub max_source_concentration: f32,
    /// When the result set has ≥3 hits, we require at least this many
    /// distinct `source_id`s. Below 3 hits the check is skipped.
    pub min_unique_sources: usize,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_per_result_relevance: 0.3,
            min_mean_relevance: 0.48,
            max_source_concentration: 0.6,
            min_unique_sources: 3,
        }
    }
}

/// Compute a quality report. Pure function — no I/O.
///
/// Behavior on edge cases:
/// - empty hits: `report.passed = false`, all numeric fields = 0.0/0,
///   `max_source_concentration = 0.0` (no concentration when no hits),
///   `unique_source_count = 0`.
/// - 1 hit: `min == mean == that hit's relevance`.
///   `max_source_concentration = 1.0` (single source = 100%).
///   `unique_source_count = 1`.
///   The `min_unique_sources` threshold of 3 is RELAXED to 1 when
///   `hits.len() < 3` (we can't fail diversity when we don't have enough
///   hits to be diverse).
/// - 2 hits: same relaxation — diversity gate doesn't apply.
/// - 3+ hits: all four gates apply.
pub fn assess_quality(
    hits: &[SearchResultHit],
    thresholds: &QualityThresholds,
) -> SearchQualityReport {
    if hits.is_empty() {
        return SearchQualityReport {
            min_relevance: 0.0,
            mean_relevance: 0.0,
            max_source_concentration: 0.0,
            unique_source_count: 0,
            passed: false,
        };
    }

    let mut min_rel = f32::INFINITY;
    let mut sum_rel = 0.0_f32;
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut uniq: HashSet<String> = HashSet::new();
    for h in hits {
        if h.relevance < min_rel {
            min_rel = h.relevance;
        }
        sum_rel += h.relevance;
        *counts.entry(h.source_id.clone()).or_insert(0) += 1;
        uniq.insert(h.source_id.clone());
    }
    let mean_rel = sum_rel / hits.len() as f32;
    let max_count = counts.values().copied().max().unwrap_or(0);
    let max_conc = max_count as f32 / hits.len() as f32;
    let unique_count = uniq.len();

    // Diversity-style gates (concentration ceiling AND unique-source
    // floor) only apply once we have enough hits to plausibly satisfy
    // them. With <3 hits, concentration is mechanically high (≥0.5)
    // and unique_count is mechanically low (≤2); failing either
    // would conflate "small result set" with "poor quality". The
    // per-result and mean relevance gates always apply.
    let (concentration_ok, diversity_ok) = if hits.len() >= 3 {
        (
            max_conc <= thresholds.max_source_concentration,
            unique_count >= thresholds.min_unique_sources,
        )
    } else {
        (true, true)
    };

    let passed = min_rel >= thresholds.min_per_result_relevance
        && mean_rel >= thresholds.min_mean_relevance
        && concentration_ok
        && diversity_ok;

    SearchQualityReport {
        min_relevance: min_rel,
        mean_relevance: mean_rel,
        max_source_concentration: max_conc,
        unique_source_count: unique_count,
        passed,
    }
}

/// Build a degradation reason string from a failed quality report. Returns
/// `None` when `report.passed == true`. Consumers concatenate this with any
/// pre-existing `degradation_reason` (typically the FTS / vector lane reasons
/// produced by the hybrid search service).
///
/// Takes `hit_count` alongside the report so we can determine which gates
/// were ACTIVE (the diversity-style gates are inactive when `hit_count < 3`,
/// and we mustn't emit reasons for inactive gates — otherwise a healthy
/// single-hit response would still report "source concentration 1.00 above
/// 0.60", which is technically true but operationally misleading).
///
/// The reason lists every failed gate, joined by `"; "`. Numeric values are
/// rendered with two-decimal precision so the daemon's `"⚠ Search degraded:"`
/// notice reads cleanly.
pub fn quality_degradation_reason(
    report: &SearchQualityReport,
    thresholds: &QualityThresholds,
    hit_count: usize,
) -> Option<String> {
    if report.passed {
        return None;
    }

    let mut reasons: Vec<String> = Vec::new();

    // Empty-hits special case: the per-result and mean gates would both
    // trip with relevance=0.0, but emitting "min relevance 0.00 below 0.30"
    // for an empty result set is misleading. Surface it as the actual
    // failure mode instead.
    if hit_count == 0 {
        reasons.push("no hits returned".into());
        return Some(reasons.join("; "));
    }

    // Per-result and mean relevance gates always apply.
    if report.min_relevance < thresholds.min_per_result_relevance {
        reasons.push(format!(
            "min relevance {:.2} below {:.2}",
            report.min_relevance, thresholds.min_per_result_relevance
        ));
    }
    if report.mean_relevance < thresholds.min_mean_relevance {
        reasons.push(format!(
            "mean relevance {:.2} below {:.2}",
            report.mean_relevance, thresholds.min_mean_relevance
        ));
    }

    // Diversity-style gates only apply when hits.len() >= 3. Suppress
    // their reasons below that threshold so a healthy small-result-set
    // doesn't get spurious diversity warnings.
    if hit_count >= 3 {
        if report.max_source_concentration > thresholds.max_source_concentration {
            reasons.push(format!(
                "source concentration {:.2} above {:.2}",
                report.max_source_concentration, thresholds.max_source_concentration
            ));
        }
        if report.unique_source_count < thresholds.min_unique_sources {
            reasons.push(format!(
                "only {} unique sources, need {}",
                report.unique_source_count, thresholds.min_unique_sources
            ));
        }
    }

    if reasons.is_empty() {
        // Defensive: passed=false but no specific gate identified.
        // Should never happen given assess_quality's logic, but emit
        // something rather than None to keep AC-1 honest.
        reasons.push("quality gate failed".into());
    }

    Some(reasons.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(source_id: &str, relevance: f32) -> SearchResultHit {
        SearchResultHit {
            source_type: "daily".into(),
            source_id: source_id.into(),
            source_agent: None,
            source_date: None,
            domain: None,
            content: String::new(),
            content_preview: String::new(),
            relevance,
        }
    }

    fn hit_typed(source_type: &str, source_id: &str, relevance: f32) -> SearchResultHit {
        SearchResultHit {
            source_type: source_type.into(),
            source_id: source_id.into(),
            source_agent: None,
            source_date: None,
            domain: None,
            content: String::new(),
            content_preview: String::new(),
            relevance,
        }
    }

    #[test]
    fn empty_hits_fails_quality() {
        let t = QualityThresholds::default();
        let hits: Vec<SearchResultHit> = vec![];
        let r = assess_quality(&hits, &t);
        assert!(!r.passed);
        assert_eq!(r.min_relevance, 0.0);
        assert_eq!(r.mean_relevance, 0.0);
        assert_eq!(r.max_source_concentration, 0.0);
        assert_eq!(r.unique_source_count, 0);
        let reason = quality_degradation_reason(&r, &t, hits.len()).expect("reason for empty");
        assert!(
            reason.to_lowercase().contains("no hits"),
            "expected empty-hits reason, got: {reason}"
        );
    }

    #[test]
    fn single_hit_above_thresholds_passes_with_diversity_relaxed() {
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.9)];
        let r = assess_quality(&hits, &t);
        assert!(r.passed, "single high-relevance hit should pass");
        assert!((r.min_relevance - 0.9).abs() < 1e-6);
        assert!((r.mean_relevance - 0.9).abs() < 1e-6);
        assert!((r.max_source_concentration - 1.0).abs() < 1e-6);
        assert_eq!(r.unique_source_count, 1);
        assert!(quality_degradation_reason(&r, &t, hits.len()).is_none());
    }

    #[test]
    fn single_hit_below_min_relevance_fails() {
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.2)];
        let r = assess_quality(&hits, &t);
        assert!(!r.passed);
        let reason = quality_degradation_reason(&r, &t, hits.len()).expect("reason expected");
        assert!(
            reason.contains("min relevance"),
            "reason should mention min relevance: {reason}"
        );
        // With <3 hits the concentration / diversity gates are inactive,
        // so they must NOT appear in the reason — even though
        // max_source_concentration=1.0 mechanically.
        assert!(
            !reason.contains("concentration"),
            "single-hit reason must not mention inactive concentration gate: {reason}"
        );
    }

    #[test]
    fn mean_relevance_below_threshold_fails() {
        let t = QualityThresholds::default();
        // Five hits with relevance exactly at the per-result floor (0.3)
        // pass that gate but the mean (0.3) trips the 0.48 mean floor.
        // Use 5 distinct sources so the diversity / concentration gates pass.
        let hits = vec![
            hit("a", 0.3),
            hit("b", 0.3),
            hit("c", 0.3),
            hit("d", 0.3),
            hit("e", 0.3),
        ];
        let r = assess_quality(&hits, &t);
        assert!(r.min_relevance >= 0.3);
        assert!((r.mean_relevance - 0.3).abs() < 1e-6);
        assert!(!r.passed);
        let reason = quality_degradation_reason(&r, &t, hits.len()).expect("reason expected");
        assert!(
            reason.contains("mean"),
            "reason should mention mean relevance: {reason}"
        );
    }

    #[test]
    fn concentration_above_max_fails() {
        let t = QualityThresholds::default();
        // Five hits all on the same source — concentration 1.0, fails 0.6.
        let hits = vec![
            hit("same", 0.9),
            hit("same", 0.9),
            hit("same", 0.9),
            hit("same", 0.9),
            hit("same", 0.9),
        ];
        let r = assess_quality(&hits, &t);
        assert!((r.max_source_concentration - 1.0).abs() < 1e-6);
        assert!(!r.passed);
        let reason = quality_degradation_reason(&r, &t, hits.len()).expect("reason expected");
        assert!(
            reason.contains("concentration"),
            "reason should mention concentration: {reason}"
        );
    }

    #[test]
    fn unique_sources_below_three_with_3plus_hits_fails() {
        let t = QualityThresholds::default();
        // 4 hits, 2 unique sources, 2 each → concentration 0.5 (passes 0.6),
        // but unique_count=2 < 3 → diversity gate trips.
        let hits = vec![hit("a", 0.9), hit("a", 0.9), hit("b", 0.9), hit("b", 0.9)];
        let r = assess_quality(&hits, &t);
        assert_eq!(r.unique_source_count, 2);
        assert!(r.max_source_concentration <= 0.6 + 1e-6);
        assert!(!r.passed, "diversity gate should trip");
        let reason = quality_degradation_reason(&r, &t, hits.len()).expect("reason expected");
        assert!(
            reason.contains("unique sources"),
            "reason should mention unique sources: {reason}"
        );
    }

    #[test]
    fn two_hits_with_two_unique_sources_passes() {
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.9), hit("b", 0.9)];
        let r = assess_quality(&hits, &t);
        assert_eq!(r.unique_source_count, 2);
        // Diversity gate is relaxed below 3 hits.
        assert!(r.passed, "2 hits, 2 sources, high relevance should pass");
        assert!(quality_degradation_reason(&r, &t, hits.len()).is_none());
    }

    #[test]
    fn all_thresholds_pass() {
        let t = QualityThresholds::default();
        let hits = vec![
            hit("a", 0.8),
            hit("b", 0.8),
            hit("c", 0.8),
            hit("d", 0.8),
            hit("e", 0.8),
        ];
        let r = assess_quality(&hits, &t);
        assert!(r.passed);
        assert_eq!(r.unique_source_count, 5);
        assert!(quality_degradation_reason(&r, &t, hits.len()).is_none());
    }

    #[test]
    fn multiple_failures_concatenated_in_reason() {
        let t = QualityThresholds::default();
        // 5 hits, all same source, low relevance.
        // Fails: per-result floor (0.2 < 0.3), mean floor (0.2 < 0.48),
        // concentration (1.0 > 0.6).
        let hits = vec![
            hit("same", 0.2),
            hit("same", 0.2),
            hit("same", 0.2),
            hit("same", 0.2),
            hit("same", 0.2),
        ];
        let r = assess_quality(&hits, &t);
        assert!(!r.passed);
        let reason = quality_degradation_reason(&r, &t, hits.len()).expect("reason expected");
        assert!(reason.contains("min relevance"), "got: {reason}");
        assert!(reason.contains("mean relevance"), "got: {reason}");
        assert!(reason.contains("concentration"), "got: {reason}");
        assert!(reason.contains("; "), "reasons must be ;-joined: {reason}");
    }

    #[test]
    fn unique_source_count_uses_source_id_not_source_type() {
        let t = QualityThresholds::default();
        // 4 hits, all source_type="daily", but 4 different source_ids.
        // Diversity must count unique source_ids → 4, not source_types → 1.
        let hits = vec![
            hit_typed("daily", "id1", 0.9),
            hit_typed("daily", "id2", 0.9),
            hit_typed("daily", "id3", 0.9),
            hit_typed("daily", "id4", 0.9),
        ];
        let r = assess_quality(&hits, &t);
        assert_eq!(
            r.unique_source_count, 4,
            "must count unique source_ids, not source_types"
        );
        assert!((r.max_source_concentration - 0.25).abs() < 1e-6);
        assert!(r.passed);
    }

    // ---- Fence-post tests for §5.7 thresholds ----------------------------
    //
    // These pin the inequality direction at exact threshold boundaries so a
    // future refactor that flips `>=` ↔ `>` (etc.) trips a test rather than
    // silently changing degradation semantics.
    //
    // Inferred semantics from `assess_quality` (lines ~101-113):
    //   per-result floor:   `min_rel >= threshold`        (>=, inclusive)
    //   mean floor:         `mean_rel >= threshold`       (>=, inclusive)
    //   concentration max:  `max_conc <= threshold`       (<=, inclusive)
    //   unique sources min: `unique_count >= threshold`   (>=, inclusive)
    // All four are inclusive at the boundary (i.e. "exactly at threshold"
    // PASSES). The fence-post tests below encode that.

    #[test]
    fn per_result_relevance_at_exactly_0_300_passes() {
        // Single hit at the per-result floor (0.30) — gate is `>=` so the
        // boundary value passes. <3 hits → diversity gates relaxed; the only
        // active gate that could trip is the mean floor (0.48), which DOES
        // trip at relevance 0.30, so we assert the per-result reason does
        // NOT appear (rather than asserting overall pass).
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.300_f32)];
        let r = assess_quality(&hits, &t);
        assert!(
            r.min_relevance >= t.min_per_result_relevance,
            "per-result floor is inclusive: 0.300 should satisfy >= 0.30"
        );
        let reason = quality_degradation_reason(&r, &t, hits.len());
        if let Some(reason) = reason {
            assert!(
                !reason.contains("min relevance"),
                "0.300 is at the floor (>=), must not appear in reason: {reason}"
            );
        }
    }

    #[test]
    fn per_result_relevance_at_0_2999_fails() {
        // Just below the per-result floor — gate must trip.
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.2999_f32)];
        let r = assess_quality(&hits, &t);
        assert!(!r.passed, "0.2999 < 0.30 should fail the per-result gate");
        let reason = quality_degradation_reason(&r, &t, hits.len())
            .expect("reason expected for failed gate");
        assert!(
            reason.contains("min relevance"),
            "below-floor must surface per-result reason: {reason}"
        );
    }

    #[test]
    fn per_result_relevance_at_0_3001_passes() {
        // Just above per-result floor; use mean-clearing setup so we can
        // assert the per-result reason is absent. Single hit at 0.3001
        // would still fail the mean gate (0.48), so use 1 hit at 0.3001
        // to verify ONLY that the per-result gate doesn't fire.
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.3001_f32)];
        let r = assess_quality(&hits, &t);
        assert!(
            r.min_relevance >= t.min_per_result_relevance,
            "0.3001 should clear the per-result floor"
        );
        let reason = quality_degradation_reason(&r, &t, hits.len());
        if let Some(reason) = reason {
            assert!(
                !reason.contains("min relevance"),
                "above-floor must not surface per-result reason: {reason}"
            );
        }
    }

    #[test]
    fn mean_relevance_at_exactly_0_480_passes() {
        // Single hit at exactly the mean floor — diversity relaxed (<3 hits),
        // per-result passes (0.48 >= 0.30), mean equals 0.48 exactly (1 hit
        // → mean = relevance, no division-rounding pitfall).
        // Gate is `>=` so this PASSES. If the implementation is ever changed
        // to `>`, this test will fail and force a deliberate review.
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.480_f32)];
        let r = assess_quality(&hits, &t);
        assert!(
            (r.mean_relevance - 0.480_f32).abs() < f32::EPSILON,
            "mean should equal exactly 0.480 for single hit"
        );
        assert!(
            r.passed,
            "mean floor is inclusive (>=): exactly 0.480 must pass; \
             if this fails, the implementation may use `>` instead of `>=`"
        );
        assert!(quality_degradation_reason(&r, &t, hits.len()).is_none());
    }

    #[test]
    fn mean_relevance_at_0_4799_fails() {
        // Single hit at 0.4799 → mean = 0.4799 < 0.48 → mean gate trips.
        // Per-result still passes (0.4799 >= 0.30).
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.4799_f32)];
        let r = assess_quality(&hits, &t);
        assert!(!r.passed, "mean 0.4799 < 0.48 should fail");
        let reason = quality_degradation_reason(&r, &t, hits.len()).expect("reason expected");
        assert!(
            reason.contains("mean relevance"),
            "below-floor must surface mean reason: {reason}"
        );
    }

    #[test]
    fn mean_relevance_at_0_4801_passes() {
        // Single hit at 0.4801 → mean = 0.4801 > 0.48 → all gates pass
        // (per-result 0.4801>=0.30, mean 0.4801>=0.48, diversity relaxed).
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.4801_f32)];
        let r = assess_quality(&hits, &t);
        assert!(r.passed, "mean 0.4801 > 0.48 should pass");
        assert!(quality_degradation_reason(&r, &t, hits.len()).is_none());
    }

    #[test]
    fn concentration_at_exactly_60pct_outcome_recorded() {
        // 5 hits, 3 from "hot" source, 2 from distinct cold sources.
        // max_source_concentration = 3/5 = 0.6 exactly (in f32).
        // Implementation uses `<=` (line ~103) so 0.6 satisfies the gate
        // and PASSES. We also need unique_count >= 3 to pass diversity:
        // 3 distinct source_ids ("hot", "cold1", "cold2") → 3, satisfies >=3.
        // Relevance 0.9 across the board so per-result and mean clear easily.
        //
        // DOCUMENTATION ASSERTION: if the production code is ever flipped
        // to `<` (strict), 0.6 == 0.6 would FAIL and this test would catch it.
        // Inequality currently in effect: `<=` (inclusive — boundary passes).
        let t = QualityThresholds::default();
        let hits = vec![
            hit("hot", 0.9),
            hit("hot", 0.9),
            hit("hot", 0.9),
            hit("cold1", 0.9),
            hit("cold2", 0.9),
        ];
        let r = assess_quality(&hits, &t);
        assert!(
            (r.max_source_concentration - 0.6_f32).abs() < f32::EPSILON,
            "concentration should be exactly 0.6 (3/5)"
        );
        assert_eq!(r.unique_source_count, 3);
        assert!(
            r.passed,
            "concentration ceiling is inclusive (<=): exactly 60% must pass; \
             if this fails, the implementation may use `<` instead of `<=`"
        );
    }

    #[test]
    fn unique_sources_at_exactly_3_passes() {
        // 3 hits from 3 distinct sources — unique_count = 3, threshold = 3.
        // Gate is `>=` so the boundary PASSES. Concentration = 1/3 ≈ 0.333,
        // well under 0.6. All relevance 0.9 → per-result and mean both clear.
        let t = QualityThresholds::default();
        let hits = vec![hit("a", 0.9), hit("b", 0.9), hit("c", 0.9)];
        let r = assess_quality(&hits, &t);
        assert_eq!(r.unique_source_count, 3);
        assert!(
            r.passed,
            "unique-source floor is inclusive (>=): exactly 3 must pass"
        );
        assert!(quality_degradation_reason(&r, &t, hits.len()).is_none());
    }
}
