//! Fitness weight tables for the three TTD stages.
//!
//! Source: `consensus/src/consensus/diffusion/fitness.py:326-351` [VERIFIED]
//!
//! Each table is a `&[(&str, f32)]` constant — dimension name paired with its
//! weight. The tables must sum to exactly 1.0 (±0.001); this is enforced by
//! the inline tests below AND at `TtdMachine` construction via
//! `fitness::check_weight_sum`.
//!
//! ## Critical constraint
//!
//! Do NOT reuse one table across all stages (Pitfall 1). Each stage has its own
//! dimension set and weight distribution. The graph stage uses groundedness as
//! the highest-weighted dimension (0.30); the synthesis stage uses faithfulness
//! (0.25); the narrative stage uses structural_clarity (0.25).

// ── Graph stage ───────────────────────────────────────────────────────────────

/// Graph stage fitness weight table (fitness.py:326-333).
///
/// Validity gate: groundedness ≥ 4 (is_valid_graph).
pub const GRAPH_WEIGHTS: &[(&str, f32)] = &[
    ("groundedness",         0.30),
    ("coverage",             0.25),
    ("atomicity",            0.15),
    ("non_redundancy",       0.10),
    ("relation_coherence",   0.10),
    ("dissent_preservation", 0.10),
];

// ── Synthesis stage ───────────────────────────────────────────────────────────

/// Synthesis stage fitness weight table (fitness.py:335-342).
///
/// Validity gate: faithfulness ≥ 4 (is_valid_synthesis).
pub const SYNTHESIS_WEIGHTS: &[(&str, f32)] = &[
    ("faithfulness",       0.25),
    ("completeness",       0.20),
    ("traceability",       0.15),
    ("neutrality",         0.15),
    ("dissent_visibility", 0.15),
    ("structural_clarity", 0.10),
];

// ── Narrative stage ───────────────────────────────────────────────────────────

/// Narrative stage fitness weight table (fitness.py:344-351).
///
/// Narrative reuses synthesis dimension names but with different weights.
/// Validity gate: faithfulness ≥ 4 (is_valid_synthesis — same predicate).
pub const NARRATIVE_WEIGHTS: &[(&str, f32)] = &[
    ("structural_clarity",  0.25),
    ("dissent_visibility",  0.20),
    ("faithfulness",        0.15),
    ("completeness",        0.15),
    ("neutrality",          0.15),
    ("traceability",        0.10),
];

// ── v2 weight tables (NEW) ────────────────────────────────────────────────────

/// v2 graph stage fitness weight table.
///
/// Faithfulness 0.30 (highest, mirrors v1's groundedness as the grounding anchor).
/// Five dims: faithfulness, coverage, tension_visibility, lineage_clarity, recency_balance.
/// All three v2 tables are distinct from each other and from the three v1 tables.
pub const V2_GRAPH_WEIGHTS: &[(&str, f32)] = &[
    ("faithfulness",       0.30),
    ("coverage",           0.25),
    ("tension_visibility", 0.15),
    ("lineage_clarity",    0.15),
    ("recency_balance",    0.15),
];

/// v2 synthesis stage fitness weight table.
///
/// Faithfulness 0.30 (highest — grounding matters most at the synthesis stage).
/// tension_visibility 0.20 (synthesis is where contradictions must be resolved or surfaced).
pub const V2_SYNTHESIS_WEIGHTS: &[(&str, f32)] = &[
    ("faithfulness",       0.30),
    ("coverage",           0.20),
    ("tension_visibility", 0.20),
    ("lineage_clarity",    0.15),
    ("recency_balance",    0.15),
];

/// v2 narrative stage fitness weight table.
///
/// Narrative weights presentation dims highest, mirroring v1 narrative's
/// structural_clarity 0.25 top slot — lineage_clarity 0.25 plays the equivalent
/// role in the v2 lit-review context (how the narrative traces intellectual lineage).
pub const V2_NARRATIVE_WEIGHTS: &[(&str, f32)] = &[
    ("lineage_clarity",    0.25),
    ("tension_visibility", 0.20),
    ("faithfulness",       0.20),
    ("coverage",           0.20),
    ("recency_balance",    0.15),
];

// ── Rubric-encoding Phase 1: plan-conformance narrative table (W-e714abb4) ───

/// Narrative weight table used when a winning `ReviewPlan` is injected into
/// Stage 3 (plan_mode != Disabled). Six dims: the five `V2_NARRATIVE_WEIGHTS`
/// dims plus `plan_conformance` — the judge that verifies the draft against
/// the WINNING PLAN (archetype, sections, term registry, planted threads),
/// not against the judge's own taste (spec §6: "draft-against-plan, not
/// draft-against-taste").
///
/// plan_conformance takes the top slot (0.25): once a plan has won the
/// tournament, divergence from it re-opens the consensus-mush channel the
/// tournament exists to close. The five v2 dims keep their relative order.
/// Selected by `NarrativeEvalFitness::weights()` ONLY when a plan is present;
/// plan-absent runs keep `V2_NARRATIVE_WEIGHTS` byte-for-byte.
pub const V3_PLANNED_NARRATIVE_WEIGHTS: &[(&str, f32)] = &[
    ("plan_conformance",   0.25),
    ("lineage_clarity",    0.20),
    ("tension_visibility", 0.15),
    ("faithfulness",       0.15),
    ("coverage",           0.15),
    ("recency_balance",    0.10),
];

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(weights: &[(&str, f32)]) -> f32 {
        weights.iter().map(|(_, w)| w).sum()
    }

    /// ENGINE-03: each weight table sums to 1.0 ± 0.001.
    #[test]
    fn graph_weights_sum_to_one() {
        let s = sum(GRAPH_WEIGHTS);
        assert!(
            (s - 1.0).abs() < 0.001,
            "GRAPH_WEIGHTS sum {s} ≠ 1.0 (±0.001)"
        );
    }

    #[test]
    fn synthesis_weights_sum_to_one() {
        let s = sum(SYNTHESIS_WEIGHTS);
        assert!(
            (s - 1.0).abs() < 0.001,
            "SYNTHESIS_WEIGHTS sum {s} ≠ 1.0 (±0.001)"
        );
    }

    #[test]
    fn narrative_weights_sum_to_one() {
        let s = sum(NARRATIVE_WEIGHTS);
        assert!(
            (s - 1.0).abs() < 0.001,
            "NARRATIVE_WEIGHTS sum {s} ≠ 1.0 (±0.001)"
        );
    }

    /// Verify no two stages share the same weight table (Pitfall 1 guard).
    #[test]
    fn weight_tables_are_distinct() {
        // Compare by pointer — they're static constants so same-pointer means same table
        assert_ne!(
            GRAPH_WEIGHTS.as_ptr(),
            SYNTHESIS_WEIGHTS.as_ptr(),
            "GRAPH_WEIGHTS must be a distinct constant from SYNTHESIS_WEIGHTS"
        );
        assert_ne!(
            GRAPH_WEIGHTS.as_ptr(),
            NARRATIVE_WEIGHTS.as_ptr(),
            "GRAPH_WEIGHTS must be a distinct constant from NARRATIVE_WEIGHTS"
        );
        assert_ne!(
            SYNTHESIS_WEIGHTS.as_ptr(),
            NARRATIVE_WEIGHTS.as_ptr(),
            "SYNTHESIS_WEIGHTS must be a distinct constant from NARRATIVE_WEIGHTS"
        );
    }

    /// Verify each table has exactly 6 dimensions (matches the 6-judge fitness eval).
    #[test]
    fn each_table_has_six_dimensions() {
        assert_eq!(GRAPH_WEIGHTS.len(), 6, "graph table must have 6 dims");
        assert_eq!(SYNTHESIS_WEIGHTS.len(), 6, "synthesis table must have 6 dims");
        assert_eq!(NARRATIVE_WEIGHTS.len(), 6, "narrative table must have 6 dims");
    }

    /// Verify graph table top dim is groundedness and synthesis/narrative is faithfulness/structural_clarity.
    #[test]
    fn top_weighted_dim_per_stage() {
        assert_eq!(GRAPH_WEIGHTS[0].0, "groundedness");
        assert_eq!(SYNTHESIS_WEIGHTS[0].0, "faithfulness");
        assert_eq!(NARRATIVE_WEIGHTS[0].0, "structural_clarity");
    }

    // ── v2 weight table tests ─────────────────────────────────────────────────

    #[test]
    fn v2_graph_weights_sum_to_one() {
        let s = sum(V2_GRAPH_WEIGHTS);
        assert!((s - 1.0).abs() < 0.001, "V2_GRAPH_WEIGHTS sum {s} ≠ 1.0 (±0.001)");
    }

    #[test]
    fn v2_synthesis_weights_sum_to_one() {
        let s = sum(V2_SYNTHESIS_WEIGHTS);
        assert!((s - 1.0).abs() < 0.001, "V2_SYNTHESIS_WEIGHTS sum {s} ≠ 1.0 (±0.001)");
    }

    #[test]
    fn v2_narrative_weights_sum_to_one() {
        let s = sum(V2_NARRATIVE_WEIGHTS);
        assert!((s - 1.0).abs() < 0.001, "V2_NARRATIVE_WEIGHTS sum {s} ≠ 1.0 (±0.001)");
    }

    #[test]
    fn v2_tables_have_five_dims() {
        assert_eq!(V2_GRAPH_WEIGHTS.len(), 5, "v2 graph must have 5 dims");
        assert_eq!(V2_SYNTHESIS_WEIGHTS.len(), 5, "v2 synthesis must have 5 dims");
        assert_eq!(V2_NARRATIVE_WEIGHTS.len(), 5, "v2 narrative must have 5 dims");
    }

    #[test]
    fn v2_tables_all_contain_faithfulness() {
        let has = |t: &[(&str, f32)]| t.iter().any(|(k, _)| *k == "faithfulness");
        assert!(has(V2_GRAPH_WEIGHTS), "V2_GRAPH_WEIGHTS must contain faithfulness");
        assert!(has(V2_SYNTHESIS_WEIGHTS), "V2_SYNTHESIS_WEIGHTS must contain faithfulness");
        assert!(has(V2_NARRATIVE_WEIGHTS), "V2_NARRATIVE_WEIGHTS must contain faithfulness");
    }

    #[test]
    fn v2_tables_are_distinct_from_each_other_and_v1() {
        // Pointer-distinct: same pointer → same table (reuse bug)
        assert_ne!(V2_GRAPH_WEIGHTS.as_ptr(), V2_SYNTHESIS_WEIGHTS.as_ptr());
        assert_ne!(V2_GRAPH_WEIGHTS.as_ptr(), V2_NARRATIVE_WEIGHTS.as_ptr());
        assert_ne!(V2_SYNTHESIS_WEIGHTS.as_ptr(), V2_NARRATIVE_WEIGHTS.as_ptr());
        // Also distinct from v1 tables
        assert_ne!(V2_GRAPH_WEIGHTS.as_ptr(), GRAPH_WEIGHTS.as_ptr());
        assert_ne!(V2_SYNTHESIS_WEIGHTS.as_ptr(), SYNTHESIS_WEIGHTS.as_ptr());
        assert_ne!(V2_NARRATIVE_WEIGHTS.as_ptr(), NARRATIVE_WEIGHTS.as_ptr());
    }

    // ── Rubric-encoding Phase 1 (W-e714abb4) ─────────────────────────────────

    #[test]
    fn v3_planned_narrative_weights_sum_to_one() {
        let s = sum(V3_PLANNED_NARRATIVE_WEIGHTS);
        assert!((s - 1.0).abs() < 0.001, "V3_PLANNED_NARRATIVE_WEIGHTS sum {s} ≠ 1.0 (±0.001)");
    }

    /// Six dims = the five v2 narrative dims + plan_conformance on top.
    /// plan_conformance MUST lead: the planned table exists to make
    /// plan-divergence expensive at selection time.
    #[test]
    fn v3_planned_narrative_weights_extend_v2_with_plan_conformance() {
        assert_eq!(V3_PLANNED_NARRATIVE_WEIGHTS.len(), 6);
        assert_eq!(V3_PLANNED_NARRATIVE_WEIGHTS[0].0, "plan_conformance");
        for (dim, _) in V2_NARRATIVE_WEIGHTS {
            assert!(
                V3_PLANNED_NARRATIVE_WEIGHTS.iter().any(|(k, _)| k == dim),
                "planned table must retain v2 dim {dim}"
            );
        }
        // Distinct constant — never aliases the v2 table.
        assert_ne!(V3_PLANNED_NARRATIVE_WEIGHTS.as_ptr(), V2_NARRATIVE_WEIGHTS.as_ptr());
    }
}
