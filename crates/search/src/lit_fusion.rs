//! Three-lane RRF fusion for the literature retrieval pipeline.
//!
//! Phase 21 Plan 04 — Task 1.
//!
//! Fuses internal hybrid hits, arxiv hits, and Semantic Scholar hits into a
//! single ranked list via Reciprocal Rank Fusion (RRF, k=60) and then passes
//! the list through the existing `quality.rs` gate.
//!
//! ## Design decisions
//!
//! - Inlines the `1/(k + rank)` math rather than exposing `hybrid.rs`'s private
//!   `fuse_rrf_with_recency` function (Open Question 1, RESOLVED in
//!   21-RESEARCH.md). `hybrid.rs` is NOT modified.
//! - Dedup is by `(source_type, source_id)` key. Keys `"internal"`, `"arxiv"`,
//!   `"s2"` never collide (21-RESEARCH.md Pattern 6 key constraint).
//! - The quality gate uses `QualityThresholds::default()` (per-result floor
//!   0.3, mean 0.48, concentration 0.6, unique sources 3). The 0.3 floor is
//!   locked for the lit path; do NOT construct a custom threshold or raise to
//!   0.57 before Phase 25 fidelity evidence. (21-CONTEXT.md locked decision.)
//! - Loud-degrade: always return whatever hits exist plus an `Option<String>`
//!   degradation reason. Never hard-fail the dispatch.

use std::collections::HashMap;

use base::SearchResultHit;

use crate::quality::{QualityThresholds, assess_quality, quality_degradation_reason};

/// Canonical Reciprocal Rank Fusion smoothing constant `k` for the three-lane
/// lit-fusion path (synthesis §5.7). Call sites passing `rrf_k` to
/// [`fuse_rrf_three_lane`] should use this rather than a bare `60.0` literal so
/// the tuned value lives in one place. Mirrors `HybridConfig::default().rrf_k`
/// (the hybrid path keeps its own copy by the 21-04-PLAN no-touch decision).
pub const RRF_K_DEFAULT: f32 = 60.0;

// ── Input hit types ───────────────────────────────────────────────────────────

/// A ranked arxiv or ar5iv hit with its cosine similarity to the query.
///
/// `arxiv_id` is the bare arxiv identifier (no URL prefix, no version suffix),
/// e.g. `"2105.14103"`. The `content` and `content_preview` fields hold the
/// raw chunk text and truncated preview respectively.
#[derive(Debug, Clone)]
pub struct ArxivHit {
    pub arxiv_id: String,
    pub title: String,
    pub section: String,
    pub content: String,
    pub content_preview: String,
    pub relevance: f32,
}

/// A ranked Semantic Scholar hit with its cosine similarity to the query.
///
/// `paper_id` is the bare S2 identifier (no `"s2:"` prefix), e.g.
/// `"204e3073870fae3d05bcbc2f6a8e263d9b72e776"`. The `content` and
/// `content_preview` fields hold the abstract text and truncated preview.
#[derive(Debug, Clone)]
pub struct S2Hit {
    pub paper_id: String,
    pub title: String,
    pub content: String,
    pub content_preview: String,
    pub relevance: f32,
}

// ── Output type ───────────────────────────────────────────────────────────────

/// One hit in the fused three-lane result list.
///
/// The `source_type` field is one of `"internal"`, `"arxiv"`, `"s2"`.
/// `content` is the raw chunk text at this point; callers (the HTTP handler)
/// wrap it in `wrap_low_authority` before returning it to agents (T-21-10).
#[derive(Debug, Clone)]
pub struct FusedHit {
    pub source_type: String,
    pub source_id: String,
    pub title: String,
    /// Section heading for arxiv chunks; empty for S2 / internal.
    pub section: Option<String>,
    pub content: String,
    pub content_preview: String,
    /// Normalised RRF score in `[0.0, 1.0]`. Top hit is 1.0.
    pub relevance: f32,
}

// ── Internal ──────────────────────────────────────────────────────────────────

type SourceKey = (String, String);

/// Reciprocal Rank Fusion kernel: the score contribution of a hit at 1-based
/// `rank` under smoothing constant `rrf_k`. Single source for the `1/(k+rank)`
/// math previously inlined once per lane in [`fuse_rrf_three_lane`].
///
/// Note: this is intentionally local to `lit_fusion`. `hybrid.rs`'s
/// `fuse_rrf_with_recency` keeps its own inlined kernel (21-04-PLAN decision:
/// hybrid.rs is not modified) — see `lib.rs`.
#[inline]
fn rrf_delta(rrf_k: f32, rank: usize) -> f32 {
    1.0 / (rrf_k + rank as f32)
}

// Accumulates RRF scoring data for a unique (source_type, source_id) pair.
struct AccRow {
    rrf_score: f32,
    // Payload: whichever lane contributed first wins for display fields.
    title: String,
    section: Option<String>,
    content: String,
    content_preview: String,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Fuse three retrieval lanes via RRF into a single ranked list.
///
/// `rrf_k` is the rank-smoothing constant (use `60.0` per synthesis §5.7).
///
/// The returned hits are sorted by RRF score descending and normalised so the
/// top hit has `relevance = 1.0`. Dedup is by `(source_type, source_id)`.
///
/// ## Key constraint
///
/// `"internal"`, `"arxiv"`, and `"s2"` source keys never collide with each
/// other, so a paper appearing in both the S2 lane and the internal hybrid
/// lane (because it was previously indexed locally) would be treated as TWO
/// separate entries. This is intentional — the fuser's job is to present the
/// best-ranked hits; the UI layer can deduplicate by content if desired.
pub fn fuse_rrf_three_lane(
    internal_hits: &[SearchResultHit],
    arxiv_hits: &[ArxivHit],
    s2_hits: &[S2Hit],
    rrf_k: f32,
) -> Vec<FusedHit> {
    let mut order: Vec<SourceKey> = Vec::new();
    let mut rows: HashMap<SourceKey, AccRow> = HashMap::new();

    // Lane 1: internal hybrid hits (source_type comes from the hit itself).
    for (i, h) in internal_hits.iter().enumerate() {
        let key = (h.source_type.clone(), h.source_id.clone());
        let delta = rrf_delta(rrf_k, i + 1);
        let entry = rows.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            AccRow {
                rrf_score: 0.0,
                title: String::new(),
                section: None,
                content: h.content.clone(),
                content_preview: h.content_preview.clone(),
            }
        });
        entry.rrf_score += delta;
    }

    // Lane 2: arxiv hits — source_type = "arxiv", source_id = arxiv_id.
    for (i, h) in arxiv_hits.iter().enumerate() {
        let key = ("arxiv".to_string(), h.arxiv_id.clone());
        let delta = rrf_delta(rrf_k, i + 1);
        let entry = rows.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            AccRow {
                rrf_score: 0.0,
                title: h.title.clone(),
                section: Some(h.section.clone()),
                content: h.content.clone(),
                content_preview: h.content_preview.clone(),
            }
        });
        entry.rrf_score += delta;
    }

    // Lane 3: S2 hits — source_type = "s2", source_id = paper_id.
    for (i, h) in s2_hits.iter().enumerate() {
        let key = ("s2".to_string(), h.paper_id.clone());
        let delta = rrf_delta(rrf_k, i + 1);
        let entry = rows.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            AccRow {
                rrf_score: 0.0,
                title: h.title.clone(),
                section: None,
                content: h.content.clone(),
                content_preview: h.content_preview.clone(),
            }
        });
        entry.rrf_score += delta;
    }

    if rows.is_empty() {
        return vec![];
    }

    // Build the scored list, preserving insertion order for stable tie-breaking.
    let mut scored: Vec<(SourceKey, f32)> = order
        .iter()
        .map(|k| (k.clone(), rows[k].rrf_score))
        .collect();

    // Sort descending. Stable so ties keep insertion order (FTS/vec/arxiv/s2
    // in that priority) for deterministic test behaviour.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Normalise top → 1.0.
    let max_score = scored.first().map(|(_, s)| *s).unwrap_or(0.0);

    scored
        .into_iter()
        .map(|(key, score)| {
            let row = rows.remove(&key).expect("key in rows");
            let relevance = if max_score > 0.0 {
                score / max_score
            } else {
                0.0
            };
            FusedHit {
                source_type: key.0,
                source_id: key.1,
                title: row.title,
                section: row.section,
                content: row.content,
                content_preview: row.content_preview,
                relevance,
            }
        })
        .collect()
}

/// Apply the quality gate to a fused hit list.
///
/// Takes an optional `lane_degraded` reason string from upstream (e.g. an
/// embedder failure on one of the lanes) and folds it with the gate's own
/// reason via `"; "`.
///
/// ## Return
///
/// `(hits, Option<String>)` — the hits are always returned (loud-degrade,
/// never hard-fail). The `Option<String>` is `Some` when either a lane
/// reported degradation or the quality gate tripped; `None` when all is well.
///
/// ## Intake floor
///
/// Uses `QualityThresholds::default()` — the 0.3 per-result floor is locked
/// for the lit path. Do NOT construct custom thresholds.
pub fn gate_fused(
    fused: Vec<FusedHit>,
    lane_degraded: Option<String>,
) -> (Vec<FusedHit>, Option<String>) {
    // ── F3: relevance floor — filter BEFORE quality assessment and panel ──────
    //
    // Normalised RRF scores (post fuse_rrf_three_lane): 0.3 floor is calibrated
    // on the normalised scale. Use the single QualityThresholds::default() binding
    // below — never construct a second ::default() call in this function.
    //
    // The healthy-no-match suppression (`original_len == 0 && lane_degraded.is_none()`)
    // keys on PRE-filter emptiness: an input that was non-empty but got fully filtered
    // is contamination blocked, not a healthy no-match.
    let thresholds = QualityThresholds::default();
    let original_len = fused.len();
    let fused: Vec<FusedHit> = fused
        .into_iter()
        .filter(|h| h.relevance >= thresholds.min_per_result_relevance)
        .collect();
    let removed = original_len - fused.len();
    if removed > 0 {
        tracing::warn!(
            target: "ttd_perf",
            removed,
            floor = thresholds.min_per_result_relevance,
            "lit_fusion: {} hit(s) below relevance floor {:.2} removed — contamination blocked",
            removed,
            thresholds.min_per_result_relevance,
        );
    }

    // Convert FusedHit → SearchResultHit for quality.rs (which takes &[SearchResultHit]).
    let hits_for_gate: Vec<SearchResultHit> = fused
        .iter()
        .map(|h| SearchResultHit {
            source_type: h.source_type.clone(),
            source_id: h.source_id.clone(),
            source_agent: None,
            source_date: None,
            domain: None,
            content: h.content.clone(),
            content_preview: h.content_preview.clone(),
            relevance: h.relevance,
        })
        .collect();

    // Quality gate — reuse the single thresholds binding (locked 0.3 floor). See 21-CONTEXT.md.
    let quality = assess_quality(&hits_for_gate, &thresholds);

    // Special case: empty results when no lane reported a problem AND the original
    // input was empty are a healthy "no matches" — match hybrid.rs:634 semantics.
    // A non-empty input that was fully filtered by the floor is contamination blocked,
    // not a healthy no-match: quality_degradation_reason must run and produce a loud reason.
    let quality_reason = if original_len == 0 && lane_degraded.is_none() {
        None
    } else {
        quality_degradation_reason(&quality, &thresholds, fused.len())
    };

    // Fold lane-level degradation with the gate's reason (same `; ` separator
    // as hybrid.rs:647-651).
    let degradation_reason = match (lane_degraded, quality_reason) {
        (Some(a), Some(b)) => Some(format!("{a}; {b}")),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    (fused, degradation_reason)
}

// ── Cross-encoder rerank (Lever B) ──────────────────────────────────────────────

/// Reorder fused hits by cross-encoder score and drop hits below `min_score`.
///
/// The bi-encoder cosine floor in [`gate_fused`] cannot separate a query's
/// methodological core from a topical false-friend (same surface vocabulary,
/// genuinely cosine-near). A cross-encoder scores those false-friends low, so
/// this stage:
///
/// 1. reorders the survivors by cross-encoder score (descending), and
/// 2. drops any hit scoring below `min_score`.
///
/// `scores` is `(index_into_fused, score)` from
/// [`JinaRerankService::rerank`](crate::JinaRerankService::rerank). It is
/// authoritative for ordering: hits are emitted in `scores` order. A hit whose
/// index is **absent** from `scores` is kept (appended after the scored hits,
/// original order) — never silently dropped, since absence means "no signal",
/// not "off-topic". This keeps the stage loud-degrade: a partial rerank result
/// cannot lose a source.
///
/// Runs **after** [`gate_fused`] so the locked 0.3 RRF floor is untouched — this
/// is precision on the survivors, not a change to intake.
///
/// ## Return
///
/// `(hits, Option<String>)` — the reordered/filtered hits plus a `Some` reason
/// when any hit was dropped by the cross-encoder floor (loud-degrade). `None`
/// when nothing was dropped.
pub fn apply_rerank(
    fused: Vec<FusedHit>,
    scores: &[crate::RerankResult],
    min_score: f32,
) -> (Vec<FusedHit>, Option<String>) {
    if fused.is_empty() || scores.is_empty() {
        return (fused, None);
    }

    let original_len = fused.len();
    // Mark which indices the reranker scored so we can append the unscored tail.
    let mut scored_idx: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // `scores` arrives sorted by score desc; emit in that order, dropping
    // below-floor. `fused.get` guards the (already index-validated) lookups.
    let mut out: Vec<FusedHit> = Vec::with_capacity(original_len);
    let mut dropped = 0usize;
    // Calibration telemetry: the score of each DROPPED hit (with a short title)
    // and the score range of the KEPT scored hits. Counts alone cannot tell
    // whether the floor cuts off-topic junk or relevant papers — the scores can.
    let mut dropped_detail: Vec<String> = Vec::new();
    let mut kept_min: Option<f32> = None;
    let mut kept_max: Option<f32> = None;
    for r in scores {
        scored_idx.insert(r.index);
        let title = fused.get(r.index).map(|h| h.title.as_str()).unwrap_or("");
        let short: String = title.chars().take(70).collect();
        if r.score < min_score {
            dropped += 1;
            dropped_detail.push(format!("{:.3}|{}", r.score, short));
            continue;
        }
        kept_min = Some(kept_min.map_or(r.score, |m| m.min(r.score)));
        kept_max = Some(kept_max.map_or(r.score, |m| m.max(r.score)));
        if let Some(h) = fused.get(r.index) {
            out.push(h.clone());
        }
    }

    // Append any hit the reranker did not score, in original order. Absence of a
    // score is "no signal", never grounds for a silent drop.
    let mut unscored = 0usize;
    for (i, h) in fused.iter().enumerate() {
        if !scored_idx.contains(&i) {
            unscored += 1;
            out.push(h.clone());
        }
    }

    if dropped > 0 {
        tracing::warn!(
            target: "ttd_perf",
            dropped,
            kept = out.len(),
            unscored,
            floor = min_score,
            kept_min = kept_min.unwrap_or(f32::NAN),
            kept_max = kept_max.unwrap_or(f32::NAN),
            dropped_scores = %dropped_detail.join(" ;; "),
            "lit_fusion: cross-encoder reranker dropped {dropped} hit(s) below floor {min_score:.2}",
        );
    } else {
        tracing::info!(
            target: "ttd_perf",
            kept = out.len(),
            unscored,
            floor = min_score,
            "lit_fusion: cross-encoder rerank reordered {} hit(s), none below floor",
            out.len(),
        );
    }

    let reason = if dropped > 0 {
        Some(format!(
            "cross-encoder reranker dropped {dropped} off-topic hit(s) below floor {min_score:.2}"
        ))
    } else {
        None
    };

    (out, reason)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RerankResult;

    fn make_internal(source_type: &str, source_id: &str) -> SearchResultHit {
        SearchResultHit {
            source_type: source_type.into(),
            source_id: source_id.into(),
            source_agent: None,
            source_date: None,
            domain: None,
            content: format!("content:{source_id}"),
            content_preview: format!("preview:{source_id}"),
            relevance: 0.8,
        }
    }

    fn make_arxiv(arxiv_id: &str) -> ArxivHit {
        ArxivHit {
            arxiv_id: arxiv_id.into(),
            title: format!("Arxiv paper {arxiv_id}"),
            section: "Introduction".into(),
            content: format!("arxiv content:{arxiv_id}"),
            content_preview: format!("arxiv preview:{arxiv_id}"),
            relevance: 0.7,
        }
    }

    fn make_s2(paper_id: &str) -> S2Hit {
        S2Hit {
            paper_id: paper_id.into(),
            title: format!("S2 paper {paper_id}"),
            content: format!("s2 content:{paper_id}"),
            content_preview: format!("s2 preview:{paper_id}"),
            relevance: 0.6,
        }
    }

    // ── rrf_dedup ─────────────────────────────────────────────────────────
    //
    // Three lanes with an overlapping id dedup by (source_type, source_id).
    // arxiv ("arxiv", id) and s2 ("s2", id) and internal ("internal", id) never
    // collide. The fused list must be sorted by RRF score descending and
    // normalised top → 1.0.

    #[test]
    fn rrf_dedup() {
        // Internal: 3 hits with different ids
        let internal = vec![
            make_internal("daily", "int-a"),
            make_internal("daily", "int-b"),
            make_internal("daily", "int-c"),
        ];
        // Arxiv: 3 hits (no id overlap with s2 or internal because source_type differs)
        let arxiv = vec![
            make_arxiv("2105.0001"),
            make_arxiv("2105.0002"),
            make_arxiv("2105.0003"),
        ];
        // S2: 3 hits
        let s2 = vec![
            make_s2("s2id-A"),
            make_s2("s2id-B"),
            make_s2("s2id-C"),
        ];

        let fused = fuse_rrf_three_lane(&internal, &arxiv, &s2, 60.0);

        // Total unique (source_type, source_id) pairs = 9 (no collisions)
        assert_eq!(fused.len(), 9, "all 9 distinct hits must be present");

        // Sorted descending
        for window in fused.windows(2) {
            assert!(
                window[0].relevance >= window[1].relevance,
                "fused list must be sorted descending: {:?} >= {:?}",
                window[0].relevance,
                window[1].relevance
            );
        }

        // Top hit normalised to 1.0
        assert!(
            (fused[0].relevance - 1.0).abs() < 1e-5,
            "top hit relevance must be 1.0, got {}",
            fused[0].relevance
        );

        // Verify no source_type collision between arxiv and s2 even if ids match
        let arxiv_hit = fuse_rrf_three_lane(
            &[],
            &[make_arxiv("same-id")],
            &[S2Hit {
                paper_id: "same-id".into(),
                title: "S2 paper".into(),
                content: "s2 content".into(),
                content_preview: "s2 preview".into(),
                relevance: 0.6,
            }],
            60.0,
        );
        // ("arxiv", "same-id") and ("s2", "same-id") are two distinct keys
        assert_eq!(arxiv_hit.len(), 2, "arxiv and s2 same bare id must NOT dedup");
        let types: Vec<&str> = arxiv_hit.iter().map(|h| h.source_type.as_str()).collect();
        assert!(types.contains(&"arxiv"), "arxiv source_type must be present");
        assert!(types.contains(&"s2"), "s2 source_type must be present");
    }

    // ── weak_retrieval_degrades ────────────────────────────────────────────
    //
    // A fused list that trips the gate (< 3 unique sources or mean < 0.48)
    // returns the hits PLUS a populated degradation_reason (loud-degrade),
    // not an empty result and not an error.

    #[test]
    fn weak_retrieval_degrades() {
        // Build five hits all from the same source_id. After dedup there will
        // be 5 distinct (source_type, source_id) keys because each has a
        // different source_id — but all from the same source_type "arxiv".
        // With 5 hits from 1 unique source_id ("same") the gate fires:
        //   concentration = 5/5 = 1.0 > 0.6 → concentration gate trips.
        //   unique_source_count = 1 < 3 → diversity gate trips.
        //
        // BUT: to get 5 hits with the SAME source_id that survive dedup,
        // we need 5 distinct (source_type, source_id) keys. Since fuse_rrf
        // dedups by (source_type, source_id), five arxiv hits with the same
        // arxiv_id will collapse to one hit. We need to use different source_ids.
        //
        // Strategy: 5 arxiv hits with different ids, but pass them as S2 hits
        // with the SAME paper_id — actually s2_id collisions dedup too.
        //
        // Simplest approach: use 5 distinct arxiv_ids all from one "arxiv" lane.
        // After fusion: 5 unique (arxiv, id) pairs. quality.rs assess_quality
        // groups by source_id (not source_type+source_id). So 5 distinct source_ids
        // → concentration = 1/5 = 0.2 (passes). unique_source_count = 5 (passes).
        //
        // To trip the gate we need: either mean relevance < 0.48, or we provide
        // 5 hits all from the SAME source_id (which requires same source_id
        // for quality.rs's source_id-based concentration check).
        //
        // quality.rs assess_quality uses h.source_id (not the compound key) for
        // concentration. So 5 FusedHit with source_id="same-paper" would give
        // concentration=1.0 but that requires 5 distinct (source_type, source_id)
        // — not possible with same source_type.
        //
        // Use a direct FusedHit list instead, bypassing fuse_rrf_three_lane, to
        // test gate_fused in isolation. This matches the plan's TDD intent.
        let fused_with_low_mean: Vec<FusedHit> = vec![
            FusedHit {
                source_type: "arxiv".into(),
                source_id: "paper-a".into(),
                title: "A".into(),
                section: None,
                content: "c".into(),
                content_preview: "p".into(),
                relevance: 0.31, // just above 0.3 floor
            },
            FusedHit {
                source_type: "arxiv".into(),
                source_id: "paper-b".into(),
                title: "B".into(),
                section: None,
                content: "c".into(),
                content_preview: "p".into(),
                relevance: 0.31,
            },
            FusedHit {
                source_type: "arxiv".into(),
                source_id: "paper-c".into(),
                title: "C".into(),
                section: None,
                content: "c".into(),
                content_preview: "p".into(),
                relevance: 0.31,
            },
            FusedHit {
                source_type: "arxiv".into(),
                source_id: "paper-d".into(),
                title: "D".into(),
                section: None,
                content: "c".into(),
                content_preview: "p".into(),
                relevance: 0.31,
            },
            FusedHit {
                source_type: "arxiv".into(),
                source_id: "paper-e".into(),
                title: "E".into(),
                section: None,
                content: "c".into(),
                content_preview: "p".into(),
                relevance: 0.31,
            },
        ];
        // Mean relevance = 0.31 < 0.48 → mean gate trips.
        let (returned_hits, reason) = gate_fused(fused_with_low_mean, None);

        // Hits are returned — not empty (loud-degrade, not hard-fail)
        assert!(
            !returned_hits.is_empty(),
            "weak retrieval must return hits, not empty (loud-degrade)"
        );

        // Degradation reason must be present
        assert!(
            reason.is_some(),
            "weak retrieval must produce a degradation_reason; got None. \
             Five hits with mean relevance 0.31 should trip the 0.48 mean gate."
        );
        let r = reason.unwrap();
        assert!(
            !r.is_empty(),
            "degradation_reason must not be an empty string"
        );
        // Should mention mean relevance gate
        assert!(
            r.contains("mean relevance"),
            "reason should mention mean relevance gate; got: {r}"
        );
    }

    // ── intake_floor_is_default ───────────────────────────────────────────
    //
    // The gate uses QualityThresholds::default() (0.3 per-result floor).
    // Assert the function does not construct a custom 0.57 threshold.

    #[test]
    fn intake_floor_is_default() {
        // A single hit at exactly 0.3 relevance: passes the 0.3 per-result
        // floor (inclusive >= per quality.rs fence-post tests). The mean gate
        // (0.48) will trip for a single hit at 0.3, but the per-result floor
        // itself must NOT trigger a per-result reason.
        //
        // We construct a FusedHit directly and call gate_fused to verify the
        // gate runs with the default thresholds.
        let hit = FusedHit {
            source_type: "arxiv".into(),
            source_id: "2105.0001".into(),
            title: "Test paper".into(),
            section: None,
            content: "content".into(),
            content_preview: "preview".into(),
            relevance: 0.3,
        };

        let (_returned_hits, reason) = gate_fused(vec![hit], None);

        // The quality gate ran — if it had constructed a custom 0.57 floor the
        // 0.3 hit would have tripped the per-result gate and the reason would
        // contain "min relevance". With the default 0.3 floor (inclusive), the
        // per-result gate must NOT fire for this hit.
        if let Some(r) = &reason {
            assert!(
                !r.contains("min relevance"),
                "default 0.3 floor must not fire for relevance=0.3 hit; \
                 got reason: {r} — possible custom threshold in use"
            );
        }
        // The mean gate (0.48) WILL fire for a single hit at mean=0.3, which
        // is expected — but this verifies the 0.3 floor is not raised to 0.57.
    }

    // ── empty_lanes_no_degradation ────────────────────────────────────────
    //
    // Empty results from all three lanes with no lane degradation should not
    // fire the quality gate (matches hybrid.rs:634 semantics — "no matches"
    // is not a degradation event).

    #[test]
    fn empty_lanes_no_degradation() {
        let fused = fuse_rrf_three_lane(&[], &[], &[], 60.0);
        assert!(fused.is_empty());
        let (_, reason) = gate_fused(fused, None);
        assert!(
            reason.is_none(),
            "empty result with no lane degradation must not produce a reason"
        );
    }

    // ── F3: relevance floor tests ─────────────────────────────────────────
    //
    // gate_fused must remove below-floor hits BEFORE quality assessment and
    // before build_panel. The 0.3 floor comes from QualityThresholds::default().

    #[test]
    fn gate_fused_removes_below_floor_hits() {
        // One hit above the 0.3 floor (relevance 1.0), one below (relevance 0.1).
        // After the filter only the above-floor hit must remain.
        let above = FusedHit {
            source_type: "arxiv".into(),
            source_id: "good-paper".into(),
            title: "Good".into(),
            section: None,
            content: "content".into(),
            content_preview: "preview".into(),
            relevance: 1.0,
        };
        let below = FusedHit {
            source_type: "arxiv".into(),
            source_id: "junk".into(),
            title: "Junk".into(),
            section: None,
            content: "content".into(),
            content_preview: "preview".into(),
            relevance: 0.1,
        };

        let (hits, _reason) = gate_fused(vec![above, below], None);
        assert_eq!(hits.len(), 1, "below-floor hit must be removed; got {} hits", hits.len());
        assert_eq!(
            hits[0].source_id, "good-paper",
            "surviving hit must be the above-floor one"
        );
    }

    #[test]
    fn gate_fused_all_below_floor_yields_degradation_reason() {
        // A non-empty input where ALL hits are below 0.3 floor must return
        // an empty vec AND Some(degradation_reason) — the healthy-no-match
        // suppression must NOT swallow a fully-filtered set.
        let fused = vec![
            FusedHit {
                source_type: "s2".into(),
                source_id: "bad-a".into(),
                title: "Bad A".into(),
                section: None,
                content: "c".into(),
                content_preview: "p".into(),
                relevance: 0.1,
            },
            FusedHit {
                source_type: "s2".into(),
                source_id: "bad-b".into(),
                title: "Bad B".into(),
                section: None,
                content: "c".into(),
                content_preview: "p".into(),
                relevance: 0.05,
            },
        ];

        let (hits, reason) = gate_fused(fused, None);
        assert!(hits.is_empty(), "all-below-floor input must yield empty hits");
        assert!(
            reason.is_some(),
            "fully-filtered non-empty input must produce a degradation reason"
        );
    }

    #[test]
    fn gate_fused_empty_input_no_lane_degraded_stays_healthy() {
        // Empty input with no lane degradation must still return (empty, None) —
        // the hybrid.rs:634 healthy-no-match semantics are preserved.
        let (hits, reason) = gate_fused(vec![], None);
        assert!(hits.is_empty(), "empty input must stay empty");
        assert!(
            reason.is_none(),
            "empty input with no lane degradation must return no reason; got: {reason:?}"
        );
    }

    // ── lane_degraded_folded ──────────────────────────────────────────────
    //
    // When a lane_degraded reason is passed in, it is folded into the
    // overall degradation_reason even when the quality gate passes.

    // ── apply_rerank (Lever B) ────────────────────────────────────────────

    fn hit(id: &str, rrf: f32) -> FusedHit {
        FusedHit {
            source_type: "s2".into(),
            source_id: id.into(),
            title: format!("title-{id}"),
            section: None,
            content: format!("content-{id}"),
            content_preview: format!("preview-{id}"),
            relevance: rrf,
        }
    }

    #[test]
    fn apply_rerank_reorders_and_drops_below_floor() {
        // RRF order is a, b, c. Cross-encoder says c is most relevant, b is the
        // off-topic false-friend (score below floor), a is middling.
        let fused = vec![hit("a", 1.0), hit("b", 0.9), hit("c", 0.8)];
        let scores = vec![
            RerankResult { index: 2, score: 0.95 }, // c
            RerankResult { index: 0, score: 0.40 }, // a
            RerankResult { index: 1, score: 0.02 }, // b — false-friend
        ];
        let (out, reason) = apply_rerank(fused, &scores, 0.1);
        // b dropped; order is now c, a.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].source_id, "c", "highest cross-encoder score first");
        assert_eq!(out[1].source_id, "a");
        assert!(reason.is_some(), "a drop must produce a loud reason");
        assert!(reason.unwrap().contains("dropped 1"));
    }

    #[test]
    fn apply_rerank_keeps_unscored_hits() {
        // Reranker only scored index 0; index 1 has no score → must be kept,
        // appended after the scored hit, never silently dropped.
        let fused = vec![hit("a", 1.0), hit("b", 0.9)];
        let scores = vec![RerankResult { index: 0, score: 0.7 }];
        let (out, reason) = apply_rerank(fused, &scores, 0.1);
        assert_eq!(out.len(), 2, "unscored hit must survive");
        assert_eq!(out[0].source_id, "a");
        assert_eq!(out[1].source_id, "b", "unscored hit appended in original order");
        assert!(reason.is_none(), "no drop → no degradation reason");
    }

    #[test]
    fn apply_rerank_empty_scores_is_noop() {
        // A failed/empty rerank result must return the hits untouched (loud-
        // degrade is the caller's job; the pure fn just preserves recall).
        let fused = vec![hit("a", 1.0), hit("b", 0.9)];
        let (out, reason) = apply_rerank(fused, &[], 0.1);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].source_id, "a");
        assert!(reason.is_none());
    }

    #[test]
    fn lane_degraded_folded() {
        // Five high-quality hits from 5 distinct sources — quality gate passes.
        let internal: Vec<SearchResultHit> = (0..5)
            .map(|i| make_internal("daily", &format!("id-{i}")))
            .collect();
        // Use high relevance so quality gate passes
        let hits_with_relevance: Vec<SearchResultHit> = internal
            .into_iter()
            .map(|mut h| {
                h.relevance = 0.9;
                h
            })
            .collect();

        let fused = fuse_rrf_three_lane(&hits_with_relevance, &[], &[], 60.0);
        let (_returned, reason) = gate_fused(fused, Some("arxiv fetch failed: 429".into()));

        assert!(
            reason.is_some(),
            "lane degradation reason must propagate even when quality gate passes"
        );
        let r = reason.unwrap();
        assert!(
            r.contains("arxiv fetch failed"),
            "lane reason must appear in folded reason: {r}"
        );
    }
}
