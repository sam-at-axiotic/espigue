//! Mechanical per-source authenticity tier.
//!
//! A paper's credibility tier is a pure, deterministic function of its
//! S2-intrinsic signals — citation count, influential-citation count, and
//! whether it was published at a venue. No model judgment: the same philosophy
//! as substring quote verification, where a mechanical check beats asking a
//! model "is this real?". Once derived, the tier is threaded to display
//! (reference tags) and to the support-level guard.
//!
//! Cutpoints are config-driven ([`TierThresholds`]) and were chosen from the
//! live corpus distribution (memory/literature.db, ~4,800 S2-resolved papers):
//! 73% of papers have zero influential citations, the citation median is ~1–2,
//! and the top citation decile runs 43 → 14,890. So `influential ≥ 5` and
//! `citations ≥ 50` each isolate roughly the top tenth of papers.

use serde::{Deserialize, Serialize};

/// Authenticity tier for one source paper. Ordered: `Unknown < Low < Moderate < High`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CredibilityTier {
    /// Signals were never fetched (S2 throttled the backfill, or the paper is
    /// not in S2). NOT a credibility judgment — "we do not know". The guard
    /// must never act on this; display renders it honestly as "unrated".
    Unknown,
    /// Zero traction / unvetted — signals fetched, no venue, no meaningful
    /// citation signal. The zero-cite preprint profile (the negotiation-conflict
    /// false-friend). This is the tier the guard acts on.
    Low,
    /// Real traction — published at a venue, or some citations / one influential
    /// citation, but not yet load-bearing for the field.
    Moderate,
    /// The field builds on it — high citation count or several influential
    /// citations.
    High,
}

impl CredibilityTier {
    /// Display label. This is the user-facing vocabulary; kept in one place so
    /// the naming is a single edit. Default is plain "<level> credibility" to
    /// avoid colliding with the claim `support_level` vocabulary (which already
    /// uses "established"/"emerging").
    pub fn label(self) -> &'static str {
        match self {
            CredibilityTier::Unknown => "unrated",
            CredibilityTier::Low => "low credibility",
            CredibilityTier::Moderate => "moderate credibility",
            CredibilityTier::High => "high credibility",
        }
    }

    /// Whether this tier is a *verified* low-credibility judgment — the only
    /// state the support-level guard may demote on. `Unknown` is not low; it is
    /// the absence of a signal and must not trigger enforcement.
    pub fn is_verified_low(self) -> bool {
        self == CredibilityTier::Low
    }
}

/// Cutpoints for [`derive_tier`]. Defaults derived from the corpus distribution
/// — see the module docs. Tune here, not at call sites.
#[derive(Debug, Clone, Copy)]
pub struct TierThresholds {
    /// Influential-citation count at or above which a paper is `High` (top ~7.5%).
    pub high_influential: i64,
    /// Citation count at or above which a paper is `High` (top ~8%).
    pub high_citation: i64,
    /// Citation count at or above which a paper is at least `Moderate`.
    pub moderate_citation: i64,
}

impl Default for TierThresholds {
    fn default() -> Self {
        Self {
            high_influential: 5,
            high_citation: 50,
            moderate_citation: 5,
        }
    }
}

/// Derive the credibility tier from a paper's signals.
///
/// When no signal was ever fetched (both counts `None` and no venue) the result
/// is `Unknown` — distinct from `Low`, so the guard never demotes a paper whose
/// signals S2 simply did not return. A blank / `None` venue is treated as absent.
///
/// - **Unknown** — no fetched signal at all
/// - **High** — `influential ≥ high_influential` OR `citations ≥ high_citation`
/// - **Moderate** — has a venue, OR `citations ≥ moderate_citation`, OR `influential ≥ 1`
/// - **Low** — fetched, none of the above (no venue, sub-threshold citations, zero influential)
pub fn derive_tier(
    citation_count: Option<i64>,
    influential_citation_count: Option<i64>,
    venue: Option<&str>,
    thresholds: &TierThresholds,
) -> CredibilityTier {
    let has_venue = venue.map(|v| !v.trim().is_empty()).unwrap_or(false);

    // No signal at all (counts never fetched, no venue) → Unknown, not Low.
    // A resolved paper with genuinely zero traction carries non-NULL 0 counts,
    // which falls through to Low below.
    if citation_count.is_none() && influential_citation_count.is_none() && !has_venue {
        return CredibilityTier::Unknown;
    }

    let citations = citation_count.unwrap_or(0);
    let influential = influential_citation_count.unwrap_or(0);

    if influential >= thresholds.high_influential || citations >= thresholds.high_citation {
        CredibilityTier::High
    } else if has_venue || citations >= thresholds.moderate_citation || influential >= 1 {
        CredibilityTier::Moderate
    } else {
        CredibilityTier::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> TierThresholds {
        TierThresholds::default()
    }

    #[test]
    fn high_by_influential() {
        // 5 influential citations alone → High, even with few raw citations.
        assert_eq!(derive_tier(Some(8), Some(5), None, &t()), CredibilityTier::High);
    }

    #[test]
    fn high_by_citation() {
        // 50 citations alone → High, even with zero influential.
        assert_eq!(derive_tier(Some(50), Some(0), None, &t()), CredibilityTier::High);
    }

    #[test]
    fn moderate_by_venue() {
        // A venued paper with no citations is still vetted enough for Moderate.
        assert_eq!(
            derive_tier(Some(0), Some(0), Some("NeurIPS"), &t()),
            CredibilityTier::Moderate
        );
    }

    #[test]
    fn moderate_by_one_influential() {
        assert_eq!(derive_tier(Some(3), Some(1), None, &t()), CredibilityTier::Moderate);
    }

    #[test]
    fn moderate_by_citation_floor() {
        assert_eq!(derive_tier(Some(5), Some(0), None, &t()), CredibilityTier::Moderate);
    }

    #[test]
    fn low_zero_traction_preprint() {
        // The negotiation-conflict false-friend: no venue, no citations, no
        // influential. This is exactly what the tier must flag.
        assert_eq!(derive_tier(Some(0), Some(0), None, &t()), CredibilityTier::Low);
        assert_eq!(derive_tier(Some(2), Some(0), Some(""), &t()), CredibilityTier::Low);
    }

    #[test]
    fn no_signals_are_unknown_not_low() {
        // NULL everywhere → Unknown (signals never fetched), NOT Low. The guard
        // must not demote on this — it would punish a rate-limit artifact.
        assert_eq!(derive_tier(None, None, None, &t()), CredibilityTier::Unknown);
        assert!(!derive_tier(None, None, None, &t()).is_verified_low());
    }

    #[test]
    fn fetched_zero_traction_is_verified_low() {
        // Signals fetched (non-NULL 0s), no venue → genuinely Low, guard-actionable.
        let tier = derive_tier(Some(0), Some(0), None, &t());
        assert_eq!(tier, CredibilityTier::Low);
        assert!(tier.is_verified_low());
    }

    #[test]
    fn unknown_but_venued_is_moderate() {
        // A venue is itself a fetched signal — rescues to Moderate, not Unknown.
        assert_eq!(derive_tier(None, None, Some("ACL"), &t()), CredibilityTier::Moderate);
    }

    #[test]
    fn ordering_holds() {
        assert!(CredibilityTier::Unknown < CredibilityTier::Low);
        assert!(CredibilityTier::Low < CredibilityTier::Moderate);
        assert!(CredibilityTier::Moderate < CredibilityTier::High);
    }

    #[test]
    fn labels_do_not_collide_with_support_level() {
        // support_level uses "established"/"emerging"; tiers must not.
        for tier in [
            CredibilityTier::Unknown,
            CredibilityTier::Low,
            CredibilityTier::Moderate,
            CredibilityTier::High,
        ] {
            let l = tier.label();
            assert!(!l.contains("established") && !l.contains("emerging"), "collision: {l}");
        }
    }
}
