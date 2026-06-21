//! Canonical agent-to-domain mapping.
//!
//! This is the single source of truth for which domain each agent belongs to.
//! `BootstrapConfig::domain_mapping` in `alzina-governance` uses this as its
//! default, and all other consumers (reflection, learnings merger, etc.) should
//! clone from here rather than maintaining their own mapping.
//!
//! The mapping can be overridden per-workspace via `[bootstrap.domain_mapping]`
//! in governance config (A-24 overlay semantics).

use std::collections::HashMap;

/// The 13 canonical agent-to-domain pairs.
///
/// Domains: orchestration, analysis, implementation, research, experimentation,
/// synthesis, governance.
pub fn canonical_domain_mapping() -> HashMap<String, String> {
    [
        ("vefr", "orchestration"),
        ("urdr", "analysis"),
        ("skuld", "analysis"),
        ("huginn", "analysis"),
        ("smidr", "implementation"),
        ("galdr", "implementation"),
        ("gna", "research"),
        ("ratatoskr", "research"),
        ("kvasir", "experimentation"),
        ("tester", "experimentation"),
        ("sjofn", "synthesis"),
        ("verdandi", "governance"),
        ("muninn", "governance"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Resolve a single agent to its canonical domain.
/// Returns `"general"` for unknown agents.
pub fn resolve_domain(agent: &str) -> &'static str {
    match agent {
        "vefr" => "orchestration",
        "urdr" | "skuld" | "huginn" => "analysis",
        "smidr" | "galdr" => "implementation",
        "gna" | "ratatoskr" => "research",
        "kvasir" | "tester" => "experimentation",
        "sjofn" => "synthesis",
        "verdandi" | "muninn" => "governance",
        _ => "general",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_mapping_has_13_entries() {
        let map = canonical_domain_mapping();
        assert_eq!(map.len(), 13);
    }

    #[test]
    fn smidr_maps_to_implementation() {
        let map = canonical_domain_mapping();
        assert_eq!(map["smidr"], "implementation");
    }

    #[test]
    fn resolve_known_agents() {
        assert_eq!(resolve_domain("smidr"), "implementation");
        assert_eq!(resolve_domain("vefr"), "orchestration");
        assert_eq!(resolve_domain("huginn"), "analysis");
        assert_eq!(resolve_domain("muninn"), "governance");
        assert_eq!(resolve_domain("kvasir"), "experimentation");
        assert_eq!(resolve_domain("sjofn"), "synthesis");
        assert_eq!(resolve_domain("gna"), "research");
    }

    #[test]
    fn resolve_unknown_agent_returns_general() {
        assert_eq!(resolve_domain("unknown"), "general");
    }

    #[test]
    fn canonical_map_matches_resolve() {
        let map = canonical_domain_mapping();
        for (agent, domain) in &map {
            assert_eq!(
                resolve_domain(agent),
                domain.as_str(),
                "mismatch for agent '{agent}': map says '{domain}', resolve says '{}'",
                resolve_domain(agent),
            );
        }
    }
}
