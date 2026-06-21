//! Synthesis artifact and argumentation graph YAML serialisation + provenance.
//!
//! Source: `consensus/src/consensus/domain/models.py:31,151-188,238-271` [VERIFIED]
//!
//! Both artifact structs carry the same five provenance fields:
//! `schema_version`, `generated_at`, `model`, `prompt_version`, `code_version`.
//!
//! The YAML serialisation uses `serde_yaml` (already a workspace dep in
//! `alzina-orchestration/Cargo.toml` line 30 — no new dep needed).
//!
//! ## Trust boundary (T-23-02)
//!
//! Only `model`, `prompt_version`, and `code_version` are stamped. No API keys,
//! no env values. The provenance struct has a fixed field set; serde cannot pick
//! up arbitrary secrets.
//!
//! ## Audit-trail emit
//!
//! The YAML artifact emit on the audit trail (ENGINE-01) is wired in Wave 3
//! (Plan 23-04 emit.rs). This plan delivers only the serde + provenance-stamping
//! helper.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ── Schema version ────────────────────────────────────────────────────────────

/// Canonical schema version string (models.py:31).
pub const SCHEMA_VERSION: &str = "1.0";

// ── Provenance fields helper ──────────────────────────────────────────────────

/// Capture the current git commit hash for the `code_version` provenance field.
///
/// Tries `git rev-parse HEAD` at runtime. Falls back to `"unknown"` when git
/// is absent or the repo is in a detached state without a commit.
///
/// Does NOT panic when git is missing — the engine must still produce an artifact
/// in environments without a git CLI (e.g. CI without git in PATH).
pub fn code_version() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

// ── Claim ─────────────────────────────────────────────────────────────────────

/// One claim in the synthesis (models.py — Claim dataclass).
///
/// v2 additive fields (`support_level`, `evidence_grade`, `method`, `year`,
/// `lineage`) are all `skip_serializing_if = "Option::is_none"` so they never
/// appear in v1 YAML output — byte-identity of v1 emit is preserved.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Claim {
    /// Claim text.
    pub text: String,
    /// Agreement level: "consensus", "majority", "divided", or "minority".
    /// v1 only — v2 uses `support_level` instead.
    #[serde(default)]
    pub agreement_level: Option<String>,
    /// Source paper IDs supporting this claim.
    #[serde(default)]
    pub sources: Vec<String>,
    /// Counterarguments or dissenting perspectives.
    #[serde(default)]
    pub counterarguments: Vec<String>,

    // ── v2 additive fields ────────────────────────────────────────────────────
    // All carry skip_serializing_if so v1 YAML output is byte-identical.

    /// v2: Epistemic support label from the closed vocabulary in `term_sheet::SUPPORT_LEVELS`.
    /// In-vocabulary values are normalised to the canonical lowercase form.
    /// Out-of-vocabulary → `None` plus a warn log (never a fabricated default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub support_level: Option<String>,

    /// v2: Evidence quality grade (free string at the schema layer; B2 defines
    /// the prompt vocabulary, B3 enforces). E.g. "high", "moderate", "low".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_grade: Option<String>,

    /// v2: Research method(s) used to establish the claim (free string).
    /// E.g. "meta-analysis", "RCT", "observational", "modelling".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,

    /// v2: Publication year(s) of the primary source(s) (stored as String, not
    /// int — sources may span a range or cite multiple years).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year: Option<String>,

    /// v2: Lineage note — how this claim relates to prior work or was derived
    /// from earlier findings in the reviewed set (free string).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<String>,

    /// v2: Verbatim quotes backing this claim, one per (source, quote) pair
    /// (quote-grounded synthesis, worklist item 4). Empty for v1 — serde-skip
    /// keeps v1 YAML byte-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quotes: Vec<ClaimQuote>,

    /// F14 (node-cited synthesis): graph node ids that support this claim, as
    /// emitted by the v2 draft model (which cites nodes instead of authoring
    /// quotes). The merger reads these to fetch each node's relevant section and
    /// author a verbatim quote; the deterministic floor uses them to attach a
    /// node's stored verified quote by exact id when authoring leaves the claim
    /// quoteless. Empty for v1 and pre-F14 v2 — serde-skip keeps YAML
    /// byte-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub node_refs: Vec<String>,

    /// F14 commit 5 (author-year citations): the rendered inline citation for
    /// this claim, e.g. `"(Smith et al., 2021; Jones, 2020)"`, built from the
    /// claim's `sources` against the papers table at the daemon emit boundary.
    /// `None` until the v2/v3 citation render runs (and for v1) — serde-skip
    /// keeps YAML byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citation: Option<String>,
}

/// One verbatim quote backing a synthesis claim, tied to its source paper.
///
/// The model may copy quotes from anywhere it saw text (graph markdown or
/// gap-retrieval context); verification keys on the cited source's STORED
/// text via the DB-backed resolver, never on prompt contents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClaimQuote {
    /// Source paper id the quote was copied from (papers/lit_chunks namespace).
    pub source: String,
    /// Verbatim quote text.
    pub text: String,
    /// Verification outcome: "verified" | "paraphrased" | "absent" |
    /// "unverified". `None` until the post-process verification pass runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// F12 honesty marker: `true` when the model's quote text was a paraphrase
    /// and the verification pass replaced it with the closest sentence from
    /// the cited source's stored prose (quote snapping). The emitted text is
    /// then a true verbatim quote, but it is not what the model wrote.
    /// Serde-skipped when false so pre-F12 output stays byte-identical.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub snapped: bool,
    /// F12-class honesty marker (Fix C / probe-17 cause 2): `true` when the
    /// quote text was attached deterministically from a DB-verified graph node,
    /// NOT written by the synthesis model. The inherited text is re-verified
    /// against the source's stored prose before stamping — no transitive trust
    /// of the graph's own verification status.
    /// Serde-skipped when false so pre-Fix-C YAML output stays byte-identical.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inherited: bool,
    /// F14 (node-cited synthesis): the graph node id this quote was authored
    /// from / attached from. Set by the Opus merger (which copies the quote from
    /// the node's relevant section) and by the deterministic floor (which
    /// attaches the node's stored verified quote by exact id). Enables exact
    /// claim↔node provenance without the lossy ≥0.5 token-containment match.
    /// `None` for v1 and pre-F14 v2 — serde-skip keeps YAML byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
}

// ── Gap ───────────────────────────────────────────────────────────────────────

/// A typed knowledge gap identified during v2 synthesis.
///
/// `uncertainties` (v1 field on `SynthesisArtifact`) stays unchanged for v1
/// back-compat. v2 populates `SynthesisArtifact.gaps` (typed) and may leave
/// `uncertainties` empty.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Gap {
    /// Human-readable description of the gap.
    pub description: String,
    /// Gap category from `term_sheet::GAP_TYPES` (e.g. "epistemic", "empirical").
    /// Stored verbatim — B3 enforces the closed vocabulary.
    /// `None` when the model did not provide a type attribute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap_type: Option<String>,
}

// ── MinorityReport ────────────────────────────────────────────────────────────

/// A minority perspective not captured in the main claims.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MinorityReport {
    /// Source paper ID(s) holding the minority view.
    pub source_ids: Vec<String>,
    /// Description of the minority position.
    pub perspective: String,
}

// ── NarrativeStatement ────────────────────────────────────────────────────────

/// One statement in the narrative with cross-references to claims and experts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NarrativeStatement {
    /// Text of the narrative statement.
    pub text: String,
    /// Inline `[Cx]` citation markers referencing claim indices.
    #[serde(default)]
    pub claim_refs: Vec<String>,
    /// Expert (paper) IDs referenced by this statement.
    #[serde(default)]
    pub expert_refs: Vec<String>,
}

// ── SynthesisArtifact ─────────────────────────────────────────────────────────

/// The primary output of the consensus TTD synthesis stage.
///
/// Matches `consensus/src/consensus/domain/models.py:151-188` [VERIFIED].
///
/// Emitted as YAML on the audit trail (ENGINE-01) after Stage 3 populates
/// the `narrative` and `narrative_statements` fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynthesisArtifact {
    // ── Provenance fields ─────────────────────────────────────────────────────
    /// Schema version — always `"1.0"` for this implementation (models.py:31).
    pub schema_version: String,
    /// Consultation / question identifier.
    pub study_id: String,
    /// Round identifier within the consultation.
    pub round_id: String,
    /// Question identifier (maps to the specific research question).
    pub question_id: String,
    /// Timestamp when the synthesis was generated.
    pub generated_at: DateTime<Utc>,
    /// LLM model used for this synthesis (e.g. `"google/gemini-2.5-flash"`).
    pub model: String,
    /// Prompt version (e.g. `"v1/synthesis"`).
    pub prompt_version: String,
    /// Git commit hash of the engine code that generated this artifact.
    pub code_version: String,

    // ── Content fields ────────────────────────────────────────────────────────
    /// Main synthesis claims with agreement levels and sources.
    #[serde(default)]
    pub claims: Vec<Claim>,
    /// Areas where experts agree.
    #[serde(default)]
    pub areas_of_agreement: Vec<String>,
    /// Areas where experts disagree.
    #[serde(default)]
    pub areas_of_disagreement: Vec<String>,
    /// Identified uncertainties or gaps in the evidence (v1 field — stays for
    /// back-compat; v2 populates `gaps` instead).
    #[serde(default)]
    pub uncertainties: Vec<String>,
    /// Minority perspectives not in the main synthesis.
    #[serde(default)]
    pub minority_reports: Vec<MinorityReport>,
    /// Narrative summary (populated after Stage 3).
    #[serde(default)]
    pub narrative: String,
    /// Parsed narrative statements with claim cross-references (Stage 3).
    #[serde(default)]
    pub narrative_statements: Vec<NarrativeStatement>,

    // ── v2 additive field ─────────────────────────────────────────────────────
    /// v2: Typed knowledge gaps with `gap_type` from `term_sheet::GAP_TYPES`.
    /// `skip_serializing_if = "Vec::is_empty"` keeps v1 YAML output byte-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<Gap>,
}

impl SynthesisArtifact {
    /// Construct with all provenance fields filled in.
    ///
    /// `code_version` is captured via `self::code_version()` (git rev-parse HEAD).
    pub fn new(
        study_id: impl Into<String>,
        round_id: impl Into<String>,
        question_id: impl Into<String>,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            study_id: study_id.into(),
            round_id: round_id.into(),
            question_id: question_id.into(),
            generated_at: Utc::now(),
            model: model.into(),
            prompt_version: prompt_version.into(),
            code_version: code_version(),
            claims: vec![],
            areas_of_agreement: vec![],
            areas_of_disagreement: vec![],
            uncertainties: vec![],
            minority_reports: vec![],
            narrative: String::new(),
            narrative_statements: vec![],
            gaps: vec![],
        }
    }

    /// Serialise to YAML (serde_yaml round-trip).
    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    /// Deserialise from YAML.
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }
}

// ── ArgumentationGraph ────────────────────────────────────────────────────────

/// Argumentation graph produced by Stage 1.
///
/// Matches `consensus/src/consensus/domain/models.py:238-271` [VERIFIED].
/// Carries the same five provenance fields as `SynthesisArtifact`.
///
/// Used as Stage 2's input when `use_graph_draft=true` (the default).
/// Both artifacts are declared here (artifact.rs OWNS both) so Stage-1
/// (graph.rs) and Wave-3 emit.rs import them rather than re-declaring —
/// this closes the Wave-3 emit.rs compile dependency (ENGINE-01).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArgumentationGraph {
    // ── Provenance fields ─────────────────────────────────────────────────────
    pub schema_version: String,
    pub study_id: String,
    pub round_id: String,
    pub question_id: String,
    pub generated_at: DateTime<Utc>,
    pub model: String,
    pub prompt_version: String,
    pub code_version: String,

    // ── Graph content ─────────────────────────────────────────────────────────
    /// Graph nodes — each node is a claim with an expert-namespaced ID.
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    /// Edges between nodes (cross-expert relationships, causal links, contradictions).
    #[serde(default)]
    pub edges: Vec<GraphEdge>,
    /// Per-node annotations (verification status, confidence).
    #[serde(default)]
    pub node_annotations: Vec<NodeAnnotation>,
}

impl ArgumentationGraph {
    pub fn new(
        study_id: impl Into<String>,
        round_id: impl Into<String>,
        question_id: impl Into<String>,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            study_id: study_id.into(),
            round_id: round_id.into(),
            question_id: question_id.into(),
            generated_at: Utc::now(),
            model: model.into(),
            prompt_version: prompt_version.into(),
            code_version: code_version(),
            nodes: vec![],
            edges: vec![],
            node_annotations: vec![],
        }
    }

    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// Render the graph as human-readable Markdown for operator perusal.
    ///
    /// Claims are grouped by source paper (expert_id) with quotes and
    /// verification status; edges carry short claim previews so relations
    /// read without cross-referencing node IDs.
    pub fn to_markdown(&self) -> String {
        use std::collections::BTreeMap;
        use std::fmt::Write;

        fn preview(text: &str, max: usize) -> String {
            let t = text.trim();
            if t.chars().count() <= max {
                t.to_string()
            } else {
                let cut: String = t.chars().take(max).collect();
                format!("{cut}…")
            }
        }

        let mut md = String::new();
        let _ = writeln!(md, "# Argumentation graph — {} / {}", self.study_id, self.round_id);
        let _ = writeln!(md);
        let _ = writeln!(md, "- question: {}", self.question_id);
        let _ = writeln!(
            md,
            "- generated: {} · model: {} · prompt: {} · code: {}",
            self.generated_at.format("%Y-%m-%d %H:%M:%SZ"),
            self.model,
            self.prompt_version,
            self.code_version,
        );

        // Group nodes by expert (paper).
        let mut by_expert: BTreeMap<&str, Vec<&GraphNode>> = BTreeMap::new();
        for node in &self.nodes {
            by_expert.entry(node.expert_id.as_str()).or_default().push(node);
        }

        let _ = writeln!(
            md,
            "- {} claims from {} sources · {} edges · {} annotations",
            self.nodes.len(),
            by_expert.len(),
            self.edges.len(),
            self.node_annotations.len(),
        );

        let _ = writeln!(md);
        let _ = writeln!(md, "## Claims by source");
        for (expert_id, nodes) in &by_expert {
            let _ = writeln!(md);
            let _ = writeln!(md, "### {} ({} claims)", expert_id, nodes.len());
            let _ = writeln!(md);
            for node in nodes {
                let status = node.verification_status.as_deref().unwrap_or("unverified");
                let _ = writeln!(md, "- **{}** [{}] — {}", node.id, status, node.claim.trim());
                if let Some(quote) = &node.quote {
                    if !quote.trim().is_empty() {
                        let _ = writeln!(md, "  > {}", quote.trim());
                    }
                }
            }
        }

        // Index claims for edge previews.
        let claim_by_id: BTreeMap<&str, &str> = self
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), n.claim.as_str()))
            .collect();
        let claim_preview = |id: &str| -> String {
            claim_by_id
                .get(id)
                .map(|c| format!(" \"{}\"", preview(c, 60)))
                .unwrap_or_default()
        };

        if !self.edges.is_empty() {
            // Group edges by relation type.
            let mut by_relation: BTreeMap<&str, Vec<&GraphEdge>> = BTreeMap::new();
            for edge in &self.edges {
                by_relation.entry(edge.relation.as_str()).or_default().push(edge);
            }
            let _ = writeln!(md);
            let _ = writeln!(md, "## Relations");
            for (relation, edges) in &by_relation {
                let _ = writeln!(md);
                let _ = writeln!(md, "### {} ({})", relation, edges.len());
                let _ = writeln!(md);
                for edge in edges {
                    let _ = writeln!(
                        md,
                        "- `{}`{} → `{}`{}",
                        edge.source,
                        claim_preview(&edge.source),
                        edge.target,
                        claim_preview(&edge.target),
                    );
                }
            }
        }

        if !self.node_annotations.is_empty() {
            let _ = writeln!(md);
            let _ = writeln!(md, "## Annotations");
            let _ = writeln!(md);
            for ann in &self.node_annotations {
                let _ = writeln!(md, "- `{}`: {}", ann.node_id, ann.annotation.trim());
            }
        }

        md
    }
}

// ── Graph sub-types ───────────────────────────────────────────────────────────

/// One node in the argumentation graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    /// Expert-namespaced node ID: `{expert_id}_{node_id}` (graph_tasks.py convention).
    pub id: String,
    /// Claim text.
    pub claim: String,
    /// Expert (paper) ID that made this claim.
    pub expert_id: String,
    /// Direct quote from the expert's response that supports this claim.
    #[serde(default)]
    pub quote: Option<String>,
    /// Verification status: "verified", "absent", or "unverified".
    #[serde(default)]
    pub verification_status: Option<String>,
}

/// A directed edge between two nodes in the argumentation graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    /// Relationship type: "supports", "contradicts", "elaborates", "causes", etc.
    pub relation: String,
}

/// Per-node annotation from the fitness evaluation or graph resolution step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeAnnotation {
    pub node_id: String,
    /// Annotation text (e.g. verification verdict, confidence note).
    pub annotation: String,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// ENGINE-01: provenance fields present after construction.
    #[test]
    fn provenance_fields_present() {
        let artifact = SynthesisArtifact::new(
            "study-001",
            "round-1",
            "q-climate-effects",
            "google/gemini-2.5-flash",
            "v1/synthesis",
        );

        assert_eq!(artifact.schema_version, "1.0");
        assert_eq!(artifact.study_id, "study-001");
        assert!(!artifact.generated_at.to_string().is_empty(), "generated_at must be set");
        assert_eq!(artifact.model, "google/gemini-2.5-flash");
        assert_eq!(artifact.prompt_version, "v1/synthesis");
        // code_version may be "unknown" if git is absent, but must never be empty
        assert!(!artifact.code_version.is_empty(), "code_version must not be empty");
    }

    /// ENGINE-01: SynthesisArtifact YAML round-trips with all provenance fields.
    #[test]
    fn synthesis_artifact_yaml_round_trip() {
        let mut artifact = SynthesisArtifact::new(
            "study-001",
            "round-1",
            "q-001",
            "google/gemini-2.5-flash",
            "v1/synthesis",
        );
        artifact.claims.push(Claim {
            text: "Climate change accelerates permafrost thaw.".into(),
            agreement_level: Some("consensus".into()),
            sources: vec!["arxiv:2105.14103".into()],
            counterarguments: vec![],
            support_level: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        artifact.narrative = "The evidence suggests rapid permafrost thaw.".into();

        let yaml = artifact.to_yaml().expect("to_yaml must succeed");
        let restored = SynthesisArtifact::from_yaml(&yaml).expect("from_yaml must succeed");

        assert_eq!(restored.schema_version, artifact.schema_version);
        assert_eq!(restored.study_id, artifact.study_id);
        assert_eq!(restored.model, artifact.model);
        assert_eq!(restored.prompt_version, artifact.prompt_version);
        assert_eq!(restored.code_version, artifact.code_version);
        assert_eq!(restored.claims.len(), 1);
        assert_eq!(restored.claims[0].text, "Climate change accelerates permafrost thaw.");
        assert_eq!(restored.narrative, artifact.narrative);
    }

    /// ENGINE-01: ArgumentationGraph YAML round-trips with provenance fields.
    #[test]
    fn argumentation_graph_yaml_round_trip() {
        let mut graph = ArgumentationGraph::new(
            "study-001",
            "round-1",
            "q-001",
            "google/gemini-2.5-flash",
            "v1/graph",
        );
        graph.nodes.push(GraphNode {
            id: "arxiv:2105.14103_c001".into(),
            claim: "Permafrost thaw releases methane.".into(),
            expert_id: "arxiv:2105.14103".into(),
            quote: Some("permafrost thaw releases significant methane".into()),
            verification_status: Some("verified".into()),
        });

        let yaml = graph.to_yaml().expect("to_yaml must succeed");
        let restored = ArgumentationGraph::from_yaml(&yaml).expect("from_yaml must succeed");

        assert_eq!(restored.schema_version, graph.schema_version);
        assert_eq!(restored.prompt_version, "v1/graph");
        assert_eq!(restored.nodes.len(), 1);
        assert_eq!(restored.nodes[0].id, "arxiv:2105.14103_c001");
    }

    /// SCHEMA_VERSION const is "1.0".
    #[test]
    fn schema_version_is_one_zero() {
        assert_eq!(SCHEMA_VERSION, "1.0");
    }

    // ── v2 additive fields — byte-identity + round-trip tests (B1) ───────────

    /// Test 3 (v1 byte-identity): a v1-shaped SynthesisArtifact (v2 fields absent)
    /// serialises to YAML that contains NONE of the v2-only strings.
    /// This test proves the `skip_serializing_if` guards work correctly.
    #[test]
    fn v1_yaml_contains_no_v2_keys() {
        let mut artifact = SynthesisArtifact::new(
            "study-001",
            "round-1",
            "q-001",
            "google/gemini-2.5-flash",
            "v1/synthesis",
        );
        artifact.claims.push(Claim {
            text: "Permafrost thaw accelerates methane release.".into(),
            agreement_level: Some("consensus".into()),
            sources: vec!["arxiv:2105.14103".into()],
            counterarguments: vec![],
            // v2 fields absent (None / empty)
            support_level: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        artifact.uncertainties.push("Rate of feedback uncertain.".into());

        let yaml = artifact.to_yaml().expect("to_yaml must succeed");

        // None of the v2-only keys must appear in v1 YAML output.
        assert!(
            !yaml.contains("support_level"),
            "v1 YAML must not contain 'support_level'; got:\n{yaml}"
        );
        assert!(
            !yaml.contains("evidence_grade"),
            "v1 YAML must not contain 'evidence_grade'; got:\n{yaml}"
        );
        assert!(
            !yaml.contains("method:"),
            "v1 YAML must not contain 'method:'; got:\n{yaml}"
        );
        assert!(
            !yaml.contains("year:"),
            "v1 YAML must not contain 'year:'; got:\n{yaml}"
        );
        assert!(
            !yaml.contains("lineage"),
            "v1 YAML must not contain 'lineage'; got:\n{yaml}"
        );
        assert!(
            !yaml.contains("gaps:"),
            "v1 YAML must not contain 'gaps:'; got:\n{yaml}"
        );
        assert!(
            !yaml.contains("gap_type"),
            "v1 YAML must not contain 'gap_type'; got:\n{yaml}"
        );
    }

    /// Test 4 (v2 round-trip): a Claim with all five v2 fields and a
    /// SynthesisArtifact with gaps survive `to_yaml` → `from_yaml` exactly.
    #[test]
    fn v2_fields_round_trip_yaml() {
        use super::Gap;

        let mut artifact = SynthesisArtifact::new(
            "study-v2",
            "round-1",
            "q-v2",
            "claude-haiku-4-5-20251001",
            "v2/lit-review",
        );
        artifact.claims.push(Claim {
            text: "Methane release is accelerating.".into(),
            agreement_level: None,
            sources: vec!["arxiv:2304.07620".into()],
            counterarguments: vec![],
            support_level: Some("converging".into()),
            evidence_grade: Some("moderate".into()),
            method: Some("observational".into()),
            year: Some("2023".into()),
            lineage: Some("Follows from earlier permafrost studies.".into()),
            quotes: vec![],
            node_refs: vec![],
            citation: None,
        });
        artifact.gaps.push(Gap {
            description: "Unknown long-term feedback rate.".into(),
            gap_type: Some("epistemic".into()),
        });
        artifact.gaps.push(Gap {
            description: "Missing field measurements.".into(),
            gap_type: None, // gap_type absent
        });

        let yaml = artifact.to_yaml().expect("to_yaml must succeed");
        let restored = SynthesisArtifact::from_yaml(&yaml).expect("from_yaml must succeed");

        let claim = &restored.claims[0];
        assert_eq!(claim.support_level.as_deref(), Some("converging"));
        assert_eq!(claim.evidence_grade.as_deref(), Some("moderate"));
        assert_eq!(claim.method.as_deref(), Some("observational"));
        assert_eq!(claim.year.as_deref(), Some("2023"));
        assert_eq!(claim.lineage.as_deref(), Some("Follows from earlier permafrost studies."));

        assert_eq!(restored.gaps.len(), 2);
        assert_eq!(restored.gaps[0].gap_type.as_deref(), Some("epistemic"));
        assert_eq!(restored.gaps[0].description, "Unknown long-term feedback rate.");
        assert!(restored.gaps[1].gap_type.is_none(), "second gap must have gap_type None");
    }

    /// to_markdown renders claims grouped by source, edges with claim
    /// previews, and annotations — operator-perusable without ID lookup.
    #[test]
    fn graph_to_markdown_renders_all_sections() {
        let mut graph = ArgumentationGraph::new(
            "study-001",
            "round-1",
            "q-001",
            "claude-haiku-4-5-20251001",
            "v1/graph",
        );
        graph.nodes.push(GraphNode {
            id: "arxiv:2105.14103_c001".into(),
            claim: "Permafrost thaw releases methane.".into(),
            expert_id: "arxiv:2105.14103".into(),
            quote: Some("permafrost thaw releases significant methane".into()),
            verification_status: Some("verified".into()),
        });
        graph.nodes.push(GraphNode {
            id: "s2:abc_c001".into(),
            claim: "Methane release rates remain uncertain.".into(),
            expert_id: "s2:abc".into(),
            quote: None,
            verification_status: None,
        });
        graph.edges.push(GraphEdge {
            source: "s2:abc_c001".into(),
            target: "arxiv:2105.14103_c001".into(),
            relation: "contradicts".into(),
        });
        graph.node_annotations.push(NodeAnnotation {
            node_id: "arxiv:2105.14103_c001".into(),
            annotation: "high confidence".into(),
        });

        let md = graph.to_markdown();

        assert!(md.contains("# Argumentation graph — study-001 / round-1"));
        assert!(md.contains("## Claims by source"));
        assert!(md.contains("### arxiv:2105.14103 (1 claims)"));
        assert!(md.contains("[verified] — Permafrost thaw releases methane."));
        assert!(md.contains("> permafrost thaw releases significant methane"));
        assert!(md.contains("[unverified] — Methane release rates remain uncertain."));
        assert!(md.contains("### contradicts (1)"));
        // Edge line carries both claim previews, not just IDs.
        assert!(md.contains(
            "- `s2:abc_c001` \"Methane release rates remain uncertain.\" \
             → `arxiv:2105.14103_c001` \"Permafrost thaw releases methane.\""
        ));
        assert!(md.contains("## Annotations"));
        assert!(md.contains("- `arxiv:2105.14103_c001`: high confidence"));
        assert!(
            md.contains("2 claims from 2 sources · 1 edges · 1 annotations"),
            "summary line must count nodes/sources/edges/annotations"
        );
    }
}
