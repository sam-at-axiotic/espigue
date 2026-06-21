//! Shared memory/weave types used across alzina-memory and alzina-orchestration.

use serde::{Deserialize, Serialize};

/// Task category for weave classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCategory {
    Routine,
    SideEffect,
    RuneCarving,
}

/// Delta severity for stitch observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaSeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl DeltaSeverity {
    /// Whether this severity is significant (MEDIUM or above).
    pub fn is_significant(&self) -> bool {
        matches!(self, Self::Medium | Self::High | Self::Critical)
    }
}

/// Stitch trigger type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StitchTrigger {
    DurableStateChange,
    SideEffect,
    RiskBoundary,
}

/// Weave lifecycle status (canonical definition).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeaveStatus {
    Open,
    Spawning,
    Synthesising,
    Closed,
    Abandoned,
    Interrupted,
    /// Phase 7: weave is paused waiting on a HITL engagement resolution.
    /// All threads inside the weave are gated at the runner before sending
    /// the next turn. Other weaves keep running.
    Engaged,
}

/// Daily entry section type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntrySection {
    WeaveOpen,
    WeaveClose,
    Stitch,
    Delta,
    Lesson,
    Note,
    RuneProposal,
    StateChange,
}

impl EntrySection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WeaveOpen => "weave_open",
            Self::WeaveClose => "weave_close",
            Self::Stitch => "stitch",
            Self::Delta => "delta",
            Self::Lesson => "lesson",
            Self::Note => "note",
            Self::RuneProposal => "rune_proposal",
            Self::StateChange => "state_change",
        }
    }

    pub fn display_header(&self) -> &'static str {
        match self {
            Self::WeaveOpen => "Open Weaves",
            Self::WeaveClose => "Completed Weaves",
            Self::Stitch => "Stitches",
            Self::Delta => "Top Deltas",
            Self::Lesson => "Lesson Candidates",
            Self::Note => "Notes",
            Self::RuneProposal => "Rune Proposals",
            Self::StateChange => "State Changes",
        }
    }
}

/// Provenance tag for learnings entries (RD-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningsProvenance {
    Envelope,
    Reflection,
}

/// Semantic memory entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticType {
    Pattern,
    Preference,
    Decision,
    Concept,
    Relationship,
}

impl SemanticType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pattern => "pattern",
            Self::Preference => "preference",
            Self::Decision => "decision",
            Self::Concept => "concept",
            Self::Relationship => "relationship",
        }
    }
}

/// Source type for search/index entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Daily,
    Learning,
    Weave,
    Stitch,
    Kb,
    Semantic,
}

impl SourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Learning => "learning",
            Self::Weave => "weave",
            Self::Stitch => "stitch",
            Self::Kb => "kb",
            Self::Semantic => "semantic",
        }
    }
}

/// Weave lifecycle events bridging orchestration and memory.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum WeaveEvent {
    Opened {
        weave_id: String,
        label: String,
        category: TaskCategory,
    },
    StitchAdded {
        weave_id: String,
        stitch_id: String,
        trigger: StitchTrigger,
    },
    StitchClosed {
        weave_id: String,
        stitch_id: String,
        delta_severity: DeltaSeverity,
    },
    Closed {
        weave_id: String,
        outcome: String,
        dod_met: bool,
    },
    Abandoned {
        weave_id: String,
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_severity_ordering() {
        assert!(DeltaSeverity::None < DeltaSeverity::Low);
        assert!(DeltaSeverity::Low < DeltaSeverity::Medium);
        assert!(DeltaSeverity::Medium < DeltaSeverity::High);
        assert!(DeltaSeverity::High < DeltaSeverity::Critical);
    }

    #[test]
    fn delta_significance() {
        assert!(!DeltaSeverity::None.is_significant());
        assert!(!DeltaSeverity::Low.is_significant());
        assert!(DeltaSeverity::Medium.is_significant());
        assert!(DeltaSeverity::High.is_significant());
        assert!(DeltaSeverity::Critical.is_significant());
    }

    #[test]
    fn weave_status_serde_round_trip() {
        let s = serde_json::to_string(&WeaveStatus::Synthesising).unwrap();
        assert_eq!(s, "\"synthesising\"");
        let v: WeaveStatus = serde_json::from_str(&s).unwrap();
        assert_eq!(v, WeaveStatus::Synthesising);
    }

    #[test]
    fn weave_status_engaged_serde_round_trip() {
        let s = serde_json::to_string(&WeaveStatus::Engaged).unwrap();
        assert_eq!(s, "\"engaged\"");
        let v: WeaveStatus = serde_json::from_str(&s).unwrap();
        assert_eq!(v, WeaveStatus::Engaged);
    }

    #[test]
    fn entry_section_as_str() {
        assert_eq!(EntrySection::WeaveOpen.as_str(), "weave_open");
        assert_eq!(EntrySection::Delta.as_str(), "delta");
    }

    #[test]
    fn semantic_type_serde_round_trip() {
        let s = serde_json::to_string(&SemanticType::Pattern).unwrap();
        assert_eq!(s, "\"pattern\"");
        let v: SemanticType = serde_json::from_str(&s).unwrap();
        assert_eq!(v, SemanticType::Pattern);
    }

    #[test]
    fn source_type_as_str() {
        assert_eq!(SourceType::Daily.as_str(), "daily");
        assert_eq!(SourceType::Learning.as_str(), "learning");
        assert_eq!(SourceType::Weave.as_str(), "weave");
        assert_eq!(SourceType::Stitch.as_str(), "stitch");
        assert_eq!(SourceType::Kb.as_str(), "kb");
        assert_eq!(SourceType::Semantic.as_str(), "semantic");
    }
}
