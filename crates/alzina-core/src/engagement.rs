//! HITL engagement primitives — the contract layer for human-in-the-loop hooks.
//!
//! `HookAction::Engage(EngagementRequest)` (defined in `hooks.rs`) hands control
//! to the engagement broker. The broker pauses the originating weave at the
//! next safe point, surfaces the request to the human (TUI modal / chat
//! prompt), and resumes with an `EngagementResolution`.
//!
//! Architectural rule: this module stays ADK-agnostic (matches the crate-level
//! rule in `lib.rs`). Concrete broker impls live in `alzina-daemon` (HTTP
//! transport) and `alzina-cli` (chat fallback). The default
//! `BlockingFallbackBroker` in this crate preserves headless-CI behaviour by
//! resolving every request via the configured `FallbackBehavior` immediately.

use crate::error::AlzinaResult;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

/// Newtype wrapper around `Uuid` for engagement identity.
///
/// Aligns with the `WeaveId` Uuid sweep tracked separately. Engagement IDs
/// are generated daemon-side when a hook returns `HookAction::Engage` and
/// flow back through the `EngagementResolution` so the originating thread
/// can match the structured reply to its request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EngagementId(pub Uuid);

impl EngagementId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for EngagementId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EngagementId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// The four interaction shapes a hook can request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EngagementMode {
    /// Yes/no with optional rationale. Resolution.value is `bool`.
    Approval,
    /// Pick one of `options`. Resolution.value is the chosen string.
    Choice { options: Vec<String> },
    /// Open text response. Resolution.value is the string.
    FreeForm,
    /// Multi-turn back-and-forth. Resolution.value is the final structured
    /// summary; transcript carries every turn.
    Dialogue,
}

/// What happens if the human times out, abandons, or the broker has no
/// surface to forward to. Constructor enforces that
/// `BlockOnPartialDialogue` is only valid with `EngagementMode::Dialogue`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FallbackBehavior {
    Continue {
        default_value: serde_json::Value,
    },
    Block,
    Reengage {
        with_mode: Box<EngagementMode>,
    },
    /// Dialogue-mode only. Treats partial conversation as ambiguous,
    /// not as approval. Constructor enforces this is paired with
    /// `EngagementMode::Dialogue`.
    BlockOnPartialDialogue,
}

/// Author of a dialogue turn — used in transcripts so post-hoc audit can
/// distinguish system probes from human replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnAuthor {
    System,
    Human,
}

/// One turn in a dialogue transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTurn {
    pub author: TurnAuthor,
    pub content: String,
    pub at: DateTime<Utc>,
}

/// What a hook hands to the broker.
///
/// Construct via `EngagementRequest::new` so the BlockOnPartialDialogue
/// invariant is enforced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementRequest {
    pub id: EngagementId,
    pub prompt: String,
    pub context: serde_json::Value,
    pub mode: EngagementMode,
    /// Pause-time clock. None = wait indefinitely (TUI surface only).
    #[serde(default, with = "crate::engagement::serde_duration_opt")]
    pub timeout: Option<Duration>,
    pub fallback: FallbackBehavior,
}

impl EngagementRequest {
    /// Construct a new request, enforcing the BlockOnPartialDialogue
    /// invariant: that fallback is only valid for Dialogue mode.
    pub fn new(
        prompt: String,
        context: serde_json::Value,
        mode: EngagementMode,
        timeout: Option<Duration>,
        fallback: FallbackBehavior,
    ) -> AlzinaResult<Self> {
        if matches!(fallback, FallbackBehavior::BlockOnPartialDialogue)
            && !matches!(mode, EngagementMode::Dialogue)
        {
            return Err(crate::error::AlzinaError::Config(
                "FallbackBehavior::BlockOnPartialDialogue is only valid with EngagementMode::Dialogue".into(),
            ));
        }
        Ok(Self {
            id: EngagementId::new(),
            prompt,
            context,
            mode,
            timeout,
            fallback,
        })
    }
}

/// What the broker hands back when the engagement closes (one way or
/// another).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementResolution {
    pub engagement_id: EngagementId,
    pub outcome: ResolutionOutcome,
    pub transcript: Vec<DialogueTurn>,
    pub terminated_at: DateTime<Utc>,
}

/// How the engagement terminated. `Resolved` carries the structured human
/// reply; `FellBack` records the fallback that fired; `Abandoned` is for
/// daemon shutdown / cancel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolutionOutcome {
    Resolved { value: serde_json::Value },
    FellBack { reason: String },
    Abandoned,
}

/// The broker contract. Concrete impls live in alzina-daemon
/// (`DaemonEngagementBroker`) and alzina-cli (chat fallback). The
/// `BlockingFallbackBroker` below is the default for headless paths.
#[async_trait]
pub trait EngagementBroker: Send + Sync {
    /// Open an engagement and block until it resolves (success, fallback,
    /// or abandonment). Implementations MUST honour
    /// `request.fallback` when no human response arrives within
    /// `request.timeout`.
    async fn request_engagement(
        &self,
        request: EngagementRequest,
    ) -> AlzinaResult<EngagementResolution>;
}

/// Default broker: never engages a human. Resolves every request by
/// applying the configured `FallbackBehavior` immediately.
///
/// This preserves today's headless-CI behaviour: with no TUI / chat
/// surface attached, `Engage` hooks degrade gracefully to the fallback
/// rather than hanging indefinitely.
pub struct BlockingFallbackBroker;

#[async_trait]
impl EngagementBroker for BlockingFallbackBroker {
    async fn request_engagement(
        &self,
        request: EngagementRequest,
    ) -> AlzinaResult<EngagementResolution> {
        let reason = match &request.fallback {
            FallbackBehavior::Continue { .. } => {
                "fallback:continue (no broker surface)".to_string()
            }
            FallbackBehavior::Block => "fallback:block (no broker surface)".to_string(),
            FallbackBehavior::Reengage { .. } => {
                "fallback:reengage-degraded-to-block (no broker surface)".to_string()
            }
            FallbackBehavior::BlockOnPartialDialogue => {
                "fallback:block_on_partial_dialogue (no broker surface)".to_string()
            }
        };
        Ok(EngagementResolution {
            engagement_id: request.id,
            outcome: ResolutionOutcome::FellBack { reason },
            transcript: Vec::new(),
            terminated_at: Utc::now(),
        })
    }
}

/// serde helper for `Option<Duration>` (round-trips as `Option<u64>` ms).
pub(crate) mod serde_duration_opt {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(
        value: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.map(|d| d.as_millis() as u64).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        Ok(Option::<u64>::deserialize(deserializer)?.map(Duration::from_millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn engagement_request_constructor_rejects_partial_dialogue_for_approval() {
        let result = EngagementRequest::new(
            "approve?".into(),
            json!(null),
            EngagementMode::Approval,
            None,
            FallbackBehavior::BlockOnPartialDialogue,
        );
        assert!(
            result.is_err(),
            "expected Err for Approval + BlockOnPartialDialogue"
        );
    }

    #[test]
    fn engagement_request_constructor_rejects_partial_dialogue_for_choice() {
        let result = EngagementRequest::new(
            "choose".into(),
            json!(null),
            EngagementMode::Choice {
                options: vec!["a".into()],
            },
            None,
            FallbackBehavior::BlockOnPartialDialogue,
        );
        assert!(
            result.is_err(),
            "expected Err for Choice + BlockOnPartialDialogue"
        );
    }

    #[test]
    fn engagement_request_constructor_rejects_partial_dialogue_for_freeform() {
        let result = EngagementRequest::new(
            "describe".into(),
            json!(null),
            EngagementMode::FreeForm,
            None,
            FallbackBehavior::BlockOnPartialDialogue,
        );
        assert!(
            result.is_err(),
            "expected Err for FreeForm + BlockOnPartialDialogue"
        );
    }

    #[test]
    fn engagement_request_constructor_accepts_partial_dialogue_for_dialogue() {
        let result = EngagementRequest::new(
            "let's talk".into(),
            json!(null),
            EngagementMode::Dialogue,
            None,
            FallbackBehavior::BlockOnPartialDialogue,
        );
        assert!(
            result.is_ok(),
            "expected Ok for Dialogue + BlockOnPartialDialogue"
        );
    }

    #[test]
    fn engagement_request_round_trip_json() {
        let req = EngagementRequest::new(
            "approve?".into(),
            json!({"agent": "smidr", "action": "write"}),
            EngagementMode::Approval,
            None,
            FallbackBehavior::Block,
        )
        .expect("constructor accepts approval+block");
        let original_id = req.id;
        let original_prompt = req.prompt.clone();

        let json = serde_json::to_string(&req).expect("serialize");
        let parsed: EngagementRequest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.id, original_id, "id must round-trip");
        assert_eq!(parsed.prompt, original_prompt, "prompt must round-trip");
        let ctx: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(ctx["context"]["agent"], "smidr");
    }

    #[test]
    fn engagement_resolution_round_trip_json() {
        let resolution = EngagementResolution {
            engagement_id: EngagementId::new(),
            outcome: ResolutionOutcome::Resolved {
                value: json!({"approve": true}),
            },
            transcript: Vec::new(),
            terminated_at: Utc::now(),
        };
        let json = serde_json::to_string(&resolution).expect("serialize");
        let parsed: EngagementResolution = serde_json::from_str(&json).expect("deserialize");
        assert!(
            matches!(parsed.outcome, ResolutionOutcome::Resolved { .. }),
            "outcome variant must round-trip as Resolved"
        );
    }

    #[test]
    fn engagement_id_new_produces_unique() {
        let a = EngagementId::new();
        let b = EngagementId::new();
        assert_ne!(a, b, "two new EngagementIds must be distinct");
    }

    #[tokio::test]
    async fn blocking_fallback_broker_resolves_continue() {
        let req = EngagementRequest::new(
            "proceed?".into(),
            json!(null),
            EngagementMode::Approval,
            None,
            FallbackBehavior::Continue {
                default_value: json!(null),
            },
        )
        .expect("valid request");
        let broker = BlockingFallbackBroker;
        let resolution = broker
            .request_engagement(req)
            .await
            .expect("broker resolves");
        match &resolution.outcome {
            ResolutionOutcome::FellBack { reason } => {
                assert!(
                    reason.contains("fallback:continue"),
                    "reason should contain 'fallback:continue', got: {reason}"
                );
            }
            other => panic!("expected FellBack, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn blocking_fallback_broker_resolves_block() {
        let req = EngagementRequest::new(
            "block?".into(),
            json!(null),
            EngagementMode::Approval,
            None,
            FallbackBehavior::Block,
        )
        .expect("valid request");
        let broker = BlockingFallbackBroker;
        let resolution = broker
            .request_engagement(req)
            .await
            .expect("broker resolves");
        match &resolution.outcome {
            ResolutionOutcome::FellBack { reason } => {
                assert!(
                    reason.contains("fallback:block"),
                    "reason should contain 'fallback:block', got: {reason}"
                );
            }
            other => panic!("expected FellBack, got {:?}", other),
        }
    }

    #[test]
    fn engagement_mode_choice_serde() {
        let mode = EngagementMode::Choice {
            options: vec!["a".into(), "b".into()],
        };
        let json = serde_json::to_string(&mode).expect("serialize");
        let val: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        assert_eq!(val["kind"], "choice", "kind must be 'choice'");
        let parsed: EngagementMode = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            EngagementMode::Choice { options } => {
                assert_eq!(options, vec!["a", "b"], "options must round-trip");
            }
            other => panic!("expected Choice, got {:?}", other),
        }
    }
}
