//! Hook lifecycle types — pre-spawn, bootstrap, and completion hooks.
//!
//! Hooks are the governance layer's extension points. They fire at defined
//! lifecycle moments and can inspect, inject, block, or route.

use crate::bootstrap::{BootstrapContext, BootstrapFragment, SessionType};
use crate::engagement::EngagementRequest;
use crate::envelope::RawEnvelope;
use crate::error::AlzinaResult;
use crate::identity::{AgentId, SessionId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Phase 7: thin orlog snapshot carried in lifecycle events.
///
/// Per CONTEXT.md § "Decision: thin OrlogSummary in alzina-core",
/// this is a snapshot — NOT the full `OrlogBlueprint` (which lives
/// in alzina-governance). Hooks that need the full blueprint should
/// re-read it from disk via `OrlogBlueprint`'s deserialiser.
///
/// `hitl_mode` is the serde-string form of governance's `HitlMode`
/// enum (e.g. "default", "auto", "force_hitl"); core does not depend
/// on the governance crate, so the string form crosses the boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrlogSummary {
    pub weave_id: String,
    pub classification: String,
    pub hitl_mode: String,
    pub goal: String,
    pub study_hook: String,
    pub phases_count: usize,
    pub stop_conditions: Vec<String>,
}

/// Phase 7: which kind of mid-execution orlog amendment fired.
///
/// Per CONTEXT.md § "Locked design decisions" #4 — orlog is a living
/// document and these mutations are HITL-gated by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmendKind {
    EmergentToMain,
    NextToMain,
    ClassificationReroute,
    ScopeChange,
}

/// Lifecycle events that trigger hook pipelines.
#[derive(Debug, Clone)]
pub enum LifecycleEvent {
    PreSpawn {
        agent_id: AgentId,
        task: String,
        parent_session: Option<SessionId>,
        /// D7 (D5-P1-3): the chat root session ID when this spawn was
        /// initiated by `dispatch_agent` from a chat turn. `None` for
        /// internal/composition spawns.
        chat_root: Option<String>,
    },
    Bootstrap {
        agent_id: AgentId,
        session_id: SessionId,
        context: BootstrapContext,
        session_type: SessionType,
    },
    Complete {
        agent_id: AgentId,
        session_id: SessionId,
        envelope: RawEnvelope,
    },
    /// Phase 7: fires once per orlog construction, after orlag gates
    /// pass and before any spawn. Drives the `OrlogSignoffHook`
    /// (Plan 7-04). `MandatoryEvent::OrlogReady` registration in the
    /// engine ensures no orlog is signed off without an explicit hook.
    OrlogReady {
        weave_id: String,
        summary: OrlogSummary,
    },
    /// Phase 7: fires for mid-execution promotions (emergent → main,
    /// next → main, classification reroute, scope change). Drives
    /// `OrlogAmendHook` (Plan 7-04).
    OrlogAmend {
        weave_id: String,
        summary: OrlogSummary,
        kind: AmendKind,
        /// Free-form diff payload (the specific fields that changed).
        /// Hooks treat this as opaque context for the human reviewer.
        diff: Value,
    },
    /// Phase 7: fires when an orlog stop condition is matched.
    /// Drives `StopConditionHook` (Plan 7-04). Auto-mode falls back
    /// to `Block` (preserves today's auto-halt).
    StopConditionTripped {
        weave_id: String,
        summary: OrlogSummary,
        condition: String,
        tripped_by: String,
    },
    /// Phase 11 (C8.1): fires when the operator overrides a tripped
    /// stop condition by picking `continue` on `StopConditionHook`'s
    /// Choice engagement (i.e. the orchestrator did NOT halt or amend
    /// the orlog — it kept going). Drives
    /// `StopConditionJustificationHook`, which engages FreeForm to
    /// capture the operator's justification for the audit trail.
    ///
    /// `choice` carries the resolved option string (`"continue"` for
    /// the override path; other values are emitted too so a future
    /// observer hook can react to `"amend-orlog"` resolutions). The
    /// `condition` and `tripped_by` fields mirror the originating
    /// `StopConditionTripped` event so the justification hook can
    /// reconstruct the override context without re-reading state.
    StopConditionOverridden {
        weave_id: String,
        summary: OrlogSummary,
        condition: String,
        tripped_by: String,
        choice: String,
    },
}

/// Action returned by a hook handler after execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookAction {
    /// Proceed to next hook in the pipeline.
    Continue,
    /// Stop pipeline and use this response directly.
    ShortCircuit(String),
    /// Reject the operation with a reason.
    Block(String),
    /// Add content to the bootstrap context.
    Inject(BootstrapFragment),
    /// Hand off to the engagement broker. The runner pauses the
    /// originating weave at the next safe point and resolves via
    /// `EngagementBroker::request_engagement`. Hooks SHOULD construct
    /// the request via `EngagementRequest::new` so the
    /// `BlockOnPartialDialogue` invariant is enforced.
    ///
    /// See `crates/alzina-core/src/engagement.rs` for the contract.
    Engage(EngagementRequest),
}

/// Shared mutable state passed through a hook pipeline for coordination.
///
/// # Design note: intentional shared-mutable pattern
///
/// `SagaState` is deliberately passed as `&mut` to each handler in sequence,
/// implementing the **saga pattern**: each hook can observe and extend the state
/// left by prior hooks. This enables cross-hook coordination (e.g., a validation
/// hook recording context that a later audit hook consumes).
///
/// Because handlers share mutable state, hooks **must be idempotent** — a hook
/// that runs twice with the same inputs should produce the same state mutations.
/// If a later handler blocks the pipeline, earlier saga mutations are NOT rolled
/// back. Hooks should write defensively (check-before-insert) rather than assuming
/// they are the sole writer to any given key.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SagaState {
    pub entries: BTreeMap<String, Value>,
}

/// A handler for a specific lifecycle event.
///
/// Implementations are registered with the `HookRunner` and executed
/// in priority order when their lifecycle event fires.
#[async_trait]
pub trait HookHandler: Send + Sync {
    /// Hook identity for ordering and logging.
    fn name(&self) -> &str;

    /// Priority for execution ordering (lower = earlier).
    fn priority(&self) -> u32;

    /// Whether this hook blocks progression on failure.
    fn blocking(&self) -> bool;

    /// Execute the hook. Returns Ok(action) or Err.
    async fn execute(
        &self,
        event: &LifecycleEvent,
        saga: &mut SagaState,
    ) -> AlzinaResult<HookAction>;
}

/// A non-blocking hook failure record — captures the hook name and error
/// so callers can inspect failures even when the pipeline continues.
#[derive(Debug, Clone)]
pub struct HookFailure {
    /// Name of the hook that failed.
    pub hook_name: String,
    /// Error message from the failed hook.
    pub error: String,
}

/// Outcome of running a complete hook pipeline.
#[derive(Debug, Clone)]
pub struct HookOutcome {
    pub actions: Vec<(String, HookAction)>,
    pub blocked: bool,
    pub block_reason: Option<String>,
    /// Non-blocking hook failures recorded during pipeline execution.
    /// Callers can inspect these even when the pipeline completed successfully.
    pub failed_hooks: Vec<HookFailure>,
    /// HITL: the engagement that early-terminated the pipeline (if
    /// any). When `Some`, callers must route `request` through an
    /// `EngagementBroker` and re-run / continue based on the
    /// `EngagementResolution`. The hook name that produced the
    /// engagement is the LAST entry of `actions`.
    pub engaged: Option<EngagedOutcome>,
}

/// Records a pending engagement on a `HookOutcome`.
#[derive(Debug, Clone)]
pub struct EngagedOutcome {
    pub hook_name: String,
    pub request: EngagementRequest,
}

/// Runs all registered hooks for a lifecycle event in priority order.
pub struct HookRunner {
    hooks: BTreeMap<u32, Vec<Box<dyn HookHandler>>>,
}

impl HookRunner {
    pub fn new() -> Self {
        Self {
            hooks: BTreeMap::new(),
        }
    }

    /// Register a hook handler.
    pub fn register(&mut self, handler: Box<dyn HookHandler>) {
        let priority = handler.priority();
        self.hooks.entry(priority).or_default().push(handler);
    }

    /// Run all hooks for a lifecycle event in priority order.
    /// Blocking hooks abort on error. Non-blocking hooks log and continue.
    pub async fn run(&self, event: &LifecycleEvent) -> AlzinaResult<HookOutcome> {
        let mut saga = SagaState::default();
        let mut actions = Vec::new();
        let mut failed_hooks = Vec::new();
        let mut blocked = false;
        let mut block_reason = None;

        for handlers in self.hooks.values() {
            for handler in handlers {
                match handler.execute(event, &mut saga).await {
                    Ok(action) => {
                        let name = handler.name().to_string();
                        if let HookAction::Engage(request) = &action {
                            let name_for_engage = name.clone();
                            let request_clone = request.clone();
                            actions.push((name, action));
                            return Ok(HookOutcome {
                                actions,
                                blocked: false,
                                block_reason: None,
                                failed_hooks,
                                engaged: Some(EngagedOutcome {
                                    hook_name: name_for_engage,
                                    request: request_clone,
                                }),
                            });
                        }
                        if let HookAction::Block(reason) = &action {
                            blocked = true;
                            block_reason = Some(reason.clone());
                            actions.push((name, action));
                            return Ok(HookOutcome {
                                actions,
                                blocked,
                                block_reason,
                                failed_hooks,
                                engaged: None,
                            });
                        }
                        actions.push((name, action));
                    }
                    Err(e) => {
                        if handler.blocking() {
                            return Err(e);
                        }
                        // Non-blocking: record failure for caller inspection
                        let hook_name = handler.name().to_string();
                        tracing::warn!(
                            hook = hook_name.as_str(),
                            error = %e,
                            "Non-blocking hook failed, continuing"
                        );
                        failed_hooks.push(HookFailure {
                            hook_name,
                            error: e.to_string(),
                        });
                    }
                }
            }
        }

        Ok(HookOutcome {
            actions,
            blocked,
            block_reason,
            failed_hooks,
            engaged: None,
        })
    }
}

impl Default for HookRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::AgentId;

    /// Phase 7 — HookAction::Engage round-trips through serde.
    #[test]
    fn hook_action_engage_round_trip() {
        use crate::engagement::{EngagementMode, FallbackBehavior};
        let req = EngagementRequest::new(
            "approve?".into(),
            serde_json::json!({"agent": "smidr"}),
            EngagementMode::Approval,
            None,
            FallbackBehavior::Block,
        )
        .expect("constructor accepts approval+block");
        let action = HookAction::Engage(req);
        let json = serde_json::to_string(&action).expect("serialize");
        let parsed: HookAction = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(parsed, HookAction::Engage(_)));
    }

    /// Phase 7 — HookRunner::run early-returns on Engage, mirroring Block.
    #[tokio::test]
    async fn hook_runner_early_returns_on_engage() {
        use crate::AlzinaResult;
        use crate::engagement::{EngagementMode, FallbackBehavior};

        struct ContinueHook;
        #[async_trait]
        impl HookHandler for ContinueHook {
            fn name(&self) -> &str {
                "continuer"
            }
            fn priority(&self) -> u32 {
                5
            }
            fn blocking(&self) -> bool {
                true
            }
            async fn execute(
                &self,
                _: &LifecycleEvent,
                _: &mut SagaState,
            ) -> AlzinaResult<HookAction> {
                Ok(HookAction::Continue)
            }
        }

        struct EngageHook;
        #[async_trait]
        impl HookHandler for EngageHook {
            fn name(&self) -> &str {
                "engager"
            }
            fn priority(&self) -> u32 {
                10
            }
            fn blocking(&self) -> bool {
                true
            }
            async fn execute(
                &self,
                _: &LifecycleEvent,
                _: &mut SagaState,
            ) -> AlzinaResult<HookAction> {
                let req = EngagementRequest::new(
                    "engage".into(),
                    serde_json::Value::Null,
                    EngagementMode::Approval,
                    None,
                    FallbackBehavior::Block,
                )?;
                Ok(HookAction::Engage(req))
            }
        }

        struct NeverReachedHook;
        #[async_trait]
        impl HookHandler for NeverReachedHook {
            fn name(&self) -> &str {
                "never"
            }
            fn priority(&self) -> u32 {
                20
            }
            fn blocking(&self) -> bool {
                true
            }
            async fn execute(
                &self,
                _: &LifecycleEvent,
                _: &mut SagaState,
            ) -> AlzinaResult<HookAction> {
                panic!("must not run after Engage");
            }
        }

        let mut runner = HookRunner::new();
        runner.register(Box::new(ContinueHook));
        runner.register(Box::new(EngageHook));
        runner.register(Box::new(NeverReachedHook));

        let event = LifecycleEvent::PreSpawn {
            agent_id: AgentId::new("smidr"),
            task: "t".into(),
            parent_session: None,
            chat_root: None,
        };
        let outcome = runner.run(&event).await.unwrap();
        assert!(!outcome.blocked);
        assert!(
            outcome.engaged.is_some(),
            "engaged must be Some after Engage"
        );
        let engaged = outcome.engaged.unwrap();
        assert_eq!(engaged.hook_name, "engager");
        // continuer + engager — never-reached must NOT be in actions
        assert_eq!(outcome.actions.len(), 2);
        assert_eq!(outcome.actions[0].0, "continuer");
        assert_eq!(outcome.actions[1].0, "engager");
    }

    /// D7 (D5-P1-3): hooks observe the chat_root field on PreSpawn so a
    /// future chat-aware hook can scope behaviour to dispatches initiated
    /// from a chat turn.
    #[test]
    fn pre_spawn_carries_chat_root_field() {
        let event = LifecycleEvent::PreSpawn {
            agent_id: AgentId::new("smidr"),
            task: "do work".into(),
            parent_session: None,
            chat_root: Some("chat-root-123".into()),
        };
        if let LifecycleEvent::PreSpawn { chat_root, .. } = &event {
            assert_eq!(chat_root.as_deref(), Some("chat-root-123"));
        } else {
            panic!("expected PreSpawn variant");
        }
    }

    #[test]
    fn pre_spawn_chat_root_none_for_internal_spawns() {
        let event = LifecycleEvent::PreSpawn {
            agent_id: AgentId::new("vefr"),
            task: "internal".into(),
            parent_session: None,
            chat_root: None,
        };
        if let LifecycleEvent::PreSpawn { chat_root, .. } = &event {
            assert!(chat_root.is_none());
        }
    }

    #[test]
    fn orlog_summary_round_trip() {
        let summary = OrlogSummary {
            weave_id: "W-7".into(),
            classification: "task-specific".into(),
            hitl_mode: "default".into(),
            goal: "ship phase 7".into(),
            study_hook: "does HITL reduce judgement-error rate?".into(),
            phases_count: 8,
            stop_conditions: vec!["build red".into(), "tests fail".into()],
        };
        let json = serde_json::to_string(&summary).unwrap();
        let parsed: OrlogSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.weave_id, "W-7");
        assert_eq!(parsed.phases_count, 8);
        assert_eq!(parsed.stop_conditions.len(), 2);
    }

    #[test]
    fn amend_kind_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&AmendKind::EmergentToMain).unwrap(),
            "\"emergent_to_main\""
        );
        assert_eq!(
            serde_json::to_string(&AmendKind::NextToMain).unwrap(),
            "\"next_to_main\""
        );
        assert_eq!(
            serde_json::to_string(&AmendKind::ClassificationReroute).unwrap(),
            "\"classification_reroute\""
        );
        assert_eq!(
            serde_json::to_string(&AmendKind::ScopeChange).unwrap(),
            "\"scope_change\""
        );
    }

    #[test]
    fn lifecycle_event_orlog_ready_constructs_with_summary() {
        let summary = OrlogSummary {
            weave_id: "W-1".into(),
            classification: "transferable-infra".into(),
            hitl_mode: "force_hitl".into(),
            goal: "g".into(),
            study_hook: "sh".into(),
            phases_count: 3,
            stop_conditions: vec!["sc".into()],
        };
        let event = LifecycleEvent::OrlogReady {
            weave_id: "W-1".into(),
            summary,
        };
        if let LifecycleEvent::OrlogReady { weave_id, summary } = &event {
            assert_eq!(weave_id, "W-1");
            assert_eq!(summary.hitl_mode, "force_hitl");
        } else {
            panic!("expected OrlogReady variant");
        }
    }

    #[test]
    fn lifecycle_event_orlog_amend_carries_kind_and_diff() {
        let summary = OrlogSummary {
            weave_id: "W-1".into(),
            classification: "task-specific".into(),
            hitl_mode: "default".into(),
            goal: "g".into(),
            study_hook: "sh".into(),
            phases_count: 1,
            stop_conditions: vec![],
        };
        let event = LifecycleEvent::OrlogAmend {
            weave_id: "W-1".into(),
            summary,
            kind: AmendKind::EmergentToMain,
            diff: serde_json::json!({"promoted_field": "emergent[0]"}),
        };
        if let LifecycleEvent::OrlogAmend { kind, diff, .. } = &event {
            assert_eq!(*kind, AmendKind::EmergentToMain);
            assert_eq!(diff["promoted_field"], "emergent[0]");
        } else {
            panic!("expected OrlogAmend variant");
        }
    }

    #[test]
    fn lifecycle_event_stop_condition_overridden_carries_choice_and_context() {
        let summary = OrlogSummary {
            weave_id: "W-1".into(),
            classification: "task-specific".into(),
            hitl_mode: "default".into(),
            goal: "g".into(),
            study_hook: "sh".into(),
            phases_count: 1,
            stop_conditions: vec!["sc".into()],
        };
        let event = LifecycleEvent::StopConditionOverridden {
            weave_id: "W-1".into(),
            summary,
            condition: "tool_failure_threshold".into(),
            tripped_by: "_system-runner:smidr".into(),
            choice: "continue".into(),
        };
        if let LifecycleEvent::StopConditionOverridden {
            weave_id,
            condition,
            tripped_by,
            choice,
            ..
        } = &event
        {
            assert_eq!(weave_id, "W-1");
            assert_eq!(condition, "tool_failure_threshold");
            assert_eq!(tripped_by, "_system-runner:smidr");
            assert_eq!(choice, "continue");
        } else {
            panic!("expected StopConditionOverridden variant");
        }
    }

    #[test]
    fn lifecycle_event_stop_condition_tripped_carries_condition_text() {
        let summary = OrlogSummary {
            weave_id: "W-1".into(),
            classification: "task-specific".into(),
            hitl_mode: "default".into(),
            goal: "g".into(),
            study_hook: "sh".into(),
            phases_count: 1,
            stop_conditions: vec!["sc".into()],
        };
        let event = LifecycleEvent::StopConditionTripped {
            weave_id: "W-1".into(),
            summary,
            condition: "build red for >1h".into(),
            tripped_by: "_system-watcher".into(),
        };
        if let LifecycleEvent::StopConditionTripped {
            condition,
            tripped_by,
            ..
        } = &event
        {
            assert_eq!(condition, "build red for >1h");
            assert_eq!(tripped_by, "_system-watcher");
        } else {
            panic!("expected StopConditionTripped variant");
        }
    }
}
