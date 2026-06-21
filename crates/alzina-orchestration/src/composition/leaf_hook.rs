//! Composition leaf-dispatch hook.
//!
//! Closes the seam between `alzina-orchestration`'s executor (which has
//! no knowledge of the daemon's `DispatchRegistry`) and `alzina-daemon`'s
//! per-parent dispatch tracking. Without this trait, composition leaves
//! would dispatch via `runner.spawn_with_id` but never register in
//! `DispatchRegistry`, breaking the Phase 7 announcement +
//! auto-continuation chain.
//!
//! Substrate-independence: the trait lives in `alzina-orchestration` so
//! the orchestration crate has no daemon-specific dependencies. The
//! daemon implements it in `alzina-daemon/src/api/dispatch_compose.rs`
//! and injects it into `OrchestratorEngine::execute_with_hook`.
//!
//! Reference: 10-RESEARCH.md § Pitfall 1 + Open Question Q1 (Option A).
//! Deviation from 10-CONTEXT.md D10-10: D10-10 said "no new mechanism";
//! research found `register_dispatch` lives in alzina-daemon and is
//! unreachable from orchestration without a seam. The trait is the
//! minimum-surface seam (one trait + one impl + one ctor arg).
//!
//! Closes register E1 — see .planning/todos/pending/2026-05-13-phase-09-deferred-debt-register.md

use std::sync::Arc;

use async_trait::async_trait;

use alzina_core::envelope::Envelope;
use alzina_core::{AlzinaResult, identity::SessionId};

/// Hook invoked once per composition leaf (per Loop iteration for body
/// leaves) BEFORE the runner dispatches the spawn.
///
/// The impl is responsible for: allocating a session id (or returning a
/// pre-allocated one), registering the dispatch in any side-tables the
/// caller maintains (e.g. the daemon's `DispatchRegistry`), and
/// returning the session id the runner should use as
/// `SpawnSpec.session_id_override`.
///
/// The `on_leaf_completed` and `on_leaf_failed` callbacks fire AFTER the
/// runner's `spawn_node.execute` returns (one or the other, never both).
/// The daemon impl publishes `SessionCompleted` / `SessionFailed` to the
/// EventBus so the `register_dispatch` watcher loop fires and decrements
/// `in_flight`. NoopLeafHook impls these as no-ops for non-daemon callers.
/// Register E1 (D11-04).
#[async_trait]
pub trait CompositionLeafHook: Send + Sync {
    /// Returns `(session_id, wrapped_task)`. The wrapped task may have
    /// artifact-directory instructions prepended by the daemon impl;
    /// non-daemon impls pass the task through unchanged.
    async fn on_leaf_dispatch(
        &self,
        compose_id: &str,
        node_id: &str,
        agent: &str,
        task: &str,
        parent_session_id: Option<&SessionId>,
    ) -> AlzinaResult<(SessionId, String)>;

    /// Invoked after a composition leaf spawn completes successfully.
    /// Daemon impl publishes `AlzinaEvent::SessionCompleted` so the
    /// `register_dispatch` watcher fires and decrements `in_flight`.
    async fn on_leaf_completed(
        &self,
        compose_id: &str,
        node_id: &str,
        agent: &str,
        session_id: &SessionId,
        envelope: &Envelope,
    );

    /// Invoked after a composition leaf spawn fails (or is cancelled).
    /// Daemon impl publishes `AlzinaEvent::SessionFailed` with the error
    /// string so the `register_dispatch` watcher fires and decrements
    /// `in_flight`. Cancelled spawns carry `error = "cancelled"` (E3 / D11-16).
    async fn on_leaf_failed(
        &self,
        compose_id: &str,
        node_id: &str,
        agent: &str,
        session_id: &SessionId,
        error: &str,
    );

    /// Plan 11-01.1 gap-closure: invoked once at the end of composition
    /// execution (success or failure). Implementations drain any
    /// pre-registered leaves that never received a terminal callback —
    /// e.g. Parallel branches aborted mid-flight by `tokio::JoinSet` drop,
    /// Sequential children short-circuited by a preceding op's failure,
    /// Conditional branches pruned by routing.
    ///
    /// Without this, those orphan leaves stay in `DispatchRegistry`
    /// in_flight forever — the wedge described in Phase 11 plan 01-01.1.
    /// The daemon impl publishes `SessionFailed{error:"composition cancelled"}`
    /// for each unfired leaf so the watcher decrements every slot.
    async fn on_composition_terminal(&self, compose_id: &str);
}

/// Default no-op hook for non-daemon callers (tests, internal patterns,
/// the legacy `engine.execute(op)` entry point).
///
/// Allocates a fresh SessionId and returns it; no side effects.
pub struct NoopLeafHook;

#[async_trait]
impl CompositionLeafHook for NoopLeafHook {
    async fn on_leaf_dispatch(
        &self,
        _compose_id: &str,
        _node_id: &str,
        _agent: &str,
        task: &str,
        _parent_session_id: Option<&SessionId>,
    ) -> AlzinaResult<(SessionId, String)> {
        Ok((SessionId::new(), task.to_string()))
    }

    async fn on_leaf_completed(
        &self,
        _compose_id: &str,
        _node_id: &str,
        _agent: &str,
        _session_id: &SessionId,
        _envelope: &Envelope,
    ) {
        // No-op: tests and internal callers don't drive bus publish.
    }

    async fn on_leaf_failed(
        &self,
        _compose_id: &str,
        _node_id: &str,
        _agent: &str,
        _session_id: &SessionId,
        _error: &str,
    ) {
        // No-op: tests and internal callers don't drive bus publish.
    }

    async fn on_composition_terminal(&self, _compose_id: &str) {
        // No-op: tests and internal callers don't have a DispatchRegistry to drain.
    }
}

/// Convenience: wrap NoopLeafHook in an Arc for ctor injection.
pub fn noop_hook() -> Arc<dyn CompositionLeafHook> {
    Arc::new(NoopLeafHook)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_core::envelope::EnvelopeStatus;
    use tokio::sync::Mutex;

    fn dummy_envelope() -> Envelope {
        Envelope {
            status: EnvelopeStatus::Complete,
            artifacts: Vec::new(),
            signal: None,
            tensions: None,
            emergent: None,
            next: None,
            context_update: None,
        }
    }

    #[tokio::test]
    async fn noop_hook_allocates_fresh_session_id() {
        let hook = NoopLeafHook;
        let (sid1, task1) = hook
            .on_leaf_dispatch("c", "n", "huginn", "do stuff", None)
            .await
            .unwrap();
        let (sid2, task2) = hook
            .on_leaf_dispatch("c", "n", "huginn", "do stuff", None)
            .await
            .unwrap();
        assert_ne!(sid1, sid2, "Noop hook must allocate fresh ids per call");
        assert_eq!(task1, "do stuff", "Noop hook passes task through");
        assert_eq!(task2, "do stuff", "Noop hook passes task through");
    }

    #[tokio::test]
    async fn noop_hook_fn_returns_arc() {
        let hook = noop_hook();
        let (sid, _task) = hook
            .on_leaf_dispatch("compose-1", "node-1", "smidr", "task", None)
            .await
            .unwrap();
        assert!(!sid.to_string().is_empty(), "sid must be non-empty");
    }

    #[tokio::test]
    async fn noop_hook_on_leaf_completed_is_noop() {
        let hook = NoopLeafHook;
        let sid = SessionId::new();
        let env = dummy_envelope();
        // Must not panic, must not block.
        hook.on_leaf_completed("c", "n", "huginn", &sid, &env).await;
    }

    #[tokio::test]
    async fn noop_hook_on_leaf_failed_is_noop() {
        let hook = NoopLeafHook;
        let sid = SessionId::new();
        // Must not panic, must not block.
        hook.on_leaf_failed("c", "n", "huginn", &sid, "boom").await;
    }

    /// Recording impl: verifies that an impl that records calls into a Vec
    /// sees on_leaf_completed and on_leaf_failed invocations work.
    struct RecordingHook {
        calls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl CompositionLeafHook for RecordingHook {
        async fn on_leaf_dispatch(
            &self,
            _compose_id: &str,
            node_id: &str,
            _agent: &str,
            task: &str,
            _parent_session_id: Option<&SessionId>,
        ) -> AlzinaResult<(SessionId, String)> {
            self.calls.lock().await.push(format!("dispatch:{node_id}"));
            Ok((SessionId::new(), task.to_string()))
        }

        async fn on_leaf_completed(
            &self,
            _compose_id: &str,
            node_id: &str,
            _agent: &str,
            _session_id: &SessionId,
            _envelope: &Envelope,
        ) {
            self.calls.lock().await.push(format!("completed:{node_id}"));
        }

        async fn on_leaf_failed(
            &self,
            _compose_id: &str,
            node_id: &str,
            _agent: &str,
            _session_id: &SessionId,
            error: &str,
        ) {
            self.calls
                .lock()
                .await
                .push(format!("failed:{node_id}:{error}"));
        }

        async fn on_composition_terminal(&self, compose_id: &str) {
            self.calls
                .lock()
                .await
                .push(format!("terminal:{compose_id}"));
        }
    }

    #[tokio::test]
    async fn recording_hook_observes_all_three_callbacks() {
        let hook = RecordingHook {
            calls: Mutex::new(Vec::new()),
        };
        let sid = SessionId::new();
        let env = dummy_envelope();
        let _ = hook
            .on_leaf_dispatch("c", "n1", "huginn", "task", None)
            .await
            .unwrap();
        hook.on_leaf_completed("c", "n2", "huginn", &sid, &env)
            .await;
        hook.on_leaf_failed("c", "n3", "huginn", &sid, "boom").await;
        let calls = hook.calls.lock().await;
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0], "dispatch:n1");
        assert_eq!(calls[1], "completed:n2");
        assert_eq!(calls[2], "failed:n3:boom");
    }
}
