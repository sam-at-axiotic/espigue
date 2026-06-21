//! A3 — StopConditionTripped evaluator.
//!
//! Tracks per-weave / per-session counters and timestamps so the runner
//! can fire `LifecycleEvent::StopConditionTripped` when one of the live
//! arms is met:
//! - **tool_failure_threshold** (default N=5): N consecutive spawn failures on
//!   the same weave_id → `condition = "tool_failure_threshold"`.
//! - **redo_count** (default N=10): N redo cycles on the same chat session →
//!   `condition = "redo_count"`.
//! - **wall_time** (default 600s): a single weave_id exceeds wall_time_seconds
//!   → `condition = "wall_time"`.
//!
//! The **budget** arm is DEFERRED to A2 per D11-13. `budget_arm` is a
//! soft-degrade no-op returning `None`; no event ever fires for the budget
//! arm until the `OrlogBlueprint` runtime-consultation parser ships in A2.
//!
//! Each arm publishes EXACTLY ONCE per weave/session lifetime — once
//! `note_failure` trips the threshold and `tool_failure_arm` returns Some,
//! subsequent calls return None until `note_success` resets the counter.
//!
//! AC-1 loud-degradation: every trip emits `tracing::warn!` so ops sees
//! the engagement happen even before the StopConditionHook handler runs.
//!
//! Closes register A3 — see .planning/todos/pending/2026-05-13-phase-09-deferred-debt-register.md

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use alzina_core::envelope::Envelope;
use alzina_core::identity::{SessionId, WeaveId};
use alzina_governance::config::StopConditionsConfig;

/// Reason strings used in `LifecycleEvent::StopConditionTripped.condition`.
/// The variant `Budget` exists as a marker for future A2 work but never
/// fires today (`budget_arm` always returns None).
pub mod reason {
    pub const TOOL_FAILURE_THRESHOLD: &str = "tool_failure_threshold";
    pub const REDO_COUNT: &str = "redo_count";
    pub const WALL_TIME: &str = "wall_time";
    /// DEFERRED to A2 — soft-degrades; this variant never appears in a
    /// published event today. See register A2/A3.
    #[deprecated(note = "deferred to A2; see register A2/A3")]
    #[allow(dead_code)]
    pub const BUDGET: &str = "budget";
}

/// Per-weave / per-session counters + timestamps for the A3 evaluator.
///
/// All maps are wrapped in `std::sync::Mutex` (not `tokio::sync::Mutex`)
/// because the inserts/reads are quick and synchronous; holding across
/// await points is unnecessary. Lock contention is bounded by the number
/// of concurrent weaves (typically < 10).
pub struct StopConditionEvaluator {
    failures_per_weave: Mutex<HashMap<WeaveId, u32>>,
    /// Set once `tool_failure_arm` returns `Some` for a weave — guarantees
    /// the event fires EXACTLY ONCE per weave until `note_success` clears.
    failure_arm_tripped: Mutex<HashMap<WeaveId, bool>>,
    redo_per_session: Mutex<HashMap<SessionId, u32>>,
    redo_arm_tripped: Mutex<HashMap<SessionId, bool>>,
    started_at: Mutex<HashMap<WeaveId, Instant>>,
    wall_arm_tripped: Mutex<HashMap<WeaveId, bool>>,
    tool_failure_threshold: u32,
    redo_threshold: u32,
    wall_time: Duration,
}

impl StopConditionEvaluator {
    pub fn new(config: &StopConditionsConfig) -> Self {
        Self {
            failures_per_weave: Mutex::new(HashMap::new()),
            failure_arm_tripped: Mutex::new(HashMap::new()),
            redo_per_session: Mutex::new(HashMap::new()),
            redo_arm_tripped: Mutex::new(HashMap::new()),
            started_at: Mutex::new(HashMap::new()),
            wall_arm_tripped: Mutex::new(HashMap::new()),
            tool_failure_threshold: config.tool_failure_threshold,
            redo_threshold: config.redo_threshold,
            wall_time: Duration::from_secs(config.wall_time_seconds),
        }
    }

    /// Test-only constructor for fine-grained threshold control.
    #[cfg(test)]
    pub fn with_thresholds(tool_failure: u32, redo: u32, wall_seconds: u64) -> Self {
        Self::new(&StopConditionsConfig {
            tool_failure_threshold: tool_failure,
            redo_threshold: redo,
            wall_time_seconds: wall_seconds,
        })
    }

    /// Record a spawn failure on the given weave. Increments the
    /// per-weave consecutive-failure counter.
    pub fn note_failure(&self, weave_id: Option<&WeaveId>) {
        if let Some(w) = weave_id {
            let mut map = self.failures_per_weave.lock().unwrap();
            *map.entry(w.clone()).or_insert(0) += 1;
        }
    }

    /// Record a spawn success on the given weave. Resets the per-weave
    /// consecutive-failure counter to 0 AND clears the
    /// `failure_arm_tripped` flag so a subsequent burst can re-trip.
    pub fn note_success(&self, weave_id: Option<&WeaveId>) {
        if let Some(w) = weave_id {
            self.failures_per_weave.lock().unwrap().remove(w);
            self.failure_arm_tripped.lock().unwrap().remove(w);
        }
    }

    /// Record a redo cycle on the given chat session.
    pub fn note_redo(&self, session_id: &SessionId) {
        let mut map = self.redo_per_session.lock().unwrap();
        *map.entry(session_id.clone()).or_insert(0) += 1;
    }

    /// Anchor the wall-time clock for a given weave. Called once per
    /// weave-id at session-start. Safe to call multiple times — only
    /// the first sets the anchor.
    pub fn note_session_start(&self, weave_id: Option<&WeaveId>) {
        if let Some(w) = weave_id {
            self.started_at
                .lock()
                .unwrap()
                .entry(w.clone())
                .or_insert_with(Instant::now);
        }
    }

    /// Tool-failure-threshold arm. Returns `Some(reason)` once the per-weave
    /// failure counter reaches `tool_failure_threshold` (default 5),
    /// otherwise None. Fires EXACTLY ONCE per weave-id until the counter
    /// is reset via `note_success`.
    pub fn tool_failure_arm(&self, weave_id: Option<&WeaveId>) -> Option<&'static str> {
        let w = weave_id?;
        let count = *self.failures_per_weave.lock().unwrap().get(w).unwrap_or(&0);
        if count < self.tool_failure_threshold {
            return None;
        }
        let mut tripped = self.failure_arm_tripped.lock().unwrap();
        if tripped.get(w).copied().unwrap_or(false) {
            return None;
        }
        tripped.insert(w.clone(), true);
        Some(reason::TOOL_FAILURE_THRESHOLD)
    }

    /// Redo-count arm. Returns `Some(reason)` once per-session redo count
    /// reaches `redo_threshold` (default 10), fires EXACTLY ONCE per session.
    pub fn redo_arm(&self, session_id: &SessionId) -> Option<&'static str> {
        let count = *self
            .redo_per_session
            .lock()
            .unwrap()
            .get(session_id)
            .unwrap_or(&0);
        if count < self.redo_threshold {
            return None;
        }
        let mut tripped = self.redo_arm_tripped.lock().unwrap();
        if tripped.get(session_id).copied().unwrap_or(false) {
            return None;
        }
        tripped.insert(session_id.clone(), true);
        Some(reason::REDO_COUNT)
    }

    /// Wall-time arm. Returns `Some(reason)` once a weave exceeds
    /// `wall_time` (default 600s). Fires EXACTLY ONCE per weave-id.
    pub fn wall_time_arm(&self, weave_id: Option<&WeaveId>) -> Option<&'static str> {
        let w = weave_id?;
        let started = self.started_at.lock().unwrap().get(w).copied()?;
        if started.elapsed() < self.wall_time {
            return None;
        }
        let mut tripped = self.wall_arm_tripped.lock().unwrap();
        if tripped.get(w).copied().unwrap_or(false) {
            return None;
        }
        tripped.insert(w.clone(), true);
        Some(reason::WALL_TIME)
    }

    /// Budget arm — DEFERRED to A2 per D11-13. Soft-degrades to None.
    /// Never publishes an event today; this is documented in the module
    /// `//!` block. Phantom-success ban applies in reverse: do NOT publish
    /// a phantom "all clear" event either; just silence.
    pub fn budget_arm(&self, _envelope: &Envelope) -> Option<&'static str> {
        None
    }
}

impl Default for StopConditionEvaluator {
    fn default() -> Self {
        Self::new(&StopConditionsConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weave(s: &str) -> WeaveId {
        WeaveId::new(s)
    }

    /// Tool-failure threshold trips after 5 consecutive failures on the same weave.
    #[test]
    fn tool_failure_threshold_trips_after_5_consecutive_failures() {
        let eval = StopConditionEvaluator::with_thresholds(5, 10, 600);
        let w = weave("W-1");
        for _ in 0..4 {
            eval.note_failure(Some(&w));
            assert!(eval.tool_failure_arm(Some(&w)).is_none(), "no trip < 5");
        }
        eval.note_failure(Some(&w));
        assert_eq!(
            eval.tool_failure_arm(Some(&w)),
            Some(reason::TOOL_FAILURE_THRESHOLD),
            "fifth failure trips the arm"
        );
        // Idempotent — second call doesn't re-fire.
        assert!(
            eval.tool_failure_arm(Some(&w)).is_none(),
            "arm fires EXACTLY ONCE per weave"
        );
    }

    /// note_success resets the counter and tripped flag — the arm can re-fire.
    #[test]
    fn tool_failure_arm_resets_on_success() {
        let eval = StopConditionEvaluator::with_thresholds(3, 10, 600);
        let w = weave("W-1");
        for _ in 0..3 {
            eval.note_failure(Some(&w));
        }
        assert_eq!(
            eval.tool_failure_arm(Some(&w)),
            Some(reason::TOOL_FAILURE_THRESHOLD)
        );
        eval.note_success(Some(&w));
        // Counter cleared — start a new burst.
        for _ in 0..3 {
            eval.note_failure(Some(&w));
        }
        assert_eq!(
            eval.tool_failure_arm(Some(&w)),
            Some(reason::TOOL_FAILURE_THRESHOLD),
            "arm re-fires after a success reset"
        );
    }

    /// Redo-count arm trips at threshold, fires once per session.
    #[test]
    fn redo_arm_trips_at_10() {
        let eval = StopConditionEvaluator::with_thresholds(5, 10, 600);
        let sid = SessionId::new();
        for _ in 0..9 {
            eval.note_redo(&sid);
            assert!(eval.redo_arm(&sid).is_none(), "no trip < 10");
        }
        eval.note_redo(&sid);
        assert_eq!(eval.redo_arm(&sid), Some(reason::REDO_COUNT));
        assert!(eval.redo_arm(&sid).is_none(), "fires exactly once");
    }

    /// Wall-time arm trips when elapsed exceeds threshold.
    #[test]
    fn wall_time_arm_trips_after_threshold_elapsed() {
        // 1-second wall time for test feasibility.
        let eval = StopConditionEvaluator::with_thresholds(5, 10, 1);
        let w = weave("W-1");
        eval.note_session_start(Some(&w));
        // Immediately — not enough elapsed.
        assert!(
            eval.wall_time_arm(Some(&w)).is_none(),
            "no trip immediately"
        );
        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(
            eval.wall_time_arm(Some(&w)),
            Some(reason::WALL_TIME),
            "trips after 1.1s elapsed"
        );
        assert!(
            eval.wall_time_arm(Some(&w)).is_none(),
            "fires exactly once per weave"
        );
    }

    /// Budget arm soft-degrades to None — no event ever fires today.
    #[test]
    fn budget_arm_soft_degrades_to_none_no_publish() {
        let eval = StopConditionEvaluator::with_thresholds(5, 10, 600);
        let env = Envelope {
            status: alzina_core::EnvelopeStatus::Complete,
            artifacts: Vec::new(),
            signal: None,
            tensions: None,
            emergent: None,
            next: None,
            context_update: None,
        };
        assert!(
            eval.budget_arm(&env).is_none(),
            "budget arm is DEFERRED to A2 — always None"
        );
    }

    /// note_failure with None weave_id is a no-op (ad-hoc dispatches).
    #[test]
    fn note_failure_with_no_weave_is_noop() {
        let eval = StopConditionEvaluator::with_thresholds(5, 10, 600);
        // Should not panic.
        for _ in 0..10 {
            eval.note_failure(None);
        }
        assert!(eval.tool_failure_arm(None).is_none());
    }
}
