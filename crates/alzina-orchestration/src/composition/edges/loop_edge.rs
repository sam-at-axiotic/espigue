//! LoopEdge — counter-based back-edge for `↻` and `↻?` operators.
//!
//! Not an ADK Node — this is edge logic used by the CompositionCompiler
//! to wire conditional back-edges in the StateGraph. Tracks iteration
//! count in the `_meta:iteration` state channel and returns `Continue`
//! or `Exhausted` based on `max_iterations`.
//!
//! On exhaustion: configurable `ExhaustAction` (error, escalate, accept-last).

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, warn};

use alzina_core::composition::ExhaustAction;

/// The decision from evaluating a loop edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopDecision {
    /// Continue iterating — increment counter and loop back.
    Continue,
    /// Loop exhausted — max iterations reached.
    Exhausted,
}

/// Configuration for a loop edge.
#[derive(Debug, Clone)]
pub struct LoopEdgeConfig {
    pub max_iterations: usize,
    pub on_exhaust: ExhaustAction,
}

impl LoopEdgeConfig {
    pub fn new(max_iterations: usize, on_exhaust: ExhaustAction) -> Self {
        Self {
            max_iterations,
            on_exhaust,
        }
    }
}

/// LoopEdge: evaluates whether to continue or exhaust a loop.
///
/// This is a pure function over state — it reads `_meta:iteration`,
/// increments it, and decides whether to continue. The state updates
/// are returned as key-value pairs for the caller to apply.
pub struct LoopEdge {
    config: LoopEdgeConfig,
}

impl LoopEdge {
    pub fn new(config: LoopEdgeConfig) -> Self {
        Self { config }
    }

    /// Evaluate the loop edge given the current iteration count from state.
    ///
    /// Returns `(decision, state_updates)` where state_updates should be
    /// applied to the graph state.
    pub fn evaluate(
        &self,
        current_iteration: Option<usize>,
    ) -> (LoopDecision, Vec<(String, serde_json::Value)>) {
        let iteration = current_iteration.unwrap_or(0);
        let next_iteration = iteration + 1;

        debug!(
            iteration = next_iteration,
            max = self.config.max_iterations,
            "loop edge evaluation"
        );

        let mut updates = vec![("_meta:iteration".to_string(), json!(next_iteration))];

        if next_iteration >= self.config.max_iterations {
            warn!(
                iteration = next_iteration,
                max = self.config.max_iterations,
                exhaust_action = ?self.config.on_exhaust,
                "loop exhausted"
            );
            updates.push(("_meta:loop_exhausted".to_string(), json!(true)));
            updates.push((
                "_meta:exhaust_action".to_string(),
                json!(format!("{:?}", self.config.on_exhaust)),
            ));
            (LoopDecision::Exhausted, updates)
        } else {
            (LoopDecision::Continue, updates)
        }
    }

    /// Convenience: extract current iteration from a serde_json::Value
    /// (as stored in state under `_meta:iteration`).
    pub fn iteration_from_state(state_value: Option<&serde_json::Value>) -> Option<usize> {
        state_value.and_then(|v| v.as_u64()).map(|n| n as usize)
    }

    /// The configured exhaustion action.
    pub fn on_exhaust(&self) -> &ExhaustAction {
        &self.config.on_exhaust
    }

    /// Maximum iterations allowed.
    pub fn max_iterations(&self) -> usize {
        self.config.max_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edge(max: usize, action: ExhaustAction) -> LoopEdge {
        LoopEdge::new(LoopEdgeConfig::new(max, action))
    }

    #[test]
    fn continues_below_max() {
        let edge = make_edge(3, ExhaustAction::Fail);

        // First iteration (0 → 1)
        let (decision, updates) = edge.evaluate(None);
        assert_eq!(decision, LoopDecision::Continue);
        assert_eq!(updates[0], ("_meta:iteration".to_string(), json!(1)));

        // Second iteration (1 → 2)
        let (decision, _) = edge.evaluate(Some(1));
        assert_eq!(decision, LoopDecision::Continue);
    }

    #[test]
    fn exhausts_at_max() {
        let edge = make_edge(3, ExhaustAction::Fail);

        // Third iteration (2 → 3) should exhaust
        let (decision, updates) = edge.evaluate(Some(2));
        assert_eq!(decision, LoopDecision::Exhausted);
        assert_eq!(updates[0], ("_meta:iteration".to_string(), json!(3)));
        assert_eq!(
            updates[1],
            ("_meta:loop_exhausted".to_string(), json!(true))
        );
    }

    #[test]
    fn exhaust_action_recorded_in_state() {
        let edge = make_edge(1, ExhaustAction::Escalate);

        let (decision, updates) = edge.evaluate(None);
        assert_eq!(decision, LoopDecision::Exhausted);

        let action_update = updates
            .iter()
            .find(|(k, _)| k == "_meta:exhaust_action")
            .unwrap();
        assert_eq!(action_update.1, json!("Escalate"));
    }

    #[test]
    fn accept_last_on_exhaust() {
        let edge = make_edge(2, ExhaustAction::AcceptLast);

        let (decision, updates) = edge.evaluate(Some(1));
        assert_eq!(decision, LoopDecision::Exhausted);

        let action_update = updates
            .iter()
            .find(|(k, _)| k == "_meta:exhaust_action")
            .unwrap();
        assert_eq!(action_update.1, json!("AcceptLast"));
    }

    #[test]
    fn iteration_from_state_helper() {
        assert_eq!(LoopEdge::iteration_from_state(None), None);
        assert_eq!(LoopEdge::iteration_from_state(Some(&json!(5))), Some(5));
        assert_eq!(
            LoopEdge::iteration_from_state(Some(&json!("not a number"))),
            None
        );
        assert_eq!(LoopEdge::iteration_from_state(Some(&json!(0))), Some(0));
    }

    #[test]
    fn single_iteration_loop() {
        let edge = make_edge(1, ExhaustAction::Fail);

        // First and only iteration should exhaust immediately
        let (decision, _) = edge.evaluate(None);
        assert_eq!(decision, LoopDecision::Exhausted);
    }

    #[test]
    fn edge_accessors() {
        let edge = make_edge(5, ExhaustAction::AcceptLast);
        assert_eq!(edge.max_iterations(), 5);
        assert!(matches!(edge.on_exhaust(), ExhaustAction::AcceptLast));
    }
}
