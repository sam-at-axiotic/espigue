//! Resource guards for the TTD denoise loop.
//!
//! Reproduces the three resource guards from consensus runner.py:509-535 and
//! runner.py:684-696. When a guard fires, the trajectory stops early and
//! returns its current best — it does NOT error.
//!
//! ## The three guards
//!
//! 1. **Wall-clock timeout** (runner.py:509-515) — checked at the TOP of each
//!    loop iteration. If `elapsed > max_stage_seconds`, stop.
//! 2. **Budget before fitness** (runner.py:516-535) — if the 6 fitness-judge
//!    spawns would push `llm_calls > max_llm_calls`, skip fitness and stop.
//! 3. **Budget after step** (runner.py:684-696) — re-checked after each
//!    denoise step's gap_identify + gap_resolve LLM calls; stop if exceeded.
//!
//! ## Placement
//!
//! These guards live in the `TtdMachine`, NOT in the CompOp `Loop` node.
//! The `Loop` node has no per-iteration budget awareness (RESEARCH note).
//!
//! ## Design
//!
//! Guards are cheap `#[inline]` predicates — zero allocation, no async.
//! The caller checks them at the seam points in `run.rs` and breaks early
//! when they fire. The caller is responsible for returning best-so-far.

/// Outcome of a resource-guard check.
///
/// When `Continue`, the caller proceeds with the next operation.
/// When `Stop`, the caller breaks the denoise loop early and returns best-so-far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardOutcome {
    /// Resource within limits — continue.
    Continue,
    /// Resource limit exceeded — stop early and return best-so-far.
    Stop,
}

/// Check the wall-clock guard (runner.py:509-515).
///
/// Called at the **top** of each denoise loop iteration, before fitness eval.
/// If `elapsed_secs > max_stage_seconds`, returns `Stop`.
///
/// Uses `>` (strictly greater) matching consensus's `if state.elapsed_time > self.max_stage_seconds`.
#[inline]
pub fn wall_clock_guard(elapsed_secs: f64, max_stage_seconds: u64) -> GuardOutcome {
    if elapsed_secs > max_stage_seconds as f64 {
        tracing::warn!(
            elapsed_secs,
            max_stage_seconds,
            "wall-clock guard fired: stopping trajectory early (runner.py:509-515)"
        );
        GuardOutcome::Stop
    } else {
        GuardOutcome::Continue
    }
}

/// Check the budget-before-fitness guard (runner.py:516-535).
///
/// Called **before** the 6 parallel fitness-judge spawns. If adding those
/// calls would push `llm_calls` over `max_llm_calls`, return `Stop`.
///
/// `fitness_call_count` is the number of calls the fitness evaluator will
/// issue (always 6 for the six parallel judge dimensions).
#[inline]
pub fn budget_before_fitness_guard(
    llm_calls: usize,
    fitness_call_count: usize,
    max_llm_calls: usize,
) -> GuardOutcome {
    if llm_calls + fitness_call_count > max_llm_calls {
        tracing::warn!(
            llm_calls,
            fitness_call_count,
            max_llm_calls,
            "budget-before-fitness guard fired: skipping fitness, stopping trajectory \
             early (runner.py:516-535)"
        );
        GuardOutcome::Stop
    } else {
        GuardOutcome::Continue
    }
}

/// Check the budget-after-step guard (runner.py:684-696).
///
/// Called **after** each denoise step's gap_identify + gap_resolve LLM calls.
/// If `llm_calls > max_llm_calls`, return `Stop` so the next iteration's
/// wall-clock check also has a chance to fire first.
#[inline]
pub fn budget_after_step_guard(llm_calls: usize, max_llm_calls: usize) -> GuardOutcome {
    if llm_calls > max_llm_calls {
        tracing::warn!(
            llm_calls,
            max_llm_calls,
            "budget-after-step guard fired: stopping trajectory early (runner.py:684-696)"
        );
        GuardOutcome::Stop
    } else {
        GuardOutcome::Continue
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::adapter::ExpertResponse;
    use crate::executor::AgentExecutor;
    use crate::ttd::fitness::FitnessEval;
    use crate::ttd::mod_types::TtdError;
    use crate::ttd::retrieval::NoopRetriever;
    use crate::ttd::stages::{DraftGen, EvalFitness, GapIdentify, GapResolve, Merger, RetrievedContext};
    use crate::ttd::state::IdentifiedGap;
    use crate::ttd::weights::GRAPH_WEIGHTS;
    use crate::ttd::{TtdConfig, TtdMachine};

    use super::*;

    // ── Unit tests for guard predicates ──────────────────────────────────────

    #[test]
    fn wall_clock_guard_continues_when_within_limit() {
        assert_eq!(
            wall_clock_guard(100.0, 1800),
            GuardOutcome::Continue,
            "100s elapsed, limit 1800s → Continue"
        );
    }

    #[test]
    fn wall_clock_guard_stops_when_exceeded() {
        assert_eq!(
            wall_clock_guard(1801.0, 1800),
            GuardOutcome::Stop,
            "1801s elapsed, limit 1800s → Stop"
        );
    }

    #[test]
    fn wall_clock_guard_at_exact_limit_continues() {
        // Exactly at the limit: `elapsed > max` is false when equal.
        assert_eq!(
            wall_clock_guard(1800.0, 1800),
            GuardOutcome::Continue,
            "exactly at limit → Continue (uses >, not >=)"
        );
    }

    #[test]
    fn budget_before_fitness_guard_continues_when_safe() {
        // 994 calls + 6 fitness = 1000; limit = 1000 → 1000 > 1000 is false → Continue.
        assert_eq!(
            budget_before_fitness_guard(994, 6, 1000),
            GuardOutcome::Continue,
            "994 + 6 = 1000 ≤ 1000 → Continue"
        );
    }

    #[test]
    fn budget_before_fitness_guard_stops_when_would_exceed() {
        // 995 + 6 = 1001 > 1000 → Stop.
        assert_eq!(
            budget_before_fitness_guard(995, 6, 1000),
            GuardOutcome::Stop,
            "995 + 6 = 1001 > 1000 → Stop"
        );
    }

    #[test]
    fn budget_after_step_guard_continues_when_within_limit() {
        assert_eq!(
            budget_after_step_guard(998, 1000),
            GuardOutcome::Continue,
            "998 calls, limit 1000 → Continue"
        );
    }

    #[test]
    fn budget_after_step_guard_stops_when_exceeded() {
        assert_eq!(
            budget_after_step_guard(1001, 1000),
            GuardOutcome::Stop,
            "1001 calls, limit 1000 → Stop"
        );
    }

    // ── Integration tests: guards wired into TtdMachine::run() ───────────────

    // Mock helpers for integration tests.

    struct CountingDraftGen {
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl DraftGen<String> for CountingDraftGen {
        async fn generate(
            &self,
            _inputs: &[ExpertResponse],
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
            _persona_prompt: Option<&str>,
            _sampling: Option<crate::executor::SamplingParams>,
        ) -> Result<String, TtdError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok("draft".to_string())
        }
    }

    struct CountingGapIdentify {
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl GapIdentify<String> for CountingGapIdentify {
        async fn identify(
            &self,
            _draft: &String,
            _fitness: &FitnessEval,
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
        ) -> Result<Vec<IdentifiedGap>, TtdError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(vec![IdentifiedGap {
                description: "test gap".into(),
                query: "test query".into(),
            }])
        }
    }

    struct NoopResolve;

    #[async_trait]
    impl GapResolve<String> for NoopResolve {
        async fn resolve(
            &self,
            draft: &String,
            _fitness: &FitnessEval,
            _gaps: &[IdentifiedGap],
            _retrieved: &[RetrievedContext],
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
        ) -> Result<String, TtdError> {
            Ok(draft.clone())
        }
    }

    struct CountingEvalFitness {
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl EvalFitness<String> for CountingEvalFitness {
        async fn evaluate(
            &self,
            _draft: &String,
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
        ) -> Result<FitnessEval, TtdError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(FitnessEval::new(vec![
                ("groundedness".to_string(), Some(4)),
                ("coverage".to_string(), Some(4)),
                ("atomicity".to_string(), Some(4)),
                ("non_redundancy".to_string(), Some(4)),
                ("relation_coherence".to_string(), Some(4)),
                ("dissent_preservation".to_string(), Some(4)),
            ]))
        }

        fn validity_fn(&self) -> fn(&FitnessEval) -> bool {
            crate::ttd::fitness::is_valid_graph
        }

        fn weights(&self) -> &'static [(&'static str, f32)] {
            GRAPH_WEIGHTS
        }
    }

    struct FirstMerger;

    #[async_trait]
    impl Merger<String> for FirstMerger {
        async fn merge(
            &self,
            candidates: &[String],
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
        ) -> Result<String, TtdError> {
            Ok(candidates.first().cloned().unwrap_or_default())
        }
    }

    struct CountingExecutor {
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AgentExecutor for CountingExecutor {
        async fn execute(
            &self,
            _agent_id: &alzina_core::identity::AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
        ) -> alzina_core::AlzinaResult<String> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok("mock_output".to_string())
        }
    }

    fn make_executor(count: Arc<AtomicUsize>) -> Arc<dyn AgentExecutor> {
        Arc::new(CountingExecutor { count })
    }

    fn make_inputs() -> Vec<ExpertResponse> {
        vec![]
    }

    /// When elapsed_secs > max_stage_seconds at loop-top, the trajectory stops
    /// early and returns its current best (not an error).
    ///
    /// We test this by setting max_stage_seconds=0 so elapsed immediately
    /// exceeds the limit on every iteration. With N=1, S=2, only gap_identify
    /// from the FIRST step fires — gap_identify count must be 0 (wall-clock
    /// fires before any step in the iteration).
    ///
    /// Actually: wall-clock fires BEFORE gap_identify in each step.
    /// With limit=0 and elapsed>0 immediately, the guard fires on step 0.
    /// gap_identify is never called.
    #[tokio::test]
    async fn wall_clock_stops_trajectory() {
        let gap_count = Arc::new(AtomicUsize::new(0));

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1;
        config.n_denoise_steps = 2;
        config.max_stage_seconds = 0; // always exceeded after any real time passes

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: gap_count.clone() }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: std::sync::Arc::new(alzina_search::bib_store::NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        let result = machine.run(&make_inputs()).await;

        // Must return Ok (not Err) — wall-clock guard returns best-so-far
        assert!(
            result.is_ok(),
            "wall-clock guard must return best-so-far (Ok), not error: {:?}", result
        );

        // Wall-clock fires before gap_identify, so gap_identify count must be 0.
        assert_eq!(
            gap_count.load(Ordering::SeqCst),
            0,
            "wall-clock guard must fire before gap_identify — gap_identify call count must be 0 \
             when max_stage_seconds=0 (runner.py:509-515)"
        );
    }

    /// When llm_calls would exceed max_llm_calls before the 6 fitness calls,
    /// fitness eval is skipped and the trajectory returns best-so-far.
    ///
    /// We set max_llm_calls=5 (< 6 fitness calls needed) so the guard fires
    /// immediately on first iteration of first trajectory.
    ///
    /// With N=1, S=2, the budget-before-fitness guard fires on step 0 of
    /// trajectory 0: eval_fitness is never called (count stays 0).
    #[tokio::test]
    async fn budget_skips_fitness() {
        let eval_count = Arc::new(AtomicUsize::new(0));

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1;
        config.n_denoise_steps = 2;
        config.max_llm_calls = 5; // < 6 needed for fitness; guard fires immediately

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: eval_count.clone() })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: std::sync::Arc::new(alzina_search::bib_store::NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        let result = machine.run(&make_inputs()).await;

        // Must return Ok — budget guard returns best-so-far
        assert!(
            result.is_ok(),
            "budget-before-fitness guard must return best-so-far (Ok): {:?}", result
        );

        // eval_fitness must not be called during the loop (budget guard fires before it)
        // NOTE: the final re-eval after the loop also checks the budget before running.
        assert_eq!(
            eval_count.load(Ordering::SeqCst),
            0,
            "budget-before-fitness guard must skip eval_fitness entirely \
             when max_llm_calls=5 < 6 fitness calls (runner.py:516-535)"
        );
    }

    /// Budget is re-checked after each step's LLM calls (gap_identify + gap_resolve).
    ///
    /// We set max_llm_calls=6 (exactly one fitness call batch) and N=1, S=2.
    /// Step 0: budget-before-fitness OK (calls=0 + 6 ≤ 6); fitness runs (calls=6);
    ///   budget-after-step: calls=6 + 1 (gap_id) + 1 (gap_res) = 8 > 6 → Stop.
    /// Step 1: never reached.
    ///
    /// gap_identify is called once (step 0) — then budget-after-step fires.
    #[tokio::test]
    async fn budget_checked_after_step() {
        let gap_count = Arc::new(AtomicUsize::new(0));

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1;
        config.n_denoise_steps = 2;
        config.max_llm_calls = 6; // one fitness batch exactly; step-calls push over

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: gap_count.clone() }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: std::sync::Arc::new(alzina_search::bib_store::NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        let result = machine.run(&make_inputs()).await;

        assert!(result.is_ok(), "budget-after-step guard must return best-so-far: {:?}", result);

        // gap_identify must have been called once (step 0 proceeds through fitness).
        assert_eq!(
            gap_count.load(Ordering::SeqCst),
            1,
            "budget-after-step guard: gap_identify must run once (step 0) then stop; \
             if 0, budget-before-fitness fired first; if 2, guard did not fire after step"
        );
    }

    /// With guards effectively unbounded (max_stage_seconds=u64::MAX, max_llm_calls=usize::MAX),
    /// the loop runs the full S=2 steps with N=1 trajectory.
    /// gap_identify count = 1 × 2 = 2.
    #[tokio::test]
    async fn guards_off_when_disabled() {
        let gap_count = Arc::new(AtomicUsize::new(0));

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1;
        config.n_denoise_steps = 2;
        config.max_stage_seconds = u64::MAX; // effectively disabled
        config.max_llm_calls = usize::MAX;   // effectively disabled

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: gap_count.clone() }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: std::sync::Arc::new(alzina_search::bib_store::NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        assert_eq!(
            gap_count.load(Ordering::SeqCst),
            2,
            "with unbounded guards, loop runs full S=2 steps: gap_identify must be called 2 \
             times (N=1 × S=2)"
        );
    }
}
