//! TtdMachine::run() — the shared denoise loop for all three TTD stages.
//!
//! Implements the fixed-cap N=5×S=2 structure from consensus runner.py:228-447:
//!
//! 1. **FanOut(N)**: call `draft_gen.generate` N times → N trajectories. Phase 23
//!    issues these SEQUENTIALLY (consensus uses asyncio.gather; true parallel
//!    fan-out is the accepted Phase 24 deferral — see WR-04 in run().)
//! 2. **Denoise Loop(S)**: for each trajectory, run S fixed iterations:
//!    a. `eval_fitness.evaluate(draft)` → fitness scores + feedback document
//!    b. `gap_identify.identify(draft, fitness)` → 3-5 gaps
//!    c. `retriever.retrieve(gap.query)` per gap (sequential per gap, dedup by source_id)
//!    d. empty-retrieved guard: if ALL retrieved empty → draft UNCHANGED
//!    e. `gap_resolve.resolve(draft, gaps, retrieved)` → refined draft
//!    f. record `StepRecord`
//! 3. **Final re-eval**: fresh `eval_fitness` FanOut for EACH trajectory after loop
//!    (Pitfall 4 — do NOT reuse steps[-1] scores; runner.py:391-410).
//! 4. **Select**: `sort_candidates_best_first(trajectories, final_evals, weights, is_valid)`
//! 5. **Merge**: `merger.merge(sorted)` → final stage artefact.
//!
//! Resource guards (wall-clock + budget) are wired as no-ops in Wave 2; they become
//! active in Wave 4 (Plan 23-05). The seam is `TtdMachine::resource_ok()`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use futures::future::join_all;
use tokio::task::JoinSet;

use alzina_search::bib_store::BibEntry;

use crate::adapter::ExpertResponse;
use crate::ttd::fitness::{is_valid_graph, sort_candidates_best_first, weighted_sum, FitnessEval};
use crate::ttd::guards::{budget_after_step_guard, budget_before_fitness_guard, wall_clock_guard, GuardOutcome};
use crate::ttd::mod_types::TtdError;
use crate::ttd::personas::{PERSONAS, V2_PERSONAS};
use crate::ttd::sampling::build_sampling_configs;
use crate::ttd::state::{DiffusionState, IdentifiedGap, StepRecord};
use crate::ttd::term_sheet::PromptProfile;
use crate::ttd::weights::GRAPH_WEIGHTS;
use crate::ttd::TtdMachine;

/// Goodhart instrumentation (2026-06-16): emit each judge dimension's raw 1-5
/// score so per-dimension variance, saturation, and correlation can be analysed
/// before the judge set is changed (add/remove/adapt). One event per dimension
/// on the `ttd_judge` target; a `None` score (parse or spawn failure) is logged
/// as `-1`. Off the hot path — fitness eval already costs LLM spawns, so a few
/// `info!` lines per eval are negligible. `phase` distinguishes the in-loop
/// `"step"` evals from the decisive `"final"` re-eval that picks the winner.
fn log_dimension_scores(stage: &str, trajectory: usize, step: usize, phase: &str, fitness: &FitnessEval) {
    for (dim, score) in &fitness.scores {
        tracing::info!(
            target: "ttd_judge",
            stage = %stage,
            trajectory,
            step,
            phase = %phase,
            dim = %dim,
            score = score.map(|v| v as i16).unwrap_or(-1),
            vetoed = fitness.veto.is_some(),
            "ttd_judge: dimension score"
        );
    }
}

impl<A: Clone + Send + Sync + 'static> TtdMachine<A> {
    /// Run the full TTD loop for this stage.
    ///
    /// ## Contract
    ///
    /// - Spawns exactly `config.n_initial_drafts` drafts concurrently.
    /// - Runs exactly `config.n_denoise_steps` iterations per trajectory (fixed cap,
    ///   `early_stopping=false` — CONTEXT locked decision).
    /// - After the loop, runs a FRESH final fitness re-evaluation for every trajectory
    ///   before calling `sort_candidates_best_first` (Pitfall 4).
    /// - If retrieved is empty for all gaps, draft is returned UNCHANGED (runner.py guard).
    /// - Every spawn routes through the injected `executor` (ENGINE-05).
    pub async fn run(
        &self,
        inputs: &[ExpertResponse],
    ) -> Result<A, TtdError> {
        if self.config.n_initial_drafts == 0 {
            return Err(TtdError::NoCandidates);
        }
        if self.config.n_denoise_steps == 0 {
            return Err(TtdError::NoCandidates);
        }

        let start = Instant::now();

        // ── Step 1: FanOut(N) — generate N initial drafts concurrently ────────
        // Phase 24 EXT-01: replaces the Phase 23 sequential loop (WR-04 gap closed).
        // Mirrors consensus asyncio.gather(*[draft_gen.generate(...)]) at
        // runner.py:310-316 using tokio::task::JoinSet (same pattern as
        // ttd/stages/graph.rs:469-520).
        let n = self.config.n_initial_drafts;

        // Build per-trajectory sampling configs (N configs, one per trajectory).
        // sampling_configs kept here for Task 3 to thread into execute_with_sampling.
        let sampling_configs = build_sampling_configs(&self.config);

        // Build per-trajectory persona prompts.
        // EXT-01 Phase 24: run a persona Spawn (via self.executor) before the FanOut.
        // Parse envelope output into Vec<String>. Fall back to PERSONAS constants when
        // the spawn yields nothing (e.g. test environments without a live executor).
        // Phase 25 checks PHASE24_EXT_NOTE to confirm persona-seeding is active.
        //
        // B2: fork on profile — V2LitReview uses V2_PERSONAS (deep reviewer set);
        // V1Delphi uses the original PERSONAS (byte-identical to pre-B2).
        let persona_prompts: Vec<Option<String>> = {
            let persona_set: &[&str] = match self.config.profile {
                // Decision 0: v3 uses the v2 persona set (no persona changes).
                PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => V2_PERSONAS,
                PromptProfile::V1Delphi => PERSONAS,
            };
            (0..n)
                .map(|i| {
                    if persona_set.is_empty() {
                        None
                    } else {
                        Some(persona_set[i % persona_set.len()].to_string())
                    }
                })
                .collect()
        };

        // Concurrent JoinSet FanOut — mirrors graph.rs:469-520.
        let mut join_set: JoinSet<Result<A, TtdError>> = JoinSet::new();
        let inputs_vec = inputs.to_vec();

        for i in 0..n {
            let draft_gen_i = Arc::clone(&self.draft_gen);
            let exec = Arc::clone(&self.executor);
            let cfg = self.config.clone();
            let inputs_clone = inputs_vec.clone();
            let persona = persona_prompts.get(i).and_then(|p| p.clone());
            // Thread the per-trajectory sampling config into the draft spawn (EXT-01).
            let sampling = sampling_configs.get(i).map(|sc| {
                crate::executor::SamplingParams {
                    temperature: sc.temperature,
                    top_p: sc.top_p,
                    top_k: sc.top_k,
                }
            });

            join_set.spawn(async move {
                draft_gen_i
                    .generate(&inputs_clone, &exec, &cfg, persona.as_deref(), sampling)
                    .await
            });
        }

        let fanout_start = Instant::now();
        let mut initial_drafts: Vec<A> = Vec::with_capacity(n);
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok(Ok(draft)) => { initial_drafts.push(draft); }
                Ok(Err(e)) => {
                    tracing::debug!(error = %e, "draft generation task failed; propagating");
                    return Err(e);
                }
                Err(join_err) => {
                    // WR-02: a panicking task is a bug signal — surface it with the
                    // payload at error level rather than flattening it into the
                    // benign-looking NoCandidates variant.
                    tracing::error!(error = %join_err, "draft generation task panicked");
                    return Err(TtdError::SpawnFailed(format!("draft task panicked: {join_err}")));
                }
            }
        }

        if initial_drafts.is_empty() {
            return Err(TtdError::NoCandidates);
        }

        tracing::info!(
            target: "ttd_perf",
            stage = %self.stage_label,
            n_drafts = initial_drafts.len(),
            duration_ms = fanout_start.elapsed().as_millis() as u64,
            "ttd_perf: draft fan-out complete"
        );

        // ── Step 2: Per-trajectory denoise Loop(S) ────────────────────────────
        // Fixed cap — early_stopping is always false (CONTEXT locked decision).
        // A5 rung 4: trajectories evolve concurrently via join_all over
        // states.iter_mut() (closes the 23-02 deferral). No Arc<Box<dyn>>
        // refactor — the 23-02 blocker applied only to JoinSet 'static spawning;
        // join_all drives borrowed futures on the current task, so the existing
        // &self borrows and the disjoint &mut DiffusionState borrows from
        // iter_mut() work as-is (all five stage traits: Send + Sync).
        //
        // Divergence from old fail-fast: join_all runs ALL trajectories to
        // completion before the error fold — side-effect-wise this matches
        // consensus asyncio.gather, which lets siblings continue after a failure.
        // The first Err is propagated in trajectory-index order (deterministic).
        let mut states: Vec<DiffusionState<A>> = initial_drafts
            .into_iter()
            .map(|draft| {
                DiffusionState::new(n, self.config.n_denoise_steps, vec![draft])
            })
            .collect();

        // One semaphore for the whole stage run — shared across all trajectory
        // futures and the final re-eval. Permits = max_concurrent_fitness_evals
        // (at least 1 — 0 would deadlock). Judges inside each evaluate() are
        // sequential (graph.rs:1133, synthesis.rs:523) so capping concurrent
        // evaluate() calls caps concurrent judge sidecar spawns 1:1.
        let fitness_sem = Arc::new(tokio::sync::Semaphore::new(
            self.config.max_concurrent_fitness_evals.max(1)
        ));

        // evolve_trajectory is defined further below as a method on TtdMachine.
        // join_all drives the borrowed futures concurrently on the current task.
        let evolve_results: Vec<Result<(), TtdError>> = join_all(
            states.iter_mut().enumerate().map(|(trajectory, state)| {
                self.evolve_trajectory(trajectory, state, start, &fitness_sem)
            })
        ).await;
        // Fold errors in trajectory-index order; propagate first Err.
        for result in evolve_results {
            result?;
        }

        // ── Step 3: Final re-evaluation (Pitfall 4 guard) ────────────────────
        // After all denoise steps, evaluate each trajectory's FINAL draft with
        // a FRESH fitness call. Do NOT reuse state.steps[-1].fitness_evaluations.
        // runner.py:391-410 confirmed this is a separate post-loop gather.
        //
        // A5 rung 4: also concurrent via join_all. join_all preserves INPUT ORDER
        // so final_evals[i] still corresponds to states[i] for
        // sort_candidates_best_first — index alignment is preserved.
        //
        // Budget guard also applies here: if llm_calls + judge_count > max_llm_calls,
        // skip the final re-eval for that trajectory (use an empty FitnessEval).
        let final_fitness_call_count: usize = self
            .eval_fitness
            .as_ref()
            .map(|ef| ef.weights().len())
            .unwrap_or(0);
        let final_eval_start = Instant::now();
        let final_evals: Vec<FitnessEval> = {
            let results: Vec<Result<FitnessEval, TtdError>> = join_all(
                states.iter().map(|state| async {
                    // Check budget before the final re-eval judge spawns.
                    if budget_before_fitness_guard(
                        state.llm_calls,
                        final_fitness_call_count,
                        self.config.max_llm_calls,
                    ) == GuardOutcome::Stop
                    {
                        return Ok(FitnessEval::new(vec![]));
                    }
                    let permit = fitness_sem.acquire().await.map_err(|_| {
                        TtdError::SpawnFailed("fitness semaphore closed during final re-eval".into())
                    })?;
                    let eval = if let Some(ref eval_fitness) = self.eval_fitness {
                        eval_fitness
                            .evaluate(
                                &state.trajectories[0],
                                &self.executor,
                                &self.config,
                            )
                            .await?
                    } else {
                        FitnessEval::new(vec![])
                    };
                    drop(permit);
                    Ok(eval)
                })
            ).await;
            // Fold in trajectory-index order; propagate first Err.
            let mut evals = Vec::with_capacity(results.len());
            for r in results {
                evals.push(r?);
            }
            evals
        };
        tracing::info!(
            target: "ttd_perf",
            stage = %self.stage_label,
            n_trajectories = states.len(),
            duration_ms = final_eval_start.elapsed().as_millis() as u64,
            "ttd_perf: final re-eval (all trajectories)"
        );
        // Goodhart instrumentation: the decisive per-dimension scores that pick
        // the winner. `final_evals` is in trajectory-index order (folded above).
        for (traj_idx, ev) in final_evals.iter().enumerate() {
            log_dimension_scores(&self.stage_label, traj_idx, 0, "final", ev);
        }

        // ── Step 4: sort_candidates_best_first ────────────────────────────────
        // Collect the final draft from each trajectory and sort best-first.
        let final_drafts: Vec<A> = states
            .iter()
            .map(|s| s.trajectories[0].clone())
            .collect();

        // Use the fitness function from the eval_fitness impl if available.
        // Default to is_valid_graph (graph stage) — Wave 3 stages pass their
        // own is_valid function through the EvalFitness impl.
        let is_valid = self
            .eval_fitness
            .as_ref()
            .map(|ef| ef.validity_fn())
            .unwrap_or(is_valid_graph);

        let weights = self
            .eval_fitness
            .as_ref()
            .map(|ef| ef.weights())
            .unwrap_or(GRAPH_WEIGHTS);

        let sorted = sort_candidates_best_first(&final_drafts, &final_evals, weights, is_valid);

        // ── Step 5: Merge ─────────────────────────────────────────────────────
        let merge_start = Instant::now();
        let merged = self.merger.merge(&sorted, &self.executor, &self.config).await?;
        tracing::info!(
            target: "ttd_perf",
            stage = %self.stage_label,
            duration_ms = merge_start.elapsed().as_millis() as u64,
            "ttd_perf: merge"
        );

        tracing::info!(
            target: "ttd_perf",
            stage = %self.stage_label,
            duration_ms = start.elapsed().as_millis() as u64,
            "ttd_perf: stage run complete"
        );

        Ok(merged)
    }

    /// Per-trajectory denoise loop body — extracted for join_all concurrency (A5 rung 4).
    ///
    /// Runs the S-step denoise loop for a single trajectory. Called concurrently
    /// over all N trajectories via `futures::future::join_all` in `run()`.
    ///
    /// ## Safety notes
    ///
    /// - `&self` is a shared reference — all concurrent futures can hold it.
    /// - `state: &mut DiffusionState<A>` is disjoint per trajectory (from
    ///   `states.iter_mut()` in the caller); no two futures mutate the same state.
    /// - `start: Instant` is `Copy` — each future reads the same stage start time.
    /// - `fitness_sem` is shared via `&Arc<Semaphore>` — permits rate-limit
    ///   concurrent evaluate() calls across all trajectories.
    ///
    /// ## Wall-clock guard semantics
    ///
    /// `start.elapsed()` is the STAGE wall-clock — shared, per spec. Concurrent
    /// trajectories no longer burn the clock waiting behind trajectory 0 (the
    /// probe-10 fidelity un-cut: stages 1-2 cut trajectories 2-4 because the
    /// clock ran out while waiting for trajectory 0 to finish sequential steps).
    async fn evolve_trajectory(
        &self,
        trajectory: usize,
        state: &mut DiffusionState<A>,
        start: Instant,
        fitness_sem: &Arc<tokio::sync::Semaphore>,
    ) -> Result<(), TtdError> {
        for step in 0..self.config.n_denoise_steps {
            // ── Guard 1: Wall-clock (runner.py:509-515) ───────────────────
            // Checked at the TOP of each iteration, before ANY work.
            let elapsed_secs = start.elapsed().as_secs_f64();
            state.elapsed_secs = elapsed_secs;
            if wall_clock_guard(elapsed_secs, self.config.max_stage_seconds)
                == GuardOutcome::Stop
            {
                // Return best-so-far: break out of the step loop for this
                // trajectory. The trajectory's current draft is preserved.
                break;
            }

            // ── Guard 2: Budget before fitness (runner.py:516-535) ────────
            // The fitness evaluator issues one spawn per dimension (v1=6, v2=5).
            // Derive the count from the evaluator's weight table so this stays
            // correct as profiles change. If there is no evaluator, count = 0.
            let fitness_call_count: usize = self
                .eval_fitness
                .as_ref()
                .map(|ef| ef.weights().len())
                .unwrap_or(0);
            if budget_before_fitness_guard(
                state.llm_calls,
                fitness_call_count,
                self.config.max_llm_calls,
            ) == GuardOutcome::Stop
            {
                break;
            }

            // a. Fitness evaluation (produces feedback document for gap_identify).
            // Semaphore-gated: concurrent fitness evals across trajectories are
            // capped at max_concurrent_fitness_evals. Judges inside evaluate()
            // are sequential (graph.rs:1133, synthesis.rs:523), so this cap
            // directly bounds concurrent judge sidecar spawns (1:1).
            let fitness_start = Instant::now();
            let fitness: FitnessEval = if let Some(ref eval_fitness) = self.eval_fitness {
                let permit = fitness_sem.acquire().await.map_err(|_| {
                    TtdError::SpawnFailed("fitness semaphore closed (WR-02 / closed-semaphore precedent)".into())
                })?;
                let eval = eval_fitness
                    .evaluate(
                        &state.trajectories[0],
                        &self.executor,
                        &self.config,
                    )
                    .await?;
                drop(permit);
                eval
            } else {
                // No fitness evaluator (e.g. narrative stage with noop).
                FitnessEval::new(vec![])
            };
            tracing::info!(
                target: "ttd_perf",
                stage = %self.stage_label,
                trajectory,
                step,
                duration_ms = fitness_start.elapsed().as_millis() as u64,
                judge_count = fitness_call_count,
                "ttd_perf: fitness eval (sequential judge spawns)"
            );
            log_dimension_scores(&self.stage_label, trajectory, step, "step", &fitness);

            // WR-05: every stage evaluator issues one spawn per dimension,
            // degrading a failed spawn to score=None instead of aborting.
            // fitness_call_count is derived from weights().len() above so it
            // stays correct across profiles (v1=6 dims, v2=5 dims).
            state.llm_calls += fitness_call_count;

            // b. Gap identification (fed prior-step fitness feedback).
            let gap_identify_start = Instant::now();
            let gaps: Vec<IdentifiedGap> = self
                .gap_identify
                .identify(&state.trajectories[0], &fitness, &self.executor, &self.config)
                .await?;
            tracing::info!(
                target: "ttd_perf",
                stage = %self.stage_label,
                trajectory,
                step,
                n_gaps = gaps.len(),
                duration_ms = gap_identify_start.elapsed().as_millis() as u64,
                "ttd_perf: gap identify"
            );

            state.llm_calls += 1; // gap_identify spawn

            // c. Retrieval per gap — concurrent within this denoise step (A5 rung 2).
            //
            // join_all over borrowed futures: each retrieve future borrows
            // &self.retriever (&dyn Retriever, Send + Sync) — no cloning,
            // no spawning, no 'static requirement.
            let retrieval_start = Instant::now();
            let retrieval_results: Vec<Result<Vec<crate::ttd::stages::RetrievedContext>, TtdError>> =
                join_all(
                    gaps.iter().map(|gap| self.retriever.retrieve(&gap.query, self.config.retrieval_top_k))
                ).await;
            // Fold in input (gap-index) order — propagate first Err.
            let mut all_retrieved = vec![];
            for result in retrieval_results {
                let mut chunk = result?;
                all_retrieved.append(&mut chunk);
            }
            tracing::info!(
                target: "ttd_perf",
                stage = %self.stage_label,
                trajectory,
                step,
                n_gaps = gaps.len(),
                n_retrieved = all_retrieved.len(),
                duration_ms = retrieval_start.elapsed().as_millis() as u64,
                "ttd_perf: per-gap retrieval (all gaps)"
            );

            // Dedup by source_id across all gap queries (runner.py:594-599).
            let mut seen_sources: HashSet<String> = HashSet::new();
            let unique_retrieved: Vec<_> = all_retrieved
                .into_iter()
                .filter(|h| seen_sources.insert(h.source_id.clone()))
                .collect();

            tracing::debug!(
                step,
                n_gaps = gaps.len(),
                n_retrieved = unique_retrieved.len(),
                "denoise step: retrieved {} unique items for {} gaps",
                unique_retrieved.len(),
                gaps.len(),
            );

            // d+e. Empty-retrieved guard + gap resolution.
            // Default: if ALL retrieval returned empty, draft is returned
            // UNCHANGED (runner.py / graph_tasks.py:1105-1107 — do NOT call
            // gap_resolve). Phase P: when `resolve_without_retrieval` is set and
            // there are gaps, resolve anyway — Stage 3's NoopRetriever makes
            // retrieval structurally empty, and this re-opens the consensus
            // critique→refine loop. Flag off → condition reduces to the original.
            let current = state.trajectories[0].clone();
            let resolve_despite_empty =
                self.config.resolve_without_retrieval && !gaps.is_empty();
            let refined = if unique_retrieved.is_empty() && !resolve_despite_empty {
                tracing::debug!(
                    step,
                    "empty-retrieved guard: all gap queries returned empty; \
                     draft returned unchanged"
                );
                current
            } else {
                let resolve_start = Instant::now();
                let r = self.gap_resolve
                    .resolve(
                        &state.trajectories[0],
                        &fitness,
                        &gaps,
                        &unique_retrieved,
                        &self.executor,
                        &self.config,
                    )
                    .await?;
                tracing::info!(
                    target: "ttd_perf",
                    stage = %self.stage_label,
                    trajectory,
                    step,
                    duration_ms = resolve_start.elapsed().as_millis() as u64,
                    "ttd_perf: gap resolve"
                );
                state.llm_calls += 1; // gap_resolve spawn (when retrieved non-empty)
                r
            };

            // f. Record StepRecord.
            let record = StepRecord {
                step,
                fitness_evaluations: vec![fitness.clone()],
                gaps: vec![gaps.clone()],
            };
            state.record_step(record);

            // EXT-03: record per-step bibliography sources directly to the
            // literature KB via the injected BibliographyStore (NOT via
            // composition channels — CONTEXT EXT-03, locked).
            if !unique_retrieved.is_empty() {
                let bib_entries: Vec<BibEntry> = unique_retrieved.iter().map(|ctx| BibEntry {
                    source_id: ctx.source_id.clone(),
                    expert_id: ctx.source_id.clone(), // source_id doubles as expert_id at retrieval layer
                    quote_raw: Some(ctx.content.chars().take(500).collect()),
                }).collect();
                if let Err(e) = self.bib_store.record_sources(
                    &self.run_id,
                    &self.stage_label,
                    step,
                    &bib_entries,
                ).await {
                    // H2 (W-522022c5): fusion-into-DB loss must be LOUD. This is
                    // the bibliography write at the literature-KB boundary; a
                    // locked/full/drifted DB silently under-populates the
                    // bibliography while the run reports success. The store
                    // builds its own degradation_reason (bib_store.rs) — surface
                    // it at `warn`, not `debug`, so partial-corpus loss is
                    // visible in operations. (Envelope-level propagation out of
                    // the engine return type is a structural follow-up — flagged
                    // to Muninn/Skuld, not bundled into this error-path guard.)
                    tracing::warn!(
                        error = %e,
                        run_id = %self.run_id,
                        stage = %self.stage_label,
                        step,
                        "bib_store.record_sources failed — bibliography fusion degraded (non-fatal); continuing"
                    );
                }
            }

            // EXT-02: config-gated plateau check (OFF by default; plateau_threshold=None).
            // Source: clawd's ≥4.6 OR Δ<0.15 rule — NOT consensus's 0.01 delta (Pitfall 7).
            if let Some(threshold) = self.config.plateau_threshold {
                let weights = self.eval_fitness.as_ref()
                    .map(|ef| ef.weights())
                    .unwrap_or(GRAPH_WEIGHTS);
                let score = weighted_sum(&fitness.score_pairs(), weights);
                if score >= threshold {
                    tracing::debug!(step, score, threshold, "plateau threshold reached — early stop");
                    break;
                }
            }

            // Update trajectory with refined draft.
            state.trajectories[0] = refined;

            // ── Guard 3: Budget after step (runner.py:684-696) ────────────
            // Re-checked after each step's gap_identify + gap_resolve calls.
            if budget_after_step_guard(state.llm_calls, self.config.max_llm_calls)
                == GuardOutcome::Stop
            {
                break;
            }
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use alzina_search::bib_store::{BibEntry, BibliographyStore, NoopBibliographyStore};

    use crate::adapter::ExpertResponse;
    use crate::executor::AgentExecutor;
    use crate::ttd::fitness::FitnessEval;
    use crate::ttd::mod_types::TtdError;
    use crate::ttd::retrieval::{NoopRetriever, Retriever};
    use crate::ttd::stages::{DraftGen, EvalFitness, GapIdentify, GapResolve, Merger, RetrievedContext};
    use crate::ttd::state::IdentifiedGap;
    use crate::ttd::weights::GRAPH_WEIGHTS;
    use crate::ttd::{TtdConfig, TtdMachine};

    // ── Mock helpers ──────────────────────────────────────────────────────────

    /// Simple mock executor — records how many times execute() is called.
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

    /// Counting DraftGen — increments counter on each generate() call.
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

    /// Recording DraftGen — records both call count and the persona_prompt received.
    struct RecordingDraftGen {
        count: Arc<AtomicUsize>,
        personas: Arc<std::sync::Mutex<Vec<Option<String>>>>,
    }

    #[async_trait]
    impl DraftGen<String> for RecordingDraftGen {
        async fn generate(
            &self,
            _inputs: &[ExpertResponse],
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
            persona_prompt: Option<&str>,
            _sampling: Option<crate::executor::SamplingParams>,
        ) -> Result<String, TtdError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            let mut log = self.personas.lock().unwrap();
            log.push(persona_prompt.map(String::from));
            Ok("draft".to_string())
        }
    }

    /// Counting GapIdentify — records identify() calls.
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
            // Return one gap so gap_resolve gets called (retrieved will be empty
            // via NoopRetriever — the empty-retrieved guard test exercises the
            // empty path).
            Ok(vec![IdentifiedGap {
                description: "test gap".into(),
                query: "test query".into(),
            }])
        }
    }

    /// GapResolve that panics if called (used for empty-retrieved guard test).
    struct PanicOnResolve;

    #[async_trait]
    impl GapResolve<String> for PanicOnResolve {
        async fn resolve(
            &self,
            _draft: &String,
            _fitness: &FitnessEval,
            _gaps: &[IdentifiedGap],
            _retrieved: &[RetrievedContext],
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
        ) -> Result<String, TtdError> {
            panic!("gap_resolve must NOT be called when retrieved is empty (empty-retrieved guard)")
        }
    }

    /// No-op GapResolve — returns the draft unchanged.
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

    /// Counting GapResolve — records resolve() calls.
    struct CountingResolve {
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl GapResolve<String> for CountingResolve {
        async fn resolve(
            &self,
            draft: &String,
            _fitness: &FitnessEval,
            _gaps: &[IdentifiedGap],
            _retrieved: &[RetrievedContext],
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
        ) -> Result<String, TtdError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(draft.clone())
        }
    }

    /// Counting EvalFitness — records evaluate() calls.
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
            // Return a valid graph eval (groundedness=4).
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

    /// No-op Merger — returns the first candidate unchanged.
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

    fn make_executor(count: Arc<AtomicUsize>) -> Arc<dyn AgentExecutor> {
        Arc::new(CountingExecutor { count })
    }

    fn make_inputs() -> Vec<ExpertResponse> {
        vec![] // Tests don't need real expert responses
    }

    // ── ENGINE-02: n_initial_drafts_is_five ───────────────────────────────────

    /// run() with N=5 config spawns exactly 5 draft_gen.generate() calls.
    #[tokio::test]
    async fn n_initial_drafts_is_five() {
        let draft_count = Arc::new(AtomicUsize::new(0));
        let draft_gen = CountingDraftGen { count: draft_count.clone() };

        let exec_count = Arc::new(AtomicUsize::new(0));

        let machine = TtdMachine {
            config: TtdConfig::default(), // n_initial_drafts=5, n_denoise_steps=2
            draft_gen: Arc::new(draft_gen),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(exec_count.clone()),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        assert_eq!(
            draft_count.load(Ordering::SeqCst),
            5,
            "run() must call draft_gen.generate exactly N=5 times (ENGINE-02)"
        );
    }

    // ── ENGINE-02: denoise_steps_is_two ──────────────────────────────────────

    /// Each trajectory's loop body executes exactly S=2 times.
    /// With N=5 trajectories, gap_identify is called 5 * 2 = 10 times total.
    #[tokio::test]
    async fn denoise_steps_is_two() {
        let gap_count = Arc::new(AtomicUsize::new(0));

        let machine = TtdMachine {
            config: TtdConfig::default(), // n_denoise_steps=2
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: gap_count.clone() }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        // N=5 trajectories × S=2 steps = 10 gap_identify calls.
        assert_eq!(
            gap_count.load(Ordering::SeqCst),
            10,
            "denoise loop must run exactly S=2 steps per trajectory × N=5 trajectories = 10 \
             gap_identify calls (ENGINE-02)"
        );
    }

    // ── Final re-eval runs after loop (Pitfall 4 guard) ──────────────────────

    /// eval_fitness is called N×S times during the loop PLUS N more times for the
    /// final re-evaluation. With N=5, S=2: loop calls = 5×2 = 10; final re-eval = 5.
    /// Total = 15.
    #[tokio::test]
    async fn final_re_eval_runs_after_loop() {
        let eval_count = Arc::new(AtomicUsize::new(0));

        let machine = TtdMachine {
            config: TtdConfig::default(), // N=5, S=2
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: eval_count.clone() })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        // Loop: N=5 × S=2 = 10 calls.
        // Final re-eval: N=5 calls.
        // Total: 15 calls.
        assert_eq!(
            eval_count.load(Ordering::SeqCst),
            15,
            "eval_fitness must be called N×S=10 times in the loop PLUS N=5 times for the \
             final re-evaluation (total 15) — Pitfall 4 guard (runner.py:391-410)"
        );
    }

    // ── Empty-retrieved guard (runner.py:1105-1107) ───────────────────────────

    /// When the Retriever returns empty for all gaps, gap_resolve must NOT be
    /// invoked and the draft is returned unchanged.
    ///
    /// This test uses `PanicOnResolve` which panics if gap_resolve is ever called.
    /// The test passes only if gap_resolve is never called (empty-retrieved guard fires).
    #[tokio::test]
    async fn empty_retrieved_returns_draft_unchanged() {
        // N=1 to keep the test simple; S=1 to test one denoise step.
        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1;
        config.n_denoise_steps = 1;

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(PanicOnResolve), // must never be called
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever), // always returns empty
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        // Must not panic (gap_resolve is not called) and must return a result.
        let result = machine.run(&make_inputs()).await;
        assert!(
            result.is_ok(),
            "empty-retrieved guard must return the draft unchanged without panicking: {result:?}"
        );
    }

    /// Phase P (C-N2): with `resolve_without_retrieval` set and gaps present,
    /// the empty-retrieved guard is bypassed — gap_resolve runs despite
    /// structurally-empty retrieval (Stage 3's NoopRetriever). This re-opens the
    /// consensus critique→refine loop. The mirror of the guard test above:
    /// flag off → resolve never runs (PanicOnResolve); flag on → resolve runs once.
    #[tokio::test]
    async fn resolve_without_retrieval_runs_resolve_despite_empty() {
        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1;
        config.n_denoise_steps = 1;
        config.resolve_without_retrieval = true; // Phase P fork

        let resolve_count = Arc::new(AtomicUsize::new(0));
        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(CountingResolve { count: resolve_count.clone() }),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever), // always returns empty
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        let result = machine.run(&make_inputs()).await;
        assert!(result.is_ok(), "Phase P resolve path must succeed: {result:?}");
        assert_eq!(
            resolve_count.load(Ordering::SeqCst),
            1,
            "resolve_without_retrieval must run gap_resolve once despite empty retrieval"
        );
    }

    // ── Stage-2 end-to-end test ───────────────────────────────────────────────

    /// Stage-2 e2e: Vec<ExpertResponse> + ArgumentationGraph → Synthesis through
    /// TtdMachine<SynthesisArtifact>::run + post_process_synthesis.
    ///
    /// Uses mock executor that returns a canned synthesis XML response.
    /// Asserts output is a well-formed SynthesisArtifact with normalised source_ids
    /// and computed agreement_levels.
    #[tokio::test]
    async fn stage2_e2e_produces_synthesis() {
        use crate::adapter::{ExpertResponse, ResponseProvenance, SourceId};
        use crate::ttd::artifact::{ArgumentationGraph, GraphNode};
        use crate::ttd::stages::synthesis::{
            SynthesisDraftGen, SynthesisGapIdentify, SynthesisGapResolve,
            SynthesisMerger, SynthesisEvalFitness,
        };
        use crate::ttd::post_process::post_process_synthesis;
        use crate::ttd::term_sheet::PromptProfile;

        // Build a minimal ArgumentationGraph (Stage-1 output)
        let mut graph = ArgumentationGraph::new(
            "study-1", "round-1", "q-1", "google/gemini-2.5-flash", "v1/graph",
        );
        graph.nodes.push(GraphNode {
            id: "arxiv:2105.14103_c001".into(),
            claim: "Permafrost thaw releases methane at scale.".into(),
            expert_id: "arxiv:2105.14103".into(),
            quote: Some("permafrost thaw releases significant methane".into()),
            verification_status: Some("verified".into()),
        });

        // Canned synthesis XML — the mock executor returns this for all spawns
        let synthesis_xml = r#"<synthesis>
  <narrative>Permafrost thaw is accelerating under warming, with significant methane release implications.</narrative>
  <claims>
    <claim id="C1">
      <text>Permafrost thaw accelerates methane release.</text>
      <agreement_level>consensus</agreement_level>
      <sources>
        <source id="arxiv:2105.14103_c001"/>
      </sources>
      <counterarguments>
        <counterargument>Rate of release is uncertain.</counterargument>
      </counterarguments>
    </claim>
  </claims>
  <areas_of_agreement>
    <area>Permafrost thaw is accelerating</area>
  </areas_of_agreement>
  <areas_of_disagreement>
    <area>Rate of methane release is disputed</area>
  </areas_of_disagreement>
  <uncertainties>
    <uncertainty>Long-term feedback loops unclear</uncertainty>
  </uncertainties>
</synthesis>"#;

        // Mock executor always returns the synthesis XML
        struct SynthesisXmlExecutor { response: String }

        #[async_trait::async_trait]
        impl AgentExecutor for SynthesisXmlExecutor {
            async fn execute(
                &self,
                _agent_id: &alzina_core::identity::AgentId,
                _instruction: &str,
                _model: &str,
                _task: &str,
            ) -> alzina_core::AlzinaResult<String> {
                Ok(self.response.clone())
            }
        }

        let executor: Arc<dyn AgentExecutor> = Arc::new(SynthesisXmlExecutor {
            response: synthesis_xml.to_string(),
        });

        // Expert responses (1 paper)
        let panel = vec![ExpertResponse {
            expert_id: SourceId::new("arxiv:2105.14103"),
            prose: "Permafrost thaw releases significant methane under warming.".into(),
            provenance: ResponseProvenance {
                source_id: SourceId::new("arxiv:2105.14103"),
                title: "Permafrost Thaw Study".into(),
                year: Some(2021),
                authors: vec![],
                credibility_tier: alzina_search::CredibilityTier::Unknown,
            },
        }];

        // Build TtdMachine<SynthesisArtifact>
        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1; // N=1 for fast e2e
        config.n_denoise_steps = 1; // S=1 for fast e2e

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(SynthesisDraftGen::new(
                "synth-agent",
                "google/gemini-2.5-flash",
                "v1/synthesis",
                Some(graph),
            )),
            gap_identify: Box::new(SynthesisGapIdentify::new(
                "gap-agent",
                "google/gemini-2.5-flash",
            )),
            gap_resolve: Box::new(SynthesisGapResolve::new(
                "resolve-agent",
                "google/gemini-2.5-flash",
            )),
            eval_fitness: Some(Box::new(SynthesisEvalFitness::new(
                "fitness-agent",
                "google/gemini-2.5-flash",
            ))),
            merger: Box::new(SynthesisMerger::new(
                "merger-agent",
                "google/gemini-2.5-flash",
                "v1/synthesis",
            )),
            retriever: Box::new(NoopRetriever), // no retrieval in e2e stub
            executor: executor.clone(),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "synthesis".to_string(),
        };

        // Run Stage 2 TTD loop
        let merged = machine.run(&panel).await.expect("Stage-2 TtdMachine::run must succeed");

        // Run post-processing
        let synthesis = post_process_synthesis(merged, &panel, &executor, PromptProfile::V1Delphi, None)
            .await
            .expect("post_process_synthesis must succeed");

        // Assertions:

        // 1. Synthesis is well-formed (has claims)
        assert!(
            !synthesis.claims.is_empty(),
            "Stage-2 e2e must produce a synthesis with at least one claim"
        );

        // 2. Source IDs are present AND normalised (no compound _c suffix).
        // The non-empty assertion makes this non-vacuous: probe 10 shipped
        // sources: [] and this loop passed because it never iterated.
        for claim in &synthesis.claims {
            assert!(
                !claim.sources.is_empty(),
                "claim '{}' must carry at least one source (provenance conserved)",
                claim.text
            );
            for src in &claim.sources {
                assert!(
                    !src.contains("_c"),
                    "source_id {src} must not contain '_c' suffix after normalisation (Pitfall 5)"
                );
            }
        }

        // 3. Agreement levels are computed deterministically
        for claim in &synthesis.claims {
            assert!(
                claim.agreement_level.is_some(),
                "agreement_level must be set after post-processing"
            );
            assert!(
                matches!(
                    claim.agreement_level.as_deref(),
                    Some("consensus") | Some("majority") | Some("divided") | Some("minority")
                ),
                "agreement_level must be one of the four canonical values: {:?}",
                claim.agreement_level
            );
        }

        // 4. Synthesis has a model and prompt_version (provenance)
        assert_eq!(synthesis.prompt_version, "v1/synthesis");
        assert!(!synthesis.model.is_empty(), "model must be set");
    }

    // ── EXT-01: concurrent_drafts_all_spawned ────────────────────────────────

    /// run() with N=5 spawns all 5 draft_gen.generate calls via JoinSet
    /// (concurrent FanOut, Phase 24 EXT-01). Same count guarantee as the
    /// Phase 23 sequential test, now concurrent.
    #[tokio::test]
    async fn concurrent_drafts_all_spawned() {
        let draft_count = Arc::new(AtomicUsize::new(0));
        let personas_log = Arc::new(std::sync::Mutex::new(Vec::new()));

        let machine = TtdMachine {
            config: TtdConfig::default(), // n_initial_drafts=5
            draft_gen: Arc::new(RecordingDraftGen {
                count: draft_count.clone(),
                personas: personas_log.clone(),
            }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        assert_eq!(
            draft_count.load(Ordering::SeqCst),
            5,
            "concurrent JoinSet FanOut must spawn exactly N=5 generate() calls (EXT-01)"
        );
    }

    // ── EXT-01: persona_spawn_seeds_draft_prompts ─────────────────────────────

    /// When persona prompts are available (PERSONAS constant), draft generation
    /// receives one persona per trajectory (trajectory i gets PERSONAS[i % len]).
    /// Asserts the per-trajectory persona reached DraftGen (RecordingDraftGen
    /// records the persona_prompt it saw).
    #[tokio::test]
    async fn persona_spawn_seeds_draft_prompts() {
        use crate::ttd::personas::PERSONAS;

        let draft_count = Arc::new(AtomicUsize::new(0));
        let personas_log = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 3; // use N=3 for simplicity

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(RecordingDraftGen {
                count: draft_count.clone(),
                personas: personas_log.clone(),
            }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
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

        let log = personas_log.lock().unwrap();

        // All 3 generate() calls must have received a non-None persona prompt.
        assert_eq!(log.len(), 3, "must have 3 generate() calls for N=3");

        // Each persona must be Some and must be a value from PERSONAS.
        // NOTE: JoinSet does not preserve order, so we check set membership,
        // not index correspondence.
        let valid_personas: std::collections::HashSet<&str> =
            PERSONAS.iter().copied().collect();

        for (i, persona) in log.iter().enumerate() {
            assert!(
                persona.is_some(),
                "trajectory {i}: persona_prompt must be Some (PERSONAS fallback active)"
            );
            let p = persona.as_deref().unwrap();
            assert!(
                valid_personas.contains(p),
                "trajectory {i}: persona_prompt must be one of the PERSONAS constants, \
                 got first 60 chars: {}",
                &p[..p.len().min(60)]
            );
        }
    }

    // ── EXT-02: plateau convergence guard (config-gated, OFF by default) ───────

    /// Mock Retriever returning one fixed source per gap — drives the non-empty
    /// retrieval path so the EXT-03 bibliography write fires (NoopRetriever
    /// returns empty and the write is skipped).
    struct OneSourceRetriever;

    #[async_trait]
    impl Retriever for OneSourceRetriever {
        async fn retrieve(
            &self,
            _query: &str,
            _top_k: usize,
        ) -> Result<Vec<RetrievedContext>, TtdError> {
            Ok(vec![RetrievedContext {
                source_id: "arxiv:2105".to_string(),
                content: "retrieved chunk content".to_string(),
                section: None,
            }])
        }
    }

    /// Recording BibliographyStore — counts record_sources calls and captures
    /// the source_ids plus the (run_id, stage) metadata it received, for the
    /// EXT-03 per-step write and CR-01 run_id-threading assertions.
    struct RecordingBibStore {
        calls: Arc<AtomicUsize>,
        source_ids: Arc<std::sync::Mutex<Vec<String>>>,
        meta: Arc<std::sync::Mutex<Vec<(String, String)>>>,
    }

    impl RecordingBibStore {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
                source_ids: Arc::new(std::sync::Mutex::new(Vec::new())),
                meta: Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl BibliographyStore for RecordingBibStore {
        async fn record_sources(
            &self,
            run_id: &str,
            stage: &str,
            _step: usize,
            sources: &[BibEntry],
        ) -> alzina_core::AlzinaResult<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.meta.lock().unwrap().push((run_id.to_string(), stage.to_string()));
            let mut log = self.source_ids.lock().unwrap();
            for e in sources {
                log.push(e.source_id.clone());
            }
            Ok(())
        }
    }

    /// With TtdConfig::default() (plateau_threshold=None) both denoise steps run —
    /// the plateau branch never fires, so the faithful fixed-cap N=5×S=2 is
    /// preserved (Pitfall 7: plateau must NOT contaminate the reproduction).
    #[tokio::test]
    async fn plateau_disabled_by_default_no_early_stop() {
        let gap_count = Arc::new(AtomicUsize::new(0));

        let machine = TtdMachine {
            config: TtdConfig::default(), // plateau_threshold = None
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: gap_count.clone() }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        // N=5 trajectories × S=2 steps = 10 gap_identify calls — both steps run.
        assert_eq!(
            gap_count.load(Ordering::SeqCst),
            10,
            "plateau OFF by default (threshold=None): both denoise steps run — \
             N=5 × S=2 = 10 gap_identify calls; reproduction preserved (Pitfall 7)"
        );
    }

    /// With plateau_threshold = Some(3.5) and CountingEvalFitness returning all
    /// dims = Some(4) (weighted_sum = 4.0 ≥ 3.5), the denoise loop breaks after
    /// step 0. Each of N=5 trajectories runs exactly one step → 5 gap_identify
    /// calls (gap_identify precedes the plateau check in the loop body).
    #[tokio::test]
    async fn plateau_fires_when_score_above_threshold() {
        let gap_count = Arc::new(AtomicUsize::new(0));

        let mut config = TtdConfig::default();
        config.plateau_threshold = Some(3.5); // 4.0 (all-Some(4) × GRAPH_WEIGHTS) ≥ 3.5

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: gap_count.clone() }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        // Step 0 fires gap_identify then the plateau check breaks the loop;
        // step 1 never runs. N=5 trajectories × 1 step = 5 gap_identify calls.
        assert_eq!(
            gap_count.load(Ordering::SeqCst),
            5,
            "plateau fires at step 0 (score 4.0 ≥ threshold 3.5): each of N=5 \
             trajectories runs exactly 1 step → 5 gap_identify calls (labelled, \
             config-gated enhancement)"
        );
    }

    // ── EXT-03: per-step bibliography externalisation ─────────────────────────

    /// A run with a recording BibliographyStore and a non-empty retriever records
    /// the step's unique sources on every denoise step. N=5 × S=2 = 10 steps, each
    /// with one retrieved source → 10 record_sources calls, all carrying the
    /// retrieved source_id (provenance conserved through the bib seam).
    #[tokio::test]
    async fn bib_store_records_per_step() {
        let calls = Arc::new(AtomicUsize::new(0));
        let source_ids = Arc::new(std::sync::Mutex::new(Vec::new()));

        let machine = TtdMachine {
            config: TtdConfig::default(), // plateau off → both steps run
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(OneSourceRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(RecordingBibStore {
                calls: calls.clone(),
                source_ids: source_ids.clone(),
                meta: Arc::new(std::sync::Mutex::new(Vec::new())),
            }),
            run_id: "run-test".to_string(),
            stage_label: "graph".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        assert_eq!(
            calls.load(Ordering::SeqCst),
            10,
            "record_sources must fire once per denoise step with non-empty \
             retrieval — N=5 × S=2 = 10 (EXT-03 per-step bibliography write)"
        );

        let ids = source_ids.lock().unwrap();
        assert_eq!(ids.len(), 10, "one source recorded per step × 10 steps");
        assert!(
            ids.iter().all(|s| s == "arxiv:2105"),
            "each recorded BibEntry carries the retrieved source_id (provenance \
             conserved through the bibliography seam)"
        );
    }

    // ── A5 rung 2: per-gap retrieval ordering-independence ────────────────────

    /// Gap-indexed retriever with artificial per-gap delays in INVERTED order:
    /// gap 0 is the slowest, last gap is the fastest. Used by
    /// `retrieval_ordering_independent_of_completion_order` to prove that
    /// join_all (input-order result) produces the same dedup output as the
    /// sequential-loop version.
    ///
    /// Gap assignments (two gaps, gap-0 shared source_id):
    /// - gap query "gap-0" → 2 results: "src:gap0-unique" + "src:shared"
    ///   (delay 4 ms — slower)
    /// - gap query "gap-1" → 1 result: "src:shared" (dup) + "src:gap1-unique"
    ///   (delay 1 ms — faster)
    ///
    /// Sequential dedup keeps FIRST occurrence of "src:shared" from gap-0.
    /// join_all preserves INPUT ORDER so the Vec is [gap-0-results, gap-1-results]
    /// regardless of which future completes first — identical dedup output.
    struct InvertedDelayRetriever;

    #[async_trait]
    impl Retriever for InvertedDelayRetriever {
        async fn retrieve(
            &self,
            query: &str,
            _top_k: usize,
        ) -> Result<Vec<RetrievedContext>, TtdError> {
            if query == "gap-0" {
                // Slowest gap. Returns gap-0-unique + shared.
                tokio::time::sleep(tokio::time::Duration::from_millis(4)).await;
                Ok(vec![
                    RetrievedContext {
                        source_id: "src:gap0-unique".into(),
                        content: "gap-0 unique content".into(),
                        section: None,
                    },
                    RetrievedContext {
                        source_id: "src:shared".into(),
                        content: "shared content from gap-0".into(),
                        section: None,
                    },
                ])
            } else {
                // Fastest gap. Returns shared (dup) + gap-1-unique.
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                Ok(vec![
                    RetrievedContext {
                        source_id: "src:shared".into(),
                        content: "shared content from gap-1".into(),
                        section: None,
                    },
                    RetrievedContext {
                        source_id: "src:gap1-unique".into(),
                        content: "gap-1 unique content".into(),
                        section: None,
                    },
                ])
            }
        }
    }

    /// Rung 2 ordering-independence test (A5-CONCURRENCY).
    ///
    /// Proves that the concurrent per-gap retrieval (join_all over gap futures)
    /// produces a dedup sequence IDENTICAL to the sequential-loop expectation,
    /// even when gaps complete in reverse order.
    ///
    /// With InvertedDelayRetriever:
    /// - gap-0 completes LAST (4ms), gap-1 completes FIRST (1ms).
    /// - join_all returns results in INPUT order (gap-0 first, gap-1 second).
    /// - dedup keeps FIRST occurrence of "src:shared" = gap-0's copy.
    /// - Expected unique sequence: ["src:gap0-unique", "src:shared", "src:gap1-unique"].
    ///
    /// A GapIdentify mock emits exactly two gaps per step with queries
    /// "gap-0" and "gap-1" in that order. N=1, S=1 for minimal run scope.
    /// The gap_resolve receives the unique_retrieved slice; a RecordingResolve
    /// captures what it saw so we can inspect the source_id sequence.
    #[tokio::test]
    async fn retrieval_ordering_independent_of_completion_order() {
        use std::sync::Mutex;

        // ── GapIdentify that emits gap-0 then gap-1 ──────────────────────────
        struct TwoGapIdentify;

        #[async_trait]
        impl GapIdentify<String> for TwoGapIdentify {
            async fn identify(
                &self,
                _draft: &String,
                _fitness: &FitnessEval,
                _executor: &Arc<dyn AgentExecutor>,
                _config: &TtdConfig,
            ) -> Result<Vec<IdentifiedGap>, TtdError> {
                Ok(vec![
                    IdentifiedGap { description: "gap zero".into(), query: "gap-0".into() },
                    IdentifiedGap { description: "gap one".into(),  query: "gap-1".into() },
                ])
            }
        }

        // ── RecordingResolve: captures the source_id sequence it receives ────
        struct RecordingResolve {
            received: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl GapResolve<String> for RecordingResolve {
            async fn resolve(
                &self,
                draft: &String,
                _fitness: &FitnessEval,
                _gaps: &[IdentifiedGap],
                retrieved: &[RetrievedContext],
                _executor: &Arc<dyn AgentExecutor>,
                _config: &TtdConfig,
            ) -> Result<String, TtdError> {
                let mut log = self.received.lock().unwrap();
                for ctx in retrieved {
                    log.push(ctx.source_id.clone());
                }
                Ok(draft.clone())
            }
        }

        let received = Arc::new(Mutex::new(Vec::<String>::new()));

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 1;
        config.n_denoise_steps = 1;

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(TwoGapIdentify),
            gap_resolve: Box::new(RecordingResolve { received: received.clone() }),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(InvertedDelayRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        // Sequential expectation: gap-0 first → gap-1 second.
        // Dedup keeps first occurrence of "src:shared" (= gap-0's copy).
        // Expected unique sequence: gap0-unique, shared, gap1-unique.
        let ids = received.lock().unwrap();
        assert_eq!(
            *ids,
            vec!["src:gap0-unique", "src:shared", "src:gap1-unique"],
            "join_all input-order preservation must produce the same dedup output \
             as the sequential-loop: got {ids:?}"
        );
    }

    /// CR-01 proof: the machine threads its own `run_id` and `stage_label` into
    /// every `record_sources` call. Before the fix the engine builders hardcoded
    /// `run_id = ""`, which collapses the SqliteBibliographyStore's
    /// UNIQUE(run_id, source_id, expert_id, quote_normalised) dedup across runs
    /// (the second run's bibliography is silently lost). The store's per-run
    /// dedup behaviour itself is covered by alzina-search's bib_store tests; here
    /// we prove the identifier actually reaches the write so dedup can be scoped.
    #[tokio::test]
    async fn machine_threads_run_id_and_stage_to_bib_store() {
        let store = Arc::new(RecordingBibStore::new());

        let machine = TtdMachine {
            config: TtdConfig::default(),
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
            merger: Box::new(FirstMerger),
            retriever: Box::new(OneSourceRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::clone(&store) as Arc<dyn BibliographyStore>,
            run_id: "weave-abc123".to_string(),
            stage_label: "graph".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        let meta = store.meta.lock().unwrap();
        assert!(!meta.is_empty(), "record_sources must have been called");
        assert!(
            meta.iter().all(|(run_id, stage)| run_id == "weave-abc123" && stage == "graph"),
            "every bibliography write must carry the machine's non-empty run_id + \
             stage_label (CR-01: a constant-empty run_id collapses cross-run dedup)"
        );
    }

    // ── A5 rung 4: trajectory concurrency + semaphore cap ────────────────────

    /// EvalFitness mock that tracks current / max in-flight evaluate() calls.
    ///
    /// On entry: increment `in_flight`, record `max_in_flight`.
    /// After a small sleep: decrement `in_flight`.
    /// Used to prove that concurrent trajectories overlap (max_in_flight > 1)
    /// and that the semaphore cap is enforced (max_in_flight == 1 with cap=1).
    struct WitnessEvalFitness {
        in_flight: Arc<AtomicUsize>,
        max_in_flight: Arc<AtomicUsize>,
    }

    impl WitnessEvalFitness {
        fn new() -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            let in_flight = Arc::new(AtomicUsize::new(0));
            let max_in_flight = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    in_flight: in_flight.clone(),
                    max_in_flight: max_in_flight.clone(),
                },
                in_flight,
                max_in_flight,
            )
        }
    }

    #[async_trait]
    impl EvalFitness<String> for WitnessEvalFitness {
        async fn evaluate(
            &self,
            _draft: &String,
            _executor: &Arc<dyn AgentExecutor>,
            _config: &TtdConfig,
        ) -> Result<FitnessEval, TtdError> {
            // Increment in-flight, record max.
            let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut max = self.max_in_flight.load(Ordering::SeqCst);
            while current > max {
                match self.max_in_flight.compare_exchange(
                    max,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(actual) => max = actual,
                }
            }
            // Hold the slot briefly to allow concurrency to manifest.
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
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

    /// Rung-4 concurrency witness: with N=3 and default semaphore cap (5),
    /// max in-flight evaluate() calls observed > 1 — trajectories genuinely overlap.
    ///
    /// This test will FAIL on the current sequential outer loop (loop serialises
    /// trajectories, so max_in_flight == 1). It passes once join_all is in place.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn trajectory_fitness_evals_overlap_with_default_cap() {
        let (witness, _in_flight, max_in_flight) = WitnessEvalFitness::new();

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 3;
        config.n_denoise_steps = 1;

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(witness)),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        let observed_max = max_in_flight.load(Ordering::SeqCst);
        assert!(
            observed_max > 1,
            "trajectories must execute fitness evals concurrently (observed max in-flight: {}; \
             expected > 1 — trajectories overlap with default semaphore cap)",
            observed_max
        );
    }

    /// Rung-4 cap enforced: with N=3 and max_concurrent_fitness_evals=1,
    /// max in-flight is exactly 1 even under concurrent trajectories.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn trajectory_fitness_eval_cap_enforced() {
        let (witness, _in_flight, max_in_flight) = WitnessEvalFitness::new();

        let mut config = TtdConfig::default();
        config.n_initial_drafts = 3;
        config.n_denoise_steps = 1;
        config.max_concurrent_fitness_evals = 1; // cap at 1

        let machine = TtdMachine {
            config,
            draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
            gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
            gap_resolve: Box::new(NoopResolve),
            eval_fitness: Some(Box::new(witness)),
            merger: Box::new(FirstMerger),
            retriever: Box::new(NoopRetriever),
            executor: make_executor(Arc::new(AtomicUsize::new(0))),
            bib_store: Arc::new(NoopBibliographyStore),
            run_id: String::new(),
            stage_label: "test".to_string(),
        };

        machine.run(&make_inputs()).await.unwrap();

        let observed_max = max_in_flight.load(Ordering::SeqCst);
        assert_eq!(
            observed_max, 1,
            "semaphore cap=1 must limit concurrent fitness evals to exactly 1 \
             (observed max in-flight: {})",
            observed_max
        );
    }

    // ── Spawn-count / budget-accounting tests ─────────────────────────────────

    /// V1 fitness impl reports 6 dims; V2 impl reports 5 dims.
    ///
    /// fitness_call_count is derived from `weights().len()`, so these two impls
    /// exercise the two branches of the budget-accounting dynamic derivation
    /// in run.rs.
    #[test]
    fn fitness_call_count_v1_is_6_v2_is_5() {
        use crate::ttd::weights::{GRAPH_WEIGHTS, V2_GRAPH_WEIGHTS};

        // V1 mock — 6 weights (same as CountingEvalFitness)
        struct V1Mock;
        #[async_trait]
        impl EvalFitness<String> for V1Mock {
            async fn evaluate(
                &self,
                _draft: &String,
                _executor: &Arc<dyn AgentExecutor>,
                _config: &TtdConfig,
            ) -> Result<FitnessEval, TtdError> {
                Ok(FitnessEval::new(vec![]))
            }
            fn validity_fn(&self) -> fn(&FitnessEval) -> bool {
                crate::ttd::fitness::is_valid_graph
            }
            fn weights(&self) -> &'static [(&'static str, f32)] {
                GRAPH_WEIGHTS
            }
        }

        // V2 mock — 5 weights
        struct V2Mock;
        #[async_trait]
        impl EvalFitness<String> for V2Mock {
            async fn evaluate(
                &self,
                _draft: &String,
                _executor: &Arc<dyn AgentExecutor>,
                _config: &TtdConfig,
            ) -> Result<FitnessEval, TtdError> {
                Ok(FitnessEval::new(vec![]))
            }
            fn validity_fn(&self) -> fn(&FitnessEval) -> bool {
                crate::ttd::fitness::is_valid_v2
            }
            fn weights(&self) -> &'static [(&'static str, f32)] {
                V2_GRAPH_WEIGHTS
            }
        }

        let v1: Box<dyn EvalFitness<String>> = Box::new(V1Mock);
        let v2: Box<dyn EvalFitness<String>> = Box::new(V2Mock);

        assert_eq!(
            v1.weights().len(),
            6,
            "V1 impl must report 6 dims for budget accounting"
        );
        assert_eq!(
            v2.weights().len(),
            5,
            "V2 impl must report 5 dims for budget accounting"
        );
    }

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║ SEAM F2 — IN-LOOP ERROR PROPAGATION  (characterisation net, W-522022c5) ║
    // ║ Pins "run-all-then-fold-first-Err" (NOT fail-fast) for the per-gap       ║
    // ║ retrieval fold + the resolve-Err `?` propagation through the trajectory.  ║
    // ╚═══════════════════════════════════════════════════════════════════════╝
    mod f2_in_loop_error_propagation {
        use super::*;

        /// A retriever that returns Err ONLY for a specific gap query, Ok otherwise.
        /// Proves the fold propagates the gap-index-FIRST error and that the later
        /// gap's retrieval still ran (no fail-fast cancellation).
        struct SelectiveFailRetriever {
            fail_query: &'static str,
            seen: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        }

        #[async_trait::async_trait]
        impl crate::ttd::retrieval::Retriever for SelectiveFailRetriever {
            async fn retrieve(
                &self,
                query: &str,
                _top_k: usize,
            ) -> Result<Vec<crate::ttd::stages::RetrievedContext>, crate::ttd::mod_types::TtdError> {
                self.seen.lock().unwrap().push(query.to_string());
                if query == self.fail_query {
                    Err(crate::ttd::mod_types::TtdError::SpawnFailed(
                        format!("retrieval failed for {query}"),
                    ))
                } else {
                    Ok(vec![crate::ttd::stages::RetrievedContext {
                        source_id: format!("src:{query}"),
                        content: "ok".into(),
                        section: None,
                    }])
                }
            }
        }

        /// GapIdentify that always emits two gaps with fixed queries gap-0, gap-1.
        struct TwoFixedGaps;
        #[async_trait::async_trait]
        impl crate::ttd::stages::GapIdentify<String> for TwoFixedGaps {
            async fn identify(
                &self,
                _draft: &String,
                _fitness: &crate::ttd::fitness::FitnessEval,
                _executor: &std::sync::Arc<dyn crate::executor::AgentExecutor>,
                _config: &crate::ttd::TtdConfig,
            ) -> Result<Vec<crate::ttd::state::IdentifiedGap>, crate::ttd::mod_types::TtdError> {
                Ok(vec![
                    crate::ttd::state::IdentifiedGap { description: "g0".into(), query: "gap-0".into() },
                    crate::ttd::state::IdentifiedGap { description: "g1".into(), query: "gap-1".into() },
                ])
            }
        }

        /// PINS: per-gap retrieval fold propagates the FIRST Err by gap-index, AND
        /// both gap retrievals executed (run-all, not fail-fast).
        #[tokio::test]
        async fn f2_retrieval_runs_all_gaps_then_propagates_first_err() {
            use std::sync::{Arc, atomic::AtomicUsize};

            let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut config = TtdConfig::default();
            config.n_initial_drafts = 1;
            config.n_denoise_steps = 1;

            let machine = TtdMachine {
                config,
                draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
                gap_identify: Box::new(TwoFixedGaps),
                gap_resolve: Box::new(NoopResolve),
                eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
                merger: Box::new(FirstMerger),
                retriever: Box::new(SelectiveFailRetriever { fail_query: "gap-0", seen: seen.clone() }),
                executor: make_executor(Arc::new(AtomicUsize::new(0))),
                bib_store: Arc::new(alzina_search::bib_store::NoopBibliographyStore),
                run_id: String::new(),
                stage_label: "test".to_string(),
            };

            let result = machine.run(&make_inputs()).await;

            assert!(result.is_err(), "F2: a gap retrieval Err must propagate out of run()");
            match result {
                Err(crate::ttd::mod_types::TtdError::SpawnFailed(msg)) => {
                    assert!(
                        msg.contains("gap-0"),
                        "F2: FIRST gap-index Err must win the fold (got: {msg})"
                    );
                }
                other => panic!("F2: expected SpawnFailed(gap-0), got {other:?}"),
            }

            // Run-all (not fail-fast): the sibling gap-1 retrieval also executed.
            let queried = seen.lock().unwrap();
            assert!(
                queried.iter().any(|q| q == "gap-1"),
                "F2: sibling gap-1 retrieval must have run (join_all is run-all, not fail-fast): saw {queried:?}"
            );
        }

        /// PINS: a gap_resolve Err inside the trajectory loop aborts via `?` and
        /// surfaces as run() Err (the trajectory fold re-raises it).
        #[tokio::test]
        async fn f2_resolve_err_aborts_trajectory_and_propagates() {
            use std::sync::{Arc, atomic::AtomicUsize};

            struct OneSourceRetriever;
            #[async_trait::async_trait]
            impl crate::ttd::retrieval::Retriever for OneSourceRetriever {
                async fn retrieve(&self, q: &str, _k: usize)
                    -> Result<Vec<crate::ttd::stages::RetrievedContext>, crate::ttd::mod_types::TtdError> {
                    Ok(vec![crate::ttd::stages::RetrievedContext {
                        source_id: format!("src:{q}"), content: "c".into(), section: None,
                    }])
                }
            }
            struct ErrResolve;
            #[async_trait::async_trait]
            impl crate::ttd::stages::GapResolve<String> for ErrResolve {
                async fn resolve(
                    &self,
                    _d: &String,
                    _f: &crate::ttd::fitness::FitnessEval,
                    _g: &[crate::ttd::state::IdentifiedGap],
                    _r: &[crate::ttd::stages::RetrievedContext],
                    _e: &Arc<dyn crate::executor::AgentExecutor>,
                    _c: &TtdConfig,
                ) -> Result<String, crate::ttd::mod_types::TtdError> {
                    Err(crate::ttd::mod_types::TtdError::ParseFailed("resolve boom".into()))
                }
            }

            let mut config = TtdConfig::default();
            config.n_initial_drafts = 1;
            config.n_denoise_steps = 1;

            let machine = TtdMachine {
                config,
                draft_gen: Arc::new(CountingDraftGen { count: Arc::new(AtomicUsize::new(0)) }),
                gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
                gap_resolve: Box::new(ErrResolve),
                eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
                merger: Box::new(FirstMerger),
                retriever: Box::new(OneSourceRetriever),
                executor: make_executor(Arc::new(AtomicUsize::new(0))),
                bib_store: Arc::new(alzina_search::bib_store::NoopBibliographyStore),
                run_id: String::new(),
                stage_label: "test".to_string(),
            };

            let result = machine.run(&make_inputs()).await;
            match result {
                Err(crate::ttd::mod_types::TtdError::ParseFailed(msg)) => {
                    assert_eq!(msg, "resolve boom", "F2: resolve Err must surface verbatim through the trajectory fold");
                }
                other => panic!("F2: expected ParseFailed(resolve boom), got {other:?}"),
            }
        }
    }

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║ SEAM F3 — CONCURRENT FAN-OUT PANIC / FOLD  (characterisation net)       ║
    // ║ Pins panic→SpawnFailed (WR-02, NOT NoCandidates/abort), Err→propagate,    ║
    // ║ N=0→NoCandidates, and the WEAK mixed-panic invariant.                    ║
    // ║ REQUIRES panic=unwind (workspace default; no panic=abort profile set).   ║
    // ╚═══════════════════════════════════════════════════════════════════════╝
    mod f3_fanout_panic_fold {
        use super::*;

        /// DraftGen whose generate() PANICS. Pins the JoinError → SpawnFailed mapping.
        struct PanicDraftGen;
        #[async_trait::async_trait]
        impl crate::ttd::stages::DraftGen<String> for PanicDraftGen {
            async fn generate(
                &self,
                _i: &[crate::adapter::ExpertResponse],
                _e: &std::sync::Arc<dyn crate::executor::AgentExecutor>,
                _c: &TtdConfig,
                _p: Option<&str>,
                _s: Option<crate::executor::SamplingParams>,
            ) -> Result<String, crate::ttd::mod_types::TtdError> {
                panic!("draft gen exploded");
            }
        }

        /// DraftGen that returns Err (task-level, not a panic). Pins Ok(Err(_)) arm.
        struct ErrDraftGen;
        #[async_trait::async_trait]
        impl crate::ttd::stages::DraftGen<String> for ErrDraftGen {
            async fn generate(
                &self,
                _i: &[crate::adapter::ExpertResponse],
                _e: &std::sync::Arc<dyn crate::executor::AgentExecutor>,
                _c: &TtdConfig,
                _p: Option<&str>,
                _s: Option<crate::executor::SamplingParams>,
            ) -> Result<String, crate::ttd::mod_types::TtdError> {
                Err(crate::ttd::mod_types::TtdError::SpawnFailed("draft gen returned Err".into()))
            }
        }

        fn machine_with_draft_gen(
            draft_gen: std::sync::Arc<dyn crate::ttd::stages::DraftGen<String>>,
            n: usize,
        ) -> TtdMachine<String> {
            use std::sync::{Arc, atomic::AtomicUsize};
            let mut config = TtdConfig::default();
            config.n_initial_drafts = n;
            config.n_denoise_steps = 1;
            TtdMachine {
                config,
                draft_gen,
                gap_identify: Box::new(CountingGapIdentify { count: Arc::new(AtomicUsize::new(0)) }),
                gap_resolve: Box::new(NoopResolve),
                eval_fitness: Some(Box::new(CountingEvalFitness { count: Arc::new(AtomicUsize::new(0)) })),
                merger: Box::new(FirstMerger),
                retriever: Box::new(NoopRetriever),
                executor: make_executor(Arc::new(AtomicUsize::new(0))),
                bib_store: Arc::new(alzina_search::bib_store::NoopBibliographyStore),
                run_id: String::new(),
                stage_label: "test".to_string(),
            }
        }

        /// PINS: a panicking draft task → TtdError::SpawnFailed("draft task
        /// panicked: ..."), NOT NoCandidates, NOT a process abort (WR-02).
        /// (Requires panic=unwind; see module caveat.)
        #[tokio::test]
        async fn f3_panicking_draft_task_maps_to_spawnfailed() {
            let machine = machine_with_draft_gen(std::sync::Arc::new(PanicDraftGen), 1);
            let result = machine.run(&make_inputs()).await;
            match result {
                Err(crate::ttd::mod_types::TtdError::SpawnFailed(msg)) => {
                    assert!(
                        msg.contains("draft task panicked"),
                        "F3: panic must map to SpawnFailed('draft task panicked: ...'), got: {msg}"
                    );
                }
                other => panic!(
                    "F3: panicking draft must NOT flatten to NoCandidates; expected SpawnFailed, got {other:?}"
                ),
            }
        }

        /// PINS: a task-level Err (not panic) propagates as that Err out of run().
        #[tokio::test]
        async fn f3_erroring_draft_task_propagates_err() {
            let machine = machine_with_draft_gen(std::sync::Arc::new(ErrDraftGen), 3);
            let result = machine.run(&make_inputs()).await;
            match result {
                Err(crate::ttd::mod_types::TtdError::SpawnFailed(msg)) => {
                    assert_eq!(
                        msg, "draft gen returned Err",
                        "F3: task-level Err must propagate verbatim (distinct from panic path)"
                    );
                }
                other => panic!("F3: expected SpawnFailed(draft gen returned Err), got {other:?}"),
            }
        }

        /// PINS A KNOWN GAP (report, do not fix): JoinSet drain order is
        /// NONDETERMINISTIC, so WHICH error wins is not guaranteed. Pin only the
        /// WEAK invariant: a panic anywhere must surface as Err, never a silent
        /// partial success.
        #[tokio::test]
        async fn f3_mixed_panic_and_success_is_never_silently_ok() {
            use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

            struct FirstPanicsRestOk { calls: Arc<AtomicUsize> }
            #[async_trait::async_trait]
            impl crate::ttd::stages::DraftGen<String> for FirstPanicsRestOk {
                async fn generate(
                    &self,
                    _i: &[crate::adapter::ExpertResponse],
                    _e: &Arc<dyn crate::executor::AgentExecutor>,
                    _c: &TtdConfig,
                    _p: Option<&str>,
                    _s: Option<crate::executor::SamplingParams>,
                ) -> Result<String, crate::ttd::mod_types::TtdError> {
                    let idx = self.calls.fetch_add(1, Ordering::SeqCst);
                    if idx == 0 { panic!("first draft exploded"); }
                    Ok("draft".to_string())
                }
            }

            let machine = machine_with_draft_gen(
                Arc::new(FirstPanicsRestOk { calls: Arc::new(AtomicUsize::new(0)) }),
                5,
            );
            let result = machine.run(&make_inputs()).await;
            assert!(
                result.is_err(),
                "F3: a panic in ANY fan-out task must surface as Err, never a silent partial success"
            );
        }

        /// PINS: zero drafts (N=0) → NoCandidates. Boundary case at the empty edge.
        #[tokio::test]
        async fn f3_zero_drafts_is_nocandidates() {
            let machine = machine_with_draft_gen(
                std::sync::Arc::new(CountingDraftGen { count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)) }),
                0,
            );
            let result = machine.run(&make_inputs()).await;
            assert!(
                matches!(result, Err(crate::ttd::mod_types::TtdError::NoCandidates)),
                "F3: zero-draft fan-out must yield NoCandidates, got {result:?}"
            );
        }
    }
}
