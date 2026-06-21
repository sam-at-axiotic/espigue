//! Stage-2 post-processing chain for the TTD synthesis stage.
//!
//! Reproduces `consensus/src/consensus/summarise/strategies.py:446-554`
//! (`_post_process_synthesis`). Runs AFTER the TTD merge, BEFORE artifact emit
//! (Pitfall 5 — not part of the TTD loop).
//!
//! ## 7-step chain
//!
//! 1. **Normalise source IDs** — graph-node compound IDs (e.g. `expert_01_c003`)
//!    → base expert IDs by matching against the known panel IDs longest-first,
//!    THEN dedup (ports strategies.py:106-212; see CR-04). Earlier drafts strip-
//!    split on the literal `_c` substring via `deduplicate_sources`; that
//!    corrupted IDs like `pmc_case_study` and is no longer used here.
//! 2. **Enrich panel_size** — fill `panel_size` from `responses.len()`.
//! 3. **Compute agreement_level** — deterministic from corroboration ratio
//!    (`_compute_agreement_level`, strategies.py:283-295): ≥0.75 consensus /
//!    ≥0.50 majority / ≥0.15 divided / else minority. The corroboration count is
//!    the unique non-system expert count (synthesis_tasks.py:1038-1044). These
//!    hard thresholds OVERRIDE any LLM judgment (T-23-08 mitigated).
//! 4. **Quote resolve** — one governed LLM spawn (dimensional-confinement quote fix).
//! 5. **Quote verify** — deterministic substring check against source text.
//! 6. **Expert coverage gate** — check all experts are cited; optional revision
//!    spawn if any are missing.
//! 7. **Minority reports** — citations + minority reports from uncovered experts.
//!
//! Steps 1-3, 5, 7 are deterministic Rust. Steps 4 and 6 are governed spawns
//! through AgentExecutor (two extra LLM calls per synthesis post-process).
//!
//! ## Trust boundary (T-23-08, T-23-09)
//!
//! - `agreement_level` is always recomputed from the corroboration ratio — a
//!   prompt-injected agreement claim cannot move the label.
//! - Quote verification (step 5) is a lightweight Phase-23 STUB, not a full
//!   trust boundary. Under the current schema (`Claim.sources: Vec<String>`,
//!   paper IDs only) there is no per-source quote/status to verify or write, so
//!   step 5 does not by itself block a hallucinated quote. Faithful per-source
//!   verification (the real T-23-09 mitigation) needs the SourceReference schema
//!   change and is Phase 25 scope. See `verify_synthesis_quotes` (WR-03) and
//!   `apply_resolved_quotes` (WR-07).

use std::collections::HashSet;
use std::sync::Arc;

use crate::adapter::ExpertResponse;
use crate::executor::AgentExecutor;
use crate::ttd::artifact::{MinorityReport, SynthesisArtifact};
use crate::ttd::mod_types::TtdError;
use crate::ttd::term_sheet::{is_valid_source_id, PromptProfile};

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the 7-step post-processing chain on a merged synthesis.
///
/// This function runs AFTER `TtdMachine::run()` returns the merged synthesis
/// and BEFORE the artifact is emitted on the audit trail (Pitfall 5 guard).
///
/// The two LLM spawns (quote_resolve, revision) route through `executor`.
///
/// ## v2 gating (B1)
///
/// Under `V2LitReview`, the vote-taxonomy steps are skipped:
/// - Step 3 (`compute_agreement_levels`) — v2 uses `support_level` (LLM-asserted);
///   vote-threshold stamps are irrelevant and would overwrite the parsed label.
/// - Post-revision recompute at step 6 — same reason.
/// - Step 7b (`compute_minority_reports`) — keyed on `agreement_level == "minority"`;
///   meaningless when that field is not set under v2.
///
/// Steps 1, 4, 5, 6 (coverage + revision), and 7a (enrich_citations) run for both
/// profiles — they are taxonomy-neutral.
pub async fn post_process_synthesis(
    synthesis: SynthesisArtifact,
    panel: &[ExpertResponse],
    executor: &Arc<dyn AgentExecutor>,
    profile: PromptProfile,
    resolver: Option<&Arc<dyn crate::ttd::engine::PanelRefresher>>,
) -> Result<SynthesisArtifact, TtdError> {
    post_process_synthesis_with_graph(synthesis, panel, executor, profile, resolver, None).await
}

/// Full post-process entry point with optional graph for Fix C quote inheritance.
///
/// Under V2LitReview, if `graph` is Some, inherited DB-verified graph quotes are
/// attached to synthesis claims after verification (and re-verified before emit).
/// Pass `None` where no graph is available (e.g. v1 path, tests without graphs).
///
/// Engine wiring (Fix C): engine.rs:434 passes `Some(&graph)` — the graph was
/// DB-re-verified at lines 300-353 before this point.
pub async fn post_process_synthesis_with_graph(
    synthesis: SynthesisArtifact,
    panel: &[ExpertResponse],
    executor: &Arc<dyn AgentExecutor>,
    profile: PromptProfile,
    resolver: Option<&Arc<dyn crate::ttd::engine::PanelRefresher>>,
    graph: Option<&crate::ttd::artifact::ArgumentationGraph>,
) -> Result<SynthesisArtifact, TtdError> {
    let panel_size = panel.len();

    // ── Step 1: Normalise source IDs ────────────────────────────────────────
    // Normalise graph-node compound IDs back to base expert IDs by matching
    // against the known panel IDs (longest-first), THEN dedup (CR-04). A wrong
    // key inflates agreement counts feeding step 3.
    let synthesis = normalise_source_ids(synthesis, panel);
    // F13 (probe-18): strip invalid source ids using the allowlist.
    // Valid = arxiv:/s2:-prefixed (shape lane) or exact panel member.
    // Supersedes Fix B's CandidateN blacklist — probe-18 proved haiku evades
    // pattern-blacklisting by mutating the label (s1_candidate1 .. s5_candidate5).
    // V2LitReview only — v1 byte-identity preserved. Decision 0: v3 = v2.
    let synthesis = match profile {
        PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
            strip_invalid_sources(synthesis, panel)
        }
        PromptProfile::V1Delphi => synthesis,
    };

    // ── Step 2: Enrich panel_size ────────────────────────────────────────────
    // Each claim needs panel_size for the corroboration ratio computation in step 3.
    // (In consensus this is attached to the Claim dataclass; here it's carried
    // implicitly via the panel length passed to _compute_agreement_level.)

    // ── Step 3: Compute agreement_level (deterministic) ──────────────────────
    // V1Delphi: hard thresholds (strategies.py:283-295) OVERRIDE any LLM judgment.
    //   T-23-08 mitigation: a prompt-injected claim cannot change this label.
    // V2LitReview: SKIP — v2 uses support_level (LLM-asserted), not vote thresholds.
    //   support_level is set at parse time; stamping agreement_level here would be
    //   meaningless and would incorrectly overwrite None agreement labels.
    let synthesis = match profile {
        PromptProfile::V1Delphi => compute_agreement_levels(synthesis, panel_size),
        // skip vote taxonomy (v3 = v2, Decision 0)
        PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => synthesis,
    };

    // ── Step 4: Quote resolve (LLM spawn) — v1 only ──────────────────────────
    // V1Delphi: resolves placeholder quotes to verbatim substrings. One
    //   governed dispatch (byte-identical v1 behaviour preserved).
    // V2LitReview: SKIPPED (worklist item 5). The spawn embeds EVERY panel
    //   member's full prose — the last full-text prompt dump — and
    //   apply_resolved_quotes discards its output (WR-07 stub). Under v2,
    //   claims carry their own quotes (item 4) verified deterministically in
    //   step 5; this spawn is pure cost.
    let synthesis = match profile {
        PromptProfile::V1Delphi => {
            tracing::debug!("post_process_synthesis: step 4 — quote_resolve spawn");
            resolve_quotes(synthesis, panel, executor).await?
        }
        PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
            tracing::debug!(
                "post_process_synthesis: step 4 skipped under v2 — claims carry \
                 verified quotes; resolve_quotes output is discarded (WR-07)"
            );
            synthesis
        }
    };

    // ── Step 5: Verify synthesis quotes (DB-backed, worklist item 4) ─────────
    // Each ClaimQuote is substring-checked against its cited source's STORED
    // text — panel prose first, then the DB-backed resolver for sources the
    // model quoted from retrieval context. Closes WR-03.
    let synthesis = verify_synthesis_quotes(synthesis, panel, resolver).await;
    // Fix C (probe-17 cause 2): after step-5 verify, inherit DB-verified graph
    // node quotes onto claims that lack a verified quote for a given source.
    // V2LitReview only. If any quotes were inherited, re-run verify so the
    // inherited text is substring-stamped against stored prose before emit
    // (no transitive trust of the graph's stamp — an unresolvable source stamps
    // absent). verify is idempotent on already-stamped quotes (cannot flip honest
    // statuses).
    // Decision 0: v3 = v2. NOTE: if-let (refutable pattern) — invisible to
    // exhaustiveness checking; or-pattern keeps v3 on the v2 inheritance path.
    let synthesis = if let (
        PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong,
        Some(g),
    ) = (profile, graph)
    {
        // F14 deterministic floor first: attach cited-node quotes by exact id
        // for any node-cited claim the merger left without a verified quote.
        // inherit_graph_quotes then backstops untagged/legacy claims via the
        // ≥0.5 token-containment match (it skips claims that carry node_refs).
        let (synthesis, n_floor) = attach_node_ref_quotes(synthesis, g);
        let (synthesis, n_inherited) = inherit_graph_quotes(synthesis, g);
        if n_floor + n_inherited > 0 {
            verify_synthesis_quotes(synthesis, panel, resolver).await
        } else {
            synthesis
        }
    } else {
        synthesis
    };

    // ── Step 6: Expert coverage gate + optional revision spawn ────────────────
    let (missing_experts, _coverage) = check_expert_coverage(&synthesis, panel);
    tracing::debug!(
        n_missing = missing_experts.len(),
        panel_size,
        "post_process_synthesis: step 6 — coverage gate"
    );

    let synthesis = if !missing_experts.is_empty() {
        tracing::debug!(
            n_missing = missing_experts.len(),
            "post_process_synthesis: triggering revision pass for missing experts"
        );
        let revised = run_revision(synthesis, panel, &missing_experts, executor, profile, graph).await?;
        // The revision output is a fresh parse — its sources were normalised
        // pre-revision but its agreement labels come from the model/parser.
        // V1: Re-run the deterministic steps so hard thresholds override any LLM
        //     judgment in the FINAL artifact too (T-23-08 holds post-revision).
        // V2: Skip vote taxonomy (same reason as step 3).
        let revised = normalise_source_ids(revised, panel);
        // F13: strip invalid source ids from the revised output too (same gate).
        let revised = match profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                strip_invalid_sources(revised, panel)
            }
            PromptProfile::V1Delphi => revised,
        };
        let revised = match profile {
            PromptProfile::V1Delphi => compute_agreement_levels(revised, panel_size),
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => revised,
        };
        // Fix A (probe-17 cause 3): re-run verify_synthesis_quotes on the
        // revised output. The revision is a fresh parse — all quote statuses
        // are None until re-stamped here. taxonomy-neutral; a no-op when
        // quotes are empty (v1 claims carry no quotes), matching the step-5
        // precedent. resolver is already in scope as a function parameter.
        let revised = verify_synthesis_quotes(revised, panel, resolver).await;
        // Fix C (post-revision): inherit graph quotes on the revised path too
        // (same gate as step-5 path above).
        // Decision 0: v3 = v2 (same if-let caveat as the step-5 gate above).
        if let (
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong,
            Some(g),
        ) = (profile, graph)
        {
            // F14 floor first, then token-containment backstop (same as step-5).
            let (revised, n_floor) = attach_node_ref_quotes(revised, g);
            let (revised, n_inherited) = inherit_graph_quotes(revised, g);
            if n_floor + n_inherited > 0 {
                verify_synthesis_quotes(revised, panel, resolver).await
            } else {
                revised
            }
        } else {
            revised
        }
    } else {
        tracing::debug!("post_process_synthesis: coverage ok — skipping revision");
        synthesis
    };

    // ── Step 7: Enrich citations + compute minority reports ───────────────────
    let synthesis = enrich_citations(synthesis);
    // V2LitReview: skip minority_reports — it's keyed on agreement_level == "minority"
    // which is not set under v2. No vote taxonomy, no vote-derived minority reports.
    let synthesis = match profile {
        PromptProfile::V1Delphi => compute_minority_reports(synthesis, panel),
        PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => synthesis,
    };

    Ok(synthesis)
}

// ── Step 1: Normalise source IDs ────────────────────────────────────────────

/// Normalise graph-node compound source IDs back to plain expert IDs, THEN dedup.
///
/// Ports `consensus/src/consensus/summarise/strategies.py:106-212`
/// (`_normalise_source_ids`). TTD graph extraction namespaces node IDs as
/// `{expert_id}_{node_id}` (e.g. `expert_01_c001`) or with a `:evidence_N`
/// suffix. This strips those back to the canonical expert ID by matching against
/// the KNOWN expert IDs sorted longest-first (so `expert_01` does not shadow
/// `expert_010`), then deduplicates the result.
///
/// Order matters: consensus normalises FIRST, dedups SECOND, so two compound IDs
/// that resolve to the same base expert collapse to one. Splitting on a literal
/// `_c` substring (the previous implementation) corrupted IDs like
/// `pmc_case_study_c001` and missed `:`/`new_` variants — see CR-04.
fn normalise_source_ids(mut synthesis: SynthesisArtifact, panel: &[ExpertResponse]) -> SynthesisArtifact {
    // Build the valid-expert-id set from the panel.
    let valid_ids: Vec<String> = panel
        .iter()
        .map(|r| r.expert_id.as_str().to_string())
        .collect();

    if valid_ids.is_empty() {
        return synthesis;
    }

    // Sort longest-first so `expert_01` does not match before `expert_010`.
    let mut sorted_ids: Vec<&str> = valid_ids.iter().map(|s| s.as_str()).collect();
    sorted_ids.sort_by(|a, b| b.len().cmp(&a.len()));

    let normalise = |source_id: &str| -> String {
        // Strip `:evidence_N` (or similar) suffix first.
        let bare = match source_id.split_once(':') {
            Some((head, _)) => head,
            None => source_id,
        };
        // Strip `new_` prefix (graph denoiser generates `new_{node_id}`).
        let bare = bare.strip_prefix("new_").unwrap_or(bare);

        // Try matching the stripped form against known expert IDs.
        for eid in &sorted_ids {
            if bare == *eid
                || bare.starts_with(&format!("{eid}_"))
                || bare.starts_with(&format!("{eid}:"))
            {
                return (*eid).to_string();
            }
        }
        // Backward-compat: try the original (un-stripped) source_id too.
        for eid in &sorted_ids {
            if source_id == *eid
                || source_id.starts_with(&format!("{eid}_"))
                || source_id.starts_with(&format!("{eid}:"))
            {
                return (*eid).to_string();
            }
        }
        // No match — return the original id unchanged.
        source_id.to_string()
    };

    for claim in &mut synthesis.claims {
        // Normalise FIRST.
        let normalised: Vec<String> = claim.sources.iter().map(|s| normalise(s)).collect();
        // Dedup SECOND, preserving first-seen order. `Claim.sources` carries no
        // quote, so the dedup key is the (normalised) source_id alone.
        let mut seen: HashSet<String> = HashSet::new();
        claim.sources = normalised
            .into_iter()
            .filter(|s| seen.insert(s.clone()))
            .collect();
        // F11 (probe 15): quote sources get the same node-id→paper-id
        // normalisation, so the DB-backed verifier (step 5) can resolve
        // them. Unmatched ids pass through unchanged — quotes citing
        // legitimate non-panel papers stay resolvable via the resolver.
        for q in &mut claim.quotes {
            q.source = normalise(&q.source);
        }
    }

    synthesis
}

// ── F13: allowlist-based strip (v2 only) ─────────────────────────────────────

/// Strip invalid source ids from claim sources and drop quotes citing them.
///
/// F13 (probe-18): replaces `strip_candidate_sources` (Fix B). Called after BOTH
/// `normalise_source_ids` call sites (step 1 and the post-revision re-normalise),
/// gated to V2LitReview.
///
/// A source id is valid iff it passes `is_valid_source_id(id, panel_ids)`:
/// - arxiv:/s2:-prefixed (shape lane, covers F11 non-panel papers), OR
/// - an exact member of the known panel expert-id set.
///
/// Everything else — including probe-18 mutated labels (`s1_candidate1` ..
/// `s5_candidate5`) and probe-17 canonical forms (`Candidate1`, etc.) — is
/// stripped from claim.sources and dropped from claim.quotes.
///
/// ## Constraint reconciliation (code comment intentional)
///
/// The locked decision says "in normalise_source_ids", but that function runs on
/// the v1 path too. Gating the strip here to V2LitReview honours the decision's
/// intent (the post-process source-normalisation surface) while preserving the
/// repo's v1-byte-identity discipline.
///
/// A claim left with zero sources after the strip survives post-process (selection
/// already happened) but is no longer falsely sourced. Selection-time protection is
/// in fitness.rs traceability VETO (F13 prong 2).
fn strip_invalid_sources(mut synthesis: SynthesisArtifact, panel: &[crate::adapter::ExpertResponse]) -> SynthesisArtifact {
    use std::collections::HashSet;

    let panel_ids: HashSet<String> = panel
        .iter()
        .map(|r| r.expert_id.as_str().to_string())
        .collect();

    let mut n_stripped_sources = 0usize;
    let mut n_dropped_quotes = 0usize;

    for claim in &mut synthesis.claims {
        let before_sources = claim.sources.len();
        claim.sources.retain(|s| is_valid_source_id(s, &panel_ids));
        n_stripped_sources += before_sources.saturating_sub(claim.sources.len());

        let before_quotes = claim.quotes.len();
        claim.quotes.retain(|q| is_valid_source_id(&q.source, &panel_ids));
        n_dropped_quotes += before_quotes.saturating_sub(claim.quotes.len());
    }

    if n_stripped_sources > 0 || n_dropped_quotes > 0 {
        tracing::warn!(
            n_stripped_sources,
            n_dropped_quotes,
            "strip_invalid_sources: removed invalid source ids from synthesis \
             (F13 — allowlist replaces CandidateN blacklist; probe-18 mutated \
             labels s1_candidate1..s5_candidate5 are now stripped)"
        );
    }

    synthesis
}

// ── Step 3: Compute agreement_level ────────────────────────────────────────

/// Compute `agreement_level` deterministically from the corroboration ratio.
///
/// Mirrors `_compute_agreement_level`
/// (`consensus/src/consensus/summarise/strategies.py:283-295`), the function the
/// post-process step ports. The corroboration count is the number of UNIQUE
/// experts excluding the synthetic `system`/`system_resolution` IDs
/// (`synthesis_tasks.py:1038-1044`), NOT the raw source count.
///
/// Hard thresholds that OVERRIDE any LLM judgment (T-23-08 mitigation) — but
/// only when the corroboration count is RESOLVABLE:
/// - ratio ≥ 0.75 → "consensus"
/// - ratio ≥ 0.50 → "majority"
/// - ratio ≥ 0.15 → "divided"
/// - else          → "minority"
///
/// ## Data-missing case (WR-01)
///
/// Consensus stores `corroboration_count` on the `Claim` (computed once in the
/// merger as `len(unique_experts)`, synthesis_tasks.py:1042); it defaults to
/// `None` (models.py:88). `_compute_agreement_level` returns `None` when the
/// count is `None`, and `_post_process_synthesis` then PRESERVES the LLM label
/// (strategies.py:487-490 `else: enriched_claims.append(c)`).
///
/// The Rust port derives the corroboration count from `claim.sources` rather
/// than a stored field. The faithful mapping of "count is None" is "no
/// resolvable non-system expert sources" — an empty corroboration set. In that
/// case we PRESERVE the model's label (matching the `level is None` branch)
/// instead of forcing "minority". We recompute the deterministic label ONLY
/// when at least one non-system expert is resolvable (the `Some(count)` case).
///
/// `panel_size = 0` → level unchanged (also the consensus `panel_size <= 0 ->
/// None` guard at strategies.py:285).
pub fn compute_agreement_levels(
    mut synthesis: SynthesisArtifact,
    panel_size: usize,
) -> SynthesisArtifact {
    if panel_size == 0 {
        return synthesis; // no data — preserve existing label
    }

    for claim in &mut synthesis.claims {
        // CR-02: corroboration_count = unique experts excluding synthetic IDs.
        let corroboration: HashSet<&str> = claim
            .sources
            .iter()
            .map(|s| s.as_str())
            .filter(|s| *s != "system" && *s != "system_resolution")
            .collect();

        // WR-01: an empty corroboration set is the Rust analogue of consensus's
        // `corroboration_count is None` — preserve the LLM label, do NOT force
        // "minority" (strategies.py:487-490).
        if corroboration.is_empty() {
            continue;
        }

        let ratio = corroboration.len() as f64 / panel_size as f64;

        // CR-01: "divided" cut is 0.15 (strategies.py:293), not 0.30.
        let level = if ratio >= 0.75 {
            "consensus"
        } else if ratio >= 0.50 {
            "majority"
        } else if ratio >= 0.15 {
            "divided"
        } else {
            "minority"
        };

        claim.agreement_level = Some(level.to_string());
    }

    synthesis
}

// ── Step 4: Quote resolve (LLM spawn) ─────────────────────────────────────

/// Resolve placeholder quotes to verbatim substrings via a governed spawn.
///
/// One LLM dispatch through AgentExecutor (ENGINE-05). WR-07 — HONESTY NOTE: the
/// spawn fires and its `<quotes>` output is parsed, but the resolved quotes are
/// NOT yet written back to the synthesis claims, because `Claim.sources` carries
/// no per-source quote field under the current schema. Write-back is Phase 25
/// scope (paired with the SourceReference schema change in WR-03).
async fn resolve_quotes(
    mut synthesis: SynthesisArtifact,
    panel: &[ExpertResponse],
    executor: &Arc<dyn AgentExecutor>,
) -> Result<SynthesisArtifact, TtdError> {
    use crate::ttd::prompts::synthesis::{render_synthesis_quote_resolve, SynthesisQuoteResolveInput};
    use alzina_core::identity::AgentId;

    // Build source texts for the quote_resolve prompt.
    let source_texts: Vec<(String, String)> = panel
        .iter()
        .map(|r| (r.expert_id.as_str().to_string(), r.prose.clone()))
        .collect();

    let input = SynthesisQuoteResolveInput {
        draft: &synthesis,
        source_texts: &source_texts,
    };
    let prompt = render_synthesis_quote_resolve(&input);

    let agent_id = AgentId::new("quote-resolver");
    // Model comes from the artifact (= EngineConfig.model), NOT a hardcoded
    // string: a pin here bypasses the deployment's model routing (live probe
    // 2026-06-10 — engine completed all three stages on the overridden model,
    // then this spawn died on the unservable hardcoded one).
    let output = executor
        .execute(&agent_id, &prompt, &synthesis.model, "quote_resolve")
        .await
        .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

    // WR-07: apply_resolved_quotes is a stub under the current schema — it does
    // not write quotes back. See its doc comment.
    apply_resolved_quotes(&mut synthesis, &output);

    Ok(synthesis)
}

/// Stub for applying quote_resolve output back to the synthesis claims.
///
/// WR-07 — HONESTY NOTE: this does NOT write resolved quotes back. The
/// quote_resolve spawn returns a `<quotes>` block keyed
/// `<claim id="..."><source id="..."><quote>...</quote></source></claim>`, but
/// `Claim.sources` is a bare `Vec<String>` of paper IDs with no quote field to
/// attach a resolved quote to. Faithful write-back needs the SourceReference
/// schema change (paired with WR-03), which is Phase 25 scope.
///
/// We deliberately do NOT parse-then-discard the XML (the previous version did,
/// which read as a working apply step while doing nothing). When the schema
/// lands, parse `output` here and match claim/source IDs to attach quotes.
fn apply_resolved_quotes(_synthesis: &mut SynthesisArtifact, _output: &str) {
    tracing::debug!(
        "quote_resolve: apply step is a Phase-25 stub (no SourceReference quote \
         field to write back under the current schema) — output not applied"
    );
}

// ── Step 5: Verify synthesis quotes (LIGHTWEIGHT STUB — see WR-03) ───────────

/// Lightweight pass-through stub for synthesis quote verification.
///
/// WR-03 — HONESTY NOTE: this is NOT the full consensus
/// `_verify_synthesis_quotes` (strategies.py:215-251). That function writes a
/// per-source `verification_status`/`match_score`/`matched_text` back onto each
/// `SourceReference` and feeds `_build_verification_summary`. The Rust
/// `Claim.sources` is a bare `Vec<String>` (paper IDs only — no quote text and
/// no per-source verification fields), so there is nothing here to verify a
/// quote against or write a status onto. Faithful per-source verification needs
/// the SourceReference schema change, which is Phase 25 scope.
///
/// This function therefore returns the synthesis UNCHANGED and performs no
/// verification. It does not enforce the T-23-09 trust boundary on its own — see
/// the downgraded module-level T-23-09 note. Kept as an explicit, named step so
/// the 7-step chain stays legible and the Phase 25 wiring point is obvious.
async fn verify_synthesis_quotes(
    mut synthesis: SynthesisArtifact,
    panel: &[ExpertResponse],
    resolver: Option<&Arc<dyn crate::ttd::engine::PanelRefresher>>,
) -> SynthesisArtifact {
    use std::collections::{HashMap, HashSet};

    // Collect cited quote sources.
    let cited: HashSet<String> = synthesis
        .claims
        .iter()
        .flat_map(|c| c.quotes.iter().map(|q| q.source.clone()))
        .collect();
    if cited.is_empty() {
        return synthesis;
    }

    // Source texts: panel prose first (already in memory)…
    let mut texts: HashMap<String, String> = panel
        .iter()
        .filter(|e| cited.contains(e.expert_id.as_str()))
        .map(|e| (e.expert_id.as_str().to_string(), e.prose.clone()))
        .collect();

    // …then the DB-backed resolver for sources the model quoted from
    // retrieval context (design rule 4: verification keys on stored text,
    // never on what was in a prompt).
    let unresolved: Vec<String> = cited
        .iter()
        .filter(|id| !texts.contains_key(*id))
        .cloned()
        .collect();
    if !unresolved.is_empty() {
        if let Some(resolver) = resolver {
            match resolver.refresh(&unresolved).await {
                Ok(resolved) => {
                    for e in resolved {
                        texts.insert(e.expert_id.as_str().to_string(), e.prose);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        n_unresolved = unresolved.len(),
                        "verify_synthesis_quotes: resolver failed — quotes from \
                         unresolved sources will stamp absent"
                    );
                }
            }
        }
    }

    // Substring-check every quote against its cited source's stored text.
    // F12: a quote that fails the substring check gets ONE deterministic
    // rescue attempt — snap it to the closest sentence in the stored prose
    // (probe 16: haiku writes fluent paraphrases and labels them quotes;
    // 0/25 verified despite resolvable sources). A snapped quote is verified
    // by construction (the sentence IS a substring) and carries `snapped:
    // true` for honesty. Below the similarity threshold the status stays
    // absent/paraphrased as before.
    let (mut n_verified, mut n_paraphrased, mut n_absent, mut n_snapped) =
        (0usize, 0usize, 0usize, 0usize);
    for claim in &mut synthesis.claims {
        for q in &mut claim.quotes {
            let status = match texts.get(&q.source) {
                Some(text) => {
                    let s = crate::ttd::stages::graph::verify_quote_status(&q.text, text);
                    if s == "verified" {
                        s
                    } else if let Some(sentence) = snap_quote_to_source(&q.text, text) {
                        q.text = sentence;
                        q.snapped = true;
                        n_snapped += 1;
                        "verified".to_string()
                    } else {
                        s
                    }
                }
                None => "absent".to_string(),
            };
            match status.as_str() {
                "verified" => n_verified += 1,
                "paraphrased" => n_paraphrased += 1,
                _ => n_absent += 1,
            }
            q.status = Some(status);
        }
    }
    tracing::info!(
        target: "ttd_perf",
        n_quotes = n_verified + n_paraphrased + n_absent,
        n_verified,
        n_snapped,
        n_paraphrased,
        n_absent,
        "ttd_perf: synthesis quote verification (WR-03 closed; F12 snapping live)"
    );
    synthesis
}

// ── F12: quote snapping ──────────────────────────────────────────────────────

/// Minimum token-overlap score for a snap (|Q ∩ S| / |Q| over normalised
/// content tokens). Below this the paraphrase shares too little with any
/// single source sentence to claim it as the grounding text.
const SNAP_THRESHOLD: f64 = 0.55;
/// Minimum count of overlapping content tokens — guards short quotes of
/// common words from spuriously snapping to an unrelated sentence.
const SNAP_MIN_OVERLAP_TOKENS: usize = 4;

/// Normalised content tokens for scoring purposes (single source of truth).
///
/// Lowercase, alphanumeric-only words of length > 2. Used by both
/// `snap_quote_to_source` and `inherit_graph_quotes` (Fix C) so the
/// normalisation is consistent across both scoring surfaces.
fn content_tokens(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Find the closest sentence in `source_text` to a paraphrased `quote`.
///
/// Deterministic, no LLM. Sentences are EXACT trimmed slices of
/// `source_text`, so a returned sentence is a verbatim substring by
/// construction — stamping it "verified" is true, not optimistic.
///
/// Scoring: normalised content tokens (lowercased, alphanumeric, length > 2);
/// score = |quote ∩ sentence| / |quote| (containment — the grounding sentence
/// may legitimately be longer than the paraphrase). Ties break to the higher
/// score, then the shorter sentence (tighter ground). Returns `None` when no
/// sentence clears both `SNAP_THRESHOLD` and `SNAP_MIN_OVERLAP_TOKENS`.
pub(crate) fn snap_quote_to_source(quote: &str, source_text: &str) -> Option<String> {
    let quote_tokens = content_tokens(quote);
    if quote_tokens.len() < SNAP_MIN_OVERLAP_TOKENS {
        return None;
    }

    // Split into sentence slices at terminators followed by whitespace, and
    // at newlines (chunked prose often lacks terminal punctuation at chunk
    // boundaries). Slices keep their terminator; trim() preserves
    // substring-ness.
    let mut sentences: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let bytes = source_text.as_bytes();
    for (i, c) in source_text.char_indices() {
        let is_terminator = matches!(c, '.' | '!' | '?')
            && bytes
                .get(i + 1)
                .is_none_or(|b| b.is_ascii_whitespace());
        if is_terminator || c == '\n' {
            let end = i + c.len_utf8();
            let slice = source_text[start..end].trim();
            if !slice.is_empty() {
                sentences.push(slice);
            }
            start = end;
        }
    }
    let tail = source_text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    let mut best: Option<(f64, &str)> = None;
    for sentence in sentences {
        let sentence_tokens = content_tokens(sentence);
        let overlap = quote_tokens.intersection(&sentence_tokens).count();
        if overlap < SNAP_MIN_OVERLAP_TOKENS {
            continue;
        }
        let score = overlap as f64 / quote_tokens.len() as f64;
        if score < SNAP_THRESHOLD {
            continue;
        }
        let better = match best {
            None => true,
            Some((best_score, best_sentence)) => {
                score > best_score
                    || (score == best_score && sentence.len() < best_sentence.len())
            }
        };
        if better {
            best = Some((score, sentence));
        }
    }

    best.map(|(_, s)| s.to_string())
}

// ── F14: deterministic node-cited quote floor ───────────────────────────────

/// Attach cited-node quotes onto F14 claims by EXACT node id (no token match).
///
/// Each F14 draft claim carries explicit `node_refs` — the graph node ids the
/// draft cited (it authors no quotes itself; the Opus merger authors verbatim
/// quotes from the cited nodes' verified evidence). This floor is the coverage
/// guarantee under the merger's relevance selection: for any claim that carries
/// `node_refs` but ends the verify pass WITHOUT a verified quote (the merger
/// omitted or mis-copied one), attach the cited nodes' DB-verified quotes by
/// exact id.
///
/// Per-claim gate: skip claims that already hold ANY verified quote — the merger
/// did its job there. This is a floor, not enrichment.
///
/// Each attached quote: source=node.expert_id, text=node.quote,
/// node_id=Some(node.id), inherited=true, status=None (re-stamped by
/// `verify_synthesis_quotes` — no transitive trust of the graph's own stamp; an
/// unresolvable source still stamps absent). Skips nodes that are not
/// DB-verified or have an empty quote.
///
/// Returns (updated synthesis, n_attached). No LLM, no async. Idempotent: skips
/// when the node_id is already attached or an identical (source, text) exists.
pub(crate) fn attach_node_ref_quotes(
    mut synthesis: SynthesisArtifact,
    graph: &crate::ttd::artifact::ArgumentationGraph,
) -> (SynthesisArtifact, usize) {
    use crate::ttd::artifact::ClaimQuote;

    let mut n_attached = 0usize;

    for claim in &mut synthesis.claims {
        if claim.node_refs.is_empty() {
            continue;
        }
        // Floor, not enrichment: if the merger already produced a verified quote
        // for this claim, leave it alone.
        if claim
            .quotes
            .iter()
            .any(|q| q.status.as_deref() == Some("verified"))
        {
            continue;
        }

        for node_ref in &claim.node_refs {
            let Some(node) = graph.nodes.iter().find(|n| &n.id == node_ref) else {
                continue;
            };
            if node.verification_status.as_deref() != Some("verified") {
                continue;
            }
            let Some(quote_text) = node
                .quote
                .as_deref()
                .filter(|q| !q.trim().is_empty())
                .map(str::to_string)
            else {
                continue;
            };
            // Idempotent: skip if this node id is already attached, or an
            // identical (source, text) quote already exists.
            if claim
                .quotes
                .iter()
                .any(|q| q.node_id.as_deref() == Some(node.id.as_str()))
            {
                continue;
            }
            if claim
                .quotes
                .iter()
                .any(|q| q.source == node.expert_id && q.text == quote_text)
            {
                continue;
            }
            claim.quotes.push(ClaimQuote {
                source: node.expert_id.clone(),
                text: quote_text,
                status: None, // re-stamped by verify_synthesis_quotes
                snapped: false,
                inherited: true,
                node_id: Some(node.id.clone()),
            });
            n_attached += 1;
        }
    }

    if n_attached > 0 {
        tracing::info!(
            target: "ttd_perf",
            n_attached,
            "ttd_perf: attach_node_ref_quotes — F14 floor attached {n_attached} cited-node quotes by exact id"
        );
    }

    (synthesis, n_attached)
}

// ── Fix C: inherit DB-verified graph quotes (probe-17 cause 2) ───────────────

/// Threshold for inheriting a graph node's quote onto a synthesis claim.
///
/// Scoring direction: overlap of the NODE's claim tokens (more focused) /
/// node_claim_token_count. Synthesis claims merge several node claims so the
/// containment runs this direction: if most of what the node says is also said
/// in the synthesis claim, the node is a plausible source.
const INHERIT_CLAIM_MATCH_THRESHOLD: f64 = 0.5;

/// Deterministically inherit DB-verified graph node quotes onto synthesis claims.
///
/// For each (claim, source) pair where the claim has NO quote with status
/// "verified" for that source: find candidate graph nodes where
/// `expert_id == source`, `verification_status == Some("verified")`, and the
/// node has a non-empty quote. Score each candidate by
/// |node_claim_tokens ∩ synthesis_claim_tokens| / |node_claim_tokens|
/// (containment). Require overlap >= SNAP_MIN_OVERLAP_TOKENS and score >=
/// INHERIT_CLAIM_MATCH_THRESHOLD. Take the best score (tie → shorter node
/// claim). Attach ONE ClaimQuote per (claim, source) pair: source=expert_id,
/// text=node.quote, status=None (re-stamped by verify_synthesis_quotes),
/// inherited=true.
///
/// Returns (updated synthesis, n_inherited).
///
/// No LLM, no async. Idempotent: skips if an identical quote text for that
/// source already exists.
pub(crate) fn inherit_graph_quotes(
    mut synthesis: SynthesisArtifact,
    graph: &crate::ttd::artifact::ArgumentationGraph,
) -> (SynthesisArtifact, usize) {
    use crate::ttd::artifact::ClaimQuote;

    let mut n_inherited = 0usize;

    for claim in &mut synthesis.claims {
        // F14: node-cited claims are owned by the deterministic floor
        // (`attach_node_ref_quotes`), which attaches by exact id. The lossy
        // ≥0.5 token-containment path here is only a backstop for untagged /
        // legacy claims that carry no node_refs.
        if !claim.node_refs.is_empty() {
            continue;
        }
        for source in &claim.sources {
            // Skip if there's already a verified quote for this source.
            if claim.quotes.iter().any(|q| &q.source == source && q.status.as_deref() == Some("verified")) {
                continue;
            }

            // Candidate nodes: same expert_id, DB-verified, non-empty quote.
            let candidates: Vec<&crate::ttd::artifact::GraphNode> = graph.nodes
                .iter()
                .filter(|n| {
                    &n.expert_id == source
                        && n.verification_status.as_deref() == Some("verified")
                        && n.quote.as_deref().map_or(false, |q| !q.trim().is_empty())
                })
                .collect();

            if candidates.is_empty() {
                continue;
            }

            let claim_tokens = content_tokens(&claim.text);

            // Score each candidate by containment: node claim tokens ∩ synthesis claim tokens.
            let mut best: Option<(&crate::ttd::artifact::GraphNode, f64)> = None;
            for node in candidates {
                let node_tokens = content_tokens(&node.claim);
                if node_tokens.len() < SNAP_MIN_OVERLAP_TOKENS {
                    continue;
                }
                let overlap = node_tokens.intersection(&claim_tokens).count();
                if overlap < SNAP_MIN_OVERLAP_TOKENS {
                    continue;
                }
                let score = overlap as f64 / node_tokens.len() as f64;
                if score < INHERIT_CLAIM_MATCH_THRESHOLD {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some((best_node, best_score)) => {
                        score > best_score
                            || (score == best_score
                                && node.claim.len() < best_node.claim.len())
                    }
                };
                if better {
                    best = Some((node, score));
                }
            }

            if let Some((node, _score)) = best {
                let quote_text = node.quote.as_deref().unwrap_or("").to_string();
                // Skip if an identical text already exists for this source.
                if claim.quotes.iter().any(|q| &q.source == source && q.text == quote_text) {
                    continue;
                }
                claim.quotes.push(ClaimQuote {
                    source: source.clone(),
                    text: quote_text,
                    status: None,   // will be stamped by verify_synthesis_quotes
                    snapped: false,
                    inherited: true,
                    node_id: None,
                });
                n_inherited += 1;
            }
        }
    }

    if n_inherited > 0 {
        tracing::info!(
            target: "ttd_perf",
            n_inherited,
            "ttd_perf: inherit_graph_quotes — attached {n_inherited} DB-verified node quotes"
        );
    }

    (synthesis, n_inherited)
}

// ── Step 6: Expert coverage gate ────────────────────────────────────────────

/// Check what fraction of panel experts are cited in the synthesis.
///
/// Returns `(missing_expert_ids, coverage_ratio)`.
/// `coverage_ratio` = proportion of experts with at least one claim citation.
pub fn check_expert_coverage(
    synthesis: &SynthesisArtifact,
    panel: &[ExpertResponse],
) -> (Vec<String>, f64) {
    // Build the set of experts cited in any claim source.
    let cited: HashSet<&str> = synthesis
        .claims
        .iter()
        .flat_map(|c| c.sources.iter().map(|s| s.as_str()))
        .collect();

    let mut missing: Vec<String> = Vec::new();
    for expert in panel {
        let expert_id = expert.expert_id.as_str();
        if !cited.contains(expert_id) {
            missing.push(expert_id.to_string());
        }
    }

    let coverage = if panel.is_empty() {
        1.0
    } else {
        (panel.len() - missing.len()) as f64 / panel.len() as f64
    };

    (missing, coverage)
}

/// Run the optional revision spawn when missing experts are detected.
///
/// Uses `committee/aggregator_revision.mustache` (versioned `v1/committee`).
/// One governed dispatch. Returns the revised synthesis.
async fn run_revision(
    synthesis: SynthesisArtifact,
    panel: &[ExpertResponse],
    missing_expert_ids: &[String],
    executor: &Arc<dyn AgentExecutor>,
    profile: PromptProfile,
    graph: Option<&crate::ttd::artifact::ArgumentationGraph>,
) -> Result<SynthesisArtifact, TtdError> {
    use crate::ttd::prompts::synthesis::{render_aggregator_revision, AggregatorRevisionInput};
    use alzina_core::identity::AgentId;

    // B2: fork on profile — V2LitReview uses paper-provenance revision prompt (D-8).
    // Decision 0: v3 = v2 (revision targets the Stage-2 synthesis, not the
    // Stage-3 narrative — the 500-800 target is untouched by Phase 0).
    let prompt = match profile {
        PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
            // F14 (option B): the revision replaces the synthesis wholesale, so it
            // must author node-grounded quotes from the same full-graph evidence
            // the merger used — else it strips them (probe 24).
            // Depth-probe B: the revision gets the SAME section-widened evidence
            // as the merger, so coverage re-authoring keeps the mechanism depth
            // instead of flattening claims back to field-level (probe-24 lesson).
            let panel_prose: Vec<(String, String)> = panel
                .iter()
                .map(|r| (r.expert_id.as_str().to_string(), r.prose.clone()))
                .collect();
            // Stage-2 soft-filter: the revision sees the SAME tier-tagged evidence
            // as the merger, so re-authored coverage keeps down-weighting weak
            // sources. Derived from the panel provenance; `Unknown` renders no tag.
            let tier_map: std::collections::BTreeMap<String, alzina_search::CredibilityTier> = panel
                .iter()
                .map(|r| {
                    (
                        r.expert_id.as_str().to_string(),
                        r.provenance.credibility_tier,
                    )
                })
                .collect();
            let node_evidence = graph
                .map(|g| {
                    crate::ttd::stages::synthesis::build_graph_evidence_with_sections(
                        g,
                        &panel_prose,
                        &tier_map,
                    )
                })
                .unwrap_or_default();
            crate::ttd::prompts::lit_review::render_revision_v2(
                &synthesis,
                panel,
                missing_expert_ids,
                "500-800",
                &node_evidence,
            )
        }
        PromptProfile::V1Delphi => {
            // V1: render original claims as text. Sources MUST be included:
            // the revision output REPLACES the synthesis wholesale, so any
            // provenance absent from this rendering is permanently laundered away
            // (probe 10: revision pass emitted every claim with sources: []).
            let original_claims: String = synthesis
                .claims
                .iter()
                .map(|c| {
                    let level = c.agreement_level.as_deref().unwrap_or("divided");
                    if c.sources.is_empty() {
                        format!("- ({level}) {text}", level = level, text = c.text)
                    } else {
                        format!(
                            "- ({level}) {text} [sources: {sources}]",
                            level = level,
                            text = c.text,
                            sources = c.sources.join(", ")
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            let original_minority_reports: String = synthesis
                .minority_reports
                .iter()
                .map(|r| format!("- {perspective}", perspective = r.perspective))
                .collect::<Vec<_>>()
                .join("\n");

            let original_uncertainties: String = synthesis.uncertainties.join("\n- ");

            let expert_responses: Vec<(String, String)> = panel
                .iter()
                .map(|r| (r.expert_id.as_str().to_string(), r.prose.clone()))
                .collect();

            let input = AggregatorRevisionInput {
                missing_expert_ids,
                question: "Revise the synthesis to include missing expert voices.",
                expert_responses: &expert_responses,
                original_claims: &original_claims,
                original_minority_reports: &original_minority_reports,
                original_uncertainties: &original_uncertainties,
                n_experts: panel.len(),
                graph_context: None,
            };

            render_aggregator_revision(&input)
        }
    };

    let agent_id = AgentId::new("revision-agent");
    // Model from the artifact, not a hardcoded pin — see resolve_quotes.
    let output = executor
        .execute(&agent_id, &prompt, &synthesis.model, "revision")
        .await
        .map_err(|e| TtdError::SpawnFailed(e.to_string()))?;

    // Parse the revised synthesis from the output.
    match crate::ttd::stages::synthesis::parse_synthesis_xml(
        &output,
        &synthesis.model,
        &synthesis.prompt_version,
        profile,
    ) {
        Ok(revised) => {
            // Provenance guard: the revision REPLACES the synthesis. If the
            // original carried sources and the revision lost them all, the
            // revision is a provenance regression — keep the original and
            // say so loudly rather than shipping untraceable claims.
            let original_had_sources =
                synthesis.claims.iter().any(|c| !c.sources.is_empty());
            let revised_lost_sources = !revised.claims.is_empty()
                && revised.claims.iter().all(|c| c.sources.is_empty());
            if original_had_sources && revised_lost_sources {
                tracing::warn!(
                    n_original_claims = synthesis.claims.len(),
                    n_revised_claims = revised.claims.len(),
                    "run_revision: revised synthesis lost ALL claim sources — \
                     rejecting revision, keeping original (provenance guard)"
                );
                return Ok(synthesis);
            }
            Ok(revised)
        }
        Err(e) => {
            tracing::debug!(
                error = %e,
                "run_revision: parse failed — returning original synthesis"
            );
            Ok(synthesis) // graceful fallback: keep original if revision fails
        }
    }
}

// ── Step 7: Enrich citations + minority reports ────────────────────────────

/// Extract URL/DOI/bracket citations from claim text (deterministic).
fn enrich_citations(synthesis: SynthesisArtifact) -> SynthesisArtifact {
    // In Phase 23 the citation extraction is a structural stub — the full
    // implementation port (URL/DOI regex) is a Phase 25 fidelity task.
    // The stub passes through the synthesis unchanged.
    synthesis
}

/// Compute minority reports from experts not well-represented in claims.
///
/// IN-04: an expert is "minority" when they are cited ONLY in claims whose
/// `agreement_level == "minority"` and appear in NO higher-agreement claim.
/// There is no percentage threshold — the rule is set membership, not a 30%
/// cut (the implementation below collects minority-only sources, then removes
/// any that also appear in a non-minority claim).
fn compute_minority_reports(
    mut synthesis: SynthesisArtifact,
    panel: &[ExpertResponse],
) -> SynthesisArtifact {
    let mut minority_sources: HashSet<String> = HashSet::new();

    // Collect sources that appear ONLY in minority-level claims.
    for claim in &synthesis.claims {
        if claim.agreement_level.as_deref() == Some("minority") {
            for src in &claim.sources {
                minority_sources.insert(src.clone());
            }
        }
    }

    // Remove sources that also appear in non-minority claims.
    let non_minority_sources: HashSet<String> = synthesis
        .claims
        .iter()
        .filter(|c| c.agreement_level.as_deref() != Some("minority"))
        .flat_map(|c| c.sources.iter().cloned())
        .collect();
    minority_sources.retain(|s| !non_minority_sources.contains(s));

    // Build minority reports for experts not covered elsewhere.
    for expert in panel {
        let eid = expert.expert_id.as_str().to_string();
        if minority_sources.contains(&eid) {
            // Expert appears only in minority claims → add a minority report.
            let already_has_report = synthesis
                .minority_reports
                .iter()
                .any(|r| r.source_ids.contains(&eid));
            if !already_has_report {
                synthesis.minority_reports.push(MinorityReport {
                    source_ids: vec![eid],
                    perspective: format!(
                        "Minority perspective from expert cited only in minority claims"
                    ),
                });
            }
        }
    }

    synthesis
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use crate::adapter::{ExpertResponse, ResponseProvenance, SourceId};
    use crate::ttd::artifact::{Claim, ClaimQuote, SynthesisArtifact};
    use crate::ttd::mod_types::TtdError;

    // ── Mock executor ─────────────────────────────────────────────────────────

    struct RecordingExecutor {
        invocations: Arc<std::sync::Mutex<Vec<String>>>,
        response: String,
    }

    #[async_trait]
    impl crate::executor::AgentExecutor for RecordingExecutor {
        async fn execute(
            &self,
            _agent_id: &alzina_core::identity::AgentId,
            _instruction: &str,
            _model: &str,
            task: &str,
        ) -> alzina_core::AlzinaResult<String> {
            self.invocations.lock().unwrap().push(task.to_string());
            Ok(self.response.clone())
        }
    }

    fn make_panel(expert_ids: &[&str]) -> Vec<ExpertResponse> {
        expert_ids
            .iter()
            .map(|id| ExpertResponse {
                expert_id: SourceId::new(*id),
                prose: format!("Prose content for {id}"),
                provenance: ResponseProvenance {
                    source_id: SourceId::new(*id),
                    title: format!("Paper {id}"),
                    year: Some(2021),
                    authors: vec![],
                    credibility_tier: alzina_search::CredibilityTier::Unknown,
                },
            })
            .collect()
    }

    fn make_synthesis(
        claims_data: &[(&str, &str, Vec<&str>)], // (text, agreement_level, sources)
    ) -> SynthesisArtifact {
        let mut art = SynthesisArtifact::new(
            "study-1", "round-1", "q-1", "model", "v1/synthesis",
        );
        for (text, level, sources) in claims_data {
            art.claims.push(Claim {
                text: text.to_string(),
                agreement_level: Some(level.to_string()),
                sources: sources.iter().map(|s| s.to_string()).collect(),
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
        }
        art
    }

    // ── Test 1: source_ids_normalised ─────────────────────────────────────────

    /// Compound graph-node IDs (`{expert_id}_{node_id}`) normalise to the base
    /// expert ID by matching against the known panel IDs (strategies.py:106-212).
    #[test]
    fn source_ids_normalised() {
        let panel = make_panel(&["arxiv:2105.14103", "arxiv:2105.14104"]);

        // A synthesis with graph-node compound IDs
        let synthesis = make_synthesis(&[
            (
                "Permafrost thaw releases methane.",
                "consensus",
                vec!["arxiv:2105.14103_c001", "arxiv:2105.14104_c002"],
            ),
        ]);

        let normalised = normalise_source_ids(synthesis, &panel);

        let sources = &normalised.claims[0].sources;
        // After normalisation, compound IDs are stripped to base expert IDs.
        assert!(
            sources.contains(&"arxiv:2105.14103".to_string()),
            "arxiv:2105.14103 must be in normalised sources: {sources:?}"
        );
        assert!(
            sources.contains(&"arxiv:2105.14104".to_string()),
            "arxiv:2105.14104 must be in normalised sources: {sources:?}"
        );
    }

    /// CR-04 regression: an expert ID containing the literal substring `_c`
    /// (e.g. `pmc_case_study`) must NOT be truncated at the first `_c`. The old
    /// `split("_c")` logic corrupted `pmc_case_study_c001` to `pmc`.
    #[test]
    fn normalise_does_not_split_on_literal_underscore_c() {
        let panel = make_panel(&["pmc_case_study", "expert_010"]);
        let synthesis = make_synthesis(&[(
            "Claim",
            "consensus",
            // compound node IDs that must resolve back to the base experts
            vec!["pmc_case_study_c001", "expert_010_n001"],
        )]);

        let normalised = normalise_source_ids(synthesis, &panel);
        let sources = &normalised.claims[0].sources;

        assert!(
            sources.contains(&"pmc_case_study".to_string()),
            "pmc_case_study must not be truncated to 'pmc': {sources:?}"
        );
        assert!(
            sources.contains(&"expert_010".to_string()),
            "expert_010 must resolve from its _n001 node suffix: {sources:?}"
        );
        assert!(
            !sources.contains(&"pmc".to_string()),
            "corrupted 'pmc' base must not appear: {sources:?}"
        );
    }

    /// CR-04: longest-first matching means `expert_01` does not shadow
    /// `expert_010`, and normalise-then-dedup collapses compound IDs that map to
    /// the same base expert.
    #[test]
    fn normalise_longest_first_then_dedup() {
        let panel = make_panel(&["expert_01", "expert_010"]);
        let synthesis = make_synthesis(&[(
            "Claim",
            "consensus",
            // two compound IDs that both map to expert_010 → dedup to one
            vec!["expert_010_c001", "expert_010_c002", "expert_01_c003"],
        )]);

        let normalised = normalise_source_ids(synthesis, &panel);
        let sources = &normalised.claims[0].sources;

        assert!(
            sources.contains(&"expert_010".to_string()),
            "expert_010 must be matched longest-first: {sources:?}"
        );
        assert!(
            sources.contains(&"expert_01".to_string()),
            "expert_01 must still resolve: {sources:?}"
        );
        // expert_010_c001 and expert_010_c002 collapse to one expert_010.
        let n_010 = sources.iter().filter(|s| *s == "expert_010").count();
        assert_eq!(n_010, 1, "duplicate expert_010 must be deduped: {sources:?}");
    }

    // ── Test 2: agreement_level_thresholds ────────────────────────────────────

    /// Corroboration ratio determines agreement_level deterministically.
    /// Thresholds: ≥0.75 consensus / ≥0.50 majority / ≥0.15 divided / else
    /// minority (strategies.py:283-295). Corroboration count is the unique
    /// non-system expert count (synthesis_tasks.py:1038-1044).
    #[test]
    fn agreement_level_thresholds() {
        let panel_size = 4; // 4 experts total

        // Test each band:
        // consensus: ≥ 0.75 → 3/4 = 0.75 → consensus
        let mut synthesis = make_synthesis(&[(
            "Consensus claim",
            "divided", // will be overridden
            vec!["s1", "s2", "s3"], // 3/4 = 0.75 → consensus
        )]);
        synthesis = compute_agreement_levels(synthesis, panel_size);
        assert_eq!(
            synthesis.claims[0].agreement_level.as_deref(),
            Some("consensus"),
            "3/4 experts → consensus (≥0.75)"
        );

        // majority: 0.50 ≤ ratio < 0.75 → 2/4 = 0.50 → majority
        let mut synthesis2 = make_synthesis(&[(
            "Majority claim",
            "divided",
            vec!["s1", "s2"], // 2/4 = 0.50 → majority
        )]);
        synthesis2 = compute_agreement_levels(synthesis2, panel_size);
        assert_eq!(
            synthesis2.claims[0].agreement_level.as_deref(),
            Some("majority"),
            "2/4 experts → majority (≥0.50)"
        );

        // divided: 0.15 ≤ ratio < 0.50 → 1/4 = 0.25 → divided (CR-01: was wrongly
        // expected to be minority under the 0.30 cut; 0.25 ≥ 0.15 → divided).
        let mut synthesis3 = make_synthesis(&[(
            "Divided claim",
            "consensus",
            vec!["s1"], // 1/4 = 0.25 → divided
        )]);
        synthesis3 = compute_agreement_levels(synthesis3, panel_size);
        assert_eq!(
            synthesis3.claims[0].agreement_level.as_deref(),
            Some("divided"),
            "1/4 experts → divided (≥0.15)"
        );

        // minority: < 0.15 → 1/8 = 0.125 → minority
        let mut synthesis4 = make_synthesis(&[(
            "Minority claim",
            "consensus",
            vec!["s1"], // 1/8 = 0.125 → minority
        )]);
        synthesis4 = compute_agreement_levels(synthesis4, 8);
        assert_eq!(
            synthesis4.claims[0].agreement_level.as_deref(),
            Some("minority"),
            "1/8 experts → minority (<0.15)"
        );
    }

    /// CR-02: synthetic `system`/`system_resolution` IDs and duplicate experts
    /// must NOT count toward corroboration (synthesis_tasks.py:1038-1044).
    #[test]
    fn agreement_level_excludes_system_and_dedups_experts() {
        let panel_size = 4;
        // Raw sources: 4 entries, but only 1 unique real expert. The naive
        // len()-based ratio would be 4/4 = consensus; the correct ratio is
        // 1/4 = 0.25 → divided.
        let synthesis = make_synthesis(&[(
            "Claim",
            "consensus",
            vec!["expert_01", "expert_01", "system", "system_resolution"],
        )]);
        let synthesis = compute_agreement_levels(synthesis, panel_size);
        assert_eq!(
            synthesis.claims[0].agreement_level.as_deref(),
            Some("divided"),
            "1 unique non-system expert / 4 → divided, not consensus"
        );
    }

    /// WR-01: a claim with NO resolvable non-system expert source is the Rust
    /// analogue of consensus's `corroboration_count is None`. The deterministic
    /// recompute must NOT fire; the LLM-provided label is PRESERVED, matching
    /// `strategies.py:487-490` (`else: enriched_claims.append(c)`). The old
    /// behaviour forced "minority" via `ratio = 0.0`.
    #[test]
    fn agreement_level_preserves_llm_label_when_no_resolvable_experts() {
        let panel_size = 4;

        // No sources at all → corroboration set empty → preserve LLM label.
        let synthesis = make_synthesis(&[("Claim", "consensus", vec![])]);
        let synthesis = compute_agreement_levels(synthesis, panel_size);
        assert_eq!(
            synthesis.claims[0].agreement_level.as_deref(),
            Some("consensus"),
            "no resolvable experts → LLM label preserved, not forced to minority"
        );

        // Only synthetic IDs → corroboration set still empty → preserve label.
        let synthesis2 = make_synthesis(&[(
            "Claim",
            "majority",
            vec!["system", "system_resolution"],
        )]);
        let synthesis2 = compute_agreement_levels(synthesis2, panel_size);
        assert_eq!(
            synthesis2.claims[0].agreement_level.as_deref(),
            Some("majority"),
            "only synthetic IDs → LLM label preserved, not forced to minority"
        );
    }

    /// IN-03: `enrich_citations` is a Phase-23 pass-through stub. Pin the
    /// pass-through contract so a future edit cannot silently start mutating
    /// claims without tripping a guard (mirrors the explicit-stub discipline
    /// used for the quote steps).
    #[test]
    fn enrich_citations_is_passthrough_stub() {
        let synthesis = make_synthesis(&[
            ("Claim one", "consensus", vec!["expert_01"]),
            ("Claim two", "minority", vec!["expert_02"]),
        ]);
        let before = synthesis.clone();
        let after = enrich_citations(synthesis);
        assert_eq!(
            after.claims, before.claims,
            "enrich_citations (Phase-23 stub) must return claims unchanged"
        );
    }

    // ── Test 3: quote_resolve spawn invoked ───────────────────────────────────

    /// post_process_synthesis runs the quote_resolve spawn as a governed dispatch.
    #[tokio::test]
    async fn quote_resolve_spawn_invoked() {
        let invocations = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let executor: Arc<dyn crate::executor::AgentExecutor> = Arc::new(RecordingExecutor {
            invocations: invocations.clone(),
            response: "<quotes/>".to_string(), // empty quotes — no-op
        });

        let panel = make_panel(&["arxiv:2105.14103"]);
        let synthesis = make_synthesis(&[(
            "Climate change is real.",
            "consensus",
            vec!["arxiv:2105.14103"],
        )]);

        let _result = post_process_synthesis(synthesis, &panel, &executor, crate::ttd::term_sheet::PromptProfile::V1Delphi, None)
            .await
            .expect("post_process must succeed");

        let invoked = invocations.lock().unwrap().clone();
        assert!(
            invoked.contains(&"quote_resolve".to_string()),
            "quote_resolve must be invoked as a governed spawn: {invoked:?}"
        );
    }

    // ── Test 4: revision runs on coverage gap ─────────────────────────────────

    /// When check_expert_coverage finds missing experts, the revision spawn fires.
    #[tokio::test]
    async fn revision_runs_on_coverage_gap() {
        let invocations = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        // Return a valid synthesis XML for the revision spawn
        let revision_xml = r#"<synthesis>
  <claims>
    <claim id="C1">
      <text>Revised claim.</text>
      <agreement_level>consensus</agreement_level>
      <sources><source id="arxiv:2105.14103"/><source id="arxiv:MISSING.001"/></sources>
    </claim>
  </claims>
  <areas_of_agreement><area>Agreement</area></areas_of_agreement>
  <areas_of_disagreement><area>Disagreement</area></areas_of_disagreement>
  <uncertainties><uncertainty>Unknown</uncertainty></uncertainties>
</synthesis>"#;

        let executor: Arc<dyn crate::executor::AgentExecutor> = Arc::new(RecordingExecutor {
            invocations: invocations.clone(),
            response: revision_xml.to_string(),
        });

        // Panel has TWO experts, but synthesis only cites ONE
        let panel = make_panel(&["arxiv:2105.14103", "arxiv:MISSING.001"]);
        let synthesis = make_synthesis(&[(
            "Climate change is real.",
            "consensus",
            vec!["arxiv:2105.14103"], // arxiv:MISSING.001 is not cited
        )]);

        let _result = post_process_synthesis(synthesis, &panel, &executor, crate::ttd::term_sheet::PromptProfile::V1Delphi, None)
            .await
            .expect("post_process must succeed even with coverage gap");

        let invoked = invocations.lock().unwrap().clone();
        assert!(
            invoked.contains(&"revision".to_string()),
            "revision must fire when experts are missing from synthesis: {invoked:?}"
        );
    }

    // ── Test 5: check_expert_coverage ────────────────────────────────────────

    #[test]
    fn check_expert_coverage_detects_missing() {
        let panel = make_panel(&["arxiv:2105.14103", "arxiv:2105.14104", "arxiv:2105.14105"]);
        let synthesis = make_synthesis(&[(
            "Claim",
            "majority",
            vec!["arxiv:2105.14103", "arxiv:2105.14104"], // missing arxiv:2105.14105
        )]);

        let (missing, coverage) = check_expert_coverage(&synthesis, &panel);
        assert!(
            missing.contains(&"arxiv:2105.14105".to_string()),
            "arxiv:2105.14105 must be in missing experts"
        );
        assert!(
            (coverage - 2.0 / 3.0).abs() < 0.01,
            "coverage must be 2/3 ≈ 0.667, got {coverage}"
        );
    }

    // ── Task 3 RED test: run_revision v2 path ────────────────────────────────

    /// Under V2LitReview, run_revision must render the v2 revision prompt —
    /// prompt contains "weaving uncovered papers" (v2 marker) and does NOT
    /// contain "missing expert voices" (v1 dressing).
    #[tokio::test]
    async fn run_revision_v2_renders_v2_prompt() {
        use crate::ttd::term_sheet::PromptProfile;
        use std::sync::Mutex;

        struct CapturingExecutor {
            prompts: Arc<Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl crate::executor::AgentExecutor for CapturingExecutor {
            async fn execute(
                &self,
                _agent_id: &alzina_core::identity::AgentId,
                instruction: &str,
                _model: &str,
                _task: &str,
            ) -> alzina_core::AlzinaResult<String> {
                self.prompts.lock().unwrap().push(instruction.to_string());
                // Return minimal valid v2 synthesis XML
                Ok(r#"<synthesis>
  <claims>
    <claim id="C1">
      <text>Permafrost thaw releases methane</text>
      <support_level>converging</support_level>
      <sources><source id="arxiv:2105.14103"/></sources>
    </claim>
  </claims>
  <areas_of_agreement><area>Thaw occurs</area></areas_of_agreement>
  <areas_of_disagreement></areas_of_disagreement>
  <uncertainties></uncertainties>
</synthesis>"#.to_string())
            }
        }

        let prompts = Arc::new(Mutex::new(Vec::<String>::new()));
        let executor: Arc<dyn crate::executor::AgentExecutor> = Arc::new(CapturingExecutor {
            prompts: prompts.clone(),
        });

        let panel = make_panel(&["arxiv:2105.14103", "arxiv:2105.14104"]);
        let synthesis = make_synthesis(&[(
            "Claim one",
            "converging",
            vec!["arxiv:2105.14103"],
        )]);
        let missing = vec!["arxiv:2105.14104".to_string()];

        // Call run_revision directly with V2LitReview
        let _result = run_revision(synthesis, &panel, &missing, &executor, PromptProfile::V2LitReview, None)
            .await
            .expect("run_revision must not error");

        let captured = prompts.lock().unwrap();
        assert!(
            !captured.is_empty(),
            "executor must have been called by run_revision"
        );
        let prompt = &captured[0];
        assert!(
            prompt.contains("weaving") || prompt.contains("uncovered papers"),
            "v2 revision prompt must contain v2 framing ('weaving'/'uncovered papers'); got start: {}",
            &prompt[..300.min(prompt.len())]
        );
        assert!(
            !prompt.contains("missing expert voices"),
            "v2 revision prompt must NOT contain 'missing expert voices' (v1 dressing)"
        );
    }

    // ── F12: quote snapping ──────────────────────────────────────────────────

    const SOURCE_PROSE: &str = "Temporal knowledge graphs extend static graphs \
with time-stamped edges. Our evaluation shows that retrieval grounded in a \
temporal graph layer reduces hallucination rates by thirty percent across \
agent benchmarks. Unrelated filler sentence about something else entirely.";

    /// A fluent paraphrase snaps to the exact source sentence (which is a
    /// verbatim substring of the stored prose by construction).
    #[test]
    fn snap_finds_closest_sentence_for_paraphrase() {
        // Paraphrase: reordered, synonyms swapped, but shares content tokens.
        let paraphrase = "retrieval grounded in temporal graph layers reduces \
hallucination rates across agent benchmarks";
        let snapped = snap_quote_to_source(paraphrase, SOURCE_PROSE)
            .expect("paraphrase must snap to its grounding sentence");
        assert!(
            SOURCE_PROSE.contains(&snapped),
            "snapped text must be an exact substring of the source: {snapped:?}"
        );
        assert!(
            snapped.contains("thirty percent"),
            "must pick the evaluation sentence, got: {snapped:?}"
        );
    }

    /// A quote sharing too few content tokens with any sentence stays unsnapped.
    #[test]
    fn snap_rejects_unrelated_quote() {
        let unrelated = "Skuld timeout rates exceeded expectations during the \
weave cancellation ladder rollout";
        assert!(
            snap_quote_to_source(unrelated, SOURCE_PROSE).is_none(),
            "unrelated quote must not snap"
        );
    }

    /// Short quotes (under the minimum content-token count) never snap —
    /// guards common-word fragments from matching arbitrary sentences.
    #[test]
    fn snap_rejects_short_quotes() {
        assert!(
            snap_quote_to_source("the graphs extend", SOURCE_PROSE).is_none(),
            "a 2-content-token quote must not snap"
        );
    }

    /// End-to-end through verify_synthesis_quotes: a paraphrased quote against
    /// panel prose comes back verified + snapped with replaced text; an
    /// unrelated quote stays absent; a verbatim quote verifies without snapping.
    #[tokio::test]
    async fn verify_synthesis_quotes_snaps_paraphrases() {
        let mut panel = make_panel(&["arxiv:2501.00001"]);
        panel[0].prose = SOURCE_PROSE.to_string();

        let mut synthesis = make_synthesis(&[(
            "Temporal grounding reduces hallucination.",
            "consensus",
            vec!["arxiv:2501.00001"],
        )]);
        synthesis.claims[0].quotes = vec![
            ClaimQuote {
                source: "arxiv:2501.00001".into(),
                text: "retrieval grounded in temporal graph layers reduces \
hallucination rates across agent benchmarks"
                    .into(),
                status: None,
                snapped: false,
                inherited: false,
                node_id: None,
            },
            ClaimQuote {
                source: "arxiv:2501.00001".into(),
                text: "weave cancellation ladder rollout exceeded Skuld timeout \
expectations entirely elsewhere"
                    .into(),
                status: None,
                snapped: false,
                inherited: false,
                node_id: None,
            },
            ClaimQuote {
                source: "arxiv:2501.00001".into(),
                text: "Temporal knowledge graphs extend static graphs with \
time-stamped edges."
                    .into(),
                status: None,
                snapped: false,
                inherited: false,
                node_id: None,
            },
        ];

        let result = verify_synthesis_quotes(synthesis, &panel, None).await;
        let quotes = &result.claims[0].quotes;

        // Paraphrase → snapped + verified, text replaced by a real substring.
        assert_eq!(quotes[0].status.as_deref(), Some("verified"));
        assert!(quotes[0].snapped, "paraphrase must carry the snapped marker");
        assert!(
            SOURCE_PROSE.contains(&quotes[0].text),
            "snapped quote text must be verbatim source text"
        );

        // Unrelated → absent, untouched.
        assert_eq!(quotes[1].status.as_deref(), Some("absent"));
        assert!(!quotes[1].snapped);
        assert!(quotes[1].text.contains("Skuld"), "absent quote text unchanged");

        // Verbatim → verified WITHOUT snapping.
        assert_eq!(quotes[2].status.as_deref(), Some("verified"));
        assert!(!quotes[2].snapped, "verbatim quote must not be marked snapped");
    }

    // CR-04 NOTE: the former `deduplicate_sources_import_strips_compound_suffix`
    // test was removed. Step 1 no longer calls `deduplicate_sources` (it splits
    // on a literal `_c`, which corrupted IDs); normalisation now matches against
    // the panel longest-first. The Phase-22 `deduplicate_sources` behaviour is
    // covered by its own tests in `adapter::conservation` and must not be
    // modified here. See `source_ids_normalised`,
    // `normalise_does_not_split_on_literal_underscore_c`, and
    // `normalise_longest_first_then_dedup` above for the step-1 coverage.

    // ── Fix A (probe-17 cause 3): re-verify after revision ───────────────────

    /// Under V2LitReview, when the coverage gate triggers a revision and the
    /// mocked executor returns v2 synthesis XML with a claim quote whose text IS
    /// a verbatim substring of a panel member's prose, the final artifact's quote
    /// must carry status Some("verified") — not None.
    ///
    /// Before Fix A: verify_synthesis_quotes was NOT called after the revision
    /// pass, so revised quotes shipped with status None.
    #[tokio::test]
    async fn fix_a_revision_quote_status_verified_when_verbatim() {
        use crate::ttd::term_sheet::PromptProfile;

        // Panel prose contains a known verbatim sentence.
        let known_sentence =
            "Permafrost thaw releases methane at measurable flux rates.";
        let panel_prose = format!("{known_sentence} More context follows.");

        // Panel: two experts — s1 (cited) and s2 (missing, triggers revision).
        let mut panel = make_panel(&["arxiv:s1.001", "arxiv:s2.002"]);
        panel[0].prose = panel_prose.clone();
        panel[1].prose = "Other paper prose content.".to_string();

        // Synthesis only cites s1 — s2 is missing, triggering revision.
        let synthesis = make_synthesis(&[(
            "Permafrost thaw releases methane.",
            "converging",
            vec!["arxiv:s1.001"],
        )]);

        // Revision XML: v2 with a quote that is a verbatim substring of s1's prose.
        let revision_xml = format!(
            r#"<synthesis>
  <claims>
    <claim id="C1">
      <text>Permafrost thaw releases methane.</text>
      <support_level>converging</support_level>
      <sources><source id="arxiv:s1.001"/><source id="arxiv:s2.002"/></sources>
      <quotes>
        <quote source="arxiv:s1.001">{}</quote>
      </quotes>
    </claim>
  </claims>
  <areas_of_agreement><area>Thaw occurs</area></areas_of_agreement>
  <areas_of_disagreement></areas_of_disagreement>
  <uncertainties></uncertainties>
</synthesis>"#,
            known_sentence
        );

        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: revision_xml,
            });

        let result = post_process_synthesis(
            synthesis,
            &panel,
            &executor,
            PromptProfile::V2LitReview,
            None,
        )
        .await
        .expect("post_process must succeed");

        assert_eq!(result.claims.len(), 1);
        let quotes = &result.claims[0].quotes;
        assert!(
            !quotes.is_empty(),
            "revised claim must carry quotes after post-process"
        );
        // Fix A: the verbatim quote must be stamped "verified", not left as None.
        assert_eq!(
            quotes[0].status.as_deref(),
            Some("verified"),
            "verbatim quote in revised output must carry status 'verified' — \
             Fix A: re-verify after revision"
        );
    }

    /// Same setup with an absent quote — final status must be Some("absent"),
    /// text untouched, snapped false.
    #[tokio::test]
    async fn fix_a_revision_quote_status_absent_when_unrelated() {
        use crate::ttd::term_sheet::PromptProfile;

        let mut panel = make_panel(&["arxiv:s1.001", "arxiv:s2.002"]);
        panel[0].prose = "Known source prose about permafrost thaw.".to_string();
        panel[1].prose = "Second paper about unrelated topics.".to_string();

        let synthesis = make_synthesis(&[(
            "Permafrost thaw releases methane.",
            "converging",
            vec!["arxiv:s1.001"],
        )]);

        let unrelated_quote = "Completely unrelated sentence about something else entirely.";
        let revision_xml = format!(
            r#"<synthesis>
  <claims>
    <claim id="C1">
      <text>Permafrost thaw releases methane.</text>
      <support_level>converging</support_level>
      <sources><source id="arxiv:s1.001"/><source id="arxiv:s2.002"/></sources>
      <quotes>
        <quote source="arxiv:s1.001">{}</quote>
      </quotes>
    </claim>
  </claims>
  <areas_of_agreement></areas_of_agreement>
  <areas_of_disagreement></areas_of_disagreement>
  <uncertainties></uncertainties>
</synthesis>"#,
            unrelated_quote
        );

        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: revision_xml,
            });

        let result = post_process_synthesis(
            synthesis,
            &panel,
            &executor,
            PromptProfile::V2LitReview,
            None,
        )
        .await
        .expect("post_process must succeed");

        let quotes = &result.claims[0].quotes;
        assert!(
            !quotes.is_empty(),
            "revised claim must carry quotes after post-process"
        );
        // Fix A: unrelated quote must be stamped "absent", text unchanged, not snapped.
        assert_eq!(
            quotes[0].status.as_deref(),
            Some("absent"),
            "unrelated quote in revised output must carry status 'absent' — Fix A"
        );
        assert!(
            !quotes[0].snapped,
            "unrelated quote must not be snapped"
        );
        assert!(
            quotes[0].text.contains("Completely unrelated"),
            "absent quote text must be unchanged"
        );
    }

    // ── Fix B (probe-17 cause 1): candidate source strip ─────────────────────

    /// Under V2LitReview, post-process must strip candidate-pattern ids from
    /// claim.sources and drop quotes whose source matches the pattern.
    /// Under V1Delphi the same input passes through unchanged.
    #[tokio::test]
    async fn fix_b_candidate_sources_stripped_under_v2_only() {
        use crate::ttd::term_sheet::PromptProfile;

        // Synthesis with one real source and two candidate-pattern ids.
        let make_candidate_synthesis = || {
            let mut art = SynthesisArtifact::new("s", "r", "q", "model", "v2/lit-review");
            art.claims.push(Claim {
                text: "A claim about permafrost.".to_string(),
                agreement_level: None,
                node_refs: vec![],
                citation: None,
                support_level: Some("converging".to_string()),
                sources: vec![
                    "arxiv:2105.14103".to_string(),
                    "Candidate1".to_string(),
                    "candidate_3".to_string(),
                ],
                quotes: vec![
                    ClaimQuote {
                        source: "arxiv:2105.14103".to_string(),
                        text: "real quote".to_string(),
                        status: None,
                        snapped: false,
                        inherited: false,
                        node_id: None,
                    },
                    ClaimQuote {
                        source: "Candidate1".to_string(),
                        text: "candidate quote".to_string(),
                        status: None,
                        snapped: false,
                        inherited: false,
                        node_id: None,
                    },
                ],
                evidence_grade: None,
                method: None,
                year: None,
                lineage: None,
                counterarguments: vec![],
            });
            art
        };

        let panel = make_panel(&["arxiv:2105.14103"]);

        let executor_v2: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<synthesis><claims></claims><areas_of_agreement></areas_of_agreement><areas_of_disagreement></areas_of_disagreement><uncertainties></uncertainties></synthesis>".to_string(),
            });

        // V2 path: candidate sources should be stripped.
        let result_v2 = post_process_synthesis(
            make_candidate_synthesis(),
            &panel,
            &executor_v2,
            PromptProfile::V2LitReview,
            None,
        )
        .await
        .expect("v2 post_process must succeed");

        let claim_v2 = &result_v2.claims[0];
        assert!(
            !claim_v2.sources.contains(&"Candidate1".to_string()),
            "V2: 'Candidate1' must be stripped from sources"
        );
        assert!(
            !claim_v2.sources.contains(&"candidate_3".to_string()),
            "V2: 'candidate_3' must be stripped from sources"
        );
        assert!(
            claim_v2.sources.contains(&"arxiv:2105.14103".to_string()),
            "V2: real source must be preserved"
        );
        assert!(
            claim_v2.quotes.iter().all(|q| q.source != "Candidate1"),
            "V2: quote citing Candidate1 must be dropped"
        );
        assert!(
            claim_v2.quotes.iter().any(|q| q.source == "arxiv:2105.14103"),
            "V2: quote citing real source must be preserved"
        );

        // V1 path: nothing should be stripped.
        let executor_v1: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<quotes/>".to_string(),
            });

        let v1_synthesis = {
            let mut art = SynthesisArtifact::new("s", "r", "q", "model", "v1/synthesis");
            art.claims.push(Claim {
                text: "A claim.".to_string(),
                agreement_level: Some("consensus".to_string()),
                support_level: None,
                sources: vec![
                    "arxiv:2105.14103".to_string(),
                    "Candidate1".to_string(),
                ],
                quotes: vec![],
                node_refs: vec![],
                citation: None,
                evidence_grade: None,
                method: None,
                year: None,
                lineage: None,
                counterarguments: vec![],
            });
            art
        };

        let result_v1 = post_process_synthesis(
            v1_synthesis,
            &panel,
            &executor_v1,
            PromptProfile::V1Delphi,
            None,
        )
        .await
        .expect("v1 post_process must succeed");

        let claim_v1 = &result_v1.claims[0];
        assert!(
            claim_v1.sources.contains(&"Candidate1".to_string()),
            "V1: candidate source must NOT be stripped — v1 path unchanged"
        );
    }

    // ── F13: probe-18 regression — allowlist strip ────────────────────────────

    /// F13 (probe-18 regression): mutated labels (sN_candidateN) must be stripped
    /// from claim sources and their quotes dropped on the v2 path, while a real
    /// arxiv: source on the same claim survives (mix scenario). A quote citing a
    /// non-panel arxiv: source also survives (F11 preserved).
    #[tokio::test]
    async fn f13_probe18_allowlist_strip_v2() {
        use crate::ttd::artifact::ClaimQuote;
        use crate::ttd::term_sheet::PromptProfile;

        // Synthesis: one claim with a real source + two probe-18 mutated labels.
        // Two quotes: one citing the real source, one citing a mutated label.
        let mut art = SynthesisArtifact::new("s", "r", "q", "model", "v2/lit-review");
        art.claims.push(Claim {
            text: "Permafrost claim with mixed provenance.".to_string(),
            agreement_level: None,
            node_refs: vec![],
            citation: None,
            support_level: Some("converging".to_string()),
            sources: vec![
                "arxiv:2502.12110".to_string(),
                "s1_candidate1".to_string(),
                "s3_candidate3".to_string(),
            ],
            quotes: vec![
                ClaimQuote {
                    source: "arxiv:2502.12110".to_string(),
                    text: "real quote text".to_string(),
                    status: None,
                    snapped: false,
                    inherited: false,
                    node_id: None,
                },
                ClaimQuote {
                    source: "s5_candidate5".to_string(),
                    text: "mutated label quote".to_string(),
                    status: None,
                    snapped: false,
                    inherited: false,
                    node_id: None,
                },
            ],
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            counterarguments: vec![],
        });

        // Panel does not include the mutated labels (they are not panel members).
        let panel = make_panel(&["arxiv:2502.12110"]);
        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<synthesis><claims></claims><areas_of_agreement></areas_of_agreement><areas_of_disagreement></areas_of_disagreement><uncertainties></uncertainties></synthesis>".to_string(),
            });

        let result = post_process_synthesis(art, &panel, &executor, PromptProfile::V2LitReview, None)
            .await
            .expect("v2 post_process must succeed");

        let claim = &result.claims[0];

        // Real arxiv: source must survive.
        assert!(
            claim.sources.contains(&"arxiv:2502.12110".to_string()),
            "arxiv:2502.12110 must survive allowlist strip"
        );

        // Mutated labels must be stripped from sources.
        assert!(
            !claim.sources.contains(&"s1_candidate1".to_string()),
            "s1_candidate1 must be stripped from sources"
        );
        assert!(
            !claim.sources.contains(&"s3_candidate3".to_string()),
            "s3_candidate3 must be stripped from sources"
        );

        // Quote citing the mutated label must be dropped.
        assert!(
            !claim.quotes.iter().any(|q| q.source == "s5_candidate5"),
            "quote citing s5_candidate5 must be dropped"
        );

        // Quote citing the real source must survive (F11).
        assert!(
            claim.quotes.iter().any(|q| q.source == "arxiv:2502.12110"),
            "quote citing arxiv:2502.12110 must survive"
        );
    }

    /// F13: a panel id that is not arxiv:/s2:-shaped survives via panel membership.
    #[tokio::test]
    async fn f13_panel_id_non_arxiv_survives_strip() {
        use crate::ttd::term_sheet::PromptProfile;

        // Panel expert with an opaque id.
        let panel = make_panel(&["pmc_case_study"]);

        let mut art = SynthesisArtifact::new("s", "r", "q", "model", "v2/lit-review");
        art.claims.push(Claim {
            text: "Claim citing a panel member by opaque id.".to_string(),
            agreement_level: None,
            support_level: None,
            sources: vec!["pmc_case_study".to_string()],
            quotes: vec![],
            node_refs: vec![],
            citation: None,
            evidence_grade: None,
            method: None,
            year: None,
            lineage: None,
            counterarguments: vec![],
        });

        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<synthesis><claims></claims><areas_of_agreement></areas_of_agreement><areas_of_disagreement></areas_of_disagreement><uncertainties></uncertainties></synthesis>".to_string(),
            });

        let result = post_process_synthesis(art, &panel, &executor, PromptProfile::V2LitReview, None)
            .await
            .expect("post_process must succeed");

        assert!(
            result.claims[0].sources.contains(&"pmc_case_study".to_string()),
            "panel member 'pmc_case_study' must survive allowlist strip via membership"
        );
    }

    // ── Fix C (probe-17 cause 2): inherit_graph_quotes ────────────────────────

    use crate::ttd::artifact::{ArgumentationGraph, GraphNode};

    fn make_graph_with_node(
        expert_id: &str,
        claim_text: &str,
        quote_text: &str,
        verification_status: Option<&str>,
    ) -> ArgumentationGraph {
        let mut graph = ArgumentationGraph::new("s", "r", "q", "model", "v2/lit-review");
        graph.nodes.push(GraphNode {
            id: format!("{expert_id}_n1"),
            claim: claim_text.to_string(),
            expert_id: expert_id.to_string(),
            quote: Some(quote_text.to_string()),
            verification_status: verification_status.map(|s| s.to_string()),
        });
        graph
    }

    /// A v2 claim citing source S with a paraphrased/absent model quote, where
    /// the graph has a verified-quoted node for S whose claim text overlaps the
    /// synthesis claim text: after post-process the claim carries an inherited
    /// quote with status "verified" and inherited=true. The model's own failing
    /// quote keeps its honest status.
    #[tokio::test]
    async fn fix_c_inherits_verified_graph_quote_onto_synthesis_claim() {
        use crate::ttd::term_sheet::PromptProfile;

        // Panel prose contains the graph node quote (for re-verification).
        let graph_quote = "Permafrost thaw releases measurable methane flux.";
        let panel_prose = format!("{graph_quote} Additional context follows here.");

        let mut panel = make_panel(&["arxiv:s1.001"]);
        panel[0].prose = panel_prose.clone();

        // Synthesis: a claim about permafrost (overlaps the graph node claim)
        // with one paraphrased quote (will stay paraphrased/absent).
        let mut synthesis = make_synthesis(&[(
            "Permafrost thaw releases methane at measurable flux rates.",
            "converging",
            vec!["arxiv:s1.001"],
        )]);
        synthesis.claims[0].support_level = Some("converging".to_string());
        synthesis.claims[0].quotes = vec![ClaimQuote {
            source: "arxiv:s1.001".to_string(),
            text: "something about thaw processes globally".to_string(), // paraphrase
            status: None,
            snapped: false,
            inherited: false,
            node_id: None,
        }];

        // Graph: a node for s1 citing the same expert, verified quote, claim text
        // overlaps the synthesis claim (many shared tokens).
        let graph = make_graph_with_node(
            "arxiv:s1.001",
            "permafrost thaw releases methane flux at measurable rates",
            graph_quote,
            Some("verified"),
        );

        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<synthesis><claims></claims><areas_of_agreement></areas_of_agreement><areas_of_disagreement></areas_of_disagreement><uncertainties></uncertainties></synthesis>".to_string(),
            });

        let result = post_process_synthesis_with_graph(
            synthesis,
            &panel,
            &executor,
            PromptProfile::V2LitReview,
            None,
            Some(&graph),
        )
        .await
        .expect("post_process must succeed");

        let quotes = &result.claims[0].quotes;

        // The inherited quote must be present.
        let inherited_quote = quotes.iter().find(|q| q.inherited);
        assert!(
            inherited_quote.is_some(),
            "Fix C: an inherited quote must be attached; quotes: {quotes:?}"
        );
        let iq = inherited_quote.unwrap();
        assert_eq!(
            iq.source, "arxiv:s1.001",
            "inherited quote source must match the graph node's expert_id"
        );
        assert_eq!(
            iq.text, graph_quote,
            "inherited quote text must equal the graph node's quote"
        );
        assert_eq!(
            iq.status.as_deref(),
            Some("verified"),
            "inherited quote must be re-verified as 'verified' against stored prose"
        );
        assert!(iq.inherited, "inherited quote must carry inherited=true");

        // The model's own paraphrased quote must retain its honest status.
        let model_quote = quotes.iter().find(|q| !q.inherited);
        assert!(
            model_quote.is_some(),
            "model's own quote must still be present"
        );
        let mq = model_quote.unwrap();
        assert!(
            mq.status.as_deref() != Some("verified") || mq.snapped,
            "model paraphrase must not be re-labelled 'verified' unless snapped"
        );
    }

    /// A node whose claim text shares too few tokens with the synthesis claim
    /// is NOT inherited. A node with verification_status absent/None is NOT
    /// inherited.
    #[tokio::test]
    async fn fix_c_does_not_inherit_low_overlap_or_unverified_node() {
        use crate::ttd::term_sheet::PromptProfile;

        let graph_quote = "Completely unrelated sentence about something else.";
        let panel_prose = format!("{graph_quote} More stuff.");

        let mut panel = make_panel(&["arxiv:s1.001"]);
        panel[0].prose = panel_prose.clone();

        let mut synthesis = make_synthesis(&[(
            "Permafrost thaw releases methane.",
            "converging",
            vec!["arxiv:s1.001"],
        )]);
        synthesis.claims[0].support_level = Some("converging".to_string());
        // No quotes at all — relies on inheritance.
        synthesis.claims[0].quotes = vec![];

        // Graph node: claim text does NOT overlap synthesis claim; "absent" status.
        let graph = make_graph_with_node(
            "arxiv:s1.001",
            "unrelated claim about something completely different elsewhere",
            graph_quote,
            Some("absent"),
        );

        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<synthesis><claims></claims><areas_of_agreement></areas_of_agreement><areas_of_disagreement></areas_of_disagreement><uncertainties></uncertainties></synthesis>".to_string(),
            });

        let result = post_process_synthesis_with_graph(
            synthesis,
            &panel,
            &executor,
            PromptProfile::V2LitReview,
            None,
            Some(&graph),
        )
        .await
        .expect("post_process must succeed");

        let quotes = &result.claims[0].quotes;
        assert!(
            quotes.iter().all(|q| !q.inherited),
            "low-overlap / absent-status node must not be inherited; quotes: {quotes:?}"
        );
    }

    /// An inherited quote whose source the resolver cannot resolve stamps "absent"
    /// (the graph stamp is never trusted transitively).
    #[tokio::test]
    async fn fix_c_inherited_quote_stamps_absent_when_resolver_cannot_resolve() {
        use crate::ttd::term_sheet::PromptProfile;

        // Panel has arxiv:s1.001 but the graph node expert_id is "arxiv:s2.999"
        // (not in panel, not in resolver) — forces absent stamp.
        let panel = make_panel(&["arxiv:s1.001"]);

        let mut synthesis = make_synthesis(&[(
            "Permafrost thaw releases methane.",
            "converging",
            vec!["arxiv:s1.001", "arxiv:s2.999"],
        )]);
        synthesis.claims[0].support_level = Some("converging".to_string());
        synthesis.claims[0].quotes = vec![];

        // Graph node for s2.999 — verified in the graph, but panel has no text
        // for it and there is no resolver. The inherited quote cannot be confirmed.
        let graph_quote = "methane flux observed in permafrost thaw studies consistently.";
        let graph = make_graph_with_node(
            "arxiv:s2.999",
            "permafrost thaw methane flux observed in consistent studies",
            graph_quote,
            Some("verified"),
        );

        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<synthesis><claims></claims><areas_of_agreement></areas_of_agreement><areas_of_disagreement></areas_of_disagreement><uncertainties></uncertainties></synthesis>".to_string(),
            });

        let result = post_process_synthesis_with_graph(
            synthesis,
            &panel,
            &executor,
            PromptProfile::V2LitReview,
            None,
            Some(&graph),
        )
        .await
        .expect("post_process must succeed");

        let quotes = &result.claims[0].quotes;
        let inherited_quote = quotes.iter().find(|q| q.inherited);
        // The inherited quote is attached (overlap check passes), but must stamp
        // "absent" because the stored prose is unresolvable — graph stamp never
        // trusted transitively.
        if let Some(iq) = inherited_quote {
            assert_eq!(
                iq.status.as_deref(),
                Some("absent"),
                "unresolvable source must stamp 'absent', not carry the graph's 'verified'"
            );
        }
        // If no inherited quote was attached (threshold not met) that's also acceptable —
        // the key constraint is no transitive trust of the graph stamp.
        for q in quotes.iter() {
            assert_ne!(
                q.status.as_deref(),
                Some("verified"),
                "no quote should be 'verified' when its source cannot be resolved"
            );
        }
    }

    /// Serde: inherited=false does not appear in serialized output (v1 byte-identical).
    #[test]
    fn fix_c_inherited_false_not_serialized() {
        let quote = ClaimQuote {
            source: "arxiv:2105.14103".to_string(),
            text: "some verbatim text".to_string(),
            status: Some("verified".to_string()),
            snapped: false,
            inherited: false,
            node_id: None,
        };
        let yaml = serde_yaml::to_string(&quote).unwrap();
        assert!(
            !yaml.contains("inherited"),
            "inherited=false must not appear in serialized YAML — v1 output byte-identical; got: {yaml}"
        );
    }

    /// F14 floor: a claim that carries `node_refs` but no verified quote gets the
    /// cited node's stored verified quote attached by EXACT id — even when the
    /// node's claim text shares no tokens with the synthesis claim (the case the
    /// ≥0.5 token-containment `inherit_graph_quotes` would starve on). Re-verify
    /// stamps it "verified" against panel prose.
    #[tokio::test]
    async fn f14_floor_attaches_cited_node_quote_by_exact_id() {
        use crate::ttd::term_sheet::PromptProfile;

        let node_quote = "Permafrost thaw releases measurable methane flux.";
        let panel_prose = format!("{node_quote} Additional context follows here.");

        let mut panel = make_panel(&["arxiv:s1.001"]);
        panel[0].prose = panel_prose;

        // Synthesis claim text deliberately shares NO content tokens with the
        // graph node's claim text — token-containment inheritance cannot fire.
        let mut synthesis = make_synthesis(&[(
            "Coastal erosion accelerates under rising seas.",
            "converging",
            vec!["arxiv:s1.001"],
        )]);
        synthesis.claims[0].support_level = Some("converging".to_string());
        synthesis.claims[0].quotes = vec![]; // merger authored none
        // The claim cites the node by exact id.
        synthesis.claims[0].node_refs = vec!["arxiv:s1.001_n1".to_string()];

        // Node claim text is unrelated to the synthesis claim — proves the floor
        // attaches by id, not by token overlap.
        let graph = make_graph_with_node(
            "arxiv:s1.001",
            "wholly unrelated statement about glacial isostatic rebound",
            node_quote,
            Some("verified"),
        );

        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<synthesis><claims></claims><areas_of_agreement></areas_of_agreement><areas_of_disagreement></areas_of_disagreement><uncertainties></uncertainties></synthesis>".to_string(),
            });

        let result = post_process_synthesis_with_graph(
            synthesis,
            &panel,
            &executor,
            PromptProfile::V2LitReview,
            None,
            Some(&graph),
        )
        .await
        .expect("post_process must succeed");

        let quotes = &result.claims[0].quotes;
        let attached = quotes
            .iter()
            .find(|q| q.node_id.as_deref() == Some("arxiv:s1.001_n1"))
            .expect("F14 floor must attach the cited node's quote by exact id");
        assert_eq!(attached.source, "arxiv:s1.001");
        assert_eq!(attached.text, node_quote);
        assert!(attached.inherited, "floor quote must carry inherited=true");
        assert_eq!(
            attached.status.as_deref(),
            Some("verified"),
            "floor quote must be re-verified against stored prose, not trusted transitively"
        );
    }

    /// F14 floor is a floor, not enrichment: a claim that already holds a
    /// verified quote is left untouched (no extra cited-node quote attached).
    #[tokio::test]
    async fn f14_floor_skips_claim_with_existing_verified_quote() {
        use crate::ttd::term_sheet::PromptProfile;

        let merger_quote = "Permafrost thaw releases measurable methane flux.";
        let panel_prose = format!("{merger_quote} More context.");

        let mut panel = make_panel(&["arxiv:s1.001"]);
        panel[0].prose = panel_prose;

        let mut synthesis = make_synthesis(&[(
            "Permafrost thaw drives methane release.",
            "converging",
            vec!["arxiv:s1.001"],
        )]);
        synthesis.claims[0].support_level = Some("converging".to_string());
        // Merger already authored a verbatim quote tagged with the node id.
        synthesis.claims[0].quotes = vec![ClaimQuote {
            source: "arxiv:s1.001".to_string(),
            text: merger_quote.to_string(),
            status: None, // stamped verified by the step-5 verify pass
            snapped: false,
            inherited: false,
            node_id: Some("arxiv:s1.001_n1".to_string()),
        }];
        synthesis.claims[0].node_refs = vec!["arxiv:s1.001_n1".to_string()];

        let graph = make_graph_with_node(
            "arxiv:s1.001",
            "permafrost thaw releases methane",
            merger_quote,
            Some("verified"),
        );

        let executor: Arc<dyn crate::executor::AgentExecutor> =
            Arc::new(RecordingExecutor {
                invocations: Arc::new(std::sync::Mutex::new(vec![])),
                response: "<synthesis><claims></claims><areas_of_agreement></areas_of_agreement><areas_of_disagreement></areas_of_disagreement><uncertainties></uncertainties></synthesis>".to_string(),
            });

        let result = post_process_synthesis_with_graph(
            synthesis,
            &panel,
            &executor,
            PromptProfile::V2LitReview,
            None,
            Some(&graph),
        )
        .await
        .expect("post_process must succeed");

        let quotes = &result.claims[0].quotes;
        assert_eq!(
            quotes.len(),
            1,
            "floor must not attach a second quote when a verified one exists; quotes: {quotes:?}"
        );
        assert!(
            !quotes[0].inherited,
            "the merger's own quote must be preserved, not replaced by a floor quote"
        );
    }
}
