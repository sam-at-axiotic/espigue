//! The synthesis pipeline, lifted out of the daemon handler.
//!
//! Port of `alzina-daemon/src/api/synthesize.rs` with every `state.X` rebind to
//! `ctx.X` ([`LitContext`]). Daemon-only concerns (sessions, auth, the event
//! bus, SSE, the `chat_service` guard) are dropped; `run_id` becomes a generated
//! UUID. The retrieval/fusion/gate/engine logic is otherwise unchanged so the
//! eval-validated behaviour carries over verbatim.
//!
//! The two `run_three_lane_fusion` call sites (initial panel + per-gap closures)
//! are preserved, as is the loud-degrade contract on every lane.

use std::sync::Arc;

use base::EmbeddingTask;
use orchestration::{
    ttd::{
        citations::{apply_author_year_citations, PaperMeta},
        engine::{run_engine_with_bib, EngineConfig},
        retrieval::{CallbackLitSearch, LitRetriever, RetrievalPolicy},
        term_sheet::PromptProfile,
    },
    AgentExecutor,
};
use search::{BibliographyStore, FusedHit, SqliteBibliographyStore};

use crate::context::LitContext;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Hard server-side cap on `top_k`.
const TOP_K_HARD_CAP: usize = 50;
/// Default `top_k`.
pub const DEFAULT_TOP_K: usize = 10;
/// Default generation model for the TTD stages (OpenRouter-shaped slug).
pub const DEFAULT_MODEL: &str = "google/gemini-2.5-flash";
/// Default v2/v3 Stage-2 merger model (provider-shaped Opus slug). Mirrors the
/// daemon's `claude-opus-4-8` pin, but in OpenRouter form (the bare daemon slug
/// 400s on OpenRouter).
pub const DEFAULT_MERGER_MODEL: &str = "anthropic/claude-opus-4.8";
/// Agent-id stamped on TTD executor spawns.
const TTD_AGENT_ID: &str = "alzina-ttd";
/// Cap on the stage-2 refreshed panel (one provenance header line per member).
const REFRESH_PANEL_CAP: usize = 30;
/// Ceiling (seconds) on waiting for initial-panel full-text promotion.
const FULLTEXT_WAIT_SECS: u64 = 180;

/// Resolve the full-text promotion wait: `ALZINA_TTD_FULLTEXT_WAIT_SECS`, else
/// [`FULLTEXT_WAIT_SECS`]; `0` restores fire-and-forget.
fn fulltext_wait_secs() -> u64 {
    std::env::var("ALZINA_TTD_FULLTEXT_WAIT_SECS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(FULLTEXT_WAIT_SECS)
}

/// Await background full-text promotion handles, bounded by [`fulltext_wait_secs`].
/// A timeout leaves the still-running promotions to finish in the background; the
/// panel grounds on whatever reached `indexed` in time (loud `timed_out` log).
async fn await_promotions(handles: Vec<tokio::task::JoinHandle<()>>, run_id: &str) {
    if handles.is_empty() {
        return;
    }
    let wait_secs = fulltext_wait_secs();
    let n_promotions = handles.len();
    let start = std::time::Instant::now();
    let waited = tokio::time::timeout(std::time::Duration::from_secs(wait_secs), async {
        for handle in handles {
            let _ = handle.await;
        }
    })
    .await;
    tracing::info!(
        target: "ttd_perf",
        run_id = %run_id,
        n_promotions,
        timed_out = waited.is_err(),
        duration_ms = start.elapsed().as_millis() as u64,
        "ttd_perf: full-text promotion wait"
    );
}

/// Plan tournament (long-form bookends). Unset → enabled; `0`/`false`/`no`/`off`
/// disables and restores the plan-less narrative path.
fn plan_tournament_enabled() -> bool {
    std::env::var("ALZINA_TTD_PLAN_TOURNAMENT")
        .ok()
        .map(|s| !matches!(s.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

/// Lever B: cross-encoder reranker toggle. Unset → enabled (when a reranker is
/// configured); `0`/`false`/`no`/`off` keeps pure RRF order.
fn rerank_enabled() -> bool {
    std::env::var("ALZINA_RERANK")
        .ok()
        .map(|s| !matches!(s.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

/// Warm-internal-lane toggle. Unset → enabled: the lit-corpus kNN runs under the
/// Live policy too, surfacing the local back-catalogue the recent-biased
/// arxiv/S2 lanes miss. `0`/`false`/`no`/`off` restores the cold behaviour.
fn warm_internal_lane_enabled() -> bool {
    std::env::var("ALZINA_WARM_INTERNAL_LANE")
        .ok()
        .map(|s| !matches!(s.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

/// Lever A: topicality gate toggle. Unset → enabled; `0`/`false`/`no`/`off`
/// keeps every retrieved source.
fn topicality_enabled() -> bool {
    std::env::var("ALZINA_TOPICALITY")
        .ok()
        .map(|s| !matches!(s.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

/// Fold two optional degradation reasons with `"; "`.
fn fold_reason(a: Option<String>, b: Option<String>) -> Option<String> {
    match (a, b) {
        (Some(a), Some(b)) => Some(format!("{a}; {b}")),
        (a, b) => a.or(b),
    }
}

// ── Public surface ──────────────────────────────────────────────────────────

/// Retrieval scope for a review run.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Local corpus only — skip the live arxiv/S2 lanes and Stage-0 exploration.
    /// Reviews cite only ingested local docs.
    CorpusOnly,
    /// Local corpus plus the live web lanes (arxiv + S2). The default.
    CorpusPlusWeb,
}

/// Knobs for one review run.
pub struct ReviewOptions {
    /// Sources retrieved per lane (clamped to 50).
    pub top_k: usize,
    /// Prompt/schema dialect.
    pub profile: PromptProfile,
    /// Generation model slug for the TTD stages.
    pub model: String,
    /// v2/v3 Stage-2 merger model slug (the hard quote-authoring step). `None`
    /// keeps the engine default (the bare daemon slug, which 400s on OpenRouter)
    /// — set this to a provider-shaped Opus slug for the standalone CLI.
    pub merger_model: Option<String>,
    /// Retrieval scope (local-only vs local+web).
    pub scope: Scope,
    /// Seed papers (arXiv id / DOI / S2 id). Empty = normal three-lane fusion.
    /// Non-empty builds the panel directly from these papers, skipping Stage-0,
    /// the initial fusion, and the topicality gate; Stage-1/2 gap-fill still
    /// runs and honours [`scope`](Self::scope).
    pub seed_papers: Vec<String>,
}

impl Default for ReviewOptions {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            profile: PromptProfile::V3LitReviewLong,
            model: DEFAULT_MODEL.to_string(),
            merger_model: Some(DEFAULT_MERGER_MODEL.to_string()),
            scope: Scope::CorpusPlusWeb,
            seed_papers: Vec::new(),
        }
    }
}

/// Outcome of a review run.
pub struct ReviewResult {
    /// YAML-serialised `SynthesisArtifact` with full provenance.
    pub synthesis_yaml: String,
    /// Markdown rendering of the Stage-1 argumentation graph.
    pub graph_markdown: String,
    /// Generated run identifier (UUID).
    pub run_id: String,
    /// Deduped rows written to `synthesis_bibliography` for this run.
    pub bib_count: usize,
    /// Final narrative text (empty on degraded paths that never synthesised).
    pub narrative: String,
    /// When true, the result is a partial/degraded synthesis.
    pub degraded: bool,
    /// Human-readable degradation notice (empty when `degraded == false`).
    pub notice: String,
}

/// Parse the optional prompt-profile string (strict allow-list).
///
/// `None` → `V3LitReviewLong` (the default sectioned long-form lit review).
pub fn parse_prompt_profile(raw: Option<&str>) -> Result<PromptProfile, String> {
    match raw {
        None => Ok(PromptProfile::V3LitReviewLong),
        Some(s) => match s.trim().to_ascii_lowercase().as_str() {
            "v1/delphi" => Ok(PromptProfile::V1Delphi),
            "v2/lit-review" => Ok(PromptProfile::V2LitReview),
            "v3/lit-review-long" => Ok(PromptProfile::V3LitReviewLong),
            other => Err(format!(
                "unknown prompt_profile '{other}'; accepted: v1/delphi, v2/lit-review, \
                 v3/lit-review-long"
            )),
        },
    }
}

// ── Seed-query decomposition ───────────────────────────────────────────────

/// Decompose the research question into 3-6 short literature search queries via
/// one executor spawn. Falls back to the raw question on any failure (loudly).
async fn decompose_seed_queries(
    executor: &Arc<dyn AgentExecutor>,
    question: &str,
    model: &str,
) -> Vec<String> {
    use base::identity::AgentId;

    let prompt = format!(
        "Decompose this research question into short literature search queries for \
         Semantic Scholar / arXiv. Produce TWO kinds:\n\
         1. FACET queries (4 to 6): 2-6 word phrases covering the question's distinct \
         facets, in the question's own terminology.\n\
         2. FOUNDATIONAL queries (1 to 2): the classical or foundational work the \
         subject is built on. Use the general, field-standard terminology and DROP \
         the modern framing words — for a question about multi-agent LLMs, omit \
         'LLM'/'agent' and name the underlying theory directly (e.g. 'Byzantine fault \
         tolerance consensus', 'Condorcet jury theorem', 'ensemble voting methods'). \
         These surface the foundational literature that the modern-framed queries miss.\n\
         Output ONLY the queries, one per line — no numbering, no group labels, no \
         bullets, no commentary.\n\n\
         Research question: {question}"
    );

    let started = std::time::Instant::now();
    match executor
        .execute(&AgentId::new(TTD_AGENT_ID), &prompt, model, "seed_query_decompose")
        .await
    {
        Ok(raw) => {
            let mut seen = std::collections::HashSet::new();
            let queries: Vec<String> = raw
                .lines()
                .map(|l| {
                    l.trim()
                        .trim_start_matches(['-', '*', '•'])
                        .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')')
                        .trim()
                        .trim_matches('"')
                        .to_string()
                })
                .filter(|l| !l.is_empty() && l.len() <= 80 && l.split_whitespace().count() <= 8)
                .filter(|l| seen.insert(l.to_lowercase()))
                .take(8)
                .collect();
            if queries.is_empty() {
                tracing::warn!(
                    "seed query decomposition returned nothing usable — \
                     falling back to the raw question"
                );
                vec![question.to_string()]
            } else {
                tracing::info!(
                    target: "ttd_perf",
                    n_queries = queries.len(),
                    queries = ?queries,
                    duration_ms = started.elapsed().as_millis() as u64,
                    "ttd_perf: seed query decomposition"
                );
                queries
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "seed query decomposition spawn failed — falling back to the raw question"
            );
            vec![question.to_string()]
        }
    }
}

// ── Topicality gate (Lever A) ──────────────────────────────────────────────

/// Parse the on-topic indices a gate reply names: every in-range integer.
fn parse_keep_indices(raw: &str, n: usize) -> std::collections::BTreeSet<usize> {
    raw.split(|c: char| !c.is_ascii_digit())
        .filter_map(|t| t.parse::<usize>().ok())
        .filter(|i| *i < n)
        .collect()
}

/// LLM topicality gate over the fused candidates (binary drop).
///
/// Loud-degrade: a spawn failure, an unparseable reply, or a verdict that would
/// empty the panel keeps EVERY candidate and returns a notice. Returns
/// `(survivors, Option<notice>)`.
async fn topicality_gate(
    executor: &Arc<dyn AgentExecutor>,
    question: &str,
    hits: Vec<FusedHit>,
    model: &str,
) -> (Vec<FusedHit>, Option<String>) {
    use base::identity::AgentId;

    if hits.len() < 2 {
        return (hits, None);
    }
    let n = hits.len();

    let listing = hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let preview: String = h.content_preview.chars().take(300).collect();
            format!("[{i}] {}\n{}", h.title, preview)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        "You are screening retrieved papers for topical relevance to a research \
         question.\n\
         A paper is ON-TOPIC if it studies the question's actual subject. It is \
         ALSO on-topic if it supplies the FOUNDATIONAL theory the subject is built \
         on — even when it predates the question's modern framing or comes from the \
         classical/adjacent field the subject inherits from (for a question about \
         consensus in multi-agent LLM systems, a classical Byzantine-fault-tolerance \
         or voting-theory paper IS on-topic foundational grounding, keep it).\n\
         A paper is OFF-TOPIC only when it shares surface vocabulary but belongs to \
         an unrelated APPLICATION domain — for example, for that same question, a \
         multi-agent-debate paper about phishing detection, video forensics, urban \
         prediction, or agriculture is OFF-TOPIC. When unsure between foundational \
         grounding and wrong-domain, keep the paper.\n\n\
         Research question:\n{question}\n\n\
         Candidates:\n{listing}\n\n\
         List the indices of the ON-TOPIC papers only, comma-separated (for example \
         `0, 2, 5`). If every candidate is on-topic, list them all. Output ONLY the \
         indices — no prose."
    );

    let started = std::time::Instant::now();
    let raw = match executor
        .execute(&AgentId::new(TTD_AGENT_ID), &prompt, model, "topicality_gate")
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "topicality gate spawn failed — keeping all candidates");
            return (
                hits,
                Some(format!("topicality gate unavailable: {e} — all sources kept")),
            );
        }
    };

    let keep = parse_keep_indices(&raw, n);

    if keep.is_empty() {
        tracing::warn!(
            target: "ttd_perf",
            n,
            duration_ms = started.elapsed().as_millis() as u64,
            "topicality gate named no on-topic candidates — keeping all (loud-degrade)"
        );
        return (
            hits,
            Some(format!(
                "topicality gate flagged all {n} candidates off-topic or did not parse \
                 — kept all (retrieval may be off-target)"
            )),
        );
    }
    if keep.len() == n {
        tracing::info!(
            target: "ttd_perf",
            n,
            duration_ms = started.elapsed().as_millis() as u64,
            "ttd_perf: topicality gate — all candidates on-topic, no drops"
        );
        return (hits, None);
    }

    let dropped = n - keep.len();
    let kept_hits: Vec<FusedHit> = hits
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, h)| h)
        .collect();
    tracing::warn!(
        target: "ttd_perf",
        dropped,
        kept = kept_hits.len(),
        duration_ms = started.elapsed().as_millis() as u64,
        "ttd_perf: topicality gate dropped {dropped} off-topic candidate(s)"
    );
    (
        kept_hits,
        Some(format!("topicality gate dropped {dropped} off-topic source(s)")),
    )
}

// ── Stage-2 panel refresher ────────────────────────────────────────────────

/// Rebuilds the stage-2 panel via `build_panel` with synthetic hits (a pure SQL
/// read of locally-indexed gap-fill papers). Ids unknown to the lit DB produce
/// empty prose and are dropped.
struct LitPanelRefresher {
    pool: sqlx::SqlitePool,
}

#[async_trait::async_trait]
impl orchestration::ttd::engine::PanelRefresher for LitPanelRefresher {
    async fn refresh(
        &self,
        source_ids: &[String],
    ) -> Result<Vec<orchestration::adapter::ExpertResponse>, String> {
        let source_ids: Vec<String> = source_ids
            .iter()
            .filter(|id| !id.trim().is_empty())
            .cloned()
            .collect();
        let ids: &[String] = if source_ids.len() > REFRESH_PANEL_CAP {
            tracing::warn!(
                requested = source_ids.len(),
                cap = REFRESH_PANEL_CAP,
                "stage-2 panel refresh truncated to cap — dropping latest graph-only ids"
            );
            &source_ids[..REFRESH_PANEL_CAP]
        } else {
            &source_ids[..]
        };

        let hits: Vec<FusedHit> = ids
            .iter()
            .map(|id| FusedHit {
                source_type: if id.starts_with("s2:") {
                    "s2".into()
                } else if id.starts_with("arxiv:") {
                    "arxiv".into()
                } else {
                    "internal".into()
                },
                source_id: id.clone(),
                title: String::new(),
                section: None,
                content: String::new(),
                content_preview: String::new(),
                relevance: 0.0,
            })
            .collect();

        let panel = orchestration::adapter::build_panel(hits, &self.pool)
            .await
            .map_err(|e| e.to_string())?;

        let n_before = panel.len();
        let panel: Vec<_> = panel
            .into_iter()
            .filter(|e| !e.prose.trim().is_empty())
            .collect();
        if panel.len() < n_before {
            tracing::warn!(
                dropped = n_before - panel.len(),
                kept = panel.len(),
                "stage-2 panel refresh: dropped ids with no local text"
            );
        }
        Ok(panel)
    }
}

// ── Three-lane fusion ──────────────────────────────────────────────────────

/// Run the three-lane RRF fusion (internal + arxiv + S2) against `ctx`.
///
/// Returns `(hits, degradation_notice, promotion_handles)`. The handles hold the
/// background full-text promotion spawns (empty under LocalOnly). The initial
/// call site awaits them (bounded) so `build_panel` sees full text; per-gap
/// closures drop them (fire-and-forget).
async fn run_three_lane_fusion(
    ctx: &LitContext,
    query: &str,
    top_k: usize,
    policy: RetrievalPolicy,
) -> (Vec<FusedHit>, Option<String>, Vec<tokio::task::JoinHandle<()>>) {
    use base::{VectorFilters, VectorStore};

    let lit_store = &ctx.lit_store;
    let s2_client = &ctx.s2_client;
    let embedder = &ctx.embedder;
    let gateway = &ctx.gateway;

    let top_k = top_k.clamp(1, TOP_K_HARD_CAP);
    let fusion_start = std::time::Instant::now();

    // Embed query once (shared across arxiv + S2 cosine ranking).
    let embed_start = std::time::Instant::now();
    let query_vec = match embedder.embed(query, EmbeddingTask::Query).await {
        Ok(v) => v,
        Err(e) => {
            let reason = format!("query embedding failed: {e}");
            tracing::warn!(query = %query, error = %e, "synthesize: query embed failed");
            return (vec![], Some(reason), vec![]);
        }
    };
    tracing::info!(
        target: "ttd_perf",
        duration_ms = embed_start.elapsed().as_millis() as u64,
        "ttd_perf: fusion query embed"
    );

    if policy == RetrievalPolicy::LocalOnly {
        tracing::info!(
            target: "ttd_perf",
            policy = "local_only",
            "ttd_perf: live arxiv/S2 lanes scoped out by retrieval policy — internal lane only"
        );
    }

    let arxiv_start = std::time::Instant::now();
    let s2_start = std::time::Instant::now();
    let internal_start = std::time::Instant::now();

    let (
        (arxiv_hits, arxiv_degraded),
        (s2_hits, s2_degraded),
        (internal_hits, internal_degraded),
    ) = tokio::join!(
        // ── arxiv lane ──────────────────────────────────────────────────────
        async {
            if policy == RetrievalPolicy::LocalOnly {
                (vec![], None)
            } else {
                run_arxiv_lane_for_synthesize(
                    query,
                    &query_vec,
                    lit_store,
                    embedder.as_ref(),
                    Some(ctx.lit_pool.as_ref()),
                    gateway,
                )
                .await
            }
        },
        // ── S2 lane ─────────────────────────────────────────────────────────
        async {
            if policy == RetrievalPolicy::LocalOnly {
                (vec![], None)
            } else {
                match gateway.acquire(search::Endpoint::S2).await {
                    search::Acquire::Proceed => {
                        run_s2_lane_for_synthesize(
                            query,
                            s2_client,
                            Some(ctx.lit_pool.as_ref()),
                            Some(lit_store.as_ref()),
                            Some(embedder.as_ref()),
                        )
                        .await
                    }
                    search::Acquire::BudgetExhausted => (
                        vec![],
                        Some("S2 per-run call budget exhausted — lane degraded to local".into()),
                    ),
                }
            }
        },
        // ── Lit-corpus kNN lane ─────────────────────────────────────────────
        async {
            if policy == RetrievalPolicy::LocalOnly || warm_internal_lane_enabled() {
                match lit_store
                    .search(&query_vec, top_k, &VectorFilters::default())
                    .await
                {
                    Ok(vec_hits) => {
                        let hits: Vec<base::SearchResultHit> = vec_hits
                            .into_iter()
                            .map(|h| base::SearchResultHit {
                                source_type: h.metadata.source_type.clone(),
                                source_id: h.metadata.source_id.clone(),
                                source_agent: h.metadata.source_agent.clone(),
                                source_date: h.metadata.source_date.clone(),
                                domain: h.metadata.domain.clone(),
                                content: h.metadata.content_preview.clone(),
                                content_preview: h.metadata.content_preview,
                                relevance: h.similarity,
                            })
                            .collect();
                        (hits, None)
                    }
                    Err(e) => {
                        let reason = format!("lit kNN lane failed: {e}");
                        tracing::warn!(query = %query, error = %e, "synthesize: lit kNN lane error");
                        (vec![], Some(reason))
                    }
                }
            } else {
                (vec![], None)
            }
        }
    );

    tracing::info!(
        target: "ttd_perf",
        n_hits = arxiv_hits.len(),
        degraded = arxiv_degraded.is_some(),
        duration_ms = arxiv_start.elapsed().as_millis() as u64,
        "ttd_perf: fusion arxiv lane (search + fetch + embed + persist)"
    );
    tracing::info!(
        target: "ttd_perf",
        n_hits = s2_hits.len(),
        degraded = s2_degraded.is_some(),
        duration_ms = s2_start.elapsed().as_millis() as u64,
        "ttd_perf: fusion S2 lane"
    );
    tracing::info!(
        target: "ttd_perf",
        n_hits = internal_hits.len(),
        degraded = internal_degraded.is_some(),
        duration_ms = internal_start.elapsed().as_millis() as u64,
        "ttd_perf: fusion internal hybrid lane"
    );

    // Fold lane degradation reasons.
    let lane_degraded: Option<String> = [arxiv_degraded, s2_degraded, internal_degraded]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("; ")
        .into();
    let lane_degraded = if lane_degraded.as_deref() == Some("") {
        None
    } else {
        lane_degraded
    };

    // Three-lane RRF fusion + quality gate.
    let fused = search::fuse_rrf_three_lane(
        &internal_hits,
        &arxiv_hits,
        &s2_hits,
        search::RRF_K_DEFAULT,
    );
    let (mut fused, mut degradation_reason) = search::gate_fused(fused, lane_degraded);

    // ── Lever B: cross-encoder rerank (reorder + drop floor) ──────────────────
    if rerank_enabled() {
        if let Some(reranker) = ctx.reranker.as_ref() {
            if !fused.is_empty() {
                let rerank_start = std::time::Instant::now();
                let docs: Vec<String> = fused
                    .iter()
                    .map(|h| format!("{}\n{}", h.title, h.content_preview))
                    .collect();
                match reranker.svc.rerank(query, &docs).await {
                    Ok(scores) => {
                        let (reranked, rerank_reason) =
                            search::apply_rerank(fused, &scores, reranker.min_score);
                        fused = reranked;
                        degradation_reason = fold_reason(degradation_reason, rerank_reason);
                        tracing::info!(
                            target: "ttd_perf",
                            n_after = fused.len(),
                            duration_ms = rerank_start.elapsed().as_millis() as u64,
                            "ttd_perf: cross-encoder rerank"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            query = %query,
                            error = %e,
                            "rerank failed — keeping RRF order (loud-degrade)"
                        );
                        degradation_reason = fold_reason(
                            degradation_reason,
                            Some(format!("cross-encoder reranker unavailable: {e}")),
                        );
                    }
                }
            }
        }
    }

    // Trim to top_k after gate + rerank.
    let fused: Vec<_> = fused.into_iter().take(top_k).collect();

    tracing::info!(
        target: "ttd_perf",
        n_fused = fused.len(),
        degraded = degradation_reason.is_some(),
        duration_ms = fusion_start.elapsed().as_millis() as u64,
        "ttd_perf: three-lane fusion complete"
    );

    // A3 contract: LocalOnly means no search, no gateway acquire, no ingest.
    if policy == RetrievalPolicy::LocalOnly {
        return (fused, degradation_reason, vec![]);
    }

    // ── Background full-text promotion (top-k hits) ───────────────────────────
    let promotion_handles = spawn_fulltext_promotion(ctx, &fused);

    (fused, degradation_reason, promotion_handles)
}

/// Spawn background full-text promotion for each hit: ar5iv HTML for `arxiv:*`,
/// open-access PDF for `s2:*`. Each handle promotes one paper from abstract-only
/// to full text and flips its `fulltext_status` to `indexed`; the caller awaits
/// the handles (bounded) before `build_panel`. Shared by the three-lane fusion
/// path and the seed-papers path so both ground on full text identically.
///
/// Gateway budget + single-flight gate every fetch, so re-running over papers
/// already `indexed` (or in flight) is a cheap no-op.
fn spawn_fulltext_promotion(
    ctx: &LitContext,
    hits: &[FusedHit],
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut promotion_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    let pool_arc = &ctx.lit_pool;
    let store_arc = &ctx.lit_store;
    let embedder_arc = &ctx.embedder;
    let gateway = &ctx.gateway;
    for hit in hits.iter() {
        // ── PDF promotion branch (s2:* hits) ────────────────────────────
        if hit.source_id.starts_with("s2:") {
            let s2_paper_id = hit.source_id.clone();
            let pool_clone_pdf = Arc::clone(pool_arc);
            let store_clone_pdf = Arc::clone(store_arc);
            let embedder_clone_pdf: Arc<dyn base::EmbeddingService> =
                Arc::clone(embedder_arc);
            let gateway_clone_pdf = Arc::clone(gateway);

            tracing::info!(
                target: "ttd_perf",
                paper_id = %s2_paper_id,
                "ttd_perf: enqueueing background PDF full-text promotion"
            );

            promotion_handles.push(tokio::spawn(async move {
                use search::pdf_fetch::PdfFetchConfig;
                use search::{
                    promote_pdf_fulltext, set_fulltext_status, Acquire, Endpoint, LitChunkConfig,
                };

                let flight_key = format!("pdf:{s2_paper_id}");

                if !gateway_clone_pdf.begin_flight(&flight_key) {
                    return;
                }

                let already_indexed = matches!(
                    sqlx::query_as::<_, (String,)>(
                        "SELECT fulltext_status FROM papers WHERE paper_id = ? LIMIT 1",
                    )
                    .bind(&s2_paper_id)
                    .fetch_optional(pool_clone_pdf.as_ref())
                    .await,
                    Ok(Some((ref s,))) if s == "indexed"
                );
                if already_indexed {
                    gateway_clone_pdf.end_flight(&flight_key);
                    return;
                }

                let row: Option<(Option<String>, String)> = sqlx::query_as(
                    "SELECT open_access_pdf_url, title FROM papers WHERE paper_id = ? LIMIT 1",
                )
                .bind(&s2_paper_id)
                .fetch_optional(pool_clone_pdf.as_ref())
                .await
                .ok()
                .flatten();

                let (pdf_url, title) = match row {
                    Some((Some(url), t)) if !url.is_empty() => (url, t),
                    _ => {
                        gateway_clone_pdf.end_flight(&flight_key);
                        return;
                    }
                };

                if gateway_clone_pdf.acquire(Endpoint::PdfFetch).await == Acquire::BudgetExhausted {
                    gateway_clone_pdf.end_flight(&flight_key);
                    tracing::warn!(
                        paper_id = %s2_paper_id,
                        "ttd_perf: PDF budget exhausted in background promotion"
                    );
                    return;
                }

                if let Err(e) =
                    set_fulltext_status(pool_clone_pdf.as_ref(), &s2_paper_id, "pending").await
                {
                    tracing::warn!(
                        paper_id = %s2_paper_id,
                        error = %e,
                        "ttd_perf: set_fulltext_status(pending) failed — status may drift from chunk state"
                    );
                }

                let pdf_cfg = PdfFetchConfig::from_env();
                let chunk_cfg = LitChunkConfig::default();
                if let Err(e) = promote_pdf_fulltext(
                    pool_clone_pdf.as_ref(),
                    store_clone_pdf.as_ref(),
                    embedder_clone_pdf.as_ref(),
                    &pdf_cfg,
                    &s2_paper_id,
                    &pdf_url,
                    &title,
                    &chunk_cfg,
                )
                .await
                {
                    tracing::warn!(
                        paper_id = %s2_paper_id,
                        error = %e,
                        "ttd_perf: background PDF promotion failed"
                    );
                }

                gateway_clone_pdf.end_flight(&flight_key);

                tracing::info!(
                    target: "ttd_perf",
                    paper_id = %s2_paper_id,
                    "ttd_perf: background PDF full-text promotion complete"
                );
            }));
            continue;
        }

        // ── ar5iv promotion branch (arxiv:* hits) ───────────────────────
        let arxiv_id = if let Some(bare) = hit.source_id.strip_prefix("arxiv:") {
            bare.to_string()
        } else if hit.source_type == "arxiv" && !hit.source_id.starts_with("s2:") {
            hit.source_id.clone()
        } else {
            continue;
        };
        let paper_id = format!("arxiv:{arxiv_id}");
        let pool_clone = Arc::clone(pool_arc);
        let store_clone = Arc::clone(store_arc);
        let embedder_clone: Arc<dyn base::EmbeddingService> = Arc::clone(embedder_arc);
        let gateway_clone = Arc::clone(gateway);

        tracing::info!(
            target: "ttd_perf",
            arxiv_id = %arxiv_id,
            "ttd_perf: enqueueing background full-text promotion"
        );

        promotion_handles.push(tokio::spawn(async move {
            use search::{Acquire, ArxivClient, ArxivConfig, Endpoint, LitChunkConfig};

            let flight_key = format!("ar5iv:{arxiv_id}");

            if !gateway_clone.begin_flight(&flight_key) {
                return;
            }

            let already_indexed = matches!(
                sqlx::query_as::<_, (String,)>(
                    "SELECT fulltext_status FROM papers WHERE paper_id = ? LIMIT 1",
                )
                .bind(&paper_id)
                .fetch_optional(pool_clone.as_ref())
                .await,
                Ok(Some((ref s,))) if s == "indexed"
            );
            if already_indexed {
                gateway_clone.end_flight(&flight_key);
                return;
            }

            if gateway_clone.acquire(Endpoint::Ar5ivFetch).await == Acquire::BudgetExhausted {
                gateway_clone.end_flight(&flight_key);
                tracing::warn!(
                    arxiv_id = %arxiv_id,
                    "ttd_perf: ar5iv budget exhausted in background promotion"
                );
                return;
            }

            if let Err(e) =
                search::set_fulltext_status(pool_clone.as_ref(), &paper_id, "pending").await
            {
                tracing::warn!(
                    arxiv_id = %arxiv_id,
                    error = %e,
                    "ttd_perf: set_fulltext_status(pending) failed — status may drift from chunk state"
                );
            }

            let client = match ArxivClient::new(ArxivConfig::default()) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        arxiv_id = %arxiv_id,
                        error = %e,
                        "ttd_perf: background promotion: arxiv client build failed"
                    );
                    if let Err(se) = search::set_fulltext_status(
                        pool_clone.as_ref(),
                        &paper_id,
                        "failed",
                    )
                    .await
                    {
                        tracing::warn!(
                            arxiv_id = %arxiv_id,
                            error = %se,
                            "ttd_perf: set_fulltext_status(failed) failed — paper may be stuck pending"
                        );
                    }
                    gateway_clone.end_flight(&flight_key);
                    return;
                }
            };

            let row: Option<(String, Option<String>)> = sqlx::query_as(
                "SELECT title, abstract FROM papers WHERE paper_id = ? LIMIT 1",
            )
            .bind(&paper_id)
            .fetch_optional(pool_clone.as_ref())
            .await
            .ok()
            .flatten();

            let (title, abstract_text) = match row {
                Some((t, a)) => (t, a.unwrap_or_default()),
                None => (arxiv_id.clone(), String::new()),
            };

            let meta = search::ArxivResult {
                arxiv_id: arxiv_id.clone(),
                title,
                abstract_text,
                authors: vec![],
                published: String::new(),
            };

            let chunk_cfg = LitChunkConfig::default();

            if let Err(e) = search::promote_arxiv_fulltext(
                pool_clone.as_ref(),
                store_clone.as_ref(),
                embedder_clone.as_ref(),
                &client,
                &meta,
                &chunk_cfg,
            )
            .await
            {
                tracing::warn!(
                    arxiv_id = %arxiv_id,
                    error = %e,
                    "ttd_perf: background promotion failed"
                );
            }

            gateway_clone.end_flight(&flight_key);

            tracing::info!(
                target: "ttd_perf",
                arxiv_id = %arxiv_id,
                "ttd_perf: background full-text promotion complete"
            );
        }));
    }

    promotion_handles
}

/// arxiv lane — abstract-first, skip-if-ingested. Full-text promotion fires
/// later on top-k hits.
async fn run_arxiv_lane_for_synthesize(
    query: &str,
    query_vec: &[f32],
    lit_store: &search::SqliteVecStore,
    embedder: &dyn base::EmbeddingService,
    lit_pool: Option<&sqlx::SqlitePool>,
    gateway: &Arc<search::LitGateway>,
) -> (Vec<search::ArxivHit>, Option<String>) {
    use base::{VectorFilters, VectorStore};
    use search::{Acquire, ArxivClient, ArxivConfig, Endpoint};

    let client = match ArxivClient::new(ArxivConfig::default()) {
        Ok(c) => c,
        Err(e) => {
            let r = format!("arxiv client build failed: {e}");
            tracing::warn!(error = %e, "synthesize: arxiv client init failed");
            return (vec![], Some(r));
        }
    };

    if gateway.acquire(Endpoint::ArxivSearch).await == Acquire::BudgetExhausted {
        return (
            vec![],
            Some("arxiv per-run call budget exhausted — lane degraded to local".into()),
        );
    }

    let search_start = std::time::Instant::now();
    let arxiv_results = match client.search(query).await {
        Ok(r) => r,
        Err(e) => {
            let r = format!("arxiv search failed: {e}");
            tracing::warn!(error = %e, "synthesize: arxiv search failed");
            return (vec![], Some(r));
        }
    };
    tracing::info!(
        target: "ttd_perf",
        n_results = arxiv_results.len(),
        duration_ms = search_start.elapsed().as_millis() as u64,
        "ttd_perf: arxiv API search"
    );

    if arxiv_results.is_empty() {
        return (vec![], None);
    }

    if let Some(pool) = lit_pool {
        let persist_start = std::time::Instant::now();
        let mut n_ingested = 0usize;
        let mut n_skipped = 0usize;
        for meta in &arxiv_results {
            let paper_id = format!("arxiv:{}", meta.arxiv_id);
            let already = match search::paper_is_ingested(pool, &paper_id).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        paper_id = %paper_id,
                        error = %e,
                        "synthesize: paper_is_ingested probe failed (DB fault, NOT a confirmed miss); proceeding to re-ingest"
                    );
                    false
                }
            };
            if already {
                n_skipped += 1;
                continue;
            }

            match search::persist_arxiv_abstract(pool, lit_store, embedder, meta).await {
                Ok(()) => n_ingested += 1,
                Err(e) => tracing::warn!(
                    arxiv_id = %meta.arxiv_id,
                    error = %e,
                    "synthesize: persist_arxiv_abstract failed; skipping"
                ),
            }
        }
        tracing::info!(
            target: "ttd_perf",
            n_papers = arxiv_results.len(),
            n_ingested,
            n_skipped,
            duration_ms = persist_start.elapsed().as_millis() as u64,
            "ttd_perf: arxiv abstract-only ingest (skip-if-ingested)"
        );
    }

    let filters = VectorFilters::default();
    let vec_hits = match lit_store.search(query_vec, 10, &filters).await {
        Ok(h) => h,
        Err(e) => {
            let r = format!("arxiv kNN search failed: {e}");
            tracing::warn!(error = %e, "synthesize: arxiv kNN failed");
            return (vec![], Some(r));
        }
    };

    let hits = vec_hits
        .into_iter()
        .map(|h| search::ArxivHit {
            arxiv_id: h.metadata.source_id.clone(),
            title: h.metadata.section.clone().unwrap_or_else(|| h.metadata.source_id.clone()),
            section: h.metadata.section.unwrap_or_default(),
            content: h.metadata.content_preview.clone(),
            content_preview: base::truncate_for_preview(&h.metadata.content_preview),
            relevance: h.similarity,
        })
        .collect();

    (hits, None)
}

/// S2 lane — abstract-first, skip-if-ingested, canonical keying.
async fn run_s2_lane_for_synthesize(
    query: &str,
    s2_client: &search::S2Client,
    lit_pool: Option<&sqlx::SqlitePool>,
    lit_store: Option<&search::SqliteVecStore>,
    embedder: Option<&dyn base::EmbeddingService>,
) -> (Vec<search::S2Hit>, Option<String>) {
    let s2_results = match s2_client.enrich(query).await {
        Ok(r) => r,
        Err(e) => {
            let r = format!("S2 search failed: {e}");
            tracing::warn!(error = %e, "synthesize: S2 enrich failed");
            return (vec![], Some(r));
        }
    };

    if s2_results.is_empty() {
        return (vec![], None);
    }

    let canonical_id = |r: &search::S2Result| -> String {
        match &r.arxiv_id {
            Some(aid) => format!("arxiv:{aid}"),
            None => format!("s2:{}", r.paper_id),
        }
    };
    if let (Some(pool), Some(store), Some(embed)) = (lit_pool, lit_store, embedder) {
        for r in &s2_results {
            let paper_id = canonical_id(r);
            let already = match search::paper_is_ingested(pool, &paper_id).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        paper_id = %paper_id,
                        error = %e,
                        "synthesize: paper_is_ingested probe failed (DB fault, NOT a confirmed miss); proceeding to re-ingest"
                    );
                    false
                }
            };
            if already {
                continue;
            }
            let persisted = if let Some(aid) = &r.arxiv_id {
                let meta = search::ArxivResult {
                    arxiv_id: aid.clone(),
                    title: r.title.clone(),
                    abstract_text: r.abstract_text.clone().unwrap_or_default(),
                    authors: r.authors.clone(),
                    published: r
                        .year
                        .map(|y| format!("{y}-01-01T00:00:00Z"))
                        .unwrap_or_default(),
                };
                search::persist_arxiv_abstract(pool, store, embed, &meta).await
            } else {
                let full = search::S2PaperFull {
                    s2_id: r.paper_id.clone(),
                    arxiv_id: None,
                    title: r.title.clone(),
                    abstract_text: r.abstract_text.clone(),
                    year: r.year,
                    citation_count: r.citation_count.unwrap_or(0),
                    influential_citation_count: 0,
                    reference_count: 0,
                    authors: r.authors.clone(),
                    venue: None,
                    doi: None,
                    open_access_pdf_url: r.open_access_pdf_url.clone(),
                };
                search::persist_s2_abstract(pool, store, embed, &full).await
            };
            if let Err(e) = persisted {
                tracing::warn!(
                    paper_id = %paper_id,
                    error = %e,
                    "synthesize: S2-lane abstract persist failed; skipping"
                );
            }
        }
    }

    let max_citations = s2_results
        .iter()
        .filter_map(|r| r.citation_count)
        .max()
        .unwrap_or(1)
        .max(1);

    let hits = s2_results
        .iter()
        .map(|r| {
            let body = r.abstract_text.clone().unwrap_or_default();
            let preview = base::truncate_for_preview(&body);
            let relevance = r
                .citation_count
                .map(|c| (c as f32 / max_citations as f32).clamp(0.0, 1.0))
                .unwrap_or(0.3);
            search::S2Hit {
                paper_id: canonical_id(r),
                title: r.title.clone(),
                content: body,
                content_preview: preview,
                relevance,
            }
        })
        .collect();

    (hits, None)
}

// ── Citation metadata ──────────────────────────────────────────────────────

/// Resolve `source_id → (authors, year, title, url)` from `papers`.
async fn fetch_paper_meta(
    pool: &sqlx::SqlitePool,
    source_ids: &[String],
) -> std::collections::BTreeMap<String, PaperMeta> {
    let mut map = std::collections::BTreeMap::new();
    if source_ids.is_empty() {
        return map;
    }
    let placeholders = std::iter::repeat("?")
        .take(source_ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT paper_id, title, year, authors, url FROM papers WHERE paper_id IN ({placeholders})"
    );
    let mut q = sqlx::query_as::<_, (String, Option<String>, Option<i64>, String, Option<String>)>(
        &sql,
    );
    for sid in source_ids {
        q = q.bind(sid);
    }
    match q.fetch_all(pool).await {
        Ok(rows) => {
            for (paper_id, title, year, authors_json, url) in rows {
                let authors: Vec<String> = serde_json::from_str(&authors_json).unwrap_or_default();
                map.insert(paper_id, PaperMeta { authors, year, title, url });
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "fetch_paper_meta: papers query failed; citations fall back to raw ids"
            );
        }
    }
    map
}

// ── Orchestration ──────────────────────────────────────────────────────────

/// Run one literature review end to end: Stage-0 exploration → three-lane fusion
/// → topicality gate → `build_panel` → TTD engine → cited synthesis.
///
/// Never panics on a missing service: a degraded lane folds a notice and the run
/// continues on whatever sources it has. An empty panel returns a loud degraded
/// [`ReviewResult`], never an empty success.
pub async fn run_review(
    question: &str,
    opts: &ReviewOptions,
    ctx: &LitContext,
) -> anyhow::Result<ReviewResult> {
    let trimmed = question.trim().to_string();
    if trimmed.is_empty() {
        anyhow::bail!("question must not be empty");
    }

    let run_id = uuid::Uuid::new_v4().to_string();
    let top_k = opts.top_k.clamp(1, TOP_K_HARD_CAP);
    let profile = opts.profile;
    let model = opts.model.as_str();

    let executor = Arc::clone(&ctx.executor);

    tracing::info!(
        run_id = %run_id,
        question_preview = %&trimmed[..trimmed.len().min(80)],
        top_k = top_k,
        profile = ?profile,
        s2_enabled = ctx.s2_enabled(),
        "synthesize: starting three-lane fusion + TTD run"
    );

    let handler_start = std::time::Instant::now();

    // Scope → policy. corpus-only scopes the live arxiv/S2 lanes out everywhere
    // (initial fusion + stage-1 graph gaps) and skips Stage-0 exploration, which
    // is S2-driven. The internal kNN lane (which surfaces source_type="local")
    // runs under both policies, so local docs are still retrieved.
    let live_policy = match opts.scope {
        Scope::CorpusOnly => RetrievalPolicy::LocalOnly,
        Scope::CorpusPlusWeb => RetrievalPolicy::Live,
    };

    // Two ways to build the initial panel:
    //  - seed-papers mode: build directly from user-supplied papers, skipping
    //    Stage-0, the three-lane fusion, and the topicality gate.
    //  - default: Stage-0 exploration → three-lane fusion → topicality gate.
    // Both yield `(fused_hits, fusion_notice, topicality_notice, seed_notice)`
    // with non-empty `fused_hits` (each path returns a degraded result early on
    // an empty panel). Stage-1/2 gap-fill runs identically afterwards.
    let (fused_hits, fusion_notice, topicality_notice, seed_notice) = if !opts.seed_papers.is_empty()
    {
        let (seed_hits, failures) = crate::seed::resolve_seeds(&opts.seed_papers, ctx).await;
        for f in &failures {
            tracing::warn!(run_id = %run_id, seed_failure = %f, "synthesize: seed paper unresolved");
        }
        let seed_note = if failures.is_empty() {
            None
        } else {
            Some(format!(
                "{} of {} seed paper(s) unresolved: {}",
                failures.len(),
                opts.seed_papers.len(),
                failures.join("; ")
            ))
        };

        if seed_hits.is_empty() {
            let detail = seed_note
                .as_ref()
                .map(|n| format!(" ({n})"))
                .unwrap_or_default();
            let notice =
                format!("⚠ Synthesis degraded: no seed papers resolved{detail}. No panel built.");
            tracing::warn!(run_id = %run_id, "synthesize: no seed papers resolved — degraded");
            return Ok(ReviewResult {
                synthesis_yaml: String::new(),
                graph_markdown: String::new(),
                run_id,
                bib_count: 0,
                narrative: String::new(),
                degraded: true,
                notice,
            });
        }

        tracing::info!(
            run_id = %run_id,
            n_seeds = seed_hits.len(),
            "synthesize: seed-papers mode — Stage-0, fusion, and topicality gate skipped"
        );

        // Seed papers ground on full text too: promote ar5iv/PDF and await
        // before build_panel, mirroring the fusion path. Skipped under
        // corpus-only (no web fetches), where seeds stay abstract-only.
        if opts.scope == Scope::CorpusPlusWeb {
            let handles = spawn_fulltext_promotion(ctx, &seed_hits);
            await_promotions(handles, &run_id).await;
        }

        (seed_hits, None, None, seed_note)
    } else {
        // Stage 0: citation-graph exploration (S2-driven). Skipped under
        // corpus-only (no web), and a cheap no-op when S2 is disabled (enrich
        // returns empty).
        if opts.scope == Scope::CorpusPlusWeb {
            let stage0_start = std::time::Instant::now();
            let explore_cfg = search::ExploreConfig::from_env();
            let seed_queries = decompose_seed_queries(&executor, &trimmed, model).await;
            let stats = search::explore_from_queries(
                &trimmed,
                &seed_queries,
                &explore_cfg,
                ctx.s2_client.as_ref(),
                &ctx.gateway,
                ctx.lit_pool.as_ref(),
                ctx.lit_store.as_ref(),
                ctx.embedder.as_ref(),
            )
            .await;
            tracing::info!(
                target: "ttd_perf",
                run_id = %run_id,
                papers_discovered = stats.papers_discovered,
                abstracts_indexed = stats.abstracts_indexed,
                s2_calls = stats.s2_calls,
                cache_hits = stats.s2_cache_hits,
                budget_exhausted = stats.budget_exhausted,
                duration_ms = stage0_start.elapsed().as_millis() as u64,
                "ttd_perf: stage-0 smart_explore"
            );
        } else {
            tracing::info!(run_id = %run_id, "scope=corpus-only — skipping Stage-0 web exploration");
        }

        // Call site 1: initial-panel fusion (policy set by scope).
        let (fused_hits, fusion_notice, promotion_handles) =
            run_three_lane_fusion(ctx, &trimmed, top_k, live_policy).await;
        tracing::info!(
            target: "ttd_perf",
            run_id = %run_id,
            n_fused = fused_hits.len(),
            duration_ms = handler_start.elapsed().as_millis() as u64,
            "ttd_perf: initial panel fusion"
        );

        if fused_hits.is_empty() {
            let notice =
                fusion_notice.unwrap_or_else(|| "no sources returned by three-lane fusion".into());
            let notice_msg = format!(
                "⚠ Synthesis degraded: {notice}. No panel built — returning empty synthesis."
            );
            tracing::warn!(run_id = %run_id, reason = %notice, "synthesize: empty panel — degraded");
            return Ok(ReviewResult {
                synthesis_yaml: String::new(),
                graph_markdown: String::new(),
                run_id,
                bib_count: 0,
                narrative: String::new(),
                degraded: true,
                notice: notice_msg,
            });
        }

        // Await initial-panel full-text promotion (bounded) BEFORE build_panel.
        await_promotions(promotion_handles, &run_id).await;

        // Lever A: topicality gate (LLM, binary drop).
        let (fused_hits, topicality_notice) = if topicality_enabled() {
            topicality_gate(&executor, &trimmed, fused_hits, model).await
        } else {
            (fused_hits, None)
        };

        (fused_hits, fusion_notice, topicality_notice, None)
    };

    let build_panel_start = std::time::Instant::now();
    let panel = match orchestration::adapter::build_panel(fused_hits, ctx.lit_pool.as_ref())
        .await
    {
        Ok(p) => p,
        Err(e) => {
            let notice =
                format!("⚠ Synthesis degraded: build_panel failed: {e}. Cannot ground synthesis.");
            tracing::warn!(run_id = %run_id, error = %e, "synthesize: build_panel failed");
            return Ok(ReviewResult {
                synthesis_yaml: String::new(),
                graph_markdown: String::new(),
                run_id,
                bib_count: 0,
                narrative: String::new(),
                degraded: true,
                notice,
            });
        }
    };

    tracing::info!(
        target: "ttd_perf",
        run_id = %run_id,
        n_experts = panel.len(),
        duration_ms = build_panel_start.elapsed().as_millis() as u64,
        "ttd_perf: build_panel"
    );

    if panel.is_empty() {
        let notice =
            "⚠ Synthesis degraded: panel built but empty — no grounded synthesis possible."
                .to_string();
        tracing::warn!(run_id = %run_id, "synthesize: panel empty after build_panel");
        return Ok(ReviewResult {
            synthesis_yaml: String::new(),
            graph_markdown: String::new(),
            run_id,
            bib_count: 0,
            narrative: String::new(),
            degraded: true,
            notice,
        });
    }

    // Full-text coverage accounting (loud, F6).
    let mut n_fulltext = 0usize;
    for expert in &panel {
        let status: Option<(String,)> = match sqlx::query_as(
            "SELECT fulltext_status FROM papers WHERE paper_id = ? LIMIT 1",
        )
        .bind(expert.expert_id.as_str())
        .fetch_optional(ctx.lit_pool.as_ref())
        .await
        {
            Ok(row) => row,
            Err(e) => {
                tracing::warn!(
                    paper_id = %expert.expert_id,
                    error = %e,
                    "synthesize: fulltext_status probe failed (DB fault, NOT abstract-only); counting as not-indexed"
                );
                None
            }
        };
        if status.map(|(s,)| s == "indexed").unwrap_or(false) {
            n_fulltext += 1;
        }
    }
    tracing::info!(
        target: "ttd_perf",
        run_id = %run_id,
        n_experts = panel.len(),
        n_fulltext,
        "ttd_perf: panel full-text coverage"
    );
    let coverage_notice: Option<String> = if n_fulltext == 0 {
        Some(format!(
            "panel grounded on abstracts only (0/{} experts have full text)",
            panel.len()
        ))
    } else {
        None
    };

    // Two retrievers: stage-1 (graph) Live, stage-2 (synthesis) LocalOnly.
    let make_callback = |ctx_c: LitContext, top_k_c: usize, policy_c: RetrievalPolicy| {
        CallbackLitSearch::new(move |query: String, _top_k_inner: usize| {
            let ctx_inner = ctx_c.clone();
            async move {
                let fused = tokio::time::timeout(
                    std::time::Duration::from_secs(180),
                    run_three_lane_fusion(&ctx_inner, &query, top_k_c, policy_c),
                )
                .await;
                let (hits, _notice, _promotion_handles) = match fused {
                    Ok(r) => r,
                    Err(_elapsed) => {
                        tracing::warn!(
                            query = %query,
                            "synthesize: per-gap retrieval timed out after 180s — \
                             degrading to empty retrieval for this gap"
                        );
                        (Vec::new(), None, Vec::new())
                    }
                };
                let contexts: Vec<orchestration::ttd::stages::RetrievedContext> = hits
                    .into_iter()
                    .map(|h| orchestration::ttd::stages::RetrievedContext {
                        source_id: h.source_id,
                        content: h.content,
                        section: h.section,
                    })
                    .collect();
                Ok(contexts)
            }
        })
    };

    // Stage-1 (graph) gaps follow the scope policy; stage-2 stays LocalOnly.
    let live_search = make_callback(ctx.clone(), top_k, live_policy);
    let live_retriever: Arc<dyn orchestration::ttd::retrieval::Retriever> =
        Arc::new(LitRetriever::new(Arc::new(live_search)));

    let local_search = make_callback(ctx.clone(), top_k, RetrievalPolicy::LocalOnly);
    let local_retriever: Arc<dyn orchestration::ttd::retrieval::Retriever> =
        Arc::new(LitRetriever::new(Arc::new(local_search)));

    let bib_store: Arc<dyn BibliographyStore> =
        Arc::new(SqliteBibliographyStore::new((*ctx.lit_pool).clone()));

    let run_prefix = &run_id[..run_id.len().min(8)];
    let question_id = uuid::Uuid::new_v4().to_string();
    let config = EngineConfig::new(
        TTD_AGENT_ID,
        model,
        format!("gna-{run_prefix}"),
        "r1",
        &question_id,
    )
    .with_run_id(run_id.clone())
    .with_stage_retrievers(live_retriever, local_retriever)
    .with_profile(profile)
    .with_panel_refresher(Arc::new(LitPanelRefresher {
        pool: (*ctx.lit_pool).clone(),
    }));

    let config = match opts.merger_model.as_deref() {
        Some(m) => config.with_merger_model(m),
        None => config,
    };

    let config = if plan_tournament_enabled() {
        tracing::info!(run_id = %run_id, "plan tournament ENABLED (PlanMode::Tournament)");
        config.with_plan_tournament()
    } else {
        config
    };

    let engine_run_start = std::time::Instant::now();
    let engine_result = run_engine_with_bib(&panel, &config, executor, bib_store).await;
    tracing::info!(
        target: "ttd_perf",
        run_id = %run_id,
        ok = engine_result.is_ok(),
        duration_ms = engine_run_start.elapsed().as_millis() as u64,
        "ttd_perf: run_engine_with_bib"
    );

    let (synthesis_yaml, graph_markdown, narrative, run_degraded, run_notice) = match engine_result {
        Ok(mut result) => {
            let graph_markdown = result.graph.to_markdown();
            let narrative = result.synthesis.narrative.clone();
            let yaml = match profile {
                PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                    let mut cited: Vec<String> = Vec::new();
                    for c in &result.synthesis.claims {
                        for s in &c.sources {
                            if !cited.contains(s) {
                                cited.push(s.clone());
                            }
                        }
                    }
                    let meta = fetch_paper_meta(ctx.lit_pool.as_ref(), &cited).await;
                    apply_author_year_citations(&mut result.synthesis, &meta);
                    match result.synthesis.to_yaml() {
                        Ok(y) => y,
                        Err(e) => {
                            tracing::warn!(
                                run_id = %run_id,
                                error = %e,
                                "synthesize: citation re-serialise failed; emitting pre-citation yaml"
                            );
                            result.yaml
                        }
                    }
                }
                PromptProfile::V1Delphi => result.yaml,
            };
            (yaml, graph_markdown, narrative, false, String::new())
        }
        Err(e) => {
            let notice = format!(
                "⚠ Synthesis degraded: engine run failed: {e}. Bibliography may be partial."
            );
            tracing::warn!(run_id = %run_id, error = %e, "synthesize: run_engine_with_bib failed");
            (String::new(), String::new(), String::new(), true, notice)
        }
    };

    // Count bibliography rows for this run_id.
    let bib_count: usize = match sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM synthesis_bibliography WHERE run_id = ?",
    )
    .bind(&run_id)
    .fetch_one(ctx.lit_pool.as_ref())
    .await
    {
        Ok(n) => n as usize,
        Err(e) => {
            tracing::warn!(
                run_id = %run_id,
                error = %e,
                "synthesize: bibliography count query failed — reporting 0 (DB fault)"
            );
            0
        }
    };

    // Compose the final notice.
    let (degraded, notice) = if run_degraded {
        (true, run_notice)
    } else if let Some(ref fusion_n) = fusion_notice {
        (true, format!("⚠ Synthesis degraded (partial sources): {fusion_n}"))
    } else {
        (false, String::new())
    };
    let (degraded, notice) = match coverage_notice {
        Some(c) if !degraded => (true, format!("⚠ Synthesis degraded: {c}")),
        Some(c) => (degraded, format!("{notice}; {c}")),
        None => (degraded, notice),
    };
    let (degraded, notice) = match topicality_notice {
        Some(t) if !degraded => (true, format!("⚠ Synthesis note: {t}")),
        Some(t) => (degraded, format!("{notice}; {t}")),
        None => (degraded, notice),
    };
    let (degraded, notice) = match seed_notice {
        Some(s) if !degraded => (true, format!("⚠ Synthesis note: {s}")),
        Some(s) => (degraded, format!("{notice}; {s}")),
        None => (degraded, notice),
    };

    tracing::info!(
        run_id = %run_id,
        bib_count = bib_count,
        degraded = degraded,
        duration_ms = handler_start.elapsed().as_millis() as u64,
        "synthesize: completed"
    );
    let gw = ctx.gateway.snapshot().await;
    tracing::info!(
        target: "ttd_perf",
        run_id = %run_id,
        arxiv_calls = gw.arxiv_calls,
        ar5iv_calls = gw.ar5iv_calls,
        s2_calls = gw.s2_calls,
        pdf_calls = gw.pdf_calls,
        backoffs = gw.backoffs,
        "ttd_perf: lit gateway run totals"
    );

    Ok(ReviewResult {
        synthesis_yaml,
        graph_markdown,
        run_id,
        bib_count,
        narrative,
        degraded,
        notice,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_keep_indices_handles_clean_and_prose_replies() {
        assert_eq!(parse_keep_indices("0, 2, 5", 6), [0, 2, 5].into_iter().collect());
        assert_eq!(
            parse_keep_indices("Papers 0 and 3 are on-topic.", 6),
            [0, 3].into_iter().collect()
        );
        assert_eq!(parse_keep_indices("1, 9", 3), [1].into_iter().collect());
        assert_eq!(parse_keep_indices("none", 3), std::collections::BTreeSet::new());
    }

    #[test]
    fn parse_prompt_profile_allows_known_rejects_unknown() {
        assert_eq!(parse_prompt_profile(None).unwrap(), PromptProfile::V3LitReviewLong);
        assert_eq!(
            parse_prompt_profile(Some("v1/delphi")).unwrap(),
            PromptProfile::V1Delphi
        );
        assert_eq!(
            parse_prompt_profile(Some("V2/Lit-Review")).unwrap(),
            PromptProfile::V2LitReview
        );
        assert!(parse_prompt_profile(Some("bogus")).is_err());
    }

    #[test]
    fn fold_reason_joins_with_separator() {
        assert_eq!(
            fold_reason(Some("a".into()), Some("b".into())),
            Some("a; b".into())
        );
        assert_eq!(fold_reason(Some("a".into()), None), Some("a".into()));
        assert_eq!(fold_reason(None, Some("b".into())), Some("b".into()));
        assert_eq!(fold_reason(None, None), None);
    }
}
