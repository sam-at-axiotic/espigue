//! Stage 0 literature exploration — port of clawd's `smart_explore.py`.
//!
//! Corpus-building citation-graph exploration that ALWAYS runs before the TTD
//! engine. Seeds from the research question via S2 search, traverses the
//! citation graph with batched S2 calls (O(depth) network calls), ranks the
//! frontier by embedding similarity to the question, keeps top-K per level,
//! ingests every discovery abstract-first through A2's primitives, and budgets
//! every call through the A1 gateway.
//!
//! ## Deliberate divergences from smart_explore.py
//!
//! 1. **LLM ranking mode NOT ported.** Embedding-only ranking.
//! 2. **Single-seed `explore()` entry NOT ported.** Only `explore_from_query` + `discover_seeds`.
//! 3. **No ar5iv full-text in Stage 0.** Abstract-first only; zero `Endpoint::Ar5ivFetch` acquires.
//! 4. **S2Cache storage = `s2_cache` table.** Not disk JSON.
//! 5. **Pacing/budget/backoff live in LitGateway.** Client methods are raw calls.
//! 6. **Seeds from S2 search only.** The arxiv half is supplied by the initial Live fusion.
//! 7. **Multi-seed depth cap is KEPT.** `effective_depth = min(max_depth, 1)` when seeds > 1.
//! 8. (synthesize.rs) **arxiv-lane kNN filter widened** — handled in Task 3, not here.
//!
//! Any divergence not listed above is a bug — flag it, do not rationalise it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use alzina_core::search::{EmbeddingService, EmbeddingTask};

use crate::lit_gateway::{Acquire, Endpoint, LitGateway, RetryAdvice};
use crate::lit_intake::{persist_arxiv_abstract, persist_s2_abstract};
use crate::lit_schema::{paper_is_ingested, s2_cache_get, s2_cache_put, s2_cache_put_if_absent};
use crate::s2_enrichment::{resolve_paper_id, S2CallError, S2Client, S2PaperFull};
use crate::sqlite_vec::SqliteVecStore;

// ── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for `explore_from_query`.
///
/// Defaults match clawd's `explore_from_query` signature
/// (smart_explore.py:612-619). All values overridable via env.
#[derive(Debug, Clone)]
pub struct ExploreConfig {
    /// Number of seed papers from S2 search. Default 3.
    pub num_seeds: usize,
    /// Max citation-graph depth per seed. Default 2.
    pub max_depth: usize,
    /// Papers to keep per depth level after ranking. Default 10.
    pub papers_per_level: usize,
    /// Minimum cosine similarity to include a candidate. Default 0.3.
    pub min_relevance: f32,
    /// Max citations/references to fetch per edge (candidates_per_edge). Default 100.
    pub candidates_per_edge: usize,
    /// Number of papers to search S2 for seeds. Default 30.
    pub seed_search_limit: usize,
}

impl Default for ExploreConfig {
    fn default() -> Self {
        Self {
            num_seeds: 3,
            max_depth: 2,
            papers_per_level: 10,
            min_relevance: 0.3,
            candidates_per_edge: 100,
            seed_search_limit: 30,
        }
    }
}

impl ExploreConfig {
    /// Build from defaults, then apply env overrides.
    ///
    /// Env vars:
    /// - `ALZINA_EXPLORE_NUM_SEEDS`
    /// - `ALZINA_EXPLORE_DEPTH`
    /// - `ALZINA_EXPLORE_PAPERS_PER_LEVEL`
    /// - `ALZINA_EXPLORE_MIN_RELEVANCE`
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default)
        }
        fn env_f32(key: &str, default: f32) -> f32 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default)
        }

        cfg.num_seeds = env_usize("ALZINA_EXPLORE_NUM_SEEDS", cfg.num_seeds);
        cfg.max_depth = env_usize("ALZINA_EXPLORE_DEPTH", cfg.max_depth);
        cfg.papers_per_level =
            env_usize("ALZINA_EXPLORE_PAPERS_PER_LEVEL", cfg.papers_per_level);
        cfg.min_relevance = env_f32("ALZINA_EXPLORE_MIN_RELEVANCE", cfg.min_relevance);
        cfg
    }
}

// ── Stats ──────────────────────────────────────────────────────────────────────

/// Telemetry returned by `explore_from_query`.
///
/// Populated incrementally; even a budget-exhausted run returns partial counts.
/// Fields map to the `ttd_perf` log line in synthesize.rs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExploreStats {
    /// Titles of seed papers chosen.
    pub seeds: Vec<String>,
    /// Total papers visited (seeds + traversal discoveries).
    pub papers_discovered: usize,
    /// Papers for which an abstract chunk was ingested.
    pub abstracts_indexed: usize,
    /// Papers already in the literature store — skipped ingest, still traversable.
    pub skipped_already_ingested: usize,
    /// Network calls granted by the gateway (each `Acquire::Proceed`).
    pub s2_calls: usize,
    /// S2 responses served from the DB cache.
    pub s2_cache_hits: usize,
    /// S2 responses that required a network call (cache miss).
    pub s2_cache_misses: usize,
    /// True if the per-run S2 budget was exhausted during traversal.
    pub budget_exhausted: bool,
    /// Human-readable breadcrumbs for loud degradation (budget notice, seed failure, etc.).
    pub notes: Vec<String>,
}

// ── Cosine similarity ──────────────────────────────────────────────────────────

/// Port of clawd `cosine_similarity` (smart_explore.py:96-103).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ── Cache + gateway sandwich ───────────────────────────────────────────────────

/// Cache key helpers mirroring clawd S2Cache scheme.
fn cache_key_paper(resolved_id: &str) -> String {
    format!("paper_{resolved_id}")
}
fn cache_key_citations(resolved_id: &str) -> String {
    format!("{resolved_id}_citations")
}
fn cache_key_references(resolved_id: &str) -> String {
    format!("{resolved_id}_references")
}

/// Cache+gateway+backoff wrapper for `search_papers`.
async fn cached_search_papers(
    query: &str,
    limit: usize,
    s2: &S2Client,
    gateway: &Arc<LitGateway>,
    lit_pool: &SqlitePool,
    stats: &mut ExploreStats,
) -> Vec<S2PaperFull> {
    // No per-query search cache (different from paper-level cache).
    match gateway.acquire(Endpoint::S2).await {
        Acquire::BudgetExhausted => {
            stats.budget_exhausted = true;
            stats.notes.push("S2 budget exhausted during seed search".to_string());
            return vec![];
        }
        Acquire::Proceed => {
            stats.s2_calls += 1;
        }
    }

    let result = gateway
        .with_backoff(Endpoint::S2, || async {
            s2.search_papers(query, limit).await.map_err(|e: S2CallError| {
                let is_retry = e.status.map(|s| s == 429 || s >= 500).unwrap_or(false);
                if is_retry {
                    RetryAdvice::Retry { error: e.message, retry_after: e.retry_after }
                } else {
                    RetryAdvice::Fatal(e.message)
                }
            })
        })
        .await;

    match result {
        Ok(papers) => {
            // Cache each paper individually (put_if_absent).
            for p in &papers {
                let resolved = resolve_paper_id(&p.s2_id);
                let key = cache_key_paper(&resolved);
                if let Ok(payload) = serde_json::to_string(p) {
                    let _ = s2_cache_put_if_absent(lit_pool, &key, &payload).await;
                }
            }
            papers
        }
        Err(e) => {
            stats.notes.push(format!("search_papers '{query}' failed: {e}"));
            vec![]
        }
    }
}

/// Cache+gateway+backoff wrapper for `get_citations`.
async fn cached_get_citations(
    paper_id: &str,
    limit: usize,
    s2: &S2Client,
    gateway: &Arc<LitGateway>,
    lit_pool: &SqlitePool,
    stats: &mut ExploreStats,
) -> Vec<S2PaperFull> {
    let resolved = resolve_paper_id(paper_id);
    let key = cache_key_citations(&resolved);

    if let Ok(Some(payload)) = s2_cache_get(lit_pool, &key).await {
        stats.s2_cache_hits += 1;
        if let Ok(papers) = serde_json::from_str::<Vec<S2PaperFull>>(&payload) {
            return papers;
        }
    }
    stats.s2_cache_misses += 1;

    match gateway.acquire(Endpoint::S2).await {
        Acquire::BudgetExhausted => {
            stats.budget_exhausted = true;
            stats.notes.push(format!("S2 budget exhausted fetching citations for {paper_id}"));
            return vec![];
        }
        Acquire::Proceed => {
            stats.s2_calls += 1;
        }
    }

    let result = gateway
        .with_backoff(Endpoint::S2, || {
            let id = paper_id.to_string();
            async move {
                s2.get_citations(&id, limit).await.map_err(|e: S2CallError| {
                    let is_retry = e.status.map(|s| s == 429 || s >= 500).unwrap_or(false);
                    if is_retry {
                        RetryAdvice::Retry { error: e.message, retry_after: e.retry_after }
                    } else {
                        RetryAdvice::Fatal(e.message)
                    }
                })
            }
        })
        .await;

    match result {
        Ok(papers) => {
            if let Ok(payload) = serde_json::to_string(&papers) {
                let _ = s2_cache_put(lit_pool, &key, &payload).await;
            }
            // Cache each embedded paper individually (put_if_absent — clawd :263-268).
            for p in &papers {
                let resolved_p = resolve_paper_id(&p.s2_id);
                let paper_key = cache_key_paper(&resolved_p);
                if let Ok(p_payload) = serde_json::to_string(p) {
                    let _ = s2_cache_put_if_absent(lit_pool, &paper_key, &p_payload).await;
                }
            }
            papers
        }
        Err(e) => {
            stats.notes.push(format!("get_citations {paper_id} failed: {e}"));
            vec![]
        }
    }
}

/// Cache+gateway+backoff wrapper for `get_references`.
async fn cached_get_references(
    paper_id: &str,
    limit: usize,
    s2: &S2Client,
    gateway: &Arc<LitGateway>,
    lit_pool: &SqlitePool,
    stats: &mut ExploreStats,
) -> Vec<S2PaperFull> {
    let resolved = resolve_paper_id(paper_id);
    let key = cache_key_references(&resolved);

    if let Ok(Some(payload)) = s2_cache_get(lit_pool, &key).await {
        stats.s2_cache_hits += 1;
        if let Ok(papers) = serde_json::from_str::<Vec<S2PaperFull>>(&payload) {
            return papers;
        }
    }
    stats.s2_cache_misses += 1;

    match gateway.acquire(Endpoint::S2).await {
        Acquire::BudgetExhausted => {
            stats.budget_exhausted = true;
            stats.notes.push(format!("S2 budget exhausted fetching references for {paper_id}"));
            return vec![];
        }
        Acquire::Proceed => {
            stats.s2_calls += 1;
        }
    }

    let result = gateway
        .with_backoff(Endpoint::S2, || {
            let id = paper_id.to_string();
            async move {
                s2.get_references(&id, limit).await.map_err(|e: S2CallError| {
                    let is_retry = e.status.map(|s| s == 429 || s >= 500).unwrap_or(false);
                    if is_retry {
                        RetryAdvice::Retry { error: e.message, retry_after: e.retry_after }
                    } else {
                        RetryAdvice::Fatal(e.message)
                    }
                })
            }
        })
        .await;

    match result {
        Ok(papers) => {
            if let Ok(payload) = serde_json::to_string(&papers) {
                let _ = s2_cache_put(lit_pool, &key, &payload).await;
            }
            // Cache each embedded paper individually (clawd :300-305).
            for p in &papers {
                let resolved_p = resolve_paper_id(&p.s2_id);
                let paper_key = cache_key_paper(&resolved_p);
                if let Ok(p_payload) = serde_json::to_string(p) {
                    let _ = s2_cache_put_if_absent(lit_pool, &paper_key, &p_payload).await;
                }
            }
            papers
        }
        Err(e) => {
            stats.notes.push(format!("get_references {paper_id} failed: {e}"));
            vec![]
        }
    }
}

/// Cache+gateway+backoff wrapper for `get_papers_batch`.
///
/// Per clawd :340-358: checks per-id paper cache first; only fetches uncached ids.
/// ON SUCCESS: cache-puts each fetched paper AND `s2_cache_put_if_absent` for
/// the direct key (clawd :369-374).
async fn cached_get_papers_batch(
    s2_ids: &[String],
    s2: &S2Client,
    gateway: &Arc<LitGateway>,
    lit_pool: &SqlitePool,
    stats: &mut ExploreStats,
) -> Vec<Option<S2PaperFull>> {
    if s2_ids.is_empty() {
        return vec![];
    }

    let mut results: Vec<Option<S2PaperFull>> = vec![None; s2_ids.len()];
    let mut uncached_indices: Vec<usize> = Vec::new();
    let mut uncached_ids: Vec<String> = Vec::new();

    // Per-id cache check (clawd :340-358).
    for (i, id) in s2_ids.iter().enumerate() {
        let resolved = resolve_paper_id(id);
        let key = cache_key_paper(&resolved);
        if let Ok(Some(payload)) = s2_cache_get(lit_pool, &key).await {
            stats.s2_cache_hits += 1;
            if let Ok(paper) = serde_json::from_str::<S2PaperFull>(&payload) {
                results[i] = Some(paper);
                continue;
            }
        }
        stats.s2_cache_misses += 1;
        uncached_indices.push(i);
        uncached_ids.push(id.clone());
    }

    if uncached_ids.is_empty() {
        return results;
    }

    // One batch call for all uncached ids.
    match gateway.acquire(Endpoint::S2).await {
        Acquire::BudgetExhausted => {
            stats.budget_exhausted = true;
            stats
                .notes
                .push(format!("S2 budget exhausted in batch fetch of {} papers", uncached_ids.len()));
            return results;
        }
        Acquire::Proceed => {
            stats.s2_calls += 1;
        }
    }

    let batch_result = gateway
        .with_backoff(Endpoint::S2, || {
            let ids = uncached_ids.clone();
            async move {
                s2.get_papers_batch(&ids).await.map_err(|e: S2CallError| {
                    let is_retry = e.status.map(|s| s == 429 || s >= 500).unwrap_or(false);
                    if is_retry {
                        RetryAdvice::Retry { error: e.message, retry_after: e.retry_after }
                    } else {
                        RetryAdvice::Fatal(e.message)
                    }
                })
            }
        })
        .await;

    match batch_result {
        Ok(fetched) => {
            for (j, maybe_paper) in fetched.into_iter().enumerate() {
                let idx = uncached_indices[j];
                if let Some(ref paper) = maybe_paper {
                    // Cache by resolved id AND by S2 id if different.
                    let resolved = resolve_paper_id(&uncached_ids[j]);
                    let key = cache_key_paper(&resolved);
                    if let Ok(payload) = serde_json::to_string(paper) {
                        let _ = s2_cache_put(lit_pool, &key, &payload).await;
                        // Also cache by s2_id directly.
                        let s2_key = cache_key_paper(&paper.s2_id);
                        if s2_key != key {
                            let _ = s2_cache_put_if_absent(lit_pool, &s2_key, &payload).await;
                        }
                    }
                }
                results[idx] = maybe_paper;
            }
        }
        Err(e) => {
            stats.notes.push(format!("get_papers_batch failed: {e}"));
        }
    }

    results
}

// ── Ingest helper ──────────────────────────────────────────────────────────────

/// Ingest one discovered paper abstract-first via A2 primitives.
///
/// Computes `paper_id`, checks skip-if-ingested, routes to
/// `persist_arxiv_abstract` or `persist_s2_abstract`. Ingest errors are
/// warn-logged and recorded in stats.notes — never abort the stage.
///
/// NO ar5iv calls anywhere in this function (divergence 3).
///
/// Returns `(was_ingested: bool)`.
async fn ingest_paper(
    paper: &S2PaperFull,
    lit_pool: &SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn EmbeddingService,
    stats: &mut ExploreStats,
) -> bool {
    // Compute the canonical paper_id.
    let paper_id = if let Some(ref aid) = paper.arxiv_id {
        format!("arxiv:{aid}")
    } else {
        format!("s2:{}", paper.s2_id)
    };

    // Skip-if-ingested — still traversable.
    match paper_is_ingested(lit_pool, &paper_id).await {
        Ok(true) => {
            stats.skipped_already_ingested += 1;
            return false;
        }
        Err(e) => {
            stats.notes.push(format!("paper_is_ingested check failed for {paper_id}: {e}"));
            // Continue and try to ingest anyway (best-effort).
        }
        Ok(false) => {}
    }

    let ingested = if let Some(ref arxiv_id) = paper.arxiv_id {
        // Has arxiv id — use persist_arxiv_abstract.
        use crate::arxiv::ArxivResult;
        let meta = ArxivResult {
            arxiv_id: arxiv_id.clone(),
            title: paper.title.clone(),
            abstract_text: paper.abstract_text.clone().unwrap_or_default(),
            authors: paper.authors.clone(),
            published: paper
                .year
                .map(|y| format!("{y}-01-01T00:00:00Z"))
                .unwrap_or_default(),
        };
        match persist_arxiv_abstract(lit_pool, lit_store, embedder, &meta).await {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(paper_id = %paper_id, error = %e,
                    "lit_explore: persist_arxiv_abstract failed — skipping");
                stats.notes.push(format!("ingest {paper_id} failed: {e}"));
                false
            }
        }
    } else {
        // No arxiv id — use persist_s2_abstract.
        match persist_s2_abstract(lit_pool, lit_store, embedder, paper).await {
            Ok(()) => paper.abstract_text.is_some(),
            Err(e) => {
                tracing::warn!(paper_id = %paper_id, error = %e,
                    "lit_explore: persist_s2_abstract failed — skipping");
                stats.notes.push(format!("ingest {paper_id} failed: {e}"));
                false
            }
        }
    };

    if ingested {
        stats.abstracts_indexed += 1;
    }
    ingested
}

// ── Public entry point ─────────────────────────────────────────────────────────

/// Stage 0 corpus-building exploration (single-query convenience wrapper).
///
/// Port of clawd `SmartExplorer.explore_from_query`
/// (smart_explore.py:612-746) with `discover_seeds` no-LLM branch (:141-156).
///
/// NEVER returns `Err`. Every failure path records a note in `ExploreStats.notes`
/// and returns the stats accumulated so far (operator lock: degradation loud,
/// never error).
pub async fn explore_from_query(
    question: &str,
    config: &ExploreConfig,
    s2: &S2Client,
    gateway: &Arc<LitGateway>,
    lit_pool: &SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn EmbeddingService,
) -> ExploreStats {
    explore_from_queries(
        question,
        &[question.to_string()],
        config,
        s2,
        gateway,
        lit_pool,
        lit_store,
        embedder,
    )
    .await
}

/// Stage 0 exploration over multiple seed queries (worklist item 6).
///
/// Probe-14 F8: a long natural-language question passed verbatim to S2
/// keyword search returned 0 results and explore exited without exploring.
/// This entry point takes decomposed short queries (the dispatch boundary
/// derives them — clawd's LLM seed branch) and unions their candidates by
/// `s2_id` before seed selection. `question` is kept separately for the
/// embedding-relevance ranking of the citation frontier.
///
/// Same never-`Err` contract as [`explore_from_query`].
#[allow(clippy::too_many_arguments)]
pub async fn explore_from_queries(
    question: &str,
    queries: &[String],
    config: &ExploreConfig,
    s2: &S2Client,
    gateway: &Arc<LitGateway>,
    lit_pool: &SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn EmbeddingService,
) -> ExploreStats {
    let mut stats = ExploreStats::default();

    // ── Step 1: Discover seeds via S2 search, one search per query ───────────
    // Union by s2_id, first-seen order (earlier queries take precedence).
    let mut candidates: Vec<S2PaperFull> = Vec::new();
    let mut seen_seed_ids: HashSet<String> = HashSet::new();
    for query in queries {
        if stats.budget_exhausted {
            break;
        }
        let found = cached_search_papers(
            query,
            config.seed_search_limit,
            s2,
            gateway,
            lit_pool,
            &mut stats,
        )
        .await;
        for p in found {
            if !p.s2_id.is_empty() && seen_seed_ids.insert(p.s2_id.clone()) {
                candidates.push(p);
            }
        }
    }

    if candidates.is_empty() {
        stats
            .notes
            .push(format!("No papers found for queries: {queries:?}"));
        return stats;
    }

    // Sort by citation_count descending; take num_seeds.
    candidates.sort_by(|a, b| b.citation_count.cmp(&a.citation_count));
    let seeds: Vec<S2PaperFull> = candidates.into_iter().take(config.num_seeds).collect();

    stats.seeds = seeds.iter().map(|s| s.title.clone()).collect();

    // Global visited map: s2_id → (depth, relevance).
    let mut all_visited: HashMap<String, (usize, f32)> = HashMap::new();

    // ── Step 2: Per-seed exploration ──────────────────────────────────────────
    for seed in &seeds {
        if stats.budget_exhausted {
            break;
        }

        // Query embedding = embed("{question}. {seed_abstract_or_title}") (clawd :649-652).
        let seed_text = seed.abstract_text.as_deref().unwrap_or(&seed.title);
        let query_text = format!("{question}. {seed_text}");
        let query_embedding = match embedder.embed(&query_text, EmbeddingTask::Query).await {
            Ok(v) => v,
            Err(e) => {
                stats.notes.push(format!("query embedding failed for seed '{}': {e}", seed.title));
                continue;
            }
        };

        // Ingest the seed.
        stats.papers_discovered += 1;
        all_visited.insert(seed.s2_id.clone(), (0, 1.0));
        ingest_paper(seed, lit_pool, lit_store, embedder, &mut stats).await;

        // effective_depth: multi-seed cap (clawd :668).
        let effective_depth = if seeds.len() > 1 {
            config.max_depth.min(1)
        } else {
            config.max_depth
        };

        let mut current_level: Vec<String> = vec![seed.s2_id.clone()];
        let mut seen_this_seed: HashSet<String> = HashSet::from([seed.s2_id.clone()]);

        for _depth in 1..=effective_depth {
            if stats.budget_exhausted {
                break;
            }

            // Gather candidates from current level.
            let mut all_candidates: Vec<S2PaperFull> = Vec::new();

            for paper_id in &current_level {
                if stats.budget_exhausted {
                    break;
                }

                // references (clawd does both).
                let refs = cached_get_references(
                    paper_id,
                    config.candidates_per_edge,
                    s2,
                    gateway,
                    lit_pool,
                    &mut stats,
                )
                .await;
                for p in refs {
                    if !seen_this_seed.contains(&p.s2_id)
                        && !all_visited.contains_key(&p.s2_id)
                        && !p.s2_id.is_empty()
                    {
                        seen_this_seed.insert(p.s2_id.clone());
                        all_candidates.push(p);
                    }
                }

                if stats.budget_exhausted {
                    break;
                }

                // citations.
                let cits = cached_get_citations(
                    paper_id,
                    config.candidates_per_edge,
                    s2,
                    gateway,
                    lit_pool,
                    &mut stats,
                )
                .await;
                for p in cits {
                    if !seen_this_seed.contains(&p.s2_id)
                        && !all_visited.contains_key(&p.s2_id)
                        && !p.s2_id.is_empty()
                    {
                        seen_this_seed.insert(p.s2_id.clone());
                        all_candidates.push(p);
                    }
                }
            }

            if all_candidates.is_empty() || stats.budget_exhausted {
                break;
            }

            // Rank ALL new candidates by cosine similarity to query embedding.
            // embed_batch owns batching (divergence 1 — no Jina sub-batch re-impl).
            let texts: Vec<String> = all_candidates
                .iter()
                .map(|c| {
                    if let Some(ref abs) = c.abstract_text {
                        let preview: String = abs.chars().take(300).collect();
                        format!("{}. {preview}", c.title)
                    } else {
                        c.title.clone()
                    }
                })
                .collect();

            let embeddings =
                match embedder.embed_batch(&texts, EmbeddingTask::Passage).await {
                    Ok(v) => v,
                    Err(e) => {
                        stats.notes.push(format!("embed_batch failed at depth {_depth}: {e}"));
                        break;
                    }
                };

            // Pair each candidate with its similarity score.
            let mut scored: Vec<(f32, &S2PaperFull)> = all_candidates
                .iter()
                .zip(embeddings.iter())
                .map(|(c, emb)| (cosine(&query_embedding, emb), c))
                .filter(|(score, _)| *score >= config.min_relevance)
                .collect();

            // Sort descending.
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(config.papers_per_level);

            if scored.is_empty() {
                break;
            }

            // Batch-fetch winners (ONE call — the O(depth) property, clawd :541-544).
            let winner_ids: Vec<String> = scored.iter().map(|(_, c)| c.s2_id.clone()).collect();
            let fetched = cached_get_papers_batch(
                &winner_ids,
                s2,
                gateway,
                lit_pool,
                &mut stats,
            )
            .await;

            let mut next_level: Vec<String> = Vec::new();

            for (maybe_paper, (score, candidate)) in
                fetched.into_iter().zip(scored.iter())
            {
                if stats.budget_exhausted {
                    break;
                }
                let paper = match maybe_paper {
                    Some(p) => p,
                    None => continue,
                };
                if all_visited.contains_key(&paper.s2_id) {
                    continue;
                }
                stats.papers_discovered += 1;
                all_visited.insert(paper.s2_id.clone(), (_depth, *score));
                ingest_paper(&paper, lit_pool, lit_store, embedder, &mut stats).await;
                next_level.push(paper.s2_id);
                let _ = candidate; // score already used above
            }

            current_level = next_level;
            if current_level.is_empty() {
                break;
            }
        }
    }

    if stats.budget_exhausted && !stats.notes.iter().any(|n| n.contains("exhausted at depth")) {
        stats.notes.push(format!(
            "S2 budget exhausted after {} calls",
            stats.s2_calls
        ));
    }

    stats
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_core::error::AlzinaResult;
    use serde_json::json;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::lit_gateway::LitGateway;
    use crate::lit_schema::in_memory_lit_pool;
    use crate::s2_enrichment::{S2Client, S2Config};
    use crate::sqlite_vec::SqliteVecStore;

    // ── Stub embedder — content-sensitive vectors for deterministic ranking ───

    /// StubEmbedder: returns a 1024-dim unit vector. The first element is set
    /// to a hash-derived value so different text gives distinguishably different
    /// vectors (deterministic ranking in tests).
    struct StubEmbedder;

    #[async_trait::async_trait]
    impl alzina_core::search::EmbeddingService for StubEmbedder {
        async fn embed(&self, text: &str, _task: EmbeddingTask) -> AlzinaResult<Vec<f32>> {
            let mut v = vec![0.0f32; 1024];
            // Simple hash: sum of byte values mod 100, normalised.
            let h: u64 = text.bytes().fold(0u64, |a, b| a.wrapping_add(b as u64));
            v[0] = (h % 100) as f32 / 100.0;
            v[1] = 1.0 - v[0]; // ensures non-zero norm
            Ok(v)
        }

        async fn embed_batch(
            &self,
            texts: &[String],
            task: EmbeddingTask,
        ) -> AlzinaResult<Vec<Vec<f32>>> {
            let mut out = Vec::new();
            for t in texts {
                out.push(self.embed(t, task).await?);
            }
            Ok(out)
        }

        fn dimensions(&self) -> usize {
            1024
        }
    }

    // ── Wiremock helpers ──────────────────────────────────────────────────────

    fn paper_json(id: &str, title: &str, arxiv: Option<&str>, citations: i64) -> serde_json::Value {
        let mut external_ids = json!({});
        if let Some(aid) = arxiv {
            external_ids["ArXiv"] = json!(aid);
        }
        json!({
            "paperId": id,
            "title": title,
            "abstract": format!("Abstract of {title}"),
            "year": 2022,
            "citationCount": citations,
            "referenceCount": 5,
            "authors": [{"name": "Test Author"}],
            "venue": "Test Venue",
            "externalIds": external_ids
        })
    }

    fn cfg_for_server(server: &MockServer) -> S2Config {
        S2Config {
            base_url: server.uri(),
            api_key: None,
            min_interval_ms: 0, // no delay in tests
            timeout_secs: 10,
            limit: 5,
        }
    }

    async fn make_pool_and_store_or_skip(
        tag: &str,
    ) -> Option<(sqlx::sqlite::SqlitePool, SqliteVecStore)> {
        let pool = in_memory_lit_pool().await.expect("in_memory_lit_pool");
        let store =
            SqliteVecStore::with_table_names(pool.clone(), 1024, "lit_vec0", "lit_chunks")
                .await
                .expect("with_table_names");
        if !store.is_enabled() {
            eprintln!("skipping lit_explore test {tag}: sqlite-vec extension not loaded");
            return None;
        }
        Some((pool, store))
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Cosine of identical vectors is 1.0; orthogonal vectors is 0.0.
    #[test]
    fn cosine_similarity_basic() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0f32, 1.0, 0.0];
        assert!((cosine(&a, &c)).abs() < 1e-6);
    }

    /// explore_from_query with a wiremock S2 server: visits seeds, keeps only
    /// candidates above min_relevance, ingests abstracts. Stats counts match.
    #[tokio::test]
    async fn explore_traverses_wiremock_fixtures() {
        let Some((pool, store)) = make_pool_and_store_or_skip("explore_basic").await else {
            return;
        };

        let server = MockServer::start().await;

        // Search response: 2 candidates (seeds = 1 after sort by citations).
        let search_resp = json!({
            "data": [
                paper_json("seed001", "Top Cited Seed Paper", None, 500),
                paper_json("seed002", "Lower Cited Seed", None, 10),
            ]
        });
        Mock::given(method("GET"))
            .and(path_regex("/paper/search.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(search_resp))
            .mount(&server)
            .await;

        // References of seed001: 3 candidates.
        let refs_resp = json!({
            "data": [
                {"citedPaper": paper_json("ref001", "Reference One", Some("1801.00001"), 100)},
                {"citedPaper": paper_json("ref002", "Reference Two", None, 50)},
                {"citedPaper": paper_json("ref003", "Reference Three", None, 25)},
            ]
        });
        Mock::given(method("GET"))
            .and(path_regex(".*/references.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(refs_resp))
            .mount(&server)
            .await;

        // Citations of seed001: 1 candidate.
        let cits_resp = json!({
            "data": [
                {"citingPaper": paper_json("cit001", "Citing Paper", None, 75)},
            ]
        });
        Mock::given(method("GET"))
            .and(path_regex(".*/citations.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(cits_resp))
            .mount(&server)
            .await;

        // Batch fetch: return all winners.
        let batch_resp = json!([
            paper_json("ref001", "Reference One", Some("1801.00001"), 100),
            paper_json("ref002", "Reference Two", None, 50),
            paper_json("ref003", "Reference Three", None, 25),
            paper_json("cit001", "Citing Paper", None, 75),
        ]);
        Mock::given(method("POST"))
            .and(path_regex(".*/paper/batch.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(batch_resp))
            .mount(&server)
            .await;

        let s2 = S2Client::with_config(cfg_for_server(&server), true).expect("S2Client");
        let gateway = Arc::new(LitGateway::new(5, 5, 20, false));
        let config = ExploreConfig {
            num_seeds: 1,
            max_depth: 1,
            papers_per_level: 10,
            min_relevance: 0.0, // accept all candidates
            candidates_per_edge: 10,
            seed_search_limit: 5,
        };

        let stats = explore_from_query(
            "test question",
            &config,
            &s2,
            &gateway,
            &pool,
            &store,
            &StubEmbedder,
        )
        .await;

        assert!(!stats.budget_exhausted, "should not exhaust budget");
        assert!(
            stats.papers_discovered >= 1,
            "at least the seed must be discovered"
        );
        assert_eq!(stats.seeds.len(), 1, "one seed selected");
        // Some abstracts should have been indexed (seed + winners).
        assert!(stats.abstracts_indexed >= 1, "at least seed abstract indexed");
        // S2 calls should be > 0 (search + references + citations + batch).
        assert!(stats.s2_calls > 0, "gateway calls must be > 0");
    }

    /// Budget exhaustion stops traversal cleanly, returns Ok(stats).
    #[tokio::test]
    async fn budget_exhaustion_stops_traversal() {
        let Some((pool, store)) = make_pool_and_store_or_skip("explore_budget").await else {
            return;
        };

        let server = MockServer::start().await;

        let search_resp = json!({
            "data": [paper_json("seedX", "Budget Test Seed", None, 100)]
        });
        Mock::given(method("GET"))
            .and(path_regex("/paper/search.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(search_resp))
            .mount(&server)
            .await;

        // references + citations: return some candidates.
        let refs_resp = json!({"data": [
            {"citedPaper": paper_json("r1", "Ref 1", None, 50)}
        ]});
        Mock::given(method("GET"))
            .and(path_regex(".*/references.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(refs_resp))
            .mount(&server)
            .await;
        let cits_resp = json!({"data": []});
        Mock::given(method("GET"))
            .and(path_regex(".*/citations.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(cits_resp))
            .mount(&server)
            .await;
        // Batch: also needs a mock.
        let batch_resp = json!([paper_json("r1", "Ref 1", None, 50)]);
        Mock::given(method("POST"))
            .and(path_regex(".*/paper/batch.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(batch_resp))
            .mount(&server)
            .await;

        let s2 = S2Client::with_config(cfg_for_server(&server), true).expect("S2Client");
        // Budget of 2: enough for search + one more call, then exhausts.
        let gateway = Arc::new(LitGateway::new(5, 5, 2, false));
        let config = ExploreConfig {
            num_seeds: 1,
            max_depth: 2,
            papers_per_level: 5,
            min_relevance: 0.0,
            candidates_per_edge: 5,
            seed_search_limit: 5,
        };

        let stats = explore_from_query(
            "budget test question",
            &config,
            &s2,
            &gateway,
            &pool,
            &store,
            &StubEmbedder,
        )
        .await;

        assert!(
            stats.budget_exhausted,
            "budget_exhausted must be true when budget runs out"
        );
        // Function must return Ok(stats) — no panic/error even with exhaustion.
        // (If it panicked, this test would fail.)
    }

    /// Second identical run: cache hits > 0, fewer gateway calls than first run.
    #[tokio::test]
    async fn second_run_hits_cache() {
        let Some((pool, store)) = make_pool_and_store_or_skip("explore_cache").await else {
            return;
        };

        let server = MockServer::start().await;

        let search_resp = json!({
            "data": [paper_json("cacheS1", "Cache Test Seed", None, 200)]
        });
        Mock::given(method("GET"))
            .and(path_regex("/paper/search.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(search_resp.clone()))
            .mount(&server)
            .await;

        let refs_resp = json!({"data": [
            {"citedPaper": paper_json("cacheR1", "Cache Ref 1", None, 30)}
        ]});
        Mock::given(method("GET"))
            .and(path_regex(".*/references.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(refs_resp))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(".*/citations.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":[]})))
            .mount(&server)
            .await;
        let batch_resp = json!([paper_json("cacheR1", "Cache Ref 1", None, 30)]);
        Mock::given(method("POST"))
            .and(path_regex(".*/paper/batch.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(batch_resp))
            .mount(&server)
            .await;

        let s2 = S2Client::with_config(cfg_for_server(&server), true).expect("S2Client");
        let gateway1 = Arc::new(LitGateway::new(5, 5, 20, false));
        let config = ExploreConfig {
            num_seeds: 1,
            max_depth: 1,
            papers_per_level: 5,
            min_relevance: 0.0,
            candidates_per_edge: 5,
            seed_search_limit: 5,
        };

        // First run: populates the cache.
        let stats1 = explore_from_query(
            "cache test question",
            &config,
            &s2,
            &gateway1,
            &pool,
            &store,
            &StubEmbedder,
        )
        .await;
        let first_calls = stats1.s2_calls;

        // Second run: same pool (cache populated), new gateway.
        let gateway2 = Arc::new(LitGateway::new(5, 5, 20, false));
        let stats2 = explore_from_query(
            "cache test question",
            &config,
            &s2,
            &gateway2,
            &pool,
            &store,
            &StubEmbedder,
        )
        .await;

        assert!(
            stats2.s2_cache_hits > 0,
            "second run must have > 0 cache hits; got hits={} misses={}",
            stats2.s2_cache_hits, stats2.s2_cache_misses
        );
        assert!(
            stats2.s2_calls <= first_calls,
            "second run must make <= gateway calls than first; first={first_calls} second={}",
            stats2.s2_calls
        );
    }
}
