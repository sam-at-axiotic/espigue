//! `BackfillJob` — historical-entry catch-up for the vector index.
//!
//! Phase 2 §5.5 establishes that the write path indexes synchronously to
//! FTS5 and asynchronously to the vector store via [`SearchIndexer::index_entry`]
//! (fire-and-forget). Async failures are swallowed with a `warn!`; reconciliation
//! is this job's responsibility.
//!
//! `BackfillJob` is the vector-side analogue of
//! [`alzina_memory::schema::backfill_fts`] (the FTS5 self-heal that Phase 1 added
//! in A4). For any row in `daily_entries`, `weave_records`, or `semantic_entries`
//! that does NOT have a corresponding `vec_index` entry, this job calls
//! [`SearchIndexer::index_entry_blocking`] to embed it via Jina (with cache) and
//! upsert it into `vec_entries` / `vec_index`.
//!
//! # Use cases
//! - First Jina API key rollout — vector index starts empty.
//! - After a fire-and-forget indexer failure (Phase 1 already swallows them with
//!   a `warn!`; backfill reconciles).
//! - After a model change (rebuild the index from scratch).
//!
//! # Not in scope (Phase 2)
//! - Learning entries (file-backed; need a different scan strategy). Skipped
//!   with a `tracing::info!` and deferred to Phase 3.
//!
//! # Production write paths
//! Production code MUST use [`SearchIndexer::index_entry`] (fire-and-forget) on
//! the write path. `BackfillJob` is only for *historical reconciliation* — it
//! is synchronous, propagates errors per entry, and is not suitable for
//! steady-state indexing.
//!
//! # AC-1
//! Per-entry errors are logged at `warn!` and counted in [`BackfillReport`];
//! they do NOT halt the job. The whole backfill is idempotent — safe to re-run.

use std::sync::Arc;

use alzina_core::{AlzinaError, AlzinaResult, VectorMetadata};
use sqlx::SqlitePool;

use crate::indexer::SearchIndexer;

/// Result of a backfill run.
///
/// Distinct from [`alzina_memory::schema::BackfillReport`] (the FTS5 self-heal
/// counter). Both names are kept short on purpose; namespace separation does
/// the disambiguation work.
#[derive(Debug, Clone, Default)]
pub struct BackfillReport {
    pub daily_indexed: usize,
    pub daily_skipped: usize,
    pub daily_errored: usize,
    pub weave_indexed: usize,
    pub weave_skipped: usize,
    pub weave_errored: usize,
    pub semantic_indexed: usize,
    pub semantic_skipped: usize,
    pub semantic_errored: usize,
    pub total_indexed: usize,
}

/// Configuration for a backfill run.
#[derive(Debug, Clone)]
pub struct BackfillConfig {
    /// Number of entries processed before the job sleeps to respect Jina rate
    /// limits. Synthesis: "Respects Jina rate limits (configurable delay between
    /// batches)."
    pub batch_size: usize,
    /// Delay applied between batches; `0` disables sleeping entirely.
    pub batch_delay_ms: u64,
    /// Log progress every N entries within a single source-type pass.
    pub progress_every: usize,
}

impl Default for BackfillConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            batch_delay_ms: 0,
            progress_every: 100,
        }
    }
}

/// Synchronous historical reconciler for the vector index.
pub struct BackfillJob {
    pool: SqlitePool,
    indexer: Arc<SearchIndexer>,
    config: BackfillConfig,
}

impl BackfillJob {
    /// Construct a job with explicit config. Use [`BackfillConfig::default`] for
    /// sensible defaults.
    pub fn new(pool: SqlitePool, indexer: Arc<SearchIndexer>, config: BackfillConfig) -> Self {
        Self {
            pool,
            indexer,
            config,
        }
    }

    /// Run the backfill. Idempotent — safe to re-run; already-indexed rows are
    /// filtered out via `LEFT JOIN vec_index ... WHERE vec_index.rowid IS NULL`.
    pub async fn run(&self) -> AlzinaResult<BackfillReport> {
        let mut report = BackfillReport::default();

        self.run_daily(&mut report).await?;
        self.run_weave(&mut report).await?;
        self.run_semantic(&mut report).await?;

        // Learnings are file-backed; deferred to Phase 3.
        tracing::info!(
            "backfill: learnings entries are file-backed; deferred to Phase 3 — Task 2.9 follow-up"
        );

        report.total_indexed =
            report.daily_indexed + report.weave_indexed + report.semantic_indexed;
        tracing::info!(report = ?report, "backfill complete");
        Ok(report)
    }

    /// Backfill `daily_entries` rows that lack a `vec_index` entry.
    async fn run_daily(&self, report: &mut BackfillReport) -> AlzinaResult<()> {
        let rows: Vec<DailyRow> = sqlx::query_as::<_, DailyRow>(
            "SELECT de.id, de.content, de.source_agent, de.date, de.weave_id, de.section \
             FROM daily_entries de \
             LEFT JOIN vec_index vi \
               ON vi.source_type = 'daily' AND vi.source_id = de.id \
             WHERE vi.rowid IS NULL \
             ORDER BY de.timestamp ASC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AlzinaError::Memory(format!("backfill daily query: {e}")))?;

        let total = rows.len();
        for (i, row) in rows.iter().enumerate() {
            let metadata = VectorMetadata {
                source_type: "daily".into(),
                source_id: row.id.clone(),
                chunk_index: 0,
                content_preview: truncate_preview(&row.content, 400),
                source_agent: row.source_agent.clone(),
                source_date: Some(row.date.clone()),
                weave_id: row.weave_id.clone(),
                section: Some(row.section.clone()),
                // Phase 1 B4: weave_id routes into the domain column so
                // domain-scoped queries work without a structured taxonomy.
                domain: row.weave_id.clone(),
                indexed_at: String::new(), // stamped by the indexer
            };
            match self
                .indexer
                .index_entry_blocking(&row.content, metadata)
                .await
            {
                Ok(_) => report.daily_indexed += 1,
                Err(AlzinaError::Search(d)) if d.degraded => {
                    tracing::warn!(
                        source_id = %row.id,
                        reason = ?d.degradation_reason,
                        "backfill daily entry skipped (degraded)"
                    );
                    report.daily_errored += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        source_id = %row.id,
                        error = %e,
                        "backfill daily entry errored"
                    );
                    report.daily_errored += 1;
                }
            }

            self.maybe_throttle(i).await;
            self.maybe_log_progress(i, total, "daily");
        }
        Ok(())
    }

    /// Backfill `weave_records` rows that lack a `vec_index` entry.
    async fn run_weave(&self, report: &mut BackfillReport) -> AlzinaResult<()> {
        let rows: Vec<WeaveRow> = sqlx::query_as::<_, WeaveRow>(
            "SELECT wr.id, wr.label, wr.outcome, wr.created_at, wr.dod_met \
             FROM weave_records wr \
             LEFT JOIN vec_index vi \
               ON vi.source_type = 'weave' AND vi.source_id = wr.id \
             WHERE vi.rowid IS NULL \
             ORDER BY wr.created_at ASC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AlzinaError::Memory(format!("backfill weave query: {e}")))?;

        let total = rows.len();
        for (i, row) in rows.iter().enumerate() {
            // Mirrors `alzina_memory::schema::backfill_fts`: prefer outcome,
            // fall back to label.
            let mut content = row.outcome.clone().unwrap_or_else(|| row.label.clone());
            // Phase 1 B1 / W6 semantic: when the DoD was explicitly NOT met,
            // tag the searchable text so abandoned weaves surface distinctly.
            if matches!(row.dod_met, Some(false)) {
                content = format!("[abandoned-DoD-fail] {content}");
            }
            // Phase 1 B1: source_date is the date portion of created_at.
            let source_date = if row.created_at.len() >= 10 {
                row.created_at[..10].to_string()
            } else {
                row.created_at.clone()
            };

            let metadata = VectorMetadata {
                source_type: "weave".into(),
                source_id: row.id.clone(),
                chunk_index: 0,
                content_preview: truncate_preview(&content, 400),
                source_agent: None,
                source_date: Some(source_date),
                weave_id: Some(row.id.clone()),
                section: None,
                // Phase 1 doesn't have a structured domain on weaves — leave None.
                domain: None,
                indexed_at: String::new(),
            };
            match self.indexer.index_entry_blocking(&content, metadata).await {
                Ok(_) => report.weave_indexed += 1,
                Err(AlzinaError::Search(d)) if d.degraded => {
                    tracing::warn!(
                        source_id = %row.id,
                        reason = ?d.degradation_reason,
                        "backfill weave entry skipped (degraded)"
                    );
                    report.weave_errored += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        source_id = %row.id,
                        error = %e,
                        "backfill weave entry errored"
                    );
                    report.weave_errored += 1;
                }
            }

            self.maybe_throttle(i).await;
            self.maybe_log_progress(i, total, "weave");
        }
        Ok(())
    }

    /// Backfill `semantic_entries` rows that lack a `vec_index` entry.
    async fn run_semantic(&self, report: &mut BackfillReport) -> AlzinaResult<()> {
        let rows: Vec<SemanticRow> = sqlx::query_as::<_, SemanticRow>(
            "SELECT se.id, se.title, se.description, se.updated_at \
             FROM semantic_entries se \
             LEFT JOIN vec_index vi \
               ON vi.source_type = 'semantic' AND vi.source_id = se.id \
             WHERE vi.rowid IS NULL \
             ORDER BY se.updated_at ASC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AlzinaError::Memory(format!("backfill semantic query: {e}")))?;

        let total = rows.len();
        for (i, row) in rows.iter().enumerate() {
            let content = format!("{}: {}", row.title, row.description);
            // source_date is the date portion of updated_at.
            let source_date = if row.updated_at.len() >= 10 {
                row.updated_at[..10].to_string()
            } else {
                row.updated_at.clone()
            };

            let metadata = VectorMetadata {
                source_type: "semantic".into(),
                source_id: row.id.clone(),
                chunk_index: 0,
                content_preview: truncate_preview(&content, 400),
                source_agent: None,
                source_date: Some(source_date),
                weave_id: None,
                section: None,
                domain: None,
                indexed_at: String::new(),
            };
            match self.indexer.index_entry_blocking(&content, metadata).await {
                Ok(_) => report.semantic_indexed += 1,
                Err(AlzinaError::Search(d)) if d.degraded => {
                    tracing::warn!(
                        source_id = %row.id,
                        reason = ?d.degradation_reason,
                        "backfill semantic entry skipped (degraded)"
                    );
                    report.semantic_errored += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        source_id = %row.id,
                        error = %e,
                        "backfill semantic entry errored"
                    );
                    report.semantic_errored += 1;
                }
            }

            self.maybe_throttle(i).await;
            self.maybe_log_progress(i, total, "semantic");
        }
        Ok(())
    }

    /// Sleep after every `batch_size` entries when `batch_delay_ms > 0`.
    async fn maybe_throttle(&self, i: usize) {
        if self.config.batch_size > 0
            && (i + 1) % self.config.batch_size == 0
            && self.config.batch_delay_ms > 0
        {
            tokio::time::sleep(std::time::Duration::from_millis(self.config.batch_delay_ms)).await;
        }
    }

    fn maybe_log_progress(&self, i: usize, total: usize, source: &str) {
        if self.config.progress_every > 0 && (i + 1) % self.config.progress_every == 0 {
            tracing::info!(progress = (i + 1), total, source, "backfill progress");
        }
    }
}

#[derive(sqlx::FromRow)]
struct DailyRow {
    id: String,
    content: String,
    source_agent: Option<String>,
    date: String,
    weave_id: Option<String>,
    section: String,
}

#[derive(sqlx::FromRow)]
struct WeaveRow {
    id: String,
    label: String,
    outcome: Option<String>,
    created_at: String,
    dod_met: Option<bool>,
}

#[derive(sqlx::FromRow)]
struct SemanticRow {
    id: String,
    title: String,
    description: String,
    updated_at: String,
}

/// Truncate `s` at most `max_chars` chars, appending `…` when truncation
/// occurred. Operates on `char` boundaries so we never split a UTF-8
/// codepoint.
fn truncate_preview(s: &str, max_chars: usize) -> String {
    let collected: String = s.chars().take(max_chars).collect();
    if collected.chars().count() < s.chars().count() {
        format!("{collected}\u{2026}")
    } else {
        collected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed_cache::EmbeddingCache;
    use crate::schema::in_memory_pool_with_search_schema;
    use alzina_core::error::SearchDetail;
    use alzina_core::search::{
        EmbeddingService, EmbeddingTask, VectorFilters, VectorHit, VectorStore,
    };
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Stub embedder — deterministic 4-dim vector. Optionally fails on the
    /// Nth call (1-indexed) to simulate transient errors mid-run.
    struct StubEmbedder {
        dim: usize,
        calls: AtomicUsize,
        fail_on_call: Option<usize>,
    }

    impl StubEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                calls: AtomicUsize::new(0),
                fail_on_call: None,
            }
        }
        fn fail_on(dim: usize, n: usize) -> Self {
            Self {
                dim,
                calls: AtomicUsize::new(0),
                fail_on_call: Some(n),
            }
        }
    }

    #[async_trait]
    impl EmbeddingService for StubEmbedder {
        async fn embed(&self, _text: &str, _task: EmbeddingTask) -> AlzinaResult<Vec<f32>> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if Some(n) == self.fail_on_call {
                return Err(AlzinaError::Search(SearchDetail {
                    message: "stub embed failure".into(),
                    degraded: true,
                    degradation_reason: Some("stub failure".into()),
                }));
            }
            Ok(vec![0.25_f32; self.dim])
        }
        async fn embed_batch(
            &self,
            texts: &[String],
            task: EmbeddingTask,
        ) -> AlzinaResult<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed(t, task).await?);
            }
            Ok(out)
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
    }

    /// Stub vector store — pushes inserts into vec_index for backfill skip
    /// queries to find them. Does NOT touch vec_entries (sqlite-vec extension
    /// may or may not be loaded; backfill correctness only depends on the
    /// metadata sidecar).
    struct StubVecStore {
        pool: SqlitePool,
        next_rowid: AtomicUsize,
    }

    impl StubVecStore {
        fn new(pool: SqlitePool) -> Self {
            Self {
                pool,
                next_rowid: AtomicUsize::new(1),
            }
        }
    }

    #[async_trait]
    impl VectorStore for StubVecStore {
        async fn insert(&self, _vector: &[f32], metadata: VectorMetadata) -> AlzinaResult<i64> {
            let rowid = self.next_rowid.fetch_add(1, Ordering::SeqCst) as i64;
            // Mirror real SqliteVecStore upsert discipline on vec_index so
            // re-runs deduplicate cleanly.
            sqlx::query(
                "DELETE FROM vec_index \
                 WHERE source_type = ? AND source_id = ? AND chunk_index = ?",
            )
            .bind(&metadata.source_type)
            .bind(&metadata.source_id)
            .bind(metadata.chunk_index)
            .execute(&self.pool)
            .await
            .map_err(|e| AlzinaError::Memory(format!("stub vec_index delete: {e}")))?;

            sqlx::query(
                "INSERT INTO vec_index (\
                    rowid, source_type, source_id, chunk_index, content_preview, \
                    source_agent, source_date, weave_id, section, domain, indexed_at\
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(rowid)
            .bind(&metadata.source_type)
            .bind(&metadata.source_id)
            .bind(metadata.chunk_index)
            .bind(&metadata.content_preview)
            .bind(metadata.source_agent.as_deref())
            .bind(metadata.source_date.as_deref())
            .bind(metadata.weave_id.as_deref())
            .bind(metadata.section.as_deref())
            .bind(metadata.domain.as_deref())
            .bind(&metadata.indexed_at)
            .execute(&self.pool)
            .await
            .map_err(|e| AlzinaError::Memory(format!("stub vec_index insert: {e}")))?;

            Ok(rowid)
        }
        async fn search(
            &self,
            _vector: &[f32],
            _top_k: usize,
            _filters: &VectorFilters,
        ) -> AlzinaResult<Vec<VectorHit>> {
            Ok(vec![])
        }
        async fn delete_by_source(
            &self,
            _source_type: &str,
            _source_id: &str,
        ) -> AlzinaResult<usize> {
            Ok(0)
        }
    }

    async fn build_job(
        embedder: Arc<dyn EmbeddingService>,
        config: BackfillConfig,
    ) -> (BackfillJob, SqlitePool) {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let store: Arc<dyn VectorStore> = Arc::new(StubVecStore::new(pool.clone()));
        let cache = Arc::new(EmbeddingCache::new(pool.clone()));
        let indexer = Arc::new(SearchIndexer::new(
            embedder,
            store,
            cache,
            "jina-embeddings-v3".into(),
        ));
        let job = BackfillJob::new(pool.clone(), indexer, config);
        (job, pool)
    }

    async fn insert_daily(pool: &SqlitePool, id: &str, content: &str, date: &str, ts: &str) {
        sqlx::query(
            "INSERT INTO daily_entries (id, date, section, timestamp, content, source_agent) \
             VALUES (?, ?, 'body', ?, ?, 'smidr')",
        )
        .bind(id)
        .bind(date)
        .bind(ts)
        .bind(content)
        .execute(pool)
        .await
        .unwrap();
    }

    async fn insert_weave(
        pool: &SqlitePool,
        id: &str,
        label: &str,
        outcome: Option<&str>,
        status: &str,
        dod_met: Option<bool>,
    ) {
        sqlx::query(
            "INSERT INTO weave_records (id, label, status, category, created_at, updated_at, outcome, dod_met) \
             VALUES (?, ?, ?, 'task', '2026-04-29T00:00:00Z', '2026-04-29T00:00:00Z', ?, ?)",
        )
        .bind(id)
        .bind(label)
        .bind(status)
        .bind(outcome)
        .bind(dod_met.map(|b| if b { 1 } else { 0 }))
        .execute(pool)
        .await
        .unwrap();
    }

    async fn insert_semantic(pool: &SqlitePool, id: &str, slug: &str, title: &str, desc: &str) {
        sqlx::query(
            "INSERT INTO semantic_entries (id, slug, title, stype, confidence, description, source_dates, source_refs, related, created_at, updated_at) \
             VALUES (?, ?, ?, 'note', 'medium', ?, '[]', '[]', NULL, '2026-04-29T00:00:00Z', '2026-04-29T00:00:00Z')",
        )
        .bind(id)
        .bind(slug)
        .bind(title)
        .bind(desc)
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn backfill_indexes_unindexed_daily_entries() {
        let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder::new(4));
        let (job, pool) = build_job(embedder, BackfillConfig::default()).await;

        insert_daily(&pool, "d1", "alpha", "2026-04-29", "2026-04-29T01:00:00Z").await;
        insert_daily(&pool, "d2", "beta", "2026-04-29", "2026-04-29T02:00:00Z").await;
        insert_daily(&pool, "d3", "gamma", "2026-04-29", "2026-04-29T03:00:00Z").await;

        let report = job.run().await.unwrap();
        assert_eq!(report.daily_indexed, 3);
        assert_eq!(report.daily_errored, 0);
        assert_eq!(report.total_indexed, 3);

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM vec_index WHERE source_type = 'daily'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 3);
    }

    #[tokio::test]
    async fn backfill_skips_already_indexed_entries() {
        let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder::new(4));
        let (job, pool) = build_job(embedder, BackfillConfig::default()).await;

        insert_daily(&pool, "d1", "alpha", "2026-04-29", "2026-04-29T01:00:00Z").await;
        insert_daily(&pool, "d2", "beta", "2026-04-29", "2026-04-29T02:00:00Z").await;
        insert_daily(&pool, "d3", "gamma", "2026-04-29", "2026-04-29T03:00:00Z").await;

        // Pre-mark d2 as already indexed.
        sqlx::query(
            "INSERT INTO vec_index (rowid, source_type, source_id, chunk_index, content_preview, indexed_at) \
             VALUES (9999, 'daily', 'd2', 0, 'beta', '2026-04-29T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let report = job.run().await.unwrap();
        assert_eq!(
            report.daily_indexed, 2,
            "the pre-indexed row must be skipped"
        );
    }

    #[tokio::test]
    async fn backfill_idempotent_on_rerun() {
        let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder::new(4));
        let (job, pool) = build_job(embedder, BackfillConfig::default()).await;

        insert_daily(&pool, "d1", "alpha", "2026-04-29", "2026-04-29T01:00:00Z").await;
        insert_daily(&pool, "d2", "beta", "2026-04-29", "2026-04-29T02:00:00Z").await;

        let r1 = job.run().await.unwrap();
        assert_eq!(r1.daily_indexed, 2);

        let r2 = job.run().await.unwrap();
        assert_eq!(r2.daily_indexed, 0, "second run is a no-op");

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM vec_index WHERE source_type = 'daily'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 2, "no duplicate vec_index rows on re-run");
    }

    #[tokio::test]
    async fn backfill_handles_weave_records() {
        let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder::new(4));
        let (job, pool) = build_job(embedder, BackfillConfig::default()).await;

        insert_weave(
            &pool,
            "w1",
            "the label",
            Some("shipped feature"),
            "closed",
            None,
        )
        .await;

        let report = job.run().await.unwrap();
        assert_eq!(report.weave_indexed, 1);

        let preview: (String,) = sqlx::query_as(
            "SELECT content_preview FROM vec_index WHERE source_type='weave' AND source_id='w1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(preview.0, "shipped feature");

        let date: (String,) = sqlx::query_as(
            "SELECT source_date FROM vec_index WHERE source_type='weave' AND source_id='w1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(date.0, "2026-04-29");
    }

    #[tokio::test]
    async fn backfill_weave_marks_abandoned_when_dod_failed() {
        let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder::new(4));
        let (job, pool) = build_job(embedder, BackfillConfig::default()).await;

        insert_weave(
            &pool,
            "w-bad",
            "the label",
            Some("did not ship"),
            "closed",
            Some(false),
        )
        .await;

        let report = job.run().await.unwrap();
        assert_eq!(report.weave_indexed, 1);

        let preview: (String,) =
            sqlx::query_as("SELECT content_preview FROM vec_index WHERE source_id='w-bad'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(
            preview.0.starts_with("[abandoned-DoD-fail] "),
            "expected abandoned tag, got {:?}",
            preview.0
        );
    }

    #[tokio::test]
    async fn backfill_handles_semantic_entries() {
        let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder::new(4));
        let (job, pool) = build_job(embedder, BackfillConfig::default()).await;

        insert_semantic(&pool, "s1", "fox-pattern", "Fox", "a clever animal").await;

        let report = job.run().await.unwrap();
        assert_eq!(report.semantic_indexed, 1);

        let preview: (String,) = sqlx::query_as(
            "SELECT content_preview FROM vec_index WHERE source_type='semantic' AND source_id='s1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(preview.0, "Fox: a clever animal");

        let date: (String,) = sqlx::query_as(
            "SELECT source_date FROM vec_index WHERE source_type='semantic' AND source_id='s1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(date.0, "2026-04-29");
    }

    #[tokio::test]
    async fn backfill_continues_on_per_entry_error() {
        // Fail the second embed call (1-indexed) — but the first call also hits
        // the cache miss path, so call #2 corresponds to the second daily row.
        let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder::fail_on(4, 2));
        let (job, pool) = build_job(embedder, BackfillConfig::default()).await;

        insert_daily(&pool, "d1", "alpha", "2026-04-29", "2026-04-29T01:00:00Z").await;
        insert_daily(&pool, "d2", "beta", "2026-04-29", "2026-04-29T02:00:00Z").await;
        insert_daily(&pool, "d3", "gamma", "2026-04-29", "2026-04-29T03:00:00Z").await;

        let report = job.run().await.unwrap();
        assert_eq!(report.daily_indexed, 2, "two entries succeeded");
        assert_eq!(report.daily_errored, 1, "one entry errored");
        // Job MUST NOT halt — the report.total_indexed reflects partial success.
        assert_eq!(report.total_indexed, 2);
    }

    #[tokio::test]
    async fn backfill_respects_batch_delay() {
        let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder::new(4));
        let config = BackfillConfig {
            batch_size: 2,
            batch_delay_ms: 50,
            progress_every: 100,
        };
        let (job, pool) = build_job(embedder, config).await;

        for i in 1..=4 {
            insert_daily(
                &pool,
                &format!("d{i}"),
                "content",
                "2026-04-29",
                &format!("2026-04-29T0{i}:00:00Z"),
            )
            .await;
        }

        let started = std::time::Instant::now();
        let report = job.run().await.unwrap();
        let elapsed = started.elapsed();

        assert_eq!(report.daily_indexed, 4);
        // 4 entries with batch_size=2 + delay=50ms triggers 2 sleeps (after #2
        // and #4). Allow generous slack for test-runner jitter.
        assert!(
            elapsed >= std::time::Duration::from_millis(100),
            "expected at least 100ms elapsed (2 batch delays), got {elapsed:?}"
        );
    }
}
