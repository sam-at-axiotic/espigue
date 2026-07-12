//! Stage-level checkpoint store for long synthesis runs.
//!
//! Checkpoint/resume foundation (harden/checkpoint-resume B1): a synthesis
//! run that dies mid-flight (crash, kill, OOM) loses hours of paid model
//! calls. This store persists two things to the literature DB so a rerun can
//! pick up at the last completed stage instead of from zero:
//!
//! - **`synthesis_runs`** — one row per run: the settings needed to resume
//!   under identical conditions (question, profile, models, embedding config,
//!   panel) plus a status lifecycle (`running | complete | failed`). Rows are
//!   kept forever as a run ledger.
//! - **`run_checkpoints`** — the latest serialised artifact per (run_id,
//!   stage). `INSERT OR REPLACE` on the composite primary key gives
//!   latest-wins with no history; rows are deleted when the run completes.
//!
//! Write path is direct parameterised SQLite, same as `bib_store.rs` — all
//! values go through `.bind()`, never string interpolation.
//!
//! The orchestration crate consumes the `CheckpointStore` trait (dep
//! direction: espigue → orchestration → search → base).

use async_trait::async_trait;
use sqlx::sqlite::SqlitePool;

use base::error::{AlzinaError, AlzinaResult, SearchDetail};

/// Helper: map sqlx errors into `AlzinaError::Search` with degradation.
fn checkpoint_err(message: impl Into<String>, reason: impl Into<String>) -> AlzinaError {
    AlzinaError::Search(SearchDetail {
        message: message.into(),
        degraded: true,
        degradation_reason: Some(reason.into()),
    })
}

// ── RunRecord ─────────────────────────────────────────────────────────────────

/// One synthesis run's restart-survivable metadata — a row of `synthesis_runs`.
///
/// Fields mirror the table columns. JSON-array fields (`seed_papers`,
/// `panel_source_ids`) are stored as their serialised strings; the caller
/// owns (de)serialisation so this store stays schema-agnostic about payloads.
#[derive(Debug, Clone)]
pub struct RunRecord {
    /// Unique run identifier.
    pub run_id: String,
    /// The research question text.
    pub question: String,
    /// Stable question identifier (for grouping reruns of the same question).
    pub question_id: String,
    /// Prompt profile string, e.g. "v3/lit-review-long" — the same strings
    /// `parse_prompt_profile` accepts.
    pub profile: String,
    /// Primary model identifier.
    pub model: String,
    /// Merger model identifier, if a distinct one is configured.
    pub merger_model: Option<String>,
    /// Retrieval top-k.
    pub top_k: i64,
    /// Retrieval scope string.
    pub scope: String,
    /// JSON array of seed paper ids.
    pub seed_papers: String,
    /// Embedding model identifier.
    pub embedding_model: String,
    /// Embedding vector width.
    pub embedding_dim: i64,
    /// JSON array of panel source ids.
    pub panel_source_ids: String,
    /// Code version that started the run (resume refuses a mismatch).
    pub code_version: String,
    /// Lifecycle status: `running | complete | failed`.
    pub status: String,
    /// RFC 3339 UTC timestamp the run started.
    pub started_at: String,
    /// RFC 3339 UTC timestamp of the last status change.
    pub updated_at: String,
}

// ── StageCheckpoint ───────────────────────────────────────────────────────────

/// The latest persisted artifact for one (run, stage) — a row of
/// `run_checkpoints`.
#[derive(Debug, Clone)]
pub struct StageCheckpoint {
    /// Run this checkpoint belongs to.
    pub run_id: String,
    /// Stage name: "graph" | "synthesis" | "narrative".
    pub stage: String,
    /// JSON-serialised stage artifact.
    pub payload: String,
    /// Payload encoding; currently always "json".
    pub format: String,
    /// Code version that wrote the checkpoint.
    pub code_version: String,
    /// RFC 3339 UTC timestamp the checkpoint was written.
    pub created_at: String,
}

// ── CheckpointStore trait ─────────────────────────────────────────────────────

/// Persist and recover stage-level synthesis run state.
///
/// `NoopCheckpointStore` is used for tests and runs without checkpointing.
/// `SqliteCheckpointStore` is used in production against the literature DB
/// pool (the same `espigue.db` that `lit_migrate` targets).
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Save (or overwrite) the artifact for one stage of a run.
    ///
    /// `INSERT OR REPLACE` on `(run_id, stage)` — latest wins, no history.
    async fn save_stage(
        &self,
        run_id: &str,
        stage: &str,
        payload: &str,
        code_version: &str,
    ) -> AlzinaResult<()>;

    /// Record a run's metadata at start. Overwrites any prior row with the
    /// same `run_id` (a deliberate restart of the same run id resets it).
    async fn record_run_start(&self, run: &RunRecord) -> AlzinaResult<()>;

    /// Load a run's metadata. `None` when the run id is unknown.
    async fn load_run(&self, run_id: &str) -> AlzinaResult<Option<RunRecord>>;

    /// Load the latest checkpoint for one stage. `None` when absent.
    async fn load_stage(&self, run_id: &str, stage: &str)
        -> AlzinaResult<Option<StageCheckpoint>>;

    /// Set a run's status (`running | complete | failed`) and bump
    /// `updated_at`.
    async fn mark_run_status(&self, run_id: &str, status: &str) -> AlzinaResult<()>;

    /// Delete all checkpoint rows for a run. The `synthesis_runs` row is
    /// kept — the run ledger survives checkpoint cleanup.
    async fn delete_checkpoints(&self, run_id: &str) -> AlzinaResult<()>;

    /// List runs whose status is not 'complete', most recently updated first.
    /// These are the candidates a resume command can offer.
    async fn list_resumable_runs(&self) -> AlzinaResult<Vec<RunRecord>>;
}

// ── NoopCheckpointStore ───────────────────────────────────────────────────────

/// No-op implementation — for tests and runs without checkpointing.
///
/// Mirrors `NoopBibliographyStore` in `bib_store.rs`. Every write succeeds
/// silently; every load returns `None` / empty.
pub struct NoopCheckpointStore;

#[async_trait]
impl CheckpointStore for NoopCheckpointStore {
    async fn save_stage(
        &self,
        _run_id: &str,
        _stage: &str,
        _payload: &str,
        _code_version: &str,
    ) -> AlzinaResult<()> {
        Ok(())
    }

    async fn record_run_start(&self, _run: &RunRecord) -> AlzinaResult<()> {
        Ok(())
    }

    async fn load_run(&self, _run_id: &str) -> AlzinaResult<Option<RunRecord>> {
        Ok(None)
    }

    async fn load_stage(
        &self,
        _run_id: &str,
        _stage: &str,
    ) -> AlzinaResult<Option<StageCheckpoint>> {
        Ok(None)
    }

    async fn mark_run_status(&self, _run_id: &str, _status: &str) -> AlzinaResult<()> {
        Ok(())
    }

    async fn delete_checkpoints(&self, _run_id: &str) -> AlzinaResult<()> {
        Ok(())
    }

    async fn list_resumable_runs(&self) -> AlzinaResult<Vec<RunRecord>> {
        Ok(Vec::new())
    }
}

// ── SqliteCheckpointStore ─────────────────────────────────────────────────────

/// SQLite-backed checkpoint store.
///
/// Uses parameterised `.bind()` SQL throughout — no value is interpolated
/// into the SQL string (same pattern as `SqliteBibliographyStore`).
pub struct SqliteCheckpointStore {
    pool: SqlitePool,
}

impl SqliteCheckpointStore {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

/// Column list shared by every `synthesis_runs` SELECT, in `RunRecord`
/// field order.
const RUN_COLUMNS: &str = "run_id, question, question_id, profile, model, merger_model, \
     top_k, scope, seed_papers, embedding_model, embedding_dim, \
     panel_source_ids, code_version, status, started_at, updated_at";

/// Row tuple matching `RUN_COLUMNS` order.
type RunRow = (
    String,         // run_id
    String,         // question
    String,         // question_id
    String,         // profile
    String,         // model
    Option<String>, // merger_model
    i64,            // top_k
    String,         // scope
    String,         // seed_papers
    String,         // embedding_model
    i64,            // embedding_dim
    String,         // panel_source_ids
    String,         // code_version
    String,         // status
    String,         // started_at
    String,         // updated_at
);

fn run_record_from_row(row: RunRow) -> RunRecord {
    RunRecord {
        run_id: row.0,
        question: row.1,
        question_id: row.2,
        profile: row.3,
        model: row.4,
        merger_model: row.5,
        top_k: row.6,
        scope: row.7,
        seed_papers: row.8,
        embedding_model: row.9,
        embedding_dim: row.10,
        panel_source_ids: row.11,
        code_version: row.12,
        status: row.13,
        started_at: row.14,
        updated_at: row.15,
    }
}

#[async_trait]
impl CheckpointStore for SqliteCheckpointStore {
    async fn save_stage(
        &self,
        run_id: &str,
        stage: &str,
        payload: &str,
        code_version: &str,
    ) -> AlzinaResult<()> {
        let created_at = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT OR REPLACE INTO run_checkpoints \
             (run_id, stage, payload, format, code_version, created_at) \
             VALUES (?, ?, ?, 'json', ?, ?)",
        )
        .bind(run_id)
        .bind(stage)
        .bind(payload)
        .bind(code_version)
        .bind(&created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| {
            checkpoint_err(
                format!("checkpoint_store save_stage: {e}"),
                format!("checkpoint save: {e}"),
            )
        })?;
        Ok(())
    }

    async fn record_run_start(&self, run: &RunRecord) -> AlzinaResult<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO synthesis_runs \
             (run_id, question, question_id, profile, model, merger_model, \
              top_k, scope, seed_papers, embedding_model, embedding_dim, \
              panel_source_ids, code_version, status, started_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&run.run_id)
        .bind(&run.question)
        .bind(&run.question_id)
        .bind(&run.profile)
        .bind(&run.model)
        .bind(&run.merger_model)
        .bind(run.top_k)
        .bind(&run.scope)
        .bind(&run.seed_papers)
        .bind(&run.embedding_model)
        .bind(run.embedding_dim)
        .bind(&run.panel_source_ids)
        .bind(&run.code_version)
        .bind(&run.status)
        .bind(&run.started_at)
        .bind(&run.updated_at)
        .execute(&self.pool)
        .await
        .map_err(|e| {
            checkpoint_err(
                format!("checkpoint_store record_run_start: {e}"),
                format!("run start insert: {e}"),
            )
        })?;
        Ok(())
    }

    async fn load_run(&self, run_id: &str) -> AlzinaResult<Option<RunRecord>> {
        let sql = format!("SELECT {RUN_COLUMNS} FROM synthesis_runs WHERE run_id = ?");
        let row: Option<RunRow> = sqlx::query_as(&sql)
            .bind(run_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| {
                checkpoint_err(
                    format!("checkpoint_store load_run: {e}"),
                    format!("run load: {e}"),
                )
            })?;
        Ok(row.map(run_record_from_row))
    }

    async fn load_stage(
        &self,
        run_id: &str,
        stage: &str,
    ) -> AlzinaResult<Option<StageCheckpoint>> {
        let row: Option<(String, String, String, String, String, String)> = sqlx::query_as(
            "SELECT run_id, stage, payload, format, code_version, created_at \
             FROM run_checkpoints WHERE run_id = ? AND stage = ?",
        )
        .bind(run_id)
        .bind(stage)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| {
            checkpoint_err(
                format!("checkpoint_store load_stage: {e}"),
                format!("checkpoint load: {e}"),
            )
        })?;
        Ok(row.map(
            |(run_id, stage, payload, format, code_version, created_at)| StageCheckpoint {
                run_id,
                stage,
                payload,
                format,
                code_version,
                created_at,
            },
        ))
    }

    async fn mark_run_status(&self, run_id: &str, status: &str) -> AlzinaResult<()> {
        let updated_at = chrono::Utc::now().to_rfc3339();
        sqlx::query("UPDATE synthesis_runs SET status = ?, updated_at = ? WHERE run_id = ?")
            .bind(status)
            .bind(&updated_at)
            .bind(run_id)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                checkpoint_err(
                    format!("checkpoint_store mark_run_status: {e}"),
                    format!("run status update: {e}"),
                )
            })?;
        Ok(())
    }

    async fn delete_checkpoints(&self, run_id: &str) -> AlzinaResult<()> {
        sqlx::query("DELETE FROM run_checkpoints WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                checkpoint_err(
                    format!("checkpoint_store delete_checkpoints: {e}"),
                    format!("checkpoint delete: {e}"),
                )
            })?;
        Ok(())
    }

    async fn list_resumable_runs(&self) -> AlzinaResult<Vec<RunRecord>> {
        let sql = format!(
            "SELECT {RUN_COLUMNS} FROM synthesis_runs \
             WHERE status != 'complete' ORDER BY updated_at DESC"
        );
        let rows: Vec<RunRow> = sqlx::query_as(&sql)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| {
                checkpoint_err(
                    format!("checkpoint_store list_resumable_runs: {e}"),
                    format!("resumable runs list: {e}"),
                )
            })?;
        Ok(rows.into_iter().map(run_record_from_row).collect())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lit_schema::in_memory_lit_pool;

    fn sample_run(run_id: &str, status: &str, updated_at: &str) -> RunRecord {
        RunRecord {
            run_id: run_id.into(),
            question: "Does X improve Y?".into(),
            question_id: "q-1".into(),
            profile: "v3/lit-review-long".into(),
            model: "test-model".into(),
            merger_model: Some("test-merger".into()),
            top_k: 40,
            scope: "lit".into(),
            seed_papers: r#"["arxiv:2105"]"#.into(),
            embedding_model: "jina-v3".into(),
            embedding_dim: 1024,
            panel_source_ids: r#"["s2:abc"]"#.into(),
            code_version: "abc123".into(),
            status: status.into(),
            started_at: "2026-07-12T00:00:00+00:00".into(),
            updated_at: updated_at.into(),
        }
    }

    #[tokio::test]
    async fn run_record_roundtrip_including_option_fields() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteCheckpointStore::new(pool.clone());

        // With merger_model set.
        let run = sample_run("run-1", "running", "2026-07-12T00:00:00+00:00");
        store.record_run_start(&run).await.unwrap();
        let loaded = store.load_run("run-1").await.unwrap().expect("run must exist");
        assert_eq!(loaded.run_id, run.run_id);
        assert_eq!(loaded.question, run.question);
        assert_eq!(loaded.question_id, run.question_id);
        assert_eq!(loaded.profile, run.profile);
        assert_eq!(loaded.model, run.model);
        assert_eq!(loaded.merger_model, run.merger_model);
        assert_eq!(loaded.top_k, run.top_k);
        assert_eq!(loaded.scope, run.scope);
        assert_eq!(loaded.seed_papers, run.seed_papers);
        assert_eq!(loaded.embedding_model, run.embedding_model);
        assert_eq!(loaded.embedding_dim, run.embedding_dim);
        assert_eq!(loaded.panel_source_ids, run.panel_source_ids);
        assert_eq!(loaded.code_version, run.code_version);
        assert_eq!(loaded.status, run.status);
        assert_eq!(loaded.started_at, run.started_at);
        assert_eq!(loaded.updated_at, run.updated_at);

        // With merger_model = None.
        let mut run2 = sample_run("run-2", "running", "2026-07-12T00:00:00+00:00");
        run2.merger_model = None;
        store.record_run_start(&run2).await.unwrap();
        let loaded2 = store.load_run("run-2").await.unwrap().expect("run must exist");
        assert_eq!(loaded2.merger_model, None);
    }

    #[tokio::test]
    async fn status_column_defaults_to_running() {
        let pool = in_memory_lit_pool().await.unwrap();

        // Insert a row without an explicit status — the schema default applies.
        sqlx::query(
            "INSERT INTO synthesis_runs \
             (run_id, question, question_id, profile, model, top_k, scope, \
              embedding_model, embedding_dim, code_version, started_at, updated_at) \
             VALUES ('run-d', 'Q', 'q-1', 'p', 'm', 40, 'lit', 'jina-v3', 1024, \
                     'abc', '2026-07-12T00:00:00+00:00', '2026-07-12T00:00:00+00:00')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let (status,): (String,) =
            sqlx::query_as("SELECT status FROM synthesis_runs WHERE run_id = 'run-d'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "running", "status default must be 'running'");
    }

    #[tokio::test]
    async fn save_stage_replace_semantics_second_write_wins() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteCheckpointStore::new(pool.clone());

        store
            .save_stage("run-1", "graph", r#"{"v":1}"#, "abc")
            .await
            .unwrap();
        store
            .save_stage("run-1", "graph", r#"{"v":2}"#, "def")
            .await
            .unwrap();

        let cp = store
            .load_stage("run-1", "graph")
            .await
            .unwrap()
            .expect("checkpoint must exist");
        assert_eq!(cp.payload, r#"{"v":2}"#, "second write must win");
        assert_eq!(cp.code_version, "def");
        assert_eq!(cp.format, "json");

        // Still exactly one row for (run-1, graph).
        let (count,): (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM run_checkpoints WHERE run_id='run-1' AND stage='graph'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1, "REPLACE must not accumulate history");
    }

    #[tokio::test]
    async fn load_missing_returns_none() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteCheckpointStore::new(pool);

        assert!(store.load_stage("no-run", "graph").await.unwrap().is_none());
        assert!(store.load_run("no-run").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn mark_run_status_transitions_and_bumps_updated_at() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteCheckpointStore::new(pool);

        // Start with an old updated_at so the bump is observable.
        let run = sample_run("run-1", "running", "2020-01-01T00:00:00+00:00");
        store.record_run_start(&run).await.unwrap();

        store.mark_run_status("run-1", "failed").await.unwrap();
        let failed = store.load_run("run-1").await.unwrap().unwrap();
        assert_eq!(failed.status, "failed");
        assert_ne!(
            failed.updated_at, "2020-01-01T00:00:00+00:00",
            "updated_at must change on status transition"
        );

        store.mark_run_status("run-1", "complete").await.unwrap();
        let complete = store.load_run("run-1").await.unwrap().unwrap();
        assert_eq!(complete.status, "complete");
    }

    #[tokio::test]
    async fn list_resumable_runs_excludes_complete_orders_recent_first() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteCheckpointStore::new(pool);

        store
            .record_run_start(&sample_run("run-old", "running", "2026-07-10T00:00:00+00:00"))
            .await
            .unwrap();
        store
            .record_run_start(&sample_run("run-new", "failed", "2026-07-12T00:00:00+00:00"))
            .await
            .unwrap();
        store
            .record_run_start(&sample_run("run-done", "complete", "2026-07-11T00:00:00+00:00"))
            .await
            .unwrap();

        let resumable = store.list_resumable_runs().await.unwrap();
        let ids: Vec<&str> = resumable.iter().map(|r| r.run_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["run-new", "run-old"],
            "must include running + failed, exclude complete, most recent first"
        );
    }

    #[tokio::test]
    async fn delete_checkpoints_keeps_run_row() {
        let pool = in_memory_lit_pool().await.unwrap();
        let store = SqliteCheckpointStore::new(pool.clone());

        store
            .record_run_start(&sample_run("run-1", "running", "2026-07-12T00:00:00+00:00"))
            .await
            .unwrap();
        store.save_stage("run-1", "graph", "{}", "abc").await.unwrap();
        store.save_stage("run-1", "synthesis", "{}", "abc").await.unwrap();

        store.delete_checkpoints("run-1").await.unwrap();

        let (cp_count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM run_checkpoints WHERE run_id='run-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(cp_count, 0, "all checkpoint rows must be deleted");

        assert!(
            store.load_run("run-1").await.unwrap().is_some(),
            "synthesis_runs row must survive checkpoint deletion"
        );
    }

    #[tokio::test]
    async fn noop_store_all_ops_succeed_loads_empty() {
        let store = NoopCheckpointStore;
        store.save_stage("r", "graph", "{}", "abc").await.unwrap();
        store
            .record_run_start(&sample_run("r", "running", "2026-07-12T00:00:00+00:00"))
            .await
            .unwrap();
        store.mark_run_status("r", "complete").await.unwrap();
        store.delete_checkpoints("r").await.unwrap();
        assert!(store.load_run("r").await.unwrap().is_none());
        assert!(store.load_stage("r", "graph").await.unwrap().is_none());
        assert!(store.list_resumable_runs().await.unwrap().is_empty());
    }
}
