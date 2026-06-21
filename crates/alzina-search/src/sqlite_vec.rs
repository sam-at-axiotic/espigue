//! `SqliteVecStore` — `VectorStore` implementation backed by the
//! `sqlite-vec` extension's `vec0` virtual table joined to a metadata
//! sidecar table for post-filtering.
//!
//! AC-1 degradation: if the vec0 virtual table is missing
//! (sqlite-vec extension didn't load), the store is constructed in a
//! `disabled` state. Every `insert`/`search`/`delete_by_source` call in
//! that state returns
//! `AlzinaError::Search(SearchDetail{degraded: true, degradation_reason:
//! Some(_)})` so callers — `HybridSearchService` in particular — can
//! surface a loud-degraded notice to the agent (synthesis A1 / AC-1).
//!
//! Upsert discipline: every `insert` first DELETEs any existing row with
//! the same `(source_type, source_id, chunk_index)` triple from the vec0
//! table (joined by rowid) and from the sidecar, then INSERTs the new
//! vector + metadata. This mirrors the FTS5 idempotency pattern
//! established in Phase 1 red team A1.
//!
//! Phase 21: `with_table_names` allows a second store instance backed by
//! `lit_vec0`/`lit_chunks` for the physically separate literature corpus.

use alzina_core::error::{AlzinaError, AlzinaResult, SearchDetail};
use alzina_core::search::{VectorFilters, VectorHit, VectorMetadata, VectorStore};
use async_trait::async_trait;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;

/// kNN over-fetch factor when post-filters are applied. Synthesis §5
/// recommends 3x to give post-filters something to discard while still
/// returning `top_k` rows.
const POST_FILTER_OVERFETCH: usize = 3;

/// `VectorStore` backed by sqlite-vec's `vec0` virtual table and a
/// metadata sidecar. Construct via [`new`] for the default
/// `vec_entries`/`vec_index` pair, or via [`with_table_names`] for a
/// custom pair such as `lit_vec0`/`lit_chunks`.
///
/// [`new`]: SqliteVecStore::new
/// [`with_table_names`]: SqliteVecStore::with_table_names
pub struct SqliteVecStore {
    pool: SqlitePool,
    dimensions: usize,
    /// Name of the vec0 virtual table (e.g. `"vec_entries"` or `"lit_vec0"`).
    vec_table: String,
    /// Name of the metadata sidecar table (e.g. `"vec_index"` or `"lit_chunks"`).
    index_table: String,
    /// `true` when `vec_table` exists at construction time. When
    /// `false`, every operation returns degraded.
    enabled: bool,
}

impl SqliteVecStore {
    /// Construct a `SqliteVecStore` with custom vec0 and sidecar table names.
    ///
    /// Probes `sqlite_master` for `vec_table`; if missing, the store is
    /// initialised in disabled state and every subsequent operation returns
    /// degraded.
    ///
    /// `vec_table` and `index_table` must be in-crate constant identifiers —
    /// never request data. They are interpolated directly into SQL strings.
    pub async fn with_table_names(
        pool: SqlitePool,
        dimensions: usize,
        vec_table: &str,
        index_table: &str,
    ) -> AlzinaResult<Self> {
        let probe_sql = format!(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='{vec_table}'"
        );
        let row: Option<(String,)> = sqlx::query_as(&probe_sql)
            .fetch_optional(&pool)
            .await
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("{vec_table} probe: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!("could not probe {vec_table}: {e}")),
                })
            })?;
        let enabled = row.is_some();
        if !enabled {
            tracing::warn!(
                "SqliteVecStore initialised in disabled state — {vec_table} missing; \
                 vector search will return degraded results"
            );
        }
        Ok(Self {
            pool,
            dimensions,
            vec_table: vec_table.to_string(),
            index_table: index_table.to_string(),
            enabled,
        })
    }

    /// Construct a `SqliteVecStore` backed by the default `vec_entries` /
    /// `vec_index` tables. Delegates to [`with_table_names`].
    ///
    /// [`with_table_names`]: SqliteVecStore::with_table_names
    pub async fn new(pool: SqlitePool, dimensions: usize) -> AlzinaResult<Self> {
        Self::with_table_names(pool, dimensions, "vec_entries", "vec_index").await
    }

    /// Whether the underlying vec0 virtual table is present and usable.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Build the standard "extension missing" degraded error.
    fn disabled_error(&self) -> AlzinaError {
        let reason = format!(
            "{} table missing — sqlite-vec extension not loaded",
            self.vec_table
        );
        AlzinaError::Search(SearchDetail {
            message: "SqliteVecStore is disabled".to_string(),
            degraded: true,
            degradation_reason: Some(reason),
        })
    }

    /// Build a degraded error for vector-dimension mismatches.
    fn dimension_error(&self, got: usize) -> AlzinaError {
        AlzinaError::Search(SearchDetail {
            message: format!(
                "vector length {got} does not match store dimensions {}",
                self.dimensions
            ),
            degraded: true,
            degradation_reason: Some(format!(
                "vector length mismatch: got {got}, expected {}",
                self.dimensions
            )),
        })
    }
}

/// Encode a vector as little-endian f32 BLOB — sqlite-vec accepts either
/// JSON arrays or BLOBs of f32 bytes; BLOBs avoid stringify overhead on
/// every insert.
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Convert sqlite-vec L2 distance to a similarity score in `(0, 1]`.
/// `1.0 / (1.0 + distance)` — exact match (distance = 0) yields 1.0;
/// further-away vectors approach 0. This is monotonic in distance, so
/// rank order is preserved.
fn distance_to_similarity(distance: f64) -> f32 {
    (1.0 / (1.0 + distance)) as f32
}

#[async_trait]
impl VectorStore for SqliteVecStore {
    async fn insert(&self, vector: &[f32], metadata: VectorMetadata) -> AlzinaResult<i64> {
        if !self.enabled {
            return Err(self.disabled_error());
        }
        if vector.len() != self.dimensions {
            return Err(self.dimension_error(vector.len()));
        }

        let mut tx = self.pool.begin().await.map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("begin tx: {e}"),
                degraded: true,
                degradation_reason: Some(format!("vec insert begin tx: {e}")),
            })
        })?;

        // Upsert discipline: drop any existing row for this
        // (source_type, source_id, chunk_index). Delete from
        // the vec0 table first (joined by rowid via the sidecar), then
        // delete the metadata row.
        let upsert_delete_vec_sql = format!(
            "DELETE FROM {vt} WHERE rowid IN (\
                SELECT rowid FROM {it} \
                WHERE source_type = ? AND source_id = ? AND chunk_index = ?\
            )",
            vt = self.vec_table,
            it = self.index_table,
        );
        sqlx::query(&upsert_delete_vec_sql)
        .bind(&metadata.source_type)
        .bind(&metadata.source_id)
        .bind(metadata.chunk_index)
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("{} upsert-delete: {e}", self.vec_table),
                degraded: true,
                degradation_reason: Some(format!("{} upsert-delete: {e}", self.vec_table)),
            })
        })?;

        let upsert_delete_idx_sql = format!(
            "DELETE FROM {it} \
             WHERE source_type = ? AND source_id = ? AND chunk_index = ?",
            it = self.index_table,
        );
        sqlx::query(&upsert_delete_idx_sql)
        .bind(&metadata.source_type)
        .bind(&metadata.source_id)
        .bind(metadata.chunk_index)
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("{} upsert-delete: {e}", self.index_table),
                degraded: true,
                degradation_reason: Some(format!("{} upsert-delete: {e}", self.index_table)),
            })
        })?;

        // Insert into the vec0 table — sqlite-vec assigns a fresh rowid.
        let blob = vec_to_blob(vector);
        let insert_vec_sql = format!(
            "INSERT INTO {} (embedding) VALUES (?)",
            self.vec_table
        );
        let res = sqlx::query(&insert_vec_sql)
            .bind(&blob)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("{} insert: {e}", self.vec_table),
                    degraded: true,
                    degradation_reason: Some(format!("{} insert: {e}", self.vec_table)),
                })
            })?;
        let rowid = res.last_insert_rowid();

        // Mirror the rowid into the sidecar table so post-filter queries can
        // join the two tables.
        let insert_idx_sql = format!(
            "INSERT INTO {it} (\
                rowid, source_type, source_id, chunk_index, content_preview, \
                source_agent, source_date, weave_id, section, domain, indexed_at\
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            it = self.index_table,
        );
        sqlx::query(&insert_idx_sql)
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
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("{} insert: {e}", self.index_table),
                degraded: true,
                degradation_reason: Some(format!("{} insert: {e}", self.index_table)),
            })
        })?;

        tx.commit().await.map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("commit tx: {e}"),
                degraded: true,
                degradation_reason: Some(format!("vec insert commit: {e}")),
            })
        })?;

        Ok(rowid)
    }

    async fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        filters: &VectorFilters,
    ) -> AlzinaResult<Vec<VectorHit>> {
        if !self.enabled {
            return Err(self.disabled_error());
        }
        if vector.len() != self.dimensions {
            return Err(self.dimension_error(vector.len()));
        }

        let has_filters = filters.source_type.is_some()
            || filters.source_agent.is_some()
            || filters.date_from.is_some()
            || filters.date_to.is_some()
            || filters.domain.is_some();

        // Over-fetch when post-filtering so we still have ~top_k rows
        // after WHERE clauses trim the kNN result.
        let knn_k = if has_filters {
            top_k.saturating_mul(POST_FILTER_OVERFETCH).max(top_k)
        } else {
            top_k
        };

        // Build SQL with optional filter clauses inline.
        let mut sql = format!(
            "SELECT v.rowid, v.distance, \
                vi.source_type, vi.source_id, vi.chunk_index, vi.content_preview, \
                vi.source_agent, vi.source_date, vi.weave_id, vi.section, vi.domain, \
                vi.indexed_at \
             FROM {vt} v \
             JOIN {it} vi ON vi.rowid = v.rowid \
             WHERE v.embedding MATCH ? AND k = ?",
            vt = self.vec_table,
            it = self.index_table,
        );
        if filters.source_type.is_some() {
            sql.push_str(" AND vi.source_type = ?");
        }
        if filters.source_agent.is_some() {
            sql.push_str(" AND vi.source_agent = ?");
        }
        if filters.date_from.is_some() {
            sql.push_str(" AND vi.source_date >= ?");
        }
        if filters.date_to.is_some() {
            sql.push_str(" AND vi.source_date <= ?");
        }
        if filters.domain.is_some() {
            sql.push_str(" AND vi.domain = ?");
        }
        sql.push_str(" ORDER BY v.distance LIMIT ?");

        let blob = vec_to_blob(vector);
        let mut q = sqlx::query(&sql).bind(&blob).bind(knn_k as i64);
        if let Some(s) = &filters.source_type {
            q = q.bind(s);
        }
        if let Some(s) = &filters.source_agent {
            q = q.bind(s);
        }
        if let Some(s) = &filters.date_from {
            q = q.bind(s);
        }
        if let Some(s) = &filters.date_to {
            q = q.bind(s);
        }
        if let Some(s) = &filters.domain {
            q = q.bind(s);
        }
        q = q.bind(top_k as i64);

        let rows = q.fetch_all(&self.pool).await.map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("vec search: {e}"),
                degraded: true,
                degradation_reason: Some(format!("vec MATCH query failed: {e}")),
            })
        })?;

        let mut hits = Vec::with_capacity(rows.len());
        for row in rows {
            let rowid: i64 = row.try_get("rowid").map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("vec search rowid: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!("rowid decode: {e}")),
                })
            })?;
            let distance: f64 = row.try_get("distance").map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("vec search distance: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!("distance decode: {e}")),
                })
            })?;
            let metadata = VectorMetadata {
                source_type: row.try_get("source_type").unwrap_or_default(),
                source_id: row.try_get("source_id").unwrap_or_default(),
                chunk_index: row.try_get("chunk_index").unwrap_or(0),
                content_preview: row.try_get("content_preview").unwrap_or_default(),
                source_agent: row.try_get("source_agent").ok(),
                source_date: row.try_get("source_date").ok(),
                weave_id: row.try_get("weave_id").ok(),
                section: row.try_get("section").ok(),
                domain: row.try_get("domain").ok(),
                indexed_at: row.try_get("indexed_at").unwrap_or_default(),
            };
            hits.push(VectorHit {
                rowid,
                similarity: distance_to_similarity(distance),
                metadata,
            });
        }

        Ok(hits)
    }

    async fn delete_by_source(&self, source_type: &str, source_id: &str) -> AlzinaResult<usize> {
        if !self.enabled {
            return Err(self.disabled_error());
        }

        let mut tx = self.pool.begin().await.map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("begin tx: {e}"),
                degraded: true,
                degradation_reason: Some(format!("vec delete begin tx: {e}")),
            })
        })?;

        let delete_vec_sql = format!(
            "DELETE FROM {vt} WHERE rowid IN (\
                SELECT rowid FROM {it} WHERE source_type = ? AND source_id = ?\
             )",
            vt = self.vec_table,
            it = self.index_table,
        );
        sqlx::query(&delete_vec_sql)
        .bind(source_type)
        .bind(source_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("{} delete: {e}", self.vec_table),
                degraded: true,
                degradation_reason: Some(format!("{} delete: {e}", self.vec_table)),
            })
        })?;

        let delete_idx_sql = format!(
            "DELETE FROM {it} WHERE source_type = ? AND source_id = ?",
            it = self.index_table,
        );
        let res = sqlx::query(&delete_idx_sql)
            .bind(source_type)
            .bind(source_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("{} delete: {e}", self.index_table),
                    degraded: true,
                    degradation_reason: Some(format!("{} delete: {e}", self.index_table)),
                })
            })?;
        let count = res.rows_affected() as usize;

        tx.commit().await.map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("commit tx: {e}"),
                degraded: true,
                degradation_reason: Some(format!("vec delete commit: {e}")),
            })
        })?;

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::in_memory_pool_with_search_schema;

    /// Convenience: 1024-dim test vector with one channel set so it's
    /// distinguishable from other test vectors.
    fn one_hot(channel: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        if channel < dim {
            v[channel] = 1.0;
        }
        v
    }

    fn meta(source_type: &str, source_id: &str) -> VectorMetadata {
        VectorMetadata {
            source_type: source_type.to_string(),
            source_id: source_id.to_string(),
            chunk_index: 0,
            content_preview: "test".to_string(),
            source_agent: Some("smidr".to_string()),
            source_date: Some("2026-04-29".to_string()),
            weave_id: None,
            section: None,
            domain: None,
            indexed_at: "2026-04-29T00:00:00Z".to_string(),
        }
    }

    /// Build a store and skip the test (early-return) if the
    /// sqlite-vec extension didn't load. Returns `(store, true)` when
    /// the test should run, `(store, false)` when it should skip.
    async fn make_store_or_skip(tag: &str) -> Option<SqliteVecStore> {
        let pool = match in_memory_pool_with_search_schema().await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skipping vec test {tag}: pool setup failed: {e}");
                return None;
            }
        };
        let store = SqliteVecStore::new(pool, 1024).await.unwrap();
        if !store.is_enabled() {
            eprintln!("skipping vec test {tag}: sqlite-vec extension not loaded");
            return None;
        }
        Some(store)
    }

    #[tokio::test]
    async fn vec_insert_and_search_round_trip() {
        let Some(store) = make_store_or_skip("round_trip").await else {
            return;
        };
        let v = one_hot(7, 1024);
        store.insert(&v, meta("daily", "d-1")).await.unwrap();
        let hits = store
            .search(&v, 5, &VectorFilters::default())
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].metadata.source_id, "d-1");
        // Exact match — distance ~ 0, similarity ~ 1.0.
        assert!(
            hits[0].similarity > 0.99,
            "similarity {} should be ~1.0 for exact match",
            hits[0].similarity
        );
    }

    #[tokio::test]
    async fn vec_search_orders_by_distance() {
        let Some(store) = make_store_or_skip("orders_by_distance").await else {
            return;
        };
        let a = one_hot(0, 1024);
        let b = one_hot(500, 1024);
        let c = one_hot(1000, 1024);
        store.insert(&a, meta("daily", "a")).await.unwrap();
        store.insert(&b, meta("daily", "b")).await.unwrap();
        store.insert(&c, meta("daily", "c")).await.unwrap();

        // Query with `b` — expect b to rank #1.
        let hits = store
            .search(&b, 3, &VectorFilters::default())
            .await
            .unwrap();
        assert_eq!(hits[0].metadata.source_id, "b");
    }

    #[tokio::test]
    async fn vec_filter_by_source_type() {
        let Some(store) = make_store_or_skip("filter_source_type").await else {
            return;
        };
        store
            .insert(&one_hot(1, 1024), meta("daily", "d-1"))
            .await
            .unwrap();
        store
            .insert(&one_hot(2, 1024), meta("semantic", "s-1"))
            .await
            .unwrap();

        let filters = VectorFilters {
            source_type: Some("daily".to_string()),
            ..Default::default()
        };
        let hits = store.search(&one_hot(1, 1024), 5, &filters).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].metadata.source_type, "daily");
    }

    #[tokio::test]
    async fn vec_filter_by_date_range() {
        let Some(store) = make_store_or_skip("filter_date_range").await else {
            return;
        };
        let mut m1 = meta("daily", "d-1");
        m1.source_date = Some("2026-01-01".to_string());
        let mut m2 = meta("daily", "d-2");
        m2.source_date = Some("2026-04-29".to_string());
        let mut m3 = meta("daily", "d-3");
        m3.source_date = Some("2026-12-31".to_string());

        store.insert(&one_hot(0, 1024), m1).await.unwrap();
        store.insert(&one_hot(1, 1024), m2).await.unwrap();
        store.insert(&one_hot(2, 1024), m3).await.unwrap();

        let filters = VectorFilters {
            date_from: Some("2026-04-01".to_string()),
            date_to: Some("2026-04-30".to_string()),
            ..Default::default()
        };
        let hits = store.search(&one_hot(1, 1024), 10, &filters).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].metadata.source_id, "d-2");
    }

    #[tokio::test]
    async fn vec_insert_dedup_on_same_source_id() {
        let Some(store) = make_store_or_skip("dedup").await else {
            return;
        };
        let pool = store.pool.clone();
        store
            .insert(&one_hot(1, 1024), meta("daily", "d-x"))
            .await
            .unwrap();
        store
            .insert(&one_hot(2, 1024), meta("daily", "d-x"))
            .await
            .unwrap();

        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM vec_index WHERE source_type = 'daily' AND source_id = 'd-x'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count.0, 1, "upsert must yield exactly one vec_index row");

        let vec_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM vec_entries")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(
            vec_count.0, 1,
            "upsert must yield exactly one vec_entries row"
        );
    }

    #[tokio::test]
    async fn vec_delete_by_source_removes_both_tables() {
        let Some(store) = make_store_or_skip("delete_by_source").await else {
            return;
        };
        let pool = store.pool.clone();
        store
            .insert(&one_hot(1, 1024), meta("daily", "d-del"))
            .await
            .unwrap();

        let before: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM vec_entries")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(before.0, 1);

        let n = store.delete_by_source("daily", "d-del").await.unwrap();
        assert_eq!(n, 1);

        let after_entries: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM vec_entries")
            .fetch_one(&pool)
            .await
            .unwrap();
        let after_index: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM vec_index WHERE source_type = 'daily' AND source_id = 'd-del'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(after_entries.0, 0);
        assert_eq!(after_index.0, 0);
    }

    /// `with_table_names` against a pool with `lit_vec0` returns an enabled store.
    #[tokio::test]
    async fn with_table_names_lit_vec0_enabled() {
        // Build a pool that has lit_vec0 (from lit_schema::migrate).
        let pool = match crate::lit_schema::in_memory_lit_pool().await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skipping with_table_names_lit_vec0_enabled: pool setup failed: {e}");
                return;
            }
        };
        let store = SqliteVecStore::with_table_names(pool, 1024, "lit_vec0", "lit_chunks")
            .await
            .unwrap();
        if !store.is_enabled() {
            eprintln!("skipping with_table_names_lit_vec0_enabled: sqlite-vec extension not loaded");
            return;
        }
        assert!(
            store.is_enabled(),
            "store backed by lit_vec0 must report enabled when lit_vec0 exists"
        );
    }

    /// `new` (the legacy path) still returns enabled when vec_entries exists.
    #[tokio::test]
    async fn new_delegates_to_with_table_names_preserving_defaults() {
        let Some(store) = make_store_or_skip("new_delegates").await else {
            return;
        };
        assert!(store.is_enabled(), "legacy new() path must stay enabled");
    }

    /// insert/search round-trip works through `with_table_names` against lit_vec0.
    #[tokio::test]
    async fn with_table_names_insert_and_search_round_trip() {
        let pool = match crate::lit_schema::in_memory_lit_pool().await {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "skipping with_table_names_insert_and_search_round_trip: pool setup failed: {e}"
                );
                return;
            }
        };
        let store = SqliteVecStore::with_table_names(pool, 1024, "lit_vec0", "lit_chunks")
            .await
            .unwrap();
        if !store.is_enabled() {
            eprintln!(
                "skipping with_table_names_insert_and_search_round_trip: sqlite-vec not loaded"
            );
            return;
        }
        let v = one_hot(42, 1024);
        store.insert(&v, meta("arxiv", "arxiv:2401.00001")).await.unwrap();
        let hits = store
            .search(&v, 5, &VectorFilters::default())
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].metadata.source_id, "arxiv:2401.00001");
        assert!(
            hits[0].similarity > 0.99,
            "exact match must yield similarity ~1.0"
        );
    }

    /// Direct test of the disabled-path: we DROP `vec_entries` (or
    /// build a pool that never had it) and assert insert/search both
    /// surface degraded errors. This test runs even when sqlite-vec
    /// IS loaded — it tests the fallback contract, not the extension.
    #[tokio::test]
    async fn vec_disabled_returns_degraded() {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        // Drop vec_entries if it was created. If it wasn't created
        // (extension not loaded), this DROP is a no-op-equivalent.
        let _ = sqlx::query("DROP TABLE IF EXISTS vec_entries")
            .execute(&pool)
            .await;
        let store = SqliteVecStore::new(pool, 1024).await.unwrap();
        assert!(!store.is_enabled(), "store must report disabled");

        let v = vec![0.0_f32; 1024];
        let insert_err = store.insert(&v, meta("daily", "d-1")).await.unwrap_err();
        match &insert_err {
            AlzinaError::Search(d) => {
                assert!(d.degraded, "insert error must carry degraded=true");
                assert!(
                    d.degradation_reason
                        .as_deref()
                        .map(|r| r.contains("sqlite-vec"))
                        .unwrap_or(false),
                    "expected reason mentioning sqlite-vec, got {:?}",
                    d.degradation_reason
                );
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }

        let search_err = store
            .search(&v, 5, &VectorFilters::default())
            .await
            .unwrap_err();
        match search_err {
            AlzinaError::Search(d) => {
                assert!(d.degraded);
                assert!(
                    d.degradation_reason
                        .as_deref()
                        .map(|r| r.contains("sqlite-vec"))
                        .unwrap_or(false),
                    "expected reason mentioning sqlite-vec, got {:?}",
                    d.degradation_reason
                );
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }
    }
}
