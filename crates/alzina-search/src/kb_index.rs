//! KB indexer — per-file path + directory-walk reconciliation.
//!
//! Phase 3 Tasks 3.3a + 3.3b. Reads Markdown files under `kb_root`, parses
//! optional YAML frontmatter, chunks the body via [`crate::chunk_markdown`],
//! embeds each chunk (cache-then-API), inserts vectors into a
//! [`VectorStore`], inserts FTS5 rows via
//! [`alzina_memory::event_sink::index_to_fts`], and updates a
//! [`KbManifest`] entry. [`KbIndexer::run`] performs a full reconciliation
//! pass: delete vectors for files unlinked from disk, re-index files whose
//! content hash drifted, and first-time-index files newly dropped under
//! `kb_root`.
//!
//! AC-1: every error returned by [`KbIndexer::index_file`] /
//! [`KbIndexer::remove_file`] is `AlzinaError::Search` with `degraded =
//! true`. [`KbIndexer::run`] captures per-file degraded errors into
//! [`KbRunReport::errors`] without halting; only walker-level failures
//! (manifest scans) and `manifest.save()` errors propagate as `Err`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use serde::Deserialize;
use sqlx::SqlitePool;

use alzina_core::{
    AlzinaError, AlzinaResult, EmbeddingService, EmbeddingTask, SearchDetail, VectorMetadata,
    VectorStore, truncate_for_preview,
};
use alzina_memory::event_sink::index_to_fts_strict;

use crate::chunking::{ChunkConfig, chunk_markdown};
use crate::embed_cache::EmbeddingCache;
use crate::manifest::KbManifest;

/// Configuration knobs for [`KbIndexer`].
#[derive(Debug, Clone)]
pub struct KbIndexConfig {
    /// Max approximate tokens per chunk (forwarded into [`ChunkConfig`]).
    pub max_tokens: usize,
    /// Embedding model identifier — recorded into the embedding cache.
    pub embedding_model: String,
}

impl Default for KbIndexConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            embedding_model: "jina-embeddings-v3".into(),
        }
    }
}

/// Per-run reconciliation report produced by [`KbIndexer::run`].
///
/// Counts successful operations and captures per-file degraded errors so
/// the caller (daemon/CLI) can decide whether to surface them, log them,
/// or fail the run. Walker-level failures (`list_*`) and `manifest.save`
/// failures do NOT land here — they propagate as `Err` from `run()`.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct KbRunReport {
    /// Files indexed for the first time this run.
    pub indexed: usize,
    /// Files whose vectors + FTS rows + manifest entries were removed.
    pub removed: usize,
    /// Files whose hash drifted and were re-indexed.
    pub stale_reindexed: usize,
    /// Per-file failures: `(relative_path, err.to_string())`. Indexing
    /// did not halt — other files in the same lane were still processed.
    pub errors: Vec<(String, String)>,
}

/// Optional YAML frontmatter recognised at the top of a KB file. All
/// fields are optional — missing fields fall back to inferred defaults
/// (e.g. domain from the path's first component).
#[derive(Debug, Default, Deserialize)]
struct KbFrontmatter {
    #[allow(dead_code)]
    title: Option<String>,
    #[allow(dead_code)]
    tags: Option<Vec<String>>,
    domain: Option<String>,
}

/// Per-file KB indexer.
///
/// Holds shared dependencies (`SqlitePool` for FTS5, embedding service,
/// vector store, embedding cache) and exposes per-file index/remove
/// operations. The directory-walk `run()` method is deferred to a
/// follow-up task — this slice only ships the per-file public surface.
pub struct KbIndexer {
    kb_root: PathBuf,
    pool: SqlitePool,
    embedder: Arc<dyn EmbeddingService>,
    vec_store: Arc<dyn VectorStore>,
    cache: Arc<EmbeddingCache>,
    config: KbIndexConfig,
}

impl KbIndexer {
    /// Wrap shared dependencies. `kb_root` is the base directory under
    /// which `relative_path` arguments resolve.
    pub fn new(
        kb_root: impl Into<PathBuf>,
        pool: SqlitePool,
        embedder: Arc<dyn EmbeddingService>,
        vec_store: Arc<dyn VectorStore>,
        cache: Arc<EmbeddingCache>,
        config: KbIndexConfig,
    ) -> Self {
        Self {
            kb_root: kb_root.into(),
            pool,
            embedder,
            vec_store,
            cache,
            config,
        }
    }

    /// Base directory under which relative paths resolve.
    pub fn kb_root(&self) -> &Path {
        &self.kb_root
    }

    /// Read `<kb_root>/<relative_path>`, parse YAML frontmatter, chunk
    /// the body, embed each chunk (cache-then-API), insert into
    /// `vec_store` and FTS5, and update the manifest entry. Returns the
    /// chunk count.
    ///
    /// Re-index discipline: stale vectors and FTS rows for this
    /// `relative_path` are deleted *before* new ones are inserted so a
    /// shrunken file leaves no orphan chunks behind.
    ///
    /// The manifest is mutated in memory (`mark_indexed`) but NOT saved —
    /// the caller (the directory-walk `run()`) controls when to flush.
    ///
    /// AC-1: every returned error is `AlzinaError::Search` with
    /// `degraded = true`.
    pub async fn index_file(
        &self,
        manifest: &mut KbManifest,
        relative_path: &str,
    ) -> AlzinaResult<usize> {
        validate_relative_path(relative_path)?;
        validate_not_symlink(&self.kb_root, relative_path)?;

        let full_path = self.kb_root.join(relative_path);

        // Hash the raw file contents BEFORE frontmatter strip so the
        // manifest's hash matches `KbManifest::hash_file` exactly.
        let content_hash = KbManifest::hash_file(&full_path)?;

        let raw = std::fs::read_to_string(&full_path).map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("kb_index: failed to read {}: {}", full_path.display(), e),
                degraded: true,
                degradation_reason: Some(format!("kb file read error: {}", e)),
            })
        })?;

        let (frontmatter, body) = parse_frontmatter(&raw);

        let chunk_cfg = ChunkConfig {
            max_tokens: self.config.max_tokens,
        };
        let chunks = chunk_markdown(body, &chunk_cfg);

        // Re-index discipline: clear stale vectors + FTS rows for this
        // source_id BEFORE inserting new ones. Defensive even if the
        // file is new — `delete_by_source` on an unknown id is cheap.
        self.vec_store
            .delete_by_source("kb", relative_path)
            .await
            .map_err(degrade)?;

        if let Err(e) =
            sqlx::query("DELETE FROM search_fts WHERE source_type = ? AND source_id = ?")
                .bind("kb")
                .bind(relative_path)
                .execute(&self.pool)
                .await
        {
            // Surface as a degraded Search error — FTS is the BM25 lane
            // and silently leaving stale rows would degrade ranking.
            return Err(AlzinaError::Search(SearchDetail {
                message: format!("kb_index: fts5 stale-clear failed: {}", e),
                degraded: true,
                degradation_reason: Some(format!("fts5 stale-clear error: {}", e)),
            }));
        }

        let domain = resolve_domain(&frontmatter, relative_path);
        let source_date = Utc::now().format("%Y-%m-%d").to_string();
        let indexed_at = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        for (idx, chunk) in chunks.iter().enumerate() {
            let section = chunk.heading_path.last().cloned();
            let preview = truncate_for_preview(&chunk.content);

            let metadata = VectorMetadata {
                source_type: "kb".into(),
                source_id: relative_path.to_string(),
                chunk_index: idx as i64,
                content_preview: preview,
                source_agent: None,
                source_date: Some(source_date.clone()),
                weave_id: None,
                section: section.clone(),
                domain: domain.clone(),
                indexed_at: indexed_at.clone(),
            };

            // Cache-then-embed.
            let hash = EmbeddingCache::hash_content(&chunk.content);
            let dims = self.embedder.dimensions();
            let vector = match self.cache.get_cached(&hash).await {
                Ok(Some(v)) if v.len() == dims => v,
                _ => {
                    let v = self
                        .embedder
                        .embed(&chunk.content, EmbeddingTask::Passage)
                        .await
                        .map_err(degrade)?;
                    if let Err(e) = self
                        .cache
                        .put_cached(&hash, &self.config.embedding_model, v.len(), &v)
                        .await
                    {
                        tracing::warn!(
                            target: "alzina_search::kb_index",
                            error = %e,
                            "embedding cache write failed (continuing)"
                        );
                    }
                    v
                }
            };

            // Vector store FIRST. If this fails, propagate the error so
            // we don't index FTS for a chunk with no vector — keeps state
            // consistent across the two indices.
            self.vec_store
                .insert(&vector, metadata)
                .await
                .map_err(degrade)?;

            index_to_fts_strict(
                &self.pool,
                &chunk.content,
                "kb",
                relative_path,
                None,
                Some(&source_date),
                domain.as_deref(),
            )
            .await
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("kb_index: fts insert failed: {}", e),
                    degraded: true,
                    degradation_reason: Some(format!("kb fts insert: {e}")),
                })
            })?;
        }

        manifest.mark_indexed(relative_path, content_hash, chunks.len());
        Ok(chunks.len())
    }

    /// Delete all vectors and FTS rows for `relative_path` and remove
    /// the manifest entry. Used when a file is unlinked from the kb tree.
    /// The manifest is mutated in memory but NOT saved — caller controls flush.
    ///
    /// AC-1: every returned error is `AlzinaError::Search` with
    /// `degraded = true`.
    pub async fn remove_file(
        &self,
        manifest: &mut KbManifest,
        relative_path: &str,
    ) -> AlzinaResult<()> {
        validate_relative_path(relative_path)?;

        self.vec_store
            .delete_by_source("kb", relative_path)
            .await
            .map_err(degrade)?;

        if let Err(e) =
            sqlx::query("DELETE FROM search_fts WHERE source_type = ? AND source_id = ?")
                .bind("kb")
                .bind(relative_path)
                .execute(&self.pool)
                .await
        {
            return Err(AlzinaError::Search(SearchDetail {
                message: format!("kb_index: fts5 delete failed: {}", e),
                degraded: true,
                degradation_reason: Some(format!("fts5 delete error: {}", e)),
            }));
        }

        manifest.remove(relative_path);
        Ok(())
    }

    /// Full reconciliation pass against `kb_root`:
    ///
    /// 1. Remove vectors + manifest entries for files unlinked from disk.
    /// 2. Re-index files whose on-disk hash drifted from the manifest.
    /// 3. First-time-index files present on disk but missing from the
    ///    manifest.
    /// 4. Persist the manifest via `save`.
    ///
    /// Per-file errors (which are `AlzinaError::Search` with `degraded =
    /// true` by AC-1) are captured into [`KbRunReport::errors`] without
    /// halting the run. Walker-level failures and `save` failures
    /// propagate as `Err` so the caller can distinguish a partially-bad
    /// run from a totally-broken one.
    pub async fn run(&self, manifest: &mut KbManifest) -> AlzinaResult<KbRunReport> {
        let mut report = KbRunReport::default();

        // Step A — removed files (manifest entries with no on-disk file).
        let removed = manifest.list_removed()?;
        for rel in removed {
            match self.remove_file(manifest, &rel).await {
                Ok(()) => report.removed += 1,
                Err(e) => report.errors.push((rel, e.to_string())),
            }
        }

        // Step B — stale files (on-disk hash differs from recorded hash).
        let stale = manifest.list_stale()?;
        for rel in stale {
            match self.index_file(manifest, &rel).await {
                Ok(_) => report.stale_reindexed += 1,
                Err(e) => report.errors.push((rel, e.to_string())),
            }
        }

        // Step C — new files (on disk but absent from the manifest).
        let new_files = manifest.list_new()?;
        for rel in new_files {
            match self.index_file(manifest, &rel).await {
                Ok(_) => report.indexed += 1,
                Err(e) => report.errors.push((rel, e.to_string())),
            }
        }

        // Step D — flush manifest. A save failure is fatal: the in-memory
        // state is now ahead of disk and the next run would re-do work or,
        // worse, skip removed-file cleanup.
        manifest.save()?;

        Ok(report)
    }
}

/// Reject path-traversal attempts. AC-1: rejection surfaces as a
/// degraded `AlzinaError::Search`.
fn validate_relative_path(relative_path: &str) -> AlzinaResult<()> {
    if relative_path.starts_with('/') || relative_path.split(['/', '\\']).any(|seg| seg == "..") {
        return Err(AlzinaError::Search(SearchDetail {
            message: format!("path traversal rejected: {}", relative_path),
            degraded: true,
            degradation_reason: Some("path traversal rejected".into()),
        }));
    }
    Ok(())
}

/// Reject paths whose final component on disk is a symlink. Skipped when
/// the path doesn't exist yet (the file may be in flight on a watcher
/// pipeline). AC-1: rejection surfaces as a degraded `AlzinaError::Search`.
fn validate_not_symlink(kb_root: &Path, relative_path: &str) -> AlzinaResult<()> {
    let full = kb_root.join(relative_path);
    match std::fs::symlink_metadata(&full) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(AlzinaError::Search(SearchDetail {
                    message: format!("symlink rejected: {}", relative_path),
                    degraded: true,
                    degradation_reason: Some("symlink rejected".into()),
                }));
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(AlzinaError::Search(SearchDetail {
            message: format!("symlink_metadata failed for {}: {}", full.display(), e),
            degraded: true,
            degradation_reason: Some(format!("symlink_metadata error: {}", e)),
        })),
    }
}

/// Map any non-`AlzinaError::Search` upstream error into a degraded
/// `AlzinaError::Search` so AC-1 holds for every surfaced error. If the
/// error is already a `Search` variant, preserve its `degraded` /
/// `degradation_reason` payload.
fn degrade(err: AlzinaError) -> AlzinaError {
    match err {
        AlzinaError::Search(d) => AlzinaError::Search(SearchDetail {
            degraded: true,
            ..d
        }),
        other => {
            let msg = other.to_string();
            AlzinaError::Search(SearchDetail {
                message: msg.clone(),
                degraded: true,
                degradation_reason: Some(msg),
            })
        }
    }
}

/// Parse optional YAML frontmatter. If line 0 is `---`, scans for the
/// next `---` and treats the in-between as YAML. On parse failure, logs
/// at `warn!` and returns the default frontmatter so indexing still
/// proceeds. Returns `(frontmatter, body_after_strip)`.
///
/// # Edge cases
///
/// A file containing only YAML frontmatter (`---\n...\n---\n` with no
/// body after the closing fence) yields zero chunks downstream. The
/// manifest will record `chunk_count: 0` and the file is not searchable.
/// This is by-design — frontmatter-only files have no body to index.
fn parse_frontmatter(source: &str) -> (KbFrontmatter, &str) {
    if !source.starts_with("---") {
        return (KbFrontmatter::default(), source);
    }
    // First line must be exactly `---` (allowing trailing whitespace).
    let mut lines = source.split_inclusive('\n');
    let first = lines.next().unwrap_or("");
    if first.trim_end() != "---" {
        return (KbFrontmatter::default(), source);
    }

    let mut yaml_end_offset: Option<usize> = None;
    let mut yaml_content_end: usize = first.len();
    let mut consumed = first.len();
    for line in lines {
        if line.trim_end() == "---" {
            yaml_end_offset = Some(consumed + line.len());
            break;
        }
        yaml_content_end = consumed + line.len();
        consumed += line.len();
    }

    let Some(end) = yaml_end_offset else {
        // Unterminated frontmatter — keep the source intact.
        return (KbFrontmatter::default(), source);
    };

    let yaml_str = &source[first.len()..yaml_content_end];
    if yaml_str.len() > 64 * 1024 {
        tracing::warn!(
            target: "alzina_search::kb_index",
            "frontmatter exceeded 64KB cap; treating as empty"
        );
        return (KbFrontmatter::default(), &source[end..]);
    }
    let frontmatter: KbFrontmatter = match serde_yaml::from_str(yaml_str) {
        Ok(fm) => fm,
        Err(e) => {
            tracing::warn!(
                target: "alzina_search::kb_index",
                error = %e,
                "kb frontmatter parse failed; continuing with defaults"
            );
            KbFrontmatter::default()
        }
    };
    (frontmatter, &source[end..])
}

/// Resolve the `domain` field for a KB file. Preference order:
/// 1. `frontmatter.domain` if set
/// 2. First path component of `relative_path` (e.g. `papers/foo.md`
///    → `Some("papers")`)
/// 3. `None` (file is at the root)
fn resolve_domain(frontmatter: &KbFrontmatter, relative_path: &str) -> Option<String> {
    if let Some(d) = frontmatter.domain.as_ref() {
        if !d.trim().is_empty() {
            return Some(d.clone());
        }
    }
    let first = relative_path.split('/').next()?;
    if first.is_empty() || first == relative_path {
        // No subdirectory component — file lives at the root.
        return None;
    }
    Some(first.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::in_memory_pool_with_search_schema;
    use alzina_core::search::{VectorFilters, VectorHit};
    use async_trait::async_trait;
    use std::sync::Mutex;
    use tempfile::tempdir;

    /// Stub embedder — produces a deterministic content-independent
    /// vector. Records call count for assertions.
    struct StubEmbedder {
        dim: usize,
        calls: Mutex<usize>,
    }

    impl StubEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                calls: Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl EmbeddingService for StubEmbedder {
        async fn embed(&self, _text: &str, _task: EmbeddingTask) -> AlzinaResult<Vec<f32>> {
            *self.calls.lock().unwrap() += 1;
            Ok(vec![0.5_f32; self.dim])
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

    /// Stub vector store — records every (vector, metadata) insert and
    /// every `delete_by_source` call. Optionally fails `insert` for a
    /// configured `source_id` so we can exercise the per-file error
    /// capture path in `KbIndexer::run`.
    #[derive(Default)]
    struct StubVecStore {
        inserts: Mutex<Vec<(Vec<f32>, VectorMetadata)>>,
        deletes: Mutex<Vec<(String, String)>>,
        fail_insert_for: Mutex<Option<String>>,
        fail_insert_all: Mutex<bool>,
        fail_insert_at_chunk: Mutex<Option<i64>>,
    }

    impl StubVecStore {
        fn new() -> Self {
            Self::default()
        }
        fn insert_snapshot(&self) -> Vec<(Vec<f32>, VectorMetadata)> {
            self.inserts.lock().unwrap().clone()
        }
        fn delete_snapshot(&self) -> Vec<(String, String)> {
            self.deletes.lock().unwrap().clone()
        }
        fn fail_insert_on(&self, source_id: &str) {
            *self.fail_insert_for.lock().unwrap() = Some(source_id.to_string());
        }
        fn fail_all_inserts(&self) {
            *self.fail_insert_all.lock().unwrap() = true;
        }
        fn fail_insert_at_chunk(&self, chunk_index: i64) {
            *self.fail_insert_at_chunk.lock().unwrap() = Some(chunk_index);
        }
    }

    #[async_trait]
    impl VectorStore for StubVecStore {
        async fn insert(&self, vector: &[f32], metadata: VectorMetadata) -> AlzinaResult<i64> {
            if *self.fail_insert_all.lock().unwrap() {
                return Err(AlzinaError::Search(SearchDetail {
                    message: "stub insert forced to fail (all)".into(),
                    degraded: true,
                    degradation_reason: Some("stub forced failure (all)".into()),
                }));
            }
            if let Some(sid) = self.fail_insert_for.lock().unwrap().as_deref() {
                if sid == metadata.source_id {
                    return Err(AlzinaError::Search(SearchDetail {
                        message: format!("stub insert forced to fail for {}", sid),
                        degraded: true,
                        degradation_reason: Some("stub forced failure".into()),
                    }));
                }
            }
            if let Some(target_idx) = *self.fail_insert_at_chunk.lock().unwrap() {
                if metadata.chunk_index == target_idx {
                    return Err(AlzinaError::Search(SearchDetail {
                        message: format!("stub insert forced to fail at chunk {}", target_idx),
                        degraded: true,
                        degradation_reason: Some("stub forced chunk failure".into()),
                    }));
                }
            }
            let mut g = self.inserts.lock().unwrap();
            g.push((vector.to_vec(), metadata));
            Ok(g.len() as i64)
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
            source_type: &str,
            source_id: &str,
        ) -> AlzinaResult<usize> {
            self.deletes
                .lock()
                .unwrap()
                .push((source_type.into(), source_id.into()));
            Ok(0)
        }
    }

    fn write_kb_file(kb_root: &Path, rel: &str, body: &str) {
        let full = kb_root.join(rel);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&full, body).unwrap();
    }

    async fn build_indexer(
        kb_root: PathBuf,
    ) -> (KbIndexer, Arc<StubEmbedder>, Arc<StubVecStore>, SqlitePool) {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let embedder = Arc::new(StubEmbedder::new(4));
        let vec_store = Arc::new(StubVecStore::new());
        let cache = Arc::new(EmbeddingCache::new(pool.clone()));
        let indexer = KbIndexer::new(
            kb_root,
            pool.clone(),
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            Arc::clone(&vec_store) as Arc<dyn VectorStore>,
            cache,
            KbIndexConfig::default(),
        );
        (indexer, embedder, vec_store, pool)
    }

    #[tokio::test]
    async fn index_file_inserts_chunks_into_vec_store() {
        let dir = tempdir().unwrap();
        write_kb_file(
            dir.path(),
            "notes/alpha.md",
            "# Alpha\n\nThe first paragraph.\n\n# Beta\n\nThe second paragraph.\n",
        );
        let (indexer, _embedder, vec_store, _pool) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let n = indexer
            .index_file(&mut manifest, "notes/alpha.md")
            .await
            .expect("index_file ok");
        assert!(n >= 2, "expected ≥2 chunks, got {n}");

        let snap = vec_store.insert_snapshot();
        assert_eq!(snap.len(), n, "every chunk produced one insert");
        for (vec, md) in &snap {
            assert_eq!(vec.len(), 4, "vector dim matches embedder");
            assert_eq!(md.source_type, "kb");
            assert_eq!(md.source_id, "notes/alpha.md");
        }
        // chunk_index is contiguous from 0..n.
        let mut indices: Vec<i64> = snap.iter().map(|(_, m)| m.chunk_index).collect();
        indices.sort();
        assert_eq!(indices, (0..n as i64).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn index_file_writes_fts_rows_for_kb_source_type() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "thoughts.md", "# Heading\n\nbody body body");
        let (indexer, _e, _v, pool) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        indexer
            .index_file(&mut manifest, "thoughts.md")
            .await
            .unwrap();

        let row: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM search_fts WHERE source_type = ? AND source_id = ?",
        )
        .bind("kb")
        .bind("thoughts.md")
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.0 >= 1, "expected ≥1 fts5 row, got {}", row.0);
    }

    #[tokio::test]
    async fn index_file_records_manifest_entry() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "foo.md", "# F\n\nbody");
        let (indexer, _e, _v, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        indexer.index_file(&mut manifest, "foo.md").await.unwrap();

        let entry = manifest.data().files.get("foo.md").expect("entry exists");
        let expected = KbManifest::hash_file(&dir.path().join("foo.md")).unwrap();
        assert_eq!(entry.content_hash, expected);
        assert!(entry.chunk_count >= 1);
    }

    #[tokio::test]
    async fn index_file_strips_yaml_frontmatter_before_chunking() {
        let dir = tempdir().unwrap();
        write_kb_file(
            dir.path(),
            "fm.md",
            "---\ntitle: Foo\n---\n# Body\n\nactual text",
        );
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        indexer.index_file(&mut manifest, "fm.md").await.unwrap();

        let snap = vec_store.insert_snapshot();
        assert!(!snap.is_empty(), "got at least one chunk");
        for (_, md) in &snap {
            assert!(
                !md.content_preview.contains("title:"),
                "frontmatter leaked into preview: {}",
                md.content_preview
            );
        }
    }

    #[tokio::test]
    async fn index_file_uses_frontmatter_domain_when_present() {
        let dir = tempdir().unwrap();
        write_kb_file(
            dir.path(),
            "notes/abc.md",
            "---\ndomain: papers\n---\n# T\n\nbody",
        );
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        indexer
            .index_file(&mut manifest, "notes/abc.md")
            .await
            .unwrap();

        let snap = vec_store.insert_snapshot();
        assert!(!snap.is_empty());
        for (_, md) in &snap {
            assert_eq!(
                md.domain.as_deref(),
                Some("papers"),
                "frontmatter domain should win over path component"
            );
        }
    }

    #[tokio::test]
    async fn index_file_falls_back_to_path_component_when_no_frontmatter_domain() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "papers/foo.md", "# T\n\nbody");
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        indexer
            .index_file(&mut manifest, "papers/foo.md")
            .await
            .unwrap();

        let snap = vec_store.insert_snapshot();
        assert!(!snap.is_empty());
        for (_, md) in &snap {
            assert_eq!(md.domain.as_deref(), Some("papers"));
        }
    }

    #[tokio::test]
    async fn index_file_rejects_path_traversal() {
        let dir = tempdir().unwrap();
        let (indexer, _e, _v, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let err = indexer
            .index_file(&mut manifest, "../etc/passwd")
            .await
            .expect_err("path traversal must be rejected");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded, "rejection must be degraded");
                let r = d.degradation_reason.unwrap_or_default();
                assert!(
                    r.contains("path traversal"),
                    "reason should mention path traversal, got {r:?}"
                );
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn index_file_deletes_stale_vectors_before_insert() {
        let dir = tempdir().unwrap();
        let rel = "shrink.md";
        write_kb_file(
            dir.path(),
            rel,
            "# A\n\npara1\n\n# B\n\npara2\n\n# C\n\npara3",
        );
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        indexer.index_file(&mut manifest, rel).await.unwrap();
        let after_first = vec_store.delete_snapshot();
        assert_eq!(
            after_first.len(),
            1,
            "first index should issue exactly one delete_by_source"
        );
        assert_eq!(after_first[0], ("kb".into(), rel.into()));

        // Shrink the file and reindex — must still issue a
        // `delete_by_source` defensively.
        write_kb_file(dir.path(), rel, "# A\n\nonly one section now");
        indexer.index_file(&mut manifest, rel).await.unwrap();

        let after_second = vec_store.delete_snapshot();
        assert_eq!(
            after_second.len(),
            2,
            "second index should issue another delete_by_source"
        );
        assert_eq!(after_second[1], ("kb".into(), rel.into()));
    }

    #[tokio::test]
    async fn remove_file_deletes_vectors_and_manifest_entry() {
        let dir = tempdir().unwrap();
        let rel = "gone.md";
        write_kb_file(dir.path(), rel, "# G\n\nbody");
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        indexer.index_file(&mut manifest, rel).await.unwrap();
        assert!(manifest.data().files.contains_key(rel));

        indexer.remove_file(&mut manifest, rel).await.unwrap();

        let snap = vec_store.delete_snapshot();
        assert!(
            snap.iter().any(|(st, sid)| st == "kb" && sid == rel),
            "expected delete_by_source for removed file"
        );
        assert!(
            !manifest.data().files.contains_key(rel),
            "manifest entry should be gone"
        );
    }

    // ── Task 3.3b: KbIndexer::run() reconciliation ────────────────────

    #[tokio::test]
    async fn run_empty_kb_root_returns_zero_counts() {
        let dir = tempdir().unwrap();
        let (indexer, _e, _v, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let report = indexer.run(&mut manifest).await.expect("run ok");
        assert_eq!(report.indexed, 0);
        assert_eq!(report.removed, 0);
        assert_eq!(report.stale_reindexed, 0);
        assert!(report.errors.is_empty());
    }

    #[tokio::test]
    async fn run_indexes_two_new_files() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "a.md", "# A\n\nbody-a");
        write_kb_file(dir.path(), "nested/b.md", "# B\n\nbody-b");
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let report = indexer.run(&mut manifest).await.expect("run ok");
        assert_eq!(report.indexed, 2);
        assert_eq!(report.removed, 0);
        assert_eq!(report.stale_reindexed, 0);
        assert!(report.errors.is_empty());

        let snap = vec_store.insert_snapshot();
        assert!(snap.iter().any(|(_, m)| m.source_id == "a.md"));
        assert!(snap.iter().any(|(_, m)| m.source_id == "nested/b.md"));
        assert!(manifest.data().files.contains_key("a.md"));
        assert!(manifest.data().files.contains_key("nested/b.md"));
    }

    #[tokio::test]
    async fn run_reindexes_stale_file_and_updates_hash() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "s.md", "# S\n\nfirst body");
        let (indexer, _e, _v, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        // Initial run indexes the file.
        let r1 = indexer.run(&mut manifest).await.unwrap();
        assert_eq!(r1.indexed, 1);
        let original_hash = manifest.data().files["s.md"].content_hash.clone();

        // Mutate the file so its on-disk hash drifts.
        write_kb_file(dir.path(), "s.md", "# S\n\ncompletely-different body");

        let r2 = indexer.run(&mut manifest).await.unwrap();
        assert_eq!(r2.stale_reindexed, 1);
        assert_eq!(r2.indexed, 0);
        assert_eq!(r2.removed, 0);
        assert!(r2.errors.is_empty());

        let new_hash = &manifest.data().files["s.md"].content_hash;
        assert_ne!(new_hash, &original_hash);
        let on_disk_hash = KbManifest::hash_file(&dir.path().join("s.md")).unwrap();
        assert_eq!(new_hash, &on_disk_hash);
    }

    #[tokio::test]
    async fn run_removes_file_no_longer_on_disk() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "keep.md", "# K\n\nkeep");
        write_kb_file(dir.path(), "gone.md", "# G\n\nbye");
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        // First run: both files indexed.
        indexer.run(&mut manifest).await.unwrap();
        assert!(manifest.data().files.contains_key("gone.md"));

        // Unlink one file from disk.
        std::fs::remove_file(dir.path().join("gone.md")).unwrap();

        let report = indexer.run(&mut manifest).await.unwrap();
        assert_eq!(report.removed, 1);
        assert_eq!(report.indexed, 0);
        assert_eq!(report.stale_reindexed, 0);
        assert!(report.errors.is_empty());

        assert!(!manifest.data().files.contains_key("gone.md"));
        let deletes = vec_store.delete_snapshot();
        assert!(
            deletes
                .iter()
                .any(|(st, sid)| st == "kb" && sid == "gone.md")
        );
    }

    #[tokio::test]
    async fn run_captures_per_file_index_error_without_halting() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "ok.md", "# O\n\nokay");
        write_kb_file(dir.path(), "bad.md", "# B\n\nbroken");
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        // Force the stub to fail on `bad.md` only.
        vec_store.fail_insert_on("bad.md");
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let report = indexer.run(&mut manifest).await.expect("walker ok");
        assert_eq!(report.indexed, 1, "ok.md still indexes");
        assert_eq!(report.errors.len(), 1, "bad.md surfaces one error");
        let (rel, msg) = &report.errors[0];
        assert_eq!(rel, "bad.md");
        assert!(!msg.is_empty(), "captured error string should be non-empty");
        // `ok.md` made it into the manifest; `bad.md` did not.
        assert!(manifest.data().files.contains_key("ok.md"));
        assert!(!manifest.data().files.contains_key("bad.md"));
    }

    #[tokio::test]
    async fn run_is_idempotent() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "i.md", "# I\n\nonce");
        let (indexer, _e, _v, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let r1 = indexer.run(&mut manifest).await.unwrap();
        assert_eq!(r1.indexed, 1);

        // Second call: nothing changed on disk → all counts zero.
        let r2 = indexer.run(&mut manifest).await.unwrap();
        assert_eq!(r2.indexed, 0);
        assert_eq!(r2.removed, 0);
        assert_eq!(r2.stale_reindexed, 0);
        assert!(r2.errors.is_empty());
    }

    #[tokio::test]
    async fn run_persists_manifest_via_save() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "persist.md", "# P\n\npersisted body");
        let (indexer, _e, _v, _p) = build_indexer(dir.path().to_path_buf()).await;
        {
            let mut manifest = KbManifest::open(dir.path()).unwrap();
            indexer.run(&mut manifest).await.unwrap();
        }
        // Open a fresh manifest from disk: the entry must be there.
        let reopened = KbManifest::open(dir.path()).unwrap();
        assert!(reopened.data().files.contains_key("persist.md"));
        assert!(dir.path().join(crate::manifest::MANIFEST_FILE).exists());
    }

    // ── Phantom-success hardening (P0#1, P1#11, P2#16, P2#17) ─────────

    /// P0#1: an FTS insert failure must propagate as a degraded
    /// `AlzinaError::Search`, not get silently swallowed.
    #[tokio::test]
    async fn index_file_propagates_fts_failure_as_degraded() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "fts.md", "# F\n\nbody");
        let (indexer, _e, _v, pool) = build_indexer(dir.path().to_path_buf()).await;
        // Drop the FTS5 virtual table so any INSERT into search_fts errors out.
        sqlx::query("DROP TABLE search_fts")
            .execute(&pool)
            .await
            .unwrap();
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let err = indexer
            .index_file(&mut manifest, "fts.md")
            .await
            .expect_err("fts insert failure must propagate");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded, "must be degraded");
                let r = d.degradation_reason.unwrap_or_default();
                assert!(r.contains("fts"), "reason should mention fts, got {r:?}");
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }
    }

    /// P0#1: when FTS insert fails, `mark_indexed` must NOT have been
    /// called — otherwise the manifest claims success despite a missing
    /// FTS row.
    #[tokio::test]
    async fn mark_indexed_skipped_on_fts_failure() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "skip.md", "# S\n\nbody");
        let (indexer, _e, _v, pool) = build_indexer(dir.path().to_path_buf()).await;
        sqlx::query("DROP TABLE search_fts")
            .execute(&pool)
            .await
            .unwrap();
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let _ = indexer.index_file(&mut manifest, "skip.md").await;
        assert!(
            !manifest.data().files.contains_key("skip.md"),
            "manifest entry must not be recorded after fts failure"
        );
    }

    /// P1#11: a path whose final component is a symlink must be rejected
    /// up-front with a degraded error.
    #[cfg(unix)]
    #[tokio::test]
    async fn index_file_rejects_symlink_target() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let real = outside.path().join("real.md");
        std::fs::write(&real, "# R\n\nbody").unwrap();
        let link = dir.path().join("link.md");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let (indexer, _e, _v, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let err = indexer
            .index_file(&mut manifest, "link.md")
            .await
            .expect_err("symlink target must be rejected");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded, "must be degraded");
                let r = d.degradation_reason.unwrap_or_default();
                assert!(
                    r.contains("symlink"),
                    "reason should mention symlink, got {r:?}"
                );
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }
        assert!(
            !manifest.data().files.contains_key("link.md"),
            "manifest must not record a rejected symlink"
        );
    }

    /// P2#17: oversized YAML frontmatter (> 64KB) must NOT panic and must
    /// NOT propagate an error — indexing falls back to default frontmatter.
    #[tokio::test]
    async fn frontmatter_over_64kb_falls_back_to_defaults() {
        let dir = tempdir().unwrap();
        // 70KB of YAML content between the fences.
        let big_yaml = "k: ".to_string() + &"a".repeat(70 * 1024);
        let body = format!("---\n{big_yaml}\n---\n# Body\n\nactual text");
        write_kb_file(dir.path(), "papers/big.md", &body);
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        indexer
            .index_file(&mut manifest, "papers/big.md")
            .await
            .expect("oversized frontmatter should NOT fail indexing");

        // Defaults kicked in: domain falls back to the path component.
        let snap = vec_store.insert_snapshot();
        assert!(!snap.is_empty(), "expected at least one chunk");
        for (_, md) in &snap {
            assert_eq!(
                md.domain.as_deref(),
                Some("papers"),
                "domain should fall back to path component when YAML is dropped"
            );
        }
    }

    /// P2#16: when every per-file index call fails, the run must report
    /// `indexed == 0` and capture an error per file.
    #[tokio::test]
    async fn run_all_files_fail_returns_zero_indexed() {
        let dir = tempdir().unwrap();
        write_kb_file(dir.path(), "a.md", "# A\n\nbody-a");
        write_kb_file(dir.path(), "b.md", "# B\n\nbody-b");
        write_kb_file(dir.path(), "c.md", "# C\n\nbody-c");
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        vec_store.fail_all_inserts();
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let report = indexer.run(&mut manifest).await.expect("walker ok");
        assert_eq!(report.indexed, 0, "no file should be marked indexed");
        assert_eq!(report.errors.len(), 3, "one error per file");
        assert!(manifest.data().files.is_empty());
    }

    /// P2#16: a mid-chunk-loop vec_store failure must early-return; later
    /// chunks must NOT be inserted, and `mark_indexed` must NOT run.
    #[tokio::test]
    async fn index_file_rolls_back_chunk_loop_on_mid_failure() {
        let dir = tempdir().unwrap();
        // Five distinct sections so chunk_markdown emits ≥5 chunks.
        let body = "# A\n\npara-a\n\n\
                    # B\n\npara-b\n\n\
                    # C\n\npara-c\n\n\
                    # D\n\npara-d\n\n\
                    # E\n\npara-e\n";
        write_kb_file(dir.path(), "five.md", body);
        let (indexer, _e, vec_store, _p) = build_indexer(dir.path().to_path_buf()).await;
        // Fail on the third chunk (zero-based index 2).
        vec_store.fail_insert_at_chunk(2);
        let mut manifest = KbManifest::open(dir.path()).unwrap();

        let err = indexer
            .index_file(&mut manifest, "five.md")
            .await
            .expect_err("mid-loop failure must propagate");
        assert!(matches!(err, AlzinaError::Search(_)));

        let snap = vec_store.insert_snapshot();
        assert!(
            snap.len() <= 3,
            "no chunks past the failure point should be inserted; got {}",
            snap.len()
        );
        // Specifically: chunk_index 3 and 4 must NEVER have been inserted —
        // the loop early-returns on the first failure.
        for (_, md) in &snap {
            assert!(
                md.chunk_index < 3,
                "chunk_index {} should not have been inserted after mid-loop failure",
                md.chunk_index
            );
        }
        assert!(
            !manifest.data().files.contains_key("five.md"),
            "manifest entry must not be recorded after mid-loop failure"
        );
    }
}
