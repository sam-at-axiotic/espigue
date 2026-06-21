//! Phase 3 Task 3.11 — KB indexer round-trip integration tests.
//!
//! Unit coverage for chunking, manifest, quality gates, S2, and per-file
//! indexer behaviour already lives in inline `mod tests`. The genuine
//! integration gap is the full pipeline: `KbIndexer::run()` writes to a
//! real `SqliteVecStore` + FTS5, then `HybridSearchService` retrieves the
//! chunks via `search_collection(Collection::Kb)`.
//!
//! Tests skip gracefully when the sqlite-vec extension isn't loaded —
//! `KbIndexer::run()` requires a working vector store, so without the
//! extension there's nothing to integrate.
//!
//! KB-only: exercises `KbIndexer` + `HybridSearchService`, which are gated
//! behind the `kb` feature. Compiled out for the literature-only build.
#![cfg(feature = "kb")]

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sqlx::SqlitePool;

use alzina_core::{
    AlzinaError, AlzinaResult, EmbeddingService, EmbeddingTask, SearchDetail, VectorStore,
    search::{SearchResults, VectorFilters},
};
use alzina_memory::FtsSearch;
use alzina_search::{
    EmbeddingCache, KbIndexConfig, KbIndexer, KbManifest, SqliteVecStore,
    hybrid::{Collection, HybridConfig, HybridSearchService},
    schema::migrate as search_migrate,
};

const DIM: usize = 1024;

/// Deterministic stub embedder — every text gets the same vector. Good
/// enough for round-trip assertions; the FTS5 lane carries the actual
/// recall.
struct StubEmbedder;

#[async_trait]
impl EmbeddingService for StubEmbedder {
    async fn embed(&self, _text: &str, _task: EmbeddingTask) -> AlzinaResult<Vec<f32>> {
        Ok(vec![0.5_f32; DIM])
    }
    async fn embed_batch(
        &self,
        texts: &[String],
        _task: EmbeddingTask,
    ) -> AlzinaResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.5_f32; DIM]).collect())
    }
    fn dimensions(&self) -> usize {
        DIM
    }
}

async fn build_pool() -> AlzinaResult<SqlitePool> {
    let pool = alzina_memory::schema::in_memory_pool().await.map_err(|e| {
        AlzinaError::Search(SearchDetail {
            message: e.clone(),
            degraded: true,
            degradation_reason: Some(format!("memory schema init: {e}")),
        })
    })?;
    search_migrate(&pool).await?;
    Ok(pool)
}

fn write_kb_file(kb_root: &Path, rel: &str, body: &str) {
    let full = kb_root.join(rel);
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&full, body).unwrap();
}

async fn build_stack(kb_root: &Path) -> Option<(KbIndexer, HybridSearchService, SqlitePool)> {
    let pool = build_pool().await.ok()?;
    let vec_store = Arc::new(SqliteVecStore::new(pool.clone(), DIM).await.ok()?);
    if !vec_store.is_enabled() {
        eprintln!("sqlite-vec extension not loaded — skipping kb round-trip test");
        return None;
    }
    let embedder: Arc<dyn EmbeddingService> = Arc::new(StubEmbedder);
    let cache = Arc::new(EmbeddingCache::new(pool.clone()));
    let fts = Arc::new(FtsSearch::new(pool.clone()));

    let indexer = KbIndexer::new(
        kb_root.to_path_buf(),
        pool.clone(),
        embedder.clone(),
        vec_store.clone() as Arc<dyn VectorStore>,
        cache.clone(),
        KbIndexConfig::default(),
    );

    let svc = HybridSearchService::new(embedder, vec_store, fts, cache, HybridConfig::default());
    Some((indexer, svc, pool))
}

fn kb_hits(res: &SearchResults) -> Vec<&str> {
    res.hits
        .iter()
        .filter(|h| h.source_type == "kb")
        .map(|h| h.source_id.as_str())
        .collect()
}

#[tokio::test]
async fn run_then_search_returns_indexed_kb_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let kb_root = dir.path();
    write_kb_file(
        kb_root,
        "alpha.md",
        "# Widgets\n\nThe widgets module discusses sprocket alignment.",
    );
    write_kb_file(
        kb_root,
        "notes/beta.md",
        "# Cogs\n\nCog torque is measured in newton-meters.",
    );

    let Some((indexer, svc, _pool)) = build_stack(kb_root).await else {
        return;
    };

    let mut manifest = KbManifest::open(kb_root).unwrap();
    let report = indexer.run(&mut manifest).await.unwrap();
    assert_eq!(report.removed, 0);
    assert_eq!(report.stale_reindexed, 0);
    assert!(
        report.indexed >= 2,
        "expected ≥2 files indexed, got {}",
        report.indexed
    );
    assert!(report.errors.is_empty(), "errors: {:?}", report.errors);

    let res = svc
        .search_collection(
            "widgets sprocket",
            &VectorFilters::default(),
            10,
            Collection::Kb,
        )
        .await
        .unwrap();
    let hits = kb_hits(&res);
    // Authority-tier wrapping may tag results with [low-authority]...[/low-authority];
    // check that at least one hit contains the bare filename.
    assert!(
        hits.iter().any(|h| h.contains("alpha.md")),
        "expected alpha.md (possibly authority-wrapped) in kb hits, got {:?}",
        hits
    );
}

#[tokio::test]
async fn run_after_delete_drops_chunks_from_search() {
    let dir = tempfile::tempdir().unwrap();
    let kb_root = dir.path();
    let target = kb_root.join("doomed.md");
    write_kb_file(
        kb_root,
        "doomed.md",
        "# Vanishing\n\nThis content will disappear.",
    );
    write_kb_file(kb_root, "kept.md", "# Preserved\n\nThis content stays put.");

    let Some((indexer, svc, _pool)) = build_stack(kb_root).await else {
        return;
    };

    let mut manifest = KbManifest::open(kb_root).unwrap();
    let _ = indexer.run(&mut manifest).await.unwrap();

    let res = svc
        .search_collection("vanishing", &VectorFilters::default(), 10, Collection::Kb)
        .await
        .unwrap();
    assert!(
        kb_hits(&res).contains(&"doomed.md"),
        "pre-delete: doomed.md must be findable"
    );

    std::fs::remove_file(&target).unwrap();
    let report2 = indexer.run(&mut manifest).await.unwrap();
    assert_eq!(
        report2.removed, 1,
        "expected 1 removed, got {}",
        report2.removed
    );

    let res2 = svc
        .search_collection("vanishing", &VectorFilters::default(), 10, Collection::Kb)
        .await
        .unwrap();
    assert!(
        !kb_hits(&res2).contains(&"doomed.md"),
        "post-delete: doomed.md must not be returned anymore"
    );
}

#[tokio::test]
async fn run_is_idempotent_under_no_changes() {
    let dir = tempfile::tempdir().unwrap();
    let kb_root = dir.path();
    write_kb_file(kb_root, "stable.md", "# Stable\n\nNothing changes here.");

    let Some((indexer, svc, _pool)) = build_stack(kb_root).await else {
        return;
    };

    let mut manifest = KbManifest::open(kb_root).unwrap();
    let r1 = indexer.run(&mut manifest).await.unwrap();
    assert!(r1.indexed >= 1);

    let r2 = indexer.run(&mut manifest).await.unwrap();
    assert_eq!(r2.indexed, 0, "second run should index nothing new");
    assert_eq!(r2.removed, 0);
    assert_eq!(r2.stale_reindexed, 0);

    let res = svc
        .search_collection("stable", &VectorFilters::default(), 10, Collection::Kb)
        .await
        .unwrap();
    let hits = kb_hits(&res);
    let stable_count = hits.iter().filter(|s| **s == "stable.md").count();
    assert_eq!(
        stable_count, 1,
        "double-run should not produce duplicate hits, got {} for stable.md",
        stable_count
    );
}
