//! `SearchIndexer` — fire-and-forget vector indexing for the write path.
//!
//! Phase 2 synthesis §5.5: FTS5 rows are inserted **synchronously** in
//! the write path (zero latency cost, transactional with the primary
//! INSERT). Vector embeddings — which require an HTTP round-trip to
//! Jina — are generated **asynchronously** via this indexer. Failures
//! are swallowed and reconciled later by the backfill job (Task 2.9).
//!
//! Two entry points:
//! - [`SearchIndexer::index_entry`] — fire-and-forget. Spawns a tokio
//!   task and returns immediately; failures log at `warn!` and never
//!   propagate.
//! - [`SearchIndexer::index_entry_blocking`] — synchronous. Used by the
//!   backfill job which needs deterministic completion and surfaced
//!   errors.
//!
//! Both paths share [`index_one`]: cache-then-embed, then upsert into
//! the vector store. The embedding cache is consulted FIRST so repeated
//! content (e.g. the same passage indexed across multiple weaves) hits
//! the workspace-global cache instead of a paid Jina call.

use std::sync::Arc;

use alzina_core::{AlzinaResult, EmbeddingService, EmbeddingTask, VectorMetadata, VectorStore};
use chrono::Utc;

use crate::embed_cache::EmbeddingCache;

/// Async fire-and-forget indexer for the vector store.
///
/// Construct once at daemon startup with shared `Arc`-wrapped
/// dependencies and clone freely — `Arc` clones are cheap and every
/// public method is `&self`.
pub struct SearchIndexer {
    embedder: Arc<dyn EmbeddingService>,
    vec_store: Arc<dyn VectorStore>,
    cache: Arc<EmbeddingCache>,
    /// Embedding model name — recorded into the cache so future
    /// multi-model rollouts can filter cache reads by model. For Phase
    /// 2 the call site passes a hardcoded `"jina-embeddings-v3"`.
    model: String,
}

impl SearchIndexer {
    /// Wrap shared dependencies. The `model` string is recorded on
    /// every cache write; pass the same value the embedder reports
    /// (Phase 2: `"jina-embeddings-v3"`).
    pub fn new(
        embedder: Arc<dyn EmbeddingService>,
        vec_store: Arc<dyn VectorStore>,
        cache: Arc<EmbeddingCache>,
        model: String,
    ) -> Self {
        Self {
            embedder,
            vec_store,
            cache,
            model,
        }
    }

    /// Fire-and-forget. Spawns a tokio task to embed and index. Returns
    /// immediately so the write path is never blocked on a Jina HTTP
    /// round-trip. Failures are logged at `warn!` and swallowed —
    /// reconciliation is the backfill job's responsibility.
    ///
    /// `chunk_index` belongs to the caller — pass `0` for non-chunked
    /// sources. `indexed_at` is stamped on the spawned task so it
    /// reflects when the index actually landed.
    pub fn index_entry(&self, content: String, mut metadata: VectorMetadata) {
        let embedder = Arc::clone(&self.embedder);
        let vec_store = Arc::clone(&self.vec_store);
        let cache = Arc::clone(&self.cache);
        let model = self.model.clone();

        tokio::spawn(async move {
            metadata.indexed_at = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

            let source_type = metadata.source_type.clone();
            let source_id = metadata.source_id.clone();

            match index_one(&*embedder, &*vec_store, &*cache, &model, &content, metadata).await {
                Ok(_) => {
                    tracing::debug!(
                        target: "alzina_search::indexer",
                        source_type = %source_type,
                        source_id = %source_id,
                        "vector indexed"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "alzina_search::indexer",
                        source_type = %source_type,
                        source_id = %source_id,
                        error = %e,
                        "vector indexing failed; backfill will reconcile"
                    );
                }
            }
        });
    }

    /// Synchronous variant — embeds, caches, and upserts inline. Used
    /// by the backfill job (Task 2.9) which needs deterministic
    /// completion. Returns errors instead of swallowing them.
    pub async fn index_entry_blocking(
        &self,
        content: &str,
        mut metadata: VectorMetadata,
    ) -> AlzinaResult<i64> {
        metadata.indexed_at = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        index_one(
            &*self.embedder,
            &*self.vec_store,
            &*self.cache,
            &self.model,
            content,
            metadata,
        )
        .await
    }
}

/// `SearchIndexHook` impl — thin pass-through to [`SearchIndexer::index_entry`]
/// so `alzina-memory` stores can schedule write-path vector indexing without
/// taking a direct dependency on `alzina-search`.
impl alzina_core::SearchIndexHook for SearchIndexer {
    fn schedule_index(&self, content: String, metadata: alzina_core::VectorMetadata) {
        self.index_entry(content, metadata);
    }
}

/// Internal: cache-then-embed, then upsert. Shared by both the
/// fire-and-forget and blocking paths so behaviour stays identical.
///
/// Cache contract: if the cached vector's dimensionality doesn't match
/// what the embedder advertises, we treat it as a miss — defensive
/// against multi-model coexistence during rollout. A cache write
/// failure is logged but doesn't abort the insert (best-effort cache).
async fn index_one(
    embedder: &dyn EmbeddingService,
    vec_store: &dyn VectorStore,
    cache: &EmbeddingCache,
    model: &str,
    content: &str,
    metadata: VectorMetadata,
) -> AlzinaResult<i64> {
    let hash = EmbeddingCache::hash_content(content);

    let vector = match cache.get_cached(&hash).await {
        Ok(Some(v)) if v.len() == embedder.dimensions() => v,
        _ => {
            let v = embedder.embed(content, EmbeddingTask::Passage).await?;
            if let Err(e) = cache.put_cached(&hash, model, v.len(), &v).await {
                tracing::warn!(
                    target: "alzina_search::indexer",
                    error = %e,
                    "embedding cache write failed (continuing)"
                );
            }
            v
        }
    };

    vec_store.insert(&vector, metadata).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::in_memory_pool_with_search_schema;
    use alzina_core::error::{AlzinaError, SearchDetail};
    use alzina_core::search::{VectorFilters, VectorHit};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Stub embedder — counts calls and returns a deterministic vector
    /// of the configured dimensionality. Optionally fails or sleeps.
    struct StubEmbedder {
        dim: usize,
        calls: AtomicUsize,
        fail: bool,
        sleep_ms: u64,
    }

    impl StubEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                calls: AtomicUsize::new(0),
                fail: false,
                sleep_ms: 0,
            }
        }
        fn failing(dim: usize) -> Self {
            Self {
                dim,
                calls: AtomicUsize::new(0),
                fail: true,
                sleep_ms: 0,
            }
        }
        fn slow(dim: usize, sleep_ms: u64) -> Self {
            Self {
                dim,
                calls: AtomicUsize::new(0),
                fail: false,
                sleep_ms,
            }
        }
        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl EmbeddingService for StubEmbedder {
        async fn embed(&self, _text: &str, _task: EmbeddingTask) -> AlzinaResult<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.sleep_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.sleep_ms)).await;
            }
            if self.fail {
                return Err(AlzinaError::Search(SearchDetail {
                    message: "stub embed failure".into(),
                    degraded: true,
                    degradation_reason: Some("stub".into()),
                }));
            }
            // Deterministic content-independent vector — sufficient for
            // round-trip and call-count assertions.
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

    /// Stub vector store — records every (vector, metadata) tuple
    /// passed to `insert`. Returns rowid=N where N is the running
    /// count.
    struct StubVecStore {
        inserts: tokio::sync::Mutex<Vec<(Vec<f32>, VectorMetadata)>>,
    }

    impl StubVecStore {
        fn new() -> Self {
            Self {
                inserts: tokio::sync::Mutex::new(Vec::new()),
            }
        }
        async fn snapshot(&self) -> Vec<(Vec<f32>, VectorMetadata)> {
            self.inserts.lock().await.clone()
        }
    }

    #[async_trait]
    impl VectorStore for StubVecStore {
        async fn insert(&self, vector: &[f32], metadata: VectorMetadata) -> AlzinaResult<i64> {
            let mut g = self.inserts.lock().await;
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
            _source_type: &str,
            _source_id: &str,
        ) -> AlzinaResult<usize> {
            Ok(0)
        }
    }

    fn sample_metadata() -> VectorMetadata {
        VectorMetadata {
            source_type: "weave".into(),
            source_id: "w-42".into(),
            chunk_index: 0,
            content_preview: "hello world".into(),
            source_agent: Some("smidr".into()),
            source_date: Some("2026-04-29".into()),
            weave_id: Some("w-42".into()),
            section: Some("body".into()),
            domain: Some("research".into()),
            indexed_at: String::new(),
        }
    }

    async fn build_indexer(
        embedder: Arc<dyn EmbeddingService>,
        vec_store: Arc<dyn VectorStore>,
    ) -> (SearchIndexer, Arc<EmbeddingCache>) {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let cache = Arc::new(EmbeddingCache::new(pool));
        let indexer = SearchIndexer::new(
            embedder,
            vec_store,
            Arc::clone(&cache),
            "jina-embeddings-v3".into(),
        );
        (indexer, cache)
    }

    #[tokio::test]
    async fn index_entry_blocking_inserts_into_vec_store() {
        let embedder = Arc::new(StubEmbedder::new(4));
        let store = Arc::new(StubVecStore::new());
        let (indexer, _cache) = build_indexer(
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            Arc::clone(&store) as Arc<dyn VectorStore>,
        )
        .await;

        let rowid = indexer
            .index_entry_blocking("hello world", sample_metadata())
            .await
            .expect("blocking insert succeeds");
        assert_eq!(rowid, 1);

        let snap = store.snapshot().await;
        assert_eq!(snap.len(), 1);
        let (vec, md) = &snap[0];
        assert_eq!(vec.len(), 4, "vector dim matches embedder");
        assert_eq!(md.source_type, "weave");
        assert_eq!(md.source_id, "w-42");
        assert_eq!(md.content_preview, "hello world");
        assert_eq!(embedder.call_count(), 1, "embedder called exactly once");
    }

    #[tokio::test]
    async fn index_entry_blocking_uses_cache_on_repeat() {
        let embedder = Arc::new(StubEmbedder::new(4));
        let store = Arc::new(StubVecStore::new());
        let (indexer, _cache) = build_indexer(
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            Arc::clone(&store) as Arc<dyn VectorStore>,
        )
        .await;

        indexer
            .index_entry_blocking("repeated content", sample_metadata())
            .await
            .unwrap();
        indexer
            .index_entry_blocking("repeated content", sample_metadata())
            .await
            .unwrap();

        assert_eq!(
            embedder.call_count(),
            1,
            "second call hits the embedding cache"
        );
        let snap = store.snapshot().await;
        assert_eq!(
            snap.len(),
            2,
            "cache is for embeddings, not vec_store dedup"
        );
    }

    #[tokio::test]
    async fn index_entry_fire_and_forget_does_not_block() {
        let embedder = Arc::new(StubEmbedder::slow(4, 500));
        let store = Arc::new(StubVecStore::new());
        let (indexer, _cache) = build_indexer(
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            Arc::clone(&store) as Arc<dyn VectorStore>,
        )
        .await;

        let started = std::time::Instant::now();
        indexer.index_entry("hello world".into(), sample_metadata());
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(50),
            "fire-and-forget returned in {elapsed:?}"
        );

        // Let the spawned task complete.
        tokio::time::sleep(Duration::from_secs(1)).await;
        let snap = store.snapshot().await;
        assert_eq!(snap.len(), 1, "spawned task eventually inserted");
    }

    #[tokio::test]
    async fn index_entry_swallows_embed_errors() {
        let embedder = Arc::new(StubEmbedder::failing(4));
        let store = Arc::new(StubVecStore::new());
        let (indexer, _cache) = build_indexer(
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            Arc::clone(&store) as Arc<dyn VectorStore>,
        )
        .await;

        // Must NOT panic; failure is logged + swallowed.
        indexer.index_entry("doomed".into(), sample_metadata());
        tokio::time::sleep(Duration::from_millis(100)).await;

        let snap = store.snapshot().await;
        assert!(snap.is_empty(), "no insert when embed fails");
    }

    #[tokio::test]
    async fn index_entry_blocking_propagates_embed_error() {
        let embedder = Arc::new(StubEmbedder::failing(4));
        let store = Arc::new(StubVecStore::new());
        let (indexer, _cache) = build_indexer(
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            Arc::clone(&store) as Arc<dyn VectorStore>,
        )
        .await;

        let err = indexer
            .index_entry_blocking("doomed", sample_metadata())
            .await
            .expect_err("must surface embed failure");
        match err {
            AlzinaError::Search(d) => {
                assert!(d.degraded, "degraded flag preserved");
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn searchindexer_impls_search_index_hook() {
        // Validates the `SearchIndexHook` impl compiles, is dyn-compatible,
        // and forwards to the spawned-task `index_entry` path. We don't
        // assert on the spawned insert here — coverage already exists in
        // `index_entry_fire_and_forget_does_not_block`.
        let embedder = Arc::new(StubEmbedder::new(4));
        let store = Arc::new(StubVecStore::new());
        let (indexer, _cache) = build_indexer(
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            Arc::clone(&store) as Arc<dyn VectorStore>,
        )
        .await;

        let hook: Arc<dyn alzina_core::SearchIndexHook> = Arc::new(indexer);
        // No panic — schedule_index returns immediately.
        hook.schedule_index("hook test".into(), sample_metadata());

        // Allow the spawned task to drain so it doesn't outlive the test.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn index_entry_blocking_records_indexed_at() {
        let embedder = Arc::new(StubEmbedder::new(4));
        let store = Arc::new(StubVecStore::new());
        let (indexer, _cache) = build_indexer(
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            Arc::clone(&store) as Arc<dyn VectorStore>,
        )
        .await;

        let before = Utc::now();
        indexer
            .index_entry_blocking("stamped", sample_metadata())
            .await
            .unwrap();
        let snap = store.snapshot().await;
        let stamp = &snap[0].1.indexed_at;
        assert!(!stamp.is_empty(), "indexed_at populated");

        // Parses as RFC3339 / ISO-8601.
        let parsed =
            chrono::DateTime::parse_from_rfc3339(stamp).expect("indexed_at parses as RFC3339");
        assert!(
            parsed.with_timezone(&Utc) >= before - chrono::Duration::seconds(1),
            "indexed_at is not absurdly in the past"
        );
    }
}
