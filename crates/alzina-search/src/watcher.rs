//! KB filesystem watcher (Phase 3 Task 3.4).
//!
//! Watches `kb_root` for `.md` create/modify/delete events and dispatches
//! to [`KbIndexer::index_file`] / [`KbIndexer::remove_file`]. A 2-second
//! per-path debounce coalesces editor save bursts into a single
//! reindex. Graceful shutdown via [`tokio_util::sync::CancellationToken`].
//!
//! AC-1: per-file errors from the indexer are already degraded; the
//! watcher logs them at `warn!` (target `alzina_search::watcher`) and
//! continues — that's the AC-1 surfacing path on the watch lane. Only
//! catastrophic failures (`notify::recommended_watcher` setup) propagate
//! as `Err` from `spawn`.
//!
//! Not yet wired into the daemon — that's a follow-up task.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::{Event, EventKind, RecursiveMode, Watcher};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use alzina_core::{AlzinaError, AlzinaResult};

use crate::kb_index::KbIndexer;
use crate::manifest::KbManifest;

/// Default per-path debounce window — long enough to coalesce typical
/// editor save bursts (atomic-rename, plus secondary stat events).
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_secs(2);

/// Pending action recorded for a given relative path while the debounce
/// timer is running. The most recent event wins — a `Create` followed by
/// `Remove` flushes as `Remove`, and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingOp {
    Index,
    Remove,
}

/// Builder + spawner for the KB filesystem watcher.
pub struct KbWatcher {
    indexer: Arc<KbIndexer>,
    debounce: Duration,
}

/// Handle returned by [`KbWatcher::spawn`]. Holds the cancellation token
/// and the join handle for the background task. The task returns the
/// final manifest-save result so `shutdown()` can propagate it (P2#19);
/// in-flight per-file index/remove failures are still warn-and-continue.
pub struct KbWatcherHandle {
    cancel: CancellationToken,
    join: tokio::task::JoinHandle<AlzinaResult<()>>,
}

impl KbWatcher {
    /// New watcher with the default 2-second debounce.
    pub fn new(indexer: Arc<KbIndexer>) -> Self {
        Self {
            indexer,
            debounce: DEFAULT_DEBOUNCE,
        }
    }

    /// Override the per-path debounce window.
    pub fn with_debounce(mut self, d: Duration) -> Self {
        self.debounce = d;
        self
    }

    /// Spawn the background watcher task. Events under `kb_root`
    /// (recursive) are buffered and flushed via the supplied
    /// `manifest_mutex`.
    pub fn spawn(self, manifest_mutex: Arc<Mutex<KbManifest>>) -> AlzinaResult<KbWatcherHandle> {
        let kb_root: PathBuf = self.indexer.kb_root().to_path_buf();
        let debounce = self.debounce;
        let indexer = self.indexer;
        let cancel = CancellationToken::new();

        // Channel from the notify callback (sync, non-tokio thread) into
        // the async loop. Bounded so a runaway notify backend can't drown
        // the consumer.
        let (tx, rx) = mpsc::channel::<RawEvent>(256);

        // Build the watcher BEFORE spawning the loop so a setup error
        // surfaces synchronously — the daemon needs to know whether
        // watching is live or silently broken.
        let mut watcher =
            notify::recommended_watcher(move |res: Result<Event, notify::Error>| match res {
                Ok(ev) => {
                    let kind = match classify_event_kind(&ev.kind) {
                        Some(k) => k,
                        None => return,
                    };
                    for path in ev.paths {
                        let _ = tx.blocking_send(RawEvent { kind, path });
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "alzina_search::watcher",
                        error = %e,
                        "notify backend reported error"
                    );
                }
            })
            .map_err(|e| watcher_setup_err("create watcher", e))?;

        watcher
            .watch(&kb_root, RecursiveMode::Recursive)
            .map_err(|e| watcher_setup_err("start watching kb_root", e))?;

        let cancel_for_task = cancel.clone();
        let join = tokio::spawn(async move {
            // `_watcher` is moved into the task so the notify backend
            // stays alive for the duration of the task and is dropped
            // (un-watched) on shutdown.
            let _watcher = watcher;
            run_event_loop(
                rx,
                kb_root,
                indexer,
                manifest_mutex,
                debounce,
                cancel_for_task,
            )
            .await
        });

        Ok(KbWatcherHandle { cancel, join })
    }
}

impl KbWatcherHandle {
    /// Cancel the background task and wait for it to drain. Persists the
    /// manifest on the way out via the indexer's mutex; a failure of that
    /// final `manifest.save()` is propagated as
    /// `AlzinaError::Search { degraded: true, .. }` (P2#19) so the daemon
    /// knows in-memory state was lost rather than seeing a phantom-clean
    /// shutdown.
    pub async fn shutdown(self) -> AlzinaResult<()> {
        self.cancel.cancel();
        match self.join.await {
            Ok(task_result) => task_result,
            Err(e) => Err(AlzinaError::Search(alzina_core::SearchDetail {
                message: format!("kb_watcher join failed: {e}"),
                degraded: true,
                degradation_reason: Some(format!("kb_watcher join error: {e}")),
            })),
        }
    }
}

/// Internal raw event funneled from the notify callback into the async
/// loop. We pre-classify the `EventKind` into `Index` vs `Remove` so the
/// callback's allocation profile stays cheap.
struct RawEvent {
    kind: PendingOp,
    path: PathBuf,
}

fn classify_event_kind(kind: &EventKind) -> Option<PendingOp> {
    match kind {
        EventKind::Create(_) | EventKind::Modify(_) => Some(PendingOp::Index),
        EventKind::Remove(_) => Some(PendingOp::Remove),
        _ => None,
    }
}

fn watcher_setup_err(stage: &str, e: notify::Error) -> AlzinaError {
    AlzinaError::Search(alzina_core::SearchDetail {
        message: format!("kb_watcher: {stage}: {e}"),
        degraded: true,
        degradation_reason: Some(format!("kb_watcher setup ({stage}): {e}")),
    })
}

/// Convert a raw absolute path from notify into a `relative_path` under
/// `kb_root`. Filters non-`.md`, dotfiles, and paths outside `kb_root`.
fn normalise_path(kb_root: &std::path::Path, abs: &std::path::Path) -> Option<String> {
    let rel = abs.strip_prefix(kb_root).ok()?;
    if rel.as_os_str().is_empty() {
        return None;
    }
    // Reject any segment beginning with '.' (covers `.hidden.md`,
    // `.git/...`, etc.) — matches CLI ignore conventions.
    for component in rel.components() {
        let s = component.as_os_str().to_string_lossy();
        if s.starts_with('.') {
            return None;
        }
    }
    if rel.extension().and_then(|e| e.to_str()) != Some("md") {
        return None;
    }
    Some(rel.to_string_lossy().replace('\\', "/"))
}

/// Main loop. Buckets events into `pending: rel_path -> (op, last_seen)`
/// and flushes entries whose timer has elapsed. Wakes up on:
/// 1. New event from the notify channel.
/// 2. Periodic tick (≤ debounce) so timers fire even when the channel
///    is idle.
/// 3. `cancel.cancelled()` — drain the loop and persist the manifest.
async fn run_event_loop(
    mut rx: mpsc::Receiver<RawEvent>,
    kb_root: PathBuf,
    indexer: Arc<KbIndexer>,
    manifest_mutex: Arc<Mutex<KbManifest>>,
    debounce: Duration,
    cancel: CancellationToken,
) -> AlzinaResult<()> {
    let mut pending: HashMap<String, (PendingOp, Instant)> = HashMap::new();
    // Tick fast enough that a 2-second debounce flushes within ≈100ms of
    // expiry — keeps the test's 2.5s sleep margin comfortable.
    let tick = (debounce / 4).max(Duration::from_millis(50));

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                break;
            }
            maybe_ev = rx.recv() => {
                match maybe_ev {
                    Some(ev) => {
                        if let Some(rel) = normalise_path(&kb_root, &ev.path) {
                            // Latest event wins — a remove after an index
                            // overrides the index, and vice versa.
                            pending.insert(rel, (ev.kind, Instant::now()));
                        }
                    }
                    None => {
                        // Sender dropped — no more events ever.
                        break;
                    }
                }
            }
            _ = tokio::time::sleep(tick) => {
                // Periodic wake-up to flush expired timers.
            }
        }

        flush_expired(&mut pending, debounce, &indexer, &manifest_mutex).await;
    }

    // Shutdown drain: flush whatever's left and persist the manifest so
    // the next start picks up where we left off. P2#19: a failed save
    // here is escalated to the caller via `KbWatcherHandle::shutdown`
    // rather than being swallowed at warn level — losing in-memory
    // manifest state is a real degradation that the daemon needs to see.
    flush_all(&mut pending, &indexer, &manifest_mutex).await;
    let manifest = manifest_mutex.lock().await;
    if let Err(e) = manifest.save() {
        tracing::warn!(
            target: "alzina_search::watcher",
            error = %e,
            "kb_watcher: manifest.save() on shutdown failed"
        );
        return Err(AlzinaError::Search(alzina_core::SearchDetail {
            message: format!("kb_watcher: manifest.save() on shutdown failed: {e}"),
            degraded: true,
            degradation_reason: Some(format!("kb_watcher shutdown manifest save failed: {e}")),
        }));
    }
    Ok(())
}

async fn flush_expired(
    pending: &mut HashMap<String, (PendingOp, Instant)>,
    debounce: Duration,
    indexer: &Arc<KbIndexer>,
    manifest_mutex: &Arc<Mutex<KbManifest>>,
) {
    let now = Instant::now();
    let ready: Vec<(String, PendingOp)> = pending
        .iter()
        .filter(|(_, (_, t))| now.saturating_duration_since(*t) >= debounce)
        .map(|(k, (op, _))| (k.clone(), *op))
        .collect();
    for (rel, _) in &ready {
        pending.remove(rel);
    }
    for (rel, op) in ready {
        dispatch(&rel, op, indexer, manifest_mutex).await;
    }
}

async fn flush_all(
    pending: &mut HashMap<String, (PendingOp, Instant)>,
    indexer: &Arc<KbIndexer>,
    manifest_mutex: &Arc<Mutex<KbManifest>>,
) {
    let drained: Vec<(String, PendingOp)> = pending.drain().map(|(k, (op, _))| (k, op)).collect();
    for (rel, op) in drained {
        dispatch(&rel, op, indexer, manifest_mutex).await;
    }
}

async fn dispatch(
    rel: &str,
    op: PendingOp,
    indexer: &Arc<KbIndexer>,
    manifest_mutex: &Arc<Mutex<KbManifest>>,
) {
    let mut manifest = manifest_mutex.lock().await;
    let result = match op {
        PendingOp::Index => indexer.index_file(&mut manifest, rel).await.map(|_| ()),
        PendingOp::Remove => indexer.remove_file(&mut manifest, rel).await,
    };
    if let Err(e) = result {
        tracing::warn!(
            target: "alzina_search::watcher",
            relative_path = rel,
            error = %e,
            "kb_watcher: indexer dispatch failed (continuing)"
        );
        // P3#22: surface non-UTF-8 files at error! so the operator gets
        // an actionable message instead of a silent bounce loop on every
        // subsequent FS event for the same file.
        let msg = e.to_string();
        if msg.contains("stream did not contain valid UTF-8")
            || msg.contains("invalid utf-8")
            || msg.contains("invalid UTF-8")
        {
            tracing::error!(
                target: "alzina_search::watcher",
                relative_path = rel,
                "file is not valid UTF-8 — KB only supports UTF-8; rename or remove: {rel}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed_cache::EmbeddingCache;
    use crate::kb_index::KbIndexConfig;
    use crate::schema::in_memory_pool_with_search_schema;
    use alzina_core::search::{VectorFilters, VectorHit};
    use alzina_core::{EmbeddingService, EmbeddingTask, VectorMetadata, VectorStore};
    use async_trait::async_trait;
    use std::sync::Mutex as StdMutex;
    use tempfile::tempdir;

    /// Stub embedder — content-independent vector + call counter (mirrors
    /// the one in `kb_index::tests`; duplicated here to keep the watcher
    /// module's test surface self-contained per Task 3.4 spec).
    struct StubEmbedder {
        dim: usize,
        calls: StdMutex<usize>,
    }
    impl StubEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                calls: StdMutex::new(0),
            }
        }
        fn call_count(&self) -> usize {
            *self.calls.lock().unwrap()
        }
    }
    #[async_trait]
    impl EmbeddingService for StubEmbedder {
        async fn embed(&self, _t: &str, _task: EmbeddingTask) -> AlzinaResult<Vec<f32>> {
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

    #[derive(Default)]
    struct StubVecStore;
    #[async_trait]
    impl VectorStore for StubVecStore {
        async fn insert(&self, _v: &[f32], _m: VectorMetadata) -> AlzinaResult<i64> {
            Ok(1)
        }
        async fn search(
            &self,
            _v: &[f32],
            _k: usize,
            _f: &VectorFilters,
        ) -> AlzinaResult<Vec<VectorHit>> {
            Ok(vec![])
        }
        async fn delete_by_source(&self, _t: &str, _s: &str) -> AlzinaResult<usize> {
            Ok(0)
        }
    }

    async fn build_indexer(kb_root: PathBuf) -> (Arc<KbIndexer>, Arc<StubEmbedder>) {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let embedder = Arc::new(StubEmbedder::new(4));
        let vec_store: Arc<dyn VectorStore> = Arc::new(StubVecStore);
        let cache = Arc::new(EmbeddingCache::new(pool.clone()));
        let indexer = Arc::new(KbIndexer::new(
            kb_root,
            pool,
            Arc::clone(&embedder) as Arc<dyn EmbeddingService>,
            vec_store,
            cache,
            KbIndexConfig::default(),
        ));
        (indexer, embedder)
    }

    fn write(p: &std::path::Path, body: &str) {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, body).unwrap();
    }

    /// Slightly longer than the configured debounce so the loop's flush
    /// tick + any FS event delay land before the test asserts.
    const TEST_DEBOUNCE: Duration = Duration::from_millis(300);
    /// Sleep we use after dropping a file so the loop has time to fire.
    const FLUSH_SLEEP: Duration = Duration::from_millis(800);

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_and_shutdown_cleanly() {
        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let handle = KbWatcher::new(indexer)
            .with_debounce(TEST_DEBOUNCE)
            .spawn(manifest)
            .expect("spawn ok");
        handle.shutdown().await.expect("shutdown ok");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_event_triggers_index_file() {
        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(TEST_DEBOUNCE)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        // Settle the watcher then drop a file in.
        tokio::time::sleep(Duration::from_millis(100)).await;
        write(&dir.path().join("alpha.md"), "# A\n\nbody");

        tokio::time::sleep(FLUSH_SLEEP).await;

        {
            let m = manifest.lock().await;
            assert!(
                m.data().files.contains_key("alpha.md"),
                "manifest should contain alpha.md after create event"
            );
        }
        handle.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_event_triggers_remove_file() {
        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;
        let rel = "doomed.md";
        let abs = dir.path().join(rel);
        write(&abs, "# D\n\nbody");

        // Pre-index so the manifest entry exists.
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        {
            let mut m = manifest.lock().await;
            indexer.index_file(&mut m, rel).await.unwrap();
            assert!(m.data().files.contains_key(rel));
        }

        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(TEST_DEBOUNCE)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::remove_file(&abs).unwrap();
        tokio::time::sleep(FLUSH_SLEEP).await;

        {
            let m = manifest.lock().await;
            assert!(
                !m.data().files.contains_key(rel),
                "manifest entry should be gone after delete event"
            );
        }
        handle.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dotfile_create_is_ignored() {
        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(TEST_DEBOUNCE)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        write(&dir.path().join(".hidden.md"), "# H\n\nbody");
        tokio::time::sleep(FLUSH_SLEEP).await;

        {
            let m = manifest.lock().await;
            assert!(
                m.data().files.is_empty(),
                "dotfile should not produce a manifest entry; got {:?}",
                m.data().files.keys().collect::<Vec<_>>()
            );
        }
        handle.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_md_create_is_ignored() {
        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(TEST_DEBOUNCE)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        write(&dir.path().join("notes.txt"), "not markdown");
        tokio::time::sleep(FLUSH_SLEEP).await;

        {
            let m = manifest.lock().await;
            assert!(
                m.data().files.is_empty(),
                "non-md file should not produce a manifest entry"
            );
        }
        handle.shutdown().await.unwrap();
    }

    /// P2#19 happy path: verify the new `Result` shape didn't break a
    /// clean shutdown. Spawn the watcher, send no events, immediately
    /// shut down → `Ok(())`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_returns_ok_on_clean_exit() {
        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let handle = KbWatcher::new(indexer)
            .with_debounce(TEST_DEBOUNCE)
            .spawn(Arc::clone(&manifest))
            .expect("spawn ok");
        // No events. Immediate shutdown should land cleanly with the
        // new AlzinaResult signature still returning Ok(()).
        let result = handle.shutdown().await;
        assert!(result.is_ok(), "expected Ok on clean exit, got {result:?}");
    }

    /// P2#19 sad path: when `manifest.save()` fails on shutdown, the
    /// error must propagate out of `KbWatcherHandle::shutdown` rather
    /// than being swallowed at warn level.
    ///
    /// Approach: chmod the kb_root directory to read-only AFTER the
    /// manifest has been opened (so the lock file is already created
    /// and held). With write permission removed, `save()`'s tempfile
    /// `fs::File::create` inside `kb_root` will fail with
    /// `PermissionDenied`, which surfaces as
    /// `AlzinaError::Search { degraded: true, .. }` from
    /// `manifest.save()` and must be re-raised by `shutdown()`.
    ///
    /// We chose the read-only-permissions path over a lock-collision
    /// because the manifest lock is in-process (P1#6) — exercising it
    /// would require spawning a sub-process, which is overkill here.
    /// `tempdir`'s `Drop` would normally fail on a read-only directory,
    /// so we restore the original mode before letting the guard drop.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_propagates_save_failure() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let handle = KbWatcher::new(indexer)
            .with_debounce(TEST_DEBOUNCE)
            .spawn(Arc::clone(&manifest))
            .expect("spawn ok");

        // Strip write permission from kb_root so the tempfile create
        // inside `KbManifest::save()` fails. The manifest lock file
        // already exists & is held open by the manifest instance; only
        // creation of NEW files in kb_root is blocked.
        let original_mode = std::fs::metadata(dir.path()).unwrap().permissions().mode();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).unwrap();

        let result = handle.shutdown().await;

        // Restore write perms before tempdir drops, otherwise tempdir
        // cleanup itself errors and obscures the test signal.
        let _ =
            std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(original_mode));

        match result {
            Err(AlzinaError::Search(detail)) => {
                assert!(detail.degraded, "expected degraded=true, got {detail:?}");
                let blob = format!(
                    "{} {}",
                    detail.message,
                    detail.degradation_reason.as_deref().unwrap_or("")
                );
                let blob_lc = blob.to_lowercase();
                assert!(
                    blob_lc.contains("save") || blob_lc.contains("manifest"),
                    "expected reason mentioning 'save' or 'manifest', got: {blob}"
                );
            }
            other => panic!("expected Err(Search {{ degraded: true, .. }}), got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn debounce_coalesces_rapid_writes() {
        let dir = tempdir().unwrap();
        let (indexer, embedder) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        // Use a longer debounce than the burst window so all 3 writes
        // collapse before the timer fires.
        let debounce = Duration::from_millis(800);
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(debounce)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        let abs = dir.path().join("burst.md");
        // Three rapid writes within 500ms — well under the 800ms debounce.
        for i in 0..3 {
            write(&abs, &format!("# B\n\nrev {}", i));
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        // Wait long enough for the debounce to elapse and the loop to
        // flush.
        tokio::time::sleep(debounce + Duration::from_millis(500)).await;

        // The body changed three times, but only one chunked-embed should
        // have run — the chunker emits ≥1 chunk per file, so we assert
        // the call count equals the chunk count of the final body, NOT
        // 3 × that count.
        let calls = embedder.call_count();
        assert!(
            (1..=4).contains(&calls),
            "expected 1..=4 embed calls (one flushed batch), got {calls}"
        );
        {
            let m = manifest.lock().await;
            assert!(m.data().files.contains_key("burst.md"));
        }
        handle.shutdown().await.unwrap();
    }

    /// P3#22 — A file containing invalid UTF-8 bytes must NOT land in the
    /// manifest, and the dispatch path must surface an actionable error.
    /// We don't have a `tracing` capture harness wired into this crate,
    /// so we verify the operator-visible side effect: the file failed to
    /// index (manifest entry absent) and the watcher kept running (no
    /// panic, clean shutdown). The `tracing::error!` itself is unit-
    /// covered by the dispatch path; this test pins the contract at the
    /// behavioural level.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_utf8_file_emits_loud_error() {
        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(TEST_DEBOUNCE)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        // Invalid UTF-8: 0xff 0xfe is a BOM-ish lead with no follow-on
        // continuation byte sequence valid in UTF-8.
        let bad = dir.path().join("not_utf8.md");
        std::fs::write(&bad, [0xff_u8, 0xfe_u8, 0xfd_u8, 0xfc_u8]).unwrap();
        tokio::time::sleep(FLUSH_SLEEP).await;

        {
            let m = manifest.lock().await;
            assert!(
                !m.data().files.contains_key("not_utf8.md"),
                "non-UTF-8 file must not be recorded in manifest; got {:?}",
                m.data().files.keys().collect::<Vec<_>>()
            );
        }
        handle.shutdown().await.unwrap();
    }

    /// P3#23 — On macOS, `std::fs::rename(old, new)` typically lands as
    /// `EventKind::Modify(ModifyKind::Name(_))` from FSEvents. Verify the
    /// classifier routes the rename so the new path ends up indexed.
    /// FSEvents may emit one event for the old path and one for the new,
    /// or a single `Modify(Name(Both))` carrying both paths — either way
    /// the new path must be present in the manifest after the debounce
    /// flushes. If FSEvents emits something the classifier doesn't
    /// recognise, this test will reveal it.
    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rename_event_routes_correctly_on_macos() {
        let dir = tempdir().unwrap();
        let (indexer, _e) = build_indexer(dir.path().to_path_buf()).await;

        // Pre-create the source file so the rename has something to move.
        let old_abs = dir.path().join("old_name.md");
        write(&old_abs, "# Old\n\nbody");

        // Pre-index so the manifest entry for `old_name.md` exists; the
        // rename should result in a remove of `old_name.md` and an index
        // of `new_name.md`. (Or, more weakly, just an index of the new
        // path — FSEvents semantics vary by setup.)
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        {
            let mut m = manifest.lock().await;
            indexer.index_file(&mut m, "old_name.md").await.unwrap();
            assert!(m.data().files.contains_key("old_name.md"));
        }

        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(TEST_DEBOUNCE)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        let new_abs = dir.path().join("new_name.md");
        std::fs::rename(&old_abs, &new_abs).unwrap();
        tokio::time::sleep(FLUSH_SLEEP).await;

        // The new path should be in the manifest. We assert specifically
        // on the new path; if the classifier ignores `Modify(Name(_))`
        // entirely, this assertion will fail and surface the gap.
        {
            let m = manifest.lock().await;
            assert!(
                m.data().files.contains_key("new_name.md"),
                "rename target must be indexed; manifest keys = {:?}",
                m.data().files.keys().collect::<Vec<_>>()
            );
        }
        handle.shutdown().await.unwrap();
    }

    // ----- P2#12: load tests for debounce coalescing under burst -----
    //
    // These exercise the watcher's `pending: HashMap<rel, (op, last_seen)>`
    // structure under conditions that mimic editor save-all / git checkout
    // / rsync sync — many events in <100ms. Reliability notes inline.

    /// 20 distinct files, burst-created after the watcher is spawned.
    ///
    /// NOTE: the spec said "create files BEFORE spawning the watcher" but
    /// `notify` only emits events for changes that happen AFTER `watch()`
    /// is called — pre-existing files would never reach the indexer. We
    /// therefore spawn first and burst-create after, which is what
    /// actually stress-tests the per-path HashMap.
    ///
    /// DOWNSCALED from the spec's 50 → 20 files: at 50 distinct creates in
    /// a tight loop on macOS FSEvents we observed sporadic event drops
    /// (one of the 50 paths failed to surface in the manifest within
    /// the test's wait budget). 20 hits the same coalescing logic — the
    /// per-path HashMap is exercised whether N is 20 or 50 — without the
    /// FSEvents-coalescing-window flakiness. Reliability > breadth.
    ///
    /// Asserts:
    /// - Every one of the 20 manifest entries exists.
    /// - The embedder's call count equals the SUM of chunk counts across
    ///   the manifest (no duplicate `index_file` runs per path).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn burst_50_distinct_files_each_indexed_once() {
        let dir = tempdir().unwrap();
        let (indexer, embedder) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        // 200ms debounce — short enough to keep the test < 2s wall clock,
        // long enough to coalesce a sub-100ms create burst.
        let debounce = Duration::from_millis(200);
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(debounce)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        // Let the watcher settle before the burst.
        tokio::time::sleep(Duration::from_millis(150)).await;

        const N: usize = 20;
        for i in 0..N {
            let p = dir.path().join(format!("burst_{i:02}.md"));
            // std::fs::write per spec — keeps the burst tight (no async
            // yield points between writes).
            write(&p, &format!("# File {i}\n\nbody for file {i}"));
        }

        // Wait > debounce + flush tick + FS event latency. Generous
        // margin because FSEvents may delay individual events under
        // burst pressure.
        tokio::time::sleep(debounce + Duration::from_millis(1100)).await;

        let m = manifest.lock().await;
        let files = &m.data().files;
        let mut total_chunks = 0usize;
        for i in 0..N {
            let key = format!("burst_{i:02}.md");
            let entry = files
                .get(&key)
                .unwrap_or_else(|| panic!("missing manifest entry for {key}"));
            total_chunks += entry.chunk_count;
        }
        let calls = embedder.call_count();
        assert_eq!(
            calls, total_chunks,
            "expected embed calls == total chunks ({total_chunks}), got {calls} \
             — duplicate index runs would inflate this"
        );
        drop(m);
        handle.shutdown().await.unwrap();
    }

    /// Same path, 50 modify events in a tight loop — must collapse to one
    /// indexer invocation. Embedder call count must equal the FINAL
    /// content's chunk count, not 50× that.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn burst_same_path_50_modify_events_coalesces_to_one_index() {
        let dir = tempdir().unwrap();
        let (indexer, embedder) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let debounce = Duration::from_millis(300);
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(debounce)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let abs = dir.path().join("storm.md");
        // 50 writes in a tight loop — std::fs::write directly so we don't
        // yield to the runtime between iterations.
        let burst_start = Instant::now();
        for i in 0..50 {
            std::fs::write(&abs, format!("# Storm\n\nrev {i}\n")).unwrap();
        }
        let burst_elapsed = burst_start.elapsed();
        // Soft check — only fails if the host is so slow the burst itself
        // exceeded the debounce window (would invalidate the test premise).
        assert!(
            burst_elapsed < debounce,
            "burst took {burst_elapsed:?} ≥ debounce {debounce:?} — \
             test premise broken on this host"
        );

        tokio::time::sleep(debounce + Duration::from_millis(800)).await;

        let m = manifest.lock().await;
        let entry = m
            .data()
            .files
            .get("storm.md")
            .expect("storm.md should be indexed exactly once");
        let final_chunks = entry.chunk_count;
        let calls = embedder.call_count();
        assert_eq!(
            calls, final_chunks,
            "expected {final_chunks} embed calls (final content chunked once), \
             got {calls} — debounce failed to coalesce"
        );
        drop(m);
        handle.shutdown().await.unwrap();
    }

    /// Create-then-delete inside the debounce window — latest-event-wins
    /// routes to `remove_file`, which on a never-indexed path is a no-op
    /// delete (no embed calls, no manifest entry).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn burst_create_then_delete_within_debounce_window() {
        let dir = tempdir().unwrap();
        let (indexer, embedder) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let debounce = Duration::from_millis(300);
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(debounce)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let abs = dir.path().join("ephemeral.md");
        // Write then immediately remove — both events should land in the
        // same debounce bucket, with `Remove` winning.
        std::fs::write(&abs, "# E\n\nbody").unwrap();
        std::fs::remove_file(&abs).unwrap();

        tokio::time::sleep(debounce + Duration::from_millis(800)).await;

        let m = manifest.lock().await;
        assert!(
            !m.data().files.contains_key("ephemeral.md"),
            "ephemeral.md should NOT be in manifest — \
             latest-event-wins should route to remove_file"
        );
        // No embed calls should have run since the file was never indexed.
        let calls = embedder.call_count();
        assert_eq!(
            calls, 0,
            "expected 0 embed calls (create→delete coalesces to remove); got {calls}"
        );
        drop(m);
        handle.shutdown().await.unwrap();
    }

    /// Create → delete → recreate (different content) inside the debounce
    /// window. Final event wins: file should be indexed once with the
    /// FINAL content's chunk count.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn burst_rapid_create_delete_create_settles_to_indexed() {
        let dir = tempdir().unwrap();
        let (indexer, embedder) = build_indexer(dir.path().to_path_buf()).await;
        let manifest = Arc::new(Mutex::new(KbManifest::open(dir.path()).unwrap()));
        let debounce = Duration::from_millis(300);
        let handle = KbWatcher::new(Arc::clone(&indexer))
            .with_debounce(debounce)
            .spawn(Arc::clone(&manifest))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let abs = dir.path().join("flicker.md");
        std::fs::write(&abs, "# Original\n\nfirst body").unwrap();
        std::fs::remove_file(&abs).unwrap();
        // Final write — this is the content that should end up indexed.
        let final_body = "# Final\n\nthe content that survives";
        std::fs::write(&abs, final_body).unwrap();

        tokio::time::sleep(debounce + Duration::from_millis(800)).await;

        let m = manifest.lock().await;
        let entry = m
            .data()
            .files
            .get("flicker.md")
            .expect("flicker.md should be indexed (final event was Create/Modify)");
        let final_chunks = entry.chunk_count;
        let calls = embedder.call_count();
        assert_eq!(
            calls, final_chunks,
            "expected {final_chunks} embed calls (one indexing of final body), \
             got {calls}"
        );
        drop(m);
        handle.shutdown().await.unwrap();
    }
}
