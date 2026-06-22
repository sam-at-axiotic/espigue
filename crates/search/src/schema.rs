//! Phase 2 schema: `vec_entries` (sqlite-vec virtual table), `vec_index`
//! (metadata sidecar joined by rowid), and `embedding_cache`
//! (SHA-256-keyed cache with 90-day TTL).
//!
//! # Ordering with Phase 1
//!
//! This migration depends on Phase 1's `schema_version` table existing and
//! must therefore be run AFTER `alzina_memory::schema::migrate()`. Calling
//! [`migrate`] on a pool that has not seen Phase 1 will fail when the
//! `INSERT INTO schema_version` step runs.
//!
//! The expected boot sequence is:
//!
//! ```ignore
//! alzina_memory::schema::migrate(&pool).await?;
//! search::schema::migrate(&pool).await?;
//! ```
//!
//! # sqlite-vec extension loading (AC-1)
//!
//! The `vec_entries` virtual table requires the `vec0` module provided by
//! the `sqlite-vec` extension. We register the extension globally via
//! [`sqlite3_auto_extension`] before any `CREATE VIRTUAL TABLE` runs, so
//! that every connection sqlx subsequently opens has `vec0` available.
//!
//! Auto-extension registration happens once per process behind a `OnceLock`.
//! If registration fails (or the underlying SQLite build refuses
//! extensions), [`migrate`] logs a `warn!` and SKIPS creating
//! `vec_entries`. The metadata-only tables (`vec_index`,
//! `embedding_cache`) are still created so the cache works and so a
//! [`crate::sqlite_vec::SqliteVecStore`] can detect the missing
//! virtual table and report itself as `enabled = false` per AC-1.

use std::sync::OnceLock;

use base::error::{AlzinaError, AlzinaResult, SearchDetail};
use sqlx::sqlite::SqlitePool;

const SCHEMA_VERSION: i64 = 2;
const SCHEMA_VERSION_DESCRIPTION: &str = "Phase 2: vec_entries, vec_index, embedding_cache";

/// Global once-registration result. `Ok(true)` means the extension was
/// successfully registered; `Ok(false)` means we ran but the C call
/// returned an error code; `Err(_)` means we never tried (shouldn't
/// happen in practice but guarded for safety).
static AUTO_EXTENSION: OnceLock<bool> = OnceLock::new();

/// Helper used by [`migrate`] to map sqlx errors into a search-degraded
/// `AlzinaError::Search` with a populated `degradation_reason`.
fn search_err(message: impl Into<String>, reason: impl Into<String>) -> AlzinaError {
    let reason = reason.into();
    AlzinaError::Search(SearchDetail {
        message: message.into(),
        degraded: true,
        degradation_reason: Some(reason),
    })
}

/// Register the sqlite-vec C extension as a SQLite auto-extension.
///
/// The `sqlite-vec` crate exposes `sqlite3_vec_init` as a raw
/// `extern "C"` symbol — the extension's init entry point. It does NOT
/// internally call `sqlite3_auto_extension`; that's the caller's job.
/// We register it via `libsqlite3_sys::sqlite3_auto_extension`, passing
/// `sqlite3_vec_init` as the entry-point function pointer. Every
/// SQLite connection opened *after* this call will have the `vec0`
/// virtual-table module available.
///
/// Gated behind a `OnceLock` so we only register once per process —
/// registering the same auto-extension twice is harmless at the SQLite
/// level but adds tracing noise we don't need.
///
/// # Safety
/// `transmute` between two `extern "C"` function-pointer types of
/// equivalent ABI shape (no-arg → SQLite's expected entry-point
/// signature) is sound. `sqlite3_auto_extension` is documented to
/// store the pointer for use on future `sqlite3_open*` calls; it has
/// no other side effects. Returns `true` if the C call returned
/// `SQLITE_OK`, `false` otherwise.
pub fn register_sqlite_vec_extension() -> bool {
    *AUTO_EXTENSION.get_or_init(|| {
        // SAFETY: see the function-level safety doc.
        let rc = unsafe {
            libsqlite3_sys::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )))
        };
        if rc == libsqlite3_sys::SQLITE_OK {
            tracing::info!("sqlite-vec auto-extension registered");
            true
        } else {
            tracing::error!(rc, "sqlite3_auto_extension returned non-OK code");
            false
        }
    })
}

/// Run all Phase 2 schema migrations. Idempotent (uses `IF NOT EXISTS`).
///
/// AC-1 degradation: if the sqlite-vec extension cannot be registered or
/// the `vec_entries` virtual table fails to create, we log a `warn!` and
/// skip creating it. `vec_index` and `embedding_cache` are still created
/// so the cache and metadata-only paths work. Callers should construct
/// [`crate::sqlite_vec::SqliteVecStore`] AFTER this — it probes for
/// `vec_entries` and reports `is_enabled() == false` when missing,
/// causing every `insert`/`search` to surface a structured
/// `AlzinaError::Search` with `degraded: true`.
pub async fn migrate(pool: &SqlitePool) -> AlzinaResult<()> {
    // Best-effort: register the sqlite-vec extension before any virtual
    // table CREATE. If this fails we still create the metadata tables and
    // proceed in a degraded state (AC-1).
    let extension_loaded = register_sqlite_vec_extension();

    // Try to create the vec0 virtual table. If extension registration
    // failed earlier, this CREATE will most likely fail too — we catch
    // that error, log a warning, and continue.
    if extension_loaded {
        match sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_entries USING vec0(\
                embedding float[1024]\
            )",
        )
        .execute(pool)
        .await
        {
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(
                    "sqlite-vec extension unavailable; vector search will degrade to FTS5-only \
                     (CREATE VIRTUAL TABLE vec_entries failed: {e})"
                );
            }
        }
    } else {
        tracing::warn!("sqlite-vec extension unavailable; vector search will degrade to FTS5-only");
    }

    // Metadata sidecar — joined to vec_entries by rowid.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS vec_index (\
            rowid INTEGER PRIMARY KEY,\
            source_type TEXT NOT NULL,\
            source_id TEXT NOT NULL,\
            chunk_index INTEGER NOT NULL DEFAULT 0,\
            content_preview TEXT NOT NULL,\
            source_agent TEXT,\
            source_date TEXT,\
            weave_id TEXT,\
            section TEXT,\
            domain TEXT,\
            indexed_at TEXT NOT NULL\
        )",
    )
    .execute(pool)
    .await
    .map_err(|e| search_err(format!("vec_index: {e}"), format!("vec_index create: {e}")))?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_vec_index_source \
         ON vec_index(source_type, source_id, chunk_index)",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("idx_vec_index_source: {e}"),
            format!("idx_vec_index_source create: {e}"),
        )
    })?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_vec_index_source_date \
         ON vec_index(source_date)",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("idx_vec_index_source_date: {e}"),
            format!("idx_vec_index_source_date create: {e}"),
        )
    })?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_vec_index_source_type_date \
         ON vec_index(source_type, source_date)",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("idx_vec_index_source_type_date: {e}"),
            format!("idx_vec_index_source_type_date create: {e}"),
        )
    })?;

    // SHA-256-keyed embedding cache (90-day TTL enforced on access).
    //
    // The cache is workspace-global: a single row keyed only on
    // (content_hash, model) — there is intentionally no per-session
    // partition. This is a deliberate design choice (synthesis §5.2):
    // identical content embedded across chat sessions hits the same
    // cache row, saving Jina API spend.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS embedding_cache (\
            content_hash TEXT PRIMARY KEY,\
            model TEXT NOT NULL,\
            dimensions INTEGER NOT NULL,\
            vector BLOB NOT NULL,\
            created_at TEXT NOT NULL,\
            accessed_at TEXT NOT NULL\
        )",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("embedding_cache: {e}"),
            format!("embedding_cache create: {e}"),
        )
    })?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_embedding_cache_accessed \
         ON embedding_cache(accessed_at)",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("idx_embedding_cache_accessed: {e}"),
            format!("idx_embedding_cache_accessed create: {e}"),
        )
    })?;

    // Record this migration. INSERT OR IGNORE so re-running migrate() is
    // safe and leaves the `(version=2, ...)` row untouched.
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT OR IGNORE INTO schema_version (version, applied_at, description) \
         VALUES (?, ?, ?)",
    )
    .bind(SCHEMA_VERSION)
    .bind(&now)
    .bind(SCHEMA_VERSION_DESCRIPTION)
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("record schema_version: {e}"),
            format!("schema_version insert: {e}"),
        )
    })?;

    Ok(())
}

/// Test helper: build an in-memory pool with both Phase 1 (memory) and
/// Phase 2 (search) schemas applied. Available for the whole crate's
/// `#[cfg(test)]` modules.
#[cfg(test)]
pub async fn in_memory_pool_with_search_schema() -> AlzinaResult<sqlx::SqlitePool> {
    // alzina-memory is not vendored here. Build the base pool inline:
    // register sqlite-vec BEFORE opening the connection (mirrors the daemon /
    // litreview ordering fix), then create the one `schema_version` table that
    // `crate::schema::migrate` requires from Phase 1.
    let pool = {
        use std::str::FromStr;
        register_sqlite_vec_extension();
        let opts = sqlx::sqlite::SqliteConnectOptions::from_str("sqlite::memory:")
            .map_err(|e| search_err("connect options", e.to_string()))?
            .create_if_missing(true);
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await
            .map_err(|e| search_err("pool connect", e.to_string()))?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .map_err(|e| search_err("schema_version", e.to_string()))?;
        pool
    };
    crate::schema::migrate(&pool).await?;
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn migrate_creates_vec_index_and_embedding_cache() {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        // Both metadata tables MUST exist regardless of whether the
        // sqlite-vec extension loaded.
        sqlx::query("SELECT COUNT(*) FROM vec_index")
            .fetch_one(&pool)
            .await
            .unwrap();
        sqlx::query("SELECT COUNT(*) FROM embedding_cache")
            .fetch_one(&pool)
            .await
            .unwrap();
    }
}
