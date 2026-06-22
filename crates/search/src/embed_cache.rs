//! SHA-256-keyed embedding cache with a 90-day TTL.
//!
//! Per synthesis §5.2 the cache is **workspace-global**: identical content
//! embedded across chat sessions hits the same cache row, saving Jina API
//! spend. The cache table is created by [`crate::schema::migrate`].
//!
//! TTL semantics: each entry has `created_at` (immutable) and
//! `accessed_at` (touched on every successful read). The 90-day TTL is
//! measured from `created_at`, NOT `accessed_at` — so re-using a cached
//! embedding does NOT extend its lifetime past 90 days. On every
//! [`get_cached`] / [`get_cached_batch`] call, expired rows are deleted
//! ("delete on access").
//!
//! BLOB encoding: vectors are stored as little-endian f32 byte sequences.
//! Decoding is byte-exact and validates that the BLOB length is a
//! multiple of 4. Length-mismatched BLOBs are surfaced as
//! `AlzinaError::Memory` with a clear reason — the upper layer decides
//! whether that's a search-degradation event.
//!
//! [`get_cached`]: EmbeddingCache::get_cached
//! [`get_cached_batch`]: EmbeddingCache::get_cached_batch

use base::error::{AlzinaError, AlzinaResult};
use chrono::Utc;
use sha2::{Digest, Sha256};
use sqlx::sqlite::SqlitePool;

/// 90-day TTL — entries are deleted on first access after this window.
const TTL_DAYS: i64 = 90;

/// Convert a vector of f32 to a little-endian byte BLOB suitable for
/// SQLite storage. The inverse of [`blob_to_vec`].
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Decode a little-endian f32 BLOB. Returns `AlzinaError::Memory` if the
/// BLOB length isn't a multiple of 4 — that should never happen for data
/// we wrote, but defends against schema-coercion surprises.
fn blob_to_vec(b: &[u8]) -> AlzinaResult<Vec<f32>> {
    if b.len() % 4 != 0 {
        return Err(AlzinaError::Memory(format!(
            "embedding_cache blob length {} is not a multiple of 4",
            b.len()
        )));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
        .collect())
}

/// SHA-256-keyed embedding cache. Construct with [`new`].
///
/// [`new`]: EmbeddingCache::new
pub struct EmbeddingCache {
    pool: SqlitePool,
}

impl EmbeddingCache {
    /// Wrap an existing pool. The pool must have been migrated with
    /// [`crate::schema::migrate`].
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// SHA-256 hex digest of `content`. Deterministic — identical input
    /// always yields identical output. Use as the primary key for
    /// `get_cached` / `put_cached`.
    pub fn hash_content(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Look up a cached embedding by content hash. Returns `None` on
    /// cache miss OR when the entry is past its 90-day TTL (in which
    /// case the row is deleted before returning).
    ///
    /// On hit, `accessed_at` is updated to now. This extends the LRU
    /// lifetime (used by potential future eviction) but does NOT extend
    /// the TTL — the TTL is anchored at `created_at`.
    pub async fn get_cached(&self, content_hash: &str) -> AlzinaResult<Option<Vec<f32>>> {
        let row: Option<(Vec<u8>, String)> =
            sqlx::query_as("SELECT vector, created_at FROM embedding_cache WHERE content_hash = ?")
                .bind(content_hash)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| AlzinaError::Memory(format!("embedding_cache select: {e}")))?;

        let Some((blob, created_at)) = row else {
            return Ok(None);
        };

        if is_expired(&created_at) {
            // Delete-on-access: TTL is measured from created_at.
            sqlx::query("DELETE FROM embedding_cache WHERE content_hash = ?")
                .bind(content_hash)
                .execute(&self.pool)
                .await
                .map_err(|e| AlzinaError::Memory(format!("embedding_cache delete-expired: {e}")))?;
            return Ok(None);
        }

        // Touch accessed_at so LRU bookkeeping stays current.
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE embedding_cache SET accessed_at = ? WHERE content_hash = ?")
            .bind(&now)
            .bind(content_hash)
            .execute(&self.pool)
            .await
            .map_err(|e| AlzinaError::Memory(format!("embedding_cache touch: {e}")))?;

        Ok(Some(blob_to_vec(&blob)?))
    }

    /// Insert or replace a cached embedding. `INSERT OR REPLACE` — so
    /// re-caching the same hash with a different model overwrites
    /// cleanly. `created_at` and `accessed_at` are both set to now.
    pub async fn put_cached(
        &self,
        content_hash: &str,
        model: &str,
        dimensions: usize,
        vector: &[f32],
    ) -> AlzinaResult<()> {
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(vector);
        sqlx::query(
            "INSERT OR REPLACE INTO embedding_cache \
             (content_hash, model, dimensions, vector, created_at, accessed_at) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(content_hash)
        .bind(model)
        .bind(dimensions as i64)
        .bind(&blob)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| AlzinaError::Memory(format!("embedding_cache insert: {e}")))?;
        Ok(())
    }

    /// Bulk lookup. Returns a vector parallel to `hashes`: index `i`
    /// holds `Some(vec)` if `hashes[i]` is a live cache hit, else
    /// `None`. Expired entries are deleted on access (same semantics as
    /// [`get_cached`]).
    pub async fn get_cached_batch(&self, hashes: &[String]) -> AlzinaResult<Vec<Option<Vec<f32>>>> {
        // Per-key calls keep the implementation simple and avoid having
        // to dynamically build an `IN (?, ?, ...)` clause. Backfill
        // typically does this in batches of ~100 — well within sqlite's
        // statement-cache budget.
        let mut out = Vec::with_capacity(hashes.len());
        for h in hashes {
            out.push(self.get_cached(h).await?);
        }
        Ok(out)
    }
}

/// Returns true if `created_at` (RFC3339) is older than `TTL_DAYS` from
/// now. Permissive on parse errors — an unparseable timestamp is treated
/// as expired so the row gets cleaned up.
fn is_expired(created_at: &str) -> bool {
    match chrono::DateTime::parse_from_rfc3339(created_at) {
        Ok(dt) => {
            let age = Utc::now().signed_duration_since(dt.with_timezone(&Utc));
            age.num_days() > TTL_DAYS
        }
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::in_memory_pool_with_search_schema;

    fn cache(pool: SqlitePool) -> EmbeddingCache {
        EmbeddingCache::new(pool)
    }

    #[tokio::test]
    async fn cache_round_trip() {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let c = cache(pool);
        let v = vec![0.1, 0.2, 0.3, 0.4];
        c.put_cached("h1", "jina-v3", 4, &v).await.unwrap();
        let got = c.get_cached("h1").await.unwrap().expect("hit");
        assert_eq!(got, v);
    }

    #[tokio::test]
    async fn cache_miss_returns_none() {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let c = cache(pool);
        let got = c.get_cached("never-inserted").await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn cache_replace_overwrites() {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let c = cache(pool);
        c.put_cached("h1", "jina-v3", 2, &[1.0, 2.0]).await.unwrap();
        c.put_cached("h1", "jina-v3", 2, &[9.0, 8.0]).await.unwrap();
        let got = c.get_cached("h1").await.unwrap().unwrap();
        assert_eq!(got, vec![9.0, 8.0]);
    }

    #[tokio::test]
    async fn cache_blob_endianness_roundtrip() {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let c = cache(pool);
        // Mix of positive, negative, zero, subnormal, and a value whose
        // bytes are not symmetric — defends against accidental BE/LE
        // swap regressions.
        let v: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE / 2.0, // subnormal
            std::f32::consts::PI,
            -1234.5678,
            f32::INFINITY,
        ];
        c.put_cached("endianness", "jina-v3", v.len(), &v)
            .await
            .unwrap();
        let got = c.get_cached("endianness").await.unwrap().unwrap();
        assert_eq!(got.len(), v.len());
        for (a, b) in got.iter().zip(v.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "byte-for-byte mismatch");
        }
    }

    #[tokio::test]
    async fn cache_get_batch_returns_parallel_options() {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let c = cache(pool);
        c.put_cached("h0", "jina-v3", 1, &[0.0]).await.unwrap();
        c.put_cached("h2", "jina-v3", 1, &[2.0]).await.unwrap();
        let hashes = vec![
            "h0".to_string(),
            "h1".to_string(),
            "h2".to_string(),
            "h3".to_string(),
        ];
        let got = c.get_cached_batch(&hashes).await.unwrap();
        assert_eq!(got.len(), 4);
        assert_eq!(got[0], Some(vec![0.0]));
        assert_eq!(got[1], None);
        assert_eq!(got[2], Some(vec![2.0]));
        assert_eq!(got[3], None);
    }

    #[tokio::test]
    async fn cache_expired_entry_is_deleted_on_access() {
        let pool = in_memory_pool_with_search_schema().await.unwrap();
        let c = cache(pool.clone());
        c.put_cached("stale", "jina-v3", 1, &[0.5]).await.unwrap();

        // Backdate created_at to 100 days ago (past 90-day TTL).
        let stale = (Utc::now() - chrono::Duration::days(100)).to_rfc3339();
        sqlx::query("UPDATE embedding_cache SET created_at = ? WHERE content_hash = ?")
            .bind(&stale)
            .bind("stale")
            .execute(&pool)
            .await
            .unwrap();

        let got = c.get_cached("stale").await.unwrap();
        assert!(got.is_none(), "expired entry must return None");

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM embedding_cache WHERE content_hash = ?")
                .bind("stale")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 0, "expired entry must be deleted on access");
    }

    #[tokio::test]
    async fn hash_content_deterministic() {
        assert_eq!(
            EmbeddingCache::hash_content("foo"),
            EmbeddingCache::hash_content("foo")
        );
        assert_ne!(
            EmbeddingCache::hash_content("foo"),
            EmbeddingCache::hash_content("bar")
        );
        // Sanity check: hex-encoded SHA-256 is 64 hex chars.
        assert_eq!(EmbeddingCache::hash_content("foo").len(), 64);
    }
}
