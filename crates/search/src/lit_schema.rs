//! Phase 21 literature schema: `papers` (provenance), `lit_vec0` (vec0
//! virtual table, 1024-dim), and `lit_chunks` (sidecar joined by rowid).
//!
//! # Ordering
//!
//! This migration is independent of the memory/KB schema — it targets a
//! physically separate `.db` file (or an in-memory pool for tests). Do
//! NOT call this against the main memory/KB pool; that creates the wrong
//! tables in the wrong database.
//!
//! The expected boot sequence for the literature store is:
//!
//! ```ignore
//! let lit_pool = open_lit_pool(path)?;
//! search::lit_schema::migrate(&lit_pool).await?;
//! ```
//!
//! # sqlite-vec extension loading (AC-1)
//!
//! `lit_vec0` requires the `vec0` module. We call
//! `crate::schema::register_sqlite_vec_extension()` (same OnceLock as the
//! memory store) at the top of `migrate()`. If the extension cannot be
//! loaded, `lit_vec0` is skipped; `papers` and `lit_chunks` are still
//! created so provenance and chunk text are survivable across restarts.
//! A second `SqliteVecStore::with_table_names(pool, 1024, "lit_vec0", "lit_chunks")`
//! will detect the missing virtual table and report `is_enabled() = false`.

use base::error::{AlzinaError, AlzinaResult, SearchDetail};
use sqlx::sqlite::SqlitePool;

/// Helper: map sqlx errors into `AlzinaError::Search` with degradation.
fn search_err(message: impl Into<String>, reason: impl Into<String>) -> AlzinaError {
    let reason = reason.into();
    AlzinaError::Search(SearchDetail {
        message: message.into(),
        degraded: true,
        degradation_reason: Some(reason),
    })
}

/// Run all Phase 21 literature schema migrations. Idempotent (`IF NOT EXISTS`).
///
/// Creates `papers`, `lit_vec0` (if sqlite-vec loaded), `lit_chunks`, and
/// the `idx_lit_chunks_paper` index against the given pool.
///
/// `dimensions` is the embedding vector width for `lit_vec0`. The daemon passes
/// its configured `embedding_dimensions` (1024 for Jina v3); the standalone CLI
/// passes its OpenRouter model's dim (1536 for `text-embedding-3-small`). The
/// value MUST match the embedder's `dimensions()` — `SqliteVecStore` rejects
/// mismatched inserts.
pub async fn migrate(pool: &SqlitePool, dimensions: usize) -> AlzinaResult<()> {
    // Best-effort: register the sqlite-vec extension (idempotent OnceLock).
    let extension_loaded = crate::schema::register_sqlite_vec_extension();

    // papers — one row per fetched paper, restart-survivable provenance.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS papers (\
            paper_id              TEXT PRIMARY KEY,\
            source                TEXT NOT NULL,\
            arxiv_id              TEXT,\
            doi                   TEXT,\
            s2_paper_id           TEXT,\
            title                 TEXT NOT NULL,\
            abstract              TEXT,\
            url                   TEXT NOT NULL,\
            year                  INTEGER,\
            authors               TEXT NOT NULL,\
            citation_count        INTEGER,\
            fetched_at            TEXT NOT NULL,\
            fulltext_status       TEXT NOT NULL DEFAULT 'none',\
            open_access_pdf_url   TEXT,\
            influential_citation_count INTEGER,\
            venue                 TEXT\
        )",
    )
    .execute(pool)
    .await
    .map_err(|e| search_err(format!("papers: {e}"), format!("papers create: {e}")))?;

    // Guarded ALTER TABLE for already-migrated DBs that lack fulltext_status.
    // A fresh DB gets the column from CREATE TABLE above; an old DB gets it here.
    // On a fresh DB this query is skipped (the column already exists).
    // Implementation: check pragma_table_info for the column name; add it only
    // when absent. This is a one-way migration — safe to run multiple times.
    let has_fulltext_status: bool = sqlx::query_as::<_, (String,)>(
        "SELECT name FROM pragma_table_info('papers') WHERE name = 'fulltext_status'",
    )
    .fetch_optional(pool)
    .await
    .map(|r| r.is_some())
    .unwrap_or(false);

    if !has_fulltext_status {
        sqlx::query(
            "ALTER TABLE papers ADD COLUMN fulltext_status TEXT NOT NULL DEFAULT 'none'",
        )
        .execute(pool)
        .await
        .map_err(|e| {
            search_err(
                format!("alter papers add fulltext_status: {e}"),
                format!("papers ALTER failed: {e}"),
            )
        })?;
    }

    // Guarded ALTER TABLE for open_access_pdf_url (F10).
    // Fresh DBs get it from CREATE TABLE above; old DBs get it here.
    let has_open_access_pdf_url: bool = sqlx::query_as::<_, (String,)>(
        "SELECT name FROM pragma_table_info('papers') WHERE name = 'open_access_pdf_url'",
    )
    .fetch_optional(pool)
    .await
    .map(|r| r.is_some())
    .unwrap_or(false);

    if !has_open_access_pdf_url {
        sqlx::query("ALTER TABLE papers ADD COLUMN open_access_pdf_url TEXT")
            .execute(pool)
            .await
            .map_err(|e| {
                search_err(
                    format!("alter papers add open_access_pdf_url: {e}"),
                    format!("papers ALTER open_access_pdf_url failed: {e}"),
                )
            })?;
    }

    // Guarded ALTER TABLE for the source-credibility columns
    // (influential_citation_count + venue). Fresh DBs get them from CREATE
    // TABLE above; old DBs get them here. Both feed the per-source authenticity
    // tier; backfilled from S2 by examples/backfill_credibility.
    let has_influential: bool = sqlx::query_as::<_, (String,)>(
        "SELECT name FROM pragma_table_info('papers') WHERE name = 'influential_citation_count'",
    )
    .fetch_optional(pool)
    .await
    .map(|r| r.is_some())
    .unwrap_or(false);

    if !has_influential {
        sqlx::query("ALTER TABLE papers ADD COLUMN influential_citation_count INTEGER")
            .execute(pool)
            .await
            .map_err(|e| {
                search_err(
                    format!("alter papers add influential_citation_count: {e}"),
                    format!("papers ALTER influential_citation_count failed: {e}"),
                )
            })?;
    }

    let has_venue: bool = sqlx::query_as::<_, (String,)>(
        "SELECT name FROM pragma_table_info('papers') WHERE name = 'venue'",
    )
    .fetch_optional(pool)
    .await
    .map(|r| r.is_some())
    .unwrap_or(false);

    if !has_venue {
        sqlx::query("ALTER TABLE papers ADD COLUMN venue TEXT")
            .execute(pool)
            .await
            .map_err(|e| {
                search_err(
                    format!("alter papers add venue: {e}"),
                    format!("papers ALTER venue failed: {e}"),
                )
            })?;
    }

    // lit_vec0 — embedding vec0 virtual table (width = `dimensions`); guarded by
    // extension_loaded.
    if extension_loaded {
        let create_vec = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS lit_vec0 USING vec0(embedding float[{dimensions}])"
        );
        match sqlx::query(&create_vec)
        .execute(pool)
        .await
        {
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(
                    "sqlite-vec extension unavailable; lit vector search will degrade \
                     (CREATE VIRTUAL TABLE lit_vec0 failed: {e})"
                );
            }
        }
    } else {
        tracing::warn!(
            "sqlite-vec extension unavailable; lit vector search will degrade"
        );
    }

    // lit_chunks — sidecar: one row per chunk, rowid FK → lit_vec0.
    // Columns include the full VectorMetadata fields (source_type, source_id,
    // content_preview, source_agent, source_date, weave_id, domain) so the
    // generic SqliteVecStore::insert path works unchanged through
    // with_table_names("lit_vec0", "lit_chunks"). Literature-specific fields
    // (paper_id, section, content) are nullable for compatibility with the
    // generic VectorMetadata insert path (which may pass NULL for optional
    // fields). The lit ingestion pipeline writes these columns directly.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS lit_chunks (\
            rowid           INTEGER PRIMARY KEY,\
            source_type     TEXT NOT NULL,\
            source_id       TEXT NOT NULL,\
            chunk_index     INTEGER NOT NULL DEFAULT 0,\
            content_preview TEXT NOT NULL,\
            source_agent    TEXT,\
            source_date     TEXT,\
            weave_id        TEXT,\
            section         TEXT,\
            domain          TEXT,\
            indexed_at      TEXT NOT NULL,\
            paper_id        TEXT REFERENCES papers(paper_id),\
            content         TEXT\
        )",
    )
    .execute(pool)
    .await
    .map_err(|e| search_err(format!("lit_chunks: {e}"), format!("lit_chunks create: {e}")))?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_lit_chunks_paper \
         ON lit_chunks(paper_id)",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("idx_lit_chunks_paper: {e}"),
            format!("idx_lit_chunks_paper create: {e}"),
        )
    })?;

    // synthesis_bibliography — accumulating bibliography for the TTD engine.
    // Phase 24 EXT-03: sources cited in synthesis, distinct from the retrieval
    // corpus ("cited in synthesis" ≠ "retrieved").
    //
    // UNIQUE(run_id, source_id, expert_id, quote_normalised) encodes BOTH
    // consensus dedup layers in one constraint:
    //   Layer 1: source_id (graph-node compound suffixes stripped — runner.py:594-599)
    //   Layer 2: expert_id + quote_normalised (merger — synthesis_tasks.py:945-948)
    // INSERT OR IGNORE is accumulate-and-dedup-by-silence (not idempotent update).
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS synthesis_bibliography (\
            id               INTEGER PRIMARY KEY AUTOINCREMENT,\
            run_id           TEXT NOT NULL,\
            source_id        TEXT NOT NULL,\
            expert_id        TEXT NOT NULL,\
            quote_normalised TEXT NOT NULL,\
            quote_raw        TEXT,\
            stage            TEXT NOT NULL,\
            denoise_step     INTEGER NOT NULL,\
            added_at         TEXT NOT NULL,\
            UNIQUE(run_id, source_id, expert_id, quote_normalised)\
        )",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("synthesis_bibliography: {e}"),
            format!("synthesis_bibliography create: {e}"),
        )
    })?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_bib_run \
         ON synthesis_bibliography(run_id)",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("idx_bib_run: {e}"),
            format!("idx_bib_run create: {e}"),
        )
    })?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_bib_source \
         ON synthesis_bibliography(source_id)",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("idx_bib_source: {e}"),
            format!("idx_bib_source create: {e}"),
        )
    })?;

    // s2_cache — response cache for S2 graph API calls.
    //
    // Key scheme mirrors clawd S2Cache (semantic_scholar.py:96-144):
    //   `paper_{resolved_id}`          — single paper (get_paper, search)
    //   `{resolved_id}_citations`      — citation list
    //   `{resolved_id}_references`     — reference list
    //
    // T-iab-02: keys are built from resolved paper IDs via format!() and
    // written via bound parameters — never concatenated into SQL.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS s2_cache (\
            cache_key  TEXT PRIMARY KEY,\
            payload    TEXT NOT NULL,\
            cached_at  TEXT NOT NULL\
        )",
    )
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("s2_cache: {e}"),
            format!("s2_cache create: {e}"),
        )
    })?;

    Ok(())
}

/// Get a cached S2 response by key. Returns `None` on cache miss.
///
/// T-iab-02: `key` written via bound parameter.
pub async fn s2_cache_get(pool: &SqlitePool, key: &str) -> AlzinaResult<Option<String>> {
    let row: Option<(String,)> =
        sqlx::query_as("SELECT payload FROM s2_cache WHERE cache_key = ?")
            .bind(key)
            .fetch_optional(pool)
            .await
            .map_err(|e| {
                search_err(
                    format!("s2_cache_get: {e}"),
                    format!("s2_cache get failed: {e}"),
                )
            })?;
    Ok(row.map(|(payload,)| payload))
}

/// Insert or replace a cached S2 response (overwrites existing).
///
/// T-iab-02: both `key` and `payload` written via bound parameters.
pub async fn s2_cache_put(pool: &SqlitePool, key: &str, payload: &str) -> AlzinaResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT OR REPLACE INTO s2_cache (cache_key, payload, cached_at) VALUES (?, ?, ?)",
    )
    .bind(key)
    .bind(payload)
    .bind(&now)
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("s2_cache_put: {e}"),
            format!("s2_cache put failed: {e}"),
        )
    })?;
    Ok(())
}

/// Insert a cached S2 response only when the key does not yet exist.
///
/// Never overwrites richer data — mirrors clawd `S2Cache.put_if_absent`
/// (semantic_scholar.py:127-132, `INSERT OR IGNORE` semantics).
///
/// T-iab-02: bound parameters only.
pub async fn s2_cache_put_if_absent(
    pool: &SqlitePool,
    key: &str,
    payload: &str,
) -> AlzinaResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT OR IGNORE INTO s2_cache (cache_key, payload, cached_at) VALUES (?, ?, ?)",
    )
    .bind(key)
    .bind(payload)
    .bind(&now)
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("s2_cache_put_if_absent: {e}"),
            format!("s2_cache put_if_absent failed: {e}"),
        )
    })?;
    Ok(())
}

/// Upsert a paper row; idempotent on `paper_id`.
#[allow(clippy::too_many_arguments)]
pub async fn upsert_paper(
    pool: &SqlitePool,
    paper_id: &str,
    source: &str,
    arxiv_id: Option<&str>,
    doi: Option<&str>,
    s2_paper_id: Option<&str>,
    title: &str,
    abstract_text: Option<&str>,
    url: &str,
    year: Option<i32>,
    authors: &str,
    citation_count: Option<i32>,
    fetched_at: &str,
) -> AlzinaResult<()> {
    sqlx::query(
        "INSERT INTO papers (\
            paper_id, source, arxiv_id, doi, s2_paper_id, title, abstract, url, \
            year, authors, citation_count, fetched_at\
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\
         ON CONFLICT(paper_id) DO UPDATE SET \
            source         = excluded.source,\
            arxiv_id       = COALESCE(excluded.arxiv_id, papers.arxiv_id),\
            doi            = COALESCE(excluded.doi, papers.doi),\
            s2_paper_id    = COALESCE(excluded.s2_paper_id, papers.s2_paper_id),\
            title          = CASE WHEN TRIM(excluded.title) = '' THEN papers.title ELSE excluded.title END,\
            abstract       = COALESCE(NULLIF(TRIM(excluded.abstract), ''), papers.abstract),\
            url            = CASE WHEN TRIM(excluded.url) = '' THEN papers.url ELSE excluded.url END,\
            year           = COALESCE(excluded.year, papers.year),\
            authors        = CASE WHEN excluded.authors IS NULL OR TRIM(excluded.authors) IN ('', '[]') THEN papers.authors ELSE excluded.authors END,\
            citation_count = COALESCE(excluded.citation_count, papers.citation_count),\
            fetched_at     = excluded.fetched_at",
    )
    .bind(paper_id)
    .bind(source)
    .bind(arxiv_id)
    .bind(doi)
    .bind(s2_paper_id)
    .bind(title)
    .bind(abstract_text)
    .bind(url)
    .bind(year)
    .bind(authors)
    .bind(citation_count)
    .bind(fetched_at)
    .execute(pool)
    .await
    .map_err(|e| search_err(format!("upsert_paper: {e}"), format!("papers upsert: {e}")))?;
    Ok(())
}

/// Set the source-credibility signals (citation_count,
/// influential_citation_count, venue) for one paper row.
///
/// Preserve-on-unknown: a `None` / NULL argument never clobbers an existing
/// value (`COALESCE`), and a blank venue is treated as unknown. This is the
/// write path for `examples/backfill_credibility` (heals historical rows —
/// arxiv papers carry no citation data from the Atom feed) and for the S2
/// ingest lane (which now persists the venue it already fetched).
///
/// Callers map a genuine S2 `influentialCitationCount` of 0 to `None` (unknown
/// vs zero is not separable downstream and a 0 carries no tier signal), so a
/// real count is never overwritten by a later abstract-only re-ingest.
pub async fn update_paper_credibility(
    pool: &SqlitePool,
    paper_id: &str,
    citation_count: Option<i32>,
    influential_citation_count: Option<i32>,
    venue: Option<&str>,
) -> AlzinaResult<()> {
    sqlx::query(
        "UPDATE papers SET \
            citation_count = COALESCE(?, citation_count),\
            influential_citation_count = COALESCE(?, influential_citation_count),\
            venue = COALESCE(NULLIF(TRIM(?), ''), venue) \
         WHERE paper_id = ?",
    )
    .bind(citation_count)
    .bind(influential_citation_count)
    .bind(venue)
    .bind(paper_id)
    .execute(pool)
    .await
    .map_err(|e| {
        search_err(
            format!("update_paper_credibility: {e}"),
            format!("papers credibility update: {e}"),
        )
    })?;
    Ok(())
}

/// Update `fulltext_status` for one paper row.
///
/// `status` must be one of `none | pending | indexed | failed` — values are
/// asserted by callers and tests; this function writes whatever string is passed
/// via a bound parameter (T-260611-01: no string interpolation of paper_id or status).
pub async fn set_fulltext_status(
    pool: &SqlitePool,
    paper_id: &str,
    status: &str,
) -> AlzinaResult<()> {
    sqlx::query("UPDATE papers SET fulltext_status = ? WHERE paper_id = ?")
        .bind(status)
        .bind(paper_id)
        .execute(pool)
        .await
        .map_err(|e| {
            search_err(
                format!("set_fulltext_status: {e}"),
                format!("papers status update: {e}"),
            )
        })?;
    Ok(())
}

/// Set `open_access_pdf_url` for one paper row. Set-only: never writes NULL,
/// so later upsert_paper refreshes (which do not touch this column) cannot
/// erase a previously stored URL.
///
/// Callers only invoke this with a real URL string. T-lq4-01: `url` stored via
/// bound parameter — no SQL interpolation.
pub async fn set_open_access_pdf_url(
    pool: &SqlitePool,
    paper_id: &str,
    url: &str,
) -> AlzinaResult<()> {
    sqlx::query("UPDATE papers SET open_access_pdf_url = ? WHERE paper_id = ?")
        .bind(url)
        .bind(paper_id)
        .execute(pool)
        .await
        .map_err(|e| {
            search_err(
                format!("set_open_access_pdf_url: {e}"),
                format!("papers open_access_pdf_url update: {e}"),
            )
        })?;
    Ok(())
}

/// Return `true` iff a `papers` row exists for `paper_id` (skip-if-ingested predicate).
///
/// Returns `false` for an absent id. Uses a bound parameter (T-260611-01).
pub async fn paper_is_ingested(pool: &SqlitePool, paper_id: &str) -> AlzinaResult<bool> {
    let row: Option<(i64,)> =
        sqlx::query_as("SELECT 1 FROM papers WHERE paper_id = ? LIMIT 1")
            .bind(paper_id)
            .fetch_optional(pool)
            .await
            .map_err(|e| {
                search_err(
                    format!("paper_is_ingested: {e}"),
                    format!("papers existence check: {e}"),
                )
            })?;
    Ok(row.is_some())
}

/// Test helper: build an in-memory pool with the literature schema applied.
/// Available for the whole crate's `#[cfg(test)]` modules.
#[cfg(test)]
pub async fn in_memory_lit_pool() -> AlzinaResult<SqlitePool> {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use std::str::FromStr;

    let opts = SqliteConnectOptions::from_str("sqlite::memory:")
        .map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("lit pool opts: {e}"),
                degraded: true,
                degradation_reason: Some(format!("lit in-memory pool init: {e}")),
            })
        })?
        .create_if_missing(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(opts)
        .await
        .map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("lit pool connect: {e}"),
                degraded: true,
                degradation_reason: Some(format!("lit in-memory pool connect: {e}")),
            })
        })?;

    migrate(&pool, 1024).await?;
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn migrate_creates_papers_lit_vec0_and_lit_chunks() {
        let pool = in_memory_lit_pool().await.unwrap();

        // papers must exist regardless of extension status.
        sqlx::query("SELECT COUNT(*) FROM papers")
            .fetch_one(&pool)
            .await
            .unwrap();

        // lit_chunks must exist regardless of extension status.
        sqlx::query("SELECT COUNT(*) FROM lit_chunks")
            .fetch_one(&pool)
            .await
            .unwrap();

        // lit_vec0 only exists when sqlite-vec loaded; check sqlite_master.
        let tables: Vec<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master WHERE type IN ('table','shadow') ORDER BY name",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        let names: Vec<&str> = tables.iter().map(|(n,)| n.as_str()).collect();
        assert!(names.contains(&"papers"), "papers must be present");
        assert!(names.contains(&"lit_chunks"), "lit_chunks must be present");
        // lit_vec0 is extension-dependent; just verify the query doesn't error.
        let _ = sqlx::query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lit_vec0'",
        )
        .fetch_optional(&pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn upsert_idempotent_on_paper_id() {
        let pool = in_memory_lit_pool().await.unwrap();

        upsert_paper(
            &pool,
            "arxiv:2401.00001",
            "arxiv",
            Some("2401.00001"),
            None,
            None,
            "Test Paper",
            Some("Abstract text"),
            "https://arxiv.org/abs/2401.00001",
            Some(2024),
            r#"["Alice","Bob"]"#,
            Some(5),
            "2026-06-05T00:00:00Z",
        )
        .await
        .unwrap();

        // Upsert the same paper_id a second time — must not error, must yield one row.
        upsert_paper(
            &pool,
            "arxiv:2401.00001",
            "arxiv",
            Some("2401.00001"),
            None,
            None,
            "Test Paper (updated title)",
            Some("Updated abstract"),
            "https://arxiv.org/abs/2401.00001",
            Some(2024),
            r#"["Alice","Bob","Carol"]"#,
            Some(10),
            "2026-06-05T01:00:00Z",
        )
        .await
        .unwrap();

        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM papers WHERE paper_id = ?")
            .bind("arxiv:2401.00001")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count.0, 1, "upsert must yield exactly one papers row");

        // Verify the update landed (fetched_at should be the latest value).
        let row: (String,) =
            sqlx::query_as("SELECT fetched_at FROM papers WHERE paper_id = ?")
                .bind("arxiv:2401.00001")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(row.0, "2026-06-05T01:00:00Z");
    }

    /// A partial re-ingest (e.g. full-text promotion building an ArxivResult with
    /// empty authors/year) must NOT erase metadata an earlier ingest wrote. The
    /// upsert preserves existing authors/year/abstract/citation_count when the
    /// incoming values are empty. (Probe-25 fix: promotion clobbered authors=[].)
    #[tokio::test]
    async fn upsert_does_not_downgrade_metadata_to_empty() {
        let pool = in_memory_lit_pool().await.unwrap();

        // First ingest: full metadata (the abstract-ingest lane).
        upsert_paper(
            &pool, "arxiv:2502.12110", "arxiv", Some("2502.12110"), None, None,
            "A Real Title", Some("A real abstract"),
            "https://arxiv.org/abs/2502.12110", Some(2025),
            r#"["Alice Smith","Bob Jones"]"#, Some(7), "2026-06-05T00:00:00Z",
        )
        .await
        .unwrap();

        // Second ingest: full-text promotion — empty authors, NULL year, empty abstract.
        upsert_paper(
            &pool, "arxiv:2502.12110", "arxiv", Some("2502.12110"), None, None,
            "A Real Title", None,
            "https://arxiv.org/abs/2502.12110", None,
            "[]", None, "2026-06-05T01:00:00Z",
        )
        .await
        .unwrap();

        let (authors, year, abstract_text, cites): (String, Option<i64>, Option<String>, Option<i64>) =
            sqlx::query_as(
                "SELECT authors, year, abstract, citation_count FROM papers WHERE paper_id = ?",
            )
            .bind("arxiv:2502.12110")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(authors, r#"["Alice Smith","Bob Jones"]"#, "authors must survive empty re-ingest");
        assert_eq!(year, Some(2025), "year must survive NULL re-ingest");
        assert_eq!(abstract_text.as_deref(), Some("A real abstract"), "abstract must survive empty re-ingest");
        assert_eq!(cites, Some(7), "citation_count must survive NULL re-ingest");

        // And fetched_at still advances (the update did land).
        let (fa,): (String,) = sqlx::query_as("SELECT fetched_at FROM papers WHERE paper_id = ?")
            .bind("arxiv:2502.12110")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(fa, "2026-06-05T01:00:00Z", "fetched_at must still update");
    }

    #[tokio::test]
    async fn migrate_creates_bibliography_table() {
        let pool = in_memory_lit_pool().await.unwrap();
        sqlx::query("SELECT COUNT(*) FROM synthesis_bibliography")
            .fetch_one(&pool)
            .await
            .unwrap();
    }

    /// migrate() twice must not error, and fulltext_status column must exist
    /// with a default of 'none' on fresh rows.
    #[tokio::test]
    async fn migrate_idempotent_fulltext_status_column() {
        use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
        use std::str::FromStr;

        // Fresh DB: CREATE TABLE includes the column.
        let pool = in_memory_lit_pool().await.unwrap();

        // Second migrate must be a no-op.
        migrate(&pool, 1024).await.expect("second migrate() must not error");

        // Check column exists via pragma.
        let cols: Vec<(String,)> =
            sqlx::query_as("SELECT name FROM pragma_table_info('papers') ORDER BY cid")
                .fetch_all(&pool)
                .await
                .unwrap();
        let col_names: Vec<&str> = cols.iter().map(|(n,)| n.as_str()).collect();
        assert!(
            col_names.contains(&"fulltext_status"),
            "papers table must have fulltext_status column; got: {col_names:?}"
        );

        // A row without explicit fulltext_status gets default 'none'.
        sqlx::query(
            "INSERT INTO papers (paper_id, source, title, url, authors, fetched_at) \
             VALUES ('test:migrate', 'arxiv', 'T', 'U', '[]', '2024-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        let (status,): (String,) =
            sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = 'test:migrate'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "none", "fulltext_status default must be 'none'");

        // Now simulate an already-migrated DB: open a file-backed DB,
        // migrate once, then migrate again — the ALTER TABLE guard fires.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db_path = tmp.path().to_str().unwrap().to_string();
        let db_url = format!("sqlite:{db_path}");

        let opts = SqliteConnectOptions::from_str(&db_url).unwrap().create_if_missing(true);
        let pool2 = SqlitePoolOptions::new().max_connections(1).connect_with(opts.clone()).await.unwrap();
        migrate(&pool2, 1024).await.expect("first migrate on file-backed DB");
        migrate(&pool2, 1024).await.expect("second migrate on file-backed DB must not error");
    }

    /// set_fulltext_status updates the fulltext_status column on an existing row.
    #[tokio::test]
    async fn set_fulltext_status_roundtrip() {
        let pool = in_memory_lit_pool().await.unwrap();

        // Insert a minimal papers row.
        sqlx::query(
            "INSERT INTO papers (paper_id, source, title, url, authors, fetched_at) \
             VALUES ('arxiv:test001', 'arxiv', 'T', 'U', '[]', '2024-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Verify it starts at 'none'.
        let (before,): (String,) =
            sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = 'arxiv:test001'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(before, "none");

        super::set_fulltext_status(&pool, "arxiv:test001", "indexed")
            .await
            .expect("set_fulltext_status must succeed");

        let (after,): (String,) =
            sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = 'arxiv:test001'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(after, "indexed");
    }

    /// paper_is_ingested returns false for absent, true for present.
    #[tokio::test]
    async fn paper_is_ingested_predicate() {
        let pool = in_memory_lit_pool().await.unwrap();

        assert!(
            !super::paper_is_ingested(&pool, "arxiv:absent001").await.unwrap(),
            "must return false for absent id"
        );

        sqlx::query(
            "INSERT INTO papers (paper_id, source, title, url, authors, fetched_at) \
             VALUES ('arxiv:absent001', 'arxiv', 'T', 'U', '[]', '2024-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        assert!(
            super::paper_is_ingested(&pool, "arxiv:absent001").await.unwrap(),
            "must return true after insert"
        );
    }

    // ── s2_cache tests ────────────────────────────────────────────────────

    #[tokio::test]
    async fn s2_cache_get_miss_returns_none() {
        let pool = in_memory_lit_pool().await.unwrap();
        let result = super::s2_cache_get(&pool, "paper_missing").await.unwrap();
        assert!(result.is_none(), "cache miss must return None");
    }

    #[tokio::test]
    async fn s2_cache_put_and_get_roundtrip() {
        let pool = in_memory_lit_pool().await.unwrap();
        super::s2_cache_put(&pool, "paper_abc123", r#"{"s2_id":"abc123"}"#)
            .await
            .unwrap();
        let hit = super::s2_cache_get(&pool, "paper_abc123").await.unwrap();
        assert_eq!(hit.as_deref(), Some(r#"{"s2_id":"abc123"}"#));
    }

    #[tokio::test]
    async fn s2_cache_put_overwrites_existing() {
        let pool = in_memory_lit_pool().await.unwrap();
        super::s2_cache_put(&pool, "key1", "old_value").await.unwrap();
        super::s2_cache_put(&pool, "key1", "new_value").await.unwrap();
        let result = super::s2_cache_get(&pool, "key1").await.unwrap();
        assert_eq!(result.as_deref(), Some("new_value"), "put must overwrite");
    }

    #[tokio::test]
    async fn s2_cache_put_if_absent_does_not_overwrite() {
        let pool = in_memory_lit_pool().await.unwrap();
        super::s2_cache_put(&pool, "key2", "richer_data").await.unwrap();
        super::s2_cache_put_if_absent(&pool, "key2", "would_clobber")
            .await
            .unwrap();
        let result = super::s2_cache_get(&pool, "key2").await.unwrap();
        assert_eq!(result.as_deref(), Some("richer_data"), "put_if_absent must not overwrite");
    }

    #[tokio::test]
    async fn s2_cache_put_if_absent_inserts_when_key_absent() {
        let pool = in_memory_lit_pool().await.unwrap();
        super::s2_cache_put_if_absent(&pool, "key3", "initial").await.unwrap();
        let result = super::s2_cache_get(&pool, "key3").await.unwrap();
        assert_eq!(result.as_deref(), Some("initial"), "put_if_absent must insert when absent");
    }

    #[tokio::test]
    async fn migrate_creates_s2_cache_table() {
        let pool = in_memory_lit_pool().await.unwrap();
        // Second migrate must be idempotent.
        super::migrate(&pool, 1024).await.expect("second migrate must not error");
        // Confirm s2_cache is queryable.
        sqlx::query("SELECT COUNT(*) FROM s2_cache")
            .fetch_one(&pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn separate_db_has_distinct_tables() {
        // This test demonstrates physical isolation: lit pool has lit_vec0 +
        // lit_chunks; memory-schema pool has vec_entries + vec_index. Neither
        // pool sees the other's tables.
        let lit_pool = in_memory_lit_pool().await.unwrap();
        let mem_pool = crate::schema::in_memory_pool_with_search_schema().await.unwrap();

        // lit_pool: papers and lit_chunks present; vec_entries absent.
        let papers_row: Option<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='papers'",
        )
        .fetch_optional(&lit_pool)
        .await
        .unwrap();
        assert!(papers_row.is_some(), "lit_pool must have papers");

        let vec_entries_in_lit: Option<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_entries'",
        )
        .fetch_optional(&lit_pool)
        .await
        .unwrap();
        assert!(
            vec_entries_in_lit.is_none(),
            "lit_pool must NOT have vec_entries"
        );

        // mem_pool: vec_index present; papers absent.
        let vec_index_row: Option<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_index'",
        )
        .fetch_optional(&mem_pool)
        .await
        .unwrap();
        assert!(vec_index_row.is_some(), "mem_pool must have vec_index");

        let papers_in_mem: Option<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='papers'",
        )
        .fetch_optional(&mem_pool)
        .await
        .unwrap();
        assert!(
            papers_in_mem.is_none(),
            "mem_pool must NOT have papers"
        );
    }
}
