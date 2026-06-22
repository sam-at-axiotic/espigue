//! Literature intake: S2Client default-on constructor + S2Result → papers
//! persistence + arxiv → papers + lit_chunks persistence with the locked
//! embedding prefix.
//!
//! Phase 21 Plan 03.
//!
//! ## S2 default-on (SC-1)
//!
//! `s2_client_for_lit()` calls `S2Client::with_config(config, true)` with
//! `enabled=true` unconditionally, overriding the `S2_LIVE_ENABLED` env gate
//! that `from_env()` reads. The lit path always fires S2; it is not opt-in.
//!
//! ## Embedding prefix (locked decision)
//!
//! Each chunk's embedding input is `"{title}\n{section}\n{content}"`.
//! This is load-bearing for chunk-vector coherence (chunks from the same paper
//! cluster near each other in vector space). Do NOT use title-only or
//! title+full-abstract — see Anti-Pattern in 21-RESEARCH.md.
//!
//! The raw `chunk.content` is stored in `lit_chunks` unchanged (display-clean).
//!
//! ## Threat notes
//!
//! - T-21-07: `S2_API_KEY` is read from env at construction time; never logged,
//!   never written to papers.
//! - T-21-08: all title/abstract text is stored via bound parameters in
//!   `upsert_paper` — no SQL interpolation of content.
//! - T-21-09: the `EmbeddingService` implementation behind the trait object
//!   already uses `embed_cache` (SHA-256, 90-day TTL) so re-fetch of the same
//!   chunk does not re-spend Jina API credits.

use base::error::{AlzinaError, AlzinaResult, SearchDetail};
use sqlx::sqlite::SqlitePool;

use crate::arxiv::{ArxivClient, ArxivResult};
use crate::lit_chunking::{LitChunk, LitChunkConfig, chunk_ar5iv_html, chunk_plain_text};
use crate::lit_schema::{
    set_fulltext_status, set_open_access_pdf_url, update_paper_credibility, upsert_paper,
};
use crate::pdf_fetch::{PdfFetchConfig, fetch_pdf_bytes, pdftotext_extract};
use crate::s2_enrichment::{S2Client, S2Config, S2PaperFull, S2Result};
use crate::sqlite_vec::SqliteVecStore;


// ── Public API ────────────────────────────────────────────────────────────

/// Build the S2Client for the lit path. `S2_API_KEY` gates whether it fires.
///
/// When the key is PRESENT the client is enabled (overriding the
/// `S2_LIVE_ENABLED` env gate). When the key is ABSENT the client is built
/// DISABLED — every S2 method returns empty without a network call — so the
/// daemon never makes ANONYMOUS S2 requests. Anonymous calls hit S2's shared
/// unauthenticated pool, get 429'd almost immediately, and read as spamming the
/// endpoint. Unkeyed S2 therefore degrades to local-only, loudly.
///
/// `limit` is set to 10 (locked decision, Pitfall 4 in 21-RESEARCH.md).
pub fn s2_client_for_lit() -> AlzinaResult<S2Client> {
    let api_key = std::env::var("S2_API_KEY").ok().filter(|s| !s.is_empty());
    let keyed = api_key.is_some();
    if !keyed {
        tracing::warn!(
            "S2_API_KEY absent — S2 lit lane DISABLED (local-only). No live \
             Semantic Scholar calls will be made; set S2_API_KEY to enable."
        );
    }
    let config = S2Config {
        api_key,
        limit: 10,
        ..S2Config::default()
    };
    // enabled = keyed: no key → zero live S2 traffic (never anonymous).
    S2Client::with_config(config, keyed)
}

/// Persist one S2Result into the `papers` table.
///
/// Maps the 7 S2Result fields (RETR-02) onto `upsert_paper`:
/// - `paper_id` = `"s2:{s2_paper_id}"`
/// - `source`   = `"s2"`
/// - `s2_paper_id` = `r.paper_id`
/// - All other fields mapped directly; `authors` JSON-encoded.
///
/// Idempotent: re-persisting the same paper_id updates metadata via
/// `ON CONFLICT DO UPDATE`.
pub async fn persist_s2_result(pool: &SqlitePool, r: &S2Result) -> AlzinaResult<()> {
    let paper_id = format!("s2:{}", r.paper_id);
    let authors =
        serde_json::to_string(&r.authors).unwrap_or_else(|_| "[]".to_string());
    let fetched_at = chrono::Utc::now().to_rfc3339();
    // citation_count: S2Result has Option<i64>, upsert_paper takes Option<i32>.
    // Clamp via try_into; values > i32::MAX are pathological — default to None.
    let citation_count = r.citation_count.and_then(|c| i32::try_from(c).ok());

    upsert_paper(
        pool,
        &paper_id,
        "s2",
        None,               // arxiv_id
        None,               // doi
        Some(&r.paper_id),  // s2_paper_id
        &r.title,
        r.abstract_text.as_deref(),
        &r.url,
        r.year,
        &authors,
        citation_count,
        &fetched_at,
    )
    .await
}

/// Produce the embedding input string for one chunk.
///
/// Format: `"{title}\n{section}\n{content}"` — the locked embedding prefix.
/// Store the raw `content` separately in `lit_chunks` for display-clean retrieval.
///
/// This function is public so Plan 21-04 can unit-test the prefix independently
/// of the full persist pipeline.
pub fn chunk_embedding_input(title: &str, section: &str, content: &str) -> String {
    format!("{title}\n{section}\n{content}")
}

/// Embed and persist a set of chunks for a paper.
///
/// This is the inner loop extracted from `persist_arxiv` so that
/// `promote_pdf_fulltext` can reuse the same embed+insert path without
/// duplicating code. `source_type` is `"arxiv"` for arxiv papers and
/// `"s2"` for PDF-promoted papers (existing kNN filter spans all source_types
/// since A4 — no kNN change needed).
///
/// `source_date` is an optional ISO 8601 string for `lit_chunks.source_date`.
///
/// Public so the standalone bring-your-own-corpora ingest path (Phase 3) can
/// reuse the same embed+insert primitive for local docs (`source_type="local"`).
pub async fn persist_chunks_for_paper(
    lit_pool: &SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn base::search::EmbeddingService,
    paper_id: &str,
    source_type: &str,
    title: &str,
    source_date: Option<&str>,
    chunks: &[LitChunk],
) -> AlzinaResult<()> {
    use base::search::{EmbeddingTask, VectorMetadata, VectorStore};

    let fetched_at = chrono::Utc::now().to_rfc3339();

    for chunk in chunks {
        let embed_input = chunk_embedding_input(title, &chunk.section, &chunk.content);

        let vector = embedder
            .embed(&embed_input, EmbeddingTask::Passage)
            .await
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("persist_chunks_for_paper embed: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!(
                        "embedding failed for {paper_id} chunk {}: {e}",
                        chunk.chunk_index
                    )),
                })
            })?;

        let metadata = VectorMetadata {
            source_type: source_type.to_string(),
            source_id: paper_id.to_string(),
            chunk_index: chunk.chunk_index as i64,
            content_preview: chunk.content.chars().take(200).collect(),
            source_agent: None,
            source_date: source_date.map(str::to_string),
            weave_id: None,
            section: Some(chunk.section.clone()),
            domain: None,
            indexed_at: fetched_at.clone(),
        };
        let rowid = lit_store.insert(&vector, metadata).await?;

        sqlx::query(
            "INSERT INTO lit_chunks (\
                rowid, source_type, source_id, chunk_index, content_preview,\
                source_date, indexed_at, paper_id, section, content\
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\
             ON CONFLICT(rowid) DO UPDATE SET \
                content    = excluded.content,\
                section    = excluded.section,\
                paper_id   = excluded.paper_id,\
                indexed_at = excluded.indexed_at",
        )
        .bind(rowid)
        .bind(source_type)
        .bind(paper_id)
        .bind(chunk.chunk_index as i64)
        .bind(chunk.content.chars().take(200).collect::<String>())
        .bind(source_date)
        .bind(&fetched_at)
        .bind(paper_id)
        .bind(&chunk.section)
        .bind(&chunk.content)
        .execute(lit_pool)
        .await
        .map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("lit_chunks insert: {e}"),
                degraded: true,
                degradation_reason: Some(format!(
                    "lit_chunks insert failed for {paper_id} chunk {}: {e}",
                    chunk.chunk_index
                )),
            })
        })?;
    }

    Ok(())
}

/// Persist one arxiv paper (metadata + chunks) into the literature store.
///
/// 1. Upserts one `papers` row (`paper_id = "arxiv:{arxiv_id}"`, `source = "arxiv"`).
/// 2. For each chunk: builds the embedding input via `chunk_embedding_input`,
///    embeds via the injected `embedder`, inserts the vector into `lit_store`,
///    and inserts a `lit_chunks` row carrying the raw `chunk.content`.
///
/// When `had_ar5iv=false` the caller should pass a single abstract chunk so the
/// paper is still indexed (provenance intact).
///
/// The `embedder` parameter is a trait object (`&dyn EmbeddingService`); Plan
/// 21-04 resolves it from `AppState.embedder` and passes `&*arc`. No concrete
/// embedder type leaks into this signature (RETR-02 anti-pattern).
pub async fn persist_arxiv(
    lit_pool: &SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn base::search::EmbeddingService,
    meta: &ArxivResult,
    chunks: &[LitChunk],
) -> AlzinaResult<()> {
    // 1. Build provenance fields.
    let paper_id = format!("arxiv:{}", meta.arxiv_id);
    let authors = serde_json::to_string(&meta.authors).unwrap_or_else(|_| "[]".to_string());
    let fetched_at = chrono::Utc::now().to_rfc3339();

    // Parse year from published (ISO 8601 prefix "YYYY-...").
    let year = meta
        .published
        .split('-')
        .next()
        .and_then(|y| y.parse::<i32>().ok());

    // ar5iv abs URL as canonical paper URL.
    let url = format!("https://ar5iv.labs.arxiv.org/abs/{}", meta.arxiv_id);

    // 2. Upsert the papers row.
    upsert_paper(
        lit_pool,
        &paper_id,
        "arxiv",
        Some(&meta.arxiv_id),
        None, // doi
        None, // s2_paper_id
        &meta.title,
        Some(&meta.abstract_text),
        &url,
        year,
        &authors,
        None, // citation_count — not from Atom feed
        &fetched_at,
    )
    .await?;

    // 3. For each chunk: embed + insert using the shared helper.
    persist_chunks_for_paper(
        lit_pool,
        lit_store,
        embedder,
        &paper_id,
        "arxiv",
        &meta.title,
        Some(&meta.published),
        chunks,
    )
    .await?;

    Ok(())
}

/// Persist abstract-only for one arxiv paper.
///
/// Writes ONE `papers` row (`fulltext_status = 'none'`) and ONE embedded
/// `lit_chunks` row (section `"Abstract"`, `chunk_index = 0`, content =
/// `meta.abstract_text`). The embedding input uses the locked prefix via
/// `chunk_embedding_input`. Calling this twice for the same `arxiv_id` yields
/// exactly one `papers` row and one `lit_chunks` row (upsert dedup on both
/// sides — `papers ON CONFLICT(paper_id)` + `lit_chunks ON CONFLICT(rowid)`).
///
/// Does NOT call `fetch_fulltext` or `chunk_ar5iv_html`. No ar5iv traffic here.
///
/// T-260611-01: all SQL uses bound parameters — `paper_id` and text fields
/// are never interpolated into query strings.
pub async fn persist_arxiv_abstract(
    lit_pool: &sqlx::SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn base::search::EmbeddingService,
    meta: &ArxivResult,
) -> base::error::AlzinaResult<()> {
    let abstract_chunk = LitChunk {
        section: "Abstract".to_string(),
        section_id: "".to_string(),
        content: meta.abstract_text.clone(),
        chunk_index: 0,
    };
    let chunks = vec![abstract_chunk];
    // persist_arxiv handles upsert_paper + lit_chunks dedup.
    persist_arxiv(lit_pool, lit_store, embedder, meta, &chunks).await
}

/// Persist abstract-only for one S2-native paper (no arxiv id).
///
/// The "S2 equivalent" of `persist_arxiv_abstract` for papers WITHOUT an arxiv id:
/// - `paper_id = "s2:{s2_id}"`
/// - `source = "s2"`
/// - Populates `doi`, `s2_paper_id`, `citation_count` from `S2PaperFull`.
/// - `url = "https://www.semanticscholar.org/paper/{s2_id}"`.
///
/// When `abstract_text` is Some: embeds ONE chunk (section "Abstract",
/// chunk_index 0) using the locked prefix `chunk_embedding_input(title, "Abstract", abstract)`,
/// inserts into lit_store with `source_type = "s2"` and a lit_chunks sidecar row.
/// When None: papers row only (provenance survives; no chunk written — mirrors
/// clawd `_index_paper` returning False for abstract-less papers, :420-421).
///
/// Idempotent: calling twice for the same s2_id yields exactly one papers row and
/// at most one lit_chunks row (ON CONFLICT upsert on both sides).
///
/// T-260611-01: SQL via bound parameters only.
pub async fn persist_s2_abstract(
    lit_pool: &sqlx::SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn base::search::EmbeddingService,
    p: &S2PaperFull,
) -> base::error::AlzinaResult<()> {
    use base::search::{EmbeddingTask, VectorMetadata, VectorStore};

    let paper_id = format!("s2:{}", p.s2_id);
    let authors = serde_json::to_string(&p.authors).unwrap_or_else(|_| "[]".to_string());
    let fetched_at = chrono::Utc::now().to_rfc3339();
    let url = format!("https://www.semanticscholar.org/paper/{}", p.s2_id);
    let citation_count = i32::try_from(p.citation_count).ok();

    // Upsert the papers row.
    upsert_paper(
        lit_pool,
        &paper_id,
        "s2",
        p.arxiv_id.as_deref(),
        p.doi.as_deref(),
        Some(&p.s2_id),
        &p.title,
        p.abstract_text.as_deref(),
        &url,
        p.year,
        &authors,
        citation_count,
        &fetched_at,
    )
    .await?;

    // Persist open-access PDF URL when present (F10).
    // set_only semantics: never writes NULL so later metadata re-upserts
    // from sources that lack the URL cannot erase it.
    if let Some(pdf_url) = &p.open_access_pdf_url {
        if let Err(e) = set_open_access_pdf_url(lit_pool, &paper_id, pdf_url).await {
            tracing::warn!(
                paper_id = %paper_id,
                error = %e,
                "persist_s2_abstract: set_open_access_pdf_url failed (non-fatal)"
            );
        }
    }

    // Persist the source-credibility signals the S2 graph API already returned
    // (influential count + venue) — these feed the per-source authenticity tier.
    // Preserve-on-unknown: 0 influential maps to None so it never clobbers a
    // real count; blank venue is treated as unknown. Non-fatal.
    let influential = i32::try_from(p.influential_citation_count)
        .ok()
        .filter(|&n| n > 0);
    if let Err(e) =
        update_paper_credibility(lit_pool, &paper_id, citation_count, influential, p.venue.as_deref())
            .await
    {
        tracing::warn!(
            paper_id = %paper_id,
            error = %e,
            "persist_s2_abstract: update_paper_credibility failed (non-fatal)"
        );
    }

    // Embed and store a single Abstract chunk when abstract is present.
    if let Some(abstract_text) = &p.abstract_text {
        let embed_input = chunk_embedding_input(&p.title, "Abstract", abstract_text);

        let vector = embedder
            .embed(&embed_input, EmbeddingTask::Passage)
            .await
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("persist_s2_abstract embed: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!(
                        "embedding failed for s2:{} abstract: {e}",
                        p.s2_id
                    )),
                })
            })?;

        let metadata = VectorMetadata {
            source_type: "s2".to_string(),
            source_id: paper_id.clone(),
            chunk_index: 0,
            content_preview: abstract_text.chars().take(200).collect(),
            source_agent: None,
            source_date: p.year.map(|y| format!("{y}-01-01T00:00:00Z")),
            weave_id: None,
            section: Some("Abstract".to_string()),
            domain: None,
            indexed_at: fetched_at.clone(),
        };
        let rowid = lit_store.insert(&vector, metadata).await?;

        // lit_chunks sidecar row.
        sqlx::query(
            "INSERT INTO lit_chunks (\
                rowid, source_type, source_id, chunk_index, content_preview,\
                source_date, indexed_at, paper_id, section, content\
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\
             ON CONFLICT(rowid) DO UPDATE SET \
                content    = excluded.content,\
                section    = excluded.section,\
                paper_id   = excluded.paper_id,\
                indexed_at = excluded.indexed_at",
        )
        .bind(rowid)
        .bind("s2")
        .bind(&paper_id)
        .bind(0i64)
        .bind(abstract_text.chars().take(200).collect::<String>())
        .bind(p.year.map(|y| format!("{y}-01-01T00:00:00Z")))
        .bind(&fetched_at)
        .bind(&paper_id)
        .bind("Abstract")
        .bind(abstract_text)
        .execute(lit_pool)
        .await
        .map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("persist_s2_abstract lit_chunks insert: {e}"),
                degraded: true,
                degradation_reason: Some(format!(
                    "lit_chunks insert failed for s2:{}: {e}",
                    p.s2_id
                )),
            })
        })?;
    }

    Ok(())
}

/// Promote an already-abstract-indexed arxiv paper to full text.
///
/// Defensive early-return `Ok(())` when the paper's `fulltext_status` is
/// already `'indexed'` (no re-fetch). Otherwise: fetch ar5iv, chunk, embed,
/// persist the full chunk set via `persist_arxiv`, then set status to
/// `'indexed'`. On any error: set status to `'failed'`, log loud at `warn`,
/// return `Ok(())` so background callers can log-and-continue.
///
/// The `paper_id` is `"arxiv:{arxiv_id}"` — callers derive it from the
/// `ArxivResult.arxiv_id` field.
///
/// T-260611-01: SQL paths are `set_fulltext_status` (bound param only).
/// T-260611-02: gateway budget + single-flight are the caller's responsibility;
/// this function performs the actual fetch+persist work only.
pub async fn promote_arxiv_fulltext(
    lit_pool: &sqlx::SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn base::search::EmbeddingService,
    client: &ArxivClient,
    meta: &ArxivResult,
    chunk_cfg: &LitChunkConfig,
) -> base::error::AlzinaResult<()> {
    let paper_id = format!("arxiv:{}", meta.arxiv_id);

    // Skip if already indexed.
    let status_row: Option<(String,)> =
        sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = ? LIMIT 1")
            .bind(&paper_id)
            .fetch_optional(lit_pool)
            .await
            .ok()
            .flatten();

    if status_row.as_ref().map(|(s,)| s.as_str()) == Some("indexed") {
        return Ok(());
    }

    // Fetch ar5iv full text.
    let full_text = match client.fetch_fulltext(&meta.arxiv_id, &meta.abstract_text).await {
        Ok(ft) => ft,
        Err(e) => {
            tracing::warn!(
                arxiv_id = %meta.arxiv_id,
                error = %e,
                "promote_arxiv_fulltext: ar5iv fetch failed — setting status 'failed'"
            );
            let _ = set_fulltext_status(lit_pool, &paper_id, "failed").await;
            return Ok(());
        }
    };

    let chunks = chunk_ar5iv_html(&full_text.body, chunk_cfg);

    if let Err(e) = persist_arxiv(lit_pool, lit_store, embedder, meta, &chunks).await {
        tracing::warn!(
            arxiv_id = %meta.arxiv_id,
            error = %e,
            "promote_arxiv_fulltext: persist_arxiv failed — setting status 'failed'"
        );
        let _ = set_fulltext_status(lit_pool, &paper_id, "failed").await;
        return Ok(());
    }

    if let Err(e) = set_fulltext_status(lit_pool, &paper_id, "indexed").await {
        tracing::warn!(
            arxiv_id = %meta.arxiv_id,
            error = %e,
            "promote_arxiv_fulltext: set_fulltext_status 'indexed' failed"
        );
    }

    Ok(())
}

/// Promote an already-abstract-indexed s2:* paper to full text via a PDF URL.
///
/// Mirrors `promote_arxiv_fulltext`'s loud-degrade contract exactly:
/// - Returns `Ok(())` on ALL paths; failures → `tracing::warn!` + status `'failed'`.
/// - The paper's existing `papers` row + abstract chunk are never touched on
///   failure — abstract-only retrieval keeps working.
///
/// Steps:
/// 1. Skip-return `Ok(())` when `fulltext_status` is already `'indexed'`.
/// 2. `fetch_pdf_bytes` — on error: warn + set `'failed'` + `Ok(())`.
/// 3. `pdftotext_extract` — on error: warn + set `'failed'` + `Ok(())`.
/// 4. `chunk_plain_text` — EMPTY chunk vec is a failure (extraction produced
///    nothing useful); warn + set `'failed'` + `Ok(())`.
/// 5. `persist_chunks_for_paper` with `source_type = "s2"` — on error: warn +
///    set `'failed'` + `Ok(())`.
/// 6. `set_fulltext_status 'indexed'` — on error: warn + still `Ok(())`.
///
/// Gateway budget + single-flight are the **caller's responsibility** (same
/// T-260611-02 note as the arxiv twin). This function performs the fetch+persist
/// work only.
pub async fn promote_pdf_fulltext(
    lit_pool: &sqlx::SqlitePool,
    lit_store: &SqliteVecStore,
    embedder: &dyn base::search::EmbeddingService,
    pdf_cfg: &PdfFetchConfig,
    paper_id: &str,
    pdf_url: &str,
    title: &str,
    chunk_cfg: &LitChunkConfig,
) -> base::error::AlzinaResult<()> {
    // 1. Skip if already indexed.
    let status_row: Option<(String,)> =
        sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = ? LIMIT 1")
            .bind(paper_id)
            .fetch_optional(lit_pool)
            .await
            .ok()
            .flatten();

    if status_row.as_ref().map(|(s,)| s.as_str()) == Some("indexed") {
        return Ok(());
    }

    // 2. Fetch PDF bytes.
    let pdf_bytes = match fetch_pdf_bytes(pdf_url, pdf_cfg).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                paper_id = %paper_id,
                error = %e,
                "promote_pdf_fulltext: PDF fetch failed — setting status 'failed'"
            );
            let _ = set_fulltext_status(lit_pool, paper_id, "failed").await;
            return Ok(());
        }
    };

    // 3. Extract text via pdftotext.
    let text = match pdftotext_extract(&pdf_bytes, pdf_cfg).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                paper_id = %paper_id,
                error = %e,
                "promote_pdf_fulltext: pdftotext extraction failed — setting status 'failed'"
            );
            let _ = set_fulltext_status(lit_pool, paper_id, "failed").await;
            return Ok(());
        }
    };

    // 4. Chunk. Empty result = extraction produced nothing usable.
    let chunks = chunk_plain_text(&text, chunk_cfg);
    if chunks.is_empty() {
        tracing::warn!(
            paper_id = %paper_id,
            "promote_pdf_fulltext: chunk_plain_text returned empty — setting status 'failed'"
        );
        let _ = set_fulltext_status(lit_pool, paper_id, "failed").await;
        return Ok(());
    }

    // 5. Embed and persist.
    if let Err(e) = persist_chunks_for_paper(
        lit_pool,
        lit_store,
        embedder,
        paper_id,
        "s2",
        title,
        None, // source_date not available from the DB papers row
        &chunks,
    )
    .await
    {
        tracing::warn!(
            paper_id = %paper_id,
            error = %e,
            "promote_pdf_fulltext: persist_chunks_for_paper failed — setting status 'failed'"
        );
        let _ = set_fulltext_status(lit_pool, paper_id, "failed").await;
        return Ok(());
    }

    // 6. Mark indexed.
    if let Err(e) = set_fulltext_status(lit_pool, paper_id, "indexed").await {
        tracing::warn!(
            paper_id = %paper_id,
            error = %e,
            "promote_pdf_fulltext: set_fulltext_status 'indexed' failed"
        );
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// Tests for the staged-ingest helpers (A2 plan + A4 persist_s2_abstract).
#[cfg(test)]
mod staged_ingest {
    use crate::arxiv::ArxivResult;
    use crate::sqlite_vec::SqliteVecStore;
    use base::error::AlzinaResult;

    struct StubEmbedder;

    #[async_trait::async_trait]
    impl base::search::EmbeddingService for StubEmbedder {
        async fn embed(
            &self,
            _text: &str,
            _task: base::search::EmbeddingTask,
        ) -> AlzinaResult<Vec<f32>> {
            let mut v = vec![0.0f32; 1024];
            v[0] = 1.0;
            Ok(v)
        }
        async fn embed_batch(
            &self,
            texts: &[String],
            task: base::search::EmbeddingTask,
        ) -> AlzinaResult<Vec<Vec<f32>>> {
            let mut out = Vec::new();
            for t in texts {
                out.push(self.embed(t, task).await?);
            }
            Ok(out)
        }
        fn dimensions(&self) -> usize { 1024 }
    }

    async fn make_pool_and_store_or_skip(tag: &str) -> Option<(sqlx::sqlite::SqlitePool, SqliteVecStore)> {
        let pool = crate::lit_schema::in_memory_lit_pool().await.expect("in_memory_lit_pool");
        let store = SqliteVecStore::with_table_names(pool.clone(), 1024, "lit_vec0", "lit_chunks")
            .await
            .expect("with_table_names");
        if !store.is_enabled() {
            eprintln!("skipping staged_ingest test {tag}: sqlite-vec extension not loaded");
            return None;
        }
        Some((pool, store))
    }

    fn test_meta(arxiv_id: &str) -> ArxivResult {
        ArxivResult {
            arxiv_id: arxiv_id.to_string(),
            title: "Test Paper Title".to_string(),
            abstract_text: "This is the abstract of the test paper.".to_string(),
            authors: vec!["Author A".to_string()],
            published: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    /// double persist_arxiv_abstract → 1 papers row + 1 lit_chunks row, fulltext_status 'none'
    #[tokio::test]
    async fn double_persist_abstract_yields_one_row_each() {
        let Some((pool, store)) = make_pool_and_store_or_skip("double_persist").await else {
            return;
        };

        let meta = test_meta("2501.00001");

        super::persist_arxiv_abstract(&pool, &store, &StubEmbedder, &meta)
            .await
            .expect("first persist_arxiv_abstract should succeed");
        super::persist_arxiv_abstract(&pool, &store, &StubEmbedder, &meta)
            .await
            .expect("second persist_arxiv_abstract should succeed (idempotent)");

        let (papers_count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM papers WHERE paper_id = ?")
                .bind("arxiv:2501.00001")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(papers_count, 1, "must be exactly one papers row after two calls");

        let (chunks_count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM lit_chunks WHERE paper_id = ?")
                .bind("arxiv:2501.00001")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(chunks_count, 1, "must be exactly one lit_chunks row after two calls");

        let (status,): (String,) =
            sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = ?")
                .bind("arxiv:2501.00001")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "none", "fulltext_status must be 'none' after abstract-only ingest");
    }

    /// abstract chunk has section='Abstract' and chunk_index=0
    #[tokio::test]
    async fn persist_abstract_stores_abstract_section() {
        let Some((pool, store)) = make_pool_and_store_or_skip("abstract_section").await else {
            return;
        };

        let meta = test_meta("2501.00002");
        super::persist_arxiv_abstract(&pool, &store, &StubEmbedder, &meta)
            .await
            .expect("persist_arxiv_abstract");

        let row: (Option<String>, i64, Option<String>) =
            sqlx::query_as("SELECT section, chunk_index, content FROM lit_chunks WHERE paper_id = ?")
                .bind("arxiv:2501.00002")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(row.0.as_deref(), Some("Abstract"), "section must be 'Abstract'");
        assert_eq!(row.1, 0, "chunk_index must be 0");
        assert_eq!(
            row.2.as_deref(),
            Some("This is the abstract of the test paper."),
            "content must be the raw abstract text"
        );
    }

    /// paper_is_ingested returns false for absent paper_id, true after ingest
    #[tokio::test]
    async fn paper_is_ingested_returns_correct_values() {
        let Some((pool, store)) = make_pool_and_store_or_skip("is_ingested").await else {
            return;
        };

        assert!(
            !crate::lit_schema::paper_is_ingested(&pool, "arxiv:2501.00003").await.unwrap(),
            "paper_is_ingested must return false for absent id"
        );

        let meta = test_meta("2501.00003");
        super::persist_arxiv_abstract(&pool, &store, &StubEmbedder, &meta)
            .await
            .expect("persist_arxiv_abstract");

        assert!(
            crate::lit_schema::paper_is_ingested(&pool, "arxiv:2501.00003").await.unwrap(),
            "paper_is_ingested must return true after ingest"
        );
    }

    // ── persist_s2_abstract tests ──────────────────────────────────────────

    fn s2_full_paper(s2_id: &str, with_abstract: bool) -> crate::s2_enrichment::S2PaperFull {
        crate::s2_enrichment::S2PaperFull {
            s2_id: s2_id.to_string(),
            arxiv_id: None,
            title: format!("S2 Paper {s2_id}"),
            abstract_text: if with_abstract {
                Some(format!("Abstract text for {s2_id}"))
            } else {
                None
            },
            year: Some(2023),
            citation_count: 10,
            influential_citation_count: 2,
            reference_count: 5,
            authors: vec!["Author X".to_string()],
            venue: Some("ICML".to_string()),
            doi: None,
            open_access_pdf_url: None,
        }
    }

    /// double persist_s2_abstract → 1 papers row + 1 lit_chunks row
    #[tokio::test]
    async fn s2_double_persist_abstract_yields_one_row_each() {
        let Some((pool, store)) = make_pool_and_store_or_skip("s2_double_persist").await else {
            return;
        };
        let p = s2_full_paper("s2test001", true);

        super::persist_s2_abstract(&pool, &store, &StubEmbedder, &p)
            .await
            .expect("first persist_s2_abstract should succeed");
        super::persist_s2_abstract(&pool, &store, &StubEmbedder, &p)
            .await
            .expect("second persist_s2_abstract should succeed (idempotent)");

        let (papers_count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM papers WHERE paper_id = ?")
                .bind("s2:s2test001")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(papers_count, 1, "must be exactly one papers row after two calls");

        let (chunks_count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM lit_chunks WHERE paper_id = ?")
                .bind("s2:s2test001")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(chunks_count, 1, "must be exactly one lit_chunks row after two calls");

        // source_type must be "s2"
        let (source_type,): (String,) =
            sqlx::query_as("SELECT source_type FROM lit_chunks WHERE paper_id = ?")
                .bind("s2:s2test001")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(source_type, "s2");
    }

    /// persist_s2_abstract with abstract=None → papers row written, zero lit_chunks rows
    #[tokio::test]
    async fn s2_persist_abstract_none_writes_papers_only() {
        let Some((pool, store)) = make_pool_and_store_or_skip("s2_no_abstract").await else {
            return;
        };
        let p = s2_full_paper("s2test002", false);

        super::persist_s2_abstract(&pool, &store, &StubEmbedder, &p)
            .await
            .expect("persist_s2_abstract with no abstract should succeed");

        let (papers_count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM papers WHERE paper_id = ?")
                .bind("s2:s2test002")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(papers_count, 1, "papers row must be written even without abstract");

        let (chunks_count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM lit_chunks WHERE paper_id = ?")
                .bind("s2:s2test002")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(chunks_count, 0, "no lit_chunks row when abstract is None");
    }

    /// persist_s2_abstract stores correct provenance fields
    #[tokio::test]
    async fn s2_persist_abstract_provenance() {
        let Some((pool, store)) = make_pool_and_store_or_skip("s2_provenance").await else {
            return;
        };
        let mut p = s2_full_paper("s2test003", true);
        p.doi = Some("10.1234/test".to_string());

        super::persist_s2_abstract(&pool, &store, &StubEmbedder, &p)
            .await
            .expect("persist_s2_abstract");

        let row: (String, Option<String>, Option<String>, String) =
            sqlx::query_as("SELECT source, doi, s2_paper_id, url FROM papers WHERE paper_id = ?")
                .bind("s2:s2test003")
                .fetch_one(&pool)
                .await
                .unwrap();
        let (source, doi, s2_paper_id, url) = row;
        assert_eq!(source, "s2");
        assert_eq!(doi.as_deref(), Some("10.1234/test"));
        assert_eq!(s2_paper_id.as_deref(), Some("s2test003"));
        assert!(url.contains("s2test003"));
    }

    /// set_fulltext_status flips a row to 'indexed' and paper_is_ingested returns true
    #[tokio::test]
    async fn set_fulltext_status_updates_row() {
        let Some((pool, _store)) = make_pool_and_store_or_skip("set_status").await else {
            return;
        };

        // Insert a paper row directly (no store needed for status test).
        crate::lit_schema::upsert_paper(
            &pool,
            "arxiv:2501.00004",
            "arxiv",
            Some("2501.00004"),
            None,
            None,
            "Status Test",
            Some("abstract"),
            "https://ar5iv.labs.arxiv.org/abs/2501.00004",
            Some(2024),
            "[]",
            None,
            "2024-01-01T00:00:00Z",
        )
        .await
        .unwrap();

        crate::lit_schema::set_fulltext_status(&pool, "arxiv:2501.00004", "indexed")
            .await
            .unwrap();

        let (status,): (String,) =
            sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = ?")
                .bind("arxiv:2501.00004")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "indexed");

        assert!(
            crate::lit_schema::paper_is_ingested(&pool, "arxiv:2501.00004").await.unwrap(),
            "paper_is_ingested must return true after status update"
        );
    }
}

#[cfg(test)]
mod s2 {
    use crate::s2_enrichment::S2Result;

    use super::{persist_s2_result, s2_client_for_lit};

    /// s2_client_for_lit() enables the client IFF S2_API_KEY is present.
    /// Unkeyed → disabled (local-only, never anonymous S2). S2_LIVE_ENABLED is
    /// irrelevant on the lit path — the key alone decides.
    ///
    /// Both states are exercised in ONE test so the process-global S2_API_KEY
    /// env var is never mutated by two tests in parallel.
    #[test]
    fn s2_lit_path_enabled_iff_keyed() {
        // Safety: single-threaded test body; the only normal-suite reader of
        // S2_API_KEY is s2_client_for_lit, called sequentially below.
        let saved = std::env::var("S2_API_KEY").ok();
        unsafe { std::env::remove_var("S2_LIVE_ENABLED"); }

        // No key → disabled (no live S2 traffic).
        unsafe { std::env::remove_var("S2_API_KEY"); }
        let unkeyed = s2_client_for_lit().expect("s2_client_for_lit should not fail");
        assert!(
            !unkeyed.is_enabled(),
            "lit-path S2 client must be DISABLED when S2_API_KEY is absent"
        );

        // Key present → enabled, even with S2_LIVE_ENABLED=false.
        unsafe {
            std::env::set_var("S2_API_KEY", "test_key_value");
            std::env::set_var("S2_LIVE_ENABLED", "false");
        }
        let keyed = s2_client_for_lit().expect("s2_client_for_lit should not fail");
        assert!(
            keyed.is_enabled(),
            "lit-path S2 client must be ENABLED when S2_API_KEY is present"
        );

        // Restore.
        unsafe {
            std::env::remove_var("S2_LIVE_ENABLED");
            match saved {
                Some(v) => std::env::set_var("S2_API_KEY", v),
                None => std::env::remove_var("S2_API_KEY"),
            }
        }
    }

    /// persist_s2_result writes all 7 S2Result fields into papers;
    /// a re-read after reopening the pool returns identical values.
    #[tokio::test]
    async fn s2_persist_all_fields() {
        let pool = crate::lit_schema::in_memory_lit_pool()
            .await
            .expect("in_memory_lit_pool");

        let r = S2Result {
            arxiv_id: None,
            paper_id: "abc123def456".to_string(),
            title: "Attention Is All You Need".to_string(),
            abstract_text: Some("Dominant sequence transduction models...".to_string()),
            year: Some(2017),
            authors: vec!["Vaswani".to_string(), "Shazeer".to_string()],
            citation_count: Some(50000),
            url: "https://www.semanticscholar.org/paper/abc123def456".to_string(),
            open_access_pdf_url: None,
        };

        persist_s2_result(&pool, &r).await.expect("persist_s2_result should succeed");

        let row: (String, String, Option<String>, Option<i32>, String, Option<i64>, String) =
            sqlx::query_as(
                "SELECT source, title, abstract, year, authors, citation_count, s2_paper_id \
                 FROM papers WHERE paper_id = ?",
            )
            .bind("s2:abc123def456")
            .fetch_one(&pool)
            .await
            .expect("papers row should exist");

        let (source, title, abstract_text, year, authors, citation_count, s2_paper_id) = row;
        assert_eq!(source, "s2");
        assert_eq!(title, "Attention Is All You Need");
        assert!(abstract_text.as_deref().unwrap_or("").contains("Dominant"));
        assert_eq!(year, Some(2017));
        let parsed_authors: Vec<String> = serde_json::from_str(&authors).unwrap();
        assert!(parsed_authors.contains(&"Vaswani".to_string()));
        assert_eq!(citation_count, Some(50000));
        assert_eq!(s2_paper_id, "abc123def456");
    }

    /// Restart-survivable: close the pool, open a new one, re-read the row.
    #[tokio::test]
    async fn s2_persist_restart_survivable() {
        use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
        use std::str::FromStr;
        use tempfile::NamedTempFile;

        let tmp = NamedTempFile::new().unwrap();
        let db_path = tmp.path().to_str().unwrap().to_string();
        let db_url = format!("sqlite:{db_path}");

        {
            let opts = SqliteConnectOptions::from_str(&db_url).unwrap().create_if_missing(true);
            let pool1 = SqlitePoolOptions::new().max_connections(1).connect_with(opts).await.unwrap();
            crate::lit_schema::migrate(&pool1, 1024).await.unwrap();

            let r = S2Result {
                arxiv_id: None,
                paper_id: "survive001".to_string(),
                title: "Restart Test Paper".to_string(),
                abstract_text: Some("Some abstract".to_string()),
                year: Some(2024),
                authors: vec!["Author A".to_string()],
                citation_count: Some(42),
                url: "https://www.semanticscholar.org/paper/survive001".to_string(),
                open_access_pdf_url: None,
            };
            persist_s2_result(&pool1, &r).await.unwrap();
        }

        {
            let opts = SqliteConnectOptions::from_str(&db_url).unwrap().create_if_missing(false);
            let pool2 = SqlitePoolOptions::new().max_connections(1).connect_with(opts).await.unwrap();

            let row: (String, String, Option<String>, Option<i32>, String, Option<i64>, String) =
                sqlx::query_as(
                    "SELECT source, title, abstract, year, authors, citation_count, s2_paper_id \
                     FROM papers WHERE paper_id = ?",
                )
                .bind("s2:survive001")
                .fetch_one(&pool2)
                .await
                .expect("row must survive pool reopen");

            let (source, title, abstract_text, year, authors, citation_count, s2_paper_id) = row;
            assert_eq!(source, "s2");
            assert_eq!(title, "Restart Test Paper");
            assert_eq!(abstract_text.as_deref(), Some("Some abstract"));
            assert_eq!(year, Some(2024));
            let parsed: Vec<String> = serde_json::from_str(&authors).unwrap();
            assert!(parsed.contains(&"Author A".to_string()));
            assert_eq!(citation_count, Some(42));
            assert_eq!(s2_paper_id, "survive001");
        }
    }
}

#[cfg(test)]
mod arxiv_tests {
    use crate::arxiv::ArxivResult;
    use crate::lit_chunking::LitChunk;
    use crate::sqlite_vec::SqliteVecStore;
    use base::error::AlzinaResult;
    use sqlx::sqlite::SqlitePool;

    use super::{chunk_embedding_input, persist_arxiv};

    // ── Stub embedder ────────────────────────────────────────────────────

    struct StubEmbedder;

    #[async_trait::async_trait]
    impl base::search::EmbeddingService for StubEmbedder {
        async fn embed(
            &self,
            _text: &str,
            _task: base::search::EmbeddingTask,
        ) -> AlzinaResult<Vec<f32>> {
            // Return a unit vector: 1.0 at index 0, rest 0.0.
            let mut v = vec![0.0f32; 1024];
            v[0] = 1.0;
            Ok(v)
        }
        async fn embed_batch(
            &self,
            texts: &[String],
            task: base::search::EmbeddingTask,
        ) -> AlzinaResult<Vec<Vec<f32>>> {
            let mut out = Vec::new();
            for t in texts {
                out.push(self.embed(t, task).await?);
            }
            Ok(out)
        }
        fn dimensions(&self) -> usize { 1024 }
    }

    // ── Helper: build pool + store; skip test if sqlite-vec not loaded ───

    async fn make_lit_pool_and_store_or_skip(
        tag: &str,
    ) -> Option<(SqlitePool, SqliteVecStore)> {
        let pool = crate::lit_schema::in_memory_lit_pool()
            .await
            .expect("in_memory_lit_pool");
        let store = SqliteVecStore::with_table_names(
            pool.clone(),
            1024,
            "lit_vec0",
            "lit_chunks",
        )
        .await
        .expect("with_table_names");
        if !store.is_enabled() {
            eprintln!("skipping lit_intake test {tag}: sqlite-vec extension not loaded");
            return None;
        }
        Some((pool, store))
    }

    // ── Tests ─────────────────────────────────────────────────────────────

    /// chunk_embedding_input returns exactly title\nsection\ncontent.
    #[test]
    fn embedding_prefix_applied() {
        let input = chunk_embedding_input("My Title", "2 Related Work", "Some content here.");
        assert_eq!(input, "My Title\n2 Related Work\nSome content here.");
    }

    /// persist_arxiv writes exactly one papers row + one lit_chunks row per chunk.
    #[tokio::test]
    async fn arxiv_persist_one_source_id() {
        let Some((pool, store)) = make_lit_pool_and_store_or_skip("one_source_id").await else {
            return;
        };

        let meta = ArxivResult {
            arxiv_id: "2105.14103".to_string(),
            title: "An Attention Free Transformer".to_string(),
            abstract_text: "We introduce AFT...".to_string(),
            authors: vec!["Zhai".to_string()],
            published: "2021-05-28T20:45:30Z".to_string(),
        };
        let chunks = vec![
            LitChunk { section: "1 Intro".to_string(), section_id: "S1".to_string(), content: "Intro text.".to_string(), chunk_index: 0 },
            LitChunk { section: "2 Related".to_string(), section_id: "S2".to_string(), content: "Related text.".to_string(), chunk_index: 1 },
        ];

        persist_arxiv(&pool, &store, &StubEmbedder, &meta, &chunks).await.expect("persist_arxiv");

        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM papers WHERE paper_id = ?")
            .bind("arxiv:2105.14103")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 1, "must be exactly one papers row per paper");

        let (source,): (String,) = sqlx::query_as("SELECT source FROM papers WHERE paper_id = ?")
            .bind("arxiv:2105.14103")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(source, "arxiv");

        let (chunk_count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM lit_chunks WHERE paper_id = ?")
            .bind("arxiv:2105.14103")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(chunk_count, 2, "must be one lit_chunks row per chunk");
    }

    /// The stored lit_chunks.content is the raw chunk text, not the prefixed embed input.
    #[tokio::test]
    async fn embedding_prefix_raw_content_in_chunks() {
        let Some((pool, store)) = make_lit_pool_and_store_or_skip("raw_content").await else {
            return;
        };

        let meta = ArxivResult {
            arxiv_id: "2105.99999".to_string(),
            title: "Some Title".to_string(),
            abstract_text: "abstract.".to_string(),
            authors: vec!["A".to_string()],
            published: "2024-01-01T00:00:00Z".to_string(),
        };
        let chunks = vec![LitChunk {
            section: "1 Introduction".to_string(),
            section_id: "S1".to_string(),
            content: "Raw chunk text here.".to_string(),
            chunk_index: 0,
        }];

        persist_arxiv(&pool, &store, &StubEmbedder, &meta, &chunks).await.expect("persist");

        let (stored_content,): (Option<String>,) =
            sqlx::query_as("SELECT content FROM lit_chunks WHERE paper_id = ?")
                .bind("arxiv:2105.99999")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(
            stored_content.as_deref(),
            Some("Raw chunk text here."),
            "lit_chunks.content must be unprefixed raw chunk text"
        );
    }

    /// Abstract-fallback (had_ar5iv=false): abstract chunk still produces
    /// one papers row and at least one lit_chunks row.
    #[tokio::test]
    async fn no_ar5iv_still_indexed() {
        let Some((pool, store)) = make_lit_pool_and_store_or_skip("no_ar5iv").await else {
            return;
        };

        let meta = ArxivResult {
            arxiv_id: "2024.abstract".to_string(),
            title: "Abstract Only Paper".to_string(),
            abstract_text: "No ar5iv render for this paper.".to_string(),
            authors: vec!["Author".to_string()],
            published: "2024-01-01T00:00:00Z".to_string(),
        };
        let chunks = vec![LitChunk {
            section: "Abstract".to_string(),
            section_id: "".to_string(),
            content: meta.abstract_text.clone(),
            chunk_index: 0,
        }];

        persist_arxiv(&pool, &store, &StubEmbedder, &meta, &chunks).await.expect("persist");

        let (p,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM papers WHERE paper_id = ?")
            .bind("arxiv:2024.abstract")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(p, 1, "abstract-fallback paper must have one papers row");

        let (c,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM lit_chunks WHERE paper_id = ?")
            .bind("arxiv:2024.abstract")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(c >= 1, "abstract-fallback paper must have at least one lit_chunks row");
    }
}

/// Tests for promote_pdf_fulltext (F10 PDF lane).
#[cfg(test)]
mod promote_pdf {
    use crate::sqlite_vec::SqliteVecStore;
    use base::error::AlzinaResult;

    struct StubEmbedder;

    #[async_trait::async_trait]
    impl base::search::EmbeddingService for StubEmbedder {
        async fn embed(
            &self,
            _text: &str,
            _task: base::search::EmbeddingTask,
        ) -> AlzinaResult<Vec<f32>> {
            let mut v = vec![0.0f32; 1024];
            v[0] = 1.0;
            Ok(v)
        }
        async fn embed_batch(
            &self,
            texts: &[String],
            task: base::search::EmbeddingTask,
        ) -> AlzinaResult<Vec<Vec<f32>>> {
            let mut out = Vec::new();
            for t in texts {
                out.push(self.embed(t, task).await?);
            }
            Ok(out)
        }
        fn dimensions(&self) -> usize { 1024 }
    }

    async fn make_pool_and_store_or_skip(
        tag: &str,
    ) -> Option<(sqlx::sqlite::SqlitePool, SqliteVecStore)> {
        let pool = crate::lit_schema::in_memory_lit_pool()
            .await
            .expect("in_memory_lit_pool");
        let store =
            SqliteVecStore::with_table_names(pool.clone(), 1024, "lit_vec0", "lit_chunks")
                .await
                .expect("with_table_names");
        if !store.is_enabled() {
            eprintln!("skipping promote_pdf test {tag}: sqlite-vec extension not loaded");
            return None;
        }
        Some((pool, store))
    }

    fn insert_s2_abstract_paper<'a>(
        pool: &'a sqlx::sqlite::SqlitePool,
        paper_id: &'a str,
    ) -> impl std::future::Future<Output = ()> + 'a {
        async move {
            // Insert a minimal papers row with abstract chunk (s2 paper).
            crate::lit_schema::upsert_paper(
                pool,
                paper_id,
                "s2",
                None,
                None,
                Some(&paper_id[3..]), // strip "s2:" prefix
                "Test PDF Paper",
                Some("Abstract of test PDF paper."),
                "https://www.semanticscholar.org/paper/test",
                Some(2023),
                "[]",
                None,
                "2024-01-01T00:00:00Z",
            )
            .await
            .unwrap();
        }
    }

    /// Failure path: promote with a URL that immediately refuses connection.
    /// Returns Ok(()), sets fulltext_status='failed', abstract row still present.
    #[tokio::test]
    async fn failure_path_connection_refused() {
        let Some((pool, store)) = make_pool_and_store_or_skip("failure_conn_refused").await else {
            return;
        };

        let paper_id = "s2:pdf_fail_001";
        insert_s2_abstract_paper(&pool, paper_id).await;

        let pdf_cfg = crate::pdf_fetch::PdfFetchConfig {
            timeout_secs: 5,
            ..crate::pdf_fetch::PdfFetchConfig::default()
        };
        let chunk_cfg = crate::lit_chunking::LitChunkConfig::default();

        let result = super::promote_pdf_fulltext(
            &pool,
            &store,
            &StubEmbedder,
            &pdf_cfg,
            paper_id,
            "http://127.0.0.1:1/x.pdf", // guaranteed connection refused, zero external traffic
            "Test PDF Paper",
            &chunk_cfg,
        )
        .await;

        // Must return Ok(()) — loud-degrade, never Err.
        assert!(result.is_ok(), "promote_pdf_fulltext must return Ok(()) on fetch failure");

        // Status must be 'failed'.
        let (status,): (String,) =
            sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = ?")
                .bind(paper_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "failed", "fulltext_status must be 'failed' after fetch error");

        // Abstract row still present (abstract-only retrieval intact).
        let (count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM papers WHERE paper_id = ?")
                .bind(paper_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count, 1, "papers row must still exist after failed promotion");
    }

    /// Missing binary: pdftotext_extract with a non-existent binary names the path.
    /// (Exercises pdftotext_extract directly — no network call needed.)
    #[tokio::test]
    async fn missing_binary_names_path_in_error() {
        let cfg = crate::pdf_fetch::PdfFetchConfig {
            pdftotext_path: "/nonexistent/pdftotext-no-such-binary".to_string(),
            ..crate::pdf_fetch::PdfFetchConfig::default()
        };
        let err = crate::pdf_fetch::pdftotext_extract(b"%PDF-1.4 test", &cfg)
            .await
            .expect_err("missing binary must return Err");
        match err {
            base::error::AlzinaError::Search(d) => {
                assert!(d.degraded);
                assert!(
                    d.message.contains("/nonexistent/pdftotext-no-such-binary"),
                    "error must name the binary path: {}",
                    d.message
                );
            }
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    /// Skip-if-indexed: set status 'indexed' first, call promote with a
    /// guaranteed-failing URL — status must STAY 'indexed' (early return fired).
    #[tokio::test]
    async fn skip_if_indexed() {
        let Some((pool, store)) = make_pool_and_store_or_skip("skip_if_indexed").await else {
            return;
        };

        let paper_id = "s2:pdf_skip_001";
        insert_s2_abstract_paper(&pool, paper_id).await;

        // Pre-set status to 'indexed'.
        crate::lit_schema::set_fulltext_status(&pool, paper_id, "indexed")
            .await
            .unwrap();

        let pdf_cfg = crate::pdf_fetch::PdfFetchConfig {
            timeout_secs: 5,
            ..crate::pdf_fetch::PdfFetchConfig::default()
        };
        let chunk_cfg = crate::lit_chunking::LitChunkConfig::default();

        super::promote_pdf_fulltext(
            &pool,
            &store,
            &StubEmbedder,
            &pdf_cfg,
            paper_id,
            "http://127.0.0.1:1/x.pdf", // would fail if we get this far
            "Test PDF Paper",
            &chunk_cfg,
        )
        .await
        .expect("promote_pdf_fulltext must return Ok(())");

        // Status must still be 'indexed' — proving early return before any fetch.
        let (status,): (String,) =
            sqlx::query_as("SELECT fulltext_status FROM papers WHERE paper_id = ?")
                .bind(paper_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(
            status, "indexed",
            "skip-if-indexed must leave status 'indexed', not overwrite with 'failed'"
        );
    }
}
