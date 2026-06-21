//! # alzina-search
//!
//! Embedding-backed hybrid search service. Defines `JinaEmbeddingService`,
//! `SqliteVecStore`, and `HybridSearchService` (FTS5 + vector + RRF fusion).
//!
//! AC-1: every search method returns `degraded` and `degradation_reason` so
//! callers can surface quality issues to agents.
//!
//! Phase 2 — see `docs/design/search/work-plan.md` and `00-synthesis.md`.
//!
//! Module stubs are added by subsequent tasks:
//! - `jina` (Task 2.3) — Jina v3 HTTP client implementing `EmbeddingService`
//! - `embed_cache` (Task 2.4) — SHA-256-keyed cache with 90-day TTL
//! - `sqlite_vec` (Task 2.5) — `SqliteVecStore` over the `vec0` virtual table
//! - `hybrid` (Task 2.6) — RRF fusion + recency weighting + degradation routing
//! - `indexer` (Task 2.7) — fire-and-forget write-path indexing
//! - `backfill` (Task 2.9) — historical-entry catch-up

pub mod jina;
pub use jina::JinaEmbeddingService;

pub mod rerank;
pub use rerank::{JinaRerankService, RerankResult};

// ── Phase 2 module registrations (Tasks 2.4 + 2.5) ────────────────────
// Tasks 2.4 (embedding_cache) and 2.5 (SqliteVecStore) share the
// schema migration so they ship together.
pub mod embed_cache;
pub mod schema;
pub mod sqlite_vec;

// ── Phase 24 EXT-03: bibliography store ──────────────────────────────────────
// `BibliographyStore` accumulates cited-in-synthesis sources to the literature
// KB across denoise iterations. Direct SQLite write path — NOT composition
// channels (CONTEXT EXT-03). `NoopBibliographyStore` is the test/no-tracking
// implementation; `SqliteBibliographyStore` is the production implementation.
pub mod bib_store;
pub use bib_store::{BibEntry, BibliographyStore, NoopBibliographyStore, SqliteBibliographyStore};

// ── Phase 21 module registration (Plan 21-01) ─────────────────────────
// `lit_schema` creates the physically separate literature DB migration:
// `papers` (provenance), `lit_vec0` (1024-dim vec0), `lit_chunks`
// (sidecar). See 21-CONTEXT.md: SC-2 wants two distinct on-disk DBs.
pub mod lit_schema;
pub use lit_schema::{
    migrate as lit_migrate, paper_is_ingested, s2_cache_get, s2_cache_put, s2_cache_put_if_absent,
    set_fulltext_status, set_open_access_pdf_url, update_paper_credibility, upsert_paper,
};

pub mod credibility;
pub use credibility::{derive_tier, CredibilityTier, TierThresholds};
#[cfg(test)]
pub use lit_schema::in_memory_lit_pool;

pub use embed_cache::EmbeddingCache;
pub use sqlite_vec::SqliteVecStore;

// ── Phase 2 module registration (Task 2.7) ────────────────────────────
// `SearchIndexer` is the fire-and-forget wrapper called by the write
// path to embed + upsert into the vector store without blocking the
// primary INSERT. Failures are swallowed and reconciled by the
// backfill job (Task 2.9). KB-only → gated behind `kb`.
#[cfg(feature = "kb")]
pub mod indexer;
#[cfg(feature = "kb")]
pub use indexer::SearchIndexer;

// ── Tasks 2.6 + 2.8 + 2.12: hybrid search service ─────────────────────
// `HybridSearchService` fuses FTS5 + vector via RRF, applies recency
// weighting, and routes degradation upward. The `fts_only` constructor
// covers the no-API-key path (Task 2.12) — every call from that mode
// emits a degraded envelope. KB-only (reads alzina-memory FTS) → gated.
#[cfg(feature = "kb")]
pub mod hybrid;
#[cfg(feature = "kb")]
pub use hybrid::{Collection, HybridConfig, HybridSearchService};

// ── Phase 2 module registration (Task 2.9) ────────────────────────────
// `BackfillJob` reconciles the vector index against canonical source
// tables — covers first-rollout catch-up, post-failure reconciliation,
// and full re-index after a model change. Distinct from the FTS5
// `BackfillReport` in `alzina_memory::schema`; namespace separation
// disambiguates. KB-only → gated.
#[cfg(feature = "kb")]
pub mod backfill;
#[cfg(feature = "kb")]
pub use backfill::{BackfillConfig, BackfillJob, BackfillReport};

// ── Phase 3 module registration (Task 3.1) ────────────────────────────
// `chunk_markdown` splits KB articles into heading-aware embeddable
// units. Pure logic — no DB, no async. Consumed by the KbIndexer
// (Task 3.3) which feeds chunks into `EmbeddingService::embed`. KB-only.
#[cfg(feature = "kb")]
pub mod chunking;
#[cfg(feature = "kb")]
pub use chunking::{Chunk, ChunkConfig, chunk_markdown};

// ── Phase 3 module registration (Task 3.7) ────────────────────────────
// `quality` evaluates a result set against synthesis §5.7 thresholds
// (per-result relevance, mean relevance, source concentration, unique
// sources). `HybridSearchService::search` calls `assess_quality` and
// folds `quality_degradation_reason` into the AC-1 degradation chain
// when gates trip — qualitatively poor results are flagged degraded
// even when the underlying lanes ran cleanly.
pub mod quality;
pub use quality::{QualityThresholds, assess_quality, quality_degradation_reason};

// ── Phase 3 module registration (Task 3.2) ────────────────────────────
// `KbManifest` tracks per-file content hashes in `kb/INDEX.toml` so the
// upcoming `KbIndexer` (Task 3.3) only re-embeds files that actually
// changed since the last run. AC-1: load failures (parse errors,
// version-too-new) surface as `AlzinaError::Search` with `degraded=true`.
// KB-only → gated.
#[cfg(feature = "kb")]
pub mod manifest;
#[cfg(feature = "kb")]
pub use manifest::{FileEntry, KbManifest, MANIFEST_FILE, MANIFEST_VERSION, ManifestData};

// ── Phase 3 module registration (Task 3.8) ────────────────────────────
// `S2Client` is an opt-in (`S2_LIVE_ENABLED=true`) Semantic Scholar
// enrichment client. Returns external paper metadata as a SEPARATE
// field on search responses — not fused into the local RRF ranking.
// AC-1: enabled-but-failing returns degraded errors with a reason;
// disabled is silent (returns Ok empty) — by design for an opt-in feature.
pub mod s2_enrichment;
pub use s2_enrichment::{
    resolve_paper_id, S2CallError, S2Client, S2Config, S2PaperFull, S2Result,
    S2_CITATION_FIELDS, S2_DEFAULT_FIELDS,
};

// ── Phase 21 Plan 04 module registrations ────────────────────────────────
// `lit_fusion` fuses internal hybrid + arxiv + S2 lanes via three-lane RRF
// (k=60, dedup by source key) and applies the quality gate (default 0.3 floor).
// hybrid.rs is NOT modified — the RRF math is inlined here.
pub mod lit_fusion;
pub use lit_fusion::{
    ArxivHit, FusedHit, RRF_K_DEFAULT, S2Hit, apply_rerank, fuse_rrf_three_lane, gate_fused,
};

// ── Phase 21 Plan 03 module registrations ────────────────────────────────
// `lit_intake` provides S2 default-on constructor, S2Result→papers persistence,
// and arxiv→papers+lit_chunks persistence with the title+section+chunk embedding
// prefix. See 21-CONTEXT.md SC-1 (default-on) and the locked embedding-prefix
// decision.
pub mod lit_intake;
pub use lit_intake::{
    chunk_embedding_input, persist_arxiv, persist_arxiv_abstract, persist_chunks_for_paper,
    persist_s2_abstract, persist_s2_result, promote_arxiv_fulltext, promote_pdf_fulltext,
    s2_client_for_lit,
};

// ── F10 PDF fetch + pdftotext extraction ─────────────────────────────────────
// `pdf_fetch` provides HTTP PDF byte fetching (scheme allowlist, size cap,
// timeout) and `pdftotext` subprocess extraction with timeout + kill.
// These are the transport layer for `promote_pdf_fulltext`.
pub mod pdf_fetch;
pub use pdf_fetch::{PdfFetchConfig, fetch_pdf_bytes, pdftotext_extract};

// ── Phase 21 Plan 02 module registrations ────────────────────────────────
// `arxiv` provides ArxivClient (Atom search + ar5iv full-text fetch +
// rate-limit + loud degradation) modelled on `s2_enrichment.rs`.
pub mod arxiv;
pub use arxiv::{ArxivClient, ArxivConfig, ArxivFullText, ArxivResult};

// `lit_chunking` provides section-aware HTML chunker returning Vec<LitChunk>
// with paragraph fallback. New code — not the generic chunking.rs token-window
// chunker — because ar5iv gives real section structure to exploit.
pub mod lit_chunking;
pub use lit_chunking::{LitChunk, LitChunkConfig, chunk_ar5iv_html, chunk_plain_text};

// ── Phase 3 module registration (Task 3.3) ────────────────────────────
// `KbIndexer` indexes KB Markdown files into the vector store and
// FTS5 — chunks via heading boundaries, embeds cache-then-API, and
// updates the per-file manifest. This slice ships only the per-file
// path (`index_file` / `remove_file`); the directory-walk `run()`
// follows in 3.3-followup. AC-1: every error is `AlzinaError::Search`
// with `degraded = true`. KB-only (reads alzina-memory FTS) → gated.
#[cfg(feature = "kb")]
pub mod kb_index;
#[cfg(feature = "kb")]
pub use kb_index::{KbIndexConfig, KbIndexer, KbRunReport};

// ── Phase 3 module registration (Task 3.4) ────────────────────────────
// `KbWatcher` watches `kb_root` for `.md` create/modify/delete events
// and dispatches to the indexer on a 2-second per-path debounce. Not
// yet wired into the daemon — that's a follow-up. KB-only → gated.
#[cfg(feature = "kb")]
pub mod watcher;
#[cfg(feature = "kb")]
pub use watcher::{DEFAULT_DEBOUNCE, KbWatcher, KbWatcherHandle};

// ── Phase 3 module registration (Task 3.12) ───────────────────────────
// `QdrantMigration` reads the existing Norn-Weave Qdrant collection
// (`literature_chunks` under
// `/Users/samj/clawd/skills/lit-review/cache/qdrant/`) and re-indexes
// it into the local `SqliteVecStore`. The daemon never depends on
// Qdrant at runtime — only this opt-in migration helper does. AC-1:
// connection / fatal scroll errors return `AlzinaError::Search` with
// `degraded = true`; per-point failures land in the report.
// KB-only (opt-in migration tool, pulls qdrant-client) → gated.
#[cfg(feature = "kb")]
pub mod qdrant_migration;
#[cfg(feature = "kb")]
pub use qdrant_migration::{QdrantMigration, QdrantMigrationConfig, QdrantMigrationReport};

// ── Literature gateway (2026-06-11) ──────────────────────────────────────
// Single chokepoint for external literature traffic: per-endpoint token
// buckets (clawd-parity spacing), per-run budgets with loud local-only
// degradation, exponential backoff, single-flight coalescing. One
// Arc<LitGateway> per synthesize run, shared across lanes/gaps/trajectories.
pub mod lit_gateway;
pub use lit_gateway::{Acquire, Endpoint, GatewaySnapshot, LitGateway, RetryAdvice};

// ── Stage 0 lit exploration (2026-06-11) ─────────────────────────────────────
// `lit_explore` provides `explore_from_query` — the port of clawd's
// `smart_explore.py`. Seeds from the research question via S2 search,
// traverses the citation graph, ranks the frontier by embedding similarity,
// ingests each discovery abstract-first, budgets every call through A1 gateway.
pub mod lit_explore;
pub use lit_explore::{ExploreConfig, ExploreStats, explore_from_queries, explore_from_query};
