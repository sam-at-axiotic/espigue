//! `LitContext` — the owned, daemon-free home for the synthesis pipeline's
//! dependencies.
//!
//! The daemon reads these off a long-lived `AppState`. The standalone CLI owns
//! them for the life of one review: a literature DB pool + vec store, the
//! OpenRouter executor + embedder, an S2 client (disabled when `S2_API_KEY` is
//! absent), an optional reranker, and a per-run `LitGateway`. The pipeline
//! ([`crate::pipeline`]) rebinds every `state.X` to `ctx.X`.

use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use alzina_core::EmbeddingService;
use alzina_orchestration::AgentExecutor;
use alzina_search::{JinaRerankService, LitGateway, S2Client, SqliteVecStore};

use crate::openrouter::embeddings::{OpenRouterEmbeddingService, DEFAULT_DIMENSIONS, DEFAULT_MODEL};
use crate::openrouter::executor::OpenRouterExecutor;
use crate::openrouter::rerank::{openrouter_reranker_with_model, DEFAULT_RERANK_MODEL};

/// A reranker paired with its cross-encoder drop floor (mirrors the daemon's
/// `LitReranker`). Hits scoring below `min_score` are dropped as off-topic.
#[derive(Clone)]
pub struct Reranker {
    pub svc: Arc<JinaRerankService>,
    pub min_score: f32,
}

/// Owned dependency bundle for one synthesis run.
///
/// `Clone` is cheap (every field is an `Arc`); the per-gap retriever closures
/// clone it the way the daemon clones `AppState`.
#[derive(Clone)]
pub struct LitContext {
    /// Literature DB pool — `papers`, `lit_chunks`, `synthesis_bibliography`.
    pub lit_pool: Arc<sqlx::SqlitePool>,
    /// kNN store over `lit_vec0` / `lit_chunks`.
    pub lit_store: Arc<SqliteVecStore>,
    /// Query + passage embeddings (OpenRouter).
    pub embedder: Arc<dyn EmbeddingService>,
    /// Generation executor for the TTD stages (OpenRouter chat).
    pub executor: Arc<dyn AgentExecutor>,
    /// Semantic Scholar client. Always present, but `is_enabled() == false`
    /// when `S2_API_KEY` is absent — `enrich` then returns empty (clean
    /// degrade, never an anonymous live call).
    pub s2_client: Arc<S2Client>,
    /// Cross-encoder reranker (Lever B). `None` disables reranking.
    pub reranker: Option<Reranker>,
    /// Per-run pacing + budget gateway, shared across lanes / gaps.
    pub gateway: Arc<LitGateway>,
}

/// Construction parameters for [`LitContext::open`].
pub struct ContextConfig<'a> {
    /// Path to the literature DB (created if missing).
    pub db_path: &'a Path,
    /// The single `OPENROUTER_API_KEY`.
    pub api_key: String,
    /// Embedding model slug (default [`DEFAULT_MODEL`]).
    pub embedding_model: String,
    /// Embedding dimension (default [`DEFAULT_DIMENSIONS`]). Threaded to the
    /// embedder, the schema migration, and the vec store in lockstep.
    pub embedding_dim: usize,
    /// Rerank model slug; `None` disables reranking entirely.
    pub rerank_model: Option<String>,
    /// Cross-encoder drop floor (only used when `rerank_model` is `Some`).
    pub rerank_min_score: f32,
}

impl<'a> ContextConfig<'a> {
    /// Defaults: `text-embedding-3-small` @ 1536 dims, `cohere/rerank-4-fast`
    /// reranking on, drop floor 0.0 (reorder only, no drop).
    pub fn new(db_path: &'a Path, api_key: String) -> Self {
        Self {
            db_path,
            api_key,
            embedding_model: DEFAULT_MODEL.to_string(),
            embedding_dim: DEFAULT_DIMENSIONS,
            rerank_model: Some(DEFAULT_RERANK_MODEL.to_string()),
            rerank_min_score: 0.0,
        }
    }
}

impl LitContext {
    /// Open the literature DB, migrate the schema at the configured dimension,
    /// and wire the OpenRouter clients. `S2_API_KEY` is read from the
    /// environment (absent → S2 lane disabled, local + arXiv only).
    pub async fn open(cfg: ContextConfig<'_>) -> anyhow::Result<Self> {
        if let Some(parent) = cfg.db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Register sqlite-vec BEFORE opening the pool. `sqlite3_auto_extension`
        // only affects connections opened after registration, and sqlx caches a
        // connection at `connect_with` time — registering first guarantees every
        // pooled connection (migrate, kNN, persist) has the vec0 module. Mirrors
        // the daemon's ordering (builder registers before opening the lit pool).
        if !alzina_search::schema::register_sqlite_vec_extension() {
            anyhow::bail!("failed to register sqlite-vec extension (vec0 module unavailable)");
        }

        // Open the pool directly (do NOT run the memory schema migration here).
        let lit_pool = sqlx::SqlitePool::connect_with(
            sqlx::sqlite::SqliteConnectOptions::from_str(&format!(
                "sqlite:{}",
                cfg.db_path.display()
            ))?
            .create_if_missing(true),
        )
        .await?;

        // Migrate at the configured embedding dimension (lit_vec0 = float[dim]).
        alzina_search::lit_migrate(&lit_pool, cfg.embedding_dim)
            .await
            .map_err(|e| anyhow::anyhow!("literature schema migrate: {e}"))?;

        let lit_pool = Arc::new(lit_pool);

        let lit_store = Arc::new(
            SqliteVecStore::with_table_names(
                (*lit_pool).clone(),
                cfg.embedding_dim,
                "lit_vec0",
                "lit_chunks",
            )
            .await
            .map_err(|e| anyhow::anyhow!("lit vec store init: {e}"))?,
        );

        let embedder: Arc<dyn EmbeddingService> = Arc::new(OpenRouterEmbeddingService::new(
            cfg.api_key.clone(),
            cfg.embedding_model,
            cfg.embedding_dim,
        )?);

        let executor: Arc<dyn AgentExecutor> =
            Arc::new(OpenRouterExecutor::new(cfg.api_key.clone())?);

        // S2 client honours S2_API_KEY: keyed → live, unkeyed → disabled.
        let s2_client = Arc::new(
            alzina_search::s2_client_for_lit()
                .map_err(|e| anyhow::anyhow!("S2 client build: {e}"))?,
        );
        let s2_keyed = s2_client.is_enabled();

        let reranker = match cfg.rerank_model {
            Some(model) => Some(Reranker {
                svc: Arc::new(openrouter_reranker_with_model(&cfg.api_key, &model)?),
                min_score: cfg.rerank_min_score,
            }),
            None => None,
        };

        // One per-run gateway: arXiv/S2/ar5iv/PDF budgets reset each process.
        let gateway = Arc::new(LitGateway::from_env(s2_keyed));

        Ok(Self {
            lit_pool,
            lit_store,
            embedder,
            executor,
            s2_client,
            reranker,
            gateway,
        })
    }

    /// Whether the S2 lane will fire live calls (true only with `S2_API_KEY`).
    pub fn s2_enabled(&self) -> bool {
        self.s2_client.is_enabled()
    }
}
