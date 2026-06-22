//! OpenRouter clients — one `OPENROUTER_API_KEY` for generation, embeddings,
//! and reranking.
//!
//! OpenRouter exposes an OpenAI-compatible API:
//! - `POST /chat/completions` → [`executor::OpenRouterExecutor`]
//! - `POST /embeddings`       → [`embeddings::OpenRouterEmbeddingService`]
//! - `POST /rerank` (Cohere shape) → [`rerank`] (re-uses `search`'s client)

pub mod embeddings;
pub mod executor;
pub mod rerank;

/// Default OpenRouter API base. Tests point the clients at a `wiremock` server.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
