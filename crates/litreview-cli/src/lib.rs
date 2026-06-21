//! `litreview` — standalone literature-review synthesis.
//!
//! A thin standalone surface over the Alzina TTD synthesis engine, backed by
//! OpenRouter for generation, embeddings, and reranking. No daemon, no sidecar,
//! no governance.

pub mod context;
pub mod ingest;
pub mod openrouter;
pub mod pipeline;
pub mod seed;

pub use context::{ContextConfig, LitContext, Reranker};
pub use ingest::{ingest_dir, IngestStats};
pub use pipeline::{run_review, ReviewOptions, ReviewResult, Scope};
