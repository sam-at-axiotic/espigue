//! Phase-1 live smoke: hit the three real OpenRouter endpoints through the
//! actual clients. Validates the wire contracts mocks cannot: chat response
//! shape, that `text-embedding-3-small` accepts `dimensions` and returns 1536,
//! and that rerank returns `relevance_score`.
//!
//! Run: `export $(grep '^OPENROUTER_API_KEY=' .env) && \
//!       cargo run -p cli --example live_smoke`

use base::identity::AgentId;
use base::search::{EmbeddingService, EmbeddingTask};
use orchestration::AgentExecutor;

use cli::openrouter::embeddings::{
    OpenRouterEmbeddingService, DEFAULT_DIMENSIONS, DEFAULT_MODEL,
};
use cli::openrouter::executor::OpenRouterExecutor;
use cli::openrouter::rerank::openrouter_reranker;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let key = std::env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set in env"))?;

    // 1. Chat completions.
    let exec = OpenRouterExecutor::new(&key)?;
    let chat = exec
        .execute(
            &AgentId::new("smoke"),
            "Reply with exactly one word: pong",
            "google/gemini-2.5-flash",
            "smoke",
        )
        .await?;
    println!("[chat]   ok -> {:?}", chat.trim());

    // 2. Embeddings — must accept `dimensions` and return that many floats.
    let emb = OpenRouterEmbeddingService::new(&key, DEFAULT_MODEL, DEFAULT_DIMENSIONS)?;
    let v = emb.embed("permafrost thaw releases methane", EmbeddingTask::Query).await?;
    println!(
        "[embed]  ok -> {} dims (expected {}) via {}",
        v.len(),
        DEFAULT_DIMENSIONS,
        DEFAULT_MODEL
    );
    anyhow::ensure!(v.len() == DEFAULT_DIMENSIONS, "embedding dim mismatch");

    // 3. Rerank — Cohere shape; proves `relevance_score` parses.
    let rr = openrouter_reranker(&key)?;
    let ranked = rr
        .rerank(
            "climate feedback from methane",
            &[
                "Permafrost thaw releases methane under warming.".into(),
                "The local football team won the cup final.".into(),
            ],
        )
        .await?;
    println!("[rerank] ok -> {:?}", ranked);
    anyhow::ensure!(!ranked.is_empty(), "rerank returned no results");
    anyhow::ensure!(ranked[0].index == 0, "rerank should rank the on-topic doc first");

    println!("\nALL THREE OPENROUTER CONTRACTS OK");
    Ok(())
}
