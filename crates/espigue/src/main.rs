//! `espigue` — standalone literature-review synthesis CLI.
//!
//! Topic in, cited literature review out. One `OPENROUTER_API_KEY` drives
//! generation, embeddings, and reranking; `S2_API_KEY` (optional) enables the
//! Semantic Scholar lane. arXiv needs no key.
//!
//! ```text
//! export OPENROUTER_API_KEY=sk-or-...
//! espigue "test-time compute scaling"        # → synthesis.yaml + graph.md
//! espigue ingest ./papers/                    # index local docs
//! espigue --scope corpus-only "my question"   # cite only local docs
//! ```

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use espigue::context::{ContextConfig, LitContext};
use espigue::ingest::ingest_dir;
use espigue::openrouter::embeddings::{DEFAULT_DIMENSIONS, DEFAULT_MODEL};
use espigue::openrouter::rerank::DEFAULT_RERANK_MODEL;
use espigue::pipeline::{
    parse_prompt_profile, run_review, ReviewOptions, Scope, DEFAULT_MERGER_MODEL, DEFAULT_TOP_K,
};

/// Standalone literature-review synthesis over OpenRouter + arXiv (+ optional S2).
#[derive(Parser, Debug)]
#[command(name = "espigue", version, about)]
struct Cli {
    /// The research question to synthesise a review for. Omit when using a
    /// subcommand (e.g. `espigue ingest ./papers/`).
    question: Option<String>,

    /// Subcommand (e.g. `ingest`). When absent, runs a review of QUESTION.
    #[command(subcommand)]
    command: Option<Command>,

    // ── Shared (review + ingest both open the same DB / embedder) ──────────
    /// Literature DB path (created if missing).
    #[arg(long, default_value = "espigue.db", global = true)]
    db: PathBuf,

    /// Embedding model slug.
    #[arg(long, default_value = DEFAULT_MODEL, global = true)]
    embedding_model: String,

    /// Embedding dimension.
    #[arg(long, default_value_t = DEFAULT_DIMENSIONS, global = true)]
    embedding_dim: usize,

    // ── Review-only ────────────────────────────────────────────────────────
    /// Sources retrieved per lane (clamped to 50).
    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    /// Prompt/schema profile: v1/delphi, v2/lit-review, or v3/lit-review-long.
    #[arg(long)]
    profile: Option<String>,

    /// Generation model slug for the TTD stages.
    #[arg(long, default_value = "google/gemini-2.5-flash")]
    model: String,

    /// Stage-2 merger model slug for v2/v3 profiles (Opus, OpenRouter-shaped).
    #[arg(long, default_value = DEFAULT_MERGER_MODEL)]
    merger_model: String,

    /// Rerank model slug (use --no-rerank to disable reranking).
    #[arg(long, default_value = DEFAULT_RERANK_MODEL)]
    rerank_model: String,

    /// Disable cross-encoder reranking (keep pure RRF order).
    #[arg(long)]
    no_rerank: bool,

    /// Cross-encoder drop floor (hits below this are dropped as off-topic).
    #[arg(long, default_value_t = 0.0)]
    rerank_min_score: f32,

    /// Retrieval scope: corpus-only (local docs only) or corpus+web.
    #[arg(long, value_enum, default_value_t = ScopeArg::CorpusPlusWeb)]
    scope: ScopeArg,

    /// Seed papers to build the panel from (comma-separated arXiv ids / DOIs /
    /// S2 ids). Skips Stage-0, fusion, and the topicality gate; gap-fill still
    /// honours --scope. DOI/S2 ids need S2_API_KEY; arXiv ids need no key.
    #[arg(long, value_delimiter = ',')]
    seed_papers: Vec<String>,

    /// Output directory for synthesis.yaml + graph.md.
    #[arg(long, default_value = ".")]
    out: PathBuf,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Ingest local .txt/.md/.pdf documents into the corpus DB.
    Ingest {
        /// Directory to walk (recursively).
        dir: PathBuf,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ScopeArg {
    #[value(name = "corpus-only")]
    CorpusOnly,
    #[value(name = "corpus+web", alias = "corpus-plus-web")]
    CorpusPlusWeb,
}

impl From<ScopeArg> for Scope {
    fn from(s: ScopeArg) -> Self {
        match s {
            ScopeArg::CorpusOnly => Scope::CorpusOnly,
            ScopeArg::CorpusPlusWeb => Scope::CorpusPlusWeb,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
        anyhow::anyhow!("OPENROUTER_API_KEY is not set — required for embeddings (and generation)")
    })?;

    // Both paths open the same context (DB + embedder + clients).
    let mut cfg = ContextConfig::new(&cli.db, api_key);
    cfg.embedding_model = cli.embedding_model.clone();
    cfg.embedding_dim = cli.embedding_dim;
    cfg.rerank_model = if cli.no_rerank {
        None
    } else {
        Some(cli.rerank_model.clone())
    };
    cfg.rerank_min_score = cli.rerank_min_score;

    let ctx = LitContext::open(cfg).await?;

    match &cli.command {
        Some(Command::Ingest { dir }) => run_ingest(dir, &ctx).await,
        None => run_review_cmd(&cli, &ctx).await,
    }
}

async fn run_ingest(dir: &std::path::Path, ctx: &LitContext) -> anyhow::Result<()> {
    let stats = ingest_dir(dir, ctx).await?;
    println!(
        "ingested {} of {} file(s) — {} chunk(s) written",
        stats.files_ingested, stats.files_seen, stats.chunks_written
    );
    if stats.pdftotext_missing {
        eprintln!(
            "espigue: pdftotext not found — PDFs were skipped. Install poppler:\n\
             \tmacOS:  brew install poppler\n\
             \tDebian: apt install poppler-utils"
        );
    }
    if !stats.skipped.is_empty() {
        eprintln!("skipped {} file(s):", stats.skipped.len());
        for (path, reason) in &stats.skipped {
            eprintln!("  {} — {}", path.display(), reason);
        }
    }
    Ok(())
}

async fn run_review_cmd(cli: &Cli, ctx: &LitContext) -> anyhow::Result<()> {
    let question = cli.question.as_deref().ok_or_else(|| {
        anyhow::anyhow!("no question given (usage: espigue \"your question\", or `espigue ingest <dir>`)")
    })?;

    let profile = parse_prompt_profile(cli.profile.as_deref()).map_err(|e| anyhow::anyhow!(e))?;
    let scope: Scope = cli.scope.into();

    if scope == Scope::CorpusPlusWeb && !ctx.s2_enabled() {
        eprintln!(
            "espigue: S2_API_KEY not set — Semantic Scholar lane disabled (arXiv + local only)."
        );
    }

    if !cli.seed_papers.is_empty() && scope == Scope::CorpusOnly {
        eprintln!(
            "espigue: --seed-papers with --scope corpus-only — seeds will be fetched, but \
             gap-fill stays local-only."
        );
    }

    let opts = ReviewOptions {
        top_k: cli.top_k,
        profile,
        model: cli.model.clone(),
        merger_model: Some(cli.merger_model.clone()),
        scope,
        seed_papers: cli.seed_papers.clone(),
    };

    let result = run_review(question, &opts, ctx).await?;

    std::fs::create_dir_all(&cli.out)?;
    let yaml_path = cli.out.join("synthesis.yaml");
    let graph_path = cli.out.join("graph.md");
    std::fs::write(&yaml_path, &result.synthesis_yaml)?;
    std::fs::write(&graph_path, &result.graph_markdown)?;

    println!("run_id:       {}", result.run_id);
    println!("bibliography: {} sources", result.bib_count);
    println!("synthesis:    {}", yaml_path.display());
    println!("graph:        {}", graph_path.display());
    if result.degraded {
        println!("\n{}", result.notice);
    } else if !result.notice.is_empty() {
        println!("\nnote: {}", result.notice);
    }
    if !result.narrative.is_empty() {
        println!("\n=== NARRATIVE ===\n{}", result.narrative);
    }

    Ok(())
}
