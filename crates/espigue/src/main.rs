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

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context as _;
use clap::{Parser, Subcommand, ValueEnum};

use espigue::context::{ContextConfig, LitContext};
use espigue::ingest::ingest_dir;
use espigue::openrouter::embeddings::{DEFAULT_DIMENSIONS, DEFAULT_MODEL};
use espigue::openrouter::rerank::DEFAULT_RERANK_MODEL;
use espigue::pipeline::{
    finalize_run, parse_prompt_profile, resume_review, run_review, ReviewOptions, ReviewResult,
    Scope, DEFAULT_MERGER_MODEL, DEFAULT_TOP_K,
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
    #[arg(long, default_value = "anthropic/claude-sonnet-5")]
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
    #[arg(long, default_value = ".", global = true)]
    out: PathBuf,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Ingest local .txt/.md/.pdf documents into the corpus DB.
    Ingest {
        /// Directory to walk (recursively).
        dir: PathBuf,
    },
    /// Resume a failed review run from its stage checkpoints.
    Resume {
        /// The run_id printed by the failed run (also in `synthesis_runs`).
        run_id: String,
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

/// Exit codes for the review path (`exit_for` maps a [`ReviewResult`] onto
/// these; `main` converts to [`ExitCode`]):
/// - `0` — clean success, outputs written.
/// - `1` — empty output; nothing written (also the generic error code).
/// - `2` — degraded but usable output; files written, notice printed.
const EXIT_CLEAN: u8 = 0;
const EXIT_EMPTY: u8 = 1;
const EXIT_DEGRADED: u8 = 2;

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    match run().await {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("Error: {err:?}");
            ExitCode::from(EXIT_EMPTY)
        }
    }
}

async fn run() -> anyhow::Result<u8> {
    let cli = Cli::parse();

    let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
        anyhow::anyhow!("OPENROUTER_API_KEY is not set — required for embeddings (and generation)")
    })?;

    // Resume must use the embedding settings the run was RECORDED with:
    // `LitContext::open` migrates the vec store with `embedding_dim`, so
    // opening with the wrong dim corrupts vec-store expectations. Pre-read
    // them from `synthesis_runs` before building the context.
    let (embedding_model, embedding_dim) = match &cli.command {
        Some(Command::Resume { run_id }) => {
            let (stored_model, stored_dim) = read_stored_embedding(&cli.db, run_id).await?;
            if stored_model != cli.embedding_model || stored_dim != cli.embedding_dim {
                tracing::warn!(
                    stored_model = %stored_model,
                    stored_dim = stored_dim,
                    cli_model = %cli.embedding_model,
                    cli_dim = cli.embedding_dim,
                    "resume: CLI embedding settings differ from the run's recorded \
                     settings — using the stored values"
                );
            }
            (stored_model, stored_dim)
        }
        _ => (cli.embedding_model.clone(), cli.embedding_dim),
    };

    // All paths open the same context (DB + embedder + clients).
    let mut cfg = ContextConfig::new(&cli.db, api_key);
    cfg.embedding_model = embedding_model;
    cfg.embedding_dim = embedding_dim;
    cfg.rerank_model = if cli.no_rerank {
        None
    } else {
        Some(cli.rerank_model.clone())
    };
    cfg.rerank_min_score = cli.rerank_min_score;

    let ctx = LitContext::open(cfg).await?;

    match &cli.command {
        Some(Command::Ingest { dir }) => {
            run_ingest(dir, &ctx).await?;
            Ok(EXIT_CLEAN)
        }
        Some(Command::Resume { run_id }) => run_resume_cmd(run_id, &cli, &ctx).await,
        None => run_review_cmd(&cli, &ctx).await,
    }
}

/// Pre-read a run's recorded embedding model + dim from `synthesis_runs`,
/// via a minimal read-only pool (no vec extension, no migration).
///
/// Bails with a targeted message for each failure mode: missing DB file,
/// pre-checkpointing DB (no `synthesis_runs` table), or unknown run_id.
async fn read_stored_embedding(db: &Path, run_id: &str) -> anyhow::Result<(String, usize)> {
    if !db.exists() {
        anyhow::bail!(
            "database {} does not exist — nothing to resume (resume needs the DB \
             the failed run wrote its checkpoints to; pass it with --db)",
            db.display()
        );
    }
    let opts = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(db)
        .create_if_missing(false)
        .read_only(true);
    let pool = sqlx::SqlitePool::connect_with(opts)
        .await
        .with_context(|| format!("opening {} read-only", db.display()))?;

    let table: Option<(String,)> = sqlx::query_as(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'synthesis_runs'",
    )
    .fetch_optional(&pool)
    .await?;
    if table.is_none() {
        pool.close().await;
        anyhow::bail!(
            "no resumable runs recorded in this database ({}) — it has no \
             synthesis_runs table (created by runs of this espigue version)",
            db.display()
        );
    }

    let row: Option<(String, i64)> =
        sqlx::query_as("SELECT embedding_model, embedding_dim FROM synthesis_runs WHERE run_id = ?")
            .bind(run_id)
            .fetch_optional(&pool)
            .await?;
    pool.close().await;
    let (model, dim) = row.ok_or_else(|| {
        anyhow::anyhow!("unknown run_id '{run_id}' — nothing to resume in {}", db.display())
    })?;
    let dim = usize::try_from(dim)
        .map_err(|_| anyhow::anyhow!("run '{run_id}' records an invalid embedding_dim ({dim})"))?;
    Ok((model, dim))
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

async fn run_review_cmd(cli: &Cli, ctx: &LitContext) -> anyhow::Result<u8> {
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
        embedding_model: cli.embedding_model.clone(),
        embedding_dim: cli.embedding_dim,
    };

    let result = run_review(question, &opts, ctx).await?;
    finish_review(cli, ctx, &result).await
}

async fn run_resume_cmd(run_id: &str, cli: &Cli, ctx: &LitContext) -> anyhow::Result<u8> {
    let result = resume_review(run_id, ctx).await?;
    finish_review(cli, ctx, &result).await
}

/// Shared output tail for fresh and resumed reviews: print the summary,
/// write the outputs (empty synthesis bails — resumable, nothing finalised),
/// then finalise the run only once the files are safely on disk.
async fn finish_review(cli: &Cli, ctx: &LitContext, result: &ReviewResult) -> anyhow::Result<u8> {
    println!("run_id:       {}", result.run_id);
    println!("bibliography: {} sources", result.bib_count);

    if result.synthesis_yaml.is_empty() {
        // Degraded to the point of no output: do NOT clobber a previous good
        // synthesis.yaml/graph.md pair, and fail loudly (exit 1 via main).
        // No finalize here — the run stays resumable via `espigue resume`.
        if !result.notice.is_empty() {
            println!("\n{}", result.notice);
        }
        let reason = if result.notice.is_empty() {
            "engine returned no output"
        } else {
            result.notice.as_str()
        };
        anyhow::bail!("review produced no synthesis — no output files written ({reason})");
    }

    let (yaml_path, graph_path) = write_outputs(&cli.out, result)?;

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

    // Outputs are safely on disk: mark the run complete and delete its
    // checkpoints. This runs for degraded-but-written results too (exit 2) —
    // the files exist, and a user wanting a better result reruns fresh
    // rather than resuming into the same degraded state. Only the empty-bail
    // path above (and a write_outputs failure) leaves the run resumable.
    // Finalisation failure is warn-and-continue: the outputs are already
    // safe, and stale checkpoints are harmless.
    if let Err(err) = finalize_run(ctx, &result.run_id).await {
        tracing::warn!(
            run_id = %result.run_id,
            error = %err,
            "failed to finalise run — outputs are written; checkpoints remain"
        );
    }

    Ok(exit_for(result))
}

/// Write `synthesis.yaml` + `graph.md` into `out`, returning their paths.
///
/// Refuses to write anything when `synthesis_yaml` is empty so a degraded run
/// never overwrites a previous good pair.
fn write_outputs(out: &Path, result: &ReviewResult) -> anyhow::Result<(PathBuf, PathBuf)> {
    if result.synthesis_yaml.is_empty() {
        anyhow::bail!("refusing to write empty synthesis output to {}", out.display());
    }
    std::fs::create_dir_all(out)?;
    let yaml_path = out.join("synthesis.yaml");
    let graph_path = out.join("graph.md");
    // yaml first: a crash between the two leaves the more valuable artifact in place.
    write_atomic(&yaml_path, result.synthesis_yaml.as_bytes())?;
    write_atomic(&graph_path, result.graph_markdown.as_bytes())?;
    Ok((yaml_path, graph_path))
}

/// Write `contents` to `path` via a `.tmp` file in the same directory plus
/// rename, so `path` is only ever missing or complete — never truncated.
/// Rename within one directory is atomic on POSIX.
fn write_atomic(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let write_result = std::fs::write(&tmp, contents)
        .and_then(|()| std::fs::rename(&tmp, path));
    if let Err(err) = write_result {
        // Best-effort cleanup; the original error is what matters.
        let _ = std::fs::remove_file(&tmp);
        return Err(err).with_context(|| format!("writing {}", path.display()));
    }
    Ok(())
}

/// Pure exit-code decision for a completed review:
/// empty output → 1, degraded-but-usable → 2, clean → 0.
fn exit_for(result: &ReviewResult) -> u8 {
    if result.synthesis_yaml.is_empty() {
        EXIT_EMPTY
    } else if result.degraded {
        EXIT_DEGRADED
    } else {
        EXIT_CLEAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stored scope string must parse back through the CLI's own parser
    /// (clap `ValueEnum`) to the same [`Scope`] — the resume path re-parses it.
    #[test]
    fn scope_str_roundtrips_through_clap_value_enum() {
        for scope in [Scope::CorpusOnly, Scope::CorpusPlusWeb] {
            let parsed: Scope =
                <ScopeArg as ValueEnum>::from_str(espigue::pipeline::scope_str(scope), false)
                    .expect("scope_str must be a valid clap value")
                    .into();
            assert_eq!(parsed, scope);
        }
    }

    #[test]
    fn cli_parses_resume_subcommand() {
        let cli = Cli::try_parse_from(["espigue", "resume", "abc123"]).unwrap();
        match cli.command {
            Some(Command::Resume { ref run_id }) => assert_eq!(run_id, "abc123"),
            other => panic!("expected Resume, got {other:?}"),
        }
        assert!(cli.question.is_none());
    }

    #[test]
    fn cli_parses_resume_with_global_out() {
        let cli =
            Cli::try_parse_from(["espigue", "resume", "abc123", "--out", "somewhere"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Resume { .. })));
        assert_eq!(cli.out, PathBuf::from("somewhere"));
    }

    #[test]
    fn cli_still_parses_bare_question() {
        let cli = Cli::try_parse_from(["espigue", "test-time compute scaling"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.question.as_deref(), Some("test-time compute scaling"));
    }

    fn result(synthesis_yaml: &str, degraded: bool) -> ReviewResult {
        ReviewResult {
            synthesis_yaml: synthesis_yaml.to_string(),
            graph_markdown: "# graph".to_string(),
            run_id: "test-run".to_string(),
            bib_count: 0,
            narrative: String::new(),
            degraded,
            notice: if degraded {
                "degraded: test notice".to_string()
            } else {
                String::new()
            },
        }
    }

    #[test]
    fn exit_for_empty_output_is_1() {
        assert_eq!(exit_for(&result("", true)), EXIT_EMPTY);
        // Empty output wins even if the degraded flag was somehow left unset.
        assert_eq!(exit_for(&result("", false)), EXIT_EMPTY);
    }

    #[test]
    fn exit_for_degraded_with_output_is_2() {
        assert_eq!(exit_for(&result("claims: []", true)), EXIT_DEGRADED);
    }

    #[test]
    fn exit_for_clean_is_0() {
        assert_eq!(exit_for(&result("claims: []", false)), EXIT_CLEAN);
    }

    #[test]
    fn write_outputs_refuses_empty_and_preserves_previous_files() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("synthesis.yaml");
        let graph_path = dir.path().join("graph.md");
        std::fs::write(&yaml_path, "previous good yaml").unwrap();
        std::fs::write(&graph_path, "previous good graph").unwrap();

        let err = write_outputs(dir.path(), &result("", true));
        assert!(err.is_err(), "empty synthesis must not be written");

        // The previous good pair is untouched.
        assert_eq!(
            std::fs::read_to_string(&yaml_path).unwrap(),
            "previous good yaml"
        );
        assert_eq!(
            std::fs::read_to_string(&graph_path).unwrap(),
            "previous good graph"
        );
    }

    #[test]
    fn write_outputs_writes_pair_on_nonempty() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("nested"); // also covers create_dir_all
        let (yaml_path, graph_path) = write_outputs(&out, &result("claims: []", false)).unwrap();
        assert_eq!(
            std::fs::read_to_string(&yaml_path).unwrap(),
            "claims: []"
        );
        assert_eq!(std::fs::read_to_string(&graph_path).unwrap(), "# graph");
    }

    #[test]
    fn write_outputs_leaves_no_tmp_files() {
        let dir = tempfile::tempdir().unwrap();
        write_outputs(dir.path(), &result("claims: []", false)).unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "tmp files left behind: {leftovers:?}");
    }
}
