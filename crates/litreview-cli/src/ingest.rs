//! Bring-your-own-corpora ingest (Phase 3).
//!
//! Walks a folder and indexes local `.txt` / `.md` / `.pdf` files into the
//! literature DB so a review can cite them. Each file becomes a synthetic
//! `local:{relative-path}` paper with `source_type="local"`; its chunks land in
//! `lit_vec0` / `lit_chunks`, so the internal kNN lane surfaces them under both
//! retrieval policies.
//!
//! PDF extraction needs `pdftotext` (poppler) on PATH. When it is absent the
//! ingest still indexes text/markdown and prints an actionable install hint —
//! the one external runtime dependency, gated loudly.

use std::path::{Path, PathBuf};

use alzina_search::{
    chunk_plain_text, persist_chunks_for_paper, pdftotext_extract, set_fulltext_status,
    upsert_paper, LitChunkConfig, PdfFetchConfig,
};

use crate::context::LitContext;

/// Outcome of an ingest run.
#[derive(Debug, Default)]
pub struct IngestStats {
    /// Candidate files seen (matching a known extension).
    pub files_seen: usize,
    /// Files successfully indexed (papers row + chunks written).
    pub files_ingested: usize,
    /// Total chunks embedded and persisted.
    pub chunks_written: usize,
    /// Files skipped, with a reason each.
    pub skipped: Vec<(PathBuf, String)>,
    /// True when `pdftotext` was not found — PDFs were skipped.
    pub pdftotext_missing: bool,
}

/// Recognised file kinds.
enum Kind {
    Text,
    Pdf,
}

fn classify(path: &Path) -> Option<Kind> {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()) {
        Some(ref e) if e == "txt" || e == "md" || e == "markdown" || e == "text" => Some(Kind::Text),
        Some(ref e) if e == "pdf" => Some(Kind::Pdf),
        _ => None,
    }
}

/// Probe whether `pdftotext` is runnable. Spawns `<path> -v`; any spawn success
/// (poppler prints its version to stderr and exits 0) means it is present.
fn pdftotext_available(cfg: &PdfFetchConfig) -> bool {
    std::process::Command::new(&cfg.pdftotext_path)
        .arg("-v")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// Collect candidate files under `dir`, recursing into subdirectories.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if classify(&path).is_some() {
            out.push(path);
        }
    }
    Ok(())
}

/// Walk `dir` and ingest every recognised file into `ctx`'s literature DB.
///
/// Never aborts the whole run on one bad file: per-file failures are collected
/// into `IngestStats.skipped` and the walk continues (loud-degrade).
pub async fn ingest_dir(dir: &Path, ctx: &LitContext) -> anyhow::Result<IngestStats> {
    if !dir.is_dir() {
        anyhow::bail!("ingest path is not a directory: {}", dir.display());
    }

    let pdf_cfg = PdfFetchConfig::from_env();
    let chunk_cfg = LitChunkConfig::default();
    let pdf_ok = pdftotext_available(&pdf_cfg);

    let mut files = Vec::new();
    collect_files(dir, &mut files)?;

    let mut stats = IngestStats {
        pdftotext_missing: !pdf_ok,
        ..Default::default()
    };

    for path in files {
        stats.files_seen += 1;
        let kind = match classify(&path) {
            Some(k) => k,
            None => continue,
        };

        // Stable, readable synthetic id from the path relative to the ingest dir.
        let rel = path.strip_prefix(dir).unwrap_or(&path);
        let paper_id = format!("local:{}", rel.to_string_lossy());
        let title = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("untitled")
            .to_string();

        // Extract text.
        let text = match kind {
            Kind::Text => match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => {
                    stats.skipped.push((path.clone(), format!("read failed: {e}")));
                    continue;
                }
            },
            Kind::Pdf => {
                if !pdf_ok {
                    stats
                        .skipped
                        .push((path.clone(), "pdftotext not found — PDF skipped".into()));
                    continue;
                }
                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(e) => {
                        stats.skipped.push((path.clone(), format!("read failed: {e}")));
                        continue;
                    }
                };
                match pdftotext_extract(&bytes, &pdf_cfg).await {
                    Ok(t) => t,
                    Err(e) => {
                        stats
                            .skipped
                            .push((path.clone(), format!("pdftotext failed: {e}")));
                        continue;
                    }
                }
            }
        };

        let chunks = chunk_plain_text(&text, &chunk_cfg);
        if chunks.is_empty() {
            stats
                .skipped
                .push((path.clone(), "no usable text after chunking".into()));
            continue;
        }

        // Write the papers row (source="local", url = on-disk path).
        let fetched_at = chrono::Utc::now().to_rfc3339();
        let url = path.to_string_lossy();
        if let Err(e) = upsert_paper(
            ctx.lit_pool.as_ref(),
            &paper_id,
            "local",
            None,
            None,
            None,
            &title,
            None,
            &url,
            None,
            "[]",
            None,
            &fetched_at,
        )
        .await
        {
            stats.skipped.push((path.clone(), format!("upsert_paper failed: {e}")));
            continue;
        }

        // Embed + persist chunks.
        let n_chunks = chunks.len();
        if let Err(e) = persist_chunks_for_paper(
            ctx.lit_pool.as_ref(),
            ctx.lit_store.as_ref(),
            ctx.embedder.as_ref(),
            &paper_id,
            "local",
            &title,
            None,
            &chunks,
        )
        .await
        {
            stats
                .skipped
                .push((path.clone(), format!("persist chunks failed: {e}")));
            continue;
        }

        // Local docs are full text by definition — mark indexed so the coverage
        // accounting in run_review counts them, not flags abstract-only.
        if let Err(e) = set_fulltext_status(ctx.lit_pool.as_ref(), &paper_id, "indexed").await {
            tracing::warn!(
                paper_id = %paper_id,
                error = %e,
                "ingest: set_fulltext_status(indexed) failed — coverage may under-count"
            );
        }

        stats.files_ingested += 1;
        stats.chunks_written += n_chunks;
        tracing::info!(
            paper_id = %paper_id,
            chunks = n_chunks,
            "ingest: indexed local document"
        );
    }

    Ok(stats)
}
