//! One-off maintenance: backfill authors/year for arxiv papers whose metadata
//! was clobbered by the pre-`f408a0e` full-text promotion bug.
//!
//! Promotion built an `ArxivResult` with `authors: vec![]` / `published: ""`,
//! and the old `upsert_paper` ON CONFLICT clause overwrote the row — leaving
//! `authors='[]'` and `year=NULL` on every `fulltext_status='indexed'` arxiv
//! row. The upsert is now fixed (never downgrades to empty), but the already
//! clobbered rows must be re-fetched. This example does that:
//!
//! 1. find arxiv rows with empty authors (and a usable arxiv_id),
//! 2. re-fetch metadata from the arxiv Atom API in batches (`id_list`),
//! 3. `UPDATE` authors + year for the rows that resolved.
//!
//! Run: `cargo run -p alzina-search --example backfill_authors [-- DB_PATH]`
//! (default DB path: `memory/literature.db`). Idempotent — re-running only
//! touches rows still missing authors. Hits the public arxiv metadata API.

use std::str::FromStr;

use alzina_search::{ArxivClient, ArxivConfig};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};

const BATCH: usize = 50;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::args().nth(1).unwrap_or_else(|| "memory/literature.db".to_string());
    eprintln!("backfill_authors: opening {db_path}");

    let pool = SqlitePool::connect_with(
        SqliteConnectOptions::from_str(&format!("sqlite:{db_path}"))?.create_if_missing(false),
    )
    .await?;

    // Rows to heal: arxiv source, empty authors, with a usable arxiv_id.
    let rows = sqlx::query(
        "SELECT paper_id, arxiv_id FROM papers \
         WHERE source = 'arxiv' \
           AND (authors IS NULL OR TRIM(authors) IN ('', '[]')) \
           AND arxiv_id IS NOT NULL AND TRIM(arxiv_id) != ''",
    )
    .fetch_all(&pool)
    .await?;

    let targets: Vec<(String, String)> = rows
        .iter()
        .map(|r| (r.get::<String, _>("paper_id"), r.get::<String, _>("arxiv_id")))
        .collect();
    eprintln!("backfill_authors: {} clobbered arxiv rows to heal", targets.len());
    if targets.is_empty() {
        return Ok(());
    }

    let client = ArxivClient::new(ArxivConfig::default())?;

    let (mut healed, mut unresolved) = (0usize, 0usize);
    for (batch_idx, chunk) in targets.chunks(BATCH).enumerate() {
        let ids: Vec<String> = chunk.iter().map(|(_, aid)| aid.clone()).collect();
        let fetched = match client.fetch_by_ids(&ids).await {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  batch {batch_idx}: fetch failed ({e}) — skipping {} ids", ids.len());
                unresolved += ids.len();
                continue;
            }
        };

        // Map arxiv_id -> (authors_json, year). parse_atom_xml strips the version.
        let mut meta: std::collections::HashMap<&str, (String, Option<i32>)> =
            std::collections::HashMap::new();
        for r in &fetched {
            if r.authors.is_empty() {
                continue; // nothing to write
            }
            let authors_json = serde_json::to_string(&r.authors)?;
            let year = r.published.get(0..4).and_then(|y| y.parse::<i32>().ok());
            meta.insert(r.arxiv_id.as_str(), (authors_json, year));
        }

        for (paper_id, arxiv_id) in chunk {
            match meta.get(arxiv_id.as_str()) {
                Some((authors_json, year)) => {
                    sqlx::query("UPDATE papers SET authors = ?, year = COALESCE(?, year) WHERE paper_id = ?")
                        .bind(authors_json)
                        .bind(year)
                        .bind(paper_id)
                        .execute(&pool)
                        .await?;
                    healed += 1;
                }
                None => unresolved += 1,
            }
        }
        eprintln!(
            "  batch {batch_idx}: {} fetched, running totals healed={healed} unresolved={unresolved}",
            fetched.len()
        );
    }

    eprintln!("backfill_authors: DONE — healed={healed}, unresolved={unresolved}");

    // Post-check.
    let remaining: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM papers WHERE source='arxiv' AND (authors IS NULL OR TRIM(authors) IN ('', '[]'))",
    )
    .fetch_one(&pool)
    .await?;
    eprintln!("backfill_authors: arxiv rows still missing authors = {remaining}");
    Ok(())
}
