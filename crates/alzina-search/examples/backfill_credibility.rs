//! One-off maintenance: backfill source-credibility signals
//! (citation_count, influential_citation_count, venue) from Semantic Scholar.
//!
//! These signals feed the per-source authenticity tier. The gap they close:
//! arxiv rows carry NO citation data (the Atom feed has none — `citation_count`
//! is NULL on every arxiv paper), and `influential_citation_count` + `venue` are
//! new columns that are NULL on every existing row. S2 has all three and resolves
//! both arxiv ids (`ARXIV:<id>`) and S2 ids in one batch endpoint.
//!
//! 1. find rows that have not been credibility-attempted (NULL influential),
//!    with a usable S2 lookup id (arxiv_id for arxiv rows, s2_paper_id for s2),
//! 2. batch-fetch from S2 `/paper/batch` (resolves ARXIV: + S2 ids),
//! 3. `update_paper_credibility` writes citation_count + influential + venue,
//!    preserve-on-unknown (a NULL argument never clobbers an existing value).
//!
//! Run (keyed S2 required — refuses to run anonymously, per the 429-storm
//! lesson in reference_s2_keyed_daemon):
//!
//! ```text
//! S2_LIVE_ENABLED=true S2_API_KEY=<key> \
//!   cargo run -p alzina-search --example backfill_credibility [-- DB_PATH [LIMIT]]
//! ```
//!
//! `DB_PATH` defaults to `memory/literature.db`. `LIMIT` (optional) caps the
//! number of rows processed — use a small value first for a trial run.
//! Idempotent: re-running only touches rows still missing the influential count
//! (rows S2 cannot resolve stay NULL and are retried; resolved rows settle).

use std::str::FromStr;
use std::time::Duration;

use alzina_search::{update_paper_credibility, S2Client};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};

/// S2 `/paper/batch` accepts up to 500 ids; 100 keeps each response small.
const BATCH: usize = 100;
/// 1.1s is S2's STRICT fastest (cumulative across all endpoints). Pace ABOVE it
/// with margin — never sit on the edge. See reference_s2_keyed_daemon.
const PACE: Duration = Duration::from_millis(1500);

/// Outcome of one batch fetch.
enum Batch {
    /// Per-id results (None = not found in S2).
    Ok(Vec<Option<alzina_search::S2PaperFull>>),
    /// A non-throttle error — skip this batch, keep going.
    Skip,
    /// Sustained 429 — the caller must ABORT the whole run. Never retry into a
    /// throttle: that stacks 429s and risks the key (Sam's hard rule). Rows are
    /// idempotently retriable another day.
    Abort,
}

/// Fetch one batch. One polite wait honouring Retry-After on a 429; if still
/// throttled, signal Abort (do NOT keep hammering).
async fn fetch_batch(client: &S2Client, ids: &[String], batch_idx: usize) -> Batch {
    match client.get_papers_batch(ids).await {
        Ok(f) => Batch::Ok(f),
        Err(e) if e.status == Some(429) => {
            let wait = e.retry_after.unwrap_or(Duration::from_secs(10)).min(Duration::from_secs(30));
            eprintln!("  batch {batch_idx}: 429 — one polite wait {wait:?}");
            tokio::time::sleep(wait).await;
            match client.get_papers_batch(ids).await {
                Ok(f) => Batch::Ok(f),
                Err(e2) if e2.status == Some(429) => {
                    eprintln!("  batch {batch_idx}: still 429 — ABORTING to protect the key");
                    Batch::Abort
                }
                Err(e2) => {
                    eprintln!("  batch {batch_idx}: fetch failed ({}) — skipping", e2.message);
                    Batch::Skip
                }
            }
        }
        Err(e) => {
            eprintln!("  batch {batch_idx}: fetch failed ({}) — skipping", e.message);
            Batch::Skip
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::args().nth(1).unwrap_or_else(|| "memory/literature.db".to_string());
    let limit: Option<usize> = std::env::args().nth(2).and_then(|s| s.parse().ok());
    eprintln!("backfill_credibility: opening {db_path}");

    // Refuse to run anonymously — an unkeyed S2 client 429-storms on the first
    // call (reference_s2_keyed_daemon). Presence-only check; never log the key.
    if std::env::var("S2_API_KEY").ok().filter(|s| !s.is_empty()).is_none() {
        return Err("S2_API_KEY must be set (keyed S2 required — refusing to run anonymously)".into());
    }
    let client = S2Client::from_env()?;
    if !client.is_enabled() {
        return Err("S2_LIVE_ENABLED must be true for this backfill".into());
    }

    let pool = SqlitePool::connect_with(
        SqliteConnectOptions::from_str(&format!("sqlite:{db_path}"))?.create_if_missing(false),
    )
    .await?;

    // Ensure the credibility columns exist on disk. migrate() is idempotent
    // (IF NOT EXISTS + guarded ALTERs); on an already-populated DB it only adds
    // the new influential_citation_count + venue columns.
    alzina_search::lit_migrate(&pool, 1024).await?;

    // Rows to attempt: not yet credibility-attempted (NULL influential), with a
    // usable S2 lookup id. arxiv → arxiv_id (resolves to ARXIV:<id>); s2 →
    // s2_paper_id (the raw S2 hash). Rows lacking a lookup id are skipped.
    let rows = sqlx::query(
        "SELECT paper_id, \
            CASE WHEN source = 'arxiv' THEN arxiv_id ELSE s2_paper_id END AS lookup_id \
         FROM papers \
         WHERE influential_citation_count IS NULL \
           AND CASE WHEN source = 'arxiv' THEN arxiv_id ELSE s2_paper_id END IS NOT NULL \
           AND TRIM(CASE WHEN source = 'arxiv' THEN arxiv_id ELSE s2_paper_id END) != ''",
    )
    .fetch_all(&pool)
    .await?;

    let mut targets: Vec<(String, String)> = rows
        .iter()
        .map(|r| (r.get::<String, _>("paper_id"), r.get::<String, _>("lookup_id")))
        .collect();
    if let Some(n) = limit {
        targets.truncate(n);
    }
    eprintln!("backfill_credibility: {} rows to attempt{}", targets.len(),
        limit.map(|n| format!(" (limited to {n})")).unwrap_or_default());
    if targets.is_empty() {
        return Ok(());
    }

    let (mut healed, mut unresolved, mut errored) = (0usize, 0usize, 0usize);
    for (batch_idx, chunk) in targets.chunks(BATCH).enumerate() {
        let ids: Vec<String> = chunk.iter().map(|(_, lid)| lid.clone()).collect();

        let fetched = match fetch_batch(&client, &ids, batch_idx).await {
            Batch::Ok(f) => f,
            Batch::Skip => {
                errored += ids.len();
                tokio::time::sleep(PACE).await;
                continue;
            }
            Batch::Abort => {
                eprintln!(
                    "backfill_credibility: ABORTED at batch {batch_idx} to protect the S2 key. \
                     Remaining rows are NULL and idempotently retriable later."
                );
                break;
            }
        };

        // get_papers_batch returns results in input order; None = not found.
        for ((paper_id, _lookup), result) in chunk.iter().zip(fetched.iter()) {
            match result {
                Some(p) => {
                    let citation = i32::try_from(p.citation_count).ok();
                    // Backfill writes the real S2 answer, including a genuine 0
                    // (a looked-up 0 is information, not "unknown") so the row
                    // settles and is not retried.
                    let influential = i32::try_from(p.influential_citation_count).ok().or(Some(0));
                    update_paper_credibility(&pool, paper_id, citation, influential, p.venue.as_deref())
                        .await?;
                    healed += 1;
                }
                None => unresolved += 1,
            }
        }
        eprintln!(
            "  batch {batch_idx}: {} resolved, running totals healed={healed} unresolved={unresolved} errored={errored}",
            fetched.iter().filter(|r| r.is_some()).count()
        );
        tokio::time::sleep(PACE).await;
    }

    eprintln!("backfill_credibility: DONE — healed={healed}, unresolved={unresolved}, errored={errored}");

    // Post-check: coverage of the three signals across the corpus.
    let (cit, infl, ven, total): (i64, i64, i64, i64) = {
        let r = sqlx::query(
            "SELECT \
                SUM(CASE WHEN citation_count IS NOT NULL THEN 1 ELSE 0 END), \
                SUM(CASE WHEN influential_citation_count IS NOT NULL THEN 1 ELSE 0 END), \
                SUM(CASE WHEN venue IS NOT NULL AND TRIM(venue) != '' THEN 1 ELSE 0 END), \
                COUNT(*) \
             FROM papers",
        )
        .fetch_one(&pool)
        .await?;
        (r.get(0), r.get(1), r.get(2), r.get(3))
    };
    eprintln!(
        "backfill_credibility: coverage — citation_count {cit}/{total}, influential {infl}/{total}, venue {ven}/{total}"
    );
    Ok(())
}
