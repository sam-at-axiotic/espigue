//! ADAPT-01 unit tests for the source → panel adapter.
//!
//! These tests build an in-memory SQLite pool directly (lit_schema::in_memory_lit_pool
//! is #[cfg(test)]-gated in alzina-search and NOT importable cross-crate).

use super::build_panel;
use alzina_search::lit_schema::{migrate, upsert_paper};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::str::FromStr;

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Build a minimal FusedHit for test use.
fn make_fused_hit(paper_id: &str) -> alzina_search::lit_fusion::FusedHit {
    alzina_search::lit_fusion::FusedHit {
        source_type: "arxiv".to_string(),
        source_id: paper_id.to_string(),
        title: format!("Test paper {paper_id}"),
        section: None,
        content: format!("Fused content for {paper_id}"),
        content_preview: format!("Preview for {paper_id}"),
        relevance: 1.0,
    }
}

/// Build an in-memory SqlitePool with the lit schema migrated.
async fn make_pool() -> SqlitePool {
    let opts = SqliteConnectOptions::from_str("sqlite::memory:")
        .expect("valid sqlite url")
        .create_if_missing(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(opts)
        .await
        .expect("in-memory pool");

    migrate(&pool, 1024).await.expect("migrate");
    pool
}

/// Insert a paper row and a lit_chunks row for testing.
async fn insert_lit_chunk(
    pool: &SqlitePool,
    paper_id: &str,
    idx: i64,
    section: Option<&str>,
    content: &str,
) {
    // Ensure the papers row exists first (upsert_paper uses ON CONFLICT DO UPDATE)
    upsert_paper(
        pool,
        paper_id,
        "arxiv",
        Some(&paper_id.trim_start_matches("arxiv:")),
        None,
        None,
        &format!("Test paper {paper_id}"),
        None,
        "https://example.com",
        Some(2024),
        r#"["Test Author"]"#,
        None,
        "2024-01-01T00:00:00Z",
    )
    .await
    .expect("upsert_paper");

    sqlx::query(
        "INSERT INTO lit_chunks \
         (source_type, source_id, chunk_index, content_preview, indexed_at, paper_id, section, content) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind("arxiv")
    .bind(paper_id)
    .bind(idx)
    .bind(content.chars().take(200).collect::<String>())
    .bind("2024-01-01T00:00:00Z")
    .bind(paper_id)
    .bind(section)
    .bind(content)
    .execute(pool)
    .await
    .expect("insert lit_chunk");
}

// ── ADAPT-01 tests ────────────────────────────────────────────────────────────

/// build_panel over a FusedHit list with N distinct paper_ids returns
/// exactly N ExpertResponses, one per distinct paper, expert_id == paper_id.
#[tokio::test]
async fn one_response_per_paper() {
    let pool = make_pool().await;
    let hits = vec![
        make_fused_hit("arxiv:2401.00001"),
        make_fused_hit("arxiv:2401.00002"),
        make_fused_hit("arxiv:2401.00003"),
    ];

    let panel = build_panel(hits.clone(), &pool).await.unwrap();

    assert_eq!(panel.len(), 3, "must return one ExpertResponse per distinct paper");

    // Check expert_id values match paper_ids
    let ids: Vec<&str> = panel.iter().map(|r| r.expert_id.as_str()).collect();
    assert!(ids.contains(&"arxiv:2401.00001"));
    assert!(ids.contains(&"arxiv:2401.00002"));
    assert!(ids.contains(&"arxiv:2401.00003"));
}

/// A paper with lit_chunks rows at chunk_index 0, 1, 2 (inserted out of order)
/// yields prose whose chunk texts appear in 0, 1, 2 order.
#[tokio::test]
async fn prose_assembles_chunks_in_order() {
    let pool = make_pool().await;
    let paper_id = "arxiv:2401.00010";

    // Insert out of order: 2, 0, 1
    insert_lit_chunk(&pool, paper_id, 2, None, "chunk two").await;
    insert_lit_chunk(&pool, paper_id, 0, None, "chunk zero").await;
    insert_lit_chunk(&pool, paper_id, 1, None, "chunk one").await;

    let hits = vec![make_fused_hit(paper_id)];
    let panel = build_panel(hits, &pool).await.unwrap();

    assert_eq!(panel.len(), 1);
    let prose = &panel[0].prose;

    let pos_zero = prose.find("chunk zero").expect("chunk zero missing");
    let pos_one = prose.find("chunk one").expect("chunk one missing");
    let pos_two = prose.find("chunk two").expect("chunk two missing");

    assert!(
        pos_zero < pos_one && pos_one < pos_two,
        "chunks must appear in chunk_index order 0→1→2; prose: {prose:?}"
    );
}

/// A chunk with section "Methods" produces a "## Methods" marker
/// immediately before that chunk's text in the prose body.
#[tokio::test]
async fn section_headings_are_inline_markers() {
    let pool = make_pool().await;
    let paper_id = "arxiv:2401.00020";

    insert_lit_chunk(&pool, paper_id, 0, Some("Introduction"), "intro text").await;
    insert_lit_chunk(&pool, paper_id, 1, Some("Methods"), "methods text").await;
    insert_lit_chunk(&pool, paper_id, 2, None, "no-section text").await;

    let hits = vec![make_fused_hit(paper_id)];
    let panel = build_panel(hits, &pool).await.unwrap();

    assert_eq!(panel.len(), 1);
    let prose = &panel[0].prose;

    assert!(
        prose.contains("## Introduction"),
        "section 'Introduction' must appear as '## Introduction'; prose: {prose:?}"
    );
    assert!(
        prose.contains("## Methods"),
        "section 'Methods' must appear as '## Methods'; prose: {prose:?}"
    );
    // Chunk with no section must NOT produce a heading
    assert!(
        !prose.contains("## \n"),
        "a chunk with no section must not produce an empty heading; prose: {prose:?}"
    );
    // The marker must appear before the chunk text
    let pos_heading = prose.find("## Methods").expect("## Methods marker");
    let pos_text = prose.find("methods text").expect("methods text");
    assert!(
        pos_heading < pos_text,
        "section heading must precede the chunk text; prose: {prose:?}"
    );
}

/// A paper with zero lit_chunks rows yields prose equal to its FusedHit.content
/// (the S2 abstract path) and is NOT dropped from the panel.
#[tokio::test]
async fn s2_paper_falls_back_to_fused_content() {
    let pool = make_pool().await;
    let paper_id = "s2:abc123def456";

    // Do NOT insert any lit_chunks for this paper — S2 abstract-only path.
    // The paper row is also absent — adapter must handle both.

    let mut hit = make_fused_hit(paper_id);
    hit.source_type = "s2".to_string();
    hit.content = "S2 abstract text for the paper".to_string();

    let hits = vec![hit];
    let panel = build_panel(hits, &pool).await.unwrap();

    assert_eq!(panel.len(), 1, "S2 paper must NOT be dropped from the panel");
    assert_eq!(
        panel[0].prose, "S2 abstract text for the paper",
        "prose must equal FusedHit.content when no lit_chunks rows exist"
    );
    assert_eq!(panel[0].expert_id.as_str(), paper_id);
}

/// WR-04: a paper WITH lit_chunks rows whose content is all empty/blank must
/// fall back to FusedHit.content rather than emit a degenerate prose body.
/// The `rows.is_empty()` gate misses this populated-but-empty case.
#[tokio::test]
async fn empty_chunk_content_falls_back_to_fused_content() {
    let pool = make_pool().await;
    let paper_id = "arxiv:2401.05050";

    // Rows EXIST (so rows.is_empty() is false) but their content is blank.
    insert_lit_chunk(&pool, paper_id, 0, None, "").await;
    insert_lit_chunk(&pool, paper_id, 1, None, "   ").await;

    let mut hit = make_fused_hit(paper_id);
    hit.content = "Populated FusedHit fallback content".to_string();

    let hits = vec![hit];
    let panel = build_panel(hits, &pool).await.unwrap();

    assert_eq!(panel.len(), 1, "paper must NOT be dropped");
    assert_eq!(
        panel[0].prose, "Populated FusedHit fallback content",
        "blank chunk content must fall back to FusedHit.content, not yield empty/heading-only prose; \
         got: {:?}",
        panel[0].prose
    );
}

// ── ADAPT-02 build_panel-driven tests ─────────────────────────────────────────

/// A FusedHit with source_id "arxiv:2401.00001" produces an ExpertResponse
/// whose expert_id.as_str() == "arxiv:2401.00001" AND
/// provenance.source_id.as_str() == the same string.
/// Neither field is transformed — the paper_id round-trips unchanged.
#[tokio::test]
async fn expert_id_equals_paper_id() {
    let pool = make_pool().await;
    let paper_id = "arxiv:2401.00001";

    let hits = vec![make_fused_hit(paper_id)];
    let panel = build_panel(hits, &pool).await.expect("build_panel must succeed");

    assert_eq!(panel.len(), 1);
    assert_eq!(
        panel[0].expert_id.as_str(),
        paper_id,
        "expert_id must equal paper_id verbatim"
    );
    assert_eq!(
        panel[0].provenance.source_id.as_str(),
        paper_id,
        "provenance.source_id must equal paper_id verbatim"
    );
}

/// build_panel over hits covering 3 distinct paper_ids returns a Vec whose
/// distinct expert_id count == 3, and build_panel succeeds (conservation_assert
/// passes on the happy path — no source_id loss).
#[tokio::test]
async fn panel_size_matches_distinct_papers() {
    let pool = make_pool().await;

    let hits = vec![
        make_fused_hit("arxiv:2401.10001"),
        make_fused_hit("arxiv:2401.10002"),
        make_fused_hit("arxiv:2401.10003"),
    ];

    let panel = build_panel(hits, &pool).await.expect("build_panel must succeed (conservation_assert passes)");

    let distinct_ids: std::collections::HashSet<&str> =
        panel.iter().map(|r| r.expert_id.as_str()).collect();

    assert_eq!(
        distinct_ids.len(),
        3,
        "distinct expert_id count must equal distinct input paper_id count (== panel_size)"
    );
}
