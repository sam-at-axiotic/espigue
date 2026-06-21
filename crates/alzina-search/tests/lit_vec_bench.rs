//! 100k-scale vector-latency benchmark for the literature store (RETR-04).
//!
//! This test is `#[ignore]`-gated and must be run manually:
//!
//! ```text
//! cargo test --test lit_vec_bench bench_lit_vec_100k -- --ignored --nocapture
//! ```
//!
//! Design:
//! - 100k synthetic 1024-dim unit vectors, fixed SEED=42 — zero Jina API calls.
//! - N=10 concurrent `tokio::spawn` queries (fan-out, not single-query).
//! - Reports p50 and p95 over 100 iterations.
//! - Asserts p95 < 200ms → keep exact sqlite-vec; fail → evaluate ANN backend.

use alzina_search::{
    lit_schema,
    schema,
    sqlite_vec::SqliteVecStore,
};
use alzina_core::{
    error::AlzinaResult,
    search::{VectorFilters, VectorMetadata, VectorStore},
};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use std::str::FromStr;
use std::sync::Arc;

/// Open a tempfile-backed SQLite pool and run the literature schema migration.
/// Uses a real on-disk file to exercise actual SQLite I/O (not :memory:).
///
/// IMPORTANT: calls `register_sqlite_vec_extension()` before opening the pool.
/// SQLite auto-extensions apply to connections opened AFTER registration, so
/// the registration must precede pool construction (Pitfall 3 from RESEARCH.md).
async fn open_tempfile_lit_pool() -> AlzinaResult<(sqlx::sqlite::SqlitePool, tempfile::NamedTempFile)> {
    // Register the sqlite-vec extension before any pool connection is opened.
    let ext_ok = schema::register_sqlite_vec_extension();
    assert!(ext_ok, "sqlite-vec extension registration failed — vec0 will be unavailable");

    let db_file = tempfile::NamedTempFile::new().expect("tempfile creation failed");
    let path = db_file.path().to_str().expect("tempfile path is valid UTF-8");

    let opts = SqliteConnectOptions::from_str(&format!("sqlite:{path}"))
        .expect("valid sqlite URL")
        .create_if_missing(true)
        // Increase connection pool to allow concurrent readers (N=10 fan-out).
        // WAL mode with multiple connections is the standard SQLite concurrency pattern.
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

    let pool = SqlitePoolOptions::new()
        .max_connections(20)
        .connect_with(opts)
        .await
        .expect("lit tempfile pool connect");

    lit_schema::migrate(&pool, 1024).await?;
    Ok((pool, db_file))
}

/// Generate a unit vector by sampling DIM f32 values in [-1,1] from the RNG
/// and normalising by the L2 norm. Zero embedder calls.
fn gen_unit_vector(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        // Pathological case — return canonical first-basis vector.
        let mut e = vec![0.0_f32; dim];
        e[0] = 1.0;
        e
    } else {
        v.into_iter().map(|x| x / norm).collect()
    }
}

/// 10k fan-out benchmark — Phase 23 / JUSTIFY-ANN decision follow-up.
///
/// Runs at 10k scale (realistic steady-state lit corpus). The 100k benchmark
/// (RETR-04) failed the <200ms p95 bar. Per the Phase 23 decision_resolution,
/// a 10k-scale benchmark with N=5 fan-out determines whether sqlite-vec exact
/// kNN is viable at the expected production corpus size, avoiding unnecessary
/// Qdrant infrastructure.
///
/// Run manually:
/// ```text
/// cargo test -p alzina-search --test lit_vec_bench bench_lit_vec_10k -- --ignored --nocapture --release
/// ```
#[tokio::test]
#[ignore = "10k benchmark — run: cargo test -p alzina-search --test lit_vec_bench bench_lit_vec_10k -- --ignored --nocapture --release"]
async fn bench_lit_vec_10k() {
    const N: usize = 10_000;
    const DIM: usize = 1024;
    const SEED: u64 = 42;
    const CONCURRENT: usize = 5; // TTD N=5 fan-out (realistic production)
    const TOP_K: usize = 25;     // TTD retrieval_top_k
    const ITERATIONS: usize = 50;

    let (pool, _db_file) = open_tempfile_lit_pool()
        .await
        .expect("lit tempfile pool init");

    let store = Arc::new(
        SqliteVecStore::with_table_names(pool, DIM, "lit_vec0", "lit_chunks")
            .await
            .expect("SqliteVecStore init"),
    );

    assert!(
        store.is_enabled(),
        "SqliteVecStore must be enabled (sqlite-vec extension loaded)"
    );

    let mut rng = SmallRng::seed_from_u64(SEED);
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| gen_unit_vector(&mut rng, DIM)).collect();

    println!("Inserting {N} vectors (dim={DIM})...");
    let insert_start = std::time::Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        let meta = VectorMetadata {
            source_type: "bench".to_string(),
            source_id: format!("bench-{i}"),
            chunk_index: 0,
            content_preview: String::new(),
            source_agent: None,
            source_date: None,
            weave_id: None,
            section: None,
            domain: None,
            indexed_at: "2026-06-07T00:00:00Z".to_string(),
        };
        store
            .insert(v, meta)
            .await
            .unwrap_or_else(|e| panic!("insert {i} failed: {e}"));
    }
    let insert_elapsed = insert_start.elapsed();
    println!(
        "Insert complete: {N} vectors in {:.1}s ({:.0}/s)",
        insert_elapsed.as_secs_f64(),
        N as f64 / insert_elapsed.as_secs_f64()
    );

    println!("Running {ITERATIONS} iterations of N={CONCURRENT} concurrent searches (top_k={TOP_K})...");
    let filters = VectorFilters::default();
    let mut latencies: Vec<u64> = Vec::with_capacity(ITERATIONS);

    for iter in 0..ITERATIONS {
        let base = iter * CONCURRENT;
        let store_ref = Arc::clone(&store);
        let queries: Vec<Vec<f32>> = (0..CONCURRENT)
            .map(|j| vectors[(base + j) % N].clone())
            .collect();
        let filters_clone = filters.clone();

        let start = std::time::Instant::now();
        let handles: Vec<_> = queries
            .into_iter()
            .map(|q| {
                let store_inner = Arc::clone(&store_ref);
                let f = filters_clone.clone();
                tokio::spawn(async move { store_inner.search(&q, TOP_K, &f).await })
            })
            .collect();

        let results = futures::future::join_all(handles).await;
        let elapsed_ms = start.elapsed().as_millis() as u64;
        latencies.push(elapsed_ms);

        if iter == 0 {
            for (j, r) in results.iter().enumerate() {
                match r {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => panic!("search {j} returned error: {e}"),
                    Err(e) => panic!("tokio task {j} panicked: {e}"),
                }
            }
        }
    }

    latencies.sort_unstable();
    let p50 = latencies[ITERATIONS / 2];
    let p95 = latencies[(ITERATIONS * 95) / 100];

    println!(
        "10k bench result: p50={p50}ms p95={p95}ms (N={N}, DIM={DIM}, CONCURRENT={CONCURRENT}, top_k={TOP_K}, ITERATIONS={ITERATIONS})"
    );
    println!("Bar: p95 < 200ms");
    println!(
        "VERDICT: {}",
        if p95 < 200 {
            "PASS — sqlite-vec meets the <200ms bar at 10k scale; keep sqlite-vec canonical"
        } else {
            "FAIL — p95 exceeds 200ms bar; Qdrant wire-up is mandatory"
        }
    );
}

/// 100k fan-out benchmark for the literature vector store (RETR-04).
///
/// Run manually (takes several minutes on first run due to 100k inserts):
/// ```text
/// cargo test --test lit_vec_bench bench_lit_vec_100k -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "100k benchmark — run manually: cargo test --test lit_vec_bench bench_lit_vec_100k -- --ignored --nocapture"]
async fn bench_lit_vec_100k() {
    const N: usize = 100_000;
    const DIM: usize = 1024;
    const SEED: u64 = 42;
    const CONCURRENT: usize = 10;
    const ITERATIONS: usize = 100;

    let (pool, _db_file) = open_tempfile_lit_pool()
        .await
        .expect("lit tempfile pool init");

    let store = Arc::new(
        SqliteVecStore::with_table_names(pool, DIM, "lit_vec0", "lit_chunks")
            .await
            .expect("SqliteVecStore init"),
    );

    assert!(
        store.is_enabled(),
        "SqliteVecStore must be enabled (sqlite-vec extension loaded)"
    );

    // --- Generate 100k seeded synthetic unit vectors (zero Jina API calls) ---
    let mut rng = SmallRng::seed_from_u64(SEED);
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| gen_unit_vector(&mut rng, DIM)).collect();

    // --- Insert all vectors into the lit store ---
    println!("Inserting {N} vectors (dim={DIM})...");
    let insert_start = std::time::Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        let meta = VectorMetadata {
            source_type: "bench".to_string(),
            source_id: format!("bench-{i}"),
            chunk_index: 0,
            content_preview: String::new(),
            source_agent: None,
            source_date: None,
            weave_id: None,
            section: None,
            domain: None,
            indexed_at: "2026-06-05T00:00:00Z".to_string(),
        };
        store
            .insert(v, meta)
            .await
            .unwrap_or_else(|e| panic!("insert {i} failed: {e}"));
    }
    let insert_elapsed = insert_start.elapsed();
    println!(
        "Insert complete: {N} vectors in {:.1}s ({:.0}/s)",
        insert_elapsed.as_secs_f64(),
        N as f64 / insert_elapsed.as_secs_f64()
    );

    // --- Fan-out latency: N=10 concurrent queries, ITERATIONS rounds ---
    // Each iteration spawns CONCURRENT tokio tasks each running one search,
    // then awaits ALL with join_all. The wall-clock elapsed for ALL tasks to
    // complete is the fan-out latency — this measures SQLite's behaviour under
    // real concurrent read pressure, not single-query throughput (Pitfall 2).
    println!("Running {ITERATIONS} iterations of N={CONCURRENT} concurrent searches...");
    let filters = VectorFilters::default();
    let mut latencies: Vec<u64> = Vec::with_capacity(ITERATIONS);

    for iter in 0..ITERATIONS {
        // Rotate the query vectors across iterations to avoid query-caching effects.
        let base = iter * CONCURRENT;
        let store_ref = Arc::clone(&store);
        let queries: Vec<Vec<f32>> = (0..CONCURRENT)
            .map(|j| vectors[(base + j) % N].clone())
            .collect();
        let filters_clone = filters.clone();

        let start = std::time::Instant::now();
        let handles: Vec<_> = queries
            .into_iter()
            .map(|q| {
                let store_inner = Arc::clone(&store_ref);
                let f = filters_clone.clone();
                tokio::spawn(async move { store_inner.search(&q, 10, &f).await })
            })
            .collect();

        // join_all awaits ALL spawned tasks — this is fan-out latency.
        let results = futures::future::join_all(handles).await;
        let elapsed_ms = start.elapsed().as_millis() as u64;
        latencies.push(elapsed_ms);

        // Verify no search errors (spot-check on first iteration).
        if iter == 0 {
            for (j, r) in results.iter().enumerate() {
                match r {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => panic!("search {j} returned error: {e}"),
                    Err(e) => panic!("tokio task {j} panicked: {e}"),
                }
            }
        }
    }

    // --- Compute and assert p50 / p95 ---
    latencies.sort_unstable();
    let p50 = latencies[ITERATIONS / 2];
    let p95 = latencies[(ITERATIONS * 95) / 100];

    println!("100k bench: p50={p50}ms p95={p95}ms");
    println!("  (N={N} vectors, DIM={DIM}, CONCURRENT={CONCURRENT}, ITERATIONS={ITERATIONS})");

    assert!(p95 < 200, "p95 {p95}ms exceeds 200ms bar — evaluate ANN backend for this corpus");
}
