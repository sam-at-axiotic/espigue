//! Qdrant -> SqliteVecStore migration tool.
//!
//! Phase 3 Task 3.12 (AC-2). Reads points from an existing Qdrant
//! collection (the Norn-Weave `literature_chunks` snapshot under
//! `/Users/samj/clawd/skills/lit-review/cache/qdrant/`) and re-indexes
//! them into our [`VectorStore`]. The daemon never depends on Qdrant —
//! only this opt-in migration helper does.
//!
//! AC-1: connection failures and fatal scroll errors return
//! `AlzinaError::Search { degraded: true, .. }`. Per-point mapping or
//! insert errors are captured into the report rather than aborting —
//! partial migration is preferable to silent loss.
//!
//! `process_point` and `map_point_to_metadata` are factored out as
//! `pub(crate)` so unit tests can drive them without spinning up a real
//! Qdrant instance — the full `run()` against live Qdrant is
//! integration-only.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;

use alzina_core::error::{AlzinaError, AlzinaResult, SearchDetail};
use alzina_core::search::{VectorMetadata, VectorStore};
use alzina_core::truncate_for_preview;

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{ScrollPointsBuilder, Value, value::Kind as ValueKind};

/// Configuration for [`QdrantMigration`].
#[derive(Debug, Clone)]
pub struct QdrantMigrationConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub collection: String,
    pub batch_size: u32,
    pub source_type: String,
}

impl Default for QdrantMigrationConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:6334".into(),
            api_key: None,
            collection: "literature_chunks".into(),
            batch_size: 256,
            source_type: "qdrant".into(),
        }
    }
}

/// Outcome of a [`QdrantMigration::run`] invocation.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct QdrantMigrationReport {
    pub points_read: usize,
    pub points_written: usize,
    pub points_skipped: usize,
    /// Per-point errors: `(point_id, err.to_string())`.
    pub errors: Vec<(String, String)>,
}

/// Migration driver. Read-only on Qdrant, write-only on `vec_store`.
pub struct QdrantMigration {
    cfg: QdrantMigrationConfig,
    vec_store: Arc<dyn VectorStore>,
}

impl QdrantMigration {
    pub fn new(cfg: QdrantMigrationConfig, vec_store: Arc<dyn VectorStore>) -> Self {
        Self { cfg, vec_store }
    }

    /// Connect to Qdrant, scroll all points in the configured
    /// collection, and re-index each into the local vector store.
    /// Connection or fatal scroll errors -> degraded
    /// `AlzinaError::Search`. Per-point errors land in the report.
    pub async fn run(&self) -> AlzinaResult<QdrantMigrationReport> {
        let client = build_client(&self.cfg)?;
        let mut report = QdrantMigrationReport::default();
        let mut next_offset: Option<qdrant_client::qdrant::PointId> = None;
        let mut batch_no: usize = 0;

        loop {
            let mut builder = ScrollPointsBuilder::new(&self.cfg.collection)
                .with_payload(true)
                .with_vectors(true)
                .limit(self.cfg.batch_size);
            if let Some(off) = next_offset.take() {
                builder = builder.offset(off);
            }

            let resp = client.scroll(builder).await.map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("qdrant scroll: {e}"),
                    degraded: true,
                    degradation_reason: Some(format!(
                        "qdrant scroll failed for collection {}: {e}",
                        self.cfg.collection
                    )),
                })
            })?;

            batch_no += 1;
            tracing::info!(
                batch = batch_no,
                points = resp.result.len(),
                collection = %self.cfg.collection,
                "qdrant migration: scrolled batch"
            );

            for point in resp.result {
                report.points_read += 1;
                self.process_point(point, &mut report).await;
            }

            // Per-batch progress heartbeat — fires every 10 batches so
            // long migrations show steady progress without spamming the
            // console on small collections. The terminal summary below
            // still logs unconditionally on completion.
            if batch_no % 10 == 0 {
                tracing::info!(
                    target: "alzina_search::qdrant_migration",
                    points_read = report.points_read,
                    points_written = report.points_written,
                    points_skipped = report.points_skipped,
                    errors = report.errors.len(),
                    "qdrant migration progress"
                );
            }

            next_offset = resp.next_page_offset;
            if next_offset.is_none() {
                break;
            }
        }

        tracing::info!(
            read = report.points_read,
            written = report.points_written,
            skipped = report.points_skipped,
            errors = report.errors.len(),
            "qdrant migration: complete"
        );
        Ok(report)
    }

    /// Process a single Qdrant `RetrievedPoint`: extract id, vector,
    /// and payload; build [`VectorMetadata`]; insert into the store or
    /// record an error in the report. Never panics.
    pub(crate) async fn process_point(
        &self,
        point: qdrant_client::qdrant::RetrievedPoint,
        report: &mut QdrantMigrationReport,
    ) {
        let id_str = point_id_to_string(point.id.as_ref());
        let vector = match extract_vector(point.vectors.as_ref()) {
            Some(v) => v,
            None => {
                tracing::warn!(point_id = %id_str, "qdrant migration: skipping point with no vector");
                report.points_skipped += 1;
                return;
            }
        };
        let metadata = map_point_to_metadata(&id_str, &point.payload, &self.cfg.source_type);
        match self.vec_store.insert(&vector, metadata).await {
            Ok(_) => report.points_written += 1,
            Err(e) => {
                tracing::warn!(point_id = %id_str, error = %e, "qdrant migration: insert failed");
                report.errors.push((id_str, e.to_string()));
            }
        }
    }
}

/// Build a connected Qdrant client; connection failures map to a
/// degraded `AlzinaError::Search`.
fn build_client(cfg: &QdrantMigrationConfig) -> AlzinaResult<Qdrant> {
    let mut builder = Qdrant::from_url(&cfg.url);
    if let Some(key) = cfg.api_key.clone() {
        builder = builder.api_key(key);
    }
    builder.build().map_err(|e| {
        AlzinaError::Search(SearchDetail {
            message: format!("qdrant connect: {e}"),
            degraded: true,
            degradation_reason: Some(format!("qdrant connect to {} failed: {e}", cfg.url)),
        })
    })
}

/// Stringify a Qdrant `PointId` (numeric or UUID) into a stable id.
fn point_id_to_string(id: Option<&qdrant_client::qdrant::PointId>) -> String {
    use qdrant_client::qdrant::point_id::PointIdOptions;
    match id.and_then(|p| p.point_id_options.as_ref()) {
        Some(PointIdOptions::Num(n)) => n.to_string(),
        Some(PointIdOptions::Uuid(u)) => u.clone(),
        None => String::new(),
    }
}

/// Extract a flat `Vec<f32>` from the optional `VectorsOutput`. Uses
/// the default (unnamed) vector lane — Norn-Weave's `literature_chunks`
/// collection uses unnamed vectors. Sparse and multi-dense variants are
/// out of scope for this slice. Empty data is treated as missing.
fn extract_vector(vectors: Option<&qdrant_client::qdrant::VectorsOutput>) -> Option<Vec<f32>> {
    use qdrant_client::qdrant::vector_output::Vector;
    match vectors?.get_vector()? {
        Vector::Dense(d) if !d.data.is_empty() => Some(d.data),
        _ => None,
    }
}

/// Defensive cap on the length (in bytes) of any *metadata* string
/// field extracted from a Qdrant payload (e.g. `source_agent`,
/// `domain`, `section`). Foreign-data integrity guard — the source
/// collection is owned by another system, so we refuse to let a
/// pathological field blow up our metadata table. The
/// `text`/`content` preview source is exempt: it goes through
/// `truncate_for_preview` (PREVIEW_MAX_CHARS = 400) instead.
const PAYLOAD_FIELD_MAX_LEN: usize = 256;

/// Truncate `s` so the returned slice is at most `max_bytes` long and
/// always ends on a UTF-8 char boundary (never splits a multi-byte
/// character). Cheap and allocation-free.
fn cap_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Find the last char start <= max_bytes; advance past it iff the
    // char ends at or before max_bytes.
    let end = s
        .char_indices()
        .take_while(|(i, _)| *i <= max_bytes)
        .last()
        .map(|(i, c)| {
            let next = i + c.len_utf8();
            if next <= max_bytes { next } else { i }
        })
        .unwrap_or(0);
    &s[..end]
}

/// Pull a string-typed payload field, ignoring non-string kinds. The
/// raw, *uncapped* contents — callers cap if they're storing the value
/// as metadata (see [`payload_str_capped`]).
fn payload_str<'a>(payload: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
    match payload.get(key)?.kind.as_ref()? {
        ValueKind::StringValue(s) => Some(s.as_str()),
        _ => None,
    }
}

/// As [`payload_str`], but truncates the result at
/// [`PAYLOAD_FIELD_MAX_LEN`] bytes on a UTF-8 char boundary. Use this
/// when the value will be persisted as metadata.
fn payload_str_capped<'a>(payload: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
    payload_str(payload, key).map(|s| cap_at_char_boundary(s, PAYLOAD_FIELD_MAX_LEN))
}

/// Map a Qdrant point's payload into our [`VectorMetadata`]. Pure
/// function — `text` then `content` is the preview-source fallback;
/// other canonical fields come through verbatim when present as strings.
pub(crate) fn map_point_to_metadata(
    point_id: &str,
    payload: &HashMap<String, Value>,
    source_type: &str,
) -> VectorMetadata {
    let preview_src = payload_str(payload, "text")
        .or_else(|| payload_str(payload, "content"))
        .unwrap_or("");
    let content_preview = if preview_src.is_empty() {
        String::new()
    } else {
        truncate_for_preview(preview_src)
    };
    VectorMetadata {
        source_type: source_type.to_string(),
        source_id: point_id.to_string(),
        chunk_index: 0,
        content_preview,
        source_agent: payload_str_capped(payload, "source_agent").map(str::to_string),
        source_date: payload_str_capped(payload, "source_date").map(str::to_string),
        weave_id: payload_str_capped(payload, "weave_id").map(str::to_string),
        section: payload_str_capped(payload, "section").map(str::to_string),
        domain: payload_str_capped(payload, "domain").map(str::to_string),
        indexed_at: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_core::search::{VectorFilters, VectorHit};
    use async_trait::async_trait;
    use std::sync::Mutex;

    fn sval(s: &str) -> Value {
        Value {
            kind: Some(ValueKind::StringValue(s.to_string())),
        }
    }

    #[derive(Default)]
    struct StubVecStore {
        inserts: Mutex<Vec<(Vec<f32>, VectorMetadata)>>,
        fail_next: Mutex<bool>,
    }

    #[async_trait]
    impl VectorStore for StubVecStore {
        async fn insert(&self, v: &[f32], m: VectorMetadata) -> AlzinaResult<i64> {
            let mut fail = self.fail_next.lock().unwrap();
            if *fail {
                *fail = false;
                return Err(AlzinaError::Search(SearchDetail {
                    message: "stub forced failure".into(),
                    degraded: true,
                    degradation_reason: Some("stub forced failure".into()),
                }));
            }
            drop(fail);
            let mut g = self.inserts.lock().unwrap();
            g.push((v.to_vec(), m));
            Ok(g.len() as i64)
        }
        async fn search(
            &self,
            _v: &[f32],
            _k: usize,
            _f: &VectorFilters,
        ) -> AlzinaResult<Vec<VectorHit>> {
            Ok(vec![])
        }
        async fn delete_by_source(&self, _t: &str, _i: &str) -> AlzinaResult<usize> {
            Ok(0)
        }
    }

    fn point(
        id: &str,
        payload: HashMap<String, Value>,
        vector: Option<Vec<f32>>,
    ) -> qdrant_client::qdrant::RetrievedPoint {
        use qdrant_client::qdrant::point_id::PointIdOptions;
        use qdrant_client::qdrant::vectors_output::VectorsOptions;
        use qdrant_client::qdrant::{PointId, RetrievedPoint, VectorOutput, VectorsOutput};

        let pid = PointId {
            point_id_options: Some(PointIdOptions::Uuid(id.into())),
        };
        // OPS (Phase 11 Wave 5): qdrant_client 1.16+ deprecated the
        // `VectorOutput::data` field in favour of `into_vector()`
        // (read-only). For test-fixture construction we still need
        // the field; migration to a new builder API is captured as
        // a follow-up. Prod build does NOT trip this warning — only
        // `--all-targets` includes the test module.
        #[allow(deprecated)]
        let vectors = vector.map(|data| VectorsOutput {
            vectors_options: Some(VectorsOptions::Vector(VectorOutput {
                data,
                ..Default::default()
            })),
        });
        // ..Default::default() so we don't break when the proto evolves.
        RetrievedPoint {
            id: Some(pid),
            payload,
            vectors,
            ..Default::default()
        }
    }

    fn migration(store: Arc<dyn VectorStore>) -> QdrantMigration {
        QdrantMigration::new(QdrantMigrationConfig::default(), store)
    }

    #[test]
    fn map_point_to_metadata_uses_text_then_content_fallback() {
        // text wins over content
        let mut p = HashMap::new();
        p.insert("text".into(), sval("hello world"));
        p.insert("content".into(), sval("ignored"));
        p.insert("section".into(), sval("intro"));
        let m = map_point_to_metadata("pid-1", &p, "qdrant");
        assert_eq!(m.source_type, "qdrant");
        assert_eq!(m.source_id, "pid-1");
        assert_eq!(m.chunk_index, 0);
        assert_eq!(m.content_preview, "hello world");
        assert_eq!(m.section.as_deref(), Some("intro"));

        // content fallback when text missing
        let mut p2 = HashMap::new();
        p2.insert("content".into(), sval("body text"));
        let m2 = map_point_to_metadata("pid-2", &p2, "qdrant");
        assert_eq!(m2.content_preview, "body text");
        assert!(m2.domain.is_none());

        // both missing -> empty preview, all canonical fields None
        let m3 = map_point_to_metadata("pid-3", &HashMap::new(), "qdrant");
        assert_eq!(m3.content_preview, "");
        assert!(m3.source_agent.is_none());
        assert!(m3.weave_id.is_none());
        assert!(!m3.indexed_at.is_empty());
    }

    #[tokio::test]
    async fn qdrant_migration_reports_skipped_when_vector_missing() {
        let store = Arc::new(StubVecStore::default());
        let mig = migration(Arc::clone(&store) as Arc<dyn VectorStore>);
        let mut report = QdrantMigrationReport::default();
        mig.process_point(point("pid-skip", HashMap::new(), None), &mut report)
            .await;
        assert_eq!(report.points_skipped, 1);
        assert_eq!(report.points_written, 0);
        assert!(report.errors.is_empty());
        assert!(store.inserts.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn qdrant_migration_writes_to_vec_store() {
        let store = Arc::new(StubVecStore::default());
        let mig = migration(Arc::clone(&store) as Arc<dyn VectorStore>);
        let mut report = QdrantMigrationReport::default();

        let mut payload = HashMap::new();
        payload.insert("text".into(), sval("hello"));
        payload.insert("domain".into(), sval("biology"));
        mig.process_point(
            point("pid-ok", payload, Some(vec![0.1, 0.2, 0.3])),
            &mut report,
        )
        .await;

        // Force the next insert to fail so the error path is exercised.
        *store.fail_next.lock().unwrap() = true;
        let mut p2_payload = HashMap::new();
        p2_payload.insert("content".into(), sval("doomed"));
        mig.process_point(
            point("pid-fail", p2_payload, Some(vec![0.4, 0.5, 0.6])),
            &mut report,
        )
        .await;

        assert_eq!(report.points_written, 1);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].0, "pid-fail");

        let snap = store.inserts.lock().unwrap().clone();
        assert_eq!(snap.len(), 1);
        let (vec, meta) = &snap[0];
        assert_eq!(vec, &vec![0.1_f32, 0.2, 0.3]);
        assert_eq!(meta.source_id, "pid-ok");
        assert_eq!(meta.content_preview, "hello");
        assert_eq!(meta.domain.as_deref(), Some("biology"));
    }

    #[test]
    fn payload_str_truncates_long_strings_at_char_boundary() {
        // Build a payload with ~10KB of multi-byte UTF-8 ("é" = 2 bytes)
        // so that a naive byte-slice could split a code point.
        let big: String = "é".repeat(5_000); // 10_000 bytes
        let mut payload = HashMap::new();
        payload.insert("domain".into(), sval(&big));

        let capped = payload_str_capped(&payload, "domain").expect("cap should yield Some");
        assert!(
            capped.len() <= PAYLOAD_FIELD_MAX_LEN,
            "capped length {} > PAYLOAD_FIELD_MAX_LEN={}",
            capped.len(),
            PAYLOAD_FIELD_MAX_LEN
        );
        // Round-tripping as &str already proves char-boundary alignment
        // (slicing on a non-boundary panics), but assert explicitly so a
        // future regression that returns bytes-not-str is caught.
        assert!(
            big.is_char_boundary(capped.len()),
            "cap point {} is not a char boundary",
            capped.len()
        );
        // And no truncated trailing byte: every byte in the cap must be
        // part of a complete UTF-8 sequence.
        assert!(std::str::from_utf8(capped.as_bytes()).is_ok());

        // Short input is returned unchanged.
        let mut small = HashMap::new();
        small.insert("section".into(), sval("intro"));
        assert_eq!(payload_str_capped(&small, "section"), Some("intro"));
    }

    #[test]
    fn map_point_to_metadata_caps_all_extracted_fields() {
        // Synthetic payload with 1KB strings in every metadata field —
        // each must be truncated to <=256 bytes in the produced metadata.
        let big = "a".repeat(1024);
        let mut payload = HashMap::new();
        payload.insert("source_agent".into(), sval(&big));
        payload.insert("source_date".into(), sval(&big));
        payload.insert("weave_id".into(), sval(&big));
        payload.insert("section".into(), sval(&big));
        payload.insert("domain".into(), sval(&big));
        // text field is intentionally unrelated to the cap — only the
        // `truncate_for_preview` rule applies there.
        payload.insert("text".into(), sval("preview ok"));

        let m = map_point_to_metadata("pid-cap", &payload, "qdrant");
        for (name, val) in [
            ("source_agent", &m.source_agent),
            ("source_date", &m.source_date),
            ("weave_id", &m.weave_id),
            ("section", &m.section),
            ("domain", &m.domain),
        ] {
            let v = val.as_deref().unwrap_or_else(|| panic!("{name} missing"));
            assert!(
                v.len() <= PAYLOAD_FIELD_MAX_LEN,
                "{name} length {} exceeds cap {}",
                v.len(),
                PAYLOAD_FIELD_MAX_LEN
            );
        }
        // Preview untouched by the metadata cap.
        assert_eq!(m.content_preview, "preview ok");
    }

    #[tokio::test]
    async fn qdrant_migration_propagates_connection_failure_as_degraded() {
        // Port 1 — nothing listens there, so connect fails fast.
        let mig = QdrantMigration::new(
            QdrantMigrationConfig {
                url: "http://127.0.0.1:1".into(),
                ..QdrantMigrationConfig::default()
            },
            Arc::new(StubVecStore::default()),
        );
        let err = mig.run().await.expect_err("expected connection failure");
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded);
                assert!(detail.degradation_reason.is_some());
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }
    }
}
