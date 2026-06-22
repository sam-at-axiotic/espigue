//! Seed-paper resolution (Phase 4).
//!
//! `--seed-papers <id,id,...>` builds the panel directly from user-supplied
//! papers, skipping three-lane fusion + the topicality gate. Each id is fetched,
//! persisted as an abstract chunk so [`build_panel`] can ground on it, and turned
//! into a [`FusedHit`] with `relevance == 1.0`.
//!
//! Id kinds accepted: arXiv (`2310.06825`, `arXiv:2310.06825`), DOI
//! (`10.1234/foo`, `DOI:10.1234/foo`), and Semantic Scholar (`s2:<hash>`,
//! `CorpusId:N`, raw hash). When `S2_API_KEY` is set the S2 client resolves all
//! three kinds; without it, only arXiv ids resolve (via the arXiv API, no key
//! needed) and DOI/S2 ids are loudly skipped.
//!
//! Loud-degrade: a per-id failure is collected, never aborts the batch. The
//! canonical `source_id` keying (`arxiv:{id}` / `s2:{hash}`) mirrors the
//! three-lane fusion path so the rest of the pipeline is unchanged.
//!
//! [`build_panel`]: orchestration::adapter::build_panel

use search::{
    persist_arxiv_abstract, persist_s2_abstract, Acquire, ArxivClient, ArxivConfig, ArxivResult,
    Endpoint, FusedHit, S2PaperFull,
};

use crate::context::LitContext;

/// A classified seed id, with provider prefixes stripped to the bare value.
enum SeedId {
    /// Bare arXiv id, e.g. `"2310.06825"` or `"2310.06825v2"`.
    Arxiv(String),
    /// Bare DOI, e.g. `"10.1234/foo"`.
    Doi(String),
    /// Raw Semantic Scholar id (hash, or `CorpusId:N`).
    S2(String),
}

/// Strip a case-insensitive prefix, returning the remainder when it matches.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

/// New-style arXiv id: `NNNN.NNNNN` optionally followed by `vN`.
fn is_arxiv_new_style(s: &str) -> bool {
    let base = match s.rfind('v') {
        Some(pos) if s[pos + 1..].chars().all(|c| c.is_ascii_digit()) && pos + 1 < s.len() => {
            &s[..pos]
        }
        _ => s,
    };
    let mut parts = base.splitn(2, '.');
    match (parts.next(), parts.next()) {
        (Some(a), Some(b)) => {
            a.len() == 4
                && a.chars().all(|c| c.is_ascii_digit())
                && (4..=5).contains(&b.len())
                && b.chars().all(|c| c.is_ascii_digit())
        }
        _ => false,
    }
}

/// Classify a raw user-supplied id into a [`SeedId`].
fn classify_seed(raw: &str) -> SeedId {
    let t = raw.trim();
    if let Some(rest) = strip_prefix_ci(t, "arxiv:") {
        return SeedId::Arxiv(rest.to_string());
    }
    if let Some(rest) = strip_prefix_ci(t, "doi:") {
        return SeedId::Doi(rest.to_string());
    }
    if let Some(rest) = strip_prefix_ci(t, "s2:") {
        return SeedId::S2(rest.to_string());
    }
    if let Some(rest) = strip_prefix_ci(t, "corpusid:") {
        return SeedId::S2(format!("CorpusId:{rest}"));
    }
    // DOI without prefix: 10.NNNN/...
    if t.starts_with("10.") && t.contains('/') {
        return SeedId::Doi(t.to_string());
    }
    if is_arxiv_new_style(t) {
        return SeedId::Arxiv(t.to_string());
    }
    // Default: a raw Semantic Scholar id (hash).
    SeedId::S2(t.to_string())
}

/// Build a `relevance == 1.0` [`FusedHit`] for a resolved seed. The
/// `source_type`/`source_id` keying mirrors the three-lane fusion path so
/// downstream promotion, citation, and panel code treat seeds identically.
fn make_hit(source_id: String, title: String, content: String) -> FusedHit {
    let source_type = if source_id.starts_with("s2:") { "s2" } else { "arxiv" };
    let content_preview = base::truncate_for_preview(&content);
    FusedHit {
        source_type: source_type.to_string(),
        source_id,
        title,
        section: None,
        content,
        content_preview,
        relevance: 1.0,
    }
}

/// Resolve one arXiv id via the keyless arXiv API and persist its abstract.
async fn resolve_arxiv(id: &str, ctx: &LitContext) -> anyhow::Result<Option<FusedHit>> {
    let client = ArxivClient::new(ArxivConfig::default())
        .map_err(|e| anyhow::anyhow!("arxiv client build failed: {e}"))?;
    let results = client
        .fetch_by_ids(&[id.to_string()])
        .await
        .map_err(|e| anyhow::anyhow!("arxiv fetch failed: {e}"))?;
    let meta = match results.into_iter().next() {
        Some(m) => m,
        None => return Ok(None),
    };
    persist_arxiv_abstract(
        ctx.lit_pool.as_ref(),
        ctx.lit_store.as_ref(),
        ctx.embedder.as_ref(),
        &meta,
    )
    .await
    .map_err(|e| anyhow::anyhow!("persist arxiv abstract failed: {e}"))?;
    let source_id = format!("arxiv:{}", meta.arxiv_id);
    Ok(Some(make_hit(source_id, meta.title, meta.abstract_text)))
}

/// Persist an S2 paper (abstract-only) and build its hit. Papers carrying an
/// arXiv id are keyed `arxiv:{id}` (canonical, mirrors `run_s2_lane`); pure-S2
/// papers are keyed `s2:{hash}`.
async fn persist_s2_hit(full: S2PaperFull, ctx: &LitContext) -> anyhow::Result<FusedHit> {
    if let Some(aid) = full.arxiv_id.clone() {
        let meta = ArxivResult {
            arxiv_id: aid.clone(),
            title: full.title.clone(),
            abstract_text: full.abstract_text.clone().unwrap_or_default(),
            authors: full.authors.clone(),
            published: full
                .year
                .map(|y| format!("{y}-01-01T00:00:00Z"))
                .unwrap_or_default(),
        };
        persist_arxiv_abstract(
            ctx.lit_pool.as_ref(),
            ctx.lit_store.as_ref(),
            ctx.embedder.as_ref(),
            &meta,
        )
        .await
        .map_err(|e| anyhow::anyhow!("persist arxiv abstract failed: {e}"))?;
        Ok(make_hit(format!("arxiv:{aid}"), meta.title, meta.abstract_text))
    } else {
        let source_id = format!("s2:{}", full.s2_id);
        let title = full.title.clone();
        let content = full.abstract_text.clone().unwrap_or_default();
        persist_s2_abstract(
            ctx.lit_pool.as_ref(),
            ctx.lit_store.as_ref(),
            ctx.embedder.as_ref(),
            &full,
        )
        .await
        .map_err(|e| anyhow::anyhow!("persist s2 abstract failed: {e}"))?;
        Ok(make_hit(source_id, title, content))
    }
}

/// Resolve one seed id to a [`FusedHit`], persisting its abstract on the way.
///
/// With S2 enabled the S2 client resolves every id kind; an S2 miss/error on an
/// arXiv id falls back to the keyless arXiv API. Without S2, only arXiv ids
/// resolve.
async fn resolve_one(seed: &SeedId, ctx: &LitContext) -> anyhow::Result<Option<FusedHit>> {
    if ctx.s2_enabled() {
        let query_id = match seed {
            SeedId::Arxiv(id) => format!("ARXIV:{id}"),
            SeedId::Doi(doi) => format!("DOI:{doi}"),
            SeedId::S2(id) => id.clone(),
        };
        match ctx.gateway.acquire(Endpoint::S2).await {
            Acquire::Proceed => {}
            Acquire::BudgetExhausted => {
                anyhow::bail!("S2 per-run call budget exhausted before seed resolution")
            }
        }
        match ctx.s2_client.get_paper(&query_id).await {
            Ok(Some(full)) => Ok(Some(persist_s2_hit(full, ctx).await?)),
            Ok(None) => {
                // S2 had no record. Fall back to the arXiv API for arXiv ids.
                if let SeedId::Arxiv(id) = seed {
                    resolve_arxiv(id, ctx).await
                } else {
                    Ok(None)
                }
            }
            Err(e) => {
                if let SeedId::Arxiv(id) = seed {
                    tracing::warn!(
                        error = %e.message,
                        "seed: S2 fetch failed — falling back to arXiv API"
                    );
                    resolve_arxiv(id, ctx).await
                } else {
                    anyhow::bail!("S2 fetch failed: {}", e.message)
                }
            }
        }
    } else {
        match seed {
            SeedId::Arxiv(id) => resolve_arxiv(id, ctx).await,
            SeedId::Doi(_) => {
                anyhow::bail!("DOI ids need S2_API_KEY (Semantic Scholar) — set it, or pass an arXiv id")
            }
            SeedId::S2(_) => {
                anyhow::bail!("S2 ids need S2_API_KEY (Semantic Scholar) — set it, or pass an arXiv id")
            }
        }
    }
}

/// Resolve every seed id into [`FusedHit`]s (abstracts persisted to the lit DB).
///
/// Returns `(hits, failures)`. Loud-degrade: each unresolved id becomes a
/// `"{id}: {reason}"` failure string and the batch continues. Duplicate
/// `source_id`s (e.g. an arXiv id and its S2 record) are collapsed to one hit.
pub async fn resolve_seeds(ids: &[String], ctx: &LitContext) -> (Vec<FusedHit>, Vec<String>) {
    let mut hits: Vec<FusedHit> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for raw in ids {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let seed = classify_seed(raw);
        match resolve_one(&seed, ctx).await {
            Ok(Some(hit)) => {
                if seen.insert(hit.source_id.clone()) {
                    hits.push(hit);
                } else {
                    tracing::info!(source_id = %hit.source_id, "seed: duplicate source_id — collapsed");
                }
            }
            Ok(None) => failures.push(format!("{raw}: not found")),
            Err(e) => failures.push(format!("{raw}: {e}")),
        }
    }

    (hits, failures)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_seed_routes_each_id_kind() {
        assert!(matches!(classify_seed("2310.06825"), SeedId::Arxiv(_)));
        assert!(matches!(classify_seed("2310.06825v2"), SeedId::Arxiv(_)));
        assert!(matches!(classify_seed("arXiv:1706.03762"), SeedId::Arxiv(_)));
        assert!(matches!(classify_seed("10.1145/3292500.3330701"), SeedId::Doi(_)));
        assert!(matches!(classify_seed("DOI:10.1234/foo"), SeedId::Doi(_)));
        assert!(matches!(classify_seed("s2:649def34f8be52c8b66281af98ae884c09aef38b"), SeedId::S2(_)));
        assert!(matches!(classify_seed("CorpusId:13756489"), SeedId::S2(_)));
        // A bare hash with no recognised shape defaults to S2.
        assert!(matches!(classify_seed("649def34f8be52c8b66281af98ae884c09aef38b"), SeedId::S2(_)));
    }

    #[test]
    fn classify_seed_strips_prefixes_to_bare_value() {
        match classify_seed("arXiv:1706.03762") {
            SeedId::Arxiv(id) => assert_eq!(id, "1706.03762"),
            _ => panic!("expected arxiv"),
        }
        match classify_seed("DOI:10.1234/foo") {
            SeedId::Doi(id) => assert_eq!(id, "10.1234/foo"),
            _ => panic!("expected doi"),
        }
        match classify_seed("corpusid:42") {
            SeedId::S2(id) => assert_eq!(id, "CorpusId:42"),
            _ => panic!("expected s2"),
        }
    }

    #[test]
    fn is_arxiv_new_style_accepts_and_rejects() {
        assert!(is_arxiv_new_style("2310.06825"));
        assert!(is_arxiv_new_style("1706.03762"));
        assert!(is_arxiv_new_style("2310.06825v3"));
        assert!(!is_arxiv_new_style("10.1234/foo"));
        assert!(!is_arxiv_new_style("hello"));
        assert!(!is_arxiv_new_style("231.06825"));
    }

    #[test]
    fn make_hit_keys_source_type_by_id() {
        let h = make_hit("s2:abc".into(), "T".into(), "body".into());
        assert_eq!(h.source_type, "s2");
        assert_eq!(h.relevance, 1.0);
        let h = make_hit("arxiv:2310.06825".into(), "T".into(), "body".into());
        assert_eq!(h.source_type, "arxiv");
    }
}
