//! Section-aware HTML chunker for ar5iv renders.
//!
//! Phase 21 Plan 02. Modelled on `chunking.rs` (pure transform, no I/O).
//!
//! ar5iv (ar5iv.labs.arxiv.org) renders LaTeX papers as HTML with a
//! rich section structure. This chunker exploits that structure to split
//! at semantic boundaries, returning one [`LitChunk`] per section with
//! heading text and ar5iv section ID for provenance.
//!
//! ## CSS selectors used
//!
//! - Sections: `section.ltx_section`
//! - Headings: `h2.ltx_title_section, h3.ltx_title_subsection`
//! - Paragraphs: `div.ltx_para`
//! - Document title: `h1.ltx_title_document`
//!
//! ## Fallback (Pitfall 6)
//!
//! When no `section.ltx_section` elements are found (e.g. flat-structure
//! or single-section papers), the chunker falls back to splitting by
//! `div.ltx_para` across the whole document and emits `tracing::warn!`.
//! This ensures every paper produces at least one indexable chunk.
//!
//! Do NOT use the generic token-window `chunking.rs` here — ar5iv gives
//! real section structure to exploit (locked decision).

use scraper::{Html, Selector};

// ── Public output types ───────────────────────────────────────────────────

/// One chunk from an ar5iv HTML document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LitChunk {
    /// Section heading text, e.g. `"2 Related Work"`.
    pub section: String,
    /// ar5iv section `id` attribute, e.g. `"S2"` or `"S3.SS1"`.
    /// Empty string when the section has no id (fallback path).
    pub section_id: String,
    /// Raw paragraph text, stripped of HTML tags.
    pub content: String,
    /// 0-based chunk index across the paper.
    pub chunk_index: usize,
}

/// Configuration for the section-aware chunker.
#[derive(Debug, Clone)]
pub struct LitChunkConfig {
    /// Maximum characters per chunk. Sections that exceed this are split
    /// at paragraph boundaries into multiple chunks; a single paragraph
    /// longer than this is hard-split at char boundaries. Keeping
    /// oversized chunks whole would push them past the embedder's token
    /// limit (Jina: 8194 tokens) and drop them from the vector lane.
    pub max_chars: usize,
}

impl Default for LitChunkConfig {
    fn default() -> Self {
        Self { max_chars: 8_000 }
    }
}

// ── Public API ────────────────────────────────────────────────────────────

/// Split an ar5iv HTML document into section-boundary chunks.
///
/// Returns `Vec<LitChunk>` in document order. Empty HTML returns empty vec.
///
/// When `section.ltx_section` elements are absent the function falls back to
/// paragraph-level chunking and emits a `tracing::warn!` so the caller can
/// surface it.
pub fn chunk_ar5iv_html(html: &str, config: &LitChunkConfig) -> Vec<LitChunk> {
    if html.trim().is_empty() {
        return Vec::new();
    }

    let document = Html::parse_document(html);
    let mut chunks = Vec::new();

    // ── Selectors ─────────────────────────────────────────────────────────
    let sel_section = Selector::parse("section.ltx_section").expect("valid selector");
    let sel_heading = Selector::parse(
        "h2.ltx_title_section, h3.ltx_title_subsection, h1.ltx_title_document",
    )
    .expect("valid selector");
    let sel_para = Selector::parse("div.ltx_para").expect("valid selector");
    let sel_doc_title = Selector::parse("h1.ltx_title_document").expect("valid selector");

    let sections: Vec<_> = document.select(&sel_section).collect();

    if sections.is_empty() {
        // ── Fallback: paragraph-level chunking ────────────────────────────
        let doc_title = document
            .select(&sel_doc_title)
            .next()
            .map(|el| element_text(&el))
            .unwrap_or_else(|| "document".to_string());

        tracing::warn!(
            "ar5iv: no ltx_section elements; falling back to paragraph-level chunking"
        );

        let mut chunk_index = 0usize;
        for para in document.select(&sel_para) {
            let content = element_text(&para);
            if content.trim().is_empty() {
                continue;
            }
            for piece in split_oversized(&content, config.max_chars) {
                chunks.push(LitChunk {
                    section: doc_title.clone(),
                    section_id: String::new(),
                    content: piece,
                    chunk_index,
                });
                chunk_index += 1;
            }
        }
        return chunks;
    }

    // ── Section-aware chunking ─────────────────────────────────────────────
    let mut chunk_index = 0usize;

    for section in &sections {
        // Section id attribute (e.g. "S1", "S2", "S3.SS1")
        let section_id = section
            .value()
            .attr("id")
            .unwrap_or("")
            .to_string();

        // Section heading: first matching heading element inside this section
        let heading = section
            .select(&sel_heading)
            .next()
            .map(|el| element_text(&el))
            .unwrap_or_else(|| section_id.clone());

        // Concatenate all div.ltx_para text content inside this section
        let mut content_parts: Vec<String> = Vec::new();
        for para in section.select(&sel_para) {
            let text = element_text(&para);
            if !text.trim().is_empty() {
                content_parts.push(text);
            }
        }

        if content_parts.iter().all(|p| p.trim().is_empty()) {
            continue;
        }

        for piece in pack_paragraphs(&content_parts, config.max_chars) {
            chunks.push(LitChunk {
                section: heading.clone(),
                section_id: section_id.clone(),
                content: piece,
                chunk_index,
            });
            chunk_index += 1;
        }
    }

    chunks
}

// ── Plain-text chunker (F10 PDF lane) ────────────────────────────────────────

/// Split pdftotext-extracted plain text into page-section chunks.
///
/// `pdftotext -layout` emits form feeds (`\x0c`) between pages. Each page's
/// paragraphs are split on blank lines. The references/bibliography section
/// and everything after it is trimmed — bibliographies must not enter the
/// embed corpus.
///
/// References trim rule: scan paragraphs in document order and STOP at the
/// first paragraph whose **first line**, trimmed and lowercased, matches:
/// - `"references"` or `"bibliography"` (bare)
/// - optionally preceded by a numeral/roman numeral and punctuation,
///   e.g. `"7 references"`, `"VII. bibliography"`, `"10. References"`.
///
/// The first-line match prevents a paragraph that merely *mentions*
/// "references" mid-sentence from triggering the trim.
///
/// Chunk fields:
/// - `section = "PDF page {n}"` (1-based) — this label IS the PDF-source
///   marker for the F10 decision metric query
///   `SELECT DISTINCT paper_id FROM lit_chunks WHERE section LIKE 'PDF page %'`.
/// - `section_id = ""` (no structural id available from plain text).
/// - `chunk_index` — sequential across the whole document.
///
/// Empty/whitespace text returns an empty vec.
pub fn chunk_plain_text(text: &str, config: &LitChunkConfig) -> Vec<LitChunk> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    // Split on form feeds to get pages.
    let pages: Vec<&str> = text.split('\x0c').collect();
    let mut chunks = Vec::new();
    let mut chunk_index = 0usize;
    let mut references_hit = false;

    for (page_idx, page_text) in pages.iter().enumerate() {
        if references_hit {
            break;
        }
        let page_num = page_idx + 1; // 1-based

        // Split page into paragraphs on blank lines.
        let paragraphs: Vec<String> = split_paragraphs(page_text);

        // Collect paragraphs for this page, stopping at references heading.
        let mut page_parts: Vec<String> = Vec::new();
        for para in &paragraphs {
            let trimmed = para.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Check first line of this paragraph for a references/bibliography heading.
            let first_line = trimmed.lines().next().unwrap_or("").trim().to_lowercase();
            if is_references_heading(&first_line) {
                references_hit = true;
                break;
            }
            page_parts.push(para.clone());
        }

        if page_parts.is_empty() {
            continue;
        }

        for piece in pack_paragraphs(&page_parts, config.max_chars) {
            chunks.push(LitChunk {
                section: format!("PDF page {page_num}"),
                section_id: String::new(),
                content: piece,
                chunk_index,
            });
            chunk_index += 1;
        }
    }

    chunks
}

/// Return true if `first_line` (already trimmed + lowercased) is a
/// references or bibliography heading.
///
/// Matches:
/// - `"references"` / `"bibliography"` (bare)
/// - Optional prefix: digits, roman numerals (I-X), followed by spaces +
///   optional `.` or `)`, e.g. `"7 references"`, `"vii. bibliography"`.
fn is_references_heading(first_line: &str) -> bool {
    // Strip optional leading numeral/roman prefix.
    let candidate = strip_leading_numeral(first_line.trim());
    candidate == "references" || candidate == "bibliography"
}

/// Strip a leading decimal or roman numeral + optional punctuation from `s`.
/// Returns `s` unchanged when no numeric prefix is found.
fn strip_leading_numeral(s: &str) -> &str {
    // Decimal prefix: one or more digits, optional `.` or `)`, then a space.
    let s = if let Some(rest) = try_strip_decimal(s) { rest } else { s };
    // Roman numeral prefix: I/V/X/L/C/D/M chars, optional `.` or `)`, then space.
    let s = if let Some(rest) = try_strip_roman(s) { rest } else { s };
    s.trim()
}

fn try_strip_decimal(s: &str) -> Option<&str> {
    let mut end = 0;
    let bytes = s.as_bytes();
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 {
        return None;
    }
    let rest = &s[end..];
    let rest = rest.trim_start_matches(['.', ')']);
    if rest.starts_with(' ') || rest.is_empty() {
        Some(rest.trim_start())
    } else {
        None
    }
}

fn try_strip_roman(s: &str) -> Option<&str> {
    // Only recognise lower-case roman chars (first_line is already lowercased).
    let roman_chars: &[char] = &['i', 'v', 'x', 'l', 'c', 'd', 'm'];
    let mut end = 0;
    for ch in s.chars() {
        if roman_chars.contains(&ch) {
            end += ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    let rest = &s[end..];
    let rest = rest.trim_start_matches(['.', ')']);
    if rest.starts_with(' ') || rest.is_empty() {
        Some(rest.trim_start())
    } else {
        None
    }
}

/// Split plain text into paragraphs on blank lines (consecutive newlines with
/// only whitespace between them).
fn split_paragraphs(text: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current = String::new();
    let mut prev_blank = false;

    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.trim().is_empty() {
                if prev_blank {
                    // Two blank lines in a row: paragraph boundary.
                    paragraphs.push(std::mem::take(&mut current).trim().to_string());
                }
            }
            prev_blank = true;
        } else {
            if prev_blank && !current.trim().is_empty() {
                paragraphs.push(std::mem::take(&mut current).trim().to_string());
            }
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
            prev_blank = false;
        }
    }
    if !current.trim().is_empty() {
        paragraphs.push(current.trim().to_string());
    }
    paragraphs
}

// ── Oversize handling ──────────────────────────────────────────────────────

/// Greedily pack paragraphs into chunks of at most `max_chars` characters,
/// joining packed paragraphs with `"\n\n"`. A single paragraph longer than
/// `max_chars` is hard-split via [`split_oversized`].
fn pack_paragraphs(parts: &[String], max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for part in parts {
        if part.len() > max_chars {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.extend(split_oversized(part, max_chars));
            continue;
        }
        let sep = if current.is_empty() { 0 } else { 2 };
        if current.len() + sep + part.len() > max_chars {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(part);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Split `text` into pieces of at most `max_chars` characters at UTF-8
/// char boundaries. Emits one `tracing::warn!` per split so oversized
/// source material stays visible in the intake logs.
fn split_oversized(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }
    tracing::warn!(
        "ar5iv: chunk exceeds max_chars ({} > {}); splitting at char boundaries",
        text.len(),
        max_chars
    );
    let mut pieces = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        let mut end = (start + max_chars).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        pieces.push(text[start..end].to_string());
        start = end;
    }
    pieces
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Extract all visible text from an element, stripping HTML tags.
fn element_text(element: &scraper::ElementRef) -> String {
    element.text().collect::<Vec<_>>().join(" ").trim().to_string()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LitChunkConfig {
        LitChunkConfig::default()
    }

    // ── Test 1: section-aware chunking ─────────────────────────────────────

    #[test]
    fn section_aware_chunks_well_structured_html() {
        let html = r#"<!DOCTYPE html>
<html>
<body>
<h1 class="ltx_title_document">My Paper</h1>
<section id="S1" class="ltx_section">
  <h2 class="ltx_title_section">1 Introduction</h2>
  <div class="ltx_para">First paragraph of the introduction.</div>
  <div class="ltx_para">Second paragraph.</div>
</section>
<section id="S2" class="ltx_section">
  <h2 class="ltx_title_section">2 Related Work</h2>
  <div class="ltx_para">Prior work paragraph.</div>
</section>
<section id="S3" class="ltx_section">
  <h3 class="ltx_title_subsection">3.1 Method Details</h3>
  <div class="ltx_para">Method details here.</div>
</section>
</body>
</html>"#;

        let chunks = chunk_ar5iv_html(html, &cfg());
        assert!(
            chunks.len() >= 2,
            "expected at least 2 section chunks, got {}",
            chunks.len()
        );

        // First chunk: section S1
        let s1 = chunks.iter().find(|c| c.section_id == "S1").expect("S1 chunk");
        assert!(
            s1.section.contains("Introduction"),
            "section heading: {}",
            s1.section
        );
        assert!(s1.content.contains("introduction"), "content: {}", s1.content);

        // Second chunk: section S2
        let s2 = chunks.iter().find(|c| c.section_id == "S2").expect("S2 chunk");
        assert!(
            s2.section.contains("Related"),
            "section heading: {}",
            s2.section
        );

        // chunk_index is 0-based and sequential
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
    }

    // ── Test 2: paragraph-level fallback ──────────────────────────────────

    #[test]
    fn falls_back_to_paragraph_chunking_when_no_sections() {
        let html = r#"<!DOCTYPE html>
<html>
<body>
<h1 class="ltx_title_document">Flat Paper</h1>
<div class="ltx_para">First flat paragraph.</div>
<div class="ltx_para">Second flat paragraph.</div>
<div class="ltx_para">Third flat paragraph.</div>
</body>
</html>"#;

        let chunks = chunk_ar5iv_html(html, &cfg());
        assert!(
            !chunks.is_empty(),
            "fallback should produce at least one chunk"
        );
        // All chunks carry the document title as the section heading
        for chunk in &chunks {
            assert_eq!(
                chunk.section, "Flat Paper",
                "expected doc title as section, got: {}",
                chunk.section
            );
            assert!(
                chunk.section_id.is_empty(),
                "fallback chunks have empty section_id"
            );
        }
        // Paragraph content is preserved
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join(" ");
        assert!(all_content.contains("First flat"), "content: {all_content}");
    }

    // ── Test 3: abstract section extracted ────────────────────────────────

    #[test]
    fn abstract_section_extracted() {
        let html = r#"<!DOCTYPE html>
<html>
<body>
<h1 class="ltx_title_document">Test Paper</h1>
<section id="Sx1" class="ltx_section">
  <h6 class="ltx_title_abstract">Abstract</h6>
  <div class="ltx_para">This paper presents a novel approach to chunking.</div>
</section>
<section id="S1" class="ltx_section">
  <h2 class="ltx_title_section">1 Introduction</h2>
  <div class="ltx_para">We introduce our method.</div>
</section>
</body>
</html>"#;

        let chunks = chunk_ar5iv_html(html, &cfg());
        assert!(
            chunks.len() >= 2,
            "expected abstract + intro chunks, got {}",
            chunks.len()
        );
        // The abstract section is captured
        let abstract_chunk = chunks.iter().find(|c| c.section_id == "Sx1");
        assert!(abstract_chunk.is_some(), "abstract chunk Sx1 not found");
        let abstract_chunk = abstract_chunk.unwrap();
        assert!(
            abstract_chunk.content.contains("chunking"),
            "abstract content: {}",
            abstract_chunk.content
        );
    }

    // ── Test 4: empty HTML returns empty vec ──────────────────────────────

    #[test]
    fn empty_html_returns_empty_chunks() {
        assert!(chunk_ar5iv_html("", &cfg()).is_empty());
        assert!(chunk_ar5iv_html("   ", &cfg()).is_empty());
    }

    // ── Test 5: oversized section splits at paragraph boundaries ──────────

    #[test]
    fn oversized_section_splits_at_paragraph_boundaries() {
        let para = "x".repeat(3_000);
        let html = format!(
            r#"<html><body>
<section id="S1" class="ltx_section">
  <h2 class="ltx_title_section">1 Big Section</h2>
  <div class="ltx_para">{para}</div>
  <div class="ltx_para">{para}</div>
  <div class="ltx_para">{para}</div>
  <div class="ltx_para">{para}</div>
</section>
</body></html>"#
        );

        let config = LitChunkConfig { max_chars: 8_000 };
        let chunks = chunk_ar5iv_html(&html, &config);

        assert!(
            chunks.len() >= 2,
            "12k chars of paragraphs must split into >1 chunk, got {}",
            chunks.len()
        );
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.content.len() <= config.max_chars,
                "chunk {} is {} chars, exceeds max {}",
                i,
                chunk.content.len(),
                config.max_chars
            );
            assert_eq!(chunk.section_id, "S1", "split chunks keep the section_id");
            assert_eq!(chunk.chunk_index, i, "chunk_index stays sequential");
        }
        // No content lost: total non-separator chars equal the input paragraphs.
        let total: usize = chunks
            .iter()
            .map(|c| c.content.matches('x').count())
            .sum();
        assert_eq!(total, 12_000, "all paragraph content survives the split");
    }

    // ── Test 6: single oversized paragraph hard-splits UTF-8-safe ─────────

    #[test]
    fn oversized_single_paragraph_hard_splits_utf8_safe() {
        // Multibyte chars (é = 2 bytes) so a naive byte slice would panic.
        let para = "é".repeat(6_000); // 12_000 bytes
        let html = format!(
            r#"<html><body>
<section id="S2" class="ltx_section">
  <h2 class="ltx_title_section">2 Huge Paragraph</h2>
  <div class="ltx_para">{para}</div>
</section>
</body></html>"#
        );

        let config = LitChunkConfig { max_chars: 8_000 };
        let chunks = chunk_ar5iv_html(&html, &config);

        assert!(chunks.len() >= 2, "12k-byte paragraph must hard-split");
        for chunk in &chunks {
            assert!(chunk.content.len() <= config.max_chars);
        }
        let total: usize = chunks.iter().map(|c| c.content.chars().count()).sum();
        assert_eq!(total, 6_000, "no chars lost in the hard split");
    }

    // ── Test 7: section_id attribute captured correctly ───────────────────

    #[test]
    fn section_id_attribute_captured() {
        let html = r#"<html><body>
<section id="S3.SS1" class="ltx_section">
  <h3 class="ltx_title_subsection">3.1 Sub-method</h3>
  <div class="ltx_para">Sub-method content.</div>
</section>
</body></html>"#;

        let chunks = chunk_ar5iv_html(html, &cfg());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section_id, "S3.SS1");
    }

    // ── chunk_plain_text tests (F10 PDF lane) ─────────────────────────────────

    /// Two-page document: page split on \x0c yields "PDF page 1"/"PDF page 2"
    /// sections with sequential chunk_index.
    #[test]
    fn plain_text_two_pages_produce_correct_sections() {
        let text = "First page content.\n\nMore first page.\x0cSecond page content.\n\nMore second.";
        let chunks = super::chunk_plain_text(text, &cfg());
        assert!(chunks.len() >= 2, "must produce at least 2 chunks for 2 pages");
        let page1: Vec<_> = chunks.iter().filter(|c| c.section == "PDF page 1").collect();
        let page2: Vec<_> = chunks.iter().filter(|c| c.section == "PDF page 2").collect();
        assert!(!page1.is_empty(), "page 1 chunks must have section 'PDF page 1'");
        assert!(!page2.is_empty(), "page 2 chunks must have section 'PDF page 2'");
        // Verify sequential chunk_index across pages.
        let indices: Vec<usize> = chunks.iter().map(|c| c.chunk_index).collect();
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, i, "chunk_index must be sequential: expected {i}, got {idx}");
        }
        // All section_ids must be empty (no structural IDs in plain text).
        for chunk in &chunks {
            assert!(chunk.section_id.is_empty(), "plain-text chunks must have empty section_id");
        }
    }

    /// References trim: content after a "References" heading paragraph is absent;
    /// case-insensitive; and "7 References" variant works.
    #[test]
    fn plain_text_references_trim() {
        // "References" bare — case insensitive.
        let text = "Introduction text.\n\nMore intro.\n\nreferences\n\nBiblio entry 1.\n\nBiblio entry 2.";
        let chunks = super::chunk_plain_text(text, &cfg());
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join(" ");
        assert!(all_content.contains("Introduction"), "pre-references content must be present");
        assert!(!all_content.contains("Biblio"), "bibliography content must be trimmed");

        // "7 References" variant.
        let text2 = "Section content.\n\n7 References\n\nRef 1.\n\nRef 2.";
        let chunks2 = super::chunk_plain_text(text2, &cfg());
        let all2: String = chunks2.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join(" ");
        assert!(all2.contains("Section content"), "pre-references content must survive");
        assert!(!all2.contains("Ref 1"), "references must be trimmed");

        // "Bibliography" variant.
        let text3 = "Body.\n\nBibliography\n\nEntry 1.";
        let chunks3 = super::chunk_plain_text(text3, &cfg());
        let all3: String = chunks3.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join(" ");
        assert!(!all3.contains("Entry 1"), "'Bibliography' heading must trigger trim");
    }

    /// "bibliography" mid-sentence inside a paragraph body does NOT trigger trim
    /// (first-line match only).
    #[test]
    fn plain_text_bibliography_mid_sentence_does_not_trim() {
        // The word "bibliography" appears mid-sentence in the paragraph body —
        // it is NOT the first line of the paragraph.
        let text = "This paper discusses the bibliography of prior work extensively.\n\nMore content here.";
        let chunks = super::chunk_plain_text(text, &cfg());
        let all: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join(" ");
        assert!(all.contains("More content here"), "mid-sentence bibliography must not trim");
    }

    /// Oversized paragraph respects max_chars via pack/split reuse.
    #[test]
    fn plain_text_oversized_paragraph_respects_max_chars() {
        let big_para = "x".repeat(10_000);
        let text = format!("{big_para}\n\nSome other paragraph.");
        let config = LitChunkConfig { max_chars: 8_000 };
        let chunks = super::chunk_plain_text(&text, &config);
        for chunk in &chunks {
            assert!(
                chunk.content.len() <= config.max_chars,
                "chunk {} exceeds max_chars: len={}",
                chunk.chunk_index,
                chunk.content.len()
            );
        }
        assert!(!chunks.is_empty(), "oversized paragraph must produce at least one chunk");
    }

    /// Empty input returns empty vec.
    #[test]
    fn plain_text_empty_input_returns_empty() {
        assert!(super::chunk_plain_text("", &cfg()).is_empty());
        assert!(super::chunk_plain_text("   ", &cfg()).is_empty());
        assert!(super::chunk_plain_text("\x0c\x0c\x0c", &cfg()).is_empty());
    }
}
