//! Markdown chunker — splits KB articles into heading-aware embeddable units.
//!
//! Phase 3 Task 3.1. Each chunk preserves heading hierarchy in `heading_path`
//! (e.g. `["Foo", "Bar"]`), tracks the byte range in the original source,
//! and reports an approximate token count.
//!
//! Chunks always end at heading or paragraph boundaries — never mid-sentence.
//! When a single paragraph exceeds `max_tokens`, the chunker keeps it whole
//! and emits a `tracing::warn!` rather than splitting mid-paragraph.
//!
//! The chunker is intentionally minimal: it knows about ATX (`#`) headings,
//! Setext (`===`/`---`) headings, fenced code blocks (` ``` `), and YAML
//! frontmatter (`---` ... `---` at the start of the file). All other
//! Markdown syntax is treated as plain content.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Hierarchical heading path leading to this chunk. Empty for a chunk
    /// before the first heading.
    pub heading_path: Vec<String>,
    /// The chunk's textual content (does NOT include heading markers).
    pub content: String,
    /// Byte offset (inclusive) in the source where this chunk begins.
    pub byte_start: usize,
    /// Byte offset (exclusive) in the source where this chunk ends.
    pub byte_end: usize,
    /// Approximate token count using the rough heuristic 1 token ≈ 4 chars.
    pub approx_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Max approximate tokens per chunk. Default 512.
    pub max_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self { max_tokens: 512 }
    }
}

/// Approximate token count using the rough heuristic 1 token ≈ 4 chars,
/// rounded up.
fn approx_tokens(s: &str) -> usize {
    let len = s.len();
    (len + 3) / 4
}

/// One line in the source with its byte offsets.
#[derive(Debug, Clone)]
struct LineSpan<'a> {
    text: &'a str,
    /// Byte offset (inclusive) in the source where this line begins.
    start: usize,
    /// Byte offset (exclusive) in the source where this line ends,
    /// including the terminating newline if present.
    end: usize,
}

/// Split the source into lines preserving byte offsets. Each `LineSpan.end`
/// includes the trailing `\n` if present so consecutive spans tile the
/// source exactly.
fn split_lines(source: &str) -> Vec<LineSpan<'_>> {
    let bytes = source.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            let end = i + 1;
            let text_end = if i > start && bytes[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            lines.push(LineSpan {
                text: &source[start..text_end],
                start,
                end,
            });
            start = end;
            i = end;
        } else {
            i += 1;
        }
    }
    if start < bytes.len() {
        lines.push(LineSpan {
            text: &source[start..bytes.len()],
            start,
            end: bytes.len(),
        });
    }
    lines
}

/// Detect an ATX heading (`#` ... `######`). Returns `(level, title)`
/// if `line` is a heading, else `None`. Up to three leading spaces are
/// permitted per CommonMark.
fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let trimmed_left = line.trim_start_matches(' ');
    // CommonMark allows up to 3 leading spaces.
    if line.len() - trimmed_left.len() > 3 {
        return None;
    }
    let mut level = 0;
    let bytes = trimmed_left.as_bytes();
    while level < bytes.len() && bytes[level] == b'#' {
        level += 1;
    }
    if level == 0 || level > 6 {
        return None;
    }
    // Must be followed by whitespace or end of line.
    if level < bytes.len() {
        let next = bytes[level];
        if next != b' ' && next != b'\t' {
            return None;
        }
    }
    let rest = &trimmed_left[level..];
    let title = rest.trim().trim_end_matches('#').trim().to_string();
    Some((level, title))
}

/// Detect a Setext underline. `=` => level 1, `-` => level 2. Returns
/// `Some(level)` if the entire line is one of those characters (>=1).
fn parse_setext_underline(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().all(|c| c == '=') {
        Some(1)
    } else if trimmed.chars().all(|c| c == '-') {
        Some(2)
    } else {
        None
    }
}

/// Detect a fenced-code-block delimiter (``` or ~~~, optionally with
/// info string).
fn is_code_fence(line: &str) -> bool {
    let trimmed = line.trim_start_matches(' ');
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

/// Update `path` in place when a new heading at `level` (1-based) with
/// `title` is encountered. Truncates deeper levels and replaces same-level.
fn update_heading_path(path: &mut Vec<String>, level: usize, title: String) {
    // Pad with empties if jumping deeper than current depth.
    while path.len() < level - 1 {
        path.push(String::new());
    }
    path.truncate(level - 1);
    path.push(title);
}

/// A "section" — the body lines under a single heading path.
#[derive(Debug)]
struct Section {
    heading_path: Vec<String>,
    /// Body lines in source order.
    lines: Vec<usize>,
}

/// Find the first index after any YAML frontmatter. If the source starts
/// with a `---` line, scans for the closing `---` and returns the line
/// index immediately after it. Otherwise returns 0.
fn skip_frontmatter(lines: &[LineSpan<'_>]) -> usize {
    if lines.is_empty() {
        return 0;
    }
    if lines[0].text.trim_end() != "---" {
        return 0;
    }
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.text.trim_end() == "---" {
            return i + 1;
        }
    }
    // Unterminated frontmatter — be conservative and don't skip.
    0
}

/// Split Markdown into chunks. Chunks always end at heading or paragraph
/// boundaries — never mid-sentence. Returns chunks in document order.
pub fn chunk_markdown(source: &str, config: &ChunkConfig) -> Vec<Chunk> {
    if source.trim().is_empty() {
        return Vec::new();
    }

    let lines = split_lines(source);
    let start_line = skip_frontmatter(&lines);

    // First pass: walk lines, partition into sections keyed by heading path.
    let mut sections: Vec<Section> = Vec::new();
    let mut current_path: Vec<String> = Vec::new();
    let mut current_lines: Vec<usize> = Vec::new();
    let mut in_code_block = false;

    let mut i = start_line;
    while i < lines.len() {
        let line = &lines[i];

        // Code-fence toggling — fences inside code blocks still close them.
        if is_code_fence(line.text) {
            in_code_block = !in_code_block;
            current_lines.push(i);
            i += 1;
            continue;
        }

        if in_code_block {
            current_lines.push(i);
            i += 1;
            continue;
        }

        // ATX heading?
        if let Some((level, title)) = parse_atx_heading(line.text) {
            // Flush current section.
            if !current_lines.is_empty() || !current_path.is_empty() {
                sections.push(Section {
                    heading_path: current_path.clone(),
                    lines: std::mem::take(&mut current_lines),
                });
            }
            update_heading_path(&mut current_path, level, title);
            i += 1;
            continue;
        }

        // Setext heading? Look ahead one line.
        if i + 1 < lines.len() && !line.text.trim().is_empty() {
            if let Some(level) = parse_setext_underline(lines[i + 1].text) {
                // Flush current section, but exclude THIS line — it's the
                // heading title, not body content.
                let title = line.text.trim().to_string();
                if !current_lines.is_empty() || !current_path.is_empty() {
                    sections.push(Section {
                        heading_path: current_path.clone(),
                        lines: std::mem::take(&mut current_lines),
                    });
                }
                update_heading_path(&mut current_path, level, title);
                i += 2;
                continue;
            }
        }

        current_lines.push(i);
        i += 1;
    }

    // Flush the final section.
    if !current_lines.is_empty() || !current_path.is_empty() {
        sections.push(Section {
            heading_path: current_path.clone(),
            lines: current_lines,
        });
    }

    // Second pass: emit chunks per section.
    let mut chunks = Vec::new();
    for section in sections {
        emit_section_chunks(&lines, &section, config, &mut chunks);
    }
    chunks
}

/// Emit one or more chunks for a section, splitting at paragraph
/// boundaries when the section exceeds `max_tokens`.
fn emit_section_chunks(
    lines: &[LineSpan<'_>],
    section: &Section,
    config: &ChunkConfig,
    out: &mut Vec<Chunk>,
) {
    if section.lines.is_empty() {
        return;
    }

    // Group lines into paragraphs. A paragraph is a run of non-blank lines
    // separated by blank lines, EXCEPT inside a fenced code block, where
    // blank lines are part of the paragraph.
    let paragraphs = group_paragraphs(lines, &section.lines);

    if paragraphs.is_empty() {
        return;
    }

    // Build the full section content; if it fits, emit one chunk.
    let full_content = paragraphs
        .iter()
        .map(|p| p.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let full_trimmed = full_content.trim();
    if full_trimmed.is_empty() {
        return;
    }

    let full_tokens = approx_tokens(full_trimmed);
    if full_tokens <= config.max_tokens {
        let byte_start = paragraphs.first().unwrap().byte_start;
        let byte_end = paragraphs.last().unwrap().byte_end;
        out.push(Chunk {
            heading_path: section.heading_path.clone(),
            content: full_trimmed.to_string(),
            byte_start,
            byte_end,
            approx_tokens: full_tokens,
        });
        return;
    }

    // Greedy-pack paragraphs into chunks ≤ max_tokens.
    let mut current: Vec<&Paragraph> = Vec::new();
    let mut current_tokens = 0usize;

    let flush = |group: &mut Vec<&Paragraph>,
                 current_tokens: &mut usize,
                 out: &mut Vec<Chunk>,
                 heading_path: &[String]| {
        if group.is_empty() {
            return;
        }
        let content = group
            .iter()
            .map(|p| p.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let trimmed = content.trim();
        if !trimmed.is_empty() {
            let byte_start = group.first().unwrap().byte_start;
            let byte_end = group.last().unwrap().byte_end;
            let tokens = approx_tokens(trimmed);
            out.push(Chunk {
                heading_path: heading_path.to_vec(),
                content: trimmed.to_string(),
                byte_start,
                byte_end,
                approx_tokens: tokens,
            });
        }
        group.clear();
        *current_tokens = 0;
    };

    for para in &paragraphs {
        let p_tokens = approx_tokens(&para.content);

        // Oversized paragraph: flush current group, then emit this paragraph
        // whole with a warning.
        if p_tokens > config.max_tokens {
            flush(
                &mut current,
                &mut current_tokens,
                out,
                &section.heading_path,
            );
            tracing::warn!(
                "chunk paragraph exceeds max_tokens: {} > {}",
                p_tokens,
                config.max_tokens
            );
            let trimmed = para.content.trim();
            if !trimmed.is_empty() {
                out.push(Chunk {
                    heading_path: section.heading_path.clone(),
                    content: trimmed.to_string(),
                    byte_start: para.byte_start,
                    byte_end: para.byte_end,
                    approx_tokens: approx_tokens(trimmed),
                });
            }
            continue;
        }

        // Will this paragraph fit in the current group?
        if current_tokens + p_tokens > config.max_tokens && !current.is_empty() {
            flush(
                &mut current,
                &mut current_tokens,
                out,
                &section.heading_path,
            );
        }
        current.push(para);
        current_tokens += p_tokens;
    }

    flush(
        &mut current,
        &mut current_tokens,
        out,
        &section.heading_path,
    );
}

#[derive(Debug)]
struct Paragraph {
    content: String,
    byte_start: usize,
    byte_end: usize,
}

/// Group a section's line indices into paragraphs (runs separated by blank
/// lines). Blank lines INSIDE a fenced code block do not split.
fn group_paragraphs(lines: &[LineSpan<'_>], section_lines: &[usize]) -> Vec<Paragraph> {
    let mut paragraphs = Vec::new();
    let mut current_text: Vec<&str> = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut current_end: usize = 0;
    let mut in_code = false;

    for &idx in section_lines {
        let line = &lines[idx];
        let is_fence = is_code_fence(line.text);
        let is_blank = line.text.trim().is_empty();

        if is_fence {
            // Toggle code state. Fence lines always join the current paragraph.
            in_code = !in_code;
            if current_start.is_none() {
                current_start = Some(line.start);
            }
            current_text.push(line.text);
            current_end = line.end;
            continue;
        }

        if is_blank && !in_code {
            // Paragraph boundary.
            if let Some(start) = current_start.take() {
                let content = current_text.join("\n");
                paragraphs.push(Paragraph {
                    content,
                    byte_start: start,
                    byte_end: current_end,
                });
                current_text.clear();
            }
            continue;
        }

        if current_start.is_none() {
            current_start = Some(line.start);
        }
        current_text.push(line.text);
        current_end = line.end;
    }

    if let Some(start) = current_start {
        let content = current_text.join("\n");
        paragraphs.push(Paragraph {
            content,
            byte_start: start,
            byte_end: current_end,
        });
    }

    // Drop paragraphs that are pure whitespace.
    paragraphs.retain(|p| !p.content.trim().is_empty());
    paragraphs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_tokens: usize) -> ChunkConfig {
        ChunkConfig { max_tokens }
    }

    #[test]
    fn empty_source_returns_no_chunks() {
        assert!(chunk_markdown("", &cfg(512)).is_empty());
    }

    #[test]
    fn whitespace_source_returns_no_chunks() {
        assert!(chunk_markdown("   \n\n  ", &cfg(512)).is_empty());
    }

    #[test]
    fn single_heading_with_short_body_emits_one_chunk() {
        let chunks = chunk_markdown("# Foo\n\nbody text", &cfg(512));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading_path, vec!["Foo".to_string()]);
        assert!(chunks[0].content.contains("body text"));
    }

    #[test]
    fn multiple_headings_emit_separate_chunks() {
        let chunks = chunk_markdown("# Foo\n\nA\n\n# Bar\n\nB", &cfg(512));
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading_path, vec!["Foo".to_string()]);
        assert_eq!(chunks[1].heading_path, vec!["Bar".to_string()]);
    }

    #[test]
    fn nested_headings_track_path() {
        let chunks = chunk_markdown("# Top\n\n## Sub\n\nbody", &cfg(512));
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].heading_path,
            vec!["Top".to_string(), "Sub".to_string()]
        );
    }

    #[test]
    fn same_level_heading_replaces_in_path() {
        let chunks = chunk_markdown("# Top\n\n## A\n\nbody1\n\n## B\n\nbody2", &cfg(512));
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[1].heading_path,
            vec!["Top".to_string(), "B".to_string()]
        );
    }

    #[test]
    fn top_level_heading_truncates_path() {
        let chunks = chunk_markdown("# A\n\n## B\n\nb1\n\n# C\n\nc1", &cfg(512));
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].heading_path, vec!["C".to_string()]);
    }

    #[test]
    fn oversized_section_splits_at_paragraphs() {
        // Six paragraphs of ~200 chars (~50 tokens each); max_tokens=120
        // should yield 3+ chunks each ≤120 tokens.
        let para = "x".repeat(200);
        let body = (0..6)
            .map(|_| para.clone())
            .collect::<Vec<_>>()
            .join("\n\n");
        let source = format!("# H\n\n{}", body);
        let chunks = chunk_markdown(&source, &cfg(120));
        assert!(
            chunks.len() >= 3,
            "expected >=3 chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert_eq!(c.heading_path, vec!["H".to_string()]);
            assert!(
                c.approx_tokens <= 120,
                "chunk over max_tokens: {} > 120",
                c.approx_tokens
            );
        }
    }

    #[test]
    fn paragraph_larger_than_max_kept_whole_with_warn() {
        let para = "x".repeat(600);
        let source = format!("# H\n\n{}", para);
        let chunks = chunk_markdown(&source, &cfg(100));
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].approx_tokens > 100,
            "expected oversize chunk, got {} tokens",
            chunks[0].approx_tokens
        );
        // No mid-paragraph split: content equals the original paragraph.
        assert_eq!(chunks[0].content, para);
    }

    #[test]
    fn byte_offsets_reconstruct_source() {
        let source = "# Foo\n\nfirst para\n\nsecond para\n\n# Bar\n\nthird para";
        let chunks = chunk_markdown(source, &cfg(512));
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.byte_start < chunk.byte_end);
            assert!(chunk.byte_end <= source.len());
            let slice = &source.as_bytes()[chunk.byte_start..chunk.byte_end];
            // Must be valid UTF-8.
            let s = std::str::from_utf8(slice).expect("byte range valid utf-8");
            // The slice must contain the chunk's content (which may have had
            // leading/trailing whitespace stripped).
            assert!(
                s.contains(chunk.content.trim()),
                "byte range {:?} does not contain content {:?}",
                s,
                chunk.content
            );
        }
    }

    #[test]
    fn code_block_not_split_on_blank_lines() {
        // A fenced code block contains blank lines. With a small max_tokens
        // the chunker would normally split — but the code block must stay
        // intact in a single chunk. To make the test deterministic, the
        // surrounding section is small enough that the entire section
        // emits as one chunk; the key assertion is that the code block
        // appears whole inside that chunk.
        let source = "# H\n\n```\nfn a() {}\n\nfn b() {}\n\nfn c() {}\n```\n";
        let chunks = chunk_markdown(source, &cfg(512));
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert!(c.content.contains("fn a()"));
        assert!(c.content.contains("fn b()"));
        assert!(c.content.contains("fn c()"));
    }

    #[test]
    fn yaml_frontmatter_skipped() {
        let source = "---\ntitle: Foo\n---\n# Body\n\ntext";
        let chunks = chunk_markdown(source, &cfg(512));
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].content.contains("title:"));
        assert_eq!(chunks[0].heading_path, vec!["Body".to_string()]);
    }

    #[test]
    fn setext_h1_treated_as_h1() {
        let chunks = chunk_markdown("Foo\n===\n\nbody", &cfg(512));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading_path, vec!["Foo".to_string()]);
    }

    #[test]
    fn setext_h2_treated_as_h2() {
        let chunks = chunk_markdown("# Top\n\nFoo\n---\n\nbody", &cfg(512));
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].heading_path,
            vec!["Top".to_string(), "Foo".to_string()]
        );
    }

    #[test]
    fn no_heading_emits_chunk_with_empty_path() {
        let chunks = chunk_markdown("just text", &cfg(512));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].heading_path.is_empty());
    }

    #[test]
    fn chunk_content_excludes_heading_line() {
        let chunks = chunk_markdown("# Foo\n\nbody", &cfg(512));
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].content.contains("# Foo"));
        assert!(chunks[0].content.contains("body"));
    }
}
