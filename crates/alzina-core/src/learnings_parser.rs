//! Shared learnings file parser (A-25).
//!
//! Canonical parsing logic for `- entry\n` format learnings files.
//! Used by both `LearningsMerger` (alzina-governance) and
//! `FileLearningsStore` (alzina-memory) to ensure format consistency
//! across the write-read feedback loop (S-12).

/// Parse entries from a learnings file.
///
/// Entries are `- ` prefixed lines. Metadata lines (provenance headers,
/// HTML comments, section headers, delimiter markers) are skipped.
///
/// This is the canonical format for on-disk learnings files:
/// - `learnings/{domain}/_index.md` (domain-mapped)
/// - `learnings/{agent_id}.md` (legacy per-agent)
pub fn parse_file_entries(content: &str) -> Vec<String> {
    let mut entries = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("- ") {
            let entry = trimmed.strip_prefix("- ").unwrap_or(trimmed).trim();
            if !entry.is_empty() {
                entries.push(entry.to_string());
            }
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_entries() {
        let content = "# Learnings\n\n- Entry one\n- Entry two\n- Entry three\n";
        let entries = parse_file_entries(content);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], "Entry one");
        assert_eq!(entries[1], "Entry two");
        assert_eq!(entries[2], "Entry three");
    }

    #[test]
    fn parse_skips_metadata() {
        let content = "<!-- provenance: author=test -->\n# Title\n\n- Actual entry\n";
        let entries = parse_file_entries(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], "Actual entry");
    }

    #[test]
    fn parse_skips_empty_lines() {
        let content = "- First\n\n\n- Second\n";
        let entries = parse_file_entries(content);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn parse_empty_content() {
        assert!(parse_file_entries("").is_empty());
    }

    #[test]
    fn parse_trims_whitespace() {
        let content = "  - Indented entry  \n";
        let entries = parse_file_entries(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], "Indented entry");
    }

    #[test]
    fn parse_skips_dash_only() {
        let content = "-  \n- Real entry\n";
        let entries = parse_file_entries(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], "Real entry");
    }

    #[test]
    fn parse_provenance_tagged_entries() {
        let content = "- [envelope] Learning from agent\n- [reflection] Learning from reflection\n";
        let entries = parse_file_entries(content);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], "[envelope] Learning from agent");
        assert_eq!(entries[1], "[reflection] Learning from reflection");
    }
}
