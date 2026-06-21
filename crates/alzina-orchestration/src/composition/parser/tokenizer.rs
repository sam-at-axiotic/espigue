//! XML tokenization: `quick-xml` Reader wrapper + byte→(line, col) `SourceMap`.
//!
//! Contract: every parser error reports `(line, column)` of the OFFENDING
//! token, not the token-after-it. This is the Pitfall-2 invariant:
//! `Reader::buffer_position()` returns the byte index of the NEXT byte to
//! read (end-of-token); using it after `read_event()` reports errors at
//! end-of-token rather than start-of-token.
//!
//! Strict reject list (`docs/composition-grammar.md` §1.5):
//! - XML namespaces (`xmlns:foo`, names containing `:`)
//! - DTD declarations and external entities (`Event::DocType`)
//! - Processing instructions (`Event::PI`)
//! - Unknown tags / unknown attributes (handled in `ast.rs` at Wave 2)
//!
//! The XML declaration `<?xml version="1.0"?>` is allowed and ignored.

use quick_xml::events::Event;
use quick_xml::reader::Reader;

use super::errors::{ParseError, ParseErrorCode, ParseErrors};

/// Maps byte offsets to 1-indexed (line, column) coordinates.
///
/// Built in a single O(n) scan over the input. Lookups are O(log n) via
/// binary search over the line-start vector.
pub struct SourceMap {
    /// Sorted byte offsets where each line begins. `line_starts[0] == 0`.
    line_starts: Vec<usize>,
}

impl SourceMap {
    pub fn new(input: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, b) in input.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Self { line_starts }
    }

    /// Translate a byte offset to 1-indexed (line, column).
    ///
    /// For offsets past the end of input, returns the column arithmetic
    /// for the last line — useful for "unexpected EOF" errors.
    pub fn locate(&self, byte_offset: usize) -> (u32, u32) {
        // partition_point yields the count of entries where pred is true;
        // (greatest `line_start <= byte_offset`) → idx = count - 1.
        let line_idx = self
            .line_starts
            .partition_point(|&s| s <= byte_offset)
            .saturating_sub(1);
        let col = byte_offset - self.line_starts[line_idx] + 1;
        ((line_idx + 1) as u32, col as u32)
    }
}

/// Build a `ParseError` at a given byte offset using the `SourceMap`.
///
/// Caller MUST capture `byte_offset` BEFORE `reader.read_event()` to
/// preserve start-of-token accuracy (Pitfall 2).
pub fn parse_err(
    code: ParseErrorCode,
    byte_offset: usize,
    source_map: &SourceMap,
    message: impl Into<String>,
    hint: impl Into<String>,
) -> ParseError {
    let (line, column) = source_map.locate(byte_offset);
    ParseError {
        category: code.category(),
        code,
        line,
        column,
        message: message.into(),
        hint: hint.into(),
    }
}

/// Convenience: wrap a single `parse_err` as a one-error `ParseErrors`.
pub fn parse_errs(
    code: ParseErrorCode,
    byte_offset: usize,
    source_map: &SourceMap,
    message: impl Into<String>,
    hint: impl Into<String>,
) -> ParseErrors {
    ParseErrors::single(parse_err(code, byte_offset, source_map, message, hint))
}

/// Streaming cursor over XML events with Pitfall-2-safe positioning.
///
/// Always capture `cursor.byte_offset_before_event()` BEFORE calling
/// `cursor.next()` to report errors at the start of the offending token.
pub struct EventCursor<'a> {
    reader: Reader<&'a [u8]>,
    last_byte_offset_before: usize,
}

impl<'a> EventCursor<'a> {
    pub fn new(xml: &'a str) -> Self {
        let mut reader = Reader::from_str(xml);
        reader.trim_text(true);
        Self {
            reader,
            last_byte_offset_before: 0,
        }
    }

    /// Byte offset captured BEFORE the most recent `next()` call.
    ///
    /// Use this for error reporting; it points to the start of the token
    /// just read, NOT to the byte after it.
    pub fn byte_offset_before_event(&self) -> usize {
        self.last_byte_offset_before
    }

    /// Read the next event. Captures byte position before reading.
    pub fn next(&mut self) -> quick_xml::Result<Event<'_>> {
        self.last_byte_offset_before = self.reader.buffer_position() as usize;
        self.reader.read_event()
    }
}

/// Check whether a tag/attribute name contains a namespace prefix (a `:`).
///
/// Used to enforce the §1.5 namespace-rejection rule. The XML declaration
/// `<?xml ... ?>` is handled separately as an `Event::Decl` (allowed).
pub fn name_has_namespace(name_bytes: &[u8]) -> bool {
    name_bytes.contains(&b':')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_first_char_returns_line_one_col_one() {
        let sm = SourceMap::new("abc\ndef\nghi");
        assert_eq!(sm.locate(0), (1, 1));
    }

    #[test]
    fn locate_end_of_first_line() {
        let sm = SourceMap::new("abc\ndef\nghi");
        assert_eq!(sm.locate(3), (1, 4)); // position of '\n'
    }

    #[test]
    fn locate_first_char_of_second_line() {
        let sm = SourceMap::new("abc\ndef\nghi");
        assert_eq!(sm.locate(4), (2, 1));
    }

    #[test]
    fn locate_first_char_of_third_line() {
        let sm = SourceMap::new("abc\ndef\nghi");
        assert_eq!(sm.locate(8), (3, 1));
    }

    #[test]
    fn locate_empty_input_at_zero_is_one_one() {
        let sm = SourceMap::new("");
        assert_eq!(sm.locate(0), (1, 1));
    }

    #[test]
    fn locate_past_end_of_single_line_extends_column() {
        let sm = SourceMap::new("a");
        assert_eq!(sm.locate(1), (1, 2));
    }

    #[test]
    fn cursor_captures_offset_before_event() {
        let mut cur = EventCursor::new("<a/>");
        assert_eq!(cur.byte_offset_before_event(), 0);
        let _ = cur.next();
        assert_eq!(
            cur.byte_offset_before_event(),
            0,
            "offset BEFORE first event must be 0 (start of input), not after the token"
        );
    }

    #[test]
    fn cursor_advances_offset_before_each_event() {
        let mut cur = EventCursor::new("<a/><b/>");
        let _ = cur.next(); // <a/>
        let pos_after_first = cur.byte_offset_before_event();
        assert_eq!(pos_after_first, 0);
        let _ = cur.next(); // <b/>
        let pos_after_second = cur.byte_offset_before_event();
        assert!(
            pos_after_second > pos_after_first,
            "offset must advance to start of second token"
        );
        assert_eq!(
            pos_after_second, 4,
            "second token starts at byte 4 (after '<a/>')"
        );
    }

    #[test]
    fn name_has_namespace_detects_colon() {
        assert!(name_has_namespace(b"foo:bar"));
        assert!(name_has_namespace(b"xmlns:foo"));
        assert!(!name_has_namespace(b"Synthesise"));
        assert!(!name_has_namespace(b"Spawn"));
    }
}
