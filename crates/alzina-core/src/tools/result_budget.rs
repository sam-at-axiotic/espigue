//! Per-call byte cap for tool output.
//!
//! Apply a byte budget to a raw tool result *before* it enters
//! conversation history. This is the cheapest stage because it operates
//! on fresh bytes that have not yet been sent to the inference backend —
//! it does not mutate existing history and therefore does not break the
//! KV-cache prefix.
//!
//! ## Truncation marker
//!
//! The marker text is LOAD-BEARING. It tells the model to re-run with a
//! narrower query. Changing the text requires retraining the model on
//! the new phrasing. Pin it via the `TRUNCATION_MARKER_VERBATIM` test.
//!
//! ## Default cap
//!
//! `DEFAULT_RESULT_BUDGET_BYTES` = 16 KB (16 × 1024 = 16 384 bytes).
//! Call `apply_budget(content, DEFAULT_RESULT_BUDGET_BYTES)` or the
//! convenience wrapper `apply_default_budget(content)`.
//!
//! Ported from openhuman @ 70fdedcdd449dca38b20bf30f69ec3c53a2b1666
//! (`src/openhuman/context/tool_result_budget.rs`). API surface adapted
//! to take `&str` (not owned `String`) and return `String`; the
//! `BudgetOutcome` struct is preserved. The `floor_char_boundary` helper
//! is inlined (openhuman's version lived in `crate::openhuman::util`
//! which is not available in alzina-core).

use std::fmt::Write as _;

/// Default per-tool-result budget (16 KB). Large raw tool payloads are
/// trimmed inline before they enter history so parent-session tool
/// output cannot grow without bound.
pub const DEFAULT_RESULT_BUDGET_BYTES: usize = 16 * 1024;

/// Bytes reserved at the tail of the budget for the truncation marker.
/// The effective head capacity is `budget - TRAILER_RESERVED`.
const TRAILER_RESERVED: usize = 256;

/// Outcome of a budget application, for tracing and testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetOutcome {
    /// Byte length of the original content.
    pub original_bytes: usize,
    /// Byte length of the returned content (`== original_bytes` when the
    /// result fit inside the budget).
    pub final_bytes: usize,
    /// `true` if the content was truncated.
    pub truncated: bool,
}

impl BudgetOutcome {
    /// Construct an "unchanged" outcome for content that fit in budget.
    pub fn unchanged(len: usize) -> Self {
        Self {
            original_bytes: len,
            final_bytes: len,
            truncated: false,
        }
    }
}

/// Find the largest byte index `<= pos` that is a valid UTF-8 character
/// boundary in `s`. Returns 0 if no boundary can be found before `pos`.
fn floor_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    let mut end = pos;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Apply a byte budget to `content`.
///
/// If `content` fits in `budget_bytes`, returns it unchanged. Otherwise
/// returns a truncated prefix followed by the load-bearing marker:
///
/// ```text
/// \n\n[… N bytes truncated by tool_result_budget — re-run with a narrower query to see the rest …]
/// ```
///
/// The cut is made at a UTF-8 character boundary so the returned string
/// is always valid UTF-8.
///
/// `budget_bytes == 0` is treated as "no budget" — content passes through
/// unchanged.
pub fn apply_budget(content: &str, budget_bytes: usize) -> String {
    let original_bytes = content.len();
    if budget_bytes == 0 || original_bytes <= budget_bytes {
        return content.to_owned();
    }

    let head_capacity = budget_bytes.saturating_sub(TRAILER_RESERVED).max(1);
    let mut cut = floor_char_boundary(content, head_capacity);

    // Extremely short content (single multi-byte char) — guarantee at
    // least one character makes it into the head.
    if cut == 0 {
        cut = content
            .char_indices()
            .next()
            .map(|(_, c)| c.len_utf8())
            .unwrap_or(0);
    }

    let dropped_bytes = original_bytes.saturating_sub(cut);
    let mut out = String::with_capacity(cut + TRAILER_RESERVED);
    out.push_str(&content[..cut]);
    let _ = write!(
        out,
        "\n\n[… {dropped_bytes} bytes truncated by tool_result_budget — re-run with a narrower query to see the rest …]"
    );
    out
}

/// Apply the default 16 KB budget to `content`.
///
/// Convenience wrapper around `apply_budget(content, DEFAULT_RESULT_BUDGET_BYTES)`.
pub fn apply_default_budget(content: &str) -> String {
    apply_budget(content, DEFAULT_RESULT_BUDGET_BYTES)
}

/// Apply the budget and return the outcome alongside the (possibly
/// truncated) content.
pub(crate) fn apply_budget_with_outcome(content: &str, budget_bytes: usize) -> (String, BudgetOutcome) {
    let original_bytes = content.len();
    let out = apply_budget(content, budget_bytes);
    let final_bytes = out.len();
    let truncated = final_bytes != original_bytes || out != content;
    (
        out,
        BudgetOutcome {
            original_bytes,
            final_bytes,
            truncated,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Content under the cap passes through unchanged with no marker.
    #[test]
    fn small_content_passes_through_unchanged() {
        let input = "hello world";
        let out = apply_budget(input, 1024);
        assert_eq!(out, input);
        assert!(!out.contains("truncated"));
    }

    /// Content at exactly the cap passes through unchanged (no off-by-one).
    #[test]
    fn content_at_exact_budget_is_unchanged() {
        let input = "x".repeat(100);
        let out = apply_budget(&input, 100);
        assert_eq!(out, input);
        assert!(!out.contains("truncated"));
    }

    /// Content one byte over the cap must be truncated and include the marker.
    #[test]
    fn content_one_byte_over_cap_is_truncated() {
        // Use ASCII so byte count == char count.
        let input = "x".repeat(1025);
        let out = apply_budget(&input, 1024);
        assert!(out.len() < input.len());
        assert!(out.contains("truncated by tool_result_budget"));
    }

    /// Oversized content is truncated and contains the marker.
    #[test]
    fn oversized_content_is_truncated_with_marker() {
        let input = "x".repeat(10_000);
        let out = apply_budget(&input, 1024);
        assert!(out.len() < 10_000);
        assert!(out.contains("truncated by tool_result_budget"));
        assert!(out.contains("bytes truncated"));
    }

    /// Truncation must respect UTF-8 boundaries — no invalid UTF-8 in output.
    #[test]
    fn truncation_respects_utf8_boundaries() {
        // Each "é" is 2 bytes. 600 of them = 1200 bytes.
        let input: String = "é".repeat(600);
        let out = apply_budget(&input, 500);
        // Must be valid UTF-8.
        let _ = out.as_str();
        // Head should contain only full "é" characters (no half-byte).
        let head_end = out.find("\n\n[").expect("marker must be present");
        let head = &out[..head_end];
        assert!(
            head.chars().all(|c| c == 'é'),
            "head contains a broken character boundary"
        );
    }

    /// Zero budget is treated as no-op — content passes through unchanged.
    #[test]
    fn zero_budget_is_noop() {
        let input = "keep me";
        let out = apply_budget(input, 0);
        assert_eq!(out, input);
    }

    /// The marker text must match the load-bearing openhuman phrasing
    /// verbatim. Any future edit to the marker will fail this test — that
    /// is intentional. Changing the marker requires a retraining decision.
    #[test]
    fn truncation_marker_verbatim() {
        let input = "x".repeat(10_000);
        let out = apply_budget(&input, 1024);
        let marker_start = out.find("\n\n[").expect("marker must be present");
        let marker = &out[marker_start..];
        // Pin the exact wording.
        assert!(
            marker.starts_with("\n\n[… ") && marker.contains("bytes truncated by tool_result_budget — re-run with a narrower query to see the rest …]"),
            "truncation marker does not match load-bearing phrasing; got: {marker:?}"
        );
    }

    /// DEFAULT_RESULT_BUDGET_BYTES constant must be 16 KB.
    #[test]
    fn default_budget_is_16_kb() {
        assert_eq!(DEFAULT_RESULT_BUDGET_BYTES, 16 * 1024);
        assert_eq!(DEFAULT_RESULT_BUDGET_BYTES, 16384);
    }

    /// apply_default_budget delegates to apply_budget with 16 KB cap.
    #[test]
    fn apply_default_budget_delegates() {
        let big = "y".repeat(DEFAULT_RESULT_BUDGET_BYTES + 1);
        let via_explicit = apply_budget(&big, DEFAULT_RESULT_BUDGET_BYTES);
        let via_default = apply_default_budget(&big);
        assert_eq!(via_explicit, via_default);
    }

    /// apply_budget_with_outcome reports correct byte counts.
    #[test]
    fn outcome_reports_correct_byte_counts() {
        let input = "x".repeat(5_000);
        let (out, outcome) = apply_budget_with_outcome(&input, 1024);
        assert_eq!(outcome.original_bytes, 5_000);
        assert_eq!(outcome.final_bytes, out.len());
        assert!(outcome.truncated);
    }

    /// apply_budget_with_outcome unchanged path.
    #[test]
    fn outcome_unchanged_when_fits() {
        let input = "short";
        let (out, outcome) = apply_budget_with_outcome(input, 1024);
        assert_eq!(out, input);
        assert!(!outcome.truncated);
        assert_eq!(outcome.original_bytes, outcome.final_bytes);
    }
}
