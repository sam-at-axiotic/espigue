//! Tokenjuice tool-output compaction adapter.
//!
//! Adapts the tokenjuice tool-output reduction engine from
//! `tinyhumansai/openhuman` (SHA `70fdedcdd449dca38b20bf30f69ec3c53a2b1666`)
//! into an Alzina-idiomatic module.  The port does NOT import the openhuman
//! rule-classification engine directly (that engine depends on `regex` and
//! `once_cell`, neither of which is in `alzina-core`'s dependency set).
//! Instead it borrows the public interface shape and the pass-through-safety
//! semantics, replaces regex-based filtering with string-scan primitives, and
//! loads the builtin rule set via `include_str!` so it ships in the binary.
//!
//! ## Pass-through guards (D13-14)
//!
//! The function is pass-through-safe.  Four skip reasons exist; they are
//! tested in the order they are applied:
//!
//! - `Small` — input shorter than 512 bytes; no benefit from rule application.
//! - `FailureDetected` — exit code != 0 OR output contains `FAILED`, `error`,
//!   or `panic`; compression is skipped so diagnostics survive to the caller.
//! - `Disabled` — no rule entry found for the supplied `tool_name`; no-op.
//! - `MarginalRatio` — rule was applied but the compacted form exceeds 95% of
//!   the input length; original is returned instead.
//!
//! ## Synthetic-engine test seam (WARNING-5 resolution)
//!
//! The `MarginalRatio` guard requires a compaction step that produces output
//! shorter than the original in order to be exercised.  Because the builtin
//! rule set ships with minimal (or empty) rules in Phase 13, the guard would
//! be untestable without a real rule body.  The `pub(crate)`
//! `apply_compaction_with_engine` function accepts an injected engine closure
//! that replaces the real rule-driven engine for test purposes.  This makes
//! both the positive-compaction path AND the marginal-ratio guard fully
//! testable with no ignore-attribute escape clauses.
//!
//! ## Builtin rule set
//!
//! Lives in the sibling file `compaction_rules.json` (same directory as this
//! source file).  It is embedded via `include_str!` at compile time so no
//! runtime filesystem access is needed.
//!
//! ## References
//!
//! - `docs/proposals/002-tokenjuice-tool-output-compression.md` — full
//!   integration architecture and rule schema this adapter targets.
//! - openhuman SHA `70fdedcdd449dca38b20bf30f69ec3c53a2b1666`:
//!   `src/openhuman/tokenjuice/tool_integration.rs` — original interface.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Threshold constants (from openhuman tool_integration.rs)
// ---------------------------------------------------------------------------

/// Skip compaction for outputs shorter than this (bytes).
/// Tiny outputs have no headroom to benefit from summarisation and risk
/// distortion by rules that were designed for long logs.
const MIN_COMPACT_INPUT_BYTES: usize = 512;

/// Keep compacted form only when its byte count is at most this fraction of
/// the original.  Between `MIN_COMPACT_RATIO` and 1.0 the compaction is
/// considered not worthwhile and the raw output is returned.
const MIN_COMPACT_RATIO: f64 = 0.95;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Reason a compaction step was skipped (or returned the original).
///
/// Carried by [`CompactionStats::skip_reason`].  The variants map 1:1 to the
/// four pass-through guards documented in D13-14.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkipReason {
    /// Input shorter than [`MIN_COMPACT_INPUT_BYTES`] (512 bytes).
    Small,
    /// Compacted form exceeded [`MIN_COMPACT_RATIO`] (95%) of the input.
    MarginalRatio,
    /// Output contained a diagnostic marker (`error`, `panic`, `failed`,
    /// `fatal`, `traceback` — word-boundary, case-insensitive), OR exit
    /// code != 0. See `has_failure_marker` for the exact rule.
    FailureDetected,
    /// No rule entry found for the supplied `tool_name`.
    Disabled,
}

/// Statistics returned alongside every `compact_tool_output` call.
///
/// When `compacted == false` the returned string is identical to the input and
/// `skip_reason` explains why.  When `compacted == true` the returned string
/// is shorter and `skip_reason` is `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionStats {
    /// Byte length of the input string.
    pub input_bytes: usize,
    /// Byte length of the returned string.
    pub output_bytes: usize,
    /// `output_bytes / input_bytes`.  1.0 means no compaction occurred.
    pub ratio: f64,
    /// `true` iff the returned string differs from the input.
    pub compacted: bool,
    /// `None` when compaction succeeded; otherwise names the skip reason.
    pub skip_reason: Option<SkipReason>,
}

// ---------------------------------------------------------------------------
// Rule-set types (schema from proposal 002)
// ---------------------------------------------------------------------------

/// A single compaction rule as loaded from `compaction_rules.json`.
///
/// Schema reference: `docs/proposals/002-tokenjuice-tool-output-compression.md`
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompactionRule {
    /// The tool name this rule applies to (e.g. `"grep"`, `"bash"`).
    tool_name: String,
    /// Ordered list of reduction strategies.
    ///
    /// An empty list means "accept the rule match but apply no
    /// transformations".  The `Disabled` guard will NOT fire (the tool IS
    /// in the rule set), but the compacted output will equal the input so
    /// the `MarginalRatio` guard will catch it.
    #[serde(default)]
    strategies: Vec<RuleStrategy>,
}

/// A reduction strategy entry inside a [`CompactionRule`].
///
/// Only a small subset of the upstream strategies are implemented here;
/// unknown `type` values are silently skipped.
///
/// PORT NOTE: The rule body is minimal / empty for most tools pending Phase
/// 14 expansion.  `fold_whitespace` and `drop_blank_lines` are implemented
/// as string-scan primitives (no `regex` dependency).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuleStrategy {
    /// Strategy type token.
    /// Supported: `"fold_whitespace"`, `"drop_blank_lines"`, `"truncate"`.
    /// Unknown values are silently skipped.
    #[serde(rename = "type")]
    strategy_type: String,
    /// For `"truncate"`: maximum number of lines to keep.
    #[serde(default)]
    max_lines: Option<usize>,
    /// For `"truncate"`: `"head"` (default) or `"tail"`.
    #[serde(default)]
    keep: Option<String>,
}

// ---------------------------------------------------------------------------
// Built-in rule set
// ---------------------------------------------------------------------------

// PORT NOTE: The builtin rule set is embedded at compile time.  The JSON
// file ships conservative rules for `grep`, `bash`, and `read`.  The rule
// body is minimal pending Phase 14 expansion.  The MarginalRatio guard is
// still testable via the synthetic-engine seam regardless.
static BUILTIN_RULES_JSON: &str = include_str!("compaction_rules.json");

/// Process-wide cache for the parsed builtin rule set (WR-06).
///
/// The embedded JSON is small and the parse is cheap, but `compact_tool_output`
/// is on a hot path — caching once per process avoids re-parsing on every
/// invocation. `OnceLock` provides lock-free post-init reads.
static BUILTIN_RULES: OnceLock<Vec<CompactionRule>> = OnceLock::new();

/// Parse and return the builtin rule set, caching the result for the lifetime
/// of the process.
///
/// On parse failure, this function:
///
///   1. Emits a `tracing::error!` event so the failure is visible in the
///      audit trail (Sam's "degradation must be loud" rule — silent fallback
///      to no-compaction would let a malformed `compaction_rules.json` edit
///      ship with no signal).
///   2. Caches the empty rule set so we do not re-attempt the failing parse
///      on every subsequent call (and do not re-emit the error log on
///      every call either — once is enough).
///
/// The compile-time test `rules_parse_at_compile_time` pins that the
/// embedded JSON parses to a non-empty set so a malformed edit fails the
/// test suite immediately.
fn load_builtin_rules() -> &'static [CompactionRule] {
    BUILTIN_RULES.get_or_init(|| {
        match serde_json::from_str::<Vec<CompactionRule>>(BUILTIN_RULES_JSON) {
            Ok(rules) => rules,
            Err(e) => {
                tracing::error!(
                    target: "compaction",
                    error = %e,
                    "compaction_rules.json failed to parse — falling back to empty \
                     rule set. Every tool will now skip with SkipReason::Disabled until \
                     the JSON is fixed."
                );
                Vec::new()
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Rule application (no-regex implementation)
// ---------------------------------------------------------------------------

/// Returns `true` if `output` carries a diagnostic marker that suggests
/// the tool reported a failure (CR-02).
///
/// The intent of this guard is: skip compaction when the output is
/// likely to contain a stack trace / error message we must preserve
/// for the caller. Two defects in the original implementation:
///
/// 1. Naked substring match — `"error".contains(...)` matches inside
///    `terror`, `mirror`, `errored`, `errorless`, `Erroneous`. Any
///    clean output that mentions these words is forever exempt from
///    compaction. Same for `"panic"` matching `panicky`/`panicking`.
/// 2. Case-sensitive on a case-insensitive vocabulary — `Error:`,
///    `ERROR:`, `Panic:`, `Failed:`, `FAILURE` all slip past. The
///    most common shell / Rust / Python error prefixes are not lowercase.
///
/// The fix is whole-token boundary matching with case-insensitive
/// comparison: the marker must be both preceded AND followed by a
/// separator (whitespace, punctuation, brackets, quotes) or by a
/// string boundary. Symmetric rule — rejects `errored` / `panicky` /
/// `errorless` (alphanumeric suffix), `error_handler` (underscore
/// suffix), and `error-free` (hyphen suffix, which the reviewer
/// flagged explicitly as a false-positive that must not trigger).
///
/// Both `compact_tool_output` and `apply_compaction_with_engine`
/// previously inlined the buggy guard verbatim. They both call this
/// helper now — single source of truth, no drift risk.
fn has_failure_marker(output: &str) -> bool {
    // Token list — every entry is a lowercased canonical diagnostic word.
    // Includes both `failed` (past tense) and `failure` (noun) since both
    // appear in test runners (`FAILED:`, `FAILURE:`).
    const TOKENS: &[&str] = &[
        "error",
        "panic",
        "failed",
        "failure",
        "fatal",
        "traceback",
    ];

    // SEPARATORS bound the diagnostic on both sides. A token only triggers
    // when:
    //   - it sits at start-of-string or is preceded by a separator;
    //   - it sits at end-of-string or is followed by a separator.
    // The symmetric rule excludes compound words like `error-free` (hyphen
    // is NOT a separator, so the match is rejected) and identifier forms
    // like `error_handler` (underscore is NOT a separator either).
    const SEPARATORS: &[char] = &[
        '\n', '\r', ' ', '\t', '[', ']', ':', '<', '>', '!', '(', ')', ',', ';', '.', '"', '\'',
    ];

    let lower = output.to_ascii_lowercase();
    let bytes = lower.as_bytes();

    for tok in TOKENS {
        let tok_bytes = tok.as_bytes();
        let mut start = 0;
        while let Some(pos) = lower[start..].find(tok) {
            let abs = start + pos;

            // Preceded-by check: start-of-string or a known separator.
            let preceded_ok = abs == 0
                || lower[..abs]
                    .chars()
                    .next_back()
                    .is_some_and(|c| SEPARATORS.contains(&c));

            // Followed-by check: end-of-string or a known separator.
            // Symmetric with preceded-by — rejects `errored`, `errorless`,
            // `panicked`, `panicky`, `error_handler` (alphanumeric or
            // underscore after) AND `error-free` (hyphen after).
            let after = abs + tok_bytes.len();
            let followed_ok = after >= bytes.len()
                || lower[after..]
                    .chars()
                    .next()
                    .is_some_and(|c| SEPARATORS.contains(&c));

            if preceded_ok && followed_ok {
                return true;
            }

            start = abs + tok_bytes.len();
        }
    }

    false
}

/// Look up a rule for `tool_name` in the loaded rule set.
///
/// Returns `None` when no rule matches — callers interpret `None` as the
/// `Disabled` skip reason.
fn find_rule<'a>(rules: &'a [CompactionRule], tool_name: &str) -> Option<&'a CompactionRule> {
    rules.iter().find(|r| r.tool_name == tool_name)
}

/// Apply a single [`RuleStrategy`] to `text`.
///
/// Returns the (potentially shorter) output string.
/// Unknown strategy types return `text` unchanged.
fn apply_strategy(text: &str, strategy: &RuleStrategy) -> String {
    match strategy.strategy_type.as_str() {
        "fold_whitespace" => {
            // WR-05(b): if the input is entirely whitespace (including the
            // empty string), there is nothing meaningful to fold. Returning
            // the empty string here would trip the MarginalRatio guard with
            // ratio = 0 and be reported as "successful compaction" —
            // silently dropping meaningful whitespace structure. Treat
            // whitespace-only inputs as a no-op instead.
            if text.chars().all(char::is_whitespace) {
                return text.to_owned();
            }
            // Collapse runs of consecutive blank lines to a single blank line.
            // Leading and trailing blank lines are stripped.
            let mut out = String::with_capacity(text.len());
            let mut blank_run: usize = 0;
            for line in text.lines() {
                if line.trim().is_empty() {
                    blank_run += 1;
                } else {
                    if blank_run > 0 && !out.is_empty() {
                        out.push('\n');
                    }
                    blank_run = 0;
                    out.push_str(line);
                    out.push('\n');
                }
            }
            if out.ends_with('\n') {
                out.truncate(out.len() - 1);
            }
            out
        }
        "drop_blank_lines" => {
            // Remove every blank line.
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        }
        "truncate" => {
            let max_lines = strategy.max_lines.unwrap_or(usize::MAX);
            let keep = strategy.keep.as_deref().unwrap_or("head");
            let lines: Vec<&str> = text.lines().collect();
            if lines.len() <= max_lines {
                return text.to_owned();
            }
            // WR-05(a): `text.lines()` strips terminators, so `join("\n")`
            // omits the trailing newline that may have been present. Most
            // line-oriented diagnostics expect newline-terminated output;
            // preserve the terminator when the input had one.
            let mut out = match keep {
                "tail" => lines[lines.len() - max_lines..].join("\n"),
                _ => lines[..max_lines].join("\n"),
            };
            if text.ends_with('\n') && !out.is_empty() {
                out.push('\n');
            }
            out
        }
        // Unknown strategy types: pass through unchanged.
        _ => text.to_owned(),
    }
}

/// Apply all strategies from `rule` in order to produce a compacted string.
fn apply_rule(rule: &CompactionRule, output: &str) -> String {
    rule.strategies
        .iter()
        .fold(output.to_owned(), |acc, s| apply_strategy(&acc, s))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compact a single tool call's output using the builtin rule set.
///
/// This is the primary entry point.  It checks the builtin rule set for
/// `tool_name`, applies the `Disabled` guard when no rule matches, then
/// delegates to [`apply_compaction_with_engine`] with the real rule engine.
///
/// ## Arguments
///
/// - `tool_name` — tool that produced `output` (used to select the rule).
/// - `args` — tool's input arguments (accepted for interface symmetry;
///   currently not used for rule selection; Phase 14 may use for
///   command-pattern matching).
/// - `output` — raw text output to compact.
/// - `exit_code` — tool exit code; non-zero triggers `FailureDetected`.
///
/// ## Returns
///
/// `(text, stats)` where `text` is either the compacted form (when
/// `stats.compacted == true`) or the original (when `stats.skip_reason`
/// is set).
pub fn compact_tool_output(
    tool_name: &str,
    args: &Value,
    output: &str,
    exit_code: i32,
) -> (String, CompactionStats) {
    let input_bytes = output.len();

    // Guard 1: Small input (cheapest — checked before the rule-set lookup).
    if input_bytes < MIN_COMPACT_INPUT_BYTES {
        return (
            output.to_owned(),
            CompactionStats {
                input_bytes,
                output_bytes: input_bytes,
                ratio: 1.0,
                compacted: false,
                skip_reason: Some(SkipReason::Small),
            },
        );
    }

    // Guard 2: Failure-pattern detection — skip compression so diagnostics
    // survive even on large outputs. CR-02: uses the word-boundary +
    // case-insensitive `has_failure_marker` helper (no naked substring).
    if exit_code != 0 || has_failure_marker(output) {
        return (
            output.to_owned(),
            CompactionStats {
                input_bytes,
                output_bytes: input_bytes,
                ratio: 1.0,
                compacted: false,
                skip_reason: Some(SkipReason::FailureDetected),
            },
        );
    }

    // Guard 3: Rule-set lookup — Disabled fires when no rule matches.
    let rules = load_builtin_rules();
    let rule = match find_rule(rules, tool_name) {
        Some(r) => r,
        None => {
            return (
                output.to_owned(),
                CompactionStats {
                    input_bytes,
                    output_bytes: input_bytes,
                    ratio: 1.0,
                    compacted: false,
                    skip_reason: Some(SkipReason::Disabled),
                },
            );
        }
    };

    // All guards before compaction have passed.  Run the rule-driven engine
    // and apply the marginal-ratio guard via the shared seam.
    //
    // We clone the matched rule so the closure has independent ownership
    // (avoids a lifetime conflict between `rules: Vec<CompactionRule>` and
    // the `FnOnce` closure).  `CompactionRule` derives `Clone`; the rule set
    // is small so the clone is cheap.
    let owned_rule = rule.clone();
    apply_compaction_with_engine(tool_name, args, output, exit_code, move |raw| {
        apply_rule(&owned_rule, raw)
    })
}

/// Test seam: run the compaction pipeline with an injected engine closure.
///
/// The public `compact_tool_output` calls this after the `Small`,
/// `FailureDetected`, and `Disabled` guards have already passed.
///
/// Tests use this directly to exercise the `MarginalRatio` guard and the
/// positive-compaction path without requiring a real rule body (WARNING-5
/// resolution):
///
/// - Marginal-ratio guard: inject `|s| s[..(s.len() * 96 / 100)].to_owned()`
/// - Positive-compaction: inject `|s| s[..s.len()/2].to_owned()`
///
/// Guard sequence applied here:
/// 1. `Small` — still checked so the seam is safe to call directly.
/// 2. `FailureDetected` — still checked.
/// 3. Engine application (no rule-set lookup — caller owns the engine).
/// 4. `MarginalRatio` — checked after engine runs.
pub(crate) fn apply_compaction_with_engine<F>(
    _tool_name: &str,
    _args: &Value,
    output: &str,
    exit_code: i32,
    engine: F,
) -> (String, CompactionStats)
where
    F: FnOnce(&str) -> String,
{
    let input_bytes = output.len();

    // Guard 1: Small.
    if input_bytes < MIN_COMPACT_INPUT_BYTES {
        return (
            output.to_owned(),
            CompactionStats {
                input_bytes,
                output_bytes: input_bytes,
                ratio: 1.0,
                compacted: false,
                skip_reason: Some(SkipReason::Small),
            },
        );
    }

    // Guard 2: Failure-pattern. CR-02: uses the shared `has_failure_marker`
    // helper so this site and `compact_tool_output` cannot drift apart.
    if exit_code != 0 || has_failure_marker(output) {
        return (
            output.to_owned(),
            CompactionStats {
                input_bytes,
                output_bytes: input_bytes,
                ratio: 1.0,
                compacted: false,
                skip_reason: Some(SkipReason::FailureDetected),
            },
        );
    }

    // Apply the engine.
    let compacted = engine(output);
    let output_bytes = compacted.len();

    let ratio = if input_bytes == 0 {
        1.0
    } else {
        output_bytes as f64 / input_bytes as f64
    };

    // Guard 3: MarginalRatio — if the engine barely compressed the input,
    // return the original.
    if ratio > MIN_COMPACT_RATIO || output_bytes >= input_bytes {
        return (
            output.to_owned(),
            CompactionStats {
                input_bytes,
                output_bytes: input_bytes,
                ratio: 1.0,
                compacted: false,
                skip_reason: Some(SkipReason::MarginalRatio),
            },
        );
    }

    // Compaction succeeded.
    (
        compacted,
        CompactionStats {
            input_bytes,
            output_bytes,
            ratio,
            compacted: true,
            skip_reason: None,
        },
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a clean string of exactly `n` bytes containing no failure
    /// substrings (`FAILED`, `error`, `panic`).
    fn clean_output(n: usize) -> String {
        // Repeating pattern well above 512 bytes.  No failure substrings.
        let unit = "hello world this is safe output line number ";
        let mut s = String::with_capacity(n + unit.len() + 5);
        let mut counter = 0usize;
        while s.len() < n {
            s.push_str(unit);
            s.push_str(&counter.to_string());
            s.push('\n');
            counter += 1;
        }
        s.truncate(n);
        s
    }

    /// Guard 1 — small input (<512 bytes) returns unchanged with `Small`.
    #[test]
    fn small_input_guard_returns_unchanged() {
        let tiny = "short output that is clearly under 512 bytes";
        assert!(tiny.len() < 512);
        let (out, stats) = compact_tool_output("bash", &json!({}), tiny, 0);
        assert_eq!(out, tiny, "output must be the original string");
        assert!(!stats.compacted, "compacted must be false");
        assert_eq!(stats.skip_reason, Some(SkipReason::Small));
        assert_eq!(stats.input_bytes, tiny.len());
        assert_eq!(stats.output_bytes, tiny.len());
    }

    /// Guard 2a — `FAILED` substring triggers `FailureDetected`.
    #[test]
    fn failure_pattern_failed_substring_returns_unchanged() {
        // Build output that is > 512 bytes but contains "FAILED".
        let output = format!("{}\nFAILED: test_foo panicked\n", clean_output(600));
        // Note: this also contains "panic" — either is sufficient to trigger.
        let without_panic = format!("{}\nFAILED: test_foo\n", clean_output(600));
        let (out, stats) = compact_tool_output("bash", &json!({}), &without_panic, 0);
        assert_eq!(out, without_panic);
        assert!(!stats.compacted);
        assert_eq!(stats.skip_reason, Some(SkipReason::FailureDetected));

        // Suppress the unused variable warning.
        let _ = output;
    }

    /// Guard 2b — `panic:` marker triggers `FailureDetected`.
    ///
    /// CR-02: the new whole-token helper requires a word boundary on
    /// both sides. `panic` followed by a colon (canonical Rust panic
    /// message prefix) is the form that triggers. Bare `panicked` no
    /// longer triggers — see `compaction_no_longer_misclassifies_panicked`.
    #[test]
    fn failure_pattern_panic_substring_returns_unchanged() {
        let output = format!("{}\npanic: assertion failed at src/lib.rs:42\n", clean_output(600));
        let (out, stats) = compact_tool_output("bash", &json!({}), &output, 0);
        assert_eq!(out, output);
        assert!(!stats.compacted);
        assert_eq!(stats.skip_reason, Some(SkipReason::FailureDetected));
    }

    /// Guard 2c — non-zero exit code triggers `FailureDetected`.
    #[test]
    fn failure_pattern_nonzero_exit_code_returns_unchanged() {
        let output = clean_output(600);
        let (out, stats) = compact_tool_output("bash", &json!({}), &output, 1);
        assert_eq!(out, output);
        assert!(!stats.compacted);
        assert_eq!(stats.skip_reason, Some(SkipReason::FailureDetected));
    }

    // ── CR-02: failure-marker helper boundary tests ──────────────────────
    //
    // These tests exercise `has_failure_marker` directly so the boundary
    // logic is pinned even when the surrounding pipeline (Small / Disabled
    // guards) would otherwise mask the change. The reviewer flagged both
    // false-positive and false-negative cases — both are covered.

    /// False-positive cases that the OLD naked-substring guard misclassified.
    /// The new whole-token helper must NOT flag these as failures.
    #[test]
    fn has_failure_marker_rejects_false_positives_from_review() {
        // From the review prose: `terror`, `mirror`, `error-free`, `errored`,
        // `errorless`, `Erroneous`, `panicky`, `panicking`, `panicked`.
        for benign in [
            "the great terror was a 1937 event",
            "see yourself in the mirror",
            "the run was error-free today",
            "the system errored gracefully",
            "the operation was errorless",
            "Erroneous assumptions are dangerous",
            "felt a bit panicky",
            "panicking is not productive",
            "the test panicked but recovered",
        ] {
            assert!(
                !has_failure_marker(benign),
                "benign text '{benign}' must not trigger has_failure_marker"
            );
        }
    }

    /// False-negative cases that the OLD case-sensitive guard missed.
    /// The new case-insensitive helper MUST flag these as failures.
    #[test]
    fn has_failure_marker_catches_false_negatives_from_review() {
        for diagnostic in [
            "Error: something broke",
            "ERROR: something broke",
            "Panic: assertion failed",
            "Failed: test_foo",
            "FAILURE: integration test",
            "WARN: error in subsystem",
            "Fatal: cannot continue",
            "FATAL: cannot continue",
            "Traceback (most recent call last):",
            "TRACEBACK: python style",
        ] {
            assert!(
                has_failure_marker(diagnostic),
                "diagnostic text '{diagnostic}' must trigger has_failure_marker"
            );
        }
    }

    /// `has_failure_marker` must trigger at start-of-string too (no leading
    /// separator needed when position is 0).
    #[test]
    fn has_failure_marker_triggers_at_start_of_string() {
        assert!(has_failure_marker("error: at position zero"));
        assert!(has_failure_marker("Panic at the disco"));
        assert!(has_failure_marker("traceback follows"));
    }

    /// `has_failure_marker` must not trigger when the token is followed
    /// by an underscore (code-identifier form, e.g. `error_handler`).
    #[test]
    fn has_failure_marker_rejects_underscore_suffix() {
        assert!(!has_failure_marker("the error_handler ran cleanly"));
        assert!(!has_failure_marker("fatal_signals are off in this build"));
    }

    /// Guard 3 — `Disabled` fires when tool name has no rule entry.
    #[test]
    fn disabled_guard_fires_for_unknown_tool() {
        let output = clean_output(600);
        let (out, stats) = compact_tool_output("unknown_tool_xyz_phase13", &json!({}), &output, 0);
        assert_eq!(out, output);
        assert!(!stats.compacted);
        assert_eq!(stats.skip_reason, Some(SkipReason::Disabled));
    }

    /// Guard 4 — `MarginalRatio` fires when synthetic engine returns ~96% of input.
    ///
    /// The engine returns 96% of the input length — just above the 0.95
    /// threshold.  The original must be returned.
    #[test]
    fn marginal_ratio_guard_returns_original() {
        let output = clean_output(1000);
        let input_len = output.len();

        let (returned, stats) = apply_compaction_with_engine(
            "bash",
            &json!({}),
            &output,
            0,
            // 96% of input length — just above the 0.95 threshold.
            |s| s[..(s.len() * 96 / 100)].to_owned(),
        );

        assert_eq!(
            returned, output,
            "original must be returned when ratio exceeds 0.95"
        );
        assert!(!stats.compacted);
        assert_eq!(stats.skip_reason, Some(SkipReason::MarginalRatio));
        assert_eq!(stats.input_bytes, input_len);
    }

    /// WR-06: the embedded `compaction_rules.json` must parse cleanly to
    /// a non-empty rule set.
    ///
    /// `load_builtin_rules` degrades loudly (via `tracing::error!`) on
    /// parse failure but still returns an empty rule set so the process
    /// stays up. That degradation path is acceptable at runtime, but a
    /// malformed JSON edit must fail the CI build immediately — this test
    /// is the safety net.
    #[test]
    fn rules_parse_at_compile_time() {
        let rules: Vec<CompactionRule> = serde_json::from_str(BUILTIN_RULES_JSON)
            .expect("compaction_rules.json must parse cleanly");
        assert!(
            !rules.is_empty(),
            "compaction_rules.json must contain at least one rule"
        );
    }

    /// WR-06: `load_builtin_rules` returns a stable `'static` reference.
    /// Two successive calls must return pointer-equal slices, proving the
    /// `OnceLock` cache is in effect (no re-parse on every call).
    #[test]
    fn load_builtin_rules_caches_via_oncelock() {
        let first = load_builtin_rules();
        let second = load_builtin_rules();
        // Pointer equality on the underlying slice — confirms the
        // `OnceLock` returned the same `Vec` both times rather than
        // re-running `from_str`.
        assert!(
            std::ptr::eq(first.as_ptr(), second.as_ptr()),
            "load_builtin_rules must return a cached static reference"
        );
        // Sanity: the cached rule set carries the expected tools.
        assert!(first.iter().any(|r| r.tool_name == "grep"));
        assert!(first.iter().any(|r| r.tool_name == "bash"));
    }

    /// WR-05(a): `truncate` strategy must preserve a trailing newline.
    ///
    /// `text.lines()` strips terminators; without the newline restore,
    /// the truncated output drops the trailing `\n` from the original
    /// and downstream line-oriented parsers see the last line as
    /// not-terminated.
    #[test]
    fn truncate_strategy_preserves_trailing_newline() {
        let strat = RuleStrategy {
            strategy_type: "truncate".to_string(),
            max_lines: Some(2),
            keep: None, // defaults to "head"
        };
        // Input ends with `\n` — head truncation to 2 lines must also end with `\n`.
        let input = "line1\nline2\nline3\n";
        let out = apply_strategy(input, &strat);
        assert_eq!(out, "line1\nline2\n", "head truncate must preserve trailing newline");

        // Tail variant must also preserve the trailing newline.
        let strat_tail = RuleStrategy {
            strategy_type: "truncate".to_string(),
            max_lines: Some(2),
            keep: Some("tail".to_string()),
        };
        let out_tail = apply_strategy(input, &strat_tail);
        assert_eq!(out_tail, "line2\nline3\n", "tail truncate must preserve trailing newline");

        // Input WITHOUT a trailing newline must NOT gain one.
        let input_nonl = "alpha\nbeta\ngamma";
        let strat_head = RuleStrategy {
            strategy_type: "truncate".to_string(),
            max_lines: Some(2),
            keep: None,
        };
        let out_nonl = apply_strategy(input_nonl, &strat_head);
        assert_eq!(out_nonl, "alpha\nbeta", "non-terminated input must not gain a trailing newline");
    }

    /// WR-05(b): `fold_whitespace` strategy must be a no-op on
    /// whitespace-only input.
    ///
    /// Without the early return, the strategy would emit `""` for an input
    /// like `"\n\n\n"`. That empty string then trips the `MarginalRatio`
    /// guard at ratio = 0 and gets reported as "successful compaction" —
    /// silently dropping meaningful whitespace structure.
    #[test]
    fn fold_whitespace_strategy_is_noop_on_whitespace_only_input() {
        let strat = RuleStrategy {
            strategy_type: "fold_whitespace".to_string(),
            max_lines: None,
            keep: None,
        };
        // Triple newline: pure whitespace, should be returned unchanged.
        let input = "\n\n\n";
        let out = apply_strategy(input, &strat);
        assert_eq!(out, input, "whitespace-only input must pass through untouched");

        // Empty string: also whitespace-only, also pass-through.
        let out_empty = apply_strategy("", &strat);
        assert_eq!(out_empty, "", "empty input must pass through untouched");

        // Mixed whitespace (tabs, spaces, newlines): still no-op.
        let mixed = "  \n\t  \n   ";
        let out_mixed = apply_strategy(mixed, &strat);
        assert_eq!(out_mixed, mixed, "mixed-whitespace-only input must pass through");

        // Sanity: non-whitespace input still folds normally — multiple blank
        // lines collapse to a single blank line between content blocks.
        let normal = "alpha\n\n\n\nbeta\n";
        let out_normal = apply_strategy(normal, &strat);
        assert_eq!(out_normal, "alpha\n\nbeta", "non-whitespace input still folds");
    }

    /// Positive compaction — synthetic engine returns 50% of input.
    ///
    /// The engine returns the first half of the string — ratio ~0.50, well
    /// below 0.95.  The compacted form must be returned.
    #[test]
    fn positive_compaction_returns_compacted_output() {
        let output = clean_output(1200);
        let half_len = output.len() / 2;

        let (returned, stats) = apply_compaction_with_engine(
            "bash",
            &json!({}),
            &output,
            0,
            // First half — well below the 0.95 threshold.
            |s| s[..s.len() / 2].to_owned(),
        );

        assert_ne!(returned, output, "compacted form must differ from original");
        assert_eq!(returned.len(), half_len);
        assert!(stats.compacted);
        assert!(
            stats.ratio < 0.95,
            "ratio must be below 0.95, got {}",
            stats.ratio
        );
        assert_eq!(stats.skip_reason, None);
        assert_eq!(stats.output_bytes, half_len);
    }
}
