//! Wall-clock timeout resolver for tool execution.
//!
//! Read once per process from `ALZINA_TOOL_TIMEOUT_SECS` and cached via
//! `OnceLock` — subsequent calls return the same `Duration` without
//! touching the environment again.
//!
//! Clamp range: 1–3600 seconds (1 second minimum prevents disabling
//! the timeout; 3600 = 1 hour ceiling). Default: 120 seconds.
//!
//! Ported from openhuman @ 70fdedcdd449dca38b20bf30f69ec3c53a2b1666
//! (`src/openhuman/tool_timeout/mod.rs`). Env var renamed from
//! `OPENHUMAN_TOOL_TIMEOUT_SECS` to `ALZINA_TOOL_TIMEOUT_SECS`.
//! Return type changed from `u64` to `std::time::Duration` so callers
//! do not need a conversion step. Pure parser exposed as `pub(crate)` so
//! unit tests can exercise all paths without env mutation (D13-13).

use std::sync::OnceLock;
use std::time::Duration;

/// Default timeout in seconds when the environment variable is absent or
/// invalid.
pub const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 120;

/// Minimum accepted timeout in seconds. Values at or below zero are
/// rejected and fall back to `DEFAULT_TOOL_TIMEOUT_SECS`.
pub const MIN_TOOL_TIMEOUT_SECS: u64 = 1;

/// Maximum accepted timeout in seconds. Values above this clamp back to
/// `DEFAULT_TOOL_TIMEOUT_SECS`.
pub const MAX_TOOL_TIMEOUT_SECS: u64 = 3600;

static TOOL_TIMEOUT: OnceLock<Duration> = OnceLock::new();

/// Parse a raw string (from an env var or test fixture) into a bounded
/// `Duration`.
///
/// - `None` → `DEFAULT_TOOL_TIMEOUT_SECS`.
/// - Non-numeric string → `DEFAULT_TOOL_TIMEOUT_SECS`.
/// - Value outside `1..=3600` → `DEFAULT_TOOL_TIMEOUT_SECS`.
/// - Valid value → `Duration::from_secs(value)`.
///
/// `pub(crate)` because external callers have no need to bypass the
/// OnceLock cache; tests inside this crate invoke it directly to avoid
/// process-level env mutation (D13-13).
pub(crate) fn parse_tool_timeout_secs(raw: Option<&str>) -> Duration {
    let secs = raw
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| (MIN_TOOL_TIMEOUT_SECS..=MAX_TOOL_TIMEOUT_SECS).contains(&n))
        .unwrap_or(DEFAULT_TOOL_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Return the tool execution timeout for this process.
///
/// The value is read from `ALZINA_TOOL_TIMEOUT_SECS` once and then
/// cached — every call after the first is a single pointer dereference.
///
/// Pass the returned `Duration` to `tokio::time::timeout` or any
/// equivalent wall-clock guard.
pub fn tool_timeout() -> Duration {
    *TOOL_TIMEOUT.get_or_init(|| {
        parse_tool_timeout_secs(std::env::var("ALZINA_TOOL_TIMEOUT_SECS").ok().as_deref())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `None` input (env var absent) must return the 120-second default.
    #[test]
    fn default_when_env_missing() {
        assert_eq!(parse_tool_timeout_secs(None), Duration::from_secs(120));
    }

    /// A well-formed numeric string within range passes through unchanged.
    #[test]
    fn accepts_valid_midrange_value() {
        assert_eq!(
            parse_tool_timeout_secs(Some("60")),
            Duration::from_secs(60)
        );
    }

    /// 0 seconds would disable the timeout — must reject and return default.
    #[test]
    fn clamps_zero_to_default() {
        assert_eq!(
            parse_tool_timeout_secs(Some("0")),
            Duration::from_secs(120)
        );
    }

    /// Values above MAX_TOOL_TIMEOUT_SECS must return the default.
    #[test]
    fn clamps_above_max_to_default() {
        assert_eq!(
            parse_tool_timeout_secs(Some("9999")),
            Duration::from_secs(120)
        );
        assert_eq!(
            parse_tool_timeout_secs(Some("99999999999")),
            Duration::from_secs(120)
        );
    }

    /// Non-numeric strings must return the default, not panic.
    #[test]
    fn default_when_value_not_numeric() {
        assert_eq!(
            parse_tool_timeout_secs(Some("not-a-number")),
            Duration::from_secs(120)
        );
    }

    /// Empty string is non-numeric — must return the default.
    #[test]
    fn default_when_empty_string() {
        assert_eq!(
            parse_tool_timeout_secs(Some("")),
            Duration::from_secs(120)
        );
    }

    /// Negative values fail u64 parsing and return the default.
    #[test]
    fn default_when_negative() {
        assert_eq!(
            parse_tool_timeout_secs(Some("-5")),
            Duration::from_secs(120)
        );
    }

    /// Boundary value 1 is the minimum accepted value.
    #[test]
    fn accepts_minimum_boundary() {
        assert_eq!(
            parse_tool_timeout_secs(Some("1")),
            Duration::from_secs(1)
        );
    }

    /// Boundary value 3600 is the maximum accepted value.
    #[test]
    fn accepts_maximum_boundary() {
        assert_eq!(
            parse_tool_timeout_secs(Some("3600")),
            Duration::from_secs(3600)
        );
    }

    /// Value exactly one above the maximum must return the default.
    #[test]
    fn rejects_just_above_max() {
        assert_eq!(
            parse_tool_timeout_secs(Some("3601")),
            Duration::from_secs(120)
        );
    }

    /// `tool_timeout()` must return the same `Duration` on repeated calls
    /// (OnceLock idempotency). This test runs first in the file so the
    /// env var has not been set, and the returned value is whatever the
    /// process's `ALZINA_TOOL_TIMEOUT_SECS` is (default 120 if absent).
    /// We just verify that two consecutive calls agree.
    ///
    /// NOTE: because `OnceLock` is process-global, the cached value is
    /// determined by whichever test in any crate calls `tool_timeout()`
    /// first. This test only asserts idempotency, not a specific value.
    #[test]
    fn tool_timeout_is_idempotent() {
        let first = tool_timeout();
        let second = tool_timeout();
        assert_eq!(first, second, "OnceLock must return the same value on every call");
    }
}
