//! `alzina_core::tools` — Tool trait and companion types.
//!
//! This module defines the **sync** `Tool` trait that every Alzina tool
//! implements, plus its companion types: `ToolResult`, `ToolContent`,
//! `ToolCallOptions`, `PermissionLevel`, `ToolCategory`, and `ToolScope`.
//!
//! ## What consumes this
//!
//! - `impl From<&dyn Tool> for CustomToolDefinition` (Plan 13-05) — the
//!   single adapter that maps every `Tool` impl onto the sidecar wire struct.
//! - `build_allowed_tools` (Plan 13-05) — the curation function that uses
//!   `permission_level()`, `scope()`, and `category()` to decide which tools
//!   a given seam sees.
//!
//! ## Wire contract
//!
//! `CustomToolDefinition` lives in `alzina_orchestration::runner::sidecar_protocol`.
//! Every method on this trait maps to exactly one field on that struct.
//!
//! ## Why the trait is SYNC
//!
//! Every method on the `Tool` trait is **pure metadata** — name, description,
//! JSON schema, endpoint URL, timeout knobs, permission level. None of these
//! performs I/O. The actual tool invocation crosses the daemon HTTP boundary;
//! that async surface lives in the daemon's handler, not on the trait.
//! Keeping `Tool` sync makes it `dyn`-compatible without `Box<dyn Future>`
//! returns, which the `&dyn Tool`-consuming adapter (Plan 13-05) depends on.
//!
//! ## Source
//!
//! Ported from `tinyhumansai/openhuman` at commit
//! `70fdedcdd449dca38b20bf30f69ec3c53a2b1666`, file
//! `src/openhuman/tools/traits.rs` and `src/openhuman/skills/types.rs`.
//! Types adapted to the Alzina dependency profile (no `anyhow`, no
//! `async-trait` on the trait).

use serde::{Deserialize, Serialize};

// ── ToolContent ────────────────────────────────────────────────────────────

/// A single content block within a [`ToolResult`].
///
/// Ported verbatim from `openhuman::skills::types::ToolContent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolContent {
    Text { text: String },
    Json { data: serde_json::Value },
}

// ── ToolResult ─────────────────────────────────────────────────────────────

/// Result of executing a tool — content blocks plus error status.
///
/// Ported from `openhuman::skills::types::ToolResult`.
/// `ToolResult` and `ToolCallOptions` are part of the full trait surface
/// for future tool implementations (Phase 14+); no consumer in Phase 13
/// calls them at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Content blocks returned by the tool.
    pub content: Vec<ToolContent>,
    /// `true` when the tool encountered an error during execution.
    #[serde(default)]
    pub is_error: bool,
    /// Optional markdown rendering. When the agent loop is configured
    /// with `prefer_markdown`, this field is sent to the model instead
    /// of the JSON-serialised content blocks (token-saving path).
    #[serde(
        default,
        rename = "markdownFormatted",
        skip_serializing_if = "Option::is_none"
    )]
    pub markdown_formatted: Option<String>,
}

impl ToolResult {
    /// Construct a successful text result.
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::Text { text: text.into() }],
            is_error: false,
            markdown_formatted: None,
        }
    }

    /// Construct an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::Text {
                text: message.into(),
            }],
            is_error: true,
            markdown_formatted: None,
        }
    }

    /// Construct a successful JSON result.
    pub fn json(data: serde_json::Value) -> Self {
        Self {
            content: vec![ToolContent::Json { data }],
            is_error: false,
            markdown_formatted: None,
        }
    }

    /// Construct a result with both a JSON payload and a markdown rendering.
    pub fn success_with_markdown(data: serde_json::Value, markdown: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::Json { data }],
            is_error: false,
            markdown_formatted: Some(markdown.into()),
        }
    }

    /// Attach (or replace) the markdown rendering on an existing result.
    pub fn with_markdown(mut self, markdown: impl Into<String>) -> Self {
        self.markdown_formatted = Some(markdown.into());
        self
    }

    /// Returns text content blocks joined by newline, skipping JSON blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                ToolContent::Text { text } => Some(text.as_str()),
                ToolContent::Json { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns all content blocks serialised to a single string.
    pub fn output(&self) -> String {
        self.content
            .iter()
            .map(|c| match c {
                ToolContent::Text { text } => text.clone(),
                ToolContent::Json { data } => {
                    serde_json::to_string_pretty(data).unwrap_or_default()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns the markdown rendering when present and non-empty,
    /// otherwise falls back to [`Self::output`].
    pub fn output_for_llm(&self, prefer_markdown: bool) -> String {
        if prefer_markdown {
            if let Some(md) = self.markdown_formatted.as_deref() {
                let trimmed = md.trim();
                if !trimmed.is_empty() {
                    return md.to_string();
                }
            }
        }
        self.output()
    }
}

// ── ToolCallOptions ────────────────────────────────────────────────────────

/// Per-invocation options threaded from the agent loop into a tool's execution.
///
/// Ported from `openhuman::tools::traits::ToolCallOptions`.
/// Tools that opt in override handling to check these flags; tools that do
/// not need them keep working unchanged (the trait's caller passes `Default`).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ToolCallOptions {
    /// When `true`, the caller prefers a markdown rendering of the result
    /// for direct LLM consumption — markdown is cheaper than JSON in tokens.
    pub prefer_markdown: bool,
}

// ── PermissionLevel ────────────────────────────────────────────────────────

/// Permission level required to execute a tool.
///
/// Channels can set a maximum permission level to restrict which tools are
/// available. Tools requiring a level above the channel's maximum are
/// rejected before execution.
///
/// Ported from `openhuman::tools::traits::PermissionLevel`.
/// The level is `PartialOrd + Ord` — the curation function uses `<` to
/// gate tools against the channel's maximum (load-bearing invariant).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum PermissionLevel {
    /// No permission needed (metadata-only operations).
    #[serde(rename = "none")]
    None = 0,
    /// Read-only operations (file reads, memory recall, listing).
    #[default]
    #[serde(rename = "read_only")]
    ReadOnly = 1,
    /// Write operations (file writes, memory store).
    #[serde(rename = "write")]
    Write = 2,
    /// Command execution (shell, scripts).
    #[serde(rename = "execute")]
    Execute = 3,
    /// Dangerous / destructive operations (hardware, system-level).
    #[serde(rename = "dangerous")]
    Dangerous = 4,
}

impl std::fmt::Display for PermissionLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::ReadOnly => write!(f, "ReadOnly"),
            Self::Write => write!(f, "Write"),
            Self::Execute => write!(f, "Execute"),
            Self::Dangerous => write!(f, "Dangerous"),
        }
    }
}

// ── ToolCategory ───────────────────────────────────────────────────────────

/// Category of a tool — used by the curation function to scope which tools
/// a given sub-agent is allowed to see.
///
/// - `System`: built-in tools with direct host access.
/// - `Skill`: integration-facing tools that reach external services.
///
/// Ported from `openhuman::tools::traits::ToolCategory`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ToolCategory {
    /// Built-in tools with direct host access (default).
    #[default]
    System,
    /// Integration-facing tools that reach external services.
    Skill,
}

impl std::fmt::Display for ToolCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::Skill => write!(f, "skill"),
        }
    }
}

// ── ToolScope ──────────────────────────────────────────────────────────────

/// Controls where a tool is available.
///
/// Ported from `openhuman::tools::traits::ToolScope` with an Alzina
/// extension: the `Region(String)` variant is ADDED here (path (b)) because
/// openhuman's `ToolScope` does not expose a region tag. This extension is
/// required unconditionally so Plan 13-05's region-scoped curation test
/// compiles without ifdefs.
///
/// # REGION VARIANT (D13-05 / WARNING-1)
///
/// `ToolScope::Region(String)` is an Alzina addition — openhuman has no
/// equivalent variant. It lets `build_allowed_tools` select tools that
/// belong to a named code region (e.g. `"auth"`, `"billing"`). The variant
/// serialises as `{"region": "..."}` via the `#[serde(rename_all)]` applied
/// to the adjacent variants; see the test below for the round-trip.
///
/// `Copy` is intentionally absent because this enum carries an owned `String`
/// in the `Region` variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ToolScope {
    /// Available in agent loop, CLI, and RPC.
    All,
    /// Only available in the autonomous agent loop.
    AgentOnly,
    /// Only available via explicit CLI / RPC invocation (not autonomous agent).
    CliRpcOnly,
    // REGION VARIANT (D13-05 / WARNING-1): added by Alzina port (path b).
    // Openhuman has no region-scoped variant; this was added to make
    // Plan 13-05's region-scoped curation test unconditional.
    /// Restricted to a named code region (e.g. `"auth"`, `"billing"`).
    Region(String),
}

impl Default for ToolScope {
    fn default() -> Self {
        Self::All
    }
}

// ── Tool trait ─────────────────────────────────────────────────────────────

/// Core tool trait — implement for any Alzina tool capability.
///
/// Every method is **pure metadata** (no I/O, no futures). Tool invocation
/// crosses the daemon HTTP boundary via `Tool::endpoint(api_base)`; that
/// async surface lives in the daemon's handler, not here.
///
/// The trait is intentionally `dyn`-compatible: no generic methods, no `Self`
/// in invalid positions. `&dyn Tool` is used by the adapter in Plan 13-05.
///
/// Optional methods have default implementations so concrete tool types only
/// override what they need.
pub trait Tool: Send + Sync {
    // ── Required methods ──────────────────────────────────────────────────

    /// Tool name as the SDK sees it.
    ///
    /// Return the bare name — **no** `mcp__alzina__` prefix. The SDK
    /// prepends the prefix per `claude_agent_sdk.rs:49`.
    fn name(&self) -> &str;

    /// Human-readable description for the model.
    fn description(&self) -> &str;

    /// JSON Schema object for the tool's input parameters.
    ///
    /// Maps to `CustomToolDefinition.input_schema`.
    fn input_schema(&self) -> serde_json::Value;

    /// Derive the HTTP endpoint for this tool given the daemon's base URL.
    ///
    /// Maps to `CustomToolDefinition.endpoint`. The daemon's handler at
    /// this URL performs the actual tool invocation and emits audit events
    /// (D13-16 — audit stays in the handler, not the adapter).
    fn endpoint(&self, api_base: &str) -> String;

    // ── Optional methods (all have defaults) ──────────────────────────────

    /// HTTP method override for the endpoint. Default: `None` (sidecar uses POST).
    fn method(&self) -> Option<&str> {
        None
    }

    /// Optional timeout in milliseconds for this tool's HTTP call.
    /// Default: `None` (sidecar applies its own default).
    fn timeout_ms(&self) -> Option<u64> {
        None
    }

    /// When `true`, the sidecar fetches the endpoint as an NDJSON progress
    /// stream rather than a single-shot HTTP call. Default: `false`.
    fn stream_progress(&self) -> bool {
        false
    }

    /// Idle timeout (ms) applied when `stream_progress` is `true`. Default:
    /// `None` (sidecar default of 60 s applies).
    fn idle_timeout_ms(&self) -> Option<u64> {
        None
    }

    /// Permission level required to execute this tool.
    ///
    /// The curation function uses `<` comparisons against the seam's
    /// maximum to gate tool access. Default: `PermissionLevel::ReadOnly`.
    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    /// Category of this tool. Default: `ToolCategory::System`.
    fn category(&self) -> ToolCategory {
        ToolCategory::System
    }

    /// Scope of this tool — where it is available. Default: `ToolScope::All`.
    fn scope(&self) -> ToolScope {
        ToolScope::All
    }

    /// Per-tool character cap on the result body sent to the model.
    ///
    /// When `Some(cap)`, the agent's tool loop truncates oversized bodies
    /// and appends a truncation marker before threading into history.
    /// When `None` (default), no per-tool cap applies.
    fn max_result_size_chars(&self) -> Option<usize> {
        None
    }

    /// Whether two concurrent invocations are safe to run in parallel inside
    /// a single LLM iteration. Default: `false`.
    fn is_concurrency_safe(&self, _args: &serde_json::Value) -> bool {
        false
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test (i): dyn-compatible compile-only test ─────────────────────────
    //
    // If the trait is not dyn-compatible this line will fail to compile.
    // Plan 13-05's `From<&dyn Tool>` adapter depends on this.
    fn _accepts_dyn(_: &dyn Tool) {}

    // ── Stub impl for default-method tests ────────────────────────────────

    struct StubTool;

    // Test (iv): all-sync impl — no async-trait annotation required.
    // Compile-only test: if the trait were not sync, a plain impl
    // would require an async-trait annotation to compile.
    impl Tool for StubTool {
        fn name(&self) -> &str {
            "stub_tool"
        }

        fn description(&self) -> &str {
            "A minimal stub tool for testing"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }

        fn endpoint(&self, api_base: &str) -> String {
            format!("{api_base}/tools/stub_tool")
        }
    }

    // ── Test (ii): default trait methods ──────────────────────────────────

    #[test]
    fn default_method_returns() {
        let tool = StubTool;
        assert_eq!(tool.name(), "stub_tool");
        assert_eq!(tool.description(), "A minimal stub tool for testing");
        assert_eq!(tool.method(), None);
        assert_eq!(tool.timeout_ms(), None);
        assert!(!tool.stream_progress());
        assert_eq!(tool.idle_timeout_ms(), None);
        assert_eq!(tool.permission_level(), PermissionLevel::ReadOnly);
        assert_eq!(tool.category(), ToolCategory::System);
        assert_eq!(tool.scope(), ToolScope::All);
        assert_eq!(tool.max_result_size_chars(), None);
        assert!(!tool.is_concurrency_safe(&serde_json::Value::Null));
    }

    #[test]
    fn endpoint_derives_from_api_base() {
        let tool = StubTool;
        let ep = tool.endpoint("http://127.0.0.1:3001");
        assert_eq!(ep, "http://127.0.0.1:3001/tools/stub_tool");
    }

    // ── Test (ii cont.): dyn dispatch with default methods ────────────────

    #[test]
    fn dyn_compatible_dispatch() {
        let tool: &dyn Tool = &StubTool;
        _accepts_dyn(tool);
        assert_eq!(tool.name(), "stub_tool");
        assert_eq!(tool.permission_level(), PermissionLevel::ReadOnly);
    }

    // ── Test (iii): serde round-trips for enums ────────────────────────────

    #[test]
    fn permission_level_serde_round_trip() {
        for level in [
            PermissionLevel::None,
            PermissionLevel::ReadOnly,
            PermissionLevel::Write,
            PermissionLevel::Execute,
            PermissionLevel::Dangerous,
        ] {
            let s = serde_json::to_string(&level).unwrap();
            let back: PermissionLevel = serde_json::from_str(&s).unwrap();
            assert_eq!(back, level, "round-trip failed for {level:?}");
        }
    }

    #[test]
    fn tool_category_serde_round_trip() {
        for cat in [ToolCategory::System, ToolCategory::Skill] {
            let s = serde_json::to_string(&cat).unwrap();
            let back: ToolCategory = serde_json::from_str(&s).unwrap();
            assert_eq!(back, cat, "round-trip failed for {cat:?}");
        }
        // snake_case serialisation is load-bearing for agent definition files.
        assert_eq!(serde_json::to_string(&ToolCategory::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&ToolCategory::Skill).unwrap(), "\"skill\"");
    }

    #[test]
    fn tool_scope_serde_round_trip() {
        for scope in [ToolScope::All, ToolScope::AgentOnly, ToolScope::CliRpcOnly] {
            let s = serde_json::to_string(&scope).unwrap();
            let back: ToolScope = serde_json::from_str(&s).unwrap();
            assert_eq!(back, scope, "round-trip failed for {scope:?}");
        }
    }

    /// REGION VARIANT round-trip — D13-05 / WARNING-1.
    ///
    /// This test is the load-bearing proof that `ToolScope::Region(String)`
    /// survives a serde round-trip. Plan 13-05's region-scoped curation test
    /// depends on deserialising `Region` variants from stored config.
    #[test]
    fn tool_scope_region_serde_round_trip() {
        let original = ToolScope::Region("auth".into());
        let serialised = serde_json::to_string(&original).unwrap();
        let restored: ToolScope = serde_json::from_str(&serialised).unwrap();
        assert_eq!(
            restored,
            ToolScope::Region("auth".into()),
            "ToolScope::Region round-trip failed; serialised form was: {serialised}"
        );
    }

    // ── PermissionLevel ordering (load-bearing for curation gating) ────────

    #[test]
    fn permission_level_ordering() {
        assert!(PermissionLevel::None < PermissionLevel::ReadOnly);
        assert!(PermissionLevel::ReadOnly < PermissionLevel::Write);
        assert!(PermissionLevel::Write < PermissionLevel::Execute);
        assert!(PermissionLevel::Execute < PermissionLevel::Dangerous);
    }

    #[test]
    fn permission_level_default_is_read_only() {
        assert_eq!(PermissionLevel::default(), PermissionLevel::ReadOnly);
    }

    // ── ToolResult tests ───────────────────────────────────────────────────

    #[test]
    fn tool_result_success_round_trip() {
        let r = ToolResult::success("hello");
        let s = serde_json::to_string(&r).unwrap();
        let back: ToolResult = serde_json::from_str(&s).unwrap();
        assert!(!back.is_error);
        assert_eq!(back.text(), "hello");
    }

    #[test]
    fn tool_result_error_flag() {
        let r = ToolResult::error("boom");
        assert!(r.is_error);
        assert_eq!(r.output(), "boom");
    }
}
