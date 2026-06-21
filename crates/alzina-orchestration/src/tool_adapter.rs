//! The single adapter from `alzina_core::tools::Tool` to `CustomToolDefinition`.
//!
//! ## Role
//!
//! This is the SINGLE conversion point (D13-15) that maps every `impl Tool`
//! onto the sidecar wire struct `CustomToolDefinition`. No other file should
//! construct a `CustomToolDefinition` from a `&dyn Tool`.
//!
//! ## Prefix discipline
//!
//! The adapter emits the bare tool name returned by `Tool::name()`. It does
//! NOT prepend `mcp__alzina__`. The SDK sidecar applies the prefix via its
//! registration gate (`claude_agent_sdk.rs` constant `MCP_ALZINA_PREFIX`).
//! Prepending the prefix here would double-apply it.
//!
//! ## Audit-event uniformity (D13-16)
//!
//! The adapter does NOT inject audit calls. Audit emission stays in the daemon
//! HTTP handler bound to `CustomToolDefinition.endpoint`, which is the same
//! path the existing `weave_*` tools use. A tool ported via this adapter
//! participates in the SAME audit-events pipeline without any code change at
//! the adapter layer — the endpoint field is the integration point.
//!
//! ## Orphan rule
//!
//! `alzina-core` does not depend on `alzina-orchestration` (alzina-core is
//! upstream). `CustomToolDefinition` lives in `alzina-orchestration`. The
//! `From` impl therefore lives here — in the crate that owns the target type.
//!
//! ## Wire struct target
//!
//! ```text
//! CustomToolDefinition {
//!   name, description, input_schema, endpoint,
//!   method, timeout_ms, stream_progress, idle_timeout_ms
//! }
//! ```
//!
//! All eight fields are populated by this adapter. See
//! `runner::sidecar_protocol::CustomToolDefinition` for field documentation.

use alzina_core::tools::Tool;

use crate::runner::sidecar_protocol::CustomToolDefinition;

/// Convert a `(&dyn Tool, api_base)` tuple into a `CustomToolDefinition`.
///
/// The tuple form is the orphan-rule-safe way to thread the `api_base`
/// argument through a `From` impl — `Tool::endpoint(api_base)` requires the
/// daemon base URL, which is not part of the trait itself.
///
/// Prefer this impl over constructing `CustomToolDefinition` manually from
/// a `&dyn Tool`. Any future field added to `CustomToolDefinition` that maps
/// from a `Tool` method should be handled here and nowhere else.
///
/// # MCP_ALZINA_PREFIX NOT applied by adapter
///
/// The adapter emits `tool.name()` verbatim. The prefix is applied by the SDK
/// sidecar at registration time. An assertion in the test module pins this.
impl<'a> From<(&'a dyn Tool, &'a str)> for CustomToolDefinition {
    fn from((tool, api_base): (&'a dyn Tool, &'a str)) -> Self {
        tool_to_definition(tool, api_base)
    }
}

/// Free-function backing the `From` impl. Exported so callers that already
/// have a concrete reference can avoid the tuple syntax if preferred.
///
/// Both paths (`From` and direct call) are equivalent — use whichever is
/// cleaner at the call site.
pub fn tool_to_definition(tool: &dyn Tool, api_base: &str) -> CustomToolDefinition {
    CustomToolDefinition {
        // Bare name — MCP_ALZINA_PREFIX NOT applied by adapter (see module doc).
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        input_schema: tool.input_schema(),
        // D13-16: endpoint is the integration point for audit events.
        // The daemon handler bound to this URL emits audit events, not the adapter.
        endpoint: tool.endpoint(api_base),
        method: tool.method().map(|s| s.to_string()),
        timeout_ms: tool.timeout_ms(),
        stream_progress: tool.stream_progress(),
        idle_timeout_ms: tool.idle_timeout_ms(),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use alzina_core::tools::Tool;
    use serde_json::json;

    use super::*;

    // ── Stub tools ────────────────────────────────────────────────────────

    /// Full-featured stub that overrides all optional fields.
    struct FullStubTool;

    impl Tool for FullStubTool {
        fn name(&self) -> &str {
            "full_stub"
        }

        fn description(&self) -> &str {
            "A fully populated stub tool for round-trip testing"
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "required": ["arg"],
                "properties": {
                    "arg": { "type": "string" }
                }
            })
        }

        fn endpoint(&self, api_base: &str) -> String {
            format!("{api_base}/api/v1/tools/full_stub")
        }

        fn method(&self) -> Option<&str> {
            Some("GET")
        }

        fn timeout_ms(&self) -> Option<u64> {
            Some(5_000)
        }

        fn stream_progress(&self) -> bool {
            true
        }

        fn idle_timeout_ms(&self) -> Option<u64> {
            Some(30_000)
        }
    }

    /// Minimal stub that relies on all default trait methods.
    struct MinimalStubTool;

    impl Tool for MinimalStubTool {
        fn name(&self) -> &str {
            "minimal_stub"
        }

        fn description(&self) -> &str {
            "Minimal stub — no optional overrides"
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }

        fn endpoint(&self, api_base: &str) -> String {
            format!("{api_base}/api/v1/tools/minimal_stub")
        }
    }

    // ── Test 1: round-trip — all eight fields ─────────────────────────────

    /// Full round-trip: every CustomToolDefinition field equals the value the
    /// stub tool reports. This is the primary D13-15 pin test.
    #[test]
    fn adapter_round_trip_all_fields() {
        let stub = FullStubTool;
        let api_base = "http://localhost:3001";

        // MCP_ALZINA_PREFIX NOT applied by adapter — assert bare name.
        let def = CustomToolDefinition::from((&stub as &dyn Tool, api_base));

        assert_eq!(def.name, "full_stub");
        assert!(!def.name.starts_with("mcp__alzina__"),
            "adapter must emit bare name, not prefixed; got: {}", def.name);
        assert_eq!(def.description, "A fully populated stub tool for round-trip testing");
        assert_eq!(
            def.input_schema,
            json!({
                "type": "object",
                "required": ["arg"],
                "properties": { "arg": { "type": "string" } }
            })
        );
        // D13-16: endpoint pin — adapter routes through Tool::endpoint, which
        // is the same path daemon handlers bind to for audit-event emission.
        assert_eq!(def.endpoint, "http://localhost:3001/api/v1/tools/full_stub");
        assert_eq!(def.method, Some("GET".to_string()));
        assert_eq!(def.timeout_ms, Some(5_000));
        assert!(def.stream_progress);
        assert_eq!(def.idle_timeout_ms, Some(30_000));
    }

    // ── Test 2: endpoint varies with api_base ─────────────────────────────

    /// Pass two different api_base values and confirm the endpoint field
    /// differs accordingly. This pins the two-argument tuple shape.
    #[test]
    fn adapter_endpoint_varies_with_api_base() {
        let stub = FullStubTool;

        let def_dev = CustomToolDefinition::from((&stub as &dyn Tool, "http://localhost:3001"));
        let def_prod = CustomToolDefinition::from((&stub as &dyn Tool, "https://api.example.com"));

        assert_eq!(def_dev.endpoint, "http://localhost:3001/api/v1/tools/full_stub");
        assert_eq!(def_prod.endpoint, "https://api.example.com/api/v1/tools/full_stub");
        assert_ne!(def_dev.endpoint, def_prod.endpoint);
    }

    // ── Test 3: default trait methods produce expected zero-values ────────

    /// A stub that does NOT override any optional method should produce
    /// method: None, timeout_ms: None, stream_progress: false, idle_timeout_ms: None.
    #[test]
    fn adapter_default_method_passthrough() {
        let stub = MinimalStubTool;
        let def = CustomToolDefinition::from((&stub as &dyn Tool, "http://localhost:3001"));

        assert_eq!(def.name, "minimal_stub");
        // MCP_ALZINA_PREFIX NOT applied by adapter
        assert!(!def.name.starts_with("mcp__alzina__"),
            "adapter must emit bare name, not prefixed");
        assert_eq!(def.method, None);
        assert_eq!(def.timeout_ms, None);
        assert!(!def.stream_progress);
        assert_eq!(def.idle_timeout_ms, None);
    }

    // ── Test 4: audit-event uniformity (D13-16) ───────────────────────────

    /// The endpoint field in the produced CustomToolDefinition is exactly the
    /// value returned by Tool::endpoint(api_base). This pins D13-16: the adapter
    /// routes through the endpoint field, which is the same path daemon HTTP
    /// handlers use to emit audit events for the existing weave_* tools.
    ///
    /// A tool ported via this adapter participates in the SAME audit pipeline
    /// without any code change — endpoint routing is the integration point.
    #[test]
    fn adapter_endpoint_equals_tool_endpoint_d13_16() {
        let stub = FullStubTool;
        let api_base = "http://127.0.0.1:3001";

        // What Tool::endpoint reports directly.
        let expected_endpoint = stub.endpoint(api_base);

        // What the adapter produces.
        let def = CustomToolDefinition::from((&stub as &dyn Tool, api_base));

        assert_eq!(
            def.endpoint,
            expected_endpoint,
            "D13-16 violation: adapter endpoint does not match Tool::endpoint(api_base)"
        );
        // Pin the exact string so the test is load-bearing.
        assert_eq!(def.endpoint, "http://127.0.0.1:3001/api/v1/tools/full_stub");
    }

    // ── Test 5: free-function alias produces identical result ─────────────

    /// tool_to_definition and From<(...)> must produce identical results.
    /// If one is changed without the other, this test catches the drift.
    #[test]
    fn tool_to_definition_matches_from_impl() {
        let stub = FullStubTool;
        let api_base = "http://localhost:3001";

        let via_from = CustomToolDefinition::from((&stub as &dyn Tool, api_base));
        let via_fn = tool_to_definition(&stub, api_base);

        assert_eq!(via_from.name, via_fn.name);
        assert_eq!(via_from.description, via_fn.description);
        assert_eq!(via_from.input_schema, via_fn.input_schema);
        assert_eq!(via_from.endpoint, via_fn.endpoint);
        assert_eq!(via_from.method, via_fn.method);
        assert_eq!(via_from.timeout_ms, via_fn.timeout_ms);
        assert_eq!(via_from.stream_progress, via_fn.stream_progress);
        assert_eq!(via_from.idle_timeout_ms, via_fn.idle_timeout_ms);
    }
}
