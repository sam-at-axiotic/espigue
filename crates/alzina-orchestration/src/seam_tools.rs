//! The single curation point for tools a sub-agent sees (D13-03).
//!
//! ## Type-name decision: `CurationContext` not `Seam`
//!
//! CONTEXT.md D13-03 originally specified `build_allowed_tools(seam: &Seam)`.
//! The name `Seam` is already taken at
//! `crates/alzina-daemon/src/observation/seam_detector.rs:78` for the
//! **observation domain**: `Seam { kind: SeamKind, severity: SeamSeverity,
//! evidence: Vec<String> }` — bug findings, not tool-curation surfaces.
//! `CurationContext` is used here instead: it is self-describing and
//! orthogonal to the observation Seam. The naming decision is recorded
//! here on disk so it is traceable without a git-blame dig.
//!
//! ## No-doubles invariant (D13-04)
//!
//! For every tool in a seam, the output of `build_allowed_tools` must never
//! contain both `X` and `mcp__alzina__X` simultaneously. A regression on this
//! invariant fails the build via `build_allowed_tools_never_contains_both_bare_and_prefixed`.
//!
//! ## Three-way classification (D13-01)
//!
//! Each tool is resolved through one of three `CurationPolicy` variants:
//! - `UseSdkBuiltin`: emit the bare name — the SDK handles it (read-only built-ins).
//! - `ShipAsMcpAlzina`: emit the `mcp__alzina__` prefixed name — Alzina-owned tool.
//! - `GovernanceShim`: emit the `mcp__alzina__` prefixed name — governance-gated shim.
//!
//! See `.planning/intel/sdk-builtin-tools-2026-05.md` for the 12 SDK built-in
//! names (Probe 1 output) and `.planning/intel/sdk-collision-shadowing-2026-05.md`
//! for the collision-rejection evidence (Probe 2 output, outcome: `SDK rejects
//! collision`). These two intel docs are the source of truth for the prefix
//! discipline encoded here.
//!
//! ## Region-scoped access (D13-05)
//!
//! `CurationContext.region` filters `available_tools` by `Tool::scope()`.
//! When `region` is `Some("auth")`, only tools whose `scope()` returns
//! `ToolScope::All` or `ToolScope::Region("auth")` are emitted. This
//! falls naturally out of the same function — no separate path is needed.

use std::collections::HashMap;

use alzina_core::tools::{Tool, ToolScope};

use crate::runner::claude_agent_sdk::MCP_ALZINA_PREFIX;

// ── CurationPolicy ────────────────────────────────────────────────────────

/// Per-tool curation policy — encodes the three-way classification (D13-01).
///
/// The default policy is inferred from `CurationContext.sdk_builtins`:
/// - If the tool name is in `sdk_builtins` → `UseSdkBuiltin`.
/// - Otherwise → `ShipAsMcpAlzina`.
///
/// Per-tool overrides in `CurationContext.policy_overrides` win over inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurationPolicy {
    /// Use the SDK built-in directly; emit the bare name in `allowed_tools`.
    ///
    /// Applied by default when `tool.name()` is in `sdk_builtins`. Training-data
    /// aligned — the model already knows these tools. No prefix needed.
    UseSdkBuiltin,

    /// Ship as an `mcp__alzina__`-prefixed MCP tool.
    ///
    /// Applied by default for any Alzina-owned tool whose name does not collide
    /// with an SDK built-in. The prefix bypasses the sidecar A7 gate, which
    /// rejects bare names that collide with `BUILTIN_TOOLS`.
    ShipAsMcpAlzina,

    /// Governance shim — ship as an `mcp__alzina__`-prefixed MCP tool.
    ///
    /// For tools that wrap an SDK built-in with an audit / governance gate.
    /// Bare-name shadowing is architecturally impossible (sidecar A7 gate — see
    /// `.planning/intel/sdk-collision-shadowing-2026-05.md`), so governance
    /// shims always use the `mcp__alzina__` prefix regardless of whether the
    /// underlying built-in is in `sdk_builtins`.
    ///
    /// // PROBE-2 DEPENDENCY: this conservative choice (always prefix for shims)
    /// // follows directly from the Probe 2 finding (`SDK rejects collision`).
    /// // The sidecar A7 gate unconditionally rejects bare-name custom tools
    /// // that collide with BUILTIN_TOOLS. Even in a future SDK version where
    /// // shadowing were possible, using the prefix keeps the audit trail
    /// // unambiguous. Cross-reference: `.planning/intel/sdk-collision-shadowing-2026-05.md`.
    GovernanceShim,
}

// ── CurationContext ───────────────────────────────────────────────────────

/// The scope input to `build_allowed_tools`.
///
/// Carries the tool registry, SDK built-in names, per-tool policy overrides,
/// and (optionally) a region tag for region-scoped access (D13-05).
pub struct CurationContext {
    /// Code-region / specialist tag. When `Some(r)`, only tools whose
    /// `scope()` is `ToolScope::All` or `ToolScope::Region(r)` are emitted.
    /// When `None`, all tools in `available_tools` pass the region gate.
    pub region: Option<String>,

    /// The registry of tools this seam can offer. Order matters only when
    /// two tools share the same name — the last one wins (no panic).
    pub available_tools: Vec<Box<dyn Tool + Send + Sync>>,

    /// Names of SDK built-in tools (Probe 1 output).
    /// Any tool whose `name()` is in this vec defaults to `UseSdkBuiltin`
    /// policy unless overridden.
    ///
    /// Populated from `.planning/intel/sdk-builtin-tools-2026-05.md`
    /// (SDK version 0.1.77, 12 built-in names).
    pub sdk_builtins: Vec<String>,

    /// Per-tool policy overrides. Key is `tool.name()`. Takes precedence over
    /// the default inference from `sdk_builtins`.
    pub policy_overrides: HashMap<String, CurationPolicy>,
}

impl Default for CurationContext {
    /// Default context: no region, no tools, sdk_builtins seeded from Probe 1.
    ///
    /// The SDK built-in names are sourced from
    /// `.planning/intel/sdk-builtin-tools-2026-05.md` (Probe 1 output,
    /// SDK version `@anthropic-ai/claude-agent-sdk@0.1.77`, observation
    /// date 2026-05-18). 12 names: Read, Write, Edit, Bash, Glob, Grep,
    /// Agent, WebFetch, WebSearch, NotebookEdit, TodoRead, TodoWrite.
    fn default() -> Self {
        Self {
            region: None,
            available_tools: Vec::new(),
            sdk_builtins: default_sdk_builtins(),
            policy_overrides: HashMap::new(),
        }
    }
}

/// The 12 SDK built-in tool names from Probe 1 (2026-05).
///
/// Source: `sidecar/claude-agent/src/index.ts:237–241` (BUILTIN_TOOLS constant).
/// SDK version: `@anthropic-ai/claude-agent-sdk@0.1.77`.
/// Cross-reference: `.planning/intel/sdk-builtin-tools-2026-05.md`.
fn default_sdk_builtins() -> Vec<String> {
    [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "Agent", "WebFetch", "WebSearch", "NotebookEdit",
        "TodoRead", "TodoWrite",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// ── build_allowed_tools ───────────────────────────────────────────────────

/// Return the list of tool names a sub-agent is allowed to call (D13-03).
///
/// The returned `Vec<String>` is the exact shape `SidecarOptions.allowed_tools`
/// consumes (see `runner::sidecar_protocol::SidecarOptions`, line 48).
///
/// ## Algorithm
///
/// For each tool in `scope.available_tools`:
///
/// 1. **Region gate (D13-05):** If `scope.region` is `Some(r)`, the tool is
///    only emitted when its `scope()` method returns `ToolScope::All` or
///    `ToolScope::Region(r)`. If `scope.region` is `None`, all tools pass.
///
/// 2. **Policy lookup:** Check `scope.policy_overrides` for a per-tool
///    override. If absent, infer: name in `sdk_builtins` → `UseSdkBuiltin`,
///    otherwise → `ShipAsMcpAlzina`.
///
/// 3. **Name emission:**
///    - `UseSdkBuiltin` → emit `tool.name()` (bare).
///    - `ShipAsMcpAlzina` | `GovernanceShim` → emit `mcp__alzina__{name}`.
///
/// 4. **No-doubles (D13-04):** The output never contains both `X` and
///    `mcp__alzina__X`. Policy lookup enforces this per-tool; the invariant
///    test pins it end-to-end.
///
/// 5. **Last-wins deduplication:** Two tools with the same `name()` in
///    `available_tools` — the last one's emitted name wins (no panic).
///
/// This function MUST NOT panic.
pub fn build_allowed_tools(scope: &CurationContext) -> Vec<String> {
    let mut seen: HashMap<String, String> = HashMap::new();

    for tool in &scope.available_tools {
        // ── Name-shape gate (WR-02) ───────────────────────────────────────
        // Reject tools whose `name()` already starts with `mcp__alzina__`.
        // Without this guard, `ShipAsMcpAlzina` would re-prefix to
        // `mcp__alzina__mcp__alzina__foo` — a double-prefix attack /
        // accidental drift. Log loudly and skip the tool rather than
        // panicking (the daemon must keep serving other seams), matching
        // Sam's "degradation must be loud" rule.
        if tool.name().starts_with(MCP_ALZINA_PREFIX) {
            tracing::error!(
                target: "seam_tools",
                tool_name = %tool.name(),
                "rejecting tool registration: name already starts with mcp__alzina__ \
                 prefix (double-prefix risk). Tools must be registered with their bare \
                 name — the prefix is applied by build_allowed_tools."
            );
            continue;
        }

        // ── Region gate (D13-05) ──────────────────────────────────────────
        if let Some(required_region) = &scope.region {
            let passes = match tool.scope() {
                ToolScope::All => true,
                ToolScope::Region(ref r) => r == required_region,
                // AgentOnly and CliRpcOnly are not region-gated but also
                // not region-scoped — they pass when no region filter is active.
                // When a region filter IS active, only All and matching Region pass.
                _ => false,
            };
            if !passes {
                continue;
            }
        }

        // ── Policy lookup ─────────────────────────────────────────────────
        let policy = scope
            .policy_overrides
            .get(tool.name())
            .copied()
            .unwrap_or_else(|| {
                if scope.sdk_builtins.iter().any(|b| b == tool.name()) {
                    CurationPolicy::UseSdkBuiltin
                } else {
                    CurationPolicy::ShipAsMcpAlzina
                }
            });

        // ── Name emission ─────────────────────────────────────────────────
        let emitted = match policy {
            CurationPolicy::UseSdkBuiltin => tool.name().to_string(),
            CurationPolicy::ShipAsMcpAlzina | CurationPolicy::GovernanceShim => {
                // PROBE-2 DEPENDENCY: always prefix governance shims.
                // Cross-reference: .planning/intel/sdk-collision-shadowing-2026-05.md
                format!("{}{}", MCP_ALZINA_PREFIX, tool.name())
            }
        };

        // Last-wins: insert by tool name key.
        seen.insert(tool.name().to_string(), emitted);
    }

    // Deterministic output order (WR-03). `HashMap::into_values()` iterates
    // in unspecified order which varies per process run — that causes:
    //   - spurious diffs in audit logs and integration test fixtures;
    //   - per-session KV-cache invalidation on the model side, because the
    //     `allowed_tools` list is part of the prompt-stable prefix;
    //   - harder incident triage when the same input produces different
    //     wire payloads.
    // Lexical sort is stable across runs and across the network boundary.
    let mut out: Vec<String> = seen.into_values().collect();
    out.sort();
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use alzina_core::tools::{
        PermissionLevel, ToolCallOptions, ToolCategory, ToolContent, ToolResult, ToolScope,
    };
    use serde_json::json;

    use super::*;
    use crate::runner::claude_agent_sdk::MCP_ALZINA_PREFIX;

    // ── Stub tools ────────────────────────────────────────────────────────

    /// Stub that mimics an SDK built-in name — "Read" collides with the SDK.
    struct StubReadTool;

    impl alzina_core::tools::Tool for StubReadTool {
        fn name(&self) -> &str {
            "Read"
        }

        fn description(&self) -> &str {
            "Alzina stub for SDK built-in Read"
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }

        fn endpoint(&self, api_base: &str) -> String {
            format!("{api_base}/api/v1/tools/read")
        }
    }

    /// Stub with an SDK-built-in name — "Grep" collides with the SDK.
    struct StubGrepTool;

    impl alzina_core::tools::Tool for StubGrepTool {
        fn name(&self) -> &str {
            "Grep"
        }

        fn description(&self) -> &str {
            "Alzina stub for SDK built-in Grep"
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }

        fn endpoint(&self, api_base: &str) -> String {
            format!("{api_base}/api/v1/tools/grep")
        }
    }

    /// Stub with a non-colliding name — "foo" is not an SDK built-in.
    struct StubFooTool;

    impl alzina_core::tools::Tool for StubFooTool {
        fn name(&self) -> &str {
            "foo"
        }

        fn description(&self) -> &str {
            "Alzina custom tool — not an SDK built-in"
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }

        fn endpoint(&self, api_base: &str) -> String {
            format!("{api_base}/api/v1/tools/foo")
        }
    }

    /// Stub whose name already starts with `mcp__alzina__` — an
    /// ill-formed registration that WR-02 rejects.
    struct AlreadyPrefixedTool;

    impl alzina_core::tools::Tool for AlreadyPrefixedTool {
        fn name(&self) -> &str {
            "mcp__alzina__bogus"
        }

        fn description(&self) -> &str {
            "Tool registered with an already-prefixed name (should be rejected)"
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }

        fn endpoint(&self, api_base: &str) -> String {
            format!("{api_base}/api/v1/tools/bogus")
        }
    }

    /// Stub with a region-scoped `scope()` override.
    struct RegionScopedTool;

    impl alzina_core::tools::Tool for RegionScopedTool {
        fn name(&self) -> &str {
            "auth_checker"
        }

        fn description(&self) -> &str {
            "Auth-region-scoped tool"
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }

        fn endpoint(&self, api_base: &str) -> String {
            format!("{api_base}/api/v1/tools/auth_checker")
        }

        fn scope(&self) -> ToolScope {
            ToolScope::Region("auth".into())
        }
    }

    // Helper to suppress unused-import warnings for types that are part of the
    // public surface but not directly exercised in every test.
    fn _use_types() {
        let _ = PermissionLevel::ReadOnly;
        let _ = ToolCategory::System;
        let _ = ToolCallOptions::default();
        let _ = ToolResult::success("ok");
        let _ = ToolContent::Text { text: "ok".into() };
    }

    // ── Test 1: no-doubles invariant (D13-04) ─────────────────────────────

    /// No-doubles invariant: for any seam fixture, the output never contains
    /// both `X` and `mcp__alzina__X` (D13-04).
    ///
    /// Fixture: one tool whose name is an SDK built-in ("Read") with default
    /// `UseSdkBuiltin` policy, and one tool whose name is not a built-in ("foo")
    /// with default `ShipAsMcpAlzina` policy. The built-in emits "Read" (bare);
    /// "foo" emits "mcp__alzina__foo". Neither pair is a bare+prefixed double.
    ///
    /// This test pins that the algorithm never emits both sides of the double.
    #[test]
    fn build_allowed_tools_never_contains_both_bare_and_prefixed() {
        let ctx = CurationContext {
            available_tools: vec![
                Box::new(StubReadTool),
                Box::new(StubFooTool),
            ],
            ..CurationContext::default()
        };

        let tools = build_allowed_tools(&ctx);

        // Core invariant: for every prefixed entry, the bare suffix is absent.
        for t in &tools {
            if let Some(bare) = t.strip_prefix(MCP_ALZINA_PREFIX) {
                assert!(
                    !tools.iter().any(|other| other == bare),
                    "no-doubles invariant violated: both `{bare}` and \
                     `{MCP_ALZINA_PREFIX}{bare}` present in output: {tools:?}"
                );
            }
        }

        // Additional directional assertions: "Read" is emitted as bare;
        // "foo" is emitted as prefixed.
        assert!(tools.contains(&"Read".to_string()),
            "SDK built-in 'Read' must be emitted as bare name");
        assert!(tools.contains(&format!("{MCP_ALZINA_PREFIX}foo")),
            "'foo' must be emitted as prefixed");
        assert!(!tools.contains(&format!("{MCP_ALZINA_PREFIX}Read")),
            "SDK built-in 'Read' must NOT be emitted as prefixed");
    }

    // ── Test 2: empty context returns empty ───────────────────────────────

    /// Empty `available_tools` → empty output. No panic, no defaults injected.
    #[test]
    fn build_allowed_tools_empty_context_returns_empty() {
        let ctx = CurationContext::default();
        let tools = build_allowed_tools(&ctx);
        assert!(tools.is_empty(), "empty context must produce empty output");
    }

    // ── Test 3: policy override wins ──────────────────────────────────────

    /// A tool whose name is in `sdk_builtins` AND has a `ShipAsMcpAlzina`
    /// override emits the prefixed name — override takes precedence.
    #[test]
    fn build_allowed_tools_policy_override_wins() {
        let mut overrides = HashMap::new();
        overrides.insert("Read".to_string(), CurationPolicy::ShipAsMcpAlzina);

        let ctx = CurationContext {
            available_tools: vec![Box::new(StubReadTool)],
            policy_overrides: overrides,
            ..CurationContext::default()
        };

        let tools = build_allowed_tools(&ctx);

        // "Read" is in sdk_builtins, but override says ShipAsMcpAlzina.
        assert!(
            tools.contains(&format!("{MCP_ALZINA_PREFIX}Read")),
            "policy override must emit prefixed name; got: {tools:?}"
        );
        assert!(
            !tools.contains(&"Read".to_string()),
            "bare 'Read' must not appear when override is ShipAsMcpAlzina"
        );
    }

    // ── Test 4: three-way classification (D13-01) ─────────────────────────

    /// One tool of each CurationPolicy produces the expected three-string
    /// output exactly. This is the primary D13-01 encode-and-test pin.
    #[test]
    fn build_allowed_tools_three_way_classification() {
        // Tool 1: SDK built-in "Read" — will infer UseSdkBuiltin.
        // Tool 2: custom "foo" — will infer ShipAsMcpAlzina.
        // Tool 3: "Grep" overridden to GovernanceShim.
        let mut overrides = HashMap::new();
        overrides.insert("Grep".to_string(), CurationPolicy::GovernanceShim);

        let ctx = CurationContext {
            available_tools: vec![
                Box::new(StubReadTool),   // UseSdkBuiltin (inferred)
                Box::new(StubFooTool),    // ShipAsMcpAlzina (inferred)
                Box::new(StubGrepTool),   // GovernanceShim (overridden)
            ],
            policy_overrides: overrides,
            ..CurationContext::default()
        };

        let tools = build_allowed_tools(&ctx);
        assert_eq!(tools.len(), 3, "expected 3 tools; got: {tools:?}");

        // UseSdkBuiltin → bare name
        assert!(tools.contains(&"Read".to_string()),
            "UseSdkBuiltin 'Read' must be bare");
        // ShipAsMcpAlzina → prefixed
        assert!(tools.contains(&format!("{MCP_ALZINA_PREFIX}foo")),
            "ShipAsMcpAlzina 'foo' must be prefixed");
        // GovernanceShim → prefixed (PROBE-2 DEPENDENCY)
        assert!(tools.contains(&format!("{MCP_ALZINA_PREFIX}Grep")),
            "GovernanceShim 'Grep' must be prefixed");

        // No-doubles as a sanity check.
        for t in &tools {
            if let Some(bare) = t.strip_prefix(MCP_ALZINA_PREFIX) {
                assert!(!tools.iter().any(|other| other == bare),
                    "no-doubles violated in three-way test: {tools:?}");
            }
        }
    }

    // ── WR-02: reject already-prefixed tool names ─────────────────────────

    /// A tool whose `name()` already starts with `mcp__alzina__` must be
    /// rejected (logged + skipped, not re-prefixed). Without this guard the
    /// algorithm would emit `mcp__alzina__mcp__alzina__bogus`. The good
    /// tool registered alongside is still emitted normally.
    #[test]
    fn build_allowed_tools_rejects_already_prefixed_tool_name() {
        let ctx = CurationContext {
            available_tools: vec![
                Box::new(AlreadyPrefixedTool),  // must be rejected
                Box::new(StubFooTool),          // must be kept
            ],
            ..CurationContext::default()
        };

        let tools = build_allowed_tools(&ctx);

        // The already-prefixed name must not appear in any form.
        assert!(
            !tools.iter().any(|t| t == "mcp__alzina__bogus"),
            "already-prefixed tool name must not be emitted; got: {tools:?}"
        );
        assert!(
            !tools.iter().any(|t| t == "mcp__alzina__mcp__alzina__bogus"),
            "already-prefixed tool must not be double-prefixed; got: {tools:?}"
        );

        // The well-formed tool registered alongside the bad one is still emitted.
        assert!(
            tools.contains(&format!("{MCP_ALZINA_PREFIX}foo")),
            "well-formed tool 'foo' must still be emitted alongside a rejected one; \
             got: {tools:?}"
        );
    }

    // ── Test 5: deterministic output order (WR-03) ────────────────────────

    /// `build_allowed_tools` must return entries in a deterministic
    /// (lexically sorted) order so:
    ///
    /// - audit-log diffs do not show spurious tool-list churn,
    /// - the model-side KV cache key (which folds in the allowed_tools
    ///   prefix) is stable across sessions,
    /// - the same `CurationContext` produces the same wire payload.
    #[test]
    fn build_allowed_tools_output_is_lexically_sorted() {
        // Construct a context whose tools intentionally produce names
        // that would land in HashMap buckets in an order that drifts
        // across runs. Three names: bare "Read", "mcp__alzina__foo",
        // "mcp__alzina__Grep" (GovernanceShim override).
        let mut overrides = HashMap::new();
        overrides.insert("Grep".to_string(), CurationPolicy::GovernanceShim);

        let ctx = CurationContext {
            available_tools: vec![
                Box::new(StubReadTool),
                Box::new(StubFooTool),
                Box::new(StubGrepTool),
            ],
            policy_overrides: overrides,
            ..CurationContext::default()
        };

        let tools = build_allowed_tools(&ctx);

        let mut expected = tools.clone();
        expected.sort();
        assert_eq!(
            tools, expected,
            "build_allowed_tools must return a lexically sorted Vec; got: {tools:?}"
        );

        // Belt-and-braces: re-running on the same context must produce
        // byte-for-byte identical output, run after run.
        for _ in 0..5 {
            assert_eq!(build_allowed_tools(&ctx), tools);
        }
    }

    // ── Test 6: region-scoped access (D13-05) ─────────────────────────────

    /// `CurationContext.region` filters tools by `Tool::scope()`.
    ///
    /// - When `region == Some("auth")`, `RegionScopedTool` (scope = Region("auth"))
    ///   and `StubFooTool` (scope = All) are emitted; `StubReadTool` (scope = All)
    ///   is also emitted.
    /// - When `region == Some("billing")`, `RegionScopedTool` is filtered out
    ///   because its region tag ("auth") does not match.
    ///
    /// This test is UNCONDITIONAL — `ToolScope::Region(String)` was committed by
    /// Plan 13-02 Task 1 and is always exercisable.
    #[test]
    fn build_allowed_tools_region_scoped_access() {
        // Sub-test A: region == "auth" → RegionScopedTool emitted.
        {
            let ctx = CurationContext {
                region: Some("auth".to_string()),
                available_tools: vec![
                    Box::new(RegionScopedTool), // scope = Region("auth")
                    Box::new(StubFooTool),      // scope = All (default)
                ],
                ..CurationContext::default()
            };

            let tools = build_allowed_tools(&ctx);
            let prefixed_auth = format!("{MCP_ALZINA_PREFIX}auth_checker");
            let prefixed_foo = format!("{MCP_ALZINA_PREFIX}foo");

            assert!(
                tools.contains(&prefixed_auth),
                "auth_checker must be emitted when region = 'auth'; got: {tools:?}"
            );
            assert!(
                tools.contains(&prefixed_foo),
                "foo (scope = All) must be emitted for any region; got: {tools:?}"
            );
        }

        // Sub-test B: region == "billing" → RegionScopedTool filtered out.
        {
            let ctx = CurationContext {
                region: Some("billing".to_string()),
                available_tools: vec![
                    Box::new(RegionScopedTool), // scope = Region("auth") — mismatch
                    Box::new(StubFooTool),      // scope = All — passes
                ],
                ..CurationContext::default()
            };

            let tools = build_allowed_tools(&ctx);
            let prefixed_auth = format!("{MCP_ALZINA_PREFIX}auth_checker");
            let prefixed_foo = format!("{MCP_ALZINA_PREFIX}foo");

            assert!(
                !tools.contains(&prefixed_auth),
                "auth_checker must NOT be emitted when region = 'billing'; got: {tools:?}"
            );
            assert!(
                tools.contains(&prefixed_foo),
                "foo (scope = All) must still be emitted for region = 'billing'; got: {tools:?}"
            );
        }
    }
}
