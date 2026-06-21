//! JSON-over-stdio protocol types for communication between
//! `ClaudeAgentSdkExecutor` and the TypeScript sidecar process.
//!
//! All messages are newline-delimited JSON. One JSON object per line.
//!
//! - **stdin** (Rust → Sidecar): `SidecarRequest`, `HookDecision`, `ShutdownRequest`
//! - **stdout** (Sidecar → Rust): `SidecarEvent`
//!
//! See `sidecar/claude-agent/src/protocol.ts` for the TypeScript counterpart.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── Outbound (Rust → Sidecar stdin) ─────────────────────────────────────────

/// Request to execute an agent query via the Claude Agent SDK.
#[derive(Debug, Clone, Serialize)]
pub struct SidecarRequest {
    /// Discriminator — always `"execute"`.
    #[serde(rename = "type")]
    pub msg_type: &'static str,
    /// Unique request ID for correlating responses.
    pub id: String,
    /// Combined instruction + task prompt for the agent.
    pub prompt: String,
    /// Agent SDK options.
    pub options: SidecarOptions,
}

impl SidecarRequest {
    /// Construct a new execute request.
    pub fn new(id: String, prompt: String, options: SidecarOptions) -> Self {
        Self {
            msg_type: "execute",
            id,
            prompt,
            options,
        }
    }
}

/// Options passed to the Agent SDK's `query()` function.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarOptions {
    /// Tools the agent is allowed to use.
    pub allowed_tools: Vec<String>,
    /// Tools stripped from the model's context entirely (SDK
    /// `disallowedTools`). `allowed_tools` only auto-approves — it does
    /// NOT hide tools — so text-generation-only spawns (TTD trajectories)
    /// list the SDK built-ins here; otherwise the model attempts them and
    /// burns one governance-blocked turn per call. Empty → field omitted
    /// from the wire (byte-identical to pre-change requests).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub disallowed_tools: Vec<String>,
    /// SDK permission mode (e.g. `"acceptEdits"`).
    pub permission_mode: String,
    /// Model to use (e.g. `"claude-sonnet-4-20250514"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Working directory for file operations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_directory: Option<String>,
    /// System prompt / instruction (if separate from prompt).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Additional directories the agent can access beyond the working directory.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<String>,
    /// Custom tool definitions to register with the SDK for this one-shot
    /// execution (plan 260515-ndk). When `None` (or absent on the wire),
    /// the sidecar's one-shot `execute` path runs with no custom tools —
    /// byte-identical to pre-260515-ndk behaviour. When `Some`, the
    /// sidecar builds an in-process MCP server exposing these tools and
    /// adds `mcp__alzina__<name>` to the SDK's `allowedTools` for each.
    ///
    /// The Rust-side `ClaudeAgentSdkExecutor` always injects
    /// `vec![return_envelope_tool()]` here for the sub-agent dispatch
    /// path so every dispatched agent gets the typed envelope-return
    /// surface unconditionally. The lenient prose fallback handles
    /// agents that ignore the tool, so there is zero downside to
    /// universal availability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_tools: Option<Vec<CustomToolDefinition>>,
    /// Per-trajectory temperature for LLM sampling diversity (EXT-01 Phase 24).
    /// None → field omitted from JSON (skip_serializing_if); sidecar uses SDK default.
    /// Serialised as `temperature` (camelCase passthrough — same name in JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Per-trajectory top_p for LLM sampling diversity (EXT-01 Phase 24).
    /// None → field omitted from JSON. Serialised as `topP` (camelCase via rename_all).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Per-trajectory top_k for LLM sampling diversity (EXT-01 Phase 24).
    /// None → field omitted from JSON. Serialised as `topK` (camelCase via rename_all).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

/// Governance hook decision sent to sidecar after a `ToolUse` event.
#[derive(Debug, Clone, Serialize)]
pub struct HookDecisionMsg {
    /// Discriminator — always `"hook_decision"`.
    #[serde(rename = "type")]
    pub msg_type: &'static str,
    /// ID matching the `hookId` in the corresponding `ToolUse` event.
    #[serde(rename = "hookId")]
    pub hook_id: String,
    /// Whether to allow or block the tool call.
    pub decision: HookVerdict,
    /// Reason for blocking (included when decision is `Block`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl HookDecisionMsg {
    /// Create an "allow" decision.
    pub fn allow(hook_id: String) -> Self {
        Self {
            msg_type: "hook_decision",
            hook_id,
            decision: HookVerdict::Allow,
            reason: None,
        }
    }

    /// Create a "block" decision with reason.
    pub fn block(hook_id: String, reason: String) -> Self {
        Self {
            msg_type: "hook_decision",
            hook_id,
            decision: HookVerdict::Block,
            reason: Some(reason),
        }
    }
}

/// Hook verdict — allow or block.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HookVerdict {
    Allow,
    Block,
}

/// Shutdown request — cleanly terminate the sidecar.
#[derive(Debug, Clone, Serialize)]
pub struct ShutdownMsg {
    /// Discriminator — always `"shutdown"`.
    #[serde(rename = "type")]
    pub msg_type: &'static str,
}

impl Default for ShutdownMsg {
    fn default() -> Self {
        Self {
            msg_type: "shutdown",
        }
    }
}

// ── Inbound (Sidecar stdout → Rust) ─────────────────────────────────────────

/// Any event emitted by the sidecar on stdout.
///
/// Discriminated on the `type` field.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SidecarEvent {
    /// Agent wants to use a tool — governance decision required.
    ToolUse {
        /// Correlates to the originating `SidecarRequest`.
        id: String,
        /// Tool name (e.g. `"Read"`, `"Edit"`, `"Bash"`).
        tool: String,
        /// Tool input arguments.
        input: HashMap<String, serde_json::Value>,
        /// Unique hook ID — respond with `HookDecisionMsg` matching this.
        /// Sidecar sends as `hookId` (JS convention); we accept both.
        #[serde(alias = "hookId")]
        hook_id: String,
    },
    /// A tool call completed.
    ToolResult {
        id: String,
        tool: String,
        /// Abbreviated tool output (for audit/progress).
        output: String,
    },
    /// Partial/streaming text output from the agent.
    Text { id: String, content: String },
    /// Agent completed — final output.
    Result {
        id: String,
        /// Complete agent output text.
        content: String,
    },
    /// Error during execution.
    Error {
        id: String,
        error: String,
        retryable: bool,
    },
    /// Session successfully initialised (persistent mode).
    SessionReady { session_id: String },
    /// Agent response to a chat_message turn (persistent mode).
    ChatResponse {
        /// Matches the chat_message turn ID.
        id: String,
        /// Final text content for this turn.
        content: String,
    },
    /// Token-usage report from the sidecar at end of a turn (Phase 4).
    ///
    /// Mirrors the TS-side `UsageEvent`. Numbers are cumulative across the
    /// turn; the daemon-side chat service maintains running totals across
    /// turns keyed off the chat session ID.
    Usage {
        /// Matches the originating request / turn ID.
        id: String,
        input_tokens: u32,
        output_tokens: u32,
        cache_read_input_tokens: u32,
        cache_creation_input_tokens: u32,
        model: String,
    },
}

impl SidecarEvent {
    /// Get the request ID this event belongs to.
    ///
    /// For `SessionReady`, returns the session ID (no request correlation).
    pub fn request_id(&self) -> &str {
        match self {
            Self::ToolUse { id, .. }
            | Self::ToolResult { id, .. }
            | Self::Text { id, .. }
            | Self::Result { id, .. }
            | Self::Error { id, .. }
            | Self::ChatResponse { id, .. }
            | Self::Usage { id, .. } => id,
            Self::SessionReady { session_id } => session_id,
        }
    }
}

// ── Persistent-mode Outbound (Rust → Sidecar stdin) ────────────────────────

/// Custom tool definition — routed via HTTP to the daemon.
///
/// Registered with the sidecar at session start so the SDK exposes them
/// to the model. Tool invocations are intercepted by the sidecar and
/// forwarded to the daemon endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolDefinition {
    /// Tool name as the SDK sees it (e.g. `"dispatch_agent"`).
    pub name: String,
    /// Human-readable description for the model.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: Value,
    /// HTTP endpoint to POST tool input to.
    pub endpoint: String,
    /// HTTP method for the endpoint. Defaults to `POST` when absent.
    /// Use `"GET"` for read-only tools that hit `axum::routing::get`
    /// handlers (e.g. `list_weaves`) — without this override the sidecar
    /// would POST to the same path and the model would see a confusing
    /// 422/parse error from a different handler bound to the route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Optional timeout in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// P5-LIVENESS: when `true`, the sidecar fetches the endpoint with
    /// `Accept: application/x-ndjson` and treats the response as a
    /// progress-event stream. Idle silence (no event for `idle_timeout_ms`)
    /// triggers an abort; ongoing event traffic resets the timer.
    /// `false`/absent keeps the existing single-shot HTTP behaviour for
    /// memory_search, weave-lifecycle, etc.
    #[serde(default, skip_serializing_if = "is_false")]
    pub stream_progress: bool,
    /// P5-LIVENESS: idle timeout the sidecar applies when
    /// `stream_progress` is `true`. Reset on every chunk read from the
    /// NDJSON body. `None` keeps the sidecar's default (60 s).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_timeout_ms: Option<u64>,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

/// Start a persistent session (Rust -> Sidecar).
///
/// Sent once after spawning the sidecar process. The sidecar responds
/// with a `SessionReady` event when initialisation is complete.
#[derive(Debug, Clone, Serialize)]
pub struct SessionStartMsg {
    /// Discriminator — always `"session_start"`.
    #[serde(rename = "type")]
    pub msg_type: &'static str,
    /// Unique session ID.
    pub session_id: String,
    /// System prompt / agent instruction.
    pub system_prompt: String,
    /// Custom tool definitions (dispatch_agent, etc).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub custom_tools: Vec<CustomToolDefinition>,
    /// Daemon API key, plumbed to the sidecar so it can attach
    /// `Authorization: Bearer <key>` when invoking custom tools (red-team A7).
    /// `None` in dev-mode deployments.
    #[serde(rename = "apiKey", skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Model id forwarded into every chat-turn `query()` call. `None` lets
    /// the SDK pick its own default. Set by the daemon to keep chat
    /// agents on the same model tier as their dispatched sub-agents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl SessionStartMsg {
    /// Construct a new session_start message.
    pub fn new(
        session_id: String,
        system_prompt: String,
        custom_tools: Vec<CustomToolDefinition>,
        api_key: Option<String>,
        model: Option<String>,
    ) -> Self {
        Self {
            msg_type: "session_start",
            session_id,
            system_prompt,
            custom_tools,
            api_key,
            model,
        }
    }
}

/// Send a user message into the active session (Rust -> Sidecar).
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessageMsg {
    /// Discriminator — always `"chat_message"`.
    #[serde(rename = "type")]
    pub msg_type: &'static str,
    /// Turn ID for correlating the response.
    pub id: String,
    /// User message content.
    pub content: String,
}

impl ChatMessageMsg {
    /// Construct a new chat_message.
    pub fn new(id: String, content: String) -> Self {
        Self {
            msg_type: "chat_message",
            id,
            content,
        }
    }
}

/// End the persistent session (Rust -> Sidecar).
#[derive(Debug, Clone, Serialize)]
pub struct SessionEndMsg {
    /// Discriminator — always `"session_end"`.
    #[serde(rename = "type")]
    pub msg_type: &'static str,
}

impl Default for SessionEndMsg {
    fn default() -> Self {
        Self {
            msg_type: "session_end",
        }
    }
}

/// Cancel the in-flight chat turn without ending the session (Rust -> Sidecar).
///
/// Signals the TypeScript sidecar to abort the active query() call so it
/// returns to SESSION_ACTIVE and is ready for the next chat_message.
/// Idempotent: safe to send when no turn is active.
#[derive(Debug, Clone, Serialize)]
pub struct TurnCancelMsg {
    /// Discriminator — always `"turn_cancel"`.
    #[serde(rename = "type")]
    pub msg_type: &'static str,
}

impl Default for TurnCancelMsg {
    fn default() -> Self {
        Self {
            msg_type: "turn_cancel",
        }
    }
}

// ── Persistent-mode Inbound (Sidecar stdout → Rust) ────────────────────────

/// Events specific to persistent-mode sessions.
///
/// These are emitted alongside the standard `SidecarEvent` variants.
/// The `SidecarEvent` enum is extended with these persistent-mode variants.
///
/// Note: `SessionReady` and `ChatResponse` are added directly to `SidecarEvent`.

// (SessionReady and ChatResponse are added to SidecarEvent below.)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_execute_request() {
        let req = SidecarRequest::new(
            "req-001".to_string(),
            "Analyse this code".to_string(),
            SidecarOptions {
                allowed_tools: vec!["Read".into(), "Glob".into()],
                disallowed_tools: Vec::new(),
                permission_mode: "acceptEdits".into(),
                model: Some("claude-sonnet-4-20250514".into()),
                working_directory: Some("/tmp/workspace".into()),
                system_prompt: None,
                additional_directories: Vec::new(),
                custom_tools: None,
                temperature: None,
                top_p: None,
                top_k: None,
            },
        );
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"type\":\"execute\""));
        assert!(json.contains("\"id\":\"req-001\""));
        assert!(json.contains("\"allowedTools\""));
    }

    // ── 260515-ndk: custom_tools wire field on SidecarOptions ──────────

    /// Test 2: SidecarOptions round-trips a non-empty custom_tools list
    /// through serde without data loss. We round-trip via `Value` because
    /// `SidecarOptions` derives `Serialize` only (no `Deserialize` — it's
    /// outbound-only wire), so we check by re-extracting the field.
    #[test]
    fn sidecar_options_custom_tools_serialises_when_some() {
        let opts = SidecarOptions {
            allowed_tools: vec!["Read".into()],
            disallowed_tools: Vec::new(),
            permission_mode: "acceptEdits".into(),
            model: None,
            working_directory: None,
            system_prompt: None,
            additional_directories: Vec::new(),
            custom_tools: Some(vec![CustomToolDefinition {
                name: "return_envelope".into(),
                description: "submit envelope".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "required": ["status"],
                    "properties": {"status": {"type": "string"}}
                }),
                endpoint: "http://127.0.0.1:0/internal/return_envelope".into(),
                method: None,
                timeout_ms: None,
                stream_progress: false,
                idle_timeout_ms: None,
            }]),
            temperature: None,
            top_p: None,
            top_k: None,
        };
        let v: serde_json::Value = serde_json::to_value(&opts).unwrap();
        // camelCase rename applies — `custom_tools` becomes `customTools`.
        let tools = v
            .get("customTools")
            .expect("customTools field must serialise when Some");
        let arr = tools.as_array().expect("customTools must be an array");
        assert_eq!(arr.len(), 1);
        assert_eq!(
            arr[0].get("name").and_then(|x| x.as_str()),
            Some("return_envelope")
        );
    }

    /// Test 3: SidecarOptions omits the field entirely when `None` so
    /// existing wire consumers stay byte-compatible (no spurious
    /// `customTools: null`).
    #[test]
    fn sidecar_options_custom_tools_absent_when_none() {
        let opts = SidecarOptions {
            allowed_tools: vec!["Read".into()],
            disallowed_tools: Vec::new(),
            permission_mode: "acceptEdits".into(),
            model: None,
            working_directory: None,
            system_prompt: None,
            additional_directories: Vec::new(),
            custom_tools: None,
            temperature: None,
            top_p: None,
            top_k: None,
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(
            !json.contains("customTools"),
            "customTools must be absent on the wire when None, got: {json}"
        );
    }

    #[test]
    fn serialize_hook_allow() {
        let msg = HookDecisionMsg::allow("hook-1".into());
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"decision\":\"allow\""));
        assert!(!json.contains("\"reason\""));
    }

    #[test]
    fn serialize_hook_block() {
        let msg = HookDecisionMsg::block("hook-2".into(), "write to governed path".into());
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"decision\":\"block\""));
        assert!(json.contains("write to governed path"));
    }

    #[test]
    fn deserialize_tool_use_event() {
        let json = r#"{"type":"tool_use","id":"req-001","tool":"Read","input":{"path":"src/main.rs"},"hook_id":"hook-1"}"#;
        let event: SidecarEvent = serde_json::from_str(json).unwrap();
        match event {
            SidecarEvent::ToolUse {
                id,
                tool,
                hook_id,
                input,
            } => {
                assert_eq!(id, "req-001");
                assert_eq!(tool, "Read");
                assert_eq!(hook_id, "hook-1");
                assert_eq!(
                    input.get("path").and_then(|v| v.as_str()),
                    Some("src/main.rs")
                );
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn deserialize_result_event() {
        let json = r#"{"type":"result","id":"req-001","content":"Analysis complete."}"#;
        let event: SidecarEvent = serde_json::from_str(json).unwrap();
        match event {
            SidecarEvent::Result { id, content } => {
                assert_eq!(id, "req-001");
                assert_eq!(content, "Analysis complete.");
            }
            _ => panic!("expected Result"),
        }
    }

    #[test]
    fn deserialize_error_event() {
        let json = r#"{"type":"error","id":"req-001","error":"rate limited","retryable":true}"#;
        let event: SidecarEvent = serde_json::from_str(json).unwrap();
        match event {
            SidecarEvent::Error {
                id,
                error,
                retryable,
            } => {
                assert_eq!(id, "req-001");
                assert_eq!(error, "rate limited");
                assert!(retryable);
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn event_request_id_accessor() {
        let json = r#"{"type":"text","id":"req-042","content":"partial..."}"#;
        let event: SidecarEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.request_id(), "req-042");
    }

    #[test]
    fn serialize_shutdown() {
        let msg = ShutdownMsg::default();
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"shutdown"}"#);
    }

    // ── Persistent-mode protocol tests ─────────────────────────────

    #[test]
    fn serialize_session_start() {
        let msg = SessionStartMsg::new(
            "sess-001".into(),
            "You are vefr.".into(),
            vec![CustomToolDefinition {
                name: "dispatch_agent".into(),
                description: "Dispatch a sub-agent".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "agent_id": { "type": "string" }
                    }
                }),
                endpoint: "http://localhost:3000/api/v1/dispatch".into(),
                method: None,
                timeout_ms: Some(30000),
                stream_progress: false,
                idle_timeout_ms: None,
            }],
            None,
            None,
        );
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"session_start\""));
        assert!(json.contains("\"session_id\":\"sess-001\""));
        assert!(json.contains("\"system_prompt\":\"You are vefr.\""));
        assert!(json.contains("dispatch_agent"));
    }

    #[test]
    fn serialize_session_start_no_tools() {
        let msg = SessionStartMsg::new("sess-002".into(), "prompt".into(), vec![], None, None);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("custom_tools"));
    }

    #[test]
    fn serialize_session_start_with_api_key_includes_apikey_field() {
        let msg = SessionStartMsg::new(
            "sess-auth".into(),
            "prompt".into(),
            vec![],
            Some("alz_sk_test".into()),
            None,
        );
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"apiKey\":\"alz_sk_test\""), "got: {json}");
    }

    #[test]
    fn serialize_session_start_without_api_key_omits_field() {
        let msg = SessionStartMsg::new("sess-noauth".into(), "prompt".into(), vec![], None, None);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("apiKey"), "got: {json}");
    }

    #[test]
    fn serialize_session_start_with_model_includes_model_field() {
        let msg = SessionStartMsg::new(
            "sess-model".into(),
            "prompt".into(),
            vec![],
            None,
            Some("claude-sonnet-4-20250514".into()),
        );
        let json = serde_json::to_string(&msg).unwrap();
        assert!(
            json.contains("\"model\":\"claude-sonnet-4-20250514\""),
            "got: {json}"
        );
    }

    #[test]
    fn serialize_session_start_without_model_omits_field() {
        let msg = SessionStartMsg::new("sess-no-model".into(), "prompt".into(), vec![], None, None);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("\"model\""), "got: {json}");
    }

    #[test]
    fn serialize_chat_message() {
        let msg = ChatMessageMsg::new("turn-001".into(), "How should we refactor auth?".into());
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"chat_message\""));
        assert!(json.contains("\"id\":\"turn-001\""));
        assert!(json.contains("refactor auth"));
    }

    #[test]
    fn serialize_session_end() {
        let msg = SessionEndMsg::default();
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"session_end"}"#);
    }

    #[test]
    fn deserialize_session_ready_event() {
        let json = r#"{"type":"session_ready","session_id":"sess-001"}"#;
        let event: SidecarEvent = serde_json::from_str(json).unwrap();
        match event {
            SidecarEvent::SessionReady { session_id } => {
                assert_eq!(session_id, "sess-001");
            }
            _ => panic!("expected SessionReady"),
        }
    }

    #[test]
    fn deserialize_chat_response_event() {
        let json = r#"{"type":"chat_response","id":"turn-001","content":"Here is my analysis..."}"#;
        let event: SidecarEvent = serde_json::from_str(json).unwrap();
        match event {
            SidecarEvent::ChatResponse { id, content } => {
                assert_eq!(id, "turn-001");
                assert_eq!(content, "Here is my analysis...");
            }
            _ => panic!("expected ChatResponse"),
        }
    }

    #[test]
    fn session_ready_request_id() {
        let json = r#"{"type":"session_ready","session_id":"sess-042"}"#;
        let event: SidecarEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.request_id(), "sess-042");
    }

    #[test]
    fn chat_response_request_id() {
        let json = r#"{"type":"chat_response","id":"turn-007","content":"done"}"#;
        let event: SidecarEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.request_id(), "turn-007");
    }
}
