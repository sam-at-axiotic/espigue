//! `ClaudeAgentSdkExecutor` — `AgentExecutor` implementation that delegates
//! LLM execution to the Claude Agent SDK via a TypeScript sidecar process.
//!
//! The sidecar (`sidecar/claude-agent/`) wraps `@anthropic-ai/claude-agent-sdk`
//! and communicates over newline-delimited JSON on stdin/stdout. This executor
//! spawns the sidecar, sends an execute request, processes events (including
//! governance hook decisions), and returns the final agent output.
//!
//! # Auth
//!
//! The sidecar inherits `ANTHROPIC_API_KEY` from the environment. This works
//! with both OAuth tokens (Claude Max/Pro) and Console API keys.
//!
//! # Governance
//!
//! Tool calls are intercepted via the sidecar's `PreToolUse` hook. When the
//! SDK wants to use a tool, the sidecar emits a `ToolUse` event. This executor
//! runs `tool_interceptor::check_tool_call()` and sends back an allow/block
//! decision. Alzina retains full visibility and control over every tool call.
//!
//! See `docs/proposals/001-claude-agent-sdk-integration.md` for the full design.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tracing::{debug, error, info, warn};

use alzina_core::identity::{AgentId, SessionId};
use alzina_core::{AlzinaError, AlzinaEvent, AlzinaResult, Envelope};

use alzina_governance::GovernanceLayer;

use super::alzina_runner::{AgentExecutor, ExecutorEventEmitter, SamplingParams};
use super::assigned_dirs::{AssignedDirGuard, AssignedDirRegistry};
use super::envelope_tool::return_envelope_tool;
use super::sidecar_protocol::{
    HookDecisionMsg, ShutdownMsg, SidecarEvent, SidecarOptions, SidecarRequest,
};
use super::tool_interceptor::{self, ToolArgs, ToolDecision};

/// MCP prefix the sidecar uses for daemon-registered custom tools (mirror
/// of `ALZINA_MCP_SERVER_NAME` in sidecar/claude-agent/src/customTools.ts).
/// 260515-ndk: stripped from `SidecarEvent::ToolUse.tool` so the bare
/// `return_envelope` name match catches the typed tool regardless of how
/// the SDK surfaces it (RESEARCH P7).
pub const MCP_ALZINA_PREFIX: &str = "mcp__alzina__";

/// The full prefixed tool name the model sees for the typed envelope-return
/// tool — `mcp__alzina__return_envelope`. Added to `allowed_tools` on the
/// outbound `SidecarOptions` so the SDK permits the model to call it.
const RETURN_ENVELOPE_TOOL_NAME: &str = "mcp__alzina__return_envelope";

// ── ClaudeAgentSdkExecutor ──────────────────────────────────────────────────

/// Executes Alzina agents via the Claude Agent SDK TypeScript sidecar.
///
/// Each `execute()` call spawns a sidecar process, sends the request,
/// streams events, enforces governance, and returns the final output.
pub struct ClaudeAgentSdkExecutor {
    /// Path to the sidecar entry point (e.g. `sidecar/claude-agent/dist/index.js`).
    sidecar_path: PathBuf,
    /// Governance layer for tool-call interception.
    governance: Arc<GovernanceLayer>,
    /// Working directory for the agent's file operations.
    working_dir: PathBuf,
    /// Tools the agent is allowed to use.
    default_tools: Vec<String>,
    /// Tools stripped from the model's context entirely (SDK
    /// `disallowedTools`). Empty for the general dispatch path; the TTD
    /// executor lists the SDK built-ins so trajectory agents stay pure
    /// text generation.
    disallowed_tools: Vec<String>,
    /// Whether to inject the typed `return_envelope` custom tool
    /// (260515-ndk). True for the dispatch path. False for
    /// text-generation-only spawns (TTD): with every other tool
    /// stripped, models wrap their answer in the one visible tool
    /// instead of emitting the prose the TTD parsers expect (live
    /// probe 2026-06-10: "no <synthesis> block" engine failure).
    inject_envelope_tool: bool,
    /// SDK permission mode (e.g. `"acceptEdits"`).
    permission_mode: String,
    /// Additional directories agents can access beyond the working directory.
    additional_directories: Vec<String>,
    /// Per-dispatch writable-dir registry. The dispatch path registers
    /// `(spawn_session_id, artifact_dir)` immediately before invoking the
    /// runner; `run_event_loop` looks the dir up on entry and threads it
    /// into every `check_tool_call` so the interceptor can enforce
    /// "writes only inside the assigned dir, only for new files".
    assigned_dirs: AssignedDirRegistry,
}

impl ClaudeAgentSdkExecutor {
    /// Construct a new executor.
    ///
    /// # Arguments
    ///
    /// * `sidecar_path` — Path to the sidecar JS entry point
    /// * `governance` — Governance layer for tool-call interception
    /// * `working_dir` — Working directory for agent file operations
    /// * `default_tools` — SDK tools the agent may use
    /// * `permission_mode` — SDK permission mode (`"default"` or `"acceptEdits"`)
    pub fn new(
        sidecar_path: PathBuf,
        governance: Arc<GovernanceLayer>,
        working_dir: PathBuf,
        default_tools: Vec<String>,
        permission_mode: String,
    ) -> Self {
        Self {
            sidecar_path,
            governance,
            working_dir,
            default_tools,
            disallowed_tools: Vec::new(),
            inject_envelope_tool: true,
            permission_mode,
            additional_directories: Vec::new(),
            assigned_dirs: AssignedDirRegistry::new(),
        }
    }

    /// Disable the `return_envelope` custom-tool injection for
    /// text-generation-only spawns. The TTD engine consumes raw text
    /// (`execute_with_sampling`) and never reads the envelope; leaving
    /// the tool as the model's ONLY visible tool makes it wrap its
    /// answer in the envelope instead of emitting parseable prose.
    pub fn without_envelope_tool(mut self) -> Self {
        self.inject_envelope_tool = false;
        self
    }

    /// Strip tools from the model's context entirely (SDK `disallowedTools`).
    /// `default_tools`/`allowedTools` only auto-approves — it does not hide
    /// tools — so text-generation-only spawn paths must disallow built-ins
    /// here or the model attempts them and burns a governance-blocked turn
    /// per call.
    pub fn with_disallowed_tools(mut self, tools: Vec<String>) -> Self {
        self.disallowed_tools = tools;
        self
    }

    /// Set additional directories agents can access.
    pub fn with_additional_directories(mut self, dirs: Vec<String>) -> Self {
        self.additional_directories = dirs;
        self
    }

    /// Inject a shared `AssignedDirRegistry`. The daemon's builder creates
    /// one registry and threads clones to both this executor and to its
    /// `AppState` so dispatch handlers can register per-spawn writable dirs
    /// that the interceptor will see.
    pub fn with_assigned_dirs(mut self, registry: AssignedDirRegistry) -> Self {
        self.assigned_dirs = registry;
        self
    }

    /// Clone-able handle to the registry. Dispatch handlers use this to
    /// register a leaf's `(SessionId, artifact_dir)` before invoking spawn.
    pub fn assigned_dirs(&self) -> AssignedDirRegistry {
        self.assigned_dirs.clone()
    }
}

#[async_trait]
impl AgentExecutor for ClaudeAgentSdkExecutor {
    async fn execute(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
    ) -> AlzinaResult<String> {
        // Single source of truth: delegate to the emitter-aware variant
        // with no emitter wired. The runner uses a synthetic SessionId
        // here since this entry point doesn't know the spawn's id;
        // streaming events would have nowhere to attribute themselves.
        self.execute_with_emitter(agent_id, instruction, model, task, &SessionId::new(), None)
            .await
    }

    async fn execute_with_emitter(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
        session_id: &SessionId,
        emitter: Option<ExecutorEventEmitter>,
    ) -> AlzinaResult<String> {
        // 260515-ndk Task 3: route through the shared event-loop helper
        // and discard the captured envelope — legacy contract preserves
        // the String-only return shape.
        let (raw, _envelope) = self
            .run_event_loop(agent_id, instruction, model, task, session_id, emitter)
            .await?;
        Ok(raw)
    }

    /// EXT-01 Phase 24: override to thread per-trajectory sampling params into
    /// SidecarOptions (temperature/top_p/top_k). None → falls through to
    /// standard execute() via the default trait impl (backwards compatible).
    async fn execute_with_sampling(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
        sampling: Option<SamplingParams>,
    ) -> AlzinaResult<String> {
        let (raw, _envelope) = self
            .run_event_loop_with_sampling(
                agent_id, instruction, model, task, &SessionId::new(), None, sampling,
            )
            .await?;
        Ok(raw)
    }

    /// 260515-ndk Task 3: override the default trait impl so the typed
    /// envelope captured via the `mcp__alzina__return_envelope` tool_use
    /// block flows back to the runner. The runner's parse step uses the
    /// typed envelope directly (skipping prose parse) when this returns
    /// `Some(env)`; when `None`, the strict-then-lenient prose parser
    /// runs unchanged.
    async fn execute_with_envelope(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
        session_id: &SessionId,
        emitter: Option<ExecutorEventEmitter>,
    ) -> AlzinaResult<(String, Option<Envelope>)> {
        self.run_event_loop(agent_id, instruction, model, task, session_id, emitter)
            .await
    }
}

impl ClaudeAgentSdkExecutor {
    /// 260515-ndk Task 3: shared event-loop helper used by both
    /// `execute_with_emitter` (which discards the envelope) and
    /// `execute_with_envelope` (which preserves it).
    ///
    /// Returns `(raw_final_text, optional_captured_envelope)`. The
    /// envelope is `Some` when the model invoked the
    /// `mcp__alzina__return_envelope` tool with a deserialisable payload;
    /// `None` otherwise (lenient handoff to the runner's prose parser).
    ///
    /// Last-wins semantics (P1): if the model calls the tool more than
    /// once, the final invocation's payload is the one returned.
    ///
    /// EOF-safe (P8): when the sidecar exits without emitting a terminal
    /// `Result` event, any envelope captured before EOF is still returned.
    async fn run_event_loop(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
        session_id: &SessionId,
        emitter: Option<ExecutorEventEmitter>,
    ) -> AlzinaResult<(String, Option<Envelope>)> {
        self.run_event_loop_with_sampling(
            agent_id, instruction, model, task, session_id, emitter, None,
        ).await
    }

    /// Shared event-loop helper that accepts optional per-trajectory sampling params
    /// (EXT-01 Phase 24). Used by `execute_with_sampling` override.
    async fn run_event_loop_with_sampling(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
        session_id: &SessionId,
        emitter: Option<ExecutorEventEmitter>,
        sampling: Option<SamplingParams>,
    ) -> AlzinaResult<(String, Option<Envelope>)> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let spawn_start = std::time::Instant::now();

        info!(
            agent = %agent_id,
            request_id = %request_id,
            model = %model,
            disallowed = self.disallowed_tools.len(),
            "spawning Claude Agent SDK sidecar"
        );

        // ── Spawn sidecar ───────────────────────────────────────────────

        let mut child = Command::new("node")
            .arg(&self.sidecar_path)
            // Headless one-shot session: suppress telemetry/autoupdate
            // (DISABLE_NONESSENTIAL_TRAFFIC) and the subagent warmup
            // (REMOTE) — cli.js p$9 fires Task("Warmup") explorers per
            // session whose denied tool calls churn through our
            // governance hook; CLAUDE_CODE_REMOTE="true" is its only
            // gate. Side effect: plan-mode tools disabled (unused here).
            .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            .env("CLAUDE_CODE_REMOTE", "true")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true) // Safety: kill orphaned sidecar on executor drop
            .spawn()
            .map_err(|e| {
                AlzinaError::Orchestration(format!(
                    "failed to spawn sidecar at {}: {e}",
                    self.sidecar_path.display()
                ))
            })?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| AlzinaError::Orchestration("sidecar stdin not available".into()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AlzinaError::Orchestration("sidecar stdout not available".into()))?;

        // Drain sidecar stderr in a background task. Pre-fix the pipe was
        // set but never `.take()`n, so any subprocess stderr stayed in the
        // OS pipe buffer and was thrown away when the child exited —
        // which is exactly when the diagnostic mattered most. Mirrors
        // `sidecar_handle::start_session::stderr_task`: lines prefixed
        // `[claude-code-stderr]` (set by the sidecar's stderr capture
        // hook on the SDK query() call) elevate to warn-level tracing so
        // they survive default log filtering; everything else stays
        // debug-only.
        if let Some(child_stderr) = child.stderr.take() {
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(child_stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if line.starts_with("[claude-code-stderr]") {
                        warn!(target: "sidecar_stderr", "{}", line);
                    } else {
                        debug!(target: "sidecar_stderr", "{}", line);
                    }
                }
            });
        }

        // ── Send execute request ────────────────────────────────────────

        // 260515-ndk Task 3: unconditionally inject the return_envelope
        // custom tool and extend allowed_tools with its prefixed name. This
        // executor is reached only via the sub-agent dispatch path; the
        // lenient prose fallback covers any agent that ignores the tool, so
        // there is zero cost to making the typed envelope-return surface
        // universally available across every envelope=true agent (gna,
        // galdr, huginn, kvasir, sjofn, muninn, skuld, urdr, smidr, tester,
        // vefr, verdandi).
        let mut allowed_tools = self.default_tools.clone();
        if self.inject_envelope_tool
            && !allowed_tools.iter().any(|t| t == RETURN_ENVELOPE_TOOL_NAME)
        {
            allowed_tools.push(RETURN_ENVELOPE_TOOL_NAME.to_string());
        }
        let custom_tools = if self.inject_envelope_tool {
            Some(vec![return_envelope_tool()])
        } else {
            None
        };

        let prompt = format!("{instruction}\n\nTask: {task}");
        let request = SidecarRequest::new(
            request_id.clone(),
            prompt,
            SidecarOptions {
                allowed_tools,
                disallowed_tools: self.disallowed_tools.clone(),
                permission_mode: self.permission_mode.clone(),
                model: Some(model.to_string()),
                working_directory: Some(self.working_dir.to_string_lossy().into_owned()),
                system_prompt: None,
                additional_directories: self.additional_directories.clone(),
                custom_tools,
                // EXT-01 Phase 24: thread per-trajectory sampling params when supplied.
                // None → fields omitted from JSON (skip_serializing_if).
                temperature: sampling.map(|s| s.temperature),
                top_p: sampling.map(|s| s.top_p),
                top_k: sampling.map(|s| s.top_k),
            },
        );

        let request_json = serde_json::to_string(&request).map_err(|e| {
            AlzinaError::Orchestration(format!("failed to serialize sidecar request: {e}"))
        })?;

        stdin
            .write_all(request_json.as_bytes())
            .await
            .map_err(|e| {
                AlzinaError::Orchestration(format!("failed to write to sidecar stdin: {e}"))
            })?;
        stdin.write_all(b"\n").await.map_err(|e| {
            AlzinaError::Orchestration(format!("failed to write newline to sidecar stdin: {e}"))
        })?;
        stdin.flush().await.map_err(|e| {
            AlzinaError::Orchestration(format!("failed to flush sidecar stdin: {e}"))
        })?;

        debug!(request_id = %request_id, "execute request sent to sidecar");

        // ── Process events ──────────────────────────────────────────────

        // Look up the per-dispatch writable dir for THIS spawn's session
        // id and install a RAII guard so the registry entry is cleared
        // on every exit path (success, error, panic, sidecar EOF). The
        // dispatch handler registers `(session_id, dir)` immediately
        // before invoking spawn; missing entries mean "no per-dispatch
        // scope" and the interceptor falls back to existing tier rules.
        let assigned_dir = self.assigned_dirs.get(session_id);
        let _assigned_dir_guard =
            AssignedDirGuard::new(self.assigned_dirs.clone(), session_id.clone());

        // Tool-interceptor session id — distinct from the spawn `session_id`
        // parameter, which is the orchestration-layer SpawnResult id used
        // to tag emitted bus events. The interceptor's id is per-execute
        // and only flows into governance hook decisions.
        let tool_session_id = SessionId::new();
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let mut final_output: Option<String> = None;
        // 260515-ndk Task 3: last-wins capture for typed envelope (P1).
        let mut final_envelope: Option<Envelope> = None;

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).await.map_err(|e| {
                AlzinaError::Orchestration(format!("failed to read sidecar stdout: {e}"))
            })?;

            if bytes_read == 0 {
                // EOF — sidecar closed stdout
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let event: SidecarEvent = match serde_json::from_str(trimmed) {
                Ok(e) => e,
                Err(e) => {
                    warn!(
                        line = %trimmed,
                        error = %e,
                        "failed to parse sidecar event, skipping"
                    );
                    continue;
                }
            };

            match event {
                SidecarEvent::ToolUse {
                    ref id,
                    ref tool,
                    ref input,
                    ref hook_id,
                } => {
                    // 260515-ndk Task 3b: intercept the typed return_envelope
                    // tool BEFORE the governance interceptor fires. Strip
                    // the `mcp__alzina__` prefix (RESEARCH P7) so the
                    // bare-name match catches the typed tool regardless of
                    // whether the model invoked it via the prefixed or
                    // unprefixed surface. On structural deserialisation
                    // failure: warn and fall through to the prose parser
                    // (Q4/P4 — lenient handoff). On success: capture into
                    // final_envelope (last-wins per P1), allow the tool so
                    // the sidecar's PreToolUse hook unblocks the static
                    // tool_result the sidecar short-circuits with, and
                    // continue the loop so the natural Result event still
                    // fires for usage accounting (Q4/P3).
                    let bare = tool.strip_prefix(MCP_ALZINA_PREFIX).unwrap_or(tool);
                    if bare == "return_envelope" {
                        // Decide allow vs block based on three stages:
                        //   1. structural deserialise (existing)
                        //   2. NEW: artifact-manifest validation against the
                        //      assigned dir — every declared path must be
                        //      inside the dir and exist on disk.
                        //   3. on success: rewrite envelope.artifacts to the
                        //      canonical workspace-relative form so
                        //      downstream consumers see one shape.
                        //
                        // Block reasons are stitched into a structured
                        // string the model receives as a tool_result and can
                        // act on (write the missing file or remove the bad
                        // entry, then re-invoke return_envelope).
                        let mut block_reason: Option<String> = None;
                        match deserialise_envelope_input(input) {
                            Ok(mut env) => {
                                if let Some(dir) = assigned_dir.as_deref() {
                                    match tool_interceptor::validate_envelope_artifacts(
                                        &env.artifacts,
                                        self.governance.workspace(),
                                        dir,
                                    ) {
                                        Ok(normalised) => {
                                            env.artifacts = normalised;
                                            debug!(
                                                request_id = %id,
                                                artifact_count = env.artifacts.len(),
                                                "captured Envelope via return_envelope; artifacts validated"
                                            );
                                            final_envelope = Some(env);
                                        }
                                        Err(errors) => {
                                            let reason = format!(
                                                "return_envelope blocked: {} artifact path(s) failed validation:\n  - {}",
                                                errors.len(),
                                                errors.join("\n  - ")
                                            );
                                            warn!(
                                                request_id = %id,
                                                fail_count = errors.len(),
                                                "return_envelope blocked — agent must correct artifacts and re-submit"
                                            );
                                            block_reason = Some(reason);
                                        }
                                    }
                                } else {
                                    debug!(
                                        request_id = %id,
                                        "captured Envelope via return_envelope; no assigned_dir so artifact validation skipped"
                                    );
                                    final_envelope = Some(env);
                                }
                            }
                            Err(e) => warn!(
                                error = %e,
                                "return_envelope deserialisation failed; falling back to prose parse"
                            ),
                        }
                        let hook_msg = match block_reason {
                            Some(reason) => HookDecisionMsg::block(hook_id.clone(), reason),
                            None => HookDecisionMsg::allow(hook_id.clone()),
                        };
                        let hook_json = serde_json::to_string(&hook_msg).map_err(|e| {
                            AlzinaError::Orchestration(format!(
                                "failed to serialize return_envelope hook decision: {e}"
                            ))
                        })?;
                        stdin.write_all(hook_json.as_bytes()).await.map_err(|e| {
                            AlzinaError::Orchestration(format!(
                                "failed to write return_envelope hook decision to sidecar: {e}"
                            ))
                        })?;
                        stdin.write_all(b"\n").await.ok();
                        stdin.flush().await.ok();
                        // Phase 2: mechanical terminal-call enforcement — shutdown immediately
                        // after capturing the envelope. This gives first-wins semantics and
                        // prevents the agent from calling more tools or returning_envelope twice.
                        // Skipped when validation blocked the envelope — the loop continues so
                        // the model receives the block reason as a tool_result and re-issues.
                        if final_envelope.is_some() {
                            let shutdown_json =
                                serde_json::to_string(&ShutdownMsg::default()).unwrap_or_default();
                            let _ = stdin.write_all(shutdown_json.as_bytes()).await;
                            let _ = stdin.write_all(b"\n").await;
                            let _ = stdin.flush().await;
                            break;
                        }
                        continue;
                    }
                    debug!(
                        request_id = %id,
                        tool = %tool,
                        hook_id = %hook_id,
                        "tool use event — checking governance"
                    );

                    // P5-LIVENESS-INNER: forward a synthetic delta so the
                    // streaming dispatch handler's idle timer resets while
                    // the sub-agent is using tools. The bus has no typed
                    // ToolUse variant; piggy-back on TextDelta which the
                    // event_belongs_to_dispatch_tree filter forwards.
                    if let Some(ref emit) = emitter {
                        let now_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as i64)
                            .unwrap_or(0);
                        emit(AlzinaEvent::TextDelta {
                            session_id: session_id.to_string(),
                            turn_id: request_id.clone(),
                            content: format!("[tool_use: {tool}]"),
                            timestamp: now_ms,
                        });
                    }

                    // Convert input to ToolArgs for the interceptor
                    let tool_args =
                        ToolArgs::from_value(&serde_json::to_value(input).unwrap_or_default());

                    // Check governance
                    let decision = tool_interceptor::check_tool_call(
                        &self.governance,
                        agent_id.as_str(),
                        &tool_session_id,
                        tool,
                        &tool_args,
                        assigned_dir.as_deref(),
                    )?;

                    // Send decision back to sidecar
                    let hook_msg = match decision {
                        ToolDecision::Allow => HookDecisionMsg::allow(hook_id.clone()),
                        ToolDecision::Block(reason) => {
                            warn!(
                                tool = %tool,
                                reason = %reason,
                                "tool call blocked by governance"
                            );
                            HookDecisionMsg::block(hook_id.clone(), reason)
                        }
                        ToolDecision::RequiresApproval => {
                            // For now, block tools that require approval
                            // (no interactive approval flow in sidecar mode)
                            warn!(
                                tool = %tool,
                                "tool requires approval — blocking in sidecar mode"
                            );
                            HookDecisionMsg::block(
                                hook_id.clone(),
                                "tool requires operator approval (not available in sidecar mode)"
                                    .into(),
                            )
                        }
                    };

                    let decision_json = serde_json::to_string(&hook_msg).map_err(|e| {
                        AlzinaError::Orchestration(format!(
                            "failed to serialize hook decision: {e}"
                        ))
                    })?;

                    stdin
                        .write_all(decision_json.as_bytes())
                        .await
                        .map_err(|e| {
                            AlzinaError::Orchestration(format!(
                                "failed to write hook decision to sidecar: {e}"
                            ))
                        })?;
                    stdin.write_all(b"\n").await.ok();
                    stdin.flush().await.ok();
                }

                SidecarEvent::ToolResult {
                    ref tool,
                    ref output,
                    ..
                } => {
                    debug!(tool = %tool, output_len = output.len(), "tool result");
                    // P5-LIVENESS-INNER: liveness ping for tool completion.
                    if let Some(ref emit) = emitter {
                        let now_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as i64)
                            .unwrap_or(0);
                        emit(AlzinaEvent::TextDelta {
                            session_id: session_id.to_string(),
                            turn_id: request_id.clone(),
                            content: format!("[tool_result: {tool}]"),
                            timestamp: now_ms,
                        });
                    }
                }

                SidecarEvent::Text { ref content, .. } => {
                    debug!(content_len = content.len(), "streaming text");
                    // P5-LIVENESS-INNER: forward each delta to the daemon
                    // event bus when an emitter is wired so the
                    // dispatch-tree SSE filter sees mid-turn traffic.
                    if let Some(ref emit) = emitter {
                        let now_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as i64)
                            .unwrap_or(0);
                        emit(AlzinaEvent::TextDelta {
                            session_id: session_id.to_string(),
                            turn_id: request_id.clone(),
                            content: content.clone(),
                            timestamp: now_ms,
                        });
                    }
                }

                SidecarEvent::Result { ref content, .. } => {
                    info!(
                        request_id = %request_id,
                        output_len = content.len(),
                        duration_ms = spawn_start.elapsed().as_millis() as u64,
                        "agent execution complete"
                    );
                    final_output = Some(content.clone());
                    break;
                }

                SidecarEvent::Error {
                    ref error,
                    retryable,
                    ..
                } => {
                    error!(
                        request_id = %request_id,
                        error = %error,
                        retryable = retryable,
                        "sidecar reported error"
                    );
                    return Err(AlzinaError::Orchestration(format!(
                        "Claude Agent SDK error: {error}"
                    )));
                }

                // P5-LIVENESS-INNER: forward token-usage events for the
                // sub-agent up to the daemon bus. Mirrors the persistent
                // sidecar path in `sidecar_handle.rs` so chat-side
                // cumulative-usage tracking sees sub-agent token spend.
                SidecarEvent::Usage {
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                    ref model,
                    ..
                } => {
                    debug!(
                        input_tokens,
                        output_tokens, "received usage event from sidecar"
                    );
                    if let Some(ref emit) = emitter {
                        let now_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as i64)
                            .unwrap_or(0);
                        emit(AlzinaEvent::TokenUsage {
                            session_id: session_id.to_string(),
                            turn_id: request_id.clone(),
                            input_tokens,
                            output_tokens,
                            cache_read_input_tokens,
                            cache_creation_input_tokens,
                            model: model.clone(),
                            timestamp: now_ms,
                        });
                    }
                }

                // Persistent-mode events — not expected in one-shot execution.
                SidecarEvent::SessionReady { .. } | SidecarEvent::ChatResponse { .. } => {
                    warn!("received persistent-mode event during one-shot execution, ignoring");
                }
            }
        }

        // ── Shutdown sidecar ────────────────────────────────────────────

        let shutdown_json = serde_json::to_string(&ShutdownMsg::default()).unwrap_or_default();
        let _ = stdin.write_all(shutdown_json.as_bytes()).await;
        let _ = stdin.write_all(b"\n").await;
        let _ = stdin.flush().await;

        // Wait briefly for clean exit, then kill if needed
        match tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await {
            Ok(Ok(status)) => {
                debug!(status = %status, "sidecar exited");
            }
            Ok(Err(e)) => {
                warn!(error = %e, "error waiting for sidecar exit");
            }
            Err(_) => {
                warn!("sidecar did not exit within 5s, killing");
                let _ = child.kill().await;
            }
        }

        // ── Return result ───────────────────────────────────────────────
        //
        // 260515-ndk Task 3: EOF-safe envelope return (P8). When the
        // sidecar exits without a terminal `Result` event but the model
        // already invoked `return_envelope`, return the captured envelope
        // with an empty `raw` rather than erroring — the runner's typed
        // path uses the typed envelope directly and re-renders `raw` from
        // it via `render_envelope_as_prose`, so the empty string here is
        // overwritten downstream.
        match (final_output, final_envelope) {
            (Some(out), env) => Ok((out, env)),
            (None, Some(env)) => Ok((String::new(), Some(env))),
            (None, None) => Err(AlzinaError::Orchestration(
                "sidecar exited without producing a result event or return_envelope tool call"
                    .into(),
            )),
        }
    }
}

/// 260515-ndk Task 3b: deserialise a `SidecarEvent::ToolUse.input`
/// `HashMap<String, serde_json::Value>` into an `alzina_core::Envelope`.
///
/// The model-facing JSON Schema in `envelope_tool::return_envelope_tool`
/// uses lowercase status strings (`"complete"` / `"partial"` / `"error"`)
/// while `alzina_core::EnvelopeStatus` serialises as PascalCase
/// (`"Complete"` / `"Partial"` / `"Error"`). Normalise the status field
/// before handing off to serde so the captured payload deserialises
/// without round-tripping through a wrapper type.
///
/// Returns an `AlzinaError::EnvelopeParse` on any structural failure so
/// the caller can fall back to the prose parser cleanly (Q4/P4).
fn deserialise_envelope_input(
    input: &std::collections::HashMap<String, serde_json::Value>,
) -> AlzinaResult<Envelope> {
    let mut obj: serde_json::Map<String, serde_json::Value> =
        input.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

    // Normalise lowercase status → PascalCase so serde matches the
    // EnvelopeStatus variant names.
    if let Some(serde_json::Value::String(s)) = obj.get("status").cloned() {
        let normalised = match s.as_str() {
            "complete" => "Complete",
            "partial" => "Partial",
            "error" => "Error",
            other => other,
        };
        obj.insert(
            "status".into(),
            serde_json::Value::String(normalised.into()),
        );
    }

    // `alzina_core::Envelope::artifacts` is `Vec<PathBuf>` WITHOUT a serde
    // default, so an absent field aborts deserialisation. The JSON Schema
    // in envelope_tool::return_envelope_tool declares `default: []` for
    // artifacts, so we honour that default here to keep the schema and
    // the deserialiser symmetric.
    obj.entry("artifacts")
        .or_insert_with(|| serde_json::Value::Array(vec![]));

    // The schema declares `artifacts` as an array, but models routinely emit
    // a single bare string (`"artifacts/foo.md"`) or a newline-bulleted block
    // — the same prose convention they use in the return-format trailer. Both
    // make `Vec<PathBuf>` deserialisation fail with
    // `invalid type: string, expected a sequence`, which dropped EVERY such
    // envelope to the prose fallback (and triggered costly re-submits). Accept
    // string-or-list by coercing a string into the array shape serde expects.
    if let Some(artifacts) = obj.get_mut("artifacts") {
        if let serde_json::Value::String(s) = artifacts {
            *artifacts = serde_json::Value::Array(coerce_artifacts_string(s));
        }
    }

    // alzina_core::Envelope::artifacts is Vec<PathBuf>, which deserialises
    // from a JSON array of strings out of the box. signal/tensions/etc are
    // Option<String> — serde maps absent fields to None automatically.
    serde_json::from_value::<Envelope>(serde_json::Value::Object(obj)).map_err(|e| {
        AlzinaError::EnvelopeParse(format!("return_envelope payload did not deserialise: {e}"))
    })
}

/// Coerce a string `artifacts` field into the array of path strings serde
/// expects for `Vec<PathBuf>`.
///
/// A model may emit a single path (`"artifacts/foo.md"`) or a newline-bulleted
/// block matching the prose return-format trailer. Split on newlines, strip a
/// leading list marker (`-`, `*`, `•`) and surrounding whitespace, and drop
/// empty lines. A single bare path yields a one-element array. An empty or
/// whitespace-only string yields an empty array.
fn coerce_artifacts_string(s: &str) -> Vec<serde_json::Value> {
    s.lines()
        .map(|line| {
            line.trim()
                .trim_start_matches(['-', '*', '•'])
                .trim()
                .to_string()
        })
        .filter(|line| !line.is_empty())
        .map(serde_json::Value::String)
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_governance::config::GovernanceConfig;
    use alzina_workspace::WorkspaceHandle;
    use std::io::Write as _;

    /// Create a mock sidecar script that echoes a canned result.
    fn write_mock_sidecar(dir: &std::path::Path, response_json: &str) -> PathBuf {
        let script_path = dir.join("mock-sidecar.js");
        let script = format!(
            r#"
const readline = require('readline');
const rl = readline.createInterface({{ input: process.stdin, terminal: false }});
rl.on('line', (line) => {{
    const msg = JSON.parse(line);
    if (msg.type === 'execute') {{
        process.stdout.write(JSON.stringify({response_json}) + '\n');
    }} else if (msg.type === 'shutdown') {{
        process.exit(0);
    }}
}});
"#,
            response_json = response_json,
        );
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    /// Create a mock sidecar that emits a tool_use event, waits for decision,
    /// then emits the result.
    fn write_hook_mock_sidecar(dir: &std::path::Path) -> PathBuf {
        let script_path = dir.join("mock-hook-sidecar.js");
        let script = r#"
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
let state = 'waiting_execute';

rl.on('line', (line) => {
    const msg = JSON.parse(line);

    if (msg.type === 'execute' && state === 'waiting_execute') {
        // Emit a tool_use event
        process.stdout.write(JSON.stringify({
            type: "tool_use",
            id: msg.id,
            tool: "Read",
            input: { path: "src/main.rs" },
            hook_id: "hook-1"
        }) + '\n');
        state = 'waiting_decision';
    } else if (msg.type === 'hook_decision' && state === 'waiting_decision') {
        if (msg.decision === 'allow') {
            process.stdout.write(JSON.stringify({
                type: "tool_result",
                id: "req",
                tool: "Read",
                output: "fn main() {}"
            }) + '\n');
            process.stdout.write(JSON.stringify({
                type: "result",
                id: "req",
                content: "Read file successfully."
            }) + '\n');
        } else {
            process.stdout.write(JSON.stringify({
                type: "result",
                id: "req",
                content: "Tool was blocked by governance."
            }) + '\n');
        }
        state = 'done';
    } else if (msg.type === 'shutdown') {
        process.exit(0);
    }
});
"#;
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    fn build_test_governance(dir: &std::path::Path) -> Arc<GovernanceLayer> {
        let workspace = Arc::new(WorkspaceHandle::open(dir.to_path_buf()).unwrap());
        let config = GovernanceConfig::default();
        Arc::new(GovernanceLayer::new(config, workspace).unwrap())
    }

    #[tokio::test]
    async fn execute_returns_result_from_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let result_json = r#"{"type":"result","id":"req-001","content":"Analysis complete."}"#;
        let sidecar_path = write_mock_sidecar(dir.path(), result_json);
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec!["Read".into()],
            "default".into(),
        );

        let result = executor
            .execute(
                &AgentId::new("test-agent"),
                "You are a test agent.",
                "test-model",
                "Do nothing.",
            )
            .await
            .unwrap();

        assert_eq!(result, "Analysis complete.");
    }

    #[tokio::test]
    async fn execute_handles_error_event() {
        let dir = tempfile::tempdir().unwrap();
        let error_json =
            r#"{"type":"error","id":"req-001","error":"rate limited","retryable":true}"#;
        let sidecar_path = write_mock_sidecar(dir.path(), error_json);
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec!["Read".into()],
            "default".into(),
        );

        let result = executor
            .execute(&AgentId::new("test-agent"), "instruction", "model", "task")
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("rate limited"), "error was: {err}");
    }

    #[tokio::test]
    async fn execute_handles_hook_protocol() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_hook_mock_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec!["Read".into()],
            "default".into(),
        );

        let result = executor
            .execute(
                &AgentId::new("test-agent"),
                "instruction",
                "model",
                "read a file",
            )
            .await
            .unwrap();

        // Read is a non-write tool, so governance allows it.
        // Mock sidecar then returns "Read file successfully."
        assert_eq!(result, "Read file successfully.");
    }

    /// Mock sidecar that emits a `text` event mid-run, then a `result`.
    /// Used to verify the P5-LIVENESS-INNER emitter callback fires for
    /// streaming text deltas.
    fn write_streaming_text_sidecar(dir: &std::path::Path) -> PathBuf {
        let script_path = dir.join("mock-streaming-sidecar.js");
        let script = r#"
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', (line) => {
    const msg = JSON.parse(line);
    if (msg.type === 'execute') {
        process.stdout.write(JSON.stringify({
            type: "text",
            id: msg.id,
            content: "thinking..."
        }) + '\n');
        process.stdout.write(JSON.stringify({
            type: "text",
            id: msg.id,
            content: "almost done."
        }) + '\n');
        process.stdout.write(JSON.stringify({
            type: "result",
            id: msg.id,
            content: "Final answer."
        }) + '\n');
    } else if (msg.type === 'shutdown') {
        process.exit(0);
    }
});
"#;
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    #[tokio::test]
    async fn execute_with_emitter_publishes_text_deltas() {
        use crate::runner::alzina_runner::ExecutorEventEmitter;
        use std::sync::Mutex;

        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_streaming_text_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec!["Read".into()],
            "default".into(),
        );

        let captured: Arc<Mutex<Vec<AlzinaEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);
        let emitter: ExecutorEventEmitter = Arc::new(move |evt: AlzinaEvent| {
            captured_clone.lock().unwrap().push(evt);
        });

        let spawn_session = SessionId::new();
        let result = executor
            .execute_with_emitter(
                &AgentId::new("test-agent"),
                "instruction",
                "model",
                "task",
                &spawn_session,
                Some(emitter),
            )
            .await
            .unwrap();

        assert_eq!(result, "Final answer.");

        let events = captured.lock().unwrap().clone();
        let deltas: Vec<&AlzinaEvent> = events
            .iter()
            .filter(|e| matches!(e, AlzinaEvent::TextDelta { .. }))
            .collect();
        assert_eq!(
            deltas.len(),
            2,
            "expected two TextDelta events, got: {:?}",
            events
        );
        // Confirm session_id tagging matches the spawn id.
        for d in &deltas {
            if let AlzinaEvent::TextDelta { session_id, .. } = d {
                assert_eq!(session_id, &spawn_session.to_string());
            }
        }
        // Confirm the content arrived.
        let payloads: Vec<String> = deltas
            .iter()
            .filter_map(|e| match e {
                AlzinaEvent::TextDelta { content, .. } => Some(content.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            payloads,
            vec!["thinking...".to_string(), "almost done.".to_string()]
        );
    }

    // ── 260515-ndk Task 3a: outbound SidecarRequest injection tests ─────
    //
    // Tests 1 and 2 from the plan: assert the outbound SidecarRequest's
    // `options.custom_tools` always carries `return_envelope_tool()` and
    // `options.allowed_tools` always includes `mcp__alzina__return_envelope`.
    // We verify by writing a mock sidecar that JSON-encodes the entire
    // received `execute` request body back inside the terminal Result
    // event — the executor returns that string verbatim, letting the
    // test parse and assert structural shape.

    /// Mock sidecar that echoes the received `execute` request as the
    /// terminal Result content. Used by the injection tests below to
    /// inspect what the executor actually sent on the wire.
    fn write_echo_request_sidecar(dir: &std::path::Path) -> PathBuf {
        let script_path = dir.join("mock-echo-request-sidecar.js");
        let script = r#"
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', (line) => {
    const msg = JSON.parse(line);
    if (msg.type === 'execute') {
        // Echo the full request back inside the Result content so the
        // Rust test can parse it and assert custom_tools / allowed_tools.
        process.stdout.write(JSON.stringify({
            type: "result",
            id: msg.id,
            content: JSON.stringify(msg)
        }) + '\n');
    } else if (msg.type === 'shutdown') {
        process.exit(0);
    }
});
"#;
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    /// Test 1: the outbound SidecarRequest carries
    /// `options.custom_tools = Some(vec![return_envelope_tool()])`
    /// unconditionally on every dispatch.
    #[tokio::test]
    async fn execute_with_envelope_injects_return_envelope_tool() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_echo_request_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec!["Read".into(), "Write".into()],
            "default".into(),
        );

        let (raw, env) = executor
            .execute_with_envelope(
                &AgentId::new("test-agent"),
                "instruction",
                "test-model",
                "task",
                &SessionId::new(),
                None,
            )
            .await
            .unwrap();

        // The echo sidecar didn't call the typed return_envelope tool
        // (it doesn't simulate the model invoking it), so envelope is None.
        assert!(env.is_none(), "echo sidecar should not produce a typed env");

        let echoed: serde_json::Value =
            serde_json::from_str(&raw).expect("Result content must be the JSON request echo");

        // options.customTools (camelCase on the wire) must be a 1-element
        // array whose first entry has name == "return_envelope".
        let tools = echoed
            .pointer("/options/customTools")
            .expect("customTools must be present on outbound request");
        let arr = tools.as_array().expect("customTools must be an array");
        assert_eq!(arr.len(), 1, "must inject exactly one custom tool");
        assert_eq!(
            arr[0].get("name").and_then(|v| v.as_str()),
            Some("return_envelope")
        );
    }

    /// Test 2: the outbound SidecarRequest's `options.allowed_tools`
    /// (`allowedTools` on the wire) includes `mcp__alzina__return_envelope`
    /// so the SDK permits the model to invoke the tool.
    #[tokio::test]
    async fn execute_with_envelope_extends_allowed_tools_with_return_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_echo_request_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec!["Read".into()],
            "default".into(),
        );

        let (raw, _env) = executor
            .execute_with_envelope(
                &AgentId::new("test-agent"),
                "instruction",
                "test-model",
                "task",
                &SessionId::new(),
                None,
            )
            .await
            .unwrap();

        let echoed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let allowed = echoed
            .pointer("/options/allowedTools")
            .and_then(|v| v.as_array())
            .expect("allowedTools must be an array");

        let names: Vec<&str> = allowed.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            names.contains(&"mcp__alzina__return_envelope"),
            "allowedTools must include mcp__alzina__return_envelope, got: {names:?}"
        );
        // Existing defaults are preserved.
        assert!(names.contains(&"Read"), "Read default must be preserved");
    }

    // ── 260515-ndk Task 3b: return_envelope interception tests ──────────
    //
    // Tests 3-6 from the plan. Each writes a mock JS sidecar that emits
    // a synthetic event stream and asserts the executor's
    // `execute_with_envelope` returns the expected (raw, Option<Envelope>).

    /// Mock sidecar that emits a single `mcp__alzina__return_envelope`
    /// tool_use with the supplied JSON input string, then a terminal
    /// `result`. The Rust executor must intercept the tool_use, capture
    /// the typed envelope, allow the tool, and continue to the Result
    /// (which carries the prose body).
    fn write_typed_envelope_sidecar(
        dir: &std::path::Path,
        envelope_input_json: &str,
        result_content: &str,
    ) -> PathBuf {
        let script_path = dir.join("mock-typed-envelope-sidecar.js");
        let script = format!(
            r#"
const readline = require('readline');
const rl = readline.createInterface({{ input: process.stdin, terminal: false }});
let state = 'waiting_execute';

rl.on('line', (line) => {{
    const msg = JSON.parse(line);

    if (msg.type === 'execute' && state === 'waiting_execute') {{
        process.stdout.write(JSON.stringify({{
            type: "tool_use",
            id: msg.id,
            tool: "mcp__alzina__return_envelope",
            input: {envelope_input_json},
            hook_id: "h-env-1"
        }}) + '\n');
        state = 'waiting_allow';
    }} else if (msg.type === 'hook_decision' && state === 'waiting_allow') {{
        // After the executor allows the typed tool, emit the result.
        process.stdout.write(JSON.stringify({{
            type: "result",
            id: "req",
            content: {result_content}
        }}) + '\n');
        state = 'done';
    }} else if (msg.type === 'shutdown') {{
        process.exit(0);
    }}
}});
"#,
            envelope_input_json = envelope_input_json,
            result_content = result_content,
        );
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    /// Test 3: feed a synthetic stream with one valid `mcp__alzina__return_envelope`
    /// tool_use → executor returns (raw, Some(env)) and writes
    /// `HookDecisionMsg::allow("h-env-1")` to the sidecar stdin.
    /// Phase 2: after capturing the envelope, the executor sends shutdown
    /// and breaks the event loop — the Result event never arrives, so raw is "".
    #[tokio::test]
    async fn return_envelope_intercept_captures_typed_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let envelope_input = r#"{
            "status": "complete",
            "signal": "captured via typed tool",
            "artifacts": ["artifacts/x.md"],
            "tensions": "none",
            "emergent": "none"
        }"#;
        let sidecar_path =
            write_typed_envelope_sidecar(dir.path(), envelope_input, "\"some prose body\"");
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec![],
            "default".into(),
        );

        let (raw, env) = executor
            .execute_with_envelope(
                &AgentId::new("test-agent"),
                "instruction",
                "test-model",
                "task",
                &SessionId::new(),
                None,
            )
            .await
            .unwrap();

        let env = env.expect("typed envelope must be captured from valid tool_use");
        assert_eq!(env.status, alzina_core::EnvelopeStatus::Complete);
        assert_eq!(env.signal.as_deref(), Some("captured via typed tool"));
        assert_eq!(env.artifacts.len(), 1);
        // Phase 2: shutdown fires immediately after envelope capture — the
        // sidecar's Result event does not arrive. raw is empty; the runner's
        // typed-path branch re-renders canonical raw from the typed envelope.
        assert!(
            raw.is_empty(),
            "raw must be empty when shutdown fires before Result: got {raw:?}"
        );
    }

    /// Test 4: structurally invalid input (missing required `status`) →
    /// executor logs warning and returns (raw, None). The runner's
    /// lenient prose-parser fallback then handles the prose body.
    #[tokio::test]
    async fn return_envelope_intercept_invalid_payload_falls_back_to_prose() {
        let dir = tempfile::tempdir().unwrap();
        // Missing required `status` field — must fail deserialisation.
        let invalid_input = r#"{"signal": "no status"}"#;
        let sidecar_path = write_typed_envelope_sidecar(
            dir.path(),
            invalid_input,
            "\"STATUS: complete\\nARTIFACTS: a.md\"",
        );
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec![],
            "default".into(),
        );

        let (raw, env) = executor
            .execute_with_envelope(
                &AgentId::new("test-agent"),
                "instruction",
                "test-model",
                "task",
                &SessionId::new(),
                None,
            )
            .await
            .unwrap();

        assert!(
            env.is_none(),
            "invalid payload must yield None — runner falls back to prose parser"
        );
        // raw still carries the prose body so the runner can attempt to
        // parse it via the unchanged strict-then-lenient path.
        assert!(raw.contains("STATUS"));
    }

    /// Test 5: two consecutive valid tool_use events → last wins (P1).
    fn write_last_wins_sidecar(dir: &std::path::Path) -> PathBuf {
        let script_path = dir.join("mock-last-wins-sidecar.js");
        let script = r#"
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
let state = 'waiting_execute';
let allows = 0;

rl.on('line', (line) => {
    const msg = JSON.parse(line);

    if (msg.type === 'execute' && state === 'waiting_execute') {
        // First (older) envelope: partial.
        process.stdout.write(JSON.stringify({
            type: "tool_use",
            id: msg.id,
            tool: "mcp__alzina__return_envelope",
            input: { status: "partial", signal: "first attempt" },
            hook_id: "h-1"
        }) + '\n');
        state = 'waiting_allow';
    } else if (msg.type === 'hook_decision' && state === 'waiting_allow') {
        allows += 1;
        if (allows === 1) {
            // Second (newer) envelope: complete — must overwrite the first.
            process.stdout.write(JSON.stringify({
                type: "tool_use",
                id: "req",
                tool: "mcp__alzina__return_envelope",
                input: { status: "complete", signal: "second attempt wins" },
                hook_id: "h-2"
            }) + '\n');
            // stay in waiting_allow for the second allow
        } else if (allows === 2) {
            process.stdout.write(JSON.stringify({
                type: "result",
                id: "req",
                content: "done"
            }) + '\n');
            state = 'done';
        }
    } else if (msg.type === 'shutdown') {
        process.exit(0);
    }
});
"#;
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    /// Phase 2: first-wins semantics — the executor shuts down immediately
    /// after capturing the first envelope. The second tool_use event
    /// (which the mock sidecar tries to emit) is never processed.
    #[tokio::test]
    async fn return_envelope_intercept_first_wins() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_last_wins_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec![],
            "default".into(),
        );

        let (_raw, env) = executor
            .execute_with_envelope(
                &AgentId::new("test-agent"),
                "instruction",
                "test-model",
                "task",
                &SessionId::new(),
                None,
            )
            .await
            .unwrap();

        let env = env.expect("envelope must be captured");
        // Phase 2: first-wins semantics — executor breaks after capturing the
        // FIRST envelope (partial). The sidecar never gets to emit the second.
        assert_eq!(
            env.status,
            alzina_core::EnvelopeStatus::Partial,
            "first-wins: first tool_use must be captured; second never processed"
        );
        assert_eq!(env.signal.as_deref(), Some("first attempt"));
    }

    /// Test 6: sidecar exits via EOF (no terminal Result) after capturing
    /// an envelope → executor returns (String::new(), Some(env))
    /// (P8: kill_on_drop semantics — never drop the captured envelope).
    ///
    /// The mock exits the node process directly after the hook_decision
    /// allow lands (rather than just `stdout.end()`-ing and waiting for
    /// shutdown). Either path proves EOF behaviour on the executor side;
    /// exit(0) is simpler and avoids relying on the executor's 5s
    /// shutdown-wait codepath inside a fast unit test.
    fn write_eof_after_envelope_sidecar(dir: &std::path::Path) -> PathBuf {
        let script_path = dir.join("mock-eof-after-envelope-sidecar.js");
        let script = r#"
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
let state = 'waiting_execute';

rl.on('line', (line) => {
    const msg = JSON.parse(line);

    if (msg.type === 'execute' && state === 'waiting_execute') {
        const out = JSON.stringify({
            type: "tool_use",
            id: msg.id,
            tool: "mcp__alzina__return_envelope",
            input: { status: "complete", signal: "before EOF" },
            hook_id: "h-eof"
        }) + '\n';
        process.stdout.write(out);
        state = 'waiting_allow';
    } else if (msg.type === 'hook_decision' && state === 'waiting_allow') {
        // Flush stdout then exit. The executor's read loop then hits
        // EOF on stdout and must still return the captured envelope.
        process.stdout.end(() => process.exit(0));
    } else if (msg.type === 'shutdown') {
        process.exit(0);
    }
});
"#;
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    #[tokio::test]
    async fn return_envelope_intercept_eof_preserves_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_eof_after_envelope_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec![],
            "default".into(),
        );

        let (raw, env) = executor
            .execute_with_envelope(
                &AgentId::new("test-agent"),
                "instruction",
                "test-model",
                "task",
                &SessionId::new(),
                None,
            )
            .await
            .unwrap();

        let env = env.expect("envelope captured before EOF must be returned");
        assert_eq!(env.status, alzina_core::EnvelopeStatus::Complete);
        assert_eq!(env.signal.as_deref(), Some("before EOF"));
        // raw is empty because no Result event arrived — the runner's
        // typed-path branch re-renders the canonical raw from `env`
        // downstream so this empty string is overwritten before any
        // downstream consumer sees it.
        assert!(raw.is_empty());
    }

    /// Unit test for the in-module `deserialise_envelope_input` helper —
    /// proves the lowercase-status normalisation closes the schema/Rust
    /// mismatch (plan must_haves: "input deserialised as alzina_core::Envelope").
    #[test]
    fn deserialise_envelope_input_normalises_lowercase_status() {
        let mut input = std::collections::HashMap::new();
        input.insert("status".to_string(), serde_json::json!("complete"));
        input.insert("signal".to_string(), serde_json::json!("ok"));
        input.insert(
            "artifacts".to_string(),
            serde_json::json!(["artifacts/foo.md"]),
        );

        let env = deserialise_envelope_input(&input)
            .expect("lowercase status must normalise to PascalCase");
        assert_eq!(env.status, alzina_core::EnvelopeStatus::Complete);
        assert_eq!(env.signal.as_deref(), Some("ok"));
        assert_eq!(env.artifacts.len(), 1);
    }

    /// A model that emits `artifacts` as a single bare string (instead of
    /// the schema-declared array) must still deserialise — this was the
    /// pervasive `invalid type: string, expected a sequence` failure that
    /// dropped every such envelope to the prose fallback.
    #[test]
    fn deserialise_envelope_input_coerces_string_artifacts() {
        let mut input = std::collections::HashMap::new();
        input.insert("status".to_string(), serde_json::json!("complete"));
        input.insert(
            "artifacts".to_string(),
            serde_json::json!("artifacts/foo.md"),
        );

        let env = deserialise_envelope_input(&input)
            .expect("a bare-string artifacts field must coerce to a one-element list");
        assert_eq!(env.artifacts.len(), 1);
        assert_eq!(
            env.artifacts[0],
            std::path::PathBuf::from("artifacts/foo.md")
        );
    }

    /// A newline-bulleted artifacts string (the prose-trailer convention)
    /// splits into one path per line, with list markers stripped.
    #[test]
    fn deserialise_envelope_input_coerces_bulleted_string_artifacts() {
        let mut input = std::collections::HashMap::new();
        input.insert("status".to_string(), serde_json::json!("complete"));
        input.insert(
            "artifacts".to_string(),
            serde_json::json!("- artifacts/a.md\n- artifacts/b.md\n"),
        );

        let env = deserialise_envelope_input(&input)
            .expect("a bulleted artifacts string must split into a list");
        assert_eq!(env.artifacts.len(), 2);
        assert_eq!(env.artifacts[0], std::path::PathBuf::from("artifacts/a.md"));
        assert_eq!(env.artifacts[1], std::path::PathBuf::from("artifacts/b.md"));
    }

    /// An empty / whitespace-only artifacts string coerces to an empty list,
    /// not a one-element list holding `""`.
    #[test]
    fn deserialise_envelope_input_coerces_empty_string_artifacts() {
        let mut input = std::collections::HashMap::new();
        input.insert("status".to_string(), serde_json::json!("complete"));
        input.insert("artifacts".to_string(), serde_json::json!("   "));

        let env = deserialise_envelope_input(&input)
            .expect("an empty artifacts string must coerce to an empty list");
        assert!(env.artifacts.is_empty());
    }

    /// Unit test for invalid payload — missing required `status`.
    #[test]
    fn deserialise_envelope_input_rejects_missing_status() {
        let mut input = std::collections::HashMap::new();
        input.insert("signal".to_string(), serde_json::json!("no status"));
        let result = deserialise_envelope_input(&input);
        assert!(
            result.is_err(),
            "missing status must fail deserialisation, got: {result:?}"
        );
    }

    /// Sanity: the legacy `execute_with_emitter` entry point also routes
    /// through the shared helper, so it also gets custom_tools injected
    /// (same dispatch surface; the only difference is whether the runner
    /// captures the optional envelope or discards it).
    #[tokio::test]
    async fn execute_with_emitter_also_injects_return_envelope_tool() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_echo_request_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec![],
            "default".into(),
        );

        let raw = executor
            .execute_with_emitter(
                &AgentId::new("test-agent"),
                "instruction",
                "test-model",
                "task",
                &SessionId::new(),
                None,
            )
            .await
            .unwrap();

        let echoed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(
            echoed
                .pointer("/options/customTools")
                .and_then(|v| v.as_array())
                .map(|a| a
                    .iter()
                    .any(|t| t.get("name").and_then(|n| n.as_str()) == Some("return_envelope")))
                .unwrap_or(false),
            "execute_with_emitter must also inject return_envelope custom tool"
        );
    }

    /// Write a sidecar that emits return_envelope, then immediately tries to emit
    /// a second tool_use event. With shutdown-on-capture, the second event must
    /// never be processed by the runner.
    fn write_post_envelope_tool_sidecar(dir: &std::path::Path) -> PathBuf {
        let script_path = dir.join("mock-post-envelope-tool-sidecar.js");
        let script = r#"
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
let state = 'waiting_execute';
let hookCount = 0;

rl.on('line', (line) => {
    const msg = JSON.parse(line);

    if (msg.type === 'execute' && state === 'waiting_execute') {
        // First: return_envelope tool
        process.stdout.write(JSON.stringify({
            type: "tool_use",
            id: msg.id,
            tool: "mcp__alzina__return_envelope",
            input: { status: "complete", signal: "envelope returned" },
            hook_id: "h-env"
        }) + '\n');
        state = 'waiting_first_allow';
    } else if (msg.type === 'hook_decision' && state === 'waiting_first_allow') {
        hookCount++;
        // Try to emit a second tool_use AFTER the envelope was captured.
        // With first-wins shutdown, the executor breaks and we never receive
        // a second hook_decision. If we do receive one, the test is wrong.
        process.stdout.write(JSON.stringify({
            type: "tool_use",
            id: "req2",
            tool: "mcp__alzina__bash",
            input: { command: "echo should_not_run" },
            hook_id: "h-bash"
        }) + '\n');
        state = 'waiting_second_allow';
    } else if (msg.type === 'hook_decision' && state === 'waiting_second_allow') {
        // This should never be reached — executor must have broken after first envelope.
        process.stdout.write(JSON.stringify({
            type: "result",
            id: "req2",
            content: "second_tool_ran"
        }) + '\n');
    } else if (msg.type === 'shutdown') {
        process.exit(0);
    }
});
"#;
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    /// return_envelope_terminates_event_loop: after the executor captures
    /// the envelope and sends shutdown, no subsequent tool calls are processed.
    ///
    /// The mock sidecar tries to emit a bash tool_use after the envelope — the
    /// executor must break before that event reaches the governance check.
    /// Verified by asserting (a) the envelope is captured and (b) the raw
    /// output does NOT contain "second_tool_ran" (which the sidecar's second
    /// branch would write only if the bash tool's hook_decision was sent back).
    #[tokio::test]
    async fn return_envelope_terminates_event_loop() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_post_envelope_tool_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            sidecar_path,
            governance,
            dir.path().to_path_buf(),
            vec![],
            "default".into(),
        );

        let (raw, env) = executor
            .execute_with_envelope(
                &AgentId::new("test-agent"),
                "instruction",
                "test-model",
                "task",
                &SessionId::new(),
                None,
            )
            .await
            .unwrap();

        // (a) Envelope is captured with correct status.
        let env = env.expect("envelope must be captured");
        assert_eq!(env.status, alzina_core::EnvelopeStatus::Complete);
        assert_eq!(env.signal.as_deref(), Some("envelope returned"));

        // (b) Second tool never ran — raw does not contain the second result.
        assert!(
            !raw.contains("second_tool_ran"),
            "event loop must terminate after return_envelope; raw: {raw:?}"
        );
    }

    #[tokio::test]
    async fn execute_handles_missing_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let governance = build_test_governance(dir.path());

        let executor = ClaudeAgentSdkExecutor::new(
            PathBuf::from("/nonexistent/sidecar.js"),
            governance,
            dir.path().to_path_buf(),
            vec![],
            "default".into(),
        );

        let result = executor
            .execute(&AgentId::new("test-agent"), "instruction", "model", "task")
            .await;

        assert!(result.is_err());
    }
}
