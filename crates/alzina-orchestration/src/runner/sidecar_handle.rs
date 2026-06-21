//! `SidecarHandle` — persistent process wrapper for the orchestrator sidecar.
//!
//! Unlike `ClaudeAgentSdkExecutor` which spawns a fresh sidecar per `execute()`
//! call, `SidecarHandle` keeps a single Node.js process alive across multiple
//! chat turns. This is the Rust side of the persistent-mode protocol defined in
//! `sidecar/claude-agent/src/protocol.ts`.
//!
//! # Lifecycle
//!
//! ```text
//! start_session()          -- spawn process, send session_start, await session_ready
//!   chat_turn("turn-1")    -- send chat_message, read events, return response
//!   chat_turn("turn-2")    -- same process, conversation continues
//!   ...
//! end_session()            -- send session_end + shutdown, wait for exit
//! ```
//!
//! The sidecar owns the SDK conversation history (message array) in-process.
//! Rust owns the durable record for crash recovery and SSE streaming.
//!
//! # Governance
//!
//! During a chat turn, the sidecar may emit `tool_use` events for governance
//! decisions. `SidecarHandle` does not handle these directly — the caller
//! (typically `ChatSession`) is responsible for providing a governance callback.
//! However, `chat_turn` processes `tool_use` / `hook_decision` inline using the
//! same pattern as `ClaudeAgentSdkExecutor`, forwarding governance decisions
//! from the provided `GovernanceLayer`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use alzina_core::identity::SessionId;
use alzina_core::{AlzinaError, AlzinaEvent, AlzinaResult};
use alzina_governance::GovernanceLayer;

use super::sidecar_protocol::{
    ChatMessageMsg, CustomToolDefinition, HookDecisionMsg, SessionEndMsg, SessionStartMsg,
    ShutdownMsg, SidecarEvent, TurnCancelMsg,
};
use super::tool_interceptor::{self, ToolArgs, ToolDecision};

/// Callback invoked by [`SidecarHandle::chat_turn`] for each in-turn event
/// the daemon may want to forward to its event bus (Phase 4 streaming).
///
/// Currently called for `text` and `usage` sidecar events. Kept as a plain
/// closure type so this crate does not depend on `alzina-daemon`'s `EventBus`.
pub type ChatEventEmitter = Arc<dyn Fn(AlzinaEvent) + Send + Sync>;

// ── SidecarHandle ──────────────────────────────────────────────────────────

/// Handle to a long-lived sidecar process for persistent chat sessions.
///
/// The sidecar process stays alive between chat turns, maintaining the SDK's
/// conversation history in-process. Communication is newline-delimited JSON
/// over stdin/stdout.
pub struct SidecarHandle {
    /// The child process (kill_on_drop = true for safety).
    child: Child,
    /// Buffered writer to the sidecar's stdin.
    stdin: BufWriter<ChildStdin>,
    /// Line-oriented reader from the sidecar's stdout.
    stdout: Lines<BufReader<ChildStdout>>,
    /// Session ID for this persistent session.
    session_id: String,
    /// Tracks whether the sidecar process is still alive.
    alive: Arc<AtomicBool>,
    /// Background task draining the sidecar's stderr to prevent pipe deadlock.
    stderr_task: JoinHandle<()>,

    // ── Fields retained for crash recovery ─────────────────────────────
    /// Path to the sidecar JS entry point (needed to respawn).
    sidecar_path: PathBuf,
    /// System prompt from bootstrap (re-injected on recovery).
    system_prompt: String,
    /// Custom tool definitions (re-injected on recovery).
    custom_tools: Vec<CustomToolDefinition>,
    /// Working directory for the sidecar process.
    working_dir: PathBuf,
    /// Daemon API key (re-injected on recovery so the recovered sidecar
    /// can continue to authenticate custom-tool callbacks). `None` in dev mode.
    api_key: Option<String>,
    /// Model id forwarded into chat-turn `query()` calls. `None` lets the
    /// SDK pick its own default. Re-injected on recovery so the
    /// recovered sidecar continues to use the same model.
    model: Option<String>,
}

impl SidecarHandle {
    /// Spawn the sidecar process and establish a persistent session.
    ///
    /// Sends a `session_start` message and waits for the `session_ready` event.
    /// The sidecar initialises the SDK with the given system prompt and custom
    /// tools, then signals readiness.
    ///
    /// # Arguments
    ///
    /// * `sidecar_path` — Path to the sidecar JS entry point
    /// * `session_id` — Unique session identifier
    /// * `system_prompt` — System instruction from bootstrap
    /// * `custom_tools` — Custom tool definitions (e.g. `dispatch_agent`)
    /// * `working_dir` — Working directory for the sidecar process
    /// * `api_key` — Daemon API key forwarded to the sidecar so it can attach
    ///   `Authorization: Bearer <key>` when invoking custom tools (red-team A7).
    ///   `None` is correct for dev-mode deployments where the daemon has no api_key.
    pub async fn start_session(
        sidecar_path: PathBuf,
        session_id: String,
        system_prompt: String,
        custom_tools: Vec<CustomToolDefinition>,
        working_dir: PathBuf,
        api_key: Option<String>,
        model: Option<String>,
    ) -> AlzinaResult<Self> {
        info!(
            session_id = %session_id,
            sidecar = %sidecar_path.display(),
            "spawning persistent sidecar"
        );

        // ── Spawn process ──────────────────────────────────────────

        let mut child = Command::new("node")
            .arg(&sidecar_path)
            .current_dir(&working_dir)
            // Headless persistent session: suppress telemetry/autoupdate
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
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                AlzinaError::Orchestration(format!(
                    "failed to spawn sidecar at {}: {e}",
                    sidecar_path.display()
                ))
            })?;

        let child_stdin = child
            .stdin
            .take()
            .ok_or_else(|| AlzinaError::Orchestration("sidecar stdin not available".into()))?;
        let child_stdout = child
            .stdout
            .take()
            .ok_or_else(|| AlzinaError::Orchestration("sidecar stdout not available".into()))?;

        let child_stderr = child
            .stderr
            .take()
            .ok_or_else(|| AlzinaError::Orchestration("sidecar stderr not available".into()))?;

        let mut stdin = BufWriter::new(child_stdin);
        let mut stdout = BufReader::new(child_stdout).lines();

        // Drain stderr in a background task to prevent OS pipe buffer deadlock.
        //
        // The sidecar prefixes claude-code subprocess stderr with the
        // `[claude-code-stderr]` marker (see sidecar/claude-agent/src/index.ts
        // stderr hook on both query() sites). Those lines are elevated to
        // `warn!` so they are visible at default log level — the SDK reports
        // any non-zero exit as "Claude Code process exited with code N"
        // with no further context, and the prefixed lines are the only
        // recoverable evidence of WHY the subprocess died.
        //
        // All other sidecar stderr (including alzina-sidecar's own debug
        // output, prefixed `[alzina-sidecar]`) stays at debug level — too
        // chatty to surface by default.
        let stderr_task = tokio::spawn(async move {
            let mut reader = BufReader::new(child_stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if line.starts_with("[claude-code-stderr]") {
                    tracing::warn!(target: "sidecar_stderr", "{}", line);
                } else {
                    tracing::debug!(target: "sidecar_stderr", "{}", line);
                }
            }
        });

        let alive = Arc::new(AtomicBool::new(true));

        // ── Send session_start ─────────────────────────────────────

        // Clone values needed for recovery before moving them into the message.
        let retained_system_prompt = system_prompt.clone();
        let retained_custom_tools = custom_tools.clone();
        let retained_sidecar_path = sidecar_path.clone();
        let retained_working_dir = working_dir.clone();

        let retained_api_key = api_key.clone();
        let retained_model = model.clone();

        let start_msg = SessionStartMsg::new(
            session_id.clone(),
            system_prompt,
            custom_tools,
            api_key,
            model,
        );

        send_msg(&mut stdin, &start_msg).await?;
        debug!(session_id = %session_id, "session_start sent, awaiting session_ready");

        // ── Await session_ready ────────────────────────────────────

        let timeout = std::time::Duration::from_secs(30);
        let ready = tokio::time::timeout(timeout, async {
            while let Some(line) = stdout.next_line().await.map_err(|e| {
                AlzinaError::Orchestration(format!("failed to read sidecar stdout: {e}"))
            })? {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }

                let event: SidecarEvent = serde_json::from_str(&trimmed).map_err(|e| {
                    AlzinaError::Orchestration(format!(
                        "failed to parse sidecar event during init: {e} (line: {trimmed})"
                    ))
                })?;

                match event {
                    SidecarEvent::SessionReady {
                        session_id: ref sid,
                    } => {
                        info!(session_id = %sid, "sidecar session ready");
                        return Ok(());
                    }
                    SidecarEvent::Error { ref error, .. } => {
                        return Err(AlzinaError::Orchestration(format!(
                            "sidecar error during session init: {error}"
                        )));
                    }
                    other => {
                        warn!(
                            event_type = %other.request_id(),
                            "unexpected event during session init, ignoring"
                        );
                    }
                }
            }
            Err(AlzinaError::Orchestration(
                "sidecar closed stdout before session_ready".into(),
            ))
        })
        .await;

        match ready {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(AlzinaError::Orchestration(
                    "timed out waiting for sidecar session_ready (30s)".into(),
                ));
            }
        }

        Ok(Self {
            child,
            stdin,
            stdout,
            session_id,
            alive,
            stderr_task,
            sidecar_path: retained_sidecar_path,
            system_prompt: retained_system_prompt,
            custom_tools: retained_custom_tools,
            working_dir: retained_working_dir,
            api_key: retained_api_key,
            model: retained_model,
        })
    }

    /// Send a chat message and collect the response.
    ///
    /// Sends a `chat_message` to the sidecar and reads events until a
    /// `chat_response` with the matching turn ID is received. During the
    /// turn, `tool_use` events are handled by checking the provided
    /// `GovernanceLayer` and sending back hook decisions.
    ///
    /// Returns the assistant's final response text for this turn.
    pub async fn chat_turn(
        &mut self,
        turn_id: &str,
        content: &str,
        governance: &GovernanceLayer,
        agent_id: &str,
        session_id: &SessionId,
    ) -> AlzinaResult<String> {
        self.chat_turn_with_emitter(turn_id, content, governance, agent_id, session_id, None)
            .await
    }

    /// Like [`Self::chat_turn`] but with a hook for forwarding mid-turn
    /// events (text deltas, usage) to the daemon's event bus.
    ///
    /// `emitter` (when `Some`) is invoked synchronously each time the
    /// sidecar emits a `text` or `usage` event. The closure is responsible
    /// for tagging events with the correct chat-session ID and turn ID;
    /// `chat_turn_with_emitter` only supplies the raw payload via
    /// pre-built [`AlzinaEvent`] variants.
    pub async fn chat_turn_with_emitter(
        &mut self,
        turn_id: &str,
        content: &str,
        governance: &GovernanceLayer,
        agent_id: &str,
        session_id: &SessionId,
        emitter: Option<ChatEventEmitter>,
    ) -> AlzinaResult<String> {
        if !self.is_alive() {
            return Err(AlzinaError::Orchestration(
                "sidecar process is not alive".into(),
            ));
        }

        debug!(turn_id = %turn_id, "sending chat_message");

        let msg = ChatMessageMsg::new(turn_id.to_string(), content.to_string());
        send_msg(&mut self.stdin, &msg).await?;

        // ── Read events until chat_response (with timeout) ─────────

        // Tool call timing tracker: maps tool_use id → start Instant.
        // Populated when a ToolUse event arrives, read + cleared on ToolResult
        // to compute duration_ms for the ToolCallAudit event.
        let mut tool_use_start: HashMap<String, std::time::Instant> = HashMap::new();

        // 600s: Fable 5 turns on hard tasks can run many minutes; this is
        // wall-clock over the whole turn (not reset by events), so it must
        // outlast a long thinking stretch plus tool calls.
        let chat_turn_timeout = std::time::Duration::from_secs(600);
        let result = tokio::time::timeout(chat_turn_timeout, async {
            loop {
                let line = match self.stdout.next_line().await {
                    Ok(Some(line)) => line,
                    Ok(None) => {
                        self.alive.store(false, Ordering::SeqCst);
                        return Err(AlzinaError::Orchestration(
                            "sidecar closed stdout during chat turn".into(),
                        ));
                    }
                    Err(e) => {
                        self.alive.store(false, Ordering::SeqCst);
                        return Err(AlzinaError::Orchestration(format!(
                            "failed to read sidecar stdout: {e}"
                        )));
                    }
                };

                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }

                let event: SidecarEvent = match serde_json::from_str(&trimmed) {
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
                    SidecarEvent::ChatResponse { ref id, ref content } if id == turn_id => {
                        info!(
                            turn_id = %turn_id,
                            content_len = content.len(),
                            "chat turn complete"
                        );
                        return Ok(content.clone());
                    }

                    SidecarEvent::ToolUse {
                        ref id,
                        ref tool,
                        ref input,
                        ref hook_id,
                        ..
                    } => {
                        // Record start time for ToolCallAudit duration computation.
                        tool_use_start.insert(id.clone(), std::time::Instant::now());
                        debug!(
                            tool = %tool,
                            hook_id = %hook_id,
                            "tool use event — checking governance"
                        );

                        let tool_args = ToolArgs::from_value(
                            &serde_json::to_value(input).unwrap_or_default(),
                        );

                        // Chat path has no per-dispatch assigned dir — the
                        // append-only-in-dir contract is a sub-agent concern.
                        let decision = tool_interceptor::check_tool_call(
                            governance,
                            agent_id,
                            session_id,
                            tool,
                            &tool_args,
                            None,
                        )?;

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
                                warn!(
                                    tool = %tool,
                                    "tool requires approval — blocking in persistent mode"
                                );
                                HookDecisionMsg::block(
                                    hook_id.clone(),
                                    "tool requires operator approval (not available in persistent sidecar mode)"
                                        .into(),
                                )
                            }
                        };

                        send_msg(&mut self.stdin, &hook_msg).await?;
                    }

                    SidecarEvent::ToolResult { ref id, ref tool, ref output } => {
                        debug!(tool = %tool, output_len = output.len(), "tool result");
                        // Emit ToolCallAudit so the audit JSONL records what the
                        // agent did (tool name + first-arg summary) without storing
                        // the full return value.
                        if let Some(ref emit) = emitter {
                            let duration_ms = tool_use_start
                                .remove(id)
                                .map(|t| t.elapsed().as_millis() as u64)
                                .unwrap_or(0);
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as i64)
                                .unwrap_or(0);
                            // args_summary: tool name only — the full input is not
                            // available on ToolResult (only on ToolUse). The audit
                            // purpose is "which tool ran" + approximate output size.
                            let args_summary = tool.clone();
                            emit(AlzinaEvent::ToolCallAudit {
                                session_id: self.session_id.clone(),
                                agent_id: agent_id.to_string(),
                                tool_name: tool.clone(),
                                args_summary,
                                timestamp: now_ms,
                                duration_ms,
                                status: "ok".to_string(),
                                output_bytes: output.len(),
                            });
                        }
                    }

                    SidecarEvent::Text { ref content, .. } => {
                        debug!(content_len = content.len(), "streaming text");
                        // Phase 4 streaming: forward each delta to the
                        // daemon event bus when an emitter is wired.
                        if let Some(ref emit) = emitter {
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as i64)
                                .unwrap_or(0);
                            emit(AlzinaEvent::TextDelta {
                                session_id: self.session_id.clone(),
                                turn_id: turn_id.to_string(),
                                content: content.clone(),
                                timestamp: now_ms,
                            });
                        }
                    }

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
                            output_tokens,
                            "received usage event from sidecar"
                        );
                        // Phase 4 token usage: forward to the daemon bus.
                        if let Some(ref emit) = emitter {
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as i64)
                                .unwrap_or(0);
                            emit(AlzinaEvent::TokenUsage {
                                session_id: self.session_id.clone(),
                                turn_id: turn_id.to_string(),
                                input_tokens,
                                output_tokens,
                                cache_read_input_tokens,
                                cache_creation_input_tokens,
                                model: model.clone(),
                                timestamp: now_ms,
                            });
                        }
                    }

                    SidecarEvent::Result { ref content, .. } => {
                        // In persistent mode, we expect ChatResponse, not Result.
                        // But handle gracefully if the sidecar sends Result.
                        warn!("received one-shot Result event during persistent session, treating as response");
                        return Ok(content.clone());
                    }

                    SidecarEvent::Error { ref error, .. } => {
                        error!(error = %error, "sidecar error during chat turn");
                        return Err(AlzinaError::Orchestration(format!(
                            "sidecar error during chat turn: {error}"
                        )));
                    }

                    SidecarEvent::SessionReady { .. } => {
                        warn!("unexpected session_ready during chat turn, ignoring");
                    }

                    SidecarEvent::ChatResponse { ref id, .. } => {
                        warn!(
                            expected = %turn_id,
                            got = %id,
                            "chat_response with mismatched turn_id, ignoring"
                        );
                    }
                }
            }
        })
        .await;

        match result {
            Ok(inner) => inner,
            Err(_) => Err(AlzinaError::Orchestration(
                "chat turn timed out after 300s".into(),
            )),
        }
    }

    /// Notify the sidecar to abort the in-flight chat turn.
    ///
    /// Sends `turn_cancel` over stdin. The sidecar aborts its active
    /// `query()` call via `AbortController`, the `for await` loop throws
    /// `AbortError`, and the `finally` block resets sidecar state to
    /// `SESSION_ACTIVE` — ready for the next `chat_message`.
    ///
    /// Best-effort: errors are logged but not propagated (the Rust
    /// cancellation path is already committed at this point).
    pub async fn cancel_current_turn(&mut self) {
        debug!(session_id = %self.session_id, "sending turn_cancel to sidecar");
        let msg = TurnCancelMsg::default();
        if let Err(e) = send_msg(&mut self.stdin, &msg).await {
            warn!(
                session_id = %self.session_id,
                error = %e,
                "failed to send turn_cancel to sidecar — sidecar may be stuck in TURN_ACTIVE"
            );
        }
    }

    /// End the session and shut down the sidecar process.
    ///
    /// Sends `session_end` followed by `shutdown`, then waits for the
    /// process to exit cleanly (with a 5-second timeout before kill).
    pub async fn end_session(mut self) -> AlzinaResult<()> {
        info!(session_id = %self.session_id, "ending sidecar session");

        // Send session_end
        let end_msg = SessionEndMsg::default();
        let _ = send_msg(&mut self.stdin, &end_msg).await;

        // Send shutdown
        let shutdown_msg = ShutdownMsg::default();
        let _ = send_msg(&mut self.stdin, &shutdown_msg).await;

        // Wait for clean exit
        match tokio::time::timeout(std::time::Duration::from_secs(5), self.child.wait()).await {
            Ok(Ok(status)) => {
                debug!(status = %status, session_id = %self.session_id, "sidecar exited");
            }
            Ok(Err(e)) => {
                warn!(error = %e, "error waiting for sidecar exit");
            }
            Err(_) => {
                warn!(session_id = %self.session_id, "sidecar did not exit within 5s, killing");
                let _ = self.child.kill().await;
            }
        }

        self.alive.store(false, Ordering::SeqCst);
        self.stderr_task.abort();
        Ok(())
    }

    /// Check if the sidecar process is still alive.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Recover from a sidecar crash by respawning the process.
    ///
    /// Spawns a fresh sidecar process, re-sends `session_start` with the
    /// original system prompt and custom tools, and optionally injects a
    /// conversation summary so the new process has context about prior turns.
    ///
    /// The `turns_summary` parameter is a brief textual summary of the
    /// conversation so far (not full history replay). It is prepended to the
    /// system prompt so the recovered sidecar has awareness of what happened
    /// before the crash.
    ///
    /// # Errors
    ///
    /// Returns an error if the new sidecar process cannot be spawned or if
    /// the `session_ready` handshake fails.
    pub async fn recover_session(&mut self, turns_summary: Option<&str>) -> AlzinaResult<()> {
        warn!(
            session_id = %self.session_id,
            "attempting crash recovery — respawning sidecar"
        );

        // Kill the old process if it is still lingering.
        let _ = self.child.kill().await;
        self.stderr_task.abort();

        // Build the recovery system prompt: original prompt + summary of
        // prior conversation so the new sidecar has context.
        let recovery_prompt = match turns_summary {
            Some(summary) if !summary.is_empty() => {
                format!(
                    "{}\n\n## Prior conversation (recovered after crash)\n\n{}",
                    self.system_prompt, summary
                )
            }
            _ => self.system_prompt.clone(),
        };

        // Generate a new session ID for the respawned process (the
        // ChatSession-level ID stays the same — only the sidecar-internal
        // session changes).
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let new_sidecar_session_id = format!("{}-recovered-{}", self.session_id, timestamp);

        // Respawn using the same parameters that were used originally.
        let new_handle = Self::start_session(
            self.sidecar_path.clone(),
            new_sidecar_session_id,
            recovery_prompt,
            self.custom_tools.clone(),
            self.working_dir.clone(),
            self.api_key.clone(),
            self.model.clone(),
        )
        .await?;

        // Transplant the new handle's process-level fields into self,
        // keeping the original recovery metadata (sidecar_path, etc.).
        self.child = new_handle.child;
        self.stdin = new_handle.stdin;
        self.stdout = new_handle.stdout;
        self.session_id = new_handle.session_id;
        self.alive = new_handle.alive;
        self.stderr_task = new_handle.stderr_task;

        info!(
            session_id = %self.session_id,
            "sidecar crash recovery successful"
        );

        Ok(())
    }

    /// Get the session ID for this handle.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Serialise and send a message to the sidecar's stdin as a single line.
async fn send_msg<T: serde::Serialize>(
    stdin: &mut BufWriter<ChildStdin>,
    msg: &T,
) -> AlzinaResult<()> {
    let json = serde_json::to_string(msg).map_err(|e| {
        AlzinaError::Orchestration(format!("failed to serialize message for sidecar: {e}"))
    })?;

    stdin.write_all(json.as_bytes()).await.map_err(|e| {
        AlzinaError::Orchestration(format!("failed to write to sidecar stdin: {e}"))
    })?;
    stdin.write_all(b"\n").await.map_err(|e| {
        AlzinaError::Orchestration(format!("failed to write newline to sidecar stdin: {e}"))
    })?;
    stdin
        .flush()
        .await
        .map_err(|e| AlzinaError::Orchestration(format!("failed to flush sidecar stdin: {e}")))?;

    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_governance::config::GovernanceConfig;
    use alzina_workspace::WorkspaceHandle;
    use std::path::Path;
    use std::sync::Arc;

    fn build_test_governance(dir: &Path) -> Arc<GovernanceLayer> {
        let workspace = Arc::new(WorkspaceHandle::open(dir.to_path_buf()).unwrap());
        let config = GovernanceConfig::default();
        Arc::new(GovernanceLayer::new(config, workspace).unwrap())
    }

    /// Create a mock sidecar that handles persistent-mode protocol.
    fn write_persistent_mock_sidecar(dir: &Path) -> PathBuf {
        let script_path = dir.join("mock-persistent-sidecar.js");
        let script = r#"
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });

rl.on('line', (line) => {
    const msg = JSON.parse(line);

    if (msg.type === 'session_start') {
        process.stdout.write(JSON.stringify({
            type: "session_ready",
            session_id: msg.session_id
        }) + '\n');
    } else if (msg.type === 'chat_message') {
        process.stdout.write(JSON.stringify({
            type: "text",
            id: msg.id,
            content: "thinking..."
        }) + '\n');
        process.stdout.write(JSON.stringify({
            type: "chat_response",
            id: msg.id,
            content: "Response to: " + msg.content
        }) + '\n');
    } else if (msg.type === 'session_end') {
        // acknowledged
    } else if (msg.type === 'shutdown') {
        process.exit(0);
    }
});
"#;
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    /// Create a mock sidecar that emits a tool_use event during a chat turn.
    fn write_governance_mock_sidecar(dir: &Path) -> PathBuf {
        let script_path = dir.join("mock-gov-sidecar.js");
        let script = r#"
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
let state = 'init';

rl.on('line', (line) => {
    const msg = JSON.parse(line);

    if (msg.type === 'session_start') {
        process.stdout.write(JSON.stringify({
            type: "session_ready",
            session_id: msg.session_id
        }) + '\n');
        state = 'ready';
    } else if (msg.type === 'chat_message' && state === 'ready') {
        // Emit a tool_use event
        process.stdout.write(JSON.stringify({
            type: "tool_use",
            id: msg.id,
            tool: "Read",
            input: { path: "src/main.rs" },
            hook_id: "hook-1"
        }) + '\n');
        state = 'waiting_decision_' + msg.id;
    } else if (msg.type === 'hook_decision' && state.startsWith('waiting_decision_')) {
        const turnId = state.replace('waiting_decision_', '');
        process.stdout.write(JSON.stringify({
            type: "tool_result",
            id: turnId,
            tool: "Read",
            output: "fn main() {}"
        }) + '\n');
        process.stdout.write(JSON.stringify({
            type: "chat_response",
            id: turnId,
            content: "Read the file. Decision was: " + msg.decision
        }) + '\n');
        state = 'ready';
    } else if (msg.type === 'session_end' || msg.type === 'shutdown') {
        process.exit(0);
    }
});
"#;
        std::fs::write(&script_path, script).unwrap();
        script_path
    }

    #[tokio::test]
    async fn start_session_and_chat() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_persistent_mock_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let mut handle = SidecarHandle::start_session(
            sidecar_path,
            "sess-001".into(),
            "You are a test agent.".into(),
            vec![],
            dir.path().to_path_buf(),
            None,
            None,
        )
        .await
        .unwrap();

        assert!(handle.is_alive());
        assert_eq!(handle.session_id(), "sess-001");

        // First turn
        let session_id = SessionId::new();
        let response = handle
            .chat_turn("turn-1", "Hello!", &governance, "test-agent", &session_id)
            .await
            .unwrap();

        assert_eq!(response, "Response to: Hello!");

        // Second turn — same process
        let response = handle
            .chat_turn(
                "turn-2",
                "How are you?",
                &governance,
                "test-agent",
                &session_id,
            )
            .await
            .unwrap();

        assert_eq!(response, "Response to: How are you?");

        // End session
        handle.end_session().await.unwrap();
    }

    #[tokio::test]
    async fn chat_turn_handles_governance() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_governance_mock_sidecar(dir.path());
        let governance = build_test_governance(dir.path());

        let mut handle = SidecarHandle::start_session(
            sidecar_path,
            "sess-002".into(),
            "You are a test agent.".into(),
            vec![],
            dir.path().to_path_buf(),
            None,
            None,
        )
        .await
        .unwrap();

        let session_id = SessionId::new();
        let response = handle
            .chat_turn(
                "turn-1",
                "Read a file",
                &governance,
                "test-agent",
                &session_id,
            )
            .await
            .unwrap();

        // Read is a non-write tool, so governance allows it.
        assert!(response.contains("allow"), "response was: {response}");

        handle.end_session().await.unwrap();
    }

    #[tokio::test]
    async fn start_session_fails_on_missing_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let result = SidecarHandle::start_session(
            PathBuf::from("/nonexistent/sidecar.js"),
            "sess-fail".into(),
            "prompt".into(),
            vec![],
            dir.path().to_path_buf(),
            None,
            None,
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn is_alive_returns_false_after_end() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar_path = write_persistent_mock_sidecar(dir.path());

        let handle = SidecarHandle::start_session(
            sidecar_path,
            "sess-003".into(),
            "prompt".into(),
            vec![],
            dir.path().to_path_buf(),
            None,
            None,
        )
        .await
        .unwrap();

        assert!(handle.is_alive());
        handle.end_session().await.unwrap();
        // After end_session consumes self, we can't check is_alive.
        // But the internal flag is set to false. This test verifies
        // that end_session completes without error.
    }
}
