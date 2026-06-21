//! `AgentExecutor` — the LLM execution seam.
//!
//! Extracted from [`crate::runner::alzina_runner`] so the TTD engine and the
//! composition layer can depend on the executor abstraction without pulling in
//! the governance-heavy `AlzinaRunner` (and its `alzina-governance` /
//! `alzina-memory` / `alzina-workspace` / `adk-rust` deps). The `runner`
//! feature gates the runner module; this module is always compiled.
//!
//! `crate::runner::alzina_runner` re-exports these types under their original
//! path, so existing imports continue to resolve when the `runner` feature is
//! enabled.

use async_trait::async_trait;

use alzina_core::identity::{AgentId, SessionId};
use alzina_core::{AlzinaEvent, AlzinaResult};

/// Per-spawn sampling parameters for LLM diversity (EXT-01 Phase 24).
///
/// Threaded from `TtdConfig` via `build_sampling_configs` → `CompNode.sampling`
/// → `AgentExecutor::execute_with_sampling` → `SidecarOptions` → TypeScript SDK
/// query().
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// Temperature for this trajectory (0.5–1.2 from runner.py:75).
    pub temperature: f32,
    /// top_p for this trajectory (0.8–1.0 from runner.py:75).
    pub top_p: f32,
    /// top_k for this trajectory (always 40 — runner.py:76).
    pub top_k: u32,
}

// ── ExecutorEventEmitter ────────────────────────────────────────────────────

/// Synchronous closure invoked by an [`AgentExecutor`] to publish bus events
/// emitted *during* a sub-agent's execution (streaming `TextDelta`,
/// `TokenUsage`, etc).
///
/// Mirrors `ChatEventEmitter` in `sidecar_handle.rs`. Kept in this crate so
/// `AlzinaRunner` does not depend on the daemon's `EventBus` directly.
pub type ExecutorEventEmitter = std::sync::Arc<dyn Fn(AlzinaEvent) + Send + Sync>;

// ── AgentExecutor trait ─────────────────────────────────────────────────────

/// Abstraction over LLM agent execution.
///
/// Real implementation wraps ADK-Rust's `LlmAgentBuilder` + `Runner`.
/// Mock implementation returns canned responses for testing.
///
/// This is the seam described in §7.3 of the architecture doc: if in-process
/// isolation proves insufficient, replace the executor without changing the
/// composition layer.
///
/// # Trust Contract
///
/// Implementations of this trait are a security-critical boundary. They **must**:
///
/// - Route all file-write operations through the tool interceptor or the
///   `WorkspaceHandle` tier system. Direct filesystem writes bypass governance.
/// - Not exfiltrate data outside the sanctioned output channels (return value,
///   state channels, learnings merge).
/// - Not construct or execute sub-processes that circumvent sandbox restrictions.
///
/// The runner cannot enforce these invariants on an untrusted executor from within
/// the same process. In production, use process-level isolation (sandbox) so the
/// executor physically cannot bypass filesystem restrictions. In testing, use
/// `MockExecutor` which is trivially safe.
///
/// Violating this contract constitutes a governance bypass.
#[async_trait]
pub trait AgentExecutor: Send + Sync {
    /// Execute an agent with the given instruction and model, returning raw output.
    async fn execute(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
    ) -> AlzinaResult<String>;

    /// Variant that lets callers receive bus events emitted DURING the run
    /// (TextDelta from streaming SDK partial messages, ToolUse, ToolResult,
    /// etc) tagged with the spawned `session_id`.
    ///
    /// Default impl just calls [`execute`](AgentExecutor::execute) and emits
    /// nothing — preserves backwards compatibility for executors that don't have
    /// an event stream (test mocks, stubs, etc).
    ///
    /// # P5-LIVENESS-INNER
    ///
    /// The streaming `dispatch_agent` handler in `api/dispatch.rs` resets
    /// its idle timer on every event matching `event_belongs_to_dispatch_tree`.
    /// Without mid-turn emission from the executor the bus is silent
    /// between `SessionSpawned` and `SessionCompleted`, so a productive
    /// multi-tool sub-agent run gets killed by `idle_timeout_ms`. Executors
    /// that drive a streaming sidecar should override this method and
    /// publish each `text` / `usage` event tagged with the spawn's
    /// `session_id`.
    async fn execute_with_emitter(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
        _session_id: &SessionId,
        _emitter: Option<ExecutorEventEmitter>,
    ) -> AlzinaResult<String> {
        self.execute(agent_id, instruction, model, task).await
    }

    /// Variant that returns both the raw agent output AND an optional
    /// typed `Envelope` captured via a backend-specific structured-return
    /// path (plan 260515-ndk).
    ///
    /// Default impl forwards to
    /// [`execute_with_emitter`](AgentExecutor::execute_with_emitter) and returns
    /// `(raw, None)` — backwards compatible with every existing
    /// `AgentExecutor` impl (mocks, stubs, future backends that haven't
    /// implemented the typed path yet). The runner's parse step then
    /// falls back to the existing strict-then-lenient prose parser
    /// unchanged.
    ///
    /// When an executor overrides this and returns `Some(env)`, the
    /// runner uses the typed envelope directly (skipping prose parse)
    /// and re-renders the canonical prose form from the typed value via
    /// `alzina_governance::envelope::parser::render_envelope_as_prose`
    /// so downstream consumers (audit_subscriber `SessionCompleted` arm,
    /// `prepare_envelope` in chat.rs, SSE `[low-authority source=...]`
    /// wrapper) see byte-identical output regardless of which branch
    /// produced the envelope.
    ///
    /// The `ClaudeAgentSdkExecutor` overrides this to intercept the
    /// `mcp__alzina__return_envelope` tool_use block — see Task 3 in
    /// plan 260515-ndk. Other backends register the same
    /// `return_envelope` `CustomToolDefinition` (`alzina_orchestration::
    /// runner::envelope_tool::return_envelope_tool`) with their wire
    /// protocol and capture the typed payload analogously.
    async fn execute_with_envelope(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
        session_id: &SessionId,
        emitter: Option<ExecutorEventEmitter>,
    ) -> AlzinaResult<(String, Option<alzina_core::Envelope>)> {
        let raw = self
            .execute_with_emitter(agent_id, instruction, model, task, session_id, emitter)
            .await?;
        Ok((raw, None))
    }

    /// Variant that forwards per-trajectory sampling params (temperature/top_p/top_k)
    /// through to the underlying LLM call (EXT-01 Phase 24).
    ///
    /// Default impl delegates to [`execute`](AgentExecutor::execute) and ignores
    /// `sampling` — preserves backwards compatibility for all existing executors
    /// and test mocks (Pitfall 3). `ClaudeAgentSdkExecutor` overrides this to
    /// populate `SidecarOptions` fields.
    ///
    /// Mirror of `execute_with_emitter` — same non-breaking default-impl pattern.
    async fn execute_with_sampling(
        &self,
        agent_id: &AgentId,
        instruction: &str,
        model: &str,
        task: &str,
        _sampling: Option<SamplingParams>,
    ) -> AlzinaResult<String> {
        self.execute(agent_id, instruction, model, task).await
    }
}
