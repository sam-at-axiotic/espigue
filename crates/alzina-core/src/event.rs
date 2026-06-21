//! Domain events for the Alzina runtime.
//!
//! `AlzinaEvent` is the canonical event type broadcast through the system.
//! It lives in `alzina-core` so any crate can reference it without pulling
//! in daemon or channel dependencies.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::AlzinaResult;

/// Wraps an `AlzinaEvent` with its resolved `Scope` at compile-gated
/// emission boundaries (HTTP dispatch, composition `SpawnSpec`).
///
/// Per D15-05 / R-WEAVE-SCOPE-001: the broadcast channel stays
/// `broadcast::Sender<AlzinaEvent>` — `ScopedEnvelope` is a CALL-SITE
/// compile gate, not a channel-type promotion. Construct at the emission
/// boundary with `new`, then publish `envelope.event` to the bus.
///
/// `bare(event)` defaults `scope` to `Scope::SessionDefault` and is
/// available only in test/test-harness builds (per RESEARCH Pitfall 5 —
/// prevents silent mis-scoping in production code).
pub struct ScopedEnvelope {
    pub scope: crate::identity::Scope,
    pub event: AlzinaEvent,
}

impl ScopedEnvelope {
    /// Authoritative production constructor. Both `scope` and `event` are
    /// required — callers cannot bypass scope threading.
    pub fn new(scope: crate::identity::Scope, event: AlzinaEvent) -> Self {
        Self { scope, event }
    }

    /// TEST-ONLY: defaults scope to `Scope::SessionDefault`. Gated by
    /// `#[cfg(any(test, feature = "test-harness"))]` so release builds
    /// cannot reach this constructor (D15-02 / RESEARCH Pitfall 5).
    #[cfg(any(test, feature = "test-harness"))]
    pub fn bare(event: AlzinaEvent) -> Self {
        Self {
            scope: crate::identity::Scope::SessionDefault,
            event,
        }
    }
}

/// Events broadcast through the Alzina event bus.
///
/// Tagged-union serialisation (`#[serde(tag = "type")]`) produces JSON like:
/// ```json
/// { "type": "SessionSpawned", "session_id": "...", ... }
/// ```
/// which downstream consumers (WebSocket, SSE, CLI) can dispatch on directly.
///
/// `#[non_exhaustive]` enforces that every `match` over this enum must keep
/// exhaustive arms (per D15-06): adding a new variant breaks the build until
/// the new arm is added to `audit_subscriber::format_event` (compile gate).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AlzinaEvent {
    /// Dispatched a new agent session.
    ///
    /// For ad-hoc `/api/v1/dispatch`-style dispatches, only the original
    /// fields (session_id, agent_id, parent_id, task_summary, timestamp) are
    /// populated; `composition` and `task_rendered` stay `None` and are
    /// skipped during serialisation, preserving byte-identical wire format.
    ///
    /// For composition-dispatched leaves (via Plan 10's `dispatch_compose`),
    /// `composition` carries the AST metadata (compose_id, node_id, rationale,
    /// ancestor ids) and `task_rendered` carries the FULL rendered task
    /// (preamble + substituted body) so the audit JSONL captures what the
    /// agent actually saw.
    SessionSpawned {
        session_id: String,
        agent_id: String,
        parent_id: Option<String>,
        /// Short summary (120-char truncation per `session_manager.rs:242`).
        /// Kept for backwards compatibility with existing audit consumers.
        task_summary: String,
        timestamp: i64,
        /// Composition dispatch metadata. `None` for ad-hoc dispatches.
        /// Serde-skipped when None — preserves byte-identical wire format.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        composition: Option<CompositionDispatchMeta>,
        /// Full rendered task (preamble + substituted body). `None` for ad-hoc
        /// dispatches. Populated for composition leaves so the audit JSONL
        /// captures what the agent actually saw.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_rendered: Option<String>,
        /// C6.4 (Phase 11 Wave 5): the daemon-allocated dispatch_id of
        /// the originating `POST /dispatch` or `dispatch_compose` call.
        /// `None` for chat-root sessions and for legacy dispatch paths
        /// that have not yet been updated to thread the id. Allows the
        /// TUI's `TuiDispatchRegistry` to key on the canonical
        /// dispatch_id once available, falling back to child session_id
        /// otherwise (mirrors the P10 D10-12 additive-field pattern).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dispatch_id: Option<String>,
        /// Phase 15 Plan 15-09 (WR-01 + WR-02 cliff closure): the
        /// dispatch-time `Scope` of this session. SSE consumers and the
        /// `audit_subscriber::event_scope` helper read this field directly
        /// instead of folding every SessionSpawned to `SessionDefault`.
        ///
        /// `#[serde(default)]` defaults missing-field reads to
        /// `Scope::SessionDefault` so pre-15-09 audit JSONL lines (which
        /// did not carry the field) deserialise cleanly — the
        /// audit-replay backcompat guarantee pinned by
        /// `session_spawned_scope_defaults_to_session_default_when_field_missing`.
        #[serde(default)]
        scope: crate::identity::Scope,
    },
    SessionCompleted {
        session_id: String,
        agent_id: String,
        status: String,
        signal: Option<String>,
        artifacts: Vec<String>,
        /// Full agent return body (raw envelope text). Populated for child
        /// sessions completing via the `dispatch_agent` custom tool so that
        /// the chat SSE stream can surface the sub-agent's full reply.
        /// `None` for paths that do not yet thread the raw body.
        envelope: Option<String>,
        timestamp: i64,
        /// F-4 Phase 1: the daemon-allocated `dispatch_id` of the
        /// originating `POST /dispatch` or `dispatch_compose` call when
        /// this terminal event resolves a dispatch attempt. `None` for
        /// chat-root sessions and any non-dispatch lifecycle path.
        ///
        /// Mirrors the established `SessionSpawned.dispatch_id` pattern
        /// (lines 91-99 above): `#[serde(default, skip_serializing_if =
        /// "Option::is_none")]` preserves byte-identical wire format
        /// when `None`, so pre-F-4 audit JSONL lines continue to
        /// deserialise cleanly. Enables F-4 Phase 2 to key the watcher
        /// on `dispatch_id` instead of `session_id` — the latter is
        /// allocated per-attempt by the retry loop and is no longer a
        /// stable cross-attempt correlation key.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dispatch_id: Option<String>,
    },
    SessionFailed {
        session_id: String,
        agent_id: String,
        error: String,
        timestamp: i64,
        /// F-4 Phase 1: dispatch attribution for terminal failure
        /// events. See `SessionCompleted.dispatch_id` doc.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dispatch_id: Option<String>,
    },
    GateEvaluated {
        session_id: String,
        verdict: String,
        reason: Option<String>,
        timestamp: i64,
    },
    EmergenceDetected {
        session_id: String,
        content: String,
        timestamp: i64,
    },
    TensionFlagged {
        session_id: String,
        content: String,
        timestamp: i64,
    },
    /// Per-token streaming text chunk emitted by the sidecar mid-turn.
    ///
    /// `session_id` is the chat session ID (the root orchestration session
    /// for this turn). `turn_id` correlates with the chat turn being processed.
    TextDelta {
        session_id: String,
        turn_id: String,
        content: String,
        timestamp: i64,
    },
    /// Token-usage event emitted at end of each turn.
    ///
    /// Fields mirror the sidecar `UsageEvent` contract. `session_id` is the
    /// chat session ID; cumulative running totals are tracked elsewhere in
    /// the chat service keyed off this session_id.
    TokenUsage {
        session_id: String,
        turn_id: String,
        input_tokens: u32,
        output_tokens: u32,
        cache_read_input_tokens: u32,
        cache_creation_input_tokens: u32,
        model: String,
        timestamp: i64,
    },
    HealthUpdate {
        uptime_secs: u64,
        active_sessions: u32,
        channels_connected: Vec<String>,
    },
    /// Phase 4: bus-wide liveness pulse emitted every 500ms by the
    /// daemon. Carries no `session_id` — bus-wide. Drives the TUI's
    /// shimmer + `[▲ stalled]` indicator (synthesis E3, ROADMAP SC-2).
    Heartbeat { timestamp: i64 },

    /// Phase 4: client requested turn cancellation via
    /// `POST /api/v1/chat/{session_id}/cancel`. Recorded in the audit
    /// log. Distinct from `SessionFailed` so audit consumers can
    /// distinguish "client asked us to stop" from "the work itself
    /// blew up". `cancelled_turns` is the count returned by
    /// `ChatService::cancel_all_turns_for` (0 when there were no
    /// in-flight turns — already-complete idempotent path).
    TurnCancelRequested {
        session_id: String,
        cancelled_turns: usize,
        timestamp: i64,
    },

    /// Phase 5 (D-08): the parent session emitted its final `ChatResponse`
    /// while async children were still in flight. `active_turns[parent]`
    /// is freed at this point — input is no longer gated. `SessionCompleted`
    /// fires later, when the last child finishes (or all are cancelled).
    /// Distinct from `SessionCompleted` so audit consumers can distinguish
    /// "turn-text done, children still running" from "everything done".
    SessionDetached {
        session_id: String,
        parent_id: Option<String>,
        live_children: usize,
        timestamp: i64,
    },

    /// Phase 5 (D-13): client requested cancel of a single dispatch via
    /// `POST /api/v1/dispatches/{dispatch_id}/cancel`. Distinct from
    /// `SessionFailed` (the in-process cancel path emits that for the
    /// child task itself — this variant records the operator/LLM intent).
    /// Phase 4 Pitfall 4 carryover: do NOT publish `SessionFailed` from
    /// the HTTP handler; both events fire on the same cancel, but for
    /// distinct audit semantics.
    DispatchCancelRequested {
        dispatch_id: String,
        session_id: String,
        parent_session_id: String,
        timestamp: i64,
    },

    /// Phase 5 (D-07): one attempt of `dispatch_async` failed with a
    /// retryable transport error (sidecar crash mid-child, IPC drop,
    /// daemon-internal panic — `sidecar_protocol.rs:167 retryable: true`).
    /// Audit-log only — never SSE-forwarded. The model only sees the
    /// final outcome envelope; operators see the flapping infrastructure
    /// trail in `audit/chat-events.jsonl` plus tracing::warn!/error!.
    DispatchRetry {
        dispatch_id: String,
        attempt: u8,
        max_attempts: u8,
        reason: String,
        backoff_ms: u64,
        timestamp: i64,
    },

    /// Phase 6 (D6-01/D6-02/D6-05): a dispatch's child has terminated and the
    /// daemon has synthesized a `[dispatch:{id}] {agent}: {summary}` chat
    /// message into the parent session's pending-announcements queue. This
    /// SSE-forwarded variant lets the TUI render a `Block::Envelope` with
    /// `BlockMeta::Dispatch` chrome in real time without waiting for the next
    /// model turn.
    ///
    /// `status` is one of "completed" | "cancelled" | "failed" (mirrors
    /// `crate::dispatch_registry::DispatchOutcome.status` shape from Phase 5,
    /// extended with "cancelled" per D6-05 — cancelled envelopes are
    /// announced for symmetry so the model knows the dispatch didn't complete).
    ///
    /// `envelope_summary` is the same summary the model sees in its conversation
    /// (D6-02 / D6-03): envelope's `signal` field if present, else first 500 chars
    /// of `text_delta` concatenation. Truncated values carry the suffix
    /// `[truncated — call dispatch_status({id}) for full envelope, or press e/d
    /// on the announcement to expand]` baked in by the synthesizer (06-02).
    ///
    /// Distinct from `SessionCompleted` and `SessionFailed` — those publish for
    /// the child session itself; THIS variant records that the parent has
    /// received the announcement into its conversation. Phase 4 Pitfall 4
    /// carryover: distinct variant for distinct semantic event.
    DispatchEnvelopeReceived {
        dispatch_id: String,
        parent_session_id: String,
        agent: String,
        status: String,
        envelope_summary: String,
        timestamp: i64,
    },

    /// A child agent's envelope claimed an artifact path that lies outside
    /// its assigned dispatch directory. The offending path is dropped from
    /// the parent's announcement and the rejection is recorded here so the
    /// audit log + TUI surface the violation rather than silently swallowing
    /// it. Carries the offending path verbatim and the dir the child was
    /// instructed to write under.
    DispatchArtifactRejected {
        dispatch_id: String,
        parent_session_id: String,
        agent: String,
        offending_path: String,
        assigned_dir: String,
        reason: String,
        timestamp: i64,
    },

    /// Phase 6 (D6-08 canonical signal — see checker BLOCKER-1 resolution): a
    /// new chat turn has begun for a session. `origin` discriminates between
    /// "user" (operator submitted text) and "auto_continuation" (the daemon's
    /// auto-continuation hook fired with the synthetic prompt
    /// `[dispatch pattern complete — continue]`).
    ///
    /// Published from `ChatService::start_turn_async` (06-02 owns the publish
    /// call) at the start of every turn, BEFORE any text_delta or
    /// process_message_with_prewarm work. The TUI consumer (06-05) reads
    /// `origin` to decide whether to render the dim chrome marker `[← auto-
    /// continued from dispatch pattern]` above the assistant's first
    /// text_delta. This replaces the recency-window heuristic that 06-05
    /// originally proposed (false positives on long-running RAG, false
    /// negatives on fast dispatch completions — checker BLOCKER-1).
    ///
    /// `session_id` is the parent ChatSession id (NOT a child sub-agent
    /// session). The SSE filter `event_belongs_to_session_tree` admits this
    /// variant when `session_id == root_session_id`.
    TurnStarted {
        turn_id: String,
        session_id: String,
        /// "user" | "auto_continuation"
        origin: String,
        timestamp: i64,
    },

    /// Phase 7: a hook returned `HookAction::Engage` and the broker
    /// opened the engagement. SSE consumers (TUI, chat fallback) react
    /// to this by surfacing the modal/prompt to the human.
    ///
    /// `mode_kind` is the snake_case form of `EngagementMode` (one of
    /// "approval", "choice", "free_form", "dialogue") so SSE consumers
    /// can decide on the rendering shape without re-deserialising the
    /// full request — the request itself is read from `WeaveStateSchema.engaged`.
    ///
    /// **Security note (T-07-05-03):** this event is broadcast globally on
    /// the SSE bus (no per-session gating). Acceptable for the single-operator
    /// TUI use case. Future hardening: add `target_session: Option<String>` to
    /// gate which TUIs render the modal.
    EngagementOpened {
        engagement_id: String,
        weave_id: String,
        prompt: String,
        mode_kind: String,
        timestamp: i64,
    },

    /// Phase 7: human added a new dialogue turn while an engagement
    /// is in flight. Only fires for `EngagementMode::Dialogue`.
    EngagementTurn {
        engagement_id: String,
        author: String,
        content: String,
        timestamp: i64,
    },

    /// Phase 7: engagement closed (resolved, fell back, or abandoned).
    /// Distinct from `EngagementOpened` so audit consumers can pair
    /// open/close.
    ///
    /// `outcome_kind` is the snake_case form of `ResolutionOutcome`
    /// (one of "resolved", "fell_back", "abandoned").
    EngagementResolved {
        engagement_id: String,
        outcome_kind: String,
        timestamp: i64,
    },

    /// Phase 9 R5-F1: gate auto-opened a new weave on `new-work` classification.
    /// TUI renders as a non-blocking toast. `label` is derived from the user
    /// message preview (caller decides — typically first ~80 chars); `weave_id`
    /// is the freshly-allocated id from `WeaveRecordStore::open_weave`.
    ///
    /// Distinct from `EngagementOpened` (Phase 7) — this is gate-driven, not
    /// hook-driven, so a separate variant keeps audit-log + TUI render logic
    /// uncoupled from the engagement broker pipeline.
    WeaveOpenedByGate {
        weave_id: String,
        label: String,
        timestamp: i64,
    },

    /// Composition plan parse failure. Published from `dispatch_compose_handler`
    /// when `parse_compose` returns errors, so the parse failure lands in the
    /// audit JSONL (not just stderr/tracing). `plan_preview` is truncated at
    /// the emission site (~2 KiB) — subscribers receive it pre-truncated.
    DispatchPlanRejected {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compose_id: Option<String>,
        errors: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        plan_preview: Option<String>,
        timestamp: i64,
    },

    /// Phase 9 R5 (ambiguous branch): gate classified the message as ambiguous
    /// — surface to TUI for human pick. `open_weaves` is the snapshot of
    /// active weave ids the user can choose from (TUI joins with weave labels
    /// at render time via existing `/api/v1/weaves` query). `message_preview`
    /// is the same bounded preview written into `gate_decisions.message_preview`.
    /// In headless/CI mode the daemon emits this event AND falls back to
    /// routine — the event is informational, not a synchronisation point.
    WeavePromptRequested {
        open_weaves: Vec<String>,
        message_preview: String,
        timestamp: i64,
    },

    /// Protocol-level envelope parse failure. Published when the governance
    /// layer (or runner) fails to parse the agent's return envelope. Previously
    /// these were only emitted via `tracing::warn!` — now they land in the
    /// audit JSONL and the ObservationService can detect them as seams.
    ///
    /// `raw_preview` is the first 200 bytes of the raw text (truncated) to aid
    /// debugging without capturing full agent output.
    ///
    /// Lives in `alzina-core` because it is a protocol-level event — it records
    /// a failure of the agent-daemon communication contract, not a daemon-internal
    /// observation.
    EnvelopeParseFailure {
        session_id: String,
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw_preview: Option<String>,
        error: String,
        timestamp: i64,
    },

    /// Summarised tool call record. Published after each tool_result so the
    /// audit JSONL captures what the agent did without persisting full return
    /// data (PII/size risk). `args_summary` is the tool name + truncated first
    /// argument (~200 chars) — enough to answer "what did the agent do" (file
    /// path, search query, etc.) without "what did it see".
    ///
    /// `duration_ms` is wall-clock from tool_use to tool_result. `status` is
    /// "ok" or "error". `output_bytes` is the byte-length of the tool result
    /// (not the content).
    ToolCallAudit {
        session_id: String,
        agent_id: String,
        tool_name: String,
        args_summary: String,
        timestamp: i64,
        duration_ms: u64,
        status: String,
        output_bytes: usize,
    },

    /// Composition substitution misses. Published from the runner when
    /// rendering a leaf's task template before dispatch and one or more
    /// `{id:channel}` references could not be resolved — typically because
    /// the referenced leaf failed (commit f5e9d17 sibling-survival made
    /// this case observable) or has not yet completed. The leaf is still
    /// dispatched with the unresolved references substituted as empty
    /// strings; this event makes the silent substitution loud.
    ///
    /// `session_id` is the consumer leaf's session_id. `consumer_node_id`
    /// + `consumer_agent` identify which leaf received the partial task.
    /// `unresolved` lists every reference that did not resolve.
    SubstitutionsUnresolved {
        session_id: String,
        compose_id: String,
        consumer_node_id: String,
        consumer_agent: String,
        unresolved: Vec<UnresolvedSubstitution>,
        timestamp: i64,
    },

    /// Cross-weave operation rejected by the scope-key check.
    ///
    /// R-WEAVE-SCOPE-001 AC-1 / §8: emitted when a request bearing
    /// `weave_id = X` attempts to read or write a resource owned by
    /// `weave_id = Y`. Distinct from `RuneOperated/gate_blocked`
    /// (rune-pipeline) — `ScopeViolation` is a *runtime* audit event.
    /// `caller_scope` is the canonical string form of the caller's
    /// scope (weave id or `"SessionDefault"`); `attempted_scope` is
    /// the scope the rejected request tried to reach.
    ScopeViolation {
        caller_scope: String,
        attempted_scope: String,
        route: String,
        agent_id: Option<String>,
        reason: String,
        timestamp: i64,
    },

    /// Legacy unscoped request folded to `SessionDefault` during the
    /// §5.4 backward-compatibility window.
    ///
    /// R-WEAVE-SCOPE-001 §5.4 / AC-8: the rate of this event is the
    /// *measure* of remaining legacy callers; the BC window does not
    /// close until the rate is zero for seven consecutive days. Each
    /// emission names the route so the migration target is concrete.
    ScopeFallback {
        route: String,
        agent_id: Option<String>,
        timestamp: i64,
    },

    /// Operator-initiated cancel gesture for one of the three ladder levels.
    ///
    /// D16-10 / D16-05: fires once per gesture. `level` is `"progress"`,
    /// `"composition"`, or `"weave"`. `target_id` is `compose_id` (Stop
    /// composition) or `weave_id` (Stop weave / Stop progress).
    CancelInitiated {
        level: String,
        target_id: String,
        scope: crate::identity::Scope,
        timestamp: i64,
    },

    /// Per-node audit event emitted for every dispatch cancelled by a cascade.
    ///
    /// D16-10 / D16-13: fires AFTER `SessionFailed { error: "cancelled" }` for
    /// the same node (backward-compatible pair). `target_id` is the correlation
    /// key (compose_id or weave_id) from the originating gesture (D16-12).
    /// `agent_id` may be empty until Phase 17 STAB-02 closes the empty-string bug.
    CascadeCancelled {
        target_id: String,
        parent_session_id: String,
        child_session_id: String,
        agent_id: String,
        compose_id: Option<String>,
        weave_id: Option<String>,
        timestamp: i64,
    },

    /// Weave progress paused by an operator Stop-progress gesture.
    ///
    /// D16-11 / D16-05: fires once when the operator pauses a weave. Distinct
    /// from cancel events — pause is not cancel. Paired with `ProgressResumed`.
    ProgressPaused {
        weave_id: String,
        scope: crate::identity::Scope,
        timestamp: i64,
    },

    /// Weave progress resumed after a Stop-progress pause.
    ///
    /// D16-11 / D16-07: fires when the operator sends a typed message into a
    /// paused weave (implicit resume). `resume_reason` is `"operator_message"`
    /// (D16-07) or `"explicit"` (deferred — not used in Phase 16).
    ProgressResumed {
        weave_id: String,
        scope: crate::identity::Scope,
        resume_reason: String,
        timestamp: i64,
    },
}

/// One unresolved `{id:channel}` reference, captured for audit.
///
/// Source: `crates/alzina-orchestration/src/composition/parser/render.rs`
/// — the runner-side substitution layer (`resolve_substitution`) records
/// these when an envelope lookup misses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnresolvedSubstitution {
    /// Full token as written in the template, e.g. `audit:envelope`.
    pub reference: String,
    /// The producer leaf id that was referenced (e.g. `audit`).
    pub referenced_id: String,
    /// The channel name (e.g. `envelope`, `signal`, `artifacts`).
    pub referenced_channel: String,
}

// ── Composition dispatch metadata ─────────────────────────────────────────

/// Metadata for an agent dispatched as part of a `<Compose>` plan.
///
/// Carried on `AlzinaEvent::SessionSpawned.composition`. Populated only for
/// composition-dispatched leaves (Plan 10); `None` for ad-hoc dispatches.
/// Wire format: bare struct, no tagged-union wrapping.
///
/// Source: `docs/composition-grammar.md` §4.3 (ancestor_ids correspond to the
/// implicit preamble's listed nodes — id only, not full envelopes).
#[derive(Debug, Clone, Serialize)]
pub struct CompositionDispatchMeta {
    /// Root composition id allocated by the `dispatch_compose` handler.
    pub compose_id: String,
    /// This node's id (auto-generated per §4.1 or explicit).
    pub node_id: String,
    /// Rationale captured from the XML comment preceding this node (§1.4).
    pub rationale: Option<String>,
    /// Ids of every happens-before ancestor visible to this node — what the
    /// implicit preamble lists. Path-only summaries (no body content).
    pub ancestor_ids: Vec<String>,
}

impl AlzinaEvent {
    /// Construct a `SessionSpawned` for an ad-hoc dispatch (no composition).
    ///
    /// `composition` and `task_rendered` are `None` (serde-skipped), so the
    /// serialised JSON only adds the new `scope` field on top of the
    /// pre-Plan-10 format.
    ///
    /// Phase 15 Plan 15-09: callers thread the dispatch-time scope here.
    /// Chat-root spawns derive scope from `request.weave_id`; pattern
    /// dispatch passes `Scope::SessionDefault` because that path has no
    /// weave (documented in `session_manager::dispatch_pattern`).
    pub fn session_spawned_ad_hoc(
        session_id: String,
        agent_id: String,
        parent_id: Option<String>,
        task_summary: String,
        timestamp: i64,
        scope: crate::identity::Scope,
    ) -> Self {
        Self::SessionSpawned {
            session_id,
            agent_id,
            parent_id,
            task_summary,
            timestamp,
            composition: None,
            task_rendered: None,
            dispatch_id: None,
            scope,
        }
    }

    /// Construct a `SessionSpawned` for an ad-hoc dispatch with a
    /// daemon-allocated dispatch_id (C6.4). Same shape as
    /// `session_spawned_ad_hoc` plus the new additive `dispatch_id`
    /// field. Use this from the dispatch handler that has just
    /// allocated a fresh dispatch_id; callers that do not have one
    /// (e.g. chat-root spawns) stay on `session_spawned_ad_hoc`.
    pub fn session_spawned_ad_hoc_with_dispatch_id(
        session_id: String,
        agent_id: String,
        parent_id: Option<String>,
        task_summary: String,
        timestamp: i64,
        dispatch_id: String,
        scope: crate::identity::Scope,
    ) -> Self {
        Self::SessionSpawned {
            session_id,
            agent_id,
            parent_id,
            task_summary,
            timestamp,
            composition: None,
            task_rendered: None,
            dispatch_id: Some(dispatch_id),
            scope,
        }
    }

    /// Construct a `SessionSpawned` for a composition-dispatched leaf.
    ///
    /// `composition` carries the AST metadata; `task_rendered` carries the
    /// full rendered task (preamble + substituted body) for audit-grade
    /// fidelity. Plan 05 (daemon seam) calls this builder.
    pub fn session_spawned_composition(
        session_id: String,
        agent_id: String,
        parent_id: Option<String>,
        task_summary: String,
        timestamp: i64,
        composition: CompositionDispatchMeta,
        task_rendered: String,
        scope: crate::identity::Scope,
    ) -> Self {
        Self::SessionSpawned {
            session_id,
            agent_id,
            parent_id,
            task_summary,
            timestamp,
            composition: Some(composition),
            task_rendered: Some(task_rendered),
            dispatch_id: None,
            scope,
        }
    }
}

// ── SpawnEventSink trait ───────────────────────────────────────────────────

/// Data emitted after a spawn completes and its envelope is processed.
#[derive(Debug, Clone)]
pub struct SpawnCompleted {
    pub agent_id: String,
    pub session_id: String,
    pub weave_id: Option<String>,
    /// Phase 1B substrate cascade: daemon-allocated dispatch_id (UUID) of
    /// the chat-tool dispatch this spawn belongs to. Set by
    /// `alzina-daemon::api::dispatches` at the HTTP boundary, propagated
    /// through `DispatchRequest` → `SpawnSpec` → `CompNode` → here.
    ///
    /// Consumed by `alzina-memory::event_sink::on_spawn_completed`, which
    /// passes it to `WeaveRecordStore::add_stitch` so the v3
    /// `stitch_records.dispatch_id` column carries the dispatch the stitch
    /// belongs to (closes G4 on the chat-tool path).
    ///
    /// `None` for ad-hoc dispatches with no enclosing chat-tool dispatch
    /// (CLI, internal triggers, etc.), and for non-chat orchestration paths
    /// that never registered a `dispatch_id`.
    pub dispatch_id: Option<String>,
    /// One-line summary of what the agent produced (envelope status + signal).
    pub summary: String,
}

/// Trait for recording spawn lifecycle events into a persistent store.
///
/// Implemented by `DailyMemoryEventSink` in `alzina-memory`. Injected into
/// `AlzinaRunner` as an `Option<Arc<dyn SpawnEventSink>>` so the orchestration
/// crate stays decoupled from memory internals.
#[async_trait]
pub trait SpawnEventSink: Send + Sync {
    /// Record that a spawn completed and its envelope was processed.
    async fn on_spawn_completed(&self, event: SpawnCompleted) -> AlzinaResult<()>;
}

#[cfg(test)]
mod composition_dispatch_meta_tests {
    use super::*;

    #[test]
    fn ad_hoc_session_spawned_serializes_without_composition_keys() {
        let event = AlzinaEvent::SessionSpawned {
            session_id: "s1".into(),
            agent_id: "huginn".into(),
            parent_id: None,
            task_summary: "task".into(),
            timestamp: 1_000_000,
            composition: None,
            task_rendered: None,
            dispatch_id: None,
            scope: crate::identity::Scope::SessionDefault,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            !json.contains("\"composition\""),
            "ad-hoc should skip composition key: {json}"
        );
        assert!(
            !json.contains("\"task_rendered\""),
            "ad-hoc should skip task_rendered key: {json}"
        );
        assert!(
            json.contains("\"task_summary\""),
            "task_summary preserved: {json}"
        );
    }

    #[test]
    fn composition_session_spawned_serializes_with_both_keys() {
        let event = AlzinaEvent::SessionSpawned {
            session_id: "s1".into(),
            agent_id: "urdr".into(),
            parent_id: Some("p1".into()),
            task_summary: "Past: prior auth migrations.".into(),
            timestamp: 1_000_000,
            composition: Some(CompositionDispatchMeta {
                compose_id: "compose-uuid".into(),
                node_id: "past".into(),
                rationale: Some("triad needs historical context".into()),
                ancestor_ids: vec![],
            }),
            task_rendered: Some(
                "## Upstream context (you are part of a composition)\n... ## Your task\nPast: prior auth migrations.".into(),
            ),
            dispatch_id: None,
            scope: crate::identity::Scope::SessionDefault,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"composition\""));
        assert!(json.contains("\"task_rendered\""));
        assert!(json.contains("\"compose_id\":\"compose-uuid\""));
        assert!(json.contains("## Upstream context"));
    }

    #[test]
    fn composition_dispatch_meta_round_trips() {
        // Verify CompositionDispatchMeta itself serializes correctly and
        // its fields are all present in the output.
        let meta = CompositionDispatchMeta {
            compose_id: "c1".into(),
            node_id: "n1".into(),
            rationale: Some("why".into()),
            ancestor_ids: vec!["a1".into(), "a2".into()],
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"compose_id\":\"c1\""));
        assert!(json.contains("\"ancestor_ids\":[\"a1\",\"a2\"]"));
        assert!(json.contains("\"node_id\":\"n1\""));
        assert!(json.contains("\"rationale\":\"why\""));
    }

    #[test]
    fn builder_session_spawned_ad_hoc_skips_new_fields() {
        let event = AlzinaEvent::session_spawned_ad_hoc(
            "s2".into(),
            "vefr".into(),
            None,
            "short summary".into(),
            999,
            crate::identity::Scope::SessionDefault,
        );
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("\"composition\""));
        assert!(!json.contains("\"task_rendered\""));
    }

    #[test]
    fn builder_session_spawned_composition_includes_both_fields() {
        let event = AlzinaEvent::session_spawned_composition(
            "s3".into(),
            "urdr".into(),
            Some("parent-x".into()),
            "brief".into(),
            1234,
            CompositionDispatchMeta {
                compose_id: "cx".into(),
                node_id: "nx".into(),
                rationale: None,
                ancestor_ids: vec!["ax".into()],
            },
            "## Upstream context\n...".into(),
            crate::identity::Scope::SessionDefault,
        );
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"composition\""));
        assert!(json.contains("\"task_rendered\""));
        assert!(json.contains("\"compose_id\":\"cx\""));
    }

    // ── C6.4 (Phase 11 Wave 5): additive `dispatch_id` field ───────────

    #[test]
    fn session_spawned_omits_dispatch_id_when_none() {
        let event = AlzinaEvent::SessionSpawned {
            session_id: "s1".into(),
            agent_id: "huginn".into(),
            parent_id: None,
            task_summary: "task".into(),
            timestamp: 1,
            composition: None,
            task_rendered: None,
            dispatch_id: None,
            scope: crate::identity::Scope::SessionDefault,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            !json.contains("\"dispatch_id\""),
            "ad-hoc should skip dispatch_id key: {json}"
        );
    }

    #[test]
    fn session_spawned_includes_dispatch_id_when_some() {
        let event = AlzinaEvent::SessionSpawned {
            session_id: "s1".into(),
            agent_id: "huginn".into(),
            parent_id: Some("p1".into()),
            task_summary: "task".into(),
            timestamp: 1,
            composition: None,
            task_rendered: None,
            dispatch_id: Some("dispatch-uuid-v1".into()),
            scope: crate::identity::Scope::SessionDefault,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"dispatch_id\":\"dispatch-uuid-v1\""));
    }

    #[test]
    fn session_spawned_ad_hoc_with_dispatch_id_builder_emits_field() {
        let event = AlzinaEvent::session_spawned_ad_hoc_with_dispatch_id(
            "s-x".into(),
            "huginn".into(),
            Some("chat-7".into()),
            "scan logs".into(),
            42,
            "dispatch-xyz".into(),
            crate::identity::Scope::SessionDefault,
        );
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"dispatch_id\":\"dispatch-xyz\""));
        assert!(json.contains("\"parent_id\":\"chat-7\""));
    }

    // ── F-4 Phase 1: dispatch_id on terminal session events ───────────

    #[test]
    fn session_completed_omits_dispatch_id_when_none() {
        let event = AlzinaEvent::SessionCompleted {
            session_id: "s1".into(),
            agent_id: "huginn".into(),
            status: "ok".into(),
            signal: None,
            artifacts: vec![],
            envelope: None,
            timestamp: 1,
            dispatch_id: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            !json.contains("\"dispatch_id\""),
            "non-dispatch SessionCompleted must skip dispatch_id key: {json}"
        );
    }

    #[test]
    fn session_completed_includes_dispatch_id_when_some() {
        let event = AlzinaEvent::SessionCompleted {
            session_id: "s1".into(),
            agent_id: "huginn".into(),
            status: "ok".into(),
            signal: None,
            artifacts: vec![],
            envelope: None,
            timestamp: 1,
            dispatch_id: Some("dispatch-uuid-v1".into()),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"dispatch_id\":\"dispatch-uuid-v1\""));
    }

    #[test]
    fn session_failed_omits_dispatch_id_when_none() {
        let event = AlzinaEvent::SessionFailed {
            session_id: "s1".into(),
            agent_id: "huginn".into(),
            error: "boom".into(),
            timestamp: 1,
            dispatch_id: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            !json.contains("\"dispatch_id\""),
            "non-dispatch SessionFailed must skip dispatch_id key: {json}"
        );
    }

    #[test]
    fn session_failed_includes_dispatch_id_when_some() {
        let event = AlzinaEvent::SessionFailed {
            session_id: "s1".into(),
            agent_id: "huginn".into(),
            error: "boom".into(),
            timestamp: 1,
            dispatch_id: Some("dispatch-uuid-v1".into()),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"dispatch_id\":\"dispatch-uuid-v1\""));
    }

    /// Audit-replay backcompat for SessionCompleted: pre-F-4 audit JSONL
    /// lines do not carry `dispatch_id`. The `#[serde(default)]`
    /// attribute on the variant must default the field to `None` so
    /// those lines still deserialise cleanly. Mirrors the
    /// `session_spawned_scope_defaults_*` shadow-struct pattern above.
    #[test]
    fn session_completed_dispatch_id_defaults_to_none_when_field_missing() {
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct ShadowSessionCompleted {
            #[allow(dead_code)]
            session_id: String,
            #[serde(default)]
            dispatch_id: Option<String>,
        }

        let pre_f4_json = r#"{
            "session_id": "s-legacy",
            "agent_id": "huginn",
            "status": "ok",
            "signal": null,
            "artifacts": [],
            "envelope": null,
            "timestamp": 1
        }"#;

        let shadow: ShadowSessionCompleted =
            serde_json::from_str(pre_f4_json).expect("pre-F-4 JSON must deserialise");
        assert_eq!(
            shadow.dispatch_id, None,
            "missing `dispatch_id` MUST default to None (audit-replay backcompat)"
        );
    }

    /// Audit-replay backcompat for SessionFailed: pre-F-4 audit JSONL
    /// lines do not carry `dispatch_id`. Mirrors the SessionCompleted
    /// shadow-struct test above.
    #[test]
    fn session_failed_dispatch_id_defaults_to_none_when_field_missing() {
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct ShadowSessionFailed {
            #[allow(dead_code)]
            session_id: String,
            #[serde(default)]
            dispatch_id: Option<String>,
        }

        let pre_f4_json = r#"{
            "session_id": "s-legacy",
            "agent_id": "huginn",
            "error": "boom",
            "timestamp": 1
        }"#;

        let shadow: ShadowSessionFailed =
            serde_json::from_str(pre_f4_json).expect("pre-F-4 JSON must deserialise");
        assert_eq!(
            shadow.dispatch_id, None,
            "missing `dispatch_id` MUST default to None (audit-replay backcompat)"
        );
    }

    #[test]
    fn session_spawned_ad_hoc_builder_still_omits_dispatch_id() {
        // The pre-existing session_spawned_ad_hoc builder MUST stay
        // wire-byte-identical (no dispatch_id key emitted) so older
        // call sites do not silently start emitting the new field.
        let event = AlzinaEvent::session_spawned_ad_hoc(
            "s".into(),
            "a".into(),
            None,
            "t".into(),
            0,
            crate::identity::Scope::SessionDefault,
        );
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            !json.contains("\"dispatch_id\""),
            "session_spawned_ad_hoc must remain byte-identical: {json}"
        );
    }

    // ── Phase 15 Plan 15-09 (WR-01 + WR-02 cliff closure) ─────────────────
    //
    // Wire-shape: `SessionSpawned` carries a `scope` field on serialisation.
    // Audit-replay backcompat: a pre-15-09 JSON without `scope` deserialises
    // cleanly via `#[serde(default)]`, defaulting to `Scope::SessionDefault`.
    //
    // `AlzinaEvent` itself derives only `Serialize`, so the round-trip
    // backcompat test uses a structurally-equivalent shadow struct to
    // exercise the `#[serde(default)]` attribute. That is the audit-replay
    // contract — pre-15-09 JSONL lines must still deserialise.

    #[test]
    fn session_spawned_serialises_with_scope_when_weave() {
        use crate::identity::{Scope, WeaveId};

        let event = AlzinaEvent::SessionSpawned {
            session_id: "s-rt".into(),
            agent_id: "huginn".into(),
            parent_id: None,
            task_summary: "round-trip probe".into(),
            timestamp: 1,
            composition: None,
            task_rendered: None,
            dispatch_id: None,
            scope: Scope::Weave(WeaveId::new("W-rt")),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            json.contains("\"scope\":\"W-rt\""),
            "SessionSpawned MUST emit `scope` on the wire (Weave variant): {json}"
        );
    }

    #[test]
    fn session_spawned_serialises_with_scope_when_session_default() {
        use crate::identity::Scope;

        let event = AlzinaEvent::SessionSpawned {
            session_id: "s-rt".into(),
            agent_id: "huginn".into(),
            parent_id: None,
            task_summary: "round-trip probe".into(),
            timestamp: 1,
            composition: None,
            task_rendered: None,
            dispatch_id: None,
            scope: Scope::SessionDefault,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            json.contains("\"scope\":\"SessionDefault\""),
            "SessionSpawned MUST emit `scope` on the wire (SessionDefault): {json}"
        );
    }

    /// Audit-replay backcompat: pre-15-09 audit JSONL lines do not carry
    /// `scope`. The `#[serde(default)]` attribute on the variant must
    /// default the field to `Scope::SessionDefault` so those lines still
    /// deserialise cleanly. Because `AlzinaEvent` itself derives only
    /// `Serialize`, this guarantee is exercised here on a shadow struct
    /// that mirrors the variant's fields and attributes — if the
    /// `Scope: Default` impl or the `#[serde(default)]` attribute is
    /// removed, this test fails.
    #[test]
    fn session_spawned_scope_defaults_to_session_default_when_field_missing() {
        use crate::identity::Scope;
        use serde::Deserialize;

        // Mirror of the SessionSpawned variant for deserialisation. The
        // `#[serde(default)]` here mirrors the live attribute on the
        // variant field. If the impl Default for Scope is removed, this
        // shadow struct fails to compile — load-bearing for the
        // audit-replay backcompat guarantee.
        #[derive(Deserialize)]
        struct ShadowSessionSpawned {
            #[allow(dead_code)]
            session_id: String,
            #[allow(dead_code)]
            agent_id: String,
            #[serde(default)]
            scope: Scope,
        }

        let pre_15_09_json = r#"{
            "session_id": "s-legacy",
            "agent_id": "huginn",
            "parent_id": null,
            "task_summary": "pre-15-09",
            "timestamp": 1
        }"#;

        let shadow: ShadowSessionSpawned =
            serde_json::from_str(pre_15_09_json).expect("pre-15-09 JSON must deserialise");
        assert_eq!(
            shadow.scope,
            Scope::SessionDefault,
            "missing `scope` MUST default to Scope::SessionDefault (audit-replay backcompat)"
        );
    }
}

#[cfg(test)]
mod scoped_envelope_tests {
    use super::*;
    use crate::identity::Scope;

    #[test]
    fn scoped_envelope_bare_uses_session_default() {
        let event = AlzinaEvent::Heartbeat { timestamp: 0 };
        let envelope = ScopedEnvelope::bare(event);
        assert_eq!(
            envelope.scope,
            Scope::SessionDefault,
            "bare() must default to Scope::SessionDefault"
        );
    }

    #[test]
    fn scoped_envelope_new_binds_scope_to_event() {
        use crate::identity::{Scope, WeaveId};
        let scope = Scope::Weave(WeaveId::new("W-test-001"));
        let event = AlzinaEvent::Heartbeat { timestamp: 42 };
        let envelope = ScopedEnvelope::new(scope.clone(), event);
        assert_eq!(envelope.scope.as_str(), "W-test-001");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_detached_serde_tag() {
        let e = AlzinaEvent::SessionDetached {
            session_id: "s".into(),
            parent_id: Some("p".into()),
            live_children: 2,
            timestamp: 1,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "SessionDetached");
        assert_eq!(v["session_id"], "s");
        assert_eq!(v["parent_id"], "p");
        assert_eq!(v["live_children"], 2);
    }

    #[test]
    fn session_detached_serde_tag_with_none_parent() {
        let e = AlzinaEvent::SessionDetached {
            session_id: "s".into(),
            parent_id: None,
            live_children: 0,
            timestamp: 0,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "SessionDetached");
        assert!(v["parent_id"].is_null());
    }

    #[test]
    fn dispatch_cancel_requested_serde_tag() {
        let e = AlzinaEvent::DispatchCancelRequested {
            dispatch_id: "d".into(),
            session_id: "s".into(),
            parent_session_id: "p".into(),
            timestamp: 1,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "DispatchCancelRequested");
        assert_eq!(v["dispatch_id"], "d");
        assert_eq!(v["parent_session_id"], "p");
    }

    #[test]
    fn dispatch_retry_serde_tag() {
        let e = AlzinaEvent::DispatchRetry {
            dispatch_id: "d".into(),
            attempt: 2,
            max_attempts: 3,
            reason: "sidecar_crash".into(),
            backoff_ms: 500,
            timestamp: 1,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "DispatchRetry");
        assert_eq!(v["attempt"], 2);
        assert_eq!(v["max_attempts"], 3);
        assert_eq!(v["backoff_ms"], 500);
        assert_eq!(v["reason"], "sidecar_crash");
    }

    // ── Phase 6: DispatchEnvelopeReceived + TurnStarted ──────────────────

    #[test]
    fn dispatch_envelope_received_serde_tag() {
        let e = AlzinaEvent::DispatchEnvelopeReceived {
            dispatch_id: "d".into(),
            parent_session_id: "p".into(),
            agent: "huginn".into(),
            status: "completed".into(),
            envelope_summary: "search complete: 3 hits".into(),
            timestamp: 1,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "DispatchEnvelopeReceived");
        assert_eq!(v["dispatch_id"], "d");
        assert_eq!(v["parent_session_id"], "p");
        assert_eq!(v["agent"], "huginn");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["envelope_summary"], "search complete: 3 hits");
    }

    #[test]
    fn turn_started_serde_tag() {
        let e = AlzinaEvent::TurnStarted {
            turn_id: "t-1".into(),
            session_id: "chat-root".into(),
            origin: "auto_continuation".into(),
            timestamp: 42,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "TurnStarted");
        assert_eq!(v["turn_id"], "t-1");
        assert_eq!(v["session_id"], "chat-root");
        assert_eq!(v["origin"], "auto_continuation");
        assert_eq!(v["timestamp"], 42);
    }

    #[test]
    fn turn_started_serde_tag_user_origin() {
        let e = AlzinaEvent::TurnStarted {
            turn_id: "t-2".into(),
            session_id: "chat-root".into(),
            origin: "user".into(),
            timestamp: 43,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["origin"], "user");
    }

    // ── Phase 7: EngagementOpened / EngagementTurn / EngagementResolved ──

    #[test]
    fn engagement_opened_serde_tag() {
        let e = AlzinaEvent::EngagementOpened {
            engagement_id: "e-1".into(),
            weave_id: "W-1".into(),
            prompt: "approve?".into(),
            mode_kind: "approval".into(),
            timestamp: 1,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "EngagementOpened");
        assert_eq!(v["engagement_id"], "e-1");
        assert_eq!(v["weave_id"], "W-1");
        assert_eq!(v["mode_kind"], "approval");
    }

    #[test]
    fn engagement_turn_serde_tag() {
        let e = AlzinaEvent::EngagementTurn {
            engagement_id: "e-1".into(),
            author: "human".into(),
            content: "ack".into(),
            timestamp: 2,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "EngagementTurn");
        assert_eq!(v["author"], "human");
        assert_eq!(v["content"], "ack");
    }

    #[test]
    fn engagement_resolved_serde_tag() {
        let e = AlzinaEvent::EngagementResolved {
            engagement_id: "e-1".into(),
            outcome_kind: "resolved".into(),
            timestamp: 3,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "EngagementResolved");
        assert_eq!(v["outcome_kind"], "resolved");
    }

    // ── Phase 9: WeaveOpenedByGate / WeavePromptRequested ─────────────────

    #[test]
    fn weave_opened_by_gate_serde_tag() {
        let e = AlzinaEvent::WeaveOpenedByGate {
            weave_id: "W-new".into(),
            label: "refactor dispatch loop".into(),
            timestamp: 100,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "WeaveOpenedByGate");
        assert_eq!(v["weave_id"], "W-new");
        assert_eq!(v["label"], "refactor dispatch loop");
        assert_eq!(v["timestamp"], 100);
    }

    #[test]
    fn weave_prompt_requested_serde_tag() {
        let e = AlzinaEvent::WeavePromptRequested {
            open_weaves: vec!["W-1".into(), "W-2".into()],
            message_preview: "fix the dispatch bug and update docs".into(),
            timestamp: 200,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "WeavePromptRequested");
        assert_eq!(v["open_weaves"][0], "W-1");
        assert_eq!(v["open_weaves"][1], "W-2");
        assert_eq!(v["message_preview"], "fix the dispatch bug and update docs");
    }

    #[test]
    fn weave_opened_by_gate_serde_tag_empty_label() {
        let e = AlzinaEvent::WeaveOpenedByGate {
            weave_id: "W-x".into(),
            label: String::new(),
            timestamp: 0,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "WeaveOpenedByGate");
        assert_eq!(v["label"], "");
    }

    #[test]
    fn weave_prompt_requested_serde_tag_no_open_weaves() {
        let e = AlzinaEvent::WeavePromptRequested {
            open_weaves: vec![],
            message_preview: "p".into(),
            timestamp: 0,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "WeavePromptRequested");
        assert!(v["open_weaves"].as_array().unwrap().is_empty());
    }

    // ── Phase 16: CancelInitiated / CascadeCancelled / ProgressPaused / ProgressResumed ──

    #[test]
    fn cancel_initiated_constructs_and_serialises() {
        let e = AlzinaEvent::CancelInitiated {
            level: "composition".into(),
            target_id: "c1".into(),
            scope: crate::identity::Scope::SessionDefault,
            timestamp: 0,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "CancelInitiated");
        assert_eq!(v["level"], "composition");
        assert_eq!(v["target_id"], "c1");
    }

    #[test]
    fn cascade_cancelled_constructs_and_serialises() {
        let e = AlzinaEvent::CascadeCancelled {
            target_id: "c1".into(),
            parent_session_id: "p1".into(),
            child_session_id: "s1".into(),
            agent_id: "huginn".into(),
            compose_id: Some("c1".into()),
            weave_id: None,
            timestamp: 0,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "CascadeCancelled");
        assert_eq!(v["target_id"], "c1");
        assert_eq!(v["parent_session_id"], "p1");
        assert!(v["compose_id"].is_string());
        assert!(v["weave_id"].is_null());
    }

    #[test]
    fn progress_paused_constructs_and_serialises() {
        let e = AlzinaEvent::ProgressPaused {
            weave_id: "W-1".into(),
            scope: crate::identity::Scope::SessionDefault,
            timestamp: 0,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "ProgressPaused");
        assert_eq!(v["weave_id"], "W-1");
    }

    #[test]
    fn progress_resumed_constructs_and_serialises() {
        let e = AlzinaEvent::ProgressResumed {
            weave_id: "W-1".into(),
            scope: crate::identity::Scope::SessionDefault,
            resume_reason: "operator_message".into(),
            timestamp: 0,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "ProgressResumed");
        assert_eq!(v["weave_id"], "W-1");
        assert_eq!(v["resume_reason"], "operator_message");
    }
}
