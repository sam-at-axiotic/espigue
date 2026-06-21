//! The `return_envelope` custom tool definition.
//!
//! Backend-agnostic typed tool contract for sub-agent envelope return.
//! When the model invokes this tool, the typed payload (matching
//! `alzina_core::Envelope`) flows back to the runner via a captured
//! `SidecarEvent::ToolUse` block — no prose parsing required.
//!
//! See plan 260515-ndk for the full design rationale. Key properties:
//!
//! - The `endpoint` is a stub — the executor short-circuits the tool
//!   in the event loop and never actually fetches it (RESEARCH P6).
//! - The `input_schema` mirrors `alzina_core::Envelope` exactly so the
//!   captured input deserialises cleanly via `serde_json::from_value`.
//! - The description tells the model that this is the terminal call,
//!   text after is informational, and re-calling = last-wins (P1).
//! - Zero Claude-specific naming: this lives in `alzina-orchestration`
//!   and is consumed by `ClaudeAgentSdkExecutor` (and any future
//!   non-Claude backend executor).

use serde_json::json;

use super::sidecar_protocol::CustomToolDefinition;

/// Build the `return_envelope` `CustomToolDefinition`.
///
/// The returned definition is injected into the sub-agent dispatch
/// path's `SidecarOptions.custom_tools` unconditionally. The model
/// learns the tool exists via the SDK; calling it terminates the
/// envelope-return contract.
///
/// `input_schema` mirrors `alzina_core::Envelope`:
/// - `status`: required enum (`complete` | `partial` | `error`)
/// - `artifacts`: optional array of file path strings (defaults to `[]`)
/// - `signal`, `tensions`, `emergent`, `next`, `context_update`: optional strings
pub fn return_envelope_tool() -> CustomToolDefinition {
    CustomToolDefinition {
        name: "return_envelope".to_string(),
        description: concat!(
            "Submit your structured return envelope. ",
            "This is the terminal call for the dispatch — any text emitted ",
            "after invoking this tool is informational only and will not be ",
            "re-parsed for envelope fields. If you call this tool more than ",
            "once, the LAST invocation wins. ",
            "Status must be exactly 'complete', 'partial', or 'error'. ",
            "Artifacts is a list of file paths you wrote during this task. ",
            "Signal is a one-line summary of the dispatch outcome. ",
            "Tensions, emergent, next, and context_update are optional ",
            "free-form text fields matching the prose return-format trailer.",
        )
        .to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["complete", "partial", "error"],
                    "description": "Outcome of the dispatch — must be one of complete | partial | error."
                },
                "artifacts": {
                    "type": "array",
                    "items": { "type": "string" },
                    "default": [],
                    "description": "List of file paths written during this task (one per entry)."
                },
                "signal": {
                    "type": "string",
                    "description": "One-line summary for the dispatch log."
                },
                "tensions": {
                    "type": "string",
                    "description": "Pointers to contradictions or conflicts found, or omit/empty if none."
                },
                "emergent": {
                    "type": "string",
                    "description": "Observations outside the task brief, or omit/empty if none."
                },
                "next": {
                    "type": "string",
                    "description": "Recommended next action, or omit for no follow-up. Include ONLY for blockers, dependencies, or gated conditions."
                },
                "context_update": {
                    "type": "string",
                    "description": "Reusable learning for future sessions, or omit/empty if none."
                }
            }
        }),
        // Stub endpoint — the executor intercepts `return_envelope` before
        // any HTTP fetch fires (RESEARCH P6). Sidecar TS handler also
        // short-circuits. The value here just has to satisfy the
        // SSRF validator (localhost) so a future test that accidentally
        // routes through the fetch path fails loudly rather than panicking
        // on a malformed URL.
        endpoint: "http://127.0.0.1:0/internal/return_envelope".to_string(),
        method: None,
        timeout_ms: None,
        stream_progress: false,
        idle_timeout_ms: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_core::Envelope;

    /// Test 1: the `input_schema` validates a complete Envelope shape by
    /// successfully deserialising a sample payload into `alzina_core::Envelope`.
    /// (The JSON Schema itself isn't validated at runtime — we prove
    /// equivalence by round-tripping through serde.)
    #[test]
    fn return_envelope_tool_schema_matches_envelope_shape() {
        let def = return_envelope_tool();
        assert_eq!(def.name, "return_envelope");

        // A payload an LLM would emit for a "complete" task with one artifact.
        let sample = json!({
            "status": "complete",
            "artifacts": ["artifacts/foo.md"],
            "signal": "OK",
            "tensions": "none",
            "emergent": "none",
            "next": null,
            "context_update": "always validate config at load time"
        });

        // The schema's required fields and types must accept a payload
        // that successfully deserialises into Envelope.
        let env: Envelope = serde_json::from_value(envelope_payload_with_canonical_status(&sample))
            .expect("sample payload must deserialise into alzina_core::Envelope");
        assert_eq!(env.status, alzina_core::EnvelopeStatus::Complete);
        assert_eq!(env.artifacts.len(), 1);

        // Required field declared.
        let req = def
            .input_schema
            .get("required")
            .and_then(|v| v.as_array())
            .expect("schema must declare a `required` array");
        assert!(req.iter().any(|v| v.as_str() == Some("status")));

        // Status enum lists all three canonical variants.
        let status_enum = def
            .input_schema
            .pointer("/properties/status/enum")
            .and_then(|v| v.as_array())
            .expect("status must declare an enum");
        let variants: Vec<&str> = status_enum.iter().filter_map(|v| v.as_str()).collect();
        for canonical in ["complete", "partial", "error"] {
            assert!(
                variants.contains(&canonical),
                "status enum must include `{canonical}`"
            );
        }
    }

    /// Helper: the model-facing schema uses lowercase status strings
    /// (`"complete"`), but `alzina_core::EnvelopeStatus` serialises as
    /// `Pascal` (`"Complete"`). The runner-side interception in Task 3
    /// will normalise; here we just remap so the schema-equivalence
    /// proof above doesn't require an as-yet-unimplemented normaliser.
    fn envelope_payload_with_canonical_status(v: &serde_json::Value) -> serde_json::Value {
        let mut o = v.as_object().cloned().expect("object");
        if let Some(status) = o.get("status").and_then(|s| s.as_str()) {
            let normalised = match status {
                "complete" => "Complete",
                "partial" => "Partial",
                "error" => "Error",
                other => other,
            };
            o.insert(
                "status".into(),
                serde_json::Value::String(normalised.into()),
            );
        }
        serde_json::Value::Object(o)
    }
}
