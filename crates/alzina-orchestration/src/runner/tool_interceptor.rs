//! Tool call interception for write-tier enforcement.
//!
//! Intercepts file-write tool calls during agent execution and checks
//! them against the GovernanceLayer before allowing execution.
//! This is the enforcement seam described in §5.2 of the architecture doc.
//!
//! # Assigned-dir policy
//!
//! When the dispatch handler registers a per-spawn writable dir via
//! [`AssignedDirRegistry`](super::assigned_dirs::AssignedDirRegistry), the
//! executor passes that dir down as `assigned_dir`. The interceptor then
//! enforces a strict append-only-in-dir contract layered on top of the
//! existing tier rules:
//!
//! - `Write` whose target is inside the dir and does NOT yet exist on
//!   disk → `Allow`. This is the path Sjöfn/leaves use to materialise the
//!   artifacts they declare in their return envelope.
//! - `Write` whose target is inside the dir but DOES exist → `Block`. The
//!   dir is no-overwrite so siblings cannot clobber each other's outputs,
//!   even when picking colliding filenames.
//! - `Edit` whose target is inside the dir → `Block`. No in-place edits;
//!   if more content is needed, use `Write` with a fresh filename.
//! - `Bash` whose command string mentions the dir → `Block`. Conservative
//!   ban — Bash is a hole big enough to drive a truck through, and parsing
//!   shell is famously a tarpit.
//!
//! Outside the assigned dir, all four tools fall through to the existing
//! tier rules (which may still block governed paths).
//!
//! When no dir is registered, the interceptor preserves pre-existing
//! behaviour byte-for-byte.

use alzina_core::AlzinaResult;
use alzina_core::identity::SessionId;
use alzina_core::tiers::{TierDecision, WriteOp};
use alzina_governance::GovernanceLayer;
use alzina_governance::profiles::ExecPolicy;
use alzina_workspace::WorkspaceHandle;
use alzina_workspace::artifacts::validate_under;
use serde_json::Value;
use std::collections::HashMap;

/// Arguments passed to a tool call.
#[derive(Debug, Clone)]
pub struct ToolArgs {
    /// Key-value arguments from the tool invocation.
    pub values: HashMap<String, Value>,
}

impl ToolArgs {
    /// Create from a serde_json::Value (expects an object).
    pub fn from_value(val: &Value) -> Self {
        let values = match val.as_object() {
            Some(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            None => HashMap::new(),
        };
        Self { values }
    }

    /// Extract a file path from common tool argument patterns.
    /// Checks: "path", "file_path", "file", "target" — in that order.
    pub fn extract_path(&self) -> Option<&str> {
        for key in &["path", "file_path", "file", "target"] {
            if let Some(s) = self
                .values
                .get(*key)
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                return Some(s);
            }
        }
        None
    }

    /// Extract the shell command string from a `Bash` tool call.
    /// Claude's Bash tool input shape is `{"command": "...", "description": "..."}`.
    pub fn extract_command(&self) -> Option<&str> {
        self.values
            .get("command")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    }
}

/// Decision from the tool interceptor.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolDecision {
    /// Tool call is allowed to proceed.
    Allow,
    /// Tool call is blocked with a reason.
    Block(String),
    /// Tool call requires operator approval (Tier 1 governed path).
    RequiresApproval,
}

/// Tools known to perform file writes.
const WRITE_TOOLS: &[&str] = &[
    "write",
    "edit",
    "create_file",
    "write_file",
    "overwrite",
    "append",
];

/// Tools that provide shell/exec access — always require approval because
/// they can perform arbitrary filesystem writes that bypass path-based checks.
const SHELL_TOOLS: &[&str] = &["exec", "shell", "bash"];

/// Tools known to perform file reads. Used for the workspace deny-list
/// branch — Claude's SDK ships in `bypassPermissions` mode, so without an
/// interceptor branch every read passes through unconditionally. The
/// existing tier-enforcement layer governs writes only.
const READ_TOOLS: &[&str] = &["read", "view", "read_file", "notebookread"];

/// Check whether a tool call should be allowed, blocked, or gated.
///
/// For write-like tools, extracts the target path from args and checks
/// against the GovernanceLayer's tier enforcement. Non-write tools pass through.
///
/// # Security: Defence-in-Depth Only
///
/// This interceptor checks the *declared* path argument, not the actual filesystem
/// write. A TOCTOU gap exists: the tool arguments checked here may differ from the
/// actual write performed by the executor. True write-tier enforcement must occur at
/// the filesystem layer (`WorkspaceHandle`), which independently validates every write
/// against governance tiers. This interceptor is a useful early-reject optimisation
/// that reduces unnecessary LLM round-trips, but it is NOT a security boundary.
pub fn check_tool_call(
    governance: &GovernanceLayer,
    agent_id: &str,
    _session_id: &SessionId,
    tool_name: &str,
    args: &ToolArgs,
    assigned_dir: Option<&str>,
) -> AlzinaResult<ToolDecision> {
    // SECURITY: defence-in-depth only — true enforcement is in WorkspaceHandle.
    // This check validates declared intent (tool args), not actual writes.
    // See doc comment above for TOCTOU considerations.

    // Workspace deny-list runs FIRST: a denied path must be blocked
    // regardless of profile, assigned dir, or tier classification. Covers
    // read tools (otherwise unmatched by the rest of this function),
    // write tools (covered redundantly by the tier check below for
    // defence-in-depth), and shell tools that mention a denied path's
    // literal prefix in the command body.
    if let Some(decision) = check_deny_list(governance, tool_name, args) {
        return Ok(decision);
    }

    // Latent #2: per-agent tool-name deny-list from identity.toml's
    // `[tools].denied`. Runs AFTER the workspace path deny-list but
    // BEFORE profile / exec_policy / tier checks so an agent's
    // explicit "I should not call Bash" pin overrides any profile
    // default that would otherwise let the call through. Case-
    // sensitive exact match against the raw tool name the SDK
    // reports.
    //
    // This is the proper per-agent override that Fix C punted on:
    // Fix C aligned vefr's profile default with its identity, but
    // a future orchestrator-profile agent that legitimately needs
    // shell could now opt-out at the identity layer instead of
    // sharing a profile-wide flag with vefr.
    for denied in governance.denied_tools_for(agent_id) {
        if denied == tool_name {
            return Ok(ToolDecision::Block(format!(
                "agent '{agent_id}' identity denies tool '{tool_name}' \
                 ([tools].denied in identity.toml)"
            )));
        }
    }

    // Per-dispatch assigned-dir policy. Runs BEFORE the existing tier
    // checks so a write to the assigned dir bypasses tier rules that
    // might otherwise block `artifacts/unattached/*` (and so an Edit or
    // overwrite inside the dir is rejected loudly with a specific reason
    // rather than a generic tier error). Outside the dir, control falls
    // through to the existing logic unchanged.
    if let Some(dir) = assigned_dir {
        if let Some(decision) =
            check_assigned_dir_policy(governance.workspace(), tool_name, args, dir)
        {
            return Ok(decision);
        }
    }

    // Shell/exec tools: consult the agent's profile capability `exec_policy`
    // rather than blanket-blocking. Pre-2026-05-05 this returned
    // `RequiresApproval` unconditionally, which meant agents whose profile
    // declared SandboxedReadonly (e.g. researcher, builder) still couldn't
    // run shell — the whole `ExecPolicy` enum was never read. Live finding
    // fixed: huginn could be dispatched but couldn't actually inspect the
    // workspace because every Bash call hit "tool requires approval —
    // blocking in sidecar mode."
    //
    // Mapping:
    //   ExecPolicy::None              → Block (agent has no shell)
    //   ExecPolicy::SandboxedReadonly → Allow (read-only sandbox)
    //   ExecPolicy::Sandboxed         → Allow (write-allowed sandbox)
    //   ExecPolicy::Gated             → RequiresApproval (orchestrator;
    //                                   blocked in sidecar mode by design)
    //
    // Sandbox enforcement at the OS level (e.g. firejail-style restrictions
    // on what Bash can read/write) is a separate concern tracked as a
    // future hardening — see TODO below. For now SandboxedReadonly trusts
    // the agent prompt + WorkspaceHandle's filesystem layer to gate writes.
    //
    // TODO(security): wrap shell invocations with an OS-level sandbox so
    // SandboxedReadonly is actually read-only at the kernel boundary, not
    // just at the agent-prompt boundary. Until then this is defense-in-
    // depth only, matching the comment above about WriteOp checks.
    if is_shell_tool(tool_name) {
        // GATE 3.5: per-agent fail-closed shell allowlist. Runs BEFORE the
        // exec_policy gate so it can only NARROW what a profile permits,
        // never widen it. The allowlist `evaluate` either vetoes
        // (`Some(reason)` → Block) or abstains (`None` → fall through to
        // exec_policy). It never returns an allow authority — exec_policy
        // remains the final allow decision.
        //
        // Fail-closed by construction: an agent with no `[tools].shell_allow`
        // entry gets a static fail-closed EMPTY allowlist, so EVERY shell
        // command is blocked here unless the identity explicitly names the
        // program. This is what makes "empty/absent ⇒ deny all" true.
        //
        // The command string is read from the Bash tool's `command` arg.
        // A shell call with no recognisable command string cannot be
        // authorised against the allowlist → block (we do not guess).
        let allowlist = governance.shell_allowlist_for(agent_id);
        // Read the command string. If the tool carries no recognisable
        // command, pass the empty string to `evaluate`: under fail-closed
        // that blocks ("nothing to authorise"), under the explicit
        // `disabled` opt-out it abstains. Either way the policy decision
        // lives entirely in `evaluate` — we never guess an allow here.
        let command = args.extract_command().unwrap_or("");
        if let Some(reason) = allowlist.evaluate(command) {
            return Ok(ToolDecision::Block(reason));
        }

        let exec_policy = governance.check_exec(agent_id);
        return Ok(match exec_policy {
            ExecPolicy::None => ToolDecision::Block(format!(
                "agent '{agent_id}' has exec_policy=none; shell tools are denied",
            )),
            ExecPolicy::SandboxedReadonly | ExecPolicy::Sandboxed => ToolDecision::Allow,
            ExecPolicy::Gated => ToolDecision::RequiresApproval,
        });
    }

    // Non-write tools always pass through.
    if !is_write_tool(tool_name) {
        return Ok(ToolDecision::Allow);
    }

    // Write tool: extract path or block if missing.
    let path = match args.extract_path() {
        Some(p) => p,
        None => {
            return Ok(ToolDecision::Block(format!(
                "write tool '{}' called without a recognisable path argument",
                tool_name
            )));
        }
    };

    // Determine write op from tool name.
    let op = write_op_for_tool(tool_name);

    // Check governance tier.
    let decision = governance.check_write_op(path, agent_id, op);

    Ok(match decision {
        TierDecision::Allowed => ToolDecision::Allow,
        TierDecision::Blocked { reason, .. } => ToolDecision::Block(reason),
        TierDecision::RequiresApproval { .. } => ToolDecision::RequiresApproval,
    })
}

/// Assigned-dir enforcement. Returns `Some(decision)` when the call is
/// scoped by the dir contract — `None` to fall through to the existing
/// tier-based checks unchanged.
fn check_assigned_dir_policy(
    workspace: &WorkspaceHandle,
    tool_name: &str,
    args: &ToolArgs,
    assigned_dir: &str,
) -> Option<ToolDecision> {
    let lower = tool_name.to_lowercase();

    // Bash: ban any command whose body mentions the assigned dir. Parsing
    // shell well is famously hard; we are explicit about over-blocking and
    // tell the agent to use the `Write` tool instead.
    if is_shell_tool(&lower) {
        let cmd = args.extract_command()?;
        if command_writes_into(cmd, workspace.root(), assigned_dir) {
            return Some(ToolDecision::Block(format!(
                "Bash command references the assigned output dir '{assigned_dir}'. \
                 The dir is write-new-only — use the Write tool with a fresh \
                 filename instead of writing via shell."
            )));
        }
        return None;
    }

    if !is_write_tool(&lower) {
        return None;
    }

    let raw_path = args.extract_path()?;
    let relative = match relativise_to_workspace(workspace.root(), raw_path) {
        Some(rel) => rel,
        // Path escapes the workspace entirely — let the existing tier
        // logic deny it with its usual error.
        None => return None,
    };

    if !path_is_inside(&relative, assigned_dir) {
        return None;
    }

    match lower.as_str() {
        "edit" => Some(ToolDecision::Block(format!(
            "Edit blocked: '{raw_path}' is inside the assigned output dir \
             '{assigned_dir}', which is append-only at the file level. \
             Use Write with a new filename instead."
        ))),
        "write" | "create_file" | "write_file" | "overwrite" | "append" => {
            match workspace.exists(&relative) {
                Ok(true) => Some(ToolDecision::Block(format!(
                    "{tool_name} blocked: '{raw_path}' already exists in the \
                     assigned output dir '{assigned_dir}', which is no-overwrite. \
                     Choose a different filename."
                ))),
                Ok(false) => Some(ToolDecision::Allow),
                // If we cannot tell whether the file exists, fall through
                // to the existing tier logic rather than guessing.
                Err(_) => None,
            }
        }
        _ => None,
    }
}

/// Validate the artifacts declared in a `return_envelope` payload against
/// the agent's assigned output dir.
///
/// Layered on top of the per-call `Write`/`Edit`/`Bash` checks as a second
/// line of defence: even if an agent slipped a write past the per-call
/// interceptor (or fabricated an artifact entry without writing the file),
/// the return-envelope step refuses to terminate the spawn until every
/// declared path is real and in the right place.
///
/// Returns the normalised (workspace-relative) `PathBuf` list on success,
/// suitable for `Envelope::artifacts` rewrite — so downstream consumers see
/// one canonical form instead of the mix of absolute and relative paths
/// agents tend to produce.
///
/// On failure, returns a list of human-readable error messages — one per
/// failing path — that the caller can stitch into a `HookDecisionMsg::block`
/// reason so the agent can correct and re-submit. Empty `artifacts` is
/// vacuously `Ok(vec![])`.
pub fn validate_envelope_artifacts(
    artifacts: &[std::path::PathBuf],
    workspace: &WorkspaceHandle,
    assigned_dir: &str,
) -> Result<Vec<std::path::PathBuf>, Vec<String>> {
    let mut errors = Vec::new();
    let mut normalised = Vec::with_capacity(artifacts.len());

    for path in artifacts {
        let raw = path.to_string_lossy();
        let relative = match relativise_to_workspace(workspace.root(), raw.as_ref()) {
            Some(r) => r,
            None => {
                errors.push(format!(
                    "artifact '{raw}' is not under workspace root \
                     — only paths inside the workspace are accepted"
                ));
                continue;
            }
        };
        if !path_is_inside(&relative, assigned_dir) {
            errors.push(format!(
                "artifact '{raw}' is outside assigned output dir '{assigned_dir}' \
                 — only paths inside the dir are accepted"
            ));
            continue;
        }
        match workspace.exists(&relative) {
            Ok(true) => normalised.push(std::path::PathBuf::from(relative)),
            Ok(false) => errors.push(format!(
                "artifact '{raw}' declared in envelope but does not exist on disk \
                 — write the file (or remove it from the artifacts list)"
            )),
            Err(e) => errors.push(format!(
                "could not verify artifact '{raw}': {e} \
                 — re-check the path and re-submit"
            )),
        }
    }

    if errors.is_empty() {
        Ok(normalised)
    } else {
        Err(errors)
    }
}

/// Normalise a tool-supplied path to a workspace-relative string. Accepts
/// either an absolute path under the workspace root or an already-relative
/// path. Returns `None` if the path is absolute and escapes the workspace.
///
/// `workspace_root` is the value `WorkspaceHandle::root()` returns, which
/// is canonicalised on construction (no symlinks). On macOS, tempdirs
/// (and any user-supplied root under a symlinked prefix like `/var` →
/// `/private/var`) will need a second canonicalisation pass on the input
/// before `strip_prefix` can match. We canonicalise the input's parent
/// rather than the input itself because the target file may not yet
/// exist (the whole point of the no-overwrite check).
fn relativise_to_workspace(workspace_root: &std::path::Path, raw_path: &str) -> Option<String> {
    let p = std::path::Path::new(raw_path);
    if !p.is_absolute() {
        return Some(raw_path.to_string());
    }
    if let Ok(rel) = p.strip_prefix(workspace_root) {
        return Some(rel.to_string_lossy().into_owned());
    }
    let parent = p.parent()?;
    let file = p.file_name()?;
    let canonical_parent = parent.canonicalize().ok()?;
    canonical_parent
        .join(file)
        .strip_prefix(workspace_root)
        .ok()
        .map(|rel| rel.to_string_lossy().into_owned())
}

/// True when `target` is a path inside `dir`. Both are workspace-relative.
/// Defers to `alzina_workspace::validate_under` for lexical normalisation
/// so `..` traversal and absolute escapes are rejected as "not inside".
fn path_is_inside(target: &str, dir: &str) -> bool {
    validate_under(target, dir).is_ok()
}

/// True when a Bash command appears to write into the assigned dir.
/// Conservative substring match against both the workspace-relative
/// form and the absolute form of the dir. We block on mere mention,
/// which is over-broad by design — agents are told to use `Write`.
fn command_writes_into(
    cmd: &str,
    workspace_root: &std::path::Path,
    assigned_dir: &str,
) -> bool {
    if cmd.contains(assigned_dir) {
        return true;
    }
    let abs = workspace_root.join(assigned_dir);
    let abs_str = abs.to_string_lossy();
    !abs_str.is_empty() && cmd.contains(abs_str.as_ref())
}

/// Is this tool name a write-like tool?
fn is_write_tool(name: &str) -> bool {
    let lower = name.to_lowercase();
    WRITE_TOOLS.iter().any(|t| lower == *t)
}

/// Is this tool name a shell/exec tool?
fn is_shell_tool(name: &str) -> bool {
    let lower = name.to_lowercase();
    SHELL_TOOLS.iter().any(|t| lower == *t)
}

/// Is this tool name a read-like tool?
fn is_read_tool(name: &str) -> bool {
    let lower = name.to_lowercase();
    READ_TOOLS.iter().any(|t| lower == *t)
}

/// Workspace deny-list check applied uniformly across read, write, and
/// shell tools. Returns `Some(Block)` when the call targets a denied
/// path, `None` to fall through to the existing branches.
///
/// Path-naming tools (read / write) check the extracted path argument
/// against the pre-compiled `GovernanceLayer::denies_path` matcher.
/// Shell tools scan the command body for substring matches against the
/// literal prefix of each deny-list pattern — over-broad by design, to
/// match the assigned-dir code's stance (parsing shell is a tarpit).
fn check_deny_list(
    governance: &GovernanceLayer,
    tool_name: &str,
    args: &ToolArgs,
) -> Option<ToolDecision> {
    if is_read_tool(tool_name) || is_write_tool(tool_name) {
        let raw = args.extract_path()?;
        let normalised = relativise_to_workspace(governance.workspace().root(), raw)
            .unwrap_or_else(|| raw.to_string());
        if governance.denies_path(&normalised) {
            return Some(ToolDecision::Block(format!(
                "tool '{tool_name}' targets '{raw}' which matches a workspace deny-list glob"
            )));
        }
        return None;
    }

    if is_shell_tool(tool_name) {
        let cmd = args.extract_command()?;
        for pattern in &governance.config().denied_path_globs {
            let literal = pattern_literal_prefix(pattern);
            if !literal.is_empty() && cmd.contains(&literal) {
                return Some(ToolDecision::Block(format!(
                    "shell command references '{literal}' which matches the workspace \
                     deny-list pattern '{pattern}'. Use a tool with a path argument so \
                     the check can be specific, or pick a different target."
                )));
            }
        }
        return None;
    }

    None
}

/// Extract the literal-character prefix of a glob, stopping at the first
/// metacharacter. Used by the shell deny-list scan because parsing
/// arbitrary shell to extract write targets is famously intractable.
///
/// Examples:
///   `.env`           → `.env`
///   `.env.*`         → `.env.`
///   `secrets/**`     → `secrets/`
///   `**/*.key`       → `` (empty — pattern has no literal prefix; shell
///                          tools cannot enforce it without real parsing)
fn pattern_literal_prefix(pattern: &str) -> String {
    let mut out = String::new();
    for c in pattern.chars() {
        if matches!(c, '*' | '?' | '[' | '{') {
            break;
        }
        out.push(c);
    }
    out
}

/// Map tool name to the most appropriate WriteOp.
fn write_op_for_tool(name: &str) -> WriteOp {
    match name.to_lowercase().as_str() {
        "append" => WriteOp::Append,
        "create_file" => WriteOp::Create,
        "edit" => WriteOp::Modify,
        _ => WriteOp::Modify,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── ToolArgs tests ──────────────────────────────────────────

    #[test]
    fn extract_path_from_path_key() {
        let args = ToolArgs::from_value(&json!({"path": "/foo/bar.rs"}));
        assert_eq!(args.extract_path(), Some("/foo/bar.rs"));
    }

    #[test]
    fn extract_path_from_file_path_key() {
        let args = ToolArgs::from_value(&json!({"file_path": "/foo/bar.rs"}));
        assert_eq!(args.extract_path(), Some("/foo/bar.rs"));
    }

    #[test]
    fn extract_path_priority() {
        // "path" wins over "file_path"
        let args = ToolArgs::from_value(&json!({
            "file_path": "/second",
            "path": "/first"
        }));
        assert_eq!(args.extract_path(), Some("/first"));
    }

    #[test]
    fn extract_path_none_when_missing() {
        let args = ToolArgs::from_value(&json!({"content": "hello"}));
        assert_eq!(args.extract_path(), None);
    }

    #[test]
    fn extract_path_skips_empty() {
        let args = ToolArgs::from_value(&json!({"path": "", "file": "/backup"}));
        assert_eq!(args.extract_path(), Some("/backup"));
    }

    // ── is_write_tool tests ─────────────────────────────────────

    #[test]
    fn recognises_write_tools() {
        assert!(is_write_tool("write"));
        assert!(is_write_tool("Write"));
        assert!(is_write_tool("EDIT"));
        assert!(is_write_tool("create_file"));
        assert!(is_write_tool("append"));
    }

    #[test]
    fn non_write_tools_pass() {
        assert!(!is_write_tool("read"));
        assert!(!is_write_tool("search"));
        assert!(!is_write_tool("exec")); // exec is not a write tool, but IS a shell tool
    }

    // ── is_shell_tool tests ─────────────────────────────────────

    #[test]
    fn recognises_shell_tools() {
        assert!(is_shell_tool("exec"));
        assert!(is_shell_tool("Exec"));
        assert!(is_shell_tool("SHELL"));
        assert!(is_shell_tool("bash"));
    }

    #[test]
    fn non_shell_tools_not_flagged() {
        assert!(!is_shell_tool("read"));
        assert!(!is_shell_tool("write"));
        assert!(!is_shell_tool("search"));
    }

    // ── check_tool_call: shell tools consult exec_policy ───────────────

    use alzina_governance::{GovernanceConfig, GovernanceLayer};
    use alzina_workspace::WorkspaceHandle;
    use std::sync::Arc;

    /// Build a GovernanceLayer with the four canonical profiles + the new
    /// `researcher` profile, mirroring `config/governance.toml` for the
    /// Norse persona. Used by the exec_policy tests below.
    fn layer_with_full_profiles() -> (tempfile::TempDir, GovernanceLayer) {
        let dir = tempfile::tempdir().unwrap();
        // GATE 3.5 ordering: the shell allowlist runs BEFORE exec_policy and
        // is fail-closed, so an agent with no allowlist is blocked there and
        // never reaches exec_policy. The exec_policy tests below want to
        // exercise the exec_policy gate specifically, so we give every agent
        // a fail-closed allowlist that NAMES the program those tests invoke
        // (`ls`). The allowlist then PERMITS that one command and control
        // falls through to exec_policy — the fail-closed way to isolate a
        // downstream gate. (Pre E-1-fix this used `mode = "disabled"` to
        // abstain; that abstain path was the fail-open trap and now denies.)
        for agent in ["vefr", "smidr", "urdr", "verdandi", "huginn"] {
            let adir = dir.path().join(format!("config/agents/{agent}"));
            std::fs::create_dir_all(&adir).unwrap();
            std::fs::write(
                adir.join("identity.toml"),
                "[tools]\nshell_allow = [\"ls\"]\n",
            )
            .unwrap();
        }
        let ws = Arc::new(WorkspaceHandle::open(dir.path().to_path_buf()).unwrap());
        let toml = r#"
            [bootstrap]
            agent_config_dir = "config/agents"

            [archetype_profiles]
            vefr = "orchestrator"
            smidr = "builder"
            urdr = "analyst"
            verdandi = "observer"
            huginn = "researcher"

            [profile_capabilities.analyst]
            max_write_tier = "free_write"
            exec_policy = "none"
            can_spawn = false
            can_govern = false
            network_policy = "denied"

            [profile_capabilities.builder]
            max_write_tier = "free_write"
            exec_policy = "sandboxed_readonly"
            can_spawn = false
            can_govern = false
            network_policy = "denied"

            [profile_capabilities.observer]
            max_write_tier = "free_write"
            exec_policy = "none"
            can_spawn = false
            can_govern = false
            network_policy = "denied"
            read_all_paths = true

            [profile_capabilities.orchestrator]
            max_write_tier = "governed"
            exec_policy = "gated"
            can_spawn = true
            can_govern = true
            network_policy = "denied"

            [profile_capabilities.researcher]
            max_write_tier = "free_write"
            exec_policy = "sandboxed_readonly"
            can_spawn = false
            can_govern = false
            network_policy = "allow_listed"
            read_all_paths = true
        "#;
        let config = GovernanceConfig::from_toml(toml).unwrap();
        let layer = GovernanceLayer::new(config, ws).unwrap();
        (dir, layer)
    }

    fn dummy_session() -> SessionId {
        SessionId::new()
    }

    /// Live-eval finding 2026-05-05: huginn (researcher / SandboxedReadonly)
    /// was being blocked from running Bash because shell tools returned
    /// `RequiresApproval` unconditionally, ignoring exec_policy entirely.
    /// Pin the new behaviour: shell decisions consult check_exec.
    #[test]
    fn shell_blocked_for_exec_policy_none() {
        let (_dir, gov) = layer_with_full_profiles();
        // `ls` is allowlisted (see layer_with_full_profiles) so the call
        // reaches the exec_policy gate, which blocks for exec_policy=none.
        let args = ToolArgs::from_value(&json!({"command": "ls"}));
        let decision = check_tool_call(&gov, "urdr", &dummy_session(), "Bash", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => {
                assert!(reason.contains("urdr"));
                assert!(reason.contains("exec_policy=none"));
            }
            other => panic!("expected Block for analyst, got {other:?}"),
        }
    }

    #[test]
    fn shell_allowed_for_exec_policy_sandboxed_readonly_builder() {
        let (_dir, gov) = layer_with_full_profiles();
        // Allowlisted `ls` passes GATE 3.5; exec_policy=sandboxed_readonly Allows.
        let args = ToolArgs::from_value(&json!({"command": "ls"}));
        let decision = check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn shell_allowed_for_exec_policy_sandboxed_readonly_researcher() {
        let (_dir, gov) = layer_with_full_profiles();
        let args = ToolArgs::from_value(&json!({"command": "ls"}));
        let decision = check_tool_call(&gov, "huginn", &dummy_session(), "bash", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn shell_requires_approval_for_exec_policy_gated() {
        let (_dir, gov) = layer_with_full_profiles();
        let args = ToolArgs::from_value(&json!({"command": "ls"}));
        let decision = check_tool_call(&gov, "vefr", &dummy_session(), "Bash", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::RequiresApproval);
    }

    #[test]
    fn shell_blocked_for_observer() {
        let (_dir, gov) = layer_with_full_profiles();
        let args = ToolArgs::from_value(&json!({"command": "ls"}));
        let decision = check_tool_call(&gov, "verdandi", &dummy_session(), "Bash", &args, None).unwrap();
        match decision {
            ToolDecision::Block(_) => {}
            other => panic!("expected Block for observer (exec_policy=none), got {other:?}"),
        }
    }

    /// Unknown agent → no `[tools].shell_allow` entry → static fail-closed
    /// EMPTY allowlist → blocked at GATE 3.5, BEFORE exec_policy. This is
    /// the strengthened defense-in-depth: even an agent that somehow
    /// reached a permissive profile cannot run shell without an explicit
    /// allowlist. We pass a real command so the allowlist sees a program
    /// token rather than the "empty command" early-out.
    #[test]
    fn shell_blocked_for_unknown_agent() {
        let (_dir, gov) = layer_with_full_profiles();
        let args = ToolArgs::from_value(&json!({"command": "git status"}));
        let decision = check_tool_call(&gov, "ghost", &dummy_session(), "exec", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => assert!(
                reason.contains("allowlist") || reason.contains("fail-closed"),
                "unknown agent must be blocked by the fail-closed allowlist: {reason}"
            ),
            other => panic!("expected Block for unknown agent, got {other:?}"),
        }
    }

    // ── Latent #2: identity [tools].denied tests ──────────────────────

    /// Build a GovernanceLayer whose workspace has an identity.toml for
    /// `agent_id` declaring a tool-name deny list. The agent is mapped
    /// to the `builder` profile (exec_policy = sandboxed_readonly) so
    /// Bash would otherwise be ALLOWED — the test pins that the identity
    /// deny-list overrides the profile default.
    fn layer_with_identity_denied(
        agent_id: &str,
        denied: &[&str],
    ) -> (tempfile::TempDir, GovernanceLayer) {
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().to_path_buf();
        // Write the identity file FIRST so the layer's eager loader
        // sees it at construction.
        let identity_dir = ws_root.join(format!("config/agents/{agent_id}"));
        std::fs::create_dir_all(&identity_dir).unwrap();
        let denied_toml = denied
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(", ");
        // A fail-closed allowlist naming `ls` lets the lowercase-`bash`
        // test reach the shell-policy gate with an allowlisted command
        // (the fail-closed gate would otherwise block before exec_policy).
        // (Pre E-1-fix this used `mode = "disabled"` to abstain — the
        // fail-open trap, now closed.)
        let identity_toml = format!(
            r#"archetype = "builder"

[tools]
denied = [{denied_toml}]
shell_allow = ["ls"]
"#
        );
        std::fs::write(identity_dir.join("identity.toml"), identity_toml).unwrap();

        let ws = Arc::new(WorkspaceHandle::open(ws_root).unwrap());
        let gov_toml = format!(
            r#"
            [bootstrap]
            agent_config_dir = "config/agents"

            [archetype_profiles]
            {agent_id} = "builder"

            [profile_capabilities.builder]
            max_write_tier = "free_write"
            exec_policy = "sandboxed_readonly"
            can_spawn = false
            can_govern = false
            network_policy = "denied"
            "#
        );
        let config = GovernanceConfig::from_toml(&gov_toml).unwrap();
        let layer = GovernanceLayer::new(config, ws).unwrap();
        (dir, layer)
    }

    #[test]
    fn identity_denied_tool_blocks_even_when_profile_would_allow() {
        // builder profile = sandboxed_readonly = Bash normally Allow.
        // Identity-level [tools].denied = ["Bash"] must override.
        let (_dir, gov) = layer_with_identity_denied("smidr", &["Bash"]);
        assert_eq!(gov.denied_tools_for("smidr"), &["Bash".to_string()]);
        let args = ToolArgs::from_value(&json!({}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => {
                assert!(reason.contains("smidr"), "reason names the agent");
                assert!(reason.contains("Bash"), "reason names the tool");
                assert!(
                    reason.contains("identity"),
                    "reason cites the identity layer so operators know where to look"
                );
            }
            other => panic!("expected identity-deny Block, got {other:?}"),
        }
    }

    #[test]
    fn identity_denied_match_is_case_sensitive() {
        // [tools].denied = ["Bash"]. A lowercase "bash" tool call should
        // NOT be caught by identity deny (it falls through to the shell
        // policy check, which Allows for builder). We keep this strict
        // so operators see exactly what they configured.
        let (_dir, gov) = layer_with_identity_denied("smidr", &["Bash"]);
        // `ls` is allowlisted (see layer_with_identity_denied) so the call
        // reaches the shell-policy gate; lowercase `bash` slips the
        // case-sensitive identity deny and exec_policy (builder) Allows.
        let args = ToolArgs::from_value(&json!({"command": "ls"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "bash", &args, None).unwrap();
        assert_eq!(
            decision,
            ToolDecision::Allow,
            "lowercase 'bash' bypasses 'Bash' identity-deny (case-sensitive match)"
        );
    }

    #[test]
    fn identity_denied_does_not_affect_other_tools() {
        // [tools].denied = ["Bash"]. A non-denied tool like "Read"
        // should pass through identity check and reach the non-write
        // pass-through.
        let (_dir, gov) = layer_with_identity_denied("smidr", &["Bash"]);
        let args = ToolArgs::from_value(&json!({}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Read", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn denied_tools_empty_for_agent_without_identity() {
        let (_dir, gov) = layer_with_full_profiles();
        // huginn's identity.toml declares only `shell_allow` (no `denied`),
        // so the deny-list accessor returns an empty slice — no block from
        // the identity deny gate. `ls` is allowlisted, so the call reaches
        // exec_policy (researcher=sandboxed_readonly) → Allow.
        assert_eq!(gov.denied_tools_for("huginn").len(), 0);
        let args = ToolArgs::from_value(&json!({"command": "ls"}));
        let decision =
            check_tool_call(&gov, "huginn", &dummy_session(), "Bash", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    // ── write_op_for_tool tests ─────────────────────────────────

    #[test]
    fn append_maps_correctly() {
        assert_eq!(write_op_for_tool("append"), WriteOp::Append);
    }

    #[test]
    fn create_file_maps_correctly() {
        assert_eq!(write_op_for_tool("create_file"), WriteOp::Create);
    }

    #[test]
    fn edit_maps_to_modify() {
        assert_eq!(write_op_for_tool("edit"), WriteOp::Modify);
    }

    #[test]
    fn unknown_write_tool_defaults_to_modify() {
        assert_eq!(write_op_for_tool("write"), WriteOp::Modify);
    }

    // ── Integration tests with GovernanceLayer ──────────────────
    //
    // Full integration tests require GovernanceLayer with a workspace.
    // These are placed in the validation test suite (tests/validation/).
    // Unit tests above cover the interceptor logic in isolation.

    // ── Assigned-dir policy tests ──────────────────────────────────────

    /// Build a governance layer plus the dir on disk so file-existence
    /// checks return real answers. `layer_with_full_profiles` already
    /// gives us a tempdir workspace; we just create the assigned dir
    /// inside it before each policy test.
    fn layer_with_assigned_dir(
        rel_dir: &str,
    ) -> (tempfile::TempDir, GovernanceLayer, String) {
        let (dir, gov) = layer_with_full_profiles();
        let abs = dir.path().join(rel_dir);
        std::fs::create_dir_all(&abs).unwrap();
        (dir, gov, rel_dir.to_string())
    }

    #[test]
    fn assigned_dir_allows_write_to_fresh_filename() {
        let (_dir, gov, ad) = layer_with_assigned_dir("artifacts/unattached/abc12345");
        let args = ToolArgs::from_value(&json!({
            "file_path": "artifacts/unattached/abc12345/huginn-findings.md"
        }));
        let decision = check_tool_call(
            &gov,
            "huginn",
            &dummy_session(),
            "Write",
            &args,
            Some(&ad),
        )
        .unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn assigned_dir_blocks_write_overwriting_existing_file() {
        let (dir, gov, ad) = layer_with_assigned_dir("artifacts/unattached/abc12345");
        // Pre-create the target so `workspace.exists()` returns true.
        std::fs::write(
            dir.path().join("artifacts/unattached/abc12345/huginn-findings.md"),
            "prior",
        )
        .unwrap();

        let args = ToolArgs::from_value(&json!({
            "file_path": "artifacts/unattached/abc12345/huginn-findings.md"
        }));
        let decision = check_tool_call(
            &gov,
            "huginn",
            &dummy_session(),
            "Write",
            &args,
            Some(&ad),
        )
        .unwrap();
        match decision {
            ToolDecision::Block(reason) => {
                assert!(reason.contains("already exists"));
                assert!(reason.contains("no-overwrite"));
            }
            other => panic!("expected Block for overwrite, got {other:?}"),
        }
    }

    #[test]
    fn assigned_dir_blocks_edit_inside_dir() {
        let (_dir, gov, ad) = layer_with_assigned_dir("artifacts/unattached/abc12345");
        let args = ToolArgs::from_value(&json!({
            "file_path": "artifacts/unattached/abc12345/huginn-findings.md"
        }));
        let decision = check_tool_call(
            &gov,
            "huginn",
            &dummy_session(),
            "Edit",
            &args,
            Some(&ad),
        )
        .unwrap();
        match decision {
            ToolDecision::Block(reason) => {
                assert!(reason.contains("append-only"));
                assert!(reason.contains("Edit"));
            }
            other => panic!("expected Block for Edit, got {other:?}"),
        }
    }

    #[test]
    fn assigned_dir_falls_through_for_paths_outside_dir() {
        let (_dir, gov, ad) = layer_with_assigned_dir("artifacts/unattached/abc12345");
        // Path is outside the assigned dir — policy returns None and
        // existing tier logic takes over (smidr is `free_write` so
        // an arbitrary workspace write is allowed).
        let args = ToolArgs::from_value(&json!({
            "file_path": "scratch/notes.md"
        }));
        let decision = check_tool_call(
            &gov,
            "smidr",
            &dummy_session(),
            "Write",
            &args,
            Some(&ad),
        )
        .unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn assigned_dir_accepts_absolute_paths_under_workspace_root() {
        let (dir, gov, ad) = layer_with_assigned_dir("artifacts/unattached/abc12345");
        let absolute = dir
            .path()
            .join("artifacts/unattached/abc12345/skuld-criteria.md")
            .to_string_lossy()
            .into_owned();
        let args = ToolArgs::from_value(&json!({ "file_path": absolute }));
        let decision = check_tool_call(
            &gov,
            "skuld",
            &dummy_session(),
            "Write",
            &args,
            Some(&ad),
        )
        .unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn assigned_dir_blocks_bash_command_writing_into_dir() {
        let (_dir, gov, ad) = layer_with_assigned_dir("artifacts/unattached/abc12345");
        let args = ToolArgs::from_value(&json!({
            "command": "echo hello > artifacts/unattached/abc12345/sneaky.md"
        }));
        // smidr has exec_policy = sandboxed_readonly → would normally Allow.
        let decision = check_tool_call(
            &gov,
            "smidr",
            &dummy_session(),
            "Bash",
            &args,
            Some(&ad),
        )
        .unwrap();
        match decision {
            ToolDecision::Block(reason) => {
                assert!(reason.contains("Bash"));
                assert!(reason.contains("assigned output dir"));
            }
            other => panic!("expected Block for Bash writing into dir, got {other:?}"),
        }
    }

    #[test]
    fn assigned_dir_allows_bash_command_not_touching_dir() {
        let (_dir, gov, ad) = layer_with_assigned_dir("artifacts/unattached/abc12345");
        let args = ToolArgs::from_value(&json!({ "command": "ls scratch/" }));
        let decision = check_tool_call(
            &gov,
            "smidr",
            &dummy_session(),
            "Bash",
            &args,
            Some(&ad),
        )
        .unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn no_assigned_dir_preserves_existing_behaviour() {
        let (_dir, gov) = layer_with_full_profiles();
        // `ls` is allowlisted; with no assigned dir the call falls through
        // GATE 3.5 to exec_policy (builder=sandboxed_readonly) → Allow.
        let args = ToolArgs::from_value(&json!({"command": "ls"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    // ── Deny-list tests ───────────────────────────────────────────────

    /// Build a layer whose `denied_path_globs` covers the canonical
    /// production set. Other config matches `layer_with_full_profiles`
    /// (kept here for self-contained deny-list tests).
    fn layer_with_deny_list() -> (tempfile::TempDir, GovernanceLayer) {
        let dir = tempfile::tempdir().unwrap();
        // As in layer_with_full_profiles: give agents a fail-closed
        // allowlist that NAMES the programs the deny-list "Allow"
        // assertions invoke (`ls`, `cat`) so those commands reach the
        // deny-list / exec_policy gates. The deny-list gate runs BEFORE the
        // allowlist, so deny-list blocks fire regardless of the allowlist;
        // the named programs only matter for the "Allow" assertions (shell
        // commands that touch no denied path). (Pre E-1-fix this used
        // `mode = "disabled"` to abstain — the fail-open trap, now closed.)
        for agent in ["smidr", "huginn"] {
            let adir = dir.path().join(format!("config/agents/{agent}"));
            std::fs::create_dir_all(&adir).unwrap();
            std::fs::write(
                adir.join("identity.toml"),
                "[tools]\nshell_allow = [\"ls\", \"cat\"]\n",
            )
            .unwrap();
        }
        let ws = Arc::new(WorkspaceHandle::open(dir.path().to_path_buf()).unwrap());
        // NOTE: top-level keys (denied_path_globs) MUST precede any table
        // header in TOML, otherwise they get parsed into that table.
        let toml = r#"
            denied_path_globs = [
              ".env",
              ".env.*",
              "secrets/**",
              "credentials/**",
              "**/*.key",
              "**/*.pem",
              ".planning/**",
            ]

            [bootstrap]
            agent_config_dir = "config/agents"

            [archetype_profiles]
            smidr = "builder"
            huginn = "researcher"

            [profile_capabilities.builder]
            max_write_tier = "free_write"
            exec_policy = "sandboxed_readonly"
            can_spawn = false
            can_govern = false
            network_policy = "denied"

            [profile_capabilities.researcher]
            max_write_tier = "free_write"
            exec_policy = "sandboxed_readonly"
            can_spawn = false
            can_govern = false
            network_policy = "allow_listed"
            read_all_paths = true
        "#;
        let config = GovernanceConfig::from_toml(toml).unwrap();
        let layer = GovernanceLayer::new(config, ws).unwrap();
        (dir, layer)
    }

    #[test]
    fn deny_list_blocks_read_of_dotenv() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({"file_path": ".env"}));
        let decision =
            check_tool_call(&gov, "huginn", &dummy_session(), "Read", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => {
                assert!(reason.contains(".env"));
                assert!(reason.contains("deny-list"));
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[test]
    fn deny_list_blocks_read_of_planning_dir() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({
            "file_path": ".planning/phases/01/PLAN.md"
        }));
        let decision =
            check_tool_call(&gov, "huginn", &dummy_session(), "Read", &args, None).unwrap();
        matches_block(&decision, "deny-list");
    }

    #[test]
    fn deny_list_blocks_write_to_secrets() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({
            "file_path": "secrets/api-token"
        }));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Write", &args, None).unwrap();
        matches_block(&decision, "deny-list");
    }

    #[test]
    fn deny_list_blocks_key_file_anywhere() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({
            "file_path": "nested/deep/cert.key"
        }));
        let decision =
            check_tool_call(&gov, "huginn", &dummy_session(), "read", &args, None).unwrap();
        matches_block(&decision, "deny-list");
    }

    #[test]
    fn deny_list_allows_read_of_safe_path() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({"file_path": "README.md"}));
        let decision =
            check_tool_call(&gov, "huginn", &dummy_session(), "Read", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn deny_list_blocks_shell_command_mentioning_planning() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({
            "command": "cat .planning/PROJECT.md"
        }));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => {
                assert!(reason.contains(".planning/"));
                assert!(reason.contains("deny-list"));
            }
            other => panic!("expected Block for shell mentioning .planning/, got {other:?}"),
        }
    }

    #[test]
    fn deny_list_blocks_shell_command_mentioning_dotenv() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({"command": "cat .env"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        matches_block(&decision, "deny-list");
    }

    #[test]
    fn deny_list_allows_shell_command_with_no_denied_substring() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({"command": "ls src/"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    /// Pure-wildcard patterns (e.g. `**/*.key`) have no literal prefix,
    /// so shell-side enforcement skips them. The path-arg branches still
    /// catch them when an agent uses Read/Write with a concrete path.
    #[test]
    fn deny_list_shell_skips_pure_wildcard_patterns() {
        let (_dir, gov) = layer_with_deny_list();
        let args = ToolArgs::from_value(&json!({"command": "cat foo.key"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        // `**/*.key` has empty literal prefix → no shell block. Surfaces
        // the known gap explicitly so a future shell-token tightening
        // is a recognisable behavioural change.
        assert_eq!(decision, ToolDecision::Allow);
    }

    #[test]
    fn deny_list_absolute_path_under_workspace_is_relativised() {
        let (dir, gov) = layer_with_deny_list();
        let absolute = dir.path().join(".env").to_string_lossy().into_owned();
        let args = ToolArgs::from_value(&json!({"file_path": absolute}));
        let decision =
            check_tool_call(&gov, "huginn", &dummy_session(), "Read", &args, None).unwrap();
        matches_block(&decision, "deny-list");
    }

    #[test]
    fn deny_list_empty_config_preserves_existing_behaviour() {
        let (_dir, gov) = layer_with_full_profiles();
        let args = ToolArgs::from_value(&json!({"file_path": ".env"}));
        let decision =
            check_tool_call(&gov, "huginn", &dummy_session(), "Read", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    // ── Shell allowlist gate (GATE 3.5) tests ─────────────────────────

    /// Build a layer where `agent_id` (builder, sandboxed_readonly → would
    /// otherwise Allow shell) has a fail-closed `[tools].shell_allow` array.
    fn layer_with_shell_allow(
        agent_id: &str,
        allow: &[&str],
    ) -> (tempfile::TempDir, GovernanceLayer) {
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().to_path_buf();
        let adir = ws_root.join(format!("config/agents/{agent_id}"));
        std::fs::create_dir_all(&adir).unwrap();
        let list = allow
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(", ");
        std::fs::write(
            adir.join("identity.toml"),
            format!("archetype = \"builder\"\n\n[tools]\nshell_allow = [{list}]\n"),
        )
        .unwrap();
        let ws = Arc::new(WorkspaceHandle::open(ws_root).unwrap());
        let gov_toml = format!(
            r#"
            [bootstrap]
            agent_config_dir = "config/agents"

            [archetype_profiles]
            {agent_id} = "builder"

            [profile_capabilities.builder]
            max_write_tier = "free_write"
            exec_policy = "sandboxed_readonly"
            can_spawn = false
            can_govern = false
            network_policy = "denied"
            "#
        );
        let config = GovernanceConfig::from_toml(&gov_toml).unwrap();
        let layer = GovernanceLayer::new(config, ws).unwrap();
        (dir, layer)
    }

    /// AC-FC: agent with no `[tools].shell_allow` is fail-closed — even
    /// though its profile (sandboxed_readonly) would Allow shell, the
    /// empty allowlist blocks every command BEFORE exec_policy.
    #[test]
    fn allowlist_absent_blocks_shell_despite_permissive_profile() {
        // layer_with_full_profiles gives smidr a *disabled* allowlist, so
        // build a bespoke builder agent with NO shell_allow at all.
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(WorkspaceHandle::open(dir.path().to_path_buf()).unwrap());
        let toml = r#"
            [archetype_profiles]
            lonework = "builder"

            [profile_capabilities.builder]
            max_write_tier = "free_write"
            exec_policy = "sandboxed_readonly"
            can_spawn = false
            can_govern = false
            network_policy = "denied"
        "#;
        let config = GovernanceConfig::from_toml(toml).unwrap();
        let gov = GovernanceLayer::new(config, ws).unwrap();
        let args = ToolArgs::from_value(&json!({"command": "ls -la"}));
        let decision =
            check_tool_call(&gov, "lonework", &dummy_session(), "Bash", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => assert!(
                reason.contains("not on this agent's allowlist")
                    || reason.contains("fail-closed"),
                "unexpected reason: {reason}"
            ),
            other => panic!("expected fail-closed Block, got {other:?}"),
        }
    }

    /// AC-AL: an allowlisted program reaches (and passes) the exec_policy gate.
    #[test]
    fn allowlist_permits_listed_program() {
        let (_dir, gov) = layer_with_shell_allow("smidr", &["git", "ls"]);
        let args = ToolArgs::from_value(&json!({"command": "git status"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        assert_eq!(decision, ToolDecision::Allow);
    }

    /// AC-DN: a non-allowlisted program is blocked by the allowlist gate.
    #[test]
    fn allowlist_blocks_unlisted_program() {
        let (_dir, gov) = layer_with_shell_allow("smidr", &["git"]);
        let args = ToolArgs::from_value(&json!({"command": "rm -rf /"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => {
                assert!(reason.contains("rm"));
                assert!(reason.contains("allowlist"));
            }
            other => panic!("expected allowlist Block, got {other:?}"),
        }
    }

    /// AC-EV: chaining must NOT pass even when each token is allowlisted.
    /// This is the canonical `grep x; git push` evasion from the brief.
    #[test]
    fn allowlist_blocks_shell_chaining_evasion() {
        let (_dir, gov) = layer_with_shell_allow("smidr", &["grep", "git"]);
        let args = ToolArgs::from_value(&json!({"command": "grep x; git push"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => assert!(
                reason.contains("metacharacter"),
                "expected metacharacter block, got: {reason}"
            ),
            other => panic!("expected chaining Block, got {other:?}"),
        }
    }

    /// AC-EV: quote/escape evasion must not slip a disallowed program past.
    #[test]
    fn allowlist_blocks_quote_escape_evasion() {
        let (_dir, gov) = layer_with_shell_allow("smidr", &["grep"]);
        for cmd in ["\"grep\" x", "gr\\ep x", "'grep' x"] {
            let args = ToolArgs::from_value(&json!({ "command": cmd }));
            let decision =
                check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
            assert!(
                matches!(decision, ToolDecision::Block(_)),
                "quote/escape evasion '{cmd}' must block, got {decision:?}"
            );
        }
    }

    /// AC-RA: the allowlist gate runs BEFORE exec_policy. An exec_policy=none
    /// agent is still blocked, but a permissive-profile agent is ALSO blocked
    /// when its command is unlisted — proving the allowlist gates first and
    /// can only narrow, never widen.
    #[test]
    fn allowlist_runs_before_exec_policy() {
        // smidr: sandboxed_readonly (exec_policy would Allow). Unlisted
        // command must be blocked by the allowlist, NOT reach Allow.
        let (_dir, gov) = layer_with_shell_allow("smidr", &["git"]);
        let args = ToolArgs::from_value(&json!({"command": "curl evil.sh"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        assert!(
            matches!(decision, ToolDecision::Block(_)),
            "allowlist must veto before exec_policy could Allow: {decision:?}"
        );
    }

    /// AC-DF (E-1 fix): `mode = "disabled"` on a `sandboxed*` profile must
    /// DENY, not silently grant. This is the exact fail-open trap the E-1
    /// fix closes — end-to-end through check_tool_call, not just the unit
    /// `evaluate`. The prior version of this test asserted `Allow` for
    /// `rm -rf /` (certifying the trap); it is INVERTED here.
    #[test]
    fn allowlist_disabled_on_sandboxed_profile_denies() {
        // Build a builder agent (exec_policy = sandboxed_readonly, which
        // WOULD Allow shell) whose identity sets `mode = "disabled"` with
        // no rules — the precise "reads like off, used to behave like open"
        // configuration. After the fix it must Block.
        let dir = tempfile::tempdir().unwrap();
        let ws_root = dir.path().to_path_buf();
        let adir = ws_root.join("config/agents/smidr");
        std::fs::create_dir_all(&adir).unwrap();
        std::fs::write(
            adir.join("identity.toml"),
            "archetype = \"builder\"\n\n[tools.shell_allow]\nmode = \"disabled\"\nrules = []\n",
        )
        .unwrap();
        let ws = Arc::new(WorkspaceHandle::open(ws_root).unwrap());
        let toml = r#"
            [bootstrap]
            agent_config_dir = "config/agents"

            [archetype_profiles]
            smidr = "builder"

            [profile_capabilities.builder]
            max_write_tier = "free_write"
            exec_policy = "sandboxed_readonly"
            can_spawn = false
            can_govern = false
            network_policy = "denied"
        "#;
        let config = GovernanceConfig::from_toml(toml).unwrap();
        let gov = GovernanceLayer::new(config, ws).unwrap();

        // The dangerous command the trap used to wave through.
        let args = ToolArgs::from_value(&json!({"command": "rm -rf /"}));
        let decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &args, None).unwrap();
        match decision {
            ToolDecision::Block(reason) => assert!(
                reason.contains("allowlist") || reason.contains("fail-closed"),
                "disabled+empty on a sandboxed profile must be blocked by the \
                 fail-closed allowlist (was a silent broad grant): {reason}"
            ),
            other => panic!(
                "E-1 regression: disabled+sandboxed must DENY, got {other:?} \
                 (this is the fail-open trap)"
            ),
        }

        // An ordinary command is denied too — `disabled` no longer abstains.
        let ls = ToolArgs::from_value(&json!({"command": "ls"}));
        let ls_decision =
            check_tool_call(&gov, "smidr", &dummy_session(), "Bash", &ls, None).unwrap();
        assert!(
            matches!(ls_decision, ToolDecision::Block(_)),
            "disabled+empty denies every command, not just dangerous ones: {ls_decision:?}"
        );
    }

    fn matches_block(decision: &ToolDecision, substring: &str) {
        match decision {
            ToolDecision::Block(reason) => assert!(
                reason.contains(substring),
                "Block reason '{reason}' missing substring '{substring}'"
            ),
            other => panic!("expected Block(_) containing '{substring}', got {other:?}"),
        }
    }

    // ── pattern_literal_prefix tests ─────────────────────────────────

    #[test]
    fn literal_prefix_extracts_full_literal() {
        assert_eq!(pattern_literal_prefix(".env"), ".env");
    }

    #[test]
    fn literal_prefix_stops_at_star() {
        assert_eq!(pattern_literal_prefix(".env.*"), ".env.");
        assert_eq!(pattern_literal_prefix("secrets/**"), "secrets/");
        assert_eq!(pattern_literal_prefix(".planning/**"), ".planning/");
    }

    #[test]
    fn literal_prefix_empty_when_no_prefix() {
        assert_eq!(pattern_literal_prefix("**/*.key"), "");
        assert_eq!(pattern_literal_prefix("*.pem"), "");
        assert_eq!(pattern_literal_prefix("[abc].txt"), "");
        assert_eq!(pattern_literal_prefix("{foo,bar}/x"), "");
    }

    #[test]
    fn literal_prefix_stops_at_question_mark() {
        assert_eq!(pattern_literal_prefix("file?.txt"), "file");
    }

    // ── validate_envelope_artifacts tests ──────────────────────────────

    /// Build a workspace + assigned-dir + populate `files` inside it.
    /// Returned tuple owns the tempdir so it lives for the test.
    fn workspace_with_dir(
        rel_dir: &str,
        files: &[&str],
    ) -> (tempfile::TempDir, Arc<WorkspaceHandle>, String) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(WorkspaceHandle::open(dir.path().to_path_buf()).unwrap());
        let abs = dir.path().join(rel_dir);
        std::fs::create_dir_all(&abs).unwrap();
        for f in files {
            std::fs::write(abs.join(f), "x").unwrap();
        }
        (dir, ws, rel_dir.to_string())
    }

    #[test]
    fn validate_artifacts_allows_existing_relative_paths() {
        let (_dir, ws, ad) =
            workspace_with_dir("artifacts/unattached/abc12345", &["huginn-findings.md"]);
        let artifacts = vec![std::path::PathBuf::from(
            "artifacts/unattached/abc12345/huginn-findings.md",
        )];
        let out = validate_envelope_artifacts(&artifacts, &ws, &ad).unwrap();
        assert_eq!(out, artifacts);
    }

    #[test]
    fn validate_artifacts_rewrites_absolute_to_relative_under_workspace() {
        let (dir, ws, ad) =
            workspace_with_dir("artifacts/unattached/abc12345", &["skuld-criteria.md"]);
        let absolute = dir
            .path()
            .join("artifacts/unattached/abc12345/skuld-criteria.md");
        let artifacts = vec![absolute];
        let out = validate_envelope_artifacts(&artifacts, &ws, &ad).unwrap();
        assert_eq!(
            out,
            vec![std::path::PathBuf::from(
                "artifacts/unattached/abc12345/skuld-criteria.md"
            )]
        );
    }

    #[test]
    fn validate_artifacts_rejects_missing_file() {
        let (_dir, ws, ad) = workspace_with_dir("artifacts/unattached/abc12345", &[]);
        let artifacts = vec![std::path::PathBuf::from(
            "artifacts/unattached/abc12345/never-written.yaml",
        )];
        let errors = validate_envelope_artifacts(&artifacts, &ws, &ad).unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("does not exist on disk"));
        assert!(errors[0].contains("never-written.yaml"));
    }

    #[test]
    fn validate_artifacts_rejects_path_outside_assigned_dir() {
        let (dir, ws, ad) =
            workspace_with_dir("artifacts/unattached/abc12345", &["inside.md"]);
        // File exists, just not under the assigned dir.
        std::fs::create_dir_all(dir.path().join("scratch")).unwrap();
        std::fs::write(dir.path().join("scratch/elsewhere.md"), "x").unwrap();
        let artifacts = vec![std::path::PathBuf::from("scratch/elsewhere.md")];
        let errors = validate_envelope_artifacts(&artifacts, &ws, &ad).unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("outside assigned output dir"));
    }

    #[test]
    fn validate_artifacts_rejects_path_above_workspace_root() {
        let (_dir, ws, ad) = workspace_with_dir("artifacts/unattached/abc12345", &[]);
        let artifacts = vec![std::path::PathBuf::from("/etc/passwd")];
        let errors = validate_envelope_artifacts(&artifacts, &ws, &ad).unwrap_err();
        assert_eq!(errors.len(), 1);
        // Reason: /etc is not under the tempdir workspace root, so neither
        // path_is_inside nor strip_prefix can place it. Either error
        // string is acceptable — both name the constraint.
        assert!(
            errors[0].contains("not under workspace root")
                || errors[0].contains("outside assigned output dir"),
            "unexpected error: {}",
            errors[0]
        );
    }

    #[test]
    fn validate_artifacts_aggregates_multiple_failures() {
        let (_dir, ws, ad) =
            workspace_with_dir("artifacts/unattached/abc12345", &["exists.md"]);
        let artifacts = vec![
            std::path::PathBuf::from("artifacts/unattached/abc12345/exists.md"),
            std::path::PathBuf::from("artifacts/unattached/abc12345/missing.md"),
            std::path::PathBuf::from("scratch/elsewhere.md"),
        ];
        let errors = validate_envelope_artifacts(&artifacts, &ws, &ad).unwrap_err();
        assert_eq!(errors.len(), 2, "two failures expected, one pass: {errors:?}");
    }

    #[test]
    fn validate_artifacts_empty_input_is_ok() {
        let (_dir, ws, ad) = workspace_with_dir("artifacts/unattached/abc12345", &[]);
        let out = validate_envelope_artifacts(&[], &ws, &ad).unwrap();
        assert!(out.is_empty());
    }
}
