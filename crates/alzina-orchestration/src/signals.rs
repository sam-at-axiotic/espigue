//! Post-completion signal processing — routes extracted signals to their
//! respective handlers (learnings merger, emergence triage, audit).
//!
//! This module is the orchestration-side counterpart to
//! `alzina-governance::envelope::SignalRouter`. The router extracts signals;
//! this module acts on them.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use alzina_core::envelope::Signal;
use alzina_core::error::{AlzinaError, AlzinaResult};
use alzina_core::identity::{AgentId, WeaveId};
use alzina_governance::{GovernanceLayer, LearningsMerger, MergeOutcome};
use alzina_memory::{SignalRecordRow, SignalRecordsStore};
use alzina_workspace::WorkspaceHandle;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Maximum size (in bytes) for CONTEXT_UPDATE content. 4KB.
const MAX_CONTEXT_UPDATE_SIZE: usize = 4096;

/// Maximum size (in bytes) for emergence/tension/next-step/cross-weave content. 4KB.
const MAX_EMERGENCE_SIZE: usize = 4096;

/// Global counter for CrossWeaveReference signals processed.
static CROSS_WEAVE_REF_COUNT: AtomicU64 = AtomicU64::new(0);

/// Strip HTML tags from content.
fn strip_html_tags(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    result
}

/// Strip instruction-like patterns (potential prompt injection).
fn strip_instruction_patterns(input: &str) -> String {
    input
        .lines()
        .filter(|line| {
            let trimmed = line.trim().to_lowercase();
            !(trimmed.starts_with("ignore previous")
                || trimmed.starts_with("disregard")
                || trimmed.starts_with("system:")
                || trimmed.starts_with("assistant:")
                || trimmed.starts_with("user:")
                || trimmed.starts_with("<|")
                || trimmed.starts_with("[inst]")
                || trimmed.starts_with("[/inst]")
                || trimmed.starts_with("<<sys>>"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Sanitise file paths in content (strip path traversal sequences).
///
/// WR-04 fix: the previous word-by-word approach discarded all directory
/// structure from absolute paths (too aggressive) while missing relative
/// traversal patterns like `../../etc/shadow` (too narrow). The replacement
/// targets actual threat vectors — path traversal sequences — without
/// destroying legitimate diagnostic path information.
fn sanitise_file_paths(input: &str) -> String {
    input.replace("../", "").replace("./", "")
}

/// Truncate content to a maximum byte length, respecting UTF-8 boundaries.
fn truncate_to_limit(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }
    let mut truncated = input[..end].to_owned();
    truncated.push_str(" [truncated]");
    truncated
}

/// Outcome of processing a batch of signals.
#[derive(Debug, Clone)]
pub struct SignalOutcome {
    /// Number of context updates merged.
    pub context_updates_merged: usize,
    /// Merge outcomes for context updates (if any).
    pub merge_outcomes: Vec<MergeOutcome>,
    /// Number of emergence signals triaged.
    pub emergences_triaged: usize,
    /// Number of tension signals logged.
    pub tensions_logged: usize,
    /// Number of next-step recommendations recorded.
    pub next_steps_recorded: usize,
    /// Signals that could not be processed (with reason).
    pub failures: Vec<SignalFailure>,
}

impl SignalOutcome {
    fn new() -> Self {
        Self {
            context_updates_merged: 0,
            merge_outcomes: Vec::new(),
            emergences_triaged: 0,
            tensions_logged: 0,
            next_steps_recorded: 0,
            failures: Vec::new(),
        }
    }

    /// Total signals successfully processed.
    pub fn total_processed(&self) -> usize {
        self.context_updates_merged
            + self.emergences_triaged
            + self.tensions_logged
            + self.next_steps_recorded
    }

    /// Whether all signals were processed without failure.
    pub fn all_succeeded(&self) -> bool {
        self.failures.is_empty()
    }
}

/// A signal that failed to process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalFailure {
    pub signal_type: String,
    pub error: String,
}

/// Process extracted signals from an agent's return envelope.
///
/// Routes each signal to its handler:
/// - `ContextUpdate` → `LearningsMerger::merge()` (uses runner's existing merger — LANDMINE 1 fix)
/// - `EmergenceDetected` → file in `well/emergence/` AND signal_records row (if signal_store provided)
/// - `TensionFlagged` → file in `well/tension/` AND signal_records row
/// - `NextStepRecommended` → file in `well/next-step/` AND signal_records row
/// - `CrossWeaveReference` → file in `well/cross-weave/` AND signal_records row
///
/// # Parameters
/// - `merger`: The runner's production LearningsMerger (LANDMINE 1 fix: do NOT construct a
///   detached FileLearningsStore — that would regress red-team A2 FTS5 indexing).
/// - `signal_store`: If `Some`, each signal also gets a row in `signal_records`. `None` for
///   callers that don't need persistence (e.g. tests without a DB pool).
/// - `envelope_id`: FK into signal_records (typically the spawn's session_id).
/// - `weave_id`: If `Some`, the signal_records row carries the weave association.
///
/// Non-fatal: individual signal failures are collected, not propagated.
/// The caller decides whether partial processing is acceptable.
///
/// Note: `governance` parameter is currently unused after the LANDMINE 1 refactor
/// but is kept for API stability (governance-aware signal routing is planned).
#[allow(clippy::too_many_arguments)]
pub async fn process_signals(
    signals: &[Signal],
    governance: &GovernanceLayer,
    workspace: &Arc<WorkspaceHandle>,
    agent_id: &AgentId,
    merger: &Arc<LearningsMerger>,
    signal_store: Option<&Arc<SignalRecordsStore>>,
    envelope_id: &str,
    weave_id: Option<&WeaveId>,
) -> AlzinaResult<SignalOutcome> {
    let mut outcome = SignalOutcome::new();

    if signals.is_empty() {
        debug!(agent_id = agent_id.as_str(), "no signals to process");
        return Ok(outcome);
    }

    // DELETED: inline FileLearningsStore + LearningsMerger::with_store construction (LANDMINE 1 fix)
    // The caller passes in the production LearningsMerger via `merger` parameter.
    let _ = governance; // kept for API stability; governance-aware routing planned

    for signal in signals {
        match signal {
            Signal::ContextUpdate { learning } => {
                // RT3-07: Sanitise and size-limit context update content
                let sanitised = strip_html_tags(learning);
                let sanitised = strip_instruction_patterns(&sanitised);
                let sanitised = truncate_to_limit(&sanitised, MAX_CONTEXT_UPDATE_SIZE);
                match merger.merge(agent_id, &sanitised).await {
                    Ok(merge_outcome) => {
                        if merge_outcome.has_merged() {
                            info!(
                                agent_id = agent_id.as_str(),
                                merged = merge_outcome.merged.len(),
                                "context update merged into learnings"
                            );
                        }
                        outcome.context_updates_merged += 1;
                        outcome.merge_outcomes.push(merge_outcome);
                        persist_signal_record(
                            signal_store,
                            envelope_id,
                            weave_id,
                            agent_id,
                            "context-update",
                            &sanitised,
                            "merged",
                            None,
                            &mut outcome,
                        )
                        .await;
                    }
                    Err(e) => {
                        warn!(
                            agent_id = agent_id.as_str(),
                            error = %e,
                            "failed to merge context update"
                        );
                        outcome.failures.push(SignalFailure {
                            signal_type: "ContextUpdate".into(),
                            error: e.to_string(),
                        });
                    }
                }
            }

            Signal::EmergenceDetected { content } => {
                handle_class_signal(
                    workspace,
                    signal_store,
                    envelope_id,
                    weave_id,
                    agent_id,
                    "emergence",
                    "well/emergence",
                    content,
                    &mut outcome,
                )
                .await;
            }

            Signal::TensionFlagged { location, content } => {
                let body = format!("location: {location}\n\n{content}");
                handle_class_signal(
                    workspace,
                    signal_store,
                    envelope_id,
                    weave_id,
                    agent_id,
                    "tension",
                    "well/tension",
                    &body,
                    &mut outcome,
                )
                .await;
            }

            Signal::NextStepRecommended { action } => {
                handle_class_signal(
                    workspace,
                    signal_store,
                    envelope_id,
                    weave_id,
                    agent_id,
                    "next-step",
                    "well/next-step",
                    action,
                    &mut outcome,
                )
                .await;
            }

            Signal::CrossWeaveReference {
                target_weave,
                content,
            } => {
                let body = format!(
                    "target_weave: {target}\n\n{content}",
                    target = target_weave.as_str()
                );
                let failures_before = outcome.failures.len();
                handle_class_signal(
                    workspace,
                    signal_store,
                    envelope_id,
                    weave_id,
                    agent_id,
                    "cross-weave",
                    "well/cross-weave",
                    &body,
                    &mut outcome,
                )
                .await;
                // IN-03 fix: only bump the global counter when the triage write
                // succeeded. Previously the counter was always bumped, making
                // it over-count when write_class_triage failed. Checking that
                // the failure list didn't grow is the most direct success signal.
                if outcome.failures.len() == failures_before {
                    let count = CROSS_WEAVE_REF_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                    info!(
                        agent_id = agent_id.as_str(),
                        target_weave = target_weave.as_str(),
                        total_cross_weave_refs = count,
                        "cross-weave reference noted"
                    );
                }
            }
        }
    }

    info!(
        agent_id = agent_id.as_str(),
        processed = outcome.total_processed(),
        failures = outcome.failures.len(),
        "signal processing complete"
    );

    Ok(outcome)
}

/// Generic per-class signal handler: sanitise → write triage file → insert signal_records row.
///
/// All 4 non-ContextUpdate signal classes flow through here for uniform sanitiser
/// handling (T-09-02-03 sanitiser parity) and fail-soft on store insert errors (T-09-02-04).
#[allow(clippy::too_many_arguments)]
async fn handle_class_signal(
    workspace: &Arc<WorkspaceHandle>,
    signal_store: Option<&Arc<SignalRecordsStore>>,
    envelope_id: &str,
    weave_id: Option<&WeaveId>,
    agent_id: &AgentId,
    class: &str,
    subdir: &str,
    content: &str,
    outcome: &mut SignalOutcome,
) {
    // T-09-02-03: uniform sanitiser stack across all 4 signal classes
    let sanitised = strip_html_tags(content);
    let sanitised = strip_instruction_patterns(&sanitised);
    let sanitised = sanitise_file_paths(&sanitised);
    let sanitised = truncate_to_limit(&sanitised, MAX_EMERGENCE_SIZE);

    match write_class_triage(workspace, agent_id, subdir, class, &sanitised) {
        Ok(path) => {
            // Bump per-class outcome counter
            match class {
                "emergence" => outcome.emergences_triaged += 1,
                "tension" => outcome.tensions_logged += 1,
                "next-step" => outcome.next_steps_recorded += 1,
                "cross-weave" => { /* CROSS_WEAVE_REF_COUNT bumped at call site */ }
                _ => {}
            }
            persist_signal_record(
                signal_store,
                envelope_id,
                weave_id,
                agent_id,
                class,
                &sanitised,
                "file-created",
                Some(&path),
                outcome,
            )
            .await;
        }
        Err(e) => {
            warn!(
                agent_id = agent_id.as_str(),
                class,
                error = %e,
                "failed to write {class} triage"
            );
            outcome.failures.push(SignalFailure {
                signal_type: class.to_string(),
                error: e.to_string(),
            });
        }
    }
}

/// Write a per-class triage file. Path: `{subdir}/{date_slug}-{agent_id}.md`.
///
/// Appends to an existing file (one file per agent per day) or creates a new
/// one. Returns the workspace-relative path for use as `signal_records.triage_path`.
///
/// Mirrors the old `write_emergence_triage` shape so `well/emergence/` behaviour
/// is unchanged. The function now handles all 4 non-ContextUpdate signal classes.
fn write_class_triage(
    workspace: &WorkspaceHandle,
    agent_id: &AgentId,
    subdir: &str,
    class: &str,
    content: &str,
) -> AlzinaResult<String> {
    let timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let date_slug = Utc::now().format("%Y-%m-%d");
    let path = format!("{subdir}/{date_slug}-{agent_id}.md");

    // Capitalise class for the Markdown header
    let class_title = {
        let mut s = class.to_string();
        if let Some(c) = s.get_mut(..1) {
            c.make_ascii_uppercase();
        }
        s
    };

    let entry = format!(
        "<!-- {class}: agent={agent_id}, ts={timestamp} -->\n\
         ## {class_title} — {agent_id}\n\n{content}\n\n---\n\n"
    );

    // Always append. workspace.append() creates the file and its parent dir
    // if missing (io.rs: OpenOptions::new().create(true).append(true) +
    // create_dir_all), so the old exists()-gated write() branch is unneeded.
    // It is also required: well/<signal>/ is Integrity-tier, where a
    // truncating workspace.write() is blocked but append() is allowed. Both
    // branches previously wrote the identical `entry`, so behaviour is
    // unchanged (one file per agent per day, appended per signal).
    workspace
        .append(&path, &entry)
        .map_err(|e| AlzinaError::Workspace(format!("failed to append {class} triage: {e}")))?;

    Ok(path)
}

/// Insert one signal_records row, fail-soft (T-09-02-04).
///
/// Failures are folded into `outcome.failures`, never propagated up.
/// The caller continues processing remaining signals regardless.
#[allow(clippy::too_many_arguments)]
async fn persist_signal_record(
    signal_store: Option<&Arc<SignalRecordsStore>>,
    envelope_id: &str,
    weave_id: Option<&WeaveId>,
    agent_id: &AgentId,
    class: &str,
    payload: &str,
    triage_action: &str,
    triage_path: Option<&str>,
    outcome: &mut SignalOutcome,
) {
    let Some(store) = signal_store else {
        return;
    };
    let row = SignalRecordRow {
        id: Uuid::new_v4().to_string(),
        weave_id: weave_id.map(|w| w.to_string()),
        signal_class: class.to_string(),
        envelope_id: envelope_id.to_string(),
        agent_id: agent_id.to_string(),
        payload: payload.to_string(),
        triage_action: triage_action.to_string(),
        triage_path: triage_path.map(String::from),
        created_at: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
    };
    if let Err(e) = store.insert(&row).await {
        warn!(class, error = %e, "signal_records insert failed (non-fatal)");
        outcome.failures.push(SignalFailure {
            signal_type: format!("{class}-persist"),
            error: e.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alzina_core::envelope::Signal;
    use alzina_core::identity::{AgentId, WeaveId};
    use alzina_governance::config::{GovernanceConfig, LearningsConfig};

    fn test_setup() -> (
        tempfile::TempDir,
        Arc<GovernanceLayer>,
        Arc<WorkspaceHandle>,
        Arc<LearningsMerger>,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(WorkspaceHandle::open(dir.path().to_path_buf()).unwrap());
        let config = GovernanceConfig::default();
        let governance = Arc::new(GovernanceLayer::new(config.clone(), ws.clone()).unwrap());

        let learnings_config = LearningsConfig::default();
        let store = Arc::new(alzina_governance::FileLearningsStore::new(
            ws.root()
                .join(&learnings_config.learnings_dir)
                .to_string_lossy()
                .to_string(),
            learnings_config.max_entries_per_agent,
        ));
        let merger = Arc::new(LearningsMerger::with_store(
            ws.clone(),
            learnings_config,
            alzina_core::canonical_domain_mapping(),
            store,
        ));

        (dir, governance, ws, merger)
    }

    // ── Convenience wrapper for tests that don't need signal_store ──────────

    async fn process_no_store(
        signals: &[Signal],
        governance: &GovernanceLayer,
        workspace: &Arc<WorkspaceHandle>,
        agent_id: &AgentId,
        merger: &Arc<LearningsMerger>,
    ) -> AlzinaResult<SignalOutcome> {
        process_signals(
            signals, governance, workspace, agent_id, merger, None, "test-env", None,
        )
        .await
    }

    // === Empty signals ===

    #[tokio::test]
    async fn empty_signals_returns_empty_outcome() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("test");
        let outcome = process_no_store(&[], &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        assert_eq!(outcome.total_processed(), 0);
        assert!(outcome.all_succeeded());
    }

    // === ContextUpdate → LearningsMerger ===

    #[tokio::test]
    async fn context_update_merges_learning() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("smidr");
        let signals = vec![Signal::ContextUpdate {
            learning: "Always validate config at load time, not at use time".into(),
        }];
        let outcome = process_no_store(&signals, &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        assert_eq!(outcome.context_updates_merged, 1);
        assert!(outcome.all_succeeded());

        // Verify file was written (store-backed: domain-mapped path)
        let domain_path = format!("{}/learnings/implementation/_index.md", ws.root().display());
        let content = std::fs::read_to_string(&domain_path).unwrap();
        assert!(content.contains("validate config at load time"));
    }

    #[tokio::test]
    async fn context_update_none_is_filtered() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("test");
        let signals = vec![Signal::ContextUpdate {
            learning: "none".into(),
        }];
        let outcome = process_no_store(&signals, &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        assert_eq!(outcome.context_updates_merged, 1);
        // Merger processed it but filtered the "none" entry
        assert!(outcome.merge_outcomes[0].merged.is_empty());
    }

    // === EmergenceDetected → triage file ===

    #[tokio::test]
    async fn emergence_writes_triage_file() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("muninn");
        let signals = vec![Signal::EmergenceDetected {
            content: "Hidden coupling between bootstrap and tier enforcement".into(),
        }];
        let outcome = process_no_store(&signals, &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        assert_eq!(outcome.emergences_triaged, 1);
        assert!(outcome.all_succeeded());

        // Verify triage file exists
        let entries = ws.list_dir("well/emergence").unwrap();
        assert!(!entries.is_empty());
        let content = ws.read(&format!("well/emergence/{}", entries[0])).unwrap();
        assert!(content.contains("Hidden coupling"));
        assert!(content.contains("muninn"));
    }

    // === TensionFlagged → triage file ===

    #[tokio::test]
    async fn tension_is_logged() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("kvasir");
        let signals = vec![Signal::TensionFlagged {
            location: "alzina-core::quality".into(),
            content: "QualityGate trait is async but all impls are sync".into(),
        }];
        let outcome = process_no_store(&signals, &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        assert_eq!(outcome.tensions_logged, 1);
        assert!(outcome.all_succeeded());

        // Verify triage file exists under well/tension/
        let entries = ws.list_dir("well/tension").unwrap();
        assert!(
            !entries.is_empty(),
            "expected well/tension/ file after TensionFlagged"
        );
    }

    // === NextStepRecommended → triage file ===

    #[tokio::test]
    async fn next_step_is_recorded() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("skuld");
        let signals = vec![Signal::NextStepRecommended {
            action: "Merge feature/phase3 to main".into(),
        }];
        let outcome = process_no_store(&signals, &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        assert_eq!(outcome.next_steps_recorded, 1);
        assert!(outcome.all_succeeded());

        // Verify triage file exists under well/next-step/
        let entries = ws.list_dir("well/next-step").unwrap();
        assert!(
            !entries.is_empty(),
            "expected well/next-step/ file after NextStepRecommended"
        );
    }

    // === CrossWeaveReference ===

    #[tokio::test]
    async fn cross_weave_reference_noted() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("urdhr");
        let signals = vec![Signal::CrossWeaveReference {
            target_weave: WeaveId::new("governance-hardening"),
            content: "TierEnforcer patterns from that weave apply here".into(),
        }];
        let outcome = process_no_store(&signals, &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        // Cross-weave refs write a triage file and are not failures
        assert!(outcome.all_succeeded());
        let entries = ws.list_dir("well/cross-weave").unwrap();
        assert!(
            !entries.is_empty(),
            "expected well/cross-weave/ file after CrossWeaveReference"
        );
    }

    // === Mixed signals ===

    #[tokio::test]
    async fn mixed_signals_all_processed() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("galdr");
        let signals = vec![
            Signal::ContextUpdate {
                learning: "Use bounded channels for streaming data processing".into(),
            },
            Signal::EmergenceDetected {
                content: "Config cascade creates hidden coupling".into(),
            },
            Signal::TensionFlagged {
                location: "module::path".into(),
                content: "tight coupling".into(),
            },
            Signal::NextStepRecommended {
                action: "Run integration tests".into(),
            },
        ];
        let outcome = process_no_store(&signals, &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        assert_eq!(outcome.context_updates_merged, 1);
        assert_eq!(outcome.emergences_triaged, 1);
        assert_eq!(outcome.tensions_logged, 1);
        assert_eq!(outcome.next_steps_recorded, 1);
        assert_eq!(outcome.total_processed(), 4);
        assert!(outcome.all_succeeded());
    }

    // === Emergence appends to existing file ===

    #[tokio::test]
    async fn emergence_appends_to_existing_triage() {
        let (_dir, governance, ws, merger) = test_setup();
        let agent = AgentId::new("test");

        // Process two emergence signals — both should end up in the same file
        let signals1 = vec![Signal::EmergenceDetected {
            content: "First emergence finding".into(),
        }];
        let signals2 = vec![Signal::EmergenceDetected {
            content: "Second emergence finding".into(),
        }];

        process_no_store(&signals1, &governance, &ws, &agent, &merger)
            .await
            .unwrap();
        process_no_store(&signals2, &governance, &ws, &agent, &merger)
            .await
            .unwrap();

        let entries = ws.list_dir("well/emergence").unwrap();
        assert_eq!(entries.len(), 1); // Same day, same agent → same file
        let content = ws.read(&format!("well/emergence/{}", entries[0])).unwrap();
        assert!(content.contains("First emergence"));
        assert!(content.contains("Second emergence"));
    }

    // === write_class_triage unit test (replaces old write_emergence_triage test) ===

    #[test]
    fn class_triage_file_structure() {
        let dir = tempfile::tempdir().unwrap();
        let ws = WorkspaceHandle::open(dir.path().to_path_buf()).unwrap();
        let agent = AgentId::new("smidr");

        write_class_triage(
            &ws,
            &agent,
            "well/emergence",
            "emergence",
            "Test emergence content",
        )
        .unwrap();

        let entries = ws.list_dir("well/emergence").unwrap();
        assert_eq!(entries.len(), 1);
        let content = ws.read(&format!("well/emergence/{}", entries[0])).unwrap();
        assert!(content.contains("<!-- emergence:"));
        assert!(content.contains("## Emergence — smidr"));
        assert!(content.contains("Test emergence content"));
    }
}
