//! Shared test helpers for alzina-orchestration unit tests.
//!
//! Also exposed under the `test-harness` feature for workspace-tier integration
//! tests (e.g. `tests/cancel_ladder_*.rs`) that need deterministic timing
//! without invoking the sidecar.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use alzina_core::identity::AgentId;
use alzina_core::{AlzinaError, AlzinaResult, TemplateEngine};
use alzina_governance::config::{GovernanceConfig, LearningsConfig};
use alzina_governance::facade::HookSet;
use alzina_governance::{BootstrapEngine, GovernanceLayer, LearningsMerger};
use alzina_workspace::WorkspaceHandle;

use crate::runner::alzina_runner::{AgentExecutor, AlzinaRunner};
use crate::session::hierarchy::SessionHierarchy;

pub fn well_formed_envelope() -> String {
    r#"Analysis complete.

STATUS: complete
ARTIFACTS: artifacts/output.md
SIGNAL: analysis done
TENSIONS: none
EMERGENT: hidden coupling detected
CONTEXT_UPDATE: always validate config at load time"#
        .to_string()
}

/// Scope-aware envelope builder. Currently identical body to
/// [`well_formed_envelope`] — scope lives at the wire/audit boundary
/// (R-WEAVE-SCOPE-001), not in the envelope payload. Reserved for
/// future scope-bearing content.
///
/// Per D15-01 Claude's-Discretion helper API: keeps existing
/// `well_formed_envelope()` for non-scope-exercising tests.
pub fn well_formed_envelope_for_scope(_scope: &alzina_core::Scope) -> String {
    well_formed_envelope()
}

// ── SleepyExecutor ──────────────────────────────────────────────────────────

/// Phase 16 (D16-18, A9): synthetic executor with configurable dwell + cancel-token
/// select arm. Used by `tests/cancel_ladder_*.rs` to drive deterministic timing
/// without invoking the sidecar.
///
/// Default response is [`well_formed_envelope()`] so envelope-parsing in the
/// orchestrator stays valid.
///
/// The `biased;` in `tokio::select!` matches the runner's existing convention
/// (`alzina_runner.rs:499`). The cancel arm is listed FIRST after `biased;` so it
/// wins deterministically when both arms are ready — important for the burst edge
/// case in plan 16-09.
pub struct SleepyExecutor {
    pub dwell: Duration,
    pub response: String,
    pub cancel_token: Option<tokio_util::sync::CancellationToken>,
}

impl SleepyExecutor {
    /// Create a new `SleepyExecutor` with the given dwell duration.
    ///
    /// Default response is [`well_formed_envelope()`].
    pub fn new(dwell: Duration) -> Self {
        Self {
            dwell,
            response: well_formed_envelope(),
            cancel_token: None,
        }
    }

    /// Attach a cancellation token. When the token fires during the dwell,
    /// `execute` returns `Err(AlzinaError::Orchestration("cancelled"))`.
    pub fn with_cancel_token(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.cancel_token = Some(token);
        self
    }

    /// Override the response returned on successful completion.
    pub fn with_response(mut self, response: String) -> Self {
        self.response = response;
        self
    }
}

#[async_trait::async_trait]
impl AgentExecutor for SleepyExecutor {
    async fn execute(
        &self,
        _agent_id: &AgentId,
        _instruction: &str,
        _model: &str,
        _task: &str,
    ) -> AlzinaResult<String> {
        match &self.cancel_token {
            Some(tok) => {
                tokio::select! {
                    biased;
                    _ = tok.cancelled() => Err(AlzinaError::Orchestration("cancelled".into())),
                    _ = tokio::time::sleep(self.dwell) => Ok(self.response.clone()),
                }
            }
            None => {
                tokio::time::sleep(self.dwell).await;
                Ok(self.response.clone())
            }
        }
    }
}

// ── Other helpers ───────────────────────────────────────────────────────────

#[allow(dead_code)]
pub fn envelope_no_context_update() -> String {
    r#"STATUS: complete
ARTIFACTS: artifacts/output.md
SIGNAL: done
TENSIONS: none
EMERGENT: none"#
        .to_string()
}

/// All agent IDs used across orchestration tests. Identity files are
/// created for each so the PreSpawn hooks (AgentIdentityHook,
/// PreSpawnGateHook) allow them through.
const TEST_AGENTS: &[&str] = &[
    "smidr",
    "galdr",
    "rogue",
    "confused",
    "slowagent",
    "muninn",
    "urdr",
    "skuld",
    "vefr",
    "test",
    "kvasir",
    "test-agent",
    "default-agent",
    "huginn",
    "sjofn",
    "verdandi",
    "a",
    "b",
    "fast_agent",
    "slow_agent",
];

pub fn setup_test_workspace(dir: &Path) {
    let tmpl_dir = dir.join("templates/bootstrap");
    std::fs::create_dir_all(&tmpl_dir).unwrap();
    std::fs::write(
        tmpl_dir.join("system-prompt.jinja"),
        "{% if spawn_essence %}{{ spawn_essence }}{% endif %}\n\
         {% if identity %}{{ identity }}{% endif %}\n\
         {% for learning in learnings %}{{ learning }}\n{% endfor %}\n\
         {% for gate in governance_gates %}{{ gate }}\n{% endfor %}",
    )
    .unwrap();

    std::fs::create_dir_all(dir.join("artifacts")).unwrap();
    std::fs::write(dir.join("artifacts/output.md"), "test content").unwrap();

    // Create identity configs for all test agents so PreSpawn hooks pass.
    for agent in TEST_AGENTS {
        let agent_dir = dir.join(format!("config/agents/{}", agent));
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("identity.toml"),
            "[identity]\nname = \"test\"\n",
        )
        .unwrap();
    }
}

fn test_governance_config(dir: &Path) -> GovernanceConfig {
    let mut config = GovernanceConfig::default();
    for agent in TEST_AGENTS {
        config
            .archetype_profiles
            .insert((*agent).to_string(), "builder".into());
    }
    config.bootstrap.agent_config_dir = dir.join("config/agents").to_string_lossy().into_owned();
    config
}

pub async fn build_test_runner(
    executor: Arc<dyn AgentExecutor>,
    hook_set: Option<HookSet>,
    workspace_dir: &Path,
) -> AlzinaResult<AlzinaRunner> {
    setup_test_workspace(workspace_dir);

    let workspace = Arc::new(WorkspaceHandle::open(workspace_dir.to_path_buf())?);
    let config = test_governance_config(workspace_dir);
    let learnings_config = LearningsConfig::default();

    let governance = match hook_set {
        Some(hs) => Arc::new(GovernanceLayer::with_hooks(
            config.clone(),
            workspace.clone(),
            hs,
        )?),
        None => Arc::new(GovernanceLayer::new(config.clone(), workspace.clone())?),
    };

    let sessions = Arc::new(SessionHierarchy::in_memory().await?);
    let store = Arc::new(alzina_governance::FileLearningsStore::new(
        workspace
            .root()
            .join(&learnings_config.learnings_dir)
            .to_string_lossy()
            .to_string(),
        learnings_config.max_entries_per_agent,
    ));
    let learnings = Arc::new(LearningsMerger::with_store(
        workspace.clone(),
        learnings_config,
        alzina_core::canonical_domain_mapping(),
        store,
    ));

    let tmpl_dir = workspace_dir.join("templates");
    let template_engine = Arc::new(TemplateEngine::new(&tmpl_dir)?);
    let bootstrap = Arc::new(BootstrapEngine::new(
        workspace.clone(),
        template_engine,
        alzina_governance::BootstrapConfig::default(),
        config,
        None,
        None,
    ));

    Ok(AlzinaRunner::new(
        governance,
        bootstrap,
        sessions,
        learnings,
        executor,
        workspace,
        "test-model".to_string(),
        std::time::Duration::from_secs(30),
    ))
}

// ── SleepyExecutor unit tests ───────────────────────────────────────────────

#[cfg(test)]
mod sleepy_executor_tests {
    use super::*;
    use alzina_core::identity::AgentId;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn sleepy_executor_returns_response_after_dwell() {
        let executor = SleepyExecutor::new(Duration::from_millis(10))
            .with_response("hello".to_string());
        let id = AgentId::new("test");
        let start = Instant::now();
        let result = executor.execute(&id, "", "model", "task").await;
        let elapsed = start.elapsed();
        assert_eq!(result.unwrap(), "hello");
        assert!(
            elapsed >= Duration::from_millis(10),
            "expected at least 10ms dwell, got {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn sleepy_executor_cancel_token_aborts_dwell() {
        let token = tokio_util::sync::CancellationToken::new();
        let executor = SleepyExecutor::new(Duration::from_secs(10))
            .with_cancel_token(token.clone());
        let id = AgentId::new("test");

        let start = Instant::now();
        // Cancel after a short delay from a background task.
        let token_clone = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            token_clone.cancel();
        });

        let result = executor.execute(&id, "", "model", "task").await;
        let elapsed = start.elapsed();

        match result {
            Err(alzina_core::AlzinaError::Orchestration(msg)) => {
                assert_eq!(msg, "cancelled");
            }
            other => panic!("expected Orchestration(cancelled), got {:?}", other),
        }
        assert!(
            elapsed < Duration::from_millis(200),
            "expected cancel within 200ms, took {:?}",
            elapsed
        );
    }
}
