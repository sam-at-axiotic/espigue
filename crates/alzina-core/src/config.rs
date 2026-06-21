//! Configuration types for daemon, channels, and cron.

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;

/// Top-level daemon configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub workspace_root: PathBuf,
    pub channels: Vec<ChannelConfig>,
    pub cron_jobs: Vec<CronJobConfig>,
    pub bind_addr: SocketAddr,
    pub pid_file: Option<PathBuf>,
}

/// Configuration for a single channel adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    pub kind: String,
    pub enabled: bool,
    pub settings: serde_json::Value,
}

/// Configuration for a scheduled cron job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJobConfig {
    pub name: String,
    pub schedule: String,
    pub agent_id: String,
    pub task_template: String,
    pub timezone: Option<String>,
    pub enabled: bool,
}

// ── Executor Configuration ──────────────────────────────────────────────────

/// Configuration for the agent executor backend.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutorConfig {
    /// Which executor backend to use.
    #[serde(default)]
    pub backend: ExecutorBackend,
    /// Configuration for the Claude Agent SDK sidecar backend.
    #[serde(default)]
    pub claude_agent_sdk: Option<ClaudeAgentSdkConfig>,
}

/// Available executor backends.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorBackend {
    /// Mock executor for testing (returns canned responses).
    #[default]
    Mock,
    /// Claude Agent SDK via TypeScript sidecar (OAuth token auth).
    ClaudeAgentSdk,
    /// Direct Anthropic Messages API (Console API key — future).
    AnthropicApi,
}

/// Configuration for the Claude Agent SDK sidecar backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeAgentSdkConfig {
    /// Tools the agent is allowed to use.
    /// Default: `["Read", "Glob", "Grep", "WebSearch", "WebFetch"]`
    #[serde(default = "default_allowed_tools")]
    pub allowed_tools: Vec<String>,
    /// SDK permission mode.
    /// Default: `"default"`
    #[serde(default = "default_permission_mode")]
    pub permission_mode: String,
    /// Path to the sidecar entry point. If not set, uses the bundled sidecar.
    #[serde(default)]
    pub sidecar_path: Option<PathBuf>,
}

impl Default for ClaudeAgentSdkConfig {
    fn default() -> Self {
        Self {
            allowed_tools: default_allowed_tools(),
            permission_mode: default_permission_mode(),
            sidecar_path: None,
        }
    }
}

fn default_allowed_tools() -> Vec<String> {
    vec![
        "Read".into(),
        "Glob".into(),
        "Grep".into(),
        "WebSearch".into(),
        "WebFetch".into(),
    ]
}

fn default_permission_mode() -> String {
    "default".into()
}
