//! Phase 3: Runner wrapper — wraps ADK-Rust Runner with governance hooks.
//!
//! AlzinaRunner orchestrates the full agent dispatch lifecycle:
//! governance → bootstrap → model resolution → execution → envelope → learnings.
//!
//! The tool interceptor enforces write tiers during execution.
//! The model resolver determines which LLM model to use.

pub mod alzina_runner;
pub mod assigned_dirs;
pub mod claude_agent_sdk;
pub mod envelope_tool;
pub mod model_resolver;
pub mod sidecar_handle;
pub mod sidecar_protocol;
pub mod stop_conditions;
pub mod tool_interceptor;

// Re-export the SDK executor and sidecar handle for external use.
pub use assigned_dirs::{AssignedDirGuard, AssignedDirRegistry};
pub use claude_agent_sdk::ClaudeAgentSdkExecutor;
pub use envelope_tool::return_envelope_tool;
pub use sidecar_handle::SidecarHandle;
pub use sidecar_protocol::CustomToolDefinition;
pub use stop_conditions::StopConditionEvaluator;
