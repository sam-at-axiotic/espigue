//! `alzina_core::tools` — Tool surface root module.
//!
//! This module is the entry point for the Alzina tool registration discipline.
//! It exports the `Tool` trait and its companion types from `types.rs`, and
//! declares four utility submodules whose bodies are filled by Wave 3 plans:
//!
//! - `url_guard` (Plan 13-03): SSRF / RFC1918 / link-local / multicast /
//!   IPv4-mapped IPv6 / DNS rebinding defence.
//! - `result_budget` (Plan 13-03): Per-call byte-cap with UTF-8-safe
//!   truncation marker.
//! - `timeout` (Plan 13-03): `ALZINA_TOOL_TIMEOUT_SECS` bounded parser and
//!   `OnceLock`-cached resolver.
//! - `compaction` (Plan 13-04): Tokenjuice tool-output compaction adapter.
//!
//! ## Adapter site
//!
//! `impl From<&dyn Tool> for CustomToolDefinition` lands in Plan 13-05
//! (alzina-orchestration). This module provides the trait and types that
//! adapter converts.

mod types;

pub use self::types::{
    PermissionLevel, Tool, ToolCallOptions, ToolCategory, ToolContent, ToolResult, ToolScope,
};

pub mod compaction;
pub mod result_budget;
pub mod timeout;
pub mod url_guard;
