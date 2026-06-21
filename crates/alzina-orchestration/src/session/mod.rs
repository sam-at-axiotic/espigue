//! Phase 3: Session hierarchy — parent→child tracking, session tree.
//!
//! SQLite-backed session tree for tracking parent→child relationships,
//! weave associations, and depth enforcement. Implements ADK-Rust's
//! `SessionService` trait for Runner integration.

pub mod hierarchy;
pub mod sqlite_session_service;

pub use hierarchy::SessionHierarchy;
pub use sqlite_session_service::SqliteSessionService;
