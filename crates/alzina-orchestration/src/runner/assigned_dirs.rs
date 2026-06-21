//! Per-dispatch writable-dir registry used by the PreToolUse interceptor
//! to scope Write/Edit/Bash decisions to the dir each dispatched agent was
//! told to write into.
//!
//! Lifecycle: the dispatch path (regular and compose) registers
//! `(SessionId, dir)` immediately before the spawn fires. The executor
//! looks the dir up by `SessionId` on every `ToolUse` event. When the
//! event loop exits — whether through completion, error, or sidecar EOF —
//! the executor drops the `AssignedDirGuard` and the registry entry is
//! removed.
//!
//! Why not a method param on every executor call: the
//! `AgentExecutor` trait already has three execute variants and a sizeable
//! impl surface (real SDK executor + several mocks across tests). A
//! shared registry keyed by the spawn `SessionId` avoids touching that
//! surface — the dispatcher and the interceptor agree on the key the
//! runner already plumbs through.
//!
//! Why not a side-table on `GovernanceLayer`: governance is shared across
//! many concerns (audit, tiers, profiles). Mixing per-dispatch transient
//! state into it would couple unrelated lifecycles. A dedicated registry
//! keeps the scope explicit.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use alzina_core::identity::SessionId;

/// Clone-able handle to the per-dispatch writable-dir map. Cloning shares
/// the underlying storage — the registry is one logical instance per
/// daemon, with handles distributed to the executor and to every
/// dispatch handler that needs to register a dir.
#[derive(Clone, Default, Debug)]
pub struct AssignedDirRegistry {
    inner: Arc<RwLock<HashMap<SessionId, String>>>,
}

impl AssignedDirRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the dir a dispatched agent is allowed to write into. Overwrites
    /// any prior entry for `session_id` (session ids are unique, so this is
    /// only a defensive overwrite).
    pub fn register(&self, session_id: SessionId, dir: String) {
        if let Ok(mut map) = self.inner.write() {
            map.insert(session_id, dir);
        }
    }

    /// Remove and return the dir for `session_id`. Called from the
    /// executor's RAII cleanup when the event loop exits.
    pub fn unregister(&self, session_id: &SessionId) -> Option<String> {
        self.inner.write().ok().and_then(|mut m| m.remove(session_id))
    }

    /// Look up the assigned dir for a session. Returns an owned String to
    /// avoid holding the read lock across an await point.
    pub fn get(&self, session_id: &SessionId) -> Option<String> {
        self.inner.read().ok().and_then(|m| m.get(session_id).cloned())
    }
}

/// RAII guard that removes a registry entry on drop. The executor holds
/// one for the lifetime of each `run_event_loop` call so the entry is
/// cleared on every exit path — including panics and early returns.
pub struct AssignedDirGuard {
    registry: AssignedDirRegistry,
    session_id: SessionId,
}

impl AssignedDirGuard {
    pub fn new(registry: AssignedDirRegistry, session_id: SessionId) -> Self {
        Self {
            registry,
            session_id,
        }
    }
}

impl Drop for AssignedDirGuard {
    fn drop(&mut self) {
        self.registry.unregister(&self.session_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_get_roundtrip() {
        let reg = AssignedDirRegistry::new();
        let sid = SessionId::new();
        reg.register(sid.clone(), "artifacts/unattached/abc12345".into());
        assert_eq!(
            reg.get(&sid).as_deref(),
            Some("artifacts/unattached/abc12345")
        );
    }

    #[test]
    fn unregister_returns_and_removes() {
        let reg = AssignedDirRegistry::new();
        let sid = SessionId::new();
        reg.register(sid.clone(), "d".into());
        assert_eq!(reg.unregister(&sid).as_deref(), Some("d"));
        assert!(reg.get(&sid).is_none());
    }

    #[test]
    fn get_missing_is_none() {
        let reg = AssignedDirRegistry::new();
        assert!(reg.get(&SessionId::new()).is_none());
    }

    #[test]
    fn guard_drops_entry() {
        let reg = AssignedDirRegistry::new();
        let sid = SessionId::new();
        reg.register(sid.clone(), "d".into());
        {
            let _g = AssignedDirGuard::new(reg.clone(), sid.clone());
            assert!(reg.get(&sid).is_some());
        }
        assert!(reg.get(&sid).is_none(), "guard drop must clear entry");
    }

    #[test]
    fn clones_share_storage() {
        let a = AssignedDirRegistry::new();
        let b = a.clone();
        let sid = SessionId::new();
        a.register(sid.clone(), "d".into());
        assert_eq!(b.get(&sid).as_deref(), Some("d"));
    }
}
