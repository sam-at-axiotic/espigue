//! Core identity types for agents, sessions, and weaves.

use crate::error::{AlzinaError, AlzinaResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Identifies an agent by name (e.g. "muninn", "skuld", "smidr").
///
/// # Examples
///
/// ```
/// use alzina_core::AgentId;
///
/// let id = AgentId::new("smidr");
/// assert_eq!(id.as_str(), "smidr");
/// assert_eq!(id.to_string(), "smidr");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

impl AgentId {
    /// Create a new `AgentId`, panicking if the name is invalid.
    ///
    /// Rejects empty strings, strings containing path separators (`/`, `\\`),
    /// null bytes, or dots (which could enable path traversal via `..`).
    ///
    /// For fallible construction, use [`AgentId::try_new`].
    pub fn new(name: impl Into<String>) -> Self {
        Self::try_new(name).expect("invalid AgentId")
    }

    /// Try to create a new `AgentId`, returning an error if the name is invalid.
    ///
    /// Rejects:
    /// - Empty strings
    /// - Strings containing path separators (`/`, `\\`)
    /// - Strings containing null bytes
    /// - Strings equal to `.` or `..`
    pub fn try_new(name: impl Into<String>) -> AlzinaResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(AlzinaError::Config("AgentId must not be empty".into()));
        }
        if name.contains('/') || name.contains('\\') {
            return Err(AlzinaError::Config(format!(
                "AgentId must not contain path separators: {name:?}"
            )));
        }
        if name.contains('\0') {
            return Err(AlzinaError::Config(format!(
                "AgentId must not contain null bytes: {name:?}"
            )));
        }
        if name == "." || name == ".." {
            return Err(AlzinaError::Config(format!(
                "AgentId must not be a relative path component: {name:?}"
            )));
        }
        // D4 (D3-P2-1): reject control codepoints (ANSI/ESC/CR/LF) and
        // bidi-override characters, which can be used to forge agent
        // identifiers in audit logs / terminal output.
        for c in name.chars() {
            if c.is_control() {
                return Err(AlzinaError::Config(format!(
                    "AgentId must not contain control characters: {name:?}"
                )));
            }
            if matches!(c, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}') {
                return Err(AlzinaError::Config(format!(
                    "AgentId must not contain bidi-override characters: {name:?}"
                )));
            }
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique session identifier backed by UUID v4.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(Uuid);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identifies a weave — a unit of nontrivial tracked work.
///
/// Per R-WEAVE-SCOPE-001 §3, `WeaveId` is a typed component of the
/// `(agent_id, weave_id)` capability key. Validation rejects the same
/// classes of input that `AgentId` rejects (path separators, nulls,
/// dots, control codepoints, bidi overrides) so weave IDs cannot be
/// forged into audit logs or file paths.
///
/// The literal string `"SessionDefault"` is reserved and rejected here —
/// it is the canonical serialised form of [`Scope::SessionDefault`] and
/// must not be constructible as a `WeaveId`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WeaveId(String);

impl WeaveId {
    /// Reserved canonical name of [`Scope::SessionDefault`].
    pub const SESSION_DEFAULT_RESERVED: &'static str = "SessionDefault";

    /// Create a `WeaveId`, panicking if invalid. Use [`WeaveId::try_new`]
    /// for fallible construction.
    pub fn new(id: impl Into<String>) -> Self {
        Self::try_new(id).expect("invalid WeaveId")
    }

    /// Try to create a `WeaveId`. Rules match [`AgentId::try_new`] plus
    /// the reservation of `"SessionDefault"`.
    pub fn try_new(id: impl Into<String>) -> AlzinaResult<Self> {
        let id = id.into();
        if id.is_empty() {
            return Err(AlzinaError::Config("WeaveId must not be empty".into()));
        }
        if id == Self::SESSION_DEFAULT_RESERVED {
            return Err(AlzinaError::Config(format!(
                "WeaveId must not equal the reserved scope literal {:?}",
                Self::SESSION_DEFAULT_RESERVED
            )));
        }
        if id.contains('/') || id.contains('\\') {
            return Err(AlzinaError::Config(format!(
                "WeaveId must not contain path separators: {id:?}"
            )));
        }
        if id.contains('\0') {
            return Err(AlzinaError::Config(format!(
                "WeaveId must not contain null bytes: {id:?}"
            )));
        }
        if id == "." || id == ".." {
            return Err(AlzinaError::Config(format!(
                "WeaveId must not be a relative path component: {id:?}"
            )));
        }
        for c in id.chars() {
            if c.is_control() {
                return Err(AlzinaError::Config(format!(
                    "WeaveId must not contain control characters: {id:?}"
                )));
            }
            if matches!(c, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}') {
                return Err(AlzinaError::Config(format!(
                    "WeaveId must not contain bidi-override characters: {id:?}"
                )));
            }
        }
        Ok(Self(id))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WeaveId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Authenticated scope key for HTTP / parser / gate boundaries.
///
/// Per R-WEAVE-SCOPE-001, `weave_id` is promoted from an advisory tag
/// to a structural scope key. `Scope` is the typed handle handlers,
/// parsers, and gates consume — `Weave(WeaveId)` for weave-bound
/// operations, `SessionDefault` for the well-defined set of operations
/// that do not belong to any weave (health, lightweight chat,
/// observation read-only). A missing or unresolved scope is a
/// parse-time rejection at boundaries that demand a scope, not a
/// runtime warning.
///
/// Serialisation is the canonical string form (`weave_id` or the
/// literal `"SessionDefault"`) — never null, never absent (AC-7).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Scope {
    Weave(WeaveId),
    SessionDefault,
}

impl Default for Scope {
    /// `Scope::SessionDefault` is the canonical unscoped value. This impl
    /// exists so `#[serde(default)]` on a `scope` field defaults missing
    /// keys to `SessionDefault` — load-bearing for audit-replay backcompat
    /// of pre-15-09 JSONL lines that predate the `SessionSpawned.scope`
    /// wire field.
    fn default() -> Self {
        Scope::SessionDefault
    }
}

impl Scope {
    /// Canonical string form. Weave-scoped: the weave_id. Unscoped:
    /// the literal `"SessionDefault"`. Used by audit emission to honour
    /// AC-7 (no null/missing weave_id field).
    pub fn as_str(&self) -> &str {
        match self {
            Scope::Weave(w) => w.as_str(),
            Scope::SessionDefault => WeaveId::SESSION_DEFAULT_RESERVED,
        }
    }

    /// Parse a scope string. The literal `"SessionDefault"` returns
    /// the unscoped variant; any other non-empty string is validated
    /// as a `WeaveId`.
    pub fn parse(s: impl AsRef<str>) -> AlzinaResult<Self> {
        let s = s.as_ref();
        if s == WeaveId::SESSION_DEFAULT_RESERVED {
            Ok(Scope::SessionDefault)
        } else {
            WeaveId::try_new(s).map(Scope::Weave)
        }
    }

    /// True if this scope is weave-bound (i.e. not `SessionDefault`).
    pub fn is_weave(&self) -> bool {
        matches!(self, Scope::Weave(_))
    }

    /// Borrow the inner `WeaveId` if weave-bound.
    pub fn weave_id(&self) -> Option<&WeaveId> {
        match self {
            Scope::Weave(w) => Some(w),
            Scope::SessionDefault => None,
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Scope {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Scope {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Scope::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// Write tier enforcement levels.
///
/// Tier 1 (Governed): SOUL.md, AGENT.md, contracts, hooks — requires rune pipeline.
/// Tier 2 (Integrity): well/log/, well/competence.yaml — append-only, system-managed.
/// Tier 3 (FreeWrite): memory/, artifacts/, learnings/ — agents write freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WriteTier {
    /// Tier 1: Governed paths — mutation requires the rune pipeline.
    Governed,
    /// Tier 2: Integrity-protected paths — system-managed, append-only.
    Integrity,
    /// Tier 3: Free-write paths — agents write freely.
    FreeWrite,
}
