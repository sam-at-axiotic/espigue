//! Workspace management — filesystem workspace operations.
//!
//! WorkspaceManager is a struct (not a trait) because there is one true
//! implementation: the filesystem. It enforces workspace root boundaries
//! and provides path resolution helpers.

use crate::error::{AlzinaError, AlzinaResult};
use std::path::{Path, PathBuf};

/// Manages a filesystem workspace — path resolution, boundary enforcement,
/// and artifact access.
#[derive(Debug, Clone)]
pub struct WorkspaceManager {
    root: PathBuf,
}

impl WorkspaceManager {
    /// Create a new workspace manager rooted at the given path.
    pub fn new(root: PathBuf) -> AlzinaResult<Self> {
        if !root.exists() {
            return Err(AlzinaError::Workspace(format!(
                "workspace root does not exist: {}",
                root.display()
            )));
        }
        Ok(Self {
            root: root
                .canonicalize()
                .map_err(|e| AlzinaError::Workspace(format!("failed to canonicalize root: {e}")))?,
        })
    }

    /// Get the workspace root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a relative path against the workspace root.
    /// Returns an error if the resolved path escapes the workspace.
    pub fn resolve(&self, relative: &str) -> AlzinaResult<PathBuf> {
        let resolved = self.root.join(relative);
        // Prevent path traversal
        if let Ok(canonical) = resolved.canonicalize() {
            if !canonical.starts_with(&self.root) {
                return Err(AlzinaError::Workspace(format!(
                    "path traversal blocked: {relative}"
                )));
            }
            Ok(canonical)
        } else {
            // File may not exist yet — check that the parent is within workspace
            let parent = resolved
                .parent()
                .ok_or_else(|| AlzinaError::Workspace("no parent directory".into()))?;
            if let Ok(canonical_parent) = parent.canonicalize()
                && !canonical_parent.starts_with(&self.root)
            {
                return Err(AlzinaError::Workspace(format!(
                    "path traversal blocked: {relative}"
                )));
            }
            Ok(resolved)
        }
    }

    // classify_path() removed — use alzina_workspace::TierClassifier as the
    // single source of truth for path-to-tier classification.
}
