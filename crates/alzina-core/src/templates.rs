//! Prompt template engine — filesystem-based Jinja2 templates.
//!
//! # Design Principle
//!
//! Prompts are **governed workspace artifacts**, not strings buried in Rust code.
//! They live as `.jinja` files on the filesystem, are git-tracked, human-readable,
//! and editable without recompilation. This is critical for the Norn-Weave governance
//! model where prompt content is a governed path.
//!
//! The `TemplateEngine` wraps minijinja to provide:
//! - Filesystem-based template loading from a configurable directory
//! - Runtime rendering with typed context (anything implementing `Serialize`)
//! - Hot-reload for development iteration
//! - Template enumeration for governance auditing
//!
//! Templates are loaded at construction time (consistent with the bootstrap-at-construction
//! pattern established in `bootstrap.rs`). The `reload()` method exists for development
//! workflows where templates are being iterated on without restarting the daemon.

use crate::error::{AlzinaError, AlzinaResult};
use minijinja::Environment;
use serde::Serialize;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Filesystem-based prompt template engine.
///
/// Wraps a minijinja `Environment` that loads `.jinja` files from a directory tree.
/// Template names are relative paths within the template directory, e.g.
/// `bootstrap/agent-identity.jinja`.
///
/// # Example
///
/// ```no_run
/// use alzina_core::templates::TemplateEngine;
/// use std::path::Path;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct AgentContext {
///     agent_name: String,
///     domain: String,
/// }
///
/// let engine = TemplateEngine::new(Path::new("templates")).unwrap();
/// let ctx = AgentContext {
///     agent_name: "Smiðr".into(),
///     domain: "workspace building".into(),
/// };
/// let rendered = engine.render("bootstrap/agent-identity.jinja", &ctx).unwrap();
/// ```
pub struct TemplateEngine {
    env: Environment<'static>,
    template_dir: PathBuf,
}

impl std::fmt::Debug for TemplateEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemplateEngine")
            .field("template_dir", &self.template_dir)
            .field("template_count", &self.env.templates().count())
            .finish()
    }
}

impl TemplateEngine {
    /// Create a new template engine, loading all `.jinja` files from `template_dir`.
    ///
    /// Recursively walks the directory tree. Template names are relative paths
    /// with forward slashes (e.g. `bootstrap/agent-identity.jinja`).
    ///
    /// Returns an error if the directory doesn't exist or contains templates
    /// with invalid Jinja2 syntax.
    pub fn new(template_dir: &Path) -> AlzinaResult<Self> {
        let template_dir = template_dir.to_path_buf();
        let mut env = Environment::new();
        Self::load_templates(&mut env, &template_dir)?;
        info!(
            dir = %template_dir.display(),
            count = env.templates().count(),
            "template engine initialised"
        );
        Ok(Self { env, template_dir })
    }

    /// Render a named template with the given context.
    ///
    /// The context can be any type implementing `Serialize` — typically a struct
    /// with the template's expected variables.
    pub fn render<S: Serialize>(&self, name: &str, context: &S) -> AlzinaResult<String> {
        let tmpl = self
            .env
            .get_template(name)
            .map_err(|e| AlzinaError::Template(format!("template not found: {name}: {e}")))?;
        tmpl.render(context)
            .map_err(|e| AlzinaError::Template(format!("render error in {name}: {e}")))
    }

    /// Re-scan the template directory, picking up new or modified templates.
    ///
    /// Intended for development workflows. In production, prefer constructing
    /// a new `TemplateEngine` to avoid partial-load states.
    pub fn reload(&mut self) -> AlzinaResult<()> {
        let mut env = Environment::new();
        Self::load_templates(&mut env, &self.template_dir)?;
        self.env = env;
        info!(
            dir = %self.template_dir.display(),
            count = self.env.templates().count(),
            "template engine reloaded"
        );
        Ok(())
    }

    /// List all loaded template names.
    ///
    /// Useful for governance auditing — enumerate what templates are available
    /// in the current workspace.
    pub fn list_templates(&self) -> Vec<String> {
        self.env
            .templates()
            .map(|(name, _)| name.to_string())
            .collect()
    }

    /// Recursively load all `.jinja` files from a directory into the environment.
    fn load_templates(env: &mut Environment<'static>, dir: &Path) -> AlzinaResult<()> {
        if !dir.is_dir() {
            return Err(AlzinaError::Template(format!(
                "template directory does not exist: {}",
                dir.display()
            )));
        }
        Self::walk_dir(env, dir, dir)
    }

    fn walk_dir(env: &mut Environment<'static>, root: &Path, dir: &Path) -> AlzinaResult<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::walk_dir(env, root, &path)?;
            } else if path.extension().is_some_and(|ext| ext == "jinja") {
                let rel = path
                    .strip_prefix(root)
                    .map_err(|e| AlzinaError::Template(format!("path error: {e}")))?;
                // Normalise to forward slashes for cross-platform template names
                let name = rel.to_string_lossy().replace('\\', "/");
                let content = std::fs::read_to_string(&path)?;
                debug!(template = %name, "loaded template");
                env.add_template_owned(name, content).map_err(|e| {
                    AlzinaError::Template(format!("syntax error in {}: {e}", path.display()))
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use std::fs;

    fn create_test_templates(dir: &Path) {
        let bootstrap_dir = dir.join("bootstrap");
        fs::create_dir_all(&bootstrap_dir).unwrap();
        fs::write(
            bootstrap_dir.join("agent-identity.jinja"),
            "You are {{ agent_name }}, the {{ domain }} agent.",
        )
        .unwrap();
        fs::write(dir.join("simple.jinja"), "Hello, {{ name }}!").unwrap();
    }

    #[derive(Serialize)]
    struct SimpleCtx {
        name: String,
    }

    #[derive(Serialize)]
    struct AgentCtx {
        agent_name: String,
        domain: String,
    }

    #[test]
    fn load_and_render() {
        let dir = tempfile::tempdir().unwrap();
        create_test_templates(dir.path());

        let engine = TemplateEngine::new(dir.path()).unwrap();
        let result = engine
            .render(
                "simple.jinja",
                &SimpleCtx {
                    name: "Samu".into(),
                },
            )
            .unwrap();
        assert_eq!(result, "Hello, Samu!");
    }

    #[test]
    fn render_nested_template() {
        let dir = tempfile::tempdir().unwrap();
        create_test_templates(dir.path());

        let engine = TemplateEngine::new(dir.path()).unwrap();
        let result = engine
            .render(
                "bootstrap/agent-identity.jinja",
                &AgentCtx {
                    agent_name: "Smiðr".into(),
                    domain: "workspace building".into(),
                },
            )
            .unwrap();
        assert_eq!(result, "You are Smiðr, the workspace building agent.");
    }

    #[test]
    fn list_templates_includes_all() {
        let dir = tempfile::tempdir().unwrap();
        create_test_templates(dir.path());

        let engine = TemplateEngine::new(dir.path()).unwrap();
        let mut names = engine.list_templates();
        names.sort();
        assert_eq!(
            names,
            vec!["bootstrap/agent-identity.jinja", "simple.jinja",]
        );
    }

    #[test]
    fn missing_template_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        create_test_templates(dir.path());

        let engine = TemplateEngine::new(dir.path()).unwrap();
        let result = engine.render(
            "nonexistent.jinja",
            &SimpleCtx {
                name: "test".into(),
            },
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("template not found"), "got: {err}");
    }

    #[test]
    fn invalid_syntax_caught_at_load() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("bad.jinja"), "{% if unclosed").unwrap();

        let result = TemplateEngine::new(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("syntax error"), "got: {err}");
    }

    #[test]
    fn nonexistent_directory_returns_error() {
        let result = TemplateEngine::new(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not exist"), "got: {err}");
    }

    #[test]
    fn reload_picks_up_new_templates() {
        let dir = tempfile::tempdir().unwrap();
        create_test_templates(dir.path());

        let mut engine = TemplateEngine::new(dir.path()).unwrap();
        assert_eq!(engine.list_templates().len(), 2);

        // Add a new template
        fs::write(dir.path().join("new.jinja"), "New: {{ value }}").unwrap();

        engine.reload().unwrap();
        assert_eq!(engine.list_templates().len(), 3);

        let result = engine
            .render("new.jinja", &serde_json::json!({"value": "hello"}))
            .unwrap();
        assert_eq!(result, "New: hello");
    }
}
