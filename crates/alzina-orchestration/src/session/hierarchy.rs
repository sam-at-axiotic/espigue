//! Session hierarchy — SQLite-backed parent→child session tree.
//!
//! Tracks session lineage for the orchestration engine. Every spawned agent
//! gets a session node linked to its parent. Weave association flows down
//! the tree: a child inherits its parent's weave unless explicitly overridden.

use alzina_core::envelope::EnvelopeStatus;
use alzina_core::identity::{AgentId, SessionId, WeaveId};
use alzina_core::session::{SessionNode, SessionStatus};
use alzina_core::{AlzinaError, AlzinaResult};
use chrono::Utc;
use sqlx::Row;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions, SqliteRow};
use tracing::{debug, instrument};

/// Default maximum session nesting depth.
pub const DEFAULT_MAX_DEPTH: u32 = 5;

/// SQLite-backed session tree.
///
/// Each session is a row with an optional parent pointer. The tree is walked
/// via recursive queries for depth checks and weave-root lookups.
pub struct SessionHierarchy {
    pool: SqlitePool,
}

impl SessionHierarchy {
    /// Create a new hierarchy backed by the given SQLite pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Connect to a SQLite database and run migrations.
    pub async fn connect(database_url: &str) -> AlzinaResult<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .map_err(|e| AlzinaError::Session(format!("failed to connect: {e}")))?;

        let hierarchy = Self { pool };
        hierarchy.migrate().await?;
        Ok(hierarchy)
    }

    /// Create an in-memory hierarchy for testing.
    ///
    /// LE-1 fix (2026-04-30): the previous URL `sqlite::memory:` produced a
    /// PRIVATE in-memory database PER pooled connection. The migration ran
    /// on whichever connection the pool handed out first, but subsequent
    /// INSERT/SELECT queries got OTHER pool members that had never seen the
    /// migration — so live dispatch failed with `no such table: sessions`
    /// the moment more than one connection was used concurrently. The fix
    /// uses a UNIQUELY-NAMED shared in-memory database so all pool
    /// connections see the same tables, while different `in_memory()`
    /// calls (e.g. parallel tests, multiple daemon instances) stay isolated
    /// from each other. `min_connections(1)` keeps at least one pool member
    /// alive so the named DB outlives any individual connection.
    pub async fn in_memory() -> AlzinaResult<Self> {
        // Per the SQLite URI spec, `file:<name>?mode=memory&cache=shared`
        // creates a named in-memory database that all connections opening
        // the same URL share. A random name per call gives us isolation
        // between independent hierarchies (e.g. parallel test fixtures).
        let unique = uuid::Uuid::new_v4().simple().to_string();
        let url = format!("sqlite:file:alzina-mem-{unique}?mode=memory&cache=shared");

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            // Keep at least one connection alive — when the pool fully drains
            // a `?mode=memory&cache=shared` database is destroyed.
            .min_connections(1)
            // sqlx's default `idle_timeout` is 10 minutes and `max_lifetime`
            // is 30 minutes. When the LAST connection to a named-shared
            // in-memory DB is recycled, SQLite destroys the database — and
            // the next connection sqlx opens hits an empty DB that's never
            // seen `migrate()`. Symptom: dispatch fails with `no such table:
            // sessions` after the daemon has been up >10 minutes. Live-eval
            // P0-LE-1B (2026-05-05). Disable both timeouts so the in-memory
            // DB persists for the daemon's full lifetime.
            .idle_timeout(None)
            .max_lifetime(None)
            .connect(&url)
            .await
            .map_err(|e| AlzinaError::Session(format!("failed to connect: {e}")))?;

        let hierarchy = Self { pool };
        hierarchy.migrate().await?;
        Ok(hierarchy)
    }

    /// Run schema migrations.
    async fn migrate(&self) -> AlzinaResult<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                parent_id TEXT,
                weave_id TEXT,
                status TEXT NOT NULL DEFAULT 'Pending',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (parent_id) REFERENCES sessions(session_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("migration failed: {e}")))?;

        // Index for parent lookups (frequent in depth checks and children queries).
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_id)")
            .execute(&self.pool)
            .await
            .map_err(|e| AlzinaError::Session(format!("index creation failed: {e}")))?;

        Ok(())
    }

    /// Create a child session linked to a parent.
    ///
    /// The child inherits the parent's weave if `weave_id` is `None`.
    #[instrument(skip(self), fields(parent = %parent, child = %child, agent = %agent_id))]
    pub async fn spawn_child(
        &self,
        parent: &SessionId,
        child: &SessionId,
        agent_id: &AgentId,
        weave_id: Option<&WeaveId>,
    ) -> AlzinaResult<()> {
        // Resolve weave: explicit override, or inherit from parent.
        let resolved_weave = match weave_id {
            Some(w) => Some(w.as_str().to_owned()),
            None => self.inherited_weave(parent).await?,
        };

        let now = Utc::now().to_rfc3339();
        let status_str = status_to_string(&SessionStatus::Pending);

        // RT3-09: INSERT OR IGNORE to handle UNIQUE constraint gracefully
        let result = sqlx::query(
            r#"
            INSERT OR IGNORE INTO sessions (session_id, agent_id, parent_id, weave_id, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(child.to_string())
        .bind(agent_id.as_str())
        .bind(parent.to_string())
        .bind(resolved_weave.as_deref())
        .bind(&status_str)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("spawn_child insert failed: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(AlzinaError::Session(format!(
                "session already exists: {child}"
            )));
        }

        debug!("spawned child session {child} under parent {parent}");
        Ok(())
    }

    /// Create a root session (no parent).
    #[instrument(skip(self), fields(session = %session_id, agent = %agent_id))]
    pub async fn create_root(
        &self,
        session_id: &SessionId,
        agent_id: &AgentId,
        weave_id: Option<&WeaveId>,
    ) -> AlzinaResult<()> {
        let now = Utc::now().to_rfc3339();
        let status_str = status_to_string(&SessionStatus::Pending);

        // RT3-09: INSERT OR IGNORE to handle UNIQUE constraint gracefully
        let result = sqlx::query(
            r#"
            INSERT OR IGNORE INTO sessions (session_id, agent_id, parent_id, weave_id, status, created_at)
            VALUES (?, ?, NULL, ?, ?, ?)
            "#,
        )
        .bind(session_id.to_string())
        .bind(agent_id.as_str())
        .bind(weave_id.map(|w| w.as_str().to_owned()).as_deref())
        .bind(&status_str)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("create_root insert failed: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(AlzinaError::Session(format!(
                "session already exists: {session_id}"
            )));
        }

        debug!("created root session {session_id}");
        Ok(())
    }

    /// Mark a session complete with an envelope status.
    #[instrument(skip(self), fields(session = %session_id))]
    pub async fn complete(
        &self,
        session_id: &SessionId,
        status: EnvelopeStatus,
    ) -> AlzinaResult<()> {
        let now = Utc::now().to_rfc3339();
        let status_str = status_to_string(&SessionStatus::Complete(status));

        let result = sqlx::query(
            r#"
            UPDATE sessions
            SET status = ?, completed_at = ?
            WHERE session_id = ?
            "#,
        )
        .bind(&status_str)
        .bind(&now)
        .bind(session_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("complete update failed: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(AlzinaError::Session(format!(
                "session not found: {session_id}"
            )));
        }

        debug!("completed session {session_id} with status {status_str}");
        Ok(())
    }

    /// Mark a session as failed with an error message.
    pub async fn fail(&self, session_id: &SessionId, reason: &str) -> AlzinaResult<()> {
        let now = Utc::now().to_rfc3339();
        let status_str = status_to_string(&SessionStatus::Failed(reason.to_string()));

        let result = sqlx::query(
            r#"
            UPDATE sessions
            SET status = ?, completed_at = ?
            WHERE session_id = ?
            "#,
        )
        .bind(&status_str)
        .bind(&now)
        .bind(session_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("fail update failed: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(AlzinaError::Session(format!(
                "session not found: {session_id}"
            )));
        }

        debug!("marked session {session_id} as failed: {reason}");
        Ok(())
    }

    /// Get all direct children of a session.
    pub async fn children(&self, session_id: &SessionId) -> AlzinaResult<Vec<SessionNode>> {
        let rows = sqlx::query(
            "SELECT session_id, agent_id, parent_id, weave_id, status, created_at, completed_at FROM sessions WHERE parent_id = ?",
        )
        .bind(session_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("children query failed: {e}")))?;

        let mut nodes = Vec::with_capacity(rows.len());
        for row in &rows {
            nodes.push(row_to_session_node(row, &self.pool).await?);
        }
        Ok(nodes)
    }

    /// Walk up to find the weave root session — the tree root of this session's
    /// weave lineage. Since children inherit weave_id from parents, the root
    /// of the weave is the tree root (the session with no parent).
    /// RT3-16: Replaced iterative traversal with recursive CTE.
    pub async fn weave_root(&self, session_id: &SessionId) -> AlzinaResult<Option<SessionNode>> {
        let row = sqlx::query(
            r#"
            WITH RECURSIVE ancestors AS (
                SELECT session_id, agent_id, parent_id, weave_id, status, created_at, completed_at
                FROM sessions WHERE session_id = ?
                UNION ALL
                SELECT s.session_id, s.agent_id, s.parent_id, s.weave_id, s.status, s.created_at, s.completed_at
                FROM sessions s
                JOIN ancestors a ON s.session_id = a.parent_id
            )
            SELECT session_id, agent_id, parent_id, weave_id, status, created_at, completed_at
            FROM ancestors
            WHERE parent_id IS NULL
            LIMIT 1
            "#,
        )
        .bind(session_id.to_string())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("weave_root CTE query failed: {e}")))?;

        match row {
            None => Ok(None),
            Some(ref r) => Ok(Some(row_to_session_node(r, &self.pool).await?)),
        }
    }

    /// Check the depth of a parent session and return it.
    ///
    /// Returns `Err(AlzinaError::Orchestration)` with `DepthExceeded` info
    /// if the depth would exceed `max_depth` when spawning a child.
    pub async fn check_depth(&self, parent: &SessionId, max_depth: u32) -> AlzinaResult<u32> {
        let depth = self.compute_depth(parent).await?;

        // depth is the parent's depth (0-indexed from root).
        // A child would be at depth + 1.
        let child_depth = depth + 1;
        if child_depth > max_depth {
            return Err(AlzinaError::Orchestration(format!(
                "depth exceeded: child would be at depth {child_depth}, max is {max_depth}"
            )));
        }

        Ok(depth)
    }

    /// RT3-16: Compute depth using a recursive CTE.
    async fn compute_depth(&self, session_id: &SessionId) -> AlzinaResult<u32> {
        let depth: Option<i32> = sqlx::query_scalar(
            r#"
            WITH RECURSIVE ancestors AS (
                SELECT session_id, parent_id, 0 AS depth
                FROM sessions WHERE session_id = ?
                UNION ALL
                SELECT s.session_id, s.parent_id, a.depth + 1
                FROM sessions s
                JOIN ancestors a ON s.session_id = a.parent_id
            )
            SELECT MAX(depth) FROM ancestors
            "#,
        )
        .bind(session_id.to_string())
        .fetch_one(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("depth CTE query failed: {e}")))?;

        Ok(depth.unwrap_or(0) as u32)
    }

    /// Get the weave_id inherited from the nearest ancestor.
    async fn inherited_weave(&self, session_id: &SessionId) -> AlzinaResult<Option<String>> {
        let mut current_id = session_id.to_string();

        for _ in 0..=DEFAULT_MAX_DEPTH {
            let row = sqlx::query("SELECT parent_id, weave_id FROM sessions WHERE session_id = ?")
                .bind(&current_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| AlzinaError::Session(format!("inherited_weave query failed: {e}")))?;

            match row {
                None => return Ok(None),
                Some(r) => {
                    let weave: Option<String> = r.get("weave_id");
                    if weave.is_some() {
                        return Ok(weave);
                    }
                    let parent: Option<String> = r.get("parent_id");
                    match parent {
                        None => return Ok(None),
                        Some(pid) => current_id = pid,
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get a single session node by ID.
    pub async fn get(&self, session_id: &SessionId) -> AlzinaResult<Option<SessionNode>> {
        let row = sqlx::query(
            "SELECT session_id, agent_id, parent_id, weave_id, status, created_at, completed_at FROM sessions WHERE session_id = ?",
        )
        .bind(session_id.to_string())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AlzinaError::Session(format!("get query failed: {e}")))?;

        match row {
            None => Ok(None),
            Some(ref r) => Ok(Some(row_to_session_node(r, &self.pool).await?)),
        }
    }

    // ── List query (added for SessionManager) ───────────────────────────

    /// List all sessions, optionally filtered by status prefix and agent_id.
    ///
    /// Returns all matching `SessionNode`s. The caller is responsible for
    /// pagination (limit/offset). Filters are applied in SQL for efficiency.
    pub async fn list(
        &self,
        status_prefix: Option<&str>,
        agent_id: Option<&AgentId>,
    ) -> AlzinaResult<Vec<SessionNode>> {
        let mut sql = String::from(
            "SELECT session_id, agent_id, parent_id, weave_id, status, created_at, completed_at FROM sessions WHERE 1=1",
        );
        let mut binds: Vec<String> = Vec::new();

        if let Some(prefix) = status_prefix {
            sql.push_str(" AND status LIKE ?");
            binds.push(format!("{prefix}%"));
        }
        if let Some(aid) = agent_id {
            sql.push_str(" AND agent_id = ?");
            binds.push(aid.as_str().to_owned());
        }

        sql.push_str(" ORDER BY created_at DESC");

        let mut query = sqlx::query(&sql);
        for b in &binds {
            query = query.bind(b);
        }

        let rows = query
            .fetch_all(&self.pool)
            .await
            .map_err(|e| AlzinaError::Session(format!("list query failed: {e}")))?;

        let mut nodes = Vec::with_capacity(rows.len());
        for row in &rows {
            nodes.push(row_to_session_node(row, &self.pool).await?);
        }
        Ok(nodes)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn status_to_string(status: &SessionStatus) -> String {
    match status {
        SessionStatus::Pending => "Pending".to_owned(),
        SessionStatus::Bootstrapping => "Bootstrapping".to_owned(),
        SessionStatus::Running => "Running".to_owned(),
        SessionStatus::AwaitingChildren => "AwaitingChildren".to_owned(),
        SessionStatus::Completing => "Completing".to_owned(),
        SessionStatus::Complete(es) => format!("Complete:{}", envelope_status_to_string(es)),
        SessionStatus::Failed(msg) => format!("Failed:{msg}"),
    }
}

fn string_to_status(s: &str) -> SessionStatus {
    match s {
        "Pending" => SessionStatus::Pending,
        "Bootstrapping" => SessionStatus::Bootstrapping,
        "Running" => SessionStatus::Running,
        "AwaitingChildren" => SessionStatus::AwaitingChildren,
        "Completing" => SessionStatus::Completing,
        s if s.starts_with("Complete:") => {
            let rest = &s["Complete:".len()..];
            SessionStatus::Complete(string_to_envelope_status(rest))
        }
        s if s.starts_with("Failed:") => SessionStatus::Failed(s["Failed:".len()..].to_owned()),
        _ => SessionStatus::Failed(format!("unknown status: {s}")),
    }
}

fn envelope_status_to_string(status: &EnvelopeStatus) -> String {
    match status {
        EnvelopeStatus::Complete => "Complete".to_owned(),
        EnvelopeStatus::Partial => "Partial".to_owned(),
        EnvelopeStatus::Error => "Error".to_owned(),
    }
}

fn string_to_envelope_status(s: &str) -> EnvelopeStatus {
    match s {
        "Complete" => EnvelopeStatus::Complete,
        "Partial" => EnvelopeStatus::Partial,
        _ => EnvelopeStatus::Error,
    }
}

/// Convert a SQLite row to a SessionNode.
///
/// Children are fetched in a separate query since they're not stored inline.
async fn row_to_session_node(row: &SqliteRow, pool: &SqlitePool) -> AlzinaResult<SessionNode> {
    let session_id_str: String = row.get("session_id");
    let agent_id_str: String = row.get("agent_id");
    let parent_str: Option<String> = row.get("parent_id");
    let weave_str: Option<String> = row.get("weave_id");
    let status_str: String = row.get("status");
    let created_str: String = row.get("created_at");
    let completed_str: Option<String> = row.get("completed_at");

    let session_id = SessionId::from_uuid(
        uuid::Uuid::parse_str(&session_id_str)
            .map_err(|e| AlzinaError::Session(format!("invalid session UUID: {e}")))?,
    );

    let parent = parent_str
        .map(|s| {
            uuid::Uuid::parse_str(&s)
                .map(SessionId::from_uuid)
                .map_err(|e| AlzinaError::Session(format!("invalid parent UUID: {e}")))
        })
        .transpose()?;

    // Fetch child IDs for this node.
    let child_rows: Vec<String> =
        sqlx::query_scalar("SELECT session_id FROM sessions WHERE parent_id = ?")
            .bind(&session_id_str)
            .fetch_all(pool)
            .await
            .map_err(|e| AlzinaError::Session(format!("children query failed: {e}")))?;

    let children: Result<Vec<SessionId>, _> = child_rows
        .iter()
        .map(|s| {
            uuid::Uuid::parse_str(s)
                .map(SessionId::from_uuid)
                .map_err(|e| AlzinaError::Session(format!("invalid child UUID: {e}")))
        })
        .collect();

    let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| AlzinaError::Session(format!("invalid created_at: {e}")))?;

    let completed_at = completed_str
        .map(|s| {
            chrono::DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| AlzinaError::Session(format!("invalid completed_at: {e}")))
        })
        .transpose()?;

    Ok(SessionNode {
        session_id,
        agent_id: AgentId::new(agent_id_str),
        parent,
        children: children?,
        status: string_to_status(&status_str),
        weave_id: weave_str.map(WeaveId::new),
        created_at,
        completed_at,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn create_root_and_get() {
        let h = SessionHierarchy::in_memory().await.unwrap();
        let sid = SessionId::new();
        let aid = AgentId::new("smidr");
        let wid = WeaveId::new("runtime-migration");

        h.create_root(&sid, &aid, Some(&wid)).await.unwrap();

        let node = h.get(&sid).await.unwrap().expect("should exist");
        assert_eq!(node.agent_id.as_str(), "smidr");
        assert!(node.parent.is_none());
        assert_eq!(
            node.weave_id.as_ref().unwrap().as_str(),
            "runtime-migration"
        );
        assert!(matches!(node.status, SessionStatus::Pending));
    }

    #[tokio::test]
    async fn spawn_child_and_verify_parent_link() {
        // AC-O4: Sessions have correct parent links.
        let h = SessionHierarchy::in_memory().await.unwrap();
        let parent = SessionId::new();
        let child = SessionId::new();

        h.create_root(&parent, &AgentId::new("vefr"), Some(&WeaveId::new("test")))
            .await
            .unwrap();
        h.spawn_child(&parent, &child, &AgentId::new("muninn"), None)
            .await
            .unwrap();

        let child_node = h.get(&child).await.unwrap().expect("child should exist");
        assert_eq!(child_node.parent.as_ref().unwrap(), &parent);
        // Weave inherited from parent.
        assert_eq!(child_node.weave_id.as_ref().unwrap().as_str(), "test");
    }

    #[tokio::test]
    async fn children_returns_direct_children() {
        let h = SessionHierarchy::in_memory().await.unwrap();
        let parent = SessionId::new();
        let c1 = SessionId::new();
        let c2 = SessionId::new();

        h.create_root(&parent, &AgentId::new("vefr"), None)
            .await
            .unwrap();
        h.spawn_child(&parent, &c1, &AgentId::new("urdr"), None)
            .await
            .unwrap();
        h.spawn_child(&parent, &c2, &AgentId::new("skuld"), None)
            .await
            .unwrap();

        let kids = h.children(&parent).await.unwrap();
        assert_eq!(kids.len(), 2);
    }

    #[tokio::test]
    async fn complete_updates_status() {
        let h = SessionHierarchy::in_memory().await.unwrap();
        let sid = SessionId::new();

        h.create_root(&sid, &AgentId::new("galdr"), None)
            .await
            .unwrap();
        h.complete(&sid, EnvelopeStatus::Complete).await.unwrap();

        let node = h.get(&sid).await.unwrap().unwrap();
        assert!(matches!(
            node.status,
            SessionStatus::Complete(EnvelopeStatus::Complete)
        ));
        assert!(node.completed_at.is_some());
    }

    #[tokio::test]
    async fn check_depth_allows_within_limit() {
        let h = SessionHierarchy::in_memory().await.unwrap();

        let root = SessionId::new();
        h.create_root(&root, &AgentId::new("vefr"), None)
            .await
            .unwrap();

        let depth = h.check_depth(&root, DEFAULT_MAX_DEPTH).await.unwrap();
        assert_eq!(depth, 0);
    }

    #[tokio::test]
    async fn check_depth_rejects_exceeding_max() {
        // AC-O7: Depth > max → error.
        let h = SessionHierarchy::in_memory().await.unwrap();

        // Build a chain: root → s1 → s2 → s3 → s4 → s5
        let root = SessionId::new();
        h.create_root(&root, &AgentId::new("vefr"), None)
            .await
            .unwrap();

        let mut prev = root;
        for i in 1..=5 {
            let next = SessionId::new();
            h.spawn_child(&prev, &next, &AgentId::new(format!("agent-{i}")), None)
                .await
                .unwrap();
            prev = next;
        }

        // prev is now at depth 5. Spawning a child would be depth 6 > max 5.
        let result = h.check_depth(&prev, DEFAULT_MAX_DEPTH).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("depth exceeded"), "error was: {err}");
    }

    #[tokio::test]
    async fn weave_root_walks_to_ancestor_with_weave() {
        let h = SessionHierarchy::in_memory().await.unwrap();

        let root = SessionId::new();
        let mid = SessionId::new();
        let leaf = SessionId::new();

        h.create_root(
            &root,
            &AgentId::new("vefr"),
            Some(&WeaveId::new("migration")),
        )
        .await
        .unwrap();
        h.spawn_child(&root, &mid, &AgentId::new("smidr"), None)
            .await
            .unwrap();
        h.spawn_child(&mid, &leaf, &AgentId::new("galdr"), None)
            .await
            .unwrap();

        let weave_root = h
            .weave_root(&leaf)
            .await
            .unwrap()
            .expect("should find root");
        assert_eq!(weave_root.session_id, root);
        assert_eq!(weave_root.weave_id.as_ref().unwrap().as_str(), "migration");
    }

    #[tokio::test]
    async fn complete_nonexistent_session_errors() {
        let h = SessionHierarchy::in_memory().await.unwrap();
        let sid = SessionId::new();

        let result = h.complete(&sid, EnvelopeStatus::Error).await;
        assert!(result.is_err());
    }

    /// LE-1 regression test (P0-LE-1, 2026-04-30): under the previous
    /// `sqlite::memory:` URL, each pooled connection got its OWN private
    /// in-memory DB. The migration ran on one, but concurrent inserts
    /// landed on empty connections and failed with `no such table: sessions`.
    ///
    /// This test fires N parallel `create_root` calls so the SQLx pool is
    /// forced to hand out >1 connection in flight. Without the fix it
    /// fails reliably; with the fix all sessions land on the shared DB.
    #[tokio::test]
    async fn concurrent_create_root_all_succeed_after_pool_race_fix() {
        use futures::future::join_all;

        // Choose N > pool's max_connections (5) to exercise multiple
        // pool members concurrently.
        let h = Arc::new(SessionHierarchy::in_memory().await.unwrap());

        let mut handles = Vec::with_capacity(16);
        for _ in 0..16 {
            let h = Arc::clone(&h);
            handles.push(tokio::spawn(async move {
                let sid = SessionId::new();
                let aid = AgentId::new("vefr");
                h.create_root(&sid, &aid, None).await.map(|_| sid)
            }));
        }

        let results: Vec<_> = join_all(handles).await;
        let mut session_ids = Vec::with_capacity(results.len());
        for (i, joined) in results.into_iter().enumerate() {
            let outcome = joined
                .unwrap_or_else(|e| panic!("task {i} panicked: {e}"))
                .unwrap_or_else(|e| {
                    panic!("task {i} create_root failed (LE-1 regression — pool race): {e}")
                });
            session_ids.push(outcome);
        }

        // Every session must round-trip through the same shared DB.
        for sid in &session_ids {
            let node = h
                .get(sid)
                .await
                .expect("get must succeed against the shared in-memory DB")
                .expect("session row should exist after create_root");
            assert_eq!(node.session_id, *sid);
        }
    }

    #[tokio::test]
    async fn weave_id_explicit_override() {
        let h = SessionHierarchy::in_memory().await.unwrap();
        let root = SessionId::new();
        let child = SessionId::new();

        h.create_root(
            &root,
            &AgentId::new("vefr"),
            Some(&WeaveId::new("parent-weave")),
        )
        .await
        .unwrap();
        h.spawn_child(
            &root,
            &child,
            &AgentId::new("smidr"),
            Some(&WeaveId::new("child-weave")),
        )
        .await
        .unwrap();

        let node = h.get(&child).await.unwrap().unwrap();
        assert_eq!(node.weave_id.as_ref().unwrap().as_str(), "child-weave");
    }
}
