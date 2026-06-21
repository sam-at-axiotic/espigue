//! SQLite-backed implementation of ADK-Rust's `SessionService` trait.
//!
//! Bridges between Alzina's `SessionHierarchy` (which tracks parent→child
//! relationships, weave associations, and depth) and ADK-Rust's `Runner`
//! (which needs a `SessionService` for session CRUD and event storage).
//!
//! This is a thin adapter: hierarchy management stays in `SessionHierarchy`,
//! while ADK session state (key-value pairs, events) lives in dedicated
//! SQLite tables managed here.

use adk_rust::session::{
    CreateRequest, DeleteRequest, GetRequest, ListRequest, Session as AdkSession, SessionService,
};
use adk_rust::{AdkError, Event, Result as AdkResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::Row;
use sqlx::sqlite::SqlitePool;
use std::collections::HashMap;
use tracing::debug;

/// SQLite-backed ADK SessionService.
///
/// Manages ADK session state (key-value state, events) in SQLite tables
/// separate from the hierarchy table. The hierarchy tracks lineage and
/// governance; this service tracks ADK Runner's operational state.
pub struct SqliteSessionService {
    pool: SqlitePool,
}

impl SqliteSessionService {
    /// Create a new service backed by the given pool.
    pub async fn new(pool: SqlitePool) -> Result<Self, String> {
        let svc = Self { pool };
        svc.migrate().await?;
        Ok(svc)
    }

    async fn migrate(&self) -> Result<(), String> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS adk_sessions (
                session_id TEXT PRIMARY KEY,
                app_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| format!("adk_sessions migration failed: {e}"))?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS adk_session_state (
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (session_id, key),
                FOREIGN KEY (session_id) REFERENCES adk_sessions(session_id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| format!("adk_session_state migration failed: {e}"))?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS adk_session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES adk_sessions(session_id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| format!("adk_session_events migration failed: {e}"))?;

        Ok(())
    }

    async fn load_state(&self, session_id: &str) -> AdkResult<HashMap<String, Value>> {
        let rows = sqlx::query("SELECT key, value FROM adk_session_state WHERE session_id = ?")
            .bind(session_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| AdkError::Session(format!("state load failed: {e}")))?;

        let mut state = HashMap::new();
        for row in &rows {
            let key: String = row.get("key");
            let value_str: String = row.get("value");
            if let Ok(v) = serde_json::from_str(&value_str) {
                state.insert(key, v);
            }
        }
        Ok(state)
    }

    async fn load_events(&self, session_id: &str) -> AdkResult<Vec<Event>> {
        let rows = sqlx::query(
            "SELECT event_json FROM adk_session_events WHERE session_id = ? ORDER BY id ASC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AdkError::Session(format!("events load failed: {e}")))?;

        let mut events = Vec::with_capacity(rows.len());
        for row in &rows {
            let json_str: String = row.get("event_json");
            let event: Event = serde_json::from_str(&json_str)
                .map_err(|e| AdkError::Session(format!("event deserialize failed: {e}")))?;
            events.push(event);
        }
        Ok(events)
    }

    async fn load_session(&self, session_id: &str) -> AdkResult<Box<dyn AdkSession>> {
        let row = sqlx::query(
            "SELECT session_id, app_name, user_id, updated_at FROM adk_sessions WHERE session_id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| AdkError::Session(format!("session load failed: {e}")))?
        .ok_or_else(|| AdkError::Session(format!("session not found: {session_id}")))?;

        let id: String = row.get("session_id");
        let app_name: String = row.get("app_name");
        let user_id: String = row.get("user_id");
        let updated_str: String = row.get("updated_at");

        let updated_at = DateTime::parse_from_rfc3339(&updated_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let state = self.load_state(&id).await?;
        let events = self.load_events(&id).await?;

        Ok(Box::new(SqliteSession {
            id,
            app_name,
            user_id,
            state: SqliteState { data: state },
            events: SqliteEvents { events },
            updated_at,
        }))
    }
}

#[async_trait]
impl SessionService for SqliteSessionService {
    async fn create(&self, req: CreateRequest) -> AdkResult<Box<dyn AdkSession>> {
        let id = req
            .session_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        sqlx::query(
            "INSERT INTO adk_sessions (session_id, app_name, user_id, updated_at) VALUES (?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(&req.app_name)
        .bind(&req.user_id)
        .bind(&now_str)
        .execute(&self.pool)
        .await
        .map_err(|e| AdkError::Session(format!("create session failed: {e}")))?;

        // Persist initial state.
        for (key, value) in &req.state {
            let value_str = serde_json::to_string(value)
                .map_err(|e| AdkError::Session(format!("state serialize failed: {e}")))?;
            sqlx::query("INSERT INTO adk_session_state (session_id, key, value) VALUES (?, ?, ?)")
                .bind(&id)
                .bind(key)
                .bind(&value_str)
                .execute(&self.pool)
                .await
                .map_err(|e| AdkError::Session(format!("state insert failed: {e}")))?;
        }

        debug!("created ADK session {id}");

        Ok(Box::new(SqliteSession {
            id,
            app_name: req.app_name,
            user_id: req.user_id,
            state: SqliteState { data: req.state },
            events: SqliteEvents { events: Vec::new() },
            updated_at: now,
        }))
    }

    async fn get(&self, req: GetRequest) -> AdkResult<Box<dyn AdkSession>> {
        self.load_session(&req.session_id).await
    }

    async fn list(&self, req: ListRequest) -> AdkResult<Vec<Box<dyn AdkSession>>> {
        let rows =
            sqlx::query("SELECT session_id FROM adk_sessions WHERE app_name = ? AND user_id = ?")
                .bind(&req.app_name)
                .bind(&req.user_id)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| AdkError::Session(format!("list sessions failed: {e}")))?;

        let mut sessions = Vec::with_capacity(rows.len());
        for row in &rows {
            let id: String = row.get("session_id");
            sessions.push(self.load_session(&id).await?);
        }
        Ok(sessions)
    }

    async fn delete(&self, req: DeleteRequest) -> AdkResult<()> {
        // Cascade deletes state + events via FK constraints.
        sqlx::query("DELETE FROM adk_sessions WHERE session_id = ?")
            .bind(&req.session_id)
            .execute(&self.pool)
            .await
            .map_err(|e| AdkError::Session(format!("delete session failed: {e}")))?;

        debug!("deleted ADK session {}", req.session_id);
        Ok(())
    }

    async fn append_event(&self, session_id: &str, event: Event) -> AdkResult<()> {
        let event_json = serde_json::to_string(&event)
            .map_err(|e| AdkError::Session(format!("event serialize failed: {e}")))?;
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO adk_session_events (session_id, event_json, created_at) VALUES (?, ?, ?)",
        )
        .bind(session_id)
        .bind(&event_json)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| AdkError::Session(format!("append event failed: {e}")))?;

        // Update timestamp.
        sqlx::query("UPDATE adk_sessions SET updated_at = ? WHERE session_id = ?")
            .bind(&now)
            .bind(session_id)
            .execute(&self.pool)
            .await
            .map_err(|e| AdkError::Session(format!("update timestamp failed: {e}")))?;

        Ok(())
    }
}

// ── ADK Session/State/Events implementations ────────────────────────────────

struct SqliteSession {
    id: String,
    app_name: String,
    user_id: String,
    state: SqliteState,
    events: SqliteEvents,
    updated_at: DateTime<Utc>,
}

impl AdkSession for SqliteSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn app_name(&self) -> &str {
        &self.app_name
    }

    fn user_id(&self) -> &str {
        &self.user_id
    }

    fn state(&self) -> &dyn adk_rust::session::State {
        &self.state
    }

    fn events(&self) -> &dyn adk_rust::session::Events {
        &self.events
    }

    fn last_update_time(&self) -> DateTime<Utc> {
        self.updated_at
    }
}

struct SqliteState {
    data: HashMap<String, Value>,
}

impl adk_rust::session::State for SqliteState {
    fn get(&self, key: &str) -> Option<Value> {
        self.data.get(key).cloned()
    }

    fn set(&mut self, key: String, value: Value) {
        self.data.insert(key, value);
    }

    fn all(&self) -> HashMap<String, Value> {
        self.data.clone()
    }
}

struct SqliteEvents {
    events: Vec<Event>,
}

impl adk_rust::session::Events for SqliteEvents {
    fn all(&self) -> Vec<Event> {
        self.events.clone()
    }

    fn len(&self) -> usize {
        self.events.len()
    }

    fn at(&self, index: usize) -> Option<&Event> {
        self.events.get(index)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn test_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn create_and_get_session() {
        let pool = test_pool().await;
        let svc = SqliteSessionService::new(pool).await.unwrap();

        let session = svc
            .create(CreateRequest {
                app_name: "alzina".to_string(),
                user_id: "operator".to_string(),
                session_id: Some("test-sess".to_string()),
                state: HashMap::new(),
            })
            .await
            .unwrap();

        assert_eq!(session.id(), "test-sess");
        assert_eq!(session.app_name(), "alzina");

        let retrieved = svc
            .get(GetRequest {
                app_name: "alzina".to_string(),
                user_id: "operator".to_string(),
                session_id: "test-sess".to_string(),
                num_recent_events: None,
                after: None,
            })
            .await
            .unwrap();

        assert_eq!(retrieved.id(), "test-sess");
    }

    #[tokio::test]
    async fn append_and_retrieve_events() {
        let pool = test_pool().await;
        let svc = SqliteSessionService::new(pool).await.unwrap();

        svc.create(CreateRequest {
            app_name: "alzina".to_string(),
            user_id: "operator".to_string(),
            session_id: Some("evt-sess".to_string()),
            state: HashMap::new(),
        })
        .await
        .unwrap();

        let event = Event::new("test-invocation");
        svc.append_event("evt-sess", event).await.unwrap();

        let session = svc
            .get(GetRequest {
                app_name: "alzina".to_string(),
                user_id: "operator".to_string(),
                session_id: "evt-sess".to_string(),
                num_recent_events: None,
                after: None,
            })
            .await
            .unwrap();

        assert_eq!(session.events().len(), 1);
    }

    #[tokio::test]
    async fn list_sessions_by_user() {
        let pool = test_pool().await;
        let svc = SqliteSessionService::new(pool).await.unwrap();

        svc.create(CreateRequest {
            app_name: "alzina".to_string(),
            user_id: "user-a".to_string(),
            session_id: Some("s1".to_string()),
            state: HashMap::new(),
        })
        .await
        .unwrap();

        svc.create(CreateRequest {
            app_name: "alzina".to_string(),
            user_id: "user-a".to_string(),
            session_id: Some("s2".to_string()),
            state: HashMap::new(),
        })
        .await
        .unwrap();

        svc.create(CreateRequest {
            app_name: "alzina".to_string(),
            user_id: "user-b".to_string(),
            session_id: Some("s3".to_string()),
            state: HashMap::new(),
        })
        .await
        .unwrap();

        let sessions = svc
            .list(ListRequest {
                app_name: "alzina".to_string(),
                user_id: "user-a".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn delete_session() {
        let pool = test_pool().await;
        let svc = SqliteSessionService::new(pool).await.unwrap();

        svc.create(CreateRequest {
            app_name: "alzina".to_string(),
            user_id: "operator".to_string(),
            session_id: Some("del-sess".to_string()),
            state: HashMap::new(),
        })
        .await
        .unwrap();

        svc.delete(DeleteRequest {
            app_name: "alzina".to_string(),
            user_id: "operator".to_string(),
            session_id: "del-sess".to_string(),
        })
        .await
        .unwrap();

        let result = svc
            .get(GetRequest {
                app_name: "alzina".to_string(),
                user_id: "operator".to_string(),
                session_id: "del-sess".to_string(),
                num_recent_events: None,
                after: None,
            })
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn initial_state_persisted() {
        let pool = test_pool().await;
        let svc = SqliteSessionService::new(pool).await.unwrap();

        let mut state = HashMap::new();
        state.insert("key1".to_string(), serde_json::json!("value1"));
        state.insert("key2".to_string(), serde_json::json!(42));

        svc.create(CreateRequest {
            app_name: "alzina".to_string(),
            user_id: "operator".to_string(),
            session_id: Some("state-sess".to_string()),
            state,
        })
        .await
        .unwrap();

        let session = svc
            .get(GetRequest {
                app_name: "alzina".to_string(),
                user_id: "operator".to_string(),
                session_id: "state-sess".to_string(),
                num_recent_events: None,
                after: None,
            })
            .await
            .unwrap();

        assert_eq!(
            session.state().get("key1"),
            Some(serde_json::json!("value1"))
        );
        assert_eq!(session.state().get("key2"), Some(serde_json::json!(42)));
    }
}
