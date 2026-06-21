//! Message types for channel communication.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// The kind of channel a message originates from or is destined to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelKind {
    Webchat,
    Discord,
    Telegram,
    Whatsapp,
    Cron,
    /// Extension point for channels not covered by built-in variants.
    Custom(String),
}

impl ChannelKind {
    /// Create a custom channel kind.
    pub fn custom(name: impl Into<String>) -> Self {
        Self::Custom(name.into())
    }
}

impl std::fmt::Display for ChannelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Webchat => write!(f, "webchat"),
            Self::Discord => write!(f, "discord"),
            Self::Telegram => write!(f, "telegram"),
            Self::Whatsapp => write!(f, "whatsapp"),
            Self::Cron => write!(f, "cron"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// Inbound message from any channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMessage {
    pub channel: ChannelKind,
    pub sender: String,
    pub content: String,
    pub thread_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, Value>,
}

/// Outbound response to a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundMessage {
    pub channel: ChannelKind,
    pub thread_id: Option<String>,
    pub content: String,
    pub attachments: Vec<Attachment>,
}

/// File or media attachment on an outbound message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub filename: String,
    pub content_type: String,
    pub data: Vec<u8>,
}
