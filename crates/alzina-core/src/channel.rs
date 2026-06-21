//! Channel adapter trait and router.

use crate::error::AlzinaResult;
use crate::message::{ChannelKind, OutboundMessage};
use async_trait::async_trait;

/// Adapter for a specific communication channel.
#[async_trait]
pub trait ChannelAdapter: Send + Sync {
    /// Channel identity.
    fn kind(&self) -> ChannelKind;

    /// Start listening for inbound messages.
    async fn connect(&self) -> AlzinaResult<()>;

    /// Send a message out through this channel.
    async fn send(&self, msg: OutboundMessage) -> AlzinaResult<()>;

    /// Graceful disconnect.
    async fn disconnect(&self) -> AlzinaResult<()>;
}

/// Routes messages between channels and the orchestration layer.
pub struct ChannelRouter {
    adapters: Vec<Box<dyn ChannelAdapter>>,
}

impl ChannelRouter {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    /// Register a channel adapter.
    pub fn register(&mut self, adapter: Box<dyn ChannelAdapter>) {
        self.adapters.push(adapter);
    }

    /// Get the number of registered adapters.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }
}

impl Default for ChannelRouter {
    fn default() -> Self {
        Self::new()
    }
}
