//! Quality gate trait — evaluates agent output against criteria.

use crate::composition::{GateCriteria, GateVerdict};
use crate::envelope::Envelope;
use crate::error::AlzinaResult;
use async_trait::async_trait;

/// Evaluates an envelope against gate criteria.
#[async_trait]
pub trait QualityGate: Send + Sync {
    /// Evaluate an envelope against the given criteria.
    async fn evaluate(
        &self,
        envelope: &Envelope,
        criteria: &GateCriteria,
    ) -> AlzinaResult<GateVerdict>;
}
