//! Phase 3: Composition nodes — agent spawn, gate, join, synthesis, fan-out, router.

pub mod fanout_node;
pub mod gate_node;
pub mod join_node;
pub mod router_node;
pub mod spawn_node;
pub mod synthesis_node;

pub use fanout_node::{FanOutNode, FanOutStrategy};
pub use gate_node::GateNode;
pub use join_node::{JoinNode, JoinResult};
pub use router_node::{RoutePredicate, RouterNode};
pub use spawn_node::SpawnNode;
pub use synthesis_node::SynthesisNode;
