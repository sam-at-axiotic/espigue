//! TTD prompt rendering — native alzina port of consensus mustache templates.
//!
//! All 8 graph templates are ported to native Rust string rendering (no mustache
//! engine — alzina uses its own channel-substitution lexer, not mustache).
//! Mustache semantics are hand-translated per the plan specification:
//!
//! 1. Section iteration `{{#list}}...{{/list}}` → Rust loop with empty-skip
//! 2. Inverted sections `{{^last}}, {{/last}}` → comma-join via enumerate
//! 3. Triple-mustache `{{{fitness_feedback}}}` → raw string insert (NO HTML escape)
//! 4. Conditional `{{#fitness_feedback}}...{{/fitness_feedback}}` → if-block
//! 5. Output discipline: "output ONLY the XML block" preserved in each prompt

pub mod graph;
pub mod lit_review;
pub mod narrative;
pub mod render;
pub mod synthesis;
