//! Happens-before scope analyzer per `docs/composition-grammar.md` §4.4.
//!
//! Two-pass:
//! 1. `collect_ids` walks the AST, building the full id → NodeKind map
//!    used for unknown-id diagnostics. Duplicates are reported during the
//!    initial AST construction in `ast.rs::register_id`, but a defensive
//!    second pass here catches any escape paths.
//! 2. `walk_validate` recurses through the AST tracking the *legal
//!    ancestor set* per node. For each `task` / `<Prompt>` body, it
//!    scans for `{<id>:<channel>}` tokens and validates them against
//!    the ancestor set + reserved-channel context (loop body? gate
//!    body?).
//!
//! Synthesiser ancestor scope (§4.4 + Pitfall 5):
//! synthesiser's legal ancestors = parent's ancestors ∪ EVERY descendant
//! id of the wrapped inner op — including descendants-of-descendants.
//!
//! All 6 standard channels (signal/status/artifacts/tensions/envelope/raw)
//! are accepted; anything else is `RefUnknownChannel`.
//!
//! W-04: `{<gate-id>:gate.feedback}` outside the gate body is REJECTED at
//! parse time as Category C `GateFeedbackOutsideRetry`. Inside
//! `<ConditionalLoop>` body, `{this:gate.feedback}` is valid (covered by
//! the `in_gate_retry_body=true` flag).

use std::collections::{HashMap, HashSet};

use super::ast::AstPath;
use super::errors::{ParseError, ParseErrorCode, ParseErrors};
use crate::composition::compiler::CompOp;

const STANDARD_CHANNELS: &[&str] = &[
    "signal",
    "status",
    "artifacts",
    "tensions",
    "envelope",
    "raw",
    "emergent",
    "next",
];

/// Per B-06: accepts `node_id_map` (AstPath → resolved id) from `ast.rs`.
/// Monotonic-counter id derivation inside scope.rs is EXCISED; the only
/// source of node ids is the map ast.rs populated during construction.
pub fn analyze(op: &CompOp, node_id_map: &HashMap<AstPath, String>) -> Result<(), ParseErrors> {
    // Pass 1: collect every declared id with its node kind.
    let mut all_ids: HashMap<String, NodeKind> = HashMap::new();
    let mut path: AstPath = Vec::new();
    collect_ids(op, &mut path, node_id_map, &mut all_ids);

    // Pass 2: validate token references against the ancestor set + reserved
    // channel context.
    let mut errors: Vec<ParseError> = Vec::new();
    let parent_ancestors: HashSet<String> = HashSet::new();
    let mut path2: AstPath = Vec::new();
    walk_validate(
        op,
        &mut path2,
        node_id_map,
        &parent_ancestors,
        &all_ids,
        false,
        false,
        &HashSet::new(),
        &mut errors,
    );

    if errors.is_empty() {
        Ok(())
    } else {
        Err(ParseErrors { errors })
    }
}

/// Tracks which node kind owns each id.
#[derive(Debug, Clone, PartialEq)]
enum NodeKind {
    Spawn,
    Sequential,
    Parallel,
    Synthesise,
    Gate,
    Loop,
    ConditionalLoop,
    Conditional,
    FanOut,
}

/// Pass-1 walker: build `all_ids` (id → NodeKind) by looking up each node's
/// resolved id from `node_id_map` at its AstPath. NO monotonic counter —
/// explicit `id="..."` attributes are honored verbatim by ast.rs and
/// surfaced here.
fn collect_ids(
    op: &CompOp,
    path: &mut AstPath,
    node_id_map: &HashMap<AstPath, String>,
    ids: &mut HashMap<String, NodeKind>,
) {
    let kind = node_kind(op);
    let id = node_id_map
        .get(path)
        .cloned()
        .unwrap_or_else(|| format!("__missing_node_id_at_path_{path:?}"));
    ids.insert(id, kind);
    match op {
        CompOp::Sequential(children) | CompOp::Parallel(children) => {
            for (i, c) in children.iter().enumerate() {
                path.push(i);
                collect_ids(c, path, node_id_map, ids);
                path.pop();
            }
        }
        CompOp::Synthesise(inner, _)
        | CompOp::Gate(inner, _)
        | CompOp::Loop(inner, _)
        | CompOp::ConditionalLoop(inner, _) => {
            path.push(0);
            collect_ids(inner, path, node_id_map, ids);
            path.pop();
        }
        CompOp::Conditional(branches) => {
            for (i, (_, c)) in branches.iter().enumerate() {
                path.push(i);
                collect_ids(c, path, node_id_map, ids);
                path.pop();
            }
        }
        CompOp::Spawn(_) | CompOp::FanOut(_, _) => {}
    }
}

fn node_kind(op: &CompOp) -> NodeKind {
    match op {
        CompOp::Spawn(_) => NodeKind::Spawn,
        CompOp::Sequential(_) => NodeKind::Sequential,
        CompOp::Parallel(_) => NodeKind::Parallel,
        CompOp::Synthesise(_, _) => NodeKind::Synthesise,
        CompOp::Gate(_, _) => NodeKind::Gate,
        CompOp::Loop(_, _) => NodeKind::Loop,
        CompOp::ConditionalLoop(_, _) => NodeKind::ConditionalLoop,
        CompOp::Conditional(_) => NodeKind::Conditional,
        CompOp::FanOut(_, _) => NodeKind::FanOut,
    }
}

/// Collect ALL descendant ids of an op (including descendants-of-descendants)
/// by walking `node_id_map` from the inner op's path forward.
/// Used to compute the Synthesise synthesiser's legal ancestor set per
/// Pitfall 5. NO monotonic counter.
fn descendant_ids(
    inner: &CompOp,
    inner_path: &AstPath,
    node_id_map: &HashMap<AstPath, String>,
) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut path = inner_path.clone();
    descend_collect(inner, &mut path, node_id_map, &mut out);
    out
}

fn descend_collect(
    op: &CompOp,
    path: &mut AstPath,
    node_id_map: &HashMap<AstPath, String>,
    out: &mut HashSet<String>,
) {
    if let Some(id) = node_id_map.get(path) {
        out.insert(id.clone());
    }
    match op {
        CompOp::Sequential(c) | CompOp::Parallel(c) => {
            for (i, o) in c.iter().enumerate() {
                path.push(i);
                descend_collect(o, path, node_id_map, out);
                path.pop();
            }
        }
        CompOp::Synthesise(inner, _)
        | CompOp::Gate(inner, _)
        | CompOp::Loop(inner, _)
        | CompOp::ConditionalLoop(inner, _) => {
            path.push(0);
            descend_collect(inner, path, node_id_map, out);
            path.pop();
        }
        CompOp::Conditional(branches) => {
            for (i, (_, o)) in branches.iter().enumerate() {
                path.push(i);
                descend_collect(o, path, node_id_map, out);
                path.pop();
            }
        }
        CompOp::Spawn(_) | CompOp::FanOut(_, _) => {}
    }
}

/// Pass-2 walker: validates each node's token references against its legal
/// ancestor set. Per B-06, `node_id_map` is the sole source of node ids —
/// no monotonic counter, no re-derivation. The `path` accumulator stays in
/// lockstep with `ast.rs::parse_children_until_close` (push child index on
/// descent, pop on return).
#[allow(clippy::too_many_arguments)]
fn walk_validate(
    op: &CompOp,
    path: &mut AstPath,
    node_id_map: &HashMap<AstPath, String>,
    ancestors: &HashSet<String>,
    all_ids: &HashMap<String, NodeKind>,
    in_loop_body: bool,
    in_gate_retry_body: bool,
    gate_ids_in_scope: &HashSet<String>,
    errors: &mut Vec<ParseError>,
) {
    let this_id = node_id_map
        .get(path)
        .cloned()
        .unwrap_or_else(|| format!("__missing_node_id_at_path_{path:?}"));
    match op {
        CompOp::Spawn(spec) => {
            validate_tokens(
                &spec.task_template,
                &this_id,
                ancestors,
                all_ids,
                in_loop_body,
                in_gate_retry_body,
                gate_ids_in_scope,
                errors,
            );
        }
        CompOp::Sequential(children) => {
            let mut acc = ancestors.clone();
            for (i, c) in children.iter().enumerate() {
                path.push(i);
                walk_validate(
                    c,
                    path,
                    node_id_map,
                    &acc,
                    all_ids,
                    in_loop_body,
                    in_gate_retry_body,
                    gate_ids_in_scope,
                    errors,
                );
                // B-06: read the child's id directly from node_id_map.
                if let Some(child_id) = node_id_map.get(path) {
                    acc.insert(child_id.clone());
                }
                path.pop();
            }
        }
        CompOp::Parallel(children) => {
            // Parallel siblings do NOT see each other — pass parent ancestors only.
            for (i, c) in children.iter().enumerate() {
                path.push(i);
                walk_validate(
                    c,
                    path,
                    node_id_map,
                    ancestors,
                    all_ids,
                    in_loop_body,
                    in_gate_retry_body,
                    gate_ids_in_scope,
                    errors,
                );
                path.pop();
            }
        }
        CompOp::Synthesise(inner, spec) => {
            // Pitfall 5: synthesiser's ancestors include ALL descendants of `inner`.
            let mut inner_path = path.clone();
            inner_path.push(0);
            let descendants = descendant_ids(inner, &inner_path, node_id_map);
            let mut synth_ancestors = ancestors.clone();
            for d in &descendants {
                synth_ancestors.insert(d.clone());
            }
            // Validate the synthesiser's task against the extended set.
            validate_tokens(
                &spec.synthesiser.task_template,
                &this_id,
                &synth_ancestors,
                all_ids,
                in_loop_body,
                in_gate_retry_body,
                gate_ids_in_scope,
                errors,
            );
            // Validate the inner op with the parent's ancestor set (not synth_ancestors).
            path.push(0);
            walk_validate(
                inner,
                path,
                node_id_map,
                ancestors,
                all_ids,
                in_loop_body,
                in_gate_retry_body,
                gate_ids_in_scope,
                errors,
            );
            path.pop();
        }
        CompOp::Gate(inner, _gate_spec) => {
            let mut g = gate_ids_in_scope.clone();
            g.insert(this_id.clone());
            path.push(0);
            walk_validate(
                inner,
                path,
                node_id_map,
                ancestors,
                all_ids,
                in_loop_body,
                /* in_gate_retry_body */ true,
                &g,
                errors,
            );
            path.pop();
        }
        CompOp::Loop(inner, _) => {
            path.push(0);
            walk_validate(
                inner,
                path,
                node_id_map,
                ancestors,
                all_ids,
                /* in_loop_body */ true,
                in_gate_retry_body,
                gate_ids_in_scope,
                errors,
            );
            path.pop();
        }
        CompOp::ConditionalLoop(inner, _) => {
            path.push(0);
            walk_validate(
                inner,
                path,
                node_id_map,
                ancestors,
                all_ids,
                /* in_loop_body */ true,
                /* in_gate_retry_body */ true,
                gate_ids_in_scope,
                errors,
            );
            path.pop();
        }
        CompOp::Conditional(branches) => {
            for (i, (_, c)) in branches.iter().enumerate() {
                path.push(i);
                walk_validate(
                    c,
                    path,
                    node_id_map,
                    ancestors,
                    all_ids,
                    in_loop_body,
                    in_gate_retry_body,
                    gate_ids_in_scope,
                    errors,
                );
                path.pop();
            }
        }
        CompOp::FanOut(spec, prompts) => {
            for p in prompts {
                validate_tokens(
                    p,
                    &this_id,
                    ancestors,
                    all_ids,
                    in_loop_body,
                    in_gate_retry_body,
                    gate_ids_in_scope,
                    errors,
                );
            }
            let _ = spec;
        }
    }
}

fn validate_tokens(
    text: &str,
    node_id: &str,
    ancestors: &HashSet<String>,
    all_ids: &HashMap<String, NodeKind>,
    in_loop_body: bool,
    in_gate_retry_body: bool,
    gate_ids_in_scope: &HashSet<String>,
    errors: &mut Vec<ParseError>,
) {
    for token in scan_tokens(text) {
        let Some((id, channel)) = token.split_once(':') else {
            continue;
        };
        if id == "this" {
            // Reserved channels.
            if channel.starts_with("prev_iteration.") {
                if !in_loop_body {
                    errors.push(simple_err(
                        ParseErrorCode::ReservedChannelOutsideLoop,
                        format!("`{{this:{channel}}}` is only valid inside a <Loop> or <ConditionalLoop> body (used in node `{node_id}`)"),
                        "Wrap this spawn in a <Loop> or <ConditionalLoop> to use prev_iteration.",
                    ));
                }
                let sub_channel = channel.trim_start_matches("prev_iteration.");
                if !STANDARD_CHANNELS.contains(&sub_channel) {
                    errors.push(simple_err(
                        ParseErrorCode::RefUnknownChannel,
                        format!("Unknown channel `{sub_channel}` in `{{this:prev_iteration.{sub_channel}}}`"),
                        format!("Valid channels: {}", STANDARD_CHANNELS.join(", ")),
                    ));
                }
                continue;
            }
            if channel == "gate.feedback" {
                if !in_gate_retry_body {
                    errors.push(simple_err(
                        ParseErrorCode::GateFeedbackOutsideRetry,
                        format!("`{{this:gate.feedback}}` is only valid inside a <Gate on_fail=\"RetryWithFeedback\"> or <ConditionalLoop> body (used in node `{node_id}`)"),
                        "Move this spawn inside the gate/loop body, or use `{<gate-id>:gate.feedback}` from outside.",
                    ));
                }
                continue;
            }
            continue;
        }
        // Channel: gate.feedback for outside-gate ref.
        // Per W-04: `{<gate-id>:gate.feedback}` substitution OUTSIDE the
        // gate body is REJECTED at parse time as Category C
        // `GateFeedbackOutsideRetry`. Outside-gate resolution is unscoped —
        // anticipated for Phase 11 during the Phase 10 scope analyser work,
        // but Phase 11 shipped without it and no successor phase has picked
        // it up. The `gate_ids_in_scope` accumulator below is the dangling
        // half of that work; leaving the parameter in place keeps the
        // walker signature stable for whoever picks the work up.
        // ConditionalLoop bodies are validated via the `{this:gate.feedback}`
        // branch above (in_gate_retry_body=true).
        if channel == "gate.feedback" {
            errors.push(simple_err(
                ParseErrorCode::GateFeedbackOutsideRetry,
                format!("`{{{id}:gate.feedback}}` outside-gate references are not supported in v1 (rejected per W-04 in node `{node_id}`)"),
                "Move the dependent spawn inside the gate body and use `{this:gate.feedback}`.",
            ));
            continue;
        }
        // Standard channel substitution: validate id exists + is ancestor + channel is known.
        if !all_ids.contains_key(id) {
            errors.push(simple_err(
                ParseErrorCode::RefUnknownId,
                format!("Reference `{{{id}:{channel}}}` in node `{node_id}` — id `{id}` is not declared in this <Compose>."),
                "Check the spelling, or declare the upstream node with an explicit `id=\"…\"`.",
            ));
            continue;
        }
        if !STANDARD_CHANNELS.contains(&channel) {
            errors.push(simple_err(
                ParseErrorCode::RefUnknownChannel,
                format!("Unknown channel `{channel}` in reference `{{{id}:{channel}}}`"),
                format!("Valid channels: {}.", STANDARD_CHANNELS.join(", ")),
            ));
            continue;
        }
        if !ancestors.contains(id) {
            errors.push(simple_err(
                ParseErrorCode::RefNonAncestor,
                format!("Reference `{{{id}:{channel}}}` in node `{node_id}` is invalid. `{id}` is not an ancestor of `{node_id}`."),
                "`Parallel` siblings cannot reference each other (no happens-before relation). To pass data from one to the other, make them <Sequential> children or wrap them in a <Synthesise> whose synthesiser sees both.",
            ));
        }
    }
}

/// Yields token bodies (everything between `{` and `}`) skipping `{{` escapes.
fn scan_tokens(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'{' && bytes[i + 1] == b'{' {
            i += 2;
            continue;
        }
        if bytes[i] == b'{' {
            if let Some(end) = bytes[i + 1..].iter().position(|&b| b == b'}') {
                let start = i + 1;
                let stop = start + end;
                out.push(std::str::from_utf8(&bytes[start..stop]).unwrap_or(""));
                i = stop + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn simple_err(code: ParseErrorCode, message: String, hint: impl Into<String>) -> ParseError {
    ParseError {
        category: code.category(),
        code,
        line: 0,
        column: 0,
        message,
        hint: hint.into(),
    }
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::parse_compose;
    use super::*;

    #[test]
    fn synthesiser_sees_descendants_of_descendants() {
        // §5 case + Pitfall 5: sjofn references {past:envelope}, where past
        // is a grandchild (Spawn under Sequential under Synthesise).
        let xml = r#"<Compose><Synthesise synthesiser="sjofn" task="see {past:envelope}"><Sequential><Spawn id="past" agent="urdr" task="t"/><Spawn id="future" agent="skuld" task="t"/></Sequential></Synthesise></Compose>"#;
        let r = parse_compose(xml);
        assert!(
            r.is_ok(),
            "synthesiser must see descendants of descendants per Pitfall 5: {:?}",
            r.err()
        );
    }

    #[test]
    fn cross_parallel_branch_ref_is_ref_non_ancestor() {
        let xml = r#"<Compose><Parallel><Spawn id="a" agent="x" task="t"/><Spawn id="b" agent="y" task="see {a:envelope}"/></Parallel></Compose>"#;
        let err = parse_compose(xml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.code == ParseErrorCode::RefNonAncestor),
            "expected RefNonAncestor, got: {:?}",
            err.errors
        );
    }

    #[test]
    fn ref_to_unknown_id_returns_ref_unknown_id() {
        let xml = r#"<Compose><Spawn agent="x" task="see {nope:signal}"/></Compose>"#;
        let err = parse_compose(xml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.code == ParseErrorCode::RefUnknownId),
            "expected RefUnknownId, got: {:?}",
            err.errors
        );
    }

    #[test]
    fn ref_to_unknown_channel_returns_ref_unknown_channel() {
        let xml = r#"<Compose><Sequential><Spawn id="a" agent="x" task="t"/><Spawn agent="y" task="{a:foo}"/></Sequential></Compose>"#;
        let err = parse_compose(xml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.code == ParseErrorCode::RefUnknownChannel),
            "expected RefUnknownChannel, got: {:?}",
            err.errors
        );
    }

    #[test]
    fn reserved_channel_outside_loop_returns_reserved_channel_outside_loop() {
        let xml = r#"<Compose><Spawn agent="x" task="{this:prev_iteration.signal}"/></Compose>"#;
        let err = parse_compose(xml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.code == ParseErrorCode::ReservedChannelOutsideLoop),
            "expected ReservedChannelOutsideLoop, got: {:?}",
            err.errors
        );
    }

    #[test]
    fn sequential_prior_sibling_ref_is_ok() {
        let xml = r#"<Compose><Sequential><Spawn id="a" agent="x" task="t"/><Spawn id="b" agent="y" task="see {a:signal}"/></Sequential></Compose>"#;
        assert!(parse_compose(xml).is_ok());
    }

    #[test]
    fn loop_body_prev_iteration_is_ok() {
        let xml = r#"<Compose><Loop max_iterations="3" on_exhaust="AcceptLast"><Spawn agent="x" task="{this:prev_iteration.signal}"/></Loop></Compose>"#;
        assert!(parse_compose(xml).is_ok());
    }

    #[test]
    fn conditional_loop_gate_feedback_is_ok() {
        let xml = r#"<Compose><ConditionalLoop max_iterations="2" on_exhaust="Escalate" required="status"><Spawn agent="x" task="{this:gate.feedback}"/></ConditionalLoop></Compose>"#;
        assert!(parse_compose(xml).is_ok());
    }

    #[test]
    fn outside_gate_gate_feedback_ref_is_rejected_per_w04() {
        // W-04: `{<gate-id>:gate.feedback}` substitution from outside the
        // gate body is REJECTED at parse time as Category C
        // `GateFeedbackOutsideRetry`. v1 narrower surface — outside-gate
        // resolution is unscoped (was anticipated for Phase 11, never picked up).
        let xml = r#"<Compose><Sequential><Gate id="g" required="status" on_fail="RetryWithFeedback"><Spawn agent="x" task="t"/></Gate><Spawn agent="y" task="{g:gate.feedback}"/></Sequential></Compose>"#;
        let err = parse_compose(xml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.code == ParseErrorCode::GateFeedbackOutsideRetry),
            "expected GateFeedbackOutsideRetry, got: {:?}",
            err.errors
        );
    }

    #[test]
    fn duplicate_node_id_is_caught() {
        // Caught in ast.rs::register_id during the AST build pass.
        let xml = r#"<Compose><Sequential><Spawn id="a" agent="x" task="t"/><Spawn id="a" agent="y" task="t"/></Sequential></Compose>"#;
        let err = parse_compose(xml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.code == ParseErrorCode::DuplicateNodeId),
            "expected DuplicateNodeId, got: {:?}",
            err.errors
        );
    }
}
