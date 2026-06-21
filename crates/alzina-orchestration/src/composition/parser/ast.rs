//! XML → CompOp AST construction.
//!
//! Per `docs/composition-grammar.md` §3 — one `parse_*` function per
//! operator. The dispatcher (`parse_inner`) matches the opening tag name
//! and routes to the matching constructor. Each `parse_*` MUST:
//! - Validate required attributes (Category B `MissingRequiredAttribute`)
//! - Reject unknown attributes (Category A `UnknownAttribute`)
//! - Enforce child-count constraints per §2 quick reference table
//! - Auto-generate node ids per §4.1 when absent
//! - Capture any preceding XML comment into `rationale_map` per §1.4
//!
//! Doc round-trip §8 invariant: tests in this module are tied 1:1 to the
//! `docs/composition-grammar.md` §3 operator examples. A grammar change
//! in the doc REQUIRES a parser change here, a tool-description update in
//! `chat.rs::build_dispatch_tools`, and a narrative update in
//! `config/agents/vefr/narrative.md` — the 4-site round trip is one logical
//! commit.

use std::collections::HashMap;
use std::time::Duration;

use quick_xml::events::Event;

use alzina_core::identity::AgentId;

use super::LeafIdent;
use super::errors::{ParseErrorCode, ParseErrors};
use super::tokenizer::{EventCursor, SourceMap, name_has_namespace, parse_errs};
use alzina_core::composition::{ExhaustAction, GateCriteria, GateFailAction};
use alzina_core::envelope::EnvelopeStatus;

use crate::composition::compiler::{
    CompOp, ConditionalLoopSpec, GateSpec, LoopSpec, Predicate, SpawnSpec, SynthesisSpec,
};

/// Per B-06: AstPath = Vec<usize> of child indices from root.
/// Stable, pointer-independent — survives moves of the AST.
pub type AstPath = Vec<usize>;

/// Parser working state — monotonic id counter, collected rationale, and the
/// resolved-id map used downstream by `scope.rs` (per B-06).
struct ParserState {
    /// Per-doc monotonic index used for auto-generated node ids per §4.1.
    next_index: u32,
    /// Captured rationale comments per §1.4, keyed by node_id.
    rationale_map: HashMap<String, String>,
    /// Last comment text seen — attached to the next operator opener.
    pending_rationale: Option<String>,
    /// All node ids declared in the document (for duplicate detection).
    declared_ids: HashMap<String, (u32, u32)>,
    /// AST-path accumulator (Vec<usize> of child indices from root).
    current_path: AstPath,
    /// AstPath → resolved node_id (explicit-then-auto). scope.rs consumes
    /// this map directly; monotonic-counter re-derivation is EXCISED.
    node_id_map: HashMap<AstPath, String>,
}

/// Top-level parse entry. Called from `parse_compose` in `mod.rs`.
///
/// Per B-06: returns `node_id_map` as a fourth element so `scope.rs::analyze`
/// consumes the resolved-id map directly. Explicit-id-bearing nodes (e.g.
/// `<Spawn id="past" .../>`) are honored verbatim; the monotonic-counter
/// fallback inside scope.rs is EXCISED. The §5 canonical composite's
/// `{past:signal}` references depend on this — see Plan 02 truth #4-5.
pub fn parse(
    xml: &str,
    sm: &SourceMap,
) -> Result<
    (
        CompOp,
        HashMap<String, String>,
        Vec<LeafIdent>,
        HashMap<AstPath, String>,
    ),
    ParseErrors,
> {
    let mut cursor = EventCursor::new(xml);
    let mut state = ParserState {
        next_index: 0,
        rationale_map: HashMap::new(),
        pending_rationale: None,
        declared_ids: HashMap::new(),
        current_path: Vec::new(),
        node_id_map: HashMap::new(),
    };

    // 1. Skip optional XML declaration; expect <Compose> opener.
    expect_compose_root(&mut cursor, sm)?;
    // 2. Parse exactly one inner operator. The root sits at AstPath = [].
    let inner = parse_inner(&mut cursor, &mut state, sm)?;
    // 3. Expect </Compose> closer.
    expect_compose_close(&mut cursor, sm)?;
    // 4. Collect leaves from the constructed CompOp, using the resolved node_id
    //    map (B-06 / IN-02 fix) so that explicit id="..." attributes are reflected
    //    in LeafIdent.node_id rather than falling back to the agent name.
    let leaves = collect_leaves_with_ids(&inner, &state.node_id_map);
    Ok((inner, state.rationale_map, leaves, state.node_id_map))
}

fn expect_compose_root(cur: &mut EventCursor<'_>, sm: &SourceMap) -> Result<(), ParseErrors> {
    loop {
        let pos = cur.byte_offset_before_event();
        match cur.next() {
            Ok(Event::Decl(_)) => continue, // <?xml ... ?> ignored
            Ok(Event::DocType(_)) => {
                return Err(parse_errs(
                    ParseErrorCode::DtdRejected,
                    pos,
                    sm,
                    "DTD declaration rejected per §1.5",
                    "Remove the <!DOCTYPE ...> declaration; the parser does not allow DTDs or external entities.",
                ));
            }
            Ok(Event::PI(_)) => {
                return Err(parse_errs(
                    ParseErrorCode::ProcessingInstructionRejected,
                    pos,
                    sm,
                    "Processing instruction rejected per §1.5",
                    "Remove the <?...?> processing instruction.",
                ));
            }
            Ok(Event::Comment(_)) => {
                // pre-root comments are not attached to any node — drop.
                continue;
            }
            Ok(Event::Start(ref e)) => {
                if name_has_namespace(e.name().as_ref()) {
                    return Err(parse_errs(
                        ParseErrorCode::NamespaceRejected,
                        pos,
                        sm,
                        "XML namespaces rejected per §1.5",
                        "Remove any `xmlns:` declaration or namespace-prefixed tags.",
                    ));
                }
                if e.name().as_ref() != b"Compose" {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    return Err(parse_errs(
                        ParseErrorCode::UnknownTag,
                        pos,
                        sm,
                        format!("expected <Compose> root, got <{tag}>"),
                        "Every composition plan must be wrapped in <Compose>...</Compose>.",
                    ));
                }
                // <Compose> has zero attributes per §1.1; reject any.
                for attr in e.attributes().flatten() {
                    let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                    return Err(parse_errs(
                        ParseErrorCode::UnknownAttribute,
                        pos,
                        sm,
                        format!("<Compose> takes no attributes, got `{key}`"),
                        "<Compose> is the document root with no attributes.",
                    ));
                }
                return Ok(());
            }
            Ok(Event::Empty(_)) => {
                return Err(parse_errs(
                    ParseErrorCode::WrongChildCount,
                    pos,
                    sm,
                    "<Compose> must contain exactly one operator",
                    "Wrap your root operator inside <Compose>...</Compose>.",
                ));
            }
            Ok(Event::Eof) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    "unexpected end of input; expected <Compose>",
                    "Wrap your plan in <Compose>...</Compose>.",
                ));
            }
            Err(e) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("XML error: {e}"),
                    "Check for unclosed tags, mismatched quotes, or invalid characters.",
                ));
            }
            _ => continue,
        }
    }
}

fn expect_compose_close(cur: &mut EventCursor<'_>, sm: &SourceMap) -> Result<(), ParseErrors> {
    loop {
        let pos = cur.byte_offset_before_event();
        match cur.next() {
            Ok(Event::End(ref e)) if e.name().as_ref() == b"Compose" => return Ok(()),
            Ok(Event::Text(_)) | Ok(Event::Comment(_)) => continue,
            Ok(Event::Eof) => return Ok(()), // permissive: allow EOF after root
            Ok(_) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    "expected </Compose>",
                    "Ensure your plan has a matching </Compose> closing tag.",
                ));
            }
            Err(e) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("XML error reading </Compose>: {e}"),
                    "Check the closing tag.",
                ));
            }
        }
    }
}

/// Dispatcher — reads the next opener and routes by tag name.
/// Also handles pending comments for rationale capture.
fn parse_inner(
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
) -> Result<CompOp, ParseErrors> {
    loop {
        let pos = cur.byte_offset_before_event();
        match cur.next() {
            Ok(Event::Comment(c)) => {
                state.pending_rationale =
                    Some(String::from_utf8_lossy(c.as_ref()).trim().to_string());
                continue;
            }
            Ok(Event::DocType(_)) => {
                return Err(parse_errs(
                    ParseErrorCode::DtdRejected,
                    pos,
                    sm,
                    "DTD declaration rejected per §1.5",
                    "Remove the <!DOCTYPE ...>.",
                ));
            }
            Ok(Event::PI(_)) => {
                return Err(parse_errs(
                    ParseErrorCode::ProcessingInstructionRejected,
                    pos,
                    sm,
                    "Processing instruction rejected per §1.5",
                    "Remove the <?...?>.",
                ));
            }
            Ok(Event::Start(ref e)) => {
                let name = e.name().as_ref().to_vec();
                if name_has_namespace(&name) {
                    return Err(parse_errs(
                        ParseErrorCode::NamespaceRejected,
                        pos,
                        sm,
                        "XML namespaces rejected per §1.5",
                        "Remove `xmlns:` or namespace prefixes.",
                    ));
                }
                let rationale = state.pending_rationale.take();
                // Clone attrs we need before e is moved
                let attrs = collect_attrs_raw(e, sm, pos)?;
                return dispatch_operator(&name, attrs, false, cur, state, sm, pos, rationale);
            }
            Ok(Event::Empty(ref e)) => {
                let name = e.name().as_ref().to_vec();
                if name_has_namespace(&name) {
                    return Err(parse_errs(
                        ParseErrorCode::NamespaceRejected,
                        pos,
                        sm,
                        "XML namespaces rejected per §1.5",
                        "Remove `xmlns:` or namespace prefixes.",
                    ));
                }
                let rationale = state.pending_rationale.take();
                let attrs = collect_attrs_raw(e, sm, pos)?;
                return dispatch_operator(&name, attrs, true, cur, state, sm, pos, rationale);
            }
            Ok(Event::End(_)) | Ok(Event::Eof) => {
                return Err(parse_errs(
                    ParseErrorCode::WrongChildCount,
                    pos,
                    sm,
                    "expected an operator tag, got closing/eof",
                    "Check that the parent operator has the required children.",
                ));
            }
            Err(e) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("XML error: {e}"),
                    "Check for unclosed tags or invalid characters.",
                ));
            }
            _ => continue,
        }
    }
}

/// Low-level: collect raw (key, value) attribute pairs from a BytesStart event,
/// rejecting unknown or namespaced attributes later in dispatch_operator.
fn collect_attrs_raw(
    e: &quick_xml::events::BytesStart,
    sm: &SourceMap,
    pos: usize,
) -> Result<HashMap<String, String>, ParseErrors> {
    let mut out = HashMap::new();
    for attr in e.attributes().flatten() {
        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
        if name_has_namespace(attr.key.as_ref()) {
            return Err(parse_errs(
                ParseErrorCode::NamespaceRejected,
                pos,
                sm,
                format!("attribute `{key}` uses XML namespace prefix"),
                "",
            ));
        }
        let val = attr
            .unescape_value()
            .map_err(|e| {
                parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("attribute parse error: {e}"),
                    "",
                )
            })?
            .into_owned();
        out.insert(key, val);
    }
    Ok(out)
}

fn dispatch_operator(
    name: &[u8],
    raw_attrs: HashMap<String, String>,
    is_empty: bool,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    match name {
        b"Spawn" => parse_spawn(raw_attrs, is_empty, cur, state, sm, pos, rationale),
        b"Sequential" => parse_sequential(raw_attrs, cur, state, sm, pos, rationale),
        b"Parallel" => parse_parallel(raw_attrs, cur, state, sm, pos, rationale),
        b"Synthesise" => parse_synthesise(raw_attrs, cur, state, sm, pos, rationale),
        b"Gate" => parse_gate(raw_attrs, cur, state, sm, pos, rationale),
        b"Loop" => parse_loop(raw_attrs, cur, state, sm, pos, rationale),
        b"ConditionalLoop" => parse_conditional_loop(raw_attrs, cur, state, sm, pos, rationale),
        b"Conditional" => parse_conditional(raw_attrs, cur, state, sm, pos, rationale),
        b"FanOut" => parse_fanout(raw_attrs, cur, state, sm, pos, rationale),
        other => {
            let tag = String::from_utf8_lossy(other).into_owned();
            let suggestion = did_you_mean(&tag);
            let hint = format!(
                "{suggestion}Valid operators: Spawn, Sequential, Parallel, Synthesise, Gate, Loop, ConditionalLoop, Conditional, FanOut."
            );
            Err(parse_errs(
                ParseErrorCode::UnknownTag,
                pos,
                sm,
                format!("Unknown operator <{tag}>"),
                hint,
            ))
        }
    }
}

fn did_you_mean(tag: &str) -> String {
    let known = [
        "Spawn",
        "Sequential",
        "Parallel",
        "Synthesise",
        "Gate",
        "Loop",
        "ConditionalLoop",
        "Conditional",
        "FanOut",
    ];
    for k in known {
        if k.to_lowercase() == tag.to_lowercase() {
            return format!("Did you mean <{k}>? ");
        }
    }
    for k in known {
        if k.chars().next() == tag.chars().next() && (k.len() as i32 - tag.len() as i32).abs() <= 2
        {
            return format!("Did you mean <{k}>? ");
        }
    }
    String::new()
}

// ── validate attrs ─────────────────────────────────────────────────────────────

fn validate_attrs(
    raw: &HashMap<String, String>,
    sm: &SourceMap,
    pos: usize,
    op: &str,
    allowed: &[&str],
) -> Result<(), ParseErrors> {
    for key in raw.keys() {
        if !allowed.contains(&key.as_str()) {
            let suggestion = closest_attr(key, allowed);
            let hint = format!(
                "{}Valid attributes on <{op}>: {}.",
                suggestion,
                allowed.join(", ")
            );
            return Err(parse_errs(
                ParseErrorCode::UnknownAttribute,
                pos,
                sm,
                format!("Unknown attribute `{key}` on <{op}>"),
                hint,
            ));
        }
    }
    Ok(())
}

fn closest_attr(needle: &str, allowed: &[&str]) -> String {
    for a in allowed {
        if a.to_lowercase() == needle.to_lowercase() {
            return format!("Did you mean `{a}`? ");
        }
    }
    String::new()
}

fn require_attr(
    attrs: &HashMap<String, String>,
    key: &str,
    op: &str,
    sm: &SourceMap,
    pos: usize,
) -> Result<String, ParseErrors> {
    attrs.get(key).cloned().ok_or_else(|| {
        parse_errs(
            ParseErrorCode::MissingRequiredAttribute,
            pos,
            sm,
            format!("<{op}> requires attribute `{key}` but it was missing"),
            match key {
                "synthesiser" => "Synthesis nodes name the agent that performs the fan-in. Example: <Synthesise synthesiser=\"sjofn\"> ... </Synthesise>".to_string(),
                "agent" => format!("<{op}> requires the `agent` attribute naming a configured specialist."),
                "task" => format!("<{op}> requires the `task` attribute with the task description."),
                _ => format!("Add `{key}=\"...\"` to <{op}>."),
            },
        )
    })
}

fn parse_duration(s: &str, sm: &SourceMap, pos: usize) -> Result<Duration, ParseErrors> {
    let split_at = s.find(|c: char| c.is_alphabetic()).unwrap_or(s.len());
    let (num_part, unit) = s.split_at(split_at);
    let n: u64 = num_part.parse().map_err(|_| {
        parse_errs(
            ParseErrorCode::AttributeValueInvalid,
            pos,
            sm,
            format!("invalid duration `{s}`"),
            "Use forms like 120s, 5m, 2h.",
        )
    })?;
    match unit {
        "s" => Ok(Duration::from_secs(n)),
        "m" => Ok(Duration::from_secs(n * 60)),
        "h" => Ok(Duration::from_secs(n * 3600)),
        other => Err(parse_errs(
            ParseErrorCode::AttributeValueInvalid,
            pos,
            sm,
            format!("unknown duration unit `{other}`"),
            "Use s, m, or h.",
        )),
    }
}

fn parse_usize(s: &str, sm: &SourceMap, pos: usize) -> Result<usize, ParseErrors> {
    s.parse().map_err(|_| {
        parse_errs(
            ParseErrorCode::AttributeValueInvalid,
            pos,
            sm,
            format!("expected integer, got `{s}`"),
            "",
        )
    })
}

fn register_id(
    id: &str,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
) -> Result<(), ParseErrors> {
    let (line, col) = sm.locate(pos);
    if let Some(prev) = state.declared_ids.get(id) {
        return Err(parse_errs(
            ParseErrorCode::DuplicateNodeId,
            pos,
            sm,
            format!(
                "Duplicate node id `{id}` at line {line}. First declared at line {}.",
                prev.0
            ),
            "Node ids must be unique within a <Compose>.",
        ));
    }
    state.declared_ids.insert(id.to_string(), (line, col));
    Ok(())
}

// ── parse_spawn ────────────────────────────────────────────────────────────────

fn parse_spawn(
    raw_attrs: HashMap<String, String>,
    is_empty: bool,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(
        &raw_attrs,
        sm,
        pos,
        "Spawn",
        &["agent", "task", "id", "model", "timeout"],
    )?;
    let agent = require_attr(&raw_attrs, "agent", "Spawn", sm, pos)?;
    let task = require_attr(&raw_attrs, "task", "Spawn", sm, pos)?;
    let model = raw_attrs.get("model").cloned();
    let timeout = raw_attrs
        .get("timeout")
        .map(|s| parse_duration(s, sm, pos))
        .transpose()?;

    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("{agent}_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp the resolved id at this AST path.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    if !is_empty {
        // <Spawn> must self-close — consume any inner content + the </Spawn>.
        expect_self_close(cur, sm, "Spawn")?;
    }

    Ok(CompOp::Spawn(SpawnSpec {
        agent_id: AgentId::new(&agent),
        task_template: task,
        model_override: model,
        timeout,
        session_id_override: None,
        weave_id: None,
        // Plan 10-05 (followup): thread the parser-assigned id so the compiler's
        // `execute_spawn` invokes the leaf hook with the SAME key that
        // `dispatch_compose` used in `static_leaf_map`. Without this the hook
        // lookup misses and the registry wedges (see compiler.rs SpawnSpec doc).
        node_id: Some(id.clone()),
        // Phase 1B substrate cascade: parser does not own dispatch_id —
        // it's set by the daemon at the dispatch boundary. Composed
        // sub-specs inherit the parent dispatch_id via the leaf-hook /
        // execute_spawn path; the parser itself emits None.
        dispatch_id: None,
    }))
}

fn expect_self_close(
    cur: &mut EventCursor<'_>,
    sm: &SourceMap,
    op: &str,
) -> Result<(), ParseErrors> {
    loop {
        let pos = cur.byte_offset_before_event();
        match cur.next() {
            Ok(Event::End(ref end)) => {
                if end.name().as_ref() == op.as_bytes() {
                    return Ok(());
                }
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("expected </{op}>"),
                    format!("<{op}> must be self-closing or have a matching close tag."),
                ));
            }
            Ok(Event::Text(_)) => continue, // tolerate whitespace
            Ok(Event::Eof) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("unexpected EOF inside <{op}>"),
                    format!("Close <{op}> with /> or </{op}>."),
                ));
            }
            Ok(_) => {
                return Err(parse_errs(
                    ParseErrorCode::WrongChildCount,
                    pos,
                    sm,
                    format!("<{op}> takes no children"),
                    format!("Use <{op} .../> as self-closing."),
                ));
            }
            Err(e) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("XML error: {e}"),
                    "",
                ));
            }
        }
    }
}

// ── parse_sequential ───────────────────────────────────────────────────────────

fn parse_sequential(
    raw_attrs: HashMap<String, String>,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(&raw_attrs, sm, pos, "Sequential", &["id"])?;
    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("sequential_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp Sequential's own resolved id.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    let children = parse_children_until_close(cur, state, sm, "Sequential")?;
    if children.is_empty() {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            "<Sequential> has zero children",
            "Sequential nodes need at least one operation. If you only have one operation, drop the <Sequential> wrapper and use the operation directly.",
        ));
    }
    Ok(CompOp::Sequential(children))
}

// ── parse_parallel ────────────────────────────────────────────────────────────

fn parse_parallel(
    raw_attrs: HashMap<String, String>,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(&raw_attrs, sm, pos, "Parallel", &["id"])?;
    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("parallel_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp Parallel's resolved id.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    let children = parse_children_until_close(cur, state, sm, "Parallel")?;
    if children.len() < 2 {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            "<Parallel> requires ≥2 children (single-child or empty parallel adds no parallelism)",
            "Use ≥2 children, or replace with the operation directly.",
        ));
    }
    Ok(CompOp::Parallel(children))
}

// ── parse_synthesise ──────────────────────────────────────────────────────────

fn parse_synthesise(
    raw_attrs: HashMap<String, String>,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(
        &raw_attrs,
        sm,
        pos,
        "Synthesise",
        &["synthesiser", "task", "id"],
    )?;
    let synth_agent = require_attr(&raw_attrs, "synthesiser", "Synthesise", sm, pos)?;
    let task = raw_attrs.get("task").cloned();
    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("synthesise_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp Synthesise's resolved id.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    let children = parse_children_until_close(cur, state, sm, "Synthesise")?;
    if children.len() != 1 {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            "<Synthesise> requires exactly 1 child operator",
            "Wrap exactly one inner operator (typically <Parallel> or <FanOut>) inside <Synthesise>.",
        ));
    }
    // The synthesiser's task_template is intentionally empty here: the runtime
    // resolves the synthesis task from SynthesisSpec.task at execution time
    // (using DEFAULT_SYNTHESIS_PROMPT as the fallback per §3.4). This separates
    // the parser-captured task string from the SpawnSpec.task_template field
    // which is only used for non-synthesis spawns.
    let synthesiser_spec = SpawnSpec {
        agent_id: AgentId::new(&synth_agent),
        task_template: String::new(),
        model_override: None,
        timeout: None,
        session_id_override: None,
        weave_id: None,
        // E2 / D11-05: stamp the synthesiser's resolved id so the
        // compiler's `execute_synthesis` consumes it and the daemon's
        // DaemonLeafHook lookup hits the pre-registered entry. Closes
        // the wedge for <Synthesise> compositions.
        node_id: Some(id.clone()),
        // Phase 1B substrate cascade: see parse_spawn — parser emits None.
        dispatch_id: None,
    };
    Ok(CompOp::Synthesise(
        Box::new(children.into_iter().next().unwrap()),
        SynthesisSpec {
            synthesiser: synthesiser_spec,
            task,
        },
    ))
}

// ── parse_gate ────────────────────────────────────────────────────────────────

fn parse_gate(
    raw_attrs: HashMap<String, String>,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(
        &raw_attrs,
        sm,
        pos,
        "Gate",
        &[
            "required",
            "on_fail",
            "id",
            "status_must_be",
            "max_tensions",
        ],
    )?;
    let required_str = require_attr(&raw_attrs, "required", "Gate", sm, pos)?;
    let envelope_required_fields: Vec<String> = required_str
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let on_fail_str = require_attr(&raw_attrs, "on_fail", "Gate", sm, pos)?;
    let on_fail = parse_gate_fail_action(&on_fail_str, sm, pos)?;
    let status_must_be = raw_attrs
        .get("status_must_be")
        .map(|s| parse_envelope_status(s, sm, pos))
        .transpose()?;
    let max_tensions = raw_attrs
        .get("max_tensions")
        .map(|s| parse_usize(s, sm, pos))
        .transpose()?;

    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("gate_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp Gate's resolved id.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    let children = parse_children_until_close(cur, state, sm, "Gate")?;
    if children.len() != 1 {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            "<Gate> requires exactly 1 child operator",
            "Wrap exactly one inner operator inside <Gate>.",
        ));
    }
    Ok(CompOp::Gate(
        Box::new(children.into_iter().next().unwrap()),
        GateSpec {
            criteria: GateCriteria {
                envelope_required_fields,
                status_must_be,
                max_tensions,
            },
            on_fail,
        },
    ))
}

fn parse_gate_fail_action(
    s: &str,
    sm: &SourceMap,
    pos: usize,
) -> Result<GateFailAction, ParseErrors> {
    if s == "RetryWithFeedback" {
        return Ok(GateFailAction::RetryWithFeedback);
    }
    if s == "Escalate" {
        return Ok(GateFailAction::Escalate);
    }
    // Degrade:"reason" — the only inline-payload value in the grammar (§3.5).
    if let Some(rest) = s.strip_prefix("Degrade:") {
        let reason = rest.trim_matches('"').to_string();
        return Ok(GateFailAction::Degrade(reason));
    }
    Err(parse_errs(
        ParseErrorCode::AttributeValueInvalid,
        pos,
        sm,
        format!("`on_fail` must be RetryWithFeedback | Escalate | Degrade:\"reason\", got `{s}`"),
        "See docs/composition-grammar.md §3.5 for the three valid forms.",
    ))
}

// ── parse_loop ────────────────────────────────────────────────────────────────

fn parse_loop(
    raw_attrs: HashMap<String, String>,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(
        &raw_attrs,
        sm,
        pos,
        "Loop",
        &["max_iterations", "on_exhaust", "id"],
    )?;
    let max_iterations = parse_usize(
        &require_attr(&raw_attrs, "max_iterations", "Loop", sm, pos)?,
        sm,
        pos,
    )?;
    let on_exhaust = parse_exhaust_action(
        &require_attr(&raw_attrs, "on_exhaust", "Loop", sm, pos)?,
        sm,
        pos,
    )?;

    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("loop_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp Loop's resolved id.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    let children = parse_children_until_close(cur, state, sm, "Loop")?;
    if children.len() != 1 {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            "<Loop> requires exactly 1 child operator",
            "Wrap exactly one operator (the loop body) inside <Loop>.",
        ));
    }
    Ok(CompOp::Loop(
        Box::new(children.into_iter().next().unwrap()),
        LoopSpec {
            max_iterations,
            on_exhaust,
        },
    ))
}

fn parse_exhaust_action(s: &str, sm: &SourceMap, pos: usize) -> Result<ExhaustAction, ParseErrors> {
    match s {
        "Escalate" => Ok(ExhaustAction::Escalate),
        "AcceptLast" => Ok(ExhaustAction::AcceptLast),
        "Fail" => Ok(ExhaustAction::Fail),
        other => Err(parse_errs(
            ParseErrorCode::AttributeValueInvalid,
            pos,
            sm,
            format!("`on_exhaust` must be Escalate | AcceptLast | Fail, got `{other}`"),
            "See docs/composition-grammar.md §3.6.",
        )),
    }
}

fn parse_envelope_status(
    s: &str,
    sm: &SourceMap,
    pos: usize,
) -> Result<EnvelopeStatus, ParseErrors> {
    match s {
        "Complete" => Ok(EnvelopeStatus::Complete),
        "Partial" => Ok(EnvelopeStatus::Partial),
        "Error" => Ok(EnvelopeStatus::Error),
        other => Err(parse_errs(
            ParseErrorCode::AttributeValueInvalid,
            pos,
            sm,
            format!("`status_must_be` must be Complete | Partial | Error, got `{other}`"),
            "See docs/composition-grammar.md §3.5.",
        )),
    }
}

// ── parse_conditional_loop ────────────────────────────────────────────────────

fn parse_conditional_loop(
    raw_attrs: HashMap<String, String>,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(
        &raw_attrs,
        sm,
        pos,
        "ConditionalLoop",
        &[
            "max_iterations",
            "on_exhaust",
            "required",
            "status_must_be",
            "max_tensions",
            "id",
        ],
    )?;
    let max_iterations = parse_usize(
        &require_attr(&raw_attrs, "max_iterations", "ConditionalLoop", sm, pos)?,
        sm,
        pos,
    )?;
    let on_exhaust = parse_exhaust_action(
        &require_attr(&raw_attrs, "on_exhaust", "ConditionalLoop", sm, pos)?,
        sm,
        pos,
    )?;
    let required_str = require_attr(&raw_attrs, "required", "ConditionalLoop", sm, pos)?;
    let envelope_required_fields: Vec<String> = required_str
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let status_must_be = raw_attrs
        .get("status_must_be")
        .map(|s| parse_envelope_status(s, sm, pos))
        .transpose()?;
    let max_tensions = raw_attrs
        .get("max_tensions")
        .map(|s| parse_usize(s, sm, pos))
        .transpose()?;

    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("conditional_loop_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp ConditionalLoop's resolved id.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    let children = parse_children_until_close(cur, state, sm, "ConditionalLoop")?;
    if children.len() != 1 {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            "<ConditionalLoop> requires exactly 1 child operator",
            "",
        ));
    }
    Ok(CompOp::ConditionalLoop(
        Box::new(children.into_iter().next().unwrap()),
        ConditionalLoopSpec {
            gate: GateCriteria {
                envelope_required_fields,
                status_must_be,
                max_tensions,
            },
            max_iterations,
            on_exhaust,
        },
    ))
}

// ── parse_conditional ─────────────────────────────────────────────────────────

fn parse_conditional(
    raw_attrs: HashMap<String, String>,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(&raw_attrs, sm, pos, "Conditional", &["id"])?;
    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("conditional_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp Conditional's resolved id.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    let mut branches: Vec<(Predicate, CompOp)> = Vec::new();
    let mut otherwise_seen = false;
    loop {
        let cpos = cur.byte_offset_before_event();
        match cur.next() {
            Ok(Event::Start(ref e2)) => {
                let n = e2.name().as_ref().to_vec();
                if name_has_namespace(&n) {
                    return Err(parse_errs(
                        ParseErrorCode::NamespaceRejected,
                        cpos,
                        sm,
                        "namespace rejected",
                        "",
                    ));
                }
                if otherwise_seen {
                    return Err(parse_errs(
                        ParseErrorCode::ChildOrderViolation,
                        cpos,
                        sm,
                        "<Otherwise> must be the last child of <Conditional>",
                        "",
                    ));
                }
                let branch_attrs = collect_attrs_raw(e2, sm, cpos)?;
                match n.as_slice() {
                    b"When" => {
                        let pred = parse_when_predicate(&branch_attrs, sm, cpos)?;
                        let branch_idx = branches.len();
                        state.current_path.push(branch_idx);
                        let inner_result = parse_inner(cur, state, sm);
                        state.current_path.pop();
                        let inner = inner_result?;
                        consume_until_end_tag(cur, sm, "When")?;
                        branches.push((pred, inner));
                    }
                    b"Otherwise" => {
                        // <Otherwise> has no attributes per §3.8.
                        for key in branch_attrs.keys() {
                            return Err(parse_errs(
                                ParseErrorCode::UnknownAttribute,
                                cpos,
                                sm,
                                format!("<Otherwise> takes no attributes, got `{key}`"),
                                "",
                            ));
                        }
                        let branch_idx = branches.len();
                        state.current_path.push(branch_idx);
                        let inner_result = parse_inner(cur, state, sm);
                        state.current_path.pop();
                        let inner = inner_result?;
                        consume_until_end_tag(cur, sm, "Otherwise")?;
                        branches.push((Predicate::Default, inner));
                        otherwise_seen = true;
                    }
                    other => {
                        let tag = String::from_utf8_lossy(other).into_owned();
                        return Err(parse_errs(
                            ParseErrorCode::UnknownTag,
                            cpos,
                            sm,
                            format!("Unknown <Conditional> child <{tag}>"),
                            "<Conditional> children must be <When> or <Otherwise>.",
                        ));
                    }
                }
            }
            Ok(Event::End(ref e2)) if e2.name().as_ref() == b"Conditional" => break,
            Ok(Event::Comment(c)) => {
                state.pending_rationale =
                    Some(String::from_utf8_lossy(c.as_ref()).trim().to_string());
                continue;
            }
            Ok(Event::Text(_)) => continue,
            Ok(Event::Eof) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    cpos,
                    sm,
                    "unexpected EOF inside <Conditional>",
                    "",
                ));
            }
            Err(e) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    cpos,
                    sm,
                    format!("XML error: {e}"),
                    "",
                ));
            }
            _ => continue,
        }
    }
    if branches.is_empty() {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            "<Conditional> requires ≥ 1 <When> child (and optional <Otherwise>)",
            "",
        ));
    }
    Ok(CompOp::Conditional(branches))
}

fn parse_when_predicate(
    attrs: &HashMap<String, String>,
    sm: &SourceMap,
    pos: usize,
) -> Result<Predicate, ParseErrors> {
    // Validate: only channel, equals, exists are allowed.
    for key in attrs.keys() {
        if !["channel", "equals", "exists"].contains(&key.as_str()) {
            return Err(parse_errs(
                ParseErrorCode::UnknownAttribute,
                pos,
                sm,
                format!("Unknown attribute `{key}` on <When>"),
                "Valid attributes on <When>: channel, equals, exists.",
            ));
        }
    }
    let channel = require_attr(attrs, "channel", "When", sm, pos)?;
    if let Some(eq) = attrs.get("equals") {
        let value: serde_json::Value = serde_json::from_str(eq)
            .or_else(|_| serde_json::from_str(&format!("\"{eq}\"")))
            .map_err(|_| {
                parse_errs(
                    ParseErrorCode::AttributeValueInvalid,
                    pos,
                    sm,
                    format!("<When equals=\"{eq}\"> — not valid JSON scalar"),
                    "Use a JSON scalar: bare string, integer, float, true, false, null.",
                )
            })?;
        return Ok(Predicate::StateEquals { channel, value });
    }
    if let Some(ex) = attrs.get("exists") {
        if ex != "true" {
            return Err(parse_errs(
                ParseErrorCode::AttributeValueInvalid,
                pos,
                sm,
                "<When exists=\"...\"> only accepts `true`; negate via <Otherwise>",
                "Use exists=\"true\" or move the alternate branch into <Otherwise>.",
            ));
        }
        return Ok(Predicate::StateExists { channel });
    }
    Err(parse_errs(
        ParseErrorCode::MissingRequiredAttribute,
        pos,
        sm,
        "<When> requires either `equals` or `exists`",
        "Example: <When channel=\"x:envelope.signal\" equals=\"ready\"> ... </When>",
    ))
}

// ── parse_fanout ──────────────────────────────────────────────────────────────

fn parse_fanout(
    raw_attrs: HashMap<String, String>,
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    pos: usize,
    rationale: Option<String>,
) -> Result<CompOp, ParseErrors> {
    validate_attrs(
        &raw_attrs,
        sm,
        pos,
        "FanOut",
        &["agent", "model", "timeout", "id"],
    )?;
    let agent = require_attr(&raw_attrs, "agent", "FanOut", sm, pos)?;
    let model = raw_attrs.get("model").cloned();
    let timeout = raw_attrs
        .get("timeout")
        .map(|s| parse_duration(s, sm, pos))
        .transpose()?;
    let idx = state.next_index;
    state.next_index += 1;
    let id = raw_attrs
        .get("id")
        .cloned()
        .unwrap_or_else(|| format!("fanout_{idx}"));
    register_id(&id, state, sm, pos)?;
    // B-06: stamp FanOut's resolved id.
    state
        .node_id_map
        .insert(state.current_path.clone(), id.clone());
    if let Some(r) = rationale {
        state.rationale_map.insert(id.clone(), r);
    }

    let mut prompts: Vec<String> = Vec::new();
    loop {
        let cpos = cur.byte_offset_before_event();
        match cur.next() {
            Ok(Event::Start(ref p)) => {
                if p.name().as_ref() != b"Prompt" {
                    let tag = String::from_utf8_lossy(p.name().as_ref()).into_owned();
                    return Err(parse_errs(
                        ParseErrorCode::UnknownTag,
                        cpos,
                        sm,
                        format!("<FanOut> only accepts <Prompt> children, got <{tag}>"),
                        "",
                    ));
                }
                // Read text until </Prompt>.
                let mut text = String::new();
                loop {
                    match cur.next() {
                        Ok(Event::Text(t)) => text.push_str(&String::from_utf8_lossy(t.as_ref())),
                        Ok(Event::End(ref end)) if end.name().as_ref() == b"Prompt" => break,
                        Ok(Event::Eof) => {
                            return Err(parse_errs(
                                ParseErrorCode::MalformedXml,
                                cpos,
                                sm,
                                "unexpected EOF inside <Prompt>",
                                "",
                            ));
                        }
                        Err(e) => {
                            return Err(parse_errs(
                                ParseErrorCode::MalformedXml,
                                cpos,
                                sm,
                                format!("{e}"),
                                "",
                            ));
                        }
                        _ => continue,
                    }
                }
                prompts.push(text);
            }
            Ok(Event::End(ref end)) if end.name().as_ref() == b"FanOut" => break,
            Ok(Event::Text(_)) => continue,
            Ok(Event::Eof) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    cpos,
                    sm,
                    "unexpected EOF inside <FanOut>",
                    "",
                ));
            }
            Err(e) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    cpos,
                    sm,
                    format!("{e}"),
                    "",
                ));
            }
            _ => continue,
        }
    }
    if prompts.is_empty() {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            "<FanOut> requires ≥ 1 <Prompt> child",
            "",
        ));
    }
    if prompts.len() > 100 {
        return Err(parse_errs(
            ParseErrorCode::WrongChildCount,
            pos,
            sm,
            format!(
                "<FanOut> has {} prompts, exceeds MAX_FANOUT_PROMPTS=100",
                prompts.len()
            ),
            "Split into multiple <FanOut> compositions or trim prompts.",
        ));
    }
    Ok(CompOp::FanOut(
        SpawnSpec {
            agent_id: AgentId::new(&agent),
            task_template: String::new(),
            model_override: model,
            timeout,
            session_id_override: None,
            weave_id: None,
            // E2 / D11-05: stamp the fanout's resolved root id; per-prompt
            // ids `{fanout_id}_p{i}` are minted at LeafIdent collection time
            // in `collect_leaves_inner_with_ids` since FanOut produces N
            // leaves from one SpawnSpec.
            node_id: Some(id.clone()),
            // Phase 1B substrate cascade: parser emits None.
            dispatch_id: None,
        },
        prompts,
    ))
}

// ── children loop ─────────────────────────────────────────────────────────────

fn parse_children_until_close(
    cur: &mut EventCursor<'_>,
    state: &mut ParserState,
    sm: &SourceMap,
    op: &str,
) -> Result<Vec<CompOp>, ParseErrors> {
    let mut children = Vec::new();
    loop {
        let pos = cur.byte_offset_before_event();
        // Peek by reading then checking what kind of event we got.
        match cur.next() {
            Ok(Event::Start(ref e)) => {
                let name = e.name().as_ref().to_vec();
                let raw_attrs = collect_attrs_raw(e, sm, pos)?;
                if name_has_namespace(&name) {
                    return Err(parse_errs(
                        ParseErrorCode::NamespaceRejected,
                        pos,
                        sm,
                        "namespace in child tag",
                        "",
                    ));
                }
                let rationale = state.pending_rationale.take();
                let child_idx = children.len();
                state.current_path.push(child_idx);
                let result =
                    dispatch_operator(&name, raw_attrs, false, cur, state, sm, pos, rationale);
                state.current_path.pop();
                let child = result?;
                children.push(child);
            }
            Ok(Event::Empty(ref e)) => {
                let name = e.name().as_ref().to_vec();
                let raw_attrs = collect_attrs_raw(e, sm, pos)?;
                if name_has_namespace(&name) {
                    return Err(parse_errs(
                        ParseErrorCode::NamespaceRejected,
                        pos,
                        sm,
                        "namespace in child tag",
                        "",
                    ));
                }
                let rationale = state.pending_rationale.take();
                let child_idx = children.len();
                state.current_path.push(child_idx);
                let result =
                    dispatch_operator(&name, raw_attrs, true, cur, state, sm, pos, rationale);
                state.current_path.pop();
                let child = result?;
                children.push(child);
            }
            Ok(Event::End(ref e)) => {
                // Verify the close tag matches our operator.
                if e.name().as_ref() != op.as_bytes() {
                    return Err(parse_errs(
                        ParseErrorCode::MalformedXml,
                        pos,
                        sm,
                        format!(
                            "expected </{op}>, got </{}>",
                            String::from_utf8_lossy(e.name().as_ref())
                        ),
                        "",
                    ));
                }
                return Ok(children);
            }
            Ok(Event::Comment(c)) => {
                state.pending_rationale =
                    Some(String::from_utf8_lossy(c.as_ref()).trim().to_string());
                continue;
            }
            Ok(Event::Text(_)) => continue,
            Ok(Event::Eof) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("unexpected EOF inside <{op}>"),
                    "",
                ));
            }
            Err(e) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("XML error: {e}"),
                    "",
                ));
            }
            _ => continue,
        }
    }
}

fn consume_until_end_tag(
    cur: &mut EventCursor<'_>,
    sm: &SourceMap,
    op: &str,
) -> Result<(), ParseErrors> {
    loop {
        let pos = cur.byte_offset_before_event();
        match cur.next() {
            Ok(Event::End(ref end)) if end.name().as_ref() == op.as_bytes() => return Ok(()),
            Ok(Event::Eof) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("unexpected EOF before </{op}>"),
                    "",
                ));
            }
            Err(e) => {
                return Err(parse_errs(
                    ParseErrorCode::MalformedXml,
                    pos,
                    sm,
                    format!("XML error: {e}"),
                    "",
                ));
            }
            _ => continue,
        }
    }
}

// ── leaf collection ────────────────────────────────────────────────────────────

pub fn collect_leaves(op: &CompOp) -> Vec<LeafIdent> {
    let mut out = Vec::new();
    collect_leaves_inner(op, &mut out);
    out
}

/// IN-02 fix: collect leaves using the resolved node_id_map so that an
/// explicit `id="past"` attribute on a `<Spawn>` produces `node_id="past"` in
/// the `LeafIdent` rather than the agent name. Falls back to agent name when
/// the AstPath is not present in the map (e.g. for programmatically-built ops).
pub fn collect_leaves_with_ids(
    op: &CompOp,
    node_id_map: &HashMap<AstPath, String>,
) -> Vec<LeafIdent> {
    let mut out = Vec::new();
    collect_leaves_inner_with_ids(
        op,
        node_id_map,
        &mut Vec::new(),
        &mut Vec::new(),
        false,
        &mut out,
    );
    out
}

fn collect_leaves_inner_with_ids(
    op: &CompOp,
    node_id_map: &HashMap<AstPath, String>,
    path: &mut AstPath,
    ancestors: &mut Vec<String>,
    deferred: bool,
    out: &mut Vec<LeafIdent>,
) {
    // Resolve the current node's id from `node_id_map` (every operator stamps
    // its id at parse time; missing entries only occur for ad-hoc programmatic
    // ops, which take the no-id collector path instead).
    let self_id: Option<String> = node_id_map.get(path).cloned();
    match op {
        CompOp::Spawn(spec) => {
            let node_id = self_id.unwrap_or_else(|| spec.agent_id.to_string());
            out.push(LeafIdent {
                node_id,
                agent: spec.agent_id.to_string(),
                deferred,
                ancestor_ids: ancestors.clone(),
            });
        }
        CompOp::Sequential(ops) => {
            let pushed = push_ancestor(ancestors, &self_id);
            // In a Sequential, only the first child's leaves are immediate;
            // subsequent children are deferred (they won't execute until the
            // preceding sibling completes).
            for (i, o) in ops.iter().enumerate() {
                let child_deferred = deferred || i > 0;
                path.push(i);
                collect_leaves_inner_with_ids(
                    o,
                    node_id_map,
                    path,
                    ancestors,
                    child_deferred,
                    out,
                );
                path.pop();
            }
            pop_ancestor(ancestors, pushed);
        }
        CompOp::Parallel(ops) => {
            let pushed = push_ancestor(ancestors, &self_id);
            // All parallel children inherit the parent's deferred state —
            // if the Parallel itself is immediate, so are all its children.
            for (i, o) in ops.iter().enumerate() {
                path.push(i);
                collect_leaves_inner_with_ids(o, node_id_map, path, ancestors, deferred, out);
                path.pop();
            }
            pop_ancestor(ancestors, pushed);
        }
        CompOp::Synthesise(inner, spec) => {
            let pushed = push_ancestor(ancestors, &self_id);
            path.push(0);
            collect_leaves_inner_with_ids(inner, node_id_map, path, ancestors, deferred, out);
            path.pop();
            // Pop the Synthesise's own id BEFORE snapshotting the synthesiser
            // leaf's ancestors. parse_synthesise stamps the same `id` attribute
            // onto the Synthesise operator AND the synthesiser SpawnSpec
            // (ast.rs:780), so the synthesiser leaf shares its node_id with its
            // container. Including the container would create a self-loop in
            // ancestor_ids — the leaf-form IS the container in collapsed view.
            pop_ancestor(ancestors, pushed);
            // E2 / D11-05: synthesiser leaf prefers its stamped SpawnSpec.node_id
            // (set by parse_synthesise). Falls back to the path-entry lookup, then
            // the agent name for programmatically-built ops.
            // The synthesiser always runs AFTER the inner op completes, so it is
            // deferred unless it is at the top level with no sequential context.
            path.push(1);
            let node_id = spec
                .synthesiser
                .node_id
                .clone()
                .or_else(|| node_id_map.get(path).cloned())
                .unwrap_or_else(|| spec.synthesiser.agent_id.to_string());
            path.pop();
            out.push(LeafIdent {
                node_id,
                agent: spec.synthesiser.agent_id.to_string(),
                // Synthesiser is deferred: it runs after all inner branches complete.
                deferred: true,
                ancestor_ids: ancestors.clone(),
            });
        }
        CompOp::Gate(inner, _) | CompOp::Loop(inner, _) | CompOp::ConditionalLoop(inner, _) => {
            let pushed = push_ancestor(ancestors, &self_id);
            path.push(0);
            collect_leaves_inner_with_ids(inner, node_id_map, path, ancestors, deferred, out);
            path.pop();
            pop_ancestor(ancestors, pushed);
        }
        CompOp::Conditional(branches) => {
            let pushed = push_ancestor(ancestors, &self_id);
            // Conditional branches are all deferred — only one actually executes,
            // determined at runtime by predicate evaluation.
            for (i, (_, o)) in branches.iter().enumerate() {
                path.push(i);
                collect_leaves_inner_with_ids(o, node_id_map, path, ancestors, true, out);
                path.pop();
            }
            pop_ancestor(ancestors, pushed);
        }
        CompOp::FanOut(spec, prompts) => {
            // E2 / D11-05: per-prompt ids in the `{fanout_id}_p{i}` format,
            // zero-indexed. The fanout's root id is stamped on `spec.node_id`
            // by `parse_fanout`; fall back to the path-resolved id, then the
            // agent name for ad-hoc compositions where neither is present.
            let fanout_id = spec
                .node_id
                .clone()
                .or_else(|| self_id.clone())
                .unwrap_or_else(|| spec.agent_id.to_string());
            // Each per-prompt leaf's immediate parent is the fanout root.
            ancestors.push(fanout_id.clone());
            for (i, _) in prompts.iter().enumerate() {
                out.push(LeafIdent {
                    node_id: format!("{fanout_id}_p{i}"),
                    agent: spec.agent_id.to_string(),
                    deferred,
                    ancestor_ids: ancestors.clone(),
                });
            }
            ancestors.pop();
        }
    }
}

fn push_ancestor(ancestors: &mut Vec<String>, self_id: &Option<String>) -> bool {
    if let Some(id) = self_id {
        ancestors.push(id.clone());
        true
    } else {
        false
    }
}

fn pop_ancestor(ancestors: &mut Vec<String>, pushed: bool) {
    if pushed {
        ancestors.pop();
    }
}

fn collect_leaves_inner(op: &CompOp, out: &mut Vec<LeafIdent>) {
    collect_leaves_inner_deferred(op, false, out);
}

fn collect_leaves_inner_deferred(op: &CompOp, deferred: bool, out: &mut Vec<LeafIdent>) {
    match op {
        CompOp::Spawn(spec) => {
            out.push(LeafIdent {
                node_id: format!("{}", spec.agent_id),
                agent: format!("{}", spec.agent_id),
                deferred,
                ancestor_ids: Vec::new(),
            });
        }
        CompOp::Sequential(ops) => {
            for (i, o) in ops.iter().enumerate() {
                collect_leaves_inner_deferred(o, deferred || i > 0, out);
            }
        }
        CompOp::Parallel(ops) => {
            for o in ops {
                collect_leaves_inner_deferred(o, deferred, out);
            }
        }
        CompOp::Synthesise(inner, spec) => {
            collect_leaves_inner_deferred(inner, deferred, out);
            out.push(LeafIdent {
                node_id: format!("{}", spec.synthesiser.agent_id),
                agent: format!("{}", spec.synthesiser.agent_id),
                deferred: true,
                ancestor_ids: Vec::new(),
            });
        }
        CompOp::Gate(inner, _) | CompOp::Loop(inner, _) | CompOp::ConditionalLoop(inner, _) => {
            collect_leaves_inner_deferred(inner, deferred, out);
        }
        CompOp::Conditional(branches) => {
            for (_, o) in branches {
                collect_leaves_inner_deferred(o, true, out);
            }
        }
        CompOp::FanOut(spec, prompts) => {
            let fanout_id = spec
                .node_id
                .clone()
                .unwrap_or_else(|| spec.agent_id.to_string());
            for (i, _) in prompts.iter().enumerate() {
                out.push(LeafIdent {
                    node_id: format!("{fanout_id}_p{i}"),
                    agent: spec.agent_id.to_string(),
                    deferred,
                    ancestor_ids: Vec::new(),
                });
            }
        }
    }
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::tokenizer::SourceMap;
    use super::*;

    fn parse_str(xml: &str) -> Result<CompOp, ParseErrors> {
        let sm = SourceMap::new(xml);
        parse(xml, &sm).map(|(op, _, _, _)| op)
    }

    #[test]
    fn round_trip_spawn() {
        let r = parse_str(r#"<Compose><Spawn agent="huginn" task="t"/></Compose>"#).unwrap();
        if let CompOp::Spawn(s) = r {
            assert_eq!(format!("{}", s.agent_id), "huginn");
            assert_eq!(s.task_template, "t");
        } else {
            panic!("expected Spawn");
        }
    }

    #[test]
    fn round_trip_sequential() {
        let r =
            parse_str(r#"<Compose><Sequential><Spawn agent="a" task="t"/></Sequential></Compose>"#)
                .unwrap();
        assert!(matches!(r, CompOp::Sequential(_)));
    }

    #[test]
    fn round_trip_parallel() {
        let r = parse_str(r#"<Compose><Parallel><Spawn agent="a" task="t"/><Spawn agent="b" task="t"/></Parallel></Compose>"#).unwrap();
        assert!(matches!(r, CompOp::Parallel(ref ops) if ops.len() == 2));
    }

    // Single-child <Parallel> is a Sequential-in-disguise and the parser has no warning
    // channel — reject it as WrongChildCount instead of silently accepting. See
    // `docs/composition-grammar.md` §3.3.
    #[test]
    fn single_child_parallel_returns_wrong_child_count() {
        let xml = r#"<Compose><Parallel><Spawn agent="a" task="t"/></Parallel></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::WrongChildCount);
    }

    #[test]
    fn empty_parallel_returns_wrong_child_count() {
        let xml = r#"<Compose><Parallel></Parallel></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::WrongChildCount);
    }

    #[test]
    fn round_trip_synthesise() {
        let r = parse_str(r#"<Compose><Synthesise synthesiser="sjofn"><Spawn agent="a" task="t"/></Synthesise></Compose>"#).unwrap();
        assert!(matches!(r, CompOp::Synthesise(_, _)));
    }

    #[test]
    fn round_trip_gate() {
        let r = parse_str(r#"<Compose><Gate required="status,signal" on_fail="Escalate"><Spawn agent="a" task="t"/></Gate></Compose>"#).unwrap();
        assert!(matches!(r, CompOp::Gate(_, _)));
    }

    #[test]
    fn round_trip_loop() {
        let r = parse_str(r#"<Compose><Loop max_iterations="3" on_exhaust="AcceptLast"><Spawn agent="a" task="t"/></Loop></Compose>"#).unwrap();
        assert!(matches!(r, CompOp::Loop(_, _)));
    }

    #[test]
    fn round_trip_conditional_loop() {
        let r = parse_str(r#"<Compose><ConditionalLoop max_iterations="2" on_exhaust="Escalate" required="status,signal"><Spawn agent="a" task="t"/></ConditionalLoop></Compose>"#).unwrap();
        assert!(matches!(r, CompOp::ConditionalLoop(_, _)));
    }

    #[test]
    fn round_trip_conditional() {
        let xml = r#"<Compose><Conditional><When channel="x" equals="ready"><Spawn agent="a" task="t"/></When><Otherwise><Spawn agent="b" task="t"/></Otherwise></Conditional></Compose>"#;
        let r = parse_str(xml).unwrap();
        assert!(matches!(r, CompOp::Conditional(ref b) if b.len() == 2));
    }

    #[test]
    fn round_trip_fanout() {
        let xml = r#"<Compose><FanOut agent="huginn"><Prompt>P1</Prompt><Prompt>P2</Prompt></FanOut></Compose>"#;
        let r = parse_str(xml).unwrap();
        assert!(matches!(r, CompOp::FanOut(_, ref p) if p.len() == 2));
    }

    #[test]
    fn canonical_section_5_composite_parses() {
        let xml = r#"<Compose>
            <Sequential>
              <Synthesise id="decision" synthesiser="sjofn" task="Synthesise tone">
                <Parallel>
                  <Spawn id="past" agent="urdr" task="Past."/>
                  <Spawn id="future" agent="skuld" task="Future."/>
                  <Spawn id="present" agent="verdandi" task="Present."/>
                </Parallel>
              </Synthesise>
              <Gate required="status,signal" status_must_be="Complete" on_fail="Escalate">
                <Spawn agent="kvasir" task="Red-team at x"/>
              </Gate>
            </Sequential>
          </Compose>"#;
        let r = parse_str(xml).unwrap();
        assert!(matches!(r, CompOp::Sequential(_)));
    }

    #[test]
    fn canonical_section_5_has_five_leaves() {
        let xml = r#"<Compose>
            <Sequential>
              <Synthesise id="decision" synthesiser="sjofn" task="Synthesise tone">
                <Parallel>
                  <Spawn id="past" agent="urdr" task="Past."/>
                  <Spawn id="future" agent="skuld" task="Future."/>
                  <Spawn id="present" agent="verdandi" task="Present."/>
                </Parallel>
              </Synthesise>
              <Gate required="status,signal" status_must_be="Complete" on_fail="Escalate">
                <Spawn agent="kvasir" task="Red-team at x"/>
              </Gate>
            </Sequential>
          </Compose>"#;
        let sm = SourceMap::new(xml);
        let (op, _, leaves, _) = parse(xml, &sm).unwrap();
        // 3 parallel spawns + 1 synthesiser + 1 kvasir spawn = 5 leaves
        assert_eq!(
            leaves.len(),
            5,
            "expected 5 leaves, got {}: {:?}",
            leaves.len(),
            leaves.iter().map(|l| &l.agent).collect::<Vec<_>>()
        );
        let _ = op;
    }

    // ── negative tests (A category) ──────────────────────────────────────────

    #[test]
    fn unknown_tag_synthesze_returns_unknown_tag() {
        let xml = r#"<Compose><Synthesze synthesiser="sjofn"><Spawn agent="a" task="t"/></Synthesze></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::UnknownTag);
    }

    #[test]
    fn unknown_attr_syntesiser_returns_unknown_attribute() {
        let xml = r#"<Compose><Synthesise syntesiser="sjofn"><Spawn agent="a" task="t"/></Synthesise></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::UnknownAttribute);
    }

    #[test]
    fn dtd_returns_dtd_rejected() {
        let xml = r#"<!DOCTYPE foo><Compose><Spawn agent="a" task="t"/></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::DtdRejected);
    }

    #[test]
    fn processing_instruction_returns_pi_rejected() {
        let xml = r#"<?xml-stylesheet href="x"?><Compose><Spawn agent="a" task="t"/></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(
            err.errors[0].code,
            ParseErrorCode::ProcessingInstructionRejected
        );
    }

    #[test]
    fn namespace_returns_namespace_rejected() {
        let xml = r#"<Compose><ns:Spawn agent="a" task="t"/></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::NamespaceRejected);
    }

    #[test]
    fn malformed_xml_returns_malformed() {
        let xml = r#"<Compose><Spawn agent="a" task="t""#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::MalformedXml);
    }

    // ── negative tests (B category) ──────────────────────────────────────────

    #[test]
    fn synthesise_missing_synthesiser_returns_missing_required_attribute() {
        let xml = r#"<Compose><Synthesise><Spawn agent="a" task="t"/></Synthesise></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::MissingRequiredAttribute);
    }

    #[test]
    fn sequential_with_zero_children_returns_wrong_child_count() {
        let xml = r#"<Compose><Sequential></Sequential></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::WrongChildCount);
        assert!(
            err.errors[0].hint.contains("drop the <Sequential> wrapper"),
            "hint was: {}",
            err.errors[0].hint
        );
    }

    #[test]
    fn loop_with_non_numeric_max_iterations_returns_attribute_invalid() {
        let xml = r#"<Compose><Loop max_iterations="abc" on_exhaust="AcceptLast"><Spawn agent="a" task="t"/></Loop></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert_eq!(err.errors[0].code, ParseErrorCode::AttributeValueInvalid);
    }

    // ── rationale_map + auto-generated ids ───────────────────────────────────

    #[test]
    fn comment_preceding_node_is_captured_in_rationale_map() {
        let xml = r#"<Compose><!-- thoughts --><Spawn id="x" agent="a" task="t"/></Compose>"#;
        let sm = SourceMap::new(xml);
        let (_, rationale, _, _) = parse(xml, &sm).unwrap();
        assert_eq!(rationale.get("x").map(|s| s.as_str()), Some("thoughts"));
    }

    #[test]
    fn auto_generated_node_ids_use_agent_index_form_for_spawn() {
        // §4.1: auto IDs use `{agent}_{index}` for Spawn.
        let xml = r#"<Compose><Sequential><Spawn agent="huginn" task="t"/><Spawn agent="smidr" task="t"/></Sequential></Compose>"#;
        let sm = SourceMap::new(xml);
        let (op, _, _, node_id_map) = parse(xml, &sm).unwrap();
        // The two spawns are at AstPath [0, 0] and [0, 1] (inside Sequential at [0])
        // node_id_map should contain "huginn_1" and "smidr_2" or "huginn_0" and "smidr_1"
        // Per §4.1, index is monotonic per <Compose>:
        // Sequential gets index 0 → id="sequential_0"
        // First Spawn gets index 1 → id="huginn_1"
        // Second Spawn gets index 2 → id="smidr_2"
        let ids: std::collections::HashSet<String> = node_id_map.values().cloned().collect();
        assert!(
            ids.iter().any(|id| id.starts_with("huginn_")),
            "expected huginn_N in {:?}",
            ids
        );
        assert!(
            ids.iter().any(|id| id.starts_with("smidr_")),
            "expected smidr_N in {:?}",
            ids
        );
        let _ = op;
    }

    #[test]
    fn explicit_id_honored_over_auto_generated() {
        let xml = r#"<Compose><Spawn id="past" agent="urdr" task="t"/></Compose>"#;
        let sm = SourceMap::new(xml);
        let (_, _, _, node_id_map) = parse(xml, &sm).unwrap();
        // The spawn at AstPath [] should have id="past" not "urdr_0"
        let id_at_root = node_id_map.get(&vec![]).cloned();
        assert_eq!(
            id_at_root.as_deref(),
            Some("past"),
            "expected explicit id 'past', got {:?}",
            id_at_root
        );
    }

    #[test]
    fn duplicate_node_id_is_caught_by_ast() {
        let xml = r#"<Compose><Sequential><Spawn id="a" agent="x" task="t"/><Spawn id="a" agent="y" task="t"/></Sequential></Compose>"#;
        let err = parse_str(xml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.code == ParseErrorCode::DuplicateNodeId),
            "expected DuplicateNodeId, got: {:?}",
            err.errors
        );
    }

    // ── E2 / D11-05 node_id stamping ──────────────────────────────────────────

    /// Parse_synthesise stamps the synthesiser's `SpawnSpec.node_id` to
    /// `Some(<resolved-id>)` (not `None`). Closes the wedge for <Synthesise>
    /// compositions where the daemon's DaemonLeafHook lookup needs a stable id.
    #[test]
    fn parse_synthesise_stamps_node_id_on_synthesiser_spec() {
        let xml = r#"<Compose>
            <Synthesise id="sjofn_synth" synthesiser="sjofn">
                <Parallel>
                    <Spawn id="left" agent="skuld" task="past"/>
                    <Spawn id="right" agent="urdr" task="present"/>
                </Parallel>
            </Synthesise>
        </Compose>"#;
        let op = parse_str(xml).unwrap();
        match op {
            CompOp::Synthesise(_, spec) => {
                assert_eq!(
                    spec.synthesiser.node_id,
                    Some("sjofn_synth".to_string()),
                    "synthesiser SpawnSpec must carry its resolved node_id"
                );
            }
            other => panic!("expected CompOp::Synthesise, got {other:?}"),
        }
    }

    /// parse_fanout stamps the fanout's root id on the SpawnSpec so the
    /// LeafIdent collector can mint per-prompt `{fanout_id}_p{i}` ids.
    #[test]
    fn parse_fanout_stamps_root_id_on_spec() {
        let xml = r#"<Compose>
            <FanOut id="discovery" agent="huginn">
                <Prompt>question A</Prompt>
                <Prompt>question B</Prompt>
            </FanOut>
        </Compose>"#;
        let op = parse_str(xml).unwrap();
        match op {
            CompOp::FanOut(spec, prompts) => {
                assert_eq!(spec.node_id, Some("discovery".to_string()));
                assert_eq!(prompts.len(), 2);
            }
            other => panic!("expected CompOp::FanOut, got {other:?}"),
        }
    }

    /// `collect_leaves_with_ids` produces one LeafIdent per fanout prompt
    /// with format `{fanout_id}_p{i}` zero-indexed.
    #[test]
    fn fanout_three_prompts_emits_three_leaf_idents_with_p_index_format() {
        let xml = r#"<Compose>
            <FanOut id="discovery" agent="huginn">
                <Prompt>question A</Prompt>
                <Prompt>question B</Prompt>
                <Prompt>question C</Prompt>
            </FanOut>
        </Compose>"#;
        let sm = SourceMap::new(xml);
        let (op, _, _, node_id_map) = parse(xml, &sm).unwrap();
        let leaves = collect_leaves_with_ids(&op, &node_id_map);
        assert_eq!(leaves.len(), 3, "fanout with 3 prompts emits 3 leaves");
        assert_eq!(leaves[0].node_id, "discovery_p0");
        assert_eq!(leaves[1].node_id, "discovery_p1");
        assert_eq!(leaves[2].node_id, "discovery_p2");
        // All carry the same agent.
        for l in &leaves {
            assert_eq!(l.agent, "huginn");
        }
    }

    /// Synthesise's LeafIdent picks up the stamped node_id (not the agent name).
    #[test]
    fn synthesise_leaf_ident_uses_stamped_node_id() {
        let xml = r#"<Compose>
            <Synthesise id="sjofn_synth" synthesiser="sjofn">
                <Parallel>
                    <Spawn id="left" agent="skuld" task="past"/>
                    <Spawn id="right" agent="urdr" task="present"/>
                </Parallel>
            </Synthesise>
        </Compose>"#;
        let sm = SourceMap::new(xml);
        let (op, _, _, node_id_map) = parse(xml, &sm).unwrap();
        let leaves = collect_leaves_with_ids(&op, &node_id_map);
        // Expect 3 leaves: skuld (left), urdr (right), synthesiser (sjofn).
        assert_eq!(leaves.len(), 3);
        // The synthesiser is last.
        let synth = leaves.last().unwrap();
        assert_eq!(synth.agent, "sjofn");
        assert_eq!(
            synth.node_id, "sjofn_synth",
            "synthesise leaf carries the stamped id, not the agent name"
        );
    }

    /// Regression net: a pure-Spawn plan (no Synthesise/FanOut) is unchanged
    /// — leaves get parser-resolved ids (`{agent}_{idx}` or explicit `id`).
    #[test]
    fn pure_spawn_plan_node_ids_unchanged_by_e2_fix() {
        let xml = r#"<Compose>
            <Sequential>
                <Spawn id="past" agent="skuld" task="t1"/>
                <Spawn id="present" agent="urdr" task="t2"/>
            </Sequential>
        </Compose>"#;
        let sm = SourceMap::new(xml);
        let (op, _, _, node_id_map) = parse(xml, &sm).unwrap();
        let leaves = collect_leaves_with_ids(&op, &node_id_map);
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].node_id, "past");
        assert_eq!(leaves[1].node_id, "present");
    }

    /// `ancestor_ids` carries the structural parent chain from root (outermost)
    /// to immediate parent (innermost), exclusive of the leaf itself. Verifies
    /// the canonical Sequential → Synthesise → Parallel nesting. The
    /// synthesiser leaf shares its node_id with its `<Synthesise>` container
    /// (ast.rs:780), so its ancestor chain stops at the container's parent —
    /// the container is the leaf in collapsed form.
    #[test]
    fn ancestor_ids_nested_sequential_synthesise_parallel() {
        let xml = r#"<Compose>
            <Sequential id="seq_root">
                <Spawn id="writer_1" agent="skuld" task="draft"/>
                <Synthesise id="synth_1" synthesiser="judge">
                    <Parallel id="par_1">
                        <Spawn id="critic_a" agent="huginn" task="ca"/>
                        <Spawn id="critic_b" agent="muninn" task="cb"/>
                    </Parallel>
                </Synthesise>
            </Sequential>
        </Compose>"#;
        let sm = SourceMap::new(xml);
        let (op, _, _, node_id_map) = parse(xml, &sm).unwrap();
        let leaves = collect_leaves_with_ids(&op, &node_id_map);

        let by_id = |needle: &str| {
            leaves
                .iter()
                .find(|l| l.node_id == needle)
                .unwrap_or_else(|| panic!("no leaf with node_id={needle}; got {leaves:#?}"))
        };

        assert_eq!(by_id("writer_1").ancestor_ids, vec!["seq_root".to_string()]);
        assert_eq!(
            by_id("critic_a").ancestor_ids,
            vec![
                "seq_root".to_string(),
                "synth_1".to_string(),
                "par_1".to_string()
            ]
        );
        assert_eq!(
            by_id("critic_b").ancestor_ids,
            vec![
                "seq_root".to_string(),
                "synth_1".to_string(),
                "par_1".to_string()
            ]
        );
        // Synthesiser leaf: node_id == container id ("synth_1"); ancestors stop
        // at "seq_root" so the leaf does not list itself. (synth_1 is the
        // container/leaf collapsed name.)
        let synth_leaf = leaves
            .iter()
            .find(|l| l.agent == "judge")
            .expect("synthesiser leaf with agent=judge");
        assert_eq!(synth_leaf.node_id, "synth_1");
        assert_eq!(
            synth_leaf.ancestor_ids,
            vec!["seq_root".to_string()],
            "synthesiser leaf must NOT list its own container id as ancestor"
        );
    }

    /// FanOut: each per-prompt leaf's ancestor chain includes the fanout root
    /// id as the immediate parent (the fanout root is the structural parent
    /// of every `_p{i}` leaf), plus any structural ancestors above the FanOut.
    #[test]
    fn ancestor_ids_fanout_per_prompt_includes_fanout_root() {
        let xml = r#"<Compose>
            <Sequential id="seq_root">
                <FanOut id="discovery" agent="huginn">
                    <Prompt>q A</Prompt>
                    <Prompt>q B</Prompt>
                </FanOut>
            </Sequential>
        </Compose>"#;
        let sm = SourceMap::new(xml);
        let (op, _, _, node_id_map) = parse(xml, &sm).unwrap();
        let leaves = collect_leaves_with_ids(&op, &node_id_map);
        assert_eq!(leaves.len(), 2);
        for leaf in &leaves {
            assert_eq!(
                leaf.ancestor_ids,
                vec!["seq_root".to_string(), "discovery".to_string()],
                "fanout per-prompt leaf {} should have [seq_root, discovery] ancestors",
                leaf.node_id
            );
        }
    }

    /// Top-level leaf with no structural parents has an empty ancestor chain.
    #[test]
    fn ancestor_ids_top_level_spawn_is_empty() {
        let xml = r#"<Compose>
            <Spawn id="solo" agent="skuld" task="t"/>
        </Compose>"#;
        let sm = SourceMap::new(xml);
        let (op, _, _, node_id_map) = parse(xml, &sm).unwrap();
        let leaves = collect_leaves_with_ids(&op, &node_id_map);
        assert_eq!(leaves.len(), 1);
        assert!(
            leaves[0].ancestor_ids.is_empty(),
            "top-level Spawn has no structural parents, got {:?}",
            leaves[0].ancestor_ids
        );
    }
}
