//! Stage-1 graph extraction task implementations.
//!
//! Implements the stage-task traits over `ArgumentationGraph` for the TTD
//! Stage-1 graph extraction pipeline. Each struct corresponds to one phase
//! in the consensus `graph_tasks.py` extraction flow.
//!
//! ## Map-Reduce structure (GraphDraftGen::generate)
//!
//! Mirrors `GraphDraftGen._generate_single_draft` (graph_tasks.py:241-345):
//!
//! 1. **Map:** `extraction_single` per expert, capped at 10 concurrent
//!    (`_EXTRACT_CONCURRENCY=10`; graph_tasks.py:271). Node IDs namespaced
//!    `{expert_id}_{node_id}`. Failed experts logged via `tracing::debug!`;
//!    only fails the draft if ALL experts fail (sibling-survival semantics).
//! 2. **Merge:** concatenate per-expert graphs, dedup overlapping node IDs
//!    (graph_tasks.py:562-595).
//! 3. **Resolve:** `resolution` spawn finds cross-expert edges + merges
//!    duplicate claims. Only runs when `len(responses) > 1`. Partial-XML
//!    recovery for truncated output.
//! 4. **Verify:** deterministic `verify_graph_quotes` (graph_tasks.py:41-170) —
//!    updates `verification_status` on each node; feeds groundedness fitness.
//!
//! ## Trust boundary (T-23-04)
//!
//! Retrieved text and `ExpertResponse.prose` stay in the data section of the
//! rendered prompt, never the instruction position — mirrors Phase 22 adapter.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::adapter::ExpertResponse;
use crate::executor::AgentExecutor;
use crate::ttd::artifact::{ArgumentationGraph, GraphEdge, GraphNode, NodeAnnotation};
use crate::ttd::config::TtdConfig;
use crate::ttd::fitness::{is_valid_graph, is_valid_v2, traceability_veto_graph, FitnessEval};
use crate::ttd::mod_types::TtdError;
use crate::ttd::stages::{
    DraftGen, EvalFitness, GapIdentify, GapResolve, Merger, RetrievedContext,
};
use crate::ttd::state::IdentifiedGap;
use crate::ttd::term_sheet::PromptProfile;
use crate::ttd::weights::{GRAPH_WEIGHTS, V2_GRAPH_WEIGHTS};

/// Concurrency cap for per-expert extraction spawns.
/// Mirrors `_EXTRACT_CONCURRENCY = 10` in graph_tasks.py:271.
const EXTRACT_CONCURRENCY: usize = 10;

// ── GraphDraftGen ─────────────────────────────────────────────────────────────

/// Stage-1 draft generation: map-reduce over expert responses.
///
/// Each call to `generate()` produces ONE `ArgumentationGraph` by:
/// 1. Extracting a per-expert sub-graph from each `ExpertResponse` (capped
///    at 10 concurrent spawns via `Semaphore`).
/// 2. Merging all per-expert sub-graphs into one combined graph.
/// 3. Resolving cross-expert relationships (only when > 1 response).
/// 4. Verifying quote groundedness deterministically.
///
/// This is called N=5 times by `TtdMachine::run()` to produce the initial
/// FanOut(5) candidate set.
pub struct GraphDraftGen {
    /// Agent ID for the extraction_single spawn.
    pub agent_id: String,
    /// Model to use for extraction spawns.
    pub model: String,
    /// Prompt version string for provenance stamping.
    pub prompt_version: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
}

impl GraphDraftGen {
    pub fn new(
        agent_id: impl Into<String>,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            model: model.into(),
            prompt_version: prompt_version.into(),
            profile: PromptProfile::V1Delphi,
        }
    }

    /// Set the prompt/schema profile (consuming builder).
    pub fn with_profile(mut self, profile: PromptProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Extract a per-expert sub-graph for one `ExpertResponse`.
    ///
    /// Namespaces node IDs as `{expert_id}_{node_id}` so they are globally
    /// unique across the merged graph (graph_tasks.py:271 convention).
    async fn extract_single(
        &self,
        response: &ExpertResponse,
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
        persona_prompt: Option<&str>,
        sampling: Option<crate::executor::SamplingParams>,
    ) -> Result<ArgumentationGraph, TtdError> {
        use crate::ttd::prompts::graph::{render_extraction_single, ExtractionSingleInput};
        use alzina_core::identity::AgentId;

        // B2: fork on profile — V2LitReview uses paper-provenance framing (D-4).
        // Decision 0: v3 = v2 at Stage 1 (long-form changes Stage-3 shape only).
        let prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                use crate::ttd::prompts::lit_review::ExtractionSingleV2Input;
                crate::ttd::prompts::lit_review::render_extraction_single_v2(&ExtractionSingleV2Input {
                    paper_id: response.expert_id.as_str().to_string(),
                    title: response.provenance.title.clone(),
                    year: response.provenance.year,
                    authors: response.provenance.authors.clone(),
                    prose: response.prose.clone(),
                    credibility_tier: response.provenance.credibility_tier,
                })
            }
            PromptProfile::V1Delphi => {
                let prompt_input = ExtractionSingleInput {
                    question: format!("Extract argumentation graph from expert {}", response.expert_id.as_str()),
                    expert_id: response.expert_id.as_str().to_string(),
                    response_text: response.prose.clone(),
                };
                render_extraction_single(&prompt_input)
            }
        };

        // EXT-01 Phase 24 (WR-01): prefix the trajectory persona so the graph
        // FanOut diversifies, mirroring Stages 2/3. None → Phase-23 template
        // behaviour preserved.
        let effective_prompt = if let Some(persona) = persona_prompt {
            format!("{}\n\n---\n\n{}", persona, prompt)
        } else {
            prompt
        };

        let agent_id = AgentId::new(self.agent_id.clone());

        // EXT-01 Phase 24 (WR-01): route through execute_with_sampling to thread
        // per-trajectory sampling params (default impl falls through to execute()
        // when sampling=None — backward compatible).
        let raw = executor
            .execute_with_sampling(&agent_id, &effective_prompt, &self.model, "extract_single", sampling)
            .await
            .map_err(TtdError::Executor)?;

        // Parse the XML graph response into ArgumentationGraph nodes/edges.
        // Namespace all node IDs with the expert_id prefix.
        let graph = parse_extraction_xml(&raw, response.expert_id.as_str(), &self)?;

        Ok(graph)
    }

    /// Merge N per-expert graphs into one combined graph.
    ///
    /// Concatenates nodes and edges, deduplicating by node ID
    /// (graph_tasks.py:562-595 `_merge_graphs`).
    fn merge_graphs(&self, graphs: Vec<ArgumentationGraph>) -> ArgumentationGraph {
        if graphs.is_empty() {
            return ArgumentationGraph::new("", "", "", &self.model, &self.prompt_version);
        }

        let template = &graphs[0];
        let mut seen_node_ids: HashSet<String> = HashSet::new();
        let mut seen_edge_ids: HashSet<String> = HashSet::new();
        let mut all_nodes: Vec<GraphNode> = Vec::new();
        let mut all_edges: Vec<GraphEdge> = Vec::new();
        let mut all_annotations: Vec<NodeAnnotation> = Vec::new();

        for g in &graphs {
            for node in &g.nodes {
                if seen_node_ids.insert(node.id.clone()) {
                    all_nodes.push(node.clone());
                }
            }
            for edge in &g.edges {
                if seen_edge_ids.insert(edge.source.clone() + "→" + &edge.target) {
                    all_edges.push(edge.clone());
                }
            }
            all_annotations.extend(g.node_annotations.iter().cloned());
        }

        ArgumentationGraph {
            schema_version: template.schema_version.clone(),
            study_id: template.study_id.clone(),
            round_id: template.round_id.clone(),
            question_id: template.question_id.clone(),
            generated_at: chrono::Utc::now(),
            model: template.model.clone(),
            prompt_version: template.prompt_version.clone(),
            code_version: template.code_version.clone(),
            nodes: all_nodes,
            edges: all_edges,
            node_annotations: all_annotations,
        }
    }

    /// Resolve cross-expert relationships via the resolution spawn.
    ///
    /// Only runs when `len(responses) > 1` (graph_tasks.py:327). Uses
    /// partial-XML recovery when output is truncated.
    async fn resolve_relationships(
        &self,
        graph: ArgumentationGraph,
        executor: &Arc<dyn AgentExecutor>,
    ) -> Result<ArgumentationGraph, TtdError> {
        use crate::ttd::prompts::graph::{render_resolution, ResolutionInput};
        use alzina_core::identity::AgentId;

        let claims_data: Vec<(String, String, Vec<String>)> = graph
            .nodes
            .iter()
            .map(|n| {
                (
                    n.id.clone(),
                    n.claim.clone(),
                    vec![n.expert_id.clone()],
                )
            })
            .collect();

        // B2: fork on profile — V2LitReview uses lit-review resolution framing.
        // Decision 0: v3 = v2.
        let prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                let graph_xml = graph.to_xml_string();
                crate::ttd::prompts::lit_review::render_resolution_v2(&claims_data, &graph_xml)
            }
            PromptProfile::V1Delphi => render_resolution(&ResolutionInput {
                question: "Resolve cross-expert relationships".to_string(),
                claims: claims_data,
            }),
        };

        let agent_id = AgentId::new(self.agent_id.clone());
        let raw = executor
            .execute(&agent_id, &prompt, &self.model, "resolution")
            .await
            .map_err(TtdError::Executor)?;

        // Apply resolution edges with partial-XML recovery.
        apply_resolution_xml(graph, &raw)
    }

    /// Deterministic quote verification (graph_tasks.py:41-170).
    ///
    /// Checks each node's quote against the source expert response text.
    /// Updates `verification_status` to "verified", "paraphrased", or "absent".
    /// Feeds groundedness fitness — poor verification lowers the score.
    fn verify_graph_quotes(
        &self,
        graph: ArgumentationGraph,
        responses: &[ExpertResponse],
    ) -> ArgumentationGraph {
        // Build expert_id → prose map.
        let response_map: std::collections::HashMap<String, String> = responses
            .iter()
            .map(|r| (r.expert_id.as_str().to_string(), r.prose.clone()))
            .collect();

        let verified_nodes: Vec<GraphNode> = graph
            .nodes
            .iter()
            .map(|node| {
                let source_text = response_map
                    .get(&node.expert_id)
                    .map(String::as_str)
                    .unwrap_or("");

                let status = if let Some(ref quote) = node.quote {
                    verify_quote_status(quote, source_text)
                } else {
                    "unverified".to_string()
                };

                GraphNode {
                    id: node.id.clone(),
                    claim: node.claim.clone(),
                    expert_id: node.expert_id.clone(),
                    quote: node.quote.clone(),
                    verification_status: Some(status),
                }
            })
            .collect();

        ArgumentationGraph {
            nodes: verified_nodes,
            ..graph
        }
    }
}

/// Simple substring-based quote verification.
///
/// Returns "verified" if the quote is found verbatim in the source text,
/// "paraphrased" if a long common subsequence is found (≥70% overlap),
/// or "absent" otherwise.
pub(crate) fn verify_quote_status(quote: &str, source_text: &str) -> String {
    if source_text.is_empty() || quote.is_empty() {
        return "unverified".to_string();
    }
    // Verbatim match (case-sensitive).
    if source_text.contains(quote.trim()) {
        return "verified".to_string();
    }
    // Case-insensitive substring check as paraphrase proxy.
    let quote_lower = quote.to_lowercase();
    let source_lower = source_text.to_lowercase();
    if source_lower.contains(&quote_lower) {
        return "paraphrased".to_string();
    }
    "absent".to_string()
}

/// Parse an `extraction_single` XML response into an `ArgumentationGraph`.
///
/// Node IDs are namespaced as `{expert_id}_{node_id}` per graph_tasks.py convention.
/// On XML parse failure, returns an empty graph (not an error — sibling survival).
fn parse_extraction_xml(
    raw: &str,
    expert_id: &str,
    draft_gen: &GraphDraftGen,
) -> Result<ArgumentationGraph, TtdError> {
    use std::io::BufReader;
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut graph = ArgumentationGraph::new(
        "",
        "",
        "",
        &draft_gen.model,
        &draft_gen.prompt_version,
    );

    // Extract <graph>...</graph> block from the response.
    let xml_content = extract_xml_block(raw, "graph").unwrap_or_else(|| raw.to_string());

    let mut reader = Reader::from_str(&xml_content);
    reader.trim_text(true);

    let mut current_node_id: Option<String> = None;
    let mut current_claim: Option<String> = None;
    let mut current_quote: Option<String> = None;
    let mut inside_text = false;
    let mut inside_quote = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"node" => {
                let id_attr = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"id")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
                // Namespace the node ID with expert_id prefix.
                // F14 (probe-22): the extraction model often already paper-scopes
                // the id (e.g. `arxiv:2501.13956_C13`); prepending expert_id again
                // yields a double-prefixed `arxiv:2501.13956_arxiv:2501.13956_C13`
                // that the synthesis draft cannot copy reliably into <node_refs>.
                // Only namespace when the id is not already expert-prefixed.
                current_node_id = id_attr.map(|id| {
                    let id = id.trim();
                    if id == expert_id || id.starts_with(&format!("{expert_id}_")) {
                        id.to_string()
                    } else {
                        format!("{expert_id}_{id}")
                    }
                });
                current_claim = None;
                current_quote = None;
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"text" => {
                inside_text = true;
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"quote" => {
                inside_quote = true;
            }
            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().to_string();
                if inside_text {
                    current_claim = Some(text);
                } else if inside_quote {
                    current_quote = Some(text);
                }
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"text" => {
                inside_text = false;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"quote" => {
                inside_quote = false;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"node" => {
                if let Some(id) = current_node_id.take() {
                    graph.nodes.push(GraphNode {
                        id: id.clone(),
                        claim: current_claim.take().unwrap_or_default(),
                        expert_id: expert_id.to_string(),
                        quote: current_quote.take(),
                        verification_status: None,
                    });
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => {
                // XML parse failure — return whatever was parsed so far.
                tracing::debug!(
                    expert_id,
                    "extraction_single XML parse error; returning partial graph"
                );
                break;
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(graph)
}

/// Apply resolution XML to a graph, adding cross-expert edges.
///
/// Uses partial-XML recovery — if the output is truncated, applies whatever
/// edges/merges were successfully parsed before the truncation point.
fn apply_resolution_xml(
    mut graph: ArgumentationGraph,
    raw: &str,
) -> Result<ArgumentationGraph, TtdError> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let xml_content = extract_xml_block(raw, "resolution").unwrap_or_else(|| raw.to_string());
    let mut reader = Reader::from_str(&xml_content);
    reader.trim_text(true);

    let mut buf = Vec::new();
    let mut inside_edge = false;
    let mut edge_from: Option<String> = None;
    let mut edge_to: Option<String> = None;
    let mut edge_type: Option<String> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"edge" => {
                inside_edge = true;
                edge_from = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"from")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
                edge_to = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"to")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
                edge_type = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"type")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"edge" && inside_edge => {
                inside_edge = false;
                if let (Some(from), Some(to), Some(rel)) =
                    (edge_from.take(), edge_to.take(), edge_type.take())
                {
                    graph.edges.push(GraphEdge {
                        source: from,
                        target: to,
                        relation: rel,
                    });
                }
            }
            // Self-closing `<edge from=".." to=".." type=".."/>` arrives as
            // Event::Empty — neither Start nor End fires, so without this arm
            // every such edge is silently dropped.
            Ok(Event::Empty(ref e)) if e.name().as_ref() == b"edge" => {
                let from = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"from")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
                let to = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"to")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
                let rel = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"type")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
                if let (Some(from), Some(to), Some(rel)) = (from, to, rel) {
                    graph.edges.push(GraphEdge {
                        source: from,
                        target: to,
                        relation: rel,
                    });
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => {
                tracing::debug!("resolution XML parse error or truncation; applying partial results");
                break;
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(graph)
}

/// Extract the whole `<{tag}...>...</{tag}>` block from raw LLM output.
///
/// T1 ruled contract: returns the WHOLE tagged block (open + body + close)
/// using prefix-open (`<{tag}` — matches `<tag>` and attributed `<tag attr=..>`)
/// / first-close matching. Returns `None` if the open or its first close is
/// absent.
fn extract_xml_block(raw: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");

    let start = raw.find(&open)?;
    let end = raw[start..].find(&close).map(|i| start + i + close.len())?;
    Some(raw[start..end].to_string())
}

#[async_trait]
impl DraftGen<ArgumentationGraph> for GraphDraftGen {
    /// Generate one complete argumentation graph via map-reduce.
    ///
    /// Concurrency cap: `EXTRACT_CONCURRENCY=10` concurrent expert extractions
    /// (matches `_EXTRACT_CONCURRENCY=10` in graph_tasks.py:271).
    async fn generate(
        &self,
        inputs: &[ExpertResponse],
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
        persona_prompt: Option<&str>,
        sampling: Option<crate::executor::SamplingParams>,
    ) -> Result<ArgumentationGraph, TtdError> {
        if inputs.is_empty() {
            // No experts → return an empty graph.
            return Ok(ArgumentationGraph::new("", "", "", &self.model, &self.prompt_version));
        }

        // EXT-01 (Phase 24, WR-01): the per-trajectory persona + sampling seed the
        // graph FanOut just as they do Stages 2/3. Captured here as owned values so
        // each per-expert extraction spawn in this trajectory carries the same lens.
        // Without this the N graph trajectories were structurally identical (the
        // Phase-23 behaviour EXT-01 set out to break).
        let persona_owned: Option<String> = persona_prompt.map(String::from);

        // ── Phase 1: Map — per-expert extraction with concurrency cap ─────────
        let semaphore = Arc::new(Semaphore::new(EXTRACT_CONCURRENCY));
        let mut join_set: JoinSet<(String, Result<ArgumentationGraph, TtdError>)> = JoinSet::new();

        for response in inputs {
            let sem = semaphore.clone();
            let executor_clone = executor.clone();
            let response_clone = response.clone();
            let agent_id = self.agent_id.clone();
            let model = self.model.clone();
            let prompt_version = self.prompt_version.clone();
            let profile = self.profile; // PromptProfile is Copy
            let persona = persona_owned.clone();
            let sampling = sampling; // SamplingParams is Copy

            join_set.spawn(async move {
                let expert_id = response_clone.expert_id.as_str().to_string();
                // WR-04: a closed semaphore must abort this extraction (return Err),
                // not run uncapped. The permit is held for the spawn's lifetime.
                let _permit = match sem.acquire().await {
                    Ok(permit) => permit,
                    Err(_) => {
                        return (
                            expert_id,
                            Err(TtdError::Executor(alzina_core::error::AlzinaError::Orchestration(
                                "semaphore closed".to_string(),
                            ))),
                        );
                    }
                };

                let draft_gen_stub = GraphDraftGen { agent_id, model, prompt_version, profile };
                let result = draft_gen_stub
                    .extract_single(
                        &response_clone,
                        &executor_clone,
                        &TtdConfig::default(),
                        persona.as_deref(),
                        sampling,
                    )
                    .await;
                (expert_id, result)
            });
        }

        // Drain all tasks; sibling-survival semantics (graph_tasks.py:290 return_exceptions=True).
        let mut expert_graphs: Vec<ArgumentationGraph> = Vec::new();
        let mut failed_experts: Vec<String> = Vec::new();

        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok((_expert_id, Ok(graph))) => {
                    expert_graphs.push(graph);
                }
                Ok((expert_id, Err(e))) => {
                    tracing::debug!(
                        expert_id = expert_id.as_str(),
                        error = %e,
                        "expert extraction failed; continuing with other experts"
                    );
                    failed_experts.push(expert_id);
                }
                Err(join_err) => {
                    tracing::debug!(
                        error = %join_err,
                        "expert extraction task panicked; continuing"
                    );
                }
            }
        }

        // Only fail if ALL experts failed.
        if expert_graphs.is_empty() {
            return Err(TtdError::NoCandidates);
        }

        if !failed_experts.is_empty() {
            tracing::debug!(
                n_failed = failed_experts.len(),
                n_total = inputs.len(),
                "some expert extractions failed; merging successful results"
            );
        }

        // ── Phase 2: Merge ────────────────────────────────────────────────────
        let merged = self.merge_graphs(expert_graphs);

        // ── Phase 3: Resolve (only when > 1 expert) ──────────────────────────
        let resolved = if inputs.len() > 1 {
            // WR-06: preserve the populated pre-resolution graph on failure.
            // `resolve_relationships` takes `merged` by value, so clone it first;
            // on a resolution spawn error we fall back to the unresolved-but-
            // POPULATED graph (matching consensus's partial-recovery fallback)
            // instead of discarding every extracted node. The previous fallback
            // merged a vec of EMPTY graphs, returning an empty graph and losing
            // all Stage-1 extraction.
            let merged_unresolved = merged.clone();
            match self.resolve_relationships(merged, executor).await {
                Ok(g) => g,
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        n_nodes = merged_unresolved.nodes.len(),
                        "resolution failed; falling back to unresolved merged graph"
                    );
                    merged_unresolved
                }
            }
        } else {
            merged
        };

        // ── Phase 4: Verify quote groundedness ────────────────────────────────
        let verified = self.verify_graph_quotes(resolved, inputs);

        Ok(verified)
    }
}

// ── GraphGapIdentify ──────────────────────────────────────────────────────────

/// Stage-1 gap identification: find 3-5 gaps in a candidate graph.
///
/// Spawns the `gap_identify` prompt with the current graph and the
/// fitness feedback document (when `use_fitness_feedback=true`).
pub struct GraphGapIdentify {
    pub agent_id: String,
    pub model: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
}

#[async_trait]
impl GapIdentify<ArgumentationGraph> for GraphGapIdentify {
    async fn identify(
        &self,
        draft: &ArgumentationGraph,
        fitness: &FitnessEval,
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<Vec<IdentifiedGap>, TtdError> {
        use crate::ttd::fitness::generate_feedback;
        use crate::ttd::prompts::graph::{render_gap_identify, GapIdentifyInput};
        use alzina_core::identity::AgentId;

        // Generate the fitness feedback document if enabled.
        let fitness_feedback = if config.use_fitness_feedback && !fitness.all_none() {
            Some(generate_feedback(fitness, config.fitness_threshold))
        } else {
            None
        };

        // B2: fork on profile — V2LitReview uses lit-coverage gap framing.
        // Decision 0: v3 = v2. NOTE: equality check (not a match) — this site is
        // invisible to exhaustiveness checking; matches! keeps v3 on the v2 path.
        let prompt = if matches!(
            self.profile,
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong
        ) {
            let graph_xml = draft.to_xml_string();
            crate::ttd::prompts::lit_review::render_gap_identify_v2(
                &graph_xml,
                "Identify gaps in the argumentation graph coverage",
            )
        } else {
            render_gap_identify(&GapIdentifyInput {
                draft_nodes: draft
                    .nodes
                    .iter()
                    .map(|n| (n.id.clone(), n.claim.clone()))
                    .collect(),
                draft_edges: draft
                    .edges
                    .iter()
                    .map(|e| (e.source.clone(), e.target.clone(), e.relation.clone()))
                    .collect(),
                fitness_feedback: fitness_feedback.clone(),
            })
        };

        let agent_id = AgentId::new(self.agent_id.clone());
        let raw = executor
            .execute(&agent_id, &prompt, &self.model, "gap_identify")
            .await
            .map_err(TtdError::Executor)?;

        let gaps = parse_gaps_xml(&raw)?;

        if gaps.is_empty() {
            tracing::debug!("gap_identify returned no gaps; using empty gap list");
        }

        Ok(gaps)
    }
}

/// Parse a `<gaps>` XML response into `IdentifiedGap` values.
///
/// T1 ruled contract: missing or empty `<gaps>` block → `Ok(vec![])` (never
/// `Err`, never a bare `Vec`). A gap is valid iff it has a non-empty
/// `<description>`; `<query>` defaults to the description when absent.
fn parse_gaps_xml(raw: &str) -> Result<Vec<IdentifiedGap>, TtdError> {
    let xml_block = match extract_xml_block(raw, "gaps") {
        Some(block) => block,
        None => return Ok(Vec::new()),
    };

    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(&xml_block);
    reader.trim_text(true);

    let mut gaps = Vec::new();
    let mut buf = Vec::new();
    let mut in_description = false;
    let mut in_query = false;
    let mut description = String::new();
    let mut query = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "gap" => {
                        description.clear();
                        query.clear();
                    }
                    "description" => { in_description = true; }
                    "query" => { in_query = true; }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                let text = e.unescape().unwrap_or_default().to_string();
                if in_description {
                    description.push_str(&text);
                } else if in_query {
                    query.push_str(&text);
                }
            }
            Ok(Event::End(e)) => {
                let tag = std::str::from_utf8(e.name().as_ref()).unwrap_or("").to_string();
                match tag.as_str() {
                    "description" => { in_description = false; }
                    "query" => { in_query = false; }
                    "gap" => {
                        if !description.is_empty() {
                            gaps.push(IdentifiedGap {
                                description: description.trim().to_string(),
                                query: if query.is_empty() {
                                    description.trim().to_string()
                                } else {
                                    query.trim().to_string()
                                },
                            });
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    Ok(gaps)
}

// ── GraphGapResolve ───────────────────────────────────────────────────────────

/// Stage-1 gap resolution: three-tier fallback chain.
///
/// Mirrors graph_tasks.py:1090-1110:
/// 1. `gap_resolve_patch` (patch-based incremental — preferred)
/// 2. On failure: `gap_resolve` (full regeneration)
/// 3. On that failure: heuristic (add one node per retrieved item)
///
/// The empty-retrieved guard lives in `run.rs` (not here) — `resolve()` is
/// only called when `retrieved` is non-empty.
pub struct GraphGapResolve {
    pub agent_id: String,
    pub model: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
}

#[async_trait]
impl GapResolve<ArgumentationGraph> for GraphGapResolve {
    async fn resolve(
        &self,
        draft: &ArgumentationGraph,
        _fitness: &FitnessEval, // Stage 1 resolve is retrieval-driven; fitness ignored (byte-stable)
        gaps: &[IdentifiedGap],
        retrieved: &[RetrievedContext],
        executor: &Arc<dyn AgentExecutor>,
        config: &TtdConfig,
    ) -> Result<ArgumentationGraph, TtdError> {
        use crate::ttd::prompts::graph::{
            render_gap_resolve, render_gap_resolve_patch, GapResolveInput,
        };
        use alzina_core::identity::AgentId;

        let agent_id = AgentId::new(self.agent_id.clone());

        // B2: build retrieved text for v2 render path.
        let retrieved_text_v2: String = retrieved
            .iter()
            .map(|r| format!("[{}]\n{}", r.source_id, r.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        // Tier 1: patch-based resolution.
        if config.incremental_resolve {
            // B2: fork on profile. Decision 0: v3 = v2.
            let patch_prompt = match self.profile {
                PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                    let graph_xml = draft.to_xml_string();
                    let gap_desc = gaps.iter().map(|g| g.description.as_str()).collect::<Vec<_>>().join("; ");
                    crate::ttd::prompts::lit_review::render_gap_resolve_patch_v2(
                        &graph_xml,
                        &gap_desc,
                        &retrieved_text_v2,
                    )
                }
                PromptProfile::V1Delphi => render_gap_resolve_patch(&GapResolveInput {
                    draft_nodes: draft
                        .nodes
                        .iter()
                        .map(|n| (n.id.clone(), n.claim.clone()))
                        .collect(),
                    draft_edges: draft
                        .edges
                        .iter()
                        .map(|e| (e.source.clone(), e.target.clone(), e.relation.clone()))
                        .collect(),
                    gaps: gaps.iter().map(|g| (g.description.clone(), g.query.clone())).collect(),
                    retrieved: retrieved
                        .iter()
                        .map(|r| (r.source_id.clone(), r.content.clone()))
                        .collect(),
                    fitness_feedback: None,
                }),
            };

            match executor
                .execute(&agent_id, &patch_prompt, &self.model, "gap_resolve_patch")
                .await
            {
                Ok(raw) => {
                    if let Some(result) = try_apply_patch(draft, &raw) {
                        return Ok(result);
                    }
                    tracing::debug!(
                        "gap_resolve_patch: patch application failed; \
                         falling through to full regeneration"
                    );
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "gap_resolve_patch executor error; falling through to full regen"
                    );
                }
            }
        }

        // Tier 2: full regeneration.
        // B2: fork on profile. Decision 0: v3 = v2.
        let full_prompt = match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                let graph_xml = draft.to_xml_string();
                let gap_desc = gaps.iter().map(|g| g.description.as_str()).collect::<Vec<_>>().join("; ");
                crate::ttd::prompts::lit_review::render_gap_resolve_v2(
                    &graph_xml,
                    &gap_desc,
                    &retrieved_text_v2,
                )
            }
            PromptProfile::V1Delphi => render_gap_resolve(&GapResolveInput {
                draft_nodes: draft
                    .nodes
                    .iter()
                    .map(|n| (n.id.clone(), n.claim.clone()))
                    .collect(),
                draft_edges: draft
                    .edges
                    .iter()
                    .map(|e| (e.source.clone(), e.target.clone(), e.relation.clone()))
                    .collect(),
                gaps: gaps.iter().map(|g| (g.description.clone(), g.query.clone())).collect(),
                retrieved: retrieved
                    .iter()
                    .map(|r| (r.source_id.clone(), r.content.clone()))
                    .collect(),
                fitness_feedback: None,
            }),
        };

        match executor
            .execute(&agent_id, &full_prompt, &self.model, "gap_resolve")
            .await
        {
            Ok(raw) => {
                if let Some(xml_content) = extract_xml_block(&raw, "graph") {
                    match parse_full_graph_xml(&xml_content, &self.model, self.profile.graph_prompt_version()) {
                        Ok(graph) => return Ok(graph),
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                "gap_resolve: full regen parse failed; using heuristic"
                            );
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "gap_resolve: full regen executor error; using heuristic"
                );
            }
        }

        // Tier 3: heuristic — add one node per retrieved item (graph_tasks.py pattern).
        let heuristic_result = apply_heuristic_resolve(draft, retrieved);
        tracing::debug!(
            n_nodes_added = retrieved.len(),
            "gap_resolve: heuristic fallback applied (one node per retrieved item)"
        );
        Ok(heuristic_result)
    }
}

/// Try to apply a `<patch>` document to a graph.
///
/// Returns `None` if the patch is empty, malformed, or fails to parse.
/// This implements the patch-apply path from gap_resolve_patch.mustache.
fn try_apply_patch(graph: &ArgumentationGraph, raw: &str) -> Option<ArgumentationGraph> {
    let xml_content = extract_xml_block(raw, "patch")?;

    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(&xml_content);
    reader.trim_text(true);
    let mut buf = Vec::new();

    let mut result = graph.clone();
    let mut inside_add = false;
    let mut inside_node = false;
    let mut current_id: Option<String> = None;
    let mut current_text: Option<String> = None;
    let mut current_expert: Option<String> = None;
    let mut inside_text = false;
    let mut inside_source = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"add" => {
                inside_add = true;
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"node" && inside_add => {
                inside_node = true;
                current_id = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"id")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"text" && inside_node => {
                inside_text = true;
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"source" && inside_node => {
                inside_source = true;
            }
            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().to_string();
                if inside_text {
                    current_text = Some(text);
                } else if inside_source {
                    current_expert = Some(text);
                }
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"text" => {
                inside_text = false;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"source" => {
                inside_source = false;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"node" && inside_add => {
                inside_node = false;
                if let Some(id) = current_id.take() {
                    result.nodes.push(GraphNode {
                        id,
                        claim: current_text.take().unwrap_or_default(),
                        expert_id: current_expert.take().unwrap_or_default(),
                        quote: None,
                        verification_status: None,
                    });
                }
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"add" => {
                inside_add = false;
            }
            Ok(Event::Eof) => break,
            Err(_) => return None,
            _ => {}
        }
        buf.clear();
    }

    Some(result)
}

/// Parse a full `<graph>` XML response into an `ArgumentationGraph`.
fn parse_full_graph_xml(
    xml_content: &str,
    model: &str,
    prompt_version: &str,
) -> Result<ArgumentationGraph, TtdError> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut graph = ArgumentationGraph::new("", "", "", model, prompt_version);
    let mut reader = Reader::from_str(xml_content);
    reader.trim_text(true);
    let mut buf = Vec::new();

    let mut current_id: Option<String> = None;
    let mut current_expert: Option<String> = None;
    let mut current_claim: Option<String> = None;
    let mut current_quote: Option<String> = None;
    let mut inside_text = false;
    let mut inside_source = false;
    let mut inside_quote = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"node" => {
                current_id = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.as_ref() == b"id")
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"text" => {
                inside_text = true;
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"source" => {
                inside_source = true;
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"quote" => {
                inside_quote = true;
            }
            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().to_string();
                if inside_text {
                    current_claim = Some(text);
                } else if inside_source {
                    current_expert = Some(text);
                } else if inside_quote {
                    current_quote = Some(text);
                }
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"text" => {
                inside_text = false;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"source" => {
                inside_source = false;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"quote" => {
                inside_quote = false;
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"node" => {
                if let Some(id) = current_id.take() {
                    graph.nodes.push(GraphNode {
                        id,
                        claim: current_claim.take().unwrap_or_default(),
                        expert_id: current_expert.take().unwrap_or_default(),
                        // Item 2: quotes survive the denoise round-trip.
                        quote: current_quote.take(),
                        verification_status: None,
                    });
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(TtdError::Executor(alzina_core::error::AlzinaError::Orchestration(
                    format!("XML parse error: {e}"),
                )));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(graph)
}

/// Heuristic gap resolve: add one node per retrieved item (graph_tasks.py pattern).
fn apply_heuristic_resolve(
    graph: &ArgumentationGraph,
    retrieved: &[RetrievedContext],
) -> ArgumentationGraph {
    let mut result = graph.clone();
    for (i, item) in retrieved.iter().enumerate() {
        result.nodes.push(GraphNode {
            id: format!("heuristic_n{i}"),
            claim: item.content.chars().take(200).collect(),
            expert_id: item.source_id.clone(),
            quote: None,
            verification_status: Some("unverified".to_string()),
        });
    }
    result
}

// ── GraphMerger ───────────────────────────────────────────────────────────────

/// Stage-1 merger: synthesise the sorted candidate graphs into one final graph.
///
/// Calls the `graph_merger` governed spawn to fold the N sorted candidates
/// (best-first) into a single `ArgumentationGraph`.
pub struct GraphMerger {
    pub agent_id: String,
    pub model: String,
    pub prompt_version: String,
}

#[async_trait]
impl Merger<ArgumentationGraph> for GraphMerger {
    async fn merge(
        &self,
        candidates: &[ArgumentationGraph],
        _executor: &Arc<dyn AgentExecutor>,
        _config: &TtdConfig,
    ) -> Result<ArgumentationGraph, TtdError> {
        if candidates.is_empty() {
            return Err(TtdError::NoCandidates);
        }

        // Phase 23: return the best candidate (first in sorted order).
        // The full merger spawn is a Wave 3 addition (Plan 23-04).
        // For Phase 23, the "merge" is the best-first candidate selection.
        Ok(candidates[0].clone())
    }
}

// ── GraphEvalFitness ──────────────────────────────────────────────────────────

/// Stage-1 fitness evaluation: sequential judge spawns per evaluate() call.
///
/// Profile fork:
/// - `V1Delphi` (default): 6 v1 judge dims via `render_fitness_judge`; `is_valid_graph`; `GRAPH_WEIGHTS`.
/// - `V2LitReview`: 5 v2 lit-review dims via `render_fitness_judge_v2_graph`; `is_valid_v2`; `V2_GRAPH_WEIGHTS`;
///   graph traceability veto attached after scoring.
///
/// Cross-trajectory concurrency is capped by the `max_concurrent_fitness_evals` semaphore
/// in TtdMachine::run() (A5 rung 4). Returns a `FitnessEval` with one `Option<u8>` per dim.
pub struct GraphEvalFitness {
    pub agent_id: String,
    pub model: String,
    /// Prompt/schema dialect. Defaults to V1Delphi — existing callers unaffected.
    pub profile: PromptProfile,
    /// Known panel expert-id set. Threaded at machine build time (stage-1 panel).
    /// Used by `traceability_veto_graph` on the v2 arm only. Empty default is safe
    /// because the shape lane (arxiv:/s2:) covers non-panel paper expert_ids.
    pub panel_ids: std::collections::HashSet<String>,
}

#[async_trait]
impl EvalFitness<ArgumentationGraph> for GraphEvalFitness {
    async fn evaluate(
        &self,
        draft: &ArgumentationGraph,
        executor: &Arc<dyn AgentExecutor>,
        _config: &TtdConfig,
    ) -> Result<FitnessEval, TtdError> {
        use alzina_core::identity::AgentId;

        let agent_id = AgentId::new(self.agent_id.clone());

        match self.profile {
            // Decision 0: v3 scores with the v2 judges unchanged (no judge changes).
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => {
                // v2 path: 5 dims from V2_JUDGE_DIMS, anchored lit-review prompts.
                use crate::ttd::prompts::lit_review::render_fitness_judge_v2_graph;
                use crate::ttd::term_sheet::V2_JUDGE_DIMS;

                let mut scores: Vec<(String, Option<u8>)> = Vec::with_capacity(5);

                // WR-05: degrade a failed spawn to None, do NOT abort.
                for dim in &V2_JUDGE_DIMS {
                    let prompt = render_fitness_judge_v2_graph(dim, draft);
                    let score = match executor.execute(&agent_id, &prompt, &self.model, dim.name).await {
                        Ok(raw) => parse_fitness_score(&raw),
                        Err(e) => {
                            tracing::debug!(
                                dimension = dim.name,
                                error = %e,
                                "GraphEvalFitness (v2): judge spawn failed — score=None"
                            );
                            None
                        }
                    };
                    scores.push((dim.name.to_string(), score));
                }

                // Deterministic traceability veto — computed on the draft structure,
                // NOT via an LLM judge. Attached before returning (T-B3-01 closure).
                // F13: pass panel_ids so the allowlist covers panel-member expert ids.
                let veto = traceability_veto_graph(draft, &self.panel_ids);
                let eval = FitnessEval::new(scores);
                Ok(if let Some(reason) = veto { eval.with_veto(reason) } else { eval })
            }

            PromptProfile::V1Delphi => {
                // v1 path: 6 dims from GRAPH_WEIGHTS, existing v1 prompts (byte-identical).
                use crate::ttd::prompts::graph::{render_fitness_judge, FitnessJudgeInput};

                let dimensions = GRAPH_WEIGHTS.iter().map(|(d, _)| *d).collect::<Vec<_>>();
                let mut scores: Vec<(String, Option<u8>)> = Vec::new();

                // Run 6 sequential fitness judge spawns (one per dimension).
                // Cross-trajectory concurrency is capped externally by the semaphore
                // in TtdMachine::run() — do NOT parallelise within evaluate().
                //
                // WR-05: a per-dimension judge spawn error degrades that dimension to
                // `None` (consensus `_default_score`, fitness.py:740-747) — it does NOT
                // abort the whole run with `?`. This makes graph consistent with the
                // synthesis/narrative evaluators AND keeps the budget accounting honest:
                // every dimension issues exactly one spawn (Ok or Err), so the run loop's
                // fixed count reflects the real spawn count.
                for dim in &dimensions {
                    let prompt = render_fitness_judge(&FitnessJudgeInput {
                        dimension: dim.to_string(),
                        draft_nodes: draft
                            .nodes
                            .iter()
                            .map(|n| (n.id.clone(), n.claim.clone()))
                            .collect(),
                    });

                    let score = match executor.execute(&agent_id, &prompt, &self.model, dim).await {
                        Ok(raw) => parse_fitness_score(&raw),
                        Err(e) => {
                            tracing::debug!(
                                dimension = *dim,
                                error = %e,
                                "GraphEvalFitness: judge spawn failed — score=None (parse failure)"
                            );
                            None
                        }
                    };
                    scores.push((dim.to_string(), score));
                }

                // v1 path: no veto (None by default in FitnessEval::new).
                Ok(FitnessEval::new(scores))
            }
        }
    }

    fn validity_fn(&self) -> fn(&FitnessEval) -> bool {
        match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => is_valid_v2,
            PromptProfile::V1Delphi => is_valid_graph,
        }
    }

    fn weights(&self) -> &'static [(&'static str, f32)] {
        match self.profile {
            PromptProfile::V2LitReview | PromptProfile::V3LitReviewLong => V2_GRAPH_WEIGHTS,
            PromptProfile::V1Delphi => GRAPH_WEIGHTS,
        }
    }
}

/// Parse a fitness score from a fitness-judge LLM response.
///
/// WR-08: delegates to the single canonical parser
/// `fitness::parse_fitness_response`, which is the one source of truth for the
/// parse ladder (empty/whitespace → `None` abstain per WR-03, XML `<score>`
/// extraction, integer CLAMP to `[1,5]` per CR-03, regex `\b([1-5])\b` fallback
/// bounded to the first 200 chars per WR-02, then `None`). Hand-rolling a
/// per-stage parser is exactly the carry-forward drift that caused CR-03/WR-03.
fn parse_fitness_score(raw: &str) -> Option<u8> {
    crate::ttd::fitness::parse_fitness_response(raw).score
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::adapter::{ExpertResponse, SourceId};
    use crate::executor::AgentExecutor;
    use crate::ttd::artifact::{ArgumentationGraph, GraphNode};
    use crate::ttd::mod_types::TtdError;
    use crate::ttd::stages::{DraftGen, GapResolve, RetrievedContext};
    use crate::ttd::state::IdentifiedGap;
    use crate::ttd::TtdConfig;

    use super::*;

    // ── Mock executor that returns pre-canned XML ─────────────────────────────

    struct XmlMockExecutor {
        response: String,
    }

    #[async_trait]
    impl AgentExecutor for XmlMockExecutor {
        async fn execute(
            &self,
            _agent_id: &alzina_core::identity::AgentId,
            _instruction: &str,
            _model: &str,
            _task: &str,
        ) -> alzina_core::AlzinaResult<String> {
            Ok(self.response.clone())
        }
    }

    fn make_executor(response: &str) -> Arc<dyn AgentExecutor> {
        Arc::new(XmlMockExecutor {
            response: response.to_string(),
        })
    }

    fn make_expert(id: &str, prose: &str) -> ExpertResponse {
        ExpertResponse {
            expert_id: SourceId::new(id.to_string()),
            prose: prose.to_string(),
            provenance: crate::adapter::ResponseProvenance {
                source_id: SourceId::new(id.to_string()),
                title: "Test Paper".to_string(),
                year: None,
                authors: vec![],
                credibility_tier: alzina_search::CredibilityTier::Unknown,
            },
        }
    }

    // ── Extraction concurrency cap test ───────────────────────────────────────

    /// With 15 experts, no more than 10 extraction spawns run concurrently.
    ///
    /// We verify this by counting max concurrent in-flight tasks.
    #[tokio::test]
    async fn extraction_concurrency_capped_at_10() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;
        use tokio::sync::Mutex;

        let in_flight = StdArc::new(AtomicUsize::new(0));
        let max_observed = StdArc::new(AtomicUsize::new(0));

        let in_flight_clone = in_flight.clone();
        let max_clone = max_observed.clone();

        struct PeakExecutor {
            in_flight: StdArc<AtomicUsize>,
            max_observed: StdArc<AtomicUsize>,
        }

        #[async_trait]
        impl AgentExecutor for PeakExecutor {
            async fn execute(
                &self,
                _agent_id: &alzina_core::identity::AgentId,
                _instruction: &str,
                _model: &str,
                _task: &str,
            ) -> alzina_core::AlzinaResult<String> {
                let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                // Track max observed concurrent tasks.
                let mut prev_max = self.max_observed.load(Ordering::SeqCst);
                while current > prev_max {
                    match self.max_observed.compare_exchange(
                        prev_max,
                        current,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(actual) => prev_max = actual,
                    }
                }
                // Simulate async work.
                tokio::task::yield_now().await;
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                // Return an empty graph XML.
                Ok("<graph><nodes></nodes><edges></edges></graph>".to_string())
            }
        }

        let executor: Arc<dyn AgentExecutor> = Arc::new(PeakExecutor {
            in_flight: in_flight_clone,
            max_observed: max_clone.clone(),
        });

        let draft_gen = GraphDraftGen::new("test-agent", "test-model", "v1/graph");

        // 15 experts — above the cap of 10.
        let inputs: Vec<ExpertResponse> = (0..15)
            .map(|i| make_expert(&format!("expert_{i:02}"), "expert prose"))
            .collect();

        let config = TtdConfig::default();
        let _ = draft_gen.generate(&inputs, &executor, &config, None, None).await;

        // The peak concurrent in-flight count must never exceed EXTRACT_CONCURRENCY=10.
        let peak = max_clone.load(Ordering::SeqCst);
        assert!(
            peak <= EXTRACT_CONCURRENCY,
            "peak concurrent extractions {peak} must not exceed cap {EXTRACT_CONCURRENCY}"
        );
    }

    // ── F14: node-id namespacing guard ─────────────────────────────────────────

    /// F14 (probe-22): when the extraction model already paper-scopes a node id,
    /// `parse_extraction_xml` must NOT prepend expert_id again. A double-prefixed
    /// id (`arxiv:X_arxiv:X_C13`) is what the synthesis draft failed to copy into
    /// `<node_refs>`, starving the merger. Plain `C13`-style ids are still
    /// namespaced; already-prefixed ids pass through unchanged.
    #[test]
    fn extraction_does_not_double_prefix_paper_scoped_node_ids() {
        let draft_gen = GraphDraftGen::new("agent", "model", "v1/graph");
        let expert_id = "arxiv:2501.13956";
        let xml = format!(
            "<graph><node id=\"{expert_id}_C13\"><text>already paper-scoped</text></node>\
             <node id=\"C7\"><text>bare id</text></node></graph>"
        );
        let g = parse_extraction_xml(&xml, expert_id, &draft_gen).unwrap();
        let ids: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(
            ids.contains(&"arxiv:2501.13956_C13"),
            "already-prefixed id must pass through (no double prefix); got {ids:?}"
        );
        assert!(
            ids.contains(&"arxiv:2501.13956_C7"),
            "bare id must still be namespaced; got {ids:?}"
        );
        assert!(
            !ids.iter().any(|id| id.contains("arxiv:2501.13956_arxiv:2501.13956")),
            "no node id may be double-prefixed; got {ids:?}"
        );
    }

    // ── Merge dedup test ──────────────────────────────────────────────────────

    /// Two experts with overlapping node IDs produce a merged graph with deduplicated nodes.
    #[test]
    fn merge_dedups_overlapping_node_ids() {
        let draft_gen = GraphDraftGen::new("agent", "model", "v1/graph");

        let mut g1 = ArgumentationGraph::new("s", "r", "q", "m", "v");
        g1.nodes.push(GraphNode {
            id: "expert_01_c001".to_string(),
            claim: "claim from expert 1".to_string(),
            expert_id: "expert_01".to_string(),
            quote: None,
            verification_status: None,
        });

        let mut g2 = ArgumentationGraph::new("s", "r", "q", "m", "v");
        g2.nodes.push(GraphNode {
            id: "expert_01_c001".to_string(), // Same ID — overlap
            claim: "duplicate claim".to_string(),
            expert_id: "expert_01".to_string(),
            quote: None,
            verification_status: None,
        });
        g2.nodes.push(GraphNode {
            id: "expert_02_c001".to_string(), // Unique ID
            claim: "claim from expert 2".to_string(),
            expert_id: "expert_02".to_string(),
            quote: None,
            verification_status: None,
        });

        let merged = draft_gen.merge_graphs(vec![g1, g2]);

        assert_eq!(
            merged.nodes.len(),
            2,
            "merge must deduplicate overlapping node IDs: expected 2, got {}",
            merged.nodes.len()
        );
        let ids: Vec<&str> = merged.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(
            ids.contains(&"expert_01_c001"),
            "expert_01_c001 must be present"
        );
        assert!(
            ids.contains(&"expert_02_c001"),
            "expert_02_c001 must be present"
        );
    }

    // ── Gap resolve fallback chain test ───────────────────────────────────────

    /// The three-tier fallback chain is exercised: patch → full-regen → heuristic.
    #[tokio::test]
    async fn gap_resolve_fallback_chain() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        let call_count = StdArc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        struct FailingExecutor {
            call_count: StdArc<AtomicUsize>,
        }

        #[async_trait]
        impl AgentExecutor for FailingExecutor {
            async fn execute(
                &self,
                _agent_id: &alzina_core::identity::AgentId,
                _instruction: &str,
                _model: &str,
                task: &str,
            ) -> alzina_core::AlzinaResult<String> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                // Return unparseable output so all tiers fail through to heuristic.
                Ok("not valid xml".to_string())
            }
        }

        let executor: Arc<dyn AgentExecutor> = Arc::new(FailingExecutor {
            call_count: call_count_clone,
        });

        let resolver = GraphGapResolve {
            agent_id: "agent".to_string(),
            model: "model".to_string(),
            profile: crate::ttd::term_sheet::PromptProfile::V1Delphi,
        };

        let draft = ArgumentationGraph::new("s", "r", "q", "m", "v1/graph");
        let gaps = vec![IdentifiedGap {
            description: "test gap".to_string(),
            query: "test query".to_string(),
        }];
        let retrieved = vec![RetrievedContext {
            source_id: "paper_01".to_string(),
            content: "some retrieved content".to_string(),
            section: None,
        }];

        let mut config = TtdConfig::default();
        config.incremental_resolve = true;

        let result = resolver
            .resolve(&draft, &FitnessEval::new(vec![]), &gaps, &retrieved, &executor, &config)
            .await;

        assert!(result.is_ok(), "fallback chain must always return a result: {result:?}");

        // The heuristic adds one node per retrieved item.
        let graph = result.unwrap();
        assert!(
            !graph.nodes.is_empty(),
            "heuristic fallback must add nodes from retrieved items"
        );

        // Both patch (tier 1) and full-regen (tier 2) must have been attempted.
        assert!(
            call_count.load(Ordering::SeqCst) >= 2,
            "at least 2 executor calls expected (patch + full-regen); got {}",
            call_count.load(Ordering::SeqCst)
        );
    }

    // ── Single-expert resolution skip test ────────────────────────────────────

    /// When there is only one expert, resolution is skipped (no cross-expert edges).
    #[tokio::test]
    async fn resolution_skipped_for_single_expert() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        let call_count = StdArc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        struct CountExecutor {
            count: StdArc<AtomicUsize>,
        }

        #[async_trait]
        impl AgentExecutor for CountExecutor {
            async fn execute(
                &self,
                _agent_id: &alzina_core::identity::AgentId,
                _instruction: &str,
                _model: &str,
                task: &str,
            ) -> alzina_core::AlzinaResult<String> {
                self.count.fetch_add(1, Ordering::SeqCst);
                // Return a minimal graph XML for extraction_single.
                Ok("<graph><nodes><node id=\"c001\" type=\"claim\"><text>test claim</text><sources><source>e1</source></sources></node></nodes><edges></edges></graph>".to_string())
            }
        }

        let executor: Arc<dyn AgentExecutor> = Arc::new(CountExecutor {
            count: call_count_clone,
        });

        let draft_gen = GraphDraftGen::new("agent", "model", "v1/graph");

        // Only ONE expert.
        let inputs = vec![make_expert("expert_01", "expert prose text here")];
        let config = TtdConfig::default();

        let _ = draft_gen.generate(&inputs, &executor, &config, None, None).await;

        // Resolution spawn is NOT called for a single-expert panel.
        // The executor is called once (extraction_single only).
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "resolution must NOT be called for single-expert panel; \
             expected 1 executor call (extraction_single only), got {}",
            call_count.load(Ordering::SeqCst)
        );
    }

    // ── WR-06: resolution-failure fallback preserves merged graph ──────────────

    /// When the resolution spawn errors, the trajectory must keep the
    /// successfully-merged (but unresolved) graph rather than returning an empty
    /// one. Regression for WR-06: the old fallback merged a vec of EMPTY graphs.
    #[tokio::test]
    async fn resolution_failure_preserves_merged_nodes() {
        /// Returns valid node XML for extraction, but ERRORS on the resolution
        /// task — exercising the WR-06 fallback path.
        struct ResolutionFailExecutor;

        #[async_trait]
        impl AgentExecutor for ResolutionFailExecutor {
            async fn execute(
                &self,
                _agent_id: &alzina_core::identity::AgentId,
                _instruction: &str,
                _model: &str,
                task: &str,
            ) -> alzina_core::AlzinaResult<String> {
                if task == "resolution" {
                    return Err(alzina_core::error::AlzinaError::Orchestration(
                        "simulated resolution spawn failure".to_string(),
                    ));
                }
                // Extraction tasks return one node per expert.
                Ok("<graph><nodes><node id=\"c001\" type=\"claim\">\
                    <text>extracted claim</text><sources><source>e1</source>\
                    </sources></node></nodes><edges></edges></graph>"
                    .to_string())
            }
        }

        let executor: Arc<dyn AgentExecutor> = Arc::new(ResolutionFailExecutor);
        let draft_gen = GraphDraftGen::new("agent", "model", "v1/graph");

        // TWO experts → resolution runs (and fails).
        let inputs = vec![
            make_expert("expert_01", "prose for expert 1"),
            make_expert("expert_02", "prose for expert 2"),
        ];
        let config = TtdConfig::default();

        let graph = draft_gen
            .generate(&inputs, &executor, &config, None, None)
            .await
            .expect("generate must succeed even when resolution fails");

        assert!(
            !graph.nodes.is_empty(),
            "resolution failure must preserve the merged nodes, not return an \
             empty graph (WR-06); got {} nodes",
            graph.nodes.len()
        );
    }

    /// Self-closing `<edge from=".." to=".." type=".."/>` arrives as
    /// Event::Empty; without the Empty arm every such edge was silently
    /// dropped (probe-10 provenance sweep). Both shapes must parse.
    #[test]
    fn apply_resolution_parses_self_closing_edges() {
        let mut graph = crate::ttd::artifact::ArgumentationGraph::new(
            "s", "r", "q", "test-model", "v1/graph",
        );
        graph.nodes.push(crate::ttd::artifact::GraphNode {
            id: "e1_c001".into(),
            claim: "A".into(),
            expert_id: "e1".into(),
            quote: None,
            verification_status: None,
        });
        graph.nodes.push(crate::ttd::artifact::GraphNode {
            id: "e2_c001".into(),
            claim: "B".into(),
            expert_id: "e2".into(),
            quote: None,
            verification_status: None,
        });

        let resolution = r#"<resolution>
  <edges>
    <edge from="e1_c001" to="e2_c001" type="supports"/>
    <edge from="e2_c001" to="e1_c001" type="attacks">
      <reasoning>contradicts the premise</reasoning>
    </edge>
  </edges>
</resolution>"#;

        let graph = apply_resolution_xml(graph, resolution)
            .expect("resolution parse must succeed");

        assert_eq!(
            graph.edges.len(),
            2,
            "both self-closing AND expanded edge forms must be captured"
        );
        assert!(graph.edges.iter().any(|e| e.relation == "supports"));
        assert!(graph.edges.iter().any(|e| e.relation == "attacks"));
    }

    // ── F2 regression tests (Task 2 / Plan 25-01) ────────────────────────────

    /// F2-extraction: v2 extraction XML following the corrected schema (using
    /// <text> for claim body, <expert_id>, <evidence>, <verification_status>)
    /// must parse through parse_extraction_xml with a non-empty claim field.
    ///
    /// RED before fix: the current v2 extraction prompt schema example uses
    /// <claim> for the claim body; parse_extraction_xml reads <text>. So a model
    /// following the prompt emits <claim> and the parsed claim comes back empty.
    ///
    /// This test uses the FIXED schema (<text>) — it passes after the prompt fix.
    /// The symmetrical broken-schema test below documents the failure mode.
    #[test]
    fn round_trip_v2_graph_dialect() {
        // XML following the v2 extraction schema AFTER the <claim>→<text> fix.
        // parse_extraction_xml reads <text> for the claim body (graph.rs:367).
        let sample_xml = r#"<graph>
  <node id="arxiv:2304.07620_C1" type="claim">
    <text>Permafrost thaw is a net methane source</text>
    <evidence>Flux measurements 2019-2023</evidence>
    <expert_id>arxiv:2304.07620</expert_id>
    <verification_status>verified</verification_status>
  </node>
</graph>"#;

        let draft_gen = GraphDraftGen::new("test-agent", "test-model", "v2/lit-review");
        let result = parse_extraction_xml(sample_xml, "arxiv:2304.07620", &draft_gen);
        assert!(result.is_ok(), "parse_extraction_xml must succeed: {:?}", result.err());
        let graph = result.unwrap();
        assert_eq!(graph.nodes.len(), 1, "must parse 1 node");
        assert!(
            !graph.nodes[0].claim.is_empty(),
            "F2 regression gate: v2 claim body must not be empty — \
             parse_extraction_xml reads <text>, not <claim>"
        );
        assert_eq!(
            graph.nodes[0].claim,
            "Permafrost thaw is a net methane source",
            "claim body must equal the input <text> content"
        );
    }

    /// F2-to_xml_string: to_xml_string must serialise using <text>/<source> tags
    /// that parse_full_graph_xml can read back.
    ///
    /// RED before fix: to_xml_string writes <claim>{claim}</claim><expert_id>{expert}</expert_id>
    /// but parse_full_graph_xml reads <text> and <source>. Claims and expert IDs
    /// come back empty after the round-trip.
    #[test]
    fn to_xml_string_round_trips_through_parse_full_graph_xml() {
        use crate::ttd::artifact::ArgumentationGraph;

        let mut graph = ArgumentationGraph::new(
            "s", "r", "q", "test-model", "v2/lit-review",
        );
        graph.nodes.push(crate::ttd::artifact::GraphNode {
            id: "arxiv:2304.07620_C1".into(),
            claim: "Permafrost thaw accelerates methane release".into(),
            expert_id: "arxiv:2304.07620".into(),
            quote: None,
            verification_status: Some("verified".into()),
        });
        graph.nodes.push(crate::ttd::artifact::GraphNode {
            id: "arxiv:2308.06046_C1".into(),
            claim: "Arctic amplification increases soil temperature".into(),
            expert_id: "arxiv:2308.06046".into(),
            quote: None,
            verification_status: None,
        });

        let xml = graph.to_xml_string();

        // Verify the serialised XML uses <text> and <source> tags (after fix).
        assert!(
            xml.contains("<text>"),
            "F2: to_xml_string must emit <text> tags (readable by parse_full_graph_xml)"
        );
        assert!(
            xml.contains("<source>"),
            "F2: to_xml_string must emit <source> tags (readable by parse_full_graph_xml)"
        );

        // Round-trip: parse back through parse_full_graph_xml.
        let result = parse_full_graph_xml(&xml, "test-model", "v2/lit-review");
        assert!(result.is_ok(), "parse_full_graph_xml must succeed on to_xml_string output: {:?}", result.err());
        let parsed = result.unwrap();

        assert_eq!(parsed.nodes.len(), 2, "must parse 2 nodes from round-tripped XML");

        let n1 = &parsed.nodes[0];
        assert_eq!(
            n1.claim,
            "Permafrost thaw accelerates methane release",
            "F2: claim must survive round-trip through to_xml_string → parse_full_graph_xml"
        );
        assert_eq!(
            n1.expert_id,
            "arxiv:2304.07620",
            "F2: expert_id must survive round-trip through to_xml_string → parse_full_graph_xml"
        );

        let n2 = &parsed.nodes[1];
        assert_eq!(
            n2.claim,
            "Arctic amplification increases soil temperature",
            "F2: second claim must survive round-trip"
        );
        assert_eq!(
            n2.expert_id,
            "arxiv:2308.06046",
            "F2: second expert_id must survive round-trip"
        );
    }

    /// Worklist item 2 (quote-grounded synthesis sketch): node quotes must
    /// survive to_xml_string → parse_full_graph_xml. RED before fix:
    /// to_xml_string emitted no <quote> element and parse_full_graph_xml
    /// hardcoded quote: None — quotes died at the first denoise round-trip
    /// (probe 14: 0/44 claims quoted).
    #[test]
    fn quote_survives_to_xml_string_round_trip() {
        let mut graph = crate::ttd::artifact::ArgumentationGraph::new(
            "s", "r", "q", "test-model", "v2/lit-review",
        );
        graph.nodes.push(crate::ttd::artifact::GraphNode {
            id: "arxiv:2304.07620_C1".into(),
            claim: "Permafrost thaw accelerates methane release".into(),
            expert_id: "arxiv:2304.07620".into(),
            quote: Some("observed methane flux increased by 38% over the thaw season".into()),
            verification_status: None,
        });
        graph.nodes.push(crate::ttd::artifact::GraphNode {
            id: "arxiv:2308.06046_C1".into(),
            claim: "Quote-less node stays quote-less".into(),
            expert_id: "arxiv:2308.06046".into(),
            quote: None,
            verification_status: None,
        });

        let xml = graph.to_xml_string();
        assert!(xml.contains("<quote>"), "to_xml_string must emit <quote> when present");

        let parsed = parse_full_graph_xml(&xml, "test-model", "v2/lit-review").unwrap();
        assert_eq!(parsed.nodes.len(), 2);
        assert_eq!(
            parsed.nodes[0].quote.as_deref(),
            Some("observed methane flux increased by 38% over the thaw season"),
            "quote must survive the round-trip"
        );
        assert_eq!(
            parsed.nodes[1].quote, None,
            "absent quote must stay None (no empty-string fabrication)"
        );
    }

    // ── F1 regression tests (Task 1 / Plan 25-01) ────────────────────────────

    /// F1-gap-resolve: GraphGapResolve with V2LitReview profile must produce a
    /// graph whose prompt_version == "v2/lit-review" after the full-regen path.
    ///
    /// RED before fix: parse_full_graph_xml is called with hardcoded "v1/graph"
    /// at line 937 regardless of self.profile; the graph carries the wrong version.
    #[tokio::test]
    async fn gap_resolve_full_regen_uses_profile_prompt_version() {
        use crate::ttd::term_sheet::PromptProfile;

        // Executor that returns valid graph XML so the full-regen path succeeds.
        struct FullRegenExecutor;

        #[async_trait]
        impl AgentExecutor for FullRegenExecutor {
            async fn execute(
                &self,
                _agent_id: &alzina_core::identity::AgentId,
                _instruction: &str,
                _model: &str,
                _task: &str,
            ) -> alzina_core::AlzinaResult<String> {
                // Return valid graph XML so parse_full_graph_xml succeeds.
                Ok(r#"<graph>
  <node id="arxiv:2304.07620_C1" type="claim">
    <text>Permafrost thaw is a net methane source</text>
    <source>arxiv:2304.07620</source>
  </node>
</graph>"#.to_string())
            }
        }

        let executor: Arc<dyn AgentExecutor> = Arc::new(FullRegenExecutor);

        let resolver = GraphGapResolve {
            agent_id: "agent".to_string(),
            model: "test-model".to_string(),
            profile: PromptProfile::V2LitReview,
        };

        let draft = ArgumentationGraph::new("s", "r", "q", "test-model", "v2/lit-review");
        let gaps = vec![IdentifiedGap {
            description: "gap needing full regen".to_string(),
            query: "permafrost methane".to_string(),
        }];
        let retrieved = vec![RetrievedContext {
            source_id: "arxiv:2304.07620".to_string(),
            content: "retrieved content".to_string(),
            section: None,
        }];

        // Force full-regen path (not patch).
        let mut config = TtdConfig::default();
        config.incremental_resolve = false;

        let result = resolver
            .resolve(&draft, &FitnessEval::new(vec![]), &gaps, &retrieved, &executor, &config)
            .await
            .expect("full-regen resolve must succeed with valid XML executor");

        assert_eq!(
            result.prompt_version,
            "v2/lit-review",
            "F1: GraphGapResolve with V2LitReview profile must stamp \
             prompt_version='v2/lit-review' onto the full-regen result \
             (self.profile.graph_prompt_version(), not hardcoded 'v1/graph')"
        );
    }

    // ── V2 profile selection tests ────────────────────────────────────────────

    /// V2LitReview profile returns V2_GRAPH_WEIGHTS (5 dims) and is_valid_v2.
    #[test]
    fn v2_graph_fitness_returns_v2_weights_and_validity_fn() {
        use crate::ttd::fitness::{is_valid_v2, FitnessEval};
        use crate::ttd::term_sheet::PromptProfile;
        use crate::ttd::weights::{GRAPH_WEIGHTS, V2_GRAPH_WEIGHTS};

        let v2 = GraphEvalFitness {
            agent_id: "agent".into(),
            model: "model".into(),
            profile: PromptProfile::V2LitReview,
            panel_ids: std::collections::HashSet::new(), // empty = shape lane only
        };
        assert_eq!(
            v2.weights(),
            V2_GRAPH_WEIGHTS,
            "V2LitReview must return V2_GRAPH_WEIGHTS"
        );
        assert_eq!(
            v2.weights().len(),
            5,
            "V2 graph weight table must have 5 dims"
        );
        let vfn = v2.validity_fn();
        // faithfulness=4 → valid under is_valid_v2
        let ok = FitnessEval::new(vec![("faithfulness".into(), Some(4))]);
        assert!(vfn(&ok), "faithfulness=4 must be valid for V2 graph");
        // faithfulness=3 → invalid
        let fail = FitnessEval::new(vec![("faithfulness".into(), Some(3))]);
        assert!(!vfn(&fail), "faithfulness=3 must be invalid for V2 graph");

        // V1 path unchanged
        let v1 = GraphEvalFitness {
            agent_id: "agent".into(),
            model: "model".into(),
            profile: PromptProfile::V1Delphi,
            panel_ids: std::collections::HashSet::new(),
        };
        assert_eq!(
            v1.weights(),
            GRAPH_WEIGHTS,
            "V1Delphi must return GRAPH_WEIGHTS"
        );
        assert_eq!(
            v1.weights().len(),
            6,
            "V1 graph weight table must have 6 dims"
        );
    }

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║ SEAM F4b — GRAPH parser (characterisation net, W-522022c5)              ║
    // ║ Reaches the private file-level `parse_gaps_xml` via super::super.        ║
    // ║ PINS THE T1 RULED CONTRACT: returns Result; a gap is valid iff non-empty ║
    // ║ <description> (query defaults to description); missing block → Ok(vec![]).║
    // ║ (Re-baselined from the prior bare-Vec / both-fields-required contract per ║
    // ║  the trip-wire protocol — the change is INTENDED under Skuld's T1 ruling: ║
    // ║  the silent desc-only drop was the BUG this rune removes.)               ║
    // ╚═══════════════════════════════════════════════════════════════════════╝
    mod f4b_graph_parser {
        #[test]
        fn f4_graph_desc_only_yields_one_gap() {
            // T1 RULED CONTRACT: desc-only is valid; query defaults to description.
            let desc_only = "<gaps><gap><description>a gap</description></gap></gaps>";
            let out = super::super::parse_gaps_xml(desc_only).expect("graph: desc-only must be Ok");
            assert_eq!(
                out.len(), 1,
                "F4: graph emits a gap on desc-only (query defaults to desc) — T1 contract"
            );
            assert_eq!(out[0].query, out[0].description, "F4: graph defaults query→description");

            let both = "<gaps><gap><description>d</description><query>q</query></gap></gaps>";
            let out2 = super::super::parse_gaps_xml(both).expect("graph: both-fields must be Ok");
            assert_eq!(out2.len(), 1, "F4: graph keeps a gap when both desc and query present");
            assert_eq!(out2[0].query, "q", "F4: graph honours an explicit query");
        }

        #[test]
        fn f4_graph_missing_block_is_ok_empty() {
            // T1 RULED CONTRACT: missing block → Ok(vec![]) (never Err, never bare Vec).
            let out = super::super::parse_gaps_xml("no gaps here");
            assert!(
                matches!(out, Ok(ref v) if v.is_empty()),
                "F4: graph returns Ok(vec![]) on missing block (T1 contract: never Err)"
            );
        }

        #[test]
        fn f4_graph_does_not_panic_on_multibyte() {
            let multibyte = "<gaps><gap><description>café</description><query>naïve—query</query></gap></gaps>";
            let r = std::panic::catch_unwind(|| super::super::parse_gaps_xml(multibyte));
            assert!(r.is_ok(), "F4: graph must not panic on multibyte content");
        }
    }
}
