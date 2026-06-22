//! Phase-0 spike: prove the TTD synthesis engine runs fully standalone.
//!
//! No daemon, no TypeScript sidecar, no governance, no ADK. A canned
//! three-stage stub executor (copied from the in-crate engine tests) drives one
//! synthesis end to end, and we print the resulting YAML artifact. If this
//! prints a non-empty synthesis, the 80%-separable estimate holds and the
//! OpenRouter executor (Phase 1) can drop straight into the same seam.
//!
//! Run: `cargo run -p cli --example spike`

use std::sync::Arc;

use async_trait::async_trait;

use base::identity::AgentId;
use base::AlzinaResult;
use orchestration::adapter::{ExpertResponse, ResponseProvenance, SourceId};
use orchestration::ttd::engine::{run_engine_with_bib, EngineConfig};
use orchestration::ttd::retrieval::NoopRetriever;
use orchestration::AgentExecutor;
use search::bib_store::{BibliographyStore, NoopBibliographyStore};
use search::CredibilityTier;

/// Returns canned, schema-valid output for each TTD stage, keyed by `task`.
/// Mirrors `ThreeStageStubExecutor` in `ttd/engine.rs` tests.
struct ThreeStageStubExecutor;

#[async_trait]
impl AgentExecutor for ThreeStageStubExecutor {
    async fn execute(
        &self,
        _agent_id: &AgentId,
        _instruction: &str,
        _model: &str,
        task: &str,
    ) -> AlzinaResult<String> {
        // Stage 1: graph extraction
        if task == "graph_extraction_single" || task == "graph_draft" || task == "graph_merger" {
            return Ok(r#"<graph>
  <nodes>
    <node id="arxiv:2105.14103_c001">
      <claim>Permafrost thaw releases methane.</claim>
      <expert_id>arxiv:2105.14103</expert_id>
      <quote>permafrost thaw releases methane</quote>
      <verification_status>verified</verification_status>
    </node>
  </nodes>
  <edges/>
</graph>"#
                .to_string());
        }
        if task == "graph_resolution" {
            return Ok("<edges/>\n<merges/>".to_string());
        }

        // Stage 2: synthesis
        if task == "synthesis_draft" || task == "synthesis_merger" {
            return Ok(r#"<synthesis>
  <narrative>Permafrost thaw is a key concern [C1].</narrative>
  <claims>
    <claim id="C1">
      <text>Permafrost thaw accelerates methane release.</text>
      <agreement_level>consensus</agreement_level>
      <sources><source id="arxiv:2105.14103_c001"/></sources>
      <counterarguments/>
    </claim>
  </claims>
  <areas_of_agreement><area>Warming accelerates permafrost thaw</area></areas_of_agreement>
  <areas_of_disagreement/>
  <uncertainties><uncertainty>Long-term feedback rates unclear</uncertainty></uncertainties>
</synthesis>"#
                .to_string());
        }
        if task == "synthesis_quote_resolve" {
            return Ok("<resolved/>".to_string());
        }

        // Stage 3: narrative
        if task == "narrative_draft" || task == "narrative_refine" || task == "narrative_final_merge"
        {
            return Ok("Permafrost thaw is accelerating under warming, with significant methane release implications [C1].".to_string());
        }
        if task == "narrative_critique" {
            return Ok("<gaps></gaps>".to_string()); // no gaps → fast path
        }

        // Fitness judges (all dimensions → score 4)
        Ok("<fitness_evaluation><score>4</score><rationale>good</rationale></fitness_evaluation>"
            .to_string())
    }
}

fn stub_panel() -> Vec<ExpertResponse> {
    vec![ExpertResponse {
        expert_id: SourceId::new("arxiv:2105.14103"),
        prose: "Permafrost thaw releases significant methane under warming.".into(),
        provenance: ResponseProvenance {
            source_id: SourceId::new("arxiv:2105.14103"),
            title: "Permafrost Thaw Study".into(),
            year: Some(2021),
            authors: vec![],
            credibility_tier: CredibilityTier::Unknown,
        },
    }]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let executor: Arc<dyn AgentExecutor> = Arc::new(ThreeStageStubExecutor);
    let bib: Arc<dyn BibliographyStore> = Arc::new(NoopBibliographyStore);

    let config = EngineConfig::new(
        "gna-spike",
        "google/gemini-2.5-flash",
        "study-spike",
        "round-1",
        "q-permafrost",
    )
    .with_run_id("spike-run")
    .with_retriever(Arc::new(NoopRetriever));

    let result = run_engine_with_bib(&stub_panel(), &config, executor, bib).await?;

    println!("=== STANDALONE TTD ENGINE: OK ===");
    println!("graph nodes: {}", result.graph.nodes.len());
    println!("\n=== SYNTHESIS YAML ===\n{}", result.yaml);

    Ok(())
}
