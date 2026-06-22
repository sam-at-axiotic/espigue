# litreview

Topic in, cited literature review out.

`litreview` is a standalone command-line tool that turns a research question into
a structured, citation-grounded literature review. It searches arXiv and Semantic
Scholar (and your own documents), grounds every claim in retrieved sources, and
synthesises a review built around the *tensions* in a field. One
`OPENROUTER_API_KEY` drives generation, embeddings, and reranking.

Two things set it apart. **Every citation and quote is mechanically verified
against the real source text** — a deterministic check, not the model's word — so
the review puts a hard floor under the hallucination that plain LLM summarisation
invites. And it keeps a **persistent local corpus**: every paper it reads, plus
anything you ingest, accumulates into a queryable memory of a field that you own.

The draft stage is an **evolutionary search**, not a single pass: a population of
parallel expert drafts competes under adversarial judges, the weakest are culled
for drifting from their sources, and survivors are rewritten round by round with
fresh evidence until the fittest is fused into the final review.

It has been evaluated end-to-end with **Haiku driving that draft swarm and a
single Opus 4.8 call to merge** — the surprising part is how much quality holds
with the bulk of the reasoning on Haiku. Defaults ship `gemini-2.5-flash` for
drafts; pass `--model anthropic/claude-haiku-4.5` to reproduce the tested setup.

```bash
pip install litreview
export OPENROUTER_API_KEY=sk-or-...
litreview "test-time compute scaling for language models"
#   → synthesis.yaml + graph.md
```

Each review is also emitted as a `synthesis.yaml` — the review as data, not just
prose: every claim carries its sources, evidence grade, support level, method,
lineage, and mechanically verified quotes, and the field's agreements,
disagreements, uncertainties, and open research gaps are first-class lists. Each
review is a small, queryable knowledge graph of its field.

Five complete, unedited example reviews — `.md` prose plus full `.synthesis.yaml`
— are in **[examples/reviews/](examples/reviews/)**. Full documentation and the
design of the three-stage pipeline are in the package README:
**[crates/cli/README.md](crates/cli/README.md)**.

## Repository layout

This is a self-contained Rust workspace. The CLI is packaged as a pip-installable
wheel via maturin (`bindings = "bin"`).

| Crate | Role |
| --- | --- |
| `crates/cli` | The `litreview` binary + the synthesis pipeline (`run_review`) |
| `crates/orchestration` | Test-time-diffusion synthesis engine (TTD + adapter) |
| `crates/search` | Literature search — RRF fusion, reranking, the vec store |
| `crates/base` | Shared types, error taxonomy, trait definitions |

## Build from source

```bash
cargo build --release -p cli      # → target/release/litreview
# or build the wheel:
pip install maturin
cd crates/cli && maturin build --release
```

## Related project

litreview is a sibling of [Symphonia](https://arc-yh.nihr.ac.uk/research/projects/symphonia-llm-assisted-expert-consensus-platform/),
an LLM-assisted expert-consensus platform, and shares its synthesis engine with
[`axiotic-ai/consensus`](https://github.com/axiotic-ai/consensus) — the original
Python implementation of the test-time-diffusion consensus method ported here.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
