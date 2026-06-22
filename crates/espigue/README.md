# espigue

Topic in, cited literature review out.

`espigue` is a standalone command-line tool that turns a research question into
a structured, citation-grounded literature review. It searches arXiv and Semantic
Scholar (and your own documents), grounds every claim in retrieved sources, and
surfaces a field's **areas of disagreement and open gaps** — then uses them to
drive further, targeted literature search, rather than flattening them into a
summary. One `OPENROUTER_API_KEY` drives generation, embeddings, and reranking.

## What it produces

Below is the unedited opening of a real review, generated from the single
question *"Consensus and agreement mechanisms in multi-agent LLM systems"*. No
hand-editing — this is the tool's output.

> ## Introduction
>
> How do you know when a multi-agent language model system has reached a safe
> commitment to an answer? This question sits at the intersection of two inherited
> frameworks that do not align. Classical distributed consensus—Paxos, Raft, and
> their game-theoretic descendants—assumes deterministic state machines whose
> outputs are fully determined by input and prior state, and treats faults as
> Byzantine. Language models violate both assumptions. [...]
>
> Aegean proposes an answer: round-based stability—the condition that an agent
> produces identical outputs across two consecutive rounds of refinement—serves as
> the necessary condition for safe commitment in multi-agent LLM reasoning (Ruan et
> al., 2025). [...] This is Tension 1 (T1): whether the sufficient conditions that
> classical consensus derives from deterministic state machines and Byzantine
> majority carry over to reasoning systems where outputs vary stochastically and
> participants are non-adversarial.

The full review continues for several thousand words, with per-claim author-year
citations and a complete bibliography. (This is the long-form `v3` profile — one
of several output shapes; see [`--profile`](#how-it-works).) Every cited paper is
a real source the tool retrieved and read, not a model recollection — and every citation and quote is mechanically
verified against that source text (see [Why the output is
trustworthy](#why-the-output-is-trustworthy)).

Five more complete, unedited reviews — a small knowledge base on multi-agent LLM
systems — are in [`examples/reviews/`](../../examples/reviews/).

espigue has been evaluated end-to-end with **Haiku driving the expert draft
swarm and a single Opus 4.8 call to merge**. The surprising result is how much of
this quality holds when the bulk of the reasoning runs on Haiku, with just one
stronger-model call to fuse the drafts. The shipped default uses
`gemini-2.5-flash` for the drafts; pass `--model anthropic/claude-haiku-4.5` to
reproduce the tested setup.

## Quickstart

```bash
pip install espigue

export OPENROUTER_API_KEY=sk-or-...          # required
export S2_API_KEY=...                        # optional — adds the Semantic Scholar lane

espigue "test-time compute scaling for language models"
#   → synthesis.yaml   (the structured review + bibliography)
#   → graph.md         (the claim/argument graph)
```

That is the whole happy path. arXiv needs no key; without `S2_API_KEY` the tool
runs on arXiv plus any local documents you ingest.

`pip install espigue` is all you need — the wheel ships a self-contained compiled
binary, so there is **no Rust toolchain to install and no Python runtime
dependencies**. It is a normal `pip install` that drops the `espigue` command on
your `PATH`.

### From Python

espigue is a CLI, but it is built to drive from Python: `pip install` it, call it
with `subprocess`, and load the structured `synthesis.yaml` — the review as data.

```python
import subprocess, yaml

# The wheel put `espigue` on PATH. Run a review like any other command:
subprocess.run(
    ["espigue", "consensus mechanisms in multi-agent LLM systems", "--out", "."],
    check=True,
)

# synthesis.yaml is plain YAML — load it and work with the structured review:
review = yaml.safe_load(open("synthesis.yaml"))

for claim in review["claims"]:
    print(claim["support_level"], claim["evidence_grade"], "—", claim["text"])
    for q in claim["quotes"]:
        if q["status"] == "verified":            # mechanically verified quote
            print("   ✓", q["source"], "—", q["text"][:80])

print("open gaps:", [g["gap_type"] for g in review["gaps"]])
```

Every field — claims, verified quotes, agreements/disagreements, gaps — is in that
YAML (see [The structured output](#the-structured-output)), so a few lines of
Python turn a review into a dataset you can filter, score, or feed onward.

### Bring your own corpus

```bash
espigue ingest ./papers/                   # index local .txt / .md / .pdf
espigue --scope corpus-only "my question"  # cite only your documents
espigue --scope corpus+web  "my question"  # your documents + arXiv/S2 (default)
```

PDF ingest needs `pdftotext` (poppler): `brew install poppler` /
`apt install poppler-utils`. Text and markdown ingest with no extra tooling.

### Seed the review with specific papers

```bash
espigue --seed-papers 2310.06825,1706.03762 "attention and efficient LLMs"
```

Seeds accept arXiv ids, DOIs, or Semantic Scholar ids. The panel is built directly
from those papers (skipping the search-and-rank stage); gap-filling still runs and
honours `--scope`. DOIs and S2 ids need `S2_API_KEY`; arXiv ids need no key.

## How it works

Three stages, one command:

```
  1. Retrieve & ground            2. Draft                  3. Merge
  ┌─────────────────────┐      ┌────────────────────┐   ┌──────────────────┐
  │ arXiv ┐             │      │ parallel expert    │   │ a stronger model │
  │ S2    ┼─ fuse (RRF) │ ───▶ │ drafts over the    │──▶│ fuses drafts into│
  │ local ┘  + rerank   │      │ grounded panel     │   │ one cited review │
  │  + full-text fetch  │      │ (test-time diffusion)  │  + bibliography  │
  └─────────────────────┘      └────────────────────┘   └──────────────────┘
```

1. **Retrieve and ground.** Three lanes — arXiv, Semantic Scholar, and your local
   corpus — are fused with reciprocal-rank fusion, reranked by a cross-encoder, and
   the top sources are promoted to full text where available. A topicality gate
   drops off-question hits. Reranking degrades loudly: if it fails, the un-reranked
   order is kept and the reason is reported — a source is never silently dropped.
2. **Draft — evolve, don't sample.** A small population of drafts is written in
   parallel, each by a different expert persona. Over fixed rounds they improve
   under selection pressure: adversarial judges score each draft against a rubric,
   its weakest points and unresolved disagreements are exposed as *gaps*, the
   retriever pulls fresh evidence for each gap, and the draft is rewritten to close
   them. A hard validity gate culls any draft that drifts from its sources. The
   fittest survivor moves on.
3. **Merge.** A stronger model fuses the surviving drafts into a single review that
   surfaces the field's disagreements and open questions instead of smoothing them
   into false consensus, with author-year citations and a bibliography.

Choose the depth with `--profile`: `v1`/`delphi` (concise), `v2`/`lit-review`, or
`v3`/`lit-review-long` (the long-form, sectioned format shown above).

### The draft stage is an evolutionary search

It is worth being literal about that middle stage, because it is where the quality
comes from. Think of it as evolution under selection rather than one model writing
an essay:

- The swarm of parallel drafts is a **population** of candidate reviews.
- The judges are the **selection pressure** — a measurable, per-dimension fitness
  function, with a validity gate that eliminates any draft whose claims are not
  grounded in real sources.
- Each round of *identify a gap → retrieve evidence → rewrite* is the **variation**
  that makes the next generation fitter than the last.

What you read is the fittest survivor of that process, fused into final form —
bred against grounding, not sampled once and hoped over. (One honest caveat: the
drafts evolve independently, with no crossover between them, so this is
population-based refinement under selection, not a genetic algorithm. The useful
claim is the accurate one — the artifact earns its place by surviving the judges,
not by sounding confident.)

## Why the output is trustworthy

**Citations and quotes are mechanically verified — the guarantee does not depend
on the model's honesty.** Every citation resolves to a real source the tool
retrieved and read, and a deterministic conservation check ensures no source is
fabricated, dropped, or duplicated between the grounded panel and the final
review. Every quote is then checked by exact substring match against the cited
source's stored text: a quote that matches is marked `verified`; a near-miss is
snapped to the actual sentence in the source; anything that cannot be grounded is
flagged (`paraphrased` / `absent`) rather than passed off as real.

This catches a failure mode we hit directly in evaluation: a small model writes a
fluent sentence, labels it a quote, and attaches a real paper — but the words are
simply not in the source. The verifier rejects those. The effect is a hard floor
on citation-and-quote hallucination that holds regardless of how confident the
generator sounds. (Quote verification runs in the `v2` / `v3` lit-review profiles,
which are the defaults.)

On top of that floor sits quality tuning. Reviews are scored by an LLM judge
against a calibrated, per-dimension rubric — grounding, citation faithfulness,
disagreement coverage, and argument structure are graded separately rather than
rolled into one vibe score. Dead or redundant dimensions were pruned by running the judge
across many corpora and dropping the ones that did not discriminate. The generator
is tuned against measurable quality, not just a prompt that sounds confident.

## The structured output

`synthesis.yaml` is not just the review as text — it is the review as data. The
prose (`graph.md`) is one rendering of it; everything behind that prose is present
in machine-readable form:

- **Claims**, each a structured record: the claim and its author-year citation,
  the sources behind it, a `support_level` (converging / established / contested),
  an `evidence_grade`, the `method` the evidence came from, the `year`, and a
  `lineage` line tracing who first showed it and who corroborated.
- **Verified quotes** per claim — the verbatim sentence, its source, the source
  node it resolves to, and a `status` (`verified` / `snapped` / `paraphrased` /
  `absent`) from the mechanical check above.
- **The shape of the field** as first-class lists: `areas_of_agreement`,
  `areas_of_disagreement`, `uncertainties`, and `minority_reports` — not buried in
  prose.
- **`narrative_statements`**, mapping each sentence of the review back to the
  claims it rests on — sentence-level traceability from prose to evidence.
- **`gaps`** — the open research questions the synthesis surfaced, each typed
  `empirical` / `methodological` / `theoretical`.
- A provenance header (`model`, `prompt_version`, `code_version`, `generated_at`)
  so any review reproduces.

Each review is therefore a small, queryable knowledge graph of a field: you can
sort claims by evidence grade, pull only the contested ones, or follow any quote
back to its source. The five [`examples/reviews/`](../../examples/reviews/) ship
their full `synthesis.yaml` alongside the rendered review.

## A growing corpus you own

espigue keeps a persistent local store (`espigue.db`). Every paper it
retrieves and reads — abstract, full text where available, chunks, embeddings, and
bibliography — is written there and reused on the next run. Documents you `ingest`
join the same store. Over time it becomes a queryable memory of a field that lives
on your disk, not a provider's.

That persistent corpus, the evolutionary draft swarm that grows reviews over it
under judge selection, and the mechanical verification above are most of the
ingredients of a self-improving research loop. The piece that would close it —
feeding verified outputs back to sharpen retrieval and panel selection — is a
direction we are pointed at, not a feature that ships today.

## Related project

espigue is a sibling of **[Symphonia][symphonia]**, an LLM-assisted
expert-consensus platform, and shares its synthesis engine with the
[`axiotic-ai/consensus`][consensus] project — the original Python implementation
of the test-time-diffusion consensus method that this Rust engine was ported from.
The shared idea: ground a panel of sources, draft competing expert views, and
converge on a verified synthesis rather than a single confident summary.

[symphonia]: https://arc-yh.nihr.ac.uk/research/projects/symphonia-llm-assisted-expert-consensus-platform/
[consensus]: https://github.com/axiotic-ai/consensus

## Configuration

| Flag | Default | Purpose |
| --- | --- | --- |
| `--profile` | `v3/lit-review-long` | Output depth and structure |
| `--scope` | `corpus+web` | `corpus-only` to cite only ingested docs |
| `--seed-papers` | — | Comma-separated arXiv/DOI/S2 ids to seed the panel |
| `--model` | `google/gemini-2.5-flash` | Generation model for the draft stages |
| `--merger-model` | `anthropic/claude-opus-4.8` | Stage-2 merge model |
| `--no-rerank` | off | Keep pure RRF order (skip the cross-encoder) |
| `--top-k` | engine default | Sources retrieved per lane |
| `--out` | `.` | Output directory for `synthesis.yaml` + `graph.md` |
| `--db` | `espigue.db` | Corpus database path |

All models are OpenRouter slugs. Embeddings default to
`openai/text-embedding-3-small` (1536 dims); reranking to a Cohere model. Override
with `--embedding-model` / `--rerank-model`.

## A note on privacy

`--scope corpus-only` keeps your question and documents away from arXiv and
Semantic Scholar, but it is **not** fully offline: embedding and generation text is
still sent to OpenRouter at ingest and query time. It is private from the public
paper indexes, not from the model provider. A fully local-embedding mode is out of
scope for now.

## Requirements

- `OPENROUTER_API_KEY` (required). `S2_API_KEY` (optional).
- `pdftotext` (poppler) only if you ingest PDFs.
- No system SQLite needed — the vector store is bundled.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
