# litreview

Topic in, cited literature review out.

`litreview` is a standalone command-line tool that turns a research question into
a structured, citation-grounded literature review. It searches arXiv and Semantic
Scholar (and your own documents), grounds every claim in retrieved sources, and
synthesises a review that surfaces the *tensions* in a field rather than a flat
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

The full review continues for several thousand words, organised as a
*problem-lattice* of competing hypotheses with per-claim author-year citations and
a complete bibliography. Every cited paper is a real source the tool retrieved and
read, not a model recollection.

## Quickstart

```bash
pip install litreview

export OPENROUTER_API_KEY=sk-or-...          # required
export S2_API_KEY=...                        # optional — adds the Semantic Scholar lane

litreview "test-time compute scaling for language models"
#   → synthesis.yaml   (the structured review + bibliography)
#   → graph.md         (the claim/tension graph)
```

That is the whole happy path. arXiv needs no key; without `S2_API_KEY` the tool
runs on arXiv plus any local documents you ingest.

### Bring your own corpus

```bash
litreview ingest ./papers/                   # index local .txt / .md / .pdf
litreview --scope corpus-only "my question"  # cite only your documents
litreview --scope corpus+web  "my question"  # your documents + arXiv/S2 (default)
```

PDF ingest needs `pdftotext` (poppler): `brew install poppler` /
`apt install poppler-utils`. Text and markdown ingest with no extra tooling.

### Seed the review with specific papers

```bash
litreview --seed-papers 2310.06825,1706.03762 "attention and efficient LLMs"
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
2. **Draft.** A test-time-diffusion engine produces parallel expert drafts grounded
   in the retrieved panel, each citing the sources it relies on.
3. **Merge.** A stronger model fuses the drafts into a single review structured
   around the field's tensions, with author-year citations and a bibliography.

Choose the depth with `--profile`: `v1`/`delphi` (concise), `v2`/`lit-review`, or
`v3`/`lit-review-long` (the long-form, tension-first format shown above).

## Why the output is trustworthy

The differentiator is the evaluation rigour behind the synthesis. Reviews are
scored by an LLM judge against a calibrated, per-dimension rubric — grounding,
citation faithfulness, tension coverage, and argument structure are graded
separately rather than rolled into one vibe score. Dead or redundant dimensions
were pruned by running the judge across many corpora and dropping the ones that did
not discriminate. The result is a generator tuned against measurable quality, not
just a prompt that sounds confident.

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
| `--db` | `litreview.db` | Corpus database path |

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
