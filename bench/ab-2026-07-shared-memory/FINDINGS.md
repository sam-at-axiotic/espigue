# A/B bench: shared memory and state management in multi-agent LLM systems

Live probes of the July 2026 prompt-system fixes, run 2026-07-02 to 2026-07-03.
Baseline is `examples/reviews/shared-memory-state.*` (generated 2026-06-17,
code 98617d5, healthy retrieval, 18-paper full-text panel).

## Setup

- Question: `shared memory and state management in multi-agent LLM systems`
  (the exact string documented in `examples/reviews/README.md`).
- Command: `espigue "<question>" --model anthropic/claude-haiku-4.5` — drafts
  on Haiku 4.5, merger on Opus 4.8 (both verified current on OpenRouter,
  2026-07-03). Profile `v3/lit-review-long`, fresh DB per run.
- Compare with `python3 compare.py <A>.synthesis.yaml <B>.synthesis.yaml`.

## Runs

| | baseline (Jun 17) | run 1 | run 2 | run 3 |
| --- | --- | --- | --- | --- |
| code | 98617d5 | 1e0b928..624ba46 | + bb1a17c (example-bleed fix) | + 984f034 (lane retries) |
| arXiv lane | healthy | dead (transport) | dead (transport) | healthy |
| S2 lane | healthy | 429, dead | partial | healthy — 13+ backoffs absorbed |
| topicality kept | n/a | 2/10 | 4/10 | 7/10 |
| verified quotes | 16/16 | 10/10 | 15/15 | 10/10 |
| distinct sources cited | 18 | 16 | 12 | 6 |
| multi-source claims | 6 | 5 | 5 | 0 |
| plan winner valid | n/a | no | yes | no |
| narrative words | 6,112 | 6,179 | 5,473 | 6,945 |

## What the runs established

1. **The verification floor held everywhere.** 100% of quotes mechanically
   verified in all four documents, across every prompt change.
2. **Question grip** (fix 1e0b928): runs 1-3 all open on the question's own
   tension; before the fix the engine never saw the question at all.
3. **Example-bleed fix confirmed on its symptom** (bb1a17c): run 1 grew
   Byzantine-fault-tolerance and classical-control sections from the prompts'
   own worked examples; runs 2-3 stay in the question's field.
4. **Lane retries confirmed live** (984f034): run 3's log shows the backoff
   ladder absorbing S2 429s and truncated-body decodes; first run with both
   live lanes `degraded=false`.

## Open directions (next session)

1. **Source-breadth collapse.** Run 3 had the healthiest retrieval of the
   three (7-paper gated panel, 21-paper stage-2 panel, 172 bibliography rows)
   yet the final claim set cites only 6 distinct sources and NO claim carries
   two independent sources — worst of any run, on a pipeline whose guards
   preach independence over vote-counting. Breadth reaches the corpus but does
   not survive drafting/merging into claims. Suspect the draft/merger
   selection dynamics, not retrieval. Start at `render_synthesis_merger_v2`'s
   union-of-sources rule versus what the Haiku drafts actually cite.
2. **Plan-draft parse reliability on Haiku.** Tournament won with an invalid
   winner in runs 1 and 3 (`calls_used=7` of ~28 — most drafts fail
   `parse_review_plan`). Either harden the plan XML contract for Haiku or
   route the 5 plan drafts to a stronger model (cheap: small calls).
3. **Run-to-run variance.** Runs 2 and 3 differ widely in scope/framing under
   thin panels. `--top-k` above the default 10 gives every downstream stage
   more to hold onto; untested.
4. **S2 429s persist even keyed** (shared-pool behaviour). Retries absorb
   them now; a disk cache for the fusion-lane S2 calls (explore already has
   one) would cut pressure further.

## Environment notes

- `.env` at repo root (gitignored): `OPENROUTER_API_KEY`, `S2_API_KEY`,
  `JINA_API_KEY`. No crate loads dotenv — `set -a && source .env` before
  running.
- A run costs ~40 min wall-clock (Haiku swarm + one Opus merge).
- Full run logs were session-local and are gone; the essentials are quoted
  above and in the commit messages.
