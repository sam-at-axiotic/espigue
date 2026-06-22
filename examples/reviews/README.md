# Example reviews

Five real, unedited espigue outputs. Each was generated from a single research
question with the `v3/lit-review-long` profile — no hand-editing. They are a
knowledge base on multi-agent LLM systems, each review built around the *tensions*
in its corner of the field with per-claim author-year citations and a full
bibliography.

All five ran end-to-end with **no quality degradation** (`degraded=False` in the
provenance comment at the top of each file). The metadata below comes from the
generation run.

Each review ships in two forms: the rendered prose (`.md`) and the full
structured output (`.synthesis.yaml`) — the same review as machine-readable data,
with every claim graded and quoted and the field's agreements, disagreements,
uncertainties, and open gaps as first-class lists. See [The structured
output](../../crates/espigue/README.md#the-structured-output) for the full
schema.

| Review | Structured | Sources | Claims | Papers | Words |
| --- | --- | ---: | ---: | ---: | ---: |
| [Roles, specialization, and division of labor](roles-specialization.md) | [yaml](roles-specialization.synthesis.yaml) | 271 | 21 | 18 | 7,951 |
| [Failure handling and robustness](failure-robustness.md) | [yaml](failure-robustness.synthesis.yaml) | 225 | 11 | 15 | 9,236 |
| [Shared memory and state management](shared-memory-state.md) | [yaml](shared-memory-state.synthesis.yaml) | 225 | 12 | 18 | 6,112 |
| [Negotiation and conflict resolution](negotiation-conflict.md) | [yaml](negotiation-conflict.synthesis.yaml) | 193 | 15 | 19 | 6,003 |
| [Communication and coordination protocols](communication-coordination.md) | [yaml](communication-coordination.synthesis.yaml) | 178 | 20 | 23 | 7,746 |

Every citation and quote in these files was mechanically verified against the
retrieved source text at generation time — see [Why the output is
trustworthy](../../crates/espigue/README.md#why-the-output-is-trustworthy).

To generate your own:

```bash
espigue "shared memory and state management in multi-agent LLM systems"
```
