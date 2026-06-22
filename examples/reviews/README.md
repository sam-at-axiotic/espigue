# Example reviews

Five real, unedited litreview outputs. Each was generated from a single research
question with the `v3/lit-review-long` profile — no hand-editing. They are a
knowledge base on multi-agent LLM systems, each review built around the *tensions*
in its corner of the field with per-claim author-year citations and a full
bibliography.

All five ran end-to-end with **no quality degradation** (`degraded=False` in the
provenance comment at the top of each file). The metadata below comes from the
generation run.

| Review | Sources retrieved | Claims | Papers cited | Words |
| --- | ---: | ---: | ---: | ---: |
| [Roles, specialization, and division of labor](roles-specialization.md) | 271 | 21 | 18 | 7,951 |
| [Failure handling and robustness](failure-robustness.md) | 225 | 11 | 15 | 9,236 |
| [Shared memory and state management](shared-memory-state.md) | 225 | 12 | 18 | 6,112 |
| [Negotiation and conflict resolution](negotiation-conflict.md) | 193 | 15 | 19 | 6,003 |
| [Communication and coordination protocols](communication-coordination.md) | 178 | 20 | 23 | 7,746 |

Every citation and quote in these files was mechanically verified against the
retrieved source text at generation time — see [Why the output is
trustworthy](../../crates/litreview-cli/README.md#why-the-output-is-trustworthy).

To generate your own:

```bash
litreview "shared memory and state management in multi-agent LLM systems"
```
