# litreview

Topic in, cited literature review out.

`litreview` is a standalone command-line tool that turns a research question into
a structured, citation-grounded literature review. It searches arXiv and Semantic
Scholar (and your own documents), grounds every claim in retrieved sources, and
synthesises a review built around the *tensions* in a field. One
`OPENROUTER_API_KEY` drives generation, embeddings, and reranking.

```bash
pip install litreview
export OPENROUTER_API_KEY=sk-or-...
litreview "test-time compute scaling for language models"
#   → synthesis.yaml + graph.md
```

Full documentation, a real generated review, and the design of the three-stage
pipeline are in the package README: **[crates/litreview-cli/README.md](crates/litreview-cli/README.md)**.

## Repository layout

This is a self-contained Rust workspace. The CLI is packaged as a pip-installable
wheel via maturin (`bindings = "bin"`).

| Crate | Role |
| --- | --- |
| `crates/litreview-cli` | The `litreview` binary + the synthesis pipeline (`run_review`) |
| `crates/alzina-orchestration` | Test-time-diffusion synthesis engine (TTD + adapter) |
| `crates/alzina-search` | Literature search — RRF fusion, reranking, the vec store |
| `crates/alzina-core` | Shared types, error taxonomy, trait definitions |

## Build from source

```bash
cargo build --release -p litreview-cli      # → target/release/litreview
# or build the wheel:
pip install maturin
cd crates/litreview-cli && maturin build --release
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).
