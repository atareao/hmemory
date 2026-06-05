# hmemory — Hermes Memory Provider (Rust)

Standalone HTTP service providing persistent, RAG-backed memory for [Hermes Agent](https://hermes-agent.nousresearch.com). Uses **pgvector** on PostgreSQL for vector storage and similarity search.

## Status

Fully implemented. See `README.md` for setup, `TODO.md` for upcoming features.

## Architecture

```
Hermes Agent  ──HTTP──>  hmemory (Rust HTTP service)
                              │
                              └── pgvector (PostgreSQL + vector extension)
```

- Exposes a JSON API aligned with Hermes's `MemoryProvider` lifecycle: `initialize`, `prefetch`, `sync_turn`, `on_session_end`, `handle_tool_call`, etc.
- Hermes integration: Hermes uses a Python memory provider plugin (in `plugins/hmprovider/`) that calls hmemory's HTTP API.
- Hybrid search: pgvector cosine similarity + ParadeDB BM25, fused with alpha-weighted score.

## Developer commands

| Command | What |
|---|---|
| `cargo build` | Debug build |
| `cargo run` | Run the service |
| `cargo test` | All tests (31 passing) |
| `cargo clippy` | Lint (run before commit) |
| `cargo fmt` | Format (run before commit) |

Order: `cargo fmt && cargo clippy && cargo test && cargo build`

## Technical constraints

- **Rust edition 2024** — `unsafe` blocks in `extern` blocks are now the default (`unsafe_extern_blocks`), `mut` is disallowed on function parameters, and `impl Trait` in return position captures all lifetime params by default. `cargo fix --edition` will catch migration issues.
- **pgvector** — requires a PostgreSQL instance with the `vector` extension. CI tests should use `testcontainers` or a GitHub Actions Postgres service container with `image: pgvector/pgvector`.
- **ParadeDB** — required for BM25 hybrid search. Install the `paradedb` Postgres extension.
- **Hermes integration** — uses `plugins/hmprovider/` Python plugin. Install it into `~/.hermes/plugins/` and set `memory.provider: hmprovider`.

## Embedding providers (env-var selected)

| Env var | Values |
|---|---|
| `EMBEDDING_PROVIDER` | `openrouter` or `ollama` (required) |
| `OPENROUTER_API_KEY` | API key for OpenRouter |
| `OPENROUTER_MODEL` | Default: `openai/text-embedding-3-small` |
| `OLLAMA_BASE_URL` | Default: `http://localhost:11434` |
| `OLLAMA_MODEL` | Default: `nomic-embed-text` |

- Trait-based: `EmbeddingProvider::embed(&self, text: &str) -> Vec<f32>`
- Two implementations: `OpenRouterEmbedder` and `OllamaEmbedder`, constructed from env at startup
- OpenRouter uses its OpenAI-compatible `/v1/embeddings` endpoint

## Key dependencies

- `axum` for HTTP
- `sqlx` for async Postgres with pgvector support
- `serde` / `serde_json` for JSON API
- `reqwest` for OpenRouter / Ollama HTTP calls
- `chrono` for timestamps
- `tower-http` for CORS

## Storage

- Postgres table: `memories` with `id`, `session_id`, `content`, `embedding` (vector(1536)), `tags` (JSONB), `metadata` (JSONB), `importance` (REAL), `source` (TEXT), `created_at`, `updated_at`
- Indexes: IVFFlat on embedding, GIN on tags, BM25 on content, B-tree on session_id
- Schema migration: additive `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (no migration framework)

## Files

- `README.md` — full documentation
- `TODO.md` — upcoming features roadmap
- `plugins/hmprovider/` — Python Hermes memory provider plugin