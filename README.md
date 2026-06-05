# hmemory

[![CI](https://github.com/atareao/hmemory/actions/workflows/ci.yml/badge.svg)](https://github.com/atareao/hmemory/actions/workflows/ci.yml)

Standalone HTTP service providing persistent, RAG-backed memory for [Hermes Agent](https://hermes-agent.nousresearch.com). Uses **pgvector** on PostgreSQL for vector storage and optional **ParadeDB** for BM25 full-text search, with hybrid search fusion.

## Features at a glance

- **Hybrid search** — vector cosine similarity + BM25, fused with alpha-weighted scoring
- **Memory decay** — configurable half-life for temporal relevance
- **8 Hermes tools** — add, search, list, get, update, delete, backup, restore
- **Session strategy** — per-session, per-directory, per-repo, or global
- **Cold/warm detection** — broader search when session is cold (< 3 turns or 1h inactivity)
- **Two-layer context** — base profile (high-importance facts) + query results
- **Tiered loading** — L0 summary (1 result), L1 overview (3), L2 details (full)
- **Conservative sync_turn** — skips chit-chat via importance heuristics
- **Conclusions/insights** — periodic insight generation stored as memories
- **Cadence throttling** — env-var controlled rate for prefetch, sync, conclusions
- **Token budget** — truncate context injection to a configurable limit
- **Reranking** — optional cross-encoder reranking step
- **Backup/restore** — full JSON export with embeddings
- **OpenAPI spec** — served at `GET /openapi.json`
- **Migration scripts** — from holographic memory (SQLite)

## Architecture

```
Hermes Agent ──HTTP──> hmemory (Rust) ──── pgvector (± ParadeDB) (PostgreSQL)
                           │
                           ├── OpenRouter / Ollama (embedding provider)
                           ├── Session management (cold/warm, strategy)
                           ├── Two-layer context (base profile + search)
                           ├── Token budget + cadence throttling
                           ├── Conclusion/insight generation
                           └── Backup/restore (full JSON)
```

hmemory maps Hermes Agent's `MemoryProvider` lifecycle to HTTP endpoints. A [Python plugin](plugins/hmprovider/) bridges the two. ParadeDB is optional — hmemory falls back to vector-only search when ParadeDB is not installed.

## Quickstart

### 1. Start Postgres

```bash
docker run -d --name hmem-db \
  -e POSTGRES_PASSWORD=hmemory \
  -e POSTGRES_DB=hmemory \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

> ParadeDB is optional for BM25 hybrid search. See [ParadeDB docs](https://docs.paradedb.com) if you want it.

### 2. Start hmemory

```bash
export EMBEDDING_PROVIDER=ollama
export DATABASE_URL=postgres://postgres:hmemory@localhost:5432/hmemory

cargo run
```

Or with Docker Compose:

```bash
docker compose up -d
```

The database is auto-created on first startup — no manual `CREATE DATABASE` needed.

### 3. Configure Hermes

```bash
cp -r plugins/hmprovider ~/.hermes/plugins/hmprovider
hermes config set memory.provider hmprovider
hermes memory setup  # sets HMEMORY_BASE_URL interactively
```

## Environment variables

### Required

| Variable | Values | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openrouter` or `ollama` | Which embedding backend to use |

### Connection

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | HTTP listen port |
| `DATABASE_URL` | `postgres://localhost:5432/hmemory` | Postgres connection string |

### Embedding provider

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required when `EMBEDDING_PROVIDER=openrouter` |
| `OPENROUTER_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Required when `EMBEDDING_PROVIDER=ollama` |
| `OLLAMA_MODEL` | `nomic-embed-text` | Embedding model |

### Search & retrieval

| Variable | Default | Description |
|---|---|---|
| `HYBRID_ALPHA` | `0.5` | BM25 vs vector weighting (`0.0` = pure BM25, `1.0` = pure vector) |
| `DECAY_HALF_LIFE_DAYS` | `30` | Memory decay half-life in days (`0` = no decay) |
| `RERANK_ENABLED` | `false` | Enable cross-encoder reranking |
| `RERANK_MODEL` | — | Reranker model name (for OpenRouter) |

### Session & cadence

| Variable | Default | Description |
|---|---|---|
| `SESSION_STRATEGY` | `per-session` | Session resolution: `per-session`, `per-directory`, `per-repo`, `global` |
| `SYNC_TURN_MIN_IMPORTANCE` | `0.3` | Minimum importance to store a turn |
| `PREFETCH_CADENCE` | `1` | Prefetch every N turns (throttling) |
| `SYNC_TURN_CADENCE` | `1` | Sync every N turns (throttling) |
| `CONCLUSION_CADENCE` | `10` | Generate conclusions every N turns |
| `CONTEXT_TOKENS` | unlimited | Max tokens for context injection |
| `BASE_CONTEXT_CADENCE` | `5` | Refresh base context every N turns |

## API

All endpoints accept and return JSON.

### Core lifecycle

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/initialize` | Create a new memory session |
| `POST` | `/prefetch` | Search memories by semantic similarity + BM25 |
| `POST` | `/sync_turn` | Store a conversation turn (conservative: skips low-importance) |
| `POST` | `/on_session_end` | Signal session end (no-op) |

#### Example: prefetch

```bash
curl -s -X POST http://localhost:8080/prefetch \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "deployment strategy",
    "session_id": "session-1",
    "limit": 5,
    "level": "overview",
    "tag_filter": {"project": "infra"},
    "min_importance": 0.3
  }'
```

### Tool dispatch

| Method | Path | Description |
|---|---|---|
| `GET` | `/tool_schemas` | JSON Schema for all Hermes tools |
| `POST` | `/handle_tool_call` | Dispatch any memory tool |

Supported tools: `memory_search`, `memory_add`, `memory_list`, `memory_get`, `memory_update`, `memory_delete`, `memory_export`, `memory_import`, `memory_backup`, `memory_restore`.

#### Example: add a memory

```bash
curl -s -X POST http://localhost:8080/handle_tool_call \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "memory_add",
    "args": {"content": "The deployment uses Docker Compose", "tags": {"project": "infra"}},
    "session_id": "session-1"
  }'
# → {"ok": true}
```

#### Example: search

```bash
curl -s -X POST http://localhost:8080/handle_tool_call \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "memory_search",
    "args": {"query": "deployment", "limit": 5},
    "session_id": "session-1"
  }'
# → {"ok": true, "memories": [...]}
```

### Memory CRUD

| Method | Path | Description |
|---|---|---|
| `POST` | `/memories/list` | List memories with pagination (per-session or global) |
| `POST` | `/memories/get` | Get a single memory by ID |
| `POST` | `/memories/update` | Update a memory (partial) |
| `POST` | `/memories/feedback` | Submit relevance feedback |
| `POST` | `/delete` | Delete a memory by ID |

#### Example: list memories

```bash
curl -s -X POST http://localhost:8080/memories/list \
  -H 'Content-Type: application/json' \
  -d '{"limit": 10, "offset": 0}'
# → {"ok": true, "memories": [...], "total": 42}
```

#### Example: get by ID

```bash
curl -s -X POST http://localhost:8080/memories/get \
  -H 'Content-Type: application/json' \
  -d '{"id": 1}'
```

### Session management

| Method | Path | Description |
|---|---|---|
| `POST` | `/session/resolve` | Resolve/create session with strategy |
| `POST` | `/session/status` | Get session status (turn count, cold/warm) |
| `POST` | `/session/conclude` | Create a conclusion/insight memory |

#### Example: resolve session with strategy

```bash
curl -s -X POST http://localhost:8080/session/resolve \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "my-project",
    "strategy": "per-repo",
    "path_or_repo": "github.com/user/my-project"
  }'
```

#### Example: check if session is cold

```bash
curl -s -X POST http://localhost:8080/session/status \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "session-1"}'
# → {"ok": true, "turn_count": 5, "is_cold": false, ...}
```

### Backup & import

| Method | Path | Description |
|---|---|---|
| `POST` | `/export` | Export all memories for a session |
| `POST` | `/import` | Bulk-import memories (with re-embedding) |
| `POST` | `/backup` | Full backup of all memories (includes embeddings) |
| `POST` | `/restore` | Restore from backup (embeddings preserved) |

```bash
# Full backup
curl -s -X POST http://localhost:8080/backup -d '{}' > backup.json

# Restore
curl -s -X POST http://localhost:8080/restore \
  -H 'Content-Type: application/json' \
  -d @backup.json
```

### Specification

| Method | Path | Description |
|---|---|---|
| `GET` | `/openapi.json` | OpenAPI 3.0 specification |

## Hybrid Search

hmemory runs **vector cosine similarity** (pgvector) and optional **BM25 full-text search** (ParadeDB) in parallel, then fuses results in Rust using a weighted sum:

```
score = alpha * vec_score + (1 - alpha) * bm25_score
```

Scores also decay over time — `DECAY_HALF_LIFE_DAYS` adjusts how fast old memories fade. A memory created `N` half-lives ago has its vec_score multiplied by `0.5^N`.

When ParadeDB is not installed, hmemory uses vector-only search without error. Hybrid mode activates automatically when ParadeDB is detected.

## Embedded fields

| Field | Type | Description |
|---|---|---|
| `tags` | JSONB | Key-value metadata for filtering |
| `metadata` | JSONB | Arbitrary metadata (conclusions store confidence here) |
| `importance` | REAL | Importance score (0.0–1.0) |
| `source` | TEXT | Origin of the memory |
| `category` | TEXT | Category (e.g. `chat`, `insight`, `fact`, `preference`) |
| `feedback_positive` | INTEGER | Positive relevance feedback count |
| `feedback_negative` | INTEGER | Negative relevance feedback count |
| `created_at` | TIMESTAMPTZ | Creation timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |

## Tiered context loading

Search results are tiered to save tokens in LLM context:

| Level | Name | Max results | Use case |
|---|---|---|---|
| `summary` | L0 | 1 | Quick topic summary (LLM sees first sentence) |
| `overview` | L1 | 3 | Broad context retrieval (LLM sees bullet list) |
| `details` | L2 | full | Full deep retrieval (LLM sees full JSON) |

The `level` parameter on `/prefetch` and the plugin's `hmemory_search` tool controls which tier to use. The plugin formats output accordingly.

## Session strategy

The `SESSION_STRATEGY` env var controls how session IDs are resolved:

| Strategy | Use case |
|---|---|
| `per-session` | Default — one session per Hermes conversation |
| `per-directory` | One session per working directory (persists across chats in same project) |
| `per-repo` | One session per git remote (persists across directories) |
| `global` | Single session — all conversations share one memory space |

When Hermes starts a new chat, the plugin calls `POST /session/resolve` with the strategy and current working directory / git remote to determine the correct session ID.

## Honcho-inspired features

hmemory includes several features inspired by [Honcho](https://github.com/grill/honcho):

- **Cold/warm detection** — cold sessions (< 3 turns or inactive > 1h) get broader search
- **Two-layer context** — base profile (high-importance facts) + query results
- **Conclusions** — insights derived from conversation patterns, stored as memories with `category: "insight"`
- **Cadence throttling** — env-var controlled cadence for prefetch, sync_turn, and conclusions
- **Token budget** — truncate context injection to `CONTEXT_TOKENS` limit
- **Base context** — aggregated high-importance memories served alongside search results

## Conservative sync_turn

`sync_turn` computes an importance score for each turn using heuristics:

| Factor | Effect |
|---|---|
| Greeting/one-word replies | −0.15 penalty |
| Code blocks (```) | +0.15 boost |
| Technical keywords (api, deploy, config…) | +0.05 each |
| Long messages (>300 chars) | +0.10 boost |
| Numeric content (>5 digits) | +0.10 boost |

Storage is skipped when `importance < SYNC_TURN_MIN_IMPORTANCE` (default `0.3`). This prevents casual chit-chat from polluting the semantic space.

## Embedding providers

### OpenRouter

```bash
export EMBEDDING_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-...
export OPENROUTER_MODEL=openai/text-embedding-3-small
```

Uses OpenAI-compatible `/v1/embeddings` endpoint. Supports any embeddings model available on OpenRouter.

### Ollama

```bash
export EMBEDDING_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=nomic-embed-text
```

Runs locally. Any Ollama-compatible embedding model works.

## Migration

See [`MIGRATION.md`](MIGRATION.md) for migrating from:

- **Holographic memory** — automatic script (`scripts/migrate_from_holographic.py`)
- **Mem0** — export via API, transform to hmemory format

## Development

```bash
cargo fmt && cargo clippy && cargo test && cargo build
```

Requires Rust edition 2024 and a Postgres instance with the `vector` extension. ParadeDB is optional.

### Test status

- **38 unit tests** (handler mocks + vector fusion + config)
- No DB required for tests (mocks cover all endpoints)
- Integration tests require a running Postgres

## File structure

```
src/
├── main.rs              # Entry point, config loading
├── config.rs            # Env-var config struct (all 17 vars)
├── api/
│   ├── mod.rs           # Router setup (20+ routes)
│   ├── state.rs         # AppState (embedder, store, strategy, cadence)
│   └── handlers.rs      # HTTP handlers + tests (38 tests)
├── embeddings/
│   ├── mod.rs           # EmbeddingProvider trait
│   ├── openrouter.rs    # OpenRouter embedder
│   └── ollama.rs        # Ollama embedder
└── storage/
    ├── mod.rs           # MemoryStore trait, MemoryRecord, SearchFilters, SessionStatus
    └── pgvector.rs      # PgVectorStore + hybrid search + sessions + conclusions

plugins/
└── hmprovider/          # Hermes Python plugin (cadence, token budget, tiered context)

.github/workflows/       # CI workflow (fmt + clippy + test + build)
openapi.json             # OpenAPI 3.0 spec
MIGRATION.md             # Migration guides
scripts/                 # Migration scripts
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `column "category" does not exist` | Old database without migration | Restart hmemory — auto-migration runs at startup |
| `operator does not exist: timestamp with time zone >= text` | Filter parameter type mismatch | Upgrade hmemory (fixed in current version) |
| `database "hmemory" does not exist` | Database not created | Restart hmemory — auto-create runs at startup |
| Search returns empty | Missing tag clause for `"null"::jsonb` | Upgrade hmemory (fixed in current version) |
| ParadeDB features not working | ParadeDB not installed | Use pgvector-only (fallback is automatic) |