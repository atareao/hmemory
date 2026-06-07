# hmemory

[![CI](https://github.com/atareao/hmemory/actions/workflows/ci.yml/badge.svg)](https://github.com/atareao/hmemory/actions/workflows/ci.yml)

Standalone HTTP service providing persistent, RAG-backed memory for [Hermes Agent](https://hermes-agent.nousresearch.com). Uses **pgvector** on PostgreSQL for vector storage and optional **ParadeDB** for BM25, with human-inspired 3-tier memory, batch consolidation, and active forgetting.

## Features at a glance

- **Three-tier memory** — `fresh` (24-48h working memory), `deep` (historical archive), `consolidated` (LLM-synthesized summaries)
- **Batch conversation flush** — groups coherent turns into semantic batches before storage (triggers: topic shift, idle timeout, batch size, session end)
- **Hybrid search** — vector cosine similarity + BM25, fused with alpha-weighted scoring
- **Human-like consolidation** — background worker clusters similar memories, calls OpenRouter to extract key insights, and stores as consolidated summaries
- **Reconsolidation** — deep memories with high relevance score (> 0.85) are automatically copied back to fresh (spreading activation)
- **Active forgetting** — low-importance, never-accessed memories are pruned after 90 days
- **Memory decay** — configurable half-life for temporal relevance
- **Emotional tagging** — keyword-based valence/arousal detection during consolidation
- **Events & Reminders** — store events with `event_date`, set `reminder` (relative or absolute), background worker polls every 30s and logs due reminders; new `reminder_in` field for from-now intervals
- **17 Hermes tools** — add, search, list, get, update, delete, export, import, backup, restore, compact, snapshot, graph, remind, feedback, associative_search, reminders_due
- **Memory links & graph** — create typed relations between memories, traverse connected subgraphs
- **Associative search** — find seeds then spider out via linked relations
- **Profile-based isolation** — each agent profile (charla, linuxdev, rustdev, etc.) has its own memory space; search can scope to one profile or all
- **Proactive recall** — relevant fresh + consolidated results returned by default; deep historical search opt-in
- **Bidirectional sync** — replicate writes between Hermes native `memory.md` and hmemory

## Architecture

```
Hermes Agent ──HTTP──> hmemory (Rust) ──── pgvector (± ParadeDB) (PostgreSQL)
                            │
                            ├── Embedding provider (OpenRouter / Ollama)
                            ├── Profile-based isolation
                            │
                            └── Storage tiers:
                                ├── memories_fresh    ── Working memory (24-48h TTL)
                                ├── memories_deep     ── Historical archive (append-only)
                                └── memories_consolid ── LLM summaries (shallow / daily)
                                     │
                                     ├── Consolidation worker (LLM + clustering)
                                     ├── Reconsolidation (score > 0.85 → fresh)
                                     ├── Active forgetting (pruning low-value)
                                     ├── Reminder worker (30s poll → stderr + log)
                                     ├── Memory graph (links, relations, traversal)
                                     ├── Associative search (seed + spider)
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
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model name (for OpenRouter) |

### Batch & consolidation

| Variable | Default | Description |
|---|---|---|
| `BATCH_SIZE` | `20` | Max conversation turns before batch flush |
| `BATCH_IDLE_SECONDS` | `300` | Seconds of inactivity before flush |
| `FRESH_TTL_HOURS` | `48` | Retention for fresh tier |
| `CONSOLIDATION_CADENCE` | `3600` | Seconds between consolidation cycles |
| `CONSOLIDATION_MODEL` | `openai/gpt-4o-mini` | LLM for summary generation |
| `OPENROUTER_API_KEY` | — | Required for LLM consolidation |

### Cadence

| Variable | Default | Description |
|---|---|---|
| `SYNC_TURN_MIN_IMPORTANCE` | `0.3` | Minimum importance to store a turn |
| `PREFETCH_CADENCE` | `1` | Prefetch every N turns (throttling) |
| `SYNC_TURN_CADENCE` | `1` | Sync every N turns (throttling) |
| `CONCLUSION_CADENCE` | `10` | Generate conclusions every N turns |
| `CONTEXT_TOKENS` | unlimited | Max tokens for context injection |
| `BASE_CONTEXT_CADENCE` | `5` | Refresh base context every N turns |

## Three-tier memory storage

hmemory separates memory into three tables, inspired by human memory models:

| Tier | Table | TTL | Purpose |
|---|---|---|---|
| **Fresh** | `memories_fresh` | 24-48h (configurable) | Working memory — recent conversations |
| **Deep** | `memories_deep` | Forever (append-only) | Historical archive — all conversations |
| **Consolidated** | `memories_consolid` | Forever | LLM-synthesized summaries and insights |

### How data flows

```
sync_turn (user + assistant turns)
    │
    ├── ConversationBuffer (per profile)
    │     └── flush triggers:
    │           ├── batch_size reached (default 20)
    │           ├── topic shift detected (cosim < 0.65)
    │           ├── idle timeout (5 min)
    │           └── on_session_end
    │                 │
    │                 ├──→ memories_fresh (with 24h expires_at)
    │                 └──→ memories_deep (permanent)
    │
consolidation_worker (every CONSOLIDATION_CADENCE sec):
    ├── Level 1 Shallow: groups fresh → LLM summary → memories_consolid
    ├── Level 2 Daily: groups 5+ deep items → daily summary
    ├── Emotional tagging (valence/arousal from keywords)
    └── Active forgetting: prune deep (imp<0.05, no access, >90d) + consolid (insight<0.05, >60d)

search():
    ├── Fresh (score × 1.15 boost for recency)
    ├── Consolidated (interleaved with fresh)
    └── Deep (only if include_deep: true)
          └── Reconsolidation: score > 0.85 → copy to fresh
```

## API

All endpoints accept and return JSON.

### Core lifecycle

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/initialize` | Create a new memory session for a profile |
| `POST` | `/prefetch` | Search memories by semantic similarity + BM25 |
| `POST` | `/sync_turn` | Buffer a conversation turn (triggers batch flush) |
| `POST` | `/on_session_end` | Force-flush buffer and end session |

### Reminders

| Method | Path | Description |
|---|---|---|
| `GET` | `/reminders/due` | Get all due (unsent) reminders |
| `POST` | `/remind` | Set a reminder on an existing memory |

#### Set a reminder on an existing memory

```bash
curl -s -X POST http://localhost:8080/remind \
  -H 'Content-Type: application/json' \
  -d '{
    "memory_id": 1,
    "reminder_interval": "30m",
    "event_date": "2026-06-10T14:00:00Z"
  }'
# → {"ok": true}
```

#### Remind in N hours (from-now)

```bash
curl -s -X POST http://localhost:8080/remind \
  -H 'Content-Type: application/json' \
  -d '{"memory_id": 42, "reminder_in": "2h"}'
# → {"ok": true}
```

#### Prefetch with deep search

```bash
curl -s -X POST http://localhost:8080/prefetch \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "deployment strategy",
    "profile": "linuxdev",
    "limit": 5,
    "level": "overview",
    "include_deep": true
  }'
```

### Tool dispatch

| Method | Path | Description |
|---|---|---|
| `GET` | `/tool_schemas` | JSON Schema for all 17 Hermes tools |
| `POST` | `/handle_tool_call` | Dispatch any memory tool |

Supported tools: `memory_search`, `memory_add`, `memory_list`, `memory_get`, `memory_update`, `memory_delete`, `memory_export`, `memory_import`, `memory_backup`, `memory_restore`, `memory_compact`, `memory_snapshot`, `memory_graph`, `memory_associative_search`, `memory_reminders_due`, `memory_feedback`, `memory_remind`.

#### Add a memory with reminder_in (from-now)

```bash
curl -s -X POST http://localhost:8080/handle_tool_call \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "memory_add",
    "args": {
      "content": "Revisar el build en 2 horas",
      "reminder_in": "2h"
    },
    "profile": "rustdev"
  }'
# → {"ok": true}
```

#### Search across all profiles

```bash
curl -s -X POST http://localhost:8080/handle_tool_call \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "memory_search",
    "args": {
      "query": "deployment",
      "limit": 5,
      "profile": "",
      "include_deep": true
    },
    "profile": "rustdev"
  }'
# → {"ok": true, "memories": [...]}
```

### Memory CRUD

| Method | Path | Description |
|---|---|---|
| `POST` | `/memories/list` | List memories with pagination (per-profile or global) |
| `POST` | `/memories/get` | Get a single memory by ID |
| `POST` | `/memories/update` | Update a memory (partial) |
| `POST` | `/memories/feedback` | Submit relevance feedback |
| `POST` | `/delete` | Delete a memory by ID |

#### List memories for all profiles

```bash
curl -s -X POST http://localhost:8080/memories/list \
  -H 'Content-Type: application/json' \
  -d '{"limit": 10, "offset": 0}'
# → {"ok": true, "memories": [...], "total": 42}
```

#### List memories for a specific profile

```bash
curl -s -X POST http://localhost:8080/memories/list \
  -H 'Content-Type: application/json' \
  -d '{"limit": 10, "offset": 0, "profile": "linuxdev"}'
```

#### Get by ID

```bash
curl -s -X POST http://localhost:8080/memories/get \
  -H 'Content-Type: application/json' \
  -d '{"id": 1}'
```

### Profile management

| Method | Path | Description |
|---|---|---|
| `POST` | `/profile/status` | Get profile status (turn count, cold/warm) |
| `POST` | `/session/conclude` | Create a conclusion/insight memory |

### Backup & import

| Method | Path | Description |
|---|---|---|
| `POST` | `/export` | Export all memories for a profile (or all if omitted) |
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

### Search routing (multi-table)

By default, search queries **fresh** (with a 1.15x recency boost) and **consolidated** tables, interleaving results. Deep historical archive is only queried when `include_deep: true`. This mimics human memory: recent + important insights are most accessible; old memories require explicit effort to recall.

Scores also decay over time — `DECAY_HALF_LIFE_DAYS` adjusts how fast old memories fade. A memory created `N` half-lives ago has its vec_score multiplied by `0.5^N`.

When ParadeDB is not installed, hmemory uses vector-only search without error. Hybrid mode activates automatically when ParadeDB is detected.

## Embedded fields

| Field | Type | Description |
|---|---|---|
| `tags` | JSONB | Key-value metadata for filtering |
| `metadata` | JSONB | Arbitrary metadata (conclusions store confidence, depth info) |
| `importance` | REAL | Importance score (0.0–1.0) |
| `source` | TEXT | Origin of the memory |
| `category` | TEXT | Category (e.g. `chat`, `insight`, `fact`, `preference`) |
| `feedback_positive` | INTEGER | Positive relevance feedback count |
| `feedback_negative` | INTEGER | Negative relevance feedback count |
| `created_at` | TIMESTAMPTZ | Creation timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |
| `event_date` | TIMESTAMPTZ | Scheduled event/appointment datetime |
| `reminder_interval` | TEXT | Relative interval (`30m`, `1h`, `2d`) or absolute RFC3339 |
| `reminder_at` | TIMESTAMPTZ | Computed reminder trigger time |
| `reminder_sent` | BOOLEAN | Whether the background worker has fired this reminder |
| `expires_at` | TIMESTAMPTZ | TTL for fresh tier memories |

### Consolidated-specific fields

| Field | Type | Description |
|---|---|---|
| `summary` | TEXT | LLM-generated insight (replaces `content`) |
| `source_ids` | BIGINT[] | References to the deep memories that generated this summary |
| `depth` | TEXT | Consolidation level: `shallow` or `daily` |
| `insight_score` | REAL | Quality of the extracted insight (0.0–1.0) |

### Fresh-specific fields

| Field | Type | Description |
|---|---|---|
| `conversation_id` | TEXT | UUID identifying the original conversation |
| `turn_range` | TEXT | Range of turn numbers in the batch (e.g. "1..20") |
| `user_msg` | TEXT | Raw user message from the turn |
| `assistant_msg` | TEXT | Raw assistant response |

## Tiered context loading

Search results are tiered to save tokens in LLM context:

| Level | Name | Max results | Use case |
|---|---|---|---|
| `summary` | L0 | 1 | Quick topic summary (LLM sees first sentence) |
| `overview` | L1 | 3 | Broad context retrieval (LLM sees bullet list) |
| `details` | L2 | full | Full deep retrieval (LLM sees full JSON) |

The `level` parameter on `/prefetch` and the plugin's tools controls which tier to use.

## Batch conversation flush

Instead of storing every individual turn, hmemory buffers turns per-profile in a `ConversationBuffer` and flushes when any trigger fires:

| Trigger | Condition | Effect |
|---|---|---|
| `batch_size` | 20 turns accumulated | Flush immediately |
| Topic shift | Cosine similarity < 0.65 between current and accumulated embedding | Flush → new batch |
| Idle timeout | 5 minutes without new turn | Flush (conversation paused) |
| `on_session_end` | Session explicitly ended | Flush + remove buffer |

Each flush produces a single row in both `memories_fresh` and `memories_deep` with the concatenated content and a coherent combined embedding.

## Consolidation worker

The consolidation worker runs every `CONSOLIDATION_CADENCE` seconds and performs:

### Level 1 — Shallow consolidation

1. Reads expired fresh memories (older than `FRESH_TTL_HOURS`)
2. Groups by content prefix (topic coherence)
3. Calls OpenRouter `gpt-4o-mini` to extract key insights
4. Stores the summary in `memories_consolid` with `depth: "shallow"`
5. Attaches emotional valence/arousal tags via keyword detection

### Level 2 — Daily consolidation

1. Groups new deep items by profile (minimum 5)
2. Calls LLM for a higher-level daily summary
3. Stores with `depth: "daily"`

### Active forgetting

At the end of each cycle:
- `prune_deep()`: deletes deep memories with `importance < 0.05`, zero accesses, and older than 90 days (skips `immortal = true`)
- `prune_consolid()`: deletes consolidated memories with `insight_score < 0.05` and older than 60 days

### Reconsolidation

When a search query includes `include_deep: true` and a deep result has `score > 0.85`, the memory is automatically copied to `memories_fresh` with a fresh 24h TTL (spreading activation — recalled memories return to working memory).

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

### Schema migration (session_id → profile)

If you have an existing database with the old `session_id` column, hmemory automatically runs:

```sql
ALTER TABLE memories RENAME TO memories_deep;
ALTER TABLE memories_deep RENAME COLUMN session_id TO profile;
```

This runs at startup. You can also run the manual script:

```bash
psql -d hmemory -f scripts/migrate_session_to_profile.sql
```

## Development

```bash
cargo fmt && cargo clippy && cargo test && cargo build
```

Requires Rust edition 2024 and a Postgres instance with the `vector` extension. ParadeDB is optional.

### Test status

- **52 unit tests** (handler mocks + vector fusion + config + compute_reminder_at + reminders + tags + consolidation helpers)
- No DB required for tests (mocks cover all endpoints)
- Integration tests require a running Postgres

## File structure

```
src/
├── main.rs                    # Entry point, config, spawns reminder + consolidation workers
├── config.rs                  # Env-var + YAML config struct
├── reminder_worker.rs         # Background poll loop (30s) for due reminders
├── consolidation_worker.rs    # Background consolidation (LLM summarization, pruning, emotional tagging)
├── api/
│   ├── mod.rs                 # Router setup (22+ routes)
│   ├── state.rs               # AppState (embedder, store, buffer, cadence)
│   └── handlers.rs            # HTTP handlers + tests (52 tests)
├── embeddings/
│   ├── mod.rs                 # EmbeddingProvider trait
│   ├── openrouter.rs          # OpenRouter embedder
│   └── ollama.rs              # Ollama embedder
└── storage/
    ├── mod.rs                 # MemoryStore trait, 3 MemoryRecord types, SearchFilters
    └── pgvector.rs            # PgVectorStore + hybrid search + 3-tier tables + migrations

plugins/
└── hmprovider/                # Hermes Python plugin (17 tools, batch config, tiered context, reminders)

scripts/                       # Migration scripts
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `column "profile" does not exist` | Old database with `session_id` | Restart hmemory — auto-migration runs at startup |
| `column "category" does not exist` | Old database without migration | Restart hmemory — auto-migration runs at startup |
| `database "hmemory" does not exist` | Database not created | Restart hmemory — auto-create runs at startup |
| Search returns empty | Missing tag clause for `"null"::jsonb` | Upgrade hmemory (fixed) |
| ParadeDB features not working | ParadeDB not installed | Use pgvector-only (fallback is automatic) |
| Consolidation worker not running | Missing `OPENROUTER_API_KEY` | Set env var or YAML config |