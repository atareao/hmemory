# Migration Guide

## From holographic memory (SQLite)

[holographic memory](https://github.com/nousresearch/hermes-agent) stores memories in a local SQLite database at `~/.hermes/memory/holographic/memories.db`.

### Steps

```bash
# 1. Install hmemory
git clone https://github.com/atareao/hmemory
cd hmemory
docker compose up -d

# 2. Run migration script
python scripts/migrate_from_holographic.py

# 3. Point Hermes to hmemory instead of holographic
hermes config set memory.provider hmprovider
```

The script reads all tables from the SQLite database, maps records to hmemory's format, and batches them via the `POST /import` API. By default it stores under session `holographic-import`.

### Options

```bash
# Custom paths
python scripts/migrate_from_holographic.py \
    --db /custom/path/memories.db \
    --base-url http://hmemory:8080 \
    --session my-migration \
    --batch 100
```

---

## From Mem0

[Mem0](https://mem0.ai) provides a managed memory API. No local database to read — you export via their API.

### Steps

```bash
# 1. Export from Mem0
curl -H "Authorization: Bearer $MEM0_API_KEY" \
     https://api.mem0.ai/v1/memories > mem0_export.json

# 2. Transform to hmemory format
python3 -c "
import json
with open('mem0_export.json') as f:
    data = json.load(f)

memories = []
for item in data.get('results', data.get('memories', [])):
    content = item.get('content', item.get('text', ''))
    if not content:
        continue
    memories.append({
        'content': content,
        'tags': item.get('metadata', {}),
        'importance': float(item.get('score', item.get('importance', 0.5))),
        'source': 'mem0',
        'category': item.get('category', 'general'),
    })

print(json.dumps({'session_id': 'mem0-import', 'memories': memories}))
" > hmemory_import.json

# 3. Import into hmemory
curl -X POST http://localhost:8080/import \
     -H 'Content-Type: application/json' \
     -d @hmemory_import.json
```

---

## Switching Hermes from holographic to hmemory

```bash
# 1. Backup existing holographic DB
cp ~/.hermes/memory/holographic/memories.db ~/holographic_backup.db

# 2. Install hmemory plugin
cp -r plugins/hmprovider ~/.hermes/plugins/hmprovider

# 3. Configure Hermes
hermes config set memory.provider hmprovider

# 4. (Optional) Set env vars in your shell or .env
export HMEMORY_BASE_URL=http://localhost:8080
export HMEMORY_SESSION_STRATEGY=per-repo
```

Hermes will call hmemory on the next session start. Existing holographic memories can be imported using the migration script above.