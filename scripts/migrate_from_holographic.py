#!/usr/bin/env python3
"""
Migrate from holographic memory (SQLite) to hmemory (HTTP API).

Usage:
    python scripts/migrate_from_holographic.py [--db PATH] [--base-url URL] [--session SESSION]

Defaults:
    --db        ~/.hermes/memory/holographic/memories.db
    --base-url  http://localhost:8080
    --session   holographic-import
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request


def load_sqlite(db_path: str) -> list[dict]:
    if not os.path.exists(db_path):
        print(f"error: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row["name"] for row in cursor.fetchall()]

    memories = []
    for table in tables:
        try:
            cur = conn.execute(f"SELECT * FROM [{table}]")
            columns = [desc[0] for desc in cur.description]
            for row in cur.fetchall():
                record = dict(row)
                record["_source_table"] = table
                record["_hmemory_category"] = table.rstrip("s")
                memories.append(record)
        except sqlite3.Error as e:
            print(f"warning: skipping table {table}: {e}", file=sys.stderr)
    conn.close()
    return memories


def map_record(r: dict) -> dict:
    content = r.get("content") or r.get("text") or r.get("memory") or json.dumps(r)
    tags = {}
    for key in ("tags", "metadata", "category", "type"):
        val = r.get(key)
        if val:
            if isinstance(val, str):
                try:
                    tags[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    tags[key] = val
            else:
                tags[key] = val
    return {
        "content": content,
        "tags": tags,
        "importance": float(r.get("importance", r.get("weight", 0.5))),
        "source": r.get("source", "holographic"),
        "category": r.get("_hmemory_category", "general"),
    }


def send_hmemory(session_id: str, base_url: str, records: list[dict]) -> int:
    url = f"{base_url.rstrip('/')}/import"
    body = json.dumps({"session_id": session_id, "memories": records}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result.get("imported", 0)
    except urllib.error.HTTPError as e:
        print(f"error: HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate from holographic memory to hmemory"
    )
    parser.add_argument(
        "--db",
        default=os.path.expanduser("~/.hermes/memory/holographic/memories.db"),
        help="Path to holographic SQLite database",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="hmemory HTTP base URL",
    )
    parser.add_argument(
        "--session",
        default="holographic-import",
        help="Session ID to store migrated memories under",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50,
        help="Batch size (default 50)",
    )
    args = parser.parse_args()

    print(f"Reading holographic database: {args.db}")
    rows = load_sqlite(args.db)
    print(
        f"Found {len(rows)} records in {len(set(r['_source_table'] for r in rows))} table(s)"
    )

    records = [map_record(r) for r in rows]
    total_imported = 0
    for i in range(0, len(records), args.batch):
        batch = records[i : i + args.batch]
        imported = send_hmemory(args.session, args.base_url, batch)
        total_imported += imported
        print(f"  batch {i // args.batch + 1}: {imported}/{len(batch)} imported")
        time.sleep(0.1)

    print(
        f"\nDone. {total_imported}/{len(records)} memories migrated to session '{args.session}'."
    )


if __name__ == "__main__":
    main()
