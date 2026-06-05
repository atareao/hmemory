import json
import os
import threading
import logging
from pathlib import Path
import requests
from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


class HMemoryProvider(MemoryProvider):
    def __init__(self):
        self._turn_count = 0
        self._conclusion_turn = 0
        self._last_resolved_session = None

    @property
    def name(self) -> str:
        return "hmemory"

    def is_available(self) -> bool:
        return True

    def _env(self, key, default):
        return os.environ.get(key, default)

    def _int_env(self, key, default):
        try:
            return int(os.environ.get(key, default))
        except (ValueError, TypeError):
            return default

    def get_config_schema(self):
        return [
            {
                "key": "base_url",
                "description": "hmemory HTTP service URL",
                "default": "http://localhost:8080",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        config_path = Path(hermes_home) / "hmemory.json"
        config_path.write_text(json.dumps(values, indent=2))

    def initialize(self, session_id: str, **kwargs) -> None:
        config_path = Path(kwargs.get("hermes_home", "~/.hermes")) / "hmemory.json"
        cfg = {"base_url": "http://localhost:8080"}
        if config_path.exists():
            cfg.update(json.loads(config_path.read_text()))
        self._base_url = cfg["base_url"].rstrip("/")

        # Session strategy
        strategy = self._env("SESSION_STRATEGY", "per-session")
        path_or_repo = (
            kwargs.get("cwd", "") if strategy in ("per-directory", "per-repo") else ""
        )

        try:
            resp = requests.post(
                f"{self._base_url}/session/resolve",
                json={
                    "session_id": session_id,
                    "strategy": strategy,
                    "path_or_repo": path_or_repo,
                },
                timeout=5,
            )
            data = resp.json()
            self._session_id = data.get("session_id", session_id)
            self._last_resolved_session = self._session_id
        except requests.RequestException as e:
            logger.warning("hmemory session resolve failed: %s", e)
            self._session_id = session_id

        # Initialize the store
        try:
            requests.post(
                f"{self._base_url}/initialize",
                json={"session_id": self._session_id},
                timeout=5,
            )
        except requests.RequestException as e:
            logger.warning("hmemory init failed: %s", e)

        # Cadence env vars
        self._prefetch_cadence = self._int_env("PREFETCH_CADENCE", 1)
        self._sync_turn_cadence = self._int_env("SYNC_TURN_CADENCE", 1)
        self._conclusion_cadence = self._int_env("CONCLUSION_CADENCE", 10)
        self._context_tokens = self._int_env("CONTEXT_TOKENS", 0) or None
        self._base_context_cadence = self._int_env("BASE_CONTEXT_CADENCE", 5)

    def shutdown(self) -> None:
        pass

    def get_tool_schemas(self):
        return [
            {
                "name": "hmemory_add",
                "description": "Manually store an important memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "tags": {"type": "object", "optional": True},
                        "importance": {"type": "number", "optional": True},
                        "category": {"type": "string", "optional": True},
                        "source": {"type": "string", "optional": True},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "hmemory_search",
                "description": "Search stored memories by semantic similarity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                        "tag_filter": {"type": "object", "optional": True},
                        "tag_match_mode": {
                            "type": "string",
                            "enum": ["all", "any"],
                            "default": "all",
                            "optional": True,
                        },
                        "category_filter": {"type": "string", "optional": True},
                        "level": {
                            "type": "string",
                            "enum": ["summary", "overview", "details"],
                            "default": "details",
                            "optional": True,
                            "description": "L0=summary, L1=overview, L2=details",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "hmemory_list",
                "description": "List memories with pagination",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 20, "optional": True},
                        "offset": {"type": "integer", "default": 0, "optional": True},
                    },
                    "required": [],
                },
            },
            {
                "name": "hmemory_get",
                "description": "Get a single memory by ID",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                },
            },
            {
                "name": "hmemory_update",
                "description": "Update an existing memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "content": {"type": "string", "optional": True},
                        "tags": {"type": "object", "optional": True},
                        "importance": {"type": "number", "optional": True},
                        "category": {"type": "string", "optional": True},
                        "source": {"type": "string", "optional": True},
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "hmemory_backup",
                "description": "Export all memories as JSON",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "hmemory_restore",
                "description": "Restore memories from a backup JSON array",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "session_id": {"type": "string"},
                                    "content": {"type": "string"},
                                    "embedding": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                    "tags": {"type": "object", "optional": True},
                                    "metadata": {"type": "object", "optional": True},
                                    "importance": {"type": "number", "optional": True},
                                    "source": {"type": "string", "optional": True},
                                    "category": {"type": "string", "optional": True},
                                },
                            },
                        },
                    },
                    "required": ["memories"],
                },
            },
            {
                "name": "hmemory_delete",
                "description": "Delete a memory by ID",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kw) -> str:
        body = {
            "tool_name": tool_name.replace("hmemory_", "memory_"),
            "args": args,
            "session_id": self._session_id,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/handle_tool_call",
                json=body,
                timeout=10,
            )
            data = resp.json()

            # Tiered formatting for memory_search results
            if (
                tool_name == "hmemory_search"
                and data.get("ok")
                and data.get("memories")
            ):
                level = args.get("level", "details")
                memories = data["memories"]
                if level == "summary" and memories:
                    data = {
                        "ok": True,
                        "level": "summary",
                        "result": f"Top match: {memories[0]['content'][:200]}",
                        "score": memories[0].get("score", 0),
                    }
                elif level == "overview" and memories:
                    lines = []
                    for m in memories[:3]:
                        tags = m.get("tags", {})
                        tag_str = f" [{tags}]" if tags else ""
                        lines.append(
                            f"- {m['content'][:300]}{tag_str} (score: {m.get('score', 0):.2f})"
                        )
                    data = {
                        "ok": True,
                        "level": "overview",
                        "result": "Top matches:\n" + "\n".join(lines),
                        "count": len(lines),
                    }

            return json.dumps(data)
        except requests.RequestException as e:
            return json.dumps({"ok": False, "error": str(e)})

    def system_prompt_block(self) -> str:
        return "hmemory external memory provider is active."

    def _truncate_to_tokens(self, text: str) -> str:
        if not self._context_tokens or self._context_tokens <= 0:
            return text
        # Approximate: ~4 chars per token
        max_chars = self._context_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[truncated...]"

    def prefetch(self, query: str, **kw) -> str | None:
        # Cadence: skip based on turn count
        if (
            self._prefetch_cadence > 1
            and self._turn_count % self._prefetch_cadence != 0
        ):
            return None

        try:
            # Two-layer context: base context + search results
            context_parts = []

            # Layer 1: Base context (high-importance memories)
            if (
                self._base_context_cadence > 0
                and self._turn_count % self._base_context_cadence == 0
            ):
                try:
                    resp = requests.post(
                        f"{self._base_url}/session/base_context",
                        json={
                            "session_id": self._session_id,
                            "limit": 3,
                        },
                        timeout=5,
                    )
                    data = resp.json()
                    if data.get("ok") and data.get("memories"):
                        items = []
                        for m in data["memories"][:3]:
                            items.append(f"- {m['content']}")
                        context_parts.append(
                            "Key context from this session:\n" + "\n".join(items)
                        )
                except requests.RequestException:
                    pass

            # Layer 2: Search results (with cold/warm flag)
            try:
                prefetch_body = {
                    "query": query,
                    "session_id": self._session_id,
                    "limit": 5,
                    "level": "overview",
                }
                resp = requests.post(
                    f"{self._base_url}/prefetch",
                    json=prefetch_body,
                    timeout=5,
                )
                data = resp.json()
                if data.get("ok") and data.get("memories"):
                    parts = []
                    for m in data["memories"]:
                        tags = m.get("tags", {})
                        tag_str = f" [{tags}]" if tags else ""
                        parts.append(
                            f"- {m['content']}{tag_str} (score: {m['score']:.2f})"
                        )
                    if parts:
                        context_parts.append("Relevant memories:\n" + "\n".join(parts))
            except requests.RequestException:
                pass

            if not context_parts:
                return None

            combined = "\n\n".join(context_parts)
            return self._truncate_to_tokens(combined)
        except Exception:
            return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id="",
        messages=None,
        **kw,
    ) -> None:
        self._turn_count += 1

        # Cadence: skip based on turn count
        if (
            self._sync_turn_cadence > 1
            and self._turn_count % self._sync_turn_cadence != 0
        ):
            return

        category = kw.get("category", "chat")
        sid = self._session_id

        def _sync():
            try:
                requests.post(
                    f"{self._base_url}/sync_turn",
                    json={
                        "user": user_content,
                        "assistant": assistant_content,
                        "session_id": sid,
                        "category": category,
                        "source": "hermes",
                    },
                    timeout=10,
                )
            except requests.RequestException as e:
                logger.warning("hmemory sync_turn failed: %s", e)

        thread = threading.Thread(target=_sync, daemon=True)
        thread.start()

        # Increment turn count on server
        try:
            requests.post(
                f"{self._base_url}/session/status",
                json={"session_id": sid},
                timeout=3,
            )
        except requests.RequestException:
            pass

        # Conclusions: trigger periodically based on conclusion_cadence
        if (
            self._conclusion_cadence > 0
            and self._turn_count >= self._conclusion_cadence
        ):
            if self._turn_count - self._conclusion_turn >= self._conclusion_cadence:
                self._conclusion_turn = self._turn_count
                self._schedule_conclusion(sid, user_content, assistant_content)

    def _schedule_conclusion(
        self, session_id: str, user_content: str, assistant_content: str
    ):
        def _conclude():
            try:
                # Simple heuristic: long assistant responses with technical content get concluded
                if len(assistant_content) > 200 and any(
                    kw in assistant_content.lower()
                    for kw in [
                        "plan",
                        "solution",
                        "approach",
                        "architecture",
                        "decision",
                    ]
                ):
                    # Find memory IDs for this turn — not trivial, so just create a lightweight
                    # conclusion that the LLM can use
                    requests.post(
                        f"{self._base_url}/session/conclude",
                        json={
                            "session_id": session_id,
                            "content": f"Insight from conversation turn: {assistant_content[:300]}",
                            "category": "insight",
                            "confidence": 0.6,
                            "source_turn_ids": [],
                        },
                        timeout=10,
                    )
            except requests.RequestException:
                pass

        thread = threading.Thread(target=_conclude, daemon=True)
        thread.start()

    def queue_prefetch(self, query: str) -> None:
        def _prefetch():
            try:
                self.prefetch(query)
            except Exception:
                pass

        thread = threading.Thread(target=_prefetch, daemon=True)
        thread.start()


def register(ctx) -> None:
    ctx.register_memory_provider(HMemoryProvider())
