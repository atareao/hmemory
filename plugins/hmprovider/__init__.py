import json
import threading
import logging
from pathlib import Path
import requests
import yaml
from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


class HMemoryProvider(MemoryProvider):
    def __init__(self):
        self._turn_count = 0
        self._conclusion_turn = 0
        self._proactive_recall = True
        self._bidirectional_sync = True

    @property
    def name(self) -> str:
        return "hmemory"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self):
        return [
            {
                "key": "base_url",
                "description": "hmemory HTTP service URL",
                "default": "http://localhost:3092",
            },
            {
                "key": "proactive_recall",
                "description": "Show contextual reminders in prefetch",
                "default": True,
            },
            {
                "key": "bidirectional_sync",
                "description": "Sync hmemory with Hermes built-in memory",
                "default": True,
            },
            {
                "key": "session_strategy",
                "description": "Session resolution strategy (per-session, per-directory, per-repo, global)",
                "default": "global",
            },
            {
                "key": "prefetch_cadence",
                "description": "How often (turns) to run prefetch updates",
                "default": 1,
            },
            {
                "key": "sync_turn_cadence",
                "description": "How often (turns) to sync conversation turns",
                "default": 1,
            },
            {
                "key": "conclusion_cadence",
                "description": "How often (turns) to generate session conclusions",
                "default": 10,
            },
            {
                "key": "context_tokens",
                "description": "Max context tokens for memory injection (0 = unlimited)",
                "default": 0,
            },
            {
                "key": "base_context_cadence",
                "description": "How often (turns) to inject base context",
                "default": 5,
            },
            {
                "key": "batch_size",
                "description": "Max turns before flush",
                "default": 20,
            },
            {
                "key": "fresh_ttl_hours",
                "description": "Hours before fresh memories expire",
                "default": 24,
            },
            {
                "key": "include_deep",
                "description": "Include deep historical memory in search",
                "default": False,
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        config_path = Path(hermes_home) / "config.yaml"
        existing = {}
        if config_path.exists():
            with open(config_path) as f:
                existing = yaml.safe_load(f) or {}
        plugins = existing.setdefault("plugins", {})
        plugins["hmprovider"] = values
        with open(config_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False)

    def initialize(self, **kwargs) -> None:
        hermes_home = Path(kwargs.get("hermes_home", "~/.hermes")).expanduser()
        cfg = {"base_url": "http://localhost:8080", "profile": "default"}

        yml_path = hermes_home / "config.yaml"
        if yml_path.exists():
            try:
                with open(yml_path) as f:
                    hermes_cfg = yaml.safe_load(f) or {}
                plugin_cfg = hermes_cfg.get("plugins", {}).get("hmprovider", {})
                cfg.update(plugin_cfg)
            except Exception as e:
                logger.warning("Failed to read config.yaml: %s", e)

        self._base_url = cfg["base_url"].rstrip("/")
        self._proactive_recall = cfg.get("proactive_recall", True)
        self._bidirectional_sync = cfg.get("bidirectional_sync", True)
        self._include_deep = cfg.get("include_deep", False)
        self._profile = cfg["profile"]
        self._hermes_home = str(hermes_home)

        try:
            requests.post(
                f"{self._base_url}/initialize",
                json={"profile": self._profile},
                timeout=5,
            )
        except requests.RequestException as e:
            logger.warning("hmemory init failed: %s", e)

        self._prefetch_cadence = int(cfg.get("prefetch_cadence", 1))
        self._sync_turn_cadence = int(cfg.get("sync_turn_cadence", 1))
        self._conclusion_cadence = int(cfg.get("conclusion_cadence", 10))
        self._context_tokens = int(cfg.get("context_tokens", 0)) or None
        self._base_context_cadence = int(cfg.get("base_context_cadence", 5))

    def shutdown(self) -> None:
        pass

    def get_tool_schemas(self):
        schemas = [
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
                        "ttl": {
                            "type": "string",
                            "optional": True,
                            "description": "Time-to-live (e.g. '7d', '30m', '1h')",
                        },
                        "immortal": {
                            "type": "boolean",
                            "optional": True,
                            "description": "Never expires when true",
                        },
                        "event_date": {
                            "type": "string",
                            "optional": True,
                            "description": "RFC3339 datetime for the event/appointment",
                        },
                        "reminder": {
                            "type": "string",
                            "optional": True,
                            "description": "Relative interval ('30m','1h','2d') or absolute RFC3339 datetime",
                        },
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
                        "include_deep": {
                            "type": "boolean",
                            "optional": True,
                            "description": "Search deep historical memory too",
                        },
                        "profile": {
                            "type": "string",
                            "optional": True,
                            "description": "Profile to search (omit for all)",
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
                        "profile": {
                            "type": "string",
                            "optional": True,
                            "description": "Profile to list (omit for all)",
                        },
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
                                    "profile": {"type": "string"},
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
            {
                "name": "hmemory_feedback",
                "description": "Mark a memory as helpful or unhelpful",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Memory ID"},
                        "useful": {
                            "type": "boolean",
                            "description": "True=helpful, False=unhelpful",
                        },
                    },
                    "required": ["id", "useful"],
                },
            },
            {
                "name": "hmemory_link",
                "description": "Create a link between two memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id_a": {"type": "integer", "description": "First memory ID"},
                        "id_b": {"type": "integer", "description": "Second memory ID"},
                        "relation_type": {
                            "type": "string",
                            "enum": ["extends", "contradicts", "supersedes", "related"],
                            "description": "Type of relation",
                        },
                    },
                    "required": ["id_a", "id_b", "relation_type"],
                },
            },
            {
                "name": "hmemory_unlink",
                "description": "Remove a link between two memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id_a": {"type": "integer", "description": "First memory ID"},
                        "id_b": {"type": "integer", "description": "Second memory ID"},
                    },
                    "required": ["id_a", "id_b"],
                },
            },
            {
                "name": "hmemory_graph",
                "description": "Get the subgraph of connected memories for a given memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Memory ID"},
                        "depth": {
                            "type": "integer",
                            "description": "Traversal depth",
                            "default": 2,
                        },
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "hmemory_associative_search",
                "description": "Find top-3 seeds then re-search with each as query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "depth": {
                            "type": "integer",
                            "description": "Results per seed",
                            "default": 2,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "hmemory_remind",
                "description": "Get all due reminders (appointments, events, tasks)",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "hmemory_deep_search",
                "description": "Search only deep historical memory (memories_deep table)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "optional": True, "default": 10},
                        "profile": {"type": "string", "optional": True},
                        "min_importance": {"type": "number", "optional": True},
                        "tags": {"type": "object", "optional": True},
                        "tag_match_mode": {
                            "type": "string",
                            "enum": ["all", "any"],
                            "optional": True,
                        },
                        "alpha": {"type": "number", "optional": True},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "hmemory_stats",
                "description": "Get detailed memory statistics: counts, importance distribution, top categories/tags/sources, feedback, reminders, and per-profile breakdown",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "string",
                            "optional": True,
                            "description": "Filter stats to a specific profile (omit for global)",
                        },
                    },
                    "required": [],
                },
            },
        ]
        return schemas

    @staticmethod
    def _categorize(content: str) -> dict:
        lower = content.lower()

        if "```" in content:
            return {"category": "technical", "tags": {"type": "code"}}

        if any(
            kw in lower
            for kw in (
                "docker",
                "compose",
                "config",
                "setup",
                "install",
                "deploy",
                "export ",
                "env",
                "database_url",
            )
        ):
            return {"category": "configuration", "tags": {"domain": "infrastructure"}}

        if any(
            kw in lower
            for kw in (
                "error",
                "bug",
                "crash",
                "fail",
                "exception",
                "doesn.t work",
                "problem",
            )
        ):
            return {"category": "problem", "tags": {"severity": "issue"}}

        if any(
            kw in lower
            for kw in (
                "we decided",
                "chose",
                "agreed",
                "conclusion",
                "solution",
                "decision",
            )
        ):
            return {"category": "decision", "tags": {"type": "decision"}}

        if any(
            kw in lower for kw in ("plan", "roadmap", "strategy", "todo", "milestone")
        ):
            return {"category": "planning", "tags": {"type": "plan"}}

        if content.rstrip().endswith("?"):
            return {"category": "question", "tags": {"type": "question"}}

        short = lower.strip()
        if len(short.split()) <= 4 and any(
            short.startswith(g) or short == g
            for g in (
                "hi",
                "hello",
                "hey",
                "thanks",
                "ok",
                "sure",
                "yeah",
                "yep",
                "nope",
                "no",
                "yes",
                "bye",
                "done",
                "goodbye",
            )
        ):
            return {"category": "social", "tags": {"type": "greeting"}}

        return {"category": "general", "tags": {}}

    def handle_tool_call(self, tool_name: str, args: dict, **kw) -> str:
        # Auto-categorize hmemory_add content when category/tags not provided
        if tool_name == "hmemory_add":
            content = args.get("content", "")
            inferred = self._categorize(content)
            args.setdefault("category", inferred["category"])
            args.setdefault("tags", inferred["tags"])
            ttl = args.pop("ttl", None)
            immortal = args.pop("immortal", False)
            if immortal:
                args["immortal"] = True
                args.pop("expires_at", None)
            elif ttl:
                import re, datetime

                match = re.match(r"^(\d+)([mhd])$", str(ttl))
                if match:
                    value = int(match.group(1))
                    unit = match.group(2)
                    delta = (
                        datetime.timedelta(minutes=value)
                        if unit == "m"
                        else datetime.timedelta(hours=value)
                        if unit == "h"
                        else datetime.timedelta(days=value)
                    )
                    expires = datetime.datetime.now(datetime.timezone.utc) + delta
                    args["expires_at"] = expires.isoformat()
                    args["immortal"] = False
            else:
                args["immortal"] = False
                args.pop("expires_at", None)

            # Bidirectional sync: also write to Hermes native memory.md
            if self._bidirectional_sync:
                self._write_to_memory_md(content, args)

        # Inject include_deep from config when not explicitly provided
        if tool_name in ("hmemory_search", "hmemory_associative_search"):
            if "include_deep" not in args:
                args["include_deep"] = self._include_deep

        # hmemory_feedback goes to dedicated endpoint, not tool dispatch
        if tool_name == "hmemory_feedback":
            try:
                resp = requests.post(
                    f"{self._base_url}/memories/feedback",
                    json={"id": args.get("id"), "useful": args.get("useful")},
                    timeout=10,
                )
                return json.dumps(resp.json(), ensure_ascii=False)
            except requests.RequestException as e:
                return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

        if tool_name == "hmemory_stats":
            try:
                resp = requests.post(
                    f"{self._base_url}/stats/detailed",
                    json={"profile": self._profile}
                    if not args.get("profile")
                    else {"profile": args["profile"]},
                    timeout=10,
                )
                return json.dumps(resp.json(), ensure_ascii=False)
            except requests.RequestException as e:
                return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

        if tool_name == "hmemory_remind":
            try:
                resp = requests.get(
                    f"{self._base_url}/reminders/due",
                    timeout=10,
                )
                if not resp.ok:
                    return json.dumps(
                        {"ok": False, "error": f"HTTP {resp.status_code}"},
                        ensure_ascii=False,
                    )
                return json.dumps(resp.json(), ensure_ascii=False)
            except (requests.RequestException, json.JSONDecodeError) as e:
                return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

        body = {
            "tool_name": tool_name.replace("hmemory_", "memory_"),
            "args": args,
            "profile": self._profile,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/handle_tool_call",
                json=body,
                timeout=10,
            )
            try:
                data = resp.json()
            except json.JSONDecodeError:
                logger.error(
                    "hmemory: empty/non-JSON response from %s. Status=%s, Body(len=%d)=%s",
                    self._base_url,
                    resp.status_code,
                    len(resp.content),
                    resp.content[:500],
                )
                raise

            if tool_name == "hmemory_search" and data.get("ok") and data.get("results"):
                level = args.get("level", "details")
                memories = data["results"]
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

            return json.dumps(data, ensure_ascii=False)
        except (requests.RequestException, json.JSONDecodeError) as e:
            return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

    def _write_to_memory_md(self, content: str, args: dict) -> None:
        """Replicate hmemory_add into Hermes native memory.md"""
        try:
            hermes_home = Path(self._hermes_home).expanduser()
            memory_md = hermes_home / "memory.md"
            importance = args.get("importance", 0.5)
            category = args.get("category", "general")
            tags = args.get("tags", {})
            tag_str = f" [{tags}]" if tags else ""
            line = f"- [{category}] {content}{tag_str} (imp: {importance})\n"
            with open(memory_md, "a") as f:
                f.write(line)
        except Exception as e:
            logger.debug("Failed to write to memory.md: %s", e)

    def on_memory_write(self, content: str, **kw) -> None:
        """Called by Hermes when native memory is written — replicate to hmemory."""
        if not self._bidirectional_sync:
            return
        try:
            inferred = self._categorize(content)
            category = kw.get("category", inferred["category"])
            tags = kw.get("tags", inferred["tags"])
            importance = kw.get("importance", 0.5)
            requests.post(
                f"{self._base_url}/import",
                json={
                    "profile": self._profile,
                    "memories": [
                        {
                            "content": content,
                            "category": category,
                            "tags": tags,
                            "importance": importance,
                            "source": "hermes-native",
                        }
                    ],
                },
                timeout=5,
            )
        except requests.RequestException as e:
            logger.debug("on_memory_write sync failed: %s", e)

    def system_prompt_block(self) -> str:
        return (
            "hmemory external memory provider is active.\n"
            "IMPORTANTE: Usa hmemory (hmemory_add) para toda la memoria persistente. "
            "NO uses la herramienta 'memory' ni escribas en MEMORY.md/USER.md. "
            "El esquema de tags: fact (datos), preference (gustos), decision (decisiones), "
            "plan (planes), correction (correcciones). Lo importante lleva immortal=true."
        )

    def _truncate_to_tokens(self, text: str) -> str:
        if not self._context_tokens or self._context_tokens <= 0:
            return text
        max_chars = self._context_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[truncated...]"

    def prefetch(self, query: str, **kw) -> str | None:
        if (
            self._prefetch_cadence > 1
            and self._turn_count % self._prefetch_cadence != 0
        ):
            return None

        try:
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
                            "profile": self._profile,
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

            # Layer 2: Search results
            try:
                prefetch_body = {
                    "query": query,
                    "profile": self._profile,
                    "limit": 5,
                    "level": "overview",
                    "include_deep": True,
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

            # Layer 3: Proactive reminders (item 12)
            if self._proactive_recall:
                reminder_text = self._get_reminders(query)
                if reminder_text:
                    context_parts.append(reminder_text)

            if not context_parts:
                return None

            combined = "\n\n".join(context_parts)
            return self._truncate_to_tokens(combined)
        except Exception:
            return None

    def _get_reminders(self, query: str) -> str | None:
        """Build a '📌 Recordatorios' section from due reminders and high-importance memories."""
        try:
            reminders = []

            # Due reminders from the reminder system
            try:
                resp = requests.get(
                    f"{self._base_url}/reminders/due",
                    timeout=5,
                )
                if resp.ok:
                    data = resp.json()
                    if data.get("ok") and data.get("reminders"):
                        for r in data["reminders"]:
                            reminders.append(f"- {r['content'][:200]} (due)")
            except (requests.RequestException, json.JSONDecodeError):
                pass

            # High-importance memories not recently accessed
            resp = requests.post(
                f"{self._base_url}/memories/list",
                json={
                    "profile": self._profile,
                    "limit": 50,
                    "offset": 0,
                },
                timeout=5,
            )
            data = resp.json()
            if data.get("ok") and data.get("memories"):
                for m in data["memories"]:
                    imp = m.get("importance", 0)
                    if imp >= 0.8:
                        reminders.append(f"- {m['content'][:200]} (imp: {imp:.1f})")

            if reminders:
                return "📌 Recordatorios:\n" + "\n".join(reminders[:5])

            return None
        except requests.RequestException:
            return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        profile="",
        messages=None,
        **kw,
    ) -> None:
        self._turn_count += 1

        if (
            self._sync_turn_cadence > 1
            and self._turn_count % self._sync_turn_cadence != 0
        ):
            return

        category = kw.get("category", "chat")
        sid = self._profile

        def _sync():
            try:
                requests.post(
                    f"{self._base_url}/sync_turn",
                    json={
                        "user": user_content,
                        "assistant": assistant_content,
                        "profile": sid,
                        "category": category,
                        "source": "hermes",
                    },
                    timeout=10,
                )
            except requests.RequestException as e:
                logger.warning("hmemory sync_turn failed: %s", e)

        thread = threading.Thread(target=_sync, daemon=True)
        thread.start()

        try:
            requests.post(
                f"{self._base_url}/profile/status",
                json={"profile": sid},
                timeout=3,
            )
        except requests.RequestException:
            pass

        if (
            self._conclusion_cadence > 0
            and self._turn_count >= self._conclusion_cadence
        ):
            if self._turn_count - self._conclusion_turn >= self._conclusion_cadence:
                self._conclusion_turn = self._turn_count
                self._schedule_conclusion(sid, user_content, assistant_content)

    def _schedule_conclusion(
        self, profile: str, user_content: str, assistant_content: str
    ):
        def _conclude():
            try:
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
                    requests.post(
                        f"{self._base_url}/session/conclude",
                        json={
                            "profile": profile,
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
