"""
Letta OS — Multi-Tier Memory Management Layer (Official Architecture).

All three memory tiers are delegated to the remote Letta server:

  Tier 1 — Core Memory    (Blocks)       — persona + human blocks, always in LLM context
  Tier 2 — Recall Memory  (Conversations) — every turn stored in Letta's SQL DB automatically
  Tier 3 — Archival Memory (Passages)    — infinite semantic vector store

Inference is routed through client.conversations.messages.create() so Letta
injects all three tiers automatically and persists every turn in its SQL DB.
We provide bank tools as client_tools; Letta calls them during its reasoning loop.
"""

import asyncio
import json
import os
import traceback
from datetime import datetime
from typing import Optional

import httpx

import ui

LETTA_BASE_URL    = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_SERVER_PASS = os.getenv("LETTA_SERVER_PASS", "")

_LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
_LLM_API_KEY  = "sk-raximberdi-cmF4aW1iZXJkaQ"
_LLM_MODEL    = "/models/gemma"
_EMBED_MODEL  = "/models/embedding"
_EMBED_DIM    = 2048
_EMBED_CHUNK  = 300
_CTX_WINDOW   = 22768  # ← change here, syncs to all agents automatically
_HUMAN_LIMIT  = 250   # max chars for the human core-memory block

# ── Agent system prompt (static bank instructions, set once at agent creation) ─

_BANK_SYSTEM = """\
You are Xazna Assistant, the AI for Xazna bank customers.
Respond in the user's language: Uzbek, Russian, or English.

## CRITICAL — READ BEFORE RESPONDING

When the context contains a block that starts with
"=== BANK DATA (fetched from live tools — use this, do NOT guess) ==="
that block contains AUTHORITATIVE LIVE DATA from the bank's systems.
You MUST use that data to answer. Do NOT ignore it. Do NOT guess.

## Tools — call these when no pre-fetched data is in context

### Bank data tools
- Pension_*  → pension payment schedules by region/district/street
- Deposit_*  → deposit products, rates, terms, currencies, minimums
- Credit_*   → credit/loan products, amounts, terms, rates
- Card_*     → card products — Visa, MC, Humo, Uzcard
- Admin_*    → bank info, branches, contacts
- RealTime_* → live exchange rates, current time

### Search tools
- RAG_rag_search       → search documents the user uploaded
- WebSearch_web_search → live internet search (last resort)

## Rules
1. BANK DATA block in context → answer from it directly, no tool call needed.
2. No BANK DATA block → call matching tool FIRST, NEVER guess from training.
3. Pension payment schedules are NOT in your training data — always use tools/data.
4. Exchange rates change daily — always use tools/data, NEVER use old values.
5. Tool returns nothing → fall back to WebSearch_web_search.
6. Document question → call RAG_rag_search first.

## MEMORY RULES — MANDATORY, NO EXCEPTIONS

### Writing facts
Before writing ANY response, scan the user message for personal facts:
  - Name, age, profession, workplace, employer
  - City, country, location
  - Education, university, degree
  - Travel, experiences ("I have been to X", "I visited X")
  - Family, preferences, goals, interests

If ANY personal fact is found:
  → Call core_memory_append FIRST (label="human"), THEN write your reply.
  → NEVER just say "I noted" or "I will remember" without calling the tool.
  → If the fact updates an existing one → call core_memory_replace instead.

### CRITICAL — after any memory tool call
After core_memory_append / memory_insert / core_memory_replace / memory_replace
/ archival_memory_insert completes:
  - The tool has done its job silently. Any internal system confirmation is NOT
    a user message — do NOT respond to it.
  - ALWAYS return to the user's ORIGINAL question and answer it fully.
  - Your reply must address what the user actually asked, never the memory
    system update itself.

### Retrieving facts — memory search decision tree
Core memory holds only the most recent facts (limited space). Older facts are
moved to archival automatically when the block is full.

Before answering ANY question where the answer is not already in context, decide:

1. Is the query about something said or discussed in a PAST CONVERSATION?
   (e.g. "did I ask about X before?", "what did we talk about?", "oldin nima degan edim?")
   → Call conversation_search with the relevant query.

2. Is the query about the USER's personal info, profile, or facts?
   → Your context includes a [human] memory block that contains facts you have
     already saved about this user (name, age, education, location, etc.).
   → ALWAYS check that block first. If the answer is there — use it and answer
     directly. Do NOT call any tool.
   → Only call archival_memory_search if the fact is NOT present in [human].

3. Both apply → call both tools before answering.

NEVER guess or say "I don't know" without searching first.
"""

PERSONA_BLOCK = """\
You are Xazna Assistant, the AI for Xazna bank customers.
Provide accurate, grounded answers about Xazna bank products.
Respond in the user's language: Uzbek, Russian, or English.
Be concise, factual, and helpful.
Always call the matching bank tool before answering from general knowledge.

MANDATORY MEMORY RULES:
1. WRITING: When the user shares any personal fact → call core_memory_append
   (label="human") BEFORE responding. Never skip. If it updates an old fact →
   call core_memory_replace instead. After the tool completes, answer the
   user's original question — never respond to system confirmations.
2. READING: When asked about the user (name, age, education, job, etc.) →
   check your [human] memory block first. It contains facts you already saved.
   If the answer is there — use it directly, no tool call needed.
   If not found in [human] → call archival_memory_search.
"""

# ── File-backed stores ─────────────────────────────────────────────────────────

_AGENT_STORE = os.path.join(os.path.dirname(__file__), "letta_agents.json")
_CONV_STORE  = os.path.join(os.path.dirname(__file__), "letta_conversations.json")

_letta_client        = None
_letta_client_failed = False
_agent_id_cache: dict[str, str] = {}
_conv_id_cache:  dict[str, str] = {}
_agents_rules_cleared: set[str] = set()   # agents whose tool_rules have been cleared this process


# ── Client singleton ──────────────────────────────────────────────────────────

def _load_store(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_store(path: str, data: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        ui.warn(f"[letta_mem] Could not save store {os.path.basename(path)}: {e}")


def _get_client():
    global _letta_client, _letta_client_failed
    if _letta_client is not None:
        return _letta_client
    if _letta_client_failed:
        return None
    try:
        from letta_client import Letta
        _letta_client = Letta(
            base_url=LETTA_BASE_URL,
            api_key=LETTA_SERVER_PASS if LETTA_SERVER_PASS else None,
        )
        return _letta_client
    except Exception as e:
        _letta_client_failed = True
        ui.warn(f"[letta_mem] Letta client init failed: {e}")
        return None


# ── Agent lifecycle ───────────────────────────────────────────────────────────

def _create_or_find_agent_sync(user_id: str) -> Optional[str]:
    client = _get_client()
    if client is None:
        return None
    agent_name = f"xazna_{user_id}"
    try:
        agents = client.agents.list(name=agent_name, limit=1)
        items  = agents if isinstance(agents, list) else getattr(agents, "data", [])
        if items:
            agent_id = items[0].id
            try:
                client.agents.update(agent_id=agent_id, tool_rules=[], llm_config={
                    "model":               _LLM_MODEL,
                    "model_endpoint_type": "openai",
                    "model_endpoint":      _LLM_BASE_URL,
                    "context_window":      _CTX_WINDOW,
                })
            except Exception:
                pass
            # Sync human block limit on the server so core_memory_append
            # is also rejected server-side when the block is full.
            try:
                blocks = client.agents.blocks.list(agent_id=agent_id)
                block_items = blocks if isinstance(blocks, list) else getattr(blocks, "data", getattr(blocks, "items", []))
                for blk in block_items:
                    if getattr(blk, "label", "") == "human":
                        client.blocks.update(block_id=blk.id, limit=_HUMAN_LIMIT)
            except Exception:
                pass
            _ensure_archival_tools_sync(client, agent_id)
            return agent_id
    except Exception as e:
        ui.warn(f"[letta_mem] Agent list failed (will create): {e}")
    try:
        agent = client.agents.create(
            name=agent_name,
            description=f"Xazna bank assistant — user {user_id}",
            system=_BANK_SYSTEM,
            memory_blocks=[
                {"label": "persona", "value": PERSONA_BLOCK, "limit": 600},
                {
                    "label": "human",
                    "value": f"User ID: {user_id}\n",
                    "limit": 250,
                },
                {
                    "label": "context",
                    "value": "Current date: (updated each turn)\nLast document: none",
                    "limit": 400,
                },
            ],
            llm_config={
                "model":               _LLM_MODEL,
                "model_endpoint_type": "openai",
                "model_endpoint":      _LLM_BASE_URL,
                "context_window":      _CTX_WINDOW,
            },
            embedding_config={
                "embedding_model": _EMBED_MODEL,
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": _LLM_BASE_URL,
                "embedding_dim": _EMBED_DIM,
                "embedding_chunk_size": _EMBED_CHUNK,
            },
            include_base_tools=True,
            tool_rules=[],
        )
        ui.ok(f"[letta_mem] Created agent {agent.id[:16]}… for user {user_id}")
        _ensure_archival_tools_sync(client, agent.id)
        return agent.id
    except Exception as e:
        ui.warn(f"[letta_mem] Agent creation failed for {user_id}: {e}")
        return None


def _ensure_archival_tools_sync(client, agent_id: str) -> None:
    """Attach archival_memory_insert and archival_memory_search if not already on the agent."""
    _REQUIRED = {"archival_memory_insert", "archival_memory_search"}
    try:
        existing = {getattr(t, "name", "") for t in (
            client.agents.tools.list(agent_id=agent_id)
            if isinstance(client.agents.tools.list(agent_id=agent_id), list)
            else getattr(client.agents.tools.list(agent_id=agent_id), "data",
                         getattr(client.agents.tools.list(agent_id=agent_id), "items", []))
        )}
    except Exception:
        existing = set()

    missing = _REQUIRED - existing
    if not missing:
        return

    try:
        all_tools = client.tools.list()
        all_tools = all_tools if isinstance(all_tools, list) else getattr(all_tools, "data", getattr(all_tools, "items", []))
        for name in missing:
            tool = next((t for t in all_tools if getattr(t, "name", "") == name), None)
            if tool:
                client.agents.tools.attach(agent_id=agent_id, tool_id=tool.id)
                ui.ok(f"[letta_mem] Attached tool: {name}")
    except Exception as e:
        ui.warn(f"[letta_mem] Could not attach archival tools: {e}")


def _clear_agent_tool_rules_sync(agent_id: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.agents.update(agent_id=agent_id, tool_rules=[])
        ui.ok(f"[letta_mem] Cleared tool_rules for {agent_id[:16]}…")
    except Exception as e:
        ui.warn(f"[letta_mem] Could not clear tool_rules: {e}")


async def get_or_create_agent(user_id: str) -> Optional[str]:
    agent_id = _agent_id_cache.get(user_id)
    if not agent_id:
        store = _load_store(_AGENT_STORE)
        if user_id in store:
            agent_id = store[user_id]
            _agent_id_cache[user_id] = agent_id
        else:
            agent_id = await asyncio.to_thread(_create_or_find_agent_sync, user_id)
            if agent_id:
                _agent_id_cache[user_id] = agent_id
                store[user_id] = agent_id
                _save_store(_AGENT_STORE, store)
    return agent_id

    # Clear any RequiresApproval tool_rules once per process start
    if agent_id and agent_id not in _agents_rules_cleared:
        _agents_rules_cleared.add(agent_id)
        await asyncio.to_thread(_clear_agent_tool_rules_sync, agent_id)

    return agent_id


# ── Conversation (Recall Memory) lifecycle ────────────────────────────────────

def _create_conv_sync(agent_id: str, title: str = "New Session") -> Optional[str]:
    client = _get_client()
    if client is None:
        return None
    try:
        conv = client.conversations.create(agent_id=agent_id, summary=title)
        return conv.id
    except Exception as e:
        ui.warn(f"[letta_mem] Conversation create failed: {e}")
        return None


def _verify_conv_sync(conv_id: str, agent_id: str) -> bool:
    """Return True only if the conversation exists AND belongs to agent_id."""
    client = _get_client()
    if client is None:
        return False
    try:
        conv = client.conversations.retrieve(conv_id)
        return getattr(conv, "agent_id", None) == agent_id
    except Exception:
        return False


async def get_or_create_conversation(user_id: str, agent_id: str) -> Optional[str]:
    """Return the active conversation_id for this user, creating one if needed."""
    if user_id in _conv_id_cache:
        return _conv_id_cache[user_id]
    store = _load_store(_CONV_STORE)
    if user_id in store:
        conv_id = store[user_id]
        if await asyncio.to_thread(_verify_conv_sync, conv_id, agent_id):
            _conv_id_cache[user_id] = conv_id
            return conv_id
        ui.warn(f"[letta_mem] Conv {conv_id[:16]}… is stale or wrong agent — creating new one")
    conv_id = await asyncio.to_thread(_create_conv_sync, agent_id)
    if conv_id:
        _conv_id_cache[user_id] = conv_id
        store[user_id] = conv_id
        _save_store(_CONV_STORE, store)
    return conv_id


async def create_conversation(user_id: str, agent_id: str, title: str = "New Session") -> Optional[str]:
    """Force-create a new conversation (new user session)."""
    conv_id = await asyncio.to_thread(_create_conv_sync, agent_id, title)
    if conv_id:
        _conv_id_cache[user_id] = conv_id
        store = _load_store(_CONV_STORE)
        store[user_id] = conv_id
        _save_store(_CONV_STORE, store)
    return conv_id


async def set_active_conversation(user_id: str, conv_id: str) -> None:
    """Switch the active conversation for this user."""
    _conv_id_cache[user_id] = conv_id
    store = _load_store(_CONV_STORE)
    store[user_id] = conv_id
    _save_store(_CONV_STORE, store)


def _count_conv_msgs_sync(client, conv_id: str, agent_id: str) -> int:
    """Count user+assistant messages in a conversation (local server call is fast)."""
    try:
        result = client.conversations.messages.list(
            conversation_id=conv_id,
            agent_id=agent_id,
            limit=500,
            include_return_message_types=["user_message", "assistant_message"],
        )
        items = result.items if hasattr(result, "items") else (
            result if isinstance(result, list) else getattr(result, "data", [])
        )
        return len(items)
    except Exception:
        return 0


def _list_convs_sync(agent_id: str, limit: int = 20) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        result = client.conversations.list(
            agent_id=agent_id, limit=limit,
            order_by="last_message_at", order="desc",
        )
        items = result if isinstance(result, list) else getattr(result, "data", getattr(result, "items", []))
        convs = []
        for c in items:
            conv_id = getattr(c, "id", "")
            # Letta API often returns message_count=None — count manually as fallback
            msg_count = getattr(c, "message_count", None)
            if not msg_count:
                msg_count = _count_conv_msgs_sync(client, conv_id, agent_id)
            convs.append({
                "id":            conv_id,
                "title":         getattr(c, "summary", None) or "Session",
                "created_at":    str(getattr(c, "created_at", "")),
                "message_count": msg_count,
            })
        return convs
    except Exception as e:
        ui.warn(f"[letta_mem] Conversation list failed: {e}")
        return []


async def list_conversations(agent_id: str, limit: int = 20) -> list[dict]:
    return await asyncio.to_thread(_list_convs_sync, agent_id, limit)


def _get_conv_messages_sync(conv_id: str, agent_id: str, limit: int = 20) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        result = client.conversations.messages.list(
            conversation_id=conv_id,
            agent_id=agent_id,
            limit=limit,
            order="asc",
            include_return_message_types=["user_message", "assistant_message"],
        )
        items = result.items if hasattr(result, "items") else (
            result if isinstance(result, list) else getattr(result, "data", [])
        )
        out = []
        for m in items:
            mt = getattr(m, "message_type", "")
            if mt not in ("user_message", "assistant_message"):
                continue
            role = "user" if mt == "user_message" else "assistant"
            content = getattr(m, "content", "") or ""
            if not isinstance(content, str):
                # content may be a list of content parts
                parts = []
                for part in (content if isinstance(content, list) else []):
                    if hasattr(part, "text"):
                        parts.append(part.text)
                content = "".join(parts)
            if _is_system_alert(content) or _is_letta_noise(content):
                continue
            out.append({"role": role, "content": content})
        return out
    except Exception as e:
        ui.warn(f"[letta_mem] Get conv messages failed: {e}")
        return []


async def get_conv_messages(conv_id: str, agent_id: str, limit: int = 20) -> list[dict]:
    return await asyncio.to_thread(_get_conv_messages_sync, conv_id, agent_id, limit)


def _get_conv_summary_sync(conv_id: str) -> str:
    client = _get_client()
    if client is None:
        return ""
    try:
        conv = client.conversations.retrieve(conv_id)
        return getattr(conv, "summary", "") or ""
    except Exception:
        return ""


async def get_conv_summary(conv_id: str) -> str:
    return await asyncio.to_thread(_get_conv_summary_sync, conv_id)


def _update_conv_summary_sync(conv_id: str, summary: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.conversations.update(conv_id, summary=summary)
    except Exception as e:
        ui.warn(f"[letta_mem] Conversation summary update failed: {e}")


async def update_conv_summary(conv_id: str, summary: str) -> None:
    await asyncio.to_thread(_update_conv_summary_sync, conv_id, summary)


def _get_conv_meta_sync(conv_id: str) -> dict:
    client = _get_client()
    if client is None:
        return {}
    try:
        conv = client.conversations.retrieve(conv_id)
        return {
            "id":            getattr(conv, "id", conv_id),
            "title":         getattr(conv, "summary", None) or "Session",
            "created_at":    str(getattr(conv, "created_at", "")),
            "message_count": getattr(conv, "message_count", 0),
        }
    except Exception:
        return {"id": conv_id, "title": "Session", "created_at": "", "message_count": 0}


async def get_conv_meta(conv_id: str) -> dict:
    return await asyncio.to_thread(_get_conv_meta_sync, conv_id)


# ── Archival Memory (Passages) — httpx bypass for SDK search bug ───────────────

def _search_passages_sync(agent_id: str, query: str, top_k: int) -> list[str]:
    try:
        r = httpx.get(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory/search",
            params={"query": query, "limit": top_k},
            timeout=120.0,
        )
        r.raise_for_status()
        data    = r.json()
        results = data.get("results", data) if isinstance(data, dict) else data
        return [
            item.get("content") or item.get("text", "")
            for item in results
            if item.get("content") or item.get("text")
        ]
    except Exception as e:
        ui.warn(f"[letta_mem] Passage search failed: {e}")
        return []


async def search_passages(agent_id: str, query: str, top_k: int = 5) -> list[str]:
    return await asyncio.to_thread(_search_passages_sync, agent_id, query, top_k)


def _insert_passage_sync(agent_id: str, text: str) -> None:
    # Try SDK first (only passages.search is known-broken; insert uses a different endpoint)
    client = _get_client()
    if client is not None:
        try:
            client.agents.passages.create(agent_id=agent_id, text=text)
            return
        except Exception as sdk_err:
            ui.warn(f"[letta_mem] SDK passage insert failed, trying httpx: {sdk_err}")

    # httpx fallback — log response body on failure for diagnosis
    try:
        r = httpx.post(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory",
            json={"text": text},
            timeout=120.0,
        )
        if not r.is_success:
            body = r.text[:300] if r.text else "(empty body)"
            ui.warn(f"[letta_mem] Passage insert HTTP {r.status_code}: {body}")
            return
    except Exception as e:
        ui.warn(f"[letta_mem] Passage insert failed: {e}")


async def insert_passage(agent_id: str, text: str) -> None:
    await asyncio.to_thread(_insert_passage_sync, agent_id, text)


# ── Built-in tool definitions (memory tools injected server-side by Letta) ───

def _get_builtin_tools_sync(agent_id: str) -> list[dict]:
    """Return the built-in tool schemas Letta injects into the LLM context."""
    client = _get_client()
    if client is None:
        return []
    try:
        result = client.agents.tools.list(agent_id=agent_id)
        items  = result if isinstance(result, list) else getattr(result, "data", getattr(result, "items", []))
        tools  = []
        for t in items:
            name = getattr(t, "name", "") or ""
            # Only built-in Letta memory/recall tools — not our client tools
            if any(kw in name for kw in ("memory", "archival", "conversation_search", "recall")):
                tools.append({
                    "name":        name,
                    "description": getattr(t, "description", "") or "",
                    "parameters":  getattr(t, "json_schema", {}) or {},
                })
        return tools
    except Exception as e:
        ui.warn(f"[letta_mem] Built-in tool list failed: {e}")
        return []


# ── Core Memory (Blocks) ──────────────────────────────────────────────────────

def _get_blocks_sync(agent_id: str) -> dict[str, str]:
    client = _get_client()
    if client is None:
        return {}
    try:
        result = client.agents.blocks.list(agent_id=agent_id)
        items  = result.items if hasattr(result, "items") else (
            result if isinstance(result, list) else getattr(result, "data", [])
        )
        return {b.label: (b.value or "") for b in items if getattr(b, "label", None)}
    except Exception as e:
        ui.warn(f"[letta_mem] Block list failed: {e}")
        return {}


async def get_core_memory(agent_id: str) -> dict[str, str]:
    return await asyncio.to_thread(_get_blocks_sync, agent_id)


def _update_block_sync(agent_id: str, label: str, value: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.agents.blocks.update(block_label=label, agent_id=agent_id, value=value)
    except Exception:
        # Block doesn't exist yet — create it on the agent
        try:
            block = client.blocks.create(label=label, value=value, limit=500)
            client.agents.blocks.attach(agent_id=agent_id, block_id=block.id)
        except Exception as e:
            ui.warn(f"[letta_mem] Block create/attach failed (label='{label}'): {e}")


async def update_core_memory(agent_id: str, label: str, value: str) -> None:
    await asyncio.to_thread(_update_block_sync, agent_id, label, value)


async def append_human_facts(agent_id: str, new_facts: list[str], limit: int = _HUMAN_LIMIT) -> None:
    """
    Append personal facts to the human core memory block.

    Overflow policy: when adding new facts would exceed `limit`, the oldest
    lines are evicted from core memory and written to archival memory first,
    so nothing is lost — it's just moved to a cheaper tier.
    """
    if not new_facts:
        return
    current = (await get_core_memory(agent_id)).get("human", "")
    lines   = [l for l in current.splitlines()
               if l.strip() and "will be populated here" not in l]
    for fact in new_facts:
        fact = fact.strip().lstrip("- ")
        if fact and fact not in current:
            lines.append(f"- {fact}")

    combined = "\n".join(lines)

    # Evict oldest lines to archival when over limit
    evicted: list[str] = []
    while len(combined) > limit and "\n" in combined:
        first_newline = combined.index("\n")
        evicted_line  = combined[:first_newline].strip()
        combined      = combined[first_newline + 1:]
        if evicted_line:
            evicted.append(evicted_line)

    combined = combined[:limit]

    if evicted:
        passage = "Evicted from core memory:\n" + "\n".join(evicted)
        await insert_passage(agent_id, passage)
        ui.warn(f"[letta_mem] Core memory full — moved {len(evicted)} fact(s) to archival")

    if combined != current:
        await update_core_memory(agent_id, "human", combined)


async def trim_human_block(agent_id: str) -> None:
    """
    Called after every inference turn to enforce _HUMAN_LIMIT on the human block.
    The LLM calls core_memory_append directly (bypassing our Python logic), so
    this post-inference trim keeps the block within its character limit.
    Oldest facts are evicted to archival — nothing is lost.
    """
    current = (await get_core_memory(agent_id)).get("human", "")
    if len(current) <= _HUMAN_LIMIT:
        return

    lines    = [l for l in current.splitlines() if l.strip()]
    combined = "\n".join(lines)
    evicted: list[str] = []
    while len(combined) > _HUMAN_LIMIT and "\n" in combined:
        idx          = combined.index("\n")
        evicted_line = combined[:idx].strip()
        combined     = combined[idx + 1:]
        if evicted_line:
            evicted.append(evicted_line)
    combined = combined[:_HUMAN_LIMIT]

    if evicted:
        passage = "Evicted from core memory (human block full):\n" + "\n".join(evicted)
        await insert_passage(agent_id, passage)
        ui.warn(f"[letta_mem] Human block trimmed — {len(evicted)} old fact(s) → archival")

    await update_core_memory(agent_id, "human", combined)


_MEMORY_TOOL_KEYWORDS = ("memory", "archival", "recall", "core_memory")

def _is_memory_tool(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _MEMORY_TOOL_KEYWORDS)


def _trim_recall_to_budget(turns: list[dict], budget_tokens: int) -> list[dict]:
    """Keep only the most recent turns that fit within budget_tokens."""
    kept, total = [], 0
    for turn in reversed(turns):
        tok = max(1, len(turn.get("content", "")) // 4)
        if total + tok > budget_tokens:
            break
        total += tok
        kept.insert(0, turn)
    return kept


def _is_system_alert(content: str) -> bool:
    """Letta injects JSON system_alert messages as user_message when context is compressed."""
    s = content.strip()
    return s.startswith("{") and "system_alert" in s


_LETTA_NOISE_PHRASES = (
    "i have received the system",
    "system alert",
    "fully operational and ready",
    "i am now fully operational",
    "previous memory tool error",
    "system update and am ready",
    "ready to continue assisting",
    "i am ready to assist you",
    "am now ready to assist",
    "please let me know how i can help you today with xazna",
    "summary of our previous interactions",
    "summary of our previous interaction",
)

def _is_letta_noise(text: str) -> bool:
    """Detect Letta's internal system-confirmation responses that leak to the user."""
    lower = text.lower()
    return any(p in lower for p in _LETTA_NOISE_PHRASES)


def _get_recent_turns_sync(conv_id: str, agent_id: str, limit: int = 300) -> list[dict]:
    """
    Fetch the most recent user+assistant messages from a specific conversation,
    returned in chronological order (oldest first).

    Uses order="desc" to get the NEWEST messages, then reverses so the caller
    sees them in correct time order. This ensures the recall display always
    shows recent context from THIS session — never from other sessions.
    """
    client = _get_client()
    if client is None:
        return []
    try:
        result = client.conversations.messages.list(
            conversation_id=conv_id,
            agent_id=agent_id,
            limit=limit,
            order="desc",   # newest first so we always get the last `limit` turns
            include_return_message_types=["user_message", "assistant_message"],
        )
        items = result.items if hasattr(result, "items") else (
            result if isinstance(result, list) else getattr(result, "data", [])
        )
        out = []
        for m in items:
            mt = getattr(m, "message_type", "")
            if mt not in ("user_message", "assistant_message"):
                continue
            role = "user" if mt == "user_message" else "assistant"
            content = getattr(m, "content", "") or ""
            if not isinstance(content, str):
                content = "".join(
                    getattr(p, "text", "") for p in (content if isinstance(content, list) else [])
                )
            if _is_system_alert(content) or _is_letta_noise(content):
                continue
            out.append({"role": role, "content": content})
        return list(reversed(out))   # back to chronological order for display
    except Exception as e:
        ui.warn(f"[letta_mem] Recent turns fetch failed: {e}")
        return []


# ── Letta Inference — routes through conversations API ────────────────────────
# Letta injects Core Memory + Recall Memory + Archival Memory automatically.
# Every turn is persisted in Letta's SQL-backed Recall tier.

def _msg_text(msg) -> str:
    """Extract plain text from a Letta message object."""
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            getattr(part, "text", "") or ""
            for part in content
        )
    return str(content)


def _call_letta_sync(client, conversation_id: str, agent_id: str, kwargs: dict):
    return client.conversations.messages.create(conversation_id, **kwargs)


def _set_effective_ctx_window_sync(agent_id: str, effective_ctx: int) -> None:
    """
    Update the agent's context_window to the effective value (full window minus
    client_tools overhead). This makes Letta allocate recall to fill exactly the
    remaining space — context is always used 100%:
        tools + system + core + recall = ctx_window
    """
    client = _get_client()
    if client is None:
        return
    try:
        client.agents.update(agent_id=agent_id, llm_config={
            "model":               _LLM_MODEL,
            "model_endpoint_type": "openai",
            "model_endpoint":      _LLM_BASE_URL,
            "context_window":      effective_ctx,
        })
    except Exception as e:
        ui.warn(f"[letta_mem] Could not update effective context window: {e}")


_APPROVAL_CONFLICT_MSG = "waiting for approval on a tool call"


def _resolve_pending_approval_sync(agent_id: str, conversation_id: str) -> bool:
    """Unblock a stuck agent by resolving its pending tool call approval via REST API."""
    # Use httpx REST directly — the SDK's include= parameter may not populate pending_approval
    try:
        r = httpx.get(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}",
            params={"include": "agent.pending_approval"},
            timeout=30.0,
        )
        if not r.is_success:
            ui.warn(f"[letta] Agent GET failed: HTTP {r.status_code}")
            return False
        data    = r.json()
        pending = data.get("pending_approval")
        if not pending:
            ui.warn("[letta] No pending_approval found in agent state")
            return False
        req_id    = pending.get("id", "")
        tc        = pending.get("tool_call") or {}
        tc_id     = tc.get("tool_call_id", "")
        tool_name = tc.get("name", "?")
        ui.warn(f"[letta] Unblocking — approving: {tool_name} (req={req_id[:12]}…)")
        # Send approval via streaming POST (drain the response)
        with httpx.stream(
            "POST",
            f"{LETTA_BASE_URL}/v1/conversations/{conversation_id}/messages",
            json={
                "agent_id": agent_id,
                "messages": [{
                    "type":                "approval",
                    "approval_request_id": req_id,
                    "approvals": [{"type": "approval", "approve": True, "tool_call_id": tc_id}],
                }],
            },
            timeout=120.0,
        ) as resp:
            for _ in resp.iter_bytes():
                pass   # drain stream
        ui.ok("[letta] Pending approval resolved — agent unblocked")
        return True
    except Exception as e:
        ui.warn(f"[letta] Recovery failed: {e}")
        return False


async def letta_inference(
    conversation_id: str,
    agent_id: str,
    user_input: str,
    mcp_server,
    dynamic_context: str = "",
    stream_callback=None,
    max_steps: int = 8,
) -> str:
    """
    Route one user turn through Letta's conversations API.

    Letta automatically:
      - injects Core Memory (persona + human blocks) into the system prompt
      - injects relevant Archival Memory passages via semantic search
      - injects recent Recall Memory turns from its SQL DB
      - stores this turn in its SQL-backed Recall DB

    We supply client_tools so Letta can call our bank APIs, RAG, and web search.
    dynamic_context carries per-turn data (date/time, user docs) not managed by Letta.
    """
    raw_tools = await mcp_server.list_tools()
    # Skip Memory_Letta_* tools — Letta manages memory internally via its own
    # built-in tools (conversation_search, memory_insert, memory_replace).
    # Sending them as client_tools wastes ~650 tokens of the context window.
    client_tools = [
        {
            "name":        t.name,
            "description": t.description or "",
            "parameters":  t.parameters or {},
        }
        for t in raw_tools
        if not t.name.startswith("Memory_Letta_")
    ]
    ui.letta_tools_list(client_tools)

    client = _get_client()
    if client is None:
        return "Memory service unavailable — Letta server not reachable."

    # Update the context block in core memory with current date/time and last document.
    # This keeps date + doc info in core memory (always visible) instead of
    # polluting every user message stored in recall memory.
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    ctx_value = f"Current date: {date_str}\nCurrent time: {time_str} (UTC+5 Tashkent)"
    if dynamic_context:
        ctx_value += f"\n{dynamic_context}"
    await asyncio.to_thread(_update_block_sync, agent_id, "context", ctx_value)

    full_input = user_input

    # Show everything Letta will inject into the LLM context this turn.
    # Fetch enough recall messages to reflect what Letta actually sends (up to 20).
    blocks, recent_turns, builtin_tools = await asyncio.gather(
        asyncio.to_thread(_get_blocks_sync, agent_id),
        asyncio.to_thread(_get_recent_turns_sync, conversation_id, agent_id, 50),
        asyncio.to_thread(_get_builtin_tools_sync, agent_id),
    )
    all_tools  = client_tools + builtin_tools
    tools_tok  = sum(
        len(t.get("name", "")) + len(t.get("description", "")) + len(json.dumps(t.get("parameters", {})))
        for t in all_tools
    ) // 4
    system_tok = len(_BANK_SYSTEM) // 4
    core_tok   = sum(len(v) for v in blocks.values()) // 4
    input_tok  = max(1, len(full_input) // 4)

    # Recall budget = whatever is left after all fixed content.
    # Cap recent_turns to this budget so the display and Letta both see ≤100%.
    recall_budget = max(200, _CTX_WINDOW - tools_tok - system_tok - core_tok - input_tok)
    recent_turns  = _trim_recall_to_budget(recent_turns, recall_budget)

    # Tell Letta the effective window = _CTX_WINDOW minus client_tools overhead.
    # Computed from _CTX_WINDOW constant, never from the agent's stored value
    # (reading it back would create a feedback loop shrinking the window each turn).
    effective_ctx = max(1500, _CTX_WINDOW - tools_tok)
    await asyncio.to_thread(_set_effective_ctx_window_sync, agent_id, effective_ctx)

    ui.letta_context_dump(_BANK_SYSTEM, blocks, recent_turns, full_input, tools_tok,
                          builtin_tool_names=[t["name"] for t in builtin_tools],
                          ctx_window=_CTX_WINDOW)

    step_kwargs: dict   = {}
    final_text: str     = ""
    is_first_step: bool = True

    for step in range(max_steps):
        if is_first_step:
            call_kwargs = {
                "agent_id":     agent_id,
                "input":        full_input,
                "client_tools": client_tools,
            }
            is_first_step = False
        else:
            call_kwargs = {
                "agent_id":     agent_id,
                "messages":     step_kwargs["tool_returns"],
                "client_tools": client_tools,
            }

        # Consume the stream inside the thread — iterating a Stream in the async
        # event loop blocks it; list() here drains the HTTP response in the pool.
        def _call(kw=call_kwargs):
            stream = client.conversations.messages.create(conversation_id, **kw)
            return list(stream)

        ui.letta_step(step + 1)

        _MAX_TIMEOUT_RETRIES = 2
        events = None
        for attempt in range(_MAX_TIMEOUT_RETRIES + 1):
            try:
                events = await asyncio.to_thread(_call)
                break
            except Exception as e:
                err_str  = str(e)
                body     = getattr(e, "body", None)
                body_str = str(body) if body else ""

                is_llm_timeout = (
                    (isinstance(body, dict) and body.get("error_type") == "llm_timeout")
                    or "llm_timeout" in err_str
                    or "llm_timeout" in body_str
                )
                is_approval_conflict = (
                    _APPROVAL_CONFLICT_MSG in err_str or
                    _APPROVAL_CONFLICT_MSG in body_str or
                    (isinstance(body, dict) and _APPROVAL_CONFLICT_MSG in str(body.get("message", "")))
                )

                if is_llm_timeout and attempt < _MAX_TIMEOUT_RETRIES:
                    wait = 3 * (attempt + 1)
                    ui.warn(f"[letta_inference] LLM timeout — retrying in {wait}s "
                            f"(attempt {attempt + 1}/{_MAX_TIMEOUT_RETRIES})…")
                    await asyncio.sleep(wait)
                    continue

                if is_approval_conflict and step == 0:
                    ui.warn("[letta_inference] Agent blocked by pending approval — attempting recovery…")
                    resolved = await asyncio.to_thread(
                        _resolve_pending_approval_sync, agent_id, conversation_id
                    )
                    if resolved:
                        try:
                            events = await asyncio.to_thread(_call)
                        except Exception as e2:
                            ui.warn(f"[letta_inference] Step {step + 1} failed after recovery: {e2}")
                    break

                ui.warn(f"[letta_inference] Step {step + 1} failed: {e}")
                ui.warn(traceback.format_exc())
                break

        if events is None:
            break

        event_types = [getattr(ev, "message_type", type(ev).__name__) for ev in events]
        ui.kv("Letta events", str(event_types) if event_types else "(empty — check Letta server)")
        if not event_types:
            ui.warn("[letta_inference] Empty event list — conversation may be in bad state; try /reset")

        pending_tool_calls  = []
        server_executed_ids = set()   # tool_call_ids Letta already ran server-side

        for ev in events:
            mt = getattr(ev, "message_type", "") or ""

            if mt == "reasoning_message":
                text = _msg_text(ev)
                if text:
                    ui.letta_thinking(text)

            elif mt == "assistant_message":
                text = _msg_text(ev)
                if text:
                    final_text = text
                    if stream_callback and not _is_letta_noise(text):
                        result = stream_callback(text)
                        if asyncio.iscoroutine(result):
                            await result

            elif mt == "tool_call_message":
                # Letta's server-side built-in tools (archival_memory_insert,
                # core_memory_append, conversation_search, …) fire this event.
                # Log immediately; the matching tool_return_message will mark
                # the tool_call_id as already-executed so we skip re-running it.
                tc = getattr(ev, "tool_call", None)
                if tc:
                    tool_name = getattr(tc, "name", "")
                    raw_args  = getattr(tc, "arguments", "{}")
                    try:
                        args_dict = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args_dict = {}
                    if _is_memory_tool(tool_name):
                        ui.memory_call(tool_name, args_dict)
                    else:
                        ui.tool_call(tool_name, json.dumps(args_dict, ensure_ascii=False))
                    if getattr(tc, "tool_call_id", None) not in {getattr(x, "tool_call_id", None) for x in pending_tool_calls}:
                        pending_tool_calls.append(tc)
                for tc2 in (getattr(ev, "tool_calls", None) or []):
                    if getattr(tc2, "tool_call_id", None) not in {getattr(x, "tool_call_id", None) for x in pending_tool_calls}:
                        pending_tool_calls.append(tc2)

            elif mt == "tool_return_message":
                # Letta already executed this tool server-side.
                # Record its tool_call_id so the execution loop won't re-run it.
                tc_id = getattr(ev, "tool_call_id", None) or ""
                if tc_id:
                    server_executed_ids.add(tc_id)
                tool_ret_name = (
                    getattr(ev, "tool_name", None)
                    or getattr(ev, "name", None)
                    or getattr(ev, "tool_call_name", None)
                    or ""
                )
                text = _msg_text(ev)
                if text:
                    label = f"[{tool_ret_name}] " if tool_ret_name else ""
                    ui.memory_result(label + text)

            elif mt == "approval_request_message":
                # Letta uses approval_request_message for client_tools — same semantics
                # as tool_call_message: execute the tool and send back a tool_return.
                tc = getattr(ev, "tool_call", None)
                if tc:
                    tc_id = getattr(tc, "tool_call_id", None)
                    if tc_id not in {getattr(x, "tool_call_id", None) for x in pending_tool_calls}:
                        pending_tool_calls.append(tc)
                        ui.warn(f"[letta] Client tool requested (via approval): {getattr(tc, 'name', '?')}")

        # Drop tools already executed server-side — only run what Letta handed back to us
        pending_tool_calls = [
            tc for tc in pending_tool_calls
            if getattr(tc, "tool_call_id", "") not in server_executed_ids
        ]

        if not pending_tool_calls:
            break

        # Execute client tools (async FastMCP calls)
        tool_returns = []
        for tc in pending_tool_calls:
            raw_args = getattr(tc, "arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception:
                args = {}

            tool_name = getattr(tc, "name", "")
            tc_id     = getattr(tc, "tool_call_id", "")

            # Memory tools → 🧠 highlight; everything else → 🔧 tool call
            if _is_memory_tool(tool_name):
                ui.memory_call(tool_name, args)
            else:
                ui.tool_call(tool_name, json.dumps(args, ensure_ascii=False))

            try:
                result = await mcp_server.call_tool(tool_name, args)
                result_str = "".join(
                    item.text if getattr(item, "type", "") == "text" else f"[{getattr(item, 'type', '')}]"
                    for item in (getattr(result, "content", []) or [])
                ) or str(result)
                status = "success"
            except Exception as e:
                result_str = f"Tool error: {e}"
                status     = "error"

            if _is_memory_tool(tool_name):
                ui.memory_result(result_str)
            else:
                ui.tool_result(result_str)
            tool_returns.append({
                "type":         "tool",
                "tool_return":  result_str,
                "status":       status,
                "tool_call_id": tc_id,
            })

        step_kwargs = {
            "tool_returns": [{"type": "tool_return", "tool_returns": tool_returns}]
        }

    # Enforce human block limit after each turn — LLM calls core_memory_append
    # directly and can exceed _HUMAN_LIMIT. Evict oldest facts to archival here.
    await trim_human_block(agent_id)

    # After memory-tool turns Letta injects a system user_message internally,
    # causing Gemma to respond to the system notification instead of the user.
    # Detect and discard that noise, then ask Letta for the real answer.
    if final_text and _is_letta_noise(final_text):
        ui.warn("[letta_inference] Letta system noise detected — requesting real answer")
        final_text = ""
        try:
            def _real_answer():
                return list(client.conversations.messages.create(
                    conversation_id,
                    agent_id=agent_id,
                    input=full_input,
                    client_tools=client_tools,
                ))
            real_events = await asyncio.to_thread(_real_answer)
            for ev in real_events:
                if getattr(ev, "message_type", "") == "assistant_message":
                    text = _msg_text(ev)
                    if text and not _is_letta_noise(text):
                        final_text = text
                        if stream_callback:
                            result = stream_callback(text)
                            if asyncio.iscoroutine(result):
                                await result
                        break
        except Exception as e:
            ui.warn(f"[letta_inference] Real answer retry failed: {e}")

    return final_text


# ── context_var binding (for concurrent requests) ─────────────────────────────

from contextvars import ContextVar

_cv_user_id:  ContextVar[str] = ContextVar("letta_user_id",  default="")
_cv_agent_id: ContextVar[str] = ContextVar("letta_agent_id", default="")


def set_letta_context(user_id: str, agent_id: Optional[str]) -> None:
    _cv_user_id.set(user_id)
    _cv_agent_id.set(agent_id or "")


def get_letta_context() -> tuple[str, str]:
    return _cv_user_id.get(), _cv_agent_id.get()
