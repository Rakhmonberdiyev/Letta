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
7. Learn a personal fact about the user → save it with memory_insert.
"""

PERSONA_BLOCK = """\
You are Xazna Assistant, the AI for Xazna bank customers.
Provide accurate, grounded answers about Xazna bank products.
Respond in the user's language: Uzbek, Russian, or English.
Be concise, factual, and helpful.
Always call the matching bank tool before answering from general knowledge.
When you learn a personal fact about the user, save it with core_memory_append to the human block so it is always in context.
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
            # Clear any RequiresApproval tool_rules so the agent never waits for human approval
            try:
                client.agents.update(agent_id=agent_id, tool_rules=[])
            except Exception:
                pass
            return agent_id
    except Exception as e:
        ui.warn(f"[letta_mem] Agent list failed (will create): {e}")
    try:
        agent = client.agents.create(
            name=agent_name,
            description=f"Xazna bank assistant — user {user_id}",
            system=_BANK_SYSTEM,
            memory_blocks=[
                {"label": "persona", "value": PERSONA_BLOCK, "limit": 2000},
                {
                    "label": "human",
                    "value": (
                        f"User ID: {user_id}\n"
                        "Known preferences, facts, and context will be populated here."
                    ),
                    "limit": 1500,
                },
            ],
            llm_config={
                "model": _LLM_MODEL,
                "model_endpoint_type": "openai",
                "model_endpoint": _LLM_BASE_URL,
                "context_window": 32768,
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
        return agent.id
    except Exception as e:
        ui.warn(f"[letta_mem] Agent creation failed for {user_id}: {e}")
        return None


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
        return [
            {
                "id":            getattr(c, "id", ""),
                "title":         getattr(c, "summary", None) or "Session",
                "created_at":    str(getattr(c, "created_at", "")),
                "message_count": getattr(c, "message_count", 0),
            }
            for c in items
        ]
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
    except Exception as e:
        ui.warn(f"[letta_mem] Block update failed (label='{label}'): {e}")


async def update_core_memory(agent_id: str, label: str, value: str) -> None:
    await asyncio.to_thread(_update_block_sync, agent_id, label, value)


async def append_human_facts(agent_id: str, new_facts: list[str], limit: int = 1500) -> None:
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
    while len(combined) > limit and "\n" in combined:
        combined = combined[combined.index("\n") + 1:]
    combined = combined[:limit]
    if combined != current:
        await update_core_memory(agent_id, "human", combined)


_MEMORY_TOOL_KEYWORDS = ("memory", "archival", "recall", "core_memory")

def _is_memory_tool(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _MEMORY_TOOL_KEYWORDS)


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

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    turn_context = f"Current date: {date_str} | Time: {time_str} (UTC+5 Tashkent)"
    if dynamic_context:
        turn_context += f"\n{dynamic_context}"

    # Prepend lightweight turn context to the user input so Letta stores it in recall
    full_input = f"[{turn_context}]\n\n{user_input}" if turn_context else user_input

    # Show everything Letta will inject into the LLM context this turn
    blocks, recent_turns = await asyncio.gather(
        asyncio.to_thread(_get_blocks_sync, agent_id),
        asyncio.to_thread(_get_conv_messages_sync, conversation_id, agent_id, 6),
    )
    tools_tok = sum(
        len(t.get("name", "")) + len(t.get("description", "")) + len(json.dumps(t.get("parameters", {})))
        for t in client_tools
    ) // 4
    ui.letta_context_dump(_BANK_SYSTEM, blocks, recent_turns, full_input, tools_tok)

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
                    if stream_callback:
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
                    pending_tool_calls.append(tc)
                for tc2 in (getattr(ev, "tool_calls", None) or []):
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
