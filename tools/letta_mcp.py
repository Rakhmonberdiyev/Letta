"""
tools/letta_mcp.py — 6 Memory Tools for the Main LLM Orchestrator.

Three tiers, full LLM control:

  CORE MEMORY  (always in context — facts about this user)
    Letta_core_memory_append(content)               → add a new fact
    Letta_core_memory_replace(old_content, new)     → update a changed fact

  RECALL MEMORY  (last N turns always in context — recent session history)
    Letta_recall_search(query)                      → keyword search older turns
    Letta_recall_insert(text)                       → explicitly save a note/summary

  ARCHIVAL MEMORY  (long-term store — only via tool, never auto-injected)
    Letta_archival_insert(content)                  → save a fact permanently
    Letta_archival_search(query)                    → semantic search all past facts

Lifecycle rules:
  • Core Memory  — updated when user facts change.  Trimmed front-first at 1500 chars.
  • Recall       — last N turns injected automatically.  Use search for older turns.
                   Use insert to save a conclusion or summary explicitly.
  • Archival     — call insert for facts that fell out of Core, or session conclusions.
                   Call search when Core + Recall don't have the answer.
"""

from fastmcp import FastMCP

from memory.letta_mem import (
    search_passages,
    insert_passage,
    get_core_memory,
    update_core_memory,
    append_human_facts,
    get_or_create_agent,
    get_letta_context,
    set_letta_context,  # re-exported so callers can import from here or letta_mem
)
from memory.session import get_session


async def _agent_id() -> str:
    user_id, aid = get_letta_context()
    if not aid:
        aid = await get_or_create_agent(user_id) or ""
    return aid


def _user_id() -> str:
    user_id, _ = get_letta_context()
    return user_id


letta_mcp = FastMCP("LettaMemory")


# ── TIER 1: CORE MEMORY ────────────────────────────────────────────────────────

@letta_mcp.tool()
async def Letta_core_memory_append(content: str) -> str:
    """
    Add a new fact about the user to Core Memory.
    Core Memory is injected into every LLM call — use this for facts that must
    always be in context: user name, active card, preferences, goals.
    Only add facts that are truly persistent and relevant to future turns.
    """
    agent_id = await _agent_id()
    if not agent_id:
        return "Core Memory unavailable."
    await append_human_facts(agent_id, [content])
    return f"Core Memory updated: added '{content[:100]}'"


@letta_mcp.tool()
async def Letta_core_memory_replace(old_content: str, new_content: str) -> str:
    """
    Update an existing fact in Core Memory by replacing old_content with new_content.
    Use this when a fact has changed — e.g. the user switched from Humo to Visa Gold,
    or their credit limit changed.  Prefer replace over append to avoid stale data.
    """
    agent_id = await _agent_id()
    if not agent_id:
        return "Core Memory unavailable."
    blocks = await get_core_memory(agent_id)
    human  = blocks.get("human", "")
    if old_content not in human:
        return f"Fact not found in Core Memory — use append instead. Searched for: '{old_content[:80]}'"
    updated = human.replace(old_content, new_content, 1)
    await update_core_memory(agent_id, "human", updated)
    return f"Core Memory updated: '{old_content[:60]}' → '{new_content[:60]}'"


# ── TIER 2: RECALL MEMORY ─────────────────────────────────────────────────────

@letta_mcp.tool()
async def Letta_recall_search(query: str) -> str:
    """
    Search Recall Memory (session history) for turns related to the query.
    The last N turns are already in your context window — use this tool only
    when you need to find something said earlier in the conversation that is
    no longer in your active context.
    """
    user_id = _user_id()
    if not user_id:
        return "Recall Memory unavailable."

    history = await get_session(user_id)
    if not history:
        return "No session history found."

    q = query.lower()
    matches = [
        f"[{m['role'].upper()}] {m['content'][:200]}"
        for m in history
        if q in (m.get("content") or "").lower()
    ]
    if not matches:
        return f"No turns in session history match: '{query}'"
    return "\n".join(matches[-10:])


@letta_mcp.tool()
async def Letta_recall_insert(text: str) -> str:
    """
    Explicitly save a note or conclusion to Recall Memory so it can be retrieved
    later in the session.  Use this to record session summaries, conclusions, or
    intermediate reasoning steps that should survive context compaction.
    Examples: 'User confirmed Visa Gold application, credit limit 7M UZS'
    """
    agent_id = await _agent_id()
    if not agent_id:
        return "Recall Memory unavailable."
    # Store as a tagged archival passage — survives context compaction,
    # retrievable via recall_search and archival_search
    await insert_passage(agent_id, f"[RECALL] {text}")
    return f"Saved to Recall Memory: '{text[:120]}'"


# ── TIER 3: ARCHIVAL MEMORY ───────────────────────────────────────────────────

@letta_mcp.tool()
async def Letta_archival_insert(content: str) -> str:
    """
    Save a fact permanently into Archival Memory (long-term semantic store).
    Use this for facts that are too detailed for Core Memory but should be
    retrievable in future sessions: full transaction history, detailed preferences,
    facts that fell out of Core Memory due to size limits.
    Archival passages are NEVER deleted — all history is always searchable.
    """
    agent_id = await _agent_id()
    if not agent_id:
        return "Archival Memory unavailable — fact not stored."
    await insert_passage(agent_id, content)
    return f"Stored in Archival Memory: '{content[:120]}'"


@letta_mcp.tool()
async def Letta_archival_search(query: str) -> str:
    """
    Semantic search over Archival Memory (long-term store of all past facts).
    Use this when Core Memory and Recall Memory don't contain the answer —
    for facts from previous sessions, historical state, or detailed user history.
    Returns the top semantically matching passages.
    """
    agent_id = await _agent_id()
    if not agent_id:
        return "Archival Memory unavailable."

    passages = await search_passages(agent_id, query, top_k=6)
    if not passages:
        return "No relevant facts found in Archival Memory."
    return "\n".join(f"- {p}" for p in passages)
