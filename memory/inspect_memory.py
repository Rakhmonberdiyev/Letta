"""
inspect_memory.py — Show everything stored in all three Letta memory tiers.

Usage:
    python inspect_memory.py [user_id]

Default user_id: 703460630
"""

import json
import sys
import os
import httpx
from letta_client import Letta

LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
_HERE       = os.path.dirname(os.path.abspath(__file__))
AGENT_STORE = os.path.join(_HERE, "letta_agents.json")
CONV_STORE  = os.path.join(_HERE, "letta_conversations.json")

client = Letta(base_url=LETTA_BASE_URL)

SEP  = "─" * 70
SEP2 = "═" * 70


def load_store(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def user_id_to_agent(user_id):
    store = load_store(AGENT_STORE)
    return store.get(user_id)


def user_id_to_conv(user_id):
    store = load_store(CONV_STORE)
    return store.get(user_id)


# ── TIER 1: Core Memory ───────────────────────────────────────────────────────

def show_system_prompt(agent_id):
    print(f"\n{SEP2}")
    print("  SYSTEM PROMPT (static, set at agent creation)")
    print(SEP2)
    try:
        agent  = client.agents.retrieve(agent_id=agent_id)
        system = getattr(agent, "system", "") or ""
        for line in system.strip().splitlines():
            print(f"    {line}")
    except Exception as e:
        print(f"  ERROR: {e}")


def show_core_memory(agent_id):
    print(f"\n{SEP2}")
    print("  TIER 1 — CORE MEMORY (always in LLM context, all sessions)")
    print(SEP2)
    try:
        result = client.agents.blocks.list(agent_id=agent_id)
        items  = result if isinstance(result, list) else getattr(result, "items", getattr(result, "data", []))
        if not items:
            print("  (no blocks found)")
            return
        for block in items:
            label = getattr(block, "label", "?")
            value = getattr(block, "value", "") or ""
            limit = getattr(block, "limit", "?")
            used  = len(value)
            print(f"\n  [{label}]  ({used} chars / {limit} limit)")
            print(SEP)
            for line in value.strip().splitlines():
                print(f"    {line}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── TIER 2: Recall Memory ─────────────────────────────────────────────────────

def show_recall_memory(agent_id, conv_id):
    print(f"\n{SEP2}")
    print("  TIER 2 — RECALL MEMORY (current conversation, SQL-backed)")
    print(SEP2)
    if not conv_id:
        print("  (no active conversation found)")
        return
    print(f"  Conversation: {conv_id}")
    _TAGS = {
        "system_message":           "SYS  ",
        "user_message":             "YOU  ",
        "assistant_message":        "BOT  ",
        "tool_call_message":        "TOOL▶",
        "tool_return_message":      "◀RET ",
        "approval_request_message": "MCP▶ ",
        "approval_response_message":"◀RESP",
        "reasoning_message":        "THINK",
        "summary_message":          "SUMRY",
    }
    try:
        result = client.conversations.messages.list(
            conversation_id=conv_id,
            agent_id=agent_id,
            limit=100,
            order="asc",
        )
        items = result.items if hasattr(result, "items") else (
            result if isinstance(result, list) else getattr(result, "data", [])
        )
        if not items:
            print("\n  (no messages yet)")
            return
        print(f"\n  {len(items)} entries stored (all types):\n")
        for i, m in enumerate(items, 1):
            mt  = getattr(m, "message_type", "?")
            tag = _TAGS.get(mt, mt[:5].upper())

            if mt in ("tool_call_message", "approval_request_message"):
                tc      = getattr(m, "tool_call", None)
                name    = getattr(tc, "name", "?") if tc else "?"
                args    = getattr(tc, "arguments", "") if tc else ""
                content = f"{name}({str(args)[:80]}{'…' if len(str(args)) > 80 else ''})"

            elif mt == "approval_response_message":
                approvals = getattr(m, "approvals", []) or []
                if approvals:
                    ap      = approvals[0]
                    status  = getattr(ap, "status", "?")
                    ret     = getattr(ap, "tool_return", "") or ""
                    content = f"[{status}] {str(ret)[:100]}{'…' if len(str(ret)) > 100 else ''}"
                else:
                    content = "(no approval data)"

            elif mt == "tool_return_message":
                tool_name = (getattr(m, "name", None)
                             or getattr(m, "tool_name", None)
                             or "?")
                status  = getattr(m, "status", "")
                ret     = getattr(m, "tool_return", "") or ""
                content = f"[{tool_name}] [{status}] {str(ret)[:100]}{'…' if len(str(ret)) > 100 else ''}"

            else:
                content = getattr(m, "content", "") or ""
                if not isinstance(content, str):
                    content = "".join(getattr(p, "text", "") for p in content)
                content = content[:120] + ("…" if len(content) > 120 else "")

            print(f"  [{i:2d}] {tag}  {content}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── TIER 3: Archival Memory ───────────────────────────────────────────────────

def show_archival_memory(agent_id):
    print(f"\n{SEP2}")
    print("  TIER 3 — ARCHIVAL MEMORY (vector store passages)")
    print(SEP2)
    try:
        r = httpx.get(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory",
            params={"limit": 100},
            timeout=30.0,
        )
        r.raise_for_status()
        data    = r.json()
        results = data.get("results", data) if isinstance(data, dict) else data
        if not results:
            print("\n  (no passages stored)")
            return
        print(f"\n  {len(results)} passages stored:\n")
        for i, p in enumerate(results, 1):
            text = p.get("text") or p.get("content") or ""
            print(f"  [{i:2d}] {text[:150]}{'…' if len(text) > 150 else ''}")
            print()
    except Exception as e:
        print(f"  ERROR: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    user_id  = sys.argv[1] if len(sys.argv) > 1 else "703460630"
    agent_id = user_id_to_agent(user_id)
    conv_id  = user_id_to_conv(user_id)

    print(f"\n{SEP2}")
    print(f"  MEMORY INSPECTOR  —  user: {user_id}")
    print(f"  Agent  : {agent_id or '(not found)'}")
    print(f"  Conv   : {conv_id  or '(not found)'}")
    print(SEP2)

    if not agent_id:
        print("\n  No agent found for this user. Run the bot first to create one.")
        sys.exit(1)

    show_system_prompt(agent_id)
    show_core_memory(agent_id)
    show_recall_memory(agent_id, conv_id)
    show_archival_memory(agent_id)

    print(f"\n{SEP2}\n")


if __name__ == "__main__":
    main()
