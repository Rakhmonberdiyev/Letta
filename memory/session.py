"""
Recall Memory — thin adapter delegating to Letta's remote server.

Official Letta architecture: all message storage is handled by Letta's SQL DB
via the Conversations API.  This module is a compatibility shim that:

  - Translates the existing session API into Letta conversations.create / .list / .messages
  - Keeps user_docs in a local SQLite table (Letta has no file-tracking concept)
  - save_turn() is a deliberate no-op — Letta persists every turn automatically
    when inference is routed through letta_mem.letta_inference()
"""

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

_DOCS_DB  = str(Path(__file__).parent / "user_docs.db")
_MAX_DOCS = 10
ACTIVE_TURNS = 5


# ── Local SQLite for user_docs only ──────────────────────────────────────────

def _init_docs_db() -> None:
    with sqlite3.connect(_DOCS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_docs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                filename  TEXT NOT NULL,
                added_at  TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_docs_user ON user_docs(user_id, id)"
        )


_init_docs_db()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_docs_sync(user_id: str) -> list[str]:
    with sqlite3.connect(_DOCS_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT filename FROM user_docs WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, _MAX_DOCS),
        ).fetchall()
        return [r["filename"] for r in rows]


def _add_doc_sync(user_id: str, filename: str) -> None:
    with sqlite3.connect(_DOCS_DB) as conn:
        conn.execute(
            "INSERT INTO user_docs (user_id, filename, added_at) VALUES (?,?,?)",
            (user_id, filename, _now()),
        )
        conn.execute(
            "DELETE FROM user_docs WHERE id IN ("
            "  SELECT id FROM user_docs WHERE user_id = ? ORDER BY id DESC LIMIT -1 OFFSET ?"
            ")",
            (user_id, _MAX_DOCS),
        )


# ── Helpers to resolve agent_id + conv_id from user_id ───────────────────────

async def _agent(user_id: str) -> str | None:
    from memory.letta_mem import get_or_create_agent
    return await get_or_create_agent(user_id)


async def _conv(user_id: str) -> str | None:
    from memory.letta_mem import get_or_create_agent, get_or_create_conversation
    agent_id = await get_or_create_agent(user_id)
    if not agent_id:
        return None
    return await get_or_create_conversation(user_id, agent_id)


# ── Session ID management ─────────────────────────────────────────────────────

async def get_current_session_id(user_id: str) -> str:
    conv_id = await _conv(user_id)
    return conv_id or ""


async def create_session(user_id: str, title: str = "New Session") -> str:
    from memory.letta_mem import get_or_create_agent, create_conversation
    agent_id = await get_or_create_agent(user_id)
    if not agent_id:
        return ""
    conv_id = await create_conversation(user_id, agent_id, title)
    return conv_id or ""


async def switch_session(user_id: str, session_id: str) -> bool:
    from memory.letta_mem import _verify_conv_sync, set_active_conversation
    valid = await asyncio.to_thread(_verify_conv_sync, session_id)
    if not valid:
        return False
    await set_active_conversation(user_id, session_id)
    return True


async def get_sessions_list(user_id: str) -> list[dict]:
    from memory.letta_mem import get_or_create_agent, list_conversations
    agent_id = await get_or_create_agent(user_id)
    if not agent_id:
        return []
    return await list_conversations(agent_id)


async def get_current_session_meta(user_id: str) -> dict | None:
    from memory.letta_mem import get_conv_meta
    conv_id = await _conv(user_id)
    if not conv_id:
        return None
    return await get_conv_meta(conv_id)


# ── Message history (reads from Letta's SQL DB) ───────────────────────────────

async def get_session(user_id: str) -> list[dict]:
    from memory.letta_mem import get_or_create_agent, get_conv_messages
    agent_id = await _agent(user_id)
    conv_id  = await _conv(user_id)
    if not agent_id or not conv_id:
        return []
    return await get_conv_messages(conv_id, agent_id, limit=40)


async def get_session_history(user_id: str, session_id: str) -> list[dict]:
    from memory.letta_mem import get_or_create_agent, get_conv_messages
    agent_id = await _agent(user_id)
    if not agent_id:
        return []
    return await get_conv_messages(session_id, agent_id, limit=40)


async def get_active_buffer(user_id: str) -> list[dict]:
    history = await get_session(user_id)
    return history[-(ACTIVE_TURNS * 2):]


async def get_older_history(user_id: str) -> list[dict]:
    history = await get_session(user_id)
    cutoff  = len(history) - ACTIVE_TURNS * 2
    return history[:cutoff] if cutoff > 0 else []


async def save_turn(user_id: str, user_msg: str, assistant_msg: str) -> None:
    # Letta stores the turn automatically when inference is routed through
    # letta_mem.letta_inference() → this is intentionally a no-op.
    pass


async def clear_session(user_id: str) -> None:
    from letta_client import Letta
    from memory.letta_mem import LETTA_BASE_URL, LETTA_SERVER_PASS, _get_client
    client  = _get_client()
    conv_id = await _conv(user_id)
    if client and conv_id:
        try:
            client.conversations.reset_messages(conv_id)
        except Exception:
            pass


# ── Session summary (stored in Letta conversation summary field) ───────────────

async def get_session_summary(user_id: str) -> str:
    from memory.letta_mem import get_conv_summary
    conv_id = await _conv(user_id)
    if not conv_id:
        return ""
    return await get_conv_summary(conv_id)


async def save_session_summary(user_id: str, summary: str) -> None:
    from memory.letta_mem import update_conv_summary
    conv_id = await _conv(user_id)
    if conv_id:
        await update_conv_summary(conv_id, summary)


# ── User-uploaded documents (local SQLite) ────────────────────────────────────

async def get_user_docs(user_id: str) -> list[str]:
    return await asyncio.to_thread(_get_docs_sync, user_id)


async def add_user_doc(user_id: str, filename: str) -> None:
    await asyncio.to_thread(_add_doc_sync, user_id, filename)
