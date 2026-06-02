# AGENT.md — Xazna AI Agent

> Live codebase reference. Updated May 2026.

---

## What This Is

A **production AI assistant for Xazna bank** — Telegram bot, FastAPI web dashboard, and a Rich terminal UI.

The agent answers questions about Xazna bank products (deposits, credits, cards, pensions, exchange rates) in Uzbek, Russian, or English. It remembers users across sessions, searches the web and internal documents, and reasons in two modes depending on query complexity.

---

## Architecture at a Glance

```
User (Telegram / Web / CLI)
         │
    agent.py:process_turn()
         │
    Safety Guard (input)
         │
    ┌────┴────────────────────────────────────────┐
    │  Is DeepThink ON?                           │
    │  YES → system2.prepare()                   │
    │         Strategy → Critique → Evidence      │
    │         returns enriched synthesis prompt   │
    │  NO  → user input passed as-is             │
    └────┬────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────────────────┐
    │  Letta OS  —  letta_mem.letta_inference()               │
    │                                                          │
    │  Letta injects automatically:                           │
    │    • Core Memory    (persona + human blocks)            │
    │    • Recall Memory  (recent turns from SQL DB)          │
    │    • Archival Memory (semantic passage search)          │
    │                                                          │
    │  client_tools → 9 namespaces executed on our side:     │
    │    Memory · WebSearch · RAG ·                           │
    │    Deposit · Credit · Pension · Card · Admin · RealTime │
    │                                                          │
    │  Every turn stored in Letta's SQL Recall DB             │
    └────┬────────────────────────────────────────────────────┘
         │
    Grounding & Hallucination Filter
    Safety Guard (output)
         │
    Response → User
    (no separate persist step — Letta handled it)
```

---

## Memory Architecture — Three Tiers (Official Letta Architecture)

**All three tiers are delegated to the remote Letta server.** No local message storage.

| Tier | Storage | Answers |
|---|---|---|
| **Core Memory** | Letta Blocks — relational DB (Docker) | Who is this user right now? |
| **Recall Memory** | Letta Conversations — relational DB (Docker) | What did we say in this conversation? |
| **Archival Memory** | Letta Passages — vector DB (Docker) | What has this user ever told us? |

Each tier has a different job — none can replace the others.

---

### Tier 1 — Core Memory (`memory/letta_mem.py`)

Two blocks always injected into every LLM system prompt:

| Block | Purpose | Limit |
|---|---|---|
| `persona` | Static Xazna assistant identity | 2 000 chars |
| `human` | Accumulated facts about this specific user | 1 500 chars |

**How `human` grows**: after every turn, `upsert_ltm()` calls the LLM to extract facts, then `append_human_facts()` appends new ones to the block. Duplicate lines are skipped. When the block exceeds 1 500 chars, the oldest lines are trimmed from the front.

**Why this tier exists**: it guarantees the LLM always knows basic user context without needing a search call. The `human` block is the user's identity card — always present, always current.

```python
# Read
client.agents.blocks.list(agent_id)           # SyncArrayPage → .items
# Write
client.agents.blocks.update(block_label=label, agent_id=agent_id, value=value)
```

---

### Tier 2 — Archival Memory (`memory/letta_mem.py` + `memory/ltm.py`)

Infinite semantic store backed by Letta's internal vector DB. Every extracted fact is stored as a separate passage and **never deleted** — old facts are always searchable.

**Why this tier exists**: long-term semantic retrieval. "What did the user tell us 3 months ago?" The Core Memory block would overflow; the Recall buffer is gone after the session. Archival keeps everything forever.

**⚠ SDK bug — bypassed with `httpx`**: `client.agents.passages.search()` hits the wrong internal endpoint and always returns 0 results. All passage operations use direct REST calls:

```python
# Insert
httpx.post(f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory",
           json={"text": fact_text})

# Search
httpx.get(f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory/search",
          params={"query": query, "limit": top_k})
# → {"results": [{"content": "...", "timestamp": "..."}], "count": N}
```

**Fact extraction flow** (`memory/ltm.py:upsert_ltm()`):
1. After each turn, the Xazna LLM reads the conversation and outputs `FACT: ...` lines
2. Each fact is inserted as an Archival passage AND appended to the Core Memory `human` block
3. If no facts are extracted, nothing is written

---

### Tier 2 — Recall Memory (`memory/session.py` + Letta Conversations)

Every turn is stored in Letta's internal relational database via the **Conversations API** — one Letta conversation per user session.

**How it works**: `letta_inference()` calls `conversations.messages.create()`, which runs the LLM and automatically persists both the user message and the assistant response in Letta's SQL DB. No separate `save_turn()` call is needed.

**Why this tier exists**: Archival Memory does semantic search — it cannot replay the last 5 messages in order. Letta injects the recent conversation history directly from its SQL store on every inference call, without any manual context assembly.

| What Letta manages | What we manage locally |
|---|---|
| Message content — every turn, stored in SQL | `user_docs` — uploaded PDF filenames (SQLite `user_docs.db`) |
| Conversation history — queryable via `conversations.messages.list()` | `letta_conversations.json` — `user_id → active_conversation_id` pointer |
| Context window — auto-compaction when full | — |
| Session metadata — summary field per conversation | — |

`memory/session.py` is now a **thin adapter**: its public API is unchanged (so `agent.py`, `telegram_bot.py`, and `Web/backend/main.py` are unaffected), but every session/message operation delegates to the Letta SDK instead of a local file.

---

### Agent Lifecycle

One Letta agent + one Letta conversation (session) per user — created on first message, persistent forever.

- Agent IDs stored in memory cache + `memory/letta_agents.json` (survives restarts)
- Conversation IDs stored in `memory/letta_conversations.json` (one per active session)
- Agent creation: ~1–2 s, one time only. All later requests use the in-memory cache.
- `include_base_tools=True` — Letta's built-in `archival_memory_insert`, `archival_memory_search`, `core_memory_append` are available; the agent uses them proactively to save user facts
- `system=_BANK_SYSTEM` — bank tool instructions are set in the agent's base system prompt at creation
- All Letta SDK calls are wrapped with `asyncio.to_thread()` (SDK is sync; pipeline is async)

---

## Tool Architecture

All tools are mounted on `main_mcp = FastMCP("Main")` in `agent.py`. The LLM sees all tools as one flat list and decides which to call.

### Memory Tools — `tools/letta_mcp.py`

6 tools give the Main LLM explicit control over all three memory tiers.
`set_letta_context(user_id, agent_id)` is called at the start of every `process_turn()` — the tools read it via `get_letta_context()` from `letta_mem`, so concurrent requests never interfere.

| Tool | Tier | What it does |
|---|---|---|
| `Memory_Letta_core_memory_append` | Core | Add a new fact to the human block |
| `Memory_Letta_core_memory_replace` | Core | Replace a changed fact (old → new) |
| `Memory_Letta_recall_search` | Recall | Keyword search older turns not in active context |
| `Memory_Letta_recall_insert` | Recall | Explicitly save a session note or conclusion |
| `Memory_Letta_archival_insert` | Archival | Persist a fact to long-term semantic store |
| `Memory_Letta_archival_search` | Archival | Semantic search over all past facts |

**When to use each:**
- Core append/replace → user fact changed (new card, changed limit, new preference)
- Recall search → need something from earlier in conversation, no longer in context window
- Recall insert → save a session summary or conclusion before context compaction
- Archival insert → fact too detailed for Core, or a fact trimmed out of Core
- Archival search → answer not in Core or Recall; may be from a previous session

### Local Tools — `tools/mcp_server.py`

| Tool | What it does |
|---|---|
| `WebSearch_web_search` | DuckDuckGo live search |
| `RAG_rag_search` | Semantic search over user-uploaded documents (Qdrant) |

### Remote Bank Tools — `tools/remote_mcp.py`

Proxied via Nginx at `localhost:8080`:

| Namespace | Domain |
|---|---|
| `Deposit_*` | Deposit products, rates, terms |
| `Credit_*` | Loan products, amounts, rates |
| `Pension_*` | Pension payment schedules |
| `Card_*` | Card products (Visa / MC / Humo / Uzcard) |
| `Admin_*` | Bank info, branches, contacts |
| `RealTime_*` | Exchange rates, current time |

---

## Reasoning Modes

### System 1 — Fast Path (`pipeline/system1.py`)

Standard agentic loop, ≤ 6 rounds:

```
LLM call (streaming) → tool calls → execute → LLM → ... → final answer
```

Used for: simple queries, factual lookups, tool-heavy requests.

### System 2 — Deepthink (`pipeline/system2.py`)

Four phases:

```
Phase 1  Strategy JSON   {goal, approach, confidence 1-10, needs_search}
Phase 2  Parse → if confidence ≥ 8 and not needs_search → skip Phase 3
Phase 3  Critique loop (≤ 3 rounds)
           LLM critiques its own plan → if needs_data → proactive_tool_call
           if validated → break
Phase 4  Synthesis: full streaming LLM call with tool loop → final answer
```

Triggered by: `deepthink` keyword, Uzbek/Russian equivalents (`yaxshilab`, `batafsil`, etc.), or explicit `/deepthink on`.

**Proactive tool call**: the critique LLM names a specific tool; the system forces `tool_choice` to that exact tool name to guarantee well-formed arguments.

---

## Full Context Assembly (`pipeline/context_ingestion.py`)

System prompt built fresh on every request, in this order:

```
1. Base instructions     — date/time, language rules, bank tool guidance
2. Core Memory human     — accumulated user facts (always present)
3. Uploaded docs list    — filenames the user has ingested
4. Archival facts        — top-K semantically matched passages
5. Session summary       — compressed summary of older turns
--- then ---
6. Active buffer         — last 5 raw turns (verbatim)
7. User message          — current input
```

---

## Project Structure

```
Letta_full/
├── agent.py                    # process_turn() — the full pipeline
├── config.py                   # LLM client, Qdrant setup, Letta URL
├── telegram_bot.py             # aiogram v3 bot
├── ui.py                       # Rich terminal + SSE queue (ContextVar)
├── docker-compose.yml          # Letta (port 8283) + Qdrant (port 6334)
├── .env                        # TELEGRAM_BOT_TOKEN, LETTA_SERVER_PASS
│
├── memory/
│   ├── letta_mem.py            # All Letta ops: agents, conversations, passages, blocks
│   │                           # + letta_inference() — routes all inference through Letta
│   ├── session.py              # Thin adapter: delegates to Letta + local user_docs SQLite
│   ├── letta_agents.json       # user_id → agent_id cache (auto-created)
│   ├── letta_conversations.json # user_id → active conversation_id (auto-created)
│   └── user_docs.db            # User-uploaded doc filenames (auto-created)
│
├── pipeline/
│   ├── system2.py              # Deepthink: prepare() phases 1-3 → enriched prompt for Letta
│   ├── safety.py               # Input + output guards
│   └── output_processor.py     # Grounding & hallucination filter
│
├── tools/
│   ├── letta_mcp.py            # Memory_Letta_* tools (archival, core, recall)
│   ├── mcp_server.py           # WebSearch + RAG
│   ├── remote_mcp.py           # Nginx-proxied bank MCP services
│   └── ingestion.py            # PDF/TXT/MD → Qdrant
│
├── test/
│   ├── test_memory_letta.py       # 3-tier isolation smoke test (Uzbek)
│   ├── eval_longmemeval_letta.py  # 20-event Uzbek benchmark (knowledge + temporal)
│   └── benchmark_letta.py         # Latency + tool-call trace benchmark
│
└── Web/backend/main.py         # FastAPI: SSE streaming + session REST API
```

---

## Infrastructure

```yaml
# docker-compose.yml
services:
  letta:   # Core Memory (blocks) + Archival Memory (passages)
    image: letta/letta:latest
    ports: ["8283:8283"]
    environment:
      - OPENAI_API_KEY=sk-raximberdi-cmF4aW1iZXJkaQ   # Xazna key

  qdrant:  # RAG only — user-uploaded documents
    image: qdrant/qdrant
    ports: ["6334:6333"]   # 6334 avoids clash on 6333
```

```bash
docker compose up -d    # starts both services
```

The Nginx gateway (`localhost:8080`) for bank MCP services is separate infrastructure — not in this compose file.

---

## Environment Variables

```env
TELEGRAM_BOT_TOKEN=...     # from BotFather
LETTA_SERVER_PASS=         # optional; leave blank
QDRANT_PORT=6334
```

LLM config is hardcoded in `config.py`:
```python
LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY  = "sk-raximberdi-cmF4aW1iZXJkaQ"
MODEL        = "/models/gemma"
EMBED_MODEL  = "/models/embedding"   # 2048-dim
```

---

## How to Run

```bash
# Start Docker services
docker compose up -d

# Install dependencies
pip install openai letta-client httpx "qdrant-client>=1.17" \
            fastmcp "aiogram>=3.0" fastapi uvicorn rich pypdf ddgs python-dotenv

# Telegram bot
python telegram_bot.py

# Web dashboard
uvicorn Web.backend.main:app --reload --port 8000

# CLI
python agent.py

# Run tests
python test/test_memory_letta.py
python test/eval_longmemeval_letta.py
```

---

## Dependencies

```
openai
letta-client
httpx              ← needed: SDK passages.search() is broken; we use REST directly
qdrant-client>=1.17
fastmcp
aiogram>=3.0
fastapi
uvicorn
rich
pypdf
ddgs
python-dotenv
```

Removed from previous architecture:
```
redis[asyncio]   → replaced by memory/sessions_data.json
mem0ai           → replaced by Letta Archival passages
neo4j            → replaced by Letta Archival passages
```

---

## Known Issues & Gotchas

**Letta SDK `passages.search()` is broken** — always returns 0 results. Fixed in `memory/letta_mem.py` by calling `/v1/agents/{id}/archival-memory/search` directly via `httpx`. Do not revert to the SDK method.

**Qdrant API version** — `qdrant-client >= 1.17` removed `.search()`. Use `.query_points()` and read `.points` from the result.

**Letta blocks API response** — `client.agents.blocks.list()` returns a `SyncArrayPage`. Read `.items` not `.data`.

**First-user latency** — a new user's first message triggers Letta agent creation (~1–2 s). Subsequent messages are instant (in-memory cache).

**SSE concurrency** — call `ui.set_log_queue(q)` and `ui.set_tool_collector(c)` at the top of each SSE request handler. They use `ContextVar` — without this, log output from concurrent users will cross-contaminate.

**Recall Memory at scale** — backed by Letta's internal PostgreSQL (or SQLite in local mode). For enterprise scale, point Letta at a dedicated PostgreSQL instance via `docker-compose.yml`; no code changes required.
