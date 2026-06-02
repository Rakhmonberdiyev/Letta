"""
benchmark_memory_trace.py — Complete Memory Tool Benchmark

Tests ALL three memory tiers via two methods:

  PHASE 1 — Direct API writes (bypasses LLM, fast)
    Archival insert  → insert_passage()       directly via REST
    Core append      → append_human_facts()   directly via REST
    Recall           → auto-populated by Letta SQL on each letta_inference() call

  PHASE 2 — LLM-driven memory tool calls (through process_turn)
    Sends crafted messages designed to trigger each Letta built-in memory tool.
    Watch the terminal — every tool call + result is printed live:

      core_memory_append      "My name is X, please remember this"
      archival_memory_insert  "Please save permanently: ..."
      archival_memory_search  "What do you know about me from before?"
      conversation_search     "What did I say at the start?"
      core_memory_replace     "My income changed — please update your memory"

  PHASE 3 — Retrieval probes (all 3 tiers)
    Verifies data written in Phase 1+2 is actually findable.
    Reports HIT/MISS + latency per tier.

Run:
  python test/benchmark_memory_trace.py
  python test/benchmark_memory_trace.py --user-id trace_abc123 --skip-ingest
"""

import asyncio
import os
import sys
import time
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import process_turn, initialize
from memory.letta_mem import (
    get_or_create_agent,
    get_or_create_conversation,
    get_core_memory,
    append_human_facts,
    insert_passage,
    search_passages,
    get_conv_messages,
    get_conv_meta,
)

# ── Terminal colors ───────────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
CYN = "\033[36m"   # core memory
BLU = "\033[34m"   # recall
YEL = "\033[33m"   # archival
GRN = "\033[32m"   # ok / timing
RED = "\033[31m"   # error
MAG = "\033[35m"   # phase headers

W62 = "─" * 62
W72 = "═" * 72


def _h1(title: str) -> None:
    print(f"\n{B}{W72}{R}\n{B}  {title}{R}\n{B}{W72}{R}")


def _h2(title: str, color: str = B) -> None:
    print(f"\n{color}{W62}{R}\n{color}  {title}{R}\n{color}{W62}{R}")


def _ms(t: float) -> str:
    ms = (time.perf_counter() - t) * 1000
    c  = GRN if ms < 500 else (YEL if ms < 2000 else RED)
    return f"{c}⏱ {ms:.0f} ms{R}"


def _tier(name: str, color: str) -> str:
    return f"{B}{color}[{name}]{R}"


def _ok(msg: str)  -> None: print(f"  {GRN}✓{R}  {msg}")
def _err(msg: str) -> None: print(f"  {RED}✗{R}  {msg}")
def _kv(k: str, v: str) -> None: print(f"  {DIM}{k:<22}{R}  {v}")


def _kw_hit(text: str, kws: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in kws)


def _diff_block(before: str, after: str) -> list[str]:
    b_lines = set(before.splitlines())
    return [l for l in after.splitlines() if l not in b_lines and l.strip()]


# ── Phase 1 scenario (5 direct-API turns) ────────────────────────────────────

SCENARIO = "Jasur Toshmatov — Visa Gold application"

DIRECT_TURNS = [
    (
        "User wants Visa Gold card application",
        ["User intent: Visa Gold card application"],
    ),
    (
        "User full name: Jasur Toshmatov. Passport: AA1234567",
        ["User full name: Jasur Toshmatov", "Passport number: AA1234567"],
    ),
    (
        "Monthly income: 8,500,000 UZS — qualifies for Visa Gold (min 3,000,000)",
        ["Monthly income: 8,500,000 UZS"],
    ),
    (
        "Co-holder added: wife Malika",
        ["Co-holder: Malika (wife)"],
    ),
    (
        "Credit limit: 15,000,000 UZS. Application submitted — approval in 3 business days",
        ["Credit limit: 15,000,000 UZS", "Application status: submitted"],
    ),
]

# ── Phase 2: LLM-driven turns targeting each memory tool ─────────────────────
# (message, description shown in terminal, tool we expect to fire)

LLM_MEMORY_TURNS = [
    (
        "My name is Jasur Toshmatov and I always prefer to chat in Uzbek. "
        "Please remember this about me.",
        "core_memory_append  — save a personal fact to Core Memory",
        "core_memory_append",
    ),
    (
        "Please save this permanently for future sessions: "
        "I applied for Visa Gold card on 2026-05-25 with credit limit 15,000,000 UZS. "
        "My passport is AA1234567.",
        "archival_memory_insert  — save a long-term fact to Archival Memory",
        "archival_memory_insert",
    ),
    (
        "What do you remember about me from our previous conversations? "
        "Please search your long-term memory.",
        "archival_memory_search  — semantic search over Archival Memory",
        "archival_memory_search",
    ),
    (
        "What was the very first thing I told you at the start of this conversation?",
        "conversation_search  — keyword search over Recall Memory (SQL DB)",
        "conversation_search",
    ),
    (
        "My monthly income actually changed to 9,000,000 UZS. "
        "Please update your memory and replace the old income figure.",
        "core_memory_replace  — update an existing fact in Core Memory",
        "core_memory_replace",
    ),
]

# ── Phase 3 retrieval probes ──────────────────────────────────────────────────

QUERIES = [
    ("Foydalanuvchining ismi va pasporti",  ["jasur", "toshmatov", "aa1234567"]),
    ("Oylik daromadi qancha",               ["8,500,000", "9,000,000", "daromad", "income"]),
    ("Hamkarta egasi kim",                  ["malika", "co-holder", "hamkarta"]),
    ("Kredit limiti qancha",                ["15,000,000", "kredit", "credit", "limit"]),
    ("Ariza holati va karta turi",          ["visa", "gold", "submitted", "tasdiqlash"]),
]


# ── Phase 1 ───────────────────────────────────────────────────────────────────

async def phase1_direct_writes(agent_id: str) -> None:
    _h1(f"PHASE 1 — DIRECT API WRITES  ({len(DIRECT_TURNS)} turns, no LLM)")
    print(f"  {DIM}Writes go directly to Letta REST API — no inference, no tool calls{R}\n")

    prev_human = (await get_core_memory(agent_id)).get("human", "")

    for i, (passage_text, facts) in enumerate(DIRECT_TURNS, 1):
        print(f"\n{B}{MAG}  TURN {i}/{len(DIRECT_TURNS)}{R}  {DIM}{passage_text[:70]}{R}")

        # — Archival insert ────────────────────────────────────────────────────
        print(f"\n  {_tier('ARCHIVAL', YEL)} insert_passage()")
        t0 = time.perf_counter()
        await insert_passage(agent_id, passage_text)
        print(f"    {YEL}→{R} {DIM}{passage_text[:80]}{R}  {_ms(t0)}")
        for fact in facts:
            t0 = time.perf_counter()
            await insert_passage(agent_id, fact)
            print(f"    {YEL}→{R} {DIM}{fact[:80]}{R}  {_ms(t0)}")

        # — Core append ────────────────────────────────────────────────────────
        print(f"\n  {_tier('CORE', CYN)} append_human_facts()")
        t0 = time.perf_counter()
        await append_human_facts(agent_id, facts)
        ms_w = _ms(t0)
        new_human = (await get_core_memory(agent_id)).get("human", "")
        new_lines = _diff_block(prev_human, new_human)
        if new_lines:
            for line in new_lines:
                print(f"    {CYN}+{R} {line.lstrip('- ')[:90]}")
        else:
            print(f"    {DIM}(no new lines — fact already exists){R}")
        print(f"    {DIM}write {ms_w}  block now {len(new_human)} chars{R}")
        prev_human = new_human

        # — Recall ─────────────────────────────────────────────────────────────
        print(f"\n  {_tier('RECALL', BLU)} auto-stored by Letta SQL on each letta_inference() call")

        await asyncio.sleep(0.2)


# ── Phase 2 ───────────────────────────────────────────────────────────────────

async def phase2_llm_memory_tools(user_id: str) -> None:
    _h1(f"PHASE 2 — LLM-DRIVEN MEMORY TOOL CALLS  ({len(LLM_MEMORY_TURNS)} turns)")
    print(f"  {DIM}Each turn runs through process_turn() — tool calls printed live below.{R}")
    print(f"  {DIM}Watch for: {CYN}🧠 MEMORY →{R}{DIM} and {MAG}🔧 TOOL CALL →{R}{DIM} panels.{R}\n")

    for i, (msg, desc, expected) in enumerate(LLM_MEMORY_TURNS, 1):
        print(f"\n{B}{MAG}{'━'*72}{R}")
        print(f"{B}{MAG}  MEMORY TURN {i}/{len(LLM_MEMORY_TURNS)}{R}")
        print(f"  {DIM}Goal:{R} {desc}")
        print(f"  {GRN}Expected tool:{R}  {B}{expected}{R}")
        print(f"{B}{MAG}{'━'*72}{R}\n")

        try:
            await process_turn(msg, user_id, deepthink=False)
        except Exception as e:
            _err(f"process_turn failed: {e}")
            import traceback; traceback.print_exc()

        await asyncio.sleep(0.5)


# ── Phase 3 ───────────────────────────────────────────────────────────────────

async def phase3_retrieval_probes(agent_id: str, conv_id: str) -> None:
    _h1(f"PHASE 3 — RETRIEVAL PROBES  ({len(QUERIES)} queries × 3 tiers)")

    # Core Memory snapshot
    _h2("CORE MEMORY — full human block", CYN)
    t0 = time.perf_counter()
    blocks = await get_core_memory(agent_id)
    elapsed = _ms(t0)
    human = blocks.get("human", "(empty)")
    print(f"\n  {_tier('CORE', CYN)}  {len(human)} chars  {elapsed}\n")
    for line in human.splitlines():
        if line.strip():
            print(f"    {CYN}│{R} {line}")

    # Recall snapshot
    _h2("RECALL MEMORY — Letta SQL DB", BLU)
    t0 = time.perf_counter()
    messages = await get_conv_messages(conv_id, agent_id, limit=40)
    elapsed  = _ms(t0)
    meta     = await get_conv_meta(conv_id)
    print(f"\n  {_tier('RECALL', BLU)}  {len(messages)} messages  {elapsed}")
    print(f"  {DIM}Conv {conv_id[:28]}…  created {meta.get('created_at','?')[:19]}{R}\n")
    for m in messages[-6:]:
        role  = "USER " if m["role"] == "user" else "AGENT"
        color = GRN if m["role"] == "user" else BLU
        print(f"  {color}[{role}]{R}  {m['content'][:90]}")

    # Per-query probes
    _h2("QUERY PROBES  (HIT / MISS per tier)", B)
    total_c = total_a = total_r = 0.0
    hits_c = hits_a = hits_r = 0
    n = len(QUERIES)

    for qi, (query, kws) in enumerate(QUERIES, 1):
        print(f"\n  {B}Query {qi}/{n}:{R}  \"{query}\"")
        print(f"  {DIM}keywords: {kws}{R}")

        # Core
        t0 = time.perf_counter()
        blocks = await get_core_memory(agent_id)
        t_c    = time.perf_counter() - t0
        total_c += t_c
        human  = blocks.get("human", "")
        hit    = _kw_hit(human, kws)
        if hit: hits_c += 1
        mk = f"{GRN}✓ HIT{R}" if hit else f"{RED}✗ MISS{R}"
        print(f"\n    {_tier('CORE', CYN)}  {mk}  {GRN}⏱ {t_c*1000:.0f} ms{R}  (block read — no search)")
        if hit:
            for line in [l for l in human.splitlines() if _kw_hit(l, kws)][:2]:
                print(f"      {CYN}│{R} {line.lstrip('- ')[:90]}")

        # Archival
        t0       = time.perf_counter()
        passages = await search_passages(agent_id, query, top_k=5)
        t_a      = time.perf_counter() - t0
        total_a += t_a
        combined = "\n".join(passages)
        hit      = _kw_hit(combined, kws)
        if hit: hits_a += 1
        mk = f"{GRN}✓ HIT{R}" if hit else f"{RED}✗ MISS{R}"
        print(f"\n    {_tier('ARCHIVAL', YEL)}  {mk}  {GRN}⏱ {t_a*1000:.0f} ms{R}  ({len(passages)} passages)")
        for p in passages[:3]:
            bullet = f"{GRN}●{R}" if _kw_hit(p, kws) else f"{DIM}○{R}"
            print(f"      {YEL}│{R} {bullet} {p[:88]}")

        # Recall
        t0      = time.perf_counter()
        history = await get_conv_messages(conv_id, agent_id, limit=40)
        t_r     = time.perf_counter() - t0
        total_r += t_r
        matched = [m for m in history if _kw_hit(m.get("content", ""), kws)]
        hit     = bool(matched)
        if hit: hits_r += 1
        mk = f"{GRN}✓ HIT{R}" if hit else f"{RED}✗ MISS{R}"
        print(f"\n    {_tier('RECALL', BLU)}  {mk}  {GRN}⏱ {t_r*1000:.0f} ms{R}  ({len(matched)} matching turn(s))")
        for m in matched[:2]:
            role = "USER " if m["role"] == "user" else "AGENT"
            print(f"      {BLU}│{R} [{role}] {m['content'][:85]}")

    # Summary
    _h1("BENCHMARK SUMMARY")
    print(f"""
  {B}Retrieval performance ({n} queries):{R}

  {_tier('CORE', CYN)}
    Avg latency : {GRN}{total_c/n*1000:.0f} ms{R}  (block read — always fastest, no search)
    Hit rate    : {GRN if hits_c==n else YEL}{hits_c}/{n}{R}  ({hits_c/n*100:.0f}%)

  {_tier('ARCHIVAL', YEL)}
    Avg latency : {GRN}{total_a/n*1000:.0f} ms{R}  (semantic vector search via Qdrant)
    Hit rate    : {GRN if hits_a==n else YEL}{hits_a}/{n}{R}  ({hits_a/n*100:.0f}%)

  {_tier('RECALL', BLU)}
    Avg latency : {GRN}{total_r/n*1000:.0f} ms{R}  (SQL DB full-message fetch)
    Hit rate    : {GRN if hits_r==n else YEL}{hits_r}/{n}{R}  ({hits_r/n*100:.0f}%)
""")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(user_id: str, skip_ingest: bool) -> None:
    started = datetime.now(timezone.utc)

    _h1("COMPLETE MEMORY TOOL BENCHMARK")
    _kv("Scenario", SCENARIO)
    _kv("User ID",  user_id)
    _kv("Started",  started.strftime("%Y-%m-%d %H:%M:%S UTC"))
    _kv("Phase 1",  f"{len(DIRECT_TURNS)} direct API turns")
    _kv("Phase 2",  f"{len(LLM_MEMORY_TURNS)} LLM-driven memory turns")
    _kv("Phase 3",  f"{len(QUERIES)} retrieval probes × 3 tiers")
    print(f"\n  Tiers: {_tier('CORE', CYN)} {_tier('RECALL', BLU)} {_tier('ARCHIVAL', YEL)}")

    # ── Init model ────────────────────────────────────────────────────────────
    await initialize()

    # ── Agent + conversation ──────────────────────────────────────────────────
    _h2("SETUP — Agent + Conversation", CYN)
    t0 = time.perf_counter()
    agent_id = await get_or_create_agent(user_id)
    if not agent_id:
        _err("Letta server unreachable — start Docker: docker compose up -d")
        return
    _ok(f"Agent {agent_id[:32]}…  {_ms(t0)}")

    t0 = time.perf_counter()
    conv_id = await get_or_create_conversation(user_id, agent_id)
    if not conv_id:
        _err("Could not create conversation")
        return
    _ok(f"Conv  {conv_id[:32]}…  {_ms(t0)}")

    if not skip_ingest:
        await phase1_direct_writes(agent_id)
        await phase2_llm_memory_tools(user_id)

    await phase3_retrieval_probes(agent_id, conv_id)

    print(f"\n  {B}Agent :{R} {agent_id}")
    print(f"  {B}Conv  :{R} {conv_id}")
    print(f"  {B}User  :{R} {user_id}")
    print(f"\n  {DIM}Re-probe:  python test/benchmark_memory_trace.py "
          f"--user-id {user_id} --skip-ingest{R}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Complete memory tool benchmark")
    parser.add_argument("--user-id",     default=None,
                        help="Reuse existing user (required with --skip-ingest)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip Phase 1+2 writes, only run Phase 3 retrieval probes")
    parser.add_argument("--phase2-only", action="store_true",
                        help="Run only Phase 2 (LLM-driven memory tool calls)")
    args = parser.parse_args()

    if args.skip_ingest and not args.user_id:
        print("Error: --skip-ingest requires --user-id")
        sys.exit(1)

    user_id = args.user_id or f"trace_{uuid.uuid4().hex[:8]}"

    if args.phase2_only:
        async def _p2():
            await initialize()
            agent_id = await get_or_create_agent(user_id)
            if not agent_id:
                _err("Letta server unreachable")
                return
            await get_or_create_conversation(user_id, agent_id)
            await phase2_llm_memory_tools(user_id)
        asyncio.run(_p2())
    else:
        asyncio.run(main(user_id, skip_ingest=args.skip_ingest))
