"""
benchmark_letta.py — Comprehensive Letta Memory Rules Benchmark

Tests all three Letta OS memory tiers + overflow eviction:

  TIER 1 — Core Memory (always in LLM context via memory blocks)
    Rule 1.1  core_memory_append / memory_insert  — called when a new user fact is learned
    Rule 1.2  core facts visible without any tool call (always-in-context)
    Rule 1.3  core_memory_replace / memory_replace — called when a fact changes
    Rule 1.4  updated fact visible without any tool call

  TIER 2 — Recall Memory (recent N turns in context; older turns via search)
    Rule 2.1  recent turn recall requires NO conversation_search
    Rule 2.2  distant turn recall DOES trigger conversation_search

    Context Window formula verified by tight 2048-token window:
      Recall = Context Window − Core − System − User Query − Tools

  TIER 3 — Archival Memory (long-term vector store)
    Rule 3.1  archival_memory_insert called when agent is asked to archive data
    Rule 3.2  archival_memory_search called when agent retrieves pre-seeded archival data
              (data seeded via API before the conversation — not in recall at all)

  CORE MEMORY OVERFLOW (cross-tier eviction — Rule 4)
    Rule 4.1  core_memory_append called for first personal fact
    Rule 4.2  core_memory_append called for second fact (block now overflows limit)
    Rule 4.3  archival_memory_search triggered when querying an evicted fact
              (fact is present ONLY in archival — evicted from core when block was full)

Fixes applied vs v1:
  • Tool names: both old (memory_insert/replace) and new (core_memory_append/replace) accepted
  • Double-counting: _parse_events deduplicates per event using tool_call_id
  • Latency: stream is drained INSIDE the timer so real inference time is measured
  • Recall flood: context window = 2048, 7 noise turns instead of 3
  • Archival search: data seeded via client.agents.passages.create() before conversation
    so it exists ONLY in the vector store, not in any conversation recall

Run:
  python test/benchmark_letta.py
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from letta_client import Letta

# ── Config ─────────────────────────────────────────────────────────────────────
_LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
_LLM_MODEL    = "/models/gemma"
_EMBED_MODEL  = "/models/embedding"
_EMBED_DIM    = 2048

# 2048-token window forces conversation_search sooner (vs 4096 which was too large).
# With system+tools overhead ~1400 tokens only ~650 tokens remain for conversation —
# that is roughly 3–4 short turn-pairs, enough to push the planted code out.
_CTX_WINDOW = 2048

client = Letta(base_url="http://localhost:8283", timeout=300.0)

# ── Memory tool names — both Letta API versions ────────────────────────────────
# Older Letta server: memory_insert / memory_replace
# Newer Letta server: core_memory_append / core_memory_replace
MEMORY_TOOLS = {
    "core_memory_append",   # new name
    "memory_insert",        # old name — same operation
    "core_memory_replace",  # new name
    "memory_replace",       # old name — same operation
    "conversation_search",
    "archival_memory_insert",
    "archival_memory_search",
}

# Core-memory write tools (either naming)
_CORE_APPEND  = ["core_memory_append", "memory_insert"]
_CORE_REPLACE = ["core_memory_replace", "memory_replace"]

# ── Noise payload — forces long LLM responses to consume context tokens ─────────
_NOISE = (
    "Please write a full Python module (200+ lines) implementing a thread-safe LRU cache "
    "with TTL expiry, hit/miss statistics, and an async refresh callback. Include type hints, "
    "unit tests using pytest, and a benchmark comparing it to functools.lru_cache. "
    "Add detailed docstrings. This is purely a coding exercise unrelated to any previous topic."
)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn: int
    section: str
    prompt: str
    tools_called: list[str] = field(default_factory=list)   # deduplicated
    response_text: str = ""
    total_latency_s: float = 0.0

    required_tools: list[str] = field(default_factory=list)  # one must be called
    forbidden_tools: list[str] = field(default_factory=list) # none may be called
    expected_in_response: Optional[str] = None

    tool_pass: Optional[bool] = None
    answer_pass: Optional[bool] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_events(events) -> tuple[list[str], str]:
    """
    Return (tools_called, response_text).

    Fix: deduplicate tool names per-event using tool_call_id so that a single
    tool invocation is not double-counted when both .tool_call and .tool_calls[]
    are populated on the same event object.
    """
    tools: list[str] = []
    seen_ids: set[str] = set()
    text = ""

    for ev in events:
        mt = getattr(ev, "message_type", None)

        if mt == "tool_call_message":
            # Primary field
            tc = getattr(ev, "tool_call", None)
            if tc:
                tc_id = getattr(tc, "tool_call_id", None) or id(tc)
                if tc_id not in seen_ids:
                    seen_ids.add(tc_id)
                    tools.append(getattr(tc, "name", ""))
            # Array field (avoid double-count via same id)
            for tc2 in (getattr(ev, "tool_calls", None) or []):
                tc_id = getattr(tc2, "tool_call_id", None) or id(tc2)
                if tc_id not in seen_ids:
                    seen_ids.add(tc_id)
                    tools.append(getattr(tc2, "name", ""))

        elif mt == "assistant_message":
            content = getattr(ev, "content", "") or ""
            if isinstance(content, list):
                content = "".join(getattr(p, "text", "") for p in content)
            text += str(content)

    return tools, text


def _send(conv_id: str, agent_id: str, prompt: str) -> tuple[list[str], str, float]:
    """
    Send one turn; return (tools_called, response_text, latency_s).

    Fix: the stream is fully drained INSIDE the timer so the measured latency
    covers real LLM inference, not just the HTTP handshake.
    """
    t0 = time.perf_counter()

    response = client.conversations.messages.create(
        conv_id,
        agent_id=agent_id,
        input=prompt,
    )

    # Drain stream inside timer window
    events = getattr(response, "messages", None)
    if events is None:
        try:
            events = list(response)
        except Exception:
            events = []

    latency = time.perf_counter() - t0   # ← real end-to-end latency

    tools, text = _parse_events(events)
    return tools, text, latency


def _evaluate(r: TurnResult) -> None:
    called = set(r.tools_called)
    if r.required_tools:
        r.tool_pass = bool(called.intersection(r.required_tools))
    elif r.forbidden_tools:
        r.tool_pass = not bool(called.intersection(r.forbidden_tools))
    else:
        r.tool_pass = None

    if r.expected_in_response is not None:
        r.answer_pass = r.expected_in_response.lower() in r.response_text.lower()
    else:
        r.answer_pass = None


def _icon(v: Optional[bool]) -> str:
    if v is True:  return "PASS"
    if v is False: return "FAIL"
    return "N/A "


def _print_result(r: TurnResult) -> None:
    print(f"  [{r.turn:2d}] {r.section}")
    mem_called   = [t for t in r.tools_called if t in MEMORY_TOOLS]
    other_called = [t for t in r.tools_called if t not in MEMORY_TOOLS]
    print(f"        Memory tools : {mem_called or ['(none)']}")
    if other_called:
        print(f"        Other  tools : {other_called}")
    if r.required_tools:
        print(f"        Tool check   : [{_icon(r.tool_pass)}]  need one of {r.required_tools}")
    elif r.forbidden_tools:
        print(f"        Tool check   : [{_icon(r.tool_pass)}]  must NOT call {r.forbidden_tools}")
    if r.expected_in_response is not None:
        snippet = r.response_text[:100].replace("\n", " ")
        print(f"        Answer check : [{_icon(r.answer_pass)}]  looking for '{r.expected_in_response}'")
        print(f"        Response     : {snippet}…")
    print(f"        Latency      : {r.total_latency_s:.2f}s")
    print()


# ── Section 1: Core Memory ─────────────────────────────────────────────────────

def _core_memory_tests(conv_id: str, agent_id: str, start: int) -> list[TurnResult]:
    results: list[TurnResult] = []
    turn = start

    # 1.1 — New fact → agent must call core_memory_append (or memory_insert)
    tools, text, lat = _send(conv_id, agent_id,
        "My name is Temur and I am a senior backend engineer at a fintech company. "
        "Please save this to your memory.")
    r = TurnResult(turn=turn, section="1.1  Core Append — new user fact",
                   prompt="My name is Temur…",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=_CORE_APPEND)
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # 1.2 — Core block is always in LLM context — no search needed
    tools, text, lat = _send(conv_id, agent_id,
        "What do you know about me? Tell me everything you have stored.")
    r = TurnResult(turn=turn, section="1.2  Core Persistence — no tool needed",
                   prompt="What do you know about me?",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   forbidden_tools=["conversation_search", "archival_memory_search"],
                   expected_in_response="Temur")
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # 1.3 — Fact changes → agent must call core_memory_replace (or memory_replace)
    tools, text, lat = _send(conv_id, agent_id,
        "I changed my name — from now on call me Bobur, not Temur. "
        "Please update your memory to replace my old name.")
    r = TurnResult(turn=turn, section="1.3  Core Replace — fact update",
                   prompt="My name changed to Bobur…",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=_CORE_REPLACE)
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # 1.4 — Updated fact still visible without any search
    tools, text, lat = _send(conv_id, agent_id,
        "What is my name?")
    r = TurnResult(turn=turn, section="1.4  Core Replace Verify — updated name in context",
                   prompt="What is my name?",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   forbidden_tools=["conversation_search", "archival_memory_search"],
                   expected_in_response="Bobur")
    _evaluate(r); _print_result(r); results.append(r)

    return results


# ── Section 2: Recall Memory ───────────────────────────────────────────────────

def _recall_memory_tests(conv_id: str, agent_id: str, start: int) -> list[TurnResult]:
    """
    Plant a code, confirm immediate recall, then flood with 7 noise turns
    (each producing a long response) to exhaust the 2048-token context window,
    then verify conversation_search is triggered for the distant message.
    """
    results: list[TurnResult] = []
    turn = start

    # 2.1 — Plant the code
    tools, text, lat = _send(conv_id, agent_id,
        "IMPORTANT: Remember this secret system code — RECALL_CODE=VX-7734. "
        "You will need it later.")
    r = TurnResult(turn=turn, section="2.1  Recall Plant — code stored in recall",
                   prompt="RECALL_CODE=VX-7734",
                   tools_called=tools, response_text=text, total_latency_s=lat)
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # 2.2 — Immediate recall: code is still in the sliding window
    tools, text, lat = _send(conv_id, agent_id,
        "What was the RECALL_CODE I just told you?")
    r = TurnResult(turn=turn, section="2.2  Recall Immediate — no search needed",
                   prompt="What was the RECALL_CODE?",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   forbidden_tools=["conversation_search"],
                   expected_in_response="VX-7734")
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # 2.3–2.9 — 7 noise turns; each produces a long response to consume context tokens.
    # With a 2048-token window and ~1400 tokens of fixed overhead (system + tools +
    # core memory blocks), only ~650 tokens remain for conversation history.
    # 7 turns × ~200 tokens/turn fills that gap and forces VX-7734 out of the window.
    for i in range(7):
        tools, text, lat = _send(conv_id, agent_id, _NOISE)
        r = TurnResult(turn=turn, section=f"2.{3+i}  Recall Flood #{i+1} — context noise",
                       prompt="(noise)",
                       tools_called=tools, response_text=text, total_latency_s=lat)
        _evaluate(r); _print_result(r); results.append(r); turn += 1

    # 2.10 — Distant recall: code is outside the context window → must search
    tools, text, lat = _send(conv_id, agent_id,
        "I need that RECALL_CODE from the beginning of our conversation. "
        "Please search your conversation history to find it.")
    r = TurnResult(turn=turn, section="2.10 Recall Search — conversation_search required",
                   prompt="Find RECALL_CODE from early conversation",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=["conversation_search"],
                   expected_in_response="VX-7734")
    _evaluate(r); _print_result(r); results.append(r)

    return results


# ── Section 3: Archival Memory ─────────────────────────────────────────────────

def _seed_archival(agent_id: str) -> None:
    """
    Insert a fact directly into the vector store via the Letta API.

    This bypasses the conversation entirely so the data is ONLY in archival
    memory — it will never appear in any conversation's recall (SQL store).
    That forces the agent to call archival_memory_search to find it.
    """
    try:
        client.agents.passages.create(
            agent_id=agent_id,
            text=(
                "Project Nexus — pre-seeded archival record: "
                "Launch date July 15, 2026. Budget $500K. Lead: Kamila Yusupova. "
                "Status: approved. Priority: critical."
            ),
        )
        print("  [seed] Project Nexus fact inserted into archival (vector store) via API.")
    except Exception as e:
        print(f"  [seed] Archival seed failed: {e}")


def _archival_memory_tests(agent_id: str, start: int) -> list[TurnResult]:
    """
    Archival insert test uses the same conversation as the rest of the benchmark.
    Archival search test uses a FRESH conversation — so the "Project Nexus" fact
    (pre-seeded via API) is only in the vector store and NOT in any recall DB.
    """
    results: list[TurnResult] = []
    turn = start

    # 3.1 — Via-conversation insert: agent should call archival_memory_insert
    conv_insert = client.conversations.create(agent_id=agent_id)
    tools, text, lat = _send(conv_insert.id, agent_id,
        "Please archive this for long-term storage using your archival memory tool: "
        "'Budget report Q1-2026: total spend $1.2M, savings $180K, flagged items: none.'")
    r = TurnResult(turn=turn, section="3.1  Archival Insert — agent called via conversation",
                   prompt="Archive budget report…",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=["archival_memory_insert"])
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # Pre-seed the "Project Nexus" fact directly (not via any conversation)
    _seed_archival(agent_id)
    print()

    # 3.2 — Fresh conversation: Project Nexus is ONLY in archival, not in recall
    conv_search = client.conversations.create(agent_id=agent_id)
    tools, text, lat = _send(conv_search.id, agent_id,
        "What do you have stored in your archival memory about Project Nexus? "
        "Please search your archival memory to find all details.")
    r = TurnResult(turn=turn, section="3.2  Archival Search — fresh conv, data only in vector store",
                   prompt="Search archival for Project Nexus",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=["archival_memory_search"],
                   expected_in_response="Nexus")
    _evaluate(r); _print_result(r); results.append(r)

    return results


# ── Section 4: Core Memory Overflow → Archival Eviction ───────────────────────

_OVERFLOW_LIMIT = 80   # chars — tiny so two personal facts force eviction


def _set_block_limit(agent_id: str, label: str, limit: int) -> None:
    try:
        blocks = client.agents.blocks.list(agent_id=agent_id)
        items  = blocks if isinstance(blocks, list) else getattr(blocks, "data", getattr(blocks, "items", []))
        for blk in items:
            if getattr(blk, "label", "") == label:
                client.blocks.update(block_id=blk.id, limit=limit)
                print(f"  [setup] '{label}' block limit → {limit} chars.")
                return
    except Exception as e:
        print(f"  [setup] Block limit update failed: {e}")


def _get_block_value(agent_id: str, label: str) -> str:
    try:
        blocks = client.agents.blocks.list(agent_id=agent_id)
        items  = blocks if isinstance(blocks, list) else getattr(blocks, "data", getattr(blocks, "items", []))
        for blk in items:
            if getattr(blk, "label", "") == label:
                return getattr(blk, "value", "") or ""
    except Exception:
        pass
    return ""


def _evict_overflow(agent_id: str, label: str, limit: int) -> list[str]:
    """
    Replicate production trim_human_block logic in the benchmark:
    evict oldest lines from a core memory block to archival if over limit.
    Returns the list of evicted strings.
    """
    try:
        blocks  = client.agents.blocks.list(agent_id=agent_id)
        items   = blocks if isinstance(blocks, list) else getattr(blocks, "data", getattr(blocks, "items", []))
        blk_obj = next((b for b in items if getattr(b, "label", "") == label), None)
        if blk_obj is None:
            return []
        value = getattr(blk_obj, "value", "") or ""
        if len(value) <= limit:
            return []
        lines    = [l for l in value.splitlines() if l.strip()]
        combined = "\n".join(lines)
        evicted: list[str] = []
        while len(combined) > limit and "\n" in combined:
            idx          = combined.index("\n")
            evicted_line = combined[:idx].strip()
            combined     = combined[idx + 1:]
            if evicted_line:
                evicted.append(evicted_line)
        combined = combined[:limit]
        if evicted:
            passage = "Evicted from core memory (human block full):\n" + "\n".join(evicted)
            client.agents.passages.create(agent_id=agent_id, text=passage)
            print(f"  [evict] {len(evicted)} line(s) → archival: {evicted[:2]}"
                  f"{'…' if len(evicted) > 2 else ''}")
        try:
            client.agents.blocks.update(block_label=label, agent_id=agent_id, value=combined)
        except Exception:
            client.blocks.update(block_id=blk_obj.id, value=combined)
        return evicted
    except Exception as e:
        print(f"  [evict] Failed: {e}")
        return []


def _core_overflow_tests(agent_id: str, start: int) -> list[TurnResult]:
    """
    Section 4 — Core Memory Overflow → Archival Eviction → Archival Retrieval.

    Rule: when a core memory block fills up, the oldest facts are evicted to
    archival (not discarded). When the user later queries an evicted fact,
    the agent must call archival_memory_search — the fact is no longer in core.

    _evict_overflow() replicates the production trim_human_block() logic so the
    benchmark is self-contained (no import from letta_mem.py).

    Steps:
      4.1  Store first personal fact → core_memory_append required
      4.2  Store second fact that together with the first overflows the tiny block
           (eviction helper runs after each turn to mirror production behaviour)
      4.3  Query the evicted first fact → archival_memory_search required
    """
    results: list[TurnResult] = []
    turn = start

    # Shrink the human block so two facts together exceed the limit
    _set_block_limit(agent_id, "human", _OVERFLOW_LIMIT)
    print()

    conv = client.conversations.create(agent_id=agent_id)
    print(f"  Conv for overflow tests: {conv.id[:20]}…")
    print()

    # 4.1 — First personal fact (may or may not overflow yet on its own)
    tools, text, lat = _send(conv.id, agent_id,
        "My close university friend is Kamol Tursunov, a software engineer. "
        "Please save this to your core memory.")
    r = TurnResult(turn=turn,
                   section="4.1  Overflow — first fact saved to core",
                   prompt="Friend Kamol Tursunov…",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=_CORE_APPEND)
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    _evict_overflow(agent_id, "human", _OVERFLOW_LIMIT)

    # 4.2 — Second fact — together both facts overflow the small block
    tools, text, lat = _send(conv.id, agent_id,
        "I am a senior data scientist specialising in NLP and LLMs. Update my profile.")
    r = TurnResult(turn=turn,
                   section="4.2  Overflow — second fact causes block overflow",
                   prompt="Senior data scientist…",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=_CORE_APPEND)
    _evaluate(r); _print_result(r); results.append(r); turn += 1

    # Force eviction: oldest facts (Kamol) moved to archival
    _evict_overflow(agent_id, "human", _OVERFLOW_LIMIT)
    remaining = _get_block_value(agent_id, "human")
    print(f"  [4.2] After eviction — human block ({len(remaining)} chars): "
          f"'{remaining[:60]}{'…' if len(remaining) > 60 else ''}'")
    print()

    # 4.3 — Query the evicted fact: Kamol is NOT in core memory → must search archival
    tools, text, lat = _send(conv.id, agent_id,
        "What is the name of the university friend I told you about earlier? "
        "Please search your archival memory to find the details.")
    r = TurnResult(turn=turn,
                   section="4.3  Overflow — evicted fact via archival_memory_search",
                   prompt="Who is my university friend? (evicted to archival)",
                   tools_called=tools, response_text=text, total_latency_s=lat,
                   required_tools=["archival_memory_search"],
                   expected_in_response="Kamol")
    _evaluate(r); _print_result(r); results.append(r)

    return results


# ── Scoring ────────────────────────────────────────────────────────────────────

def _report(all_results: list[TurnResult]) -> None:
    tool_checks   = [r for r in all_results if r.tool_pass   is not None]
    answer_checks = [r for r in all_results if r.answer_pass is not None]
    tool_passed   = sum(1 for r in tool_checks   if r.tool_pass)
    answer_passed = sum(1 for r in answer_checks if r.answer_pass)
    total_latency = sum(r.total_latency_s for r in all_results)
    avg_latency   = total_latency / len(all_results) if all_results else 0.0

    print("=" * 62)
    print("SCORECARD")
    print("=" * 62)
    print(f"  Tool-call correctness : {tool_passed}/{len(tool_checks)}")
    print(f"  Answer correctness    : {answer_passed}/{len(answer_checks)}")
    print(f"  Total latency         : {total_latency:.2f}s  "
          f"(avg {avg_latency:.2f}s / turn, {len(all_results)} turns)")
    print()

    print("Per-rule results:")
    for r in all_results:
        t_icon = f"[{_icon(r.tool_pass)}]"   if r.tool_pass   is not None else "      "
        a_icon = f"[{_icon(r.answer_pass)}]" if r.answer_pass is not None else "      "
        print(f"  Turn {r.turn:2d}  {r.section:<52s}  "
              f"tool={t_icon}  answer={a_icon}  {r.total_latency_s:.2f}s")
    print()

    # Slowest turns that called a memory tool
    mem_turns = sorted(
        [r for r in all_results if any(t in MEMORY_TOOLS for t in r.tools_called)],
        key=lambda r: r.total_latency_s, reverse=True,
    )
    if mem_turns:
        print("Slowest memory-tool turns:")
        for r in mem_turns[:5]:
            mt = [t for t in r.tools_called if t in MEMORY_TOOLS]
            print(f"  {r.total_latency_s:.2f}s  turn {r.turn:2d}  {mt}")
        print()

    total_checks = len(tool_checks) + len(answer_checks)
    total_passed = tool_passed + answer_passed
    pct = 100 * total_passed // total_checks if total_checks else 0
    print(f"Overall: {total_passed}/{total_checks} checks passed ({pct}%)")
    if pct == 100:
        print("All memory rules working correctly.")
    elif pct >= 75:
        print("Most rules working — review FAIL rows above.")
    else:
        print("Multiple rules failing — check Letta server and model configuration.")


# ── Entry point ────────────────────────────────────────────────────────────────

def run_benchmark() -> None:
    print("Letta Memory Rules Benchmark (v2 — all root causes fixed)")
    print("Sections: Core Memory | Recall Memory | Archival Memory")
    print()

    agent = client.agents.create(
        name="mem_benchmark_agent_v2",
        description="Comprehensive memory-rules benchmark agent",
        memory_blocks=[
            {"label": "persona", "value": (
                "You are a careful assistant that manages memory precisely.\n"
                "Rules you MUST follow:\n"
                "- When you learn a new fact about the user → call core_memory_append (or memory_insert).\n"
                "- When a user fact changes → call core_memory_replace (or memory_replace).\n"
                "- When asked to archive data for long-term storage → call archival_memory_insert.\n"
                "- When you need to retrieve previously archived data → call archival_memory_search.\n"
                "- When you need to look up older conversation history → call conversation_search.\n"
                "Never skip these tool calls — they are required for correct memory management."
            )},
            {"label": "human", "value": "User profile: no facts known yet."},
        ],
        llm_config={
            "model":               _LLM_MODEL,
            "model_endpoint_type": "openai",
            "model_endpoint":      _LLM_BASE_URL,
            "context_window":      _CTX_WINDOW,   # 2048 — tight enough to force recall search
        },
        embedding_config={
            "embedding_model":         _EMBED_MODEL,
            "embedding_endpoint_type": "openai",
            "embedding_endpoint":      _LLM_BASE_URL,
            "embedding_dim":           _EMBED_DIM,
            "embedding_chunk_size":    300,
        },
        include_base_tools=True,
        tool_rules=[],
    )
    print(f"Agent     : {agent.id[:24]}…")

    # One shared conversation for Sections 1 & 2
    conv = client.conversations.create(agent_id=agent.id)
    print(f"Conv      : {conv.id[:24]}…")
    print(f"Ctx window: {_CTX_WINDOW} tokens")
    print()

    all_results: list[TurnResult] = []

    print("━" * 62)
    print("SECTION 1 — Core Memory (append / replace / always-in-context)")
    print("━" * 62)
    all_results.extend(_core_memory_tests(conv.id, agent.id, start=1))

    print("━" * 62)
    print("SECTION 2 — Recall Memory (in-window recall vs conversation_search)")
    print("━" * 62)
    all_results.extend(_recall_memory_tests(conv.id, agent.id, start=len(all_results) + 1))

    print("━" * 62)
    print("SECTION 3 — Archival Memory (insert + search with pre-seeded data)")
    print("━" * 62)
    all_results.extend(_archival_memory_tests(agent.id, start=len(all_results) + 1))

    print("━" * 62)
    print("SECTION 4 — Core Memory Overflow (eviction → archival retrieval)")
    print("━" * 62)
    all_results.extend(_core_overflow_tests(agent.id, start=len(all_results) + 1))

    print("━" * 62)
    _report(all_results)

    try:
        client.agents.delete(agent.id)
        print(f"\nAgent {agent.id[:24]}… deleted.")
    except Exception as e:
        print(f"\nAgent cleanup failed: {e}")


if __name__ == "__main__":
    run_benchmark()
