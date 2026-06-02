"""
eval_longmemeval_letta.py — Mini-LongMemEval: Banking Temporal Knowledge Benchmark
                             Adapted for the Official Letta OS three-tier architecture.

20 sequential banking events across 30 simulated days.
Two evaluation axes:
  A) Knowledge Updates  — does the system surface the CURRENT state?
  B) Temporal Reasoning — can the system locate WHEN changes happened?

Scoring: PASS = 2 pts | PARTIAL = 1 pt | FAIL = 0 pts | max = 20 pts

Architecture tested (Official Letta OS — three tiers, all on remote server):
  Tier 1 — Core Memory    (Letta Blocks)         → human block = current-state summary,
                                                    always injected into LLM context window
  Tier 2 — Recall Memory  (Letta Conversations)   → every turn stored in Letta's SQL DB
                                                    automatically via letta_inference()
  Tier 3 — Archival Memory (Letta Passages)       → ALL inserted facts retained,
                                                    semantic ranking surfaces the most
                                                    relevant passage first

Key architectural difference vs Mem0:
  Mem0/Qdrant auto-deduplicates: "Visa Gold" overwrites "Humo Classic" entry.
  Letta Archival does NOT deduplicate: both facts exist as separate passages.
  ✓ Temporal reasoning scores HIGHER  — old facts are never lost from Archival.
  ⚠  Knowledge updates rely on semantic RANKING + Core Memory human block
     to surface the most current answer above stale passages.

Run:
  python test/eval_longmemeval_letta.py               # full run (ingest + eval)
  python test/eval_longmemeval_letta.py --skip-ingest # re-evaluate existing user data
"""

import asyncio
import os
import sys
import uuid
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.letta_mem import (
    get_or_create_agent,
    get_core_memory,
    insert_passage,
    search_passages,
    append_human_facts,
)

from letta_client import Letta as _LettaRaw

_LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
_LLM_BASE_URL   = "https://ai.xazna.uz/llm/v1"
_LLM_MODEL      = "/models/gemma"
_EMBED_MODEL    = "/models/embedding"
_EMBED_DIM      = 2048
_RECALL_CTX     = 2048   # tight window for Phase 4 — forces conversation_search

_raw = _LettaRaw(base_url=_LETTA_BASE_URL, timeout=300.0)

_MEMORY_TOOLS = {
    "core_memory_append", "memory_insert",
    "core_memory_replace", "memory_replace",
    "conversation_search",
    "archival_memory_insert",
    "archival_memory_search",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  20-EVENT BANKING TIMELINE  (30 simulated days)
# ─────────────────────────────────────────────────────────────────────────────
# "Day N:" prefixes anchor facts temporally so Archival passages store them
# with enough context to distinguish Day-1 "Humo Classic" from Day-10 "Visa Gold".

TIMELINE: list[tuple[int, str, str]] = [
    (1,
     "1-kun: Humo kartasi ochmoqchiman",
     "Ajoyib! Humo karta arizangizni boshladim. "
     "Pasport va daromad tasdiqnomasi kerak bo'ladi."),

    (2,
     "2-kun: Humo Classic uchun yillik to'lov qancha?",
     "Humo Classic birinchi yil bepul, keyin yiliga 50,000 so'm."),

    (3,
     "3-kun: Mening oylik daromadim 5,000,000 so'm",
     "Oylik 5,000,000 so'm daromadingiz Humo Classic yoki Premium uchun mos keladi."),

    (4,
     "4-kun: Men aniq Humo Classic darajasini tanlayman",
     "Qabul qilindi. Arizangiz Humo Classic karta uchun tasdiqlandi."),

    (5,
     "5-kun: Humo Classic kontaktsiz to'lovni qo'llab-quvvatlaydimi?",
     "Ha, Humo Classic HumoPass texnologiyasi orqali kontaktsiz to'lovni qo'llab-quvvatlaydi."),

    (6,
     "6-kun: Mening to'lov manzilim: Toshkent, Amir Temur ko'chasi 12-uy",
     "To'lov manzili saqlandi: Toshkent, Amir Temur ko'chasi 12-uy."),

    (7,
     "7-kun: Karta hisobotlarini o'zbek tilida olishni xohlayman",
     "Hisobot tili afzalligi hisobingizda o'zbek tiliga o'rnatildi."),

    (8,
     "8-kun: Humo Classic uchun kunlik naqd pul yechib olish limiti qancha?",
     "Humo Classic kuniga 2,000,000 so'mgacha naqd pul yechib olishga ruxsat beradi."),

    (9,
     "9-kun: Hisobimga SMS bildirishnomalarni yoqing",
     "SMS bildirishnomalar hisobingizda yoqildi."),

    # ── ASOSIY O'ZGARISH: Humo → Visa ────────────────────────────────────────
    (10,
     "10-kun: Humo arizamni bekor qiling — o'rniga Visa karta bering",
     "Tushundim. Humo Classic arizangiz bekor qilindi. "
     "Yangi Visa karta arizangizni boshlayapman."),

    (11,
     "11-kun: Men aniq Visa Gold xohlayman",
     "Visa Gold tasdiqlandi. Oylik 5,000,000 so'm daromadingiz mos keladi "
     "(minimal 3,000,000 so'm talab qilinadi). Ariza yangilandi."),

    (12,
     "12-kun: Visa Gold qanday sayohat imtiyozlarini taklif etadi?",
     "Visa Gold quyidagilarni o'z ichiga oladi: xalqaro sayohat sug'urtasi, "
     "Priority Pass aeroporti zali va nol xorijiy tranzaksiya to'lovlari."),

    (14,
     "14-kun: Men tez-tez ish safariga boraman, shuning uchun Visa Gold to'g'ri tanlov",
     "Tushundim. Visa Gold tez-tez safar qiladigan ishbilarmonlar uchun ideal. "
     "Arizangiz Visa Gold uchun tasdiqlandi."),

    (15,
     "15-kun: Visa Gold kartamga xotinim Nilufar'ni hamkarta egasi sifatida qo'shing",
     "Hamkarta egasi Nilufar Visa Gold karta arizangizga qo'shildi."),

    (16,
     "16-kun: Kredit limitimni 10,000,000 so'mga o'rnating",
     "Kredit limiti Visa Gold kartangizda 10,000,000 so'mga o'rnatildi."),

    (18,
     "18-kun: Visa Gold kartamda xorijiy valyuta operatsiyalarini yoqing",
     "Xorijiy valyuta operatsiyalari Visa Gold kartangizda yoqildi."),

    # ── ASOSIY O'ZGARISH: hamkarta egasi olib tashlandi ──────────────────────
    (20,
     "20-kun: Hamkarta egasi so'rovini bekor qiling — karta faqat mening nomimda bo'lsin",
     "Hamkarta egasi Nilufar olib tashlandi. Visa Gold kartangiz "
     "faqat siz nomingizda (yagona egasi sifatida) beriladi."),

    # ── ASOSIY O'ZGARISH: kredit limiti kamaytirildi ──────────────────────────
    (22,
     "22-kun: Kredit limitimni 10,000,000 dan 7,000,000 so'mga kamaytiring",
     "Kredit limiti Visa Gold kartangizda 10,000,000 so'mdan 7,000,000 so'mga yangilandi."),

    (25,
     "25-kun: Visa Gold kartam orqali Toshkent elektr energiyasi to'lovlari uchun avtomatik to'lov o'rnating",
     "Toshkent elektr energiyasi to'lovlari uchun avtomatik to'lov Visa Gold kartangizda sozlandi."),

    (28,
     "28-kun: Visa Gold arizamning joriy holati qanday?",
     "Sizning Visa Gold arizangiz — yagona egasi, 7,000,000 so'm limit, "
     "xorijiy valyuta yoqilgan — ko'rib chiqilmoqda. 3 ish kuni ichida tasdiqlash kutilmoqda."),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  EVALUATION QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalQ:
    qid: str
    category: Literal["knowledge_update", "temporal_reasoning"]
    question: str
    expected_answer: str
    required_kws: list[str]      # must appear for a PASS
    stale_kws: list[str]         # presence as primary answer signals stale data
    core_kws: list[str] = field(default_factory=list)  # look for in Core Memory human block


EVAL_QUESTIONS: list[EvalQ] = [
    # ── BILIM YANGILANISHI / KNOWLEDGE UPDATE (5) ─────────────────────────────
    EvalQ(
        qid="KU-1",
        category="knowledge_update",
        question="Foydalanuvchi hozirda qanday karta xohlaydi yoki ariza bergan?",
        expected_answer="Visa Gold (10-kunda Humo Classic'dan o'tgan)",
        required_kws=["visa", "gold"],
        stale_kws=["humo"],
        core_kws=["visa", "gold"],
    ),
    EvalQ(
        qid="KU-2",
        category="knowledge_update",
        question="Hozirda kartada hamkarta egasi bormi?",
        expected_answer="Yo'q — hamkarta egasi Nilufar 20-kunda bekor qilindi; karta yagona egasida",
        required_kws=["nilufar", "sole", "removed", "cancelled", "yagona", "bekor", "olib tashlandi"],
        stale_kws=[],
        core_kws=["nilufar", "sole", "yagona"],
    ),
    EvalQ(
        qid="KU-3",
        category="knowledge_update",
        question="Foydalanuvchining joriy kredit limiti qancha?",
        expected_answer="7,000,000 so'm (22-kunda 10M dan kamaytirilgan)",
        required_kws=["7,000,000", "7000000", "7 million", "7m"],
        stale_kws=["10,000,000", "10 million", "10m"],
        core_kws=["7,000,000", "7000000"],
    ),
    EvalQ(
        qid="KU-4",
        category="knowledge_update",
        question="Foydalanuvchi karta hisobotlari uchun qaysi tilni afzal ko'radi?",
        expected_answer="O'zbek tili (7-kunda o'rnatilgan)",
        required_kws=["uzbek", "o'zbek", "uzbekcha"],
        stale_kws=[],
        core_kws=["uzbek", "o'zbek"],
    ),
    EvalQ(
        qid="KU-5",
        category="knowledge_update",
        question="Foydalanuvchi uchun xorijiy valyuta operatsiyalari yoqilganmi?",
        expected_answer="Ha — 18-kunda yoqilgan",
        required_kws=["foreign", "currency", "enabled", "xorijiy", "valyuta", "yoqilgan"],
        stale_kws=[],
        core_kws=["foreign", "xorijiy", "valyuta"],
    ),
    # ── VAQTINCHALIK MANTIQ / TEMPORAL REASONING (5) ─────────────────────────
    EvalQ(
        qid="TR-1",
        category="temporal_reasoning",
        question="Foydalanuvchi o'zgartishdan oldin dastlab qanday kartaga ariza bergan edi?",
        expected_answer="Humo Classic (1–9-kunlar), 10-kunda Visa Gold'ga o'tgan",
        required_kws=["humo", "classic"],
        stale_kws=[],
        core_kws=["humo"],
    ),
    EvalQ(
        qid="TR-2",
        category="temporal_reasoning",
        question="Kredit limiti kamaytirilishdan oldin qancha edi?",
        expected_answer="10,000,000 so'm (16-kunda o'rnatilgan, 22-kunda 7M ga kamaytirilgan)",
        required_kws=["10,000,000", "10 million", "10m", "10000000"],
        stale_kws=[],
        core_kws=["10,000,000"],
    ),
    EvalQ(
        qid="TR-3",
        category="temporal_reasoning",
        question="Foydalanuvchining to'lov manzili qanday?",
        expected_answer="Toshkent, Amir Temur ko'chasi 12-uy (6-kunda o'rnatilgan, o'zgarmagan)",
        required_kws=["amir temur", "tashkent", "12"],
        stale_kws=[],
        core_kws=["amir temur", "tashkent"],
    ),
    EvalQ(
        qid="TR-4",
        category="temporal_reasoning",
        question="Bu kartada hamkarta egasi so'rovi bo'lganmi?",
        expected_answer="Ha — Nilufar 15-kunda qo'shilgan, 20-kunda olib tashlangan",
        required_kws=["nilufar"],
        stale_kws=[],
        core_kws=["nilufar"],
    ),
    EvalQ(
        qid="TR-5",
        category="temporal_reasoning",
        question="Foydalanuvchi qanday oylik daromad e'lon qilgan?",
        expected_answer="Oyiga 5,000,000 so'm (3-kunda e'lon qilgan)",
        required_kws=["5,000,000", "5 million", "5000000", "5m", "daromad", "income"],
        stale_kws=[],
        core_kws=["5,000,000", "daromad", "income"],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SCORING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _kw_match(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def score_question(
    q: EvalQ,
    retrieved_facts: str,      # from search_passages (Archival passages, joined)
    core_memory_text: str,     # from get_core_memory human block
) -> tuple[int, str]:
    """
    Returns (score, reason).
    0 = FAIL | 1 = PARTIAL | 2 = PASS

    Letta-specific scoring rules:
      Knowledge Updates — Core Memory human block is the primary authoritative source
        (it is always injected into the LLM context window).
        Archival passages are secondary.  Stale contamination is checked across both.
      Temporal Reasoning — Archival passages are authoritative because Letta retains
        ALL inserted passages without deduplication, making historical facts reliably
        searchable.  Core Memory is a fallback.
    """
    facts_lower = retrieved_facts.lower()
    core_lower  = core_memory_text.lower()
    combined    = facts_lower + " " + core_lower

    found_required    = _kw_match(combined,    q.required_kws)
    found_stale       = _kw_match(combined,    q.stale_kws)    if q.stale_kws else False
    found_in_archival = _kw_match(facts_lower, q.required_kws)
    found_in_core     = _kw_match(core_lower,  q.required_kws)
    stale_in_core     = _kw_match(core_lower,  q.stale_kws)    if q.stale_kws else False

    # First passage = highest-ranked by semantic similarity
    first_passage = (retrieved_facts.splitlines() or [""])[0].lower()
    first_has_req   = _kw_match(first_passage, q.required_kws)
    first_has_stale = _kw_match(first_passage, q.stale_kws) if q.stale_kws else False

    if q.category == "knowledge_update":
        # Core Memory is append-only: stale_in_core is almost always True when facts change
        # (e.g. "Humo Classic" days 1-9 stay in the block alongside "Visa Gold" day 10+).
        # Fix: check which keyword appears LATER in the block — the last-appended fact wins.
        core_current_is_newer = False
        if stale_in_core and found_in_core and q.stale_kws:
            cl = core_memory_text.lower()
            last_req   = max((cl.rfind(k.lower()) for k in q.required_kws if k.lower() in cl), default=-1)
            last_stale = max((cl.rfind(k.lower()) for k in q.stale_kws   if k.lower() in cl), default=-1)
            core_current_is_newer = last_req > last_stale

        # PASS: current fact in Core AND it appears after any stale fact (more recent append)
        if found_in_core and (not stale_in_core or core_current_is_newer):
            return 2, "PASS — current fact in Core Memory human block (appears after stale fact)"
        # PASS: top Archival result is current with no stale contamination in top result
        if first_has_req and not first_has_stale:
            return 2, "PASS — current fact ranked first in Archival; no stale contamination"
        # PARTIAL: current fact found somewhere but stale also present
        if found_required and found_stale:
            return 1, "PARTIAL — current fact present but stale fact also retrieved (Letta retains all passages)"
        # PARTIAL: current fact in Archival but not yet propagated to Core Memory
        if found_in_archival and not found_in_core:
            return 1, "PARTIAL — found in Archival passages but not yet in Core Memory human block"
        if found_required:
            return 2, "PASS — current fact present; no stale contamination detected"
        return 0, "FAIL — expected current fact not found in Archival or Core Memory"

    else:  # temporal_reasoning
        # Letta excels here: ALL passages are retained → historical facts stay searchable
        if found_in_archival:
            return 2, "PASS — historical fact preserved in Letta Archival passages (never deleted)"
        if found_in_core:
            return 1, "PARTIAL — found in Core Memory human block; not surfaced by Archival search"
        return 0, "FAIL — historical fact absent from Archival passages and Core Memory"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "═" * 64
SEP2 = "─" * 64

GRADE_LABELS = {2: "✅ PASS   ", 1: "⚠️  PARTIAL", 0: "❌ FAIL   "}


def _h1(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _h2(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PIPELINE PHASES
# ─────────────────────────────────────────────────────────────────────────────

async def ingest_timeline(user_id: str) -> None:
    _h1("PHASE 1 — INGESTING 20 BANKING EVENTS")
    print(f"  User ID  : {user_id}")
    print(f"  Events   : {len(TIMELINE)} across 30 simulated days")
    print(f"  Per turn : insert_passage() → Archival  |  append_human_facts() → Core Memory\n")
    print("  Note: Recall Memory is populated automatically by letta_inference() in production.\n")

    agent_id = await get_or_create_agent(user_id)
    if not agent_id:
        print("  ✗  Letta server unavailable — aborting ingestion")
        return
    print(f"  Agent ID : {agent_id[:20]}…\n")

    for i, (day, user_msg, asst_msg) in enumerate(TIMELINE, 1):
        sys.stdout.write(f"  [{i:02d}/20] Day {day:>2d}: {user_msg[:55]}…")
        sys.stdout.flush()
        t0 = time.perf_counter()

        # Archival: full conversation turn as a searchable passage
        passage = f"Day {day}: {user_msg} | {asst_msg}"
        await insert_passage(agent_id, passage)

        # Core Memory: append assistant response as a one-line fact summary
        await append_human_facts(agent_id, [f"Day {day}: {asst_msg[:150]}"])

        elapsed = time.perf_counter() - t0
        print(f"  ({elapsed:.1f}s)")
        await asyncio.sleep(0.2)   # let Letta settle between inserts

    print(f"\n  ✓ Ingestion complete.")


async def evaluate(user_id: str) -> tuple[list[dict], str]:
    _h1("PHASE 2 — EVALUATION  (10 questions)")
    print(f"  Querying Archival Memory + Core Memory for user: {user_id}\n")

    # Fetch agent_id (cache hit after ingest)
    agent_id = await get_or_create_agent(user_id)
    blocks   = await get_core_memory(agent_id) if agent_id else {}
    human_block = (blocks or {}).get("human", "")

    if not agent_id:
        print("  ⚠  Letta server unavailable — Core Memory checks will score as FAIL")
    else:
        print(f"  Agent ID    : {agent_id[:20]}…")
        print(f"  Human block : {len(human_block)} chars\n")

    results = []

    for q in EVAL_QUESTIONS:
        print(f"\n  [{q.qid}] {q.question}")
        passages = await search_passages(agent_id, q.question, top_k=7) if agent_id else []
        facts    = "\n".join(passages) if passages else ""

        score, reason = score_question(q, facts, human_block)
        grade = GRADE_LABELS[score]
        print(f"         Grade    : {grade}  ({score}/2)")
        print(f"         Expected : {q.expected_answer}")
        print(f"         Reason   : {reason}")

        # Show top 3 Archival passages
        for p in passages[:3]:
            print(f"         Archival : {p[:88]}")

        # Show Core Memory match
        core_match = _kw_match(human_block, q.core_kws)
        if q.core_kws:
            marker = "✓" if core_match else "✗"
            print(f"         Core Mem : {marker}  keywords {q.core_kws} in human block")

        results.append({
            "qid":       q.qid,
            "category":  q.category,
            "question":  q.question,
            "expected":  q.expected_answer,
            "score":     score,
            "reason":    reason,
            "retrieved": facts,
        })

    return results, human_block


def print_core_memory_audit(human_block: str, agent_id: str | None) -> None:
    _h1("PHASE 3 — CORE MEMORY AUDIT  (Letta human block)")
    print("  Core Memory human block is always injected into the LLM context window.")
    print("  It holds the most recently appended user facts (trimmed front-first at 1,500 chars).\n")

    if not agent_id:
        print("  Letta server unavailable — audit skipped.")
        return

    print(f"  Agent ID    : {(agent_id or '')[:20]}…")
    print(f"  Block size  : {len(human_block)} / 1,500 chars\n")

    if not human_block:
        print("  Human block is empty — facts may still be processing.")
        print("  Tip: re-run with --skip-ingest after 10–15 s to see populated blocks.")
        return

    print("  ── Human block contents ──────────────────────────────────────────")
    for line in human_block.splitlines():
        print(f"    {line[:100]}")
    print()

    # Verify key temporal milestones are captured
    print("  ── Key-event tracking ────────────────────────────────────────────")
    key_checks = [
        ("Card switch: Humo → Visa Gold",        ["humo", "visa"]),
        ("Credit limit 10M (historical)",        ["10,000,000"]),
        ("Credit limit 7M (current)",            ["7,000,000"]),
        ("Co-holder Nilufar (added+removed)",    ["nilufar"]),
        ("Billing address Tashkent",             ["tashkent", "amir temur"]),
        ("Income 5,000,000 UZS",                 ["5,000,000", "income"]),
        ("Uzbek statement preference",           ["uzbek"]),
        ("Foreign currency enabled",             ["foreign", "currency"]),
    ]
    h = human_block.lower()
    for label, kws in key_checks:
        found  = any(kw in h for kw in kws)
        marker = "✓" if found else "✗"
        print(f"    {marker}  {label}")

    print()
    print("  Note: Letta Core Memory does NOT store a change-log.")
    print("  Historical state lives in Archival passages (all retained forever).")
    print("  The human block reflects accumulated facts, trimmed when it exceeds 1,500 chars.")


def print_scorecard(results: list[dict]) -> None:
    _h1("SCORECARD")

    ku = [r for r in results if r["category"] == "knowledge_update"]
    tr = [r for r in results if r["category"] == "temporal_reasoning"]

    def _section(title: str, rows: list[dict]) -> None:
        total   = sum(r["score"] for r in rows)
        max_pts = len(rows) * 2
        print(f"\n  {title}  ({total}/{max_pts} pts)")
        print(f"  {'QID':<8} {'Score':<8} Verdict")
        print(f"  {'─'*6}  {'─'*6}  {'─'*50}")
        for r in rows:
            grade = GRADE_LABELS[r["score"]]
            print(f"  {r['qid']:<8} {r['score']}/2     {grade}  {r['reason']}")

    _section("A) Knowledge Updates", ku)
    _section("B) Temporal Reasoning", tr)

    grand = sum(r["score"] for r in results)
    max_g = len(results) * 2
    pct   = grand / max_g * 100

    ku_score = sum(r["score"] for r in ku)
    tr_score = sum(r["score"] for r in tr)

    print(f"\n{SEP2}")
    print(f"  TOTAL SCORE : {grand} / {max_g} pts  ({pct:.0f}%)")
    print(SEP2)

    print(f"""
  BOSS REPORT — Official Letta OS Architecture Analysis
  ──────────────────────────────────────────────────────

  Knowledge Updates  {ku_score}/{len(ku)*2} pts
    Core Memory human block is always present in the LLM context window,
    giving the model direct access to the most recent user state without
    needing a search call.  Unlike Mem0/Qdrant, Letta does NOT deduplicate
    passages — both "Humo Classic" and "Visa Gold" exist in Archival.
    Correct answers rely on semantic ranking surfacing the more recent fact
    first, plus the human block acting as a current-state anchor.

  Temporal Reasoning {tr_score}/{len(tr)*2} pts
    Letta Archival passages are NEVER automatically deleted or overwritten.
    Every inserted fact survives, making historical queries ("what card did
    the user originally apply for?") reliably answerable directly from
    Archival search — no separate change-log database needed.
    This is a structural advantage over Mem0, where deduplication can erase
    historical state, forcing reliance on a SQLite change-log.

  Architecture Tradeoff Summary
    Mem0:  Auto-dedup → cleaner current-state answers | change-log needed for history
    Letta: All passages retained → richer history | semantic ranking is critical for
           surfacing the current answer above stale passages

  Result: {pct:.0f}% accuracy — Official Letta OS three-tier memory demonstrated on
    a 30-day, 20-event banking scenario with deliberate state transitions.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RAW-TURN HELPERS  (used by Phases 4 & 5)
# ─────────────────────────────────────────────────────────────────────────────

def _drain(response) -> list:
    """Fully drain a streaming Letta response into an event list."""
    events = getattr(response, "messages", None)
    if events is None:
        try:
            events = list(response)
        except Exception:
            events = []
    return events


def _tools_from(events: list) -> list[str]:
    """Extract deduplicated tool names, avoiding double-count from .tool_call + .tool_calls[]."""
    seen: set = set()
    names: list[str] = []
    for ev in events:
        if getattr(ev, "message_type", None) != "tool_call_message":
            continue
        for tc in ([getattr(ev, "tool_call", None)] + list(getattr(ev, "tool_calls", None) or [])):
            if tc is None:
                continue
            tc_id = getattr(tc, "tool_call_id", None) or id(tc)
            if tc_id not in seen:
                seen.add(tc_id)
                names.append(getattr(tc, "name", ""))
    return names


def _text_from(events: list) -> str:
    out = ""
    for ev in events:
        if getattr(ev, "message_type", None) == "assistant_message":
            c = getattr(ev, "content", "") or ""
            if isinstance(c, list):
                c = "".join(getattr(p, "text", "") for p in c)
            out += str(c)
    return out


def _turn(conv_id: str, agent_id: str, msg: str) -> tuple[list[str], str, float]:
    """Send one raw turn; return (tools_called, response_text, latency_s)."""
    t0 = time.perf_counter()
    resp = _raw.conversations.messages.create(conv_id, agent_id=agent_id, input=msg)
    events = _drain(resp)
    lat = time.perf_counter() - t0
    return _tools_from(events), _text_from(events), lat


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PHASE 4 — RECALL MEMORY  (conversation_search)
# ─────────────────────────────────────────────────────────────────────────────

# Noise prompt that forces a long LLM response so each flood turn consumes
# many context-window tokens, pushing the planted fact out of the sliding window.
_RECALL_NOISE = (
    "Describe every field in a SWIFT MT103 payment message in detail: "
    "sender BIC, receiver BIC, value date, currency, amount, ordering customer, "
    "beneficiary, remittance information, and charges field. Provide a full example message."
)


async def eval_recall_memory() -> dict:
    """
    Tests Tier 2 (Recall Memory / conversation_search).

    Creates a dedicated 2048-token-context agent so that 7 noise turns consume
    the entire sliding window and force the agent to call conversation_search
    to find the fact planted at Turn 1.

    Two sub-checks:
      4.2 — Immediate recall: planted fact retrieved WITHOUT conversation_search
      4.4 — Distant recall:   planted fact retrieved WITH    conversation_search
    """
    _h1("PHASE 4 — RECALL MEMORY  (Tier 2: conversation_search)")
    print(f"  Context window : {_RECALL_CTX} tokens")
    print(f"  Flood turns    : 7  (forces context overflow)\n")

    agent = _raw.agents.create(
        name="eval_recall_phase4",
        description="Phase 4 recall memory eval — temporary",
        memory_blocks=[
            {"label": "persona", "value": (
                "You are a banking assistant. "
                "When you need information from earlier in the conversation, "
                "call conversation_search. Never guess or invent data."
            )},
            {"label": "human", "value": "Customer session."},
        ],
        llm_config={
            "model":               _LLM_MODEL,
            "model_endpoint_type": "openai",
            "model_endpoint":      _LLM_BASE_URL,
            "context_window":      _RECALL_CTX,
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
    conv = _raw.conversations.create(agent_id=agent.id)
    checks: list[tuple[str, int, int]] = []   # (label, score, max)

    # 4.1 — Plant a banking fact
    iban = "UZ12345678901234567890"
    tools, _, lat = _turn(conv.id, agent.id,
        f"Important: my bank IBAN is {iban}. Please note this down.")
    print(f"  [4.1] Plant IBAN in recall")
    print(f"        Tools    : {tools or ['(none)']}")
    print(f"        Latency  : {lat:.2f}s\n")

    # 4.2 — Immediate recall: IBAN is still in the context window → NO search needed
    tools, text, lat = _turn(conv.id, agent.id, "What is my IBAN?")
    no_search  = "conversation_search" not in tools
    iban_found = iban in text
    s42 = 2 if (no_search and iban_found) else (1 if iban_found else 0)
    print(f"  [4.2] Immediate recall — must NOT call conversation_search")
    print(f"        Tools called      : {tools or ['(none)']}")
    print(f"        no_search         : {'PASS' if no_search else 'FAIL'}")
    print(f"        IBAN in response  : {'YES' if iban_found else 'NO'}")
    print(f"        Score             : {s42}/2   ({lat:.2f}s)\n")
    checks.append(("4.2 Immediate recall (no search)", s42, 2))

    # 4.3 — Flood context window with 7 noise turns
    print(f"  [4.3] Flooding context window…")
    for i in range(7):
        _, _, lat_n = _turn(conv.id, agent.id, _RECALL_NOISE)
        sys.stdout.write(f"  noise {i+1}/7 ({lat_n:.2f}s)  ")
        sys.stdout.flush()
    print("\n")

    # 4.4 — Distant recall: IBAN is outside context → MUST call conversation_search
    tools, text, lat = _turn(conv.id, agent.id,
        "Search your conversation history — what IBAN did I tell you at the very start?")
    search_called = "conversation_search" in tools
    iban_found2   = iban in text
    s44 = 2 if (search_called and iban_found2) else (1 if iban_found2 else 0)
    print(f"  [4.4] Distant recall — MUST call conversation_search")
    print(f"        Tools called      : {tools or ['(none)']}")
    print(f"        search_called     : {'PASS' if search_called else 'FAIL'}")
    print(f"        IBAN retrieved    : {'YES' if iban_found2 else 'NO'}")
    print(f"        Score             : {s44}/2   ({lat:.2f}s)\n")
    checks.append(("4.4 Distant recall (search required)", s44, 2))

    try:
        _raw.agents.delete(agent.id)
    except Exception:
        pass

    total = sum(s for _, s, _ in checks)
    maxt  = sum(m for _, _, m in checks)
    print(f"  Recall subtotal: {total}/{maxt}")
    return {"checks": checks, "score": total, "max": maxt}


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PHASE 5 — LLM-DRIVEN STORAGE DECISIONS
# ─────────────────────────────────────────────────────────────────────────────

async def eval_llm_storage_decisions(agent_id: str) -> dict:
    """
    Tests whether the LLM spontaneously calls the correct memory tools when
    given real banking conversation turns — without being told which tool to call.

    This is the gap that Phases 1–3 miss: those phases inject facts directly via
    the API, bypassing the LLM entirely.  Here the LLM must decide on its own.

    Sub-checks:
      5.1 — New user fact    → core_memory_append / memory_insert
      5.2 — Archive request  → archival_memory_insert
      5.3 — Fact change      → core_memory_replace / memory_replace
      5.4 — Updated fact visible without search (core memory always-in-context)
    """
    _h1("PHASE 5 — LLM-DRIVEN STORAGE DECISIONS")
    print("  Sends real conversation turns and checks whether the LLM\n"
          "  spontaneously calls the correct memory tool each time.\n")

    conv = _raw.conversations.create(agent_id=agent_id)
    checks: list[tuple[str, int, int]] = []

    # 5.1 — New user fact → LLM should call core_memory_append / memory_insert
    tools, _, lat = _turn(conv.id, agent_id,
        "My name is Aziz Karimov and I am a premium Xazna bank customer. Please save this.")
    called_append = bool({"core_memory_append", "memory_insert"}.intersection(tools))
    s51 = 2 if called_append else 0
    print(f"  [5.1] New user fact → core_memory_append / memory_insert")
    print(f"        Tools called : {tools or ['(none)']}")
    print(f"        PASS         : {'YES' if called_append else 'NO'}   ({lat:.2f}s)\n")
    checks.append(("5.1 Core append for new fact", s51, 2))

    # 5.2 — Archive request → LLM should call archival_memory_insert
    tools, _, lat = _turn(conv.id, agent_id,
        "Please archive this for long-term storage: "
        "'Investment portfolio Day-30: 50,000,000 UZS in time deposits, "
        "20,000,000 UZS in government bonds, risk profile: conservative.'")
    called_archival = "archival_memory_insert" in tools
    s52 = 2 if called_archival else 0
    print(f"  [5.2] Archive request → archival_memory_insert")
    print(f"        Tools called : {tools or ['(none)']}")
    print(f"        PASS         : {'YES' if called_archival else 'NO'}   ({lat:.2f}s)\n")
    checks.append(("5.2 Archival insert for portfolio", s52, 2))

    # 5.3 — Fact change → LLM should call core_memory_replace / memory_replace
    tools, _, lat = _turn(conv.id, agent_id,
        "I changed my name — from now on I am Bobur Karimov, not Aziz. Please update your memory.")
    called_replace = bool({"core_memory_replace", "memory_replace"}.intersection(tools))
    s53 = 2 if called_replace else 0
    print(f"  [5.3] Fact change → core_memory_replace / memory_replace")
    print(f"        Tools called : {tools or ['(none)']}")
    print(f"        PASS         : {'YES' if called_replace else 'NO'}   ({lat:.2f}s)\n")
    checks.append(("5.3 Core replace for name change", s53, 2))

    # 5.4 — Updated fact should be visible without any search tool (always in context)
    tools, text, lat = _turn(conv.id, agent_id, "What is my name?")
    no_search    = not {"conversation_search", "archival_memory_search"}.intersection(tools)
    name_correct = "bobur" in text.lower()
    s54 = 2 if (no_search and name_correct) else (1 if name_correct else 0)
    print(f"  [5.4] Updated fact visible without search (always-in-context)")
    print(f"        Tools called : {tools or ['(none)']}")
    print(f"        No search    : {'PASS' if no_search else 'FAIL'}")
    print(f"        Name correct : {'YES — ' + text[:60] if name_correct else 'NO'}")
    print(f"        Score        : {s54}/2   ({lat:.2f}s)\n")
    checks.append(("5.4 Core visible no search", s54, 2))

    total = sum(s for _, s, _ in checks)
    maxt  = sum(m for _, _, m in checks)
    print(f"  LLM-storage subtotal: {total}/{maxt}")
    return {"checks": checks, "score": total, "max": maxt}


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PHASES 4 & 5 SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_live_phases_summary(
    recall_result: dict,
    llm_result: dict,
    content_score: int,
    content_max: int,
) -> None:
    _h1("COMBINED SCORECARD — All Phases")

    def _row(label: str, score: int, max_score: int) -> None:
        pct   = 100 * score // max_score if max_score else 0
        grade = "PASS   " if score == max_score else ("PARTIAL" if score > 0 else "FAIL   ")
        print(f"    [{grade}]  {label:<45s}  {score}/{max_score}  ({pct}%)")

    print("\n  Phase 1–3  — Content accuracy (Archival + Core Memory)")
    _row("Knowledge Updates + Temporal Reasoning (10 Qs)", content_score, content_max)

    print("\n  Phase 4    — Recall Memory (conversation_search)")
    for label, score, max_score in recall_result["checks"]:
        _row(label, score, max_score)

    print("\n  Phase 5    — LLM-Driven Storage Decisions")
    for label, score, max_score in llm_result["checks"]:
        _row(label, score, max_score)

    grand     = content_score + recall_result["score"] + llm_result["score"]
    grand_max = content_max   + recall_result["max"]   + llm_result["max"]
    pct       = 100 * grand // grand_max if grand_max else 0

    print(f"\n{SEP2}")
    print(f"  GRAND TOTAL : {grand} / {grand_max} pts  ({pct}%)")
    print(SEP2)
    if pct == 100:
        print("  All memory rules — content accuracy, recall search, and LLM storage decisions — working correctly.")
    elif pct >= 75:
        print("  Most rules working. Review FAIL rows above for targeted fixes.")
    else:
        print("  Multiple rules failing. Check Letta server config and model tool-use capability.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main(skip_ingest: bool, user_id: str) -> None:
    print(f"\n{'═'*64}")
    print("  MINI-LongMemEval — Official Letta OS Banking Temporal Knowledge Benchmark")
    print(f"{'═'*64}")
    print(f"  User ID   : {user_id}   ← pass to --user-id to reuse with --skip-ingest")
    print(f"  Events    : {len(TIMELINE)} | Questions : {len(EVAL_QUESTIONS)}")
    print(f"  Storage   : ALL delegated to remote Letta server")
    print(f"  Memory    : Core (Blocks) | Recall (SQL DB) | Archival (Passages)")
    print(f"  Mode      : {'evaluation only (--skip-ingest)' if skip_ingest else 'full run (ingest + eval)'}")

    if not skip_ingest:
        await ingest_timeline(user_id)
        print("\n  Waiting 5 s for Letta passage indexing to settle…")
        await asyncio.sleep(5)
    else:
        print(f"\n  Skipping ingestion — using existing data for user {user_id}")

    results, human_block = await evaluate(user_id)

    agent_id = await get_or_create_agent(user_id)
    print_core_memory_audit(human_block, agent_id)

    # Phases 1–3 content score (for combined scorecard)
    content_score = sum(r["score"] for r in results)
    content_max   = len(results) * 2

    print_scorecard(results)

    # Phase 4 — Recall Memory (conversation_search)
    recall_result = await eval_recall_memory()

    # Phase 5 — LLM-Driven Storage Decisions (uses the same Letta agent as Phases 1–3)
    llm_result = await eval_llm_storage_decisions(agent_id)

    # Combined scorecard across all phases
    print_live_phases_summary(recall_result, llm_result, content_score, content_max)

    # Print user_id again at the end so it's easy to copy for --skip-ingest
    print(f"  User ID (for --skip-ingest) : {user_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mini-LongMemEval — Official Letta OS banking benchmark"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion and evaluate existing data",
    )
    parser.add_argument(
        "--user-id",
        default=f"eval_letta_{uuid.uuid4().hex[:8]}",
        help="Explicit user ID (default: random)",
    )
    args = parser.parse_args()

    asyncio.run(main(skip_ingest=args.skip_ingest, user_id=args.user_id))
