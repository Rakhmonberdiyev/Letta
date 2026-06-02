"""
agent.py — Main entry point (CLI + shared core for Telegram bot and Web backend).

Full pipeline per turn (Official Letta OS Architecture):

  User Input
      │
      ├─ [Letta OS]  get_or_create_agent(user_id)       ← per-user Letta agent
      ├─ [Letta OS]  get_or_create_conversation(user_id) ← per-session SQL recall store
      │
      ├─ Safety Guard (input)
      │
      ├─ ROUTING:
      │     deepthink=True  → system2.prepare()           (Strategy → Critique → evidence)
      │     deepthink=False → user_input passed as-is
      │
      ├─ [Letta OS]  letta_inference()
      │     Letta injects automatically:
      │       • Core Memory    (persona + human blocks)
      │       • Recall Memory  (recent turns from SQL DB)
      │       • Archival Memory (semantic passages)
      │     Letta executes client_tools (bank APIs, RAG, web search) via our FastMCP server.
      │     Letta stores this turn in its SQL-backed Recall DB automatically.
      │
      ├─ Grounding & Hallucination Filter
      ├─ Safety Guard (output)
      │
      └─ (no background persist needed — Letta handled it)
"""

import asyncio
import json
import re
import time
import config

from fastmcp import FastMCP

import ui
from memory.session import get_user_docs
from memory.letta_mem import (
    get_or_create_agent,
    get_or_create_conversation,
    letta_inference,
    set_letta_context,
    append_human_facts,
)
from tools.mcp_server  import search_mcp, rag_mcp
from tools.letta_mcp   import letta_mcp
from tools.remote_mcp  import (
    deposit_mcp, credit_mcp, pension_mcp,
    card_mcp, admin_mcp, realtime_mcp,
)
from pipeline import safety
from pipeline import system2
from pipeline.output_processor import ground_and_filter


# ── Deepthink intent detection ────────────────────────────────────────────────

_DEEPTHINK_RE = re.compile(
    r"\b("
    r"deepthink|deep\s*think"
    r"|deep\s*research|deep\s*dive|research\s+this|research\s+thoroughly"
    r"|think\s+carefully|think\s+deeply|think\s+step.by.step"
    r"|reason\s+carefully|reason\s+through"
    r"|analyze\s+carefully|careful\s+analysis|thorough\s+analysis|detailed\s+analysis"
    r"|explain\s+in\s+detail|elaborate|in[\s-]depth"
    r")\b"
    r"|yaxshilab|batafsil|chuqur\s*o['']yla|chuqur\s*tahlil|sinchiklab",
    re.IGNORECASE,
)


def _wants_deepthink(text: str) -> bool:
    return bool(_DEEPTHINK_RE.search(text))


# ── Personal fact extraction via LLM ─────────────────────────────────────────

_FACT_EXTRACT_PROMPT = """\
Extract any personal facts about the user from the message below.
Personal facts include: name, age, profession, workplace, city, country, \
education, university, family, travel experience, preferences, goals — \
anything worth remembering long-term about this specific person.

Return ONLY a JSON array of short English strings, or [] if nothing personal.
Examples:
  "meni ismim Raximberdi"          → ["Name: Raximberdi"]
  "men halq bankida ishlayman"     → ["Works at: Halq Bank"]
  "I graduated from NewUU"         → ["Graduated from: NewUU University"]
  "sometimes i have been in China" → ["Has visited: China"]
  "what is the dollar rate?"       → []
  "men Toshkentda yashaman"        → ["Lives in: Tashkent"]

Message: {message}"""


async def _auto_save_facts(agent_id: str, user_input: str, model: str) -> None:
    """Ask the LLM to extract personal facts and write them to the human block."""
    try:
        resp = await config.llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _FACT_EXTRACT_PROMPT.format(message=user_input)}],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        facts = json.loads(clean)
        if isinstance(facts, list) and facts:
            await append_human_facts(agent_id, facts)
            ui.ok(f"[agent] Auto-saved facts: {facts}")
    except Exception:
        pass


# ── Shared MCP server (all tools as client_tools for Letta) ──────────────────

main_mcp = FastMCP("Main")
main_mcp.mount(search_mcp,   namespace="WebSearch")
main_mcp.mount(rag_mcp,      namespace="RAG")
main_mcp.mount(letta_mcp,    namespace="Memory")
main_mcp.mount(deposit_mcp,  namespace="Deposit")
main_mcp.mount(credit_mcp,   namespace="Credit")
main_mcp.mount(pension_mcp,  namespace="Pension")
main_mcp.mount(card_mcp,     namespace="Card")
main_mcp.mount(admin_mcp,    namespace="Admin")
main_mcp.mount(realtime_mcp, namespace="RealTime")

_FALLBACK_MODEL = "/models/gemma"

# ── Bank query pre-fetch (prevents Gemma from answering without calling tools) ──

_PENSION_KW  = {"pensiya", "pension", "nafaqa", "to'lov kun", "tolov kun",
                "payment day", "пенсия", "to'lov", "tolov"}
_DEPOSIT_KW  = {"omonat", "deposit", "депозит", "foiz stavka", "foiz", "омонат"}
_CREDIT_KW   = {"kredit", "credit", "кредит", "qarz", "loan", "ipoteka",
                "ипотека", "avtokredit"}
_CARD_KW     = {"karta", "card", "карта", "visa", "mastercard", "humo",
                "uzcard", "plastik"}
_REALTIME_KW = {"valyuta", "kurs", "курс", "dollar", "euro", "евро", "доллар",
                "exchange rate", "usd", "eur", "rub", "рубл"}


def _matches(text: str, keywords: set) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


async def _call_tool_safe(tool_name: str, args: dict) -> str:
    try:
        result = await main_mcp.call_tool(tool_name, args)
        return "".join(
            item.text if getattr(item, "type", "") == "text" else ""
            for item in (getattr(result, "content", []) or [])
        ).strip()
    except Exception as e:
        return f"(tool error: {e})"


async def _prefetch_bank_data(user_input: str) -> str:
    """
    For queries about bank products, pre-call the relevant tools and return
    the data as a string block. This is injected into dynamic_context so the
    model never needs to guess — it has the real data in front of it.
    """
    fetched: list[str] = []

    # ── Pension ─────────────────────────────────────────────────────────────────
    # Pension queries need region/district/street extracted — the pension server
    # requires exact slugs and doesn't fuzzy-match reliably from free text.
    # Skip pre-fetch here; the improved system prompt ensures Letta calls the tool.
    if _matches(user_input, _PENSION_KW):
        ui.stage("Pre-fetch", "Pension query — Letta will call Pension_* tools")

    # ── Exchange rates ───────────────────────────────────────────────────────────
    if _matches(user_input, _REALTIME_KW):
        ui.stage("Pre-fetch", "Currency query detected — fetching exchange rates…")
        for code in ["USD", "EUR", "RUB"]:
            if any(k in user_input.lower() for k in [code.lower(), code,
                   {"USD": "dollar", "EUR": "euro евро", "RUB": "rub рубл"}.get(code, "")]):
                data = await _call_tool_safe("RealTime_exchange_rate", {"currency_code": code})
                if data and "error" not in data.lower():
                    fetched.append(f"[Exchange rate {code}]\n{data}")
        if not fetched:
            # Fetch all three major rates
            for code in ["USD", "EUR", "RUB"]:
                data = await _call_tool_safe("RealTime_exchange_rate", {"currency_code": code})
                if data and "error" not in data.lower():
                    fetched.append(f"[Exchange rate {code}]\n{data}")

    # ── Deposits ─────────────────────────────────────────────────────────────────
    if _matches(user_input, _DEPOSIT_KW):
        ui.stage("Pre-fetch", "Deposit query detected — fetching deposit list…")
        data = await _call_tool_safe("Deposit_get_all_deposit_name", {})
        if data and "error" not in data.lower():
            fetched.append(f"[Deposit products]\n{data}")

    # ── Credits ──────────────────────────────────────────────────────────────────
    if _matches(user_input, _CREDIT_KW):
        ui.stage("Pre-fetch", "Credit query detected — fetching credit list…")
        data = await _call_tool_safe("Credit_get_all_credit_name", {})
        if data and "error" not in data.lower():
            fetched.append(f"[Credit products]\n{data}")

    # ── Cards ────────────────────────────────────────────────────────────────────
    if _matches(user_input, _CARD_KW):
        ui.stage("Pre-fetch", "Card query detected — fetching card list…")
        data = await _call_tool_safe("Card_get_all_card_name", {})
        if data and "error" not in data.lower():
            fetched.append(f"[Card products]\n{data}")

    if fetched:
        block = "\n\n".join(fetched)
        ui.ok(f"Pre-fetched {len(fetched)} tool result(s) — injecting into context")
        return (
            "=== BANK DATA (fetched from live tools — use this, do NOT guess) ===\n"
            + block
            + "\n=== END BANK DATA ==="
        )
    return ""


async def initialize() -> str:
    try:
        models   = await config.llm_client.models.list()
        model_id = models.data[0].id
    except Exception as exc:
        model_id = _FALLBACK_MODEL
        ui.warn(f"Could not fetch model list ({exc}) — using fallback: {model_id}")
    config.MODEL_ID = model_id
    ui.console.print(
        f"\n[dim]Model:[/dim] [bold cyan]{model_id}[/bold cyan]"
        f"  [dim]│[/dim]  [bold green]Deepthink ON[/bold green] by default\n"
    )
    return model_id


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def process_turn(
    user_input: str,
    user_id: str,
    deepthink: bool = False,
    stream_callback=None,
    metadata: dict | None = None,
) -> str:
    """
    Process one user turn through the official Letta OS pipeline.
    Returns the assistant response string.

    All three memory tiers are managed by the Letta remote server:
      • Core Memory    → Letta blocks  (always in context)
      • Recall Memory  → Letta SQL DB  (auto-persisted by letta_inference)
      • Archival Memory → Letta passages (auto-searched and auto-inserted)
    """
    model     = config.MODEL_ID
    t_overall = time.perf_counter()

    ui.blank()
    ui.user_panel(user_input)

    # ── 0. Letta OS — per-user agent + conversation ────────────────────────────
    ui.stage("Letta OS", "Resolving agent + conversation…")
    t_agent  = time.perf_counter()
    agent_id = await get_or_create_agent(user_id)
    if not agent_id:
        ui.warn("Letta server unavailable — cannot process turn")
        return "Service temporarily unavailable. Please try again later."

    conversation_id = await get_or_create_conversation(user_id, agent_id)
    if not conversation_id:
        ui.warn("Could not create/find Letta conversation")
        return "Service temporarily unavailable. Please try again later."

    set_letta_context(user_id, agent_id)
    ui.timing("Letta agent + conversation lookup", time.perf_counter() - t_agent)
    ui.kv("Agent",        f"{agent_id[:20]}…")
    ui.kv("Conversation", f"{conversation_id[:20]}…")

    # ── 1. Build dynamic context (user docs — not managed by Letta) ────────────
    user_docs = await get_user_docs(user_id)
    dynamic_context = ""
    if user_docs:
        doc_list        = "\n".join(f"  - {d}" for d in user_docs)
        dynamic_context = (
            f"Documents uploaded by this user (searchable via RAG_rag_search, "
            f"most recent first):\n{doc_list}\n"
            f"When the user says 'this file' they mean: '{user_docs[0]}'"
        )
    ui.kv("Uploaded docs", ", ".join(user_docs) if user_docs else "none")

    # ── 2. Safety Guard (input) ────────────────────────────────────────────────
    ui.section("Safety Guard (input)")
    ui.kv("Checking", user_input[:200] + ("…" if len(user_input) > 200 else ""))
    is_safe, reason = safety.check_input(user_input)
    if not is_safe:
        ui.err(f"Input BLOCKED — {reason}")
        return "I'm sorry, I can't process that request."
    ui.ok("Input passed safety check")

    # ── 3. Routing ─────────────────────────────────────────────────────────────
    ui.section("Routing")
    evidence         = ""
    letta_input      = user_input
    use_deepthink    = deepthink or _wants_deepthink(user_input)

    if use_deepthink:
        ui.stage("System 2", "Deepthink — Strategy → Critique → evidence (phases 1-3)")
        t_s2 = time.perf_counter()
        letta_input, evidence = await system2.prepare(user_input, main_mcp, model)
        ui.timing("System 2 phases 1-3", time.perf_counter() - t_s2)
        ui.kv("Evidence pieces", f"{len(evidence.splitlines())} lines")
        if metadata is not None:
            metadata["evidence"] = evidence
    else:
        ui.stage("System 1", "Fast mode — pre-fetching bank data then Letta")
        # In fast mode System 2 doesn't run, so we pre-call bank tools ourselves
        # to ensure the model has real data and doesn't answer from training.
        bank_data = await _prefetch_bank_data(user_input)
        if bank_data:
            dynamic_context = (bank_data + "\n\n" + dynamic_context).strip()
        if metadata is not None:
            metadata["evidence"] = ""

    # ── 4. Letta inference ─────────────────────────────────────────────────────
    # Letta injects Core Memory + Recall Memory + Archival Memory automatically.
    # Every turn is stored in Letta's SQL-backed Recall DB.
    ui.section("Letta OS — Inference + Memory Injection + Recall Persistence")
    ui.stage("Letta", "Core Memory + Recall + Archival injected → tool loop → SQL store")

    t_inference = time.perf_counter()
    response    = await letta_inference(
        conversation_id=conversation_id,
        agent_id=agent_id,
        user_input=letta_input,
        mcp_server=main_mcp,
        dynamic_context=dynamic_context,
        stream_callback=stream_callback,
    )
    ui.timing("Letta inference", time.perf_counter() - t_inference)

    # Fire-and-forget: extract personal facts via LLM and save to human block.
    # Runs after the response is already returned so it adds zero latency.
    asyncio.create_task(_auto_save_facts(agent_id, user_input, model))

    if not response:
        response = "I was unable to generate a response. Please try again."

    if metadata is not None:
        metadata["ltm_facts"] = []

    # ── 5. Grounding & Hallucination Filter ────────────────────────────────────
    ui.section("Output Processing")
    t_ground = time.perf_counter()
    response = await ground_and_filter(response, evidence, model)
    ui.timing("Grounding filter", time.perf_counter() - t_ground)
    ui.ok("Grounding complete")
    ui.kv("Response tokens", f"≈{len(response)//4:,} tok  ({len(response):,} chars)")

    # ── 6. Output Safety Guard ─────────────────────────────────────────────────
    ui.section("Safety Guard (output)")
    ui.kv("Checking", response[:200] + ("…" if len(response) > 200 else ""))
    out_safe, out_reason = safety.check_output(response)
    if not out_safe:
        ui.err(f"Output BLOCKED — {out_reason}")
        response = safety.redact_unsafe_output(response)
    else:
        ui.ok("Output passed safety check")

    ui.blank()
    ui.response_panel(response)
    ui.total_time("Overall pipeline", time.perf_counter() - t_overall)
    ui.blank()

    return response


# ── CLI loop ──────────────────────────────────────────────────────────────────

async def main() -> None:
    from rich.panel import Panel
    ui.console.print(Panel(
        "[bold]Xazna AI Agent[/bold] — Official Letta OS Architecture\n"
        "All memory delegated to Letta server: Core · Recall · Archival\n"
        "Commands: [cyan]/deepthink on[/cyan] | [cyan]/deepthink off[/cyan] | [cyan]exit[/cyan]",
        border_style="cyan",
        padding=(0, 2),
    ))

    await initialize()

    USER_ID   = "user_001"
    deepthink = False

    while True:
        try:
            raw = await asyncio.to_thread(input, "\n[You] → ")
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\n[dim]Goodbye.[/dim]")
            break

        raw = raw.strip()
        if not raw:
            continue
        if raw.lower() == "exit":
            ui.console.print("[dim]Goodbye.[/dim]")
            break
        if raw.lower() == "/deepthink off":
            deepthink = False
            ui.warn("Deepthink OFF — using System 1 (fast)")
            continue
        if raw.lower() == "/deepthink on":
            deepthink = True
            ui.ok("Deepthink ON — using System 2")
            continue

        try:
            await process_turn(raw, USER_ID, deepthink=deepthink)
        except Exception as e:
            ui.err(f"Pipeline error: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
