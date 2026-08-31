#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Measure what memory actually costs, so the README can carry real numbers.

Needs a real database and a real OpenAI key, and it spends money.

    export OPENAI_API_KEY=... VOICEMEM_TEST_DSN=...
    python scripts/bench_latency.py --turns 8 --queries 12

The read path is reported broken out by stage. A single total hides where the
time goes, and the answer matters: if it is dominated by the embedding round
trip then a local embedder is worth its dependency weight, and if it is
dominated by SQL then it is not.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

from livekit.plugins import voicemem
from livekit.plugins.voicemem.types import TurnRecord

CORPUS = [
    ("I'm vegetarian and allergic to shellfish.", "Noted."),
    ("I work as a product designer at Figma, my manager is Priya.", "Got it."),
    ("I get really anxious before design reviews, every time.", "That sounds hard."),
    ("We adopted a beagle named Max last weekend.", "Congratulations!"),
    ("I'm training for the Toronto marathon in October.", "Good luck!"),
    ("I much prefer written updates over stand-up meetings.", "Understood."),
    ("My sister Anya is visiting from Vancouver next month.", "That'll be nice."),
    ("I've been trying to cut down on caffeine after 2pm.", "Sensible."),
]

QUERIES = [
    "what can I eat?",
    "who is my manager?",
    "how do I feel about reviews?",
    "do I have any pets?",
    "am I doing any races?",
    "how do I like to get updates?",
    "is anyone visiting me?",
    "what am I cutting back on?",
    "what do I have on next week?",
    "tell me about my job",
    "what should you know about me?",
    "any dietary restrictions?",
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[idx]


def table(title: str, rows: dict[str, list[float]], total_key: str | None = None) -> None:
    print(f"\n{title}")
    print(f"  {'stage':<22}{'n':>4}{'p50':>9}{'p95':>9}{'max':>9}{'% total':>9}")
    print("  " + "-" * 62)
    grand = statistics.median(rows[total_key]) if total_key and rows.get(total_key) else 0.0
    for name, values in rows.items():
        if not values:
            continue
        share = f"{100 * statistics.median(values) / grand:>8.0f}%" if grand else " " * 9
        print(
            f"  {name:<22}{len(values):>4}{pct(values, 0.5):>9.1f}"
            f"{pct(values, 0.95):>9.1f}{max(values):>9.1f}{share}"
        )


async def main(args: argparse.Namespace) -> None:
    cfg = voicemem.Config(
        pg_dsn=os.environ["VOICEMEM_TEST_DSN"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        tenant_id="bench",
    )
    runtime = await voicemem.build(cfg)
    memory = runtime.session("bench_user")
    await runtime.graph.delete_user(memory.scope)

    print(f"corpus: {args.turns} turns, then {args.queries} queries")
    print(f"chat model: {cfg.chat_model}   embed model: {cfg.embed_model}")

    write_ms: list[float] = []
    calls_per_turn: list[int] = []
    for text, reply in CORPUS[: args.turns]:
        rec = voicemem.Recorder()
        session = runtime.session("bench_user", recorder=rec)
        rec.begin_turn()
        t0 = time.perf_counter()
        await session.remember(
            TurnRecord(
                tenant_id="bench", user_id="bench_user", user_text=text, agent_reply=reply
            )
        )
        write_ms.append((time.perf_counter() - t0) * 1000)
        turn = rec.end_turn()
        calls_per_turn.append(turn.llm_call_count)

    stages: dict[str, list[float]] = {}
    for query in QUERIES[: args.queries]:
        result = await memory.recall(query)
        for stage, ms in result.timing.items():
            stages.setdefault(stage, []).append(ms)

    order = ["embed", "classify", "slot_filter", "rank", "rb", "total"]
    ordered = {k: stages[k] for k in order if k in stages}
    table("READ PATH, milliseconds (this is what lands inside the voice turn)", ordered, "total")

    table("WRITE PATH, milliseconds (background, never inside the turn)", {"ingest": write_ms})
    print(
        f"\n  LLM calls per ingested turn: "
        f"min {min(calls_per_turn)}, median {int(statistics.median(calls_per_turn))}, "
        f"max {max(calls_per_turn)}"
    )
    print("\nenvironment")
    print(f"  postgres: {cfg.pg_dsn.split('@')[-1] if '@' in cfg.pg_dsn else 'local'}")
    print("  Report both a same-host container and a managed database. A local")
    print("  container understates a real deployment badly enough to mislead.")

    await runtime.aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=len(CORPUS))
    p.add_argument("--queries", type=int, default=len(QUERIES))
    asyncio.run(main(p.parse_args()))
