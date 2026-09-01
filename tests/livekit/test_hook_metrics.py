# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Turn accounting, through the hook, with no room and no database.

``Recorder`` opens a turn inside ``recall`` and something has to close it. For a
long time nothing did, outside the benchmark and the recorder's own unit tests,
so in the only context anyone would attach instrumentation to, ``turns`` stayed
at zero, ``prefetch_hit_rate`` divided by zero and always answered 0.0, and the
``metrics_hook`` documented as the way to export these never emitted once.
"""

from __future__ import annotations

import pytest
from livekit.agents.llm import ChatContext, ChatMessage

from livekit.plugins.voicemem.config import Config
from livekit.plugins.voicemem.extraction.extractor import ConflictResolver, Extractor
from livekit.plugins.voicemem.hooks import MemoryHooks
from livekit.plugins.voicemem.instrument import Recorder
from livekit.plugins.voicemem.leftbrain.classify import SlotClassifier
from livekit.plugins.voicemem.memory import VoiceMemory
from livekit.plugins.voicemem.types import MemoryRecord, Scope
from tests.fakes import FakeEmbedder, FakeGraphStore, FakeLLM, FakeVectorStore

pytestmark = pytest.mark.livekit


def build(recorder: Recorder | None = None):
    embedder = FakeEmbedder()
    vectors = FakeVectorStore(embedder)
    memory = VoiceMemory(
        config=Config(pg_dsn="postgresql://u:p@localhost/db", openai_api_key="k"),
        scope=Scope("acme", "alice"),
        vectors=vectors,
        graph=FakeGraphStore(),
        embedder=embedder,
        classifier=SlotClassifier(embedder),
        extractor=Extractor(FakeLLM([])),
        resolver=ConflictResolver(FakeLLM([])),
        recorder=recorder,
    )
    return memory, vectors


def _turn(text: str) -> tuple[ChatContext, ChatMessage]:
    return ChatContext.empty(), ChatMessage(role="user", content=[text])


async def test_a_completed_turn_is_counted() -> None:
    recorder = Recorder()
    memory, _ = build(recorder)
    hooks = MemoryHooks(memory)

    for text in ("what can I eat?", "and this month?"):
        await hooks.on_user_turn_completed(*_turn(text))

    assert recorder.counters.get("turns") == 2


async def test_metrics_hook_receives_each_turn() -> None:
    """The documented way to export these. It fired for nothing at all before."""
    seen: list[tuple[str, dict]] = []
    recorder = Recorder(metrics_hook=lambda event, payload: seen.append((event, payload)))
    memory, _ = build(recorder)

    await MemoryHooks(memory).on_user_turn_completed(*_turn("what can I eat?"))

    assert [e for e, _p in seen] == ["turn"]
    assert "recall_total_ms" in seen[0][1]


async def test_prefetch_hit_rate_has_a_denominator() -> None:
    """It divides by `turns`, so with nothing closing a turn it could only ever
    answer 0.0, however well prefetching was working."""
    recorder = Recorder()
    memory, _ = build(recorder)
    hooks = MemoryHooks(memory)

    await hooks.on_user_turn_completed(*_turn("what can I eat?"))
    assert recorder.counters.get("turns") == 1
    assert recorder.prefetch_hit_rate == 0.0  # no prefetch ran, honestly zero

    recorder.count("prefetch_hit")
    assert recorder.prefetch_hit_rate == 1.0


async def test_an_empty_turn_is_not_counted() -> None:
    """StopResponse is raised before the turn opens, so it must not inflate the
    denominator every other rate is measured against."""
    recorder = Recorder()
    memory, _ = build(recorder)

    from livekit.agents import StopResponse

    with pytest.raises(StopResponse):
        await MemoryHooks(memory).on_user_turn_completed(*_turn("   "))

    assert recorder.counters.get("turns") is None


async def test_a_turn_is_closed_even_when_recall_fails() -> None:
    """Recall failures are contained rather than raised, so the accounting has
    to survive them or one bad turn skews every rate afterwards."""
    recorder = Recorder()
    memory, vectors = build(recorder)

    async def boom(*a, **k):
        raise RuntimeError("store is down")

    vectors.search = boom  # type: ignore[method-assign]

    await MemoryHooks(memory).on_user_turn_completed(*_turn("what can I eat?"))
    assert recorder.counters.get("turns") == 1


async def test_recorded_timings_reach_the_turn(monkeypatch) -> None:
    recorder = Recorder()
    memory, vectors = build(recorder)
    await vectors.add_records(memory.scope, [MemoryRecord("", "User avoids dairy")])

    captured: list[dict] = []
    recorder._hook = lambda event, payload: captured.append(payload)  # type: ignore[assignment]

    await MemoryHooks(memory).on_user_turn_completed(*_turn("what can I eat?"))
    assert captured and captured[0]["recall_total_ms"] >= 0.0
