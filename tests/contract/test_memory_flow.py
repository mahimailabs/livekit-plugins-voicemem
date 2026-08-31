# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""The whole retrieve-and-store chain, with no database and no network.

This is what the protocol seams buy: the same constructors the container uses,
given in-memory doubles.
"""

from __future__ import annotations

import pytest

from livekit.plugins.voicemem.config import Config
from livekit.plugins.voicemem.extraction.extractor import ConflictResolver, Extractor
from livekit.plugins.voicemem.leftbrain.classify import SlotClassifier
from livekit.plugins.voicemem.memory import VoiceMemory, render_context
from livekit.plugins.voicemem.types import (
    MemoryHit,
    MemoryRecord,
    RightBrainHit,
    Scope,
    TurnRecord,
)
from tests.fakes import FakeEmbedder, FakeGraphStore, FakeLLM, FakeVectorStore

DSN = "postgresql://u:p@localhost/db"


def build(llm_responses=None):
    embedder = FakeEmbedder()
    vectors = FakeVectorStore(embedder)
    graph = FakeGraphStore()
    llm = FakeLLM(llm_responses or [])
    cfg = Config(pg_dsn=DSN, openai_api_key="k", tenant_id="acme")
    mem = VoiceMemory(
        config=cfg,
        scope=Scope("acme", "alice"),
        vectors=vectors,
        graph=graph,
        embedder=embedder,
        classifier=SlotClassifier(embedder),
        extractor=Extractor(llm),
        resolver=ConflictResolver(llm),
    )
    return mem, vectors, graph, llm


# --- retrieval -------------------------------------------------------------


async def test_recall_returns_nothing_on_an_empty_store():
    mem, *_ = build()
    result = await mem.recall("what am I allergic to?")
    assert result.block == ""
    assert result.hits == ()


async def test_recall_finds_a_stored_fact():
    mem, vectors, *_ = build()
    await vectors.add_records(mem.scope, [MemoryRecord("", "User is allergic to peanuts")])
    result = await mem.recall("User is allergic to peanuts")
    assert [h.text for h in result.hits] == ["User is allergic to peanuts"]
    assert "allergic to peanuts" in result.block


async def test_recall_never_returns_another_tenants_memories():
    mem, vectors, *_ = build()
    await vectors.add_records(Scope("globex", "alice"), [MemoryRecord("", "Globex secret")])
    result = await mem.recall("Globex secret")
    assert result.hits == ()


async def test_the_agents_own_replies_are_not_retrieved():
    # Otherwise the agent quotes itself and answers drift every turn.
    mem, vectors, *_ = build()
    await vectors.add_text(mem.scope, "I think you should try almonds", attributed_to="assistant")
    result = await mem.recall("I think you should try almonds")
    assert result.hits == ()


async def test_recall_records_timing_per_stage():
    mem, vectors, *_ = build()
    await vectors.add_records(mem.scope, [MemoryRecord("", "User likes tea")])
    result = await mem.recall("User likes tea")
    assert "rank" in result.timing and "total" in result.timing


async def test_timings_do_not_accumulate_across_recalls():
    mem, vectors, *_ = build()
    await vectors.add_records(mem.scope, [MemoryRecord("", "User likes tea")])
    first = (await mem.recall("User likes tea")).timing["rank"]
    for _ in range(4):
        last = (await mem.recall("User likes tea")).timing["rank"]
    # Without begin_turn() per recall this grows without bound.
    assert last < first * 5 + 50


# --- ingest ----------------------------------------------------------------


async def test_remember_stores_extracted_facts():
    mem, vectors, _, _llm = build(
        [
            {"memory": [{"text": "User is vegetarian"}], "emotion": "", "traits": []},
            # No existing memories, so the resolver is skipped entirely.
        ]
    )
    ids = await mem.remember(
        TurnRecord(tenant_id="acme", user_id="alice", user_text="I'm vegetarian")
    )
    assert len(ids) == 1
    assert [r.text for r in await vectors.list_entries(mem.scope)] == ["User is vegetarian"]


async def test_empty_turn_stores_nothing_and_calls_no_model():
    mem, _, _, llm = build()
    assert await mem.remember(TurnRecord(tenant_id="acme", user_id="alice", user_text="  ")) == []
    assert llm.purposes == []


async def test_conflict_resolution_is_skipped_when_the_store_is_empty():
    # The single most expensive call on the write path.
    mem, _, _, llm = build(
        [{"memory": [{"text": "User is vegetarian"}], "emotion": "", "traits": []}]
    )
    await mem.remember(TurnRecord(tenant_id="acme", user_id="alice", user_text="I'm vegetarian"))
    assert llm.purposes == ["extract"]


async def test_intra_turn_duplicate_facts_are_collapsed():
    # One utterance producing two paraphrases of the same thing must store one.
    mem, _vectors, _, _llm = build(
        [
            {
                "memory": [
                    {"text": "User always orders oat milk because of lactose intolerance"},
                    {"text": "User prefers oat milk due to lactose intolerance"},
                ],
                "emotion": "",
                "traits": [],
            }
        ]
    )
    ids = await mem.remember(
        TurnRecord(tenant_id="acme", user_id="alice", user_text="I always get oat milk")
    )
    assert len(ids) == 1


async def test_traits_are_written_to_the_right_brain():
    mem, _, graph, _ = build(
        [
            {
                "memory": [{"text": "User has design reviews at work"}],
                "emotion": "anxious",
                "traits": [{"slot": "emotion", "label": "gets anxious before reviews"}],
            }
        ]
    )
    await mem.remember(
        TurnRecord(tenant_id="acme", user_id="alice", user_text="Reviews make me anxious")
    )
    claims = [t.claim for t in graph.traits[("acme", "alice")]]
    assert claims == ["gets anxious before reviews"]


async def test_a_trait_in_an_unknown_slot_is_discarded():
    # A model improvising slot names would silently fragment the trait space.
    mem, _, graph, _ = build(
        [
            {
                "memory": [],
                "emotion": "",
                "traits": [{"slot": "vibes", "label": "likes jazz"}],
            }
        ]
    )
    await mem.remember(TurnRecord(tenant_id="acme", user_id="alice", user_text="I like jazz"))
    assert graph.traits.get(("acme", "alice")) is None


async def test_the_agent_reply_is_stored_but_not_retrievable():
    mem, vectors, _, _ = build(
        [{"memory": [{"text": "User likes tea"}], "emotion": "", "traits": []}]
    )
    await mem.remember(
        TurnRecord(
            tenant_id="acme", user_id="alice", user_text="I like tea", agent_reply="Noted!"
        )
    )
    stored = [r.text for r in await vectors.list_entries(mem.scope)]
    assert "Noted!" in stored
    # The fake embedder is not semantic, so other rows may still rank. What
    # matters is that the assistant's own line is never among them.
    assert "Noted!" not in [h.text for h in (await mem.recall("Noted!")).hits]


# --- rendering -------------------------------------------------------------


def test_render_separates_facts_from_internal_notes():
    block = render_context(
        (MemoryHit(memory_id="1", text="User is vegetarian", score=0.9),),
        (RightBrainHit(content="wants reassurance under pressure", source="coping"),),
    )
    assert "User is vegetarian" in block
    assert "never say these aloud" in block.lower()
    assert block.index("Things you remember") < block.index("How to speak")


def test_render_of_nothing_is_empty_not_a_header():
    assert render_context((), ()) == ""


@pytest.mark.parametrize("max_rb", [1, 2, 3])
def test_right_brain_notes_are_capped(max_rb):
    rb = tuple(RightBrainHit(content=f"note {i}", source="emotion") for i in range(5))
    block = render_context((), rb, max_rb=max_rb)
    assert block.count("- note") == max_rb
