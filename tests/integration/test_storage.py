# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Storage against a real PostgreSQL with pgvector.

No OpenAI key needed: a deterministic fake embedder is injected, because what is
under test is the SQL and the isolation, not the embedding model.
"""

from __future__ import annotations

import pytest

from livekit.plugins.voicemem.adapters.pg_graph import PgGraphStore
from livekit.plugins.voicemem.adapters.pg_vectors import PgVectorStore
from livekit.plugins.voicemem.adapters.pool import Database, make_pool
from livekit.plugins.voicemem.types import MemoryRecord, Scope
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

# The real store is built for 1536-wide vectors, which is what the schema was
# migrated with.
DIMS = 1536


@pytest.fixture
async def db(test_dsn):
    pool = await make_pool(test_dsn, schema="voicemem", min_size=1, max_size=2)
    database = Database(pool, schema="voicemem")
    yield database
    await database.close()


@pytest.fixture
def stores(db):
    embedder = FakeEmbedder(dimensions=DIMS)
    return PgVectorStore(db, embedder), PgGraphStore(db)


async def test_round_trip(stores, unique_user):
    vectors, _ = stores
    scope = Scope("acme", unique_user)
    ids = await vectors.add_records(
        scope, [MemoryRecord("", "User is allergic to peanuts")]
    )
    assert len(ids) == 1
    hits = await vectors.search("User is allergic to peanuts", scope=scope, top_k=3)
    assert [h.text for h in hits] == ["User is allergic to peanuts"]


async def test_tenants_cannot_see_each_other(stores, unique_user):
    vectors, _ = stores
    await vectors.add_records(Scope("acme", unique_user), [MemoryRecord("", "Acme salary is 100k")])
    hits = await vectors.search("Acme salary is 100k", scope=Scope("globex", unique_user), top_k=5)
    assert hits == []


async def test_assistant_rows_are_excluded_by_default(stores, unique_user):
    vectors, _ = stores
    scope = Scope("acme", unique_user)
    text = "You should try the pasta"
    await vectors.add_text(scope, text, attributed_to="assistant")
    assert text not in [h.text for h in await vectors.search(text, scope=scope, top_k=5)]
    included = await vectors.search(text, scope=scope, top_k=5, include_assistant=True)
    assert text in [h.text for h in included]


async def test_memory_id_filter_is_applied_in_sql(stores, unique_user):
    vectors, _ = stores
    scope = Scope("acme", unique_user)
    ids = await vectors.add_records(
        scope,
        [MemoryRecord("", "User likes tea"), MemoryRecord("", "User likes coffee")],
    )
    hits = await vectors.search("drink", scope=scope, top_k=5, memory_id_filter=[ids[0]])
    assert {h.memory_id for h in hits} <= {ids[0]}


async def test_an_empty_filter_returns_nothing_rather_than_everything(stores, unique_user):
    vectors, _ = stores
    scope = Scope("acme", unique_user)
    await vectors.add_records(scope, [MemoryRecord("", "User likes tea")])
    assert await vectors.search("tea", scope=scope, memory_id_filter=[]) == []


async def test_archived_memories_disappear_from_search(stores, unique_user):
    vectors, _ = stores
    scope = Scope("acme", unique_user)
    (mid,) = await vectors.add_records(scope, [MemoryRecord("", "User used to smoke")])
    assert await vectors.archive_memory(scope, mid)
    assert await vectors.search("User used to smoke", scope=scope) == []
    assert await vectors.unarchive_memory(scope, mid)
    assert await vectors.search("User used to smoke", scope=scope)


async def test_update_and_delete(stores, unique_user):
    vectors, _ = stores
    scope = Scope("acme", unique_user)
    (mid,) = await vectors.add_records(scope, [MemoryRecord("", "User is vegetarian")])
    assert await vectors.update_memory(scope, mid, "User is vegan")
    assert [r.text for r in await vectors.list_entries(scope)] == ["User is vegan"]
    assert await vectors.delete_memory(scope, mid)
    assert await vectors.list_entries(scope) == []


async def test_slot_tags_narrow_retrieval(stores, unique_user):
    vectors, graph = stores
    scope = Scope("acme", unique_user)
    ids = await vectors.add_records(
        scope,
        [MemoryRecord("", "User is allergic to peanuts"), MemoryRecord("", "User works at Figma")],
    )
    await graph.upsert_memory_tags(scope, ids[0], [("health", 0.9)])
    await graph.upsert_memory_tags(scope, ids[1], [("work", 0.9)])
    assert await graph.memory_ids_for_slots(scope, ["health"]) == [ids[0]]
    assert await graph.memory_tag_counts(scope) == {"health": 1, "work": 1}


async def test_traits_round_trip_and_merge_near_duplicates(stores, unique_user):
    _, graph = stores
    scope = Scope("acme", unique_user)
    embedder = FakeEmbedder(dimensions=DIMS)
    vec = await embedder.embed_query("dislikes being interrupted")

    first = await graph.add_trait(scope, slot="emotion", claim="dislikes being interrupted", embedding=vec)
    again = await graph.add_trait(scope, slot="emotion", claim="dislikes being interrupted", embedding=vec)
    # Identical claim and vector must merge, not accumulate paraphrases.
    assert first == again
    assert len(await graph.all_traits(scope)) == 1
    assert (await graph.search_traits(scope, embedding=vec, top_k=3))[0].claim == (
        "dislikes being interrupted"
    )


async def test_pop_touched_is_atomic_take_and_clear(stores, unique_user):
    _, graph = stores
    scope = Scope("acme", unique_user)
    await graph.touch(scope, "pending", "ref-1")
    await graph.touch(scope, "pending", "ref-2")
    assert sorted(await graph.pop_touched(scope, "pending")) == ["ref-1", "ref-2"]
    assert await graph.pop_touched(scope, "pending") == []


async def test_delete_user_removes_everything(stores, unique_user):
    vectors, graph = stores
    scope = Scope("acme", unique_user)
    (mid,) = await vectors.add_records(scope, [MemoryRecord("", "User likes tea")])
    await graph.upsert_memory_tags(scope, mid, [("daily_life", 0.9)])
    await graph.delete_user(scope)
    assert await vectors.list_entries(scope) == []
    assert await graph.memory_tag_counts(scope) == {}
