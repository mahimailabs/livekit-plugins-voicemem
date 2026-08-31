# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Replaces the SQLite stores in VoiceMem (Apache-2.0):
#   voicemem/leftbrain/cognitive_graph/store.py, store_v2.py
#   voicemem/leftbrain/slot_split/graph_entity_store.py
#   voicemem/rightbrain/store.py, traits_store.py
#   voicemem/utils/common/session_tracker.py
# Changes: async psycopg; tenant scoped; the O(n) Python cosine scans replaced
#   with nearest-neighbour SQL; select-then-delete replaced by DELETE RETURNING;
#   read-modify-write metadata merge replaced by a jsonb concat.
# See CHANGES-FROM-UPSTREAM.md.
"""The relational half of memory: slots, entities, traits and anchors."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from psycopg.types.json import Jsonb

from ..types import Evidence, RightBrainHit, Scope, StoredTrait
from .pg_types import Vector

if TYPE_CHECKING:
    from .pool import Database

__all__ = ["PgGraphStore"]


class PgGraphStore:
    """Implements the seven narrow graph protocols against PostgreSQL."""

    __slots__ = ("_db",)

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- SlotIndex ----------------------------------------------------------

    async def upsert_memory_tags(
        self, scope: Scope, memory_id: str, tags: Sequence[tuple[str, float]]
    ) -> None:
        if not tags:
            return
        async with self._db.tx(scope) as conn:
            cur = conn.cursor()
            await cur.executemany(
                """
                INSERT INTO memory_tags (tenant_id, memory_id, user_id, slot, confidence)
                VALUES (%s, %s::uuid, %s, %s, %s)
                ON CONFLICT (tenant_id, memory_id, slot)
                DO UPDATE SET confidence = GREATEST(memory_tags.confidence, EXCLUDED.confidence)
                """,
                [
                    (scope.tenant_id, memory_id, scope.user_id, slot, conf)
                    for slot, conf in tags
                ],
            )

    async def memory_ids_for_slots(
        self, scope: Scope, slots: Sequence[str], *, min_confidence: float = 0.0
    ) -> list[str]:
        """Ids tagged with any of these slots.

        This is the narrowing step: it turns "what am I allergic to" into the
        subset of memories filed under health, which is what makes retrieval
        more accurate than flat vector search over everything.
        """
        if not slots:
            return []
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                SELECT DISTINCT memory_id::text AS id
                FROM memory_tags
                WHERE tenant_id = %s AND user_id = %s
                  AND slot = ANY(%s) AND confidence >= %s
                """,
                (scope.tenant_id, scope.user_id, list(slots), min_confidence),
            )
            return [r["id"] for r in await cur.fetchall()]

    async def memory_tag_counts(self, scope: Scope) -> dict[str, int]:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "SELECT slot, count(*) AS n FROM memory_tags "
                "WHERE tenant_id = %s AND user_id = %s GROUP BY slot",
                (scope.tenant_id, scope.user_id),
            )
            return {r["slot"]: r["n"] for r in await cur.fetchall()}

    async def upsert_slot_summary(self, scope: Scope, slot: str, summary: str) -> None:
        async with self._db.tx(scope) as conn:
            await conn.execute(
                """
                INSERT INTO slot_summaries (tenant_id, user_id, slot, summary, updated_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (tenant_id, user_id, slot)
                DO UPDATE SET summary = EXCLUDED.summary, updated_at = now()
                """,
                (scope.tenant_id, scope.user_id, slot, summary),
            )

    async def get_slot_summaries(self, scope: Scope, slots: Sequence[str]) -> dict[str, str]:
        if not slots:
            return {}
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "SELECT slot, summary FROM slot_summaries "
                "WHERE tenant_id = %s AND user_id = %s AND slot = ANY(%s)",
                (scope.tenant_id, scope.user_id, list(slots)),
            )
            return {r["slot"]: r["summary"] for r in await cur.fetchall()}

    async def record_slot_cooccurrence(self, scope: Scope, slots: Sequence[str]) -> None:
        """Note that these slots were activated together.

        This is how related-slot edges emerge from actual use rather than being
        declared up front.
        """
        pairs = [(a, b) for a in slots for b in slots if a < b]
        if not pairs:
            return
        async with self._db.tx(scope) as conn:
            cur = conn.cursor()
            await cur.executemany(
                """
                INSERT INTO slot_macro_edges (tenant_id, user_id, slot_a, slot_b, weight, updated_at)
                VALUES (%s, %s, %s, %s, 1, now())
                ON CONFLICT (tenant_id, user_id, slot_a, slot_b)
                DO UPDATE SET weight = slot_macro_edges.weight + 1, updated_at = now()
                """,
                [(scope.tenant_id, scope.user_id, a, b) for a, b in pairs],
            )

    async def get_macro_related_slots(
        self, scope: Scope, slot: str, *, limit: int = 3
    ) -> list[str]:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                SELECT CASE WHEN slot_a = %(slot)s THEN slot_b ELSE slot_a END AS other
                FROM slot_macro_edges
                WHERE tenant_id = %(tenant)s AND user_id = %(user)s
                  AND (slot_a = %(slot)s OR slot_b = %(slot)s)
                ORDER BY weight DESC
                LIMIT %(limit)s
                """,
                {
                    "tenant": scope.tenant_id,
                    "user": scope.user_id,
                    "slot": slot,
                    "limit": limit,
                },
            )
            return [r["other"] for r in await cur.fetchall()]

    # -- EntityGraph --------------------------------------------------------

    async def upsert_entity(
        self,
        scope: Scope,
        *,
        name: str,
        entity_type: str,
        slot: str,
        description: str = "",
        embedding: Sequence[float] | None = None,
        similarity_threshold: float = 0.88,
    ) -> str:
        """Create or merge an entity by semantic nearest neighbour.

        Upstream loaded every entity of the type into Python and compared
        cosines one at a time on each mention, which is O(n) per entity per
        turn. Here the nearest neighbour is one indexed query scoped to
        (tenant, user, entity_type), which is far more selective than any
        approximate search would be.
        """
        name_norm = name.strip().lower()
        if not name_norm:
            raise ValueError("entity name must not be empty")

        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                SELECT id FROM entities
                WHERE tenant_id = %s AND user_id = %s AND name_norm = %s AND entity_type = %s
                """,
                (scope.tenant_id, scope.user_id, name_norm, entity_type),
            )
            if row := await cur.fetchone():
                return str(row["id"])

            if embedding is not None:
                cur = await conn.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s) AS sim
                    FROM entities
                    WHERE tenant_id = %s AND user_id = %s AND entity_type = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT 1
                    """,
                    (
                        Vector(embedding),
                        scope.tenant_id,
                        scope.user_id,
                        entity_type,
                        Vector(embedding),
                    ),
                )
                if (row := await cur.fetchone()) and float(row["sim"]) >= similarity_threshold:
                    return str(row["id"])

            entity_id = f"{entity_type}_{uuid.uuid4().hex[:12]}"
            await conn.execute(
                """
                INSERT INTO entities
                    (tenant_id, id, user_id, entity_type, name, name_norm, slot,
                     description, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, id) DO NOTHING
                """,
                (
                    scope.tenant_id,
                    entity_id,
                    scope.user_id,
                    entity_type,
                    name.strip(),
                    name_norm,
                    slot,
                    description,
                    Vector(embedding) if embedding is not None else None,
                ),
            )
            return entity_id

    async def link_memory(
        self,
        scope: Scope,
        *,
        memory_id: str,
        entity_id: str,
        role: str = "context",
        relation_hint: str | None = None,
    ) -> None:
        async with self._db.tx(scope) as conn:
            await conn.execute(
                """
                INSERT INTO entity_memory_links
                    (tenant_id, id, memory_id, entity_id, user_id, role, relation_hint)
                VALUES (%s, %s, %s::uuid, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, memory_id, entity_id) DO NOTHING
                """,
                (
                    scope.tenant_id,
                    str(uuid.uuid4()),
                    memory_id,
                    entity_id,
                    scope.user_id,
                    role,
                    relation_hint,
                ),
            )

    async def find_entities_by_name(self, scope: Scope, names: Sequence[str]) -> list[str]:
        if not names:
            return []
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "SELECT id FROM entities WHERE tenant_id = %s AND user_id = %s "
                "AND name_norm = ANY(%s)",
                (scope.tenant_id, scope.user_id, [n.strip().lower() for n in names]),
            )
            return [str(r["id"]) for r in await cur.fetchall()]

    async def memory_ids_for_entities(
        self, scope: Scope, entity_ids: Sequence[str]
    ) -> list[str]:
        if not entity_ids:
            return []
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "SELECT DISTINCT memory_id::text AS id FROM entity_memory_links "
                "WHERE tenant_id = %s AND user_id = %s AND entity_id = ANY(%s)",
                (scope.tenant_id, scope.user_id, list(entity_ids)),
            )
            return [r["id"] for r in await cur.fetchall()]

    async def record_memory_hits(self, scope: Scope, memory_ids: Sequence[str]) -> None:
        """Mark memories as retrieved, feeding the heat score used for archival."""
        if not memory_ids:
            return
        async with self._db.tx(scope) as conn:
            await conn.execute(
                "UPDATE memories SET heat = heat + 1, last_hit_at = now() "
                "WHERE tenant_id = %s AND id = ANY(%s::uuid[])",
                (scope.tenant_id, [str(m) for m in memory_ids]),
            )

    # -- TraitGraph ---------------------------------------------------------

    async def add_trait(
        self,
        scope: Scope,
        *,
        slot: str,
        claim: str,
        evidence: Evidence | None = None,
        embedding: Sequence[float] | None = None,
        similarity_threshold: float = 0.93,
    ) -> str:
        """Store a judgement about the user, merging near-duplicates.

        Without the merge the same observation restated slightly differently
        accumulates as separate traits, and the profile becomes a list of
        paraphrases.
        """
        claim = claim.strip()
        if not claim:
            raise ValueError("claim must not be empty")

        async with self._db.tx(scope) as conn:
            trait_id: str | None = None
            if embedding is not None:
                cur = await conn.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s) AS sim
                    FROM rb_traits
                    WHERE tenant_id = %s AND user_id = %s AND slot = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT 1
                    """,
                    (Vector(embedding), scope.tenant_id, scope.user_id, slot, Vector(embedding)),
                )
                if (row := await cur.fetchone()) and float(row["sim"]) >= similarity_threshold:
                    trait_id = str(row["id"])
                    await conn.execute(
                        "UPDATE rb_traits SET updated_at = now() "
                        "WHERE tenant_id = %s AND id = %s::uuid",
                        (scope.tenant_id, trait_id),
                    )

            if trait_id is None:
                trait_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO rb_traits
                        (tenant_id, id, user_id, slot, claim, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        scope.tenant_id,
                        trait_id,
                        scope.user_id,
                        slot,
                        claim,
                        Vector(embedding) if embedding is not None else None,
                    ),
                )

            if evidence is not None and evidence.quote:
                await conn.execute(
                    """
                    INSERT INTO rb_evidence
                        (tenant_id, id, trait_id, user_id, quote, emotion, cause,
                         cause_id, observed_at)
                    VALUES (%s, %s, %s::uuid, %s, %s, %s, %s, %s::uuid, %s::timestamptz)
                    """,
                    (
                        scope.tenant_id,
                        str(uuid.uuid4()),
                        trait_id,
                        scope.user_id,
                        evidence.quote,
                        evidence.emotion,
                        evidence.cause,
                        evidence.cause_id or None,
                        evidence.at or None,
                    ),
                )
            return trait_id

    async def search_traits(
        self, scope: Scope, *, embedding: Sequence[float], top_k: int = 5
    ) -> list[StoredTrait]:
        """Traits relevant to this query, rather than the whole profile.

        Upstream returned the same static summary lines on every turn, which
        reads to the user as the agent reciting a dossier instead of
        remembering something.
        """
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                SELECT id::text AS id, slot, claim, confidence,
                       1 - (embedding <=> %s) AS sim
                FROM rb_traits
                WHERE tenant_id = %s AND user_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (Vector(embedding), scope.tenant_id, scope.user_id, Vector(embedding), top_k),
            )
            return [
                StoredTrait(
                    trait_id=r["id"],
                    slot=r["slot"],
                    claim=r["claim"],
                    confidence=float(r["confidence"]),
                    score=float(r["sim"]),
                )
                for r in await cur.fetchall()
            ]

    async def all_traits(self, scope: Scope, *, per_slot: int = 8) -> list[StoredTrait]:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                SELECT id::text AS id, slot, claim, confidence FROM (
                    SELECT *, row_number() OVER (PARTITION BY slot ORDER BY updated_at DESC) AS rn
                    FROM rb_traits WHERE tenant_id = %s AND user_id = %s
                ) t WHERE rn <= %s
                """,
                (scope.tenant_id, scope.user_id, per_slot),
            )
            return [
                StoredTrait(
                    trait_id=r["id"],
                    slot=r["slot"],
                    claim=r["claim"],
                    confidence=float(r["confidence"]),
                )
                for r in await cur.fetchall()
            ]

    # -- AnchorGraph --------------------------------------------------------

    async def upsert_right_memory(
        self,
        scope: Scope,
        *,
        memory_class: str,
        content: str,
        priority: float = 0.5,
        confidence: float = 1.0,
        ttl_class: str = "long_term",
        condition: str | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_memory_ids: Sequence[str] = (),
    ) -> str:
        memory_id = str(uuid.uuid4())
        async with self._db.tx(scope) as conn:
            await conn.execute(
                """
                INSERT INTO right_brain_memories
                    (tenant_id, id, user_id, memory_class, content, condition_text,
                     priority, confidence, ttl_class, metadata, evidence_memory_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    scope.tenant_id,
                    memory_id,
                    scope.user_id,
                    memory_class,
                    content,
                    condition,
                    priority,
                    confidence,
                    ttl_class,
                    Jsonb(metadata or {}),
                    Jsonb(list(evidence_memory_ids)),
                ),
            )
        return memory_id

    async def link_anchor(
        self,
        scope: Scope,
        *,
        right_memory_id: str,
        anchor_type: str,
        anchor_id: str | None,
        role: str = "context",
        weight: float = 1.0,
        confidence: float = 1.0,
    ) -> None:
        async with self._db.tx(scope) as conn:
            await conn.execute(
                """
                INSERT INTO right_brain_anchor_links
                    (tenant_id, id, user_id, right_memory_id, anchor_type, anchor_id,
                     role, weight, confidence)
                VALUES (%s, %s, %s, %s::uuid, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    scope.tenant_id,
                    str(uuid.uuid4()),
                    scope.user_id,
                    right_memory_id,
                    anchor_type,
                    anchor_id,
                    role,
                    weight,
                    confidence,
                ),
            )

    async def search_by_anchors(
        self,
        scope: Scope,
        *,
        emotion: str | None = None,
        entity_names: Sequence[str] = (),
        top_k: int = 5,
    ) -> list[RightBrainHit]:
        """Right-brain notes reachable from the current turn's anchors.

        Zero LLM calls, and it runs concurrently with the left brain's ranking,
        so it costs no wall clock inside the turn.
        """
        anchors: list[tuple[str, str]] = []
        if emotion:
            anchors.append(("emotion", emotion))
        anchors.extend(("entity", n.strip().lower()) for n in entity_names if n.strip())

        async with self._db.tx(scope) as conn:
            if anchors:
                cur = await conn.execute(
                    """
                    SELECT DISTINCT m.id::text AS id, m.content, m.memory_class,
                           m.priority, l.weight
                    FROM right_brain_memories m
                    JOIN right_brain_anchor_links l
                      ON l.tenant_id = m.tenant_id AND l.right_memory_id = m.id
                    WHERE m.tenant_id = %s AND m.user_id = %s
                      AND (l.anchor_type, coalesce(l.anchor_id, '')) = ANY(%s::record[])
                    ORDER BY m.priority DESC
                    LIMIT %s
                    """,
                    (scope.tenant_id, scope.user_id, anchors, top_k),
                )
            else:
                cur = await conn.execute(
                    """
                    SELECT id::text AS id, content, memory_class, priority, 1.0 AS weight
                    FROM right_brain_memories
                    WHERE tenant_id = %s AND user_id = %s
                    ORDER BY priority DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (scope.tenant_id, scope.user_id, top_k),
                )
            return [
                RightBrainHit(
                    content=r["content"],
                    source=r["memory_class"],
                    priority=float(r["priority"]) * float(r["weight"]),
                )
                for r in await cur.fetchall()
            ]

    # -- SessionState -------------------------------------------------------

    async def record_turn(self, scope: Scope, session_id: str) -> dict[str, Any]:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                INSERT INTO session_state (tenant_id, user_id, last_session_id, turn_count, updated_at)
                VALUES (%s, %s, %s, 1, now())
                ON CONFLICT (tenant_id, user_id) DO UPDATE SET
                    turn_count = CASE
                        WHEN session_state.last_session_id IS DISTINCT FROM EXCLUDED.last_session_id
                        THEN 1 ELSE session_state.turn_count + 1 END,
                    last_session_id = EXCLUDED.last_session_id,
                    updated_at = now()
                RETURNING turn_count, (turn_count = 1) AS new_session
                """,
                (scope.tenant_id, scope.user_id, session_id),
            )
            row = await cur.fetchone()
            return dict(row) if row else {}

    async def touch(self, scope: Scope, namespace: str, ref: str) -> None:
        async with self._db.tx(scope) as conn:
            await conn.execute(
                "INSERT INTO touched_refs (tenant_id, user_id, namespace, ref) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (scope.tenant_id, scope.user_id, namespace, ref),
            )

    async def pop_touched(self, scope: Scope, namespace: str) -> list[str]:
        """Take and clear, atomically.

        Upstream did a SELECT then a DELETE in two statements, which under
        concurrent ingests either double-processes a ref or loses one.
        """
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "DELETE FROM touched_refs WHERE tenant_id = %s AND user_id = %s "
                "AND namespace = %s RETURNING ref",
                (scope.tenant_id, scope.user_id, namespace),
            )
            return [r["ref"] for r in await cur.fetchall()]

    async def delete_user(self, scope: Scope) -> None:
        """Erase everything for one user. Cascades handle the dependent rows."""
        async with self._db.tx(scope) as conn:
            for table in (
                "memories",
                "entities",
                "right_brain_memories",
                "rb_traits",
                "memory_tags",
                "slot_summaries",
                "slot_profiles",
                "slot_macro_edges",
                "dynamic_slots",
                "graph_entities",
                "session_state",
                "touched_refs",
                "kv",
                "ingest_outbox",
            ):
                await conn.execute(
                    f"DELETE FROM {table} WHERE tenant_id = %s AND user_id = %s",
                    (scope.tenant_id, scope.user_id),
                )
