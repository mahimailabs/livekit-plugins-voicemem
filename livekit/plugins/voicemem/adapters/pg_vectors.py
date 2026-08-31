# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Replaces VoiceMem's voicemem/leftbrain/mem0_backend_store.py (Apache-2.0),
# commit d587d424a02727d9ff3ef6f8e672a39a19ce64a7.
# Changes: mem0 and Qdrant removed entirely in favour of PostgreSQL with
#   pgvector; async; tenant scoped; the two id spaces merged into one table;
#   memory_id_filter pushed into SQL instead of fetching 10,000 rows and
#   filtering in Python. The ordering and scoring contract from its search()
#   at lines 366-440 is preserved and now lives in leftbrain/rerank.py so the
#   real store and the test fake cannot diverge on it.
# See CHANGES-FROM-UPSTREAM.md.
"""Facts and their vectors, in PostgreSQL.

Upstream stored fact metadata in SQLite and the text plus vector in an embedded
Qdrant instance, joining the two by zipping id lists together. That is why a
foreign key from ``memory_tags`` could reference a row that only existed in the
vector store, and why the failure was swallowed into a log line. Here it is one
table with one id.

Two performance notes, both carried over from real upstream behaviour:

* When the caller has already narrowed to a candidate set, the filter goes into
  the SQL. Upstream fetched up to ten thousand rows and filtered them in Python.
* Ranking runs on exact distance over the narrowed set, not on the ANN index.
  That is deliberate. The HNSW index serves the unnarrowed case; once the graph
  has cut the candidates to a few dozen, an approximate search over a filtered
  set is both slower and less accurate than reading them.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from psycopg.types.json import Jsonb

from ..leftbrain.rerank import rank_hits
from ..types import MemoryHit, MemoryRecord, Scope
from .pg_types import Vector

if TYPE_CHECKING:
    from ..protocols import Embedder
    from .pool import Database

__all__ = ["PgVectorStore"]

#: How many rows to pull for reranking when nothing has narrowed the candidates.
#: Upstream used top_k * 3 with a floor of 20.
def _fetch_width(top_k: int) -> int:
    return max(top_k * 3, 20)


class PgVectorStore:
    """Satisfies :class:`~livekit.plugins.voicemem.protocols.VectorStore`."""

    __slots__ = ("_db", "_embedder")

    def __init__(self, db: Database, embedder: Embedder) -> None:
        self._db = db
        self._embedder = embedder

    # -- read ---------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        scope: Scope,
        top_k: int = 10,
        threshold: float | None = None,
        memory_id_filter: Sequence[str] | None = None,
        rescue_k: int = 0,
        include_assistant: bool = False,
        query_vector: Sequence[float] | None = None,
    ) -> list[MemoryHit]:
        q = query.strip()
        if not q:
            return []

        # Reuse the caller's vector when it has one. Recall needs the same
        # embedding for slot classification, ranking and trait search, and
        # embedding the sentence once per consumer costs about 200ms each.
        vec = Vector(
            query_vector if query_vector is not None else await self._embedder.embed_query(q)
        )

        # `<=>` is cosine distance, so similarity is 1 - distance. Selecting the
        # distance and converting here keeps the ORDER BY on the operator, which
        # is what the index can serve.
        conditions = ["tenant_id = %(tenant)s", "user_id = %(user)s", "archived_at IS NULL",
                      "embedding IS NOT NULL"]
        params: dict[str, Any] = {"tenant": scope.tenant_id, "user": scope.user_id, "vec": vec}

        if not include_assistant:
            # Without this the agent retrieves its own past replies and quotes
            # them back, and answers drift further from the truth every turn.
            conditions.append("role <> 'assistant'")

        if memory_id_filter is not None:
            ids = [str(m) for m in memory_id_filter]
            if not ids:
                return []
            conditions.append("id = ANY(%(ids)s::uuid[])")
            params["ids"] = ids
            limit = len(ids)
        else:
            limit = _fetch_width(top_k)
        params["limit"] = limit

        sql = f"""
            SELECT id::text AS id,
                   content,
                   attributed_to,
                   metadata,
                   to_char(observed_on, 'YYYY-MM-DD') AS observed_at,
                   1 - (embedding <=> %(vec)s) AS cosine
            FROM memories
            WHERE {' AND '.join(conditions)}
            ORDER BY embedding <=> %(vec)s
            LIMIT %(limit)s
        """

        async with self._db.tx(scope) as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()

        return rank_hits(
            q,
            (
                (
                    r["id"],
                    r["content"],
                    float(r["cosine"]),
                    r["attributed_to"],
                    r["metadata"] or {},
                    r["observed_at"] or "",
                )
                for r in rows
            ),
            top_k=top_k,
            rescue_k=rescue_k,
            threshold=threshold,
        )

    async def list_ids(self, scope: Scope) -> list[str]:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "SELECT id::text AS id FROM memories "
                "WHERE tenant_id = %s AND user_id = %s AND archived_at IS NULL",
                (scope.tenant_id, scope.user_id),
            )
            return [r["id"] for r in await cur.fetchall()]

    async def list_entries(self, scope: Scope) -> list[MemoryRecord]:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                SELECT id::text AS id, content, role, attributed_to, metadata, slot,
                       to_char(observed_on, 'YYYY-MM-DD') AS observed_at
                FROM memories
                WHERE tenant_id = %s AND user_id = %s AND archived_at IS NULL
                ORDER BY created_at
                """,
                (scope.tenant_id, scope.user_id),
            )
            return [_to_record(r) for r in await cur.fetchall()]

    async def existing_for_extractor(
        self, scope: Scope, *, limit: int = 50
    ) -> list[MemoryRecord]:
        """Most recent facts, shown to the conflict resolver.

        Assistant rows are excluded: the resolver decides what is true about the
        user, and the agent's own replies are not evidence of that.
        """
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                SELECT id::text AS id, content, role, attributed_to, metadata, slot,
                       to_char(observed_on, 'YYYY-MM-DD') AS observed_at
                FROM memories
                WHERE tenant_id = %s AND user_id = %s AND archived_at IS NULL
                  AND role <> 'assistant'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (scope.tenant_id, scope.user_id, limit),
            )
            return [_to_record(r) for r in await cur.fetchall()]

    async def memory_ids_with_time_expr(self, scope: Scope, *, kind: str) -> set[str]:
        """Ids whose text carries a date or a duration.

        The regex runs in Postgres rather than over every row in Python.
        ``~*`` is case-insensitive POSIX matching; the alternations mirror the
        recognisers in ``leftbrain/rerank.py``.
        """
        if kind == "duration":
            pattern = r"[0-9]+ *(year|month|week|day|hour|minute)s?|[0-9]+ *(年|个月|周|天|小时|分钟)"
        else:
            pattern = (
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* +[0-9]{1,2},? +(19|20)[0-9]{2}"
                r"|(19|20)[0-9]{2}-[0-9]{2}-[0-9]{2}"
                r"|[0-9]{1,2}月[0-9]{1,2}[日号]"
            )
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "SELECT id::text AS id FROM memories "
                "WHERE tenant_id = %s AND user_id = %s AND archived_at IS NULL "
                "AND content ~* %s",
                (scope.tenant_id, scope.user_id, pattern),
            )
            return {r["id"] for r in await cur.fetchall()}

    # -- write --------------------------------------------------------------

    async def add_records(self, scope: Scope, records: Sequence[MemoryRecord]) -> list[str]:
        texts = [r.text for r in records if r.text.strip()]
        if not texts:
            return []
        vectors = await self._embedder.embed_documents(texts)

        rows: list[tuple[Any, ...]] = []
        ids: list[str] = []
        vi = 0
        for rec in records:
            if not rec.text.strip():
                continue
            new_id = rec.memory_id or str(uuid.uuid4())
            ids.append(new_id)
            rows.append(
                (
                    scope.tenant_id,
                    new_id,
                    scope.user_id,
                    rec.text,
                    Vector(vectors[vi]),
                    rec.slot,
                    rec.role,
                    rec.attributed_to,
                    Jsonb(rec.metadata or {}),
                    rec.observed_at or None,
                )
            )
            vi += 1

        async with self._db.tx(scope) as conn:
            cur = conn.cursor()
            await cur.executemany(
                """
                INSERT INTO memories
                    (tenant_id, id, user_id, content, embedding, slot, role,
                     attributed_to, metadata, observed_on)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::date)
                ON CONFLICT (tenant_id, id) DO NOTHING
                """,
                rows,
            )
        return ids

    async def add_text(
        self,
        scope: Scope,
        text: str,
        *,
        attributed_to: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        role = "assistant" if attributed_to == "assistant" else "user"
        ids = await self.add_records(
            scope,
            [
                MemoryRecord(
                    memory_id="",
                    text=text,
                    role=role,  # type: ignore[arg-type]
                    attributed_to=attributed_to,
                    metadata=metadata or {},
                )
            ],
        )
        if not ids:
            raise ValueError("cannot store empty text")
        return ids[0]

    async def update_memory(
        self,
        scope: Scope,
        memory_id: str,
        new_text: str,
        *,
        session_id: str | None = None,
        observed_at: str | None = None,
    ) -> bool:
        text = new_text.strip()
        if not text:
            return False
        vec = Vector((await self._embedder.embed_documents([text]))[0])
        extra: dict[str, Any] = {}
        if session_id is not None:
            extra["session_id"] = session_id

        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                """
                UPDATE memories
                   SET content = %s,
                       embedding = %s,
                       metadata = metadata || %s::jsonb,
                       observed_on = COALESCE(%s::date, observed_on),
                       updated_at = now()
                 WHERE tenant_id = %s AND id = %s::uuid
                """,
                (text, vec, Jsonb(extra), observed_at, scope.tenant_id, memory_id),
            )
            return cur.rowcount > 0

    async def delete_memory(self, scope: Scope, memory_id: str) -> bool:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "DELETE FROM memories WHERE tenant_id = %s AND id = %s::uuid",
                (scope.tenant_id, memory_id),
            )
            return cur.rowcount > 0

    async def archive_memory(self, scope: Scope, memory_id: str) -> bool:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "UPDATE memories SET archived_at = now() "
                "WHERE tenant_id = %s AND id = %s::uuid AND archived_at IS NULL",
                (scope.tenant_id, memory_id),
            )
            return cur.rowcount > 0

    async def unarchive_memory(self, scope: Scope, memory_id: str) -> bool:
        async with self._db.tx(scope) as conn:
            cur = await conn.execute(
                "UPDATE memories SET archived_at = NULL "
                "WHERE tenant_id = %s AND id = %s::uuid",
                (scope.tenant_id, memory_id),
            )
            return cur.rowcount > 0


def _to_record(row: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row["id"],
        text=row["content"],
        role=row.get("role", "user"),
        attributed_to=row.get("attributed_to", "user"),
        metadata=row.get("metadata") or {},
        observed_at=row.get("observed_at") or "",
        slot=row.get("slot") or "",
    )
