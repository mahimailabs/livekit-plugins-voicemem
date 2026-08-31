# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""In-memory doubles for the four seams.

These exist because of the protocol design: nothing here inherits from anything
in the package, it just has the right methods. That is what lets the whole
retrieval chain run in a test with no database, no network and no API key.

The scoring is deliberately NOT reimplemented. Both this and the real Postgres
store call ``leftbrain.rerank.rank_hits``, so the two cannot drift apart on the
thing that actually decides retrieval quality.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Sequence
from typing import Any

from livekit.plugins.voicemem.leftbrain.rerank import rank_hits
from livekit.plugins.voicemem.types import (
    Evidence,
    MemoryHit,
    MemoryRecord,
    RightBrainHit,
    Scope,
    StoredTrait,
)

__all__ = ["FakeEmbedder", "FakeGraphStore", "FakeLLM", "FakeVectorStore"]


class FakeEmbedder:
    """Deterministic hash embeddings.

    Not semantic, and not pretending to be. Same text gives the same vector and
    different text gives a different one, which is all the storage and ranking
    layers actually require. Anything asserting on semantic similarity belongs
    in the integration suite with a real embedder.
    """

    def __init__(self, dimensions: int = 16, model: str = "fake-embed") -> None:
        self._dimensions = dimensions
        self._model = model
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dimensions)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vector(text)


class FakeLLM:
    """Returns queued responses and records what it was asked.

    ``purposes`` is the list a call-budget test asserts on, which is how a
    future reduction in the five-calls-per-turn cost stays reviewable.
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.purposes: list[str] = []
        self.prompts: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "fake-llm"

    def _next(self, default: Any) -> Any:
        return self.responses.pop(0) if self.responses else default

    async def complete_json(
        self, *, system: str, user: str, purpose: str = "", **_: Any
    ) -> dict[str, Any]:
        self.purposes.append(purpose)
        self.prompts.append((system, user))
        return self._next({})

    async def complete_text(
        self, *, system: str, user: str, purpose: str = "", **_: Any
    ) -> str:
        self.purposes.append(purpose)
        self.prompts.append((system, user))
        return self._next("")


class FakeVectorStore:
    """Dict-backed store with the real scoring contract."""

    def __init__(self, embedder: FakeEmbedder | None = None) -> None:
        self._embedder = embedder or FakeEmbedder()
        # (tenant, user) -> memory_id -> record plus its vector
        self._rows: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    def _bucket(self, scope: Scope) -> dict[str, dict[str, Any]]:
        return self._rows.setdefault((scope.tenant_id, scope.user_id), {})

    async def add_records(self, scope: Scope, records: Sequence[MemoryRecord]) -> list[str]:
        bucket = self._bucket(scope)
        ids: list[str] = []
        for rec in records:
            if not rec.text.strip():
                continue
            mid = rec.memory_id or str(uuid.uuid4())
            bucket[mid] = {
                "record": rec,
                "vector": (await self._embedder.embed_documents([rec.text]))[0],
                "archived": False,
            }
            ids.append(mid)
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
        return ids[0]

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
        qv = list(query_vector) if query_vector is not None else await self._embedder.embed_query(query)
        allowed = set(memory_id_filter) if memory_id_filter is not None else None
        if allowed is not None and not allowed:
            return []

        candidates = []
        for mid, row in self._bucket(scope).items():
            if row["archived"]:
                continue
            rec: MemoryRecord = row["record"]
            if not include_assistant and rec.role == "assistant":
                continue
            if allowed is not None and mid not in allowed:
                continue
            cosine = sum(a * b for a, b in zip(qv, row["vector"], strict=True))
            candidates.append(
                (mid, rec.text, cosine, rec.attributed_to, rec.metadata, rec.observed_at)
            )

        # The same function the Postgres store uses.
        return rank_hits(
            query, candidates, top_k=top_k, rescue_k=rescue_k, threshold=threshold
        )

    async def list_ids(self, scope: Scope) -> list[str]:
        return [m for m, r in self._bucket(scope).items() if not r["archived"]]

    async def list_entries(self, scope: Scope) -> list[MemoryRecord]:
        return [r["record"] for r in self._bucket(scope).values() if not r["archived"]]

    async def existing_for_extractor(
        self, scope: Scope, *, limit: int = 50
    ) -> list[MemoryRecord]:
        out = [
            r["record"]
            for r in self._bucket(scope).values()
            if not r["archived"] and r["record"].role != "assistant"
        ]
        return out[-limit:]

    async def memory_ids_with_time_expr(self, scope: Scope, *, kind: str) -> set[str]:
        from livekit.plugins.voicemem.leftbrain.rerank import _DATE_RE, _DURATION_RE

        pattern = _DURATION_RE if kind == "duration" else _DATE_RE
        return {
            mid
            for mid, row in self._bucket(scope).items()
            if pattern.search(row["record"].text)
        }

    async def update_memory(
        self,
        scope: Scope,
        memory_id: str,
        new_text: str,
        *,
        session_id: str | None = None,
        observed_at: str | None = None,
    ) -> bool:
        bucket = self._bucket(scope)
        if memory_id not in bucket:
            return False
        old: MemoryRecord = bucket[memory_id]["record"]
        bucket[memory_id]["record"] = MemoryRecord(
            memory_id=memory_id,
            text=new_text,
            role=old.role,
            attributed_to=old.attributed_to,
            metadata=old.metadata,
            observed_at=observed_at or old.observed_at,
            slot=old.slot,
        )
        bucket[memory_id]["vector"] = (await self._embedder.embed_documents([new_text]))[0]
        return True

    async def delete_memory(self, scope: Scope, memory_id: str) -> bool:
        return self._bucket(scope).pop(memory_id, None) is not None

    async def archive_memory(self, scope: Scope, memory_id: str) -> bool:
        row = self._bucket(scope).get(memory_id)
        if row is None or row["archived"]:
            return False
        row["archived"] = True
        return True

    async def unarchive_memory(self, scope: Scope, memory_id: str) -> bool:
        row = self._bucket(scope).get(memory_id)
        if row is None:
            return False
        row["archived"] = False
        return True


class FakeGraphStore:
    """In-memory slots, traits and anchors."""

    def __init__(self) -> None:
        self.tags: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {}
        self.traits: dict[tuple[str, str], list[StoredTrait]] = {}
        self.right: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.cooccurrence: list[tuple[str, str]] = []
        self.hits_recorded: list[str] = []
        self.touched: dict[tuple[str, str, str], list[str]] = {}

    @staticmethod
    def _key(scope: Scope) -> tuple[str, str]:
        return (scope.tenant_id, scope.user_id)

    async def upsert_memory_tags(
        self, scope: Scope, memory_id: str, tags: Sequence[tuple[str, float]]
    ) -> None:
        self.tags.setdefault(self._key(scope), {})[memory_id] = list(tags)

    async def memory_ids_for_slots(
        self, scope: Scope, slots: Sequence[str], *, min_confidence: float = 0.0
    ) -> list[str]:
        wanted = set(slots)
        return [
            mid
            for mid, tags in self.tags.get(self._key(scope), {}).items()
            if any(s in wanted and c >= min_confidence for s, c in tags)
        ]

    async def memory_tag_counts(self, scope: Scope) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tags in self.tags.get(self._key(scope), {}).values():
            for slot, _ in tags:
                counts[slot] = counts.get(slot, 0) + 1
        return counts

    async def record_slot_cooccurrence(self, scope: Scope, slots: Sequence[str]) -> None:
        self.cooccurrence.extend((a, b) for a in slots for b in slots if a < b)

    async def record_memory_hits(self, scope: Scope, memory_ids: Sequence[str]) -> None:
        self.hits_recorded.extend(memory_ids)

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
        trait_id = str(uuid.uuid4())
        self.traits.setdefault(self._key(scope), []).append(
            StoredTrait(trait_id=trait_id, slot=slot, claim=claim)
        )
        return trait_id

    async def search_traits(
        self, scope: Scope, *, embedding: Sequence[float], top_k: int = 5
    ) -> list[StoredTrait]:
        return self.traits.get(self._key(scope), [])[:top_k]

    async def all_traits(self, scope: Scope, *, per_slot: int = 8) -> list[StoredTrait]:
        return self.traits.get(self._key(scope), [])

    async def upsert_right_memory(self, scope: Scope, **kw: Any) -> str:
        memory_id = str(uuid.uuid4())
        self.right.setdefault(self._key(scope), []).append({"id": memory_id, **kw})
        return memory_id

    async def link_anchor(self, scope: Scope, **kw: Any) -> None:
        return None

    async def search_by_anchors(
        self,
        scope: Scope,
        *,
        emotion: str | None = None,
        entity_names: Sequence[str] = (),
        top_k: int = 5,
    ) -> list[RightBrainHit]:
        return [
            RightBrainHit(content=m["content"], source=m.get("memory_class", "heartnote"))
            for m in self.right.get(self._key(scope), [])[:top_k]
        ]

    async def upsert_entity(self, scope: Scope, *, name: str, **kw: Any) -> str:
        return f"entity_{name.lower().replace(' ', '_')}"

    async def link_memory(self, scope: Scope, **kw: Any) -> None:
        return None

    async def touch(self, scope: Scope, namespace: str, ref: str) -> None:
        self.touched.setdefault((*self._key(scope), namespace), []).append(ref)

    async def pop_touched(self, scope: Scope, namespace: str) -> list[str]:
        return self.touched.pop((*self._key(scope), namespace), [])

    async def delete_user(self, scope: Scope) -> None:
        key = self._key(scope)
        self.tags.pop(key, None)
        self.traits.pop(key, None)
        self.right.pop(key, None)
