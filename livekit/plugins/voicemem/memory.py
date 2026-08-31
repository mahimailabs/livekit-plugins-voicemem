# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""``VoiceMemory``: the public surface.

Two operations matter. :meth:`VoiceMemory.recall` runs inside the voice turn and
is budgeted; :meth:`VoiceMemory.remember` runs after it and is not. Everything
else exists to keep those two honest.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date
from typing import TYPE_CHECKING

from .config import Config
from .instrument import Recorder
from .leftbrain.rerank import dedupe_near
from .leftbrain.timeexpand import expand_relative_dates
from .log import logger
from .types import (
    Evidence,
    MemoryHit,
    MemoryRecord,
    RecallResult,
    RightBrainHit,
    Scope,
    TurnRecord,
)

if TYPE_CHECKING:
    from .adapters.pg_graph import PgGraphStore
    from .adapters.pg_vectors import PgVectorStore
    from .extraction.extractor import ConflictResolver, Extractor
    from .leftbrain.classify import SlotClassifier
    from .protocols import Embedder

__all__ = ["VoiceMemory", "render_context"]


def render_context(
    hits: tuple[MemoryHit, ...], rb_hits: tuple[RightBrainHit, ...], *, max_rb: int = 3
) -> str:
    """Turn retrieved memory into the block that goes into the prompt.

    The two halves are labelled differently and that is load-bearing. Facts may
    be repeated back to the user. Right-brain notes are internal: they read like
    "wants reassurance under pressure", and an agent that says that out loud
    sounds like it is reading a file about you rather than talking to you.
    """
    lines: list[str] = []
    if hits:
        lines.append("Things you remember about this user:")
        for h in hits:
            lines.append(f"- [{h.observed_at}] {h.text}" if h.observed_at else f"- {h.text}")
    if rb_hits:
        if lines:
            lines.append("")
        lines.append("How to speak to this user (internal, never say these aloud):")
        lines.extend(f"- {h.content}" for h in rb_hits[:max_rb])
        lines.append(
            "Let these shape your tone and what you bring up. Never state them back."
        )
    return "\n".join(lines)


def _dedupe_texts(texts: list[str], *, threshold: float = 0.5) -> list[str]:
    """Drop facts from one extraction that restate each other.

    Content-word overlap rather than trigrams: the duplicates this catches are
    paraphrases ("always orders oat milk" and "prefers oat milk"), which share
    vocabulary but little character-level structure. The threshold is lower than
    the retrieval-side one for the same reason.

    The longer text wins, since it is the one carrying the extra detail.
    """
    from .leftbrain.rerank import content_words

    kept: list[tuple[str, set[str]]] = []
    for text in sorted(texts, key=len, reverse=True):
        words = content_words(text)
        if not words:
            continue
        if any(
            len(words & other) / min(len(words), len(other)) >= threshold
            for _, other in kept
        ):
            logger.debug("voicemem: dropping intra-turn duplicate fact %r", text[:60])
            continue
        kept.append((text, words))
    return [t for t, _ in kept]


class VoiceMemory:
    """Long-term memory for one session.

    One instance per session, because it owns the prefetch state and the write
    queue. Build it with :func:`~livekit.plugins.voicemem.container.build`.
    """

    def __init__(
        self,
        *,
        config: Config,
        scope: Scope,
        vectors: PgVectorStore,
        graph: PgGraphStore,
        embedder: Embedder,
        classifier: SlotClassifier,
        extractor: Extractor,
        resolver: ConflictResolver,
        recorder: Recorder | None = None,
    ) -> None:
        self.config = config
        self.scope = scope
        self._vectors = vectors
        self._graph = graph
        self._embedder = embedder
        self._classifier = classifier
        self._extractor = extractor
        self._resolver = resolver
        self.recorder = recorder or Recorder()
        #: The last reply the agent gave, so the next turn can be attributed.
        self._last_reply = ""

    # -- read ---------------------------------------------------------------

    async def recall(self, query: str, *, top_k: int | None = None) -> RecallResult:
        """Retrieve memory relevant to ``query``.

        The left and right brains run concurrently. They do not depend on each
        other's results, and the right brain is pure SQL, so it finishes well
        inside the time the vector search takes.
        """
        started = time.perf_counter()
        top_k = top_k or self.config.top_k
        rec = self.recorder
        # Without this the per-stage timings accumulate across every recall for
        # the life of the session, so the reported "rank" time grows without
        # bound and the README numbers would be nonsense.
        turn = rec.begin_turn()

        # Relative expressions are dead in vector space: the store holds
        # absolute dates and the question holds none.
        expanded = expand_relative_dates(query)

        # Embed once. Slot classification, vector ranking and trait search all
        # need this same vector; a measured run with three separate round trips
        # spent roughly 200ms on each, which was most of the turn budget.
        with rec.stage("embed"):
            qvec = await self._embedder.embed_query(expanded)

        with rec.stage("classify"):
            classification = await self._classifier.classify(expanded, vector=qvec)

        candidate_ids: list[str] | None = None
        if classification.slots:
            with rec.stage("slot_filter"):
                ids = await self._graph.memory_ids_for_slots(
                    self.scope, list(classification.slots)
                )
            # Narrowing to nothing means the slot guess was wrong, not that the
            # user has no memories. Fall back to the whole corpus.
            candidate_ids = ids or None

        async def left() -> list[MemoryHit]:
            with rec.stage("rank"):
                return await self._vectors.search(
                    expanded,
                    scope=self.scope,
                    top_k=top_k,
                    memory_id_filter=candidate_ids,
                    rescue_k=self.config.rescue_k,
                    query_vector=qvec,
                )

        async def right() -> list[RightBrainHit]:
            if not self.config.enable_right_brain:
                return []
            with rec.stage("rb"):
                try:
                    traits = await self._graph.search_traits(
                        self.scope, embedding=qvec, top_k=self.config.right_brain_top_k
                    )
                    return [
                        RightBrainHit(content=t.claim, source=t.slot, priority=t.score)
                        for t in traits
                    ]
                except Exception:
                    # The right brain is an enhancement. Losing it must not lose
                    # the facts alongside it.
                    logger.debug("voicemem: right brain recall failed", exc_info=True)
                    return []

        hits, rb_hits = await asyncio.gather(left(), right())
        hits = dedupe_near(hits)

        total_ms = (time.perf_counter() - started) * 1000
        rec.record_recall(
            total_ms=total_ms, hits=len(hits), rb_hits=len(rb_hits), prefetched=False
        )

        if hits:
            # Fire and forget: heat feeds archival, and nothing waits on it.
            asyncio.create_task(  # noqa: RUF006
                self._safe(self._graph.record_memory_hits(self.scope, [h.memory_id for h in hits]))
            )

        timing = {k: round(v, 1) for k, v in turn.recall_ms.items()}
        timing["total"] = round(total_ms, 1)
        return RecallResult(
            block=render_context(
                tuple(hits), tuple(rb_hits), max_rb=self.config.right_brain_top_k
            ),
            hits=tuple(hits),
            rb_hits=tuple(rb_hits),
            timing=timing,
        )

    # -- write --------------------------------------------------------------

    async def remember(self, turn: TurnRecord) -> list[str]:
        """Extract and store one turn. Never call this on the critical path.

        Five LLM calls in the worst case, one of which upstream measured at 10.2
        seconds against a store of 95 memories.
        """
        text = turn.user_text.strip()
        if not text:
            return []

        observed = turn.observed_at or date.today().isoformat()

        existing_records = await self._vectors.existing_for_extractor(self.scope, limit=30)
        existing = [{"id": r.memory_id, "text": r.text} for r in existing_records]

        extraction = await self._extractor.extract(
            user_text=text,
            agent_reply=turn.agent_reply,
            existing=existing[:10],
            observation_date=observed[:10],
        )
        if not extraction.facts and not extraction.traits:
            return []

        stored: list[str] = []
        if extraction.facts:
            # Deduplicate within the turn before resolving. The conflict
            # resolver compares new facts against what is already stored, never
            # against each other, so one utterance yielding both "always orders
            # oat milk" and "prefers oat milk" stores both.
            new_texts = _dedupe_texts([f.text for f in extraction.facts])
            resolutions = await self._resolver.resolve(
                existing=existing, new_facts=new_texts
            )
            adds = [r for r in resolutions if r.action == "ADD"]
            if adds:
                stored = await self._vectors.add_records(
                    self.scope,
                    [MemoryRecord(memory_id="", text=r.text, observed_at=observed[:10]) for r in adds],
                )
                await self._tag(stored, [r.text for r in adds])

            for r in resolutions:
                if r.action == "UPDATE" and r.memory_id:
                    await self._vectors.update_memory(
                        self.scope, r.memory_id, r.text, observed_at=observed[:10]
                    )
                elif r.action == "DELETE" and r.memory_id:
                    await self._vectors.delete_memory(self.scope, r.memory_id)

        if self.config.enable_right_brain:
            await self._write_right_brain(extraction, turn=turn, observed=observed)

        # Store the agent's own reply so the next turn can be disambiguated.
        # Written with an assistant role, which search() excludes by default;
        # without that the agent starts retrieving and quoting itself.
        if turn.agent_reply.strip():
            await self._vectors.add_text(
                self.scope, turn.agent_reply.strip(), attributed_to="assistant"
            )
        return stored

    async def _tag(self, memory_ids: list[str], texts: list[str]) -> None:
        """File each stored fact under its life-domain slots."""
        for memory_id, text in zip(memory_ids, texts, strict=False):
            tags = await self._classifier.tag(text)
            if tags:
                await self._graph.upsert_memory_tags(self.scope, memory_id, tags)
                await self._graph.record_slot_cooccurrence(
                    self.scope, [slot for slot, _ in tags]
                )

    async def _write_right_brain(
        self, extraction, *, turn: TurnRecord, observed: str
    ) -> None:
        """Store what this turn revealed about the person, not the facts."""
        if not extraction.traits:
            return
        claims = [t.label for t in extraction.traits]
        vectors = await self._embedder.embed_documents(claims)
        for trait, vec in zip(extraction.traits, vectors, strict=True):
            try:
                await self._graph.add_trait(
                    self.scope,
                    slot=trait.slot,
                    claim=trait.label,
                    embedding=vec,
                    evidence=Evidence(
                        quote=turn.user_text[:500],
                        emotion=extraction.emotion,
                        at=observed[:10],
                    ),
                )
            except Exception:
                logger.debug("voicemem: trait write failed", exc_info=True)

        if extraction.emotion:
            # An emotion anchor is what makes this note findable later. Upstream
            # measured that essentially every right-brain note is reached
            # through one.
            note = await self._graph.upsert_right_memory(
                self.scope,
                memory_class="heartnote",
                content=f"Felt {extraction.emotion} when saying: {turn.user_text[:200]}",
                priority=0.6,
                metadata={"emotion": extraction.emotion, "observed_at": observed[:10]},
            )
            await self._graph.link_anchor(
                self.scope,
                right_memory_id=note,
                anchor_type="emotion",
                anchor_id=extraction.emotion,
                role="subject",
            )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    async def _safe(coro) -> None:
        try:
            await coro
        except Exception:
            logger.debug("voicemem: background operation failed", exc_info=True)

    def note_reply(self, reply: str) -> None:
        self._last_reply = (reply or "").strip()

    @property
    def last_reply(self) -> str:
        return self._last_reply
