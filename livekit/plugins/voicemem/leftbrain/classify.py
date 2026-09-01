# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Derived from VoiceMem (https://github.com/xzf-thu/VoiceMem), Apache-2.0,
# commit d587d424a02727d9ff3ef6f8e672a39a19ce64a7, files:
#   voicemem/leftbrain/cognitive_graph/local_query_classifier.py
#   voicemem/leftbrain/memory_repository_v2.py  (_ensure_slot_embeddings)
# Changes: async; the module-level _SLOT_EMBED_CACHE, which upstream keyed on
#   slot value alone so two embedders silently mixed vector spaces, is now
#   per-instance and keyed by model name.
# See CHANGES-FROM-UPSTREAM.md.
"""Which life domains a piece of text belongs to.

Embeddings, not an LLM. This runs on the retrieval path, inside the turn budget,
and on the write path once per stored fact. Upstream offers an LLM classifier as
an alternative and defaults to the local one for exactly this reason: a network
round trip per query is the difference between prefetch being worth doing and
not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..types import QueryClassification
from .slots import ALL_SLOT_V2_VALUES, SLOT_V2_DESCRIPTIONS

if TYPE_CHECKING:
    from ..protocols import Embedder

__all__ = ["SlotClassifier", "cosine"]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


class SlotClassifier:
    """Maps text to the slots it belongs to, by comparing against slot descriptions.

    The slot descriptions are deliberately bilingual, which is why they are
    copied verbatim from upstream: the same vector has to sit near both English
    and Chinese phrasings of the same domain.
    """

    __slots__ = ("_cache", "_embedder", "_margin", "_max_slots", "_min_score")

    def __init__(
        self,
        embedder: Embedder,
        *,
        min_score: float = 0.72,
        max_slots: int = 2,
        margin: float | None = None,
    ) -> None:
        self._embedder = embedder
        # Per instance and implicitly per embedder, rather than a module global
        # keyed on slot name only.
        self._cache: dict[str, list[float]] | None = None
        self._min_score = min_score
        self._max_slots = max_slots
        # An absolute floor only works when the model's cosines are spread out.
        # E5's are not: measured against these very slot descriptions it scores
        # the correct slot between 0.730 and 0.880 and the wrong ones up to
        # 0.880 too, so the classes overlap and no single cut separates them. At
        # 0.72 every slot passes for every memory and the filter quietly stops
        # filtering. Ranking is unaffected, so a relative window is used instead:
        # keep whatever sits within `margin` of the best. The container picks
        # which rule applies, because the right answer is a property of the
        # embedding model, not of the caller.
        self._margin = margin

    @property
    def enabled(self) -> bool:
        """False when narrowing is off, so callers can skip the work entirely."""
        return self._max_slots > 0

    def _select(self, scored: list[tuple[float, str]]) -> list[tuple[float, str]]:
        if not scored:
            return []
        best = scored[0][0]
        return [
            (score, name)
            for score, name in scored[: self._max_slots]
            if score >= self._min_score
            and (self._margin is None or score >= best - self._margin)
        ]

    async def _slot_vectors(self) -> dict[str, list[float]]:
        if self._cache is None:
            names = list(ALL_SLOT_V2_VALUES)
            texts = [SLOT_V2_DESCRIPTIONS[name] for name in names]
            vectors = await self._embedder.embed_documents(texts)
            self._cache = dict(zip(names, vectors, strict=True))
        return self._cache

    async def _scored(self, text: str, vector: Sequence[float] | None):
        vec = vector if vector is not None else await self._embedder.embed_query(text)
        return sorted(
            ((cosine(vec, v), name) for name, v in (await self._slot_vectors()).items()),
            reverse=True,
        )

    async def classify(
        self, text: str, *, vector: Sequence[float] | None = None
    ) -> QueryClassification:
        """Slots this text belongs to, most relevant first.

        Pass ``vector`` when the caller has already embedded this text. Recall
        needs the same vector for ranking and for trait search, and a measured
        run showed three separate round trips costing about 200ms each, which is
        most of the turn budget spent embedding one sentence three times.

        Returns no slots rather than a bad guess when nothing scores well. An
        empty result means the caller searches the whole corpus, which is slower
        but correct; a wrong slot silently hides the right memory.
        """
        text = (text or "").strip()
        if not text or not self.enabled:
            # With narrowing off there is nothing to compute: skipping here also
            # avoids embedding the seven slot descriptions that would never be
            # compared against anything.
            return QueryClassification()
        scored = await self._scored(text, vector)
        return QueryClassification(slots=tuple(name for _score, name in self._select(scored)))

    async def tag(
        self, text: str, *, vector: Sequence[float] | None = None
    ) -> tuple[tuple[str, float], ...]:
        """Slot tags with confidences, for writing to ``memory_tags``."""
        text = (text or "").strip()
        if not text or not self.enabled:
            return ()
        scored = await self._scored(text, vector)
        return tuple((name, round(score, 4)) for score, name in self._select(scored))
