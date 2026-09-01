# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""The real ONNX model, not a stub.

``tests/unit/test_local_embed.py`` proves the arithmetic. This proves the thing
the arithmetic is applied to: that the pinned revision downloads, loads, returns
384 dimensions, and puts semantically related text near each other in both
English and Chinese.

Skipped unless the weights are already in the local HuggingFace cache, so a
fresh clone and CI stay fast and offline. Populate it with ``voicemem-db
fetch-model`` (or set ``VOICEMEM_TEST_LOCAL_MODEL=1`` to allow the download).
"""

from __future__ import annotations

import math
import os

import pytest

from livekit.plugins.voicemem.adapters.local_embed import (
    DIMENSIONS,
    MODEL_REPO,
    LocalEmbedder,
    download,
)

_ALLOW_NETWORK = os.environ.get("VOICEMEM_TEST_LOCAL_MODEL") == "1"


def _available() -> bool:
    if _ALLOW_NETWORK:
        return True
    try:
        download(allow_network=False)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _available(),
    reason="local model not in the HuggingFace cache; run 'voicemem-db fetch-model'",
)


@pytest.fixture(scope="module")
async def embedder():
    emb = await LocalEmbedder.load(allow_network=_ALLOW_NETWORK)
    yield emb
    await emb.aclose()


async def test_reports_the_pinned_model_and_width(embedder) -> None:
    assert embedder.model_name == MODEL_REPO
    assert embedder.dimensions == DIMENSIONS


async def test_real_vectors_are_the_declared_width_and_unit_length(embedder) -> None:
    """`dimensions` is a promise the schema's vector(N) column relies on."""
    vec = await embedder.embed_query("I am lactose intolerant")
    assert len(vec) == DIMENSIONS
    assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0, abs=1e-5)


async def test_related_text_scores_above_unrelated(embedder) -> None:
    docs = await embedder.embed_documents(
        [
            "User is lactose intolerant and always takes oat milk.",
            "User is training for the Toronto Marathon in October.",
        ]
    )
    query = await embedder.embed_query("I am at a cafe, should I order the latte?")
    dairy = sum(a * b for a, b in zip(docs[0], query, strict=True))
    running = sum(a * b for a, b in zip(docs[1], query, strict=True))
    assert dairy > running, "the dairy memory should win a question about a latte"


async def test_it_is_actually_multilingual(embedder) -> None:
    """An English-only model would pass every other test in this file and then
    quietly degrade slot routing, whose descriptions are bilingual."""
    docs = await embedder.embed_documents(
        ["用户对乳糖不耐受，总是喝燕麦奶。", "User is training for the Toronto Marathon."]
    )
    query = await embedder.embed_query("我可以喝牛奶吗？")
    dairy = sum(a * b for a, b in zip(docs[0], query, strict=True))
    running = sum(a * b for a, b in zip(docs[1], query, strict=True))
    assert dairy > running


async def test_query_and_passage_prefixes_produce_different_vectors(embedder) -> None:
    """If the prefixes were dropped these would be identical, and the model
    would be running in a mode it was not trained for."""
    as_query = await embedder.embed_query("oat milk")
    (as_passage,) = await embedder.embed_documents(["oat milk"])
    assert as_query != as_passage


async def test_batching_does_not_move_a_vector_enough_to_matter(embedder) -> None:
    """Batch size must not change what gets retrieved.

    It does change the vector slightly, and that is expected rather than a bug:
    the tokenizer pads to the longest text in the batch, so an int8 model runs
    different tensor shapes and its kernels are not bit-identical across them.
    Measured agreement is cosine 0.9991 at worst, and the ranking is unchanged,
    so this asserts the property that matters instead of exact equality, which a
    quantised model cannot offer.
    """
    texts = [
        "alpha",
        "beta and a somewhat longer sentence here",
        "User is lactose intolerant and always takes oat milk.",
        "delta",
    ]
    batched = await embedder.embed_documents(texts)

    alone: list[list[float]] = []
    for text in texts:
        alone.extend(await embedder.embed_documents([text]))

    for a, b in zip(batched, alone, strict=True):
        assert sum(x * y for x, y in zip(a, b, strict=True)) > 0.998

    query = await embedder.embed_query("should I order a latte?")

    def rank(vectors: list[list[float]]) -> list[int]:
        scores = [sum(x * y for x, y in zip(v, query, strict=True)) for v in vectors]
        return sorted(range(len(scores)), key=lambda i: -scores[i])

    assert rank(batched) == rank(alone), "batch size must not reorder results"
