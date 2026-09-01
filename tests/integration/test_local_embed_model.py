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


async def test_batching_drift_stays_below_the_retrieval_signal(embedder) -> None:
    """Batch size must not change the answer.

    Vectors do move a little with batch size, and that is expected: the
    tokenizer pads to the longest text in the batch, so an int8 model runs
    different tensor shapes and its kernels are not bit-identical across them.
    The size of that drift is architecture-dependent, because the pinned build
    is quantised for AVX512-VNNI and takes a fallback path on arm64.

    So neither the drift nor the scores are asserted against constants. What
    must hold is the relationship between them: the drift has to stay smaller
    than the gap between the best document and the next one, because that is
    exactly the condition under which it cannot change what is retrieved. Both
    quantities are measured here, on whatever machine is running, so the test
    calibrates itself.

    The corpus is three real memories with genuinely different relevance. An
    earlier version padded it with the word "alpha", on the assumption that
    nonsense would score low. It does not: E5 put it at 0.7808 against a latte
    question where the correct memory scored 0.7881, a gap of 0.007 that the
    drift duly crossed on x86. E5's absolute cosines sit in a narrow band and
    only large relative differences carry meaning, which is the same property
    that makes an absolute slot threshold unusable.
    """
    texts = [
        "User is lactose intolerant and always takes oat milk.",
        "User is training for the Toronto Marathon in October.",
        "User's daughter starts school in September.",
    ]
    query = await embedder.embed_query("I am at a cafe, should I order the latte?")

    def scores(vectors: list[list[float]]) -> list[float]:
        return [sum(x * y for x, y in zip(v, query, strict=True)) for v in vectors]

    batched = scores(await embedder.embed_documents(texts))
    alone: list[list[float]] = []
    for text in texts:
        alone.extend(await embedder.embed_documents([text]))
    single = scores(alone)

    assert max(range(len(batched)), key=lambda i: batched[i]) == 0
    assert max(range(len(single)), key=lambda i: single[i]) == 0

    ordered = sorted(single, reverse=True)
    margin = ordered[0] - ordered[1]
    drift = max(abs(b - s) for b, s in zip(batched, single, strict=True))
    assert drift < margin, (
        f"batching moved a score by {drift:.4f}, which is more than the "
        f"{margin:.4f} separating the best document from the next. At that "
        f"point batch size decides what gets retrieved."
    )
