# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""The local embedder's pooling, prefixes and shape contract.

Stubbed rather than loaded, so this runs on a fresh clone with no model, no
network and no key. What is under test is the arithmetic between the tokenizer
and the vector, which is where a local embedder goes silently wrong: CLS instead
of mean pooling, or pooling over padding, produces vectors that embed without
error and retrieve badly.

The real model is exercised in ``tests/integration/test_local_embed_model.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from livekit.plugins.voicemem.adapters.local_embed import (
    DIMENSIONS,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    LocalEmbedder,
)


class _Encoded:
    def __init__(self, ids: list[int], attention_mask: list[int]) -> None:
        self.ids = ids
        self.attention_mask = attention_mask


class _Tokenizer:
    """Records what it was asked to encode, and pads to the longest input."""

    def __init__(self, masks: list[list[int]] | None = None) -> None:
        self.seen: list[str] = []
        # When given, these are returned verbatim, so a test can pin the exact
        # padding layout instead of inferring it from word counts.
        self._masks = masks

    def encode_batch(self, texts: list[str]) -> list[_Encoded]:
        self.seen.extend(texts)
        if self._masks is not None:
            return [_Encoded([1] * len(m), list(m)) for m in self._masks]
        # One token per word of the text itself. The E5 prefix is dropped so a
        # test can state its expected token count without counting the prefix.
        counts = [len(t.split(": ", 1)[-1].split()) for t in texts]
        width = max(counts)
        return [
            _Encoded([1] * n + [0] * (width - n), [1] * n + [0] * (width - n)) for n in counts
        ]


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class _Session:
    """Returns a known hidden state so pooling can be checked by hand."""

    def __init__(
        self,
        hidden: np.ndarray | list[np.ndarray],
        inputs: tuple[str, ...] = ("input_ids", "attention_mask"),
    ) -> None:
        # A list is consumed one entry per call, so a test can tell which output
        # came back for which input. One fixed array for every call makes every
        # vector identical, and any assertion about ordering vacuous.
        self._queue = list(hidden) if isinstance(hidden, list) else None
        self._fixed = None if isinstance(hidden, list) else hidden
        self._inputs = inputs
        self.feeds: list[dict] = []

    def get_inputs(self) -> list[_Input]:
        return [_Input(n) for n in self._inputs]

    def run(self, _outputs, feed):
        self.feeds.append(feed)
        if self._fixed is not None:
            return [self._fixed]
        assert self._queue, "the stub ran out of prepared outputs"
        return [self._queue.pop(0)]


def _embedder(
    hidden: np.ndarray, masks: list[list[int]] | None = None, **kw
) -> tuple[LocalEmbedder, _Tokenizer, _Session]:
    tok, sess = _Tokenizer(masks), _Session(hidden, **kw)
    return LocalEmbedder(sess, tok), tok, sess


async def test_pooling_ignores_padding() -> None:
    """The whole point of masking. Position 1 is padding and must not count."""
    # One sequence, two positions. The second is padding, so the pooled vector
    # must equal the first position alone, not the average of both.
    hidden = np.array([[[3.0, 4.0], [100.0, 100.0]]], dtype=np.float32)
    emb, _, _ = _embedder(hidden, masks=[[1, 0]])
    (vec,) = await emb.embed_documents(["one"])
    assert vec == pytest.approx([0.6, 0.8])  # (3,4) normalised, padding excluded


async def test_output_is_l2_normalised() -> None:
    """The HNSW index is vector_cosine_ops, so unit length is the contract."""
    hidden = np.array([[[5.0, 12.0]]], dtype=np.float32)
    emb, _, _ = _embedder(hidden)
    (vec,) = await emb.embed_documents(["x"])
    assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)


async def test_mean_is_taken_over_real_tokens() -> None:
    hidden = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    emb, _, _ = _embedder(hidden, masks=[[1, 1]])
    (vec,) = await emb.embed_documents(["two words"])
    assert vec == pytest.approx([0.7071, 0.7071], abs=1e-4)


async def test_query_and_document_prefixes_differ() -> None:
    """E5 is asymmetric; collapsing the prefixes costs real recall."""
    hidden = np.array([[[1.0, 0.0]]], dtype=np.float32)
    emb, tok, _ = _embedder(hidden)
    await emb.embed_query("hello")
    await emb.embed_documents(["hello"])
    assert tok.seen == [f"{QUERY_PREFIX}hello", f"{PASSAGE_PREFIX}hello"]


async def test_blank_text_still_yields_a_usable_vector() -> None:
    """Callers zip results against inputs with strict=True, so a dropped element
    is a crash. Counting the results is not enough: that also accepts an
    all-zero vector, which pgvector's cosine operator cannot rank."""
    hidden = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    emb, tok, _ = _embedder(hidden, masks=[[1], [1]])
    out = await emb.embed_documents(["real", "   "])
    assert len(out) == 2
    for vec in out:
        assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)
    # The blank one was sent as a space, not dropped before tokenizing.
    assert len(tok.seen) == 2


async def test_empty_input_returns_empty() -> None:
    emb, _, sess = _embedder(np.zeros((1, 1, 2), dtype=np.float32))
    assert await emb.embed_documents([]) == []
    assert sess.feeds == [], "an empty batch must not reach the model"


async def test_order_is_preserved_across_batch_boundaries() -> None:
    """Batching is where order gets lost, so each call returns a distinguishable
    vector. Given one fixed output for every call the results are identical and
    a reordering could never be observed."""
    tok = _Tokenizer()
    sess = _Session(
        [
            np.array([[[1.0, 0.0]]], dtype=np.float32),
            np.array([[[0.0, 1.0]]], dtype=np.float32),
            np.array([[[-1.0, 0.0]]], dtype=np.float32),
        ]
    )
    emb = LocalEmbedder(sess, tok, batch_size=1)
    out = await emb.embed_documents(["a", "b", "c"])

    assert len(sess.feeds) == 3, "batch_size=1 should mean three separate runs"
    expected = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    for got, want in zip(out, expected, strict=True):
        assert got == pytest.approx(want)
    assert tok.seen == [f"{PASSAGE_PREFIX}a", f"{PASSAGE_PREFIX}b", f"{PASSAGE_PREFIX}c"]


async def test_token_type_ids_supplied_only_when_the_export_wants_them() -> None:
    hidden = np.array([[[1.0, 0.0]]], dtype=np.float32)
    tok = _Tokenizer()
    wants = _Session(hidden, inputs=("input_ids", "attention_mask", "token_type_ids"))
    await LocalEmbedder(wants, tok).embed_query("x")
    assert "token_type_ids" in wants.feeds[0]

    tok2 = _Tokenizer()
    without = _Session(hidden, inputs=("input_ids", "attention_mask"))
    await LocalEmbedder(without, tok2).embed_query("x")
    assert "token_type_ids" not in without.feeds[0]


async def test_with_recorder_rebinds_without_reloading_the_model() -> None:
    """Both halves matter: counters must reach the session's recorder, and the
    118MB of weights must not be loaded again to achieve it."""
    from livekit.plugins.voicemem.instrument import Recorder

    hidden = np.array([[[1.0, 0.0]]], dtype=np.float32)
    emb, _, sess = _embedder(hidden)
    recorder = Recorder()
    bound = emb.with_recorder(recorder)

    assert bound._session is sess, "the model must be shared, not reloaded"
    await bound.embed_query("x")
    assert recorder.counters.get("embed_calls") == 1
    assert emb._recorder is None, "the original must not be mutated"


async def test_closing_a_view_does_not_release_the_shared_model() -> None:
    """A per-session view must not take the model away from every other session
    in the worker when it closes."""
    hidden = np.array([[[1.0, 0.0]]], dtype=np.float32)
    emb, _, sess = _embedder(hidden)
    view = emb.with_recorder(None)
    await view.aclose()
    assert emb._session is sess
    await emb.embed_query("still works")


def test_dimensions_matches_the_model_card() -> None:
    assert DIMENSIONS == 384
