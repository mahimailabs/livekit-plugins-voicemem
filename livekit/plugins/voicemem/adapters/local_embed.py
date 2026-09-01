# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Derived from VoiceMem (https://github.com/xzf-thu/VoiceMem), Apache-2.0,
# commit d587d424a02727d9ff3ef6f8e672a39a19ce64a7, file:
#   voicemem/leftbrain/local_e5_embedder.py
# Changes: ONNX Runtime instead of sentence-transformers, so no torch; async,
#   with inference offloaded off the event loop; the module-level lru_cache
#   singleton is now an injected instance; pooling is implemented here rather
#   than inherited from SentenceTransformer.
# See CHANGES-FROM-UPSTREAM.md.
"""The embedding seam, run in process. No network on the read path.

One OpenAI embedding round trip is 180.5ms p50 and 812.5ms p95, against a 0.6s
recall budget: the tail does not merely run slow, it blows the budget and
``hooks.py`` injects nothing at all. The same query embeds here in 2.6ms.

ONNX Runtime rather than sentence-transformers, because sentence-transformers
means torch, and a voice agent does not need two gigabytes of CUDA-capable
tensor library to turn a sentence into 384 floats. Almost every LiveKit agent
already has onnxruntime installed: both ``livekit-plugins-silero`` and
``livekit-plugins-turn-detector`` depend on it.

The model is multilingual on purpose. The slot descriptions this feeds are
deliberately bilingual, and an English-only model would quietly degrade slot
routing rather than fail.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..log import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..instrument import Recorder

__all__ = [
    "DIMENSIONS",
    "MODEL_FILE",
    "MODEL_REPO",
    "MODEL_REVISION",
    "LocalEmbedder",
    "download",
    "is_cached",
]

#: Pinned to a revision, not a tag. The weights are executable content and a
#: moving tag is a supply chain hole.
MODEL_REPO = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
#: The official repository ships its own ONNX export, so no third-party port is
#: in the trust path. Despite the name this runs correctly on arm64; ONNX
#: Runtime falls back when the AVX512-VNNI kernels are unavailable.
MODEL_FILE = "onnx/model_qint8_avx512_vnni.onnx"
TOKENIZER_FILE = "tokenizer.json"

DIMENSIONS = 384
MAX_TOKENS = 512

#: E5 is asymmetric and these prefixes are load-bearing, not decorative.
#: Dropping them costs real recall, which is why the Embedder protocol splits
#: embed_query from embed_documents at all.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_MISSING = (
    "voicemem: the local embedding model is not downloaded. Run "
    "'voicemem-db fetch-model', or 'python your_agent.py download-files' which "
    "fetches it alongside the Silero and turn detector models."
)


def download(*, allow_network: bool = True) -> tuple[str, str]:
    """Resolve the model and tokenizer, fetching them if allowed.

    Returns ``(model_path, tokenizer_path)``. Blocking: call it from a thread or
    at startup, never inside a turn.
    """
    from huggingface_hub import hf_hub_download

    try:
        model = hf_hub_download(
            MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION, local_files_only=not allow_network
        )
        tokenizer = hf_hub_download(
            MODEL_REPO, TOKENIZER_FILE, revision=MODEL_REVISION, local_files_only=not allow_network
        )
    except Exception as exc:
        raise RuntimeError(_MISSING) from exc
    return model, tokenizer


def is_cached() -> bool:
    """Whether the weights are already local, so a caller can warn before a
    118MB download happens somewhere it should not."""
    try:
        download(allow_network=False)
    except Exception:
        return False
    return True


def _build(model_path: str, tokenizer_path: str, threads: int) -> tuple[Any, Any]:
    """Load the session and tokenizer. Blocking, and slow enough to matter.

    Imported here rather than at module scope so that importing the package
    costs nothing for anyone on the OpenAI backend.
    """
    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    # One thread per session. The agent's event loop and audio pipeline are on
    # this machine too, and an embedder that grabs every core to save two
    # milliseconds is a bad trade inside a voice turn.
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    tokenizer.enable_padding()
    return session, tokenizer


class LocalEmbedder:
    """Satisfies :class:`~livekit.plugins.voicemem.protocols.Embedder`.

    Construct with :meth:`load`, which does the blocking work off the event
    loop. Building one of these inside a turn would stall the audio pipeline for
    the whole model load.
    """

    __slots__ = ("_batch_size", "_owns_model", "_recorder", "_session", "_tokenizer")

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        *,
        batch_size: int = 32,
        recorder: Recorder | None = None,
        owns_model: bool = True,
    ) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._recorder = recorder
        #: Views made by :meth:`with_recorder` share this session and must not
        #: drop it: closing a per-session view would otherwise appear to release
        #: a model that every other session is still using.
        self._owns_model = owns_model

    @classmethod
    async def load(
        cls,
        *,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        threads: int = 1,
        batch_size: int = 32,
        recorder: Recorder | None = None,
        allow_network: bool = True,
    ) -> LocalEmbedder:
        if model_path is None or tokenizer_path is None:
            model_path, tokenizer_path = await asyncio.to_thread(
                download, allow_network=allow_network
            )
        session, tokenizer = await asyncio.to_thread(_build, model_path, tokenizer_path, threads)
        logger.debug("voicemem: local embedder ready (%s)", MODEL_REPO)
        return cls(session, tokenizer, batch_size=batch_size, recorder=recorder)

    def with_recorder(self, recorder: Recorder | None) -> LocalEmbedder:
        """A view bound to a session's recorder, sharing the loaded model.

        The model is the expensive part and is deliberately shared. Rebuilding
        it per session would load 118MB of weights per caller.
        """
        return type(self)(
            self._session,
            self._tokenizer,
            batch_size=self._batch_size,
            recorder=recorder,
            owns_model=False,
        )

    @property
    def model_name(self) -> str:
        return MODEL_REPO

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Tokenize, run the model, pool. Blocking and CPU bound."""
        import numpy as np

        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        wanted = {i.name for i in self._session.get_inputs()}
        if "token_type_ids" in wanted:
            # XLM-R does not use segment ids, but the export declares the input.
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, {k: v for k, v in feed.items() if k in wanted})[0]

        # Mean pooling over real tokens. Not CLS: E5 is trained with average
        # pooling, and using CLS produces vectors that embed without error and
        # retrieve badly, which is the worst kind of wrong.
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        # L2 normalised because the HNSW index is vector_cosine_ops.
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return pooled.tolist()  # type: ignore[no-any-return]

    async def _run(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        # Blank input must still produce a vector. Callers zip the result
        # against the input with strict=True, so a dropped element is a crash.
        prepared = [f"{prefix}{t if t.strip() else ' '}" for t in texts]

        out: list[list[float]] = []
        for start in range(0, len(prepared), self._batch_size):
            batch = prepared[start : start + self._batch_size]
            out.extend(await asyncio.to_thread(self._encode, batch))

        if self._recorder is not None:
            self._recorder.count("embed_calls")
            self._recorder.count("embed_texts", len(texts))
        return out

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._run(list(texts), PASSAGE_PREFIX)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._run([text], QUERY_PREFIX))[0]

    async def aclose(self) -> None:
        """Drop this instance's hold on the model.

        ONNX Runtime exposes no explicit close, so releasing means dropping the
        last reference. A view from :meth:`with_recorder` therefore does nothing
        here: it shares one session with every other session in the worker, and
        a per-session close that tore that down would take memory out from under
        callers still using it.
        """
        if not self._owns_model:
            return
        self._session = None
        self._tokenizer = None
