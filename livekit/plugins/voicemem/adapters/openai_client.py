# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""OpenAI adapters: one async client, two seams.

Upstream constructs a fresh synchronous ``openai.OpenAI`` at every call site.
Inside a LiveKit agent that blocks the event loop for the whole request, and
upstream's own comments measure one conflict-resolution call at 10.2 seconds
against a store of 95 memories. Everything here is async, and the client is
built once and injected.

Retries use ``asyncio.sleep``. Upstream's annotator sleeps with ``time.sleep``
for up to fifteen seconds inside its retry loop, which in an agent is fifteen
seconds of frozen audio.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Sequence
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from ..instrument import LLMCall, Recorder
from ..log import logger

__all__ = ["OpenAIEmbedder", "OpenAILLM", "make_client"]

_RETRYABLE = (RateLimitError, APITimeoutError)


def make_client(
    api_key: str, *, base_url: str | None = None, timeout: float = 30.0
) -> AsyncOpenAI:
    """One client for the process, injected rather than rebuilt per call."""
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


async def _with_retry(fn: Any, *, attempts: int, what: str) -> Any:
    """Exponential backoff with jitter, awaited rather than slept through."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except _RETRYABLE as exc:
            last = exc
            if attempt == attempts - 1:
                break
            delay = min(2.0**attempt, 8.0) * (0.5 + random.random() / 2)
            logger.warning(
                "voicemem: %s failed (%s), retrying in %.1fs", what, type(exc).__name__, delay
            )
            await asyncio.sleep(delay)
        except APIError:
            raise
    assert last is not None
    raise last


class OpenAIEmbedder:
    """Satisfies :class:`~livekit.plugins.voicemem.protocols.Embedder`.

    OpenAI's embedding models treat query and document text identically, so both
    methods route to the same call. The seam keeps them separate anyway because
    an asymmetric model such as E5 needs the distinction, and a future local
    adapter should not require changing every caller.
    """

    __slots__ = ("_client", "_dimensions", "_model", "_recorder")

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        recorder: Recorder | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._recorder = recorder

    def with_recorder(self, recorder: Recorder | None) -> OpenAIEmbedder:
        """A view bound to a session's recorder, sharing the one HTTP client.

        Every embedder offers this, so ``Runtime.session`` can rebind for
        instrumentation without knowing which backend it holds. It used to name
        this class directly, which meant an instrumented session silently
        reverted to OpenAI whatever the runtime had been built with.
        """
        return type(self)(
            self._client, model=self._model, dimensions=self._dimensions, recorder=recorder
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [t if t.strip() else " " for t in texts]
        if not cleaned:
            return []

        async def call() -> Any:
            return await self._client.embeddings.create(
                model=self._model, input=list(cleaned), dimensions=self._dimensions
            )

        started = time.perf_counter()
        resp = await _with_retry(call, attempts=3, what="embedding")
        if self._recorder is not None:
            self._recorder.count("embed_calls")
            self._recorder.count("embed_texts", len(cleaned))
        logger.debug(
            "voicemem embed", extra={"n": len(cleaned), "ms": round((time.perf_counter() - started) * 1000, 1)}
        )
        # The API does not promise input order, and the index field is the only
        # thing that does.
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class OpenAILLM:
    """Satisfies :class:`~livekit.plugins.voicemem.protocols.LLMClient`."""

    __slots__ = ("_client", "_model", "_recorder")

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "gpt-4o-mini",
        recorder: Recorder | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._recorder = recorder

    @property
    def model(self) -> str:
        return self._model

    async def _complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None,
        temperature: float,
        json_mode: bool,
        purpose: str,
    ) -> tuple[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        ok = True
        try:
            resp = await _with_retry(
                lambda: self._client.chat.completions.create(**kwargs),
                attempts=3,
                what=f"llm:{purpose or 'chat'}",
            )
            return resp.choices[0].message.content or "", resp
        except Exception:
            ok = False
            raise
        finally:
            if self._recorder is not None:
                usage = getattr(locals().get("resp", None), "usage", None)
                self._recorder.record_llm(
                    LLMCall(
                        purpose=purpose or "chat",
                        model=self._model,
                        duration_s=time.perf_counter() - started,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        ok=ok,
                    )
                )

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        purpose: str = "",
    ) -> dict[str, Any]:
        text, _ = await self._complete(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
            purpose=purpose,
        )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("voicemem: %s returned unparseable JSON", purpose or "llm")
            return {}
        # json_object mode guarantees an object, but a model behind a
        # compatible base_url may not honour that.
        return parsed if isinstance(parsed, dict) else {}

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        purpose: str = "",
    ) -> str:
        text, _ = await self._complete(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=False,
            purpose=purpose,
        )
        return text.strip()
