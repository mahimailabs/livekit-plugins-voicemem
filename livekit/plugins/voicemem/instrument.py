# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Measurement.

The README makes two claims that only mean something if they are measured: that
retrieval fits inside a voice turn, and that ingest costs five LLM calls. This
module is what turns both into numbers, and it is injected rather than global so
that two sessions in one worker never mix their counters.

It replaces upstream's ``utils/common/cost_log.py``, which appends to a file
path derived from the current working directory and reads its configuration from
the environment at import time.

Deliberately not a metrics library. A plugin should not choose Prometheus or
OpenTelemetry on its host's behalf. Records go to a structured logger at DEBUG,
and a caller who wants them elsewhere passes ``metrics_hook``.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .log import logger

__all__ = ["LLMCall", "Recorder", "TurnMetrics"]


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One chat completion, labelled by what it was for.

    ``purpose`` is the load-bearing field. Five calls a turn is only actionable
    if you can see which five, so the labels match the upstream call sites:
    ``extract``, ``conflict``, ``slot_tag``, ``inner_os``, ``attribution``.
    """

    purpose: str
    model: str
    duration_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = True


@dataclass(slots=True)
class TurnMetrics:
    """What one turn cost, on both paths."""

    #: Per-stage recall milliseconds. Keys mirror upstream's timing dict so the
    #: numbers stay comparable: classify, slot_filter, entity_narrow, rank, rb.
    recall_ms: dict[str, float] = field(default_factory=dict)
    recall_total_ms: float = 0.0
    #: True when the injected block came from a prefetch rather than a lookup
    #: made after the user stopped speaking.
    prefetch_hit: bool = False
    #: Recall exceeded its budget, so nothing was injected. Invisible to the
    #: user, which is exactly why it needs a counter.
    budget_exhausted: bool = False
    hits: int = 0
    rb_hits: int = 0
    llm_calls: list[LLMCall] = field(default_factory=list)

    @property
    def llm_call_count(self) -> int:
        return len(self.llm_calls)

    @property
    def llm_total_s(self) -> float:
        return sum(c.duration_s for c in self.llm_calls)


class Recorder:
    """Collects timings for one memory instance.

    One per :class:`~livekit.plugins.voicemem.memory.VoiceMemory`, which is one
    per session. Not a singleton, and it holds no module state.
    """

    __slots__ = ("_counters", "_hook", "_turn", "_turns")

    def __init__(self, metrics_hook: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        #: Called synchronously with (event, payload). Kept sync on purpose: a
        #: hook is not a place to start awaiting things mid-turn. Exceptions
        #: raised here are swallowed; a broken metrics sink must not break memory.
        self._hook = metrics_hook
        self._counters: Counter[str] = Counter()
        self._turns: int = 0
        self._turn = TurnMetrics()

    # -- turn lifecycle -----------------------------------------------------

    def begin_turn(self) -> TurnMetrics:
        self._turn = TurnMetrics()
        return self._turn

    def end_turn(self) -> TurnMetrics:
        t = self._turn
        self._turns += 1
        self._counters["turns"] += 1
        if t.prefetch_hit:
            self._counters["prefetch_hit"] += 1
        if t.budget_exhausted:
            self._counters["budget_exhausted"] += 1
        self._counters["llm_calls"] += t.llm_call_count
        logger.debug(
            "voicemem turn complete",
            extra={
                "recall_total_ms": round(t.recall_total_ms, 1),
                "recall_ms": {k: round(v, 1) for k, v in t.recall_ms.items()},
                "prefetch_hit": t.prefetch_hit,
                "budget_exhausted": t.budget_exhausted,
                "hits": t.hits,
                "rb_hits": t.rb_hits,
                "llm_calls": t.llm_call_count,
                "llm_total_s": round(t.llm_total_s, 3),
            },
        )
        self._emit("turn", {"recall_total_ms": t.recall_total_ms, "llm_calls": t.llm_call_count})
        return t

    # -- instrumentation points --------------------------------------------

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one recall stage.

        Uses ``perf_counter`` rather than wall time: these are sub-second
        intervals and a clock adjustment mid-turn would otherwise produce a
        negative duration.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._turn.recall_ms[name] = self._turn.recall_ms.get(name, 0.0) + elapsed_ms

    def record_llm(self, call: LLMCall) -> None:
        self._turn.llm_calls.append(call)
        self._counters[f"llm.{call.purpose}"] += 1
        if not call.ok:
            self._counters[f"llm.{call.purpose}.error"] += 1

    def record_recall(
        self, *, total_ms: float, hits: int, rb_hits: int, prefetched: bool
    ) -> None:
        self._turn.recall_total_ms = total_ms
        self._turn.hits = hits
        self._turn.rb_hits = rb_hits
        self._turn.prefetch_hit = prefetched

    def record_budget_exhausted(self) -> None:
        self._turn.budget_exhausted = True

    def count(self, name: str, n: int = 1) -> None:
        """Bump a named counter. Used for prefetch started, cancelled, stale and
        wasted, which together say whether the prefetch design earns its keep."""
        self._counters[name] += n

    # -- reporting ----------------------------------------------------------

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    @property
    def prefetch_hit_rate(self) -> float:
        """Fraction of turns where a prefetch was actually reusable.

        This is the number that decides whether prefetching justifies its
        complexity. A low rate with interims present means the thresholds are
        wrong; a zero rate usually means the STT is not emitting interims at all.
        """
        turns = self._counters["turns"]
        return self._counters["prefetch_hit"] / turns if turns else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "turns": self._counters["turns"],
            "prefetch_hit_rate": round(self.prefetch_hit_rate, 3),
            "llm_calls_total": self._counters["llm_calls"],
            "llm_calls_per_turn": (
                round(self._counters["llm_calls"] / self._counters["turns"], 2)
                if self._counters["turns"]
                else 0.0
            ),
            "budget_exhausted": self._counters["budget_exhausted"],
            "counters": dict(self._counters),
        }

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._hook is None:
            return
        try:
            self._hook(event, payload)
        except Exception:
            # A metrics sink is never allowed to take down the memory path.
            logger.debug("voicemem metrics_hook raised", exc_info=True)
