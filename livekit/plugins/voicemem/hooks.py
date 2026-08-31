# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""The LiveKit integration, and the only file that imports from ``livekit.agents``.

Everything below this module is a plain async memory library. Keeping the
coupling in one place is what makes the rest testable without a room, and what
would make a Pipecat adapter a second file rather than a rewrite.

Every claim in the comments below was verified by reading installed
livekit-agents source (1.7.1), not documentation. The comments carry the
specific file and line each one came from, so a version bump can be re-checked
against them directly. ``tests/livekit/`` asserts the behavioural half, which is
what actually fails a build when a minor release changes semantics.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from livekit.agents import Agent, AgentSession, StopResponse
from livekit.agents.llm import ChatContext, ChatMessage

from .log import logger
from .types import TurnRecord

if TYPE_CHECKING:
    from .memory import VoiceMemory

__all__ = ["MemoryHooks", "inject_memory"]


async def inject_memory(
    memory: VoiceMemory, turn_ctx: ChatContext, query: str, *, budget_s: float
) -> bool:
    """Retrieve and inject, within a budget. Returns whether anything was added.

    The memory block goes in as an ASSISTANT message, which is the convention
    LiveKit's own RAG documentation uses.

    ``turn_ctx`` is a copy of the agent's chat context, so what is added here
    reaches this turn's LLM request and is then discarded. That is intentional:
    persisting it would grow the context by a memory block every single turn.
    """
    try:
        result = await asyncio.wait_for(memory.recall(query), timeout=budget_s)
    except TimeoutError:
        # Skipping is invisible to the user; an audible pause is not. The
        # counter is how this stays visible to whoever is tuning the budget.
        memory.recorder.record_budget_exhausted()
        logger.debug("voicemem: recall exceeded %.2fs budget, injecting nothing", budget_s)
        return False

    if not result.block:
        return False

    turn_ctx.add_message(
        role="assistant",
        content=(
            "Relevant information from previous conversations with this user:\n"
            f"{result.block}"
        ),
    )
    return True


class MemoryHooks:
    """Attaches memory to an ``AgentSession``.

    Usage::

        memory = runtime.session(user_id="alice")
        hooks = MemoryHooks(memory)
        hooks.attach(session)          # once, in the entrypoint

    and from the agent::

        async def on_user_turn_completed(self, turn_ctx, new_message):
            await hooks.on_user_turn_completed(turn_ctx, new_message)
    """

    def __init__(self, memory: VoiceMemory) -> None:
        self._memory = memory
        self._cfg = memory.config
        self._session: AgentSession | None = None
        self._attached = False
        self._preemptive_warned = False

        # Prefetch state
        self._prefetch: asyncio.Task | None = None
        self._prefetch_text = ""
        self._prefetch_at = 0.0
        self._last_kick = 0.0
        self._interims_seen = 0
        self._turns = 0
        self._no_interim_warned = False

        # Write path
        self._writes: set[asyncio.Task] = set()

    # -- attachment ---------------------------------------------------------

    def attach(self, session: AgentSession) -> None:
        """Register listeners. Call once, from the entrypoint.

        Not from ``Agent.on_enter``: that runs again for every agent in a
        handoff chain, which would stack duplicate handlers on one session.
        """
        if self._attached:
            logger.debug("voicemem: attach() called twice, ignoring")
            return
        self._attached = True
        self._session = session
        self._warn_about_preemptive(session)

        # Must be a plain def. rtc.EventEmitter.on() raises ValueError when
        # handed a coroutine function, and tells you to spawn a task instead.
        @session.on("user_input_transcribed")
        def _on_transcribed(ev) -> None:
            try:
                if ev.is_final:
                    return
                self._maybe_prefetch(ev.transcript or "")
            except Exception:
                # A throwing handler is swallowed by the emitter, so anything
                # not caught here disappears silently.
                logger.debug("voicemem: interim handler failed", exc_info=True)

    def _warn_about_preemptive(self, session: AgentSession) -> None:
        """Say, once, that injection cancels preemptive generation.

        This is on by default in 1.6 and 1.7. Injecting into the turn context
        changes it, so the framework's ``is_equivalent`` check fails and it
        cancels the warm generation it had already started. The cost is the
        point: you pay for a full discarded LLM generation on every turn.
        """
        if self._preemptive_warned:
            return
        self._preemptive_warned = True
        try:
            enabled = bool(session.preemptive_generation.get("enabled"))
        except Exception:
            return
        if not enabled:
            return

        message = (
            "voicemem: preemptive_generation is enabled (it is the default). Injecting "
            "memory changes the turn's chat context, so LiveKit cancels the speculative "
            "generation it already started, on every turn. You pay for a discarded LLM "
            "call per turn and gain nothing from it. Disable it with "
            'AgentSession(turn_handling={"preemptive_generation": {"enabled": False}}). '
            "This plugin overlaps retrieval with speech using interim transcripts instead."
        )
        if self._cfg.strict:
            raise RuntimeError(message)
        logger.warning(message)

    # -- read path ----------------------------------------------------------

    def _maybe_prefetch(self, text: str) -> None:
        """Start speculative retrieval while the user is still speaking."""
        cfg = self._cfg.prefetch
        if not cfg.enabled:
            return

        text = text.strip()
        if not text:
            # Realtime models emit a synthetic empty interim at end of speech.
            return
        self._interims_seen += 1

        # CJK carries more meaning per character, so it needs fewer of them.
        has_cjk = any("一" <= c <= "鿿" for c in text)
        if len(text) < (cfg.min_chars_cjk if has_cjk else cfg.min_chars_latin):
            return
        if len(text) - len(self._prefetch_text) < cfg.min_growth_chars and self._prefetch_text:
            return

        now = time.monotonic()
        if now - self._last_kick < cfg.debounce_s:
            return
        self._last_kick = now

        if self._prefetch is not None and not self._prefetch.done():
            self._prefetch.cancel()

        self._prefetch_text = text
        self._prefetch_at = now
        self._prefetch = asyncio.create_task(self._memory.recall(text))
        self._memory.recorder.count("prefetch_started")

    def _take_prefetch(self, final_text: str):
        """The prefetched result, if it is still applicable to this utterance."""
        task, self._prefetch = self._prefetch, None
        started_on, self._prefetch_text = self._prefetch_text, ""
        if task is None or not started_on:
            return None

        cfg = self._cfg.prefetch
        if time.monotonic() - self._prefetch_at > cfg.ttl_s:
            task.cancel()
            self._memory.recorder.count("prefetch_stale")
            return None

        # Reusable when the final transcript is what we searched on plus a
        # little more. Compared case and punctuation insensitively because the
        # final adds capitalisation and a full stop the interim lacked.
        def norm(s: str) -> str:
            return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()

        a, b = norm(started_on), norm(final_text)
        if not b.startswith(a[: max(len(a) - 4, 1)]) or len(b) - len(a) > cfg.prefix_growth_cap:
            task.cancel()
            self._memory.recorder.count("prefetch_wasted")
            return None
        if not task.done():
            task.cancel()
            self._memory.recorder.count("prefetch_late")
            return None
        return task

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Inject memory before the LLM sees the turn.

        Every failure below is contained. An exception escaping this method is
        caught by the framework, logged, and the user's turn is dropped
        entirely: no reply at all. A memory bug must never do that.
        """
        query = (new_message.text_content or "").strip()
        if not query:
            # Documented empty-turn guard, raised before the try so it is not
            # swallowed by it.
            raise StopResponse

        self._turns += 1
        try:
            injected = False
            task = self._take_prefetch(query)
            if task is not None:
                with contextlib.suppress(Exception):
                    result = task.result()
                    if result.block:
                        turn_ctx.add_message(
                            role="assistant",
                            content=(
                                "Relevant information from previous conversations "
                                f"with this user:\n{result.block}"
                            ),
                        )
                        self._memory.recorder.count("prefetch_hit")
                        injected = True

            if not injected:
                await inject_memory(
                    self._memory, turn_ctx, query, budget_s=self._cfg.recall_budget_s
                )

            self._warn_if_no_interims()
        except Exception:
            logger.exception("voicemem: recall failed; continuing without memory")

    def _warn_if_no_interims(self) -> None:
        """Say once, after a few turns, if prefetch never had anything to work with."""
        if self._no_interim_warned or self._turns < 3 or self._interims_seen:
            return
        self._no_interim_warned = True
        logger.info(
            "voicemem: no interim transcripts seen after %d turns, so retrieval cannot "
            "overlap with speech and runs inside the turn instead. This happens with a "
            "non-streaming STT wrapped in StreamAdapter, or with a realtime model that "
            "does its own transcription.",
            self._turns,
        )

    # -- write path ---------------------------------------------------------

    def remember_turn(
        self, user_text: str, agent_reply: str, *, session_id: str = ""
    ) -> None:
        """Queue a completed exchange for background ingestion.

        Returns immediately. Ingestion is up to five LLM calls and must never
        sit between the user finishing and the agent speaking.
        """
        if not self._cfg.writer.enabled or not user_text.strip():
            return
        if len(self._writes) >= self._cfg.writer.queue_size:
            # Bounded on purpose: an unbounded queue turns a slow LLM into
            # unbounded memory growth and hides the backpressure.
            self._memory.recorder.count("write_dropped")
            logger.warning("voicemem: ingest queue full, dropping a turn")
            return

        record = TurnRecord(
            tenant_id=self._memory.scope.tenant_id,
            user_id=self._memory.scope.user_id,
            user_text=user_text,
            agent_reply=agent_reply,
            prior_agent_reply=self._memory.last_reply,
            session_id=session_id,
        )
        self._memory.note_reply(agent_reply)

        task = asyncio.create_task(self._ingest(record))
        # A task with no strong reference can be garbage collected mid-flight.
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _ingest(self, record: TurnRecord) -> None:
        try:
            await self._memory.remember(record)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Nothing awaits this, so an uncaught exception here is a turn that
            # vanishes with no trace anywhere.
            logger.exception("voicemem: ingest failed for one turn")

    async def aclose(self, reason: str = "") -> None:
        """Finish outstanding writes. Register with ``ctx.add_shutdown_callback``.

        That callback is async and actually awaited by the worker, unlike the
        session ``close`` event, which is synchronous and may not complete
        before the process exits.
        """
        if self._prefetch is not None and not self._prefetch.done():
            self._prefetch.cancel()
        if not self._writes:
            return
        logger.info("voicemem: draining %d pending write(s)", len(self._writes))
        _done, pending = await asyncio.wait(
            set(self._writes), timeout=self._cfg.writer.drain_timeout_s
        )
        for task in pending:
            task.cancel()
        if pending:
            logger.warning(
                "voicemem: %d write(s) did not finish within %.1fs and were dropped",
                len(pending), self._cfg.writer.drain_timeout_s,
            )


class MemoryAgent(Agent):
    """An ``Agent`` with memory already wired in.

    Convenience for the common case. Anything more involved should hold a
    :class:`MemoryHooks` and call it from its own ``on_user_turn_completed``.
    """

    def __init__(self, *args, memory: VoiceMemory, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hooks = MemoryHooks(memory)

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        await self.hooks.on_user_turn_completed(turn_ctx, new_message)
