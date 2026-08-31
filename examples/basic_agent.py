# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""A voice agent that remembers, across sessions.

    export OPENAI_API_KEY=sk-...
    export VOICEMEM_PG_DSN=postgresql://voicemem_app:...@localhost:5432/app
    voicemem-db --dsn "$VOICEMEM_ADMIN_DSN" upgrade
    python examples/basic_agent.py dev

Say something about yourself, hang up, reconnect, and ask about it.
"""

from __future__ import annotations

import logging
import os

from livekit.agents import Agent, AgentSession, JobContext

from livekit import agents
from livekit.plugins import deepgram, openai, silero, voicemem

logger = logging.getLogger("basic-agent")


class Assistant(Agent):
    def __init__(self, hooks: voicemem.MemoryHooks) -> None:
        super().__init__(
            instructions=(
                "You are a warm, concise voice assistant. When you are given "
                "information you remember about the user, use it naturally. Never "
                "read it out as a list and never mention that you have notes."
            )
        )
        self._hooks = hooks

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        # One line. Everything else in this file is ordinary LiveKit setup.
        await self._hooks.on_user_turn_completed(turn_ctx, new_message)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    runtime = await voicemem.build(
        voicemem.Config(
            pg_dsn=os.environ["VOICEMEM_PG_DSN"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            tenant_id=os.environ.get("VOICEMEM_TENANT_ID", "default"),
        )
    )

    # Identity is yours to decide. Keying on the participant means the same
    # caller is recognised on a later call; keying on the room would give
    # everyone a fresh memory each time.
    participant = await ctx.wait_for_participant()
    memory = runtime.session(user_id=participant.identity)
    hooks = voicemem.MemoryHooks(memory)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
        # Injecting memory invalidates preemptive generation on every turn, so
        # leaving it on means paying for a speculative call that is always
        # thrown away. This plugin overlaps retrieval with speech instead.
        turn_handling={"preemptive_generation": {"enabled": False}},
    )

    # Once, here. Not in Agent.on_enter, which re-runs on every agent handoff.
    hooks.attach(session)

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        # Ingest after the agent has replied, so the five LLM calls it costs
        # never sit between the user finishing and the agent speaking.
        if ev.item.role != "assistant":
            return
        user_text = next(
            (i.text_content for i in reversed(session.history.items) if i.role == "user"), ""
        )
        if user_text:
            hooks.remember_turn(user_text, ev.item.text_content or "", session_id=ctx.room.name)

    # Async and actually awaited by the worker, unlike the session close event.
    ctx.add_shutdown_callback(hooks.aclose)
    ctx.add_shutdown_callback(runtime.aclose)

    await session.start(agent=Assistant(hooks), room=ctx.room)
    await session.generate_reply(instructions="Greet the user briefly and warmly.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
