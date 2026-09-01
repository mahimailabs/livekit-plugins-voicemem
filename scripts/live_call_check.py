# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Two real calls against a real LiveKit server. Does the agent actually remember?

Not a pytest module, and deliberately so: this needs a LiveKit server, Deepgram,
OpenAI and a migrated PostgreSQL, which is not something CI should depend on.
Run it by hand before a release.

    python scripts/live_call_check.py

Why a real room, when ``AgentSession.run()`` exists and is so much cheaper: it
does not work for this. ``run()`` drives ``generate_reply()`` directly and never
enters the user-turn pipeline, so ``on_user_turn_completed`` is not called at
all. That was confirmed by injecting an unmissable codeword into the turn
context from the hook and finding it absent from every reply, in both text and
audio modality, with the hook's own call counter still at zero. A harness built
on ``run()`` reports that memory is broken when it is fine. So the caller here is
a second real participant publishing synthesised speech into the room, and the
agent hears it over WebRTC and transcribes it with Deepgram, exactly as it would
from a phone.

Two things are checked.

1. Memory crosses the call boundary. Call one states two facts. Call two joins a
   different room with a session object that no longer exists, asks two oblique
   questions, and the answers have to come back carrying the facts. Nothing
   connects the calls except PostgreSQL.

2. One recall per committed turn. ``on_user_turn_completed`` has exactly one call
   site in livekit-agents 1.7.1 (``agent_activity.py``), reached once per
   committed turn, so the hook count and the commit count must match. They are
   counted separately here because that invariant is what distinguishes a real
   double-injection bug from the ordinary case of one utterance being committed
   as two turns, which is what a pause longer than ``min_delay`` produces and
   which is not a defect.

Environment: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY,
DEEPGRAM_API_KEY, VOICEMEM_PG_DSN.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import numpy as np
from livekit.agents import Agent, AgentSession
from livekit.agents.utils import http_context

from livekit import api, rtc
from livekit.plugins import deepgram, openai, silero, voicemem

logging.basicConfig(level=logging.ERROR)

USER_IDENTITY = "voicemem-live-check"
TENANT = "live-check"
SAMPLE_RATE = 24000

REQUIRED = [
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
    "OPENAI_API_KEY", "DEEPGRAM_API_KEY", "VOICEMEM_PG_DSN",
]

TEACH = [
    "Hi there. Quick thing about me, I am lactose intolerant so I always take oat milk.",
    "Also I am training for the Toronto marathon in October.",
]
ASK = [
    "I am at a cafe right now. Should I order the latte?",
    "What should I keep in mind for this month?",
]
#: Neither question mentions milk or running, so a hit cannot come from the
#: wording of the question itself.
EXPECT = {
    "the dairy constraint": ("oat", "lactose", "dairy"),
    "the marathon": ("marathon", "training", "run"),
}


def token(room: str, identity: str) -> str:
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
        )
        .to_jwt()
    )


_COMMITTED = [0]
_COUNTER_INSTALLED = [False]


def reset_turn_counter() -> None:
    """Zero the committed-turn count, installing the counter on first use.

    Reaches into ``AgentActivity`` because the framework exposes no public count
    of committed turns. Version coupled on purpose: if a future release renames
    this, the check disables itself rather than reporting a false result.

    Installed exactly once. Patching per call would wrap the previous wrapper,
    so by call two every turn would be counted twice and the invariant below
    would report a double-injection that is not there.
    """
    _COMMITTED[0] = 0
    if _COUNTER_INSTALLED[0]:
        return
    _COUNTER_INSTALLED[0] = True

    try:
        from livekit.agents.voice.agent_activity import AgentActivity
    except ImportError:
        return

    original = getattr(AgentActivity, "on_end_of_turn", None)
    if original is None:
        return

    def patched(self, info):
        committed = original(self, info)
        if committed:
            _COMMITTED[0] += 1
            print(f"    [turn committed] {info.new_transcript!r}")
        return committed

    AgentActivity.on_end_of_turn = patched


class Assistant(Agent):
    def __init__(self, hooks: voicemem.MemoryHooks) -> None:
        super().__init__(
            instructions=(
                "You are a warm, concise voice assistant. Reply in one or two short "
                "sentences. When you are given information you remember about the user, "
                "use it naturally. Never read it out as a list and never mention notes."
            )
        )
        self.hooks = hooks
        self.hook_calls = 0
        self.injected = 0

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        self.hook_calls += 1
        before = len(turn_ctx.items)
        await self.hooks.on_user_turn_completed(turn_ctx, new_message)
        if len(turn_ctx.items) > before:
            self.injected += 1


async def _silence(source: rtc.AudioSource, seconds: float) -> None:
    chunk = np.zeros(SAMPLE_RATE // 10, dtype=np.int16)
    for _ in range(int(seconds * 10)):
        await source.capture_frame(
            rtc.AudioFrame(chunk.tobytes(), SAMPLE_RATE, 1, len(chunk))
        )
        await asyncio.sleep(0.1)


async def speak(room: rtc.Room, sentences: list[str], replied: asyncio.Event) -> None:
    """Publish synthesised speech, one utterance per turn, waiting for each reply."""
    source = rtc.AudioSource(SAMPLE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("caller-voice", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    tts = openai.TTS(voice="nova")
    for sentence in sentences:
        print(f"  caller  {sentence}")
        stream = tts.synthesize(sentence)
        async for ev in stream:
            await source.capture_frame(ev.frame)
        await stream.aclose()
        # capture_frame only queues. Without draining, the trailing silence is
        # appended behind audio that has not played, and the pause the turn
        # detector sees is nothing like the one intended.
        await source.wait_for_playout()
        await _silence(source, 1.8)

        try:
            await asyncio.wait_for(replied.wait(), timeout=25)
        except TimeoutError:
            print("          (no reply within 25s)")
        replied.clear()


async def call(mode: str, runtime) -> tuple[list[str], int, int]:
    """One call. Returns the agent's replies, hook calls and committed turns."""
    room_name = f"voicemem-check-{mode}-{uuid.uuid4().hex[:6]}"
    memory = runtime.session(user_id=USER_IDENTITY)

    if mode == "teach":
        # Start from nothing, so a pass cannot be a leftover from a previous run.
        await runtime.graph.delete_user(memory.scope)
        print("  (memory wiped, starting from nothing)")

    hooks = voicemem.MemoryHooks(memory)
    agent = Assistant(hooks)
    reset_turn_counter()

    agent_room, caller_room = rtc.Room(), rtc.Room()
    url = os.environ["LIVEKIT_URL"]
    await agent_room.connect(url, token(room_name, "agent"))
    await caller_room.connect(url, token(room_name, USER_IDENTITY))
    print(f"  room {room_name}")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=deepgram.TTS(model="aura-2-thalia-en"),
        vad=silero.VAD.load(),
        turn_handling={"preemptive_generation": {"enabled": False}},
    )
    hooks.attach(session)

    replies: list[str] = []
    replied = asyncio.Event()

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = getattr(ev, "item", None)
        if getattr(item, "role", None) != "assistant":
            return
        text = getattr(item, "text_content", "") or ""
        replies.append(text)
        print(f"  agent   {text}")
        user_text = next(
            (i.text_content for i in reversed(session.history.items)
             if getattr(i, "role", None) == "user"), ""
        )
        if user_text:
            hooks.remember_turn(user_text, text)
        replied.set()

    await session.start(agent=agent, room=agent_room)
    try:
        await speak(caller_room, TEACH if mode == "teach" else ASK, replied)
        await asyncio.sleep(2)
    finally:
        # aclose drains the background writes, as the shutdown callback would.
        await hooks.aclose()
        await session.aclose()
        await caller_room.disconnect()
        await agent_room.disconnect()

    print(f"  injected into {agent.injected} of {agent.hook_calls} turns")
    return replies, agent.hook_calls, _COMMITTED[0]


async def main() -> int:
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        print(f"missing environment: {', '.join(missing)}")
        return 2

    runtime = await voicemem.build(
        voicemem.Config(
            pg_dsn=os.environ["VOICEMEM_PG_DSN"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            tenant_id=TENANT,
        )
    )
    failures: list[str] = []
    try:
        print("\n=== CALL 1: the caller states two facts ===")
        _, hooks_1, turns_1 = await call("teach", runtime)

        print("\n=== CALL 2: new room, new session, oblique questions ===")
        replies, hooks_2, turns_2 = await call("ask", runtime)

        blob = " ".join(replies).lower()
        for label, words in EXPECT.items():
            if not any(w in blob for w in words):
                failures.append(f"did not recall {label}")

        # One hook call per committed turn. More turns than utterances is normal
        # (a pause past min_delay splits one utterance in two); more hook calls
        # than committed turns would be a real double-injection bug.
        for n, (calls, turns) in enumerate([(hooks_1, turns_1), (hooks_2, turns_2)], 1):
            if turns and calls != turns:
                failures.append(
                    f"call {n}: {calls} hook call(s) for {turns} committed turn(s)"
                )
    finally:
        await runtime.aclose()

    print("\n=== VERDICT ===")
    for label in EXPECT:
        ok = not any(label in f for f in failures)
        print(f"  {'PASS' if ok else 'FAIL'}  recalled {label}")
    invariant = [f for f in failures if "hook call" in f]
    print(f"  {'PASS' if not invariant else 'FAIL'}  one recall per committed turn")
    for f in invariant:
        print(f"        {f}")
    return 1 if failures else 0


async def wrapped() -> int:
    # The plugins reach for a shared aiohttp session that normally only exists
    # inside a running job. Outside the worker we open one ourselves.
    async with http_context.open():
        return await main()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(wrapped()))
