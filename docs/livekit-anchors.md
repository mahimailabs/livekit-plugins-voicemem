# Verified LiveKit anchors

Facts this plugin depends on, each verified by reading installed source rather than docs.
When bumping the `livekit-agents` pin, re-run these checks and update the table.

Verified against **livekit-agents 1.7.1** on 2026-08-30 (the version `>=1.6.0,<1.8` resolves to
today). Line numbers from 1.6.10 are kept alongside because the design was originally traced
against that release; the behaviour is unchanged, only the offsets moved.

| Fact | 1.7.1 | 1.6.10 | Why we care |
|---|---|---|---|
| `on_user_turn_completed(self, turn_ctx, new_message)` | `voice/agent.py:325-333` | `:315-323` | The injection point. |
| `turn_ctx` is `self._agent.chat_ctx.copy()` | `voice/agent_activity.py:2586` | `:2482` | Mutation applies to this turn only and is discarded after. Do not call `update_chat_ctx` from the hook. |
| Hook is awaited | `voice/agent_activity.py:2589` | `:2485` | Slow work here delays the reply. Hence the recall budget. |
| `StopResponse` aborts the turn cleanly | `voice/agent_activity.py:2592` | `:2488` | The documented empty-turn guard. |
| **Any other exception is logged and the turn is dropped** | `voice/agent_activity.py:2595` | `:2490-2492` | A memory bug must never reach this. Catch `Exception` inside the hook. |
| Preemptive reuse requires `preemptive.chat_ctx.is_equivalent(temp_mutable_chat_ctx)` | `voice/agent_activity.py:2625` | `:2517-2524` | Our injection breaks equality. |
| "preemptive generation invalidated after `on_user_turn_completed`" | `voice/agent_activity.py:2644` | `:2538-2543` | The warm handle is cancelled every turn we inject. |
| `_PREEMPTIVE_GENERATION_DEFAULTS = {"enabled": True, ...}` | `voice/turn.py:223-228` | same | **On by default.** Every user pays for a discarded generation per turn unless they opt out. |
| Accessor is `session.preemptive_generation` -> `PreemptiveGenerationOptions` | `voice/agent_session.py:310-311` | n/a | Read `["enabled"]` here for the startup warning. NOT `session.options.preemptive_generation`. |
| Opt-out is `turn_handling={"preemptive_generation": {"enabled": False}}` | `voice/turn.py:260-285` | same | The exact string our warning must print. |
| `UserInputTranscribedEvent(transcript, is_final, item_id, speaker_id, language, created_at)` | `voice/events.py:323-331` | `:334-342` | The prefetch trigger. |
| `EventEmitter.on()` raises `ValueError` on a coroutine function | `livekit/rtc/event_emitter.py:160-163` | same | The interim handler must be a plain `def` that spawns a task. |
| `ctx.add_shutdown_callback` is async and awaited | `job.py:562-579` | same | The durable flush point. `session.on("close")` is sync and may not finish. |

## Checks

`tests/livekit/test_anchors.py` asserts the behavioural half of this table against whatever
version is installed, so a version bump that changes semantics fails CI rather than production.
Line numbers are documentation; the tests assert behaviour.
