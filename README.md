# livekit-plugins-voicemem

Long-term memory for [LiveKit Agents](https://docs.livekit.io/agents/) voice agents, backed by
PostgreSQL and pgvector. Your agent remembers what a caller told it last week.

Three runtime dependencies: `livekit-agents`, `openai`, `psycopg`. No torch, no local models.

> **Status: 0.1.0, first release.** The schema may change between minor versions while 0.x.

## Install

```bash
pip install livekit-plugins-voicemem
```

Requires PostgreSQL 14 or later with [pgvector](https://github.com/pgvector/pgvector) 0.8+,
an OpenAI key, and LiveKit Agents 1.6 or 1.7.

## Set up the schema

The plugin owns its tables in their own `voicemem` schema, so it never collides with yours.

```bash
export VOICEMEM_ADMIN_DSN=postgresql://admin@host/db   # a role with DDL rights
voicemem-db --dsn "$VOICEMEM_ADMIN_DSN" upgrade
voicemem-db --dsn "$VOICEMEM_ADMIN_DSN" status
```

Migrations do not run at startup, on purpose. Twenty workers booting at once and racing DDL is a
real failure, not a theoretical one: `CREATE TABLE IF NOT EXISTS` is not race-safe in PostgreSQL.
Use `voicemem-db sql` to print the DDL if your shop reviews schema changes.

## Use it

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import voicemem

class Assistant(Agent):
    def __init__(self, hooks):
        super().__init__(instructions="You are a helpful assistant.")
        self._hooks = hooks

    async def on_user_turn_completed(self, turn_ctx, new_message):
        await self._hooks.on_user_turn_completed(turn_ctx, new_message)   # the one line

async def entrypoint(ctx):
    await ctx.connect()
    runtime = await voicemem.build(voicemem.Config(
        pg_dsn=os.environ["VOICEMEM_PG_DSN"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
    ))
    participant = await ctx.wait_for_participant()
    hooks = voicemem.MemoryHooks(runtime.session(user_id=participant.identity))

    session = AgentSession(
        stt=..., llm=..., tts=..., vad=...,
        turn_handling={"preemptive_generation": {"enabled": False}},   # see Limitations
    )
    hooks.attach(session)                     # once, here, not in Agent.on_enter

    @session.on("conversation_item_added")
    def _on_item(ev):
        if ev.item.role == "assistant":
            user_text = next((i.text_content for i in reversed(session.history.items)
                              if i.role == "user"), "")
            hooks.remember_turn(user_text, ev.item.text_content or "")

    ctx.add_shutdown_callback(hooks.aclose)
    ctx.add_shutdown_callback(runtime.aclose)
    await session.start(agent=Assistant(hooks), room=ctx.room)
```

See `examples/basic_agent.py` for the whole file.

## How it works

Two stores, following [VoiceMem](https://github.com/xzf-thu/VoiceMem)'s design.

**The left brain stores facts.** It extracts atomic statements from each turn, files them under one
of seven life domains, and retrieves them by vector search. "User is allergic to shellfish."

**The right brain stores what the person is like.** Preferences, habits, and emotional patterns:
"gets anxious before design reviews". These are injected as internal notes the agent must never
repeat aloud, and they are what stops the agent sounding like it is reading your file back to you.

**Reading** runs inside the voice turn: embed the query once, classify it into slots, narrow to
those candidates, rank, and fetch relevant traits concurrently. No LLM call.

**Writing** runs after the turn, never between the user finishing and the agent speaking. One
extraction call, plus a conflict-resolution call when there is anything to conflict with.

## Measured latency and cost

Real numbers from `scripts/bench_latency.py`, not from the VoiceMem paper. 8 stored turns,
12 queries, `text-embedding-3-small` and `gpt-4o-mini`.

**Read path**, the part inside the voice turn:

| stage | p50 | p95 | share |
|---|---|---|---|
| embed query (OpenAI round trip) | 180.5 ms | 812.5 ms | 93% |
| classify into slots | 1.8 ms | 1.9 ms | 1% |
| rank (pgvector + rerank) | 8.9 ms | 11.9 ms | 5% |
| right brain (traits) | 9.1 ms | 19.5 ms | 5% |
| **total** | **192 ms** | **826 ms** | |

**The single OpenAI embedding call is 93% of it.** Everything this package does costs about 19ms.
If that ratio matters to you, a local embedder is the lever, and it is one line at the `Embedder`
seam. That is also why retrieval is prefetched on interim transcripts: the round trip overlaps with
the user still speaking, so on a prefetch hit it costs nothing inside the turn.

**Write path**, background: 3.8 s p50, 5.5 s p95, at **2 LLM calls per ingested turn**
(1 when the store is empty and conflict resolution is skipped).

*Environment: same-host Docker `pgvector/pgvector:pg17`, Apple Silicon, Python 3.12,
livekit-agents 1.7.1. A managed database in another region will be slower; the `embed` row will
not change, the others will.*

## Limitations

Stated up front rather than buried.

- **Emotion is inferred from text**, by the extraction model reading what was said. There is no
  prosody analysis, no acoustic emotion, no voiceprint and no speaker identification. Upstream
  ships those; this plugin does not, because they require torch, funasr and modelscope. Text
  catches "I'm frustrated". It cannot catch a flat "fine" said bitterly.
- **Memory injection disables LiveKit's preemptive generation.** Injecting changes the turn's chat
  context, so the framework's equivalence check fails and it cancels the speculative generation it
  had already started. Preemptive generation is **on by default**, so unless you turn it off you
  pay for a discarded LLM call on every turn. Disable it as shown in the quickstart. The plugin
  logs a warning once at startup if you have not.
- **Relative-date expansion covers English and Chinese only**, and its output format must match how
  the extractor writes dates. "next week" becomes `August 31, 2026`.
- **The plugin owns 24 tables** in your database, in the `voicemem` schema. The 0.x schema is not
  stable.
- **`on_user_turn_completed` does not fire** for realtime models using server-side turn detection,
  so memory is neither injected nor ingested on those turns.

## Multi-tenancy

Every table carries `tenant_id` and every query scopes by it. Migration `0002_rls.sql` adds
PostgreSQL row-level security on top, so a missed `WHERE` clause cannot cross tenants.

**This only works if you connect as a constrained role.** Superusers and `BYPASSRLS` roles walk
straight through row-level security no matter what the policies say. Migration 0002 creates
`voicemem_app` for this; give it a password and point `pg_dsn` at it.

```bash
voicemem-db --dsn "$DSN" status     # reports whether isolation is actually in force
```

## Swapping implementations

Four seams, defined as `typing.Protocol`, so your implementation never imports anything from here.

```python
class MyEmbedder:                       # satisfies voicemem.protocols.Embedder
    @property
    def model_name(self) -> str: ...
    @property
    def dimensions(self) -> int: ...
    async def embed_documents(self, texts): ...
    async def embed_query(self, text): ...
```

`Embedder`, `LLMClient`, `VectorStore` and `GraphStore`. `container.py` is the only file that names
a concrete class.

## Development

```bash
uv sync --group dev
docker compose up -d
export VOICEMEM_TEST_DSN=postgresql://voicemem_app:apppass@localhost:55432/voicemem_test
uv run pytest                 # integration tests skip without a DSN
uv run ruff check .
```

`pytest` on a fresh clone with no database and no API key passes: the contract suite runs the whole
retrieval chain against in-memory fakes.

## Attribution

Derived from [VoiceMem](https://github.com/xzf-thu/VoiceMem) (Apache-2.0). Prompt templates
originate from [mem0](https://github.com/mem0ai/mem0) (Apache-2.0). See `NOTICE` for attribution
and `CHANGES-FROM-UPSTREAM.md` for what was changed and why.

Apache-2.0. Not affiliated with, endorsed by, or sponsored by LiveKit, the VoiceMem authors, or
mem0.
