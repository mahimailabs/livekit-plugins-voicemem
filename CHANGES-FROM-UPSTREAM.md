# Changes from upstream

Apache-2.0 section 4(b) requires derivative works to carry prominent notices
stating that files have been changed. This file is that notice.

Upstream: [VoiceMem](https://github.com/xzf-thu/VoiceMem) at commit
`d587d424a02727d9ff3ef6f8e672a39a19ce64a7`, retrieved 2026-08-30.

Every file under `livekit/plugins/voicemem/` derived from that source carries an
`SPDX-License-Identifier: Apache-2.0` header naming its upstream path. This
document records what changed and why. `scripts/check_headers.py` enforces that
every Apache-tagged file appears in the table below and that every row points at
a file that exists.

## Changes applying to all derived code

1. **Vendored, not depended upon.** The upstream `voicemem` distribution
   hard-installs torch, torchvision, transformers, accelerate, funasr,
   modelscope, sherpa-onnx, sentence-transformers, soundfile, scipy, fastapi,
   uvicorn, sounddevice and pywebrtc-audio. None are used by the code paths this
   package needs. Required modules were copied in and adapted rather than
   imported.

2. **Storage moved from SQLite plus embedded Qdrant to PostgreSQL with
   pgvector.** Upstream keeps one SQLite file per memory space and one embedded
   Qdrant directory. Neither survives more than one process; upstream's own
   source documents the Qdrant limitation. All persistence was rewritten against
   Postgres.

3. **Multi-tenancy added.** Upstream's tenancy boundary is the SQLite file
   itself, so no table carries a tenant column. A `tenant_id` column was added to
   every table, to every index, and to every query, with row-level security
   policies as defence in depth.

4. **Synchronous I/O converted to asyncio.** Every upstream LLM and embedding
   call constructs a synchronous `openai.OpenAI` client and blocks. In a LiveKit
   agent that stalls the event loop. All network-touching code was rewritten
   async.

5. **Audio removed.** ASR, VAD, voiceprint, scene and environment detection,
   music recognition and acoustic emotion are not ported. LiveKit owns turn
   detection and speech recognition. Emotion is taken from the text judgement
   upstream already prefers over its own acoustic path.

6. **Process-global state removed.** Module-level caches, `threading.local`
   scratchpads, `os.environ` mutation, `atexit` hooks and lazy singletons behind
   shared locks were replaced with constructor injection and per-turn value
   objects. See the table in the project plan for the full list.

## Per-module changes

| This package | Upstream source | Change |
|---|---|---|
| `livekit/plugins/voicemem/adapters/pg_graph.py` | `voicemem/leftbrain/cognitive_graph/store.py, voicemem/leftbrain/slot_split/graph_entity_store.py, voicemem/rightbrain/store.py, voicemem/utils/common/session_tracker.py` | async psycopg; tenant scoped; the O(n) Python cosine scans replaced with nearest-neighbour SQL; select-then-delete replaced by DELETE RETURNING; read-modify-write metadata merge replaced... |
| `livekit/plugins/voicemem/adapters/pg_vectors.py` | `voicemem/leftbrain/mem0_backend_store.py` | mem0 and Qdrant removed entirely in favour of PostgreSQL with pgvector; async; tenant scoped; the two id spaces merged into one table; memory_id_filter pushed into SQL instead of... |
| `livekit/plugins/voicemem/extraction/extractor.py` | `voicemem/leftbrain/extract_facts_openai.py` | async; the module-level _MERGED_UTTERANCE dict and the threading.local scratchpad are gone, replaced by returning one Extraction value object; comments translated. Junk filtering, the... |
| `livekit/plugins/voicemem/extraction/prompts.py` | `voicemem/leftbrain/mem0_additive_prompt_build.py, voicemem/leftbrain/merged_extraction.py` | the merge addendum is REWRITTEN IN ENGLISH. Upstream's version is written in Chinese and instructs the model to emit Chinese trait labels and a single Chinese emotion word, which is... |
| `livekit/plugins/voicemem/leftbrain/classify.py` | `voicemem/leftbrain/cognitive_graph/local_query_classifier.py, voicemem/leftbrain/memory_repository_v2.py` | async; the module-level _SLOT_EMBED_CACHE, which upstream keyed on slot value alone so two embedders silently mixed vector spaces, is now per-instance and keyed by model name. |
| `livekit/plugins/voicemem/leftbrain/records.py` | `voicemem/leftbrain/cognitive_graph/types.py` | header only so far. Explanatory comments remain in the original Chinese; translation is pending. Field names, defaults and semantics are unchanged. Import updated for the slot_v2 ->... |
| `livekit/plugins/voicemem/leftbrain/rerank.py` | `voicemem/leftbrain/local_memory_store.py, voicemem/leftbrain/mem0_backend_store.py` | comments translated to English; the two halves gathered into one pure module with no I/O so the scoring can be tested on its own and shared by the real store and the in-memory fake.... |
| `livekit/plugins/voicemem/leftbrain/slots.py` | `voicemem/leftbrain/cognitive_graph/slot_v2.py` | verbatim; header only. This module is already fully English. Slot descriptions feed the embedder and the extraction prompt, so the text must not drift. |
| `livekit/plugins/voicemem/leftbrain/timeexpand.py` | `voicemem/leftbrain/time_expand.py` | comments and docstrings translated to English. ENGLISH SUPPORT ADDED: upstream recognised only Chinese expressions and emitted only the Chinese date format, so the function was a no-op... |
| `livekit/plugins/voicemem/rightbrain/records.py` | `voicemem/rightbrain/types.py` | header only so far. Explanatory comments remain in the original Chinese; translation is pending. Field names, defaults and semantics are unchanged. |

## Code removed as dead or broken

These upstream functions are not ported. Each was verified against the source.

| Upstream | Why |
|---|---|
| `leftbrain/memory_repository.py:301-324` `search_with_graph` | References `self._graph_store`, never assigned, and the names `GraphSearchHit` / `GraphMemoryContext`, never imported. Raises on call. |
| `leftbrain/brain.py:941-978` `_get_user_name` | Runs `SELECT text FROM memories`; that column is named `content`. The body is wrapped in a bare `except`, so it has always returned `None` and cached that result. |
| `leftbrain/memory_repository.py:119-138` JSON mirror | Serialises every memory into a single row on each write. O(n) per ingest with a lost-update race between concurrent writers. |
| `leftbrain/memory_repository_v2.py:185-449` | Not reachable from the orchestrator's search path. Includes a daemon thread that calls OpenAI. |
| `leftbrain/merged_extraction.py` | Replaced entirely. Its `threading.local()` scratchpad assumes one thread per conversation; under asyncio one thread serves every session, so one user's emotion and traits are consumed by another user's write. Superseded by a per-turn `Extraction` value object. |
| `rightbrain/graph_store.py` | Superseded upstream by the traits store. Its tables have no readers on the ported path. |
| `rightbrain/attribution_manager.py` | Batch LLM consolidation at session boundaries. Out of scope for the first release. |
