# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-01

### Added

- **Local embeddings, and they are the new default.** An ONNX Runtime adapter
  for `intfloat/multilingual-e5-small` (384 dims), pinned to a revision rather
  than a tag. Measured against OpenAI on the same machine and the same queries:
  **2.2 ms p50 against 232 ms, and 2.6 ms p95 against 3209 ms**. The tail is the
  real prize. `recall_budget_s` is 0.6 s, so an OpenAI p95 blows through it and
  injects nothing at all, silently. The model loads once at startup, in 787 ms.
- Multilingual on purpose. The slot descriptions are bilingual and the prefetch
  heuristics branch on CJK, so an English-only model would have degraded slot
  routing rather than failed visibly.
- `voicemem-db fetch-model`, and `Plugin.download_files`, so weights arrive the
  way Silero's and the turn detector's already do and `python agent.py
  download-files` needs no change.
- `Config.embed_backend`, `embed_threads`, `embed_allow_download`, and slot
  tuning overrides.
- A cold-start migration test. It creates its own database, migrates it and
  drops it again, because `CREATE EXTENSION` is per database and reusing the
  test database silently skips the statement that broke 0.1.0. Verified by
  reverting the fix, which fails four of its five cases with the original
  `type "vector" does not exist`.
- `scripts/live_call_check.py`, a pre-release check that runs two real calls
  through a LiveKit room and asserts a fact stated in the first is recalled in
  the second. Not a pytest module: it needs a LiveKit server, Deepgram and
  OpenAI. Its docstring records why `AgentSession.run()` cannot be used for
  this, since that path never invokes `on_user_turn_completed` at all.

### Changed

- **The embedding backend now follows the database.** A fresh schema gets local;
  a schema already stamped with an OpenAI model keeps using it. Flipping the
  default outright would have made every 0.1.x database fail its dimension guard
  at startup, which is an outage rather than an upgrade. Explicit
  `Config.embed_backend` beats both.
- Slot classification takes a relative window instead of an absolute cosine
  floor when the model calls for it. Measured over 22 memories, E5 scores the
  correct slot between 0.730 and 0.880 and *wrong* slots up to 0.880, so the
  classes overlap and no single floor separates them: at the inherited 0.72
  every slot passed for every memory and the filter silently stopped filtering.
  Ranking was never the problem, so a 0.01 window with three slots retains the
  right slot 22 times out of 22. OpenAI keeps the floor it was calibrated with.
- Runtime dependencies go from three to seven: `onnxruntime`, `tokenizers`,
  `huggingface-hub` and `numpy`. The claim that mattered was never the number,
  it was the absence of a two-gigabyte torch install, and that still holds.
  `livekit-plugins-silero` and `livekit-plugins-turn-detector` already depend on
  onnxruntime, so for most agents the real delta is close to zero, and
  onnxruntime is not imported at package import time.
- `voicemem-db --embed-dim` and `--embed-model` now resolve together or refuse.
  Defaulting them independently let `--embed-dim 1536` stamp OpenAI's width
  under the local model's name, a schema no embedder can satisfy.
- The quickstart documents `endpointing.min_delay`. A caller who pauses longer
  than it mid-thought is committed as two turns, and the leading fragment
  carries nothing to retrieve on.

### Fixed

- A database that was migrated but never stamped is no longer read as empty.
  `voicemem-db sql | psql` creates `vm_meta` without populating it, because
  `record_meta` is Python and never appears in that SQL. Treating that as a
  fresh install selected a 384-wide model for 1536-wide columns, and since the
  hooks contain every exception on both paths, the result was an agent whose
  memory was silently and permanently dead. The declared column width is now
  the authority, and the startup guard compares the embedder against the
  database rather than against a config value that was just derived from it.

## [0.1.1] - 2026-08-31

### Fixed

- `voicemem-db upgrade` now creates the pgvector extension before applying
  migrations. On a genuinely fresh database 0.1.0 failed with
  `type "vector" does not exist`, so the documented quickstart did not work
  unless someone had already run `CREATE EXTENSION vector` by hand. The `sql`
  subcommand printed the line but `upgrade` never executed it.

### Changed

- Rebranded to VoiceMemory. The distribution name stays
  `livekit-plugins-voicemem`, which describes what it derives from; upstream
  VoiceMem is a separate project by different authors and Apache-2.0 grants no
  trademark rights.
- README carries the logo and links to the docs site.

## [0.1.0] - 2026-08-31

First release. Long-term memory for LiveKit voice agents on PostgreSQL and pgvector,
derived from [VoiceMem](https://github.com/xzf-thu/VoiceMem) (Apache-2.0).

### Added

- **Left brain**: fact extraction, ADD/UPDATE/DELETE/NONE conflict resolution, slot
  classification over seven life domains, and vector retrieval with the upstream lexical,
  date-overlap and time-rescue scoring contract preserved.
- **Right brain**: personality and emotional traits with evidence, retrieved against the
  query rather than returned as a static profile, and rendered as internal notes the agent
  is told never to repeat aloud.
- **PostgreSQL storage**: 24 tables in a dedicated `voicemem` schema, pgvector with an HNSW
  index, and forward-only SQL migrations run through `voicemem-db`.
- **Multi-tenancy**: `tenant_id` on every table and in every query, plus optional PostgreSQL
  row-level security (`0002_rls.sql`) and a constrained `voicemem_app` role.
- **LiveKit integration**: `MemoryHooks` for `on_user_turn_completed`, speculative retrieval
  on interim transcripts, and background ingest that never blocks the reply.
- **Four protocol seams** (`Embedder`, `LLMClient`, `VectorStore`, `GraphStore`) with
  constructor injection throughout, so any of them can be replaced without subclassing.
- **English relative-date expansion**, emitting `August 31, 2026` to match how the extractor
  normalises English dates. Upstream recognised only Chinese expressions.
- `voicemem-db` CLI: `status`, `upgrade`, `sql`, `drop`.
- Latency and cost instrumentation, and `scripts/bench_latency.py` behind the published numbers.

### Notes

- Three runtime dependencies: `livekit-agents`, `openai`, `psycopg`. Upstream's package
  hard-installs torch, torchvision, transformers, funasr, modelscope, sherpa-onnx,
  sounddevice and pywebrtc-audio; none are needed here and a test asserts none are imported.
- Measured on a same-host container: 192 ms p50 for retrieval, of which 180 ms is the OpenAI
  embedding round trip and roughly 19 ms is this package. Ingest is 2 LLM calls per turn.
- Memory injection cancels LiveKit's preemptive generation, which is enabled by default.
  Disable it, or pay for a discarded LLM call every turn. The plugin warns once at startup.

### Not included

Audio-derived features from upstream: prosody and acoustic emotion, voiceprint and speaker
identification, and scene or environment detection. Emotion is inferred from text instead.

[Unreleased]: https://github.com/mahimailabs/livekit-plugins-voicemem/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mahimailabs/livekit-plugins-voicemem/releases/tag/v0.2.0
[0.1.1]: https://github.com/mahimailabs/livekit-plugins-voicemem/releases/tag/v0.1.1
[0.1.0]: https://github.com/mahimailabs/livekit-plugins-voicemem/releases/tag/v0.1.0
