# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No runtime changes. Tests, docs and tooling only.

### Added

- A cold-start migration test. It creates its own database, migrates it and
  drops it again, because `CREATE EXTENSION` is per database and reusing the
  test database silently skips the statement that broke 0.1.0. Verified by
  reverting the fix, which fails four of its five cases with the original
  `type "vector" does not exist`.

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

[Unreleased]: https://github.com/mahimailabs/livekit-plugins-voicemem/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/mahimailabs/livekit-plugins-voicemem/releases/tag/v0.1.1
[0.1.0]: https://github.com/mahimailabs/livekit-plugins-voicemem/releases/tag/v0.1.0
