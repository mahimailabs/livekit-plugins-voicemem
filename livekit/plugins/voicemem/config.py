# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Configuration, as one frozen object.

This replaces roughly seventeen environment variables that upstream reads at
import time into module constants, plus a scattering more read per call on the
hot path. Import-time reads cannot be configured per instance, which means two
sessions in one worker process cannot differ, and a test cannot change a setting
without reimporting the module.

Nothing in this package reads ``os.environ`` except :meth:`Config.from_env`, and
that only when a caller explicitly asks for it. In particular the OpenAI key is
never written back into the environment: upstream does
``os.environ["OPENAI_API_KEY"] = api_key`` in its constructor, so building one
memory instance silently reassigns credentials for every other tenant in the
same process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["Config", "PrefetchConfig", "WriterConfig"]


@dataclass(frozen=True, slots=True)
class PrefetchConfig:
    """When to speculatively retrieve while the user is still speaking.

    The whole point is to overlap retrieval with speech so that by end of turn
    the answer is already in hand. The thresholds exist to stop us spending a
    lookup on every keystroke of a growing transcript.
    """

    enabled: bool = True
    #: Minimum characters before the first lookup fires. Latin scripts need more
    #: characters than CJK to carry the same amount of meaning.
    min_chars_latin: int = 12
    min_chars_cjk: int = 6
    #: Interim transcripts arrive roughly every 100ms. Without a debounce we
    #: would cancel and restart faster than a lookup can finish.
    debounce_s: float = 0.35
    #: Re-fire only once the transcript has grown by this much, so trailing
    #: punctuation corrections do not trigger a fresh search.
    min_growth_chars: int = 8
    #: A prefetch is reusable if the final transcript starts with the text we
    #: searched on, allowing this much extra growth.
    prefix_growth_cap: int = 24
    #: Discard a prefetch older than this. A stale result is worse than none.
    ttl_s: float = 5.0


@dataclass(frozen=True, slots=True)
class WriterConfig:
    """The background ingest path.

    Ingest costs five LLM calls and must never sit between the user finishing
    and the agent speaking.
    """

    enabled: bool = True
    #: Bounded on purpose. An unbounded queue turns a slow LLM into unbounded
    #: memory growth and hides the backpressure that tells you something is wrong.
    queue_size: int = 8
    #: How long shutdown waits for in-flight ingests before giving up.
    #:
    #: Must exceed the p95 of a full ingest or shutdown routinely discards the
    #: last turn of every call. Measured on gpt-4o-mini: 3.8s p50, 5.5s p95, so
    #: a 3s default silently dropped writes that were about to succeed. 12s
    #: leaves headroom for a slow conflict-resolution call, which grows with the
    #: size of the store.
    drain_timeout_s: float = 12.0
    #: Attempts per turn before the outbox row is left for the next worker.
    max_attempts: int = 3


@dataclass(frozen=True, slots=True)
class Config:
    """Everything this package needs to run.

    Required: a Postgres DSN and an OpenAI key. Everything else has a default
    that is safe in production.
    """

    pg_dsn: str
    openai_api_key: str

    #: Tables live in their own schema. The vendored table names include
    #: `memories`, `entities`, `history` and `kv`; a plugin must not squat those
    #: in a customer's `public` schema.
    pg_schema: str = "voicemem"

    #: Isolation boundary. Upstream had none, because its boundary was the
    #: SQLite file. Every row and every query carries this.
    tenant_id: str = "default"

    # -- models -------------------------------------------------------------
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    #: Must match the width the schema was migrated with. Recorded in `vm_meta`
    #: and asserted at startup, because a mismatched embedder otherwise fails
    #: silently: upstream's trait store skips shape-mismatched rows with a bare
    #: `continue`, so the right brain returns nothing, forever, with no error.
    embed_dim: int = 1536
    openai_base_url: str | None = None

    # -- retrieval ----------------------------------------------------------
    top_k: int = 5
    #: Right-brain notes injected per turn. Upstream measured that all five
    #: delayed the reply model's first token; three did not.
    right_brain_top_k: int = 3
    #: Hard ceiling on recall inside the turn. On expiry we inject nothing
    #: rather than delay speech, and count it.
    recall_budget_s: float = 0.6
    #: Extra time-boosted hits appended after the top_k cut, to recover a
    #: time-relevant memory that ranked poorly on cosine alone.
    rescue_k: int = 2

    # -- behaviour ----------------------------------------------------------
    enable_right_brain: bool = True
    #: Fold annotation and trait extraction into the extraction call. On, this
    #: costs 5 LLM calls per turn; off, 7. Kept switchable because it is the
    #: escape hatch if the merged prompt degrades extraction on some model.
    merged_extraction: bool = True
    #: Session-boundary cluster emergence. Off by default: up to three
    #: sequential LLM calls for a feature most deployments will not notice.
    enable_subgraph: bool = False
    #: Skip conflict resolution and always insert. Upstream measured one
    #: resolution call at 10.2s against a 95-memory store, growing with size.
    always_add: bool = False

    # -- infrastructure -----------------------------------------------------
    pool_min_size: int = 1
    pool_max_size: int = 4
    #: Migrations are a deliberate operator action by default. Twenty workers
    #: booting at once racing DDL is a real failure, not a theoretical one:
    #: `CREATE TABLE IF NOT EXISTS` is not race-safe in Postgres.
    auto_migrate: bool = False
    #: Raise instead of warning when the session has preemptive generation on.
    strict: bool = False

    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)

    def __post_init__(self) -> None:
        if not self.pg_dsn:
            raise ValueError("pg_dsn is required")
        if not self.openai_api_key:
            raise ValueError("openai_api_key is required")
        if not self.tenant_id:
            raise ValueError("tenant_id must not be empty")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.recall_budget_s <= 0:
            raise ValueError("recall_budget_s must be positive")
        if self.pool_max_size < self.pool_min_size:
            raise ValueError("pool_max_size must be >= pool_min_size")
        if not self.pg_schema.replace("_", "").isalnum():
            # This value is interpolated into DDL and into search_path, where it
            # cannot be parameterised.
            raise ValueError(f"pg_schema must be alphanumeric or underscore, got {self.pg_schema!r}")

    def replace(self, **changes: Any) -> Config:
        """A copy with fields changed. The object stays frozen."""
        return replace(self, **changes)

    @classmethod
    def from_env(cls, **overrides: Any) -> Config:
        """Build from environment variables. Opt-in, never automatic.

        Reads ``VOICEMEM_PG_DSN``, ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``,
        ``VOICEMEM_TENANT_ID``, ``VOICEMEM_PG_SCHEMA``, ``VOICEMEM_CHAT_MODEL``,
        ``VOICEMEM_EMBED_MODEL`` and ``VOICEMEM_EMBED_DIM``. Explicit keyword
        arguments win over the environment.
        """
        env: dict[str, Any] = {
            "pg_dsn": os.environ.get("VOICEMEM_PG_DSN", ""),
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "openai_base_url": os.environ.get("OPENAI_BASE_URL") or None,
            "tenant_id": os.environ.get("VOICEMEM_TENANT_ID", "default"),
            "pg_schema": os.environ.get("VOICEMEM_PG_SCHEMA", "voicemem"),
            "chat_model": os.environ.get("VOICEMEM_CHAT_MODEL", "gpt-4o-mini"),
            "embed_model": os.environ.get("VOICEMEM_EMBED_MODEL", "text-embedding-3-small"),
        }
        if raw := os.environ.get("VOICEMEM_EMBED_DIM"):
            try:
                env["embed_dim"] = int(raw)
            except ValueError as exc:
                raise ValueError(f"VOICEMEM_EMBED_DIM must be an integer, got {raw!r}") from exc
        env.update(overrides)
        return cls(**env)

    def redacted(self) -> dict[str, Any]:
        """Loggable form. Never emit the key or the DSN, which carries a password."""
        out: dict[str, Any] = {}
        for name in self.__slots__:
            value = getattr(self, name)
            if name == "openai_api_key":
                value = "***" if value else ""
            elif name == "pg_dsn":
                value = _redact_dsn(value)
            out[name] = value
        return out


def _redact_dsn(dsn: str) -> str:
    """Strip the password from a DSN so it can be logged.

    Handles the URL form. Anything else is reported as present but hidden,
    rather than guessed at and leaked.
    """
    if "://" not in dsn:
        return "<dsn>"
    scheme, _, rest = dsn.partition("://")
    if "@" not in rest:
        return f"{scheme}://{rest}"
    creds, _, host = rest.partition("@")
    user = creds.partition(":")[0]
    return f"{scheme}://{user}:***@{host}"
