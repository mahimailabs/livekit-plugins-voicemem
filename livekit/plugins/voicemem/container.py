# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Wiring. The only module that names a concrete implementation.

Every other module receives its collaborators through its constructor and knows
them only as protocols. That is what makes ``tests/fakes.py`` possible: the same
constructors, given in-memory doubles, run the whole retrieval chain with no
database and no network.

Upstream builds its components lazily behind a shared ``threading.Lock``, and
holds that lock while constructing objects that perform network I/O, including
one that probes the embedding dimension with a live API call. Under asyncio that
blocks the event loop. Everything here is built eagerly, once, at startup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from psycopg import sql
from psycopg.errors import UndefinedTable

from .adapters import local_embed
from .adapters.openai_client import OpenAIEmbedder, OpenAILLM, make_client
from .adapters.pg_graph import PgGraphStore
from .adapters.pg_vectors import PgVectorStore
from .adapters.pool import Database, make_pool
from .config import Config
from .extraction.extractor import ConflictResolver, Extractor
from .instrument import Recorder
from .leftbrain.classify import SlotClassifier
from .log import logger
from .memory import VoiceMemory
from .migrations import runner
from .types import Scope

if TYPE_CHECKING:
    from .protocols import Embedder

__all__ = ["Runtime", "build"]


@dataclass(slots=True)
class Runtime:
    """Process-scoped resources, shared by every session in this worker.

    Held separately from :class:`~livekit.plugins.voicemem.memory.VoiceMemory`
    because the pool and the HTTP client should be built once per process, while
    memory state is per session.
    """

    config: Config
    db: Database
    client: AsyncOpenAI
    embedder: Embedder
    llm: OpenAILLM
    vectors: PgVectorStore
    graph: PgGraphStore
    classifier: SlotClassifier

    def session(self, user_id: str, *, recorder: Recorder | None = None) -> VoiceMemory:
        """A memory instance for one user's session.

        When a recorder is given, the LLM and embedder are rebound to it so the
        per-turn call count and timings are attributed to this session. They
        share the one HTTP client, so this costs nothing but a wrapper.
        """
        llm = self.llm
        embedder = self.embedder
        if recorder is not None:
            llm = OpenAILLM(self.client, model=self.config.chat_model, recorder=recorder)
            # Ask the embedder to rebind itself. Naming a concrete class here
            # meant an instrumented session reverted to OpenAI no matter what
            # the runtime held, which with a different vector width is not a
            # degraded result but a wrong one.
            rebind = getattr(embedder, "with_recorder", None)
            if rebind is not None:
                embedder = rebind(recorder)
        return VoiceMemory(
            config=self.config,
            scope=Scope(tenant_id=self.config.tenant_id, user_id=user_id),
            vectors=PgVectorStore(self.db, embedder) if recorder is not None else self.vectors,
            graph=self.graph,
            embedder=embedder,
            classifier=self.classifier,
            extractor=Extractor(llm, merged=self.config.merged_extraction),
            resolver=ConflictResolver(llm),
            recorder=recorder,
        )

    async def aclose(self) -> None:
        await self.db.close()
        await self.client.close()
        # Duck-typed rather than on the protocol: adding a required method
        # would break tests/fakes.py and every embedder a user already wrote.
        closer = getattr(self.embedder, "aclose", None)
        if closer is not None:
            await closer()


#: Slot routing, per embedding model.
#:
#: OpenAI keeps the absolute floor it was calibrated with. Local switches slot
#: narrowing OFF, and that is a measured decision rather than a cautious one.
#:
#: E5's cosines against the slot descriptions are bunched into roughly two
#: hundredths of a point. Over 22 slot-bearing memories and 8 deliberately vague
#: questions, four candidate statistics were tried for telling the two apart:
#: the gap to the second slot, the total spread, the gap as a fraction of the
#: spread, and the z-score of the best. Every one of them overlaps, so there is
#: no threshold that says "this text has a slot" without also firing on text
#: that has none.
#:
#: That matters because narrowing is not advisory. ``memory.py`` falls back to
#: the whole corpus only when the slots match *nothing*; a partly-wrong guess
#: returns a non-empty id list and hard-excludes everything else. Observed on a
#: real call: "What should I keep in mind for this month?" scored all seven
#: slots within 0.02, took FINANCE/DAILY_LIFE/GOALS, and silently hid a
#: memory tagged HEALTH that the question was really about.
#:
#: So local searches the whole corpus, which is what classify()'s own docstring
#: prescribes: slower but correct, against a wrong slot that hides the right
#: memory. It also skips the classify stage entirely. Set slot_max_slots to
#: re-enable it.
_SLOT_TUNING: dict[str, dict[str, float | None]] = {
    "openai": {"min_score": 0.72, "margin": None, "max_slots": 2},
    "local": {"min_score": 0.0, "margin": 0.01, "max_slots": 0},
}


def _slot_classifier(config: Config, embedder: Embedder, backend: str) -> SlotClassifier:
    tuning = _SLOT_TUNING[backend]
    min_score = config.slot_min_score
    margin = config.slot_margin
    max_slots = config.slot_max_slots
    return SlotClassifier(
        embedder,
        min_score=float(tuning["min_score"] if min_score is None else min_score),
        margin=(tuning["margin"] if margin is None else margin),
        max_slots=int(tuning["max_slots"] if max_slots is None else max_slots),  # type: ignore[arg-type]
    )


async def _database_state(db: Database, schema: str) -> tuple[str, int] | None:
    """What this database is already committed to, or ``None`` if it is empty.

    Three states used to collapse into ``None`` here, and only one of them means
    "free to choose": the schema not existing, ``vm_meta`` holding no row, and
    the query failing outright. The middle one is not hypothetical. The
    documented ``voicemem-db sql | psql`` path creates ``vm_meta`` and never
    populates it, because ``record_meta`` is Python and never appears in that
    SQL. Treating such a database as fresh selects a 384-wide model for
    1536-wide columns, and since ``hooks.py`` contains every exception on both
    the read and write paths, the result is an agent whose memory is silently
    and permanently dead. That is the exact failure ``vm_meta`` was added to
    prevent.

    So the column itself is the authority. ``vm_meta`` is consulted for the
    model name, but the width comes from ``memories.embedding`` when it must.
    """
    async with db.admin_tx() as conn:
        await conn.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
        )
        cur = await conn.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS declared
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = 'memories' AND a.attname = 'embedding'
            """,
            (schema,),
        )
        row = await cur.fetchone()
        if row is None:
            # No memories table: genuinely nothing here yet.
            return None
        width = _parse_vector_width(str(row["declared"]))

        try:
            cur = await conn.execute("SELECT embed_model, embed_dim FROM vm_meta")
            meta = await cur.fetchone()
        except UndefinedTable:
            meta = None

    if meta is not None:
        return str(meta["embed_model"]), int(meta["embed_dim"])
    if width is None:
        return None
    # Migrated but never stamped. The width is still a hard fact, so honour it
    # and leave the model name empty rather than inventing one.
    logger.warning(
        "voicemem: schema %r has tables but no vm_meta row, which is what "
        "'voicemem-db sql | psql' produces. Falling back to the declared column "
        "width of %d. Run 'voicemem-db upgrade' to stamp it properly.",
        schema, width,
    )
    return "", width


def _parse_vector_width(declared: str) -> int | None:
    """``vector(1536)`` to ``1536``. Returns None for an undeclared width."""
    match = re.search(r"\((\d+)\)", declared)
    return int(match.group(1)) if match else None


def _choose_backend(config: Config, state: tuple[str, int] | None) -> str:
    """Explicit config wins; otherwise follow the database.

    A schema already carrying OpenAI vectors keeps using OpenAI, because the
    alternative is a startup crash on every existing deployment the day they
    upgrade. A fresh schema gets the local model, which is where "local by
    default" actually means something.
    """
    if config.embed_backend != "auto":
        return config.embed_backend
    if state is None:
        return "local"
    model, dim = state
    if model:
        return "local" if model == local_embed.MODEL_REPO else "openai"
    # Unstamped: the width is all we have, and it is enough.
    return "local" if dim == local_embed.DIMENSIONS else "openai"


async def _make_embedder(
    config: Config, client: AsyncOpenAI, recorder: Recorder | None, backend: str
) -> tuple[Embedder, Config]:
    """Build the embedder, and reconcile Config with what it actually is.

    The local model reports its own name and width, and those are what get
    stamped into ``vm_meta``, so the config is rewritten to match rather than
    letting the two drift into a mismatch nothing checks.
    """
    if backend != "local":
        return (
            OpenAIEmbedder(
                client,
                model=config.embed_model,
                dimensions=config.embed_dim,
                recorder=recorder,
            ),
            config,
        )
    if config.embed_allow_download and not local_embed.is_cached():
        logger.warning(
            "voicemem: the local embedding model is not cached, so startup will "
            "download about 118MB now. Pre-fetch it with 'voicemem-db fetch-model' "
            "to keep this out of a worker's boot, or set embed_allow_download=False "
            "to make a missing model an error instead."
        )
    embedder = await local_embed.LocalEmbedder.load(
        threads=config.embed_threads,
        recorder=recorder,
        allow_network=config.embed_allow_download,
    )
    return embedder, config.replace(
        embed_model=embedder.model_name, embed_dim=embedder.dimensions
    )


async def build(config: Config, *, recorder: Recorder | None = None) -> Runtime:
    """Construct everything, verify the database matches, and return the runtime."""
    pool = await make_pool(
        config.pg_dsn,
        schema=config.pg_schema,
        min_size=config.pool_min_size,
        max_size=config.pool_max_size,
    )
    db = Database(pool, schema=config.pg_schema)

    client = make_client(config.openai_api_key, base_url=config.openai_base_url)

    try:
        stamped = await _database_state(db, config.pg_schema)
        backend = _choose_backend(config, stamped)
        embedder, config = await _make_embedder(config, client, recorder, backend)
    except BaseException:
        # Without this a failure here strands an open pool and an open HTTP
        # client for the life of the process.
        await db.close()
        await client.close()
        raise
    logger.info(
        "voicemem: %s embeddings, %s at %d dims%s",
        backend,
        config.embed_model,
        config.embed_dim,
        "" if stamped is None else f" (database stamped {stamped[0]} at {stamped[1]})",
    )
    if backend == "openai" and stamped is not None and config.embed_backend == "auto":
        # Kept on OpenAI only because the database is already stamped that way.
        logger.info(
            "voicemem: this database predates local embeddings, so it is still paying "
            "for an OpenAI round trip on every turn. Local embeddings need a different "
            "vector width, so moving means a new schema and re-ingesting."
        )

    llm = OpenAILLM(client, model=config.chat_model, recorder=recorder)

    # Compare the embedder against the DATABASE, not against Config. Config was
    # just reconciled to the embedder a few lines above, so checking the two
    # against each other can never fail and proves nothing.
    if stamped is not None and stamped[1] != embedder.dimensions:
        await db.close()
        await client.close()
        raise RuntimeError(
            f"embedding width mismatch: {config.pg_schema!r} stores "
            f"vector({stamped[1]}) but {embedder.model_name!r} produces "
            f"{embedder.dimensions}. Existing vectors are not comparable across "
            f"models. Point at a different schema, or re-create this one."
        )

    if config.auto_migrate:
        # Development convenience only. In production the runtime role should
        # not hold DDL rights, and concurrent workers racing DDL is a real
        # failure rather than a theoretical one.
        logger.warning(
            "voicemem: auto_migrate is on. Prefer running 'voicemem-db upgrade' "
            "as a deliberate step with a role that has DDL rights."
        )
        async with db.admin_tx() as conn:
            await runner.upgrade(conn, schema=config.pg_schema, embed_dim=config.embed_dim)
            await runner.record_meta(
                conn,
                schema=config.pg_schema,
                embed_model=config.embed_model,
                embed_dim=config.embed_dim,
            )

    # Fails loudly on a dimension mismatch rather than returning nothing forever.
    async with db.admin_tx() as conn:
        await runner.verify_meta(
            conn,
            schema=config.pg_schema,
            embed_model=config.embed_model,
            embed_dim=config.embed_dim,
        )

    if not await db.check_rls():
        logger.info(
            "voicemem: tenant isolation is not enforced by the database for this "
            "connection. Queries still scope by tenant_id, but a missed WHERE clause "
            "would not be caught. Connect as the voicemem_app role to enable it."
        )

    return Runtime(
        config=config,
        db=db,
        client=client,
        embedder=embedder,
        llm=llm,
        vectors=PgVectorStore(db, embedder),
        graph=PgGraphStore(db),
        classifier=_slot_classifier(config, embedder, backend),
    )
