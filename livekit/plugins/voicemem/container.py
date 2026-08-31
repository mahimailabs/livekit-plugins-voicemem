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

from dataclasses import dataclass

from openai import AsyncOpenAI

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
    embedder: OpenAIEmbedder
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
            embedder = OpenAIEmbedder(
                self.client,
                model=self.config.embed_model,
                dimensions=self.config.embed_dim,
                recorder=recorder,
            )
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
    embedder = OpenAIEmbedder(
        client, model=config.embed_model, dimensions=config.embed_dim, recorder=recorder
    )
    llm = OpenAILLM(client, model=config.chat_model, recorder=recorder)

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
        classifier=SlotClassifier(embedder),
    )
