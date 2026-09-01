# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Which embedder a database gets, and why.

This is the load-bearing behaviour of the local-embeddings change. Flipping the
default naively would make every database created before it fail at startup on
the dimension guard, pointing at a re-embed command, which is an outage rather
than an upgrade. So the default follows the database: fresh schemas get local,
already-stamped ones keep what they have, and explicit config beats both.

Each test owns a throwaway schema, so they can be read in any order and none of
them depends on how the others left the database.
"""

from __future__ import annotations

import os
import uuid

import pytest
from psycopg import AsyncConnection, sql

from livekit.plugins.voicemem import Config, build
from livekit.plugins.voicemem.adapters.local_embed import DIMENSIONS, MODEL_REPO, download
from livekit.plugins.voicemem.container import _choose_backend

pytestmark = pytest.mark.integration

OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIMS = 1536


def _model_cached() -> bool:
    try:
        download(allow_network=False)
    except Exception:
        return False
    return True


needs_model = pytest.mark.skipif(
    not _model_cached(), reason="local model not cached; run 'voicemem-db fetch-model'"
)


@pytest.fixture
async def schema(test_dsn):
    """A throwaway schema, dropped afterwards.

    These tests need DDL rights, which the runtime role deliberately does not
    have: CI connects as the constrained ``voicemem_app``. Probe first and skip
    with a reason, rather than failing on a permission error that says nothing
    about the code under test.
    """
    name = f"vm_sel_{uuid.uuid4().hex[:10]}"
    conn = await AsyncConnection.connect(test_dsn, autocommit=True)
    try:
        try:
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
            await conn.execute(
                sql.SQL("CREATE TABLE {}.probe (id int)").format(sql.Identifier(name))
            )
            await conn.execute(sql.SQL("DROP TABLE {}.probe").format(sql.Identifier(name)))
        except Exception as exc:
            pytest.skip(f"this role cannot create schema objects, so backend "
                        f"selection cannot be exercised: {exc}")

        yield name
    finally:
        try:
            await conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name))
            )
        finally:
            await conn.close()


def _config(dsn: str, schema: str, **kw) -> Config:
    # A real key is not needed: nothing here makes an LLM call, and the client
    # performs no I/O when it is constructed.
    return Config(
        pg_dsn=dsn,
        openai_api_key=os.environ.get("OPENAI_API_KEY") or "sk-not-used-in-this-test",
        pg_schema=schema,
        auto_migrate=True,
        **kw,
    )


# -- the pure decision, no database needed ---------------------------------


def test_a_fresh_database_chooses_local() -> None:
    cfg = _config("postgresql://x/y", "vm_x")
    assert _choose_backend(cfg, None) == "local"


def test_an_openai_stamped_database_keeps_openai() -> None:
    """The whole point. Upgrading must not break a running deployment."""
    cfg = _config("postgresql://x/y", "vm_x")
    assert _choose_backend(cfg, (OPENAI_MODEL, OPENAI_DIMS)) == "openai"


def test_a_local_stamped_database_keeps_local() -> None:
    cfg = _config("postgresql://x/y", "vm_x")
    assert _choose_backend(cfg, (MODEL_REPO, DIMENSIONS)) == "local"


@pytest.mark.parametrize("chosen", ["local", "openai"])
def test_explicit_config_beats_the_database(chosen: str) -> None:
    cfg = _config("postgresql://x/y", "vm_x", embed_backend=chosen)
    other = (OPENAI_MODEL, OPENAI_DIMS) if chosen == "local" else (MODEL_REPO, DIMENSIONS)
    assert _choose_backend(cfg, other) == chosen
    assert _choose_backend(cfg, None) == chosen


# -- against a real database -----------------------------------------------


@needs_model
async def test_a_fresh_schema_is_built_and_stamped_local(test_dsn, schema) -> None:
    runtime = await build(_config(test_dsn, schema))
    try:
        assert runtime.embedder.model_name == MODEL_REPO
        assert runtime.embedder.dimensions == DIMENSIONS
        # Config was reconciled to what the embedder actually is, so the value
        # stamped into vm_meta and the vectors in the table agree.
        assert runtime.config.embed_dim == DIMENSIONS
        assert runtime.config.embed_model == MODEL_REPO
    finally:
        await runtime.aclose()


async def test_an_openai_database_still_starts(test_dsn, schema) -> None:
    """Migrate at 1536 as 0.1.x did, then boot with the new defaults."""
    first = await build(_config(test_dsn, schema, embed_backend="openai"))
    await first.aclose()

    second = await build(_config(test_dsn, schema))  # auto
    try:
        assert second.embedder.model_name == OPENAI_MODEL
        assert second.config.embed_dim == OPENAI_DIMS
    finally:
        await second.aclose()


@needs_model
async def test_forcing_local_onto_an_openai_database_refuses_loudly(test_dsn, schema) -> None:
    """It must not quietly write 384-wide vectors into a 1536-wide column."""
    first = await build(_config(test_dsn, schema, embed_backend="openai"))
    await first.aclose()

    with pytest.raises(RuntimeError, match="width mismatch"):
        await build(_config(test_dsn, schema, embed_backend="local"))


@needs_model
async def test_an_instrumented_session_keeps_the_local_embedder(test_dsn, schema) -> None:
    """The rebind used to name OpenAIEmbedder outright, so passing a recorder
    silently swapped the backend, and with it the vector width."""
    from livekit.plugins.voicemem import Recorder

    runtime = await build(_config(test_dsn, schema))
    try:
        memory = runtime.session("alice", recorder=Recorder())
        assert memory._embedder.model_name == MODEL_REPO
        assert memory._embedder.dimensions == DIMENSIONS
    finally:
        await runtime.aclose()


# -- the state that is neither fresh nor stamped ---------------------------


def test_an_unstamped_schema_is_read_from_its_column_width() -> None:
    """`voicemem-db sql | psql` creates vm_meta and never fills it, because
    record_meta is Python and never appears in that SQL. Reading the empty table
    as "fresh" selects a 384-wide model for 1536-wide columns, and hooks.py
    swallows every resulting error, so memory dies silently and permanently."""
    cfg = _config("postgresql://x/y", "vm_x")
    assert _choose_backend(cfg, ("", OPENAI_DIMS)) == "openai"
    assert _choose_backend(cfg, ("", DIMENSIONS)) == "local"


async def test_a_schema_provisioned_by_the_sql_command_does_not_get_local(
    test_dsn, schema
) -> None:
    """The regression, end to end. Provision exactly as `voicemem-db sql`
    does: DDL only, no vm_meta row, no bookkeeping table."""
    from livekit.plugins.voicemem.migrations.runner import discover, render

    conn = await AsyncConnection.connect(test_dsn, autocommit=True)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        await conn.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
        )
        for migration in discover():
            await conn.execute(render(migration, embed_dim=OPENAI_DIMS))  # type: ignore[arg-type]

        cur = await conn.execute(
            sql.SQL("SELECT count(*) FROM {}.vm_meta").format(sql.Identifier(schema))
        )
        row = await cur.fetchone()
        assert row is not None and row[0] == 0, "this path must leave vm_meta empty"
    finally:
        await conn.close()

    runtime = await build(
        Config(pg_dsn=test_dsn, openai_api_key="sk-not-used", pg_schema=schema)
    )
    try:
        assert runtime.embedder.dimensions == OPENAI_DIMS, (
            "a 1536-wide schema must not be handed a 384-wide embedder"
        )
    finally:
        await runtime.aclose()
