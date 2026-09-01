# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Migration against a database nothing has touched.

0.1.0 shipped a quickstart that could not work. ``CREATE EXTENSION vector``
appears in no migration file, and ``upgrade`` did not run it, so ``0001`` died on
a fresh database with ``type "vector" does not exist``. The whole integration
suite stayed green throughout, because every test ran against a database some
earlier run had already prepared. The bug only surfaced when a container died
and took its volume with it.

That is the shape of the gap: a cold start is the one path a warm suite can
never exercise. So this module refuses to reuse anything. It creates its own
database, migrates it with the same call the CLI makes, and drops it again.

Requires a role with CREATEDB. Skipped, with a reason, when it does not have it.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from psycopg import AsyncConnection, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from livekit.plugins.voicemem.cli import main as cli_main
from livekit.plugins.voicemem.migrations import runner

pytestmark = pytest.mark.integration

SCHEMA = "voicemem"
DIMS = 1536


def _with_dbname(dsn: str, dbname: str) -> str:
    info = conninfo_to_dict(dsn)
    info["dbname"] = dbname
    return make_conninfo(**info)


async def _connect(dsn: str) -> AsyncConnection:
    conn = await AsyncConnection.connect(dsn, autocommit=True)
    conn.row_factory = dict_row  # type: ignore[assignment]
    return conn


async def _scalar(conn: AsyncConnection, query: str, params=None):
    cur = await conn.execute(query, params)
    row = await cur.fetchone()
    return None if row is None else next(iter(row.values()))


async def _has_vector(conn: AsyncConnection) -> bool:
    return bool(await _scalar(conn, "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"))


@pytest.fixture
async def virgin_dsn(test_dsn):
    """A brand new database, dropped again afterwards.

    A new schema would not do. ``CREATE EXTENSION`` is per database, not per
    schema, so reusing the test database would silently skip the very statement
    this module exists to cover.
    """
    name = f"voicemem_cold_{uuid.uuid4().hex[:12]}"
    admin = await AsyncConnection.connect(test_dsn, autocommit=True)
    try:
        try:
            await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        except Exception as exc:  # insufficient_privilege, most likely
            pytest.skip(
                f"cannot CREATE DATABASE as this role, so a cold start cannot be tested: {exc}"
            )

        try:
            yield _with_dbname(test_dsn, name)
        finally:
            # Our own connections are closed by now; anything else still holding
            # one would block the drop.
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (name,),
            )
            await admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
    finally:
        await admin.close()


@pytest.fixture
async def cold_conn(virgin_dsn):
    """A connection to the virgin database, having proved it is actually cold.

    If pgvector is installed into ``template1`` then every new database inherits
    it and nothing here can demonstrate anything, so say so rather than pass.
    """
    conn = await _connect(virgin_dsn)
    try:
        if await _has_vector(conn):
            pytest.skip(
                "the new database already has pgvector, which means template1 carries "
                "it. This cluster cannot demonstrate a cold start."
            )
        yield conn
    finally:
        await conn.close()


async def test_the_database_really_is_cold(cold_conn):
    """Guard the guard. If the fixture handed back a prepared database, every
    assertion below it would pass without proving anything."""
    assert await _scalar(
        cold_conn,
        "SELECT count(*) FROM information_schema.schemata WHERE schema_name = %s",
        (SCHEMA,),
    ) == 0


async def test_upgrade_creates_the_extension_it_depends_on(cold_conn):
    """The 0.1.1 regression. ``vector`` has to exist before 0001 declares a
    column of that type, and no migration file creates it."""
    applied = await runner.upgrade(cold_conn, schema=SCHEMA, embed_dim=DIMS)

    assert applied == [1, 2], "a cold start must apply every migration"
    assert await _has_vector(cold_conn)


async def test_the_schema_is_usable_afterwards(cold_conn):
    """Applying cleanly is not the same as producing a working schema. The
    embedding column has to come out at the configured width, or every later
    query fails on a dimension mismatch instead."""
    await runner.upgrade(cold_conn, schema=SCHEMA, embed_dim=DIMS)

    assert await _scalar(
        cold_conn,
        "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relname = 'memories' AND a.attname = 'embedding'",
        (SCHEMA,),
    ) == f"vector({DIMS})"

    # 0002 is the half that makes tenancy real rather than advisory.
    assert await _scalar(
        cold_conn, "SELECT count(*) FROM pg_roles WHERE rolname = 'voicemem_app'"
    ) == 1
    assert await _scalar(
        cold_conn,
        "SELECT relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relname = 'memories'",
        (SCHEMA,),
    ) is True


async def test_upgrading_twice_is_a_no_op(cold_conn):
    """Workers boot in parallel and every one of them may call this."""
    assert await runner.upgrade(cold_conn, schema=SCHEMA, embed_dim=DIMS) == [1, 2]
    assert await runner.upgrade(cold_conn, schema=SCHEMA, embed_dim=DIMS) == []


async def test_the_documented_command_works_on_a_cold_database(cold_conn, virgin_dsn, capsys):
    """What the quickstart actually tells people to run.

    Driving the CLI rather than the runner is the point: 0.1.0's failure was
    reachable from ``voicemem-db upgrade`` and from nowhere else CI exercised.
    Run in a worker thread because ``main`` opens its own event loop.
    """
    assert await asyncio.to_thread(cli_main, ["--dsn", virgin_dsn, "upgrade"]) == 0
    assert "applied: 0001, 0002" in capsys.readouterr().out

    assert await asyncio.to_thread(cli_main, ["--dsn", virgin_dsn, "status"]) == 0
    assert "0001" in capsys.readouterr().out
