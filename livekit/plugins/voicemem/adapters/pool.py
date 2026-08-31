# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""The connection pool, and the transaction that carries a tenant.

The pool is constructed in ``container.py`` and injected. Nothing here is
global: upstream cached one client per storage path in a module dict, which is
what made two concurrent users in one process a real problem rather than a
theoretical one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ..log import logger
from ..types import Scope
from .pg_types import register_vector_async

__all__ = ["Database", "make_pool"]


async def make_pool(
    dsn: str,
    *,
    schema: str,
    min_size: int = 1,
    max_size: int = 4,
) -> AsyncConnectionPool:
    """Build a pool whose connections are ready to use.

    Every connection gets ``search_path`` set once and the vector type
    registered once, rather than each query paying for it.
    """

    async def configure(conn: AsyncConnection) -> None:
        conn.row_factory = dict_row  # type: ignore[assignment]
        await conn.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
        )
        await register_vector_async(conn)
        # The pool requires configure to hand back an idle connection. Both
        # statements above open an implicit transaction, and leaving it open
        # makes the pool discard every connection it builds and then time out
        # with "connection left in status INTRANS by configure function".
        # search_path and the type registration are session state, so committing
        # here does not undo them.
        await conn.commit()

    pool = AsyncConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        configure=configure,
        open=False,
        # Fail fast at startup rather than hanging the first turn of a call.
        timeout=10.0,
    )
    await pool.open(wait=True, timeout=15.0)
    return pool


class Database:
    """Owns the pool and hands out tenant-scoped transactions.

    The only way to get a cursor is through :meth:`tx`, which sets the tenant
    for the transaction. That is what makes the row-level security policies in
    migration 0002 effective: without the setting a connection reads nothing, so
    a code path that forgets the tenant fails loudly and empty rather than
    quietly returning another customer's rows.
    """

    __slots__ = ("_pool", "_schema")

    def __init__(self, pool: AsyncConnectionPool, *, schema: str = "voicemem") -> None:
        self._pool = pool
        self._schema = schema

    @property
    def schema(self) -> str:
        return self._schema

    @asynccontextmanager
    async def tx(self, scope: Scope) -> AsyncIterator[AsyncConnection]:
        """A transaction bound to one tenant.

        ``SET LOCAL`` rather than ``SET``: it is undone at commit, so a pooled
        connection handed to a different tenant next cannot inherit this one's
        context. That is what keeps this safe behind pgbouncer in transaction
        pooling mode.
        """
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(
                "SELECT set_config('voicemem.tenant_id', %s, true)", (scope.tenant_id,)
            )
            yield conn

    @asynccontextmanager
    async def admin_tx(self) -> AsyncIterator[AsyncConnection]:
        """A transaction with no tenant set, for migrations and startup checks.

        Reads nothing from any tenant-scoped table by design.
        """
        async with self._pool.connection() as conn, conn.transaction():
            yield conn

    async def check_rls(self) -> bool:
        """Whether tenant isolation is actually in force for this connection.

        Reports rather than assumes. Policies can be present in ``pg_policies``
        and still do nothing: superusers and roles with ``BYPASSRLS`` walk
        straight through them, and so does the table owner unless the table was
        created with ``FORCE ROW LEVEL SECURITY``. This was a real finding
        during development, where a superuser connection saw every tenant's rows
        with all 22 policies in place.
        """
        async with self.admin_tx() as conn:
            cur = await conn.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            )
            row = await cur.fetchone()
            if row is None:
                return False
            if row["rolsuper"] or row["rolbypassrls"]:
                logger.warning(
                    "voicemem: connected as %s, which bypasses row-level security. "
                    "Tenant isolation is enforced only by the WHERE clauses in this "
                    "package, not by the database. Connect as the unprivileged "
                    "voicemem_app role to get the policies from migration 0002.",
                    "a superuser" if row["rolsuper"] else "a BYPASSRLS role",
                )
                return False

            cur = await conn.execute(
                """
                SELECT count(*) AS n
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = %s AND c.relkind = 'r'
                  AND c.relrowsecurity AND c.relforcerowsecurity
                """,
                (self._schema,),
            )
            row = await cur.fetchone()
            return bool(row and row["n"] > 0)

    async def close(self) -> None:
        await self._pool.close()
