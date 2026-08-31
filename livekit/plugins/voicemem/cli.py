# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""``voicemem-db``: schema management for the plugin's tables.

Separate from the runtime on purpose. The role your agent connects with should
not hold DDL rights, and schema changes should be something an operator does
deliberately rather than something twenty workers race each other to do at boot.

    voicemem-db status    what is applied, and whether isolation is real
    voicemem-db upgrade   apply outstanding migrations
    voicemem-db sql       print the SQL instead of running it
    voicemem-db drop      remove the schema entirely
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from psycopg import AsyncConnection, sql

from .migrations import runner
from .migrations.runner import discover, render

__all__ = ["main"]

_DEFAULT_SCHEMA = "voicemem"


def _dsn(args: argparse.Namespace) -> str:
    dsn = args.dsn or os.environ.get("VOICEMEM_PG_DSN") or ""
    if not dsn:
        raise SystemExit(
            "no database given. Pass --dsn or set VOICEMEM_PG_DSN.\n"
            "Use a role with DDL rights here, not the runtime role."
        )
    return dsn


async def _status(args: argparse.Namespace) -> int:
    from .adapters.pool import Database, make_pool

    pool = await make_pool(_dsn(args), schema=args.schema, min_size=1, max_size=1)
    db = Database(pool, schema=args.schema)
    try:
        async with db.admin_tx() as conn:
            done = await runner.applied_versions(conn, args.schema)
            available = discover()

            print(f"schema: {args.schema}")
            print(f"migrations: {len(done)} of {len(available)} applied")
            for m in available:
                mark = "applied" if m.version in done else "PENDING"
                print(f"  {m.version:04d}_{m.name:<24} {mark}")

            cur = await conn.execute(
                "SELECT embed_model, embed_dim FROM vm_meta"
                if 1 in done
                else "SELECT NULL AS embed_model, NULL AS embed_dim"
            )
            row = await cur.fetchone()
            if row and row["embed_model"]:
                print(f"embeddings: {row['embed_model']} ({row['embed_dim']} dims)")

        # Reported rather than assumed. Policies can exist and do nothing.
        isolated = await db.check_rls()
        print(f"tenant isolation enforced by the database: {'yes' if isolated else 'no'}")
        if not isolated and 2 in done:
            print(
                "  migration 0002 is applied but this connection bypasses it. That is\n"
                "  expected for an admin role. Your agent should connect as voicemem_app."
            )
        return 0
    finally:
        await db.close()


async def _upgrade(args: argparse.Namespace) -> int:
    conn = await AsyncConnection.connect(_dsn(args), autocommit=True)
    try:
        from psycopg.rows import dict_row

        conn.row_factory = dict_row  # type: ignore[assignment]
        applied = await runner.upgrade(conn, schema=args.schema, embed_dim=args.embed_dim)
        async with conn.transaction():
            await runner.record_meta(
                conn,
                schema=args.schema,
                embed_model=args.embed_model,
                embed_dim=args.embed_dim,
            )
        if applied:
            print("applied: " + ", ".join(f"{v:04d}" for v in applied))
        else:
            print("already up to date")
        return 0
    finally:
        await conn.close()


def _sql(args: argparse.Namespace) -> int:
    """Print the DDL for shops that require reviewed schema changes."""
    print(f"CREATE SCHEMA IF NOT EXISTS {args.schema};")
    print(f"SET search_path TO {args.schema}, public;")
    print("CREATE EXTENSION IF NOT EXISTS vector;\n")
    for m in discover():
        print(f"-- ---- {m.version:04d}_{m.name} ----")
        print(render(m, embed_dim=args.embed_dim))
    return 0


async def _drop(args: argparse.Namespace) -> int:
    if not args.yes:
        raise SystemExit(
            f"this deletes every memory in schema {args.schema!r} and cannot be undone.\n"
            "Re-run with --yes if that is what you want."
        )
    conn = await AsyncConnection.connect(_dsn(args), autocommit=True)
    try:
        await conn.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(args.schema))
        )
        print(f"dropped schema {args.schema}")
        return 0
    finally:
        await conn.close()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voicemem-db", description=__doc__.split("\n")[0])
    p.add_argument("--dsn", help="PostgreSQL DSN. Defaults to $VOICEMEM_PG_DSN.")
    p.add_argument("--schema", default=_DEFAULT_SCHEMA)
    p.add_argument("--embed-dim", type=int, default=1536, dest="embed_dim")
    p.add_argument("--embed-model", default="text-embedding-3-small", dest="embed_model")

    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("status", help="show applied migrations and isolation status")
    subs.add_parser("upgrade", help="apply outstanding migrations")
    subs.add_parser("sql", help="print the SQL without running it")
    d = subs.add_parser("drop", help="drop the schema and everything in it")
    d.add_argument("--yes", action="store_true", help="confirm the deletion")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "sql":
            return _sql(args)
        if args.command == "status":
            return asyncio.run(_status(args))
        if args.command == "upgrade":
            return asyncio.run(_upgrade(args))
        if args.command == "drop":
            return asyncio.run(_drop(args))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"voicemem-db: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
