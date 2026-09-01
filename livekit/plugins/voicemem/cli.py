# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""``voicemem-db``: schema management for the plugin's tables.

Separate from the runtime on purpose. The role your agent connects with should
not hold DDL rights, and schema changes should be something an operator does
deliberately rather than something twenty workers race each other to do at boot.

    voicemem-db status       what is applied, and whether isolation is real
    voicemem-db upgrade      apply outstanding migrations
    voicemem-db fetch-model  download the local embedding model
    voicemem-db sql       print the SQL instead of running it
    voicemem-db drop      remove the schema entirely
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

from psycopg import AsyncConnection, sql

from .adapters import local_embed
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
        await _assert_width_matches(conn, args.schema, args.embed_dim)
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
        print(render(m, embed_dim=args.embed_dim, schema=args.schema))
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


#: Widths we know, so a model named on the command line implies its dimension
#: rather than needing both flags kept in sync by hand.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def _resolve_embedding(args: argparse.Namespace) -> None:
    """Fill in model and dimension together, or refuse.

    They describe one decision, so allowing half of it to be specified produces
    an incoherent stamp that no embedder can satisfy and that only surfaces much
    later as a failed insert.
    """
    known = dict(_KNOWN_DIMS)
    known[local_embed.MODEL_REPO] = local_embed.DIMENSIONS

    if args.embed_model is None and args.embed_dim is None:
        # The runtime's own default, so `upgrade` then `run` agree.
        args.embed_model = local_embed.MODEL_REPO
        args.embed_dim = local_embed.DIMENSIONS
        return
    if args.embed_model is None:
        raise SystemExit(
            "--embed-dim was given without --embed-model. Say which model this "
            "width belongs to, or pass neither to use "
            f"{local_embed.MODEL_REPO} at {local_embed.DIMENSIONS}."
        )
    if args.embed_dim is None:
        if args.embed_model not in known:
            raise SystemExit(
                f"unknown embedding model {args.embed_model!r}, so its width cannot "
                "be inferred. Pass --embed-dim as well."
            )
        args.embed_dim = known[args.embed_model]
        return
    if args.embed_model == local_embed.MODEL_REPO and args.embed_dim != local_embed.DIMENSIONS:
        # A local model cannot be truncated the way OpenAI's can.
        raise SystemExit(
            f"{local_embed.MODEL_REPO} produces {local_embed.DIMENSIONS} dimensions, "
            f"not {args.embed_dim}."
        )


async def _assert_width_matches(conn, schema: str, embed_dim: int) -> None:
    """Refuse to stamp a width the columns contradict.

    Migrations are skipped once applied, so `upgrade` on an existing schema does
    no DDL. Without this it would happily record a dimension the table cannot
    hold, and every later insert would fail inside a handler that swallows it.
    """
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
        return
    match = re.search(r"\((\d+)\)", str(row["declared"]))
    if match and int(match.group(1)) != embed_dim:
        raise SystemExit(
            f"schema {schema!r} stores vector({match.group(1)}) but you asked to stamp "
            f"{embed_dim}. Existing vectors are not comparable across widths: use a "
            f"different schema, or drop this one and re-ingest."
        )


def _fetch_model(args: argparse.Namespace) -> int:
    """Download the local embedding model.

    Also reachable as ``python your_agent.py download-files``, which fetches it
    alongside the Silero and turn detector weights. This exists so the model can
    be pre-fetched in a Dockerfile without importing an agent.
    """
    from .adapters import local_embed

    model, tokenizer = local_embed.download()
    print(f"model:     {model}")
    print(f"tokenizer: {tokenizer}")
    print(f"{local_embed.MODEL_REPO} at revision {local_embed.MODEL_REVISION[:12]}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voicemem-db", description=__doc__.split("\n")[0])
    p.add_argument("--dsn", help="PostgreSQL DSN. Defaults to $VOICEMEM_PG_DSN.")
    p.add_argument("--schema", default=_DEFAULT_SCHEMA)
    # Deliberately no defaults here; see _resolve_embedding. Defaulting these
    # independently means `--embed-dim 1536` alone stamps OpenAI's width under
    # the local model's name, which is a schema no embedder can ever satisfy.
    p.add_argument("--embed-dim", type=int, default=None, dest="embed_dim")
    p.add_argument("--embed-model", default=None, dest="embed_model")

    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("status", help="show applied migrations and isolation status")
    subs.add_parser("upgrade", help="apply outstanding migrations")
    subs.add_parser("sql", help="print the SQL without running it")
    d = subs.add_parser("drop", help="drop the schema and everything in it")
    d.add_argument("--yes", action="store_true", help="confirm the deletion")
    subs.add_parser("fetch-model", help="download the local embedding model")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in ("upgrade", "sql", "status"):
            _resolve_embedding(args)
        if args.command == "fetch-model":
            return _fetch_model(args)
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
