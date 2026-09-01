# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Migration runner.

Numbered SQL files plus about a hundred lines, rather than alembic. Alembic is
the right tool for an application and the wrong one for a library: it pulls in
SQLAlchemy, it assumes an app-shaped ``env.py``, and it writes a single
``alembic_version`` table into a database the host application very likely also
manages with alembic. At that point running ``alembic upgrade head`` in their
repo tries to autogenerate DROP statements for our two dozen tables.

Migrations do not run at startup by default. Twenty workers booting at once and
racing DDL is a real failure rather than a theoretical one: ``CREATE TABLE IF
NOT EXISTS`` is not race-safe in PostgreSQL, and concurrent creators collide on
``pg_type_typname_nsp_index``. An advisory lock serialises them when they do run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING

from psycopg import AsyncConnection, sql

from ..log import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Migration", "applied_versions", "discover", "upgrade"]

#: One arbitrary but fixed key, so every process serialises on the same lock.
_LOCK_KEY = 0x766D_656D  # "vmem"

_FILENAME_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql_text: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql_text.encode()).hexdigest()[:16]


def discover() -> list[Migration]:
    """Load the migration files shipped inside the package.

    Read through ``importlib.resources`` rather than by path, so this works from
    a wheel, a zipimport and an editable install alike. A wheel that silently
    omitted these files would fail here rather than at the first query.
    """
    out: list[Migration] = []
    for entry in resources.files(__package__).iterdir():
        m = _FILENAME_RE.match(entry.name)
        if not m:
            continue
        out.append(
            Migration(
                version=int(m.group(1)),
                name=m.group(2),
                sql_text=entry.read_text(encoding="utf-8"),
            )
        )
    if not out:
        raise RuntimeError(
            "no migration files found in the installed package. The wheel is "
            "missing its .sql package data."
        )
    return sorted(out, key=lambda x: x.version)


#: A plain SQL identifier. Checked here as well as in Config, because this is
#: the point where the value is interpolated into DDL and cannot be a bind
#: parameter.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def render(migration: Migration, *, embed_dim: int, schema: str) -> str:
    """Substitute the values that cannot be bind parameters.

    ``embed_dim`` is part of a type declaration (``vector(1536)``), and
    ``schema`` appears in ``GRANT ... ON SCHEMA``, which takes an identifier
    rather than an expression. Neither can be parameterised, so both are
    validated here instead.

    Tables do not need the substitution: they are created unqualified under a
    ``search_path`` the runner sets. Grants do, and 0002 used to name the
    default schema literally, so any other value failed with "schema voicemem
    does not exist" and ``pg_schema`` did not work at all.
    """
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError(f"embed_dim must be a positive int, got {embed_dim!r}")
    if not _IDENTIFIER.match(schema):
        raise ValueError(f"schema must be a plain SQL identifier, got {schema!r}")
    return migration.sql_text.replace("{{embed_dim}}", str(embed_dim)).replace(
        "{{schema}}", schema
    )


async def _ensure_extension(conn: AsyncConnection) -> None:
    """Create the pgvector extension before any migration needs its types.

    Without this, 0001 fails on a genuinely fresh database with
    ``type "vector" does not exist``, because ``CREATE EXTENSION`` is not part
    of any migration file. The extension is global to the database rather than
    to our schema, which is why it lives here and not in 0001.
    """
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:
        raise RuntimeError(
            "could not create the pgvector extension. Either install it "
            "(https://github.com/pgvector/pgvector) or use an image that ships it, "
            "such as pgvector/pgvector:pg17. On managed Postgres, creating an "
            "extension usually needs the database owner. If a superuser has already "
            "run 'CREATE EXTENSION vector' for you, this step is a no-op and the "
            f"error below means something else is wrong.\n  {exc}"
        ) from exc


async def _ensure_bookkeeping(conn: AsyncConnection, schema: str) -> None:
    await _ensure_extension(conn)
    await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    await conn.execute(
        sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS voicemem_schema_migrations (
            version     INTEGER     PRIMARY KEY,
            name        TEXT        NOT NULL,
            checksum    TEXT        NOT NULL,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


async def applied_versions(conn: AsyncConnection, schema: str) -> dict[int, str]:
    """Version to checksum for what is already applied. Empty on a fresh database."""
    await conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    # to_regclass rather than a SELECT in a try block. A failed SELECT aborts
    # the whole transaction, so every later statement fails too: that is what
    # made `voicemem-db status` report every migration as PENDING and then die
    # with InFailedSqlTransaction when the role could not read the table.
    cur = await conn.execute(
        "SELECT to_regclass(%s) AS oid", (f"{schema}.voicemem_schema_migrations",)
    )
    row = await cur.fetchone()
    if row is None or row["oid"] is None:
        return {}
    cur = await conn.execute(
        "SELECT version, checksum FROM voicemem_schema_migrations ORDER BY version"
    )
    return {r["version"]: r["checksum"] for r in await cur.fetchall()}


async def upgrade(
    conn: AsyncConnection,
    *,
    schema: str,
    embed_dim: int,
    migrations: Sequence[Migration] | None = None,
) -> list[int]:
    """Apply everything outstanding. Returns the versions applied.

    Each migration runs in its own transaction, so a failure half way leaves the
    ones before it applied and recorded rather than rolling back work that
    succeeded.
    """
    migrations = list(migrations) if migrations is not None else discover()

    # Serialise concurrent upgraders. Session-level rather than transaction
    # level, because each migration commits separately below.
    await conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
    try:
        async with conn.transaction():
            await _ensure_bookkeeping(conn, schema)
        done = await applied_versions(conn, schema)

        for m in migrations:
            if m.version in done:
                if done[m.version] != m.checksum:
                    logger.warning(
                        "voicemem: migration %04d_%s has changed since it was applied "
                        "(recorded %s, now %s). Migrations are immutable once applied; "
                        "add a new one instead.",
                        m.version, m.name, done[m.version], m.checksum,
                    )
                continue

            logger.info("voicemem: applying migration %04d_%s", m.version, m.name)
            async with conn.transaction():
                await conn.execute(
                    sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
                )
                await conn.execute(  # type: ignore[arg-type]
                    render(m, embed_dim=embed_dim, schema=schema)
                )
                await conn.execute(
                    "INSERT INTO voicemem_schema_migrations (version, name, checksum) "
                    "VALUES (%s, %s, %s)",
                    (m.version, m.name, m.checksum),
                )

        applied_now = await applied_versions(conn, schema)
        return sorted(set(applied_now) - set(done))
    finally:
        await conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))


async def record_meta(
    conn: AsyncConnection, *, schema: str, embed_model: str, embed_dim: int
) -> None:
    """Stamp the embedding model into ``vm_meta`` so a later swap is detectable."""
    await conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    await conn.execute(
        """
        INSERT INTO vm_meta (id, embed_model, embed_dim) VALUES (TRUE, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (embed_model, embed_dim),
    )


async def verify_meta(
    conn: AsyncConnection, *, schema: str, embed_model: str, embed_dim: int
) -> None:
    """Fail loudly when the configured embedder does not match the schema.

    Upstream had no equivalent, and its failure mode was silent: the trait store
    skipped shape-mismatched vectors with a bare ``continue``, so a changed
    embedding model produced a right brain that returned nothing at all, with no
    error, indefinitely.
    """
    await conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    cur = await conn.execute("SELECT embed_model, embed_dim FROM vm_meta")
    row = await cur.fetchone()
    if row is None:
        return
    if row["embed_dim"] != embed_dim:
        raise RuntimeError(
            f"embedding dimension mismatch: this database was migrated for "
            f"vector({row['embed_dim']}) using {row['embed_model']!r}, but the current "
            f"config asks for vector({embed_dim}) using {embed_model!r}. "
            f"Existing vectors are not comparable across models, and the column "
            f"type cannot change while it holds data. Point at a different schema "
            f"with pg_schema, or drop this one and re-ingest."
        )
    if row["embed_model"] != embed_model:
        logger.warning(
            "voicemem: this database was embedded with %r but the config says %r. "
            "The dimensions match so queries will run, but scores mix two vector "
            "spaces and retrieval quality will be poor.",
            row["embed_model"], embed_model,
        )
