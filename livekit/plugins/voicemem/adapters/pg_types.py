# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""pgvector wire format for psycopg 3, without the ``pgvector`` package.

Twenty-five lines against one more dependency. The text format pgvector accepts
is ``[1,2,3]``, which is trivial to produce and parse, and keeping it here means
the runtime dependency list stays at three packages.

Registration is per connection, done by the pool's configure hook, so nothing
here mutates global adapter state that another library in the host process might
also be using.
"""

from __future__ import annotations

from collections.abc import Sequence

from psycopg import Connection
from psycopg.abc import AdaptContext
from psycopg.adapt import Dumper, Loader
from psycopg.types import TypeInfo

__all__ = ["Vector", "register_vector", "register_vector_async"]


class Vector(list):
    """A float sequence that psycopg will send as a pgvector ``vector``.

    Wrapping rather than adapting ``list`` directly is deliberate: adapting the
    builtin would change how every other list in the host application is sent.
    """

    __slots__ = ()

    def __init__(self, values: Sequence[float] = ()) -> None:
        super().__init__(float(v) for v in values)


class _VectorDumper(Dumper):
    def dump(self, obj: Sequence[float]) -> bytes:
        # repr() on a float round-trips exactly; str() is the same in py3 but
        # repr states the intent, which is that no precision is lost here.
        return ("[" + ",".join(repr(float(v)) for v in obj) + "]").encode()


class _VectorLoader(Loader):
    def load(self, data: bytes | memoryview) -> list[float]:
        raw = bytes(data).decode()
        inner = raw.strip()[1:-1]
        return [float(x) for x in inner.split(",")] if inner else []


def _register(info: TypeInfo | None, context: AdaptContext) -> None:
    if info is None:
        raise RuntimeError(
            "the 'vector' type was not found. Run CREATE EXTENSION vector in this "
            "database, or use an image that ships pgvector such as pgvector/pgvector:pg17."
        )
    info.register(context)

    dumper = type("VectorDumper", (_VectorDumper,), {"oid": info.oid})
    loader = type("VectorLoader", (_VectorLoader,), {})

    adapters = context.adapters
    adapters.register_dumper(Vector, dumper)
    adapters.register_loader(info.oid, loader)


def register_vector(conn: Connection) -> None:
    _register(TypeInfo.fetch(conn, "vector"), conn)


async def register_vector_async(conn: object) -> None:
    """Register on an async connection. Called from the pool's configure hook."""
    from psycopg import AsyncConnection

    assert isinstance(conn, AsyncConnection)
    _register(await TypeInfo.fetch(conn, "vector"), conn)
