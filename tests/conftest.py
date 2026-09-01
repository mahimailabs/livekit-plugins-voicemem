# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
from __future__ import annotations

import os
import re
import uuid

import pytest

TEST_DSN_ENV = "VOICEMEM_TEST_DSN"
#: Set in CI. Turns "the database container failed to start" from a green build
#: with everything skipped into a red one.
REQUIRE_DB_ENV = "VOICEMEM_REQUIRE_DB"


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless a database is configured."""
    dsn = os.environ.get(TEST_DSN_ENV)
    if dsn:
        return
    if os.environ.get(REQUIRE_DB_ENV) == "1":
        raise pytest.UsageError(
            f"{REQUIRE_DB_ENV}=1 but {TEST_DSN_ENV} is not set. Refusing to pass by "
            "skipping the entire integration layer."
        )
    skip = pytest.mark.skip(reason=f"set {TEST_DSN_ENV} to run integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def test_dsn() -> str:
    dsn = os.environ.get(TEST_DSN_ENV)
    if not dsn:
        pytest.skip(f"{TEST_DSN_ENV} not set")
    return dsn


@pytest.fixture(scope="session")
def schema_dims(test_dsn) -> int:
    """The width the test schema was actually migrated with.

    Read rather than assumed. The default embedding model, and therefore the
    default width, is a property of the release: hardcoding it here means the
    storage tests break the day it changes, for a reason that has nothing to do
    with storage.
    """
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(test_dsn, row_factory=dict_row) as conn:
        row = conn.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS declared
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'voicemem' AND c.relname = 'memories'
              AND a.attname = 'embedding'
            """
        ).fetchone()
    if row is None:
        pytest.skip("the voicemem schema is not migrated; run 'voicemem-db upgrade'")
    match = re.search(r"\((\d+)\)", str(row["declared"]))
    assert match, f"embedding column has no declared width: {row['declared']}"
    return int(match.group(1))


@pytest.fixture
def unique_user() -> str:
    """A fresh user per test, so tests never see each other's memories."""
    return f"test_{uuid.uuid4().hex[:12]}"
