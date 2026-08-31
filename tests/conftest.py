# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
from __future__ import annotations

import os
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


@pytest.fixture
def unique_user() -> str:
    """A fresh user per test, so tests never see each other's memories."""
    return f"test_{uuid.uuid4().hex[:12]}"
