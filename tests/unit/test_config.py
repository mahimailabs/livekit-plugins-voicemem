# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
import dataclasses

import pytest

from livekit.plugins.voicemem.config import Config

DSN = "postgresql://bob:hunter2@db.example.com:5432/app"


def cfg(**kw):
    return Config(**{"pg_dsn": DSN, "openai_api_key": "sk-test", **kw})


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg().top_k = 9


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"pg_dsn": ""}, "pg_dsn"),
        ({"openai_api_key": ""}, "openai_api_key"),
        ({"tenant_id": ""}, "tenant_id"),
        ({"embed_dim": 0}, "embed_dim"),
        ({"recall_budget_s": 0}, "recall_budget_s"),
        ({"pool_min_size": 5, "pool_max_size": 2}, "pool_max_size"),
    ],
)
def test_invalid_values_are_rejected(kw, msg):
    base = {"pg_dsn": DSN, "openai_api_key": "sk-test"}
    with pytest.raises(ValueError, match=msg):
        Config(**{**base, **kw})


def test_schema_name_is_validated_because_it_reaches_ddl_unparameterised():
    with pytest.raises(ValueError, match="pg_schema"):
        cfg(pg_schema="public; DROP TABLE users")


def test_replace_returns_a_copy():
    a = cfg()
    b = a.replace(top_k=9)
    assert a.top_k == 5 and b.top_k == 9


def test_redacted_hides_key_and_dsn_password():
    r = cfg().redacted()
    assert r["openai_api_key"] == "***"
    assert "hunter2" not in r["pg_dsn"]
    assert r["pg_dsn"] == "postgresql://bob:***@db.example.com:5432/app"


def test_redacted_never_leaks_an_unparseable_dsn():
    r = Config(pg_dsn="host=db password=hunter2", openai_api_key="k").redacted()
    assert "hunter2" not in r["pg_dsn"]


def test_no_module_reads_env_unless_asked(monkeypatch):
    # Nothing implicit: a bare Config never consults the environment.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert cfg(openai_api_key="sk-explicit").openai_api_key == "sk-explicit"


def test_from_env_is_explicit_and_overrides_win(monkeypatch):
    monkeypatch.setenv("VOICEMEM_PG_DSN", DSN)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("VOICEMEM_TENANT_ID", "acme")
    c = Config.from_env(top_k=7)
    assert c.tenant_id == "acme" and c.openai_api_key == "sk-env" and c.top_k == 7


def test_from_env_rejects_a_non_integer_dimension(monkeypatch):
    monkeypatch.setenv("VOICEMEM_PG_DSN", DSN)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("VOICEMEM_EMBED_DIM", "big")
    with pytest.raises(ValueError, match="VOICEMEM_EMBED_DIM"):
        Config.from_env()


def test_constructing_config_does_not_mutate_the_environment(monkeypatch):
    # Upstream writes the key into os.environ, which reassigns credentials for
    # every other tenant in the same process.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg(openai_api_key="sk-secret")
    import os

    assert "OPENAI_API_KEY" not in os.environ
