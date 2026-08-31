-- SPDX-License-Identifier: Apache-2.0
-- Copyright 2026 Mahimai Labs
--
-- Schema derived from VoiceMem (https://github.com/xzf-thu/VoiceMem), Apache-2.0,
-- commit d587d424a02727d9ff3ef6f8e672a39a19ce64a7.
-- See CHANGES-FROM-UPSTREAM.md.
--
-- Ported from SQLite. Four changes apply throughout:
--
--   1. tenant_id on every table, in the leading position of every primary key
--      and every index. Upstream's tenancy boundary was the SQLite file itself,
--      so no table carried a tenant column and one shared database would have
--      had no separation at all.
--   2. Real types. JSON-in-TEXT becomes JSONB, ISO-8601 strings become
--      TIMESTAMPTZ, 0/1 integers become BOOLEAN, and vectors stored as JSON text
--      or as raw numpy bytes become pgvector columns.
--   3. The `memories` table is merged. Upstream kept fact metadata here and the
--      fact text plus its vector in Qdrant, joined by zipping two id lists
--      together. One table, one id, real foreign keys.
--   4. Schema changes live in numbered migrations. Upstream ran CREATE TABLE
--      plus fifteen ALTER TABLE statements on every single construction, each
--      wrapped in try/except; in Postgres the first duplicate-column error
--      aborts the surrounding transaction.
--
-- {{embed_dim}} is substituted by the migration runner from Config.embed_dim.
-- It cannot be a bind parameter: it is part of a type declaration.

-- ---------------------------------------------------------------------------
-- Bookkeeping
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS vm_meta (
    id            BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    embed_model   TEXT        NOT NULL,
    embed_dim     INTEGER     NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE vm_meta IS
  'Single row. The container asserts embed_model and embed_dim against Config at
   startup. Upstream had no such check: its trait store skipped shape-mismatched
   vectors with a bare continue, so switching embedding model left the right
   brain silently returning nothing, with no error, forever.';

-- ---------------------------------------------------------------------------
-- Left brain: facts
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memories (
    tenant_id     TEXT        NOT NULL,
    id            UUID        NOT NULL,
    user_id       TEXT        NOT NULL,
    content       TEXT        NOT NULL,
    embedding     vector({{embed_dim}}),
    slot          TEXT        NOT NULL DEFAULT '',
    memory_type   TEXT        NOT NULL DEFAULT 'fact',
    -- 'assistant' rows are the agent's own replies. search() excludes them by
    -- default; without that the agent retrieves and quotes itself, and answers
    -- drift further from the truth every turn.
    role          TEXT        NOT NULL DEFAULT 'user'
                              CHECK (role IN ('user', 'assistant')),
    attributed_to TEXT        NOT NULL DEFAULT 'user',
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    confidence    REAL        NOT NULL DEFAULT 1.0,
    sensitivity   REAL        NOT NULL DEFAULT 0.0,
    heat          REAL        NOT NULL DEFAULT 1.0,
    last_hit_at   TIMESTAMPTZ,
    -- The day the remembered event happened, as opposed to when the row was
    -- written. Backfilled history must set this or every memory looks like today.
    observed_on   DATE,
    archived_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_mem_scope
    ON memories (tenant_id, user_id, slot);
CREATE INDEX IF NOT EXISTS idx_mem_observed
    ON memories (tenant_id, user_id, observed_on DESC);

-- The only ANN index in the schema. HNSW rather than ivfflat because a library
-- creates its schema on an empty database: ivfflat needs training rows to pick
-- a useful list count, and there is no natural moment to rebuild it afterwards.
-- Partial, because archived rows are excluded from every search anyway.
CREATE INDEX IF NOT EXISTS idx_mem_embedding
    ON memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE archived_at IS NULL;

-- ---------------------------------------------------------------------------
-- Left brain: the cognitive graph
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    tenant_id   TEXT        NOT NULL,
    id          TEXT        NOT NULL,
    user_id     TEXT        NOT NULL,
    entity_type TEXT        NOT NULL,
    name        TEXT        NOT NULL,
    name_norm   TEXT        NOT NULL,
    slot        TEXT        NOT NULL,
    description TEXT        NOT NULL DEFAULT '',
    -- Upstream declared this only in an ALTER run at construction time, while
    -- insert code already wrote to it. A fresh database worked purely because
    -- that migration loop happened to run first.
    embedding   vector({{embed_dim}}),
    confidence  REAL        NOT NULL DEFAULT 1.0,
    importance  REAL        NOT NULL DEFAULT 0.5,
    aliases     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    properties  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_ent_scope ON entities (tenant_id, user_id);
CREATE INDEX IF NOT EXISTS idx_ent_norm
    ON entities (tenant_id, user_id, name_norm, entity_type);
-- No ANN index here on purpose. Entity deduplication is always scoped to one
-- (tenant, user, entity_type), which is far more selective than any
-- approximate search, and an HNSW index would only cause over-filtering. The
-- exact scan happens inside Postgres over a handful of rows, which already
-- replaces upstream's json.loads-plus-Python-cosine over every entity.
CREATE INDEX IF NOT EXISTS idx_ent_type_scope
    ON entities (tenant_id, user_id, entity_type);

CREATE TABLE IF NOT EXISTS entity_edges (
    tenant_id           TEXT        NOT NULL,
    id                  UUID        NOT NULL,
    user_id             TEXT        NOT NULL,
    from_entity_id      TEXT        NOT NULL,
    to_entity_id        TEXT        NOT NULL,
    relation_type       TEXT        NOT NULL,
    role_label          TEXT,
    confidence          REAL        NOT NULL DEFAULT 1.0,
    weight              REAL        NOT NULL DEFAULT 1.0,
    edge_type           TEXT        NOT NULL DEFAULT 'weak',
    status              TEXT        NOT NULL DEFAULT 'active',
    evidence_memory_ids JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, user_id, from_entity_id, to_entity_id, relation_type),
    FOREIGN KEY (tenant_id, from_entity_id) REFERENCES entities (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, to_entity_id)   REFERENCES entities (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edge_from ON entity_edges (tenant_id, user_id, from_entity_id);
CREATE INDEX IF NOT EXISTS idx_edge_to   ON entity_edges (tenant_id, user_id, to_entity_id);

CREATE TABLE IF NOT EXISTS entity_memory_links (
    tenant_id     TEXT        NOT NULL,
    id            UUID        NOT NULL,
    memory_id     UUID        NOT NULL,
    entity_id     TEXT        NOT NULL,
    user_id       TEXT        NOT NULL,
    role          TEXT        NOT NULL DEFAULT 'context',
    relation_hint TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, memory_id, entity_id),
    -- Enforceable only because memories is now one table. Upstream's equivalent
    -- link could reference an id that existed solely in Qdrant.
    FOREIGN KEY (tenant_id, memory_id) REFERENCES memories (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, entity_id) REFERENCES entities (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eml_mem ON entity_memory_links (tenant_id, memory_id);
CREATE INDEX IF NOT EXISTS idx_eml_ent ON entity_memory_links (tenant_id, entity_id);

CREATE TABLE IF NOT EXISTS affective_edges (
    tenant_id           TEXT        NOT NULL,
    id                  UUID        NOT NULL,
    user_id             TEXT        NOT NULL,
    from_entity_id      TEXT        NOT NULL,
    to_entity_id        TEXT,
    trigger_frame       TEXT,
    emotion             TEXT,
    appraisal           TEXT,
    response_policy     TEXT,
    confidence          REAL        NOT NULL DEFAULT 1.0,
    evidence_memory_ids JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, from_entity_id) REFERENCES entities (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ae_scope ON affective_edges (tenant_id, user_id, from_entity_id);

CREATE TABLE IF NOT EXISTS query_activations (
    tenant_id  TEXT        NOT NULL,
    id         UUID        NOT NULL,
    user_id    TEXT        NOT NULL,
    query_id   TEXT        NOT NULL,
    entity_id  TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_qa_query  ON query_activations (tenant_id, user_id, query_id);
CREATE INDEX IF NOT EXISTS idx_qa_entity ON query_activations (tenant_id, user_id, entity_id);
-- Append-only with no retention policy anywhere upstream. See 0001's companion
-- note in the runner: prune_activations() exists for this reason.
CREATE INDEX IF NOT EXISTS idx_qa_created ON query_activations (tenant_id, created_at);

-- ---------------------------------------------------------------------------
-- Left brain: slot routing
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_tags (
    tenant_id  TEXT NOT NULL,
    memory_id  UUID NOT NULL,
    user_id    TEXT NOT NULL,
    -- Overloaded upstream: base slot values, emergent dynamic-slot names, and
    -- namespaced tags such as 'scene:x' queried with LIKE. Postgres LIKE is
    -- case-sensitive where SQLite's is not, so prefix queries need the
    -- text_pattern_ops index below to stay on an index.
    slot       TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (tenant_id, memory_id, slot),
    FOREIGN KEY (tenant_id, memory_id) REFERENCES memories (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_slot ON memory_tags (tenant_id, user_id, slot);
CREATE INDEX IF NOT EXISTS idx_tags_prefix
    ON memory_tags (tenant_id, user_id, slot text_pattern_ops);

CREATE TABLE IF NOT EXISTS slot_summaries (
    tenant_id  TEXT        NOT NULL,
    user_id    TEXT        NOT NULL,
    slot       TEXT        NOT NULL,
    summary    TEXT        NOT NULL,
    mem_count  INTEGER     NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, slot)
);

CREATE TABLE IF NOT EXISTS slot_profiles (
    tenant_id    TEXT        NOT NULL,
    user_id      TEXT        NOT NULL,
    slot         TEXT        NOT NULL,
    summary      TEXT        NOT NULL DEFAULT '',
    entity_ids   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    entity_count INTEGER     NOT NULL DEFAULT 0,
    memory_count INTEGER     NOT NULL DEFAULT 0,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, slot)
);

CREATE TABLE IF NOT EXISTS slot_macro_edges (
    tenant_id  TEXT        NOT NULL,
    user_id    TEXT        NOT NULL,
    slot_a     TEXT        NOT NULL,
    slot_b     TEXT        NOT NULL,
    weight     REAL        NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, slot_a, slot_b)
);

-- ---------------------------------------------------------------------------
-- Left brain: emergent sub-slots
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dynamic_slots (
    tenant_id    TEXT        NOT NULL,
    user_id      TEXT        NOT NULL,
    name         TEXT        NOT NULL,
    description  TEXT        NOT NULL,
    embedding    vector({{embed_dim}}),
    parent_slots JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, name)
);

CREATE TABLE IF NOT EXISTS graph_entities (
    tenant_id   TEXT        NOT NULL,
    id          UUID        NOT NULL,
    user_id     TEXT        NOT NULL,
    slot_ref    TEXT        NOT NULL,
    name        TEXT        NOT NULL,
    description TEXT        NOT NULL DEFAULT '',
    -- Upstream stored this as raw numpy float32 bytes and scanned every row of
    -- the slot in Python on each mention.
    embedding   vector({{embed_dim}}),
    can_split   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, user_id, slot_ref, name)
);

CREATE INDEX IF NOT EXISTS idx_ge_slot ON graph_entities (tenant_id, user_id, slot_ref);

CREATE TABLE IF NOT EXISTS graph_entity_memories (
    tenant_id  TEXT        NOT NULL,
    id         UUID        NOT NULL,
    entity_id  UUID        NOT NULL,
    user_id    TEXT        NOT NULL,
    memory_id  UUID        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, entity_id, memory_id),
    FOREIGN KEY (tenant_id, entity_id) REFERENCES graph_entities (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, memory_id) REFERENCES memories (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_gem_entity ON graph_entity_memories (tenant_id, entity_id);
CREATE INDEX IF NOT EXISTS idx_gem_memory ON graph_entity_memories (tenant_id, user_id, memory_id);

CREATE TABLE IF NOT EXISTS graph_query_activations (
    tenant_id  TEXT        NOT NULL,
    id         UUID        NOT NULL,
    user_id    TEXT        NOT NULL,
    query_id   TEXT        NOT NULL,
    entity_id  UUID        NOT NULL,
    session_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_gqa_query   ON graph_query_activations (tenant_id, user_id, query_id);
CREATE INDEX IF NOT EXISTS idx_gqa_session ON graph_query_activations (tenant_id, user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_gqa_created ON graph_query_activations (tenant_id, created_at);

-- ---------------------------------------------------------------------------
-- Right brain
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS right_brain_memories (
    tenant_id           TEXT        NOT NULL,
    id                  UUID        NOT NULL,
    user_id             TEXT        NOT NULL,
    memory_class        TEXT        NOT NULL,
    content             TEXT        NOT NULL,
    condition_text      TEXT,
    priority            REAL        NOT NULL DEFAULT 0.5,
    confidence          REAL        NOT NULL DEFAULT 1.0,
    -- Renamed from `ttl`. Upstream has two different `ttl` columns of two
    -- different types under one name; SQLite's dynamic typing hid the clash.
    ttl_class           TEXT        NOT NULL DEFAULT 'long_term',
    metadata            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    evidence_turn_ids   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    evidence_memory_ids JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_rbm_scope ON right_brain_memories (tenant_id, user_id, memory_class);

CREATE TABLE IF NOT EXISTS right_brain_anchor_links (
    tenant_id       TEXT        NOT NULL,
    id              UUID        NOT NULL,
    user_id         TEXT        NOT NULL,
    right_memory_id UUID        NOT NULL,
    anchor_type     TEXT        NOT NULL,
    anchor_id       TEXT,
    role            TEXT        NOT NULL DEFAULT 'context',
    weight          REAL        NOT NULL DEFAULT 1.0,
    confidence      REAL        NOT NULL DEFAULT 1.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    -- Upstream declared this FK but never enabled foreign keys on the
    -- connection, so it was decorative. Here it is enforced.
    FOREIGN KEY (tenant_id, right_memory_id)
        REFERENCES right_brain_memories (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rbal_mem ON right_brain_anchor_links (tenant_id, right_memory_id);
CREATE INDEX IF NOT EXISTS idx_rbal_anchor
    ON right_brain_anchor_links (tenant_id, user_id, anchor_type, anchor_id);

-- Upstream re-links the same (right_memory_id, anchor) pair repeatedly because
-- its ON CONFLICT had no conflict target and so could never fire. anchor_id is
-- nullable and NULL is distinct by default, hence the expression index rather
-- than a plain UNIQUE. UNIQUE NULLS NOT DISTINCT would be cleaner but needs
-- PostgreSQL 15, and the floor here is 14.
CREATE UNIQUE INDEX IF NOT EXISTS uq_rbal_pair
    ON right_brain_anchor_links
       (tenant_id, right_memory_id, anchor_type, COALESCE(anchor_id, ''), role);

CREATE TABLE IF NOT EXISTS rb_traits (
    tenant_id  TEXT        NOT NULL,
    id         UUID        NOT NULL,
    user_id    TEXT        NOT NULL,
    slot       TEXT        NOT NULL,
    claim      TEXT        NOT NULL,
    -- Upstream: raw numpy float32 bytes in a BLOB, with mismatched shapes
    -- silently skipped. That is the failure that makes a swapped embedding
    -- model look like an empty right brain rather than an error.
    embedding  vector({{embed_dim}}),
    confidence REAL        NOT NULL DEFAULT 0.9,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_tr_scope ON rb_traits (tenant_id, user_id, slot);

CREATE TABLE IF NOT EXISTS rb_evidence (
    tenant_id  TEXT        NOT NULL,
    id         UUID        NOT NULL,
    trait_id   UUID        NOT NULL,
    user_id    TEXT        NOT NULL,
    quote      TEXT        NOT NULL,
    emotion    TEXT        NOT NULL DEFAULT '',
    cause      TEXT        NOT NULL DEFAULT '',
    cause_id   UUID,
    -- Upstream overloaded created_at as the event time, so it held a mixture of
    -- write timestamps and caller-supplied observation dates. Split in two.
    observed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, trait_id) REFERENCES rb_traits (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ev_trait ON rb_evidence (tenant_id, trait_id);

-- ---------------------------------------------------------------------------
-- Session state
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS session_state (
    tenant_id       TEXT        NOT NULL,
    user_id         TEXT        NOT NULL,
    last_session_id TEXT,
    turn_count      INTEGER     NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id)
);

CREATE TABLE IF NOT EXISTS touched_refs (
    tenant_id  TEXT        NOT NULL,
    user_id    TEXT        NOT NULL,
    namespace  TEXT        NOT NULL,
    ref        TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, namespace, ref)
);

CREATE TABLE IF NOT EXISTS kv (
    tenant_id TEXT  NOT NULL,
    -- Upstream's kv was a per-file singleton with no user column, so two users
    -- in one space overwrote each other's state.
    user_id   TEXT  NOT NULL,
    k         TEXT  NOT NULL,
    v         JSONB NOT NULL,
    PRIMARY KEY (tenant_id, user_id, k)
);

-- ---------------------------------------------------------------------------
-- The ingest outbox
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ingest_outbox (
    tenant_id         TEXT        NOT NULL,
    id                UUID        NOT NULL,
    user_id           TEXT        NOT NULL,
    user_text         TEXT        NOT NULL,
    agent_reply       TEXT        NOT NULL DEFAULT '',
    prior_agent_reply TEXT        NOT NULL DEFAULT '',
    session_id        TEXT        NOT NULL DEFAULT '',
    observed_at       TIMESTAMPTZ,
    interrupted       BOOLEAN     NOT NULL DEFAULT FALSE,
    attempts          INTEGER     NOT NULL DEFAULT 0,
    leased_until      TIMESTAMPTZ,
    done_at           TIMESTAMPTZ,
    last_error        TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
);

COMMENT ON TABLE ingest_outbox IS
  'Durability for the write path. Stage one is a single fast insert with no LLM
   call, so a turn survives the worker dying. Stage two claims a row with
   FOR UPDATE SKIP LOCKED, runs the five-call pipeline and marks it done; the
   lease makes an abandoned row recoverable by the next worker.';

CREATE INDEX IF NOT EXISTS idx_outbox_claimable
    ON ingest_outbox (tenant_id, created_at)
    WHERE done_at IS NULL;
