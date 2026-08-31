-- SPDX-License-Identifier: Apache-2.0
-- Copyright 2026 Mahimai Labs
--
-- Row-level security: defence in depth for multi-tenancy.
--
-- Every query this package issues already carries tenant_id, and
-- tests/integration/test_tenancy_schema.py asserts that for all 24 tables. This
-- migration exists because that assertion covers the code we wrote today, and a
-- single forgotten WHERE clause in future vendored code is a cross-tenant read.
-- With RLS the database refuses the row regardless of what the SQL asked for.
--
-- The setting is per transaction, not per session, which is what keeps it safe
-- behind pgbouncer in transaction pooling mode: a pooled connection handed to
-- another tenant mid-flight carries nothing over, because SET LOCAL is undone
-- at commit. A connection that somehow arrives without the setting reads zero
-- rows rather than everyone's.
--
-- RLS ONLY PROTECTS A ROLE THAT CAN BE CONSTRAINED. Two roles bypass it
-- unconditionally, and no amount of policy or FORCE changes that:
--
--   * superusers
--   * roles with the BYPASSRLS attribute
--
-- FORCE ROW LEVEL SECURITY makes the table *owner* obey its own policies, but a
-- superuser owner still walks straight through. This was verified rather than
-- assumed: with policies in place and FORCE set, a superuser connection saw
-- every tenant's rows, including with no tenant set at all.
--
-- So this migration also creates a dedicated, deliberately unprivileged role.
-- Connecting as anything else leaves the policies inert, and
-- voicemem-db status reports that as a warning rather than letting it pass for
-- protection you do not have.

-- ---------------------------------------------------------------------------
-- The accessor
-- ---------------------------------------------------------------------------

-- STABLE, not VOLATILE, so the planner may hoist it out of the row loop rather
-- than calling it once per candidate row. Marked STRICT-free deliberately: it
-- must return a value for the missing case rather than NULL, because
-- `tenant_id = NULL` is NULL, not false, and a NULL policy result is treated as
-- false anyway but reads confusingly in EXPLAIN.
CREATE OR REPLACE FUNCTION voicemem_current_tenant()
RETURNS TEXT
LANGUAGE sql
STABLE
AS $$
    SELECT coalesce(current_setting('voicemem.tenant_id', true), '')
$$;

COMMENT ON FUNCTION voicemem_current_tenant() IS
  'Reads the tenant set by SET LOCAL voicemem.tenant_id. The second argument to
   current_setting is missing_ok: without it, a connection that has not set the
   GUC raises rather than returning no rows, which turns a policy into an
   outage.';

-- ---------------------------------------------------------------------------
-- Policies
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    t TEXT;
    tables TEXT[] := ARRAY[
        'memories',
        'entities',
        'entity_edges',
        'entity_memory_links',
        'affective_edges',
        'query_activations',
        'memory_tags',
        'slot_summaries',
        'slot_profiles',
        'slot_macro_edges',
        'dynamic_slots',
        'graph_entities',
        'graph_entity_memories',
        'graph_query_activations',
        'right_brain_memories',
        'right_brain_anchor_links',
        'rb_traits',
        'rb_evidence',
        'session_state',
        'touched_refs',
        'kv',
        'ingest_outbox'
    ];
BEGIN
    FOREACH t IN ARRAY tables LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        -- Without FORCE, the table owner bypasses the policy entirely. In most
        -- deployments the migration role and the runtime role are the same, so
        -- omitting this would make the whole migration a no-op.
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);

        EXECUTE format('DROP POLICY IF EXISTS voicemem_tenant_isolation ON %I', t);
        -- USING governs what is visible to SELECT, UPDATE and DELETE.
        -- WITH CHECK governs what INSERT and UPDATE may write, and is what stops
        -- a caller inserting a row belonging to somebody else.
        EXECUTE format(
            'CREATE POLICY voicemem_tenant_isolation ON %I
                 USING (tenant_id = voicemem_current_tenant())
                 WITH CHECK (tenant_id = voicemem_current_tenant())',
            t
        );
    END LOOP;
END
$$;

-- vm_meta is deliberately excluded. It holds the embedding model and dimension
-- for the whole database, not for a tenant, and the container reads it at
-- startup before any tenant context exists.

-- ---------------------------------------------------------------------------
-- The application role
-- ---------------------------------------------------------------------------

-- NOINHERIT so membership in another role cannot quietly widen it. No
-- SUPERUSER, no BYPASSRLS, and it does not own the tables, which together are
-- the only reason the policies above have any effect.
--
-- No password is set here. A migration file is the wrong place for a
-- credential, and it would end up in shell history and in the output of
-- voicemem-db sql. Set one out of band:
--
--     ALTER ROLE voicemem_app WITH LOGIN PASSWORD '...';
--
-- then point Config.pg_dsn at that role.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'voicemem_app') THEN
        CREATE ROLE voicemem_app NOLOGIN NOINHERIT NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
END
$$;

-- DML only. The runtime role has no DDL rights on purpose: schema changes are a
-- deliberate operator action, and twenty workers booting at once racing DDL is
-- a real failure rather than a theoretical one.
GRANT USAGE ON SCHEMA voicemem TO voicemem_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA voicemem TO voicemem_app;
GRANT SELECT ON vm_meta TO voicemem_app;
GRANT EXECUTE ON FUNCTION voicemem_current_tenant() TO voicemem_app;

-- Tables added by later migrations inherit the same grants, so a new migration
-- cannot accidentally ship a table the application cannot read.
ALTER DEFAULT PRIVILEGES IN SCHEMA voicemem
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO voicemem_app;

COMMENT ON ROLE voicemem_app IS
  'Runtime role for livekit-plugins-voicemem. Must not be granted SUPERUSER or
   BYPASSRLS, and must not own the tables: any of those silently disables tenant
   isolation while leaving the policies visible in pg_policies.';
