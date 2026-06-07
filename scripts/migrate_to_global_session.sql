-- migrate_to_global_session.sql
-- Migrates all existing memories and sessions to use '__global__' as their
-- session_id, enabling cross-profile memory sharing when using global strategy.
--
-- Usage:
--   psql -d hmemory -f scripts/migrate_to_global_session.sql
--
-- Or from hmemory config:
--   psql "$DATABASE_URL" -f scripts/migrate_to_global_session.sql

BEGIN;

-- 1. Update all memories to the global session
UPDATE memories SET session_id = '__global__', updated_at = now()
WHERE session_id IS DISTINCT FROM '__global__';

-- 2. Upsert a single global session record
INSERT INTO sessions (session_id, strategy, path_or_repo, created_at, last_active_at, turn_count)
VALUES ('__global__', 'global', '', now(), now(), 0)
ON CONFLICT (session_id) DO UPDATE SET last_active_at = now();

-- 3. Clean up old per-session records (they no longer map to any memory)
DELETE FROM sessions WHERE session_id IS DISTINCT FROM '__global__';

COMMIT;