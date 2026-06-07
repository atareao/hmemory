-- migrate_session_to_profile.sql
-- Migrates from session_id-based isolation to profile-based grouping.
-- profile = Hermes profile name (e.g. "charla", "linuxdev", "rustdev")
--
-- Usage:
--   psql -d hmemory -f scripts/migrate_session_to_profile.sql

BEGIN;

-- 1. Add profile column and backfill from existing session_id
ALTER TABLE memories ADD COLUMN IF NOT EXISTS profile TEXT NOT NULL DEFAULT '';
UPDATE memories SET profile = COALESCE(session_id, 'default');

-- 2. Drop old session_id index and column
DROP INDEX IF EXISTS idx_memories_session;
ALTER TABLE memories DROP COLUMN IF EXISTS session_id;

-- 3. Create profile index
CREATE INDEX IF NOT EXISTS idx_memories_profile ON memories (profile);

-- 4. Drop BM25 index on old columns, recreate with profile
DROP INDEX IF EXISTS idx_memories_bm25;
CREATE INDEX IF NOT EXISTS idx_memories_bm25 ON memories
  USING bm25 (id, content, tags, source, profile)
  WITH (key_field = 'id');

-- 5. Replace sessions table with profile_stats
DROP TABLE IF EXISTS sessions;
CREATE TABLE IF NOT EXISTS profile_stats (
    profile TEXT PRIMARY KEY,
    turn_count BIGINT DEFAULT 0,
    last_active_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 6. Backfill profile_stats
INSERT INTO profile_stats (profile, turn_count, last_active_at, created_at)
SELECT DISTINCT ON (profile) profile, 0, now(), now()
FROM memories
WHERE profile != ''
ON CONFLICT (profile) DO NOTHING;

COMMIT;