-- Application keys for the cache data path.
--
-- The secret is never stored. What lives here is its SHA-256 hash, the application the key
-- was issued to, and enough of the secret's prefix that a person can tell two keys apart in
-- a list without either being usable from what they can see. A copy of this table is
-- therefore not a set of working credentials.
--
-- It exists at all because the engine's key registry is in memory, and a platform that
-- restarts containers freely would otherwise invalidate every key an operator handed out.
--
-- Idempotent, like every other migration here.

create table if not exists public.aura_api_keys (
    id          text primary key,
    application text        not null,
    key_hash    text        not null unique,
    hint        text        not null,
    created_at  timestamptz not null default now(),
    revoked     boolean     not null default false,
    constraint aura_api_keys_application_not_blank check (length(btrim(application)) > 0)
);

create index if not exists aura_api_keys_application_idx
    on public.aura_api_keys (application)
    where revoked = false;

-- Service role only. There is no policy granting anon or authenticated any access, which
-- with RLS enabled means the browser cannot read key hashes even if it holds a valid login.
alter table public.aura_api_keys enable row level security;

comment on table public.aura_api_keys is
    'SHA-256 hashes of application keys issued by the AURA engine. Never contains a secret.';
