-- Console accounts for the AURA engine.
--
-- Password hashes only: PBKDF2-HMAC-SHA256 with 120,000 iterations and a per-user random
-- salt, stored as `pbkdf2$iterations$salt$hash` so the parameters travel with the hash and
-- can be raised later without invalidating anyone. A password itself never leaves the
-- browser that typed it, and a copy of this table is not a set of working logins.
--
-- The engine seeds a root account from AURA_ROOT_EMAIL and AURA_ROOT_PASSWORD on first boot
-- and writes it here, so a recycled container does not lose the only way in.
--
-- Idempotent, like every other migration here.

create table if not exists public.aura_users (
    id            text primary key,
    email         text        not null unique,
    role          text        not null default 'operator',
    password_hash text        not null,
    created_at    timestamptz not null default now(),
    constraint aura_users_role_known check (role in ('root', 'operator')),
    constraint aura_users_email_shaped check (position('@' in email) > 1),
    -- A plaintext password could not be written here even by mistake.
    constraint aura_users_hash_is_pbkdf2 check (password_hash like 'pbkdf2$%')
);

-- Service role only. No policy grants anon or authenticated anything, so the browser cannot
-- read password hashes even holding a valid login.
alter table public.aura_users enable row level security;

comment on table public.aura_users is
    'Console accounts for the AURA engine. Password hashes only (PBKDF2-HMAC-SHA256, per-user salt); never a password.';
