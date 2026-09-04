-- The durable half of the audit log.
--
-- The engine keeps a few hundred entries in memory, which answers "what just happened" and
-- nothing else. This table answers "why did we serve that value on Tuesday", which is the
-- question that actually gets asked, and it has to survive the process that made the
-- decision. Written by the engine's shipper in batches; nothing reads it on the hot path.

create table if not exists public.aura_audit_log (
    id           bigserial primary key,
    seq          bigint      not null,
    t_ms         double precision not null,
    at           text        not null,
    kind         text        not null,
    label        text        not null,
    severity     text        not null,
    subject      text        not null,
    application  text        not null,
    -- The sentence. This is the column a person reads.
    message      text        not null,
    -- The numbers behind the sentence, kept structured so the dashboard can filter on them
    -- without parsing prose.
    facts        jsonb       not null default '[]'::jsonb,
    usd_impact   double precision not null default 0,
    created_at   timestamptz not null default now()
);

-- Reading is always "the most recent N", usually narrowed to one application or to the
-- events that cost money, so those are the three indexes that matter.
create index if not exists aura_audit_log_recent_idx
    on public.aura_audit_log (created_at desc);
create index if not exists aura_audit_log_app_idx
    on public.aura_audit_log (application, created_at desc);
create index if not exists aura_audit_log_kind_idx
    on public.aura_audit_log (kind, created_at desc);

alter table public.aura_audit_log enable row level security;

-- The engine writes with the service role, which bypasses RLS. Anonymous readers get the
-- log read-only, because the dashboard is the point of it.
drop policy if exists aura_audit_log_read on public.aura_audit_log;
create policy aura_audit_log_read
    on public.aura_audit_log for select
    using (true);

-- `aura_events.kind` is a foreign key onto a lookup table, so an event kind that is not
-- listed there is rejected with a 409 rather than inserted. That is the right design -- it
-- stops a typo becoming a new category nobody ever queries -- but it means every new kind
-- has to be declared. These are the ones the engine now emits.
insert into public.aura_event_kinds (code) values
    ('engine_started'), ('Invalidate'), ('VersionBump'), ('ModelReload'),
    ('PolicyShift'), ('ScaleUp'), ('ScaleDown'), ('AttackStart'), ('SimStart')
on conflict (code) do nothing;
