-- World Trade Radar — Pilot Security Phase 1
-- Account membership + ownership foundations.
-- Run in Supabase SQL Editor. service_role bypasses RLS; policies deny anon.

-- ---------------------------------------------------------------------------
-- account_members
-- ---------------------------------------------------------------------------
create table if not exists public.account_members (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  account_id uuid not null references public.accounts (id) on delete cascade,
  role text not null default 'member'
    check (role in ('owner', 'member', 'admin')),
  created_at timestamptz not null default now(),
  unique (user_id, account_id)
);

create index if not exists account_members_user_id_idx
  on public.account_members (user_id);

create index if not exists account_members_account_id_idx
  on public.account_members (account_id);

comment on table public.account_members is
  'JWT sub (user_id) → accounts. Client X-Account-Slug is not trusted.';

alter table public.account_members enable row level security;

-- No permissive policies: browser clients cannot read/write membership.
-- Backend uses service_role and application-level authorization.

-- ---------------------------------------------------------------------------
-- Optional account_id on logs/runs (nullable for backfill)
-- ---------------------------------------------------------------------------
alter table public.agent_logs
  add column if not exists account_id uuid references public.accounts (id) on delete set null;

create index if not exists agent_logs_account_id_idx
  on public.agent_logs (account_id);

alter table public.agent_runs
  add column if not exists account_id uuid references public.accounts (id) on delete set null;

create index if not exists agent_runs_account_id_idx
  on public.agent_runs (account_id);

-- ---------------------------------------------------------------------------
-- consultant_profiles: add account_id alongside legacy account_slug
-- ---------------------------------------------------------------------------
alter table public.consultant_profiles
  add column if not exists account_id uuid references public.accounts (id) on delete cascade;

create index if not exists consultant_profiles_account_id_idx
  on public.consultant_profiles (account_id);

alter table public.consultant_memories
  add column if not exists account_id uuid references public.accounts (id) on delete cascade;

create index if not exists consultant_memories_account_id_idx
  on public.consultant_memories (account_id);

comment on column public.consultant_profiles.account_id is
  'Preferred tenant key; account_slug retained for legacy rows.';
comment on column public.consultant_memories.account_id is
  'Preferred tenant key for account-owned memory.';
