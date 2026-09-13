-- Phase 2.5 activation — idempotent, non-destructive
-- Aligns with Phase 1 schema_account_members.sql + billing/accounts foundations.
-- service_role bypasses RLS; no permissive policies = deny anon/authenticated direct access.
-- Does NOT drop data. No DROP TABLE / DROP COLUMN / DELETE / TRUNCATE.
-- Does NOT insert demo as customer identity.
-- Does NOT add Upper plan (Phase 3 entitlement migration).

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- ---------------------------------------------------------------------------
-- plans (Free + Pro only)
-- ---------------------------------------------------------------------------
create table if not exists public.plans (
  id text primary key
    check (id in ('free', 'pro')),
  name text not null,
  daily_search_limit integer not null check (daily_search_limit >= 0),
  daily_token_limit integer not null check (daily_token_limit >= 0),
  monthly_price_usd numeric(10, 2) not null default 0,
  features jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

insert into public.plans (
  id, name, daily_search_limit, daily_token_limit, monthly_price_usd, features
) values
  ('free', 'Free', 8, 15000, 0, '[]'::jsonb),
  ('pro', 'Pro', 250, 500000, 49, '[]'::jsonb)
on conflict (id) do nothing;

alter table public.plans enable row level security;
-- No permissive policies: browser cannot mutate plans via PostgREST.

-- ---------------------------------------------------------------------------
-- accounts
-- ---------------------------------------------------------------------------
create table if not exists public.accounts (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  display_name text not null default 'Account',
  email text,
  plan_id text not null default 'free' references public.plans (id),
  plan_started_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists accounts_plan_id_idx on public.accounts (plan_id);

alter table public.accounts enable row level security;
-- No permissive policies: account rows are backend/service_role only.

comment on table public.accounts is
  'SaaS tenant. Authorization via JWT → account_members, never client slug.';

-- ---------------------------------------------------------------------------
-- account_members (Phase 1)
-- ---------------------------------------------------------------------------
create table if not exists public.account_members (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
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
-- No permissive policies: deny-by-default for anon/authenticated.
-- Backend uses service_role + application-level authorization.

-- ---------------------------------------------------------------------------
-- agent_runs (aligned with agent_runs.sql; plus account_id)
-- ---------------------------------------------------------------------------
create table if not exists public.agent_runs (
  id uuid primary key default gen_random_uuid(),
  query text not null,
  intent text
    check (
      intent is null or intent in (
        'trade_advisor',
        'supplier_finder',
        'product_matching'
      )
    ),
  selected_agent text,
  status text not null default 'running'
    check (status in ('pending', 'running', 'success', 'error')),
  steps jsonb not null default '[]'::jsonb,
  result jsonb not null default '{}'::jsonb,
  error_message text,
  duration_ms integer,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists agent_runs_created_at_idx
  on public.agent_runs (created_at desc);
create index if not exists agent_runs_intent_idx
  on public.agent_runs (intent);

alter table public.agent_runs enable row level security;

alter table public.agent_runs
  add column if not exists account_id uuid references public.accounts (id) on delete set null;
create index if not exists agent_runs_account_id_idx
  on public.agent_runs (account_id);

-- ---------------------------------------------------------------------------
-- agent_logs.account_id (nullable for backfill; preserves existing rows)
-- ---------------------------------------------------------------------------
alter table public.agent_logs
  add column if not exists account_id uuid references public.accounts (id) on delete set null;
create index if not exists agent_logs_account_id_idx
  on public.agent_logs (account_id);

-- ---------------------------------------------------------------------------
-- consultant_profiles / memories
-- Temporary dual-key state: account_slug remains; account_id added for Phase 2.5.
-- Phase 3: migrate to account-owned profile/memory (account_id authoritative).
-- ---------------------------------------------------------------------------
create table if not exists public.consultant_profiles (
  account_slug text primary key,
  company_name text,
  sector text,
  products jsonb not null default '[]'::jsonb,
  capacity text,
  current_markets text,
  target_markets text,
  sales_channels text,
  pricing_notes text,
  moq text,
  logistics text,
  brand text,
  target_customer text,
  commercial_goals text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.consultant_profiles
  add column if not exists account_id uuid references public.accounts (id) on delete cascade;
create index if not exists consultant_profiles_account_id_idx
  on public.consultant_profiles (account_id);

comment on column public.consultant_profiles.account_id is
  'Preferred tenant key; account_slug retained for legacy rows (Phase 3 debt).';

create table if not exists public.consultant_memories (
  id uuid primary key default gen_random_uuid(),
  account_slug text not null,
  session_id text,
  project_id uuid,
  kind text not null default 'fact'
    check (kind in ('fact', 'decision', 'preference', 'project')),
  content text not null,
  embedding extensions.vector(1024),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

alter table public.consultant_memories
  add column if not exists account_id uuid references public.accounts (id) on delete cascade;
create index if not exists consultant_memories_account_id_idx
  on public.consultant_memories (account_id);
create index if not exists consultant_memories_account_idx
  on public.consultant_memories (account_slug, created_at desc);

comment on column public.consultant_memories.account_id is
  'Preferred tenant key; account_slug still required (temporary Phase 2.5 state).';

alter table public.consultant_profiles enable row level security;
alter table public.consultant_memories enable row level security;
