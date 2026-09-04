-- World Trade Radar — danışman bellek + şirket profili
-- Mevcut organizations (hedef/kaynak firma) tablosunu değiştirmez.
-- SQL Editor'de çalıştırın. Tablo yoksa Python katmanı sessizce bellek-içi yedek kullanır.

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

drop trigger if exists consultant_profiles_set_updated_at on public.consultant_profiles;
create trigger consultant_profiles_set_updated_at
  before update on public.consultant_profiles
  for each row
  execute function public.set_updated_at();

create table if not exists public.consultant_projects (
  id uuid primary key default gen_random_uuid(),
  account_slug text not null,
  title text not null,
  status text not null default 'active'
    check (status in ('active', 'paused', 'done')),
  notes jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists consultant_projects_account_idx
  on public.consultant_projects (account_slug);

drop trigger if exists consultant_projects_set_updated_at on public.consultant_projects;
create trigger consultant_projects_set_updated_at
  before update on public.consultant_projects
  for each row
  execute function public.set_updated_at();

create table if not exists public.consultant_memories (
  id uuid primary key default gen_random_uuid(),
  account_slug text not null,
  session_id text,
  project_id uuid references public.consultant_projects (id) on delete set null,
  kind text not null default 'fact'
    check (kind in ('fact', 'decision', 'preference', 'project')),
  content text not null,
  embedding extensions.vector(1024),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists consultant_memories_account_idx
  on public.consultant_memories (account_slug, created_at desc);

create index if not exists consultant_memories_kind_idx
  on public.consultant_memories (account_slug, kind);

alter table public.consultant_profiles enable row level security;
alter table public.consultant_projects enable row level security;
alter table public.consultant_memories enable row level security;

comment on table public.consultant_profiles is
  'Kullanıcının kendi şirketi. organizations tablosu hedef/kaynak firmadır.';
comment on table public.consultant_memories is
  'Kalıcı ticari bellek. account_slug ile kiracı izolasyonu.';
