-- World Trade Radar — Supabase şeması
-- SQL Editor'de çalıştırın. BGE-M3 dense embedding boyutu: 1024.

-- ---------------------------------------------------------------------------
-- Eklentiler
-- ---------------------------------------------------------------------------
create extension if not exists vector with schema extensions;
create extension if not exists pgcrypto with schema extensions;

-- ---------------------------------------------------------------------------
-- Ortak: updated_at tetikleyicisi
-- ---------------------------------------------------------------------------
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
-- organizations
-- ---------------------------------------------------------------------------
create table if not exists public.organizations (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  slug text unique,
  country_code char(2),
  city text,
  organization_type text not null default 'unknown'
    check (organization_type in (
      'importer',
      'exporter',
      'trader',
      'broker',
      'logistics',
      'unknown'
    )),
  tax_id text,
  website text,
  contact_email text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists organizations_country_code_idx
  on public.organizations (country_code);

create index if not exists organizations_type_idx
  on public.organizations (organization_type);

create trigger organizations_set_updated_at
  before update on public.organizations
  for each row
  execute function public.set_updated_at();

comment on table public.organizations is
  'İthalatçı, ihracatçı ve diğer ticaret kuruluşları.';

-- ---------------------------------------------------------------------------
-- trade_items
-- BGE-M3 dense vektör: vector(1024), kosinüs benzerliği
-- ---------------------------------------------------------------------------
create table if not exists public.trade_items (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid references public.organizations (id) on delete set null,
  hs_code text,
  product_name text not null,
  description text,
  origin_country char(2),
  destination_country char(2),
  direction text
    check (direction is null or direction in ('import', 'export')),
  quantity numeric(18, 4),
  unit text,
  value_usd numeric(18, 2),
  trade_date date,
  source text,
  embedding extensions.vector(1024),
  embedding_text text,
  embedding_model text not null default 'bge-m3',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists trade_items_organization_id_idx
  on public.trade_items (organization_id);

create index if not exists trade_items_hs_code_idx
  on public.trade_items (hs_code);

create index if not exists trade_items_trade_date_idx
  on public.trade_items (trade_date desc);

create index if not exists trade_items_origin_destination_idx
  on public.trade_items (origin_country, destination_country);

create index if not exists trade_items_embedding_hnsw_idx
  on public.trade_items
  using hnsw (embedding extensions.vector_cosine_ops);

create trigger trade_items_set_updated_at
  before update on public.trade_items
  for each row
  execute function public.set_updated_at();

comment on table public.trade_items is
  'Ticaret kalemleri. embedding alanı BAAI/bge-m3 (1024 boyut) dense vektörüdür.';
comment on column public.trade_items.embedding is
  'BGE-M3 dense embedding, 1024 boyut. Semantik arama için kosinüs mesafesi (<=>) kullanın.';
comment on column public.trade_items.embedding_text is
  'Vektöre dönüştürülen ham metin (ürün adı + açıklama + HS kodu).';

-- Semantik arama RPC: 1 - (embedding <=> query) = kosinüs benzerliği
create or replace function public.match_trade_items(
  query_embedding extensions.vector(1024),
  match_count integer default 10,
  match_threshold double precision default 0.5,
  filter_organization_id uuid default null,
  filter_query text default null
)
returns table (
  id uuid,
  organization_id uuid,
  hs_code text,
  product_name text,
  description text,
  origin_country char(2),
  destination_country char(2),
  similarity double precision
)
language sql
stable
as $$
  select
    ti.id,
    ti.organization_id,
    ti.hs_code,
    ti.product_name,
    ti.description,
    ti.origin_country,
    ti.destination_country,
    (1 - (ti.embedding <=> query_embedding))::double precision as similarity
  from public.trade_items ti
  where ti.embedding is not null
    and (filter_organization_id is null or ti.organization_id = filter_organization_id)
    and 1 - (ti.embedding <=> query_embedding) >= match_threshold
    and (
      filter_query is null
      or length(trim(filter_query)) = 0
      or ti.product_name ilike '%' || trim(filter_query) || '%'
      or coalesce(ti.description, '') ilike '%' || trim(filter_query) || '%'
      or coalesce(ti.embedding_text, '') ilike '%' || trim(filter_query) || '%'
      or coalesce(ti.hs_code, '') ilike '%' || trim(filter_query) || '%'
    )
  order by ti.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

-- ---------------------------------------------------------------------------
-- agent_logs
-- ---------------------------------------------------------------------------
create table if not exists public.agent_logs (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid references public.organizations (id) on delete set null,
  trade_item_id uuid references public.trade_items (id) on delete set null,
  agent_name text not null,
  action text not null,
  status text not null default 'pending'
    check (status in ('pending', 'running', 'success', 'error')),
  input jsonb not null default '{}'::jsonb,
  output jsonb not null default '{}'::jsonb,
  error_message text,
  duration_ms integer,
  created_at timestamptz not null default now()
);

create index if not exists agent_logs_organization_id_idx
  on public.agent_logs (organization_id);

create index if not exists agent_logs_trade_item_id_idx
  on public.agent_logs (trade_item_id);

create index if not exists agent_logs_agent_name_idx
  on public.agent_logs (agent_name);

create index if not exists agent_logs_status_idx
  on public.agent_logs (status);

create index if not exists agent_logs_created_at_idx
  on public.agent_logs (created_at desc);

comment on table public.agent_logs is
  'Ajan (ingest, embedding, tarama) çalışma kayıtları.';

-- ---------------------------------------------------------------------------
-- agent_runs — orchestrator adım günlüğü
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

create trigger agent_runs_set_updated_at
  before update on public.agent_runs
  for each row
  execute function public.set_updated_at();

comment on table public.agent_runs is
  'Orchestrator çalıştırmaları ve şeffaf ajan adımları.';

-- ---------------------------------------------------------------------------
-- Row Level Security
-- FastAPI service_role anahtarı RLS'i atlar.
-- Anon/authenticated istemciler varsayılan olarak erişemez.
-- ---------------------------------------------------------------------------
alter table public.organizations enable row level security;
alter table public.trade_items enable row level security;
alter table public.agent_logs enable row level security;
alter table public.agent_runs enable row level security;

-- ---------------------------------------------------------------------------
-- Aşama 4: SaaS Billing + Quota Engine
-- Tam DDL: backend/billing.sql
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

drop trigger if exists plans_set_updated_at on public.plans;
create trigger plans_set_updated_at
  before update on public.plans
  for each row
  execute function public.set_updated_at();

insert into public.plans (
  id, name, daily_search_limit, daily_token_limit, monthly_price_usd, features
) values
  (
    'free', 'Free', 8, 15000, 0,
    '["Günde 8 arama / danışmanlık","15.000 AI token / gün","Orchestrator + 3 ajan","Şeffaf ajan akışı"]'::jsonb
  ),
  (
    'pro', 'Pro', 250, 500000, 49,
    '["Günde 250 arama / danışmanlık","500.000 AI token / gün","Öncelikli kota","Tüm ajanlar ve geçmiş raporlar","Paket yükseltme dahil"]'::jsonb
  )
on conflict (id) do update set
  name = excluded.name,
  daily_search_limit = excluded.daily_search_limit,
  daily_token_limit = excluded.daily_token_limit,
  monthly_price_usd = excluded.monthly_price_usd,
  features = excluded.features;

create table if not exists public.accounts (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  display_name text not null default 'Demo hesap',
  email text,
  plan_id text not null default 'free' references public.plans (id),
  plan_started_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

drop trigger if exists accounts_set_updated_at on public.accounts;
create trigger accounts_set_updated_at
  before update on public.accounts
  for each row
  execute function public.set_updated_at();

insert into public.accounts (id, slug, display_name, email, plan_id)
values (
  '00000000-0000-4000-8000-000000000001',
  'demo',
  'World Trade Radar Demo',
  'demo@worldtraderadar.local',
  'free'
)
on conflict (slug) do nothing;

create table if not exists public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  account_id uuid not null references public.accounts (id) on delete cascade,
  plan_id text not null references public.plans (id),
  status text not null default 'active'
    check (status in ('trialing', 'active', 'canceled', 'past_due')),
  provider text not null default 'internal',
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

drop trigger if exists subscriptions_set_updated_at on public.subscriptions;
create trigger subscriptions_set_updated_at
  before update on public.subscriptions
  for each row
  execute function public.set_updated_at();

create table if not exists public.quota_usage (
  account_id uuid not null references public.accounts (id) on delete cascade,
  usage_date date not null default ((timezone('utc', now()))::date),
  search_count integer not null default 0 check (search_count >= 0),
  token_count integer not null default 0 check (token_count >= 0),
  updated_at timestamptz not null default now(),
  primary key (account_id, usage_date)
);

drop trigger if exists quota_usage_set_updated_at on public.quota_usage;
create trigger quota_usage_set_updated_at
  before update on public.quota_usage
  for each row
  execute function public.set_updated_at();

create table if not exists public.billing_events (
  id uuid primary key default gen_random_uuid(),
  account_id uuid references public.accounts (id) on delete set null,
  event_type text not null,
  plan_id text,
  amount_usd numeric(10, 2),
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create or replace function public.consume_quota(
  p_account_id uuid,
  p_searches integer default 0,
  p_tokens integer default 0
)
returns jsonb
language plpgsql
as $$
declare
  v_plan public.plans%rowtype;
  v_account public.accounts%rowtype;
  v_today date := (timezone('utc', now()))::date;
  v_search integer;
  v_token integer;
begin
  if p_searches < 0 or p_tokens < 0 then
    raise exception 'kota artışı negatif olamaz';
  end if;

  select * into v_account from public.accounts where id = p_account_id;
  if not found then
    raise exception 'hesap bulunamadı';
  end if;

  select * into v_plan from public.plans where id = v_account.plan_id;
  if not found then
    raise exception 'plan bulunamadı';
  end if;

  insert into public.quota_usage (account_id, usage_date, search_count, token_count)
  values (p_account_id, v_today, 0, 0)
  on conflict (account_id, usage_date) do nothing;

  select search_count, token_count
    into v_search, v_token
  from public.quota_usage
  where account_id = p_account_id and usage_date = v_today
  for update;

  if v_search + p_searches > v_plan.daily_search_limit then
    return jsonb_build_object(
      'allowed', false,
      'reason', 'search_limit',
      'search_count', v_search,
      'token_count', v_token,
      'daily_search_limit', v_plan.daily_search_limit,
      'daily_token_limit', v_plan.daily_token_limit,
      'plan_id', v_plan.id
    );
  end if;

  if v_token >= v_plan.daily_token_limit and p_tokens > 0 then
    return jsonb_build_object(
      'allowed', false,
      'reason', 'token_limit',
      'search_count', v_search,
      'token_count', v_token,
      'daily_search_limit', v_plan.daily_search_limit,
      'daily_token_limit', v_plan.daily_token_limit,
      'plan_id', v_plan.id
    );
  end if;

  update public.quota_usage
  set
    search_count = search_count + p_searches,
    token_count = token_count + p_tokens
  where account_id = p_account_id and usage_date = v_today
  returning search_count, token_count into v_search, v_token;

  return jsonb_build_object(
    'allowed', true,
    'reason', null,
    'search_count', v_search,
    'token_count', v_token,
    'daily_search_limit', v_plan.daily_search_limit,
    'daily_token_limit', v_plan.daily_token_limit,
    'plan_id', v_plan.id
  );
end;
$$;

alter table public.plans enable row level security;
alter table public.accounts enable row level security;
alter table public.subscriptions enable row level security;
alter table public.quota_usage enable row level security;

-- Danışman bellek + kullanıcı şirket profili: backend/schema_memory.sql
-- (organizations hedef/kaynak firmadır; consultant_profiles kullanıcının kendi şirketidir.)
alter table public.billing_events enable row level security;
