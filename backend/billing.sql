-- World Trade Radar — Aşama 4: SaaS Billing + Quota Engine
-- Supabase SQL Editor'de çalıştırın.

-- ---------------------------------------------------------------------------
-- plans
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

comment on table public.plans is
  'Free ve Pro kota tavanları (günlük arama + AI token).';

insert into public.plans (
  id, name, daily_search_limit, daily_token_limit, monthly_price_usd, features
) values
  (
    'free',
    'Free',
    8,
    15000,
    0,
    '["Günde 8 arama / danışmanlık","15.000 AI token / gün","Orchestrator + 3 ajan","Şeffaf ajan akışı"]'::jsonb
  ),
  (
    'pro',
    'Pro',
    250,
    500000,
    49,
    '["Günde 250 arama / danışmanlık","500.000 AI token / gün","Öncelikli kota","Tüm ajanlar ve geçmiş raporlar","Paket yükseltme dahil"]'::jsonb
  )
on conflict (id) do update set
  name = excluded.name,
  daily_search_limit = excluded.daily_search_limit,
  daily_token_limit = excluded.daily_token_limit,
  monthly_price_usd = excluded.monthly_price_usd,
  features = excluded.features;

-- ---------------------------------------------------------------------------
-- accounts
-- ---------------------------------------------------------------------------
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

create index if not exists accounts_plan_id_idx on public.accounts (plan_id);

drop trigger if exists accounts_set_updated_at on public.accounts;
create trigger accounts_set_updated_at
  before update on public.accounts
  for each row
  execute function public.set_updated_at();

comment on table public.accounts is
  'SaaS kiracı hesabı. X-Account-Slug başlığı ile çözülür.';

insert into public.accounts (id, slug, display_name, email, plan_id)
values (
  '00000000-0000-4000-8000-000000000001',
  'demo',
  'World Trade Radar Demo',
  'demo@worldtraderadar.local',
  'free'
)
on conflict (slug) do nothing;

-- ---------------------------------------------------------------------------
-- subscriptions
-- ---------------------------------------------------------------------------
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

create index if not exists subscriptions_account_id_idx
  on public.subscriptions (account_id);

drop trigger if exists subscriptions_set_updated_at on public.subscriptions;
create trigger subscriptions_set_updated_at
  before update on public.subscriptions
  for each row
  execute function public.set_updated_at();

-- ---------------------------------------------------------------------------
-- quota_usage — günlük arama + token sayaçları
-- ---------------------------------------------------------------------------
create table if not exists public.quota_usage (
  account_id uuid not null references public.accounts (id) on delete cascade,
  usage_date date not null default ((timezone('utc', now()))::date),
  search_count integer not null default 0 check (search_count >= 0),
  token_count integer not null default 0 check (token_count >= 0),
  updated_at timestamptz not null default now(),
  primary key (account_id, usage_date)
);

create index if not exists quota_usage_date_idx
  on public.quota_usage (usage_date desc);

drop trigger if exists quota_usage_set_updated_at on public.quota_usage;
create trigger quota_usage_set_updated_at
  before update on public.quota_usage
  for each row
  execute function public.set_updated_at();

comment on table public.quota_usage is
  'Hesap başına UTC günü arama ve AI token tüketimi.';

-- ---------------------------------------------------------------------------
-- billing_events
-- ---------------------------------------------------------------------------
create table if not exists public.billing_events (
  id uuid primary key default gen_random_uuid(),
  account_id uuid references public.accounts (id) on delete set null,
  event_type text not null,
  plan_id text,
  amount_usd numeric(10, 2),
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists billing_events_account_id_idx
  on public.billing_events (account_id);

create index if not exists billing_events_created_at_idx
  on public.billing_events (created_at desc);

-- ---------------------------------------------------------------------------
-- Atomik kota tüketimi
-- ---------------------------------------------------------------------------
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

  if v_token + p_tokens > v_plan.daily_token_limit and p_tokens > 0 and v_token >= v_plan.daily_token_limit then
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

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
alter table public.plans enable row level security;
alter table public.accounts enable row level security;
alter table public.subscriptions enable row level security;
alter table public.quota_usage enable row level security;
alter table public.billing_events enable row level security;
