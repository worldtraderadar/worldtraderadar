-- Supabase SQL Editor'de çalıştırın (agent_runs tablosu).

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

drop trigger if exists agent_runs_set_updated_at on public.agent_runs;
create trigger agent_runs_set_updated_at
  before update on public.agent_runs
  for each row
  execute function public.set_updated_at();

alter table public.agent_runs enable row level security;
