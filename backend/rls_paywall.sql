-- API katmanı plan bazlı redaksiyon ile Free kullanıcıya
-- contact_email, website, tax_id, quantity, value_usd göndermez.
-- FastAPI service_role RLS'i atladığı için paywall uygulama katmanındadır.

comment on column public.organizations.contact_email is
  'Hassas: yalnızca Pro planda API yanıtına dahil edilir.';
comment on column public.organizations.website is
  'Hassas: yalnızca Pro planda API yanıtına dahil edilir.';
comment on column public.trade_items.quantity is
  'Hassas hacim: yalnızca Pro planda API yanıtına dahil edilir.';
comment on column public.trade_items.value_usd is
  'Hassas hacim: yalnızca Pro planda API yanıtına dahil edilir.';
