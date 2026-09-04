"use client"

import { useEffect, useState } from "react"
import { Building2, LoaderCircle } from "lucide-react"
import { OrganizationCard, SensitiveDetails } from "@/components/sensitive-details"
import { listSuppliers, type Organization, type TradeItem } from "@/lib/api"

export function SuppliersDesk() {
  const [items, setItems] = useState<TradeItem[]>([])
  const [organizations, setOrganizations] = useState<Organization[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listSuppliers(50)
      .then((data) => {
        setItems(data.items)
        setOrganizations(data.organizations)
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Tedarikçiler yüklenemedi.")
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="space-y-8">
      <header className="space-y-3">
        <p className="text-xs font-medium tracking-[0.22em] text-teal-300/80 uppercase">
          Tedarik ağı
        </p>
        <h1 className="font-heading text-3xl text-zinc-50 sm:text-4xl">
          Tedarikçi ve firmalar
        </h1>
        <p className="max-w-2xl text-sm leading-6 text-zinc-400">
          Ürün, rota ve firma özeti herkese açık. İletişim ve hacim Free planda
          kilitli; dokununca Pro paywall açılır.
        </p>
      </header>

      {error ? (
        <p className="rounded-xl border border-amber-400/20 bg-amber-400/8 px-3 py-2 text-sm text-amber-200">
          {error}
        </p>
      ) : null}

      {organizations.length > 0 ? (
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-xs tracking-[0.18em] text-zinc-500 uppercase">
            <Building2 className="size-3.5" />
            Firma dizini
          </h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {organizations.map((org) => (
              <OrganizationCard key={org.id} org={org} />
            ))}
          </div>
        </section>
      ) : null}

      <section className="space-y-3">
        <h2 className="text-xs tracking-[0.18em] text-zinc-500 uppercase">
          Tedarikçi kalemleri
        </h2>
        {loading ? (
          <p className="flex items-center gap-2 text-sm text-zinc-500">
            <LoaderCircle className="size-4 animate-spin" />
            Yükleniyor…
          </p>
        ) : items.length === 0 ? (
          <p className="rounded-2xl border border-white/8 p-6 text-sm text-zinc-500">
            Henüz tedarikçi kalemi yok.
          </p>
        ) : (
          <div className="grid gap-3">
            {items.map((item) => (
              <article
                key={item.id}
                className="rounded-2xl border border-white/8 bg-[#0b1c27]/70 p-4"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-zinc-100">
                      {item.organization_name || item.product_name}
                    </p>
                    <p className="mt-1 text-xs text-zinc-500">
                      {item.product_name}
                    </p>
                  </div>
                  <span className="rounded-full border border-white/8 px-2 py-0.5 text-[11px] text-zinc-400">
                    {item.direction === "import" ? "ithalat" : "ihracat"}
                  </span>
                </div>
                <p className="mt-2 font-mono text-xs text-zinc-500">
                  HS {item.hs_code || "—"} · {(item.origin_country || "?").trim()} →{" "}
                  {(item.destination_country || "?").trim()}
                </p>
                <SensitiveDetails item={item} />
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
