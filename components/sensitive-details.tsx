"use client"

import type { Organization, PaywallMeta } from "@/lib/api"
import { LockedValue } from "@/components/locked-value"

function formatUsd(value: number | null | undefined) {
  if (value == null) return "Kayıt yok"
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value)
}

function formatQty(quantity?: number | null, unit?: string | null) {
  if (quantity == null) return "Kayıt yok"
  const amount = new Intl.NumberFormat("tr-TR").format(quantity)
  return unit ? `${amount} ${unit}` : amount
}

export function SensitiveDetails({
  item,
  compact = false,
}: {
  item: PaywallMeta
  compact?: boolean
}) {
  const locked = item.locked
  return (
    <div
      className={
        compact
          ? "mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2"
          : "mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2"
      }
    >
      <LockedValue
        reason="contact"
        locked={locked?.contact}
        label="İletişim"
        placeholder="ops••••@firma.trade"
      >
        {item.contact_email || item.website || "Kayıt yok"}
      </LockedValue>
      <LockedValue
        reason="volume"
        locked={locked?.volume}
        label="Hacim"
        placeholder="$ ••••••  ·  •••• kg"
      >
        {`${formatQty(item.quantity, item.unit)} · ${formatUsd(item.value_usd)}`}
      </LockedValue>
    </div>
  )
}

export function OrganizationCard({ org }: { org: Organization }) {
  return (
    <article className="rounded-2xl border border-white/8 bg-[#0b1c27]/70 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-zinc-100">{org.name}</p>
          <p className="mt-1 font-mono text-xs text-zinc-500">
            {(org.country_code || "?").toUpperCase()}
            {org.city ? ` · ${org.city}` : ""}
            {org.organization_type ? ` · ${org.organization_type}` : ""}
          </p>
        </div>
      </div>
      <SensitiveDetails item={org} />
    </article>
  )
}
