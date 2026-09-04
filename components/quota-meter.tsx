"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Sparkles } from "lucide-react"
import { useBilling } from "@/components/billing-provider"
import { cn } from "@/lib/utils"

function formatTokens(value: number) {
  if (value >= 1000) {
    const kilo = value / 1000
    return `${kilo >= 10 ? Math.round(kilo) : kilo.toFixed(1)}k`
  }
  return String(value)
}

function MeterBar({
  label,
  used,
  limit,
}: {
  label: string
  used: number
  limit: number
}) {
  const ratio = limit <= 0 ? 1 : Math.min(used / limit, 1)
  const tight = ratio >= 0.85
  return (
    <div className="min-w-[4.25rem]">
      <div className="mb-0.5 flex justify-between gap-2 font-mono text-[10px] text-zinc-500">
        <span>{label}</span>
        <span className={tight ? "text-amber-300" : "text-zinc-400"}>
          {label === "Token" ? formatTokens(used) : used}/
          {label === "Token" ? formatTokens(limit) : limit}
        </span>
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-white/8">
        <div
          className={cn(
            "h-full rounded-full",
            tight ? "bg-amber-400" : "bg-teal-400"
          )}
          style={{ width: `${Math.max(ratio * 100, used > 0 ? 6 : 0)}%` }}
        />
      </div>
    </div>
  )
}

export function QuotaMeter() {
  const pathname = usePathname()
  const { snapshot } = useBilling()
  if (!snapshot) return null

  const isPro = snapshot.plan_id === "pro"
  const exhausted =
    snapshot.remaining.searches <= 0 || snapshot.remaining.tokens <= 0

  return (
    <div
      className="flex items-center gap-2 sm:gap-3"
      aria-label={`Kota ${snapshot.plan_name}, arama ${snapshot.usage.searches}/${snapshot.limits.daily_searches}, token ${snapshot.usage.tokens}/${snapshot.limits.daily_tokens}`}
    >
      <div className="hidden items-center gap-3 md:flex">
        <MeterBar
          label="Arama"
          used={snapshot.usage.searches}
          limit={snapshot.limits.daily_searches}
        />
        <MeterBar
          label="Token"
          used={snapshot.usage.tokens}
          limit={snapshot.limits.daily_tokens}
        />
      </div>
      <span
        className={cn(
          "rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase",
          isPro
            ? "border-teal-400/30 bg-teal-400/10 text-teal-200"
            : "border-white/10 bg-white/4 text-zinc-400"
        )}
      >
        {snapshot.plan_name}
      </span>
      {pathname !== "/paket" ? (
        <Link
          href="/paket"
          className={cn(
            "inline-flex min-h-9 items-center gap-1 rounded-md px-2.5 py-1 text-xs transition",
            exhausted || !isPro
              ? "bg-teal-400 text-[#062026] hover:bg-teal-300"
              : "text-zinc-400 hover:text-zinc-200"
          )}
        >
          <Sparkles className="size-3" />
          {isPro ? "Paket" : "Yükselt"}
        </Link>
      ) : null}
    </div>
  )
}
