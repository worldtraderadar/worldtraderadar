"use client"

import type { ReactNode } from "react"
import { Lock } from "lucide-react"
import { useBilling } from "@/components/billing-provider"
import { cn } from "@/lib/utils"

type Reason = "contact" | "volume"

export function LockedValue({
  reason,
  locked,
  label,
  children,
  placeholder,
}: {
  reason: Reason
  locked?: boolean
  label: string
  children?: ReactNode
  placeholder: string
}) {
  const { isPro, openPaywall } = useBilling()
  const sealed = locked ?? !isPro

  if (!sealed) {
    return (
      <div className="min-w-0">
        <p className="text-[10px] tracking-[0.14em] text-zinc-500 uppercase">
          {label}
        </p>
        <p className="mt-0.5 truncate text-sm text-zinc-200">{children}</p>
      </div>
    )
  }

  return (
    <button
      type="button"
      onClick={() => openPaywall(reason)}
      className="group min-h-11 min-w-0 rounded-xl border border-white/8 bg-white/3 px-3 py-2 text-left transition hover:border-teal-400/30 active:border-teal-400/40"
    >
      <p className="flex items-center gap-1 text-[10px] tracking-[0.14em] text-zinc-500 uppercase">
        <Lock className="size-2.5 text-teal-300" />
        {label}
      </p>
      <p
        className={cn(
          "mt-0.5 truncate text-sm text-zinc-400 select-none",
          "blur-[5px] group-hover:blur-[3px]"
        )}
        aria-hidden
      >
        {placeholder}
      </p>
      <span className="sr-only">Pro ile {label} bilgisini aç</span>
    </button>
  )
}
