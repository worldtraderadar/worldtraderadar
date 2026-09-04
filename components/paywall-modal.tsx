"use client"

import { useEffect } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Crown, Lock, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useBilling } from "@/components/billing-provider"

const COPY = {
  contact: {
    title: "Firma iletişim bilgisi kilitli",
    body: "E-posta, web sitesi ve vergi kimliği Free planda gizlenir. Pro ile tedarikçi ve firma dizinini açın.",
  },
  volume: {
    title: "Hacim detayları kilitli",
    body: "Miktar ve USD tutarı Free planda bulanık gösterilir. Gerçek sevkiyat hacmine Pro ile ulaşın.",
  },
}

export function PaywallModal() {
  const pathname = usePathname()
  const { paywallOpen, paywallReason, closePaywall } = useBilling()

  useEffect(() => {
    if (pathname === "/paket") closePaywall()
  }, [pathname, closePaywall])

  useEffect(() => {
    if (!paywallOpen) return
    const previous = document.body.style.overflow
    document.body.style.overflow = "hidden"
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") closePaywall()
    }
    window.addEventListener("keydown", onKey)
    return () => {
      document.body.style.overflow = previous
      window.removeEventListener("keydown", onKey)
    }
  }, [paywallOpen, closePaywall])

  if (!paywallOpen) return null
  const copy = COPY[paywallReason]

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/65 p-4 pb-[calc(1rem+env(safe-area-inset-bottom))] sm:items-center"
      role="dialog"
      aria-modal="true"
      aria-labelledby="paywall-title"
      onClick={closePaywall}
    >
      <div
        className="w-full max-w-md rounded-t-3xl border border-teal-400/25 bg-[#0b1c27] p-6 shadow-[0_-20px_80px_rgba(13,148,136,0.18)] sm:rounded-3xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <span className="flex size-10 items-center justify-center rounded-xl border border-teal-400/30 bg-teal-400/10 text-teal-300">
            <Lock className="size-4" />
          </span>
          <button
            type="button"
            onClick={closePaywall}
            className="flex size-10 items-center justify-center rounded-full text-zinc-500 hover:text-zinc-200"
            aria-label="Kapat"
          >
            <X className="size-4" />
          </button>
        </div>
        <h2 id="paywall-title" className="font-heading text-2xl text-zinc-50">
          {copy.title}
        </h2>
        <p className="mt-2 text-sm leading-6 text-zinc-400">{copy.body}</p>
        <div className="mt-6 flex flex-col gap-2 sm:flex-row">
          <Link
            href="/paket"
            className="inline-flex h-11 flex-1 items-center justify-center gap-1.5 rounded-lg bg-teal-400 text-sm font-medium text-[#062026] hover:bg-teal-300"
          >
            <Crown className="size-4" />
            Pro&apos;ya yükselt
          </Link>
          <Button variant="ghost" className="h-11" onClick={closePaywall}>
            Şimdi değil
          </Button>
        </div>
      </div>
    </div>
  )
}
