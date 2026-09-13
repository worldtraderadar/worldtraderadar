"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react"
import { useAuth } from "@/components/auth-provider"
import { getBillingMe, type BillingSnapshot } from "@/lib/api"
import { QUOTA_EVENT } from "@/lib/quota"

type PaywallReason = "contact" | "volume"

type BillingContextValue = {
  snapshot: BillingSnapshot | null
  isPro: boolean
  loading: boolean
  refresh: () => Promise<void>
  paywallOpen: boolean
  paywallReason: PaywallReason
  openPaywall: (reason?: PaywallReason) => void
  closePaywall: () => void
}

const BillingContext = createContext<BillingContextValue | null>(null)

export function BillingProvider({ children }: { children: ReactNode }) {
  const { status } = useAuth()
  const [snapshot, setSnapshot] = useState<BillingSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [paywallOpen, setPaywallOpen] = useState(false)
  const [paywallReason, setPaywallReason] = useState<PaywallReason>("contact")

  const refresh = useCallback(async () => {
    if (status !== "authenticated") {
      setSnapshot(null)
      setLoading(false)
      return
    }
    try {
      setSnapshot(await getBillingMe())
    } catch {
      setSnapshot(null)
    } finally {
      setLoading(false)
    }
  }, [status])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refresh()
    }, 0)
    const onChange = () => {
      void refresh()
    }
    window.addEventListener(QUOTA_EVENT, onChange)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener(QUOTA_EVENT, onChange)
    }
  }, [refresh])

  const value = useMemo<BillingContextValue>(
    () => ({
      snapshot,
      isPro: snapshot?.plan_id === "pro",
      loading,
      refresh,
      paywallOpen,
      paywallReason,
      openPaywall: (reason = "contact") => {
        setPaywallReason(reason)
        setPaywallOpen(true)
      },
      closePaywall: () => setPaywallOpen(false),
    }),
    [snapshot, loading, refresh, paywallOpen, paywallReason]
  )

  return (
    <BillingContext.Provider value={value}>{children}</BillingContext.Provider>
  )
}

export function useBilling() {
  const context = useContext(BillingContext)
  if (!context) {
    throw new Error("useBilling BillingProvider içinde kullanılmalı.")
  }
  return context
}
