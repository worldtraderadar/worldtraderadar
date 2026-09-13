"use client"

import type { ReactNode } from "react"
import { AuthGate } from "@/components/auth-gate"
import { AuthProvider } from "@/components/auth-provider"
import { BillingProvider } from "@/components/billing-provider"
import { InstallBanner } from "@/components/install-banner"
import { MobileTabBar } from "@/components/mobile-tab-bar"
import { PaywallModal } from "@/components/paywall-modal"
import { PwaRegister } from "@/components/pwa-register"
import { SiteHeader } from "@/components/site-header"

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <AuthProvider>
      <AuthGate>
        <BillingProvider>
          <PwaRegister />
          <SiteHeader />
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 sm:py-10 pb-[calc(6.25rem+env(safe-area-inset-bottom))] lg:pb-10">
            {children}
          </main>
          <InstallBanner />
          <MobileTabBar />
          <PaywallModal />
        </BillingProvider>
      </AuthGate>
    </AuthProvider>
  )
}
