"use client"

import { useEffect, useState } from "react"
import { Check, Crown, LoaderCircle, Shield, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  changePlan,
  confirmCheckout,
  getBillingMe,
  getBillingPlans,
  startCheckout,
  type BillingPlan,
  type BillingSnapshot,
  type CheckoutSession,
} from "@/lib/api"
import { notifyQuotaChanged } from "@/lib/quota"
import { cn } from "@/lib/utils"

export function UpgradeDesk() {
  const [plans, setPlans] = useState<BillingPlan[]>([])
  const [account, setAccount] = useState<BillingSnapshot | null>(null)
  const [checkout, setCheckout] = useState<CheckoutSession | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [paid, setPaid] = useState(false)

  async function refresh() {
    const [planPayload, me] = await Promise.all([
      getBillingPlans(),
      getBillingMe(),
    ])
    setPlans(planPayload.plans)
    setAccount(me)
  }

  useEffect(() => {
    refresh()
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Paketler yüklenemedi.")
      })
      .finally(() => setLoading(false))
  }, [])

  async function beginUpgrade() {
    setBusy(true)
    setError(null)
    setPaid(false)
    try {
      const session = await startCheckout("pro")
      setCheckout(session)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ödeme oturumu açılamadı.")
    } finally {
      setBusy(false)
    }
  }

  async function confirm() {
    if (!checkout) return
    setBusy(true)
    setError(null)
    try {
      const result = await confirmCheckout(checkout.id)
      setCheckout(result.checkout)
      setAccount(result.account)
      setPaid(true)
      notifyQuotaChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ödeme tamamlanamadı.")
    } finally {
      setBusy(false)
    }
  }

  async function downgrade() {
    setBusy(true)
    setError(null)
    try {
      const next = await changePlan("free")
      setAccount(next)
      setCheckout(null)
      setPaid(false)
      notifyQuotaChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Plan düşürülemedi.")
    } finally {
      setBusy(false)
    }
  }

  const current = account?.plan_id ?? "free"

  return (
    <div className="space-y-8">
      <section className="rounded-3xl border border-white/8 bg-[#0b1c27]/80 p-8 sm:p-10">
        <p className="mb-3 text-xs font-medium tracking-[0.22em] text-teal-300/80 uppercase">
          SaaS billing
        </p>
        <h1 className="font-heading max-w-2xl text-3xl leading-tight text-zinc-50 sm:text-5xl">
          Kota ve paket.
        </h1>
        <p className="mt-4 max-w-xl text-base leading-relaxed text-zinc-400">
          Free planda günlük arama ve AI token tavanı vardır. Pro, yüksek hacimli
          tarama ve danışmanlık için kotayı açar.
        </p>
        {account ? (
          <p className="mt-4 font-mono text-xs text-zinc-500">
            Bugün {account.usage.searches}/{account.limits.daily_searches} arama
            · {account.usage.tokens}/{account.limits.daily_tokens} token ·{" "}
            {account.backend === "supabase" ? "kalıcı kota" : "oturum kotası"}
          </p>
        ) : null}
      </section>

      {loading ? (
        <p className="text-sm text-zinc-500">Paketler yükleniyor…</p>
      ) : (
        <div className="grid gap-5 lg:grid-cols-2">
          {plans.map((plan) => {
            const active = plan.id === current
            const isPro = plan.id === "pro"
            return (
              <article
                key={plan.id}
                className={cn(
                  "rounded-3xl border p-6",
                  isPro
                    ? "border-teal-400/30 bg-teal-400/6"
                    : "border-white/8 bg-[#0b1c27]/70"
                )}
              >
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-teal-300">
                    {isPro ? (
                      <Crown className="size-4" />
                    ) : (
                      <Shield className="size-4" />
                    )}
                    <h2 className="font-heading text-2xl text-zinc-50">
                      {plan.name}
                    </h2>
                  </div>
                  {active ? (
                    <span className="rounded-full border border-white/10 px-2 py-0.5 font-mono text-[10px] text-zinc-400 uppercase">
                      Aktif
                    </span>
                  ) : null}
                </div>
                <p className="font-heading text-3xl text-zinc-50">
                  {plan.monthly_price_usd === 0
                    ? "Ücretsiz"
                    : `$${plan.monthly_price_usd}`}
                  {plan.monthly_price_usd > 0 ? (
                    <span className="ml-1 text-sm font-sans text-zinc-500">
                      / ay
                    </span>
                  ) : null}
                </p>
                <ul className="mt-5 space-y-2">
                  {(plan.features || []).map((feature) => (
                    <li
                      key={feature}
                      className="flex items-start gap-2 text-sm text-zinc-300"
                    >
                      <Check className="mt-0.5 size-3.5 shrink-0 text-teal-400" />
                      {feature}
                    </li>
                  ))}
                </ul>
                {isPro && !active ? (
                  <Button
                    className="mt-6 bg-teal-400 text-[#062026] hover:bg-teal-300"
                    onClick={() => void beginUpgrade()}
                    disabled={busy}
                  >
                    {busy && !checkout ? (
                      <LoaderCircle className="animate-spin" />
                    ) : (
                      <Sparkles />
                    )}
                    Pro&apos;ya yükselt
                  </Button>
                ) : null}
                {isPro && active ? (
                  <Button
                    variant="outline"
                    className="mt-6 border-white/10"
                    onClick={() => void downgrade()}
                    disabled={busy}
                  >
                    Free&apos;e dön
                  </Button>
                ) : null}
              </article>
            )
          })}
        </div>
      )}

      {checkout && !paid ? (
        <section className="rounded-3xl border border-teal-400/25 bg-[#0b1c27] p-6">
          <h2 className="font-heading text-xl text-zinc-50">Ödeme onayı</h2>
          <p className="mt-2 text-sm leading-6 text-zinc-400">
            Stripe bağlanmadan dahili checkout: {checkout.plan_name} paketi{" "}
            <span className="text-zinc-200">${checkout.amount_usd}/ay</span>{" "}
            olarak etkinleşir. Kota tavanı {checkout.limits.daily_searches} arama
            ve {checkout.limits.daily_tokens.toLocaleString("tr-TR")} token olur.
          </p>
          <div className="mt-5 flex flex-wrap gap-3">
            <Button
              className="bg-teal-400 text-[#062026] hover:bg-teal-300"
              onClick={() => void confirm()}
              disabled={busy}
            >
              {busy ? <LoaderCircle className="animate-spin" /> : <Crown />}
              Ödemeyi tamamla
            </Button>
            <Button
              variant="ghost"
              onClick={() => setCheckout(null)}
              disabled={busy}
            >
              Vazgeç
            </Button>
          </div>
        </section>
      ) : null}

      {paid ? (
        <p className="rounded-2xl border border-teal-400/20 bg-teal-400/8 px-4 py-3 text-sm text-teal-100">
          Pro paket aktif. Günlük arama ve token kotalarınız yükseltildi.
        </p>
      ) : null}

      {error ? (
        <p className="rounded-2xl border border-amber-400/20 bg-amber-400/8 px-4 py-3 text-sm text-amber-200">
          {error}
        </p>
      ) : null}
    </div>
  )
}
