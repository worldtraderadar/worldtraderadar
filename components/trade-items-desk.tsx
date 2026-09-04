"use client"

import { useEffect, useState, type FormEvent } from "react"
import Link from "next/link"
import { LoaderCircle, Plus, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { SensitiveDetails } from "@/components/sensitive-details"
import {
  createTradeItem,
  listTradeItems,
  QuotaError,
  searchTradeItems,
  type TradeItem,
  type TradeItemCreate,
  type TradeItemMatch,
} from "@/lib/api"
import { notifyQuotaChanged } from "@/lib/quota"

const EMPTY: TradeItemCreate = {
  product_name: "",
  hs_code: "",
  origin_country: "",
  destination_country: "",
  direction: "export",
  description: "",
}

export function TradeItemsDesk() {
  const [items, setItems] = useState<TradeItem[]>([])
  const [form, setForm] = useState<TradeItemCreate>(EMPTY)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [quotaBlocked, setQuotaBlocked] = useState(false)
  const [query, setQuery] = useState("")
  const [searching, setSearching] = useState(false)
  const [matches, setMatches] = useState<TradeItemMatch[] | null>(null)

  async function refresh() {
    const data = await listTradeItems()
    setItems(data)
  }

  useEffect(() => {
    refresh()
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Kalemler yüklenemedi.")
      })
      .finally(() => setLoading(false))
  }, [])

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    if (!form.product_name.trim()) return
    setSaving(true)
    setError(null)
    try {
      const payload: TradeItemCreate = {
        product_name: form.product_name.trim(),
        direction: form.direction,
      }
      if (form.hs_code?.trim()) payload.hs_code = form.hs_code.trim()
      if (form.description?.trim()) payload.description = form.description.trim()
      if (form.origin_country?.trim()) {
        payload.origin_country = form.origin_country.trim().toUpperCase()
      }
      if (form.destination_country?.trim()) {
        payload.destination_country = form.destination_country
          .trim()
          .toUpperCase()
      }
      await createTradeItem(payload)
      setForm(EMPTY)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kayıt başarısız.")
    } finally {
      setSaving(false)
    }
  }

  async function onSearch(event: FormEvent) {
    event.preventDefault()
    const next = query.trim()
    if (!next) return
    setSearching(true)
    setError(null)
    setQuotaBlocked(false)
    try {
      const data = await searchTradeItems(next)
      setMatches(data.results)
      notifyQuotaChanged()
    } catch (err) {
      if (err instanceof QuotaError) {
        setQuotaBlocked(true)
        setError(err.message)
      } else {
        setError(err instanceof Error ? err.message : "Arama başarısız.")
      }
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
      <form
        onSubmit={onSubmit}
        className="h-fit space-y-4 rounded-3xl border border-white/8 bg-[#0b1c27]/80 p-6"
      >
        <h1 className="font-heading text-2xl text-zinc-50">Yeni ticaret kalemi</h1>
        <p className="text-sm text-zinc-500">
          Kayıt sırasında Ollama bge-m3 otomatik embedding üretir.
        </p>
        <Field
          label="Ürün adı"
          value={form.product_name}
          onChange={(value) => setForm((prev) => ({ ...prev, product_name: value }))}
        />
        <Field
          label="HS kodu"
          value={form.hs_code ?? ""}
          onChange={(value) => setForm((prev) => ({ ...prev, hs_code: value }))}
        />
        <div className="grid grid-cols-2 gap-3">
          <Field
            label="Menşe (ISO2)"
            value={form.origin_country ?? ""}
            onChange={(value) =>
              setForm((prev) => ({ ...prev, origin_country: value }))
            }
          />
          <Field
            label="Hedef (ISO2)"
            value={form.destination_country ?? ""}
            onChange={(value) =>
              setForm((prev) => ({ ...prev, destination_country: value }))
            }
          />
        </div>
        <label className="block space-y-1.5 text-xs text-zinc-500">
          Yön
          <select
            value={form.direction}
            onChange={(event) =>
              setForm((prev) => ({
                ...prev,
                direction: event.target.value as "import" | "export",
              }))
            }
            className="h-10 w-full rounded-xl border border-white/10 bg-[#061018] px-3 text-sm text-zinc-100 outline-none"
          >
            <option value="export">İhracat</option>
            <option value="import">İthalat</option>
          </select>
        </label>
        <label className="block space-y-1.5 text-xs text-zinc-500">
          Açıklama
          <textarea
            value={form.description ?? ""}
            onChange={(event) =>
              setForm((prev) => ({ ...prev, description: event.target.value }))
            }
            rows={3}
            className="w-full resize-none rounded-xl border border-white/10 bg-[#061018] px-3 py-2 text-sm text-zinc-100 outline-none"
          />
        </label>
        {error ? (
          <div className="text-sm text-amber-200">
            <p>{error}</p>
            {quotaBlocked ? (
              <Link href="/paket" className="mt-1 inline-block text-teal-300 hover:underline">
                Pro pakete yükselt →
              </Link>
            ) : null}
          </div>
        ) : null}
        <Button
          type="submit"
          disabled={saving || !form.product_name.trim()}
          className="bg-teal-400 text-[#062026] hover:bg-teal-300"
        >
          {saving ? <LoaderCircle className="animate-spin" /> : <Plus />}
          {saving ? "Gömülüyor…" : "Kalemi kaydet"}
        </Button>
      </form>

      <section className="space-y-3">
        <h2 className="font-heading text-xl text-zinc-50">Kayıtlı kalemler</h2>
        <form onSubmit={onSearch} className="flex flex-col gap-2 sm:flex-row">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Semantik arama (kota sayılır)…"
            className="h-10 flex-1 rounded-xl border border-white/10 bg-[#061018] px-3 text-sm text-zinc-100 outline-none focus:border-teal-400/40"
          />
          <Button
            type="submit"
            disabled={searching || !query.trim()}
            className="bg-teal-400 text-[#062026] hover:bg-teal-300"
          >
            {searching ? <LoaderCircle className="animate-spin" /> : <Search />}
            Ara
          </Button>
        </form>
        {matches ? (
          <p className="text-xs text-zinc-500">
            {matches.length} semantik sonuç (bu arama günlük kotadan düşer).
          </p>
        ) : null}
        {matches && matches.length > 0 ? (
          <div className="space-y-2">
            {matches.map((item) => (
              <article
                key={item.id}
                className="rounded-2xl border border-teal-400/15 bg-[#0b1c27]/70 p-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm font-medium text-zinc-100">
                    {item.product_name}
                  </p>
                  <span className="font-mono text-[11px] text-teal-300">
                    {Math.round(item.similarity * 100)}%
                  </span>
                </div>
                <p className="mt-2 font-mono text-xs text-zinc-500">
                  HS {item.hs_code || "—"} · {(item.origin_country || "?").trim()} →{" "}
                  {(item.destination_country || "?").trim()}
                </p>
                <SensitiveDetails item={item} compact />
              </article>
            ))}
          </div>
        ) : null}
        {loading ? (
          <p className="text-sm text-zinc-500">Yükleniyor…</p>
        ) : items.length === 0 ? (
          <p className="rounded-2xl border border-white/8 p-6 text-sm text-zinc-500">
            Henüz ticaret kalemi yok. Soldan ekleyin; danışmanlık bu veriyi
            bağlam olarak kullanır.
          </p>
        ) : (
          items.map((item) => (
            <article
              key={item.id}
              className="rounded-2xl border border-white/8 bg-[#0b1c27]/70 p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm font-medium text-zinc-100">
                  {item.product_name}
                </p>
                <span className="rounded-full border border-white/8 px-2 py-0.5 text-[11px] text-zinc-400">
                  {item.direction === "import" ? "ithalat" : "ihracat"}
                </span>
              </div>
              <p className="mt-2 font-mono text-xs text-zinc-500">
                HS {item.hs_code || "—"} · {(item.origin_country || "?").trim()} →{" "}
                {(item.destination_country || "?").trim()}
              </p>
              <SensitiveDetails item={item} compact />
            </article>
          ))
        )}
      </section>
    </div>
  )
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string
  value: string
  onChange: (value: string) => void
}) {
  return (
    <label className="block space-y-1.5 text-xs text-zinc-500">
      {label}
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-10 w-full rounded-xl border border-white/10 bg-[#061018] px-3 text-sm text-zinc-100 outline-none focus:border-teal-400/40"
      />
    </label>
  )
}
