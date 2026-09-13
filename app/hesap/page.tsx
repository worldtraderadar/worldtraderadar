"use client"

import Link from "next/link"
import { Building2, LogOut } from "lucide-react"
import { useAuth } from "@/components/auth-provider"
import { Button } from "@/components/ui/button"

export default function AccountPage() {
  const { account, user, signOut, status } = useAuth()

  if (status !== "authenticated") {
    return null
  }

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-teal-300/80">
          Hesabım
        </p>
        <h1 className="font-heading text-3xl text-zinc-50">Şirket hesabı</h1>
        <p className="text-sm text-zinc-400">
          Kimlik ve plan bilgisi JWT → membership üzerinden gelir. İstemci
          account seçemez. Profil düzenleme ve görsel yükleme sonraki fazlarda
          açılacak.
        </p>
      </div>

      <section className="space-y-3 rounded-xl border border-white/10 bg-white/[0.03] p-5">
        <div className="flex items-center gap-2 text-zinc-200">
          <Building2 className="size-4 text-teal-300" />
          <h2 className="text-sm font-medium">Account context</h2>
        </div>
        <dl className="grid gap-3 text-sm sm:grid-cols-2">
          <div>
            <dt className="text-zinc-500">E-posta</dt>
            <dd className="text-zinc-200">{user?.email || "—"}</dd>
          </div>
          <div>
            <dt className="text-zinc-500">Plan</dt>
            <dd className="uppercase text-zinc-200">
              {account?.plan_id || "—"}
            </dd>
          </div>
          <div>
            <dt className="text-zinc-500">Account slug</dt>
            <dd className="font-mono text-xs text-zinc-300">
              {account?.account_slug || "—"}
            </dd>
          </div>
          <div>
            <dt className="text-zinc-500">Rol</dt>
            <dd className="text-zinc-200">{account?.role || "—"}</dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-zinc-500">Account ID</dt>
            <dd className="break-all font-mono text-xs text-zinc-400">
              {account?.account_id || "—"}
            </dd>
          </div>
        </dl>
      </section>

      <div className="flex flex-wrap gap-3">
        <Link
          href="/"
          className="inline-flex h-8 items-center rounded-lg border border-white/15 px-3 text-sm text-zinc-200 hover:bg-white/5"
        >
          Danışmanlığa dön
        </Link>
        <Button
          variant="ghost"
          onClick={() => {
            void signOut()
          }}
        >
          <LogOut className="size-4" />
          Çıkış
        </Button>
      </div>
    </div>
  )
}
