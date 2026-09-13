"use client"

import { FormEvent, useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { LoaderCircle } from "lucide-react"
import { useAuth } from "@/components/auth-provider"
import { Button } from "@/components/ui/button"

export default function LoginPage() {
  const { signIn, status, configured } = useAuth()
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (status === "authenticated") {
      router.replace("/")
    }
  }, [status, router])

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    setError(null)
    setBusy(true)
    try {
      await signIn(email.trim(), password)
      router.replace("/")
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Giriş başarısız. Davet hesabınızı kontrol edin."
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto flex min-h-[70vh] max-w-md flex-col justify-center gap-6">
      <div className="space-y-2">
        <h1 className="font-heading text-3xl tracking-tight text-zinc-50">
          Giriş
        </h1>
        <p className="text-sm text-zinc-400">
          Pilot davet hesabınızla giriş yapın. Açık kayıt kapalıdır.
        </p>
      </div>

      {!configured ? (
        <p className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
          Supabase Auth env eksik: NEXT_PUBLIC_SUPABASE_URL ve
          NEXT_PUBLIC_SUPABASE_ANON_KEY gerekli.
        </p>
      ) : null}

      <form onSubmit={onSubmit} className="space-y-4">
        <label className="block space-y-1.5 text-sm">
          <span className="text-zinc-300">E-posta</span>
          <input
            type="email"
            autoComplete="username"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-md border border-white/10 bg-black/30 px-3 py-2 text-zinc-50 outline-none ring-teal-400/40 focus:ring"
          />
        </label>
        <label className="block space-y-1.5 text-sm">
          <span className="text-zinc-300">Şifre</span>
          <input
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-md border border-white/10 bg-black/30 px-3 py-2 text-zinc-50 outline-none ring-teal-400/40 focus:ring"
          />
        </label>
        {error ? (
          <p className="text-sm text-rose-300" role="alert">
            {error}
          </p>
        ) : null}
        <Button type="submit" disabled={busy || !configured} className="w-full">
          {busy ? (
            <>
              <LoaderCircle className="size-4 animate-spin" />
              Giriş yapılıyor
            </>
          ) : (
            "Giriş yap"
          )}
        </Button>
      </form>
    </div>
  )
}
