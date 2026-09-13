"use client"

import { useEffect, type ReactNode } from "react"
import { usePathname, useRouter } from "next/navigation"
import { LoaderCircle } from "lucide-react"
import { useAuth } from "@/components/auth-provider"

const PUBLIC_PATHS = new Set(["/login"])

export function AuthGate({ children }: { children: ReactNode }) {
  const { status } = useAuth()
  const pathname = usePathname()
  const router = useRouter()
  const isPublic = PUBLIC_PATHS.has(pathname)

  useEffect(() => {
    if (status === "loading") return
    if (!isPublic && status === "unauthenticated") {
      router.replace("/login")
    }
    if (isPublic && status === "authenticated") {
      router.replace("/")
    }
  }, [status, isPublic, router])

  if (status === "loading") {
    return (
      <div className="flex min-h-[50vh] items-center justify-center gap-2 text-sm text-zinc-400">
        <LoaderCircle className="size-4 animate-spin" />
        Oturum kontrol ediliyor…
      </div>
    )
  }

  if (!isPublic && status !== "authenticated") {
    return (
      <div className="flex min-h-[40vh] items-center justify-center text-sm text-zinc-400">
        Giriş sayfasına yönlendiriliyorsunuz…
      </div>
    )
  }

  return <>{children}</>
}
