"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useEffect, useState } from "react"
import { Radar } from "lucide-react"
import { QuotaMeter } from "@/components/quota-meter"
import { getApiUrl, getHealth, type HealthResponse } from "@/lib/api"
import { cn } from "@/lib/utils"

export const NAV = [
  { href: "/", label: "Danışmanlık" },
  { href: "/kalemler", label: "Kalemler" },
  { href: "/firmalar", label: "Firmalar" },
  { href: "/gecmis", label: "Geçmiş" },
  { href: "/paket", label: "Paket" },
]

export function SiteHeader() {
  const pathname = usePathname()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [offline, setOffline] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function ping() {
      try {
        const data = await getHealth()
        if (!cancelled) {
          setHealth(data)
          setOffline(false)
        }
      } catch {
        if (!cancelled) {
          setHealth(null)
          setOffline(true)
        }
      }
    }

    ping()
    const id = setInterval(ping, 20_000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const ok = !offline && health?.status === "ok"

  return (
    <header className="sticky top-0 z-40 border-b border-white/8 bg-[#07131c]/85 pt-[env(safe-area-inset-top)] backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <Link href="/" className="flex min-h-11 min-w-0 items-center gap-2.5">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-md border border-teal-400/30 bg-teal-400/10 text-teal-300">
            <Radar className="size-4" />
          </span>
          <span className="font-heading truncate text-base tracking-tight text-zinc-50 sm:text-lg">
            <span className="sm:hidden">Trade Radar</span>
            <span className="hidden sm:inline">World Trade Radar</span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 text-sm lg:flex">
          {NAV.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href)
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "min-h-9 rounded-md px-3 py-1.5 transition-colors",
                  active
                    ? "bg-white/8 text-zinc-50"
                    : "text-zinc-400 hover:text-zinc-200"
                )}
              >
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="flex items-center gap-3">
          <QuotaMeter />
          <div className="hidden items-center gap-2 text-xs text-zinc-400 sm:flex">
            <span
              className={cn(
                "size-1.5 rounded-full",
                ok
                  ? "bg-teal-400 shadow-[0_0_8px_rgba(45,212,191,0.8)]"
                  : "bg-amber-400"
              )}
            />
            <span className="hidden md:inline">
              {offline ? "API çevrimdışı" : ok ? "API hazır" : "API kısmi"}
            </span>
            <span className="hidden font-mono text-[11px] text-zinc-600 xl:inline">
              {getApiUrl().replace(/^https?:\/\//, "")}
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}
