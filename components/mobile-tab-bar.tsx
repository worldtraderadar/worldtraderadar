"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Building2, Compass, History, Package, Warehouse } from "lucide-react"
import { cn } from "@/lib/utils"

const TABS = [
  { href: "/", label: "Radar", icon: Compass },
  { href: "/kalemler", label: "Kalemler", icon: Warehouse },
  { href: "/firmalar", label: "Firmalar", icon: Building2 },
  { href: "/gecmis", label: "Geçmiş", icon: History },
  { href: "/paket", label: "Paket", icon: Package },
]

export function MobileTabBar() {
  const pathname = usePathname()
  if (pathname === "/login") return null
  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-40 border-t border-white/8 bg-[#07131c]/92 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl lg:hidden"
      aria-label="Mobil gezinme"
    >
      <ul className="mx-auto grid max-w-6xl grid-cols-5">
        {TABS.map((tab) => {
          const active =
            tab.href === "/"
              ? pathname === "/"
              : pathname.startsWith(tab.href)
          const Icon = tab.icon
          return (
            <li key={tab.href}>
              <Link
                href={tab.href}
                className={cn(
                  "flex min-h-14 flex-col items-center justify-center gap-0.5 text-[10px] tracking-wide",
                  active ? "text-teal-300" : "text-zinc-500"
                )}
              >
                <Icon className="size-4" />
                {tab.label}
              </Link>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
