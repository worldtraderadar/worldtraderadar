"use client"

import { useEffect, useState } from "react"
import { Download, Share, X } from "lucide-react"

type BeforeInstallPromptEvent = Event & {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>
}

export function InstallBanner() {
  const [visible, setVisible] = useState(false)
  const [ios, setIos] = useState(false)
  const [deferred, setDeferred] = useState<BeforeInstallPromptEvent | null>(
    null
  )

  useEffect(() => {
    const standalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      ("standalone" in navigator &&
        Boolean((navigator as Navigator & { standalone?: boolean }).standalone))
    if (standalone) return

    const dismissed = window.sessionStorage.getItem("wtr-install-dismissed")
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
    setIos(isIOS)
    if (!dismissed && isIOS) setVisible(true)

    const onPrompt = (event: Event) => {
      event.preventDefault()
      setDeferred(event as BeforeInstallPromptEvent)
      if (!dismissed) setVisible(true)
    }
    window.addEventListener("beforeinstallprompt", onPrompt)
    return () => window.removeEventListener("beforeinstallprompt", onPrompt)
  }, [])

  if (!visible) return null

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-[calc(3.75rem+env(safe-area-inset-bottom))] z-30 px-3 lg:bottom-4">
      <div className="pointer-events-auto mx-auto flex max-w-6xl items-center gap-3 rounded-2xl border border-white/10 bg-[#0b1c27]/95 px-3 py-2.5 shadow-lg backdrop-blur-xl">
        {ios ? (
          <Share className="size-4 shrink-0 text-teal-300" />
        ) : (
          <Download className="size-4 shrink-0 text-teal-300" />
        )}
        <p className="min-w-0 flex-1 text-xs leading-5 text-zinc-300">
          {ios
            ? "Ana ekrana eklemek için Paylaş → Ana Ekrana Ekle."
            : "Trade Radar’ı ana ekrana yükleyin; çevrimdışı kabuk ve tam ekran PWA."}
        </p>
        {deferred ? (
          <button
            type="button"
            className="inline-flex h-9 shrink-0 items-center rounded-lg bg-teal-400 px-3 text-xs font-medium text-[#062026]"
            onClick={async () => {
              await deferred.prompt()
              await deferred.userChoice
              setDeferred(null)
              setVisible(false)
              window.sessionStorage.setItem("wtr-install-dismissed", "1")
            }}
          >
            Yükle
          </button>
        ) : null}
        <button
          type="button"
          className="flex size-9 shrink-0 items-center justify-center text-zinc-500"
          aria-label="Kurulum ipucunu kapat"
          onClick={() => {
            window.sessionStorage.setItem("wtr-install-dismissed", "1")
            setVisible(false)
          }}
        >
          <X className="size-4" />
        </button>
      </div>
    </div>
  )
}
