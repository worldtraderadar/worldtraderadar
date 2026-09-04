"use client"

import { useEffect, useRef, useState } from "react"
import { LoaderCircle, Pause, Volume2 } from "lucide-react"
import { speakText, stopSpeaking, unlockAudioPlayback } from "@/lib/speech"
import { cn } from "@/lib/utils"

export function SpeakButton({
  text,
  className,
  autoPlay = false,
  onBusyChange,
  onFinished,
}: {
  text: string
  className?: string
  autoPlay?: boolean
  onBusyChange?: (busy: boolean) => void
  onFinished?: () => void
}) {
  const [speaking, setSpeaking] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const onBusyChangeRef = useRef(onBusyChange)
  const onFinishedRef = useRef(onFinished)
  onBusyChangeRef.current = onBusyChange
  onFinishedRef.current = onFinished

  function setBusy(next: boolean) {
    onBusyChangeRef.current?.(next)
  }

  function play() {
    setError(null)
    setLoading(true)
    setBusy(true)
    void speakText(text, {
      onStart: () => {
        setLoading(false)
        setSpeaking(true)
        setBusy(true)
      },
      onEnd: () => {
        setLoading(false)
        setSpeaking(false)
        setBusy(false)
        onFinishedRef.current?.()
      },
      onError: (message) => {
        setError(message)
        setLoading(false)
        setSpeaking(false)
        setBusy(false)
        onFinishedRef.current?.()
      },
    })
  }

  useEffect(() => {
    return () => {
      stopSpeaking()
      onBusyChangeRef.current?.(false)
    }
  }, [])

  useEffect(() => {
    stopSpeaking()
    setSpeaking(false)
    setError(null)
    if (!autoPlay || !text.trim()) {
      setLoading(false)
      setBusy(false)
      return
    }
    play()
    return () => stopSpeaking()
  }, [text, autoPlay])

  if (!text.trim()) return null

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        type="button"
        disabled={loading}
        onClick={() => {
          unlockAudioPlayback()
          if (speaking || loading) {
            stopSpeaking()
            setSpeaking(false)
            setLoading(false)
            setBusy(false)
            return
          }
          play()
        }}
        aria-pressed={speaking}
        className={cn(
          "inline-flex h-9 items-center gap-1.5 rounded-lg border px-2.5 text-xs transition",
          speaking || loading
            ? "border-teal-400/40 bg-teal-400/12 text-teal-200"
            : "border-white/10 bg-white/4 text-zinc-300 hover:border-teal-400/30 hover:text-teal-200",
          className
        )}
      >
        {loading ? (
          <LoaderCircle className="size-3.5 animate-spin" />
        ) : speaking ? (
          <Pause className="size-3.5" />
        ) : (
          <Volume2 className="size-3.5" />
        )}
        {loading ? "Hazırlanıyor…" : speaking ? "Durdur" : "Sesli Dinle"}
      </button>
      {error ? (
        <span className="max-w-52 text-right text-[10px] leading-4 text-amber-200">
          {error}
        </span>
      ) : null}
    </div>
  )
}
