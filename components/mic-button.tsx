"use client"

import { useEffect, useRef, useState } from "react"
import { Mic, MicOff } from "lucide-react"
import {
  collectRecognitionTranscripts,
  composeVoiceTranscript,
  getSpeechRecognitionCtor,
  joinUniqueUtterance,
  speechRecognitionSupported,
  stopSpeaking,
  unlockAudioPlayback,
  SPEECH_LANG,
  VAD_ECHO_GUARD_MS,
  VAD_SILENCE_MS,
  type SpeechRecognitionLike,
} from "@/lib/speech"
import { cn } from "@/lib/utils"

const ERROR_COPY: Record<string, string> = {
  "not-allowed": "Mikrofon izni verilmedi.",
  "audio-capture": "Mikrofon bulunamadı.",
  "service-not-allowed": "Ses tanıma bu tarayıcıda kapalı.",
  "network": "Ses tanıma ağına ulaşılamadı.",
  "aborted": "",
  "no-speech": "",
}

export function MicButton({
  value,
  onTranscript,
  onCommit,
  disabled,
  listenKey = 0,
}: {
  value: string
  onTranscript: (text: string) => void
  onCommit?: (text: string) => void
  disabled?: boolean
  listenKey?: number
}) {
  const [supported, setSupported] = useState(false)
  const [listening, setListening] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const listeningRef = useRef(false)
  const valueRef = useRef(value)
  const disabledRef = useRef(disabled)
  const onTranscriptRef = useRef(onTranscript)
  const onCommitRef = useRef(onCommit)
  const commitTimerRef = useRef<number | null>(null)
  const lastSpeechAtRef = useRef(0)
  const ignoreUntilRef = useRef(0)
  const appliedListenKey = useRef(0)
  const committedRef = useRef("")
  const sessionFinalRef = useRef("")
  const composedRef = useRef("")

  onTranscriptRef.current = onTranscript
  onCommitRef.current = onCommit
  valueRef.current = value
  disabledRef.current = disabled

  useEffect(() => {
    setSupported(speechRecognitionSupported())
  }, [])

  useEffect(() => {
    return () => {
      listeningRef.current = false
      if (commitTimerRef.current) window.clearTimeout(commitTimerRef.current)
      const rec = recognitionRef.current
      recognitionRef.current = null
      rec?.abort()
    }
  }, [])

  function clearCommitTimer() {
    if (commitTimerRef.current) {
      window.clearTimeout(commitTimerRef.current)
      commitTimerRef.current = null
    }
  }

  function resetBuffers() {
    committedRef.current = ""
    sessionFinalRef.current = ""
    composedRef.current = ""
  }

  function publish(text: string) {
    const next = text.replace(/\s+/g, " ").trim()
    if (next === composedRef.current) return
    composedRef.current = next
    onTranscriptRef.current(next)
  }

  function haltRecognition() {
    listeningRef.current = false
    clearCommitTimer()
    const rec = recognitionRef.current
    recognitionRef.current = null
    if (rec) {
      rec.onresult = null
      rec.onend = null
      rec.onerror = null
      rec.onstart = null
      try {
        rec.stop()
      } catch {
        try {
          rec.abort()
        } catch {
          /* zaten kapalı */
        }
      }
    }
    resetBuffers()
    setListening(false)
  }

  function commitNow() {
    const text = (composedRef.current || valueRef.current).trim()
    haltRecognition()
    if (text) onCommitRef.current?.(text)
  }

  function scheduleCommit() {
    if (!onCommitRef.current) return
    clearCommitTimer()
    commitTimerRef.current = window.setTimeout(() => {
      commitTimerRef.current = null
      if (!listeningRef.current || disabledRef.current) return
      const quietFor = Date.now() - lastSpeechAtRef.current
      if (quietFor < VAD_SILENCE_MS) {
        scheduleCommit()
        return
      }
      commitNow()
    }, VAD_SILENCE_MS)
  }

  function noteSpeech() {
    lastSpeechAtRef.current = Date.now()
    scheduleCommit()
  }

  function start() {
    const Ctor = getSpeechRecognitionCtor()
    if (!Ctor || disabledRef.current) return
    if (recognitionRef.current) haltRecognition()
    unlockAudioPlayback()
    stopSpeaking()
    setError(null)
    clearCommitTimer()
    resetBuffers()
    composedRef.current = ""
    onTranscriptRef.current("")
    ignoreUntilRef.current = Date.now() + VAD_ECHO_GUARD_MS
    lastSpeechAtRef.current = Date.now()
    const recognition = new Ctor()
    recognition.lang = SPEECH_LANG
    recognition.continuous = true
    recognition.interimResults = true
    recognition.maxAlternatives = 1
    recognition.onstart = () => {
      listeningRef.current = true
      setListening(true)
    }
    recognition.onresult = (event) => {
      if (Date.now() < ignoreUntilRef.current) return
      const { finalTranscript, interimTranscript } =
        collectRecognitionTranscripts(event.results)
      sessionFinalRef.current = finalTranscript.replace(/\s+/g, " ").trim()
      const committed = joinUniqueUtterance(
        committedRef.current,
        sessionFinalRef.current
      )
      const live = composeVoiceTranscript(
        committed,
        interimTranscript
      )
      publish(live)
      if (finalTranscript || interimTranscript) noteSpeech()
    }
    recognition.onerror = (event) => {
      const message = ERROR_COPY[event.error]
      if (message) setError(message)
      if (event.error === "no-speech" || event.error === "aborted") return
      listeningRef.current = false
      clearCommitTimer()
      setListening(false)
    }
    recognition.onend = () => {
      if (recognitionRef.current !== recognition) {
        setListening(false)
        return
      }
      if (listeningRef.current && !disabledRef.current) {
        committedRef.current = joinUniqueUtterance(
          committedRef.current,
          sessionFinalRef.current
        )
        sessionFinalRef.current = ""
        try {
          recognition.start()
          return
        } catch {
          listeningRef.current = false
        }
      }
      try {
        recognition.stop()
      } catch {
        /* kapalı */
      }
      recognitionRef.current = null
      setListening(false)
    }
    recognitionRef.current = recognition
    try {
      recognition.start()
    } catch {
      recognitionRef.current = null
      setError("Mikrofon başlatılamadı.")
    }
  }

  useEffect(() => {
    if (disabled) haltRecognition()
  }, [disabled])

  useEffect(() => {
    if (listenKey <= 0 || listenKey === appliedListenKey.current) return
    if (disabled) return
    appliedListenKey.current = listenKey
    start()
  }, [listenKey, disabled])

  if (!supported) {
    return (
      <button
        type="button"
        disabled
        title="Bu tarayıcı sesli girişi desteklemiyor (Chrome veya Edge deneyin)."
        className="flex size-12 shrink-0 items-center justify-center rounded-xl border border-white/8 text-zinc-600 sm:size-10"
        aria-label="Mikrofon desteklenmiyor"
      >
        <MicOff className="size-5 sm:size-4" />
      </button>
    )
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        type="button"
        disabled={disabled}
        onClick={() => (listening ? commitNow() : start())}
        aria-pressed={listening}
        aria-label={listening ? "Gönder ve durdur" : "Sesli soru söyle"}
        title={
          listening
            ? "Dokunun: gönder"
            : "Mikrofon — Türkçe konuşun, bitince otomatik gider"
        }
        className={cn(
          "flex size-12 shrink-0 items-center justify-center rounded-xl border transition sm:size-10",
          listening
            ? "border-teal-400/50 bg-teal-400/15 text-teal-200"
            : "border-white/10 bg-white/4 text-zinc-300 hover:border-teal-400/30 hover:text-teal-200",
          disabled && "opacity-50"
        )}
      >
        <Mic className={cn("size-5 sm:size-4", listening && "animate-pulse")} />
      </button>
      {listening ? (
        <span className="text-[10px] tracking-wide text-teal-300/80">
          Dinleniyor…
        </span>
      ) : error ? (
        <span className="max-w-36 text-right text-[10px] leading-4 text-amber-200">
          {error}
        </span>
      ) : null}
    </div>
  )
}
