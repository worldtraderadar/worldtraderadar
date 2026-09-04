import { fetchSpeakAudio } from "@/lib/api"

export const SPEECH_LANG = "tr-TR"

/** Nefes / duraksama cümle bitişi sayılmasın. */
export const VAD_SILENCE_MS = 1500
/** Asistan sesinin yankısını yutmak için. */
export const VAD_ECHO_GUARD_MS = 350

export type SpeechRecognitionLike = {
  lang: string
  continuous: boolean
  interimResults: boolean
  maxAlternatives: number
  start: () => void
  stop: () => void
  abort: () => void
  onstart: (() => void) | null
  onend: (() => void) | null
  onerror: ((event: { error: string }) => void) | null
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
}

export type SpeechRecognitionEventLike = {
  resultIndex: number
  results: ArrayLike<{
    isFinal: boolean
    0: { transcript: string }
  }>
}

export function collectRecognitionTranscripts(
  results: SpeechRecognitionEventLike["results"]
): { finalTranscript: string; interimTranscript: string } {
  let finalTranscript = ""
  let interimTranscript = ""
  for (let i = 0; i < results.length; i += 1) {
    const piece = results[i]?.[0]?.transcript ?? ""
    if (results[i]?.isFinal) finalTranscript += piece
    else interimTranscript += piece
  }
  return { finalTranscript, interimTranscript }
}

function foldTr(text: string) {
  return text.replace(/\s+/g, " ").trim().toLocaleLowerCase("tr-TR")
}

/** Nihai + anlık parçaları tek metinde birleştir; tekrarlayan hipotezi yut. */
export function composeVoiceTranscript(
  finalTranscript: string,
  interimTranscript: string
): string {
  const finalText = finalTranscript.replace(/\s+/g, " ").trim()
  const interimText = interimTranscript.replace(/\s+/g, " ").trim()
  if (!interimText) return finalText
  if (!finalText) return interimText
  const foldedFinal = foldTr(finalText)
  const foldedInterim = foldTr(interimText)
  if (foldedInterim === foldedFinal) return finalText
  if (foldedInterim.startsWith(foldedFinal)) return interimText
  if (foldedFinal.startsWith(foldedInterim)) return finalText
  return `${finalText} ${interimText}`.replace(/\s+/g, " ").trim()
}

/** Önceki oturum metnine yeni cümleyi ekle; aynı cümleyi iki kez yazma. */
export function joinUniqueUtterance(prefix: string, next: string): string {
  const left = prefix.replace(/\s+/g, " ").trim()
  const right = next.replace(/\s+/g, " ").trim()
  if (!left) return right
  if (!right) return left
  const foldedLeft = foldTr(left)
  const foldedRight = foldTr(right)
  if (foldedRight === foldedLeft) return left
  if (foldedRight.startsWith(foldedLeft)) return right
  if (foldedLeft.startsWith(foldedRight)) return left
  if (foldedLeft.endsWith(foldedRight)) return left
  if (foldedRight.endsWith(foldedLeft)) return right
  return `${left} ${right}`.replace(/\s+/g, " ").trim()
}

type SpeechWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognitionLike
  webkitSpeechRecognition?: new () => SpeechRecognitionLike
}

export function getSpeechRecognitionCtor():
  | (new () => SpeechRecognitionLike)
  | null {
  if (typeof window === "undefined") return null
  const speechWindow = window as SpeechWindow
  return (
    speechWindow.SpeechRecognition ??
    speechWindow.webkitSpeechRecognition ??
    null
  )
}

export function speechRecognitionSupported() {
  return getSpeechRecognitionCtor() != null
}

let speakGeneration = 0
let currentAudio: HTMLAudioElement | null = null
let currentUrl: string | null = null

const SILENCE_WAV =
  "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"

export function unlockAudioPlayback() {
  if (typeof window === "undefined") return
  try {
    const AudioCtx =
      window.AudioContext ||
      (window as Window & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext
    if (AudioCtx) void new AudioCtx().resume()
  } catch {
    /* tarayıcı kilidi */
  }
  const audio = new Audio(SILENCE_WAV)
  audio.volume = 0.01
  void audio.play().catch(() => {})
}

export function stopSpeaking() {
  speakGeneration += 1
  if (currentAudio) {
    currentAudio.pause()
    currentAudio.removeAttribute("src")
    currentAudio.load()
    currentAudio = null
  }
  if (currentUrl) {
    URL.revokeObjectURL(currentUrl)
    currentUrl = null
  }
}

export async function speakText(
  text: string,
  {
    onEnd,
    onStart,
    onError,
  }: {
    lang?: string
    onEnd?: () => void
    onStart?: () => void
    onError?: (message: string) => void
  } = {}
) {
  const cleaned = text.replace(/\s+/g, " ").trim()
  if (!cleaned) {
    onEnd?.()
    return
  }

  stopSpeaking()
  const generation = speakGeneration
  try {
    const blob = await fetchSpeakAudio(cleaned)
    if (generation !== speakGeneration) return
    const url = URL.createObjectURL(blob)
    currentUrl = url
    const audio = new Audio(url)
    currentAudio = audio
    audio.onplay = () => onStart?.()
    audio.onended = () => {
      if (generation !== speakGeneration) return
      stopSpeaking()
      onEnd?.()
    }
    audio.onerror = () => {
      if (generation !== speakGeneration) return
      onError?.("Ses dosyası oynatılamadı.")
      stopSpeaking()
      onEnd?.()
    }
    await audio.play()
  } catch (error) {
    if (generation !== speakGeneration) return
    const message =
      error instanceof Error ? error.message : "Yerel ses motoruna ulaşılamadı."
    onError?.(message)
    onEnd?.()
  }
}
