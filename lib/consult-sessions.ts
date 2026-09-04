import type { ChatTurn, ConsultResponse, TradeItemMatch } from "@/lib/api"

export const ACTIVE_SESSION_KEY = "wtr-consult-session-id"
const CATALOG_KEY = "wtr-consult-sessions"
const LEGACY_HISTORY_KEY = "wtr-consult-history"

const MARKET_RE =
  /\b(almanya|italya|fransa|hollanda|ispanya|ingiltere|avrupa|amerika|belçika|belcika|polonya)\b/i

export type StoredChat = {
  id: string
  title: string
  updatedAt: number
  history: ChatTurn[]
  product?: string | null
  capacity?: string | null
  market?: string | null
  lastAgent?: string | null
  lastAdvice?: string | null
  context?: TradeItemMatch[]
}

function prettyTitle(text: string) {
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((word) =>
      word.length <= 3 && word === word.toUpperCase()
        ? word
        : word.charAt(0).toUpperCase() + word.slice(1)
    )
    .join(" ")
}

export function titleFromTurns(
  history: ChatTurn[],
  extras?: { product?: string | null; market?: string | null }
) {
  const product = prettyTitle(extras?.product || "")
  let market = (extras?.market || "").trim()
  if (!market) {
    const blob = history.map((turn) => turn.content).join(" ")
    const match = blob.match(MARKET_RE)
    if (match) {
      const raw = match[1].toLocaleLowerCase("tr-TR")
      const map: Record<string, string> = {
        almanya: "Almanya",
        italya: "İtalya",
        fransa: "Fransa",
        hollanda: "Hollanda",
        ispanya: "İspanya",
        ingiltere: "İngiltere",
        avrupa: "Avrupa",
        amerika: "Amerika",
        belcika: "Belçika",
        belçika: "Belçika",
        polonya: "Polonya",
      }
      market = map[raw] || prettyTitle(match[1])
    }
  }
  if (product && market) return `${product} - ${market}`
  if (product) return product
  const first = history.find((turn) => turn.role === "user")?.content.trim()
  if (first) return first.slice(0, 48)
  return "Yeni sohbet"
}

export function readCatalog(): StoredChat[] {
  if (typeof window === "undefined") return []
  try {
    const raw = window.localStorage.getItem(CATALOG_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as StoredChat[]
      if (Array.isArray(parsed)) {
        return parsed.filter((chat) => chat && typeof chat.id === "string")
      }
    }
    const legacyId = window.localStorage.getItem(ACTIVE_SESSION_KEY)
    const legacyHistory = window.localStorage.getItem(LEGACY_HISTORY_KEY)
    if (legacyId && legacyHistory) {
      const history = JSON.parse(legacyHistory) as ChatTurn[]
      if (Array.isArray(history) && history.length > 0) {
        const migrated: StoredChat = {
          id: legacyId,
          title: titleFromTurns(history),
          updatedAt: Date.now(),
          history,
        }
        writeCatalog([migrated])
        return [migrated]
      }
    }
  } catch {
    /* private mode */
  }
  return []
}

export function writeCatalog(chats: StoredChat[]) {
  try {
    window.localStorage.setItem(CATALOG_KEY, JSON.stringify(chats.slice(0, 40)))
  } catch {
    /* private mode */
  }
}

export function readActiveId(fallback: string) {
  try {
    return window.localStorage.getItem(ACTIVE_SESSION_KEY) || fallback
  } catch {
    return fallback
  }
}

export function writeActiveId(id: string) {
  try {
    window.localStorage.setItem(ACTIVE_SESSION_KEY, id)
  } catch {
    /* private mode */
  }
}

export function upsertChat(
  chats: StoredChat[],
  next: StoredChat
): StoredChat[] {
  const rest = chats.filter((chat) => chat.id !== next.id)
  return [next, ...rest].slice(0, 40)
}

export function removeChat(chats: StoredChat[], id: string): StoredChat[] {
  return chats.filter((chat) => chat.id !== id)
}

export function chatFromResult(
  sessionId: string,
  history: ChatTurn[],
  result?: ConsultResponse | null
): StoredChat {
  return {
    id: sessionId,
    title:
      result?.session_title ||
      titleFromTurns(history, {
        product: result?.product,
        market: result?.market,
      }),
    updatedAt: Date.now(),
    history,
    product: result?.product,
    capacity: result?.capacity,
    market: result?.market,
    lastAgent: result?.selected_agent,
    lastAdvice: result?.advice,
    context: result?.context,
  }
}
