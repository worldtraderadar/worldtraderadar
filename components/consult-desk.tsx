"use client"

import Link from "next/link"
import { useEffect, useRef, useState } from "react"
import { ArrowRight, Compass, LoaderCircle, Sparkles } from "lucide-react"
import { AgentFlow } from "@/components/agent-flow"
import { ChatSidebar } from "@/components/chat-sidebar"
import { MicButton } from "@/components/mic-button"
import { SensitiveDetails } from "@/components/sensitive-details"
import { SpeakButton } from "@/components/speak-button"
import { Button } from "@/components/ui/button"
import {
  consultStream,
  createConsultSession,
  deleteConsultSession,
  QuotaError,
  type AgentStep,
  type ChatTurn,
  type ConsultResponse,
  type TradeItemMatch,
} from "@/lib/api"
import {
  chatFromResult,
  readActiveId,
  readCatalog,
  removeChat,
  titleFromTurns,
  upsertChat,
  writeActiveId,
  writeCatalog,
  type StoredChat,
} from "@/lib/consult-sessions"

const PROMPTS = [
  "Almanya'ya zeytinyağı ihracatında HS kodu ve pazar riskleri neler?",
  "Zeytinyağı için Avrupa'ya satış yapabilecek tedarikçileri bul",
  "1509 HS koduna benzer ürünleri eşleştir",
]

const INTENT_LABEL: Record<string, string> = {
  chat: "Sohbet",
  intake: "Keşif",
  sector_chat: "Sektör sohbeti",
  trade_advisor: "Trade Advisor",
  buyer_finder: "Buyer Finder",
  supplier_finder: "Supplier Finder",
  product_matching: "Product Matching",
}

function emptyChat(id: string, updatedAt = 0): StoredChat {
  return {
    id,
    title: "Yeni sohbet",
    updatedAt,
    history: [],
  }
}

function resultFromChat(chat: StoredChat): ConsultResponse | null {
  if (!chat.lastAdvice && (!chat.context || chat.context.length === 0)) {
    return null
  }
  return {
    question: "",
    advice: chat.lastAdvice || "",
    embed_model: "",
    chat_model: "",
    context: chat.context || [],
    session_id: chat.id,
    history: chat.history,
    selected_agent: chat.lastAgent,
    session_title: chat.title,
    product: chat.product,
    capacity: chat.capacity,
    market: chat.market,
  }
}

export function ConsultDesk() {
  const [question, setQuestion] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [quotaBlocked, setQuotaBlocked] = useState(false)
  const [result, setResult] = useState<ConsultResponse | null>(null)
  const [steps, setSteps] = useState<AgentStep[]>([])
  const [messages, setMessages] = useState<ChatTurn[]>([])
  const [voiceHandsFree, setVoiceHandsFree] = useState(false)
  const [listenKey, setListenKey] = useState(0)
  const [ttsBusy, setTtsBusy] = useState(false)
  const [chats, setChats] = useState<StoredChat[]>([
    { id: "pending", title: "Yeni sohbet", updatedAt: 0, history: [] },
  ])
  const [activeId, setActiveId] = useState("pending")
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const voiceHandsFreeRef = useRef(false)
  const listenTimerRef = useRef<number | null>(null)
  const sessionIdRef = useRef("")
  const messagesRef = useRef<ChatTurn[]>([])
  const chatsRef = useRef<StoredChat[]>([])
  const resultRef = useRef<ConsultResponse | null>(null)

  useEffect(() => {
    let cancelled = false
    async function boot() {
      let catalog = readCatalog()
      let active = readActiveId("")
      if (catalog.length === 0) {
        const local = emptyChat(crypto.randomUUID(), Date.now())
        catalog = [local]
        writeCatalog(catalog)
        writeActiveId(local.id)
        active = local.id
      } else if (!active || !catalog.some((chat) => chat.id === active)) {
        active = catalog[0].id
        writeActiveId(active)
      }
      const current = catalog.find((chat) => chat.id === active) || catalog[0]
      sessionIdRef.current = current.id
      messagesRef.current = current.history
      chatsRef.current = catalog
      resultRef.current = resultFromChat(current)
      if (cancelled) return
      setChats(catalog)
      setActiveId(current.id)
      setMessages(current.history)
      setResult(resultFromChat(current))
      try {
        setSidebarCollapsed(
          window.localStorage.getItem("wtr-consult-sidebar-collapsed") === "1"
        )
      } catch {
        /* private mode */
      }
      if (current.history.length === 0) {
        try {
          const snap = await createConsultSession()
          if (cancelled) return
          if (sessionIdRef.current !== current.id) return
          const fresh = { ...current, id: snap.session_id }
          const next = catalog.map((chat) =>
            chat.id === current.id ? fresh : chat
          )
          sessionIdRef.current = fresh.id
          persistCatalog(next)
          writeActiveId(fresh.id)
          setActiveId(fresh.id)
        } catch {
          /* yerel oturum yeterli */
        }
      }
    }
    void boot()
    return () => {
      cancelled = true
      if (listenTimerRef.current) window.clearTimeout(listenTimerRef.current)
    }
  }, [])

  function persistCatalog(next: StoredChat[]) {
    chatsRef.current = next
    setChats(next)
    writeCatalog(next)
  }

  function snapshotCurrent(): StoredChat {
    const id = sessionIdRef.current
    const history = messagesRef.current
    const existing = chatsRef.current.find((chat) => chat.id === id)
    return {
      ...(existing || emptyChat(id)),
      ...chatFromResult(id, history, resultRef.current),
      title:
        resultRef.current?.session_title ||
        titleFromTurns(history, {
          product: resultRef.current?.product || existing?.product,
          market: resultRef.current?.market || existing?.market,
        }),
      history,
      updatedAt: Date.now(),
    }
  }

  function activateChat(chat: StoredChat) {
    sessionIdRef.current = chat.id
    messagesRef.current = chat.history
    resultRef.current = resultFromChat(chat)
    writeActiveId(chat.id)
    setActiveId(chat.id)
    setMessages(chat.history)
    setResult(resultFromChat(chat))
    setSteps([])
    setError(null)
    setQuestion("")
    setDrawerOpen(false)
  }

  async function startNewChat() {
    const current = snapshotCurrent()
    const kept = chatsRef.current.filter(
      (chat) => chat.history.length > 0 && chat.id !== current.id
    )
    const archived =
      current.history.length > 0 ? upsertChat(kept, current) : kept
    let id = crypto.randomUUID()
    try {
      id = (await createConsultSession()).session_id
    } catch {
      /* yerel yedek */
    }
    const fresh = emptyChat(id, Date.now())
    persistCatalog(upsertChat(archived, fresh))
    activateChat(fresh)
  }

  function selectChat(id: string) {
    if (id === sessionIdRef.current) {
      setDrawerOpen(false)
      return
    }
    const current = snapshotCurrent()
    const withCurrent = upsertChat(chatsRef.current, current)
    const target = withCurrent.find((chat) => chat.id === id)
    persistCatalog(withCurrent)
    if (target) activateChat(target)
  }

  function deleteChat(id: string) {
    const remaining = removeChat(chatsRef.current, id)
    persistCatalog(remaining)
    void deleteConsultSession(id).catch(() => {
      /* yerel listeden zaten düştü */
    })
    if (id !== sessionIdRef.current) return
    if (remaining[0]) {
      activateChat(remaining[0])
      return
    }
    const local = emptyChat(crypto.randomUUID(), Date.now())
    persistCatalog([local])
    activateChat(local)
    void (async () => {
      try {
        const snap = await createConsultSession()
        if (sessionIdRef.current !== local.id) return
        const fresh = { ...local, id: snap.session_id }
        persistCatalog([fresh])
        activateChat(fresh)
      } catch {
        /* yerel yedek */
      }
    })()
  }

  function armMicListen() {
    if (listenTimerRef.current) window.clearTimeout(listenTimerRef.current)
    listenTimerRef.current = window.setTimeout(() => {
      listenTimerRef.current = null
      setListenKey((key) => key + 1)
    }, 450)
  }

  function applyResult(data: ConsultResponse) {
    setResult(data)
    resultRef.current = data
    if (data.session_id) {
      sessionIdRef.current = data.session_id
    }
    const nextHistory =
      data.history && data.history.length > 0
        ? data.history
        : [
            ...messagesRef.current,
            ...(messagesRef.current.at(-1)?.content === data.question
              ? []
              : [{ role: "user" as const, content: data.question }]),
            { role: "assistant" as const, content: data.advice },
          ]
    messagesRef.current = nextHistory
    setMessages(nextHistory)
    const stored = chatFromResult(sessionIdRef.current, nextHistory, data)
    persistCatalog(upsertChat(chatsRef.current, stored))
    writeActiveId(sessionIdRef.current)
    setActiveId(sessionIdRef.current)
    if (data.steps?.length) {
      setSteps(data.steps)
    }
  }

  async function submit(value = question, fromVoice = false) {
    const next = value.trim()
    if (!next || loading) return
    if (!sessionIdRef.current) {
      await startNewChat()
    }
    voiceHandsFreeRef.current = fromVoice
    setVoiceHandsFree(fromVoice)
    setQuestion("")
    setLoading(true)
    setError(null)
    setQuotaBlocked(false)
    setSteps([])
    setTtsBusy(false)
    const prior = messagesRef.current
    const pending = [...prior, { role: "user" as const, content: next }]
    messagesRef.current = pending
    setMessages(pending)
    persistCatalog(
      upsertChat(
        chatsRef.current,
        chatFromResult(sessionIdRef.current, pending, resultRef.current)
      )
    )
    try {
      const data = await consultStream(
        {
          question: next,
          session_id: sessionIdRef.current,
          history: prior,
        },
        (event) => {
          if (event.type === "step" && event.step) {
            setSteps((current) => upsertStep(current, event.step!))
          }
          if (event.type === "complete" && event.result) {
            applyResult(event.result)
          }
          if (event.type === "error") {
            setError(event.detail || "Danışmanlık akışı kesildi.")
            messagesRef.current = prior
            setMessages(prior)
          }
        }
      )
      if (data) applyResult(data)
    } catch (err) {
      messagesRef.current = prior
      setMessages(prior)
      if (err instanceof QuotaError) {
        setQuotaBlocked(true)
        setError(err.message)
      } else {
        setError(err instanceof Error ? err.message : "Danışmanlık alınamadı.")
      }
      if (voiceHandsFreeRef.current) armMicListen()
    } finally {
      setLoading(false)
    }
  }

  const lastAdvice =
    [...messages].reverse().find((turn) => turn.role === "assistant")?.content ||
    result?.advice ||
    ""

  return (
    <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:gap-6">
      <ChatSidebar
        chats={chats}
        activeId={activeId}
        drawerOpen={drawerOpen}
        collapsed={sidebarCollapsed}
        onDrawerOpenChange={setDrawerOpen}
        onCollapsedChange={(next) => {
          setSidebarCollapsed(next)
          try {
            window.localStorage.setItem(
              "wtr-consult-sidebar-collapsed",
              next ? "1" : "0"
            )
          } catch {
            /* private mode */
          }
        }}
        onNewChat={() => void startNewChat()}
        onSelect={selectChat}
        onDelete={deleteChat}
      />
      <div className="min-w-0 flex-1 space-y-10">
      <section className="relative overflow-hidden rounded-3xl border border-white/8 bg-[#0b1c27]/80 p-5 shadow-[0_0_80px_rgba(13,148,136,0.08)] sm:p-10">
        <div className="pointer-events-none absolute -right-16 -top-16 size-64 rounded-full border border-teal-400/10" />
        <div className="pointer-events-none absolute -right-8 -top-8 size-40 rounded-full border border-teal-400/15" />
        <p className="mb-3 text-xs font-medium tracking-[0.22em] text-teal-300/80 uppercase">
          Kahve masası
        </p>
        <h1 className="font-heading max-w-2xl text-3xl leading-tight text-zinc-50 sm:text-5xl">
          Dış ticarette yönünüzü netleştirin.
        </h1>
        <p className="mt-4 max-w-xl text-base leading-relaxed text-zinc-400">
          Orchestrator niyeti okur; Trade Advisor, Supplier Finder veya
          Product Matching ajanını seçer. Adımlar canlı izlenir.{" "}
          <Link href="/gecmis" className="text-teal-300 hover:underline">
            Geçmiş sohbetler
          </Link>
        </p>

        <form
          className="mt-8 space-y-4"
          onSubmit={(event) => {
            event.preventDefault()
            void submit()
          }}
        >
          <div className="flex items-end gap-2">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={() => {
                if (!voiceHandsFreeRef.current) return
                voiceHandsFreeRef.current = false
                setVoiceHandsFree(false)
              }}
              placeholder="Hedef pazar, tedarikçi veya HS eşleşmesi sorun… ya da mikrofona konuşun."
              rows={4}
              className="min-w-0 flex-1 resize-none rounded-2xl border border-white/10 bg-[#061018] px-4 py-3 text-sm leading-6 text-zinc-100 outline-none placeholder:text-zinc-600 focus:border-teal-400/40 focus:ring-3 focus:ring-teal-400/15"
            />
            <MicButton
              value={question}
              onTranscript={setQuestion}
              onCommit={(text) => void submit(text, true)}
              disabled={loading || ttsBusy}
              listenKey={listenKey}
            />
          </div>
          {voiceHandsFree ? (
            <p className="text-[11px] tracking-wide text-teal-300/75">
              {loading
                ? "Yanıt hazırlanıyor — giriş kutusu yeni komut için boş."
                : ttsBusy
                  ? "Asistan konuşuyor. Bitince mikrofon yeniden açılır."
                  : "Mikrofon dinlemede. Konuşun; metni silmeniz gerekmez."}
            </p>
          ) : null}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div className="flex flex-wrap gap-2">
              {PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => void submit(prompt)}
                  className="rounded-full border border-white/8 bg-white/3 px-3 py-1 text-left text-xs text-zinc-400 transition hover:border-teal-400/30 hover:text-zinc-200"
                >
                  {prompt}
                </button>
              ))}
            </div>
            <Button
              type="submit"
              size="lg"
              disabled={loading || !question.trim()}
              className="w-full bg-teal-400 text-[#062026] hover:bg-teal-300 sm:w-auto sm:self-end"
            >
              {loading ? (
                <LoaderCircle className="animate-spin" />
              ) : (
                <Sparkles />
              )}
              {loading ? "Ajanlar çalışıyor…" : "Danışmanlık al"}
              {!loading && <ArrowRight data-icon="inline-end" />}
            </Button>
          </div>
        </form>
        {error ? (
          <div className="mt-4 rounded-xl border border-amber-400/20 bg-amber-400/8 px-3 py-2 text-sm text-amber-200">
            <p>{error}</p>
            {quotaBlocked ? (
              <Link
                href="/paket"
                className="mt-2 inline-flex text-teal-300 hover:underline"
              >
                Pro pakete yükselt →
              </Link>
            ) : null}
          </div>
        ) : null}
      </section>

      {loading || steps.length > 0 ? <AgentFlow steps={steps} /> : null}
      {messages.length > 0 || result ? (
        <Briefing
          messages={messages}
          result={result}
          lastAdvice={lastAdvice}
          autoPlay={voiceHandsFree}
          onTtsBusyChange={setTtsBusy}
          onSpeakFinished={() => {
            if (voiceHandsFreeRef.current) armMicListen()
          }}
        />
      ) : null}
      </div>
    </div>
  )
}

function upsertStep(current: AgentStep[], next: AgentStep) {
  const index = current.findIndex(
    (step) => step.agent === next.agent && step.key === next.key
  )
  if (index === -1) return [...current, next]
  const copy = current.slice()
  copy[index] = next
  return copy
}

function Briefing({
  messages,
  result,
  lastAdvice,
  autoPlay,
  onTtsBusyChange,
  onSpeakFinished,
}: {
  messages: ChatTurn[]
  result: ConsultResponse | null
  lastAdvice: string
  autoPlay: boolean
  onTtsBusyChange: (busy: boolean) => void
  onSpeakFinished: () => void
}) {
  const agent =
    (result?.selected_agent && INTENT_LABEL[result.selected_agent]) ||
    result?.selected_agent
  const context = result?.context ?? []
  return (
    <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
      <article className="rounded-3xl border border-white/8 bg-[#0b1c27]/70 p-7">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2 text-teal-300">
            <Compass className="size-4" />
            <h2 className="font-heading text-xl text-zinc-50">Sohbet</h2>
            {agent ? (
              <span className="rounded-full border border-teal-400/20 bg-teal-400/8 px-2 py-0.5 font-mono text-[10px] text-teal-200">
                {agent}
              </span>
            ) : null}
          </div>
          {lastAdvice ? (
            <SpeakButton
              text={lastAdvice}
              autoPlay={autoPlay}
              onBusyChange={onTtsBusyChange}
              onFinished={onSpeakFinished}
            />
          ) : null}
        </div>
        <div className="space-y-3">
          {messages.map((turn, index) => (
            <div
              key={`${turn.role}-${index}`}
              className={
                turn.role === "user"
                  ? "rounded-2xl border border-white/8 bg-white/4 px-4 py-3 text-sm leading-6 text-zinc-300"
                  : "rounded-2xl border border-teal-400/15 bg-teal-400/6 px-4 py-3 text-sm leading-7 whitespace-pre-wrap text-zinc-100"
              }
            >
              <p className="mb-1 font-mono text-[10px] tracking-[0.14em] text-zinc-500 uppercase">
                {turn.role === "user" ? "Siz" : "Asistan"}
              </p>
              {turn.content}
            </div>
          ))}
        </div>
        {result ? (
          <p className="mt-6 font-mono text-[11px] text-zinc-600">
            {result.run_id ? `run ${result.run_id.slice(0, 8)}` : ""}
            {result.session_id
              ? `${result.run_id ? " · " : ""}oturum ${result.session_id.slice(0, 8)}`
              : ""}
          </p>
        ) : null}
      </article>
      <aside className="space-y-3">
        <h3 className="px-1 text-xs tracking-[0.18em] text-zinc-500 uppercase">
          {result?.selected_agent === "buyer_finder"
            ? "Eşleşen alıcı kalemleri"
            : result?.selected_agent === "supplier_finder"
              ? "Eşleşen tedarik kalemleri"
              : "Bağlam kalemleri"}
        </h3>
        {context.length === 0 ? (
          <p className="rounded-2xl border border-white/8 p-5 text-sm text-zinc-500">
            {result?.selected_agent === "buyer_finder" ||
            result?.selected_agent === "supplier_finder"
              ? "Bu ürünle ilgili alıcı/tedarik kaydı yok. Kartlar yalnızca sorgu ürününe göre süzülür."
              : "Bu soruya yeterince benzer ticaret kalemi bulunamadı."}
          </p>
        ) : (
          context.map((item) => <ContextCard key={item.id} item={item} />)
        )}
      </aside>
    </section>
  )
}

function ContextCard({ item }: { item: TradeItemMatch }) {
  const similarity = Math.round(item.similarity * 100)
  return (
    <div className="rounded-2xl border border-white/8 bg-[#08141c] p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {item.organization_name ? (
            <p className="text-sm font-medium text-zinc-100">
              {item.organization_name}
            </p>
          ) : null}
          <p
            className={
              item.organization_name
                ? "mt-0.5 text-xs text-zinc-400"
                : "text-sm font-medium text-zinc-100"
            }
          >
            {item.product_name}
          </p>
        </div>
        <span className="font-mono text-[11px] text-teal-300">{similarity}%</span>
      </div>
      <p className="mt-2 font-mono text-xs text-zinc-500">
        HS {item.hs_code || "—"} · {(item.origin_country || "?").trim()} →{" "}
        {(item.destination_country || "?").trim()}
      </p>
      {item.description ? (
        <p className="mt-2 line-clamp-3 text-xs leading-5 text-zinc-400">
          {item.description}
        </p>
      ) : null}
      <SensitiveDetails item={item} compact />
    </div>
  )
}
