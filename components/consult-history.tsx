"use client"

import { useEffect, useMemo, useState } from "react"
import { ChevronDown, Clock3, FileText, LoaderCircle, Trash2 } from "lucide-react"
import { SpeakButton } from "@/components/speak-button"
import {
  deleteConsultSession,
  listConsultHistory,
  type AgentLogStatus,
  type ConsultHistoryLog,
} from "@/lib/api"
import {
  readActiveId,
  readCatalog,
  removeChat,
  writeActiveId,
  writeCatalog,
} from "@/lib/consult-sessions"
import { cn } from "@/lib/utils"

type Filter = "all" | "success" | "error"

const EMPTY_COPY = "Kayıt bulunamadı"

export function ConsultHistory() {
  const [logs, setLogs] = useState<ConsultHistoryLog[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<Filter>("all")
  const [openId, setOpenId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const data = await listConsultHistory(80)
        if (cancelled) return
        const rows = Array.isArray(data) ? data : []
        setLogs(rows)
        const firstWithAdvice = rows.find((log) => Boolean(reportOf(log)))
        setOpenId((firstWithAdvice ?? rows[0])?.id ?? null)
      } catch {
        if (cancelled) return
        setLogs([])
        setOpenId(null)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  async function deleteLog(log: ConsultHistoryLog) {
    const sessionId = sessionIdOf(log) || log.id
    setDeletingId(log.id)
    const next = logs.filter((row) => {
      if (row.id === log.id) return false
      const rowSession = sessionIdOf(row)
      return !rowSession || rowSession !== sessionId
    })
    setLogs(next)
    setOpenId((open) =>
      open && !next.some((row) => row.id === open) ? next[0]?.id ?? null : open
    )
    try {
      const catalog = removeChat(readCatalog(), sessionId)
      writeCatalog(catalog)
      if (readActiveId("") === sessionId) {
        writeActiveId(catalog[0]?.id ?? "")
      }
      await deleteConsultSession(sessionId)
    } catch {
      /* liste zaten güncellendi */
    } finally {
      setDeletingId(null)
    }
  }

  const visible = useMemo(
    () =>
      logs.filter((log) => (filter === "all" ? true : log.status === filter)),
    [filter, logs]
  )

  return (
    <div className="space-y-8">
      <header className="space-y-3">
        <p className="text-xs font-medium tracking-[0.22em] text-teal-300/80 uppercase">
          Arşiv
        </p>
        <h1 className="font-heading text-4xl text-zinc-50">
          Danışmanlık geçmişi
        </h1>
        <p className="max-w-2xl text-sm leading-6 text-zinc-400">
          `/consult` sohbetleri ve llama3 cevapları ajan kayıtlarından
          kronolojik olarak listelenir.
        </p>
      </header>

      <div className="flex flex-wrap gap-2">
        {(
          [
            ["all", "Tümü"],
            ["success", "Cevaplar"],
            ["error", "Hatalar"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            onClick={() => setFilter(value)}
            className={cn(
              "rounded-full border px-3 py-1 text-xs transition",
              filter === value
                ? "border-teal-400/40 bg-teal-400/10 text-teal-200"
                : "border-white/8 text-zinc-500 hover:text-zinc-300"
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {loading ? (
        <p className="flex items-center gap-2 text-sm text-zinc-500">
          <LoaderCircle className="size-4 animate-spin" />
          Kayıtlar yükleniyor…
        </p>
      ) : visible.length === 0 ? (
        <p className="rounded-2xl border border-white/8 p-6 text-sm text-zinc-500">
          {EMPTY_COPY}
        </p>
      ) : (
        <ol className="space-y-3">
          {visible.map((log) => (
            <HistoryCard
              key={log.id}
              log={log}
              open={openId === log.id}
              deleting={deletingId === log.id}
              onToggle={() =>
                setOpenId((current) => (current === log.id ? null : log.id))
              }
              onDelete={() => void deleteLog(log)}
            />
          ))}
        </ol>
      )}
    </div>
  )
}

function HistoryCard({
  log,
  open,
  deleting,
  onToggle,
  onDelete,
}: {
  log: ConsultHistoryLog
  open: boolean
  deleting: boolean
  onToggle: () => void
  onDelete: () => void
}) {
  const question = questionOf(log)
  const advice = reportOf(log)
  const context = Array.isArray(log.output?.context) ? log.output.context : []
  const when = formatWhen(log.created_at)

  return (
    <li className="overflow-hidden rounded-2xl border border-white/8 bg-[#0b1c27]/75">
      <div className="flex items-start gap-1 pr-2">
        <button
          type="button"
          onClick={onToggle}
          className="flex min-w-0 flex-1 items-start gap-4 px-5 py-4 text-left"
        >
          <span
            className={cn(
              "mt-1 size-2 shrink-0 rounded-full",
              log.status === "success"
                ? "bg-teal-400"
                : log.status === "error"
                  ? "bg-amber-400"
                  : "bg-zinc-500"
            )}
          />
          <span className="min-w-0 flex-1">
            <span className="flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
              <Clock3 className="size-3" />
              {when}
              <StatusBadge status={log.status} />
              {log.duration_ms != null ? (
                <span>{(log.duration_ms / 1000).toFixed(1)} sn</span>
              ) : null}
            </span>
            <span className="mt-1.5 block text-sm font-medium text-zinc-100">
              {question}
            </span>
            {advice && !open ? (
              <span className="mt-1 line-clamp-2 block text-xs leading-5 text-zinc-500">
                {advice}
              </span>
            ) : null}
          </span>
          <ChevronDown
            className={cn(
              "mt-1 size-4 shrink-0 text-zinc-500 transition",
              open && "rotate-180"
            )}
          />
        </button>
        <button
          type="button"
          aria-label="Sohbeti sil"
          disabled={deleting}
          onClick={(event) => {
            event.preventDefault()
            event.stopPropagation()
            onDelete()
          }}
          className="mt-3 flex size-9 shrink-0 items-center justify-center rounded-lg text-zinc-500 transition hover:bg-red-400/10 hover:text-red-300 disabled:opacity-40"
        >
          {deleting ? (
            <LoaderCircle className="size-3.5 animate-spin" />
          ) : (
            <Trash2 className="size-3.5" />
          )}
        </button>
      </div>

      {open ? (
        <div className="space-y-4 border-t border-white/6 px-5 py-4">
          {log.error_message ? (
            <p className="rounded-xl border border-amber-400/20 bg-amber-400/8 px-3 py-2 text-sm text-amber-200">
              {log.error_message}
            </p>
          ) : null}
          {advice ? (
            <article>
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-teal-300">
                <div className="flex items-center gap-2">
                  <FileText className="size-3.5" />
                  <h2 className="text-xs tracking-[0.16em] uppercase">Cevap</h2>
                </div>
                <SpeakButton text={advice} />
              </div>
              <div className="text-sm leading-7 whitespace-pre-wrap text-zinc-200">
                {advice}
              </div>
            </article>
          ) : log.status === "success" ? (
            <p className="text-sm text-zinc-500">
              Bu kayıtta cevap metni saklanmamış (eski log). Yeni
              danışmanlıklarda tam metin burada görünür.
            </p>
          ) : null}
          {context.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {context.map((item, index) => (
                <span
                  key={item.id ?? `${item.product_name}-${index}`}
                  className="rounded-full border border-white/8 px-2.5 py-1 font-mono text-[11px] text-zinc-400"
                >
                  {item.product_name}
                  {item.hs_code ? ` · ${item.hs_code}` : ""}
                  {item.similarity != null
                    ? ` · ${Math.round(item.similarity * 100)}%`
                    : ""}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </li>
  )
}

function questionOf(log: ConsultHistoryLog) {
  const fromOutput = log.output?.question
  const fromInput = log.input?.question
  if (typeof fromOutput === "string" && fromOutput.trim()) return fromOutput
  if (typeof fromInput === "string" && fromInput.trim()) return fromInput
  return "Soru kaydedilmedi"
}

function sessionIdOf(log: ConsultHistoryLog) {
  const fromInput = log.input?.session_id
  const fromOutput = log.output?.session_id
  if (typeof fromInput === "string" && fromInput.trim()) return fromInput
  if (typeof fromOutput === "string" && fromOutput.trim()) return fromOutput
  return null
}

function reportOf(log: ConsultHistoryLog) {
  const advice = log.output?.advice
  return typeof advice === "string" && advice.trim() ? advice : null
}

function formatWhen(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date)
}

function StatusBadge({ status }: { status: AgentLogStatus }) {
  const label =
    status === "success"
      ? "tamamlandı"
      : status === "error"
        ? "hata"
        : status
  return (
    <span
      className={cn(
        "rounded-full px-2 py-0.5",
        status === "success" && "bg-teal-400/12 text-teal-300",
        status === "error" && "bg-amber-400/12 text-amber-200",
        status !== "success" && status !== "error" && "bg-white/6 text-zinc-400"
      )}
    >
      {label}
    </span>
  )
}
