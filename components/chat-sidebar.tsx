"use client"

import { useEffect, useState } from "react"
import { ChevronLeft, Menu, MessageSquare, PanelLeft, Plus, Trash2, X } from "lucide-react"
import { cn } from "@/lib/utils"
import type { StoredChat } from "@/lib/consult-sessions"

export function ChatSidebar({
  chats,
  activeId,
  drawerOpen,
  collapsed,
  onDrawerOpenChange,
  onCollapsedChange,
  onNewChat,
  onSelect,
  onDelete,
}: {
  chats: StoredChat[]
  activeId: string
  drawerOpen: boolean
  collapsed: boolean
  onDrawerOpenChange: (open: boolean) => void
  onCollapsedChange: (collapsed: boolean) => void
  onNewChat: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}) {
  const [isDesktop, setIsDesktop] = useState(false)
  const panelVisible = isDesktop ? !collapsed : drawerOpen

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)")
    const sync = () => setIsDesktop(mq.matches)
    sync()
    mq.addEventListener("change", sync)
    return () => mq.removeEventListener("change", sync)
  }, [])

  useEffect(() => {
    if (!drawerOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onDrawerOpenChange(false)
    }
    const previous = document.body.style.overflow
    document.body.style.overflow = "hidden"
    window.addEventListener("keydown", onKey)
    return () => {
      document.body.style.overflow = previous
      window.removeEventListener("keydown", onKey)
    }
  }, [drawerOpen, onDrawerOpenChange])

  return (
    <>
      <div className="flex items-center gap-2 lg:hidden">
        <button
          type="button"
          aria-expanded={drawerOpen}
          aria-controls="consult-chat-sidebar"
          onClick={() => onDrawerOpenChange(true)}
          className="inline-flex min-h-10 items-center gap-2 rounded-xl border border-white/10 bg-white/4 px-3 text-sm text-zinc-200"
        >
          <Menu className="size-4 text-teal-300" />
          Sohbetler
        </button>
        <button
          type="button"
          onClick={onNewChat}
          className="inline-flex min-h-10 flex-1 items-center justify-center gap-2 rounded-xl border border-teal-400/30 bg-teal-400/10 px-3 text-sm text-teal-100"
        >
          <Plus className="size-4" />
          Yeni sohbet
        </button>
      </div>

      {drawerOpen ? (
        <button
          type="button"
          aria-label="Sohbet menüsünü kapat"
          className="fixed inset-0 z-40 bg-black/55 backdrop-blur-[2px] lg:hidden"
          onClick={() => onDrawerOpenChange(false)}
        />
      ) : null}

      {collapsed ? (
        <button
          type="button"
          aria-label="Sohbet menüsünü göster"
          onClick={() => onCollapsedChange(false)}
          className="sticky top-20 hidden size-11 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-[#08141c] text-teal-200 transition hover:border-teal-400/30 hover:bg-teal-400/10 lg:flex"
        >
          <PanelLeft className="size-4" />
        </button>
      ) : null}

      <aside
        id="consult-chat-sidebar"
        aria-hidden={!panelVisible}
        inert={!panelVisible}
        className={cn(
          "flex min-w-0 shrink-0 flex-col overflow-hidden border-white/8 bg-[#08141c]/96",
          "max-lg:fixed max-lg:inset-y-0 max-lg:left-0 max-lg:z-50 max-lg:w-72 max-lg:max-w-[85vw] max-lg:border-r max-lg:pt-[env(safe-area-inset-top)] max-lg:shadow-2xl max-lg:transition-transform max-lg:duration-300 max-lg:ease-out",
          drawerOpen
            ? "max-lg:translate-x-0"
            : "max-lg:pointer-events-none max-lg:-translate-x-full",
          "lg:sticky lg:top-20 lg:max-h-[calc(100vh-7.5rem)] lg:rounded-3xl lg:transition-[width,opacity] lg:duration-300 lg:ease-out",
          collapsed
            ? "lg:w-0 lg:border-0 lg:opacity-0 lg:pointer-events-none"
            : "lg:w-64 lg:border lg:opacity-100"
        )}
      >
        <div className="flex items-center gap-2 border-b border-white/8 p-3">
          <button
            type="button"
            onClick={onNewChat}
            className="flex min-h-11 min-w-0 flex-1 items-center justify-center gap-2 rounded-xl border border-teal-400/30 bg-teal-400/12 text-sm font-medium text-teal-100 transition hover:bg-teal-400/20"
          >
            <Plus className="size-4" />
            Yeni sohbet
          </button>
          <button
            type="button"
            aria-label="Sohbet menüsünü gizle"
            onClick={() => onCollapsedChange(true)}
            className="hidden size-11 shrink-0 items-center justify-center rounded-xl border border-white/10 text-zinc-400 transition hover:border-teal-400/25 hover:text-teal-200 lg:flex"
          >
            <ChevronLeft className="size-4" />
          </button>
          <button
            type="button"
            aria-label="Sohbet menüsünü kapat"
            onClick={() => onDrawerOpenChange(false)}
            className="flex size-11 shrink-0 items-center justify-center rounded-xl border border-white/10 text-zinc-300 lg:hidden"
          >
            <X className="size-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {chats.length === 0 ? (
            <p className="px-3 py-6 text-center text-xs leading-5 text-zinc-500">
              Henüz sohbet yok. Yeni sohbet ile başlayın.
            </p>
          ) : (
            <ul className="space-y-1">
              {chats.map((chat) => {
                const active = chat.id === activeId
                return (
                  <li key={chat.id} className="flex items-start gap-0.5">
                    <button
                      type="button"
                      onClick={() => onSelect(chat.id)}
                      className={cn(
                        "flex min-w-0 flex-1 items-start gap-2 rounded-xl px-3 py-2.5 text-left transition",
                        active
                          ? "border border-teal-400/25 bg-teal-400/10 text-zinc-50"
                          : "border border-transparent text-zinc-400 hover:bg-white/5 hover:text-zinc-200"
                      )}
                    >
                      <MessageSquare
                        className={cn(
                          "mt-0.5 size-3.5 shrink-0",
                          active ? "text-teal-300" : "text-zinc-600"
                        )}
                      />
                      <span className="min-w-0">
                        <span className="block truncate text-sm leading-5">
                          {chat.title || "Yeni sohbet"}
                        </span>
                        {chat.history.length > 0 ? (
                          <span className="mt-0.5 block font-mono text-[10px] tracking-wide text-zinc-600">
                            {chat.history.filter((turn) => turn.role === "user").length}{" "}
                            mesaj
                          </span>
                        ) : (
                          <span className="mt-0.5 block font-mono text-[10px] text-zinc-600">
                            Boş oturum
                          </span>
                        )}
                      </span>
                    </button>
                    <button
                      type="button"
                      aria-label="Sohbeti sil"
                      onClick={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        onDelete(chat.id)
                      }}
                      className="mt-1.5 flex size-8 shrink-0 items-center justify-center rounded-lg text-zinc-600 transition hover:bg-red-400/10 hover:text-red-300"
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </aside>
    </>
  )
}
