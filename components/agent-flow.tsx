"use client"

import { Check, CircleAlert, LoaderCircle, Waypoints } from "lucide-react"
import type { AgentStep } from "@/lib/api"
import { cn } from "@/lib/utils"

const AGENT_TONE: Record<string, string> = {
  orchestrator: "text-teal-300 border-teal-400/20 bg-teal-400/8",
  chat: "text-rose-200 border-rose-400/20 bg-rose-400/8",
  intake: "text-cyan-200 border-cyan-400/20 bg-cyan-400/8",
  sector_chat: "text-lime-200 border-lime-400/20 bg-lime-400/8",
  trade_advisor: "text-sky-300 border-sky-400/20 bg-sky-400/8",
  buyer_finder: "text-amber-200 border-amber-400/20 bg-amber-400/8",
  supplier_finder: "text-amber-200 border-amber-400/20 bg-amber-400/8",
  product_matching: "text-violet-300 border-violet-400/20 bg-violet-400/8",
}

export function AgentFlow({ steps }: { steps: AgentStep[] }) {
  if (steps.length === 0) {
    return (
      <div className="rounded-2xl border border-white/8 bg-[#0b1c27]/70 p-5 text-sm text-zinc-500">
        Orchestrator bekleniyor…
      </div>
    )
  }

  return (
    <section className="rounded-2xl border border-white/8 bg-[#0b1c27]/70 p-5">
      <div className="mb-4 flex items-center gap-2 text-teal-300">
        <Waypoints className="size-4" />
        <h2 className="text-xs tracking-[0.18em] uppercase">Ajan akışı</h2>
      </div>
      <ol className="space-y-3">
        {steps.map((step, index) => (
          <li key={`${step.agent}-${step.key}-${index}`} className="flex gap-3">
            <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center">
              {step.status === "running" ? (
                <LoaderCircle className="size-3.5 animate-spin text-teal-300" />
              ) : step.status === "error" ? (
                <CircleAlert className="size-3.5 text-amber-300" />
              ) : (
                <Check className="size-3.5 text-teal-400" />
              )}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={cn(
                    "rounded-full border px-2 py-0.5 font-mono text-[10px]",
                    AGENT_TONE[step.agent] ?? "text-zinc-400 border-white/10"
                  )}
                >
                  {step.agent}
                </span>
                <p className="text-sm text-zinc-100">{step.label}</p>
              </div>
              {step.detail ? (
                <p className="mt-1 text-xs leading-5 text-zinc-500">{step.detail}</p>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
