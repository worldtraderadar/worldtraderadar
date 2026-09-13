import { notifyQuotaChanged } from "@/lib/quota"
import { getSupabaseBrowserClient } from "@/lib/supabase/client"

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "")

async function authHeaders(): Promise<Record<string, string>> {
  const client = getSupabaseBrowserClient()
  if (!client) return {}
  const { data } = await client.auth.getSession()
  const token = data.session?.access_token
  if (!token) return {}
  return { Authorization: `Bearer ${token}` }
}

export type QuotaExceededDetail = {
  code: "quota_exceeded"
  reason: "search_limit" | "token_limit" | string
  message: string
  upgrade_path?: string
  plan_id?: string
  plan_name?: string
  usage?: { searches: number; tokens: number }
  limits?: { daily_searches: number; daily_tokens: number }
}

export class QuotaError extends Error {
  detail: QuotaExceededDetail
  constructor(detail: QuotaExceededDetail) {
    super(detail.message)
    this.name = "QuotaError"
    this.detail = detail
  }
}

export type BillingPlan = {
  id: string
  name: string
  daily_search_limit: number
  daily_token_limit: number
  monthly_price_usd: number
  features: string[]
}

export type BillingSnapshot = {
  account_id: string
  slug: string
  display_name: string
  plan_id: string
  plan_name: string
  usage_date: string
  backend: string
  monthly_price_usd: number
  features: string[]
  usage: { searches: number; tokens: number }
  limits: { daily_searches: number; daily_tokens: number }
  remaining: { searches: number; tokens: number }
}

export type CheckoutSession = {
  id: string
  plan_id: string
  plan_name: string
  amount_usd: number
  currency: string
  status: string
  limits: { daily_searches: number; daily_tokens: number }
}

function isQuotaDetail(value: unknown): value is QuotaExceededDetail {
  return (
    !!value &&
    typeof value === "object" &&
    "code" in value &&
    (value as { code: unknown }).code === "quota_exceeded" &&
    "message" in value
  )
}

function throwApiError(statusText: string, body: unknown, status?: number): never {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail
    if (isQuotaDetail(detail)) {
      notifyQuotaChanged()
      throw new QuotaError(detail)
    }
    const message =
      typeof detail === "string" ? detail : JSON.stringify(detail)
    if (status === 429) {
      throw new QuotaError({
        code: "quota_exceeded",
        reason: "search_limit",
        message,
        upgrade_path: "/paket",
      })
    }
    throw new Error(message)
  }
  throw new Error(statusText)
}

export type LockedFlags = {
  contact: boolean
  volume: boolean
}

export type PaywallMeta = {
  quantity?: number | null
  unit?: string | null
  value_usd?: number | null
  organization_name?: string | null
  contact_email?: string | null
  website?: string | null
  has_volume?: boolean
  has_contact?: boolean
  plan_id?: string | null
  locked?: LockedFlags
}

export type TradeItemMatch = {
  id: string
  organization_id: string | null
  hs_code: string | null
  product_name: string
  description: string | null
  origin_country: string | null
  destination_country: string | null
  similarity: number
} & PaywallMeta

export type AgentStep = {
  key: string
  agent: string
  label: string
  status: "pending" | "running" | "success" | "error" | string
  detail?: string
}

export type ChatTurn = {
  role: "user" | "assistant"
  content: string
}

export type ConsultResponse = {
  question: string
  advice: string
  embed_model: string
  chat_model: string
  context: TradeItemMatch[]
  run_id?: string | null
  intent?: string | null
  selected_agent?: string | null
  steps?: AgentStep[]
  speak_url?: string
  session_id?: string | null
  history?: ChatTurn[]
  session_title?: string | null
  product?: string | null
  capacity?: string | null
  market?: string | null
}

export type TtsStatus = {
  ready: boolean
  engine: string
  language: string
  device: string
  error: string | null
  cloud: boolean
}

export type HealthResponse = {
  status: string
  env_local?: {
    path: string
    exists: boolean
    NEXT_PUBLIC_SUPABASE_URL: boolean
    SUPABASE_SERVICE_ROLE_KEY: boolean
  }
  supabase: {
    configured: boolean
    connected: boolean
    error: string | null
  }
  ollama: {
    ok: boolean
    base_url: string
    embed_model: string
    chat_model: string
    bge_m3_available: boolean
    llama3_available: boolean
    error: string | null
  }
  tts?: TtsStatus
}

export type TradeItem = {
  id: string
  organization_id: string | null
  hs_code: string | null
  product_name: string
  description: string | null
  origin_country: string | null
  destination_country: string | null
  direction: "import" | "export" | null
  quantity: number | null
  unit: string | null
  value_usd: number | null
  trade_date: string | null
  source: string | null
  embedding_model: string
  created_at: string
} & PaywallMeta

export type Organization = {
  id: string
  name: string
  slug?: string | null
  country_code?: string | null
  city?: string | null
  organization_type?: string | null
  website?: string | null
  contact_email?: string | null
  tax_id?: string | null
  created_at?: string
} & PaywallMeta

export type TradeItemCreate = {
  product_name: string
  hs_code?: string
  description?: string
  origin_country?: string
  destination_country?: string
  direction?: "import" | "export"
  quantity?: number
  unit?: string
  value_usd?: number
  source?: string
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(await authHeaders()),
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    let body: unknown = null
    try {
      body = await response.json()
    } catch {
      throw new Error(response.statusText)
    }
    throwApiError(response.statusText, body, response.status)
  }

  const data = (await response.json()) as T
  return data
}

export function getApiUrl() {
  return API_URL
}

export function getHealth() {
  return request<HealthResponse>("/health")
}

export async function fetchSpeakAudio(text: string) {
  const response = await fetch("/api/speak", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "audio/wav",
      ...(await authHeaders()),
    },
    body: JSON.stringify({ text, lang: "tr" }),
    signal: AbortSignal.timeout(180_000),
  })
  if (!response.ok) {
    let detail = "Yerel ses üretilemedi."
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      /* wav/plain */
    }
    throw new Error(detail)
  }
  return response.blob()
}

export type ConsultPayload = {
  question: string
  session_id?: string | null
  history?: ChatTurn[]
  match_count?: number
  match_threshold?: number
}

function consultBody(input: string | ConsultPayload) {
  if (typeof input === "string") {
    return {
      question: input,
      match_count: 8,
      match_threshold: 0.35,
    }
  }
  return {
    question: input.question,
    session_id: input.session_id || undefined,
    history: input.history || [],
    match_count: input.match_count ?? 8,
    match_threshold: input.match_threshold ?? 0.35,
  }
}

export function consult(input: string | ConsultPayload) {
  return request<ConsultResponse>("/consult", {
    method: "POST",
    body: JSON.stringify(consultBody(input)),
    signal: AbortSignal.timeout(180_000),
  })
}

export async function consultStream(
  input: string | ConsultPayload,
  onEvent: (event: {
    type: "step" | "complete" | "error"
    step?: AgentStep
    result?: ConsultResponse
    detail?: string
  }) => void,
  signal?: AbortSignal
) {
  const response = await fetch(`${API_URL}/consult/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(await authHeaders()),
    },
    body: JSON.stringify(consultBody(input)),
    signal,
  })
  if (!response.ok || !response.body) {
    let body: unknown = null
    try {
      body = await response.json()
    } catch {
      throw new Error(response.statusText)
    }
    throwApiError(response.statusText, body, response.status)
  }

  const stream = response.body
  if (!stream) {
    throw new Error("Danışmanlık akışı okunamadı.")
  }

  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split("\n\n")
    buffer = chunks.pop() ?? ""
    for (const chunk of chunks) {
      const line = chunk
        .split("\n")
        .filter((row) => row.startsWith("data:"))
        .map((row) => row.slice(5).trim())
        .join("")
      if (!line) continue
      const parsed = JSON.parse(line) as {
        type: "step" | "complete" | "error"
        step?: AgentStep
        result?: ConsultResponse
        detail?: string
      }
      onEvent(parsed)
      if (parsed.type === "complete" || parsed.type === "error") {
        notifyQuotaChanged()
        return parsed.result
      }
    }
  }
  throw new Error("Danışmanlık akışı tamamlanamadı.")
}

export function searchTradeItems(query: string) {
  return request<{ query: string; model: string; results: TradeItemMatch[] }>(
    "/search",
    {
      method: "POST",
      body: JSON.stringify({ query, match_count: 12, match_threshold: 0.35 }),
    }
  )
}

export function listTradeItems(limit = 40) {
  return request<TradeItem[]>(`/trade-items?limit=${limit}`)
}

export function createTradeItem(item: TradeItemCreate) {
  return request<TradeItem>("/trade-items", {
    method: "POST",
    body: JSON.stringify(item),
    signal: AbortSignal.timeout(90_000),
  })
}

export type AgentLogStatus = "pending" | "running" | "success" | "error"

export type ConsultHistoryContext = {
  id?: string
  product_name?: string
  hs_code?: string | null
  origin_country?: string | null
  destination_country?: string | null
  similarity?: number
}

export type ConsultHistoryLog = {
  id: string
  agent_name: string
  action: string
  status: AgentLogStatus
  input: {
    question?: string
    session_id?: string | null
    match_count?: number
    match_threshold?: number
  } & Record<string, unknown>
  output: {
    question?: string
    session_id?: string | null
    advice?: string
    chat_model?: string
    embed_model?: string
    context_count?: number
    context?: ConsultHistoryContext[]
  } & Record<string, unknown>
  error_message: string | null
  duration_ms: number | null
  created_at: string
}

export async function listConsultHistory(limit = 50) {
  const capped = Math.min(Math.max(limit, 1), 200)
  try {
    const logs = await request<ConsultHistoryLog[]>(
      `/consult/history?limit=${capped}`,
      { signal: AbortSignal.timeout(8_000) }
    )
    return Array.isArray(logs) ? logs : []
  } catch {
    throw new Error("Kayıt bulunamadı")
  }
}

export function getBillingMe() {
  return request<BillingSnapshot>("/billing/me")
}

export type AuthMeResponse = {
  user_id: string
  account_id: string
  account_slug: string
  role: string
  plan_id: string
}

export function getAuthMe() {
  return request<AuthMeResponse>("/auth/me")
}

export function getBillingPlans() {
  return request<{ plans: BillingPlan[] }>("/billing/plans")
}

export function startCheckout(planId = "pro") {
  return request<CheckoutSession>("/billing/checkout", {
    method: "POST",
    body: JSON.stringify({ plan_id: planId }),
  })
}

export function confirmCheckout(checkoutId: string) {
  return request<{ checkout: CheckoutSession; account: BillingSnapshot }>(
    "/billing/checkout/confirm",
    {
      method: "POST",
      body: JSON.stringify({ checkout_id: checkoutId }),
    }
  )
}

export function changePlan(planId: "free" | "pro") {
  return request<BillingSnapshot>("/billing/plan", {
    method: "POST",
    body: JSON.stringify({ plan_id: planId }),
  })
}

export type ConsultSessionSnapshot = {
  session_id: string
  title: string
  product: string | null
  capacity: string | null
  market: string | null
  last_intent: string | null
  history: ChatTurn[]
}

export async function createConsultSession(): Promise<ConsultSessionSnapshot> {
  return request<ConsultSessionSnapshot>("/consult/sessions", {
    method: "POST",
    body: "{}",
    signal: AbortSignal.timeout(8_000),
  })
}

export function listConsultSessions() {
  return request<ConsultSessionSnapshot[]>("/consult/sessions", {
    signal: AbortSignal.timeout(8_000),
  })
}

export function getConsultSession(sessionId: string) {
  return request<ConsultSessionSnapshot>(
    `/consult/sessions/${encodeURIComponent(sessionId)}`,
    { signal: AbortSignal.timeout(8_000) }
  )
}

export function deleteConsultSession(sessionId: string) {
  return request<{ ok: boolean; session_id: string; deleted: boolean }>(
    `/consult/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "DELETE",
      signal: AbortSignal.timeout(8_000),
    }
  )
}

export function listSuppliers(limit = 40) {
  return request<{
    plan_id: string
    items: TradeItem[]
    organizations: Organization[]
  }>(`/suppliers?limit=${limit}`)
}
