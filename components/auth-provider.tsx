"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react"
import type { Session, User } from "@supabase/supabase-js"
import {
  getSupabaseBrowserClient,
  isSupabaseAuthConfigured,
} from "@/lib/supabase/client"

export type AuthStatus = "loading" | "authenticated" | "unauthenticated"

export type AccountContext = {
  user_id: string
  account_id: string
  account_slug: string
  role: string
  plan_id: string
}

type AuthContextValue = {
  status: AuthStatus
  session: Session | null
  user: User | null
  accessToken: string | null
  account: AccountContext | null
  configured: boolean
  signIn: (email: string, password: string) => Promise<void>
  signOut: () => Promise<void>
  refreshAccount: () => Promise<void>
  getAccessToken: () => Promise<string | null>
}

const AuthContext = createContext<AuthContextValue | null>(null)

async function fetchAccountContext(
  accessToken: string
): Promise<AccountContext | null> {
  const api = (
    process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
  ).replace(/\/$/, "")
  try {
    const response = await fetch(`${api}/auth/me`, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
        Accept: "application/json",
      },
    })
    if (!response.ok) return null
    return (await response.json()) as AccountContext
  } catch {
    return null
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const configured = isSupabaseAuthConfigured()
  const [status, setStatus] = useState<AuthStatus>(() =>
    configured ? "loading" : "unauthenticated"
  )
  const [session, setSession] = useState<Session | null>(null)
  const [account, setAccount] = useState<AccountContext | null>(null)

  const applySession = useCallback(async (next: Session | null) => {
    setSession(next)
    if (!next?.access_token) {
      setAccount(null)
      setStatus("unauthenticated")
      return
    }
    setStatus("authenticated")
    const ctx = await fetchAccountContext(next.access_token)
    setAccount(ctx)
  }, [])

  useEffect(() => {
    if (!configured) return
    const client = getSupabaseBrowserClient()
    if (!client) return

    let cancelled = false
    client.auth.getSession().then(({ data }) => {
      if (cancelled) return
      void applySession(data.session)
    })

    const { data: sub } = client.auth.onAuthStateChange((_event, next) => {
      void applySession(next)
    })

    return () => {
      cancelled = true
      sub.subscription.unsubscribe()
    }
  }, [configured, applySession])

  const signIn = useCallback(async (email: string, password: string) => {
    const client = getSupabaseBrowserClient()
    if (!client) {
      throw new Error(
        "Supabase Auth yapılandırılmamış (NEXT_PUBLIC_SUPABASE_URL / ANON_KEY)."
      )
    }
    const { error } = await client.auth.signInWithPassword({ email, password })
    if (error) throw error
  }, [])

  const signOut = useCallback(async () => {
    const client = getSupabaseBrowserClient()
    if (client) {
      await client.auth.signOut()
    }
    setSession(null)
    setAccount(null)
    setStatus("unauthenticated")
  }, [])

  const refreshAccount = useCallback(async () => {
    const token = session?.access_token
    if (!token) {
      setAccount(null)
      return
    }
    setAccount(await fetchAccountContext(token))
  }, [session])

  const getAccessToken = useCallback(async () => {
    const client = getSupabaseBrowserClient()
    if (!client) return session?.access_token ?? null
    const { data } = await client.auth.getSession()
    return data.session?.access_token ?? null
  }, [session])

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      session,
      user: session?.user ?? null,
      accessToken: session?.access_token ?? null,
      account,
      configured,
      signIn,
      signOut,
      refreshAccount,
      getAccessToken,
    }),
    [
      status,
      session,
      account,
      configured,
      signIn,
      signOut,
      refreshAccount,
      getAccessToken,
    ]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error("useAuth AuthProvider içinde kullanılmalı.")
  }
  return ctx
}
