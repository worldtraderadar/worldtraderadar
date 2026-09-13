import { createClient, type SupabaseClient } from "@supabase/supabase-js"

let browserClient: SupabaseClient | null = null

export function getSupabaseBrowserClient(): SupabaseClient | null {
  const url = (process.env.NEXT_PUBLIC_SUPABASE_URL || "").trim()
  const anon = (process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "").trim()
  if (!url || !anon) return null
  if (browserClient) return browserClient
  browserClient = createClient(url, anon, {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: true,
    },
  })
  return browserClient
}

export function isSupabaseAuthConfigured() {
  return Boolean(
    (process.env.NEXT_PUBLIC_SUPABASE_URL || "").trim() &&
      (process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "").trim()
  )
}
