import { NextRequest } from "next/server"

export async function POST(request: NextRequest) {
  const api = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
    /\/$/,
    ""
  )
  const body = await request.text()
  const upstream = await fetch(`${api}/speak`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "audio/wav",
      "X-Account-Slug": request.headers.get("X-Account-Slug") ?? "demo",
    },
    body,
    signal: AbortSignal.timeout(180_000),
  })

  const headers = new Headers()
  const contentType = upstream.headers.get("Content-Type")
  if (contentType) headers.set("Content-Type", contentType)
  const engine = upstream.headers.get("X-TTS-Engine")
  if (engine) headers.set("X-TTS-Engine", engine)
  const language = upstream.headers.get("X-TTS-Language")
  if (language) headers.set("X-TTS-Language", language)

  return new Response(upstream.body, {
    status: upstream.status,
    headers,
  })
}
