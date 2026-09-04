import { renderAppIcon } from "@/lib/app-icon"

export const contentType = "image/png"

export function GET() {
  return renderAppIcon(512)
}
