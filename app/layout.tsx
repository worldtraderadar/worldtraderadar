import type { Metadata, Viewport } from "next"
import { Geist, Geist_Mono, Newsreader } from "next/font/google"
import { AppShell } from "@/components/app-shell"
import "./globals.css"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin", "latin-ext"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

const newsreader = Newsreader({
  variable: "--font-newsreader",
  subsets: ["latin", "latin-ext"],
  style: ["normal", "italic"],
})

export const metadata: Metadata = {
  title: "World Trade Radar",
  description:
    "BGE-M3 ve llama3 ile stratejik dış ticaret danışmanlığı.",
  applicationName: "World Trade Radar",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Trade Radar",
  },
  formatDetection: {
    telephone: false,
  },
}

export const viewport: Viewport = {
  themeColor: "#061018",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  colorScheme: "dark",
}

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="tr"
      className={`dark ${geistSans.variable} ${geistMono.variable} ${newsreader.variable} h-full antialiased`}
    >
      <body className="radar-body flex min-h-full flex-col">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  )
}
