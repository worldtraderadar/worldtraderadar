import { ImageResponse } from "next/og"

export function renderAppIcon(size: number) {
  const fontSize = Math.round(size * 0.42)
  const inner = Math.round(size * 0.72)
  const radius = Math.round(size * 0.18)

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#061018",
        }}
      >
        <div
          style={{
            width: inner,
            height: inner,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#0b1c27",
            color: "#2dd4bf",
            fontSize,
            fontWeight: 700,
            borderRadius: radius,
            border: `${Math.max(2, Math.round(size * 0.012))}px solid #2dd4bf55`,
          }}
        >
          R
        </div>
      </div>
    ),
    { width: size, height: size }
  )
}
