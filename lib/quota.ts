export const QUOTA_EVENT = "wtr:quota"

export function notifyQuotaChanged() {
  if (typeof window === "undefined") return
  window.dispatchEvent(new Event(QUOTA_EVENT))
}
