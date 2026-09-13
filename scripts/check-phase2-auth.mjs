/**
 * Phase 2 static checks: API client must not trust X-Account-Slug.
 * Run: node --experimental-strip-types scripts/check-phase2-auth.mjs
 * (or plain node on the emitted sources via fs read)
 */
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const apiTs = fs.readFileSync(path.join(root, "lib/api.ts"), "utf8")
const speak = fs.readFileSync(path.join(root, "app/api/speak/route.ts"), "utf8")

const errors = []

if (apiTs.includes("X-Account-Slug") || apiTs.includes("wtr-account-slug")) {
  errors.push("lib/api.ts still references account slug identity")
}
if (!apiTs.includes("Authorization") || !apiTs.includes("Bearer")) {
  errors.push("lib/api.ts missing Bearer Authorization")
}
if (speak.includes("X-Account-Slug") || speak.includes('"demo"')) {
  errors.push("speak route still uses slug/demo identity")
}
if (!speak.toLowerCase().includes("authorization")) {
  errors.push("speak route missing Authorization forwarding")
}

if (errors.length) {
  console.error("PHASE2 AUTH CHECK FAIL")
  for (const e of errors) console.error("-", e)
  process.exit(1)
}
console.log("PHASE2 AUTH CHECK PASS")
