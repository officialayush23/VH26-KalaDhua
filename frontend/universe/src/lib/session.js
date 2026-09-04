import { createClient } from "@supabase/supabase-js"

/// The console's half of authentication.
///
/// People sign in through Supabase Auth in the browser; the engine verifies the token
/// Supabase issued and keeps no user table of its own. That means there is exactly one
/// account system, and this file never sees a password: it hands one to Supabase and holds
/// the resulting access token.
///
/// With no Supabase project configured the module reports itself unavailable rather than
/// throwing. A local engine runs open, and a console that refused to load without a login
/// server would make the local demo harder for no security gain.

const URL = import.meta.env.VITE_SUPABASE_URL
const ANON = import.meta.env.VITE_SUPABASE_ANON_KEY

export const authAvailable = Boolean(URL && ANON)

export const supabase = authAvailable
  ? createClient(URL, ANON, {
      auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
    })
  : null

let cached = null

/// The current access token, or null. Read synchronously by the socket and the fetch
/// helpers, which cannot await on every call.
export function token() {
  return cached?.access_token ?? null
}

export function currentUser() {
  return cached?.user ?? null
}

/// Start watching the session. Returns an unsubscribe function. The callback fires once
/// immediately with whatever is already stored, so a reload does not flash a login screen at
/// someone who is signed in.
export function watchSession(onChange) {
  if (!supabase) {
    onChange(null)
    return () => {}
  }
  supabase.auth.getSession().then(({ data }) => {
    cached = data?.session ?? null
    onChange(cached)
  })
  const { data } = supabase.auth.onAuthStateChange((_event, session) => {
    cached = session ?? null
    onChange(cached)
  })
  return () => data?.subscription?.unsubscribe?.()
}

export async function signIn(email, password) {
  if (!supabase) throw new Error("no Supabase project is configured for this console")
  const { data, error } = await supabase.auth.signInWithPassword({ email, password })
  if (error) throw error
  cached = data.session
  return data.session
}

export async function signOut() {
  if (!supabase) return
  await supabase.auth.signOut()
  cached = null
}

/// Headers for a call to the engine. Absent when signed out, which is correct: an engine
/// running open will answer anyway, and one running enforced should say so rather than be
/// handed an empty credential.
export function authHeaders() {
  const t = token()
  return t ? { authorization: `Bearer ${t}` } : {}
}
