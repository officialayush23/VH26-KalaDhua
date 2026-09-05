/// The console's session, issued by the engine.
///
/// There is no third-party identity provider here and that is deliberate: a cache that can
/// only be administered while some other service is reachable cannot be administered during
/// the incident you most need it in. The engine owns the accounts, signs the sessions and
/// verifies them itself, so the console needs exactly one piece of configuration -
/// VITE_AURA_URL - and nothing else.
///
/// The token lives in this browser. It is a bearer credential with a twelve-hour life, and
/// the engine ends every session signed under a password the moment that password changes.

const BASE = import.meta.env.VITE_AURA_URL || "http://localhost:8080"
const STORE_KEY = "aura.session"

let session = read()
const listeners = new Set()

function read() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    // An expired token is worse than none: it produces 401s that look like a broken engine.
    if (parsed?.expires_at && parsed.expires_at * 1000 < Date.now()) return null
    return parsed
  } catch {
    return null
  }
}

function write(next) {
  session = next
  try {
    if (next) window.localStorage.setItem(STORE_KEY, JSON.stringify(next))
    else window.localStorage.removeItem(STORE_KEY)
  } catch {
    // A browser that refuses storage still works for the length of this page view.
  }
  listeners.forEach((fn) => fn(session))
}

export function token() {
  return session?.token ?? null
}

export function currentUser() {
  return session?.user ?? null
}

export function isSignedIn() {
  return Boolean(session?.token)
}

export function watchSession(onChange) {
  listeners.add(onChange)
  onChange(session)
  return () => listeners.delete(onChange)
}

export function authHeaders() {
  const t = token()
  return t ? { authorization: `Bearer ${t}` } : {}
}

/// Sign in. The engine answers with one error for a wrong address and a wrong password, so
/// this cannot be used to find out who has an account.
export async function signIn(email, password) {
  const res = await fetch(`${BASE}/v1/auth/login`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password }),
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const err = new Error(body?.error || `sign-in failed (${res.status})`)
    err.fix = body?.fix
    throw err
  }
  write({ token: body.token, expires_at: body.expires_at, user: body.user })
  return body.user
}

export function signOut() {
  write(null)
}

/// What the engine will accept before anything is typed: whether it is enforcing at all,
/// and whether any account exists to sign in as. The difference matters - "wrong password"
/// and "nobody has been created yet" are different problems with different fixes.
export async function authState() {
  try {
    const res = await fetch(`${BASE}/v1/auth/me`, { headers: authHeaders() })
    return await res.json()
  } catch {
    return { enforced: false, accounts_exist: true, offline: true }
  }
}

export async function changePassword(email, currentPassword, newPassword) {
  const res = await fetch(`${BASE}/v1/auth/password`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders() },
    body: JSON.stringify({
      email,
      current_password: currentPassword,
      new_password: newPassword,
    }),
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body?.error || "could not change the password")
  // Every session signed under the old password is over, including this one.
  signOut()
  return body
}
