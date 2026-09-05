import { useEffect, useState } from "react"

import { authState, currentUser, isSignedIn, signIn, signOut, watchSession } from "@/lib/session"
import { cn } from "@/lib/utils"

export function useSession() {
  const [session, setSession] = useState(null)
  const [state, setState] = useState(null)

  useEffect(() => watchSession(setSession), [])

  // Asked once at load, and again whenever the session changes, because signing in is
  // exactly when "does this engine enforce anything" stops being hypothetical.
  useEffect(() => {
    let alive = true
    authState().then((s) => alive && setState(s))
    return () => {
      alive = false
    }
  }, [session])

  return {
    session,
    user: session?.user ?? currentUser(),
    signedIn: Boolean(session?.token) || isSignedIn(),
    enforced: Boolean(state?.enforced),
    accountsExist: state?.accounts_exist !== false,
  }
}

export function SessionChip({ user, signedIn }) {
  if (!signedIn) {
    return (
      <span className="rounded-xl border border-border bg-card px-3 py-2 text-[12.5px] text-muted-foreground">
        not signed in
      </span>
    )
  }
  return (
    <div className="flex items-center gap-2 rounded-xl border border-border bg-card px-3 py-2">
      <span className="h-1.5 w-1.5 rounded-full bg-primary" />
      <span className="text-[12.5px]">{user?.email ?? "signed in"}</span>
      {user?.role === "root" && (
        <span className="rounded bg-primary/15 px-1.5 text-[11px] font-semibold text-primary">
          root
        </span>
      )}
      <button
        onClick={signOut}
        className="text-[12.5px] font-medium text-muted-foreground hover:text-foreground"
      >
        sign out
      </button>
    </div>
  )
}

/// The whole console behind one form.
///
/// A half-loaded page that renders empty panels and a stream of 401s in the console is a
/// worse answer than a login screen: it looks like the engine is broken. So when the engine
/// enforces and there is no session, this is the page.
export function SignInScreen({ accountsExist }) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [fix, setFix] = useState(null)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    setFix(null)
    try {
      await signIn(email, password)
    } catch (err) {
      setError(err?.message ?? String(err))
      setFix(err?.fix ?? null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-svh items-center justify-center bg-background px-4">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <h1 className="text-[30px] font-semibold tracking-tight">AURA</h1>
          <p className="mt-1.5 text-[13.5px] text-muted-foreground">
            The cache is running and refusing anonymous callers, which is what it should be
            doing. Sign in to see it.
          </p>
        </div>

        <form
          onSubmit={submit}
          className="rounded-2xl border border-border bg-card p-6 shadow-sm"
        >
          {!accountsExist && (
            <div className="mb-4 rounded-xl border border-amber-400/45 bg-amber-400/10 px-3.5 py-3">
              <p className="text-[13px] font-semibold text-amber-300">
                This engine has no accounts yet
              </p>
              <p className="mt-1 text-[12.5px] leading-snug text-muted-foreground">
                Set <code className="font-mono">AURA_ROOT_EMAIL</code> and{" "}
                <code className="font-mono">AURA_ROOT_PASSWORD</code> on the service and
                restart it. The first boot creates the account and writes it to the control
                plane, so a redeploy will not lose it.
              </p>
            </div>
          )}

          <label className="block">
            <span className="text-[12.5px] font-medium text-muted-foreground">Email</span>
            <input
              type="email"
              required
              autoFocus
              autoComplete="username"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1.5 h-11 w-full rounded-xl border border-border bg-background px-3 text-[14px] outline-none focus:border-primary/60"
            />
          </label>

          <label className="mt-4 block">
            <span className="text-[12.5px] font-medium text-muted-foreground">Password</span>
            <input
              type="password"
              required
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1.5 h-11 w-full rounded-xl border border-border bg-background px-3 text-[14px] outline-none focus:border-primary/60"
            />
          </label>

          {error && (
            <div className="mt-4 rounded-xl border border-rose-400/45 bg-rose-400/10 px-3.5 py-2.5">
              <p className="text-[12.5px] text-rose-300">{error}</p>
              {fix && <p className="mt-1 text-[12px] text-muted-foreground">{fix}</p>}
            </div>
          )}

          <button
            type="submit"
            disabled={busy}
            className="mt-5 h-11 w-full rounded-xl bg-primary text-[14px] font-semibold text-primary-foreground disabled:opacity-50"
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <p className="mt-4 text-center text-[12px] text-muted-foreground">
          Sessions last twelve hours and end when the password changes.
        </p>
      </div>
    </div>
  )
}

export function AuthNotice({ enforced, signedIn }) {
  if (!enforced || signedIn) return null
  return (
    <div className={cn("mb-5 rounded-2xl border border-amber-400/45 bg-amber-400/[0.07] p-4")}>
      <h2 className="text-[14.5px] font-semibold">This engine requires a sign-in</h2>
      <p className="mt-1 text-[13px] text-muted-foreground">
        Telemetry and controls are refused until you sign in. Applications are unaffected:
        they carry their own keys.
      </p>
    </div>
  )
}
