import { useEffect, useState } from "react"

import { authAvailable, currentUser, signIn, signOut, watchSession } from "@/lib/session"
import { cn } from "@/lib/utils"

/// Sign in with the account system the project already has.
///
/// The console holds no user table and never sees a password twice: the form hands the
/// credentials to Supabase Auth, Supabase issues a token, and the engine verifies that
/// token against the project's JWT secret. One account system, one place to revoke someone.

export function useSession() {
  const [session, setSession] = useState(null)
  const [ready, setReady] = useState(!authAvailable)

  useEffect(() => {
    const stop = watchSession((s) => {
      setSession(s)
      setReady(true)
    })
    return stop
  }, [])

  return { session, ready, user: session?.user ?? currentUser() }
}

export function SessionChip({ session }) {
  const [open, setOpen] = useState(false)

  if (!authAvailable) {
    return (
      <span className="rounded-xl border border-border bg-card px-3 py-2 text-[12.5px] text-muted-foreground">
        local console
      </span>
    )
  }

  if (session) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-border bg-card px-3 py-2">
        <span className="h-1.5 w-1.5 rounded-full bg-primary" />
        <span className="text-[12.5px]">{session.user?.email ?? "signed in"}</span>
        <button
          onClick={() => signOut()}
          className="text-[12.5px] font-medium text-muted-foreground hover:text-foreground"
        >
          sign out
        </button>
      </div>
    )
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="rounded-xl bg-primary px-3.5 py-2 text-[13px] font-semibold text-primary-foreground"
      >
        Sign in
      </button>
      {open && <SignInDialog onClose={() => setOpen(false)} />}
    </>
  )
}

function SignInDialog({ onClose }) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await signIn(email, password)
      onClose()
    } catch (err) {
      setError(err?.message ?? String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
        className="w-full max-w-sm rounded-2xl border border-border bg-card p-5 shadow-xl"
      >
        <h2 className="text-[16px] font-semibold">Sign in to the console</h2>
        <p className="mt-1 text-[12.5px] text-muted-foreground">
          Your Supabase account. The engine verifies the token Supabase issues; this page
          never stores a password.
        </p>

        <label className="mt-4 block">
          <span className="text-[12.5px] font-medium text-muted-foreground">Email</span>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mt-1.5 h-10 w-full rounded-xl border border-border bg-background px-3 text-[14px] outline-none focus:border-primary/60"
          />
        </label>
        <label className="mt-3 block">
          <span className="text-[12.5px] font-medium text-muted-foreground">Password</span>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mt-1.5 h-10 w-full rounded-xl border border-border bg-background px-3 text-[14px] outline-none focus:border-primary/60"
          />
        </label>

        {error && (
          <p className="mt-3 rounded-lg border border-rose-400/40 bg-rose-400/10 px-3 py-2 text-[12.5px] text-rose-300">
            {error}
          </p>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-border px-3 py-1.5 text-[13px]"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={busy}
            className="rounded-lg bg-primary px-3.5 py-1.5 text-[13px] font-semibold text-primary-foreground disabled:opacity-50"
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </div>
      </form>
    </div>
  )
}

/// Shown when the engine requires a credential and the console does not have one. It is a
/// notice rather than a wall: the page still renders whatever the engine will serve, so an
/// operator can see that the engine is up and refusing them, which is a different problem
/// from the engine being down.
export function AuthNotice({ enforced, session }) {
  if (!enforced || session) return null
  return (
    <div className={cn("mb-5 rounded-2xl border border-amber-400/40 bg-amber-400/[0.07] p-4")}>
      <h2 className="text-[14.5px] font-semibold">This engine requires a sign-in</h2>
      <p className="mt-1 text-[13px] text-muted-foreground">
        Telemetry and controls are refused until you sign in. Applications are unaffected:
        they carry their own keys.
      </p>
    </div>
  )
}
