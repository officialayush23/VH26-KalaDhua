# AURA — deployment

Written as steps you can follow top to bottom. Local first, because a hosted deploy that
was never run locally fails in a place where you cannot attach a debugger.

Status: the engine and the two applications read `PORT`, the engine image copies the right
binary, and the analytics app can reach Supabase over either the direct host or the
transaction pooler. Those four things were the blockers; they are fixed.

---

## 0. What gets deployed

| Piece | Image | Port | Needs |
|---|---|---|---|
| engine (`aura`) | `deploy/Dockerfile.engine` | `$PORT` or 8080 | Supabase URL + service key, to pull model bundles |
| recommendation | `deploy/Dockerfile.apps` | `$PORT` or 8101 | the engine's address |
| analytics | `deploy/Dockerfile.apps` | `$PORT` or 8102 | the engine's address, a Supabase Postgres DSN |
| dashboard | `deploy/Dockerfile.dashboard` | 80 | the engine's **public** URL, baked at build time |
| invalidator | `deploy/Dockerfile.apps` | none | the engine's address |
| simulator | `deploy/Dockerfile.apps` | none | the application addresses |

The content app still exists and still builds; it is behind the `content` compose profile
because recommendation (CPU heavy) and analytics (database heavy) are the two cost shapes
the argument needs.

---

## 1. Local, without Docker

Fastest loop while developing. Four terminals.

```powershell
# 1 — engine
cd D:\GITHUB\VH26-KalaDhua\engine
cargo run --release -p aura-server -- --real-values

# 2 — applications
cd D:\GITHUB\VH26-KalaDhua\apps
pip install -r requirements.txt
python -m recommendation.main      # :8101
python -m analytics.main           # :8102  (second window)

# 3 — traffic
cd D:\GITHUB\VH26-KalaDhua\apps
python -m simulator.driver --rps 120 --duration 300 `
  --endpoint recommendation=http://localhost:8101 `
  --endpoint analytics=http://localhost:8102

# 4 — dashboard
cd D:\GITHUB\VH26-KalaDhua\frontend\universe
npm run dev                        # http://localhost:5173
```

No `--scenario` on the engine. That keeps the internal generator off, so every request the
cache sees came from an application that actually did the work.

Analytics picks its database in this order: `SUPABASE_DIRECT_CONNECTION_URL`, then
`SUPABASE_TRANSACTION_URL`, then a local SQLite fixture. Check which one it chose:

```powershell
curl http://localhost:8102/health
```

`"dialect": "postgres"` means real Supabase. `"sqlite"` means it fell back and the
regeneration cost is a local fixture, which you must not present as a database result.

---

## 2. Local, with Docker

Needs Docker Desktop **running**, not just installed. `docker info` printing a `Server:`
block is the test; `where docker` finding the CLI is not.

```powershell
cd D:\GITHUB\VH26-KalaDhua
copy deploy\.env.example deploy\.env       # then edit
docker compose -f deploy/docker-compose.yml up --build
```

Build context is the repository root for every service, so run it from the root. Compose
reads secrets from `backend/.env`; that file must exist or compose refuses to start.

```powershell
docker compose -f deploy/docker-compose.yml --profile load up -d   # add the traffic
docker compose -f deploy/docker-compose.yml logs -f engine
docker compose -f deploy/docker-compose.yml down -v                # also drops the model volume
```

Engine on http://localhost:8080/healthz, dashboard on http://localhost:5173.

---

## 3. Railway

Railway builds the image on its own machines, so a local Docker daemon is not required to
deploy. It is required to *test the image* before you push, which is a different thing.

Three services in one project, all from the same GitHub repo.

**Service 1 — engine**

- Root directory: `/`
- Build: Dockerfile, path `deploy/Dockerfile.engine`
- Start command: leave empty. The image's `CMD` is correct and the engine reads `PORT`.
- Health check path: `/healthz`
- Variables:

```
RUST_LOG=info
AURA__CACHE__CAPACITY_BYTES=134217728
AURA__CAPACITY__HOST_BUDGET_BYTES=402653184
AURA__CAPACITY__AUTO=true
AURA__PREDICTOR__KIND=gbdt
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_SECRET_KEY=...
```

The capacity numbers matter. `--real-values` holds real bytes, and 800 MB of pool measured
1.19 GB resident. On a 512 MB container a 512 MB pool is an OOM kill, not a cache.

Generate a domain for this service. That URL is what the dashboard talks to.

**Service 2 — recommendation**

- Build: Dockerfile, path `deploy/Dockerfile.apps`
- Start command: `python -m recommendation.main`
- Variables: `AURA_APPS_AURA_BASE_URL=http://<engine-service>.railway.internal:8080`,
  `LOG_LEVEL=INFO`

Use the private network address, not the public one. The applications talk to the cache on
every request; sending that through the public internet doubles the latency you are trying
to measure.

**Service 3 — analytics**

- Same image, start command `python -m analytics.main`
- Variables: the same `AURA_APPS_AURA_BASE_URL`, plus
  `SUPABASE_DIRECT_CONNECTION_URL=...`

Railway has IPv6 egress, so the direct Supabase host works there. Set
`SUPABASE_TRANSACTION_URL` as well and the app falls back to the pooler on its own.

**Traffic.** Run the simulator from your laptop against the public URLs of the two
applications, or add it as a fourth service with `python -m simulator.driver` and the
internal endpoints. Local is easier to start and stop during a demo.

---

## 4. Render

`deploy/render.yaml` is a blueprint for the same three services. New → Blueprint, point it
at the repo, then fill the five secrets it marks `sync: false`.

The one difference from Railway: **Render containers have no IPv6 route**, and Supabase's
direct host `db.<ref>.supabase.co` is IPv6 only. On Render the analytics app must use the
transaction pooler DSN (`...pooler.supabase.com:6543`). The pool disables asyncpg's
prepared statement cache automatically when it sees that host, because pgbouncer in
transaction mode hands a different backend to every statement.

---

## 5. Console on Vercel

`VITE_AURA_URL` is substituted into the bundle at build time, so it is a build variable,
not a runtime one. Changing it means a rebuild.

- New project → import the repo
- **Root directory: `frontend/universe`**. `vercel.json` lives there and carries the
  single-page rewrite, without which a refresh on any panel returns 404.
- Framework preset: Vite (detected). Build `npm run build`, output `dist`
- Environment variables:
  - `VITE_AURA_URL=https://vh26-kaladhua.onrender.com`

  That is the only variable the console needs. Sign-in goes to the engine, so there is no
  second identity system to configure and nothing else to keep in step.

The live socket URL is derived by swapping the scheme, so an `https` engine gives `wss`
and there is no mixed-content block. An engine on plain `http` behind an `https` dashboard
will fail silently in the browser console; that is the one combination to avoid.

Alternative: deploy `deploy/Dockerfile.dashboard` as a fourth Railway service and pass
`VITE_AURA_URL` as a build argument. One platform instead of two, slower builds.

---

## 6. After the first deploy, verify in this order

```bash
curl https://<engine>/healthz            # up
curl https://<engine>/v1/supabase        # configured, reachable, active models
curl https://<engine>/v1/stats           # predictor should read gbdt, not heuristic
curl https://<recommendation>/health
curl https://<analytics>/health          # dialect must be postgres
```

If `/v1/stats` says `heuristic`, the engine could not pull bundles: check
`SUPABASE_SERVICE_ROLE_SECRET_KEY` on the engine service. The cache still works, it is just
running on the online logistic fallback, and you should not claim a trained model.

---

## 7. Authentication

Two callers, two credentials, and the engine owns both.

**People** sign in to the console with an email and a password the engine itself holds.
There is no third-party identity provider, deliberately: a cache that can only be
administered while some other service is reachable cannot be administered during the
incident you most need it in. Passwords are stored as PBKDF2-HMAC-SHA256, 120,000
iterations, per-user random salt, in `aura_users`. Sessions are twelve-hour tokens the
engine signs, and changing a password ends every session signed under the old one.

The root account comes from the environment on first boot and is written to the control
plane, so a recycled container does not lose the only way in:

```
AURA_ROOT_EMAIL=you@example.com
AURA_ROOT_PASSWORD=<at least 10 characters>
```

Only a root account can create or remove accounts. Apply `training/sql/008_users.sql`
first, or accounts will not survive a restart.

**Applications** carry a key: `Authorization: Bearer aura_sk_...`, minted on the console's
Connect tab and shown exactly once. The engine stores a SHA-256 hash and so does
`aura_api_keys`, so neither a memory dump nor a copy of the table is a working credential.
The key is also the application's identity: the engine attributes every request under it to
the application it was issued to and ignores any name in the request body. An application
key deliberately cannot reach a control route, so a leaked key cannot re-tune the cache for
everyone else.

`GET /v1/connections` is what the console's "Connected services" panel reads: which keys
exist, which have been used, how long ago each was last seen, and what its application has
actually done. It is an observation, not a registry.

Mode is explicit, never inferred:

```
AURA_AUTH=open       # every route answers. Local demos.
AURA_AUTH=enforced   # required for anything with a public address.
```

`/healthz`, `/metrics`, `/v1/auth/login` and `/v1/auth/me` stay public under enforcement -
the way in cannot be behind the door.

## 8. Known gaps at deploy time
- **Model bundles are not in the image.** `engine/models/` is gitignored, so a fresh
  container starts empty and pulls from Supabase on boot. That is the intended path, but it
  makes Supabase a boot dependency for the trained model.
- **`reuse_linear_h60s` does not parse** — `missing field bias`. Three GBDT bundles load
  fine. It is a trainer export bug and it logs a warning on every boot.
- **`/v1/nodes` returns one hardcoded node.** No consistent hashing, no ScaleOut.
- Free tiers sleep. A cold engine has an empty cache and a hit rate of zero for the first
  minute, which is a bad first thirty seconds of a demo. Warm it before you present.
