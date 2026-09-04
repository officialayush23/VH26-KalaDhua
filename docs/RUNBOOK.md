# AURA — runbook

Exact steps, in order. Assumes Windows with the repository at `D:\GITHUB\VH26-KalaDhua`.

---

## Step 0 — install what you need

You need three things. Check each:

```powershell
cargo --version     # Rust. If missing: https://rustup.rs
node --version      # Node 20+. If missing: https://nodejs.org
python --version    # Python 3.11+. Only needed for the example apps.
```

If `cargo` is missing, install rustup, close the terminal, open a new one, and check again.
Nothing else about Rust matters to you — you never edit it, you only run it.

---

## Step 1 — start the engine

```powershell
cd D:\GITHUB\VH26-KalaDhua\engine
cargo run --release -p aura-server -- --scenario mixed_production
```

The first build takes **three to six minutes** and prints hundreds of `Compiling` lines.
That is normal. It is done when you see:

```
INFO aura: aura listening addr=0.0.0.0:8080
```

Leave this window open. Every later run of the same command starts in about a second.

**Check it:** open `http://localhost:8080/healthz`. You should get `{"ok":true,...}`.

Useful flags:

| Flag | Effect |
|---|---|
| `--scenario expensive_tail` | Start on the pattern where AURA wins most clearly. |
| `--scenario` omitted | No simulated traffic. The cache only holds what applications put in it. |
| `--bind 0.0.0.0:9000` | Different port. Then set `VITE_AURA_URL` for the dashboard. |
| `--models models` | Folder to load trained bundles from. Defaults to `engine/models`. |

---

## Step 2 — start the dashboard

New terminal:

```powershell
cd D:\GITHUB\VH26-KalaDhua\frontend\universe
npm install
npm run dev
```

Open `http://localhost:5173`.

If it says **engine offline**, step 1 is not running or not finished. The page is fine; it
just has nothing to talk to.

---

## Step 3 — Supabase, in this order

Go to your project, open the **SQL Editor**, and run these two files. Order matters: the
second one fills tables the first one creates.

1. `training/sql/003_supabase_schema.sql`
2. `training/sql/004_supabase_seed.sql`

Paste the whole file, press Run, wait for it to finish, then do the next one. The seed takes
about a minute on Supabase. When it finishes it prints a table you should check:

| table_name | count |
|---|---|
| regions | 25 |
| products | 4000 |
| customers | 40000 |
| orders | 120000 |
| order_items | 360000 |

Both files are safe to run more than once. The schema skips what already exists; the seed
clears the `app_*` tables and refills them.

**Ignore `001_schema.sql` and `002_seed_analytics.sql`.** They are the earlier versions.
`003` and `004` replace them and match what the code actually writes.

### Connecting the engine to it

`backend/.env` already holds your keys. The engine finds it automatically. Restart the
engine and check:

```
http://localhost:8080/v1/supabase
```

You want `"configured": true, "reachable": true`. The dashboard shows the same thing in its
Supabase panel.

If it says configured but unreachable, the key is wrong or the project is paused. If it says
not configured, the engine did not find `backend/.env` — start it from inside the repository,
not from somewhere else.

---

## Step 4 — train the model (optional for the demo, needed for the full story)

The engine works without this. It runs on a logistic model that learns online, and the
dashboard will say `heuristic` where it would otherwise say `gbdt`.

1. Upload `training/notebooks/aura_training_colab.ipynb` to Google Colab.
2. In Colab, open the key icon in the left sidebar and add two secrets:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_SECRET_KEY`
3. Run every cell top to bottom.

**About the dataset.** You do not need to find one. Cell 6 tries to fetch public traces and
falls back to `aura_train/synthetic.py`, which generates traffic with the same structure the
Rust simulator produces. That is the intended path. The labels are built by looking forward
in the stream: was this key requested again within 10 s, 60 s, 600 s.

What the cells do, in groups:

| Cells | What happens |
|---|---|
| 0–4 | Install packages, pull the training package out of the repo. |
| 5–6 | Get or generate traces. |
| 7–10 | Build the dataset. The split is by traffic regime, not random — a random split leaks the future into training. |
| 11–13 | Train the model plus ablations, so you can say which features earn their place. |
| 14–19 | Evaluate. AUC and PR-AUC, then a cache replay, because a model can score well and still not improve the cache. |
| 20–22 | Export `model_bundle.json` and check it against the schema the Rust side expects. |
| 23–25 | Upload to Supabase Storage and register it in `aura_models`. |
| 26–27 | Tell a running engine to pick it up. |

The last cell posts to `http://localhost:8080/v1/model/reload`, which Colab cannot reach on
your machine. Skip it and do this locally instead:

```powershell
curl -X POST http://localhost:8080/v1/model/reload -H "content-type: application/json" -d "{\"source\":\"supabase\"}"
```

The engine downloads the active bundle from Supabase and swaps it in without restarting.
Alternatively drop the exported `.json` files into `engine/models/` and restart.

---

## Step 5 — the example applications (optional)

These are what make a cache miss cost a real database query rather than a simulated one.

```powershell
cd D:\GITHUB\VH26-KalaDhua\apps
pip install -r requirements.txt
python -m driver.run_universe
```

Starts three services on 8101, 8102, 8103. The analytics one queries the `app_*` tables you
seeded in step 3, so its miss cost is a real Supabase query.

---

## The demo, in order

1. Dashboard open, engine running on `mixed_production`.
2. Point at **Requests served from cache** and **Money not spent**. Those are the headline.
3. Press **Scan**. Watch the traffic pattern flip to `Scan`, refusals climb in the live
   decisions feed, and the hit rate hold instead of collapsing. That is scan resistance:
   a plain LRU would flush its working set here.
4. Press **Cost spike**. Watch the policy blend shift toward the cost-aware experts.
5. Scroll to **Should we buy more memory?** and read the sentence at the bottom. It is a
   dollars-per-hour argument, not a utilisation rule.
6. Scroll to **Head to head benchmark**, choose `expensive_tail`, press Run. Wait a few
   seconds. AURA has the lowest cost. Point at the Belady row and say that is the offline
   optimum no online policy can reach.

---

## When something goes wrong

**`cargo` is not recognised.** Rust is not installed or the terminal predates the install.
Install rustup, open a new terminal.

**Build fails on `reqwest` or `ring`.** You need the MSVC build tools. Install "Desktop
development with C++" from the Visual Studio Installer.

**Port 8080 already in use.** `--bind 0.0.0.0:8090`, then create
`frontend/universe/.env.local` containing `VITE_AURA_URL=http://localhost:8090`.

**Dashboard says offline but the engine is running.** Check
`http://localhost:8080/healthz` in a browser. If that works and the page still says offline,
you changed the port and did not set `VITE_AURA_URL`.

**Supabase SQL errors on `storage.buckets`.** Harmless. It means your project layout differs;
the buckets get created by the client when the notebook uploads.

**Benchmark returns 500.** Lower `requests` — it runs in memory and 400k requests through
five policies needs a lot of it.
