//! Supabase client for the control plane.
//!
//! The cache data path deliberately does not go through here. Putting Postgres between
//! the cache and every request would defeat the point of the cache: on a miss it is the
//! application service that queries Supabase, measures what that cost, and hands the
//! measured cost back on the PUT. This module covers the other three jobs, which are all
//! control plane and all off the hot path:
//!
//!   1. pull the active model bundle out of Storage on boot or on demand
//!   2. publish a benchmark run and its per-policy rows
//!   3. append events
//!
//! Every call is fire-and-report: a Supabase outage degrades the engine to local files
//! and local telemetry, and never fails a cache request.

use std::time::Duration;

use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Value};

const MODELS_TABLE: &str = "aura_models";
const RUNS_TABLE: &str = "aura_benchmark_runs";
const RESULTS_TABLE: &str = "aura_benchmark_results";
const EVENTS_TABLE: &str = "aura_events";
const AUDIT_TABLE: &str = "aura_audit_log";
const KEYS_TABLE: &str = "aura_api_keys";
const USERS_TABLE: &str = "aura_users";
const MODEL_BUCKET: &str = "aura-models";

#[derive(Debug, Clone)]
pub struct Supabase {
    base_url: String,
    key: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct ModelRow {
    name: String,
    version: String,
    storage_path: String,
    #[serde(default)]
    kind: String,
}

impl Supabase {
    /// Reads `SUPABASE_URL` and a service key from the environment. Returns `None` when
    /// either is absent, which is the normal state for a local run with no cloud project.
    pub fn from_env() -> Option<Self> {
        let base_url = first_env(&["SUPABASE_URL"])?.trim_end_matches('/').to_string();
        let key = first_env(&[
            "SUPABASE_SERVICE_ROLE_SECRET_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_ANON_PUBLIC_KEY",
            "SUPABASE_KEY",
        ])?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .ok()?;
        Some(Self { base_url, key, client })
    }

    pub fn project(&self) -> &str {
        &self.base_url
    }

    fn rest(&self, table: &str) -> String {
        format!("{}/rest/v1/{table}", self.base_url)
    }

    fn req(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder
            .header("apikey", &self.key)
            .header("authorization", format!("Bearer {}", self.key))
            .header("content-type", "application/json")
    }

    /// Downloads the bundle whose `is_active` row wins for `name`. The Postgres row is the
    /// source of truth for which version is live; Storage only holds the bytes.
    pub async fn active_bundle(&self, name: &str) -> anyhow::Result<(String, Vec<u8>)> {
        let url = self.rest(MODELS_TABLE);
        let res = self
            .req(self.client.get(&url))
            .query(&[
                ("name", format!("eq.{name}")),
                ("is_active", "eq.true".to_string()),
                ("select", "name,version,storage_path,kind".to_string()),
                ("limit", "1".to_string()),
            ])
            .send()
            .await?;
        if !res.status().is_success() {
            anyhow::bail!("model lookup failed: {} {}", res.status(), res.text().await.unwrap_or_default());
        }
        let rows: Vec<ModelRow> = res.json().await?;
        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no active model row for {name}"))?;

        // storage_path is stored as `bucket/key`; strip the bucket if it is present so the
        // same value works whether the writer included it or not.
        let key = row
            .storage_path
            .strip_prefix(&format!("{MODEL_BUCKET}/"))
            .unwrap_or(&row.storage_path)
            .to_string();
        let obj_url = format!("{}/storage/v1/object/{MODEL_BUCKET}/{key}", self.base_url);
        let obj = self.req(self.client.get(&obj_url)).send().await?;
        if !obj.status().is_success() {
            anyhow::bail!("bundle download failed: {} {}", obj.status(), key);
        }
        let bytes = obj.bytes().await?.to_vec();
        tracing::info!(model = %row.name, version = %row.version, kind = %row.kind, "pulled bundle from supabase");
        Ok((row.version, bytes))
    }

    /// Lists every active bundle, so a reload can pick up all three horizons at once.
    pub async fn active_models(&self) -> anyhow::Result<Vec<String>> {
        let url = self.rest(MODELS_TABLE);
        let res = self
            .req(self.client.get(&url))
            .query(&[
                ("is_active", "eq.true".to_string()),
                ("select", "name".to_string()),
            ])
            .send()
            .await?;
        let rows: Vec<Value> = res.json().await?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r.get("name").and_then(|n| n.as_str()).map(String::from))
            .collect())
    }

    /// Publishes a benchmark report: the run row, then one row per policy. Column names
    /// match `training/aura_train/supabase_io.py` so both writers agree.
    pub async fn publish_benchmark(&self, report: &Value, engine_version: &str) -> anyhow::Result<()> {
        let run_id = report
            .get("run_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("report has no run_id"))?;

        let run_row = json!({
            "run_id": run_id,
            "scenario": report.get("scenario").and_then(|v| v.as_str()).unwrap_or("unknown"),
            "seed": report.get("seed").and_then(|v| v.as_i64()).unwrap_or(0),
            "capacity_bytes": report.get("capacity_bytes").and_then(|v| v.as_i64()).unwrap_or(0),
            "requests": report.get("requests").and_then(|v| v.as_i64()).unwrap_or(0),
            "engine_version": engine_version,
            "summary": {
                "winner": report.get("winner"),
                "improvement_vs": report.get("improvement_vs"),
                "belady_upper_bound": report.get("belady_upper_bound"),
            }
        });

        let res = self
            .req(self.client.post(self.rest(RUNS_TABLE)))
            .header("prefer", "resolution=merge-duplicates")
            .query(&[("on_conflict", "run_id")])
            .json(&json!([run_row]))
            .send()
            .await?;
        if !res.status().is_success() {
            anyhow::bail!("run insert failed: {} {}", res.status(), res.text().await.unwrap_or_default());
        }

        let empty = Vec::new();
        let rows = report.get("rows").and_then(|v| v.as_array()).unwrap_or(&empty);
        let payload: Vec<Value> = rows
            .iter()
            .map(|r| {
                json!({
                    "run_id": run_id,
                    "policy": r.get("policy"),
                    "object_hit_rate": r.get("object_hit_rate"),
                    "byte_hit_rate": r.get("byte_hit_rate"),
                    "p95_latency_ms": r.get("p95_latency_ms"),
                    "backend_requests": r.get("backend_requests"),
                    "total_cost_usd": r.get("total_cost_usd"),
                    "regen_cost_usd": r.get("regen_cost_usd"),
                    "sla_penalty_usd": r.get("sla_penalty_usd"),
                    "decision_overhead_us": r.get("decision_overhead_us_p50"),
                    "extra": { "memory_overhead_bytes": r.get("memory_overhead_bytes") }
                })
            })
            .collect();

        if !payload.is_empty() {
            let res = self
                .req(self.client.post(self.rest(RESULTS_TABLE)))
                .header("prefer", "resolution=merge-duplicates")
                .query(&[("on_conflict", "run_id,policy")])
                .json(&payload)
                .send()
                .await?;
            if !res.status().is_success() {
                anyhow::bail!("results insert failed: {} {}", res.status(), res.text().await.unwrap_or_default());
            }
        }

        tracing::info!(run_id, rows = payload.len(), "published benchmark to supabase");
        Ok(())
    }

    /// Ship a batch of audit entries so the explanation of what the cache did outlives the
    /// process that made the decisions.
    ///
    /// The in-memory log is a ring buffer a few hundred entries deep — enough to answer
    /// "what just happened", useless for "why did we serve a wrong price on Tuesday". This
    /// is the durable half. Failure is the caller's problem to requeue, which is why the
    /// count that was accepted comes back rather than a bare unit.
    pub async fn push_audit(&self, entries: &[Value]) -> anyhow::Result<usize> {
        if entries.is_empty() {
            return Ok(0);
        }
        let res = self
            .req(self.client.post(self.rest(AUDIT_TABLE)))
            .json(entries)
            .send()
            .await?;
        if !res.status().is_success() {
            anyhow::bail!(
                "audit insert failed: {} {}",
                res.status(),
                res.text().await.unwrap_or_default()
            );
        }
        Ok(entries.len())
    }

    /// Ask the identity provider who a token belongs to.
    ///
    /// This is the fallback for projects whose tokens are signed with an asymmetric key,
    /// where the project's JWT secret verifies nothing. It is a network call, so it happens
    /// once per token and the answer is cached until the token's own expiry: the identity
    /// provider must never end up on the cache's request path.
    pub async fn whoami(&self, token: &str) -> anyhow::Result<(String, Option<String>, u64)> {
        let url = format!("{}/auth/v1/user", self.base_url);
        let res = self
            .client
            .get(&url)
            .header("apikey", &self.key)
            .header("authorization", format!("Bearer {token}"))
            .timeout(Duration::from_secs(5))
            .send()
            .await?;
        if !res.status().is_success() {
            anyhow::bail!("token rejected by the identity provider: {}", res.status());
        }
        let body: Value = res.json().await?;
        let subject = body
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("no subject in the provider's answer"))?
            .to_string();
        let email = body.get("email").and_then(|v| v.as_str()).map(str::to_string);
        // The provider does not return the expiry here, so trust the token's own claim and
        // fall back to a short window. Erring short costs one extra call; erring long would
        // keep a revoked login working.
        let exp = token
            .split('.')
            .nth(1)
            .and_then(|part| {
                base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(part).ok()
            })
            .and_then(|raw| serde_json::from_slice::<Value>(&raw).ok())
            .and_then(|claims| claims.get("exp").and_then(|v| v.as_u64()))
            .unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() + 300)
                    .unwrap_or(0)
            });
        Ok((subject, email, exp))
    }

    /// Console accounts. Password hashes only; the passwords themselves never exist outside
    /// the browser that typed them.
    pub async fn users(&self) -> anyhow::Result<Vec<Value>> {
        let url = format!("{}?select=*&order=created_at.asc", self.rest(USERS_TABLE));
        let res = self.req(self.client.get(&url)).send().await?;
        if !res.status().is_success() {
            anyhow::bail!("account fetch failed: {}", res.status());
        }
        Ok(res.json::<Vec<Value>>().await?)
    }

    /// Create or update an account, keyed on the address.
    pub async fn upsert_user(&self, row: &Value) -> anyhow::Result<()> {
        let res = self
            .req(self.client.post(self.rest(USERS_TABLE)))
            .header("Prefer", "resolution=merge-duplicates")
            .json(&json!([row]))
            .send()
            .await?;
        if !res.status().is_success() {
            anyhow::bail!(
                "account upsert failed: {} {}",
                res.status(),
                res.text().await.unwrap_or_default()
            );
        }
        Ok(())
    }

    pub async fn delete_user(&self, email: &str) -> anyhow::Result<()> {
        let url = format!("{}?email=eq.{}", self.rest(USERS_TABLE), urlencode(email));
        let res = self.req(self.client.delete(&url)).send().await?;
        if !res.status().is_success() {
            anyhow::bail!("account delete failed: {}", res.status());
        }
        Ok(())
    }

    /// Application keys, as hashes. Read once at boot so a restart does not invalidate every
    /// key an operator handed out, which on a platform that restarts containers freely would
    /// make keys useless.
    pub async fn api_keys(&self) -> anyhow::Result<Vec<Value>> {
        let url = format!("{}?select=*&order=created_at.asc", self.rest(KEYS_TABLE));
        let res = self.req(self.client.get(&url)).send().await?;
        if !res.status().is_success() {
            anyhow::bail!("key fetch failed: {}", res.status());
        }
        Ok(res.json::<Vec<Value>>().await?)
    }

    /// Record a minted key. The secret is not sent: only its hash, which is all the engine
    /// itself keeps, so a copy of this table is not a set of working credentials.
    pub async fn insert_api_key(&self, row: &Value) -> anyhow::Result<()> {
        let res = self
            .req(self.client.post(self.rest(KEYS_TABLE)))
            .json(&json!([row]))
            .send()
            .await?;
        if !res.status().is_success() {
            anyhow::bail!(
                "key insert failed: {} {}",
                res.status(),
                res.text().await.unwrap_or_default()
            );
        }
        Ok(())
    }

    pub async fn revoke_api_key(&self, id: &str) -> anyhow::Result<()> {
        let url = format!("{}?id=eq.{id}", self.rest(KEYS_TABLE));
        let res = self
            .req(self.client.patch(&url))
            .json(&json!({ "revoked": true }))
            .send()
            .await?;
        if !res.status().is_success() {
            anyhow::bail!("key revoke failed: {}", res.status());
        }
        Ok(())
    }

    pub async fn push_event(&self, kind: &str, detail: Value) -> anyhow::Result<()> {
        let res = self
            .req(self.client.post(self.rest(EVENTS_TABLE)))
            .json(&json!([{ "kind": kind, "detail": detail }]))
            .send()
            .await?;
        if !res.status().is_success() {
            anyhow::bail!("event insert failed: {}", res.status());
        }
        Ok(())
    }

    /// Round-trips the REST endpoint so the dashboard can show whether the project is
    /// actually reachable rather than merely configured.
    /// Runs a genuine aggregate against the seeded analytics tables and returns how long
    /// Supabase actually took. This is the difference between a cost model fed by a made-up
    /// number and one fed by a measurement: the query is executed by the database, over the
    /// network, and the wall clock is what the engine learns from.
    pub async fn probe_analytics(&self, region_id: i32, days: i64) -> anyhow::Result<(f64, u64)> {
        // PostgREST compares against a literal, so the cutoff has to be a real timestamp.
        // Sending `now()-90days` would be matched as a string and quietly return nothing.
        let since = iso_days_ago(days);
        let url = format!("{}/rest/v1/app_order_totals", self.base_url);
        let started = std::time::Instant::now();
        let res = self
            .req(self.client.get(&url))
            .header("prefer", "count=exact")
            .query(&[
                ("region_id", format!("eq.{region_id}")),
                ("placed_at", format!("gte.{since}")),
                ("select", "order_id,order_total,line_count".to_string()),
                ("limit", "1000".to_string()),
            ])
            .send()
            .await?;
        let status = res.status();
        let body = res.bytes().await?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        if !status.is_success() {
            anyhow::bail!("analytics probe failed: {status}");
        }
        Ok((elapsed_ms, body.len() as u64))
    }

    pub async fn health(&self) -> bool {
        let url = self.rest(MODELS_TABLE);
        match self
            .req(self.client.get(&url))
            .query(&[("select", "name"), ("limit", "1")])
            .send()
            .await
        {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }
}

/// UTC timestamp `days` in the past, formatted for PostgREST. Written out rather than
/// pulling in a date library for one call.
fn iso_days_ago(days: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let secs = now - days * 86_400;
    let days_since_epoch = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);

    // Civil-from-days, Howard Hinnant's algorithm.
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, tod / 3600, (tod % 3600) / 60, tod % 60
    )
}

fn first_env(names: &[&str]) -> Option<String> {
    for n in names {
        if let Ok(v) = std::env::var(n) {
            let v = v.trim().trim_matches('"').to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

/// Loads `KEY=value` lines from the first `.env` found by walking up from the working
/// directory. Existing environment variables always win, so a shell export overrides the
/// file. Kept deliberately small: this is a convenience for local runs, not a config system.
pub fn load_dotenv() -> Option<std::path::PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..5 {
        for candidate in ["backend/.env", ".env"] {
            let path = dir.join(candidate);
            if path.is_file() {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }
                        if let Some((k, v)) = line.split_once('=') {
                            let k = k.trim();
                            let v = v.trim().trim_matches('"').trim_matches('\'');
                            if std::env::var_os(k).is_none() && !k.is_empty() {
                                std::env::set_var(k, v);
                            }
                        }
                    }
                    return Some(path);
                }
            }
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::iso_days_ago;

    #[test]
    fn iso_timestamp_is_well_formed_and_in_the_past() {
        let now = iso_days_ago(0);
        let past = iso_days_ago(90);
        assert_eq!(now.len(), 20, "{now}");
        assert!(now.ends_with('Z'));
        assert!(past < now, "{past} should sort before {now}");
        let year: i32 = now[..4].parse().unwrap();
        assert!((2020..2100).contains(&year), "{now}");
        let month: u32 = now[5..7].parse().unwrap();
        let day: u32 = now[8..10].parse().unwrap();
        assert!((1..=12).contains(&month) && (1..=31).contains(&day), "{now}");
    }
}

/// Minimal percent-encoding for the one place a value reaches a query string: an email
/// address in a filter. Pulling in a URL crate for `@` and `+` would be the wrong trade.
fn urlencode(value: &str) -> String {
    value
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            other => format!("%{:02X}", other as u32),
        })
        .collect()
}
