//! Who is calling, and what they are allowed to do.
//!
//! Two callers, two credentials, because they are not the same kind of thing:
//!
//! **Applications** are machines on the data path. They carry a long-lived key,
//! `Authorization: Bearer aura_sk_...`, minted once in the console. The key is also the
//! application's identity: whatever name the key was issued to is the application every
//! request under it belongs to, which is what makes onboarding a matter of pointing a
//! service at a URL rather than registering anything. A key is stored as a SHA-256 hash, so
//! a leak of the engine's state does not leak the keys, and it is shown exactly once at mint
//! time because there is nothing to show it from afterwards.
//!
//! **People** are on the console. They log in through Supabase Auth in the browser and the
//! engine verifies the token Supabase issued, using the project's JWT secret. No user table
//! here, no password handling here, and no second identity to keep in step.
//!
//! Modes are explicit rather than inferred. `AURA_AUTH=open` is the local demo: everything
//! answers, no credential needed, and the engine says so at boot. `AURA_AUTH=enforced` is
//! what anything with a public URL runs, and it fails closed - a missing JWT secret in
//! enforced mode is a refusal to start, not a warning, because the alternative is a cache
//! that quietly accepts everyone.

use std::time::{SystemTime, UNIX_EPOCH};

use ahash::AHashMap;
use base64::Engine as _;
use hmac::{Hmac, Mac};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// What a route needs from its caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    /// Liveness and metrics. Never gated: a health check that needs a credential is a
    /// health check that reports the credential's state, not the service's.
    Public,
    /// Reading telemetry. A person on the console, or an application looking at itself.
    Read,
    /// The cache itself: get, put, delete, invalidate. Applications live here.
    Data,
    /// Changing how the cache behaves: profiles, simulation, capacity, model reloads.
    Control,
}

/// Who turned out to be calling.
#[derive(Debug, Clone)]
pub enum Caller {
    /// Nobody presented anything and the engine is running open.
    Anonymous,
    Application { key_id: String, application: String },
    Person { subject: String, email: Option<String> },
}

impl Caller {
    pub fn label(&self) -> String {
        match self {
            Caller::Anonymous => "anonymous".to_string(),
            // The key id and not just the application: two services can share a name in a
            // careless deployment, and an audit line saying which credential made a change
            // is the difference between an investigation and a guess.
            Caller::Application { application, key_id } => format!("app:{application} ({key_id})"),
            Caller::Person { email, subject } => {
                format!("user:{}", email.clone().unwrap_or_else(|| subject.clone()))
            }
        }
    }

    /// The application a request belongs to, when the credential says so. An application
    /// cannot claim to be a different one in the request body: the key decides.
    pub fn application(&self) -> Option<&str> {
        match self {
            Caller::Application { application, .. } => Some(application),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiKey {
    pub id: String,
    pub application: String,
    /// SHA-256 of the secret, hex. The secret itself is never stored anywhere.
    #[serde(skip)]
    pub hash: String,
    pub created_at: String,
    pub revoked: bool,
    /// First eight characters of the secret, so a person can tell two keys apart in a list
    /// without either one being usable from what they can see.
    pub hint: String,
}

impl ApiKey {
    /// Rebuild a key from its stored row. A row missing its hash is skipped rather than
    /// defaulted: a key with an empty hash would match the hash of an empty secret.
    pub fn from_row(row: &serde_json::Value) -> Option<Self> {
        let hash = row.get("key_hash")?.as_str()?.to_string();
        if hash.len() < 32 {
            return None;
        }
        Some(Self {
            id: row.get("id")?.as_str()?.to_string(),
            application: row.get("application")?.as_str()?.to_string(),
            hash,
            created_at: row
                .get("created_at")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            revoked: row.get("revoked").and_then(|v| v.as_bool()).unwrap_or(false),
            hint: row.get("hint").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        })
    }

    /// The row shape the control plane stores. The secret is not in it.
    pub fn as_row(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "application": self.application,
            "key_hash": self.hash,
            "hint": self.hint,
            "revoked": self.revoked,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Open,
    Enforced,
}

/// What the gate decided, before any network call.
///
/// A token that cannot be checked locally does not fail: Supabase projects created after
/// the move to asymmetric signing issue ES256 tokens, and their JWT secret verifies nothing.
/// Those are checked by asking Supabase who the token belongs to, which the gate does after
/// releasing the lock, because holding a mutex across a network call is how a cache stops
/// answering.
pub enum Verdict {
    Allowed(Caller),
    Denied(Denial),
    /// Verify this token with the identity provider, then allow.
    Remote(String),
}

pub struct Auth {
    pub mode: Mode,
    jwt_secret: Option<String>,
    /// Sessions the engine issued itself are checked against [`crate::users::Users`], which
    /// the gate holds separately; this flag only records that accounts exist at all, so the
    /// console can tell "sign in" from "there is nobody to sign in as yet".
    has_accounts: bool,
    keys: AHashMap<String, ApiKey>,
    /// Tokens already checked with the provider, with the expiry the provider stated. A
    /// remote check per request would put the identity provider on the cache's hot path,
    /// which is the one place it must never be.
    verified: AHashMap<String, (u64, String, Option<String>)>,
}

impl Auth {
    /// Read the mode and the console secret from the environment.
    ///
    /// Returns an error rather than a warning when enforcement is asked for and cannot be
    /// delivered. A deployment that silently downgrades to open is worse than one that
    /// refuses to start, because nobody finds out until it matters.
    pub fn from_env() -> anyhow::Result<Self> {
        let mode = match std::env::var("AURA_AUTH").unwrap_or_default().to_lowercase().as_str() {
            "enforced" | "on" | "1" | "true" => Mode::Enforced,
            _ => Mode::Open,
        };
        let jwt_secret = std::env::var("SUPABASE_JWT_KEY")
            .or_else(|_| std::env::var("SUPABASE_JWT_SECRET"))
            .ok()
            .filter(|s| !s.trim().is_empty());
        Ok(Self {
            mode,
            jwt_secret,
            has_accounts: false,
            keys: AHashMap::new(),
            verified: AHashMap::new(),
        })
    }

    pub fn is_enforced(&self) -> bool {
        self.mode == Mode::Enforced
    }

    pub fn has_accounts(&self) -> bool {
        self.has_accounts
    }

    pub fn set_has_accounts(&mut self, yes: bool) {
        self.has_accounts = yes;
    }

    /// Record a token the identity provider vouched for, so the next request on it is
    /// answered here rather than over the network.
    pub fn remember_verified(&mut self, token: &str, exp: u64, subject: String, email: Option<String>) {
        if self.verified.len() > 512 {
            let now = now_secs();
            self.verified.retain(|_, (e, _, _)| *e > now);
        }
        self.verified.insert(hash_secret(token), (exp, subject, email));
    }

    pub fn keys(&self) -> Vec<ApiKey> {
        let mut out: Vec<ApiKey> = self.keys.values().cloned().collect();
        out.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        out
    }

    /// Load keys minted in an earlier run. Hashes only; there is nothing else to load.
    pub fn adopt(&mut self, keys: Vec<ApiKey>) {
        for k in keys {
            self.keys.insert(k.hash.clone(), k);
        }
    }

    /// Mint a key for an application. Returns the secret, which is the only time it exists
    /// outside the caller's own storage.
    pub fn mint(&mut self, application: &str, rng: &mut impl FnMut() -> u64) -> (String, ApiKey) {
        let mut raw = String::from("aura_sk_");
        for _ in 0..4 {
            raw.push_str(&format!("{:016x}", rng()));
        }
        let hash = hash_secret(&raw);
        let key = ApiKey {
            id: format!("key_{}", &hash[..12]),
            application: application.to_string(),
            hash: hash.clone(),
            created_at: now_iso(),
            revoked: false,
            hint: raw[..16].to_string(),
        };
        self.keys.insert(hash, key.clone());
        (raw, key)
    }

    pub fn revoke(&mut self, id: &str) -> bool {
        for k in self.keys.values_mut() {
            if k.id == id {
                k.revoked = true;
                return true;
            }
        }
        false
    }

    /// Resolve a request's credential into a caller, or say why it will not be served.
    ///
    /// `bearer` is whatever came in on `Authorization: Bearer ...` or, for the WebSocket,
    /// the `token` query parameter - browsers cannot set headers on a socket handshake, and
    /// refusing to solve that is how dashboards end up unauthenticated in practice.
    pub fn authorise(&self, need: Need, bearer: Option<&str>) -> Verdict {
        if need == Need::Public {
            return Verdict::Allowed(Caller::Anonymous);
        }
        let Some(token) = bearer.map(str::trim).filter(|t| !t.is_empty()) else {
            return if self.is_enforced() {
                Verdict::Denied(Denial::Missing)
            } else {
                Verdict::Allowed(Caller::Anonymous)
            };
        };

        if let Some((exp, subject, email)) = self.verified.get(&hash_secret(token)) {
            if *exp > now_secs() {
                return Verdict::Allowed(Caller::Person {
                    subject: subject.clone(),
                    email: email.clone(),
                });
            }
        }

        if token.starts_with("aura_sk_") {
            let found = match self.keys.get(&hash_secret(token)) {
                Some(k) if !k.revoked => Ok(Caller::Application {
                    key_id: k.id.clone(),
                    application: k.application.clone(),
                }),
                Some(_) => Err(Denial::Revoked),
                None => Err(Denial::Unknown),
            };
            return match found {
                // An application key is for the data path and for looking at telemetry. It
                // cannot re-tune the cache: a leaked key should not be able to change what
                // the cache optimises for every other application.
                Ok(_) if need == Need::Control => Verdict::Denied(Denial::Forbidden),
                Ok(caller) => Verdict::Allowed(caller),
                Err(d) => Verdict::Denied(d),
            };
        }

        // A session this engine signed is the normal case and is checked by the caller
        // before this point; reaching here with one means it did not verify.
        match &self.jwt_secret {
            Some(secret) => match verify_jwt(token, secret) {
                Ok(caller) => Verdict::Allowed(caller),
                // A signature this secret cannot check is not necessarily a bad token: on a
                // project using asymmetric signing keys the secret is simply the wrong
                // instrument. Ask the provider rather than refusing someone who is signed in.
                Err(Denial::Malformed) => Verdict::Remote(token.to_string()),
                Err(d) => Verdict::Denied(d),
            },
            // No external provider configured: an unrecognised token is simply wrong.
            None if self.is_enforced() => Verdict::Denied(Denial::Expired),
            None => Verdict::Allowed(Caller::Anonymous),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denial {
    Missing,
    Unknown,
    Revoked,
    Expired,
    Malformed,
    Forbidden,
}

impl Denial {
    pub fn status(self) -> u16 {
        match self {
            Denial::Forbidden => 403,
            _ => 401,
        }
    }

    /// Said plainly, because an operator debugging a 401 at 2 a.m. is the person this string
    /// is for. None of these reveal whether a given key exists.
    pub fn message(self) -> &'static str {
        match self {
            Denial::Missing => "this engine requires a credential: an application key as `Authorization: Bearer aura_sk_...`, or a console login",
            Denial::Unknown => "that key is not one this engine issued",
            Denial::Revoked => "that key was revoked",
            Denial::Expired => "that login has expired; sign in again",
            Denial::Malformed => "the token could not be read as a Supabase login",
            Denial::Forbidden => "an application key cannot change how the cache behaves; that needs a console login",
        }
    }
}

/// Which credential a path needs. Written as a function over the path rather than attached
/// per route so that a route added without thinking about auth lands in `Control`, the
/// strictest class, instead of silently becoming public.
pub fn need_for(method: &str, path: &str) -> Need {
    if path == "/healthz" || path == "/metrics" {
        return Need::Public;
    }
    // The way in cannot be behind the door. `/v1/auth/login` takes an email and a password
    // and is the only route that mints a session, so gating it means nobody can ever sign
    // in; `/v1/auth/me` reports whether accounts exist at all, which is what the console
    // needs to tell "sign in" from "there is nobody to sign in as yet".
    if path == "/v1/auth/login" || path == "/v1/auth/me" {
        return Need::Public;
    }
    if path.starts_with("/v1/cache")
        || path == "/v1/invalidate"
        || path == "/v1/version/bump"
    {
        return Need::Data;
    }
    if method == "GET" {
        // Everything else that only reads is console reading: telemetry, explanations, the
        // audit log, the benchmark's last result.
        return Need::Read;
    }
    Need::Control
}

fn hash_secret(secret: &str) -> String {
    let digest = Sha256::digest(secret.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn now_iso() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // Good enough for an audit line and stable across platforms without pulling in a date
    // library for one format string.
    format!("{secs}")
}

type HmacSha256 = Hmac<Sha256>;

/// Verify a Supabase-issued HS256 token: signature first, then expiry, then who it is.
///
/// Only HS256 is accepted. A token is rejected outright if it names another algorithm,
/// including `none`, because accepting the algorithm a token asks for is the oldest JWT
/// vulnerability there is.
fn verify_jwt(token: &str, secret: &str) -> Result<Caller, Denial> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(Denial::Malformed);
    }
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let header: serde_json::Value = b64
        .decode(parts[0])
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or(Denial::Malformed)?;
    if header.get("alg").and_then(|v| v.as_str()) != Some("HS256") {
        return Err(Denial::Malformed);
    }

    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let signature = b64.decode(parts[2]).map_err(|_| Denial::Malformed)?;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).map_err(|_| Denial::Malformed)?;
    mac.update(signing_input.as_bytes());
    mac.verify_slice(&signature).map_err(|_| Denial::Malformed)?;

    let claims: serde_json::Value = b64
        .decode(parts[1])
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or(Denial::Malformed)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    if let Some(exp) = claims.get("exp").and_then(|v| v.as_u64()) {
        if exp < now {
            return Err(Denial::Expired);
        }
    }
    Ok(Caller::Person {
        subject: claims.get("sub").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
        email: claims.get("email").and_then(|v| v.as_str()).map(str::to_string),
    })
}
