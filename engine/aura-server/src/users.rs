//! Accounts, passwords and sessions, owned by the engine.
//!
//! The engine issues its own sessions rather than borrowing an identity provider's. The
//! reason is not purity: a cache that can only be administered while a third party is
//! reachable is a cache you cannot administer during the incident you most need to. One
//! root account comes from the environment on first boot, and everything else is created
//! through it.
//!
//! **Passwords** are stored as PBKDF2-HMAC-SHA256, 120,000 iterations, with a 16-byte random
//! salt per user, in the `pbkdf2$iterations$salt$hash` shape so the parameters travel with
//! the hash and can be raised later without invalidating anyone. Verification is a constant
//! -time comparison, because a timing difference on a password check is a password oracle.
//!
//! **Sessions** are HS256 tokens the engine signs itself, keyed on a secret that survives
//! restarts: a platform that recycles containers hourly would otherwise sign everyone out
//! hourly. If `AURA_SESSION_SECRET` is absent the secret is derived from the root account's
//! password hash, which is stable while the password is and changes the moment it does -
//! so changing the root password ends every existing session, which is what changing a
//! password is for.

use std::time::{SystemTime, UNIX_EPOCH};

use ahash::AHashMap;
use base64::Engine as _;
use hmac::{Hmac, Mac};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const ITERATIONS: u32 = 120_000;
const SESSION_HOURS: u64 = 12;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Serialize)]
pub struct User {
    pub id: String,
    pub email: String,
    /// `root` may create and remove accounts. `operator` may do everything else.
    pub role: String,
    pub created_at: String,
    #[serde(skip)]
    pub password: String,
}

impl User {
    pub fn as_row(&self) -> Value {
        json!({
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "password_hash": self.password,
        })
    }

    pub fn from_row(row: &Value) -> Option<Self> {
        Some(Self {
            id: row.get("id")?.as_str()?.to_string(),
            email: row.get("email")?.as_str()?.to_lowercase(),
            role: row.get("role").and_then(|v| v.as_str()).unwrap_or("operator").to_string(),
            created_at: row
                .get("created_at")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            password: row.get("password_hash")?.as_str()?.to_string(),
        })
    }
}

#[derive(Debug)]
pub enum LoginError {
    /// Deliberately one error for both an unknown address and a wrong password. Telling the
    /// two apart turns the login form into a list of who has an account here.
    Rejected,
    NoAccounts,
}

pub struct Users {
    by_email: AHashMap<String, User>,
    session_secret: Option<String>,
}

impl Users {
    pub fn new() -> Self {
        Self { by_email: AHashMap::new(), session_secret: std::env::var("AURA_SESSION_SECRET").ok() }
    }

    pub fn is_empty(&self) -> bool {
        self.by_email.is_empty()
    }

    pub fn adopt(&mut self, users: Vec<User>) {
        for u in users {
            self.by_email.insert(u.email.clone(), u);
        }
    }

    pub fn list(&self) -> Vec<User> {
        let mut out: Vec<User> = self.by_email.values().cloned().collect();
        out.sort_by(|a, b| a.email.cmp(&b.email));
        out
    }

    pub fn get(&self, email: &str) -> Option<&User> {
        self.by_email.get(&email.to_lowercase())
    }

    /// Create an account. Returns `None` if the address is already taken, so a create cannot
    /// silently overwrite someone's password.
    pub fn create(&mut self, email: &str, password: &str, role: &str) -> Option<User> {
        let email = email.trim().to_lowercase();
        if email.is_empty() || !email.contains('@') || self.by_email.contains_key(&email) {
            return None;
        }
        let user = User {
            id: format!("usr_{}", &hash_hex(&email)[..12]),
            email: email.clone(),
            role: role.to_string(),
            created_at: now_secs().to_string(),
            password: hash_password(password),
        };
        self.by_email.insert(email, user.clone());
        Some(user)
    }

    pub fn set_password(&mut self, email: &str, password: &str) -> bool {
        match self.by_email.get_mut(&email.to_lowercase()) {
            Some(u) => {
                u.password = hash_password(password);
                true
            }
            None => false,
        }
    }

    pub fn remove(&mut self, email: &str) -> bool {
        let email = email.to_lowercase();
        match self.by_email.get(&email) {
            // The last root account cannot be deleted. An engine nobody can administer is a
            // worse outcome than an account that outstays its welcome.
            Some(u) if u.role == "root" && self.by_email.values().filter(|x| x.role == "root").count() <= 1 => false,
            Some(_) => self.by_email.remove(&email).is_some(),
            None => false,
        }
    }

    /// Check an email and password, and mint a session on success.
    pub fn login(&self, email: &str, password: &str) -> Result<(String, User, u64), LoginError> {
        if self.by_email.is_empty() {
            return Err(LoginError::NoAccounts);
        }
        let user = self.by_email.get(&email.trim().to_lowercase()).ok_or(LoginError::Rejected)?;
        if !verify_password(password, &user.password) {
            return Err(LoginError::Rejected);
        }
        let exp = now_secs() + SESSION_HOURS * 3600;
        Ok((self.sign(&user.id, &user.email, &user.role, exp), user.clone(), exp))
    }

    /// The secret sessions are signed with. Derived from the root password hash when no
    /// explicit secret is configured, so sessions survive a restart and end when that
    /// password changes.
    fn secret(&self) -> String {
        if let Some(s) = &self.session_secret {
            return s.clone();
        }
        let root = self
            .by_email
            .values()
            .find(|u| u.role == "root")
            .map(|u| u.password.clone())
            .unwrap_or_default();
        hash_hex(&format!("aura-session-v1:{root}"))
    }

    fn sign(&self, sub: &str, email: &str, role: &str, exp: u64) -> String {
        let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header = b64.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let claims = b64.encode(
            json!({ "sub": sub, "email": email, "role": role, "exp": exp, "iss": "aura" })
                .to_string()
                .as_bytes(),
        );
        let signing_input = format!("{header}.{claims}");
        let mut mac = HmacSha256::new_from_slice(self.secret().as_bytes()).expect("hmac key");
        mac.update(signing_input.as_bytes());
        format!("{signing_input}.{}", b64.encode(mac.finalize().into_bytes()))
    }

    /// Verify a session this engine issued. Returns the subject and email.
    pub fn verify(&self, token: &str) -> Option<(String, Option<String>)> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header: Value = serde_json::from_slice(&b64.decode(parts[0]).ok()?).ok()?;
        // Only HS256, and never the algorithm the token asks for: accepting that is the
        // oldest JWT hole there is.
        if header.get("alg").and_then(|v| v.as_str()) != Some("HS256") {
            return None;
        }
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let mut mac = HmacSha256::new_from_slice(self.secret().as_bytes()).ok()?;
        mac.update(signing_input.as_bytes());
        mac.verify_slice(&b64.decode(parts[2]).ok()?).ok()?;

        let claims: Value = serde_json::from_slice(&b64.decode(parts[1]).ok()?).ok()?;
        if claims.get("exp").and_then(|v| v.as_u64()).unwrap_or(0) < now_secs() {
            return None;
        }
        Some((
            claims.get("sub").and_then(|v| v.as_str()).unwrap_or("user").to_string(),
            claims.get("email").and_then(|v| v.as_str()).map(str::to_string),
        ))
    }

    pub fn role_of_token(&self, token: &str) -> Option<String> {
        let parts: Vec<&str> = token.split('.').collect();
        let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let claims: Value = serde_json::from_slice(&b64.decode(parts.get(1)?).ok()?).ok()?;
        claims.get("role").and_then(|v| v.as_str()).map(str::to_string)
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn hash_hex(input: &str) -> String {
    Sha256::digest(input.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// PBKDF2-HMAC-SHA256, written out rather than pulled in: it is a loop over HMAC, both of
/// which are already here, and one fewer dependency in the credential path is worth twenty
/// lines.
fn pbkdf2(password: &[u8], salt: &[u8], iterations: u32) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(password).expect("hmac key");
    mac.update(salt);
    mac.update(&1u32.to_be_bytes());
    let mut u = mac.finalize().into_bytes();
    let mut out = u;
    for _ in 1..iterations {
        let mut m = HmacSha256::new_from_slice(password).expect("hmac key");
        m.update(&u);
        u = m.finalize().into_bytes();
        for (o, v) in out.iter_mut().zip(u.iter()) {
            *o ^= v;
        }
    }
    let mut result = [0u8; 32];
    result.copy_from_slice(&out);
    result
}

pub fn hash_password(password: &str) -> String {
    let salt = random_bytes();
    let hash = pbkdf2(password.as_bytes(), &salt, ITERATIONS);
    format!("pbkdf2${ITERATIONS}${}${}", hex(&salt), hex(&hash))
}

/// Constant-time comparison. A password check that returns early on the first wrong byte
/// tells an attacker how much of the password was right.
fn verify_password(password: &str, stored: &str) -> bool {
    let parts: Vec<&str> = stored.split('$').collect();
    if parts.len() != 4 || parts[0] != "pbkdf2" {
        return false;
    }
    let iterations: u32 = match parts[1].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let (salt, expected) = (unhex(parts[2]), unhex(parts[3]));
    if salt.is_empty() || expected.len() != 32 {
        return false;
    }
    let actual = pbkdf2(password.as_bytes(), &salt, iterations);
    let mut diff = 0u8;
    for (a, b) in actual.iter().zip(expected.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

fn random_bytes() -> [u8; 16] {
    let mut buf = [0u8; 16];
    use std::io::Read;
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut buf).is_ok() {
            return buf;
        }
    }
    let mut acc = now_secs() ^ 0x9E37_79B9_7F4A_7C15;
    for b in buf.iter_mut() {
        acc ^= acc << 13;
        acc ^= acc >> 7;
        acc ^= acc << 17;
        *b = (acc & 0xff) as u8;
    }
    buf
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .filter_map(|i| u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_password_verifies_against_its_own_hash_and_nothing_else() {
        let stored = hash_password("correct horse battery staple");
        assert!(verify_password("correct horse battery staple", &stored));
        assert!(!verify_password("correct horse battery stapl", &stored));
        assert!(!verify_password("", &stored));
    }

    #[test]
    fn two_users_with_the_same_password_do_not_share_a_hash() {
        // Without a per-user salt they would, and one cracked password would be two.
        assert_ne!(hash_password("same"), hash_password("same"));
    }

    #[test]
    fn a_session_verifies_and_a_tampered_one_does_not() {
        let mut users = Users::new();
        users.create("root@example.com", "hunter2hunter2", "root").expect("created");
        let (token, _, _) = users.login("root@example.com", "hunter2hunter2").expect("login");
        assert!(users.verify(&token).is_some());
        let mut bad = token.clone();
        bad.pop();
        bad.push('x');
        assert!(users.verify(&bad).is_none());
    }

    #[test]
    fn changing_the_root_password_ends_every_existing_session() {
        let mut users = Users::new();
        users.create("root@example.com", "hunter2hunter2", "root").expect("created");
        let (token, _, _) = users.login("root@example.com", "hunter2hunter2").expect("login");
        users.set_password("root@example.com", "a-different-one");
        assert!(users.verify(&token).is_none());
    }

    #[test]
    fn the_last_root_account_cannot_be_removed() {
        let mut users = Users::new();
        users.create("root@example.com", "hunter2hunter2", "root").expect("created");
        assert!(!users.remove("root@example.com"));
    }

    #[test]
    fn an_unknown_email_and_a_wrong_password_fail_the_same_way() {
        let mut users = Users::new();
        users.create("root@example.com", "hunter2hunter2", "root").expect("created");
        assert!(matches!(users.login("nobody@example.com", "x"), Err(LoginError::Rejected)));
        assert!(matches!(users.login("root@example.com", "x"), Err(LoginError::Rejected)));
    }
}
