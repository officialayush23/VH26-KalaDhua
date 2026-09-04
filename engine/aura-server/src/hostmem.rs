//! What the machine actually has free.
//!
//! The capacity controller used to reason about `host_budget_bytes`, a number in a config
//! file. With `--real-values` the pool is resident memory, so a budget that is larger than
//! the machine is not a budget, it is a crash: the allocator refuses, and a Rust binary
//! built with `panic = "abort"` does not get to explain itself. That is exactly what
//! happened on a laptop asked for a 512 MB pool that the controller was free to grow to
//! four gigabytes.
//!
//! Readings are cached for a second. Asking the operating system for a memory summary on
//! every controller tick would cost more than the decision it informs.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use sysinfo::{MemoryRefreshKind, System};

struct Cached {
    sys: System,
    at: Instant,
    available: u64,
    total: u64,
}

static STATE: Mutex<Option<Cached>> = Mutex::new(None);

const MAX_AGE: Duration = Duration::from_millis(1_000);

/// Free memory the process could plausibly obtain, and the machine's total, in bytes.
/// Returns `(0, 0)` if the platform will not say, which callers must treat as "no limit
/// known" rather than "no memory".
pub fn snapshot() -> (u64, u64) {
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        // A poisoned lock here means another thread panicked mid-reading. The memory figure
        // is advisory, so degrade to "unknown" rather than propagating someone else's panic.
        Err(_) => return (0, 0),
    };
    let fresh = match guard.as_ref() {
        Some(c) if c.at.elapsed() < MAX_AGE => Some((c.available, c.total)),
        _ => None,
    };
    if let Some(v) = fresh {
        return v;
    }
    let kind = MemoryRefreshKind::nothing().with_ram();
    match guard.as_mut() {
        Some(c) => {
            c.sys.refresh_memory_specifics(kind);
            c.available = c.sys.available_memory();
            c.total = c.sys.total_memory();
            c.at = Instant::now();
            (c.available, c.total)
        }
        None => {
            let mut sys = System::new();
            sys.refresh_memory_specifics(kind);
            let available = sys.available_memory();
            let total = sys.total_memory();
            *guard = Some(Cached { sys, at: Instant::now(), available, total });
            (available, total)
        }
    }
}

pub fn available_bytes() -> u64 {
    snapshot().0
}

/// The largest pool this machine should be asked to hold right now.
///
/// Two thirds of what is free, less a floor the rest of the process needs for model
/// bundles, sketches, the journal and the allocator's own overhead. Measured overhead on
/// the payload path is about 1.5x the logical pool, so the logical bound is divided by that
/// rather than handed out whole: a 400 MB allowance means a 266 MB pool, not a 400 MB one
/// that becomes 600 MB resident.
pub fn safe_pool_bytes() -> Option<u64> {
    let available = available_bytes();
    if available == 0 {
        return None;
    }
    const RESERVE: u64 = 512 * 1024 * 1024;
    let spare = available.saturating_sub(RESERVE);
    if spare == 0 {
        // A machine this tight gets the smallest pool that is still a cache rather than
        // nothing at all, and the caller says so out loud.
        return Some(64 * 1024 * 1024);
    }
    Some(((spare as f64 * 0.66) / 1.5) as u64)
}
