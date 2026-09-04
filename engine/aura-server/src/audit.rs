//! The audit log: everything the cache did, in words a person can read.
//!
//! A cache that cannot explain itself is a cache nobody will put in front of their
//! database. `/v1/explain/{key}` already answers "why this object" on demand, but that is a
//! question you have to know to ask. This is the running account: every admission,
//! eviction, refresh, invalidation, scaling decision and policy shift, written as a
//! sentence at the moment it happens.
//!
//! ```text
//! 14:22:31  KEPT       recommendation:user:1842
//!           87% likely to be wanted again within a minute, and rebuilding it costs
//!           320 ms of GPU time. Worth 4.3x the weakest object it displaced.
//!
//! 14:22:31  REFUSED    content:blob:88213
//!           84 MB for an 11% chance of reuse. It would have displaced better objects.
//!
//! 14:22:33  INVALIDATED row:product:1292
//!           Changed in the database. Dropped 27 cached objects built from it.
//!
//! 14:22:41  GREW       512 MB -> 768 MB
//!           The extra memory should lift the hit rate 6.6 points, saving $7.20/hour
//!           against $1.40/hour of rent.
//! ```
//!
//! Two rules govern the writing:
//!
//! **Every sentence carries the numbers that produced it.** "Evicted because its value was
//! low" is not an explanation. "Cheapest per byte of the 32 sampled, and its value had
//! fallen 60% since admission" is one, and it can be checked.
//!
//! **The structured facts travel with the prose.** The message is for a human reading the
//! dashboard; `facts` is for anything that needs to filter, aggregate or replay. Neither is
//! derived from the other after the fact, so they cannot drift apart.

use std::collections::VecDeque;

use serde::Serialize;

/// What happened. Kept separate from severity: an eviction is routine, an eviction that
/// throws away something expensive is not, and the difference is not in the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditKind {
    Admit,
    Reject,
    Evict,
    Refresh,
    Expire,
    Invalidate,
    VersionBump,
    ScaleUp,
    ScaleDown,
    ScaleHold,
    PolicyShift,
    ModelLoad,
    RegimeChange,
    Pressure,
}

impl AuditKind {
    /// The short label the dashboard shows in its own column.
    pub fn label(self) -> &'static str {
        match self {
            AuditKind::Admit => "KEPT",
            AuditKind::Reject => "REFUSED",
            AuditKind::Evict => "EVICTED",
            AuditKind::Refresh => "REFRESHED",
            AuditKind::Expire => "EXPIRED",
            AuditKind::Invalidate => "INVALIDATED",
            AuditKind::VersionBump => "RETIRED",
            AuditKind::ScaleUp => "GREW",
            AuditKind::ScaleDown => "SHRANK",
            AuditKind::ScaleHold => "HELD",
            AuditKind::PolicyShift => "RE-WEIGHTED",
            AuditKind::ModelLoad => "MODEL",
            AuditKind::RegimeChange => "WORKLOAD",
            AuditKind::Pressure => "PRESSURE",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AuditKind::Admit => "admit",
            AuditKind::Reject => "reject",
            AuditKind::Evict => "evict",
            AuditKind::Refresh => "refresh",
            AuditKind::Expire => "expire",
            AuditKind::Invalidate => "invalidate",
            AuditKind::VersionBump => "version_bump",
            AuditKind::ScaleUp => "scale_up",
            AuditKind::ScaleDown => "scale_down",
            AuditKind::ScaleHold => "scale_hold",
            AuditKind::PolicyShift => "policy_shift",
            AuditKind::ModelLoad => "model_load",
            AuditKind::RegimeChange => "regime_change",
            AuditKind::Pressure => "pressure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Routine. The overwhelming majority.
    Info,
    /// Worth a glance: a decision that cost money, or a correctness event.
    Notice,
    /// Something a person should look at.
    Warning,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    pub seq: u64,
    /// Engine clock, so an entry can be lined up against a telemetry frame.
    pub t_ms: f64,
    /// Wall clock, so a person can line it up against their own logs.
    pub at: String,
    pub kind: &'static str,
    pub label: &'static str,
    pub severity: Severity,
    pub subject: String,
    pub application: String,
    /// The sentence. This is the point of the module.
    pub message: String,
    /// The numbers behind the sentence, for filtering and replay.
    pub facts: Vec<Fact>,
    /// Money this decision saved (positive) or spent (negative), where it is meaningful.
    pub usd_impact: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Fact {
    pub name: String,
    pub value: String,
}

impl Fact {
    pub fn new(name: &str, value: impl Into<String>) -> Self {
        Self { name: name.to_string(), value: value.into() }
    }
}

// --------------------------------------------------------------------------- humanising

/// Bytes as a person writes them. Decimal units, because that is how memory is sold.
pub fn bytes(n: u64) -> String {
    const KB: f64 = 1_000.0;
    let n = n as f64;
    if n < KB {
        format!("{n:.0} B")
    } else if n < KB * KB {
        format!("{:.0} KB", n / KB)
    } else if n < KB * KB * KB {
        let mb = n / (KB * KB);
        if mb < 10.0 {
            format!("{mb:.1} MB")
        } else {
            format!("{mb:.0} MB")
        }
    } else {
        format!("{:.2} GB", n / (KB * KB * KB))
    }
}

/// Money at the scale a single cache decision actually involves.
///
/// A rebuild often costs a few millionths of a dollar. Printing `$0.00` is worse than
/// useless, so small amounts are given in cents with enough precision to be compared.
pub fn usd(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 {
        "$0".to_string()
    } else if a < 0.0001 {
        format!("{:.4} cents", v * 100.0)
    } else if a < 0.01 {
        format!("{:.2} cents", v * 100.0)
    } else if a < 100.0 {
        format!("${v:.2}")
    } else {
        format!("${v:.0}")
    }
}

pub fn usd_per_hour(v: f64) -> String {
    format!("{}/hour", usd(v))
}

/// Durations the way people say them.
pub fn ms(v: f64) -> String {
    if v < 1.0 {
        format!("{:.1} ms", v)
    } else if v < 1_000.0 {
        format!("{:.0} ms", v)
    } else if v < 60_000.0 {
        format!("{:.1} s", v / 1_000.0)
    } else {
        format!("{:.1} min", v / 60_000.0)
    }
}

pub fn percent(p: f64) -> String {
    format!("{:.0}%", p * 100.0)
}

/// A ratio as "4.3x", or "just above" when it is barely over one — because "1.0x above the
/// bar" reads as though nothing happened, which is exactly what it means.
pub fn multiple(x: f64) -> String {
    if !x.is_finite() {
        "far".to_string()
    } else if x >= 100.0 {
        format!("{x:.0}x")
    } else if x >= 1.15 {
        format!("{x:.1}x")
    } else if x >= 1.0 {
        "just above".to_string()
    } else {
        format!("{:.0}% of", x * 100.0)
    }
}

/// Describe a cost vector in whichever resource actually dominates it.
///
/// "320 ms of work" is much less useful than "320 ms of GPU time", and which resource
/// dominates is the whole reason two objects with equal latency can differ 5x in value.
pub fn dominant_cost(cpu_ms: f64, gpu_ms: f64, db_ms: f64, api_usd: f64, pricing_usd: f64) -> String {
    let cpu = cpu_ms * 0.000_000_011_6;
    let gpu = gpu_ms * 0.000_000_255;
    let db = db_ms * 0.000_000_032_0;
    let mut parts: Vec<(&str, f64, f64)> =
        vec![("GPU", gpu, gpu_ms), ("database", db, db_ms), ("CPU", cpu, cpu_ms)];
    parts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if api_usd > 0.0 && api_usd >= parts[0].1 {
        return format!("a paid API call costing {}", usd(api_usd));
    }
    let (name, share, amount) = parts[0];
    if amount <= 0.0 || pricing_usd <= 0.0 {
        return format!("{} to rebuild", usd(pricing_usd));
    }
    let fraction = share / pricing_usd;
    if fraction > 0.6 {
        format!("{} of {} time", ms(amount), name)
    } else {
        format!("{} of mixed {} and other work", ms(amount), name)
    }
}

// --------------------------------------------------------------------------- the log

#[derive(Debug)]
pub struct AuditLog {
    entries: VecDeque<AuditEntry>,
    capacity: usize,
    seq: u64,
    /// Entries not yet handed to the Supabase writer.
    unshipped: VecDeque<AuditEntry>,
    /// Routine admissions and evictions are the overwhelming majority of events. Logging
    /// every one at several thousand per second would bury the interesting ones and cost
    /// more than the decisions themselves, so they are sampled; anything that costs money
    /// or touches correctness is always kept.
    routine_sample: u64,
    routine_counter: u64,
    pub suppressed: u64,
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new(500, 32)
    }
}

impl AuditLog {
    pub fn new(capacity: usize, routine_sample: u64) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            seq: 0,
            unshipped: VecDeque::new(),
            routine_sample: routine_sample.max(1),
            routine_counter: 0,
            suppressed: 0,
        }
    }

    fn keep_routine(&mut self) -> bool {
        self.routine_counter += 1;
        if self.routine_counter % self.routine_sample == 0 {
            true
        } else {
            self.suppressed += 1;
            false
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        t_ms: f64,
        kind: AuditKind,
        severity: Severity,
        subject: impl Into<String>,
        application: impl Into<String>,
        message: impl Into<String>,
        facts: Vec<Fact>,
        usd_impact: f64,
    ) {
        // Sample the routine, keep everything that matters.
        let routine = matches!(kind, AuditKind::Admit | AuditKind::Evict | AuditKind::Reject)
            && severity == Severity::Info;
        if routine && !self.keep_routine() {
            return;
        }

        self.seq += 1;
        let entry = AuditEntry {
            seq: self.seq,
            t_ms,
            at: wall_clock(),
            kind: kind.as_str(),
            label: kind.label(),
            severity,
            subject: subject.into(),
            application: application.into(),
            message: message.into(),
            facts,
            usd_impact,
        };

        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry.clone());

        // Bounded independently: if the Supabase writer is down, the in-memory log must keep
        // working rather than growing until the process dies.
        if self.unshipped.len() >= 5_000 {
            self.unshipped.pop_front();
        }
        self.unshipped.push_back(entry);
    }

    /// Most recent first, which is the order a person reads a log in.
    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        self.entries.iter().rev().take(limit).cloned().collect()
    }

    pub fn recent_of(&self, kind: &str, limit: usize) -> Vec<AuditEntry> {
        self.entries.iter().rev().filter(|e| e.kind == kind).take(limit).cloned().collect()
    }

    /// Take entries for shipping to Supabase. Destructive, so a failed push must re-queue
    /// rather than silently drop.
    pub fn drain_unshipped(&mut self, n: usize) -> Vec<AuditEntry> {
        let take = n.min(self.unshipped.len());
        self.unshipped.drain(..take).collect()
    }

    pub fn requeue(&mut self, entries: Vec<AuditEntry>) {
        for e in entries.into_iter().rev() {
            if self.unshipped.len() >= 5_000 {
                break;
            }
            self.unshipped.push_front(e);
        }
    }

    pub fn pending_shipment(&self) -> usize {
        self.unshipped.len()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// `HH:MM:SS` in UTC, without pulling in a date library for one format string.
fn wall_clock() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3_600, (s % 3_600) / 60, s % 60)
}

// --------------------------------------------------------------------------- sentences

/// The writers. Each takes the numbers a decision produced and returns the sentence a
/// person would write about it.
pub mod say {
    use super::*;

    pub fn admitted(
        key: &str,
        reuse_60s: f64,
        cost_usd: f64,
        cpu_ms: f64,
        gpu_ms: f64,
        db_ms: f64,
        api_usd: f64,
        density: f64,
        bar: f64,
        evicted: usize,
    ) -> String {
        let over = if bar > 0.0 { density / bar } else { f64::INFINITY };
        let displaced = match evicted {
            0 => "There was room for it.".to_string(),
            1 => "It displaced one object worth less.".to_string(),
            n => format!("It displaced {n} objects worth less."),
        };
        format!(
            "Kept {key}. {} likely to be wanted again within a minute, and rebuilding it costs \
             {}. Worth {} the weakest thing it would displace. {displaced}",
            percent(reuse_60s),
            dominant_cost(cpu_ms, gpu_ms, db_ms, api_usd, cost_usd),
            multiple(over),
        )
    }

    pub fn rejected(key: &str, reason_code: &str, size: u64, reuse_60s: f64, density: f64, bar: f64) -> String {
        match reason_code {
            "object_exceeds_quarter_capacity" => format!(
                "Refused {key}. At {} it would take more than a quarter of the whole cache — \
                 no single object is allowed to do that.",
                bytes(size)
            ),
            "single_touch_scan_signature" => format!(
                "Refused {key}. The workload is a scan right now: keys are arriving once and \
                 never coming back, so admitting them would flush everything valuable.",
            ),
            "could_not_free_enough" => format!(
                "Could not fit {key} ({}). Nothing resident was worth less than it, so the \
                 cache kept what it had.",
                bytes(size)
            ),
            _ => format!(
                "Refused {key}. {} for a {} chance of being wanted again — worth {} the \
                 weakest object it would have displaced.",
                bytes(size),
                percent(reuse_60s),
                multiple(if bar > 0.0 { density / bar } else { 0.0 })
            ),
        }
    }

    pub fn evicted(key: &str, size: u64, density: f64, incoming_density: f64, saved_usd: f64) -> String {
        let comparison = if incoming_density > 0.0 && density > 0.0 {
            format!(
                " The arriving object was worth {} as much per byte.",
                multiple(incoming_density / density)
            )
        } else {
            String::new()
        };
        format!(
            "Evicted {key} to make room — {}, and the cheapest per byte of the 32 sampled.{} \
             Rebuilding it later will cost {}.",
            bytes(size),
            comparison,
            usd(saved_usd)
        )
    }

    pub fn refreshed(key: &str, ttl_remaining: f64, reuse_60s: f64) -> String {
        format!(
            "Rebuilt {key} before anyone asked. Only {} of its lifetime was left and it is \
             {} likely to be wanted within the minute, so waiting for it to expire would \
             have handed someone a slow request.",
            percent(ttl_remaining),
            percent(reuse_60s)
        )
    }

    pub fn invalidated(tag: &str, keys: usize, hard: bool, source: &str) -> String {
        let action = if hard { "Dropped" } else { "Marked stale" };
        let why = if hard {
            "Serving the old value would be wrong, not merely out of date."
        } else {
            "The next reader gets the old value once while a rebuild runs behind it, which is \
             far cheaper than making everyone wait."
        };
        format!(
            "{tag} changed in {source}. {action} {} cached object{} built from it. {why}",
            keys,
            if keys == 1 { "" } else { "s" }
        )
    }

    pub fn version_bumped(namespace: &str, version: u64) -> String {
        format!(
            "Retired every {namespace} object to version {version}. Nothing was deleted: new \
             requests carry the new version and the old generation ages out under ordinary \
             pressure. Flushing instead would have sent the whole miss stream at the origin."
        )
    }

    pub fn scaled(from: u64, to: u64, delta_hit: f64, savings_hr: f64, rent_hr: f64) -> String {
        let grew = to > from;
        let net = savings_hr - rent_hr;
        if grew {
            format!(
                "Grew the pool from {} to {}. The extra memory should lift the hit rate {:.1} \
                 points, saving {} against {} of rent — {} net.",
                bytes(from),
                bytes(to),
                delta_hit * 100.0,
                usd_per_hour(savings_hr),
                usd_per_hour(rent_hr),
                usd_per_hour(net)
            )
        } else {
            format!(
                "Shrank the pool from {} to {}. The memory was no longer paying for itself: it \
                 was buying {} of savings against {} of rent.",
                bytes(from),
                bytes(to),
                usd_per_hour(savings_hr),
                usd_per_hour(rent_hr)
            )
        }
    }

    pub fn held(current: u64, wanted: u64, roi: f64, threshold: f64) -> String {
        if wanted > current {
            format!(
                "Considered growing from {} to {} and decided against it. The extra memory \
                 would return {:.2}x its cost, below the {:.2}x we require before spending.",
                bytes(current),
                bytes(wanted),
                roi,
                threshold
            )
        } else {
            format!("Held the pool at {}. Nothing about the workload justified a change.", bytes(current))
        }
    }

    pub fn policy_shifted(toward: &str, from_weight: f64, to_weight: f64, because: &str) -> String {
        format!(
            "Shifted weight toward {toward} eviction, {} to {}. {because} Nothing was \
             reconfigured — the strategy that is being rewarded changed.",
            percent(from_weight),
            percent(to_weight)
        )
    }

    pub fn regime_changed(from: &str, to: &str, confidence: f64) -> String {
        format!(
            "The workload looks like {to} now rather than {from} ({} confident). Admission \
             tightens or loosens accordingly.",
            percent(confidence)
        )
    }

    pub fn model_loaded(name: &str, features: usize, source: &str, auc: Option<f64>) -> String {
        let quality = match auc {
            Some(a) => format!(" It scored {:.3} AUC on held-out data.", a),
            None => String::new(),
        };
        format!(
            "Loaded {name} from {source}, using {features} features.{quality} The learned \
             preference starts at 20% influence and rises only as it proves itself."
        )
    }

    pub fn pressure(used: u64, capacity: u64, evictions_per_s: f64) -> String {
        format!(
            "The cache is {} full ({} of {}) and evicting {:.0} objects a second. Admission is \
             now refusing anything that cannot beat what it would displace.",
            percent(used as f64 / capacity.max(1) as f64),
            bytes(used),
            bytes(capacity),
            evictions_per_s
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_read_the_way_people_write_them() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(40_960), "41 KB");
        assert_eq!(bytes(1_854_200), "1.9 MB");
        assert_eq!(bytes(84_000_000), "84 MB");
        // 536,870,912 bytes is 0.5 GiB but 537 MB in the decimal units cloud providers
        // bill in, and decimal is what the pricing table uses. Reporting "0.54 GB" here
        // would disagree with the money in the same sentence.
        assert_eq!(bytes(536_870_912), "537 MB");
        assert_eq!(bytes(2_000_000_000), "2.00 GB");
    }

    #[test]
    fn tiny_amounts_of_money_stay_legible() {
        // The failure this prevents: a rebuild costing $0.0000089 printing as "$0.00",
        // which tells the reader nothing and makes the economics look fake.
        assert_eq!(usd(0.0), "$0");
        assert_eq!(usd(0.0000089), "0.0009 cents");
        assert_eq!(usd(0.0021), "0.21 cents");
        assert_eq!(usd(8.42), "$8.42");
        assert_eq!(usd(1234.0), "$1234");
    }

    #[test]
    fn durations_change_unit_where_a_person_would() {
        assert_eq!(ms(0.4), "0.4 ms");
        assert_eq!(ms(320.0), "320 ms");
        assert_eq!(ms(2_100.0), "2.1 s");
        assert_eq!(ms(120_000.0), "2.0 min");
    }

    #[test]
    fn a_ratio_barely_above_one_does_not_claim_to_be_impressive() {
        assert_eq!(multiple(4.3), "4.3x");
        assert_eq!(multiple(1.02), "just above");
        assert_eq!(multiple(0.5), "50% of");
        assert_eq!(multiple(f64::INFINITY), "far");
    }

    #[test]
    fn the_dominant_resource_is_named_not_averaged() {
        // A GPU-heavy rebuild and a database-heavy one of identical latency must read
        // differently, because they are worth different amounts.
        let gpu = dominant_cost(20.0, 300.0, 10.0, 0.0, 300.0 * 0.000_000_255 + 20.0 * 1.16e-8 + 10.0 * 3.2e-8);
        assert!(gpu.contains("GPU"), "expected GPU to dominate: {gpu}");

        let db = dominant_cost(20.0, 0.0, 300.0, 0.0, 300.0 * 3.2e-8 + 20.0 * 1.16e-8);
        assert!(db.contains("database"), "expected database to dominate: {db}");

        let api = dominant_cost(5.0, 0.0, 5.0, 0.02, 0.02);
        assert!(api.contains("API"), "expected the paid call to dominate: {api}");
    }

    #[test]
    fn sentences_contain_the_numbers_that_produced_them() {
        let s = say::admitted("rec:user:1842", 0.87, 0.000_0522, 100.0, 200.0, 0.0, 0.0, 4.3, 1.0, 2);
        assert!(s.contains("87%"), "missing the probability: {s}");
        assert!(s.contains("GPU"), "missing the dominant resource: {s}");
        assert!(s.contains("4.3x"), "missing the margin over the bar: {s}");
        assert!(s.contains("2 objects"), "missing what it displaced: {s}");
    }

    #[test]
    fn a_scan_rejection_explains_the_workload_not_the_object() {
        let s = say::rejected("k", "single_touch_scan_signature", 1000, 0.02, 0.1, 5.0);
        assert!(s.contains("scan"), "{s}");
        assert!(s.contains("flush"), "should say what admitting them would cost: {s}");
    }

    #[test]
    fn scaling_reports_the_trade_it_made() {
        let s = say::scaled(536_870_912, 805_306_368, 0.066, 7.20, 1.40);
        assert!(s.contains("6.6 points"), "{s}");
        assert!(s.contains("$7.20/hour"), "{s}");
        assert!(s.contains("$1.40/hour"), "{s}");
        assert!(s.contains("$5.80/hour"), "should state the net: {s}");
    }

    #[test]
    fn declining_to_scale_is_also_explained() {
        let s = say::held(536_870_912, 805_306_368, 1.08, 1.25);
        assert!(s.contains("1.08x") && s.contains("1.25x"), "{s}");
        assert!(s.contains("against it"), "{s}");
    }

    #[test]
    fn a_version_bump_says_why_it_did_not_flush() {
        let s = say::version_bumped("recommendation", 8);
        assert!(s.contains("Nothing was deleted"), "{s}");
        assert!(s.contains("miss stream"), "should name the failure it avoided: {s}");
    }

    #[test]
    fn routine_events_are_sampled_and_important_ones_never_are() {
        let mut log = AuditLog::new(1_000, 10);
        for i in 0..100 {
            log.record(
                i as f64,
                AuditKind::Admit,
                Severity::Info,
                format!("k{i}"),
                "app",
                "routine",
                vec![],
                0.0,
            );
        }
        assert_eq!(log.len(), 10, "routine admissions should be sampled 1 in 10");
        assert_eq!(log.suppressed, 90);

        for i in 0..20 {
            log.record(
                i as f64,
                AuditKind::Invalidate,
                Severity::Notice,
                "row:product:1",
                "analytics",
                "correctness",
                vec![],
                0.0,
            );
        }
        assert_eq!(log.len(), 30, "correctness events must never be sampled away");
    }

    #[test]
    fn the_log_is_bounded_and_newest_first() {
        let mut log = AuditLog::new(5, 1);
        for i in 0..20 {
            log.record(
                i as f64,
                AuditKind::Evict,
                Severity::Notice,
                format!("k{i}"),
                "app",
                format!("message {i}"),
                vec![Fact::new("size", "1 KB")],
                0.0,
            );
        }
        assert_eq!(log.len(), 5);
        let recent = log.recent(3);
        assert_eq!(recent[0].subject, "k19", "most recent must come first");
        assert_eq!(recent[2].subject, "k17");
    }

    #[test]
    fn failed_shipments_are_requeued_not_lost() {
        let mut log = AuditLog::new(100, 1);
        for i in 0..10 {
            log.record(i as f64, AuditKind::Evict, Severity::Notice, "k", "app", "m", vec![], 0.0);
        }
        let batch = log.drain_unshipped(10);
        assert_eq!(batch.len(), 10);
        assert_eq!(log.pending_shipment(), 0);
        log.requeue(batch);
        assert_eq!(log.pending_shipment(), 10, "a failed push must not lose the entries");
        let again = log.drain_unshipped(10);
        assert_eq!(again[0].seq, 1, "requeued entries must keep their order");
    }
}
