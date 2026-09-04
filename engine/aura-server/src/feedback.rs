//! The decision journal: delayed, realised outcomes.
//!
//! Every claim about learning in this project depends on this file. A cache decision is a
//! prediction about the future, and the future has not happened yet when the decision is
//! made. Crediting a bandit arm or updating a model at decision time therefore teaches the
//! system nothing except what it already believed — it grades its own homework.
//!
//! So each decision is written down with the features it was made from and the probability
//! it was made with, and left alone. When enough time has passed that the prediction is
//! either confirmed or refuted by what actually happened, the record *settles*: the bandit
//! is credited with a realised reward and the online predictor is corrected against a real
//! label.
//!
//! ```text
//!   t0        decision recorded          predicted P(reuse<60s) = 0.72
//!    │
//!    ├── t0+3s   key requested again  ──► reused[10s] = true, reused[60s] = true
//!    │
//!    ▼
//!   t0+60s     settles                ──► bandit reward, predictor label = 1
//!    │
//!    ▼
//!   t0+600s    retires                ──► training row written, record dropped
//! ```
//!
//! Two horizons matter for two different consumers. The bandit and the online model settle
//! at 60 seconds, because a control loop that waits ten minutes for feedback cannot track a
//! workload that changes in one. Training rows retire at 600 seconds, when all three labels
//! are final.

use std::collections::VecDeque;

use ahash::AHashMap;
use aura_core::features::Features;
use aura_core::types::KeyId;
use serde::Serialize;

use crate::policy::Policy;

/// The reuse horizons, in milliseconds. Must match the trained bundle's `horizons_ms` and
/// `training/aura_train/config.py`.
pub const HORIZONS_MS: [f64; 3] = [10_000.0, 60_000.0, 600_000.0];

/// Horizon the bandit and the online model settle on. The 60-second label is the one that
/// is both meaningful and available quickly enough to steer a live cache.
const SETTLE_HORIZON: f64 = HORIZONS_MS[1];

/// Horizon at which all three labels are final and the record can be written out.
const RETIRE_HORIZON: f64 = HORIZONS_MS[2];

/// Ceiling on outstanding records, so a pathological key space cannot grow the journal
/// without bound. Oldest records are dropped first and simply never settle.
const MAX_PENDING: usize = 300_000;

/// A decision that has been made and whose outcome is not yet known.
#[derive(Debug, Clone)]
pub struct PendingDecision {
    pub id: u64,
    pub key: KeyId,
    pub application: String,
    pub features: Features,
    /// What the predictor said at decision time, for the three horizons.
    pub predicted: [f64; 3],
    /// Which arm the bandit was following when this decision was taken.
    pub policy: Policy,
    pub admitted: bool,
    pub value_density: f64,
    pub threshold: f64,
    /// Priced rebuild cost — the money a future hit would save.
    pub cost_usd: f64,
    /// Money spent holding the object for the settle window, had it been admitted.
    pub holding_usd: f64,
    pub size_bytes: u64,
    pub decided_ms: f64,
    /// Filled in by [`Journal::observe_access`] as the future actually arrives.
    pub reused: [bool; 3],
    settled: bool,
}

impl PendingDecision {
    /// Was this decision right, and by how much? Range `[0, 1]`.
    ///
    /// The four cases are not symmetric, and that asymmetry is deliberate:
    ///
    /// - **Admitted and reused** — correct, and worth more when the rebuild it avoided was
    ///   expensive relative to the rent paid. A cheap object that gets reused is a mild
    ///   success; an expensive one is the whole point of the cache.
    /// - **Admitted and not reused** — wrong. We paid rent and displaced something else for
    ///   nothing.
    /// - **Rejected and not reused** — correct, and cheap to be right about. Scored well but
    ///   not perfectly, because refusing everything would otherwise look like genius.
    /// - **Rejected and reused** — the expensive mistake. We forced a rebuild we could have
    ///   avoided, so this is scored near zero.
    pub fn realised_reward(&self) -> f64 {
        let reused = self.reused[1];
        match (self.admitted, reused) {
            (true, true) => {
                let ratio = self.cost_usd / self.holding_usd.max(1e-12);
                // A 10x return on the rent is already excellent; saturate there so one
                // enormous object cannot dominate the posterior.
                0.55 + 0.45 * (ratio / 10.0).clamp(0.0, 1.0)
            }
            (true, false) => 0.15,
            (false, false) => 0.80,
            (false, true) => 0.05,
        }
    }
}

/// A decision whose 60-second outcome is now known.
#[derive(Debug, Clone)]
pub struct Settled {
    pub features: Features,
    pub predicted: [f64; 3],
    pub reused: [bool; 3],
    pub policy: Policy,
    pub reward: f64,
    pub admitted: bool,
}

/// A fully matured record, ready to become a training row.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingRow {
    pub decided_ms: f64,
    pub key: KeyId,
    pub application: String,
    pub features: Vec<f64>,
    pub label_h10s: u8,
    pub label_h60s: u8,
    pub label_h600s: u8,
    pub predicted_h60s: f64,
    pub admitted: bool,
}

/// Running quality of the system's own predictions, for the dashboard.
///
/// Calibration is the number worth watching: if the model says 0.70 and reality comes back
/// 0.42, the model is confidently wrong and the confidence floor is the only thing keeping
/// the cache sane.
#[derive(Debug, Clone, Default, Serialize)]
pub struct JournalStats {
    pub pending: usize,
    pub settled: u64,
    pub retired: u64,
    pub dropped_overflow: u64,
    /// Fraction of admissions that were actually reused within 60 s.
    pub admit_precision: f64,
    /// Fraction of rejections that were correctly refused.
    pub reject_precision: f64,
    /// Mean predicted P(reuse within 60 s) over settled decisions.
    pub mean_predicted: f64,
    /// Mean realised reuse rate over the same decisions.
    pub mean_realised: f64,
    /// `mean_predicted - mean_realised`. Positive means the model is over-optimistic.
    pub calibration_error: f64,
    pub mean_reward: f64,
}

#[derive(Debug)]
pub struct Journal {
    next_id: u64,
    pending: AHashMap<u64, PendingDecision>,
    by_key: AHashMap<KeyId, Vec<u64>>,
    /// Insertion order. Because every record has the same horizons, insertion order is
    /// maturity order, so settling is a scan from the front rather than a sort.
    order: VecDeque<u64>,

    settled_count: u64,
    retired_count: u64,
    dropped: u64,

    admits: u64,
    admits_reused: u64,
    rejects: u64,
    rejects_not_reused: u64,
    sum_predicted: f64,
    sum_realised: f64,
    sum_reward: f64,

    /// Retired rows waiting to be drained by the trace writer.
    completed: VecDeque<TrainingRow>,
    completed_cap: usize,
}

impl Default for Journal {
    fn default() -> Self {
        Self::new()
    }
}

impl Journal {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            pending: AHashMap::new(),
            by_key: AHashMap::new(),
            order: VecDeque::new(),
            settled_count: 0,
            retired_count: 0,
            dropped: 0,
            admits: 0,
            admits_reused: 0,
            rejects: 0,
            rejects_not_reused: 0,
            sum_predicted: 0.0,
            sum_realised: 0.0,
            sum_reward: 0.0,
            completed: VecDeque::new(),
            completed_cap: 100_000,
        }
    }

    /// Write down a decision. Returns its id, which the explain record carries so a viewer
    /// can later look up whether the decision turned out to be right.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        key: KeyId,
        application: &str,
        features: Features,
        predicted: [f64; 3],
        policy: Policy,
        admitted: bool,
        value_density: f64,
        threshold: f64,
        cost_usd: f64,
        holding_usd: f64,
        size_bytes: u64,
        now_ms: f64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.pending.insert(
            id,
            PendingDecision {
                id,
                key,
                application: application.to_string(),
                features,
                predicted,
                policy,
                admitted,
                value_density,
                threshold,
                cost_usd,
                holding_usd,
                size_bytes,
                decided_ms: now_ms,
                reused: [false; 3],
                settled: false,
            },
        );
        self.by_key.entry(key).or_default().push(id);
        self.order.push_back(id);

        while self.order.len() > MAX_PENDING {
            if let Some(old) = self.order.pop_front() {
                self.forget(old);
                self.dropped += 1;
            }
        }
        id
    }

    /// The future arriving. Called on every access to `key`, hit or miss.
    ///
    /// A miss counts as reuse just as much as a hit does: the question the label answers is
    /// "was this object wanted again", and whether the cache happened to still hold it is
    /// exactly the thing we are trying to learn to get right. Counting only hits would
    /// teach the model that whatever the cache kept was the right thing to keep.
    pub fn observe_access(&mut self, key: KeyId, now_ms: f64) {
        let Some(ids) = self.by_key.get(&key) else { return };
        for id in ids.iter() {
            let Some(p) = self.pending.get_mut(id) else { continue };
            let elapsed = now_ms - p.decided_ms;
            if elapsed <= 0.0 {
                continue;
            }
            for h in 0..3 {
                if elapsed <= HORIZONS_MS[h] {
                    p.reused[h] = true;
                }
            }
        }
    }

    /// Decisions whose 60-second verdict is in. Feeds the bandit and the online predictor.
    ///
    /// Scans from the front and stops at the first record that has not matured, which is
    /// correct because insertion order is maturity order.
    pub fn settle(&mut self, now_ms: f64, limit: usize) -> Vec<Settled> {
        // Collect the ids first: the loop below mutates several fields of `self`, and
        // holding an iterator over `order` across those writes is exactly the kind of
        // aliasing the borrow checker exists to stop.
        let mut ready: Vec<u64> = Vec::new();
        for id in self.order.iter().copied() {
            if ready.len() >= limit {
                break;
            }
            let Some(p) = self.pending.get(&id) else { continue };
            if p.settled {
                continue;
            }
            if now_ms - p.decided_ms < SETTLE_HORIZON {
                break;
            }
            ready.push(id);
        }

        let mut out = Vec::with_capacity(ready.len());
        for id in ready {
            let Some(p) = self.pending.get_mut(&id) else { continue };
            p.settled = true;
            let reward = p.realised_reward();
            let (predicted, reused, policy, admitted, features) =
                (p.predicted, p.reused, p.policy, p.admitted, p.features);

            self.settled_count += 1;
            self.sum_predicted += predicted[1];
            self.sum_realised += if reused[1] { 1.0 } else { 0.0 };
            self.sum_reward += reward;
            if admitted {
                self.admits += 1;
                if reused[1] {
                    self.admits_reused += 1;
                }
            } else {
                self.rejects += 1;
                if !reused[1] {
                    self.rejects_not_reused += 1;
                }
            }

            out.push(Settled { features, predicted, reused, policy, reward, admitted });
        }
        out
    }

    /// Drop records whose longest horizon has passed, turning each into a training row.
    pub fn retire(&mut self, now_ms: f64, limit: usize) {
        let mut done = 0;
        while done < limit {
            let Some(&id) = self.order.front() else { break };
            let Some(p) = self.pending.get(&id) else {
                self.order.pop_front();
                continue;
            };
            if now_ms - p.decided_ms < RETIRE_HORIZON {
                break;
            }
            let row = TrainingRow {
                decided_ms: p.decided_ms,
                key: p.key,
                application: p.application.clone(),
                features: p.features.to_vec(),
                label_h10s: p.reused[0] as u8,
                label_h60s: p.reused[1] as u8,
                label_h600s: p.reused[2] as u8,
                predicted_h60s: p.predicted[1],
                admitted: p.admitted,
            };
            if self.completed.len() >= self.completed_cap {
                self.completed.pop_front();
            }
            self.completed.push_back(row);
            self.retired_count += 1;

            self.order.pop_front();
            self.forget(id);
            done += 1;
        }
    }

    fn forget(&mut self, id: u64) {
        if let Some(p) = self.pending.remove(&id) {
            if let Some(ids) = self.by_key.get_mut(&p.key) {
                ids.retain(|x| *x != id);
                if ids.is_empty() {
                    self.by_key.remove(&p.key);
                }
            }
        }
    }

    /// Take up to `n` finished training rows. The caller writes them to the trace file.
    pub fn drain_completed(&mut self, n: usize) -> Vec<TrainingRow> {
        let take = n.min(self.completed.len());
        self.completed.drain(..take).collect()
    }

    pub fn completed_len(&self) -> usize {
        self.completed.len()
    }

    pub fn stats(&self) -> JournalStats {
        let settled = self.settled_count.max(1) as f64;
        let mean_predicted = self.sum_predicted / settled;
        let mean_realised = self.sum_realised / settled;
        JournalStats {
            pending: self.pending.len(),
            settled: self.settled_count,
            retired: self.retired_count,
            dropped_overflow: self.dropped,
            admit_precision: if self.admits == 0 {
                0.0
            } else {
                self.admits_reused as f64 / self.admits as f64
            },
            reject_precision: if self.rejects == 0 {
                0.0
            } else {
                self.rejects_not_reused as f64 / self.rejects as f64
            },
            mean_predicted,
            mean_realised,
            calibration_error: mean_predicted - mean_realised,
            mean_reward: self.sum_reward / settled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features() -> Features {
        [0.0; aura_core::features::N_FEATURES]
    }

    fn journal_with(admitted: bool, reuse_at: Option<f64>) -> (Journal, Settled) {
        let mut j = Journal::new();
        j.record(
            7,
            "recommendation",
            features(),
            [0.6, 0.7, 0.8],
            Policy::Gdsf,
            admitted,
            5.0,
            1.0,
            0.004,
            0.0004,
            1024,
            0.0,
        );
        if let Some(t) = reuse_at {
            j.observe_access(7, t);
        }
        let settled = j.settle(61_000.0, 10);
        assert_eq!(settled.len(), 1);
        let s = settled[0].clone();
        (j, s)
    }

    #[test]
    fn nothing_settles_before_the_horizon() {
        let mut j = Journal::new();
        j.record(1, "a", features(), [0.5; 3], Policy::Lru, true, 1.0, 1.0, 0.001, 0.0001, 100, 0.0);
        assert!(j.settle(30_000.0, 10).is_empty(), "settled early");
        assert_eq!(j.settle(61_000.0, 10).len(), 1);
    }

    #[test]
    fn a_decision_settles_exactly_once() {
        let mut j = Journal::new();
        j.record(1, "a", features(), [0.5; 3], Policy::Lru, true, 1.0, 1.0, 0.001, 0.0001, 100, 0.0);
        assert_eq!(j.settle(61_000.0, 10).len(), 1);
        assert_eq!(j.settle(62_000.0, 10).len(), 0, "settled twice");
    }

    #[test]
    fn access_marks_only_the_horizons_that_have_not_elapsed() {
        let mut j = Journal::new();
        j.record(1, "a", features(), [0.5; 3], Policy::Lru, true, 1.0, 1.0, 0.001, 0.0001, 100, 0.0);
        // 30 s later: inside the 60 s and 600 s horizons, outside the 10 s one.
        j.observe_access(1, 30_000.0);
        let s = j.settle(61_000.0, 10);
        assert_eq!(s[0].reused, [false, true, true]);
    }

    #[test]
    fn rewards_order_the_four_outcomes_correctly() {
        let (_, admit_hit) = journal_with(true, Some(5_000.0));
        let (_, admit_miss) = journal_with(true, None);
        let (_, reject_right) = journal_with(false, None);
        let (_, reject_wrong) = journal_with(false, Some(5_000.0));

        assert!(admit_hit.reward > reject_right.reward, "a useful admission should beat a correct refusal");
        assert!(reject_right.reward > admit_miss.reward, "a correct refusal should beat a wasted admission");
        assert!(admit_miss.reward > reject_wrong.reward, "refusing something that was wanted is the worst case");
        assert!(reject_wrong.reward < 0.1);
    }

    #[test]
    fn an_expensive_saved_rebuild_is_rewarded_above_a_cheap_one() {
        let mut j = Journal::new();
        // Same size and reuse; only the rebuild cost differs.
        j.record(1, "a", features(), [0.5; 3], Policy::Gdsf, true, 1.0, 1.0, 0.0001, 0.0001, 100, 0.0);
        j.record(2, "a", features(), [0.5; 3], Policy::Gdsf, true, 1.0, 1.0, 0.0100, 0.0001, 100, 0.0);
        j.observe_access(1, 1_000.0);
        j.observe_access(2, 1_000.0);
        let s = j.settle(61_000.0, 10);
        assert!(s[1].reward > s[0].reward, "the expensive save should score higher");
    }

    #[test]
    fn retirement_produces_labelled_training_rows() {
        let mut j = Journal::new();
        j.record(1, "analytics", features(), [0.5; 3], Policy::Lru, true, 1.0, 1.0, 0.001, 0.0001, 100, 0.0);
        j.observe_access(1, 5_000.0);
        j.settle(61_000.0, 10);
        j.retire(601_000.0, 10);
        let rows = j.drain_completed(10);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].label_h10s, 1);
        assert_eq!(rows[0].label_h60s, 1);
        assert_eq!(rows[0].label_h600s, 1);
        assert_eq!(rows[0].features.len(), aura_core::features::N_FEATURES);
    }

    #[test]
    fn calibration_reports_an_over_optimistic_model() {
        let mut j = Journal::new();
        // The model claims 0.9 every time; only one in five is actually reused.
        for i in 0..10u64 {
            j.record(i, "a", features(), [0.9; 3], Policy::Lru, true, 1.0, 1.0, 0.001, 0.0001, 100, 0.0);
            if i % 5 == 0 {
                j.observe_access(i, 1_000.0);
            }
        }
        j.settle(61_000.0, 100);
        let st = j.stats();
        assert!(st.calibration_error > 0.5, "expected a large positive calibration error, got {}", st.calibration_error);
        assert!((st.admit_precision - 0.2).abs() < 1e-9);
    }

    #[test]
    fn the_journal_stays_bounded() {
        let mut j = Journal::new();
        for i in 0..(MAX_PENDING as u64 + 5_000) {
            j.record(i, "a", features(), [0.5; 3], Policy::Lru, true, 1.0, 1.0, 0.001, 0.0001, 100, 0.0);
        }
        assert!(j.pending.len() <= MAX_PENDING);
        assert!(j.stats().dropped_overflow >= 5_000);
    }
}
