//! Small, cheap estimators that run on every access.
//!
//! Nothing here allocates during steady state and nothing here is allowed to be more than
//! a few nanoseconds, because these are the only structures the hot path touches.

use crate::types::KeyId;

const LN2: f64 = std::f64::consts::LN_2;

/// Exponentially decayed access counter.
///
/// Reads decay the counter to *now* and return it; folding an access in is a separate
/// step. Keeping those two operations separate is what lets the feature builder emit a
/// value that reflects strictly prior accesses — the training pipeline does the same, and
/// getting this order wrong is the single easiest way to leak the present into a feature.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecayCounter {
    value: f64,
}

impl DecayCounter {
    /// Decay by an elapsed time and a *time constant* (exponential, `exp(-dt/tau)`).
    pub fn decayed_tau(&self, dt_ms: f64, tau_ms: f64) -> f64 {
        if dt_ms > 0.0 {
            self.value * (-dt_ms / tau_ms).exp()
        } else {
            self.value
        }
    }

    /// Decay by an elapsed time and a *half-life* in seconds.
    pub fn decayed_half_life(&self, dt_ms: f64, half_life_s: f64) -> f64 {
        if dt_ms > 0.0 {
            self.value * (-LN2 * (dt_ms / 1000.0) / half_life_s).exp()
        } else {
            self.value
        }
    }

    pub fn set(&mut self, value: f64) {
        self.value = value;
    }

    pub fn get(&self) -> f64 {
        self.value
    }
}

/// Count-min sketch with 4-bit counters and periodic halving.
///
/// This is the admission filter's memory. Four bits per counter keeps a sketch for
/// hundreds of thousands of keys inside a few hundred kilobytes; the halving pass is what
/// stops a key that was popular an hour ago from holding a slot forever, which is exactly
/// the failure mode that makes plain LFU collapse under a popularity shift.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    /// Packed 4-bit counters, two per byte, `depth` rows of `width` counters.
    table: Vec<u8>,
    width: usize,
    depth: usize,
    mask: u64,
    seeds: Vec<u64>,
    additions: u64,
    reset_at: u64,
}

impl CountMinSketch {
    /// `capacity` is the number of distinct keys the sketch is sized for.
    pub fn new(capacity: usize) -> Self {
        let width = capacity.next_power_of_two().max(1024);
        let depth = 4;
        let seeds = vec![
            0x9E37_79B9_7F4A_7C15,
            0xC2B2_AE3D_27D4_EB4F,
            0x1656_67B1_9E37_79F9,
            0xD6E8_FEB8_6659_FD93,
        ];
        Self {
            table: vec![0u8; width * depth / 2],
            width,
            depth,
            mask: (width - 1) as u64,
            seeds,
            additions: 0,
            // Halve once the sketch has seen roughly ten additions per counter.
            reset_at: (width as u64) * 10,
        }
    }

    #[inline]
    fn index(&self, key: KeyId, row: usize) -> usize {
        let mut h = key ^ self.seeds[row];
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
        h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        h ^= h >> 33;
        row * self.width + (h & self.mask) as usize
    }

    #[inline]
    fn get_nibble(&self, slot: usize) -> u8 {
        let byte = self.table[slot / 2];
        if slot % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        }
    }

    #[inline]
    fn set_nibble(&mut self, slot: usize, value: u8) {
        let byte = &mut self.table[slot / 2];
        if slot % 2 == 0 {
            *byte = (*byte & 0xF0) | (value & 0x0F);
        } else {
            *byte = (*byte & 0x0F) | ((value & 0x0F) << 4);
        }
    }

    /// Estimated frequency of `key`, saturating at 15.
    pub fn estimate(&self, key: KeyId) -> u8 {
        let mut min = u8::MAX;
        for row in 0..self.depth {
            let slot = self.index(key, row);
            min = min.min(self.get_nibble(slot));
        }
        min
    }

    /// Record an access. Counters saturate rather than wrap.
    pub fn increment(&mut self, key: KeyId) {
        let mut min = u8::MAX;
        let mut slots = [0usize; 8];
        for row in 0..self.depth {
            let slot = self.index(key, row);
            slots[row] = slot;
            min = min.min(self.get_nibble(slot));
        }
        if min < 15 {
            // Conservative update: only raise the counters that are at the minimum. This
            // meaningfully reduces over-estimation from collisions.
            for row in 0..self.depth {
                let slot = slots[row];
                if self.get_nibble(slot) == min {
                    self.set_nibble(slot, min + 1);
                }
            }
        }
        self.additions += 1;
        if self.additions >= self.reset_at {
            self.halve();
        }
    }

    /// Age the whole sketch. Frequency history has to fade or the sketch becomes a
    /// permanent record of who was popular first.
    pub fn halve(&mut self) {
        for byte in self.table.iter_mut() {
            *byte = (*byte >> 1) & 0x77;
        }
        self.additions = 0;
    }

    pub fn memory_bytes(&self) -> usize {
        self.table.len()
    }
}

/// Streaming quantile estimator using the pinball (quantile-loss) gradient step.
///
/// A per-group histogram would be exact, but the engine tracks quantiles for every
/// (application, object_type) pair and must survive on the hot path. The training pipeline
/// implements the identical update, so the `cost_variance_ratio` feature means the same
/// thing offline and online.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantilePair {
    pub p50: f64,
    pub p95: f64,
    pub seen: bool,
    pub count: u64,
}

impl QuantilePair {
    pub fn observe(&mut self, x: f64, lr: f64) {
        if !self.seen {
            self.p50 = x;
            self.p95 = x;
            self.seen = true;
            self.count = 1;
            return;
        }
        let step50 = lr * self.p50.max(1.0);
        self.p50 += if x > self.p50 { step50 * 0.5 } else { -step50 * 0.5 };
        let step95 = lr * self.p95.max(1.0);
        self.p95 += if x > self.p95 { step95 * 0.95 } else { -step95 * 0.05 };
        if self.p95 < self.p50 {
            self.p95 = self.p50;
        }
        self.count += 1;
    }

    pub fn variance_ratio(&self) -> f64 {
        self.p95 / self.p50.max(1.0)
    }
}

/// Fixed-capacity ring of `f64` used for windowed statistics (latency percentiles,
/// inter-arrival spread, burstiness).
#[derive(Debug, Clone)]
pub struct Reservoir {
    buf: Vec<f64>,
    next: usize,
    filled: bool,
}

impl Reservoir {
    pub fn new(capacity: usize) -> Self {
        Self { buf: vec![0.0; capacity.max(1)], next: 0, filled: false }
    }

    pub fn push(&mut self, x: f64) {
        self.buf[self.next] = x;
        self.next = (self.next + 1) % self.buf.len();
        if self.next == 0 {
            self.filled = true;
        }
    }

    pub fn len(&self) -> usize {
        if self.filled {
            self.buf.len()
        } else {
            self.next
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn quantile(&self, q: f64) -> f64 {
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.buf[..n].to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((n - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
        sorted[idx]
    }

    pub fn mean(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        self.buf[..n].iter().sum::<f64>() / n as f64
    }

    pub fn std_dev(&self) -> f64 {
        let n = self.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let var = self.buf[..n].iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
        var.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sketch_tracks_relative_frequency() {
        let mut cms = CountMinSketch::new(4096);
        for _ in 0..12 {
            cms.increment(7);
        }
        cms.increment(9);
        assert!(cms.estimate(7) > cms.estimate(9));
        assert!(cms.estimate(7) <= 15);
    }

    #[test]
    fn sketch_saturates_and_ages() {
        let mut cms = CountMinSketch::new(1024);
        for _ in 0..100 {
            cms.increment(1);
        }
        let before = cms.estimate(1);
        assert_eq!(before, 15);
        cms.halve();
        assert!(cms.estimate(1) < before);
    }

    #[test]
    fn quantiles_converge_and_stay_ordered() {
        let mut q = QuantilePair::default();
        // A distribution with a heavy tail: p95 must end up well above p50.
        for i in 0..8000 {
            let x = if i % 20 == 0 { 2000.0 } else { 100.0 };
            q.observe(x, 0.02);
        }
        assert!(q.p95 >= q.p50);
        assert!(q.p50 < 400.0, "p50 drifted into the tail: {}", q.p50);
        assert!(q.variance_ratio() > 1.5);
    }

    #[test]
    fn decay_counter_halves_over_one_half_life() {
        let mut c = DecayCounter::default();
        c.set(8.0);
        let after = c.decayed_half_life(5_000.0, 5.0);
        assert!((after - 4.0).abs() < 1e-9);
    }

    #[test]
    fn reservoir_quantiles() {
        let mut r = Reservoir::new(101);
        for i in 0..=100 {
            r.push(i as f64);
        }
        assert!((r.quantile(0.5) - 50.0).abs() < 1.0);
        assert!((r.quantile(0.95) - 95.0).abs() < 1.0);
    }
}
