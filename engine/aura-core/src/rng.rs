//! Deterministic random number generation.

#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed_from_u64(seed: u64) -> Self {
        // SplitMix64 to expand one seed into the four words of state; seeding all four
        // from the same value would correlate the first outputs.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // 53 significant bits is exactly the mantissa of an f64; taking the top bits
        // avoids the low-bit weakness some generators have.
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[0, n)`. Debiased by rejection, so small ranges stay exactly uniform.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % n;
            }
        }
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Standard normal via Box-Muller. Only used off the hot path.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn log_normal(&mut self, mu: f64, sigma: f64) -> f64 {
        (mu + sigma * self.normal()).exp()
    }

    /// Exponential with the given mean. Drives Poisson arrival processes.
    pub fn exponential(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.next_f64()).max(1e-12).ln()
    }

    pub fn bool_with(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Sample an index from unnormalised weights.
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return 0;
        }
        let mut target = self.next_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// Zipf sampler over `n` items with exponent `alpha`.
#[derive(Debug, Clone)]
pub struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    pub fn new(n: usize, alpha: f64) -> Self {
        let n = n.max(1);
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0;
        for i in 1..=n {
            acc += 1.0 / (i as f64).powf(alpha);
            cdf.push(acc);
        }
        let total = *cdf.last().unwrap_or(&1.0);
        for v in cdf.iter_mut() {
            *v /= total;
        }
        Self { cdf }
    }

    /// Returns a rank in `[0, n)`, most popular first.
    pub fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.next_f64();
        match self.cdf.binary_search_by(|p| p.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(i) => i,
            Err(i) => i.min(self.cdf.len() - 1),
        }
    }

    pub fn len(&self) -> usize {
        self.cdf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cdf.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_stream() {
        let mut a = Rng::seed_from_u64(42);
        let mut b = Rng::seed_from_u64(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_output_is_actually_uniform() {
        let mut rng = Rng::seed_from_u64(7);
        let mut buckets = [0u32; 10];
        for _ in 0..100_000 {
            buckets[(rng.next_f64() * 10.0) as usize] += 1;
        }
        for b in buckets {
            assert!((b as i64 - 10_000).abs() < 600, "bucket {b} is off");
        }
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = Rng::seed_from_u64(11);
        for n in [1u64, 2, 7, 32, 1000] {
            for _ in 0..1000 {
                assert!(rng.below(n) < n);
            }
        }
    }

    #[test]
    fn zipf_concentrates_on_the_head() {
        let z = Zipf::new(10_000, 1.2);
        let mut rng = Rng::seed_from_u64(3);
        let mut top100 = 0;
        for _ in 0..50_000 {
            if z.sample(&mut rng) < 100 {
                top100 += 1;
            }
        }
        // With alpha = 1.2 the top 1% of keys should take a large majority of requests.
        assert!(top100 > 25_000, "expected a heavy head, got {top100}/50000");
    }

    #[test]
    fn weighted_respects_the_weights() {
        let mut rng = Rng::seed_from_u64(5);
        let mut counts = [0; 3];
        for _ in 0..30_000 {
            counts[rng.weighted(&[0.1, 0.3, 0.6])] += 1;
        }
        assert!(counts[2] > counts[1] && counts[1] > counts[0]);
    }
}
