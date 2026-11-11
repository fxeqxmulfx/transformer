//! MT19937 as NumPy's legacy `RandomState` drives it: the seeding of `init_genrand`, 32-bit
//! outputs, the rejection sampler `rk_interval`, the 53-bit doubles of `rk_double`, and the
//! Fisher-Yates shuffle of `RandomState.shuffle`. The author's data generator shuffles the
//! equations with `np.random.RandomState(seed=0)`, so the split is reproduced only by this
//! generator. The tests hold it to published outputs of MT19937 and NumPy.

const N: usize = 624;
const M: usize = 397;

/// The generator's state: 624 words and the position of the next one.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Mt {
    key: [u32; N],
    pos: usize,
}

impl Mt {
    /// The state `np.random.RandomState(seed)` starts from, for an integer seed.
    pub fn new(seed: u32) -> Mt {
        let mut key = [0u32; N];
        let mut s = seed;
        for (pos, slot) in key.iter_mut().enumerate() {
            *slot = s;
            s = 1_812_433_253u32.wrapping_mul(s ^ (s >> 30)).wrapping_add(pos as u32 + 1);
        }
        Mt { key, pos: N }
    }

    fn refill(&mut self) {
        for i in 0..N {
            let y = (self.key[i] & 0x8000_0000) | (self.key[(i + 1) % N] & 0x7fff_ffff);
            let mag = if y & 1 == 1 { 0x9908_b0df } else { 0 };
            self.key[i] = self.key[(i + M) % N] ^ (y >> 1) ^ mag;
        }
        self.pos = 0;
    }

    /// The next 32-bit output, `rk_random`.
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == N {
            self.refill();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// A uniform integer of `0..=max`, as `rk_interval` draws it: outputs masked to the
    /// smallest `2^b - 1 >= max`, rejected while above `max`.
    pub fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for shift in [1, 2, 4, 8, 16] {
            mask |= mask >> shift;
        }
        loop {
            let value = self.next_u32() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// A double of `[0, 1)` with 53 random bits, `rk_double`.
    pub fn double(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// `RandomState.shuffle`: for `i` from the last index down to 1, swap `x[i]` with
    /// `x[interval(i)]`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.interval(i as u32) as usize;
            x.swap(i, j);
        }
    }

    /// A standard normal by Box-Muller from two doubles: `sqrt(-2 ln(1 - u1)) cos(2 pi u2)`.
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.double();
        let u2 = self.double();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// The state as 625 words: the key, then the position.
    pub fn words(&self) -> Vec<u32> {
        let mut out = self.key.to_vec();
        out.push(self.pos as u32);
        out
    }

    /// The generator of the 625 words that [`Mt::words`] gave.
    pub fn from_words(words: &[u32]) -> Mt {
        assert_eq!(words.len(), N + 1, "an MT19937 state has 625 words");
        let mut key = [0u32; N];
        key.copy_from_slice(&words[..N]);
        let pos = words[N] as usize;
        assert!(pos <= N, "the position of an MT19937 state is at most 624");
        Mt { key, pos }
    }
}
