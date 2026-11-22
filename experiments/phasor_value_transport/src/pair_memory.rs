//! Learned values of the preceding and current tokens at every position.
//! Build the incidence lists from input tokens alone. Their transpose gathers
//! gradients in minibatch order, including repeated addresses and short tails.
use crate::engine::{Config, Result};
use kernite_cuda::{Buffer, Cuda};

pub const TABLES: usize = 17;
// With no experiment selector a binary contains the selected successor model.
const DEFAULT_MODEL: bool = !cfg!(any(
    feature = "pair-v2",
    feature = "prefix-xor",
    feature = "prefix-sum"
));
const HALF_GAIN: bool = DEFAULT_MODEL || cfg!(feature = "prefix-successor-half");
pub const SUCCESSOR: bool = HALF_GAIN || cfg!(feature = "prefix-successor");
pub const XOR_PREFIX: bool = SUCCESSOR || cfg!(feature = "prefix-xor");
pub const INITIAL_GAIN: f32 = if HALF_GAIN { 0.5 } else { 0.25 };
pub const DISPLACEMENT: f32 = if cfg!(feature = "successor-displacement-quarter") {
    0.25
} else if cfg!(feature = "successor-displacement") {
    1.0
} else {
    0.0
};

#[cfg(all(feature = "prefix-xor", feature = "prefix-sum"))]
compile_error!("choose one prefix architecture per binary");
#[cfg(all(
    feature = "pair-v2",
    any(feature = "prefix-xor", feature = "prefix-sum")
))]
compile_error!("pair-v2 replays the original pair-address control");
#[cfg(all(feature = "prefix-xor-cubic", feature = "prefix-successor"))]
compile_error!("the successor trial uses the linear prefix sum");

pub const MODEL: &str = if cfg!(feature = "transition-count") {
    if cfg!(feature = "successor-displacement-quarter") {
        "phasor-transitions-displacement-quarter-v1"
    } else if cfg!(feature = "successor-displacement") {
        "phasor-transitions-displacement-v1"
    } else {
        "phasor-transitions-v1"
    }
} else if cfg!(feature = "successor-displacement-quarter") {
    "phasor-successor-displacement-quarter-v1"
} else if cfg!(feature = "successor-displacement") {
    "phasor-successor-displacement-v1"
} else if HALF_GAIN {
    "phasor-prefix-successor-half-v1"
} else if cfg!(feature = "prefix-successor") {
    "phasor-prefix-successor-v1"
} else if cfg!(feature = "prefix-xor-cubic") {
    "phasor-prefix-xor-cubic-v1"
} else if cfg!(feature = "prefix-xor") {
    "phasor-prefix-xor-v1"
} else if cfg!(feature = "prefix-sum") {
    "phasor-prefix-sum-v1"
} else {
    "phasor-causal-value-residual-v2"
};

pub fn parameters(cfg: &Config) -> usize {
    if cfg.shared_values {
        TABLES * cfg.vocab * cfg.width + 1
    } else {
        0
    }
}

pub fn start(cfg: &Config) -> usize {
    crate::layout::score_offset(cfg) + crate::layout::score_count(cfg)
}

pub fn gain(cfg: &Config) -> usize {
    start(cfg) + TABLES * cfg.vocab * cfg.width
}

fn power_factor(x: usize, p: usize) -> usize {
    if x == 0 {
        return 0;
    }
    let (mut base, mut exp, mut acc) = (x, p.saturating_sub(2), 1);
    while exp > 0 {
        if exp % 2 == 1 {
            acc = acc * base % p;
        }
        base = base * base % p;
        exp /= 2;
    }
    acc
}

fn addresses(a: usize, b: usize, p: usize, factor_b: usize) -> [u32; TABLES] {
    let powers = |x| [x, x * x % p, (x * x % p) * x % p];
    let (left, right) = (powers(a), powers(b));
    let mut result = [0; TABLES];
    let mut keys = result.iter_mut().enumerate();
    let mut put = |value| {
        let (table, dest) = keys.next().unwrap();
        *dest = (table * p + value) as u32;
    };
    for value in [b, (a + p - b) % p, a.abs_diff(b), a * factor_b % p] {
        put(value);
    }
    for &x in &left {
        for &y in &right {
            put((x + y) % p);
        }
    }
    for &x in &left[..2] {
        for &y in &right[..2] {
            put(x * y % p);
        }
    }
    result
}

pub struct PairMemory {
    pub kernels: crate::pair_ops::kernels::Kernels,
    pub keys: Buffer<u32>,
    pub offsets: Buffer<u32>,
    pub owners: Buffer<u32>,
    pub causal: crate::causal_values::kernels::Kernels,
    pub dscores: Buffer<f32>,
    pub merged: Buffer<f32>,
    pub scaled: Buffer<f32>,
    pub partial: Buffer<f32>,
    indices: Vec<u32>,
    starts: Vec<u32>,
    cursor: Vec<u32>,
    rows: Vec<u32>,
    factors: Vec<usize>,
    successors: Vec<Option<u32>>,
}

impl PairMemory {
    pub fn new(cuda: &Cuda, cfg: &Config) -> Result<Self> {
        let words = TABLES * cfg.vocab;
        let rows = cfg.max_batch * cfg.max_length;
        let entries = TABLES * rows;
        Ok(Self {
            kernels: crate::pair_ops::kernels::Kernels::load(cuda)?,
            keys: cuda.alloc(entries)?,
            offsets: cuda.alloc(words + 1)?,
            owners: cuda.alloc(entries)?,
            causal: crate::causal_values::kernels::Kernels::load(cuda)?,
            dscores: cuda.alloc(rows * cfg.vocab)?,
            merged: cuda.alloc(rows * cfg.width)?,
            scaled: cuda.alloc(rows * cfg.width)?,
            partial: cuda.alloc(rows)?,
            indices: vec![0; entries],
            starts: vec![0; words + 1],
            cursor: vec![0; words],
            rows: vec![0; entries],
            factors: (0..cfg.vocab).map(|b| power_factor(b, cfg.vocab)).collect(),
            successors: vec![None; if SUCCESSOR { cfg.vocab } else { 0 }],
        })
    }

    pub fn prepare(&mut self, tokens: &[u32], length: usize) -> Result<()> {
        let p = self.factors.len();
        let used = tokens.len() * TABLES;
        self.starts.fill(0);
        let mut prefix = 0;
        let mut transitions = 0;
        for (row, (&token, keys)) in tokens
            .iter()
            .zip(self.indices.as_chunks_mut::<TABLES>().0)
            .enumerate()
        {
            let prev = if row.is_multiple_of(length) {
                row
            } else {
                row - 1
            };
            let (a, b) = (tokens[prev] as usize, token as usize);
            keys.copy_from_slice(&addresses(a, b, p, self.factors[b]));
            if SUCCESSOR {
                let pos = row % length;
                if pos == 0 {
                    self.successors.fill(None);
                } else if pos >= 2 {
                    self.successors[tokens[row - 2] as usize] = Some(tokens[row - 1]);
                }
                keys[0] = self.successors[b].unwrap_or(token);
                if cfg!(feature = "transition-count") {
                    if pos == 0 {
                        transitions = 0;
                    } else if pos >= 2 && tokens[row - 2] != tokens[row - 1] {
                        transitions += 1;
                    }
                    let squared_product = TABLES - 1;
                    let base = squared_product * p;
                    let address = keys[squared_product] as usize - base;
                    keys[squared_product] = (base + (address + transitions) % p) as u32;
                }
            }
            if XOR_PREFIX || cfg!(feature = "prefix-sum") {
                let left = if prev == row { b } else { prefix };
                if cfg!(feature = "prefix-xor-cubic") {
                    let cube = |x: usize| x * x % p * x % p;
                    keys[12] = (12 * p + (cube(left) + cube(b)) % p) as u32;
                } else {
                    keys[4] = (4 * p + (left + b) % p) as u32;
                }
                let earlier = if prev == row { 0 } else { prefix };
                prefix = if XOR_PREFIX {
                    earlier ^ b
                } else {
                    (earlier + b) % p
                };
            }
            for &k in keys.iter() {
                self.starts[k as usize + 1] += 1;
            }
        }
        for i in 1..self.starts.len() {
            self.starts[i] += self.starts[i - 1];
        }
        self.cursor.copy_from_slice(&self.starts[..TABLES * p]);
        for (i, &k) in self.indices[..used].iter().enumerate() {
            let offset = &mut self.cursor[k as usize];
            self.rows[*offset as usize] = (i / TABLES) as u32;
            *offset += 1;
        }
        self.keys.write(&self.indices)?;
        self.offsets.write(&self.starts)?;
        self.owners.write(&self.rows)?;
        Ok(())
    }
}
