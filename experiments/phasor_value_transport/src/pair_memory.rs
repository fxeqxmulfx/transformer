//! Learned hidden values for the field coordinates in Phasor.PairValues.
//! Build the incidence lists from input tokens alone. Their transpose gathers
//! gradients in minibatch order, including repeated addresses and short tails.
use crate::engine::{Config, Result};
use kernite_cuda::{Buffer, Cuda};

pub const TABLES: usize = 17;

pub fn active(cfg: &Config) -> bool {
    let p = cfg.vocab;
    cfg.pair_values
        && cfg.max_length == 2
        && p >= 2
        && (2..)
            .take_while(|&i| i <= p / i)
            .all(|i| !p.is_multiple_of(i))
}

pub fn parameters(cfg: &Config) -> usize {
    if active(cfg) {
        TABLES * cfg.vocab * cfg.width
    } else {
        0
    }
}

pub fn start(cfg: &Config) -> usize {
    crate::layout::score_offset(cfg) + crate::layout::score_count(cfg)
}

fn inverse(x: usize, p: usize) -> usize {
    if x == 0 {
        return 0;
    }
    let (mut base, mut exp, mut acc) = (x, p - 2, 1);
    while exp > 0 {
        if exp % 2 == 1 {
            acc = acc * base % p;
        }
        base = base * base % p;
        exp /= 2;
    }
    acc
}

fn addresses(a: usize, b: usize, p: usize, inverse_b: usize) -> [u32; TABLES] {
    let powers = |x| [x, x * x % p, (x * x % p) * x % p];
    let (left, right) = (powers(a), powers(b));
    let mut result = [0; TABLES];
    let mut keys = result.iter_mut().enumerate();
    let mut put = |value| {
        let (table, dest) = keys.next().unwrap();
        *dest = (table * p + value) as u32;
    };
    for value in [b, (a + p - b) % p, a.abs_diff(b), a * inverse_b % p] {
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
    pub output: Buffer<f32>,
    indices: Vec<u32>,
    starts: Vec<u32>,
    cursor: Vec<u32>,
    rows: Vec<u32>,
    inverses: Vec<usize>,
}

impl PairMemory {
    pub fn new(cuda: &Cuda, cfg: &Config) -> Result<Self> {
        let words = TABLES * cfg.vocab;
        let entries = TABLES * cfg.max_batch;
        Ok(Self {
            kernels: crate::pair_ops::kernels::Kernels::load(cuda)?,
            keys: cuda.alloc(entries)?,
            offsets: cuda.alloc(words + 1)?,
            owners: cuda.alloc(entries)?,
            output: cuda.alloc(2 * cfg.max_batch * cfg.vocab)?,
            indices: vec![0; entries],
            starts: vec![0; words + 1],
            cursor: vec![0; words],
            rows: vec![0; entries],
            inverses: (0..cfg.vocab).map(|b| inverse(b, cfg.vocab)).collect(),
        })
    }

    pub fn prepare(&mut self, tokens: &[u32]) -> Result<()> {
        let p = self.inverses.len();
        let used = tokens.len() / 2 * TABLES;
        self.starts.fill(0);
        for (pair, keys) in tokens
            .as_chunks::<2>()
            .0
            .iter()
            .zip(self.indices.as_chunks_mut::<TABLES>().0.iter_mut())
        {
            let (a, b) = (pair[0] as usize, pair[1] as usize);
            keys.copy_from_slice(&addresses(a, b, p, self.inverses[b]));
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
