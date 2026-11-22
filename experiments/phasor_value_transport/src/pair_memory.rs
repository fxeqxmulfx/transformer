//! Learned values of the preceding and current tokens at every position.
//! Build the incidence lists from input tokens alone. Their transpose gathers
//! gradients in minibatch order, including repeated addresses and short tails.
use crate::engine::{Config, Result};
use kernite_cuda::{Buffer, Cuda};

const PAIR_TABLES: usize = 17;
pub const TABLES: usize = PAIR_TABLES + cfg!(feature = "transition-table") as usize;
// A build without a historical selector uses the promoted closed-relation reader.
pub const DEFAULT_MODEL: bool = !cfg!(any(
    feature = "pair-v2",
    feature = "prefix-xor",
    feature = "prefix-sum"
));
#[cfg(all(feature = "relation-paths", feature = "prefix-frequency"))]
compile_error!("choose one additional prefix read per trial");
#[cfg(all(
    feature = "closed-relation",
    any(feature = "relation-paths", feature = "prefix-frequency")
))]
compile_error!("choose one relation read per trial");
const THREE_QUARTER_GAIN: bool = DEFAULT_MODEL || cfg!(feature = "successor-gain-three-quarters");
const HALF_GAIN: bool = cfg!(feature = "prefix-successor-half");
pub const SUCCESSOR: bool = DEFAULT_MODEL || cfg!(feature = "prefix-successor");
pub const XOR_PREFIX: bool = SUCCESSOR || cfg!(feature = "prefix-xor");
pub const CLOSED_RELATION: bool = DEFAULT_MODEL || cfg!(feature = "closed-relation");
pub const CLOSED_GAIN: bool = DEFAULT_MODEL || cfg!(feature = "closed-relation-boost");
pub const LEARN_BONUS: bool = cfg!(feature = "successor-bonus");
pub const HIT_GAIN: bool = LEARN_BONUS || CLOSED_GAIN || cfg!(feature = "successor-hit-gain");
pub const INITIAL_BONUS: f32 = 0.5;
pub const HIT_FACTOR: f32 = if cfg!(feature = "successor-hit-gain-seven-quarters") {
    1.75
} else {
    2.0
};
pub const INITIAL_GAIN: f32 = if THREE_QUARTER_GAIN {
    0.75
} else if HALF_GAIN {
    0.5
} else {
    0.25
};
pub const DISPLACEMENT: f32 = if cfg!(feature = "successor-displacement-quarter") {
    0.25
} else if DEFAULT_MODEL || cfg!(feature = "successor-displacement") {
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
#[cfg(all(
    feature = "successor-gain-three-quarters",
    any(
        feature = "transition-count",
        feature = "successor-hit-gain",
        feature = "successor-displacement-quarter"
    )
))]
compile_error!(
    "the three-quarter gain trial specifies full displacement and the original table addresses"
);

#[cfg(all(
    feature = "successor-hit-gain",
    any(
        feature = "transition-count",
        feature = "successor-displacement-quarter"
    )
))]
compile_error!(
    "the retrieval-dependent gain trial uses full displacement and the original table addresses"
);

#[cfg(all(
    feature = "successor-bonus",
    any(
        feature = "successor-hit-gain",
        feature = "successor-gain-three-quarters",
        feature = "successor-displacement-quarter",
        feature = "transition-count"
    )
))]
compile_error!("the independent bonus trial uses a half-gain base and full successor displacement");

pub const MODEL: &str = if LEARN_BONUS {
    "phasor-successor-independent-bonus-v1"
} else if cfg!(feature = "successor-agreement") {
    if cfg!(feature = "successor-hit-gain-seven-quarters") {
        "phasor-successor-agreement-gain175-v1"
    } else {
        "phasor-successor-agreement-v1"
    }
} else if cfg!(feature = "successor-hit-gain-seven-quarters") {
    "phasor-successor-hit-gain175-v1"
} else if cfg!(feature = "inclusive-transitions") {
    "phasor-inclusive-transition-table-v1"
} else if cfg!(feature = "transition-table") {
    "phasor-independent-transition-table-v1"
} else if cfg!(feature = "single-write-gain") {
    "phasor-single-write-or-closed-gain-v1"
} else if CLOSED_GAIN {
    "phasor-closed-relation-gain-v1"
} else if HIT_GAIN {
    "phasor-successor-hit-gain-v1"
} else if CLOSED_RELATION {
    "phasor-closed-relation-address-v1"
} else if cfg!(feature = "relation-support") {
    "phasor-composable-relation-support-v1"
} else if cfg!(feature = "relation-path-mean") {
    "phasor-mean-two-hop-relation-values-v1"
} else if cfg!(feature = "relation-paths") {
    "phasor-two-hop-relation-values-v1"
} else if cfg!(feature = "prefix-frequency") {
    "phasor-centered-prefix-frequency-v1"
} else if THREE_QUARTER_GAIN {
    "phasor-successor-displacement-gain075-v1"
} else if cfg!(feature = "transition-count") {
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
    if cfg.head == crate::head::Head::Relations {
        TABLES * cfg.vocab * cfg.width
            + 1
            + usize::from(LEARN_BONUS)
            + if cfg!(feature = "relation-paths") {
                cfg.vocab
            } else {
                0
            }
    } else {
        0
    }
}

pub use crate::architecture::start;

pub fn gain(cfg: &Config) -> usize {
    start(cfg) + TABLES * cfg.vocab * cfg.width
}

pub fn initialize_gains(cfg: &Config, parameters: &mut [f32]) {
    if cfg.head == crate::head::Head::Relations {
        let index = gain(cfg);
        parameters[index] = INITIAL_GAIN;
        if LEARN_BONUS {
            parameters[index + 1] = INITIAL_BONUS;
        }
        if cfg!(feature = "relation-path-mean") {
            parameters[index + 1..index + 1 + cfg.vocab].fill(1.);
        }
    }
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

struct GainEvidence {
    reads: Vec<u32>,
    device: Buffer<u32>,
}

pub struct PairMemory {
    pub kernels: crate::pair_ops::kernels::Kernels,
    pub keys: Buffer<u32>,
    pub offsets: Buffer<u32>,
    pub owners: Buffer<u32>,
    pub causal: crate::causal_values::kernels::Kernels,
    pub frequency: Option<crate::frequency_read::kernels::Kernels>,
    pub relations: Option<crate::relation_paths::RelationRead>,
    pub dscores: Buffer<f32>,
    pub merged: Buffer<f32>,
    pub scaled: Buffer<f32>,
    pub partial: Buffer<f32>,
    indices: Vec<u32>,
    active_shape: Option<(usize, usize)>,
    starts: Vec<u32>,
    cursor: Vec<u32>,
    rows: Vec<u32>,
    factors: Vec<usize>,
    successors: Vec<Option<u32>>,
    closed: Option<crate::relation_support::ClosedRead>,
    ambiguous: Vec<bool>,
    writes: Vec<u8>,
    evidence: Option<GainEvidence>,
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
            frequency: cfg!(feature = "prefix-frequency")
                .then(|| crate::frequency_read::kernels::Kernels::load(cuda))
                .transpose()?,
            relations: cfg!(feature = "relation-paths")
                .then(|| crate::relation_paths::RelationRead::new(cuda, rows, cfg.vocab))
                .transpose()?,
            dscores: cuda.alloc(rows * cfg.vocab)?,
            merged: cuda.alloc(rows * cfg.width)?,
            scaled: cuda.alloc(rows * cfg.width)?,
            partial: cuda.alloc(rows * (1 + usize::from(LEARN_BONUS)))?,
            indices: vec![0; entries],
            active_shape: None,
            starts: vec![0; words + 1],
            cursor: vec![0; words],
            rows: vec![0; entries],
            factors: (0..cfg.vocab).map(|b| power_factor(b, cfg.vocab)).collect(),
            successors: vec![None; if SUCCESSOR { cfg.vocab } else { 0 }],
            closed: CLOSED_RELATION.then(|| crate::relation_support::ClosedRead::new(cfg.vocab)),
            ambiguous: vec![
                false;
                if cfg!(feature = "successor-agreement") {
                    cfg.vocab
                } else {
                    0
                }
            ],
            writes: vec![
                0;
                if cfg!(feature = "single-write-gain") {
                    cfg.vocab
                } else {
                    0
                }
            ],
            evidence: if CLOSED_GAIN || cfg!(feature = "successor-agreement") {
                Some(GainEvidence {
                    reads: vec![0; rows],
                    device: cuda.alloc(rows)?,
                })
            } else {
                None
            },
        })
    }

    // Gated reads have a separate evidence address. A plain hit multiplier
    // uses the first address in each row of the ordinary table indices.
    pub fn gain_addresses(&self) -> (&Buffer<u32>, u32) {
        self.evidence
            .as_ref()
            .map_or((&self.keys, TABLES as u32), |a| (&a.device, 1))
    }

    pub fn prepare(&mut self, tokens: &[u32], length: usize) -> Result<()> {
        let p = self.factors.len();
        self.active_shape = None;
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
                    self.ambiguous.fill(false);
                    self.writes.fill(0);
                } else if pos >= 2 {
                    let (key, value) = (tokens[row - 2] as usize, tokens[row - 1]);
                    if cfg!(feature = "successor-agreement")
                        && self.successors[key].is_some_and(|old| old != value)
                    {
                        self.ambiguous[key] = true;
                    }
                    self.successors[key] = Some(value);
                    if cfg!(feature = "single-write-gain") {
                        self.writes[key] = (self.writes[key] + 1).min(2);
                    }
                }
                keys[0] = self.successors[b].unwrap_or(token);
                let closed_end = if let Some(closed) = &mut self.closed {
                    if pos == 0 {
                        closed.clear();
                    } else if pos >= 3 {
                        closed.write(tokens[row - 3], tokens[row - 2], tokens[row - 1]);
                    }
                    closed.address(token)
                } else {
                    None
                };
                keys[0] = closed_end.unwrap_or(keys[0]);
                if let Some(evidence) = &mut self.evidence {
                    evidence.reads[row] = if CLOSED_GAIN {
                        if cfg!(feature = "single-write-gain") && self.writes[b] == 1 {
                            keys[0]
                        } else {
                            closed_end.unwrap_or(token)
                        }
                    } else if self.ambiguous[b] {
                        token
                    } else {
                        keys[0]
                    };
                }
                if cfg!(any(
                    feature = "transition-count",
                    feature = "transition-table"
                )) {
                    let changed = if cfg!(feature = "inclusive-transitions") {
                        a != b
                    } else {
                        pos >= 2 && tokens[row - 2] != tokens[row - 1]
                    };
                    if pos == 0 {
                        transitions = 0;
                    } else if changed {
                        transitions += 1;
                    }
                    if cfg!(feature = "transition-table") {
                        let count_table = TABLES - 1;
                        let bucket = if cfg!(feature = "inclusive-transitions") {
                            transitions.min(p - 1)
                        } else {
                            transitions % p
                        };
                        keys[count_table] = (count_table * p + bucket) as u32;
                    } else {
                        let squared_product = PAIR_TABLES - 1;
                        let base = squared_product * p;
                        let address = keys[squared_product] as usize - base;
                        keys[squared_product] = (base + (address + transitions) % p) as u32;
                    }
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
        }
        self.keys.write(&self.indices)?;
        if let Some(evidence) = &mut self.evidence {
            evidence.device.write(&evidence.reads)?;
        }
        if let Some(relations) = &mut self.relations {
            relations.prepare(tokens, length)?;
        }
        self.active_shape = Some((tokens.len() / length, length));
        Ok(())
    }

    // Only differentiation needs the transpose of the forward gather.
    // Build it in row order for the exact batch that produced the logits.
    pub fn prepare_transpose(&mut self, batch: usize, length: usize) -> Result<()> {
        if self.active_shape != Some((batch, length)) {
            return Err("table gradients require a forward pass of the same shape".into());
        }
        let active = &self.indices[..batch * length * TABLES];
        self.starts.fill(0);
        for &key in active {
            self.starts[key as usize + 1] += 1;
        }
        for i in 1..self.starts.len() {
            self.starts[i] += self.starts[i - 1];
        }
        self.cursor
            .copy_from_slice(&self.starts[..self.starts.len() - 1]);
        for (i, &k) in active.iter().enumerate() {
            let offset = &mut self.cursor[k as usize];
            self.rows[*offset as usize] = (i / TABLES) as u32;
            *offset += 1;
        }
        self.offsets.write(&self.starts)?;
        self.owners.write(&self.rows)?;
        Ok(())
    }
}
