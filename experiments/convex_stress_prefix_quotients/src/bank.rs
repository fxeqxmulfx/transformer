//! Keep every observed signature up to an explicit cap, choosing the smallest
//! syntax within the same count-depth bound. CM58 covers observed substitution.
use crate::{
    pool::{Pool, mask},
    *,
};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Candidate {
    pub formula: Formula,
    pub nodes: usize,
    pub bits: Vec<u64>,
    pub counts: Vec<i16>,
}
pub struct Bank {
    pub rows: Vec<Candidate>,
    lookup: HashMap<Vec<u64>, usize>,
    pub visited: u64,
}
pub struct Expansion {
    pub bank: Bank,
    pub refusal: Option<String>,
}
impl Bank {
    pub fn new() -> Self {
        Self {
            rows: vec![],
            lookup: HashMap::new(),
            visited: 0,
        }
    }
    pub fn add(
        &mut self,
        pool: &Pool,
        bits: &[u64],
        nodes: usize,
        cap: usize,
        make: impl FnOnce() -> Formula,
    ) -> Result<()> {
        self.visited += 1;
        if let Some(&i) = self.lookup.get(bits) {
            if nodes < self.rows[i].nodes {
                self.rows[i].formula = make();
                self.rows[i].nodes = nodes;
            }
            return Ok(());
        }
        if self.rows.len() == cap {
            return Err("observed signature class cap exhausted".into());
        }
        let formula = make();
        if formula.nodes() != nodes {
            return Err("grammar size forecast mismatch".into());
        }
        self.lookup.insert(bits.to_vec(), self.rows.len());
        self.rows.push(Candidate {
            formula,
            nodes,
            bits: bits.to_vec(),
            counts: pool.counts(bits),
        });
        Ok(())
    }
    pub fn initial(pool: &Pool, cap: usize) -> Result<Self> {
        let mut out = Self::new();
        for f in grammar::base() {
            let signature = pool.signature(&f)?;
            out.add(pool, &signature, f.nodes(), cap, || f)?;
        }
        Ok(out)
    }
    pub fn expanded(pool: &Pool, children: &Self, cap: usize) -> Result<Expansion> {
        let mut out = Self::initial(pool, cap)?;
        for c in &children.rows {
            out.add(pool, &c.bits, c.nodes, cap, || c.formula.clone())?;
        }
        let literals: Vec<_> = (0..6)
            .map(|i| pool.signature(&grammar::literal(i)))
            .collect::<Result<_>>()?;
        let counts: Vec<_> = literals[..3].iter().map(|b| pool.counts(b)).collect();
        let mut comparison = vec![0u64; pool.nodes.len().div_ceil(64)];
        let mut signature = comparison.clone();
        for a in &children.rows {
            for b in &children.rows {
                for extra in [None, Some(0), Some(1), Some(2)] {
                    for offset in -2..=2 {
                        comparison.fill(0);
                        for i in 0..pool.nodes.len() {
                            if i32::from(a.counts[i])
                                < i32::from(b.counts[i])
                                    + extra.map_or(0, |s| i32::from(counts[s][i]))
                                    + offset
                            {
                                comparison[i / 64] |= 1 << (i % 64);
                            }
                        }
                        for negated in [false, true] {
                            for gate in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)]
                            {
                                let size =
                                    grammar::size(a.nodes, b.nodes, extra, offset, negated, gate);
                                if size > 127 {
                                    continue;
                                }
                                for (i, &x) in comparison.iter().enumerate() {
                                    signature[i] = (if negated { !x } else { x })
                                        & gate.map_or(u64::MAX, |g| literals[g][i]);
                                }
                                mask(&mut signature, pool.nodes.len());
                                if let Err(error) = out.add(pool, &signature, size, cap, || {
                                    grammar::make(
                                        &a.formula, &b.formula, extra, offset, negated, gate,
                                    )
                                }) {
                                    if error.to_string() != "observed signature class cap exhausted"
                                    {
                                        return Err(error);
                                    }
                                    return Ok(Expansion {
                                        bank: out,
                                        refusal: Some(error.to_string()),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Expansion {
            bank: out,
            refusal: None,
        })
    }
}
impl Default for Bank {
    fn default() -> Self {
        Self::new()
    }
}
