//! Bounded, training-only search over a supplied nested-count grammar.
//! This discrete feature discovery precedes convex CE. No formula seed, task
//! identity or heldout label is an input. Beam truncation is reported explicitly.
use crate::{
    observations::Observation,
    prefix::{Formula, Term},
    program::Program,
    *,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub depth: usize,
    pub beam: usize,
    pub retained: usize,
    pub max_candidates: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            depth: 3,
            beam: 64,
            retained: 32,
            max_candidates: 2_500_000,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage {
    pub depth: usize,
    pub children: usize,
    pub visited: usize,
    pub perfect_syntaxes: usize,
    pub retained_children: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Search {
    pub policy: Policy,
    pub stages: Vec<Stage>,
    pub programs: Vec<Program>,
    pub skipped: Option<String>,
    pub scalar_forecast: u64,
    pub buffer_forecast: u64,
}

#[derive(Clone)]
struct Candidate {
    formula: Formula,
    bits: Vec<u64>,
    errors: usize,
    hash: u64,
}
struct Bucket {
    capacity: usize,
    kind: usize,
    rows: Vec<Candidate>,
}
impl Bucket {
    fn key(&self, c: &Candidate) -> (usize, usize, u64) {
        match self.kind {
            0 => (c.errors, c.formula.nodes(), c.hash),
            1 | 3 => (c.formula.nodes(), c.errors, c.hash),
            _ => (0, 0, c.hash),
        }
    }
    fn add(
        &mut self,
        bits: &[u64],
        errors: usize,
        hash: u64,
        nodes: usize,
        make: impl FnOnce() -> Formula,
    ) {
        if self.capacity == 0 {
            return;
        }
        let key = match self.kind {
            0 => (errors, nodes, hash),
            1 | 3 => (nodes, errors, hash),
            _ => (0, 0, hash),
        };
        if self.rows.len() == self.capacity && key >= self.key(self.rows.last().unwrap()) {
            return;
        }
        if self.kind != 3
            && let Some(i) = self.rows.iter().position(|r| r.bits == bits)
        {
            if self.key(&self.rows[i]) <= key {
                return;
            }
            self.rows.remove(i);
        }
        let entry = Candidate {
            formula: make(),
            bits: bits.to_vec(),
            errors,
            hash,
        };
        if self.kind == 3 && self.rows.iter().any(|r| r.formula == entry.formula) {
            return;
        }
        let i = self.rows.partition_point(|r| self.key(r) <= key);
        self.rows.insert(i, entry);
        if self.rows.len() > self.capacity {
            self.rows.pop();
        }
    }
}
fn hash(bits: &[u64]) -> u64 {
    bits.iter().fold(0xcbf29ce484222325u64, |h, x| {
        (h ^ x).wrapping_mul(0x100000001b3)
    })
}
fn literal(index: usize) -> Formula {
    let f = Formula::Symbol([9, 10, 23][index % 3]);
    if index < 3 {
        f
    } else {
        Formula::Not(Box::new(f))
    }
}
fn make(
    a: &Formula,
    b: &Formula,
    extra: Option<usize>,
    offset: i32,
    negated: bool,
    gate: Option<usize>,
) -> Formula {
    let left = Term::Count(Box::new(a.clone()));
    let mut right = Term::Count(Box::new(b.clone()));
    if let Some(s) = extra {
        right = Term::Add(Box::new(right), Box::new(Term::Count(Box::new(literal(s)))));
    }
    if offset != 0 {
        right = Term::Add(Box::new(right), Box::new(Term::Constant(offset)));
    }
    let mut f = Formula::Less(Box::new(left), Box::new(right));
    if negated {
        f = Formula::Not(Box::new(f));
    }
    if let Some(g) = gate {
        f = Formula::And(Box::new(f), Box::new(literal(g)));
    }
    f
}
fn bits(values: impl Iterator<Item = bool>, n: usize) -> Vec<u64> {
    let mut out = vec![0; n.div_ceil(64)];
    for (i, b) in values.enumerate() {
        if b {
            out[i / 64] |= 1 << (i % 64);
        }
    }
    out
}

pub fn run(rows: &[Observation], vocabulary: usize, policy: Policy) -> Result<Search> {
    crate::observations::validate(rows, vocabulary)?;
    if !(1..=3).contains(&policy.depth)
        || !(16..=128).contains(&policy.beam)
        || !(1..=64).contains(&policy.retained)
        || !(1..=10_000_000).contains(&policy.max_candidates)
    {
        return Err("prefix-search policy refused".into());
    }
    let work = policy.max_candidates as u128 * rows.len() as u128 * 128
        + policy.depth as u128
            * policy.beam as u128
            * rows
                .iter()
                .map(|r| (r.tokens.len() as u128 + 1).pow(2) * 4096)
                .sum::<u128>();
    let memory = 512 * 1024 * 1024
        + 64 * (policy.beam as u128 + policy.retained as u128) * (rows.len() as u128 * 32 + 16384);
    if work > 600_000_000_000_000 || memory > joint::MAX_BYTES {
        return Err("prefix-search forecast refused".into());
    }
    let mut out = Search {
        policy: policy.clone(),
        stages: vec![],
        programs: vec![],
        skipped: None,
        scalar_forecast: work as u64,
        buffer_forecast: memory as u64,
    };
    if rows
        .iter()
        .any(|r| crate::prefix::payload(&r.tokens).is_none())
    {
        out.skipped = Some("input prefixes do not use the supplied three-symbol alphabet".into());
        return Ok(out);
    }
    let labels: BTreeSet<_> = rows.iter().map(|r| r.label).collect();
    if labels.len() != 2 {
        out.programs = (0..6).map(|i| Program::Prefix(literal(i))).collect();
        out.skipped =
            Some("binary search requires two observed raw classes; retained literal bank".into());
        return Ok(out);
    }
    let target = bits(
        rows.iter().map(|r| r.label == *labels.last().unwrap()),
        rows.len(),
    );
    let gates: Vec<_> = (0..6)
        .map(|i| {
            bits(
                rows.iter().map(|r| {
                    let equal = *r.tokens.last().unwrap() == [9, 10, 23][i % 3];
                    if i < 3 { equal } else { !equal }
                }),
                rows.len(),
            )
        })
        .collect();
    let symbol_counts: Vec<Vec<i32>> = (0..3)
        .map(|i| {
            rows.iter()
                .map(|r| r.tokens.iter().filter(|&&x| x == [9, 10, 23][i]).count() as i32)
                .collect()
        })
        .collect();
    let mut children: Vec<_> = (0..6).map(literal).collect();
    let mut visited = 0usize;
    for depth in 1..=policy.depth {
        let counts: Vec<Vec<i32>> = children
            .iter()
            .map(|f| {
                rows.iter()
                    .map(|r| {
                        Ok(f.evaluate(
                            crate::prefix::payload(&r.tokens).ok_or("invalid counted word")?,
                        )?
                        .iter()
                        .filter(|&&x| x)
                        .count() as i32)
                    })
                    .collect::<Result<_>>()
            })
            .collect::<Result<_>>()?;
        let capacity = policy.beam - 6;
        let mut buckets = [
            Bucket {
                capacity: capacity / 2,
                kind: 0,
                rows: vec![],
            },
            Bucket {
                capacity: capacity / 4,
                kind: 1,
                rows: vec![],
            },
            Bucket {
                capacity: capacity - capacity / 2 - capacity / 4,
                kind: 2,
                rows: vec![],
            },
        ];
        let mut perfect = Bucket {
            capacity: policy.retained,
            kind: 3,
            rows: vec![],
        };
        let before = visited;
        let mut perfect_syntaxes = 0;
        for (ai, a) in children.iter().enumerate() {
            for (bi, b) in children.iter().enumerate() {
                for extra in [None, Some(0), Some(1), Some(2)] {
                    for offset in -2..=2 {
                        let comparison = bits(
                            (0..rows.len()).map(|i| {
                                counts[ai][i]
                                    < counts[bi][i]
                                        + extra.map_or(0, |s| symbol_counts[s][i])
                                        + offset
                            }),
                            rows.len(),
                        );
                        for negated in [false, true] {
                            for gate in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)]
                            {
                                visited += 1;
                                if visited > policy.max_candidates {
                                    return Err("prefix candidate work cap exhausted".into());
                                }
                                let nodes = 3
                                    + a.nodes()
                                    + b.nodes()
                                    + usize::from(extra.is_some()) * 3
                                    + usize::from(offset != 0) * 2
                                    + usize::from(negated)
                                    + gate.map_or(0, |g| if g < 3 { 2 } else { 3 });
                                if nodes > 127 {
                                    continue;
                                }
                                let mut signature: Vec<_> = comparison
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &x)| {
                                        (if negated { !x } else { x })
                                            & gate.map_or(u64::MAX, |g| gates[g][i])
                                    })
                                    .collect();
                                if !rows.len().is_multiple_of(64) {
                                    *signature.last_mut().unwrap() &=
                                        (1u64 << (rows.len() % 64)) - 1;
                                }
                                let errors: usize = signature
                                    .iter()
                                    .zip(&target)
                                    .map(|(a, b)| (a ^ b).count_ones() as usize)
                                    .sum();
                                let errors = errors.min(rows.len() - errors);
                                let digest = hash(&signature);
                                if errors == 0 {
                                    perfect_syntaxes += 1;
                                    perfect.add(&signature, errors, digest, nodes, || {
                                        make(a, b, extra, offset, negated, gate)
                                    });
                                }
                                for bucket in &mut buckets {
                                    bucket.add(&signature, errors, digest, nodes, || {
                                        make(a, b, extra, offset, negated, gate)
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        let old = children.len();
        let mut next: Vec<Formula> = (0..6).map(literal).collect();
        let mut signatures: Vec<Vec<u64>> = next
            .iter()
            .map(|f| {
                bits(
                    rows.iter().map(|r| {
                        *f.evaluate(crate::prefix::payload(&r.tokens).unwrap())
                            .unwrap()
                            .last()
                            .unwrap()
                    }),
                    rows.len(),
                )
            })
            .collect();
        for bucket in buckets {
            for c in bucket.rows {
                if !signatures.contains(&c.bits) {
                    signatures.push(c.bits);
                    next.push(c.formula);
                }
            }
        }
        out.stages.push(Stage {
            depth,
            children: old,
            visited: visited - before,
            perfect_syntaxes,
            retained_children: next.len(),
        });
        if !perfect.rows.is_empty() {
            out.programs = perfect
                .rows
                .into_iter()
                .map(|c| Program::Prefix(c.formula))
                .collect();
            return Ok(out);
        }
        children = next;
    }
    Ok(out)
}
