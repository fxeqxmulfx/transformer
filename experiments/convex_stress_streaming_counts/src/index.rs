//! CM62 exact integer-threshold joins in either orientation, ported from
//! prefix_quotients/pairs.rs at b388f4ff. All known constraints are intersected.
use crate::{bank::Bank, latent::Meter, pool::Pool, transitions::Config, *};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub queries: u64,
    pub threshold_lookups: u64,
    pub word_intersections: u64,
    pub matching_pairs: u64,
    pub size_rejections: u64,
    pub index_buffer_bytes: u64,
}
struct Row {
    node: usize,
    depth: usize,
    use_above: bool,
    extra: i32,
    above: Vec<u64>,
    cards: Vec<usize>,
}
pub struct Index<'a> {
    anchor: &'a Bank,
    side: Side,
    config: Config,
    rows: Vec<Row>,
    impossible: bool,
    thresholds: Vec<usize>,
    strengths: Vec<usize>,
    order: Vec<usize>,
    candidates: Vec<u64>,
    active: Vec<usize>,
    pub stats: Stats,
}
impl<'a> Index<'a> {
    pub fn new(
        pool: &Pool,
        observed: &BTreeMap<usize, bool>,
        anchor: &'a Bank,
        side: Side,
        config: &Config,
        extras: &[Vec<i16>],
        meter: &mut Meter,
    ) -> Result<Self> {
        let bytes = previous::pairs::buffer_bound(pool, anchor.rows.len());
        if bytes > prior::joint::MAX_BYTES {
            return Err("stream index buffer forecast refused".into());
        }
        let words = anchor.rows.len().div_ceil(64);
        let mut rows = vec![];
        let mut impossible = anchor.rows.is_empty();
        for (&node, &target) in observed {
            if !config.passes(pool.nodes[node].symbol) {
                impossible |= target ^ config.flipped;
                continue;
            }
            let depth = pool.nodes[node].depth;
            meter.charge(
                anchor.rows.len() as u64 * 64 + (depth as u64 + 2) * words as u64 * 32 + 4096,
            )?;
            let mut above = vec![0u64; (depth + 2) * words];
            let mut cards = vec![0usize; depth + 2];
            for (a, candidate) in anchor.rows.iter().enumerate() {
                let count = usize::try_from(candidate.counts[node])?;
                if count > depth {
                    return Err("anchor count exceeds prefix length".into());
                }
                above[count * words + a / 64] |= 1 << (a % 64);
                cards[count] += 1;
            }
            for threshold in (0..depth).rev() {
                for word in 0..words {
                    above[threshold * words + word] |= above[(threshold + 1) * words + word];
                }
                cards[threshold] += cards[threshold + 1];
            }
            let positive = target ^ config.flipped ^ config.negated;
            rows.push(Row {
                node,
                depth,
                use_above: if side == Side::Right {
                    positive
                } else {
                    !positive
                },
                extra: config.extra.map_or(0, |s| i32::from(extras[s][node])) + config.offset,
                above,
                cards,
            });
        }
        let n = rows.len();
        Ok(Self {
            anchor,
            side,
            config: config.clone(),
            rows,
            impossible,
            thresholds: vec![0; n],
            strengths: vec![0; n],
            order: (0..n).collect(),
            candidates: vec![0; words],
            active: Vec::with_capacity(words),
            stats: Stats {
                index_buffer_bytes: u64::try_from(bytes)?,
                ..Default::default()
            },
        })
    }

    /// Return every fitting anchor ID under the actual combined AST-size cap.
    /// An error means the query was not completed and must not enter the cache.
    pub fn query(&mut self, counts: &[i16], nodes: usize, meter: &mut Meter) -> Result<Vec<usize>> {
        let words = self.candidates.len();
        meter.charge(
            self.rows.len() as u64 * (u64::from(self.rows.len().max(1).ilog2()) + 2) * 64
                + words as u64 * 32
                + 1024,
        )?;
        self.stats.queries += 1;
        if self.impossible {
            return Ok(vec![]);
        }
        let (mut best, mut smallest) = (0, usize::MAX);
        for (i, row) in self.rows.iter().enumerate() {
            self.stats.threshold_lookups += 1;
            let count = *counts.get(row.node).ok_or("streamed count shape differs")?;
            if count < 0 || count as usize > row.depth {
                return Err("streamed count exceeds prefix length".into());
            }
            let t = if self.side == Side::Right {
                i32::from(count) - row.extra + 1
            } else {
                i32::from(count) + row.extra
            };
            self.thresholds[i] = t.clamp(0, row.depth as i32 + 1) as usize;
            let count = row.cards[self.thresholds[i]];
            let count = if row.use_above {
                count
            } else {
                self.anchor.rows.len() - count
            };
            self.strengths[i] = count;
            if count < smallest {
                best = i;
                smallest = count;
            }
            if smallest == 0 {
                return Ok(vec![]);
            }
        }
        self.candidates.fill(u64::MAX);
        if !self.rows.is_empty() {
            self.order.sort_unstable_by_key(|&i| (self.strengths[i], i));
            let row = &self.rows[best];
            let start = self.thresholds[best] * words;
            for (j, candidate) in self.candidates.iter_mut().enumerate() {
                let bits = row.above[start + j];
                *candidate = if row.use_above { bits } else { !bits };
            }
        }
        pool::mask(&mut self.candidates, self.anchor.rows.len());
        self.active.clear();
        self.active.extend(
            self.candidates
                .iter()
                .enumerate()
                .filter(|(_, b)| **b != 0)
                .map(|(j, _)| j),
        );
        for &i in &self.order {
            if i == best {
                continue;
            }
            let row = &self.rows[i];
            meter.charge(self.active.len() as u64 * 32 + 128)?;
            let start = self.thresholds[i] * words;
            self.active.retain(|&j| {
                self.stats.word_intersections += 1;
                let bits = row.above[start + j];
                self.candidates[j] &= if row.use_above { bits } else { !bits };
                self.candidates[j] != 0
            });
            if self.active.is_empty() {
                return Ok(vec![]);
            }
        }
        let mut matched = vec![];
        for &word in &self.active {
            let mut bits = self.candidates[word];
            while bits != 0 {
                meter.charge(32768)?;
                let i = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if grammar::size(
                    nodes,
                    self.anchor.rows[i].nodes,
                    self.config.extra,
                    self.config.offset,
                    self.config.negated,
                    self.config.gate,
                ) > 127
                {
                    self.stats.size_rejections += 1;
                    continue;
                }
                self.stats.matching_pairs += 1;
                matched.push(i);
            }
        }
        Ok(matched)
    }
    pub fn assemble(&self, query: &Formula, ids: &[usize], retained: usize) -> Vec<Formula> {
        let mut out: Vec<_> = ids
            .iter()
            .map(|&i| {
                let a = &self.anchor.rows[i].formula;
                let (left, right) = if self.side == Side::Left {
                    (a, query)
                } else {
                    (query, a)
                };
                grammar::make(
                    left,
                    right,
                    self.config.extra,
                    self.config.offset,
                    self.config.negated,
                    self.config.gate,
                )
            })
            .collect();
        out.sort_unstable_by(|a, b| (a.nodes(), a).cmp(&(b.nodes(), b)));
        out.dedup();
        out.truncate(retained);
        out
    }
}
