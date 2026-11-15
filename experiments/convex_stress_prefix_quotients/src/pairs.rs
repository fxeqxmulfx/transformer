//! Exact right-count index from CM59 comparison_right_lower/upper. Every
//! observed constraint is intersected; tail bits never name a candidate.
use crate::{
    bank::Bank,
    latent::Meter,
    pool::{Pool, mask},
    transitions::Config,
    *,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub possible_pairs: u64,
    pub left_queries: u64,
    pub threshold_lookups: u64,
    pub word_intersections: u64,
    pub matching_pairs: u64,
    pub index_buffer_bytes: u64,
    pub work_exhausted: bool,
}
struct Row {
    node: usize,
    depth: usize,
    positive: bool,
    extra: i32,
    above: Vec<u64>,
    cards: Vec<usize>,
}
pub fn buffer_bound(pool: &Pool, cap: usize) -> u128 {
    let words = cap.div_ceil(64) as u128;
    pool.nodes
        .iter()
        .map(|n| (n.depth as u128 + 2) * (words * 8 + 16) + 4096)
        .sum::<u128>()
        + words * 32
        + 1024 * 1024
}
pub fn run(
    pool: &Pool,
    observed: &BTreeMap<usize, bool>,
    banks: (&Bank, &Bank),
    config: &Config,
    extras: &[Vec<i16>],
    retained: usize,
    meter: &mut Meter,
) -> Result<(Vec<Formula>, Stats)> {
    let (left, right) = banks;
    let words = right.rows.len().div_ceil(64);
    let mut stats = Stats {
        possible_pairs: left.rows.len() as u64 * right.rows.len() as u64,
        index_buffer_bytes: u64::try_from(buffer_bound(pool, right.rows.len()))?,
        ..Default::default()
    };
    if u128::from(stats.index_buffer_bytes) > prior::joint::MAX_BYTES {
        return Err("count index buffer forecast refused".into());
    }
    macro_rules! charge {
        ($n:expr) => {
            if meter.charge($n).is_err() {
                stats.work_exhausted = true;
                return Ok((vec![], stats));
            }
        };
    }
    let mut index = vec![];
    for (&node, &target) in observed {
        if !config.passes(pool.nodes[node].symbol) {
            if target ^ config.flipped {
                return Ok((vec![], stats));
            }
            continue;
        }
        let depth = pool.nodes[node].depth;
        charge!(right.rows.len() as u64 * 64 + (depth as u64 + 2) * words as u64 * 32 + 4096);
        let mut above = vec![0u64; (depth + 2) * words];
        let mut cards = vec![0usize; depth + 2];
        for (b, candidate) in right.rows.iter().enumerate() {
            let count = usize::try_from(candidate.counts[node])?;
            if count > depth {
                return Err("cached count exceeds prefix length".into());
            }
            above[count * words + b / 64] |= 1 << (b % 64);
            cards[count] += 1;
        }
        for threshold in (0..depth).rev() {
            for word in 0..words {
                above[threshold * words + word] |= above[(threshold + 1) * words + word];
            }
            cards[threshold] += cards[threshold + 1];
        }
        index.push(Row {
            node,
            depth,
            positive: target ^ config.flipped ^ config.negated,
            extra: config.extra.map_or(0, |s| i32::from(extras[s][node])),
            above,
            cards,
        });
    }
    if index.is_empty() || right.rows.is_empty() {
        return Ok((vec![], stats));
    }
    let mut thresholds = vec![0usize; index.len()];
    let mut candidates = vec![0u64; words];
    let mut active = Vec::with_capacity(words);
    let mut perfect: Vec<Formula> = vec![];
    for a in &left.rows {
        charge!(index.len() as u64 * 64 + words as u64 * 32 + 1024);
        stats.left_queries += 1;
        let (mut best, mut smallest) = (0, usize::MAX);
        for (i, row) in index.iter().enumerate() {
            stats.threshold_lookups += 1;
            let t = i32::from(a.counts[row.node]) - row.extra - config.offset + 1;
            thresholds[i] = t.clamp(0, row.depth as i32 + 1) as usize;
            let count = row.cards[thresholds[i]];
            let count = if row.positive {
                count
            } else {
                right.rows.len() - count
            };
            if count < smallest {
                best = i;
                smallest = count;
            }
            if smallest == 0 {
                break;
            }
        }
        if smallest == 0 {
            continue;
        }
        let row = &index[best];
        let start = thresholds[best] * words;
        for (j, candidate) in candidates.iter_mut().enumerate() {
            let bits = row.above[start + j];
            *candidate = if row.positive { bits } else { !bits };
        }
        mask(&mut candidates, right.rows.len());
        active.clear();
        active.extend(
            candidates
                .iter()
                .enumerate()
                .filter(|(_, b)| **b != 0)
                .map(|(j, _)| j),
        );
        for (i, row) in index.iter().enumerate() {
            if i == best {
                continue;
            }
            charge!(active.len() as u64 * 32 + 128);
            let start = thresholds[i] * words;
            active.retain(|&j| {
                stats.word_intersections += 1;
                let bits = row.above[start + j];
                candidates[j] &= if row.positive { bits } else { !bits };
                candidates[j] != 0
            });
            if active.is_empty() {
                break;
            }
        }
        for &word in &active {
            let mut bits = candidates[word];
            while bits != 0 {
                charge!(32768);
                let index = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let b = &right.rows[index];
                let size = grammar::size(
                    a.nodes,
                    b.nodes,
                    config.extra,
                    config.offset,
                    config.negated,
                    config.gate,
                );
                if size > 127 {
                    continue;
                }
                stats.matching_pairs += 1;
                let f = grammar::make(
                    &a.formula,
                    &b.formula,
                    config.extra,
                    config.offset,
                    config.negated,
                    config.gate,
                );
                let key = (size, &f);
                if perfect.len() == retained
                    && key >= (perfect.last().unwrap().nodes(), perfect.last().unwrap())
                {
                    continue;
                }
                if perfect.contains(&f) {
                    continue;
                }
                let place = perfect.partition_point(|p| (p.nodes(), p) < key);
                perfect.insert(place, f);
                if perfect.len() > retained {
                    perfect.pop();
                }
            }
        }
    }
    Ok((perfect, stats))
}
