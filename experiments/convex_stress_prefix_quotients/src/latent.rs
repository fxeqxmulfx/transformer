//! Exhaustive child-root constraints until an explicit signature/work cap.
//! A capped bank is reported as truncated. Retained programs can still be
//! checked against every original label; no search completeness is claimed.
use crate::{
    bank::Bank,
    pool::{Pool, bit},
    transitions::Partial,
    *,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub pairs: u64,
    pub row_comparisons: u64,
    pub matching_syntaxes: u64,
    pub signature_positions: u64,
    pub classes: usize,
    pub capped: bool,
    pub work_exhausted: bool,
    pub count_bound_rejections: u64,
}
pub struct Meter {
    pub work: u64,
    pub limit: u64,
}
impl Meter {
    pub fn charge(&mut self, n: u64) -> Result<()> {
        if n > self.limit.saturating_sub(self.work) {
            return Err("transition search work cap exhausted".into());
        }
        self.work += n;
        Ok(())
    }
}
pub fn synthesize(
    pool: &Pool,
    children: &Bank,
    wanted: &Partial,
    cap: usize,
    meter: &mut Meter,
) -> Result<(Bank, Stats)> {
    synthesize_bounded(pool, children, wanted, cap, meter, None)
}
pub fn synthesize_bounded(
    pool: &Pool,
    children: &Bank,
    wanted: &Partial,
    cap: usize,
    meter: &mut Meter,
    bounds: Option<&count_bounds::Ranges>,
) -> Result<(Bank, Stats)> {
    let mut out = Bank::new();
    let mut stats = Stats::default();
    if let Some(b) = bounds {
        if b.low.len() != pool.nodes.len() || b.high.len() != pool.nodes.len() {
            return Err("count-range shape differs".into());
        }
        if b.impossible() {
            return Ok((out, stats));
        }
    }
    macro_rules! charge {
        ($n:expr) => {
            if meter.charge($n).is_err() {
                stats.work_exhausted = true;
                stats.classes = out.rows.len();
                return Ok((out, stats));
            }
        };
    }
    for child in &children.rows {
        charge!(wanted.len() as u64 * 16 + pool.nodes.len() as u64 * 32 + 8192);
        if wanted.iter().all(|&(i, b)| bit(&child.bits, i) == b)
            && bounds.is_none_or(|b| {
                child
                    .counts
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| b.allows(i, v))
            })
            && let Err(error) = out.add(pool, &child.bits, child.nodes, cap, || {
                child.formula.clone()
            })
        {
            if error.to_string() != "observed signature class cap exhausted" {
                return Err(error);
            }
            stats.capped = true;
            stats.classes = out.rows.len();
            return Ok((out, stats));
        }
    }
    let literal_bits: Vec<_> = (0..6)
        .map(|g| pool.signature(&grammar::literal(g)))
        .collect::<Result<_>>()?;
    let counts: Vec<_> = literal_bits[..3].iter().map(|b| pool.counts(b)).collect();
    let cases: Vec<_> = [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)]
        .into_iter()
        .filter_map(|gate| {
            if wanted
                .iter()
                .any(|&(i, b)| b && gate.is_some_and(|g| !bit(&literal_bits[g], i)))
            {
                return None;
            }
            Some((
                gate,
                wanted
                    .iter()
                    .filter(|&&(i, _)| gate.is_none_or(|g| bit(&literal_bits[g], i)))
                    .copied()
                    .collect::<Vec<_>>(),
            ))
        })
        .collect();
    let mut signature = vec![0u64; pool.nodes.len().div_ceil(64)];
    let mut signature_counts = vec![0i16; pool.nodes.len()];
    for a in &children.rows {
        for b in &children.rows {
            charge!(8192);
            stats.pairs += 1;
            for extra in [None, Some(0), Some(1), Some(2)] {
                for (gate, active) in &cases {
                    for negated in [false, true] {
                        // Reserve the conservative complete scan before executing it.
                        charge!(active.len() as u64 * 32 + 256);
                        let (mut lower, mut upper) = (-2, 2);
                        for &(i, target) in active {
                            stats.row_comparisons += 1;
                            let delta = i32::from(a.counts[i])
                                - i32::from(b.counts[i])
                                - extra.map_or(0, |s| i32::from(counts[s][i]));
                            if target ^ negated {
                                lower = lower.max(delta + 1);
                            } else {
                                upper = upper.min(delta);
                            }
                            if lower > upper {
                                break;
                            }
                        }
                        for offset in lower..=upper {
                            let size =
                                grammar::size(a.nodes, b.nodes, extra, offset, negated, *gate);
                            if size > 127 {
                                continue;
                            }
                            charge!(pool.nodes.len() as u64 * 64 + 32768);
                            stats.matching_syntaxes += 1;
                            signature.fill(0);
                            let mut allowed = true;
                            for i in 0..pool.nodes.len() {
                                stats.signature_positions += 1;
                                let v = i32::from(a.counts[i])
                                    < i32::from(b.counts[i])
                                        + extra.map_or(0, |s| i32::from(counts[s][i]))
                                        + offset;
                                let value =
                                    (v ^ negated) && gate.is_none_or(|g| bit(&literal_bits[g], i));
                                if value {
                                    signature[i / 64] |= 1 << (i % 64);
                                }
                                signature_counts[i] =
                                    pool.nodes[i].parent.map_or(0, |p| signature_counts[p])
                                        + i16::from(value);
                                if bounds.is_some_and(|b| !b.allows(i, signature_counts[i])) {
                                    allowed = false;
                                    stats.count_bound_rejections += 1;
                                    break;
                                }
                            }
                            if !allowed {
                                continue;
                            }
                            if let Err(error) = out.add(pool, &signature, size, cap, || {
                                grammar::make(&a.formula, &b.formula, extra, offset, negated, *gate)
                            }) {
                                if error.to_string() != "observed signature class cap exhausted" {
                                    return Err(error);
                                }
                                stats.capped = true;
                                stats.classes = out.rows.len();
                                return Ok((out, stats));
                            }
                        }
                    }
                }
            }
        }
    }
    stats.classes = out.rows.len();
    Ok((out, stats))
}
