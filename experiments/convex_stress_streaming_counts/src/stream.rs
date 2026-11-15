//! Stream the same child grammar as prefix_quotients/latent.rs at b388f4ff.
//! The visitor tests each admitted syntax; no distinct-signature cap stops it.
use crate::{
    bank::Bank,
    latent::Meter,
    pool::{Pool, bit},
    transitions::Partial,
    *,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub child_pairs: u64,
    pub row_comparisons: u64,
    pub matching_syntaxes: u64,
    pub signature_positions: u64,
    pub count_bound_rejections: u64,
    pub visited_queries: u64,
    pub complete: bool,
    pub found: bool,
}
pub struct Query<'a> {
    pub bits: &'a [u64],
    pub counts: &'a [i16],
    pub nodes: usize,
    pub make: &'a dyn Fn() -> Formula,
}
pub fn enumerate(
    pool: &Pool,
    children: &Bank,
    wanted: &Partial,
    bounds: &count_bounds::Ranges,
    meter: &mut Meter,
    stats: &mut Stats,
    mut visit: impl FnMut(Query<'_>, &mut Meter) -> Result<bool>,
) -> Result<()> {
    if bounds.low.len() != pool.nodes.len() || bounds.high.len() != pool.nodes.len() {
        return Err("stream count-range shape differs".into());
    }
    if bounds.impossible() {
        stats.complete = true;
        return Ok(());
    }
    for child in &children.rows {
        meter.charge(wanted.len() as u64 * 16 + pool.nodes.len() as u64 * 32 + 8192)?;
        if wanted.iter().all(|&(i, b)| bit(&child.bits, i) == b)
            && child
                .counts
                .iter()
                .enumerate()
                .all(|(i, &v)| bounds.allows(i, v))
        {
            stats.visited_queries += 1;
            if visit(
                Query {
                    bits: &child.bits,
                    counts: &child.counts,
                    nodes: child.nodes,
                    make: &|| child.formula.clone(),
                },
                meter,
            )? {
                stats.found = true;
                return Ok(());
            }
        }
    }
    meter.charge(pool.nodes.len() as u64 * 4096 * 128)?;
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
            meter.charge(8192)?;
            stats.child_pairs += 1;
            for extra in [None, Some(0), Some(1), Some(2)] {
                for (gate, active) in &cases {
                    for negated in [false, true] {
                        meter.charge(active.len() as u64 * 32 + 256)?;
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
                            meter.charge(pool.nodes.len() as u64 * 64 + 32768)?;
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
                                if !bounds.allows(i, signature_counts[i]) {
                                    stats.count_bound_rejections += 1;
                                    allowed = false;
                                    break;
                                }
                            }
                            if !allowed {
                                continue;
                            }
                            stats.visited_queries += 1;
                            if visit(
                                Query {
                                    bits: &signature,
                                    counts: &signature_counts,
                                    nodes: size,
                                    make: &|| {
                                        grammar::make(
                                            &a.formula, &b.formula, extra, offset, negated, *gate,
                                        )
                                    },
                                },
                                meter,
                            )? {
                                stats.found = true;
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
    }
    stats.complete = true;
    Ok(())
}
