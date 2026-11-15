//! CM59 necessary child bits, conditional on each proposed root. These are
//! search constraints only; the CE dataset remains the original observations.
use crate::{pool::Pool, *};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
pub type Partial = Vec<(usize, bool)>;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub extra: Option<usize>,
    pub offset: i32,
    pub negated: bool,
    pub gate: Option<usize>,
    pub flipped: bool,
}
#[derive(Clone, Debug)]
pub struct Branch {
    pub config: Config,
    pub left: Partial,
    pub right: Partial,
}
impl Config {
    pub fn comparison(&self, a: i16, b: i16, symbol_count: i16) -> bool {
        let raw = i32::from(a) < i32::from(b) + i32::from(symbol_count) + self.offset;
        raw ^ self.negated
    }
    pub fn passes(&self, symbol: usize) -> bool {
        self.gate.is_none_or(|g| {
            let equal = symbol == [9, 10, 23][g % 3];
            if g < 3 { equal } else { !equal }
        })
    }
}
pub fn branches(pool: &Pool, rows: &[Observation]) -> Result<Vec<Branch>> {
    let high = rows.iter().map(|r| r.label).max().ok_or("missing labels")?;
    let mut observed = vec![None; pool.nodes.len()];
    for (&i, r) in pool.row_nodes.iter().zip(rows) {
        let b = r.label == high;
        if observed[i].is_some_and(|old| old != b) {
            return Err("conflicting observed root labels".into());
        }
        observed[i] = Some(b);
    }
    let mut out = vec![];
    let mut seen = BTreeSet::new();
    for extra in [None, Some(0), Some(1), Some(2)] {
        for offset in [0, -1, 1, -2, 2] {
            for gate in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)] {
                for negated in [false, true] {
                    for flipped in [false, true] {
                        let config = Config {
                            extra,
                            offset,
                            negated,
                            gate,
                            flipped,
                        };
                        let mut known = vec![None; pool.nodes.len()];
                        let mut possible = true;
                        for (i, &label) in observed.iter().enumerate() {
                            if let Some(label) = label {
                                let wanted = label ^ flipped;
                                if config.passes(pool.nodes[i].symbol) {
                                    known[i] = Some(wanted ^ negated);
                                } else if wanted {
                                    possible = false;
                                    break;
                                }
                            }
                        }
                        if !possible || !seen.insert((extra, offset, known.clone())) {
                            continue;
                        }
                        let (mut left, mut right) = (BTreeMap::new(), BTreeMap::new());
                        for (i, node) in pool.nodes.iter().enumerate() {
                            let Some(after) = known[i] else {
                                continue;
                            };
                            let e = extra.is_some_and(|s| node.symbol == [9, 10, 23][s]);
                            if let Some(parent) = node.parent {
                                let Some(before) = known[parent] else {
                                    continue;
                                };
                                if before && !after {
                                    if e {
                                        possible = false;
                                        break;
                                    }
                                    left.insert(i, true);
                                    right.insert(i, false);
                                } else if !before && after && !e {
                                    left.insert(i, false);
                                    right.insert(i, true);
                                }
                            } else {
                                // Empty-prefix difference is exactly zero. Enumerate four
                                // possible child bits rather than invent an observed label.
                                let allowed: Vec<_> = [false, true]
                                    .into_iter()
                                    .flat_map(|a| [false, true].map(move |b| (a, b)))
                                    .filter(|&(a, b)| {
                                        ((i32::from(a) - i32::from(b) - i32::from(e)) < offset)
                                            == after
                                    })
                                    .collect();
                                if allowed.is_empty() {
                                    possible = false;
                                    break;
                                }
                                if allowed.iter().all(|x| x.0 == allowed[0].0) {
                                    left.insert(i, allowed[0].0);
                                }
                                if allowed.iter().all(|x| x.1 == allowed[0].1) {
                                    right.insert(i, allowed[0].1);
                                }
                            }
                        }
                        if possible {
                            out.push(Branch {
                                config,
                                left: left.into_iter().collect(),
                                right: right.into_iter().collect(),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}
