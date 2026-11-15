//! Training constraints prune root comparisons before materializing their ASTs.
//! Every retained result is checked independently against all observed labels.
use crate::{bank::Bank, pool::Pool, *};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone)]
struct Constraints {
    gate: Option<usize>,
    rows: Vec<(usize, bool)>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub pairs: u64,
    pub interval_tests: u64,
    pub row_comparisons: u64,
    pub fitting_syntaxes: u64,
    pub retained: usize,
}
fn constraints(pool: &Pool, rows: &[Observation]) -> Result<Vec<Constraints>> {
    let high = rows
        .iter()
        .map(|r| r.label)
        .max()
        .ok_or("missing raw labels")?;
    let mut labels = BTreeMap::new();
    for (&i, r) in pool.row_nodes.iter().zip(rows) {
        if let Some(old) = labels.insert(i, r.label == high)
            && old != (r.label == high)
        {
            return Err("identical input has conflicting observed labels".into());
        }
    }
    let mut out = vec![];
    for gate in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)] {
        for flipped in [false, true] {
            let mut active = vec![];
            let mut possible = true;
            for (&i, &target) in &labels {
                let passes = gate.is_none_or(|g| {
                    let equal = pool.nodes[i].symbol == [9, 10, 23][g % 3];
                    if g < 3 { equal } else { !equal }
                });
                let wanted = target ^ flipped;
                if passes {
                    active.push((i, wanted));
                } else if wanted {
                    possible = false;
                    break;
                }
            }
            if possible && !active.is_empty() {
                out.push(Constraints { gate, rows: active });
            }
        }
    }
    Ok(out)
}
pub fn run(
    pool: &Pool,
    rows: &[Observation],
    bank: &Bank,
    retained: usize,
) -> Result<(Vec<Formula>, Stats)> {
    let cases = constraints(pool, rows)?;
    let counts: Vec<_> = (0..3)
        .map(|s| {
            let bits = pool.signature(&grammar::literal(s))?;
            Ok(pool.counts(&bits))
        })
        .collect::<Result<_>>()?;
    let mut stats = Stats::default();
    let mut perfect: Vec<Formula> = vec![];
    for a in &bank.rows {
        for b in &bank.rows {
            stats.pairs += 1;
            for extra in [None, Some(0), Some(1), Some(2)] {
                for c in &cases {
                    for negated in [false, true] {
                        stats.interval_tests += 1;
                        let (mut lower, mut upper) = (-2, 2);
                        for &(i, wanted) in &c.rows {
                            stats.row_comparisons += 1;
                            let delta = i32::from(a.counts[i])
                                - i32::from(b.counts[i])
                                - extra.map_or(0, |s| i32::from(counts[s][i]));
                            if wanted ^ negated {
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
                                grammar::size(a.nodes, b.nodes, extra, offset, negated, c.gate);
                            if size > 127 {
                                continue;
                            }
                            stats.fitting_syntaxes += 1;
                            let f = grammar::make(
                                &a.formula, &b.formula, extra, offset, negated, c.gate,
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
                            let index = perfect.partition_point(|x| (x.nodes(), x) < key);
                            perfect.insert(index, f);
                            if perfect.len() > retained {
                                perfect.pop();
                            }
                        }
                    }
                }
            }
        }
    }
    for f in &perfect {
        let program = Program::Prefix(f.clone());
        let mut decoder = BTreeMap::new();
        for r in rows {
            let atom = program.run(&r.tokens).ok_or("retained root undefined")?;
            let independent = program
                .interpret(&r.tokens)
                .ok_or("independent root undefined")?;
            if atom != independent {
                return Err("independent root interpreter mismatch".into());
            }
            if let Some(old) = decoder.insert(atom, r.label)
                && old != r.label
            {
                return Err("retained root fails raw-label consistency".into());
            }
        }
    }
    stats.retained = perfect.len();
    Ok((perfect, stats))
}
