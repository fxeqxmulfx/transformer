//! CM10: propose fixed row/column groups using training labels alone.
//! These finite, data-dependent proposals are not a joint convex optimization.
//! Once fixed, their lifted probability-table family is convex by Quotient.lean.
use crate::{Result, Sample};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    pub axis: usize,
    pub first: usize,
    pub second: usize,
    pub overlap: usize,
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Groups {
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub row_count: usize,
    pub column_count: usize,
    pub evidence: Vec<Evidence>,
}

fn root(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}

fn compact(parent: &[usize]) -> (Vec<usize>, usize) {
    let roots: std::collections::BTreeSet<_> = (0..parent.len()).map(|i| root(parent, i)).collect();
    let ids: std::collections::BTreeMap<_, _> =
        roots.into_iter().enumerate().map(|(i, r)| (r, i)).collect();
    (
        (0..parent.len()).map(|i| ids[&root(parent, i)]).collect(),
        ids.len(),
    )
}

impl Groups {
    pub fn identity(n: usize) -> Self {
        Self {
            rows: (0..n).collect(),
            columns: (0..n).collect(),
            row_count: n,
            column_count: n,
            evidence: vec![],
        }
    }

    pub fn validate(&self, n: usize) -> Result<()> {
        for (map, count) in [
            (&self.rows, self.row_count),
            (&self.columns, self.column_count),
        ] {
            let actual: std::collections::BTreeSet<_> = map.iter().copied().collect();
            if map.len() != n || count == 0 || actual != (0..count).collect() {
                return Err("invalid compact input groups".into());
            }
        }
        Ok(())
    }

    pub fn cell(&self, raw: usize) -> usize {
        let n = self.rows.len();
        self.rows[raw % n] + n * self.columns[raw / n]
    }

    pub fn map_samples(&self, samples: &[Sample]) -> Result<Vec<Sample>> {
        let n = self.rows.len();
        self.validate(n)?;
        let mut labels = vec![None; n * n];
        let mut mapped = Vec::with_capacity(samples.len());
        for s in samples {
            if s.cell >= n * n || s.label >= n {
                return Err("invalid original observation".into());
            }
            let cell = self.cell(s.cell);
            if labels[cell].is_some_and(|label| label != s.label) {
                return Err("grouped observations have conflicting labels".into());
            }
            labels[cell] = Some(s.label);
            // Keep every original observation, including mapped duplicates.
            mapped.push(Sample {
                cell,
                label: s.label,
            });
        }
        Ok(mapped)
    }

    /// A pair must agree on at least `minimum_overlap` observed counterparts.
    /// Each proposed union is then checked against all training observations
    /// under both current maps. Unobserved agreement is never assumed proved.
    pub fn discover(n: usize, samples: &[Sample], minimum_overlap: usize) -> Result<Self> {
        if !(2..=256).contains(&n) || minimum_overlap < 2 || samples.is_empty() {
            return Err("invalid grouping dimensions or overlap threshold".into());
        }
        let mut table = vec![None; n * n];
        for s in samples {
            if s.cell >= n * n || s.label >= n || table[s.cell].is_some() {
                return Err("group discovery requires unique original cells".into());
            }
            table[s.cell] = Some(s.label);
        }
        let mut proposals = vec![];
        for axis in 0..2 {
            for first in 0..n {
                for second in first + 1..n {
                    let (mut overlap, mut agrees) = (0, true);
                    for other in 0..n {
                        let cell = |x| {
                            if axis == 0 {
                                x + n * other
                            } else {
                                other + n * x
                            }
                        };
                        if let (Some(a), Some(b)) = (table[cell(first)], table[cell(second)]) {
                            overlap += 1;
                            agrees &= a == b;
                        }
                    }
                    if agrees && overlap >= minimum_overlap {
                        proposals.push(Evidence {
                            axis,
                            first,
                            second,
                            overlap,
                            accepted: false,
                        });
                    }
                }
            }
        }
        proposals.sort_by_key(|p| (std::cmp::Reverse(p.overlap), p.axis, p.first, p.second));
        let mut parents = [(0..n).collect::<Vec<_>>(), (0..n).collect::<Vec<_>>()];
        for proposal in &mut proposals {
            let a = root(&parents[proposal.axis], proposal.first);
            let b = root(&parents[proposal.axis], proposal.second);
            if a == b {
                proposal.accepted = true;
                continue;
            }
            let mut trial = parents.clone();
            trial[proposal.axis][a.max(b)] = a.min(b);
            let mut seen = vec![None; n * n];
            let consistent = samples.iter().all(|s| {
                let cell = root(&trial[0], s.cell % n) + n * root(&trial[1], s.cell / n);
                let ok = seen[cell].is_none_or(|label| label == s.label);
                seen[cell] = Some(s.label);
                ok
            });
            if consistent {
                parents = trial;
                proposal.accepted = true;
            }
        }
        let (rows, row_count) = compact(&parents[0]);
        let (columns, column_count) = compact(&parents[1]);
        let groups = Self {
            rows,
            columns,
            row_count,
            column_count,
            evidence: proposals,
        };
        groups.map_samples(samples)?;
        Ok(groups)
    }
}
