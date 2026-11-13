//! CM26 / Lean13711438: observed categorical groups propose fiber equations.
//! Group size and common mass are guesses, checked against each actual endpoint.
use crate::finite_system::{Policy, Problem, System};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    Row,
    Column,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    pub axis: Axis,
    pub fixed: usize,
    pub label: usize,
    pub members: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partition {
    pub groups: Vec<Group>,
    pub missing_rows: Vec<usize>,
    pub missing_columns: Vec<usize>,
    pub sizes: Vec<usize>,
}
pub fn collect(n: usize, observations: &[[usize; 2]]) -> Partition {
    let mut groups = vec![];
    let (mut rows, mut columns) = (vec![false; n], vec![false; n]);
    for &[cell, _] in observations {
        rows[cell % n] = true;
        columns[cell / n] = true;
    }
    for axis in [Axis::Column, Axis::Row] {
        for fixed in 0..n {
            let mut local: Vec<Group> = vec![];
            for &[cell, label] in observations {
                let (i, j) = (cell % n, cell / n);
                let (a, b) = match axis {
                    Axis::Column => (j, i),
                    Axis::Row => (i, j),
                };
                if a != fixed {
                    continue;
                }
                if let Some(g) = local.iter_mut().find(|g| g.label == label) {
                    g.members.push(b);
                } else {
                    local.push(Group {
                        axis,
                        fixed,
                        label,
                        members: vec![b],
                    });
                }
            }
            for g in &mut local {
                g.members.sort_unstable();
            }
            groups.extend(local);
        }
    }
    let mut sizes: Vec<_> = groups
        .iter()
        .map(|g| g.members.len())
        .filter(|k| (2..=8).contains(k))
        .collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes.dedup();
    Partition {
        groups,
        missing_rows: (0..n).filter(|&i| !rows[i]).collect(),
        missing_columns: (0..n).filter(|&i| !columns[i]).collect(),
        sizes,
    }
}
pub fn problem(n: usize, groups: &[Group], indices: &[usize], policy: Policy) -> Problem {
    let variables = 2 * n + 1;
    let mut rows = vec![];
    for &index in indices {
        let g = &groups[index];
        let mut row = vec![0; variables];
        let (offset, fixed) = match g.axis {
            Axis::Column => (0, n + g.fixed),
            Axis::Row => (n, g.fixed),
        };
        for &i in &g.members {
            row[offset + i] = 1;
        }
        row[fixed] = (g.members.len() % n) as u16;
        row[2 * n] = (n - 1) as u16;
        rows.push(row);
    }
    let mut right = vec![0; rows.len()];
    // Gauge choices concern unknown categorical phases, not token arithmetic.
    for (column, value) in [(0, 0), (n, 0), (1, 1)] {
        let mut row = vec![0; variables];
        row[column] = 1;
        rows.push(row);
        right.push(value);
    }
    Problem {
        system: System {
            prime: n as u16,
            variables,
            rows,
            right,
        },
        policy,
    }
}
