//! CM23 / Lean 0bb5496f: integer dual certificates for an axis assignment.
//! Shortest augmenting paths ported from convex_stress_decode/assignment.rs at
//! d88f7f37; adds retained potentials and an independent inequality checker.
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Certificate {
    pub permutation: Vec<usize>,
    pub row_dual: Vec<i64>,
    pub column_dual: Vec<i64>,
    pub objective: usize,
    pub relaxations: usize,
}
pub fn permutation(values: &[usize], n: usize) -> bool {
    values.len() == n && (0..n).all(|i| values.iter().filter(|&&j| j == i).count() == 1)
}
fn validate(cost: &[usize], n: usize) -> Result<()> {
    if !(2..=193).contains(&n) || cost.len() != n * n || cost.iter().any(|&c| c > n * n) {
        return Err("invalid bounded assignment matrix".into());
    }
    Ok(())
}
pub fn solve(cost: &[usize], n: usize) -> Result<Certificate> {
    validate(cost, n)?;
    let (mut u, mut v, mut p, mut way) = (
        vec![0i64; n + 1],
        vec![0i64; n + 1],
        vec![0usize; n + 1],
        vec![0usize; n + 1],
    );
    let mut relaxations = 0;
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut min = vec![i64::MAX / 4; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = i64::MAX / 4;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    relaxations += 1;
                    let c = cost[(i0 - 1) * n + j - 1] as i64 - u[i0] - v[j];
                    if c < min[j] {
                        min[j] = c;
                        way[j] = j0;
                    }
                    if min[j] < delta {
                        delta = min[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut perm = vec![0; n];
    for j in 1..=n {
        perm[p[j] - 1] = j - 1;
    }
    let out = Certificate {
        objective: perm.iter().enumerate().map(|(i, &j)| cost[i * n + j]).sum(),
        permutation: perm,
        row_dual: u[1..].to_vec(),
        column_dual: v[1..].to_vec(),
        relaxations,
    };
    check(cost, n, &out)?;
    Ok(out)
}

/// Independently verifies every dual inequality and tight edge using i128,
/// including malformed extreme saved i64 potentials without overflow.
pub fn check(cost: &[usize], n: usize, c: &Certificate) -> Result<()> {
    validate(cost, n)?;
    if !permutation(&c.permutation, n)
        || c.row_dual.len() != n
        || c.column_dual.len() != n
        || c.relaxations > n * n * n
    {
        return Err("invalid assignment certificate shape or work".into());
    }
    let mut objective = 0;
    for i in 0..n {
        for j in 0..n {
            let dual = i128::from(c.row_dual[i]) + i128::from(c.column_dual[j]);
            if dual > cost[i * n + j] as i128
                || (j == c.permutation[i] && dual != cost[i * n + j] as i128)
            {
                return Err("assignment dual inequality or tight edge failed".into());
            }
        }
        objective += cost[i * n + c.permutation[i]];
    }
    let dual: i128 = c
        .row_dual
        .iter()
        .chain(&c.column_dual)
        .map(|&x| i128::from(x))
        .sum();
    if objective != c.objective || dual != objective as i128 {
        return Err("assignment primal and dual objectives disagree".into());
    }
    Ok(())
}
