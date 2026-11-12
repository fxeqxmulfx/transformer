//! Independent dense-membership replay, without calling the sorted-domain sweep.
//! Verifies the exact complete branch list, deletions, output union and costs.
use crate::*;

#[derive(Clone, Debug)]
pub struct Verified {
    certificate: Certificate,
}

impl Verified {
    pub fn answer(&self, row: usize, column: usize) -> Option<usize> {
        let side = self.certificate.problem.side;
        if row >= side || column >= side {
            return None;
        }
        match self.certificate.answers[row * side + column].as_slice() {
            [label] => Some(*label),
            _ => None,
        }
    }

    /// This refutes only the declared bank, not all conceivable target models.
    pub fn inconsistent(&self) -> bool {
        self.certificate.answers.iter().all(Vec::is_empty)
    }

    pub fn certificate(&self) -> &Certificate {
        &self.certificate
    }
}

fn tick(cost: &mut Costs, output: bool, c: Config) -> Result<()> {
    cost.tick(output, c.max_checks)
        .ok_or_else(|| "certificate exceeds work budget".into())
}

fn advance(
    p: &Problem,
    d: &[usize],
    rows: &[bool],
    cols: &[bool],
    c: Config,
    cost: &mut Costs,
) -> Result<(Vec<bool>, Vec<bool>)> {
    let q = p.states;
    let (mut rr, mut cc) = (rows.to_vec(), cols.to_vec());
    for i in 0..p.side {
        let incident: Vec<_> = p.observations.iter().filter(|t| t.row == i).collect();
        for a in 0..q {
            if !rows[i * q + a] {
                continue;
            }
            for t in &incident {
                let mut support = false;
                for b in 0..q {
                    if !cols[t.column * q + b] {
                        continue;
                    }
                    tick(cost, false, c)?;
                    if d[(a + b) % q] == t.label {
                        support = true;
                        break;
                    }
                }
                if !support {
                    rr[i * q + a] = false;
                    break;
                }
            }
        }
    }
    // Both sides read the old arrays, including when a row was just removed.
    for j in 0..p.side {
        let incident: Vec<_> = p.observations.iter().filter(|t| t.column == j).collect();
        for b in 0..q {
            if !cols[j * q + b] {
                continue;
            }
            for t in &incident {
                let mut support = false;
                for a in 0..q {
                    if !rows[t.row * q + a] {
                        continue;
                    }
                    tick(cost, false, c)?;
                    if d[(a + b) % q] == t.label {
                        support = true;
                        break;
                    }
                }
                if !support {
                    cc[j * q + b] = false;
                    break;
                }
            }
        }
    }
    Ok((rr, cc))
}

/// Check against separately supplied inputs and config. Never trust serialized
/// coverage, phase domains, branch omissions, operation counts or answer sets.
pub fn verify(problem: &Problem, c: Config, saved: &Certificate) -> Result<Verified> {
    let plan = preflight(problem, c)?;
    if plan.memory_bound > c.memory_limit || plan.total_branches > c.max_branches {
        return Err("certificate requested outside memory or branch budget".into());
    }
    if saved.version != 1
        || saved.config != c
        || saved.memory_bound != plan.memory_bound
        || saved.branches.len() != plan.total_branches
        || saved.answers.len() != problem.side * problem.side
    {
        return Err("certificate header, branch count or output dimensions changed".into());
    }
    let mut p = problem.clone();
    p.observations.sort_unstable();
    if saved.problem != p {
        return Err("certificate input binding changed".into());
    }
    let mut costs = Costs::default();
    let (n, q, v) = (p.side, p.states, p.vocab);
    let mut union = vec![false; n * n * v];
    let mut index = 0;
    for (k, d) in p.decoders.iter().enumerate() {
        for (z, &label) in d.iter().enumerate() {
            if label != c.anchor.label {
                continue;
            }
            let branch = &saved.branches[index];
            if branch.decoder != k
                || branch.anchor_state != z
                || branch.removed_per_sweep.len() != c.steps
                || branch.domains.rows.len() != n
                || branch.domains.columns.len() != n
            {
                return Err("branch identity or dimensions changed".into());
            }
            let (mut rows, mut cols) = (vec![true; n * q], vec![true; n * q]);
            rows[c.anchor.row * q..(c.anchor.row + 1) * q].fill(false);
            cols[c.anchor.column * q..(c.anchor.column + 1) * q].fill(false);
            rows[c.anchor.row * q] = true;
            cols[c.anchor.column * q + z] = true;
            for &removed in &branch.removed_per_sweep {
                let (rr, cc) = advance(&p, d, &rows, &cols, c, &mut costs)?;
                let actual = rows
                    .iter()
                    .zip(&rr)
                    .chain(cols.iter().zip(&cc))
                    .filter(|(old, next)| **old && !**next)
                    .count();
                if actual != removed {
                    return Err("sweep deletion count changed".into());
                }
                (rows, cols) = (rr, cc);
            }
            let mut live = true;
            for i in 0..n {
                let r: Vec<_> = (0..q).filter(|&a| rows[i * q + a]).collect();
                let s: Vec<_> = (0..q).filter(|&b| cols[i * q + b]).collect();
                if r != branch.domains.rows[i] || s != branch.domains.columns[i] {
                    return Err("retained domain changed".into());
                }
                live &= !r.is_empty() && !s.is_empty();
            }
            if live != branch.live {
                return Err("empty-domain rejection changed".into());
            }
            if live {
                for i in 0..n {
                    for j in 0..n {
                        for a in 0..q {
                            if !rows[i * q + a] {
                                continue;
                            }
                            for b in 0..q {
                                if !cols[j * q + b] {
                                    continue;
                                }
                                tick(&mut costs, true, c)?;
                                union[(i * n + j) * v + d[(a + b) % q]] = true;
                            }
                        }
                    }
                }
            }
            index += 1;
        }
    }
    for (cell, membership) in union.chunks_exact(v).enumerate() {
        let values: Vec<_> = (0..v).filter(|&y| membership[y]).collect();
        if saved.answers[cell] != values {
            return Err("decoded answer union changed".into());
        }
    }
    if saved.costs != costs {
        return Err("operation count changed".into());
    }
    Ok(Verified {
        certificate: saved.clone(),
    })
}
