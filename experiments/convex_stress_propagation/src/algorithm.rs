//! Sorted-domain implementation of CM19's exact finite simultaneous sweeps.
use crate::*;

struct Incident {
    rows: Vec<Vec<usize>>,
    columns: Vec<Vec<usize>>,
}

impl Incident {
    fn new(p: &Problem) -> Self {
        let mut result = Self {
            rows: vec![vec![]; p.side],
            columns: vec![vec![]; p.side],
        };
        for (index, t) in p.observations.iter().enumerate() {
            result.rows[t.row].push(index);
            result.columns[t.column].push(index);
        }
        result
    }
}

fn initial(p: &Problem, c: Config, z: usize) -> Domains {
    Domains {
        rows: (0..p.side)
            .map(|i| {
                if i == c.anchor.row {
                    vec![0]
                } else {
                    (0..p.states).collect()
                }
            })
            .collect(),
        columns: (0..p.side)
            .map(|j| {
                if j == c.anchor.column {
                    vec![z]
                } else {
                    (0..p.states).collect()
                }
            })
            .collect(),
    }
}

fn sweep(
    p: &Problem,
    decoder: &[usize],
    old: &Domains,
    c: Config,
    cost: &mut Costs,
    incident: &Incident,
) -> Option<Domains> {
    let mut next = Domains {
        rows: Vec::with_capacity(p.side),
        columns: Vec::with_capacity(p.side),
    };
    for row_side in [true, false] {
        let (own, other) = if row_side {
            (&old.rows, &old.columns)
        } else {
            (&old.columns, &old.rows)
        };
        for (i, values) in own.iter().enumerate() {
            let mut kept = Vec::with_capacity(values.len());
            for &a in values {
                let mut supported = true;
                let edges = if row_side {
                    &incident.rows[i]
                } else {
                    &incident.columns[i]
                };
                for &index in edges {
                    let t = p.observations[index];
                    let there = if row_side { t.column } else { t.row };
                    let mut witness = false;
                    for &b in &other[there] {
                        cost.tick(false, c.max_checks)?;
                        if decoder[(a + b) % p.states] == t.label {
                            witness = true;
                            break;
                        }
                    }
                    if !witness {
                        supported = false;
                        break;
                    }
                }
                if supported {
                    kept.push(a);
                }
            }
            if row_side {
                next.rows.push(kept);
            } else {
                next.columns.push(kept);
            }
        }
    }
    Some(next)
}

pub(crate) fn refused(reason: RefusalReason, cost: Costs, done: usize, total: usize) -> Outcome {
    Outcome::Refused(Refusal {
        reason,
        costs: cost,
        completed_branches: done,
        total_branches: total,
    })
}

/// A complete result unions every declared decoder and every compatible anchor
/// state. Budget exhaustion never returns a partial set as a certificate.
pub fn propagate(problem: &Problem, config: Config) -> Result<Outcome> {
    let plan = preflight(problem, config)?;
    if plan.memory_bound > config.memory_limit {
        return Ok(refused(
            RefusalReason::Memory {
                required: plan.memory_bound,
                limit: config.memory_limit,
            },
            Costs::default(),
            0,
            plan.total_branches,
        ));
    }
    if plan.total_branches > config.max_branches {
        return Ok(refused(
            RefusalReason::BranchBudget {
                required: plan.total_branches,
                limit: config.max_branches,
            },
            Costs::default(),
            0,
            plan.total_branches,
        ));
    }
    let mut p = problem.clone();
    p.observations.sort_unstable(); // preserve repetitions and contradictions
    let incident = Incident::new(&p);
    let mut costs = Costs::default();
    let mut branches = Vec::with_capacity(plan.total_branches);
    let mut union = vec![false; p.side * p.side * p.vocab];
    for (k, decoder) in p.decoders.iter().enumerate() {
        for (z, &label) in decoder.iter().enumerate() {
            if label != config.anchor.label {
                continue;
            }
            let mut domains = initial(&p, config, z);
            let mut removed_per_sweep = Vec::with_capacity(config.steps);
            for _ in 0..config.steps {
                let Some(next) = sweep(&p, decoder, &domains, config, &mut costs, &incident) else {
                    return Ok(refused(
                        RefusalReason::WorkBudget,
                        costs,
                        branches.len(),
                        plan.total_branches,
                    ));
                };
                let size =
                    |d: &Domains| d.rows.iter().chain(&d.columns).map(Vec::len).sum::<usize>();
                removed_per_sweep.push(size(&domains) - size(&next));
                domains = next;
            }
            let live = domains.live();
            if live {
                for i in 0..p.side {
                    for j in 0..p.side {
                        for &a in &domains.rows[i] {
                            for &b in &domains.columns[j] {
                                if costs.tick(true, config.max_checks).is_none() {
                                    return Ok(refused(
                                        RefusalReason::WorkBudget,
                                        costs,
                                        branches.len(),
                                        plan.total_branches,
                                    ));
                                }
                                union[(i * p.side + j) * p.vocab + decoder[(a + b) % p.states]] =
                                    true;
                            }
                        }
                    }
                }
            }
            branches.push(Branch {
                decoder: k,
                anchor_state: z,
                domains,
                removed_per_sweep,
                live,
            });
        }
    }
    let answers = union
        .chunks_exact(p.vocab)
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(y, &yes)| yes.then_some(y))
                .collect()
        })
        .collect();
    Ok(Outcome::Complete(Box::new(Certificate {
        version: 1,
        problem: p,
        config,
        memory_bound: plan.memory_bound,
        branches,
        answers,
        costs,
    })))
}
