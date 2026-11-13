use super::*;

pub(super) fn partition(input: &Input) -> Partition {
    let (mut classes, mut class_indices) = (vec![], vec![]);
    let (mut rows, mut columns) = (vec![false; input.n], vec![false; input.n]);
    for &[cell, label] in &input.observations {
        let index = classes.iter().position(|&c| c == label).unwrap_or_else(|| {
            classes.push(label);
            classes.len() - 1
        });
        class_indices.push(index);
        rows[cell % input.n] = true;
        columns[cell / input.n] = true;
    }
    Partition {
        classes,
        class_indices,
        missing_rows: (0..input.n).filter(|&i| !rows[i]).collect(),
        missing_columns: (0..input.n).filter(|&i| !columns[i]).collect(),
        anchor_row: input.observations[0][0] % input.n,
        retained_observations: input
            .observations
            .len()
            .min(input.policy.observation_equations),
    }
}
pub(super) fn system(input: &Input, p: &Partition) -> System {
    let variables = 2 * input.n + p.classes.len();
    let mut rows = vec![];
    for (s, &class) in input
        .observations
        .iter()
        .zip(&p.class_indices)
        .take(p.retained_observations)
    {
        let mut r = vec![0; variables];
        r[s[0] % input.n] = 1;
        r[input.n + s[0] / input.n] = 1;
        r[2 * input.n + class] = -1;
        rows.push(r);
    }
    // The two gauge pins are valid shifts; missing axes are explicit choices
    // unconstrained by observations. They provide no unseen-answer guarantee.
    for i in [p.anchor_row, 2 * input.n]
        .into_iter()
        .chain(p.missing_rows.iter().copied())
        .chain(p.missing_columns.iter().map(|&j| input.n + j))
    {
        let mut r = vec![0; variables];
        r[i] = 1;
        rows.push(r);
    }
    System {
        right: vec![0; rows.len()],
        variables,
        rows,
    }
}
pub(super) fn factors(q: usize) -> Vec<Factor> {
    let mut remaining = q;
    let mut factors = vec![];
    for prime in 2..=q {
        if !remaining.is_multiple_of(prime) {
            continue;
        }
        let (mut exponent, mut modulus) = (0, 1);
        while remaining.is_multiple_of(prime) {
            remaining /= prime;
            exponent += 1;
            modulus *= prime;
        }
        factors.push(Factor {
            prime: prime as u16,
            exponent,
            modulus: modulus as u16,
        });
    }
    factors
}
pub(super) fn anchored(system: &System, n: usize, class: usize) -> System {
    let mut system = system.clone();
    let mut row = vec![0; system.variables];
    row[2 * n + class] = 1;
    system.rows.push(row);
    system.right.push(1);
    system
}
pub(super) fn field(
    system: &System,
    prime: u16,
    policy: finite_system::Policy,
) -> finite_system::Problem {
    finite_system::Problem {
        system: finite_system::System {
            prime,
            variables: system.variables,
            rows: system
                .rows
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|a| a.rem_euclid(prime as i64) as u16)
                        .collect()
                })
                .collect(),
            right: system
                .right
                .iter()
                .map(|b| b.rem_euclid(prime as i64) as u16)
                .collect(),
        },
        policy,
    }
}
pub(super) fn stage_policy(p: &Policy, work: &Work, nullity: usize) -> finite_system::Policy {
    finite_system::Policy {
        scalar_updates: p
            .scalar_updates_per_stage
            .min(p.total_scalar_updates - work.finite_scalar_updates),
        row_operations: p.row_operations_per_stage,
        nullity,
    }
}
pub(super) fn charge(work: &mut Work, out: &finite_system::Outcome) {
    work.finite_stages += 1;
    work.finite_scalar_updates += out.work.scalar_updates;
    work.finite_row_operations += out.work.row_operations;
    work.finite_pivot_tests += out.work.pivot_tests;
    work.finite_inverse_tests += out.work.inverse_tests;
}
