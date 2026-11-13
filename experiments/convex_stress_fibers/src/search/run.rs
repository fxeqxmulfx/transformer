use super::*;

pub fn run(input: Input) -> Result<Report> {
    let buffer_bound_bytes = input.validate()?;
    let partition = groups::collect(input.n, &input.observations);
    let mut out = Report {
        work: Work {
            grouping_cell_visits: 2 * input.n * input.observations.len(),
            ..Work::default()
        },
        input,
        buffer_bound_bytes,
        partition,
        attempts: vec![],
        candidates: vec![],
        selected: None,
        reason: "no_admissible_candidate".into(),
    };
    if !finite_system::prime(out.input.n as u16) {
        out.reason = "unsupported_prime_domain".into();
        return Ok(out);
    }
    if !out.partition.missing_rows.is_empty() || !out.partition.missing_columns.is_empty() {
        out.reason = "missing_observed_axis".into();
        return Ok(out);
    }
    for (guess, &size) in out.partition.sizes.iter().enumerate() {
        let all: Vec<_> = out
            .partition
            .groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.members.len() == size)
            .map(|(i, _)| i)
            .collect();
        let mut attempt = Attempt {
            size,
            constraints_truncated: all.len() > out.input.policy.constraints,
            group_indices: all.into_iter().take(out.input.policy.constraints).collect(),
            elimination: None,
            elimination_audit: None,
            total_coefficients: None,
            trials: vec![],
            reason: "guess_budget".into(),
        };
        if guess < out.input.policy.guesses {
            let problem = groups::problem(
                out.input.n,
                &out.partition.groups,
                &attempt.group_indices,
                out.input.policy.elimination.clone(),
            );
            let solved = finite_system::solve(problem.clone())?;
            attempt.elimination_audit = Some(finite_system::check(&problem, &solved)?);
            out.work.elimination_scalar_updates += solved.work.scalar_updates;
            out.work.elimination_row_operations += solved.work.row_operations;
            attempt.reason = solved.reason.clone();
            if let Some(space) = &solved.space {
                let total = (out.input.n as u64).pow(space.directions.len() as u32);
                attempt.total_coefficients = Some(total);
                attempt.reason = "complete_enumeration".into();
                for ordinal in 0..total {
                    if out.candidates.len() == out.input.policy.models {
                        attempt.reason = "model_cap".into();
                        break;
                    }
                    if attempt.trials.len() == out.input.policy.enumeration {
                        attempt.reason = "enumeration_cap".into();
                        break;
                    }
                    let mut remaining = ordinal;
                    let coefficients: Vec<_> = (0..space.directions.len())
                        .map(|_| {
                            let z = (remaining % out.input.n as u64) as u16;
                            remaining /= out.input.n as u64;
                            z
                        })
                        .collect();
                    let x = candidate::point(space, &coefficients, out.input.n);
                    out.work.coefficient_vectors += 1;
                    out.work.affine_products += x.len() * coefficients.len();
                    let decision = match candidate::classify(
                        &out.input,
                        &out.partition,
                        size,
                        &x,
                        &mut out.work,
                    ) {
                        Err(reason) => reason,
                        Ok(decoder) => {
                            let model = Model::construct(
                                out.input.vocab,
                                x[..out.input.n].iter().map(|&v| v as usize).collect(),
                                x[out.input.n..2 * out.input.n]
                                    .iter()
                                    .map(|&v| v as usize)
                                    .collect(),
                                decoder,
                            )?;
                            if let Some(index) = out
                                .candidates
                                .iter()
                                .position(|c| c.model.output_labels == model.output_labels)
                            {
                                Decision::Duplicate { candidate: index }
                            } else {
                                let audit = model.audit(&out.input.samples())?;
                                if audit["training_errors"] != 0 {
                                    return Err("admitted categorical fiber candidate does not fit its observations".into());
                                }
                                out.work.raw_frequencies += model.q;
                                let index = out.candidates.len();
                                out.candidates.push(Candidate {
                                    guess,
                                    trial: attempt.trials.len(),
                                    common_mass: x[2 * out.input.n] as usize,
                                    model,
                                    audit,
                                });
                                // Every admitted vertex is an exact unit-logit CE minimum.
                                // Preserve the first one; this is no unseen-answer certificate.
                                out.selected.get_or_insert(index);
                                Decision::Accepted { candidate: index }
                            }
                        }
                    };
                    attempt.trials.push(Trial {
                        coefficients,
                        decision,
                    });
                }
            }
            attempt.elimination = Some(solved);
        }
        out.attempts.push(attempt);
    }
    if out.selected.is_some() {
        out.reason = "admitted_proposals".into();
    } else if out.partition.sizes.is_empty() {
        out.reason = "no_equal_label_groups".into();
    }
    Ok(out)
}
