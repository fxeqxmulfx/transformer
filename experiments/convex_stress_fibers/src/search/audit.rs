use super::*;
use serde_json::json;

pub fn check(input: &Input, report: &Report) -> Result<Value> {
    let bound = input.validate()?;
    if &report.input != input
        || report.buffer_bound_bytes != bound
        || report.partition != groups::collect(input.n, &input.observations)
        || report.candidates.len() > input.policy.models
        || report.work.ce_gradients != 0
    {
        return Err("fiber input/partition/storage/gradient provenance mismatch".into());
    }
    let refused = if !finite_system::prime(input.n as u16) {
        Some("unsupported_prime_domain")
    } else if !report.partition.missing_rows.is_empty()
        || !report.partition.missing_columns.is_empty()
    {
        Some("missing_observed_axis")
    } else {
        None
    };
    if let Some(reason) = refused {
        if report.reason != reason
            || !report.attempts.is_empty()
            || !report.candidates.is_empty()
            || report.selected.is_some()
        {
            return Err("unsupported domain or missing axes admitted discovery".into());
        }
        return Ok(json!({"refusal_checked":reason,"candidate_count":0}));
    }
    if report.attempts.len() != report.partition.sizes.len() {
        return Err("missing multiplicity guesses".into());
    }
    let (
        mut accepted,
        mut trials,
        mut products,
        mut updates,
        mut operations,
        mut original_products,
    ) = (0, 0, 0, 0u64, 0, 0u64);
    for (guess, a) in report.attempts.iter().enumerate() {
        let all: Vec<_> = report
            .partition
            .groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.members.len() == a.size)
            .map(|(i, _)| i)
            .collect();
        let expected: Vec<_> = all.iter().copied().take(input.policy.constraints).collect();
        if a.size != report.partition.sizes[guess]
            || a.group_indices != expected
            || a.constraints_truncated != (all.len() > input.policy.constraints)
        {
            return Err("fiber group guess, equation order or cap differs".into());
        }
        if guess >= input.policy.guesses {
            if a.reason != "guess_budget"
                || a.elimination.is_some()
                || a.elimination_audit.is_some()
                || a.total_coefficients.is_some()
                || !a.trials.is_empty()
            {
                return Err("skipped guess contains uncharged work".into());
            }
            continue;
        }
        let problem = groups::problem(
            input.n,
            &report.partition.groups,
            &expected,
            input.policy.elimination.clone(),
        );
        let solved = a
            .elimination
            .as_ref()
            .ok_or("missing finite-system outcome")?;
        if Some(finite_system::check(&problem, solved)?) != a.elimination_audit {
            return Err("independent elimination audit differs".into());
        }
        updates += solved.work.scalar_updates;
        operations += solved.work.row_operations;
        let Some(space) = &solved.space else {
            if a.total_coefficients.is_some() || !a.trials.is_empty() || a.reason != solved.reason {
                return Err("incomplete elimination supplied phase proposals".into());
            }
            continue;
        };
        let total = (input.n as u64).pow(space.directions.len() as u32);
        if a.total_coefficients != Some(total)
            || a.trials.len() as u64 > total
            || a.trials.len() > input.policy.enumeration
        {
            return Err("enumeration extent exceeds the actual affine space or cap".into());
        }
        for (ordinal, trial) in a.trials.iter().enumerate() {
            if accepted == input.policy.models
                || trial.coefficients.len() != space.directions.len()
                || trial.coefficients.iter().any(|&c| c as usize >= input.n)
            {
                return Err("invalid or unbounded affine coefficient vector".into());
            }
            let encoded: u64 = trial
                .coefficients
                .iter()
                .enumerate()
                .map(|(i, &v)| v as u64 * (input.n as u64).pow(i as u32))
                .sum();
            if encoded != ordinal as u64 {
                return Err("enumeration skips or repeats a coefficient tuple".into());
            }
            let mut x = space.particular.clone();
            for (d, &c) in space.directions.iter().zip(&trial.coefficients) {
                for (z, &v) in x.iter_mut().zip(d) {
                    *z = ((*z as usize + c as usize * v as usize) % input.n) as u16;
                }
            }
            witness::check(
                input,
                &report.partition,
                a.size,
                &x,
                trial,
                &report.candidates,
                accepted,
            )?;
            if let Decision::Accepted { candidate } = trial.decision {
                let c = &report.candidates[candidate];
                if c.guess != guess || c.trial != ordinal {
                    return Err("candidate origin differs".into());
                }
                // Independently check every original equation for each admitted
                // endpoint, beyond the affine-basis certificate for all tuples.
                for (row, &rhs) in problem.system.rows.iter().zip(&problem.system.right) {
                    let sum: u64 = row.iter().zip(&x).map(|(&a, &b)| a as u64 * b as u64).sum();
                    original_products += x.len() as u64;
                    if sum % input.n as u64 != rhs as u64 {
                        return Err("candidate violates an original equation".into());
                    }
                }
                accepted += 1;
            }
            trials += 1;
            products += x.len() * trial.coefficients.len();
        }
        let reason = if a.trials.len() as u64 == total {
            "complete_enumeration"
        } else if accepted == input.policy.models {
            "model_cap"
        } else if a.trials.len() == input.policy.enumeration {
            "enumeration_cap"
        } else {
            return Err("uncapped enumeration stopped early".into());
        };
        if a.reason != reason {
            return Err("enumeration stopping reason differs".into());
        }
    }
    let reason = if accepted > 0 {
        "admitted_proposals"
    } else if report.partition.sizes.is_empty() {
        "no_equal_label_groups"
    } else {
        "no_admissible_candidate"
    };
    if accepted != report.candidates.len()
        || report.selected != if accepted > 0 { Some(0) } else { None }
        || report.reason != reason
        || report.work.coefficient_vectors != trials
        || report.work.affine_products != products
        || report.work.elimination_scalar_updates != updates
        || report.work.elimination_row_operations != operations
        || report.work.raw_frequencies != accepted * input.n
        || report.work.grouping_cell_visits != 2 * input.n * input.observations.len()
    {
        return Err("fiber selection, construction extent or work differs".into());
    }
    Ok(
        json!({"candidate_count":accepted,"trial_witnesses_checked":trials,
        "original_endpoint_equation_products":original_products,"all_actual_fiber_premises_checked":true,
        "raw_frequencies":accepted*input.n,"unseen_answers_identified":false,
        "scope":"declared guessed fiber systems and bounded proposal enumeration"}),
    )
}
