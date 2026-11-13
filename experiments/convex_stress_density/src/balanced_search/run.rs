use super::*;
use crate::{Sample, matrix, random::Random};

pub fn run(
    original: &interval_search::Input,
    prior: &[Model],
    parent: refinement_search::Report,
    policy: Policy,
) -> Result<Report> {
    let samples = search::validate(&original.categorical)?;
    let (n, vocab) = (original.categorical.n, original.categorical.vocab);
    policy.validate(n)?;
    refinement_search::replay(original, prior, &parent.parent, &parent)?;
    let (mut rows, mut columns) = (vec![false; n], vec![false; n]);
    let mut classes = vec![];
    for s in &samples {
        rows[s.cell % n] = true;
        columns[s.cell / n] = true;
        if !classes.contains(&s.label) {
            classes.push(s.label);
        }
    }
    let anchors: Vec<_> = (0..n).filter(|&i| rows[i]).take(2).collect();
    let bound = 4 * parent.buffer_bound_bytes
        + 512 * 1024 * 1024
        + 1024 * policy.total_iterations as u64
        + policy.refinement.seeds as u64
            * (policy.refinement.rounds as u64 * 65536 + 4 * 1024 * 1024);
    if bound > 32 * 1024 * 1024 * 1024 {
        return Err("balanced buffer preflight exceeds32GiB".into());
    }
    let mut out = Report {
        format: 1,
        parent,
        policy,
        observed_rows: rows,
        observed_columns: columns,
        subspaces: vec![],
        attempts: vec![],
        selected: None,
        cost: Cost::default(),
        buffer_bound_bytes: bound,
    };
    out.cost.observation_coverage_visits = samples.len();
    let mut rng = Random::new(original.categorical.options.seed);
    for trial in 0..original.categorical.options.code_trials {
        let sketch = search::sketch(n, vocab, &samples, &classes, &mut rng);
        out.cost.repeated_sketches += 1;
        out.cost.repeated_sketch_observations += 2 * samples.len();
        out.cost.centering_cells += 2 * n * n;
        let slots: Vec<_> = out
            .parent
            .parent
            .proposal
            .exact
            .attempts
            .iter()
            .enumerate()
            .filter(|(_, a)| a.trial == trial)
            .map(|(i, a)| (i, a.rank, a.features.clone()))
            .collect();
        for (index, rank, features) in slots {
            let mut sub = Subspace {
                exact_attempt: index,
                reason: "missing_exact_subspace".into(),
                balanced: None,
                audit: None,
            };
            if let Some(b) = features {
                out.cost.feature_energy_terms += n * rank;
                match feature_balance::build(&b, n, rank, &out.observed_rows) {
                    Err(e) => sub.reason = format!("feature_balance_refused:{e}"),
                    Ok(b) => {
                        match feature_balance::audit(&b) {
                            Err(e) => sub.reason = format!("raw_balance_audit_failed:{e}"),
                            Ok(audit) => {
                                sub.reason = "audited_uniform_feasible_density".into();
                                sub.audit = Some(audit);
                            }
                        }
                        sub.balanced = Some(b);
                    }
                }
            }
            let k = out.subspaces.len();
            out.subspaces.push(sub);
            for variant in std::iter::once(None).chain((0..out.policy.variants.len()).map(Some)) {
                let mut a = Attempt::new(k, variant);
                if anchors.len() < 2 {
                    a.reason = "fewer_than_two_observed_rows".into();
                } else if let Some(b) = out.subspaces[k]
                    .balanced
                    .as_ref()
                    .filter(|_| out.subspaces[k].audit.is_some())
                {
                    let b = b.clone();
                    let left = extract(&mut out, &mut a, &b, [anchors[0], anchors[1]])?;
                    if let Some(left) = left {
                        propose_model(&mut out, &mut a, left, &sketch, &samples, &classes)?;
                    }
                }
                out.attempts.push(a);
            }
        }
    }
    let mut best = f64::INFINITY;
    for (id, _, ce) in candidates(&out)? {
        if !ce.is_finite() {
            return Err("nonfinite balanced candidate CE".into());
        }
        if ce + 1e-12 < best {
            best = ce;
            out.selected = Some(id);
        }
    }
    Ok(out)
}
fn extract(
    out: &mut Report,
    a: &mut Attempt,
    b: &feature_balance::Balanced,
    rows: [usize; 2],
) -> Result<Option<Vec<C>>> {
    let Some(v) = a.variant else {
        out.cost.uniform_column_products += b.n * b.rank * b.rank;
        a.reason = "uniform_feasible_candidate_no_optimality_claim".into();
        return Ok(Some(
            (0..b.n)
                .map(|x| matrix::reading(&b.features, b.n, b.rank, &b.uniform, x, rows[0]))
                .collect(),
        ));
    };
    let remaining = out.policy.total_iterations - out.cost.iterations_charged;
    if remaining == 0 {
        a.reason = "balanced_iteration_budget_exhausted".into();
        return Ok(None);
    }
    let options = interval::Options {
        iterations: out.policy.variants[v].iterations.min(remaining),
        ..out.policy.variants[v]
    };
    let input = interval::Input {
        n: b.n,
        rank: b.rank,
        features: b.features.clone(),
        rows,
        options,
    };
    let result = match interval::solve(input) {
        Err(e) => {
            a.reason = format!("balanced_interval_error:{e}");
            a.iterations_charged = options.iterations;
            out.cost.iterations_charged += options.iterations;
            return Ok(None);
        }
        Ok(result) => result,
    };
    a.iterations_charged = result.trace.len();
    out.cost.iterations += a.iterations_charged;
    out.cost.iterations_charged += a.iterations_charged;
    a.reason = result.reason.clone();
    let (raw, left) = match interval_audit::endpoint(&result.input, &result) {
        Err(e) => {
            a.reason = format!("raw_balanced_interval_audit_failed:{e}");
            (Value::Null, None)
        }
        Ok(raw) => (raw, result.kernel_column.clone()),
    };
    a.extraction = Some(interval_audit::Artifact {
        outcome: result,
        raw_audit: raw,
    });
    Ok(left)
}
fn propose_model(
    out: &mut Report,
    a: &mut Attempt,
    left: Vec<C>,
    sketch: &[C],
    samples: &[Sample],
    classes: &[usize],
) -> Result<()> {
    let (n, vocab) = (
        out.parent.parent.proposal.input.categorical.n,
        out.parent.parent.proposal.input.categorical.vocab,
    );
    let right: Vec<_> = (0..n)
        .map(|j| {
            (0..n)
                .map(|i| sketch[i + n * j].conj() * left[i])
                .sum::<C>()
                .conj()
        })
        .collect();
    out.cost.right_phase_products += n * n;
    let rc = feature_balance::clock(&left, &out.observed_rows);
    let cc = feature_balance::clock(&right, &out.observed_columns);
    a.left = Some(left);
    a.right = Some(right);
    match (rc, cc) {
        (Ok(rc), Ok(cc)) => {
            a.row_clock = Some(rc);
            a.column_clock = Some(cc);
        }
        (rc, cc) => {
            a.reason = format!(
                "partial_clock_refused:row={:?},column={:?}",
                rc.err(),
                cc.err()
            );
            return Ok(());
        }
    }
    let rows = &a.row_clock.as_ref().unwrap().phases;
    let columns = &a.column_clock.as_ref().unwrap().phases;
    out.cost.decoder_observations += samples.len();
    let Some(decoder) = search::decoder(n, vocab, samples, classes, rows, columns) else {
        a.reason = "uncovered_decoder_state".into();
        return Ok(());
    };
    if out.cost.model_audits >= out.policy.model_audits {
        a.reason = "balanced_model_audit_budget_exhausted".into();
        return Ok(());
    }
    let seed = Model::construct(vocab, rows.clone(), columns.clone(), decoder)?;
    out.cost.model_audits += 1;
    out.cost.latent_frequency_audit_allowance += seed.q;
    a.model = Some(seed.clone());
    match seed.audit(samples) {
        Err(e) => {
            a.reason = format!("raw_balanced_model_audit_failed:{e}");
            return Ok(());
        }
        Ok(raw) => {
            a.model_audit = Some(raw);
            a.reason = "audited_balanced_phase_candidate".into();
        }
    }
    refine(out, a, seed)
}
fn refine(out: &mut Report, a: &mut Attempt, seed: Model) -> Result<()> {
    let remaining = out.policy.refinement.assignments - out.cost.assignments_charged;
    if out.cost.seeds_run >= out.policy.refinement.seeds {
        a.refinement_reason = Some("balanced_seed_budget_exhausted".into());
        return Ok(());
    }
    if remaining < 2 {
        a.refinement_reason = Some("balanced_assignment_budget_exhausted".into());
        return Ok(());
    }
    let rounds = out.policy.refinement.rounds.min(remaining / 2);
    let input = refinement::Input {
        seed,
        observations: out
            .parent
            .parent
            .proposal
            .input
            .categorical
            .observations
            .clone(),
        rounds,
    };
    out.cost.seeds_run += 1;
    match refinement::run(input.clone()) {
        Err(e) => {
            a.assignments_charged = 2 * rounds;
            a.refinement_reason = Some(format!("balanced_refinement_error:{e}"));
        }
        Ok(value) => {
            a.assignments_charged = value.cost.assignments;
            a.refinement_reason = Some(value.reason.clone());
            if value.best.is_some() {
                out.cost.endpoint_audits += 1;
                out.cost.latent_frequency_audit_allowance += input.seed.q;
            }
            match refinement_audit::check(&input, &value) {
                Err(e) => a.refinement_reason = Some(format!("raw_balanced_refinement_failed:{e}")),
                Ok(raw) => a.refinement_audit = Some(raw),
            }
            a.refinement = Some(value);
        }
    }
    out.cost.assignments_charged += a.assignments_charged;
    Ok(())
}
