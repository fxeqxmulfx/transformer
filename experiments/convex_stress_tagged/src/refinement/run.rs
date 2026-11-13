//! Bounded CM25 free-axis updates. Only raw observed labels enter the costs.
use super::*;
use std::collections::BTreeSet;

fn record(
    out: &mut Outcome,
    samples: &[Sample],
    mut state: State,
    kind: Kind,
    minimum_costs: Option<Vec<usize>>,
) -> Result<()> {
    let (n, q) = (out.input.seed.n, out.input.seed.q);
    let mut seen = vec![false; q];
    state.errors = 0;
    for s in samples {
        let z = (state.rows[s.cell % n] + state.columns[s.cell / n]) % q;
        state.errors += usize::from(state.decoder[z] != s.label);
        seen[z] = true;
    }
    state.unseen_latents = (0..q).filter(|&z| !seen[z]).collect();
    out.cost.state_observations += samples.len();
    if out
        .steps
        .last()
        .is_some_and(|old| old.state.errors < state.errors)
    {
        return Err("free coordinate update increased actual observed CE".into());
    }
    if state.unseen_latents.is_empty()
        && out
            .best
            .is_none_or(|i| state.errors < out.steps[i].state.errors)
    {
        out.best = Some(out.steps.len());
    }
    out.steps.push(Step {
        kind,
        state,
        minimum_costs,
    });
    Ok(())
}
fn decoder(out: &mut Outcome, samples: &[Sample], classes: &[usize]) -> Result<()> {
    let mut state = out.steps.last().unwrap().state.clone();
    let (n, q, v) = (out.input.seed.n, out.input.seed.q, out.input.seed.vocab);
    let mut counts = vec![0; q * v];
    for s in samples {
        counts[((state.rows[s.cell % n] + state.columns[s.cell / n]) % q) * v + s.label] += 1;
    }
    out.cost.decoder_observations += samples.len();
    for (z, d) in state.decoder.iter_mut().enumerate() {
        for &c in classes {
            out.cost.decoder_comparisons += 1;
            if counts[z * v + c] > counts[z * v + *d] {
                *d = c;
            }
        }
    }
    record(out, samples, state, Kind::Decoder, None)
}
fn axis(out: &mut Outcome, samples: &[Sample], row: bool) -> Result<()> {
    let mut state = out.steps.last().unwrap().state.clone();
    let (n, q) = (out.input.seed.n, out.input.seed.q);
    let mut costs = vec![0; n * q];
    for s in samples {
        let (a, other) = if row {
            (s.cell % n, state.columns[s.cell / n])
        } else {
            (s.cell / n, state.rows[s.cell % n])
        };
        for z in 0..q {
            costs[a * q + z] += usize::from(state.decoder[(z + other) % q] != s.label);
        }
    }
    let target = if row {
        &mut state.rows
    } else {
        &mut state.columns
    };
    let mut minima = Vec::with_capacity(n);
    for (a, phase) in target.iter_mut().enumerate() {
        for z in 0..q {
            if costs[a * q + z] < costs[a * q + *phase] {
                *phase = z;
            }
        }
        minima.push(costs[a * q + *phase]);
    }
    out.cost.axis_steps += 1;
    out.cost.axis_observation_comparisons += q * samples.len();
    out.cost.axis_minimum_comparisons += n * q;
    record(
        out,
        samples,
        state,
        if row { Kind::Row } else { Kind::Column },
        Some(minima),
    )
}
pub fn run(input: Input) -> Result<Outcome> {
    let samples = validate(&input)?;
    let state = State {
        rows: input.seed.rows.clone(),
        columns: input.seed.columns.clone(),
        decoder: input.seed.decoder.clone(),
        errors: 0,
        unseen_latents: vec![],
    };
    let mut classes = vec![];
    for s in &samples {
        if !classes.contains(&s.label) {
            classes.push(s.label);
        }
    }
    let mut out = Outcome {
        input,
        steps: vec![],
        best: None,
        reason: "round_cap".into(),
        cost: Cost {
            class_observations: samples.len(),
            ..Cost::default()
        },
    };
    record(&mut out, &samples, state, Kind::Seed, None)?;
    let mut visited = BTreeSet::new();
    'rounds: for _ in 0..out.input.rounds {
        let s = &out.steps.last().unwrap().state;
        if s.errors == 0 {
            break;
        }
        if !visited.insert((s.rows.clone(), s.columns.clone(), s.decoder.clone())) {
            out.reason = "repeated_state".into();
            break;
        }
        decoder(&mut out, &samples, &classes)?;
        for row in [true, false] {
            if out.cost.axis_steps == out.input.axis_budget {
                out.reason = "axis_budget_exhausted".into();
                break 'rounds;
            }
            axis(&mut out, &samples, row)?;
            decoder(&mut out, &samples, &classes)?;
        }
    }
    let last = &out.steps.last().unwrap().state;
    if last.errors == 0 {
        out.reason = if last.unseen_latents.is_empty() {
            "exact_observed_fit"
        } else {
            "exact_fit_uncovered"
        }
        .into();
    }
    Ok(out)
}
pub fn replay(input: &Input, out: &Outcome) -> Result<()> {
    if &out.input != input || run(input.clone())? != *out {
        return Err("free-phase original input/state/work replay mismatch".into());
    }
    Ok(())
}
