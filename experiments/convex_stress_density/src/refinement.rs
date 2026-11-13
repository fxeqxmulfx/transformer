//! CM23 / Lean 0bb5496f: bounded observed-CE refinement of a supplied vertex.
//! No hidden labels, new spectral bound, global oracle or continuous gradient.
use crate::{Model, Result, Sample, assignment_certificate as assignment};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub seed: Model,
    pub observations: Vec<[usize; 2]>,
    pub rounds: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub decoder: Vec<usize>,
    pub errors: usize,
    pub covered: bool,
}
impl State {
    pub fn model(&self, vocab: usize) -> Result<Model> {
        Model::construct(
            vocab,
            self.rows.clone(),
            self.columns.clone(),
            self.decoder.clone(),
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Seed,
    Decoder,
    Row,
    Column,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub kind: Kind,
    pub state: State,
    pub certificate: Option<assignment::Certificate>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cost {
    pub class_observations: usize,
    pub state_observations: usize,
    pub decoder_observations: usize,
    pub decoder_comparisons: usize,
    pub assignments: usize,
    pub assignment_observation_comparisons: usize,
    pub assignment_relaxations: usize,
    pub certificate_inequalities: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub input: Input,
    pub steps: Vec<Step>,
    pub cost: Cost,
    pub best: Option<usize>,
    pub reason: String,
}
pub fn validate(input: &Input) -> Result<Vec<Sample>> {
    let m = &input.seed;
    m.validate_shape()?;
    if m.n != m.q
        || !assignment::permutation(&m.row_phases, m.n)
        || !assignment::permutation(&m.column_phases, m.n)
        || !(1..=128).contains(&input.rounds)
    {
        return Err("refinement requires bounded rounds and permutation phase seeds".into());
    }
    let samples: Vec<_> = input
        .observations
        .iter()
        .map(|&[cell, label]| Sample { cell, label })
        .collect();
    convex_stress_decode::validate_observations(m.n, m.vocab, &samples)?;
    Ok(samples)
}
fn record(
    out: &mut Outcome,
    samples: &[Sample],
    mut state: State,
    kind: Kind,
    certificate: Option<assignment::Certificate>,
) -> Result<()> {
    let n = state.rows.len();
    let mut covered = vec![false; n];
    state.errors = 0;
    for s in samples {
        let z = (state.rows[s.cell % n] + state.columns[s.cell / n]) % n;
        state.errors += usize::from(state.decoder[z] != s.label);
        covered[z] = true;
    }
    state.covered = covered.iter().all(|&x| x);
    out.cost.state_observations += samples.len();
    if out
        .steps
        .last()
        .is_some_and(|s| state.errors > s.state.errors)
    {
        return Err("certified update increased observed CE".into());
    }
    if state.covered
        && out
            .best
            .is_none_or(|i| state.errors < out.steps[i].state.errors)
    {
        out.best = Some(out.steps.len());
    }
    out.steps.push(Step {
        kind,
        state,
        certificate,
    });
    Ok(())
}
fn decoder(out: &mut Outcome, samples: &[Sample], classes: &[usize]) -> Result<()> {
    let mut state = out.steps.last().unwrap().state.clone();
    let (n, vocab) = (out.input.seed.n, out.input.seed.vocab);
    let mut counts = vec![0; n * vocab];
    for s in samples {
        counts[((state.rows[s.cell % n] + state.columns[s.cell / n]) % n) * vocab + s.label] += 1;
    }
    out.cost.decoder_observations += samples.len();
    for (l, d) in state.decoder.iter_mut().enumerate() {
        for &c in classes {
            out.cost.decoder_comparisons += 1;
            if counts[l * vocab + c] > counts[l * vocab + *d] {
                *d = c;
            }
        }
    }
    record(out, samples, state, Kind::Decoder, None)
}
fn axis(out: &mut Outcome, samples: &[Sample], row: bool) -> Result<()> {
    let mut state = out.steps.last().unwrap().state.clone();
    let n = out.input.seed.n;
    let mut costs = vec![0; n * n];
    for s in samples {
        let (i, other) = if row {
            (s.cell % n, state.columns[s.cell / n])
        } else {
            (s.cell / n, state.rows[s.cell % n])
        };
        for z in 0..n {
            costs[i * n + z] += usize::from(state.decoder[(z + other) % n] != s.label);
        }
    }
    out.cost.assignment_observation_comparisons += n * samples.len();
    let certificate = assignment::solve(&costs, n)?;
    out.cost.assignments += 1;
    out.cost.assignment_relaxations += certificate.relaxations;
    out.cost.certificate_inequalities += n * n;
    if row {
        state.rows = certificate.permutation.clone();
    } else {
        state.columns = certificate.permutation.clone();
    }
    record(
        out,
        samples,
        state,
        if row { Kind::Row } else { Kind::Column },
        Some(certificate),
    )
}
pub fn run(input: Input) -> Result<Outcome> {
    let samples = validate(&input)?;
    let state = State {
        rows: input.seed.row_phases.clone(),
        columns: input.seed.column_phases.clone(),
        decoder: input.seed.decoder.clone(),
        errors: 0,
        covered: false,
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
    for _ in 0..out.input.rounds {
        let state = &out.steps.last().unwrap().state;
        if state.errors == 0 {
            out.reason = "exact_observed_fit".into();
            break;
        }
        if !visited.insert((
            state.rows.clone(),
            state.columns.clone(),
            state.decoder.clone(),
        )) {
            out.reason = "repeated_state".into();
            break;
        }
        decoder(&mut out, &samples, &classes)?;
        axis(&mut out, &samples, true)?;
        decoder(&mut out, &samples, &classes)?;
        axis(&mut out, &samples, false)?;
        decoder(&mut out, &samples, &classes)?;
    }
    if out.steps.last().unwrap().state.errors == 0 {
        out.reason = "exact_observed_fit".into();
    }
    Ok(out)
}
pub fn replay(input: &Input, out: &Outcome) -> Result<()> {
    if &out.input != input || run(input.clone())? != *out {
        return Err("refinement original-input or full-state replay mismatch".into());
    }
    Ok(())
}
