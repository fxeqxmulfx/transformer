//! Training-only categorical proposal search for the CM15 convex decoder hull.
//! Random centered label sketches propose singular-vector phases. Alternating
//! exact row/column assignments and empirical decoder updates reduce the number
//! of observed wrong one-hot logits. At unit scale that count equals ordinary
//! CE excess times the sample count. No input-token arithmetic labels are used.
use crate::{C, Model, Result, Sample, assignment, random::Random, validate_observations};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    pub code_trials: usize,
    pub power_steps: usize,
    pub refinement_steps: usize,
    pub seed: u32,
}
#[derive(Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cost {
    pub code_trials: usize,
    pub power_steps: usize,
    pub power_matvecs: usize,
    pub assignment_sweeps: usize,
    pub validation_observations: usize,
    pub class_observations: usize,
    pub sketch_observations: usize,
    pub template_observations: usize,
    pub verification_observations: usize,
    pub assignment_comparisons: usize,
}
#[derive(Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trial {
    pub trial_index: usize,
    pub reason: String,
    pub power_steps: usize,
    pub power_matvecs: usize,
    pub assignment_sweeps: usize,
    pub states_scored: usize,
    pub best_errors: Option<usize>,
    pub best_covered_errors: Option<usize>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub model: Option<Model>,
    pub exact_fit: bool,
    pub best_errors: Option<usize>,
    pub reason: String,
    pub cost: Cost,
    pub improvement_history: Vec<usize>,
    pub trials: Vec<Trial>,
}

fn normalize(x: &mut [C]) -> bool {
    let norm = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return false;
    }
    for z in x {
        *z /= norm;
    }
    true
}
fn clock(x: &[C]) -> Option<Vec<usize>> {
    if x.iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.norm() < 1e-12)
    {
        return None;
    }
    let mut order: Vec<_> = (0..x.len()).collect();
    order.sort_by(|&i, &j| x[i].arg().total_cmp(&x[j].arg()).then(i.cmp(&j)));
    let mut rank = vec![0; x.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }
    Some(rank)
}
fn phases(
    x: &[C],
    n: usize,
    rng: &mut Random,
    steps: usize,
    cost: &mut Cost,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut right: Vec<_> = (0..n)
        .map(|_| C::new(rng.gauss(0., 1.), rng.gauss(0., 1.)))
        .collect();
    if !normalize(&mut right) {
        return None;
    }
    let mut left = vec![C::default(); n];
    for _ in 0..steps {
        cost.power_matvecs += 1;
        left.fill(C::default());
        for j in 0..n {
            for i in 0..n {
                left[i] += x[i + n * j] * right[j];
            }
        }
        if !normalize(&mut left) {
            return None;
        }
        cost.power_matvecs += 1;
        right.fill(C::default());
        for j in 0..n {
            for i in 0..n {
                right[j] += x[i + n * j].conj() * left[i];
            }
        }
        if !normalize(&mut right) {
            return None;
        }
        cost.power_steps += 1;
    }
    cost.power_matvecs += 1;
    left.fill(C::default());
    for j in 0..n {
        for i in 0..n {
            left[i] += x[i + n * j] * right[j];
        }
    }
    Some((
        clock(&left)?,
        clock(&right.iter().map(|z| z.conj()).collect::<Vec<_>>())?,
    ))
}
fn template(
    n: usize,
    vocab: usize,
    classes: &[usize],
    samples: &[Sample],
    rows: &[usize],
    columns: &[usize],
    cost: &mut Cost,
) -> (Vec<usize>, usize, bool) {
    let mut counts = vec![0usize; n * vocab];
    for s in samples {
        counts[((rows[s.cell % n] + columns[s.cell / n]) % n) * vocab + s.label] += 1;
    }
    cost.template_observations += samples.len();
    let mut decoder = vec![0; n];
    let mut correct = 0;
    let mut covered = true;
    for phase in 0..n {
        let mut best = classes[0];
        for &c in &classes[1..] {
            if counts[phase * vocab + c] > counts[phase * vocab + best] {
                best = c;
            }
        }
        decoder[phase] = best;
        correct += counts[phase * vocab + best];
        covered &= counts[phase * vocab + best] > 0;
    }
    (decoder, samples.len() - correct, covered)
}
fn assign(
    n: usize,
    samples: &[Sample],
    other: &[usize],
    decoder: &[usize],
    row_axis: bool,
    cost: &mut Cost,
) -> Result<Vec<usize>> {
    let mut loss = vec![0usize; n * n];
    for s in samples {
        let (i, j) = if row_axis {
            (s.cell % n, s.cell / n)
        } else {
            (s.cell / n, s.cell % n)
        };
        for phase in 0..n {
            loss[i * n + phase] += usize::from(decoder[(phase + other[j]) % n] != s.label);
        }
    }
    cost.assignment_sweeps += 1;
    cost.assignment_comparisons += n * samples.len();
    assignment::minimize(&loss, n)
}

pub fn propose(n: usize, vocab: usize, samples: &[Sample], options: Options) -> Result<Proposal> {
    validate_observations(n, vocab, samples)?;
    if !(1..=64).contains(&options.code_trials)
        || !(1..=128).contains(&options.power_steps)
        || !(1..=128).contains(&options.refinement_steps)
    {
        return Err("invalid bounded proposal options".into());
    }
    let mut seen = vec![false; vocab];
    let mut classes = vec![];
    for s in samples {
        if !seen[s.label] {
            seen[s.label] = true;
            classes.push(s.label);
        }
    }
    let mut out = Proposal {
        model: None,
        exact_fit: false,
        best_errors: None,
        reason: "no_covered_candidate".into(),
        cost: Cost {
            validation_observations: samples.len(),
            class_observations: samples.len(),
            ..Cost::default()
        },
        improvement_history: vec![],
        trials: vec![],
    };
    let mut rng = Random::new(options.seed);
    for trial_index in 0..options.code_trials {
        out.cost.code_trials += 1;
        let mut trial = Trial {
            trial_index,
            reason: "refinement_cap".into(),
            ..Trial::default()
        };
        let before_power = out.cost.power_steps;
        let before_matvecs = out.cost.power_matvecs;
        let before_sweeps = out.cost.assignment_sweeps;
        // First-occurrence class order makes categorical output renaming preserve
        // this sketch when the observation order and RNG seed are preserved.
        let mut codes = vec![C::default(); vocab];
        for &c in &classes {
            codes[c] = C::new(rng.gauss(0., 1.), rng.gauss(0., 1.));
        }
        let mean = samples.iter().map(|s| codes[s.label]).sum::<C>() / samples.len() as f64;
        let mut sketch = vec![C::default(); n * n];
        for s in samples {
            sketch[s.cell] = codes[s.label] - mean;
        }
        out.cost.sketch_observations += 2 * samples.len();
        let Some((mut rows, mut columns)) =
            phases(&sketch, n, &mut rng, options.power_steps, &mut out.cost)
        else {
            trial.reason = "power_or_clock_failed".into();
            trial.power_steps = out.cost.power_steps - before_power;
            trial.power_matvecs = out.cost.power_matvecs - before_matvecs;
            out.trials.push(trial);
            continue;
        };
        trial.power_steps = out.cost.power_steps - before_power;
        trial.power_matvecs = out.cost.power_matvecs - before_matvecs;
        let mut visited = BTreeSet::new();
        for iteration in 0..=options.refinement_steps {
            if !visited.insert((rows.clone(), columns.clone())) {
                trial.reason = "repeated_state".into();
                break;
            }
            let (decoder, errors, covered) =
                template(n, vocab, &classes, samples, &rows, &columns, &mut out.cost);
            trial.states_scored += 1;
            trial.best_errors = Some(trial.best_errors.map_or(errors, |old| old.min(errors)));
            if covered {
                trial.best_covered_errors = Some(
                    trial
                        .best_covered_errors
                        .map_or(errors, |old| old.min(errors)),
                );
            }
            if covered && out.best_errors.is_none_or(|old| errors < old) {
                let model =
                    Model::construct(vocab, rows.clone(), columns.clone(), decoder.clone())?;
                let actual = samples
                    .iter()
                    .filter(|s| model.output_labels[s.cell] != s.label)
                    .count();
                out.cost.verification_observations += samples.len();
                if actual != errors {
                    return Err("proposal score and saved table differ".into());
                }
                out.model = Some(model);
                out.best_errors = Some(errors);
                out.improvement_history.push(errors);
                out.reason = "best_covered_candidate_without_exact_fit".into();
                if errors == 0 {
                    out.exact_fit = true;
                    out.reason = "exact_observed_fit".into();
                    trial.reason = "exact_observed_fit".into();
                    trial.assignment_sweeps = out.cost.assignment_sweeps - before_sweeps;
                    out.trials.push(trial);
                    return Ok(out);
                }
            }
            if iteration == options.refinement_steps {
                break;
            }
            rows = assign(n, samples, &columns, &decoder, true, &mut out.cost)?;
            let (decoder, _, _) =
                template(n, vocab, &classes, samples, &rows, &columns, &mut out.cost);
            columns = assign(n, samples, &rows, &decoder, false, &mut out.cost)?;
        }
        trial.assignment_sweeps = out.cost.assignment_sweeps - before_sweeps;
        out.trials.push(trial);
    }
    Ok(out)
}
