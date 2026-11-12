//! CM21-to-CM15 categorical proposal bridge. Complex label sketches provide
//! estimated centered subspaces; their exactness is not assumed by the code.
//! Extraction may refuse. Right phases always use the same original sketch.
use crate::{
    C, Model, Result, Sample, audit, matrix,
    random::Random,
    solver::{self, Extraction, Options},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchOptions {
    pub code_trials: usize,
    pub ranks: Vec<usize>,
    pub seed: u32,
    pub extraction: Options,
    pub total_iterations: usize,
    pub model_audits: usize,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            code_trials: 4,
            ranks: vec![1, 2, 3, 4],
            seed: 73,
            extraction: Options::default(),
            total_iterations: 16_384,
            model_audits: 4,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Input {
    pub n: usize,
    pub vocab: usize,
    /// Original observed categorical cell and label. No validation/test data.
    pub observations: Vec<[usize; 2]>,
    pub options: SearchOptions,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cost {
    pub sketches: usize,
    pub sketch_observations: usize,
    pub centering_cells: usize,
    pub gram_products: usize,
    pub subspace_eigensolves: usize,
    pub admm_iterations: usize,
    /// Actual iterations for completed calls; the full reserved allowance on errors.
    pub iteration_budget_used: usize,
    pub extracted_phase_products: usize,
    pub decoder_observations: usize,
    pub model_audits: usize,
    pub latent_frequency_audit_allowance: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub trial: usize,
    pub rank: usize,
    pub reason: String,
    pub features: Option<Vec<C>>,
    pub extraction: Option<Extraction>,
    pub raw_audit: Option<Value>,
    pub model: Option<Model>,
    pub model_audit: Option<Value>,
    pub train_ce: Option<f64>,
    pub iterations_charged: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub format: u32,
    pub input: Input,
    pub buffer_bound_bytes: u64,
    pub attempts: Vec<Attempt>,
    pub selected: Option<usize>,
    pub cost: Cost,
}

pub fn validate(input: &Input) -> Result<Vec<Sample>> {
    crate::memory_bound(input.n, 1)?;
    let o = &input.options;
    o.extraction.validate()?;
    if input.observations.len() > input.n * input.n {
        return Err("too many observations".into());
    }
    if !(1..=16).contains(&o.code_trials)
        || o.ranks.is_empty()
        || o.ranks.len() > 8
        || o.ranks
            .iter()
            .any(|&r| !(1..=8).contains(&r) || r >= input.n)
        || o.ranks
            .iter()
            .enumerate()
            .any(|(i, r)| o.ranks[..i].contains(r))
        || !(2..=131_072).contains(&o.total_iterations)
        || !(1..=32).contains(&o.model_audits)
    {
        return Err("invalid bounded subspace proposal policy".into());
    }
    let samples: Vec<_> = input
        .observations
        .iter()
        .map(|&[cell, label]| Sample { cell, label })
        .collect();
    convex_stress_decode::validate_observations(input.n, input.vocab, &samples)?;
    Ok(samples)
}

fn sketch(
    n: usize,
    vocab: usize,
    samples: &[Sample],
    classes: &[usize],
    rng: &mut Random,
) -> Vec<C> {
    let mut codes = vec![C::default(); vocab];
    for &c in classes {
        codes[c] = C::new(rng.gauss(0., 1.), rng.gauss(0., 1.));
    }
    let mean = samples.iter().map(|s| codes[s.label]).sum::<C>() / samples.len() as f64;
    let mut out = vec![C::default(); n * n];
    for s in samples {
        out[s.cell] = codes[s.label] - mean;
    }
    let mut rows = vec![C::default(); n];
    let mut columns = rows.clone();
    for j in 0..n {
        for i in 0..n {
            rows[i] += out[i + n * j] / n as f64;
            columns[j] += out[i + n * j] / n as f64;
        }
    }
    let grand = rows.iter().sum::<C>() / n as f64;
    for j in 0..n {
        for i in 0..n {
            out[i + n * j] += grand - rows[i] - columns[j];
        }
    }
    out
}
fn left_gram(s: &[C], n: usize) -> Vec<C> {
    let mut out = vec![C::default(); n * n];
    for j in 0..n {
        for i in 0..n {
            out[i + n * j] = (0..n).map(|k| s[i + n * k] * s[j + n * k].conj()).sum();
        }
    }
    out
}
fn clock(x: &[C]) -> Option<Vec<usize>> {
    if x.iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.norm() < 1e-12)
    {
        return None;
    }
    let phase: Vec<_> = x.iter().map(|z| *z / z.norm()).collect();
    if (0..x.len()).any(|i| (0..i).any(|j| (phase[i] - phase[j]).norm() < 1e-7)) {
        return None;
    }
    let mut order: Vec<_> = (0..x.len()).collect();
    order.sort_by(|&i, &j| x[i].arg().total_cmp(&x[j].arg()).then(i.cmp(&j)));
    let mut ranks = vec![0; x.len()];
    for (rank, &i) in order.iter().enumerate() {
        ranks[i] = rank;
    }
    Some(ranks)
}

fn decoder(
    n: usize,
    vocab: usize,
    samples: &[Sample],
    classes: &[usize],
    rows: &[usize],
    cols: &[usize],
) -> Option<Vec<usize>> {
    let mut counts = vec![0usize; n * vocab];
    for s in samples {
        counts[((rows[s.cell % n] + cols[s.cell / n]) % n) * vocab + s.label] += 1;
    }
    let mut out = vec![0; n];
    for (phase, value) in out.iter_mut().enumerate() {
        *value = classes[0];
        for &c in &classes[1..] {
            if counts[phase * vocab + c] > counts[phase * vocab + *value] {
                *value = c;
            }
        }
        if counts[phase * vocab + *value] == 0 {
            return None;
        }
    }
    Some(out)
}

pub fn propose(input: Input) -> Result<Proposal> {
    let samples = validate(&input)?;
    let (n, vocab) = (input.n, input.vocab);
    let mut classes = vec![];
    for s in &samples {
        if !classes.contains(&s.label) {
            classes.push(s.label);
        }
    }
    let o = input.options.clone();
    let attempts = o.code_trials * o.ranks.len();
    let bound = 256 * 1024 * 1024
        + 256 * o.total_iterations as u64
        + attempts as u64 * (32 * (n as u64).pow(2) + 256 * n as u64);
    let mut out = Proposal {
        format: 1,
        input,
        buffer_bound_bytes: bound,
        attempts: vec![],
        selected: None,
        cost: Cost::default(),
    };
    let mut rng = Random::new(o.seed);
    for trial in 0..o.code_trials {
        let available = o.total_iterations - out.cost.iteration_budget_used;
        let data = if available >= 2 {
            let s = sketch(n, vocab, &samples, &classes, &mut rng);
            out.cost.sketches += 1;
            out.cost.sketch_observations += 2 * samples.len();
            out.cost.centering_cells += 2 * n * n;
            out.cost.gram_products += n * n * n;
            out.cost.subspace_eigensolves += 1;
            Some(matrix::eigen(&left_gram(&s, n), n).map(|e| (s, e)))
        } else {
            None
        };
        for &rank in &o.ranks {
            let mut attempt = Attempt {
                trial,
                rank,
                reason: "iteration_budget_exhausted".into(),
                features: None,
                extraction: None,
                raw_audit: None,
                model: None,
                model_audit: None,
                train_ce: None,
                iterations_charged: 0,
            };
            let remaining = o.total_iterations - out.cost.iteration_budget_used;
            if remaining < 2 {
                out.attempts.push(attempt);
                continue;
            }
            let (s, e) = match &data {
                Some(Ok(value)) => value,
                Some(Err(error)) => {
                    attempt.reason = format!("subspace_error: {error}");
                    out.attempts.push(attempt);
                    continue;
                }
                None => {
                    out.attempts.push(attempt);
                    continue;
                }
            };
            if e.values[n - rank] <= 1e-12 * e.values[n - 1].max(1.) {
                attempt.reason = "subspace_rank_deficient".into();
                out.attempts.push(attempt);
                continue;
            }
            let mut b = vec![C::default(); n * rank];
            for j in 0..rank {
                for i in 0..n {
                    b[i + n * j] = e.vectors[i + n * (n - 1 - j)] * (n as f64).sqrt();
                }
            }
            attempt.features = Some(b.clone());
            let options = Options {
                iterations: o.extraction.iterations.min(remaining / 2),
                ..o.extraction
            };
            let extraction = match solver::extract(&b, n, rank, [0, 1], options) {
                Ok(e) => e,
                Err(e) => {
                    attempt.iterations_charged = 2 * options.iterations;
                    out.cost.iteration_budget_used += attempt.iterations_charged;
                    attempt.reason = format!("extraction_error: {e}");
                    out.attempts.push(attempt);
                    continue;
                }
            };
            attempt.iterations_charged = extraction
                .stages
                .iter()
                .map(|s| s.trace.len())
                .sum::<usize>();
            out.cost.admm_iterations += attempt.iterations_charged;
            out.cost.iteration_budget_used += attempt.iterations_charged;
            match audit::endpoint(&b, &extraction) {
                Ok(audit) => attempt.raw_audit = Some(audit),
                Err(error) => {
                    attempt.reason = format!("raw_density_audit_failed: {error}");
                    attempt.extraction = Some(extraction);
                    out.attempts.push(attempt);
                    continue;
                }
            }
            attempt.reason = extraction.reason.clone();
            let left = extraction.phase_column.clone();
            attempt.extraction = Some(extraction);
            if let Some(left) = left {
                let right: Vec<_> = (0..n)
                    .map(|j| {
                        (0..n)
                            .map(|i| s[i + n * j].conj() * left[i])
                            .sum::<C>()
                            .conj()
                    })
                    .collect();
                out.cost.extracted_phase_products += n * n;
                if let (Some(rows), Some(cols)) = (clock(&left), clock(&right)) {
                    out.cost.decoder_observations += samples.len();
                    if let Some(d) = decoder(n, vocab, &samples, &classes, &rows, &cols) {
                        if out.cost.model_audits < o.model_audits {
                            let model = Model::construct(vocab, rows, cols, d)?;
                            out.cost.model_audits += 1;
                            out.cost.latent_frequency_audit_allowance += n;
                            let audit = match model.audit(&samples) {
                                Ok(audit) => audit,
                                Err(error) => {
                                    attempt.model = Some(model);
                                    attempt.reason = format!("raw_latent_audit_failed: {error}");
                                    out.attempts.push(attempt);
                                    continue;
                                }
                            };
                            let ce = audit["train_ce"].as_f64().ok_or("missing raw model CE")?;
                            attempt.reason = "audited_phase_candidate".into();
                            attempt.train_ce = Some(ce);
                            attempt.model = Some(model);
                            attempt.model_audit = Some(audit);
                            if out
                                .selected
                                .is_none_or(|i| ce + 1e-12 < out.attempts[i].train_ce.unwrap())
                            {
                                out.selected = Some(out.attempts.len());
                            }
                        } else {
                            attempt.reason = "model_audit_budget_exhausted".into();
                        }
                    } else {
                        attempt.reason = "uncovered_decoder_state".into();
                    }
                } else {
                    attempt.reason = "phase_clock_not_injective".into();
                }
            }
            out.attempts.push(attempt);
        }
    }
    Ok(out)
}

pub fn replay(original: &Input, saved: &Proposal) -> Result<()> {
    if saved.format != 1 || &saved.input != original {
        return Err("proposal input mismatch".into());
    }
    if propose(original.clone())? != *saved {
        return Err("proposal replay mismatch".into());
    }
    Ok(())
}
