//! Ordinary diagnostic CE on the simplex. Conditional-gradient directions
//! use CM45's complete coefficient slopes; line candidates keep unit mass.
//! This finite algorithm claims no convergence rate or held-out correctness.
use crate::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub canonical_ce: f64,
    pub correct_records: usize,
    pub records: usize,
    pub excluded_state_records: usize,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_records: usize,
    pub score_records: usize,
    pub optimizer_updates: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub start: usize,
    pub count: usize,
    pub vertex: usize,
    pub gamma: f64,
    pub before_ce: f64,
    pub after_ce: f64,
    pub weights_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub candidates: Vec<Candidate>,
    pub initial_vertex: usize,
    pub weights: Vec<f64>,
    pub canonical_ce: f64,
    pub correct_records: usize,
    pub retained_epoch: usize,
    pub steps: Vec<Step>,
    pub work: Work,
}

pub(crate) fn select(
    parent: &parent::Model,
    bank: &[Context],
    endpoint: &Endpoint,
) -> Result<Selection> {
    let cache = diagnostic::Cache::new(parent, bank)?;
    let n = cache.rows.len();
    let candidates: Vec<_> = bank
        .iter()
        .enumerate()
        .map(|(i, context)| {
            let result = cache.vertex(i);
            Candidate {
                canonical_ce: result.loss,
                correct_records: result.correct,
                records: n,
                excluded_state_records: if context.neighbors == 0 {
                    0
                } else {
                    cache.excluded_records
                },
            }
        })
        .collect();
    let old = parent
        .selection()
        .candidates
        .iter()
        .find(|c| c.scale == parent.selection().selected_scale)
        .ok_or("missing parent diagnostic")?;
    if (candidates[0].canonical_ce - old.canonical_ce).abs() > 1e-10
        || candidates[0].correct_records != old.correct_records
        || candidates[0].excluded_state_records != old.excluded_state_records
    {
        return Err("baseline bank vertex differs from reviewed diagnostic".into());
    }
    let mut initial_vertex = 0;
    for i in 1..bank.len() {
        if candidates[i].canonical_ce + 1e-10 < candidates[initial_vertex].canonical_ce {
            initial_vertex = i;
        }
    }
    let mut weights = vec![0.; bank.len()];
    weights[initial_vertex] = 1.;
    let mut result = Selection {
        canonical_ce: candidates[initial_vertex].canonical_ce,
        correct_records: candidates[initial_vertex].correct_records,
        candidates,
        initial_vertex,
        weights: weights.clone(),
        retained_epoch: 0,
        steps: vec![],
        work: Work {
            score_records: n * bank.len(),
            ..Work::default()
        },
    };
    let epochs = match endpoint.mode {
        Mode::Full => FULL_STEPS,
        Mode::Mini => MINI_EPOCHS,
        Mode::Forward => 0,
    };
    let batch = if endpoint.mode == Mode::Mini {
        endpoint.mixture_batch
    } else {
        n
    };
    for epoch in 0..epochs {
        for start in (0..n).step_by(batch) {
            let count = batch.min(n - start);
            let current = cache.evaluate(&weights, start, count, true);
            result.work.gradient_batches += 1;
            result.work.gradient_records += count;
            let vertex = current
                .slopes
                .iter()
                .enumerate()
                .min_by(|(i, a), (j, b)| a.total_cmp(b).then(i.cmp(j)))
                .unwrap()
                .0;
            let losses = cache.line_scores(&weights, vertex, start, count);
            result.work.score_records += LINE_SEARCH.len() * count;
            let mut selected = 0;
            for i in 1..LINE_SEARCH.len() {
                if losses[i] + 1e-12 < losses[selected] {
                    selected = i;
                }
            }
            let gamma = LINE_SEARCH[selected];
            if gamma > 0. {
                for a in &mut weights {
                    *a *= 1. - gamma;
                }
                weights[vertex] += gamma;
                result.work.optimizer_updates += 1;
            }
            result.steps.push(Step {
                start,
                count,
                vertex,
                gamma,
                before_ce: current.loss,
                after_ce: losses[selected],
                weights_sha256: digest(&weights)?,
            });
        }
        let scored = cache.evaluate(&weights, 0, n, false);
        result.work.score_records += n;
        if scored.loss + 1e-10 < result.canonical_ce {
            result.canonical_ce = scored.loss;
            result.correct_records = scored.correct;
            result.weights.clone_from(&weights);
            result.retained_epoch = epoch + 1;
        }
    }
    if result.weights.iter().any(|a| !a.is_finite() || *a < 0.)
        || (result.weights.iter().sum::<f64>() - 1.).abs() > 1e-12
        || !result.canonical_ce.is_finite()
    {
        return Err("invalid learned simplex or diagnostic".into());
    }
    Ok(result)
}
