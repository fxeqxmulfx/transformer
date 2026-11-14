//! CM46 ordinary CE on a product of local simplexes. Fixed one-hot gates make
//! initialization separable. Conditional gradients keep absent gates unchanged.
use crate::*;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub canonical_ce: f64,
    pub correct_records: usize,
    pub records: usize,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_records: usize,
    pub score_records: usize,
    pub optimizer_updates: usize,
    pub local_row_updates: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub start: usize,
    pub count: usize,
    pub vertices: Vec<Option<usize>>,
    pub gamma: f64,
    pub before_ce: f64,
    pub after_ce: f64,
    pub changed_rows: usize,
    pub weights_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub anchors: Vec<Candidate>,
    pub candidates: Vec<Vec<Candidate>>,
    pub initial_vertices: Vec<Option<usize>>,
    pub initial_ce: f64,
    pub weights: Vec<Vec<f64>>,
    pub canonical_ce: f64,
    pub correct_records: usize,
    pub excluded_records: usize,
    pub retained_epoch: usize,
    pub steps: Vec<Step>,
    pub work: Work,
}

pub(crate) fn update(weights: &mut [Vec<f64>], vertices: &[Option<usize>], gamma: f64) -> usize {
    let mut changed = 0;
    if gamma == 0. {
        return changed;
    }
    for (row, vertex) in weights.iter_mut().zip(vertices) {
        if let Some(vertex) = *vertex {
            if row[vertex] == 1. {
                continue;
            }
            for a in row.iter_mut() {
                *a *= 1. - gamma;
            }
            row[vertex] += gamma;
            changed += 1;
        }
    }
    changed
}

pub(crate) fn select(parent: &parent::Model, gates: &[usize]) -> Result<Selection> {
    let cache = diagnostic::Cache::new(parent, gates)?;
    let n = cache.rows.len();
    let anchors = cache.anchors();
    let candidates = cache.vertices();
    let old_ce = anchors
        .iter()
        .map(|c| c.canonical_ce * c.records as f64)
        .sum::<f64>()
        / n as f64;
    if (old_ce - parent.selection().canonical_ce).abs() > 1e-10
        || anchors.iter().map(|c| c.correct_records).sum::<usize>()
            != parent.selection().correct_records
    {
        return Err("local anchor differs from reviewed CM45 diagnostic".into());
    }
    let mut weights = vec![parent.selection().weights.clone(); cache.groups];
    let mut initial_vertices = vec![None; cache.groups];
    for g in 0..cache.groups {
        let mut best = anchors[g].canonical_ce;
        for (j, candidate) in candidates[g].iter().enumerate() {
            if candidate.records > 0 && candidate.canonical_ce + 1e-10 < best {
                best = candidate.canonical_ce;
                initial_vertices[g] = Some(j);
            }
        }
        if let Some(j) = initial_vertices[g] {
            weights[g].fill(0.);
            weights[g][j] = 1.;
        }
    }
    let initial = cache.evaluate(&weights, 0, n, false);
    let mut result = Selection {
        anchors,
        candidates,
        initial_vertices,
        initial_ce: initial.loss,
        weights: weights.clone(),
        canonical_ce: initial.loss,
        correct_records: initial.correct,
        excluded_records: cache.excluded_records,
        retained_epoch: 0,
        steps: vec![],
        work: Work {
            score_records: n * (cache.bank_size + 2),
            ..Work::default()
        },
    };
    let endpoint = parent.endpoint();
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
            let vertices: Vec<_> = current
                .slopes
                .iter()
                .zip(&current.active)
                .map(|(slopes, active)| {
                    active.then(|| {
                        slopes
                            .iter()
                            .enumerate()
                            .min_by(|(i, a), (j, b)| a.total_cmp(b).then(i.cmp(j)))
                            .unwrap()
                            .0
                    })
                })
                .collect();
            let losses = cache.line_scores(&weights, &vertices, start, count);
            result.work.score_records += LINE_SEARCH.len() * count;
            let mut chosen = 0;
            for i in 1..LINE_SEARCH.len() {
                if losses[i] + 1e-12 < losses[chosen] {
                    chosen = i;
                }
            }
            let gamma = LINE_SEARCH[chosen];
            let changed_rows = update(&mut weights, &vertices, gamma);
            result.work.optimizer_updates += usize::from(changed_rows > 0);
            result.work.local_row_updates += changed_rows;
            result.steps.push(Step {
                start,
                count,
                vertices,
                gamma,
                before_ce: current.loss,
                after_ce: losses[chosen],
                changed_rows,
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
    if result.weights.iter().any(|row| {
        row.iter().any(|a| !a.is_finite() || *a < 0.)
            || (row.iter().sum::<f64>() - 1.).abs() > 1e-12
    }) || !result.canonical_ce.is_finite()
    {
        return Err("invalid local simplex or diagnostic".into());
    }
    Ok(result)
}
