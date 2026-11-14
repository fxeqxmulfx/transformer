//! Finite feasible segment search, not an exact constrained linear oracle.
//! Ordinary CE supplies every coefficient slope; no held-out label is read.
use crate::*;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub cap: f64,
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
    pub cap_class_visits: usize,
    pub guard_query_visits: usize,
    pub guard_class_visits: usize,
    pub actual_head_terms: usize,
    pub feature_scalar_work_bound: usize,
    pub rejected_proposals: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub start: usize,
    pub count: usize,
    pub vertices: Vec<Option<usize>>,
    pub caps: Vec<f64>,
    pub proposed_gamma: f64,
    pub gamma: f64,
    pub backtracks: usize,
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
    pub initial_step: Step,
    pub initial_ce: f64,
    pub weights: Vec<Vec<f64>>,
    pub canonical_ce: f64,
    pub correct_records: usize,
    pub excluded_records: usize,
    pub retained_epoch: usize,
    pub steps: Vec<Step>,
    pub work: Work,
}

pub(crate) fn update_row(row: &mut [f64], vertex: usize, step: f64) {
    if step == 0. || row[vertex] == 1. {
        return;
    }
    for a in row.iter_mut() {
        *a *= 1. - step;
    }
    row[vertex] += step;
}

fn proposal(
    weights: &[Vec<f64>],
    vertices: &[Option<usize>],
    caps: &[f64],
    gamma: f64,
) -> Vec<Vec<f64>> {
    let mut result = weights.to_vec();
    for (g, row) in result.iter_mut().enumerate() {
        if let Some(j) = vertices[g] {
            update_row(row, j, caps[g] * gamma);
        }
    }
    result
}

struct Search<'a> {
    parent: &'a parent::Model,
    cache: &'a diagnostic::Cache,
    guard: &'a guard::Guard,
    work: Work,
}

impl Search<'_> {
    fn apply(
        &mut self,
        weights: &mut Vec<Vec<f64>>,
        scores: &mut Vec<Vec<f64>>,
        mut step: Step,
        threshold: f64,
    ) -> Result<Step> {
        let mut gamma = step.proposed_gamma;
        for attempt in 0..=BACKTRACKS {
            if gamma == 0. {
                break;
            }
            let next = proposal(weights, &step.vertices, &step.caps, gamma);
            let active: Vec<_> = weights.iter().zip(&next).map(|(a, b)| a != b).collect();
            if !active.iter().any(|x| *x) {
                break;
            }
            let scored = self.cache.evaluate(&next, step.start, step.count, false);
            self.work.score_records += step.count;
            if scored.loss + threshold < step.before_ce {
                if let Some(next_scores) =
                    self.guard
                        .verify(self.parent, &next, scores, &active, &mut self.work)
                {
                    *weights = next;
                    *scores = next_scores;
                    step.gamma = gamma;
                    step.backtracks = attempt;
                    step.after_ce = scored.loss;
                    step.changed_rows = active.iter().filter(|v| **v).count();
                    break;
                }
                self.work.rejected_proposals += 1;
            }
            gamma *= 0.5;
            step.backtracks = attempt + 1;
        }
        step.weights_sha256 = digest(weights)?;
        Ok(step)
    }
}

fn step(
    start: usize,
    count: usize,
    vertices: Vec<Option<usize>>,
    caps: Vec<f64>,
    gamma: f64,
    before: f64,
) -> Step {
    Step {
        start,
        count,
        vertices,
        caps,
        proposed_gamma: gamma,
        gamma: 0.,
        backtracks: 0,
        before_ce: before,
        after_ce: before,
        changed_rows: 0,
        weights_sha256: String::new(),
    }
}

pub(crate) fn select(
    parent: &parent::Model,
    gates: &[usize],
    guard: &guard::Guard,
) -> Result<(Selection, Certificate)> {
    let cache = diagnostic::Cache::new(parent, gates)?;
    let n = cache.rows.len();
    let mut search = Search {
        parent,
        cache: &cache,
        guard,
        work: Work::default(),
    };
    let anchors = cache.anchors();
    let mut weights = vec![anchor(cache.bank_size); GATE_COUNT];
    let old = cache.evaluate(&weights, 0, n, false);
    search.work.score_records += 2 * n;
    if (old.loss - parent.selection().canonical_ce).abs() > 1e-10
        || old.correct != parent.selection().correct_records
    {
        return Err("margin anchor differs from reviewed CM46 diagnostic".into());
    }
    let all_caps = guard.initial_caps(cache.bank_size, &mut search.work);
    let candidates = cache.vertices(&all_caps);
    search.work.score_records += n * cache.bank_size;
    let mut vertices = vec![None; GATE_COUNT];
    let mut caps = vec![0.; GATE_COUNT];
    for g in 0..GATE_COUNT {
        let mut best = anchors[g].canonical_ce;
        for (j, c) in candidates[g].iter().enumerate() {
            if c.records > 0 && c.canonical_ce + 1e-10 < best {
                best = c.canonical_ce;
                vertices[g] = Some(j);
                caps[g] = c.cap;
            }
        }
    }
    let mut scores: Vec<_> = guard.queries.iter().map(|q| q.reference.clone()).collect();
    let initial_step = search.apply(
        &mut weights,
        &mut scores,
        step(0, n, vertices, caps, 1., old.loss),
        1e-10,
    )?;
    let initial = cache.evaluate(&weights, 0, n, false);
    search.work.score_records += n;
    let mut result = Selection {
        anchors,
        candidates,
        initial_step,
        initial_ce: initial.loss,
        weights: weights.clone(),
        canonical_ce: initial.loss,
        correct_records: initial.correct,
        excluded_records: cache.excluded_records,
        retained_epoch: 0,
        steps: vec![],
        work: Work::default(),
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
            search.work.gradient_batches += 1;
            search.work.gradient_records += count;
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
            let caps = guard.caps(&scores, &vertices, &mut search.work);
            let (mut best, mut gamma) = (current.loss, 0.);
            for candidate in LINE_SEARCH.into_iter().skip(1) {
                let proposed = proposal(&weights, &vertices, &caps, candidate);
                let ce = cache.evaluate(&proposed, start, count, false).loss;
                search.work.score_records += count;
                if ce + 1e-12 < best {
                    best = ce;
                    gamma = candidate;
                }
            }
            let update = search.apply(
                &mut weights,
                &mut scores,
                step(start, count, vertices, caps, gamma, current.loss),
                1e-12,
            )?;
            search.work.optimizer_updates += usize::from(update.changed_rows > 0);
            search.work.local_row_updates += update.changed_rows;
            result.steps.push(update);
        }
        let scored = cache.evaluate(&weights, 0, n, false);
        search.work.score_records += n;
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
        return Err("invalid guarded simplex or diagnostic".into());
    }
    let final_scores = guard
        .verify(
            parent,
            &result.weights,
            &scores,
            &[true; GATE_COUNT],
            &mut search.work,
        )
        .ok_or("retained endpoint failed actual constraints")?;
    let certificate = guard.certificate(&final_scores, parent.source().vocabulary)?;
    // Certificate repeats the complete class check and the exact argmax.
    search.work.guard_query_visits += guard.queries.len();
    search.work.guard_class_visits += guard.queries.len() * parent.source().vocabulary;
    result.work = search.work;
    Ok((result, certificate))
}
