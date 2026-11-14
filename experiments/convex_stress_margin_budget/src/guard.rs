//! CM47 constraints on actual raw scores. Cached affine proposals are only
//! proposals: each accepted step passes recomputed complete complex products.
use crate::*;

/// Exact metric convention from convex_stress/src/channel.rs, including signed zero.
pub fn argmax(z: &[f64]) -> usize {
    (0..z.len())
        .max_by(|&a, &b| z[a].total_cmp(&z[b]).then(b.cmp(&a)))
        .unwrap()
}

/// No comparison tolerance: even a subnormal raw decision flip is rejected.
pub fn margins_hold(reference: &[f64], actual: &[f64]) -> bool {
    if reference.is_empty()
        || actual.len() != reference.len()
        || reference.iter().chain(actual).any(|v| !v.is_finite())
    {
        return false;
    }
    let y = argmax(reference);
    let stable = argmax(actual) == y;
    let margins = (0..actual.len()).fold(true, |pass, c| {
        let old = reference[y] - reference[c];
        let new = actual[y] - actual[c];
        pass & (old.is_finite() && new.is_finite() && new >= RETENTION * old)
    });
    stable && margins
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Certificate {
    pub guarded_cells: usize,
    pub classes: usize,
    pub reference_scores_sha256: String,
    pub final_scores_sha256: String,
    pub minimum_slack: f64,
    pub raw_decisions_unchanged: bool,
}

pub(crate) struct Query {
    pub cell: usize,
    pub gate: usize,
    pub reference: Vec<f64>,
    pub winner: usize,
    pub maps: Vec<Vec<Weight>>,
    pub logits: Vec<Vec<f64>>,
}
pub(crate) struct Guard {
    pub queries: Vec<Query>,
}

/// Maximum affine segment fraction. Native verification still checks its endpoint.
pub(crate) fn cap(reference: &[f64], current: &[f64], end: &[f64], y: usize) -> f64 {
    let mut result: f64 = 1.;
    for c in 0..reference.len() {
        let now = current[y] - current[c];
        let slope = end[y] - end[c] - now;
        if slope < 0. {
            result =
                result.min(((now - RETENTION * (reference[y] - reference[c])) / -slope).max(0.));
        }
    }
    result
}

impl Guard {
    pub(crate) fn new(parent: &parent::Model, gates: &[usize]) -> Result<Self> {
        let source = parent.source();
        let mut state_cache = vec![None; source.blocks.len()];
        let mut queries = Vec::new();
        for (cell, &gate) in gates.iter().enumerate() {
            if parent
                .parent()
                .parent()
                .context()
                .neighbors(cell)?
                .is_none()
            {
                continue;
            }
            let block = source.location(cell).0;
            let states = state_cache[block].get_or_insert_with(|| {
                (0..source.blocks[block].states())
                    .map(|s| source.logits_state(block, s))
                    .collect::<Vec<_>>()
            });
            let reference = parent.logits_cell(cell)?;
            if reference.iter().any(|z| !z.is_finite()) {
                return Err("nonfinite actual reference".into());
            }
            let sorted = bank::distances(source, cell);
            let maps: Vec<_> = parent
                .bank()
                .iter()
                .map(|c| bank::weights(c, &sorted))
                .collect();
            let mut logits = vec![
                reference.clone(),
                reference.iter().map(|z| REFERENCE_SCALE * z).collect(),
            ];
            for map in &maps {
                let mut z = vec![0.; source.vocabulary];
                for w in map {
                    for (a, v) in z.iter_mut().zip(&states[w.state]) {
                        *a += w.mass * v;
                    }
                }
                logits.push(z);
            }
            if logits.iter().flatten().any(|z| !z.is_finite()) {
                return Err("nonfinite bank product".into());
            }
            queries.push(Query {
                cell,
                gate,
                winner: argmax(&reference),
                reference,
                maps,
                logits,
            });
        }
        Ok(Self { queries })
    }

    fn vertex(q: &Query, j: usize) -> Vec<f64> {
        if j == 0 {
            return q.reference.clone();
        }
        q.reference
            .iter()
            .zip(&q.logits[j])
            .map(|(a, b)| (1. - RADIUS) * a + RADIUS * b)
            .collect()
    }

    pub(crate) fn initial_caps(&self, j: usize, work: &mut Work) -> Vec<Vec<f64>> {
        let mut caps: Vec<Vec<f64>> = vec![vec![1.; j]; GATE_COUNT];
        for q in &self.queries {
            for (vertex, value) in caps[q.gate].iter_mut().enumerate() {
                *value = value.min(cap(
                    &q.reference,
                    &q.reference,
                    &Self::vertex(q, vertex),
                    q.winner,
                ));
                work.cap_class_visits += q.reference.len();
            }
        }
        caps
    }

    pub(crate) fn caps(
        &self,
        current: &[Vec<f64>],
        vertices: &[Option<usize>],
        work: &mut Work,
    ) -> Vec<f64> {
        let mut result: Vec<f64> = vertices
            .iter()
            .map(|v| if v.is_some() { 1. } else { 0. })
            .collect();
        for (q, z) in self.queries.iter().zip(current) {
            if let Some(j) = vertices[q.gate] {
                result[q.gate] =
                    result[q.gate].min(cap(&q.reference, z, &Self::vertex(q, j), q.winner));
                work.cap_class_visits += z.len();
            }
        }
        result
    }

    /// Returns a new full score state only if every changed gate passes.
    pub(crate) fn verify(
        &self,
        parent: &parent::Model,
        weights: &[Vec<f64>],
        current: &[Vec<f64>],
        active: &[bool],
        work: &mut Work,
    ) -> Option<Vec<Vec<f64>>> {
        let mut next = current.to_vec();
        for (q, z) in self.queries.iter().zip(&mut next) {
            if !active[q.gate] {
                continue;
            }
            if model::is_anchor(&weights[q.gate]) {
                z.clone_from(&q.reference);
            } else {
                let row = model::row(parent.source(), q.cell, &weights[q.gate], &q.maps);
                *z = model::logits(parent.source(), q.cell, &row, &q.reference);
                let s = row.residual.len();
                if row.residual.iter().any(|p| p.re != 0. || p.im != 0.) {
                    work.actual_head_terms += s * z.len();
                }
                work.feature_scalar_work_bound += weights[q.gate].len() * 17 + 8 * s * s;
            }
            work.guard_query_visits += 1;
            work.guard_class_visits += z.len();
            if !margins_hold(&q.reference, z) {
                return None;
            }
        }
        Some(next)
    }

    pub(crate) fn certificate(
        &self,
        scores: &[Vec<f64>],
        vocabulary: usize,
    ) -> Result<Certificate> {
        let mut minimum_slack = f64::INFINITY;
        for (q, z) in self.queries.iter().zip(scores) {
            if !margins_hold(&q.reference, z) {
                return Err("final margin certificate failed".into());
            }
            for c in 0..vocabulary {
                if c == q.winner {
                    continue;
                }
                minimum_slack = minimum_slack
                    .min(z[q.winner] - z[c] - RETENTION * (q.reference[q.winner] - q.reference[c]));
            }
        }
        if self.queries.is_empty() {
            minimum_slack = 0.;
        }
        Ok(Certificate {
            guarded_cells: self.queries.len(),
            classes: vocabulary,
            reference_scores_sha256: digest(
                &self
                    .queries
                    .iter()
                    .map(|q| (q.cell, &q.reference))
                    .collect::<Vec<_>>(),
            )?,
            final_scores_sha256: digest(&scores)?,
            minimum_slack,
            raw_decisions_unchanged: true,
        })
    }
}
