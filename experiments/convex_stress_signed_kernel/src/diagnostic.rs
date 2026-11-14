//! CM45 diagnostic at14b8ad6e, extended with CM51 signed kernel rows. The
//! canonical conjugate identity allows signed label masses; absent classes
//! still contribute all of their exponentials to ordinary full-vocabulary CE.
use crate::*;

#[derive(Clone, Serialize)]
pub(crate) struct Row {
    pub labels: Vec<usize>,
    pub target: usize,
    pub logits: Vec<Vec<f64>>,
}
#[derive(Serialize)]
pub(crate) struct Cache {
    pub vocabulary: usize,
    pub rows: Vec<Row>,
    pub excluded_records: usize,
}
pub(crate) struct Evaluation {
    pub loss: f64,
    pub correct: usize,
    pub slopes: Vec<f64>,
}

impl Row {
    pub(crate) fn mixture(&self, weights: &[f64]) -> Vec<f64> {
        let mut z = vec![0.; self.labels.len()];
        for (alpha, bank) in weights.iter().zip(&self.logits) {
            for (value, &raw) in z.iter_mut().zip(bank) {
                *value += alpha * raw;
            }
        }
        z
    }

    pub(crate) fn statistics(&self, z: &[f64], vocabulary: usize) -> (f64, bool, Vec<f64>) {
        let top = z.iter().copied().fold(0., f64::max);
        let mut probabilities: Vec<_> = z.iter().map(|v| (v - top).exp()).collect();
        let total = probabilities.iter().sum::<f64>()
            + (vocabulary - self.labels.len()) as f64 * (-top).exp();
        for p in &mut probabilities {
            *p /= total;
        }
        let mut winner = (0, f64::NEG_INFINITY);
        for (&label, &value) in self.labels.iter().zip(z) {
            if value > winner.1 {
                winner = (label, value);
            }
        }
        if let Some(missing) = (0..vocabulary).find(|c| self.labels.binary_search(c).is_err())
            && (winner.1 < 0. || (winner.1 == 0. && missing < winner.0))
        {
            winner = (missing, 0.);
        }
        (
            total.ln() + top - z[self.target],
            winner.0 == self.labels[self.target],
            probabilities,
        )
    }
}

impl Cache {
    pub(crate) fn new(
        parent: &parent::Model,
        fits: &kernels::Fits,
        work: &mut Work,
    ) -> Result<Self> {
        let source = parent.source();
        let mut excluded_records = 0;
        let mut rows = Vec::with_capacity(parent.records().len());
        let mut templates: Vec<Option<Row>> = vec![None; source.n.pow(2)];
        for record in parent.records() {
            let (block, state) = source.location(record.cell);
            let b = &source.blocks[block];
            let ambiguous = b
                .observed_labels
                .first()
                .is_some_and(|y| b.observed_labels.iter().any(|v| v != y));
            excluded_records += usize::from(ambiguous);
            if let Some(row) = &templates[record.cell] {
                rows.push(row.clone());
                continue;
            }
            if b.observed_labels.get(state) != Some(&record.label) {
                return Err("kernel diagnostic record differs from its observed source".into());
            }
            let mut labels = b.observed_labels.clone();
            labels.push(b.default_label);
            labels.sort_unstable();
            labels.dedup();
            let target = labels
                .binary_search(&record.label)
                .map_err(|_| "absent diagnostic target")?;
            let baseline = baseline::logits(parent, record.cell, &labels);
            let mut logits = vec![baseline.clone()];
            for fit in fits[block].iter().skip(1) {
                let Some(fit) = fit else {
                    logits.push(baseline.clone());
                    continue;
                };
                let weights = fit.excluded(state, work)?;
                let mut z = vec![0.; labels.len()];
                for (state, mass) in weights.into_iter().enumerate() {
                    z[labels.binary_search(&b.observed_labels[state]).unwrap()] +=
                        gated::scale::context::MARGIN * mass;
                }
                logits.push(z);
            }
            let row = Row {
                labels,
                target,
                logits,
            };
            templates[record.cell] = Some(row.clone());
            rows.push(row);
        }
        Ok(Self {
            vocabulary: source.vocabulary,
            rows,
            excluded_records,
        })
    }

    pub(crate) fn vertex(&self, index: usize) -> Evaluation {
        let (mut loss, mut correct) = (0., 0);
        for row in &self.rows {
            let (ce, hit, _) = row.statistics(&row.logits[index], self.vocabulary);
            loss += ce;
            correct += usize::from(hit);
        }
        Evaluation {
            loss: loss / self.rows.len() as f64,
            correct,
            slopes: vec![],
        }
    }

    pub(crate) fn evaluate(
        &self,
        weights: &[f64],
        start: usize,
        count: usize,
        gradient: bool,
    ) -> Evaluation {
        let mut result = Evaluation {
            loss: 0.,
            correct: 0,
            slopes: vec![0.; weights.len()],
        };
        for row in &self.rows[start..start + count] {
            let z = row.mixture(weights);
            let (ce, hit, probabilities) = row.statistics(&z, self.vocabulary);
            result.loss += ce;
            result.correct += usize::from(hit);
            if gradient {
                for (slope, bank) in result.slopes.iter_mut().zip(&row.logits) {
                    *slope += probabilities
                        .iter()
                        .zip(bank)
                        .map(|(p, z)| p * z)
                        .sum::<f64>()
                        - bank[row.target];
                }
            }
        }
        result.loss /= count as f64;
        for slope in &mut result.slopes {
            *slope /= count as f64;
        }
        result
    }

    pub(crate) fn line_scores(
        &self,
        weights: &[f64],
        vertex: usize,
        start: usize,
        count: usize,
    ) -> [f64; 6] {
        let mut losses = [0.; 6];
        for row in &self.rows[start..start + count] {
            let z = row.mixture(weights);
            for (loss, gamma) in losses.iter_mut().zip(LINE_SEARCH) {
                let line: Vec<_> = z
                    .iter()
                    .zip(&row.logits[vertex])
                    .map(|(a, b)| (1. - gamma) * a + gamma * b)
                    .collect();
                *loss += row.statistics(&line, self.vocabulary).0;
            }
        }
        losses.map(|loss| loss / count as f64)
    }
}
