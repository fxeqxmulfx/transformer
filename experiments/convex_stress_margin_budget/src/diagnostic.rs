//! CM47 diagnostic_ce_convex permits this fixed source-excluded loss bank.
//! Ported from5e89c616; bank0 reconstructs the reviewed CM46 diagnostic exactly.
use crate::*;

pub(crate) struct Row {
    pub gate: usize,
    pub anchor: Vec<f64>,
    pub labels: Vec<usize>,
    pub target: usize,
    pub logits: Vec<Vec<f64>>,
}
pub(crate) struct Cache {
    pub vocabulary: usize,
    pub rows: Vec<Row>,
    pub excluded_records: usize,
    pub groups: usize,
    pub bank_size: usize,
}
pub(crate) struct Evaluation {
    pub loss: f64,
    pub correct: usize,
    pub slopes: Vec<Vec<f64>>,
    pub active: Vec<bool>,
}

impl Row {
    pub(crate) fn mixture(&self, weights: &[f64]) -> Vec<f64> {
        let mass = model::reference_mass(weights);
        let mut z: Vec<_> = self.anchor.iter().map(|v| mass * v).collect();
        for (alpha, bank) in weights[2..].iter().zip(&self.logits[2..]) {
            for (value, &raw) in z.iter_mut().zip(bank) {
                *value += RADIUS * alpha * raw;
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
    pub(crate) fn new(parent: &parent::Model, gates: &[usize]) -> Result<Self> {
        let source = parent.source();
        let bank = parent.bank();
        let mut excluded_records = 0;
        let mut rows = Vec::with_capacity(parent.records().len());
        for record in parent.records() {
            let (block, _) = source.location(record.cell);
            let b = &source.blocks[block];
            let ambiguous = bank::ambiguous(b);
            excluded_records += usize::from(ambiguous);
            let sorted = bank::distances(source, record.cell);
            let mut labels = b.observed_labels.clone();
            labels.push(b.default_label);
            labels.sort_unstable();
            labels.dedup();
            let target = labels
                .binary_search(&record.label)
                .map_err(|_| "absent diagnostic target")?;
            let mut logits = Vec::with_capacity(bank.len());
            for context in bank {
                let weights = if ambiguous {
                    bank::weights(context, &sorted)
                } else {
                    vec![]
                };
                let mut z = vec![0.; labels.len()];
                if weights.is_empty() {
                    z[labels.binary_search(&b.default_label).unwrap()] = scale::context::MARGIN;
                } else {
                    for w in weights {
                        z[labels.binary_search(&b.observed_labels[w.state]).unwrap()] +=
                            scale::context::MARGIN * w.mass;
                    }
                }
                logits.push(z);
            }
            let mut anchor = vec![0.; labels.len()];
            for (&a, z) in parent.parent().selection().weights.iter().zip(&logits) {
                for (value, &raw) in anchor.iter_mut().zip(z) {
                    *value += a * raw;
                }
            }
            for value in &mut anchor {
                *value *= 1. - parent.radius();
            }
            for (&a, z) in parent.selection().weights[parent.gates()[record.cell]]
                .iter()
                .zip(&logits)
            {
                for (value, &raw) in anchor.iter_mut().zip(z) {
                    *value += parent.radius() * a * raw;
                }
            }
            logits.insert(0, anchor.iter().map(|x| REFERENCE_SCALE * x).collect());
            logits.insert(0, anchor.clone());
            rows.push(Row {
                gate: gates[record.cell],
                anchor,
                labels,
                target,
                logits,
            });
        }
        Ok(Self {
            vocabulary: source.vocabulary,
            rows,
            excluded_records,
            groups: GATE_COUNT,
            bank_size: bank.len() + 2,
        })
    }

    /// Score every anchored vertex by gate without collapsing repeated records.
    pub(crate) fn vertices(&self, caps: &[Vec<f64>]) -> Vec<Vec<crate::Candidate>> {
        let mut result = vec![vec![crate::Candidate::default(); self.bank_size]; self.groups];
        let weights: Vec<Vec<_>> = caps
            .iter()
            .map(|group| {
                group
                    .iter()
                    .enumerate()
                    .map(|(j, &cap)| {
                        let mut a = anchor(self.bank_size);
                        selection::update_row(&mut a, j, cap);
                        a
                    })
                    .collect()
            })
            .collect();
        for row in &self.rows {
            for (j, candidate) in result[row.gate].iter_mut().enumerate() {
                candidate.cap = caps[row.gate][j];
                let z = row.mixture(&weights[row.gate][j]);
                let (ce, hit, _) = row.statistics(&z, self.vocabulary);
                candidate.canonical_ce += ce;
                candidate.correct_records += usize::from(hit);
                candidate.records += 1;
            }
        }
        for group in &mut result {
            for c in group {
                if c.records > 0 {
                    c.canonical_ce /= c.records as f64;
                }
            }
        }
        result
    }

    pub(crate) fn anchors(&self) -> Vec<crate::Candidate> {
        let mut result = vec![crate::Candidate::default(); self.groups];
        for row in &self.rows {
            let (ce, hit, _) = row.statistics(&row.anchor, self.vocabulary);
            let c = &mut result[row.gate];
            c.canonical_ce += ce;
            c.correct_records += usize::from(hit);
            c.records += 1;
        }
        for c in &mut result {
            if c.records > 0 {
                c.canonical_ce /= c.records as f64;
            }
        }
        result
    }

    pub(crate) fn evaluate(
        &self,
        weights: &[Vec<f64>],
        start: usize,
        count: usize,
        gradient: bool,
    ) -> Evaluation {
        let mut result = Evaluation {
            loss: 0.,
            correct: 0,
            slopes: if gradient {
                vec![vec![0.; self.bank_size]; self.groups]
            } else {
                vec![]
            },
            active: vec![false; self.groups],
        };
        for row in &self.rows[start..start + count] {
            let z = row.mixture(&weights[row.gate]);
            result.active[row.gate] = true;
            let (ce, hit, probabilities) = row.statistics(&z, self.vocabulary);
            result.loss += ce;
            result.correct += usize::from(hit);
            if gradient {
                for (slope, bank) in result.slopes[row.gate].iter_mut().zip(&row.logits) {
                    *slope += RADIUS
                        * (probabilities
                            .iter()
                            .zip(bank)
                            .map(|(p, z)| p * z)
                            .sum::<f64>()
                            - bank[row.target]);
                }
            }
        }
        result.loss /= count as f64;
        for slope in result.slopes.iter_mut().flatten() {
            *slope /= count as f64;
        }
        result
    }
}
