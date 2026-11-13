use super::*;

impl Policy {
    fn validate(&self) -> Result<()> {
        if !(1..=4096).contains(&self.equations) || self.models > 8 {
            return Err("learner equation or model policy exceeds declared limits".into());
        }
        // Shape-only policy checks also apply to unsupported class counts.
        // These temporary inputs are never emitted as an observation certificate.
        composite::Request {
            n: 2,
            classes: 2,
            system: System {
                variables: 6,
                rows: vec![vec![0; 6]],
                right: vec![0],
            },
            component_policy: self.component.clone(),
            policy: self.composite.clone(),
        }
        .validate()?;
        decoded::Request {
            n: 2,
            vocabulary: 2,
            modulus: 2,
            observations: vec![[0, 0], [1, 1]],
            predictions: vec![0; 4],
            prefix_policy: self.prefix.clone(),
            policy: self.decoded.clone(),
        }
        .validate()?;
        Ok(())
    }
}
impl Request {
    pub fn validate(&self) -> Result<(Partition, u64)> {
        self.policy.validate()?;
        convex_stress_tagged::validate_observations(self.n, self.vocabulary, &self.samples())?;
        let (mut classes, mut class_indices) = (vec![], vec![]);
        let (mut rows, mut columns) = (vec![false; self.n], vec![false; self.n]);
        for &[cell, label] in &self.observations {
            let index = classes.iter().position(|&c| c == label).unwrap_or_else(|| {
                classes.push(label);
                classes.len() - 1
            });
            class_indices.push(index);
            rows[cell % self.n] = true;
            columns[cell / self.n] = true;
        }
        let partition = Partition {
            classes,
            class_indices,
            missing_rows: (0..self.n).filter(|&i| !rows[i]).collect(),
            missing_columns: (0..self.n).filter(|&j| !columns[j]).collect(),
            anchor_row: self.observations[0][0] % self.n,
            retained_observations: self.observations.len().min(self.policy.equations),
        };
        let q = partition.classes.len();
        let mut bound = 4 * 1024 * 1024 * 1024;
        if (2..=257).contains(&q) {
            bound += self.problem(&partition).validate()?;
            // Only the shape enters this forecast. Actual predictions are bound
            // to a constructed model before any real certificate is produced.
            let shape = self.certificate(q, vec![partition.classes[0]; self.n * self.n]);
            bound += self.policy.models as u64
                * (shape.validate()?.1
                    + convex_stress_tagged::model::memory_bound(self.n, q, self.vocabulary)?);
        }
        if bound > 56 * 1024 * 1024 * 1024 {
            return Err("learner code/model/certificate forecast exceeds56GiB".into());
        }
        Ok((partition, bound))
    }
    pub(super) fn samples(&self) -> Vec<Sample> {
        self.observations
            .iter()
            .map(|&[cell, label]| Sample { cell, label })
            .collect()
    }
    pub(super) fn problem(&self, p: &Partition) -> composite::Request {
        let variables = 2 * self.n + p.classes.len();
        let mut rows = vec![];
        for (&[cell, _], &class) in self
            .observations
            .iter()
            .zip(&p.class_indices)
            .take(p.retained_observations)
        {
            let mut row = vec![0; variables];
            row[cell % self.n] = 1;
            row[self.n + cell / self.n] = 1;
            row[2 * self.n + class] = -1;
            rows.push(row);
        }
        // Two actual shift gauges for proposal discovery. Missing axes remain
        // free; decoded identification constructs its own observation-only matrix.
        for coordinate in [p.anchor_row, 2 * self.n] {
            let mut row = vec![0; variables];
            row[coordinate] = 1;
            rows.push(row);
        }
        composite::Request {
            n: self.n,
            classes: p.classes.len(),
            system: System {
                variables,
                right: vec![0; rows.len()],
                rows,
            },
            component_policy: self.policy.component.clone(),
            policy: self.policy.composite.clone(),
        }
    }
    pub(super) fn certificate(&self, q: usize, predictions: Vec<usize>) -> decoded::Request {
        decoded::Request {
            n: self.n,
            vocabulary: self.vocabulary,
            modulus: q as u16,
            observations: self.observations.clone(),
            predictions,
            prefix_policy: self.policy.prefix.clone(),
            policy: self.policy.decoded.clone(),
        }
    }
}

pub(super) fn select(candidates: &[Candidate]) -> Result<Option<usize>> {
    let mut selected: Option<usize> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        let ce = candidate.raw_audit["train_ce"]
            .as_f64()
            .ok_or("missing raw training CE")?;
        if !ce.is_finite() {
            return Err("nonfinite raw training CE".into());
        }
        let better = match selected {
            None => true,
            Some(earlier) => {
                let old = &candidates[earlier];
                let previous = old.raw_audit["train_ce"]
                    .as_f64()
                    .ok_or("missing earlier CE")?;
                ce < previous - 1e-12
                    || ((ce - previous).abs() <= 1e-12
                        && candidate.certificate.certified
                        && !old.certificate.certified)
            }
        };
        if better {
            selected = Some(index);
        }
    }
    Ok(selected)
}
pub(super) fn reason(candidates: &[Candidate]) -> String {
    if candidates.is_empty() {
        "no_admissible_candidate"
    } else if candidates.iter().any(|c| c.certificate.certified) {
        "admitted_with_decoded_certificates"
    } else {
        "admitted_uncertified_proposals"
    }
    .into()
}
