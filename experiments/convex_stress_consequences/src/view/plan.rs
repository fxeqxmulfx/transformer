use super::*;

impl Input {
    pub(super) fn samples(&self) -> Vec<Sample> {
        self.categorical
            .observations
            .iter()
            .map(|&[cell, label]| Sample { cell, label })
            .collect()
    }
    pub(super) fn inner(&self, v: &views::View) -> learner::Request {
        learner::Request {
            n: v.n,
            vocabulary: self.categorical.vocabulary,
            observations: v.observations.clone(),
            policy: self.categorical.policy.clone(),
        }
    }
}
pub fn catalogue(input: &Input) -> Result<Vec<Slot>> {
    if input.max_views > 4 {
        return Err("fixed-view policy exceeds four attempted views".into());
    }
    // Validate every policy even when all actual views will be skipped. These
    // shape-only observations never enter a model or a query certificate.
    learner::Request {
        n: 2,
        vocabulary: 2,
        observations: vec![[0, 0], [1, 1]],
        policy: input.categorical.policy.clone(),
    }
    .validate()?;
    let proposed = views::propose(
        input.categorical.n,
        input.categorical.vocabulary,
        &input.samples(),
        input.minimum_strip_support,
    )?;
    let mut slots = vec![];
    for (index, view) in proposed.into_iter().enumerate() {
        let disposition = if index >= input.max_views {
            Disposition::ViewCap
        } else if view.n < 2 || view.observations.is_empty() {
            Disposition::EmptyInterior
        } else {
            let bound = input.inner(&view).validate().and_then(|(p, bytes)| {
                let mut full = 2 * 1024 * 1024 * 1024 + bytes;
                if (2..=257).contains(&p.classes.len()) {
                    let q = p.classes.len() * if view.constant.is_some() { 3 } else { 1 };
                    full += input.categorical.policy.models as u64
                        * convex_stress_tagged::model::memory_bound(
                            input.categorical.n,
                            q,
                            input.categorical.vocabulary,
                        )?;
                }
                if full > 60 * 1024 * 1024 * 1024 {
                    return Err("one-view model/certificate/replay forecast exceeds60GiB".into());
                }
                Ok(full)
            });
            match bound {
                Ok(buffer_preflight_bytes) => Disposition::Ready {
                    buffer_preflight_bytes,
                },
                Err(error) => Disposition::PreflightRefusal {
                    error: error.to_string(),
                },
            }
        };
        slots.push(Slot { view, disposition });
    }
    Ok(slots)
}
pub(super) fn select(candidates: &[Candidate]) -> Result<Option<usize>> {
    let mut selected: Option<usize> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        let ce = candidate.raw_audit["train_ce"]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or("missing finite original-view CE")?;
        let better = match selected {
            None => true,
            Some(previous) => {
                let old = &candidates[previous];
                let old_ce = old.raw_audit["train_ce"]
                    .as_f64()
                    .ok_or("missing earlier view CE")?;
                ce < old_ce - 1e-12
                    || ((ce - old_ce).abs() <= 1e-12
                        && candidate.scope.full_table_certified_within_family
                        && !old.scope.full_table_certified_within_family)
            }
        };
        if better {
            selected = Some(index);
        }
    }
    Ok(selected)
}
