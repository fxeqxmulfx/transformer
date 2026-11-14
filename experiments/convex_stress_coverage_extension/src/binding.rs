//! CM39 binds the actual CM34/35 observed equations, complete codes, Fourier
//! tensors and ordered original records. Numerical audits verify feasible
//! coordinates; an importing adapter must additionally bind saved CE histories.
use crate::*;
use std::collections::{BTreeMap, BTreeSet};

fn dictionary(model: &kernel::Model, observations: &[[usize; 2]]) -> Result<()> {
    let mut actual: Vec<_> = model
        .blocks
        .iter()
        .flat_map(|b| {
            b.observed
                .iter()
                .copied()
                .zip(b.observed_labels.iter().copied())
        })
        .map(|(x, y)| [x, y])
        .collect();
    let mut expected = observations.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    if actual != expected {
        return Err("saved observed-input dictionary differs from original data".into());
    }
    model.encode(observations)?;
    Ok(())
}

impl Model {
    pub fn new(
        prior: kernel::Model,
        supplement: kernel::Model,
        raw: &kernel::discovery::Report,
        grouped: &grouped::discovery::Report,
        records: Vec<[usize; 2]>,
    ) -> Result<Self> {
        if raw.request.input.n != grouped.request.input.n
            || raw.request.input.vocabulary != grouped.request.input.vocabulary
            || raw.request.input.observations != grouped.request.input.observations
            || raw.request.prime != grouped.request.prime
            || prior.n != raw.request.input.n
            || prior.vocabulary != raw.request.input.vocabulary
            || supplement.n != prior.n
            || supplement.vocabulary != prior.vocabulary
        {
            return Err("heads do not share original data, field and vocabulary".into());
        }
        let grouped_certificate = grouped
            .grouped
            .as_ref()
            .ok_or("missing grouped certificate")?;
        let certificates = raw
            .buffer_preflight_bytes
            .checked_add(grouped_certificate.buffer_preflight_bytes)
            .ok_or("certificate forecast overflow")?;
        let bounds = bounds::combine(
            &prior.bounds()?,
            &supplement.bounds()?,
            certificates,
            records.len(),
        )?;
        kernel::discovery::check(&raw.request, raw)?;
        grouped::discovery::check(&grouped.request, grouped)?;
        if Some(&prior.encoding.code) != raw.code.as_ref()
            || Some(&supplement.encoding.code) != grouped.raw_code.as_ref()
        {
            return Err("actual support codes differ from the complete certificates".into());
        }
        prior.validate()?;
        supplement.validate()?;
        let observations = &raw.request.input.observations;
        dictionary(&prior, observations)?;
        dictionary(&supplement, observations)?;
        let labels: BTreeMap<_, _> = observations.iter().map(|&[x, y]| (x, y)).collect();
        let mut seen = BTreeSet::new();
        for &[cell, label] in &records {
            if labels.get(&cell) != Some(&label) || !prior.known(cell) || !supplement.known(cell) {
                return Err("original record lacks consistent observed support".into());
            }
            seen.insert(cell);
        }
        if seen.len() != labels.len() {
            return Err("original records omit an observed input".into());
        }
        prior.audit(observations)?;
        supplement.audit(observations)?;
        let binding = Binding {
            prior_sha256: digest(&prior)?,
            supplement_sha256: digest(&supplement)?,
            raw_discovery_sha256: digest(raw)?,
            grouped_discovery_sha256: digest(grouped)?,
            records_sha256: digest(&records)?,
        };
        let mut coverage = Coverage::default();
        for cell in 0..prior.n * prior.n {
            let (old, new) = (prior.known(cell), supplement.known(cell));
            if old && !new {
                return Err("checked grouped kernel loses prior match coverage".into());
            }
            coverage.prior_supported += usize::from(old);
            coverage.newly_covered += usize::from(!old && new);
            coverage.jointly_refused += usize::from(!old && !new);
            coverage.unknown_prior_axes += usize::from(prior.latent(cell).is_none());
            coverage.unmatched_prior_cells += usize::from(prior.latent(cell).is_some() && !old);
        }
        Ok(Self {
            prior,
            supplement,
            records,
            binding,
            bounds,
            coverage,
        })
    }
}
