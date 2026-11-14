//! CM39/40 binding, ported from coverage_extension at 08e4108b. Check the
//! actual augmented certificate and retain original IDs using CM40 records.
//! An importing adapter must additionally bind and replay saved CE histories.
use crate::*;

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
        shared: &shared::discovery::Report,
        records: Vec<records::Record>,
    ) -> Result<Self> {
        if raw.request.input.n != shared.request.input.n
            || raw.request.input.vocabulary != shared.request.input.vocabulary
            || raw.request.input.observations != shared.request.input.observations
            || raw.request.prime != shared.request.prime
            || prior.n != raw.request.input.n
            || prior.vocabulary != raw.request.input.vocabulary
            || supplement.n != prior.n
            || supplement.vocabulary != prior.vocabulary
        {
            return Err("heads do not share original data, field and vocabulary".into());
        }
        records::validate(&raw.request.input, &records)?;
        let certificates = raw
            .buffer_preflight_bytes
            .checked_add(shared.buffer_preflight_bytes)
            .ok_or("certificate forecast overflow")?;
        let bounds = bounds::combine(
            &prior.bounds()?,
            &supplement.bounds()?,
            certificates,
            records.len(),
        )?;
        kernel::discovery::check(&raw.request, raw)?;
        shared::discovery::check(&shared.request, shared)?;
        if Some(&prior.encoding.code) != raw.code.as_ref()
            || Some(&supplement.encoding.code) != shared.raw_code.as_ref()
        {
            return Err("actual support codes differ from the complete certificates".into());
        }
        prior.validate()?;
        supplement.validate()?;
        let observations = &raw.request.input.observations;
        dictionary(&prior, observations)?;
        dictionary(&supplement, observations)?;
        records::encode(&prior, &records)?;
        records::encode(&supplement, &records)?;
        for record in &records {
            if !prior.known(record.cell) || !supplement.known(record.cell) {
                return Err("original record lacks consistent observed support".into());
            }
        }
        prior.audit(observations)?;
        supplement.audit(observations)?;
        let binding = Binding {
            prior_sha256: digest(&prior)?,
            supplement_sha256: digest(&supplement)?,
            raw_discovery_sha256: digest(raw)?,
            shared_discovery_sha256: digest(shared)?,
            records_sha256: digest(&records)?,
        };
        let mut coverage = Coverage::default();
        for cell in 0..prior.n * prior.n {
            let (old, new) = (prior.known(cell), supplement.known(cell));
            if old && !new {
                return Err("checked shared kernel loses prior match coverage".into());
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
