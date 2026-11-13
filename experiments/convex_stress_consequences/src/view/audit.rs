use super::*;

pub fn check(input: &Request, out: &Report) -> Result<()> {
    let slot = catalogue(&input.input)?
        .get(input.view)
        .cloned()
        .ok_or("checked view index outside catalogue")?;
    if out.input != *input || out.slot != slot {
        return Err("original fixed-view partition, policy or preflight differs".into());
    }
    if !matches!(slot.disposition, Disposition::Ready { .. }) {
        if out.interior.is_some()
            || !out.decisions.is_empty()
            || !out.candidates.is_empty()
            || out.selected.is_some()
            || out.work != Work::default()
        {
            return Err("skipped or refused view contains undeclared work".into());
        }
        return Ok(());
    }
    let inner = out
        .interior
        .as_ref()
        .ok_or("attempted view lost its original learner report")?;
    learner::check(&input.input.inner(&slot.view), inner)?;
    if out.decisions.len() != inner.candidates.len() {
        return Err("view omitted an admitted interior proposal".into());
    }
    let mut work = Work {
        interior: Some(inner.work.clone()),
        ..Work::default()
    };
    let mut admitted = 0;
    for (interior, base) in inner.candidates.iter().enumerate() {
        let model = binding::reconstruct(&slot.view, &base.model)?;
        work.lifts += 1;
        work.full_prediction_cells += model.n * model.n;
        work.raw_audits += 1;
        work.raw_frequency_extents.push(None);
        let decision = match model.audit(&input.input.samples()) {
            Err(error) => Decision::RawRefusal {
                error: error.to_string(),
            },
            Ok(raw_audit) => {
                let c = out
                    .candidates
                    .get(admitted)
                    .ok_or("view lost an actual full-model proposal")?;
                let scope = binding::scope(&slot.view, base, &model)?;
                let duplicate_of = out.candidates[..admitted]
                    .iter()
                    .position(|c| c.model.output_labels == model.output_labels);
                if c.interior != interior
                    || c.model != model
                    || c.raw_audit != raw_audit
                    || c.scope != scope
                    || c.duplicate_of != duplicate_of
                    || raw_audit["training_errors"] != 0
                {
                    return Err(
                        "actual fixed-view model, raw spectra, CE or conditional scope differs"
                            .into(),
                    );
                }
                work.completed_raw_frequencies += model.q;
                *work
                    .raw_frequency_extents
                    .last_mut()
                    .expect("current checked full audit") = Some(model.q);
                work.scope_cells += scope.origins.len();
                let candidate = admitted;
                admitted += 1;
                Decision::Admitted { candidate }
            }
        };
        if decision != out.decisions[interior] {
            return Err("full-model admission or numerical refusal differs".into());
        }
    }
    if admitted != out.candidates.len()
        || out.work != work
        || out.selected != plan::select(&out.candidates)?
    {
        return Err("view candidate extent, work or observed-only selection differs".into());
    }
    Ok(())
}
