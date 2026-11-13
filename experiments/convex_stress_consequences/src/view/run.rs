use super::*;

pub fn run(input: Request) -> Result<Report> {
    let slot = catalogue(&input.input)?
        .get(input.view)
        .cloned()
        .ok_or("view index outside catalogue")?;
    let mut out = Report {
        input,
        slot,
        interior: None,
        decisions: vec![],
        candidates: vec![],
        selected: None,
        work: Work::default(),
    };
    if !matches!(out.slot.disposition, Disposition::Ready { .. }) {
        return Ok(out);
    }
    let request = out.input.input.inner(&out.slot.view);
    let inner = learner::run(request.clone())?;
    learner::check(&request, &inner)?;
    out.work.interior = Some(inner.work.clone());
    for (interior, base) in inner.candidates.iter().enumerate() {
        let model = out.slot.view.lift(&base.model)?;
        out.work.lifts += 1;
        out.work.full_prediction_cells += model.n * model.n;
        out.work.raw_audits += 1;
        out.work.raw_frequency_extents.push(None);
        let decision = match model.audit(&out.input.input.samples()) {
            Err(error) => Decision::RawRefusal {
                error: error.to_string(),
            },
            Ok(raw_audit) => {
                if raw_audit["training_errors"] != 0 {
                    return Err("actual fixed-view model contradicts original observations".into());
                }
                out.work.completed_raw_frequencies += model.q;
                *out.work
                    .raw_frequency_extents
                    .last_mut()
                    .expect("current full audit") = Some(model.q);
                let scope = binding::scope(&out.slot.view, base, &model)?;
                out.work.scope_cells += scope.origins.len();
                let duplicate_of = out
                    .candidates
                    .iter()
                    .position(|c| c.model.output_labels == model.output_labels);
                let candidate = out.candidates.len();
                out.candidates.push(Candidate {
                    interior,
                    duplicate_of,
                    model,
                    raw_audit,
                    scope,
                });
                Decision::Admitted { candidate }
            }
        };
        out.decisions.push(decision);
    }
    out.selected = plan::select(&out.candidates)?;
    out.interior = Some(inner);
    Ok(out)
}
