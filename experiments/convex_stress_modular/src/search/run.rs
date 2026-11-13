use super::*;

pub fn run(input: Input) -> Result<Report> {
    let buffer_bound_bytes = input.validate()?;
    let proposed = views::propose(
        input.categorical.n,
        input.categorical.vocab,
        &input.samples(),
        input.minimum_strip_support,
    )?;
    let mut out = Report {
        input,
        buffer_bound_bytes,
        attempts: vec![],
        candidates: vec![],
        selected: None,
        reason: "no_admissible_candidate".into(),
        work: Work {
            views_proposed: proposed.len(),
            ..Work::default()
        },
    };
    for (index, view) in proposed.into_iter().enumerate() {
        let mut a = Attempt {
            view,
            coding: None,
            audit: None,
            decision: Decision::ViewCap,
        };
        if index < out.input.max_views {
            if a.view.n < 2 || a.view.observations.is_empty() {
                a.decision = Decision::EmptyInterior;
            } else {
                let input = out.input.inner(&a.view);
                let code = coding::run(input.clone())?;
                a.audit = Some(coding::check(&input, &code)?);
                out.work.charge(&code);
                a.decision = Decision::Refused;
                if let Some(base) = &code.model {
                    let model = a.view.lift(base)?;
                    a.decision = if let Some(candidate) = out
                        .candidates
                        .iter()
                        .position(|c| c.model.output_labels == model.output_labels)
                    {
                        Decision::Duplicate { candidate }
                    } else {
                        let audit = model.audit(&out.input.samples())?;
                        if audit["training_errors"] != 0 {
                            return Err(
                                "categorical strip extension violates original observations".into(),
                            );
                        }
                        let candidate = out.candidates.len();
                        out.work.full_raw_frequencies += model.q;
                        out.candidates.push(Candidate {
                            view: index,
                            model,
                            audit,
                        });
                        // Every accepted model fits all observations at the
                        // same unit-logit CE minimum; retain the first.
                        out.selected.get_or_insert(candidate);
                        Decision::Accepted { candidate }
                    };
                }
                a.coding = Some(code);
            }
        }
        out.attempts.push(a);
    }
    if out.selected.is_some() {
        out.reason = "admitted_proposals".into();
    }
    Ok(out)
}
