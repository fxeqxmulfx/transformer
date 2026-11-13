use super::*;
use serde_json::json;

pub fn check(input: &Input, out: &Report) -> Result<Value> {
    if &out.input != input || out.buffer_bound_bytes != input.validate()? {
        return Err("composite strip input or bound differs".into());
    }
    let samples = input.samples();
    let saved: Vec<_> = out.attempts.iter().map(|a| a.view.clone()).collect();
    views::replay(
        input.categorical.n,
        input.categorical.vocab,
        &samples,
        input.minimum_strip_support,
        &saved,
    )?;
    let mut work = Work {
        views_proposed: saved.len(),
        ..Work::default()
    };
    let mut accepted = 0;
    for (index, a) in out.attempts.iter().enumerate() {
        let skipped = if index >= input.max_views {
            Some(Decision::ViewCap)
        } else if a.view.n < 2 || a.view.observations.is_empty() {
            Some(Decision::EmptyInterior)
        } else {
            None
        };
        if let Some(decision) = skipped {
            if a.decision != decision || a.coding.is_some() || a.audit.is_some() {
                return Err("skipped strip view has undeclared work".into());
            }
            continue;
        }
        let inner = input.inner(&a.view);
        let code = a
            .coding
            .as_ref()
            .ok_or("attempted strip lacks its code report")?;
        if a.audit.as_ref() != Some(&coding::check(&inner, code)?) {
            return Err("strip code audit changed".into());
        }
        work.charge(code);
        let Some(base) = &code.model else {
            if a.decision != Decision::Refused {
                return Err("refused strip code supplied a candidate".into());
            }
            continue;
        };
        let model = a.view.lift(base)?;
        // Check every full-table prediction from the category maps, including
        // unseen cells, without trusting the stored full output table.
        for (j, column) in a.view.columns.iter().enumerate() {
            for (i, row) in a.view.rows.iter().enumerate() {
                let expected = if let (Some(r), Some(c)) = (row, column) {
                    base.output_labels[r + base.n * c]
                } else {
                    a.view.constant.ok_or("strip coordinate has no constant")?
                };
                if model.output_labels[i + input.categorical.n * j] != expected {
                    return Err("actual strip lift differs from its categorical map".into());
                }
            }
        }
        if let Some(candidate) = out.candidates[..accepted]
            .iter()
            .position(|c| c.model.output_labels == model.output_labels)
        {
            if a.decision != (Decision::Duplicate { candidate }) {
                return Err("duplicate strip endpoint was not retained as a duplicate".into());
            }
        } else {
            let c = out
                .candidates
                .get(accepted)
                .ok_or("missing admitted strip candidate")?;
            let raw = model.audit(&samples)?;
            if a.decision
                != (Decision::Accepted {
                    candidate: accepted,
                })
                || c.view != index
                || c.model != model
                || c.audit != raw
                || raw["training_errors"] != 0
            {
                return Err(
                    "strip actual model, original fit or raw spectral/CE audit differs".into(),
                );
            }
            work.full_raw_frequencies += model.q;
            accepted += 1;
        }
    }
    let (selected, reason) = if accepted == 0 {
        (None, "no_admissible_candidate")
    } else {
        (Some(0), "admitted_proposals")
    };
    if accepted != out.candidates.len()
        || selected != out.selected
        || reason != out.reason
        || work != out.work
    {
        return Err("strip candidate extent, CE choice or work differs".into());
    }
    Ok(
        json!({"views_checked":saved.len(),"candidates_checked":accepted,
        "interior_raw_frequencies":work.interior_raw_frequencies,"full_raw_frequencies":work.full_raw_frequencies,
        "original_categorical_maps_checked":true,"unseen_answers_identified":false}),
    )
}
