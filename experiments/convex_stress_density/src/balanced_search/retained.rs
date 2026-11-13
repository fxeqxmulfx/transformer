//! Preserve the complete earlier candidate order from cohort3311e13c.
use super::*;
pub(super) fn candidates(r: &refinement_search::Report) -> Result<Vec<(String, Model, f64)>> {
    let mut out = vec![];
    for (i, p) in r.parent.earlier.prior.iter().enumerate() {
        out.push((format!("prior-{i}"), p.model.clone(), p.train_ce));
    }
    for (i, a) in r.parent.proposal.exact.attempts.iter().enumerate() {
        if let Some(ce) = a.train_ce {
            out.push((
                format!("exact-{i}"),
                a.model.clone().ok_or("missing exact model")?,
                ce,
            ));
        }
    }
    for (i, a) in r.parent.proposal.attempts.iter().enumerate() {
        if let Some(ce) = a.train_ce {
            out.push((
                format!("interval-{i}"),
                a.model.clone().ok_or("missing interval model")?,
                ce,
            ));
        }
    }
    for (i, a) in r.attempts.iter().enumerate() {
        if let Some(ce) = a.train_ce {
            let o = a.outcome.as_ref().ok_or("missing refinement")?;
            out.push((
                format!("refined-{i}"),
                o.steps[o.best.ok_or("missing refined endpoint")?]
                    .state
                    .model(o.input.seed.vocab)?,
                ce,
            ));
        }
    }
    Ok(out)
}
