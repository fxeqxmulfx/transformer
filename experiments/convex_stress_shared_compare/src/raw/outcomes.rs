//! Original scoring of actual saved complex coefficients, including pooled
//! answers. Candidate readout compatibility does not identify an unknown target.
use super::*;
pub(crate) fn compatible(d: &discovery::Report, readout: &[usize]) -> bool {
    let Some(code) = &d.code else {
        return false;
    };
    if readout.len() != d.request.input.vocabulary
        || code
            .validate(d.request.input.n, d.request.input.vocabulary)
            .is_err()
    {
        return false;
    }
    let mut groups = BTreeMap::new();
    for (&class, key) in code.names.classes.iter().zip(&code.classes) {
        let Some(&answer) = readout.get(class) else {
            return false;
        };
        if groups
            .insert(key, answer)
            .is_some_and(|previous| previous != answer)
        {
            return false;
        }
    }
    true
}
pub(crate) fn offers(
    task: &str,
    prime: u16,
    sha: &str,
    r: &binding::Report,
    eligible: bool,
) -> Result<Vec<comparison::Offer>> {
    let mut offers = vec![];
    if r.endpoints.len() != 3 {
        return Err("complete-kernel report lost a mode".into());
    }
    for (i, e) in r.endpoints.iter().enumerate() {
        match (&e.training, &e.audit, &e.refusal) {
            (Some(t), Some(a), None) => {
                if mode(t.policy.mode) != MODES[i]
                    || t.policy != r.policies[i]
                    || (t.final_ce - a.train_ce).abs() > 1e-8
                {
                    return Err("saved mode/CE/audit differs".into());
                }
                if eligible {
                    offers.push(comparison::Offer {
                        task: task.into(),
                        mode: t.policy.mode,
                        prime,
                        train_ce: a.train_ce,
                        model_sha256: hash(&serde_json::to_vec(&t.model)?),
                        stage_sha256: sha.into(),
                    });
                }
            }
            (None, None, Some(reason)) if !reason.is_empty() => {}
            _ => return Err("ambiguous actual mode availability".into()),
        }
    }
    Ok(offers)
}
