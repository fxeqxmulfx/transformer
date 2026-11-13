use super::*;

pub fn run(parent: &search::Report, policy: Policy) -> Result<Report> {
    let audit = search::check(&parent.input, parent)?;
    let mut out = Report {
        buffer_preflight_bytes: binding::forecast(parent, &policy)?,
        policy,
        parent_audit: audit,
        parent_views: binding::views(parent),
        entries: vec![],
        certificate_count: 0,
        ce_gradients: 0,
    };
    for (index, c) in parent.candidates.iter().enumerate() {
        let request = binding::request(parent, index, &out.policy)?;
        let checked_predictions = binding::predictions(parent, index, &request)?;
        let certificate = categorical::run(request.clone())?;
        categorical::check(&request, &certificate)?;
        out.certificate_count += usize::from(certificate.certified);
        out.entries.push(Entry {
            candidate: index,
            view: c.view,
            model_sha256: hash(&serde_json::to_vec(&c.model)?),
            request_sha256: hash(&serde_json::to_vec(&request)?),
            full_prediction_checks: checked_predictions,
            scope: binding::scope(parent, index)?,
            certificate,
        });
    }
    Ok(out)
}
pub fn check(parent: &search::Report, policy: &Policy, out: &Report) -> Result<()> {
    if out.policy != *policy
        || out.buffer_preflight_bytes != binding::forecast(parent, policy)?
        || out.parent_audit != search::check(&parent.input, parent)?
        || out.parent_views != binding::views(parent)
        || out.ce_gradients != 0
        || out.entries.len() != parent.candidates.len()
    {
        return Err(
            "identification policy, complete parent, extent or aggregate bound differs".into(),
        );
    }
    let mut certified = 0;
    for (index, (c, saved)) in parent.candidates.iter().zip(&out.entries).enumerate() {
        let request = binding::request(parent, index, policy)?;
        if saved.candidate != index
            || saved.view != c.view
            || saved.model_sha256 != hash(&serde_json::to_vec(&c.model)?)
            || saved.request_sha256 != hash(&serde_json::to_vec(&request)?)
            || saved.scope != binding::scope(parent, index)?
            || saved.full_prediction_checks != binding::predictions(parent, index, &request)?
        {
            return Err(
                "identification endpoint, actual prediction or conditional scope changed".into(),
            );
        }
        categorical::check(&request, &saved.certificate)?;
        certified += usize::from(saved.certificate.certified);
    }
    if certified != out.certificate_count {
        return Err("identification certificate count differs".into());
    }
    Ok(())
}
pub fn replay(parent: &search::Report, out: &Report) -> Result<()> {
    check(parent, &out.policy, out)?;
    for entry in &out.entries {
        categorical::replay(&entry.certificate.input, &entry.certificate)?;
    }
    Ok(())
}
