use super::*;
use serde_json::json;

pub fn check(input: &Input, out: &Report) -> Result<Value> {
    if &out.input != input
        || out.buffer_bound_bytes != input.validate()?
        || out.partition != plan::partition(input)
    {
        return Err("categorical input, partition or storage provenance differs".into());
    }
    let q = out.partition.classes.len();
    let mut work = Work::default();
    if !(2..=257).contains(&q) {
        if out.system.is_some()
            || !out.factors.is_empty()
            || !out.components.is_empty()
            || !out.crt.is_empty()
            || !out.crt_audits.is_empty()
            || out.point.is_some()
            || out.collision.is_some()
            || out.observation_conflict.is_some()
            || out.model.is_some()
            || out.audit.is_some()
            || out.reason != "unsupported_latent_domain"
            || out.work != work
        {
            return Err("unsupported categorical domain supplied state".into());
        }
        return Ok(json!({"refusal_checked":out.reason,"unseen_answers_identified":false}));
    }
    let system = plan::system(input, &out.partition);
    system.validate()?;
    if out.system.as_ref() != Some(&system)
        || out.factors != plan::factors(q)
        || out.components.len() != out.factors.len()
    {
        return Err("categorical original equations, factors or component extent differs".into());
    }
    let mut failed = false;
    for (component, factor) in out.components.iter().zip(&out.factors) {
        if &component.factor != factor {
            return Err("component factor changed".into());
        }
        component_audit::check(input, &system, component, &mut work, failed)?;
        failed |= component.point.is_none();
    }
    if failed {
        if !out.crt.is_empty()
            || !out.crt_audits.is_empty()
            || out.point.is_some()
            || out.collision.is_some()
            || out.observation_conflict.is_some()
            || out.model.is_some()
            || out.audit.is_some()
            || out.reason != "component_refused"
            || out.work != work
        {
            return Err("refused component produced a final code or uncharged work".into());
        }
        return Ok(
            json!({"components_checked":out.components.len(),"refusal_checked":out.reason,"unseen_answers_identified":false}),
        );
    }
    if out.crt.len() != out.factors.len() - 1 || out.crt_audits.len() != out.crt.len() {
        return Err("wrong CRT extent".into());
    }
    let mut point = out.components[0]
        .point
        .clone()
        .ok_or("absent first final component")?;
    let mut modulus = out.factors[0].modulus;
    for (index, component) in out.components[1..].iter().enumerate() {
        let step = &out.crt[index];
        let expected = recombine::Input {
            system: system.clone(),
            left_modulus: modulus,
            right_modulus: component.factor.modulus,
            left: point,
            right: component.point.clone().ok_or("absent final component")?,
        };
        if recombine::check(&expected, step)? != out.crt_audits[index] {
            return Err("CRT audit changed".into());
        }
        point = step
            .point
            .clone()
            .ok_or("coprime factors did not recombine")?;
        modulus *= component.factor.modulus;
        work.inverse_tests += step.inverse_tests;
        work.crt_original_products += step.original_products;
    }
    if modulus as usize != q || out.point.as_ref() != Some(&point) {
        return Err("final categorical code is not the CRT representative".into());
    }
    crate::checked_residuals(&system, modulus, &point)?;
    endpoint::check(input, out, &point, &mut work)?;
    if work != out.work {
        return Err(
            "categorical construction, original observation or raw-audit work differs".into(),
        );
    }
    Ok(
        json!({"components_checked":out.components.len(),"digit_lifts_checked":out.components.iter().map(|c|c.lifts.len()).sum::<usize>(),
        "original_observations_checked":work.observation_checks,"raw_frequencies":work.raw_frequencies,
        "reason":out.reason,"unseen_answers_identified":false,
        "scope":"declared cyclic domain and normalized component policy; no global discovery guarantee"}),
    )
}
