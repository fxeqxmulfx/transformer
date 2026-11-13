//! CM35 ports kernel/binding.rs at166bebc3 for declared readout discovery.
//! Original raw observations, full-vocabulary CE, independent actual complex
//! audits and update replay are unchanged. Unknown-target premises stay open.
use crate::*;
use construct::initialize;
use sha2::{Digest, Sha256};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    pub training: Option<training::Outcome>,
    pub audit: Option<audit::Audit>,
    pub refusal: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub discovery_sha256: String,
    pub input_sha256: String,
    pub policies: [training::Policy; 3],
    pub initialization: Option<Initialization>,
    pub initialization_refusal: Option<String>,
    pub endpoints: Vec<Endpoint>,
    pub raw_label_identification: bool,
    pub target_readout_compatibility_proved: bool,
}
fn fingerprint<T: Serialize>(x: &T) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(x)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
fn validate(policies: &[training::Policy; 3]) -> Result<()> {
    if policies.each_ref().map(|p| p.mode)
        != [
            training::Mode::Full,
            training::Mode::Mini,
            training::Mode::Forward,
        ]
    {
        return Err("missing or reordered kernel training modes".into());
    }
    Ok(())
}
fn refused(reason: &str) -> Endpoint {
    Endpoint {
        training: None,
        audit: None,
        refusal: Some(reason.into()),
    }
}
pub fn run(discovery: &discovery::Report, policies: [training::Policy; 3]) -> Result<Report> {
    validate(&policies)?;
    discovery::check(&discovery.request, discovery)?;
    let mut out = Report {
        discovery_sha256: fingerprint(discovery)?,
        input_sha256: fingerprint(&discovery.request.input)?,
        policies,
        initialization: None,
        initialization_refusal: None,
        endpoints: vec![],
        raw_label_identification: false,
        target_readout_compatibility_proved: false,
    };
    let initial = match initialize(discovery) {
        Ok(initial) => initial,
        Err(reason) => {
            let reason = reason.to_string();
            out.endpoints = (0..3).map(|_| refused(&reason)).collect();
            out.initialization_refusal = Some(reason);
            return Ok(out);
        }
    };
    let samples = &discovery.request.input.observations;
    let encoded = initial.model.encode(samples)?;
    for policy in &out.policies {
        match preflight::work(policy, &initial.model, &encoded) {
            Err(reason) => out.endpoints.push(refused(&reason.to_string())),
            Ok(forecast) => {
                let training = training::run(&initial.model, samples, policy.clone())?;
                if training.scalar_work_forecast != forecast {
                    return Err("readout preflight differs from unchanged raw CE trainer".into());
                }
                let audit = training.model.audit(samples)?;
                out.endpoints.push(Endpoint {
                    training: Some(training),
                    audit: Some(audit),
                    refusal: None,
                });
            }
        }
    }
    out.initialization = Some(initial);
    Ok(out)
}
pub fn check(
    discovery: &discovery::Report,
    policies: &[training::Policy; 3],
    out: &Report,
) -> Result<()> {
    validate(policies)?;
    discovery::check(&discovery.request, discovery)?;
    if out.discovery_sha256 != fingerprint(discovery)?
        || out.input_sha256 != fingerprint(&discovery.request.input)?
        || out.policies != *policies
        || out.endpoints.len() != 3
        || out.raw_label_identification
        || out.target_readout_compatibility_proved
    {
        return Err("kernel source, policies or conditional scope changed".into());
    }
    let expected = match initialize(discovery) {
        Ok(initial) => initial,
        Err(reason) => {
            let reason = reason.to_string();
            if out.initialization.is_some()
                || out.initialization_refusal.as_deref() != Some(&reason)
                || out.endpoints.iter().any(|e| *e != refused(&reason))
            {
                return Err("kernel constructor refusal was altered".into());
            }
            return Ok(());
        }
    };
    if out.initialization_refusal.is_some() || out.initialization.as_ref() != Some(&expected) {
        return Err("actual kernel initialization changed".into());
    }
    construct::check(discovery, &expected)?;
    let initial = &expected.model;
    let samples = &discovery.request.input.observations;
    let encoded = initial.encode(samples)?;
    for (policy, endpoint) in policies.iter().zip(&out.endpoints) {
        match preflight::work(policy, initial, &encoded) {
            Err(reason) => {
                if *endpoint != refused(&reason.to_string()) {
                    return Err("kernel mode refusal changed".into());
                }
            }
            Ok(forecast) => {
                let training = endpoint
                    .training
                    .as_ref()
                    .ok_or("missing feasible kernel mode")?;
                if endpoint.refusal.is_some()
                    || training.policy != *policy
                    || training.scalar_work_forecast != forecast
                {
                    return Err("kernel mode policy or status differs".into());
                }
                training::check(initial, samples, training)?;
                if endpoint.audit.as_ref() != Some(&training.model.audit(samples)?) {
                    return Err("kernel actual-state audit differs".into());
                }
            }
        }
    }
    Ok(())
}
pub fn replay(
    discovery: &discovery::Report,
    policies: &[training::Policy; 3],
    out: &Report,
) -> Result<()> {
    check(discovery, policies, out)?;
    if run(discovery, policies.clone())? != *out {
        return Err("complete kernel construction/update replay differs".into());
    }
    Ok(())
}
