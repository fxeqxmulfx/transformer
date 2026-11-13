//! Bind original observed data to each mode without accepting a readout,
//! held-out set, task name or target rule in construction/training.
use crate::*;
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
    pub initialization: Initialization,
    pub endpoints: Vec<Endpoint>,
    pub raw_label_identification: bool,
    pub target_readout_compatibility_proved: bool,
    pub new_discovery_work: usize,
}
fn fingerprint<T: Serialize>(value: &T) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(value)?)
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
        return Err("missing or reordered refined full/mini/forward modes".into());
    }
    Ok(())
}
pub fn run(discovery: &quotient::Report, policies: [training::Policy; 3]) -> Result<Report> {
    validate(&policies)?;
    let initialization = initialize(discovery)?;
    let samples = &discovery.request.input.observations;
    let encoded = initialization.model.encode(samples)?;
    let mut endpoints = vec![];
    for policy in &policies {
        if let Err(reason) = policy.preflight(&initialization.model, &encoded) {
            endpoints.push(Endpoint {
                training: None,
                audit: None,
                refusal: Some(reason.to_string()),
            });
            continue;
        }
        let training = training::run(&initialization.model, samples, policy.clone())?;
        let audit = training.model.audit(samples)?;
        endpoints.push(Endpoint {
            training: Some(training),
            audit: Some(audit),
            refusal: None,
        });
    }
    Ok(Report {
        discovery_sha256: fingerprint(discovery)?,
        input_sha256: fingerprint(&discovery.request.input)?,
        policies,
        initialization,
        endpoints,
        raw_label_identification: false,
        target_readout_compatibility_proved: false,
        new_discovery_work: 0,
    })
}
pub fn check(
    discovery: &quotient::Report,
    policies: &[training::Policy; 3],
    out: &Report,
) -> Result<()> {
    validate(policies)?;
    if out.discovery_sha256 != fingerprint(discovery)?
        || out.input_sha256 != fingerprint(&discovery.request.input)?
        || out.policies != *policies
        || out.endpoints.len() != 3
        || out.raw_label_identification
        || out.target_readout_compatibility_proved
        || out.new_discovery_work != 0
    {
        return Err("original refined input, modes or conditional scope changed".into());
    }
    construct::check(discovery, &out.initialization)?;
    let samples = &discovery.request.input.observations;
    let initial = &out.initialization.model;
    let encoded = initial.encode(samples)?;
    for (policy, endpoint) in policies.iter().zip(&out.endpoints) {
        match policy.preflight(initial, &encoded) {
            Err(reason) => {
                if endpoint.refusal.as_deref() != Some(reason.to_string().as_str())
                    || endpoint.training.is_some()
                    || endpoint.audit.is_some()
                {
                    return Err("refined mode refusal changed".into());
                }
            }
            Ok(_) => {
                let training = endpoint
                    .training
                    .as_ref()
                    .ok_or("missing feasible refined mode")?;
                if endpoint.refusal.is_some() || training.policy != *policy {
                    return Err("refined mode policy/status differs".into());
                }
                training::check(initial, samples, training)?;
                if endpoint.audit.as_ref() != Some(&training.model.audit(samples)?) {
                    return Err("refined actual-state spectral/CE audit differs".into());
                }
            }
        }
    }
    Ok(())
}
pub fn replay(
    discovery: &quotient::Report,
    policies: &[training::Policy; 3],
    out: &Report,
) -> Result<()> {
    check(discovery, policies, out)?;
    if run(discovery, policies.clone())? != *out {
        return Err("refined complete construction/update replay differs".into());
    }
    Ok(())
}
