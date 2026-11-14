//! CM40 ports readout_kernel/binding.rs at ec5f839d. The new matrix certificate,
//! ordered original record IDs, actual raw CE and all modes are bound separately.
//! Unknown shared-role target realizability and readout compatibility stay open.
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
    pub records_sha256: String,
    pub original_records: usize,
    pub buffer_preflight_bytes: Option<u64>,
    pub policies: [training::Policy; 3],
    pub initialization: Option<Initialization>,
    pub initialization_refusal: Option<String>,
    pub endpoints: Vec<Endpoint>,
    pub raw_label_identification: bool,
    pub target_readout_compatibility_proved: bool,
    pub shared_role_target_proved: bool,
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
pub fn run(
    discovery: &discovery::Report,
    records: &[records::Record],
    policies: [training::Policy; 3],
) -> Result<Report> {
    validate(&policies)?;
    discovery::check(&discovery.request, discovery)?;
    records::validate(&discovery.request.input, records)?;
    let mut out = Report {
        discovery_sha256: fingerprint(discovery)?,
        input_sha256: fingerprint(&discovery.request.input)?,
        records_sha256: fingerprint(&records)?,
        original_records: records.len(),
        buffer_preflight_bytes: None,
        policies,
        initialization: None,
        initialization_refusal: None,
        endpoints: vec![],
        raw_label_identification: false,
        target_readout_compatibility_proved: false,
        shared_role_target_proved: false,
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
    out.buffer_preflight_bytes = Some(construct::buffer_bytes(discovery, &initial.bounds)?);
    let samples = records;
    let encoded = records::encode(&initial.model, samples)?;
    for policy in &out.policies {
        match policy.preflight(&initial.model, &encoded) {
            Err(reason) => out.endpoints.push(refused(&reason.to_string())),
            Ok(forecast) => {
                let training = training::run(&initial.model, samples, policy.clone())?;
                if training.scalar_work_forecast != forecast {
                    return Err("readout preflight differs from unchanged raw CE trainer".into());
                }
                let audit = audit::run(&training.model, &discovery.request.input, samples)?;
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
    records: &[records::Record],
    policies: &[training::Policy; 3],
    out: &Report,
) -> Result<()> {
    validate(policies)?;
    discovery::check(&discovery.request, discovery)?;
    records::validate(&discovery.request.input, records)?;
    if out.discovery_sha256 != fingerprint(discovery)?
        || out.input_sha256 != fingerprint(&discovery.request.input)?
        || out.records_sha256 != fingerprint(&records)?
        || out.original_records != records.len()
        || out.policies != *policies
        || out.endpoints.len() != 3
        || out.raw_label_identification
        || out.target_readout_compatibility_proved
        || out.shared_role_target_proved
    {
        return Err("kernel source, policies or conditional scope changed".into());
    }
    let expected = match initialize(discovery) {
        Ok(initial) => initial,
        Err(reason) => {
            let reason = reason.to_string();
            if out.initialization.is_some()
                || out.buffer_preflight_bytes.is_some()
                || out.initialization_refusal.as_deref() != Some(&reason)
                || out.endpoints.iter().any(|e| *e != refused(&reason))
            {
                return Err("kernel constructor refusal was altered".into());
            }
            return Ok(());
        }
    };
    if out.initialization_refusal.is_some()
        || out.initialization.as_ref() != Some(&expected)
        || out.buffer_preflight_bytes != Some(construct::buffer_bytes(discovery, &expected.bounds)?)
    {
        return Err("actual kernel initialization changed".into());
    }
    construct::check(discovery, &expected)?;
    let initial = &expected.model;
    let samples = records;
    let encoded = records::encode(initial, samples)?;
    for (policy, endpoint) in policies.iter().zip(&out.endpoints) {
        match policy.preflight(initial, &encoded) {
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
                if endpoint.audit.as_ref()
                    != Some(&audit::run(
                        &training.model,
                        &discovery.request.input,
                        samples,
                    )?)
                {
                    return Err("kernel actual-state audit differs".into());
                }
            }
        }
    }
    Ok(())
}
pub fn replay(
    discovery: &discovery::Report,
    records: &[records::Record],
    policies: &[training::Policy; 3],
    out: &Report,
) -> Result<()> {
    check(discovery, records, policies, out)?;
    if run(discovery, records, policies.clone())? != *out {
        return Err("complete kernel construction/update replay differs".into());
    }
    Ok(())
}
