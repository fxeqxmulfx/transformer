//! Preserve every independent mode outcome, including empty-program and work
//! refusals; source-code flatness never asserts empirical target realizability.
use crate::{
    model::Model,
    training::{Mode, Outcome, Policy},
    *,
};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    pub policy: Policy,
    pub training: Option<Outcome>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub discovery: discovery::Report,
    pub initial: Option<Model>,
    pub audit: Option<head::Audit>,
    pub initialization_error: Option<String>,
    pub endpoints: Vec<Endpoint>,
    pub target_realizability_certified: bool,
}

pub fn run(input: Input) -> Result<Report> {
    let discovery = discovery::run(input)?;
    let mut report = Report {
        discovery,
        initial: None,
        audit: None,
        initialization_error: None,
        endpoints: vec![],
        target_realizability_certified: false,
    };
    discovery::check(&report.discovery.input, &report.discovery)?;
    let initial = Model::new(&report.discovery);
    match initial {
        Err(error) => report.initialization_error = Some(error.to_string()),
        Ok(model) => {
            match model.head.audit(&model.weights) {
                Ok(audit) => report.audit = Some(audit),
                Err(error) => report.initialization_error = Some(error.to_string()),
            }
            report.initial = Some(model);
        }
    }
    for policy in training::policies()? {
        let mut endpoint = Endpoint {
            policy: policy.clone(),
            training: None,
            error: report.initialization_error.clone(),
        };
        if endpoint.error.is_none() {
            match training::run(
                report.initial.as_ref().unwrap(),
                &report.discovery.input,
                policy,
            ) {
                Ok(out) => endpoint.training = Some(out),
                Err(error) => endpoint.error = Some(error.to_string()),
            }
        }
        report.endpoints.push(endpoint);
    }
    Ok(report)
}

pub fn check(input: &Input, out: &Report) -> Result<()> {
    if out.target_realizability_certified
        || out.discovery.input != *input
        || out.endpoints.len() != 3
        || out
            .endpoints
            .iter()
            .map(|e| e.policy.mode)
            .collect::<Vec<_>>()
            != vec![Mode::Full, Mode::Mini, Mode::Forward]
    {
        return Err("changed input, missing mode or unsupported realization certificate".into());
    }
    discovery::check(input, &out.discovery)?;
    if run(input.clone())? != *out {
        return Err("bound shared program model, audit or refusals changed".into());
    }
    for endpoint in &out.endpoints {
        if let Some(training) = &endpoint.training {
            training::check(out.initial.as_ref().unwrap(), input, training)?;
        }
    }
    Ok(())
}
