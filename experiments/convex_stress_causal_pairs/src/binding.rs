//! Retain every discovery, initialization and mode-specific refusal. A failed
//! iterative budget must not discard an independently available forward model.
use crate::*;
#[cfg(test)]
mod tests;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub discovery: crate::Request,
    pub training: [training::Policy; 3],
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    pub policy: training::Policy,
    pub training: Option<training::Outcome>,
    pub audit: Option<model::Audit>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub request: Request,
    pub discovery: discovery::Report,
    pub initialization: Option<model::Initialization>,
    pub initialization_error: Option<String>,
    pub endpoints: Vec<Endpoint>,
    pub buffer_preflight_bytes: u64,
    pub unseen_identification: bool,
}
impl Request {
    pub fn validate(&self) -> Result<u64> {
        let discovery = self.discovery.validate()?;
        if self.training.each_ref().map(|p| p.mode)
            != [
                training::Mode::Full,
                training::Mode::Mini,
                training::Mode::Forward,
            ]
        {
            return Err("missing or reordered full/mini/forward policy".into());
        }
        // Four retained models, replay temporaries and serialization, using
        // the absolute model buffer caps before the selected states are known.
        let model = 512 * 1024 * 1024u64 + 128 * 4_000_000 + 128 * 16_000_000;
        let bytes = discovery + 8 * model;
        if bytes > 56 * 1024 * 1024 * 1024 {
            return Err("complete binding exceeds 56 GiB within the 64 GiB cap".into());
        }
        Ok(bytes)
    }
}
fn endpoint(initial: &model::Model, input: &Input, policy: training::Policy) -> Endpoint {
    let mut out = Endpoint {
        policy,
        training: None,
        audit: None,
        error: None,
    };
    match training::run(initial, input, out.policy.clone()) {
        Err(error) => out.error = Some(error.to_string()),
        Ok(training) => {
            match training.model.audit(input) {
                Ok(audit) => out.audit = Some(audit),
                Err(error) => out.error = Some(error.to_string()),
            }
            out.training = Some(training);
        }
    }
    out
}
pub fn run(request: Request) -> Result<Report> {
    let bytes = request.validate()?;
    let discovery = discovery::run(request.discovery.clone())?;
    let mut out = Report {
        request,
        discovery,
        initialization: None,
        initialization_error: None,
        endpoints: vec![],
        buffer_preflight_bytes: bytes,
        unseen_identification: false,
    };
    match model::initialize(&out.discovery) {
        Err(error) => out.initialization_error = Some(error.to_string()),
        Ok(initial) => {
            if let Err(error) = model::check_initialization(&out.discovery, &initial) {
                out.initialization_error = Some(error.to_string());
            } else {
                for policy in &out.request.training {
                    out.endpoints.push(endpoint(
                        &initial.model,
                        &out.request.discovery.input,
                        policy.clone(),
                    ));
                }
            }
            // Even a model whose independent inversion check exceeds its
            // forecast remains evidence; it cannot be offered for evaluation.
            out.initialization = Some(initial);
        }
    }
    Ok(out)
}
pub fn check(request: &Request, out: &Report) -> Result<()> {
    if out.request != *request
        || out.buffer_preflight_bytes != request.validate()?
        || out.unseen_identification
    {
        return Err("bound request, process forecast or identification claim changed".into());
    }
    discovery::replay(&request.discovery, &out.discovery)?;
    match model::initialize(&out.discovery) {
        Err(error) => {
            if out.initialization.is_some()
                || !out.endpoints.is_empty()
                || out.initialization_error != Some(error.to_string())
            {
                return Err("initialization refusal was changed or omitted".into());
            }
        }
        Ok(initial) => {
            if out.initialization.as_ref() != Some(&initial) {
                return Err("actual saved complex initializer changed".into());
            }
            if let Err(error) = model::check_initialization(&out.discovery, &initial) {
                if !out.endpoints.is_empty() || out.initialization_error != Some(error.to_string())
                {
                    return Err("initializer audit refusal changed".into());
                }
                return Ok(());
            }
            if out.initialization_error.is_some() || out.endpoints.len() != 3 {
                return Err("independent mode outcomes missing".into());
            }
            for (saved, policy) in out.endpoints.iter().zip(&request.training) {
                if &saved.policy != policy {
                    return Err("mode policy changed".into());
                }
                if let Some(t) = &saved.training {
                    if &t.policy != policy {
                        return Err("training policy differs from mode".into());
                    }
                    training::check(&initial.model, &request.discovery.input, t)?;
                }
                if *saved != endpoint(&initial.model, &request.discovery.input, policy.clone()) {
                    return Err("actual mode training, audit, work or refusal differs".into());
                }
            }
        }
    }
    Ok(())
}
pub fn replay(request: &Request, out: &Report) -> Result<()> {
    check(request, out)?;
    if run(request.clone())? != *out {
        return Err("complete paired complex binding replay differs".into());
    }
    Ok(())
}
