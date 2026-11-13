//! Bind CM30 discovery, CM25 latent bounds and CD8 CE to original raw rows.
//! A compatible candidate readout is only a check on this learned code. The
//! unknown target's additive representation and readout compatibility remain
//! assumptions of the CM30 theorem, not conclusions of the native audit.
use crate::{Result, certificate, model, row_program, training};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub discovery: crate::Request,
    pub training: [training::Policy; 3],
    pub prefix_policy: row_program::Policy,
    pub certificate_policy: certificate::Policy,
    /// Optional declared evaluation readout; never used by discovery/training.
    pub readout: Option<Vec<usize>>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    pub training: training::Outcome,
    pub audit: model::Audit,
    pub certificate: certificate::Report,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub request: Request,
    pub discovery: crate::Report,
    pub initialization: Option<model::Initialization>,
    pub endpoints: Vec<Endpoint>,
    pub candidate_readout_compatible: Option<bool>,
    pub target_readout_compatibility_proved: bool,
    pub raw_label_identification: bool,
    pub buffer_preflight_bytes: u64,
}
impl Request {
    fn certificate(&self, predictions: Vec<usize>) -> certificate::Request {
        let input = &self.discovery.input;
        certificate::Request {
            n: input.n,
            vocabulary: input.vocabulary,
            modulus: self.discovery.prime,
            observations: input.observations.clone(),
            predictions,
            prefix_policy: self.prefix_policy.clone(),
            policy: self.certificate_policy.clone(),
        }
    }
    fn validate(&self) -> Result<u64> {
        let (_, _, discovery) = self.discovery.problem()?;
        let input = &self.discovery.input;
        if self.training.each_ref().map(|p| p.mode)
            != [
                training::Mode::Full,
                training::Mode::Mini,
                training::Mode::Forward,
            ]
            || self
                .readout
                .as_ref()
                .is_some_and(|q| q.len() != input.vocabulary)
        {
            return Err(
                "missing or reordered training mode, or readout vocabulary mismatch".into(),
            );
        }
        let (_, certificate) = self.certificate(vec![0; input.n * input.n]).validate()?;
        for policy in &self.training {
            policy.preflight(
                input.n,
                self.discovery.prime as usize,
                input.vocabulary,
                input.observations.len(),
            )?;
        }
        let model = model::memory_bound(input.n, self.discovery.prime as usize, input.vocabulary)?;
        let bytes = discovery + 3 * certificate + 4 * model + 1024 * 1024 * 1024;
        if bytes > 56 * 1024 * 1024 * 1024 {
            return Err("complete binding forecast exceeds56GiB within64GiB process limit".into());
        }
        Ok(bytes)
    }
}
fn compatibility(request: &Request, discovery: &crate::Report) -> Option<bool> {
    let readout = request.readout.as_ref()?;
    discovery.code.as_ref()?;
    Some(discovery.code_fibers.iter().all(|fiber| {
        fiber
            .first()
            .is_none_or(|&first| fiber.iter().all(|&c| readout[c] == readout[first]))
    }))
}
pub fn run(request: Request) -> Result<Report> {
    let bytes = request.validate()?;
    let discovery = crate::run(request.discovery.clone())?;
    let mut out = Report {
        candidate_readout_compatible: compatibility(&request, &discovery),
        request,
        discovery,
        initialization: None,
        endpoints: vec![],
        target_readout_compatibility_proved: false,
        raw_label_identification: false,
        buffer_preflight_bytes: bytes,
    };
    if out.discovery.code.is_none() {
        return Ok(out);
    }
    let initial = model::initialize(&out.discovery)?;
    for policy in &out.request.training {
        let training = training::run(
            &initial.model,
            &out.request.discovery.input.observations,
            policy.clone(),
        )?;
        let audit = training
            .model
            .audit(&out.request.discovery.input.observations)?;
        let certificate = certificate::run(out.request.certificate(training.model.predictions()))?;
        out.endpoints.push(Endpoint {
            training,
            audit,
            certificate,
        });
    }
    out.initialization = Some(initial);
    Ok(out)
}
pub fn check(request: &Request, out: &Report) -> Result<()> {
    if out.request != *request
        || out.buffer_preflight_bytes != request.validate()?
        || out.target_readout_compatibility_proved
        || out.raw_label_identification
    {
        return Err("original request, bound or conditional claim changed".into());
    }
    crate::check(&request.discovery, &out.discovery)?;
    if out.candidate_readout_compatible != compatibility(request, &out.discovery) {
        return Err("candidate-fiber/readout compatibility differs".into());
    }
    let Some(initial) = &out.initialization else {
        if out.discovery.code.is_some() || !out.endpoints.is_empty() {
            return Err("admitted code or endpoints lack an initializer".into());
        }
        return Ok(());
    };
    model::check_initialization(&out.discovery, initial)?;
    if out.endpoints.len() != 3 {
        return Err("full/mini/forward endpoint missing".into());
    }
    for (endpoint, policy) in out.endpoints.iter().zip(&request.training) {
        if &endpoint.training.policy != policy {
            return Err("training policy binding differs".into());
        }
        training::check(
            &initial.model,
            &request.discovery.input.observations,
            &endpoint.training,
        )?;
        if endpoint.audit
            != endpoint
                .training
                .model
                .audit(&request.discovery.input.observations)?
        {
            return Err("actual raw Fourier norms, head budget or ordinary CE differs".into());
        }
        certificate::check(
            &request.certificate(endpoint.training.model.predictions()),
            &endpoint.certificate,
        )?;
    }
    Ok(())
}
pub fn replay(request: &Request, out: &Report) -> Result<()> {
    check(request, out)?;
    if run(request.clone())? != *out {
        return Err("complete original discovery/model/training/certificate replay differs".into());
    }
    Ok(())
}
