use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Attempt {
    pub request: binding::Request,
    pub report: Option<binding::Report>,
    pub returned_error: Option<String>,
}
impl Attempt {
    pub fn run(request: binding::Request) -> Self {
        match binding::run(request.clone()) {
            Ok(report) => Self {
                request,
                report: Some(report),
                returned_error: None,
            },
            Err(error) => Self {
                request,
                report: None,
                returned_error: Some(error.to_string()),
            },
        }
    }
    pub fn check(&self, expected: &binding::Request) -> Result<()> {
        if self.request != *expected {
            return Err("stage original request differs".into());
        }
        match (&self.report, &self.returned_error) {
            (Some(report), None) => binding::replay(expected, report),
            (None, Some(error)) if !error.is_empty() => {
                if binding::run(expected.clone())
                    .err()
                    .map(|e| e.to_string())
                    .as_ref()
                    != Some(error)
                {
                    return Err("returned stage error does not replay".into());
                }
                Ok(())
            }
            _ => Err("stage must contain exactly one report or returned error".into()),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Stage {
    pub prime: u16,
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    pub retained: Value,
}
pub(crate) fn stage(root: &Path, index: usize, attempt: &Attempt) -> Result<Stage> {
    let prime = attempt.request.discovery.prime;
    let file = format!("stage-{index:02}-p{prime}.json");
    let retained = if let Some(report) = &attempt.report {
        let endpoints: Vec<_> = report.endpoints.iter().map(|e| Ok(json!({
            "mode":e.training.policy.mode,"train_ce":e.audit.train_ce,
            "model_sha256":hash(&serde_json::to_vec(&e.training.model)?),"audit":e.audit,
            "training_work":e.training.work,"training_forecast":e.training.scalar_work_forecast,
            "certificate_work":e.certificate.work,"code_certified":e.certificate.certified,
            "certified_queries":e.certificate.queries.iter().filter(|q|q.decision==certificate::Decision::Passed).count(),
            "certificate_reason":e.certificate.reason
        }))).collect::<Result<_>>()?;
        json!({"discovery_reason":report.discovery.reason,"discovery_work":report.discovery.work,
            "code_admitted":report.discovery.code.is_some(),"code_fibers":report.discovery.code_fibers,
            "initializer_work":report.initialization.as_ref().map(|i|json!({
                "observed_increments":i.observed_increments,"logarithms":i.logarithms,
                "phase_cells":i.phase_cells,"ce_gradients":i.ce_gradients})),
            "candidate_readout_compatible":report.candidate_readout_compatible,
            "raw_label_identification":false,"target_readout_compatibility_proved":false,
            "buffer_preflight_bytes":report.buffer_preflight_bytes,"endpoints":endpoints})
    } else {
        json!({"returned_error":attempt.returned_error,"endpoints":[]})
    };
    Ok(Stage {
        prime,
        sha256: digest(&root.join(&file))?,
        bytes: fs::metadata(root.join(&file))?.len(),
        file,
        retained,
    })
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Ledger {
    pub metadata: Value,
    pub stages: Vec<Stage>,
    pub seconds_per_stage: Vec<f64>,
    pub selected_stages: [Option<usize>; 3],
}
pub(crate) fn select(stages: &[Stage]) -> Result<[Option<usize>; 3]> {
    let mut chosen = [None; 3];
    let mut best = [f64::INFINITY; 3];
    for (index, stage) in stages.iter().enumerate() {
        let endpoints = stage.retained["endpoints"]
            .as_array()
            .ok_or("missing endpoint ledger")?;
        if !endpoints.is_empty() && endpoints.len() != 3 {
            return Err("stage lost a training mode".into());
        }
        for (mode, e) in endpoints.iter().enumerate() {
            if e["mode"]
                != serde_json::to_value(
                    [
                        training::Mode::Full,
                        training::Mode::Mini,
                        training::Mode::Forward,
                    ][mode],
                )?
            {
                return Err("stage modes reordered".into());
            }
            let ce = e["train_ce"]
                .as_f64()
                .filter(|ce| ce.is_finite() && *ce >= 0.)
                .ok_or("invalid actual training CE")?;
            if ce < best[mode] - 1e-12 {
                chosen[mode] = Some(index);
                best[mode] = ce;
            }
        }
    }
    Ok(chosen)
}
