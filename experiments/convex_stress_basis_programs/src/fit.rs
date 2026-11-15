//! Fit every original record with fixed discovery, decoder prior, initializer
//! and anchor policy. Nothing in this module reads heldout or clean-label pools.
use crate::{
    discovery::Discovery,
    model::{Model, Prior},
    *,
};
use basis::data::Example;
use joint::training::{Mode, Policy};
use serde::{Deserialize, Serialize};

pub fn policies(exact: bool) -> Result<Vec<Policy>> {
    let p: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    let f = &p["fit"];
    Ok([Mode::Forward, Mode::Full, Mode::Mini]
        .into_iter()
        .map(|mode| Policy {
            mode,
            epochs: match mode {
                Mode::Forward => 0,
                Mode::Full => f["full_epochs"].as_u64().unwrap() as usize,
                Mode::Mini => f["mini_epochs"].as_u64().unwrap() as usize,
            },
            batch: f["batch"].as_u64().unwrap() as usize,
            learning_rate: f[if exact {
                "exact_learning_rate"
            } else {
                "relaxed_learning_rate"
            }]
            .as_f64()
            .unwrap(),
            initial_total_mass: f[if exact { "exact_mass" } else { "relaxed_mass" }]
                .as_f64()
                .unwrap(),
            projection_cycles: f["projection_cycles"].as_u64().unwrap() as usize,
            projection_tolerance: f["projection_tolerance"].as_f64().unwrap(),
        })
        .collect())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub prior: Prior,
    pub policy: Policy,
    pub model: Option<Model>,
    pub training: Option<weighted::training::Outcome>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub discovery: Discovery,
    pub endpoints: Vec<Endpoint>,
    pub corpus_bindings: Vec<serde_json::Value>,
    pub scalar_forecast: u64,
    pub buffer_forecast: u64,
    pub complete_scalar_forecast: u64,
    pub scoring_forecast: Option<evaluation::Forecast>,
    pub all_anchor_diagnostics: Vec<serde_json::Value>,
}

pub fn construct(vocabulary: usize, rows: &[Example]) -> Result<Stage> {
    construct_bounded(vocabulary, rows, None)
}
pub fn construct_bounded(
    vocabulary: usize,
    rows: &[Example],
    scoring: Option<evaluation::Forecast>,
) -> Result<Stage> {
    let discovery = discovery::run(vocabulary, rows)?;
    let selected = discovery
        .selected
        .as_ref()
        .ok_or("fixed bank has no defined training-compatible candidate")?;
    let policies = policies(selected.exact)?;
    let mut scalar = discovery.scalar_forecast as u128;
    let mut bytes = discovery.buffer_forecast as u128 + 8 * 1024u128.pow(3);
    let mut inputs = vec![];
    let mut corpus_bindings = vec![];
    for prior in [Prior::Coordinates, Prior::Categorical] {
        let corpus = model::corpus(&discovery, rows, prior)?;
        let spec = model::specification(&discovery, rows, &corpus, prior)?;
        let mut arm_bytes = 0;
        for policy in &policies {
            let f = weighted::training::forecast(&corpus, policy)?;
            scalar += f.scalar_work as u128;
            arm_bytes += f.buffer_bytes as u128;
        }
        if !selected.exact {
            scalar += 2048 * corpus.groups().len() as u128
                + 128 * (16384 + 64 * corpus.groups().len() as u128);
        }
        bytes += arm_bytes;
        corpus_bindings.push(serde_json::json!({"prior":prior,"sha256":corpus.fingerprint()?,
            "records":corpus.limits().records,"groups":corpus.groups().len(),"limits":corpus.limits(),"anchors":spec.anchors}));
        inputs.push((prior, corpus, spec));
    }
    // Include the subsequent independently regenerated complete construction.
    let complete_scalar = 2 * scalar + 24 * scoring.as_ref().map_or(0, |f| f.scalar_work as u128);
    bytes =
        bytes.max(12 * 1024u128.pow(3) + scoring.as_ref().map_or(0, |f| f.buffer_bytes as u128));
    if complete_scalar > 600_000_000_000_000 || bytes > joint::MAX_BYTES {
        return Err("complete six-endpoint weighted fit forecast refused".into());
    }
    let mut endpoints = vec![];
    let mut all_anchor_diagnostics = vec![];
    for (prior, corpus, spec) in inputs {
        if !selected.exact {
            let mut full = spec.clone();
            full.anchors = weighted::domain::Anchors::All;
            let result = weighted::domain::Domain::new(&corpus, &full).and_then(|d| {
                d.project(
                    &[[0., 0., 0., 0., 0., policies[0].initial_total_mass]],
                    128,
                    policies[0].projection_tolerance,
                )
            });
            all_anchor_diagnostics.push(serde_json::json!({"prior":prior,"projection_cycles_limit":128,
                "feasible":result.is_ok(),"error":result.err().map(|e|e.to_string()),"all_observed_groups":corpus.groups().len()}));
        }
        for policy in &policies {
            let unit = if prior == Prior::Coordinates {
                selected.unit
            } else {
                [1., 0., 0., 0., 1., 1.]
            };
            let initial = vec![unit.map(|x| x * policy.initial_total_mass)];
            let result = weighted::training::fit_from(&corpus, &spec, policy.clone(), initial);
            let (model, training, error) = match result {
                Ok(training) => {
                    let model = Model {
                        prior,
                        programs: vec![selected.clone()],
                        head: training.head.clone(),
                        feature_bases: training.domain.constraints().feature_basis.clone(),
                    };
                    model.validate()?;
                    (Some(model), Some(training), None)
                }
                Err(e) => (None, None, Some(e.to_string())),
            };
            endpoints.push(Endpoint {
                prior,
                policy: policy.clone(),
                model,
                training,
                error,
            });
        }
    }
    Ok(Stage {
        discovery,
        endpoints,
        corpus_bindings,
        scalar_forecast: scalar as u64,
        buffer_forecast: bytes as u64,
        complete_scalar_forecast: complete_scalar as u64,
        scoring_forecast: scoring,
        all_anchor_diagnostics,
    })
}
