//! Seal all original training endpoints before any heldout scoring. Only
//! pure model failures become refusals; filesystem/resource errors propagate.
use super::*;
use joint::{Work, training::Mode};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub mode: Mode,
    pub model: Option<model::State>,
    pub refusal: Option<String>,
    pub charged_work: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub case: String,
    pub support: usize,
    pub preparation_work: u64,
    pub heldout_pools_read: usize,
    pub endpoints: BTreeMap<String, Endpoint>,
}
pub fn run(root: &Path, workspace: &Path, name: &str) -> Result<()> {
    let (inv, guard) = load(root, workspace)?;
    if root.join("training-seal.json").exists() {
        return Err("training is already sealed".into());
    }
    let case = case(&inv, name)?;
    let dir = root.join("cases").join(name);
    fs::create_dir(&dir)?;
    let mut preparation = Work::new(joint::MAX_WORK)?;
    let prepared = if case.source_refusal.is_none() && case.support > 0 {
        Some(model::prepare(
            &training(root, &inv, name)?,
            &selection(root, &inv, name)?,
            &mut preparation,
        )?)
    } else {
        None
    };
    if let Some(p) = &prepared
        && Some(p.domain.shape) != case.shape
    {
        return Err("actual model allocation differs from forecast".into());
    }
    let mut endpoints = BTreeMap::new();
    for mode in model::MODES {
        guard.check()?;
        let reserve = case.shape.map_or(0, |sh| {
            32 * (sh.parameters() as u64 + 2 * sh.cells() as u64)
        });
        let mut work = Work::new(joint::MAX_WORK - reserve)?;
        let endpoint = if let Some(p) = &prepared {
            match joint::training::train(&p.domain, &p.records, pairs::policy(mode), &mut work) {
                Ok(trained) => {
                    work.limit = joint::MAX_WORK;
                    let residual = p.domain.check(&trained.parameters, 1e-8)?;
                    let tensor = tensor::save(
                        &dir.join(format!("{}.bin", model::name(mode))),
                        &trained.parameters,
                        &trained.head,
                        &guard,
                        &mut work,
                    )?;
                    Endpoint {
                        mode,
                        model: Some(model::State::from_trained(
                            &trained, tensor, work.used, residual,
                        )),
                        refusal: None,
                        charged_work: work.used,
                    }
                }
                Err(error) => Endpoint {
                    mode,
                    model: None,
                    refusal: Some(error.to_string()),
                    charged_work: work.used,
                },
            }
        } else {
            Endpoint {mode,model:None,refusal:Some(case.source_refusal.clone().unwrap_or_else(||"empty observationally compatible support; all fixed candidates have checked collision witnesses".into())),charged_work:0}
        };
        endpoints.insert(model::name(mode).into(), endpoint);
    }
    disk::json(
        &dir.join("training.json"),
        &Stage {
            case: name.into(),
            support: case.support,
            preparation_work: preparation.used,
            heldout_pools_read: 0,
            endpoints,
        },
        JSON_LIMIT,
        &guard,
    )?;
    Ok(())
}
pub fn validate_stage(value: &Stage, case: &Case) -> Result<()> {
    if value.case != case.name
        || value.support != case.support
        || value.heldout_pools_read != 0
        || value.preparation_work > joint::MAX_WORK
        || value.endpoints.len() != 3
    {
        return Err("complete training stage changed".into());
    }
    for mode in model::MODES {
        let endpoint = value
            .endpoints
            .get(model::name(mode))
            .ok_or("missing training endpoint")?;
        if endpoint.mode != mode || endpoint.charged_work > joint::MAX_WORK {
            return Err("endpoint policy/work changed".into());
        }
        match (&endpoint.model, &endpoint.refusal) {
            (Some(state), None)
                if case.source_refusal.is_none()
                    && case.shape == Some(state.shape)
                    && state.config == pairs::policy(mode)
                    && state.total_operations == endpoint.charged_work =>
            {
                let batch = if mode == Mode::Mini {
                    state.config.batch_size
                } else {
                    case.training_records
                };
                if state.steps.len() != state.config.epochs * case.training_records.div_ceil(batch)
                {
                    return Err("actual update/short-tail count changed".into());
                }
                for (index, step) in state.steps.iter().enumerate() {
                    let per_epoch = case.training_records.div_ceil(batch);
                    let first = index % per_epoch * batch;
                    if step.epoch != index / per_epoch
                        || step.first_record != first
                        || step.end_record_exclusive != (first + batch).min(case.training_records)
                        || !step.before_ce.is_finite()
                        || !step.after_ce.is_finite()
                        || step.after_ce > step.before_ce + 1e-8
                        || step.maximum_feasibility_residual > 1e-8
                        || step.maximum_kkt_residual > 1e-7
                    {
                        return Err("actual original-record training trace changed".into());
                    }
                }
            }
            (None, Some(reason)) if !reason.is_empty() => {}
            _ => return Err("model/refusal accounting changed".into()),
        }
    }
    Ok(())
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Seal {
    pub inventory: disk::Artifact,
    pub cases: usize,
    pub endpoints: usize,
    pub heldout_pools_scored: usize,
    pub training: BTreeMap<String, disk::Artifact>,
}
fn collect(root: &Path, inv: &Inventory, hash_tensors: bool) -> Result<Seal> {
    let mut files = BTreeMap::new();
    for case in &inv.cases {
        let file = format!("cases/{}/training.json", case.name);
        let value: Stage = read(&root.join(&file))?;
        validate_stage(&value, case)?;
        files.insert(file.clone(), disk::hash(&root.join(file))?);
        for endpoint in value.endpoints.values() {
            if let Some(state) = &endpoint.model {
                let file = format!("cases/{}/{}.bin", case.name, model::name(endpoint.mode));
                if state.tensor.bytes != tensor::length(state.shape)?
                    || fs::metadata(root.join(&file))?.len() != state.tensor.bytes
                {
                    return Err("sealed tensor shape/length changed".into());
                }
                if hash_tensors && disk::hash(&root.join(&file))? != state.tensor {
                    return Err("training tensor hash changed before seal".into());
                }
                files.insert(file, state.tensor.clone());
            }
        }
    }
    Ok(Seal {
        inventory: disk::hash(&root.join("inventory.json"))?,
        cases: inv.cases.len(),
        endpoints: 3 * inv.cases.len(),
        heldout_pools_scored: 0,
        training: files,
    })
}
pub fn seal(root: &Path, workspace: &Path) -> Result<()> {
    let (inv, guard) = load(root, workspace)?;
    create_seal(root, &inv, &guard)
}
pub(crate) fn create_seal(root: &Path, inv: &Inventory, guard: &resources::Guard) -> Result<()> {
    let sealed = collect(root, inv, true)?;
    disk::json(&root.join("training-seal.json"), &sealed, JSON_LIMIT, guard)?;
    Ok(())
}
pub fn require_seal(root: &Path, inv: &Inventory) -> Result<disk::Artifact> {
    let saved: Seal = read(&root.join("training-seal.json"))?;
    if saved != collect(root, inv, false)? {
        return Err("complete pre-heldout training seal changed".into());
    }
    // Each endpoint's full tensor hash is also checked immediately before use.
    disk::hash(&root.join("training-seal.json"))
}
