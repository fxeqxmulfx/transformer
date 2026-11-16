//! Seal all original training endpoints before any heldout scoring. Only
//! pure model failures become refusals; filesystem/resource errors propagate.
use super::*;
use joint::training::Mode;

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
    pub preparation_phases: preparation::Phases,
    pub heldout_pools_read: usize,
    pub endpoints: BTreeMap<String, Endpoint>,
}
pub fn run(root: &Path, workspace: &Path, name: &str) -> Result<()> {
    let (inv, guard) = load(root, workspace)?;
    execute(root, name, inv, guard, resources::process_limit()?)
}
pub(crate) fn execute(
    root: &Path,
    name: &str,
    inv: Inventory,
    guard: resources::Guard,
    process_limit: u64,
) -> Result<()> {
    if root.join("training-seal.json").exists() {
        return Err("training is already sealed".into());
    }
    let case = case(&inv, name)?;
    let dir = root.join("cases").join(name);
    fs::create_dir(&dir)?;
    let mut preparation = preparation::Phases::new()?;
    let input = if case.source_refusal.is_none() && case.forecast.is_some() {
        Some((training(root, &inv, name)?, selection(root, &inv, name)?))
    } else {
        None
    };
    let prepared = if case.source_refusal.is_none() && case.forecast.is_some() {
        let (training, expected) = input.as_ref().ok_or("missing original input")?;
        match preparation::prepare(
            training,
            expected,
            process_limit,
            memory::caller(&inv, &guard)?,
            &mut preparation,
        ) {
            Ok(p) => Some(p),
            Err(error) => {
                disk::json(
                    &dir.join("preparation-failure.json"),
                    &json!({"case":name,"preparation_phases":preparation,
                        "charged_work":preparation.total()?,"reason":error.to_string(),
                        "heldout_pools_read":0}),
                    JSON_LIMIT,
                    &guard,
                )?;
                return Err(error);
            }
        }
    } else {
        None
    };
    if let Some(p) = &prepared
        && (Some(p.domain.shape) != case.shape
            || Some(u64::try_from(
                joint::domain::allocation_bytes(&p.domain) + p.records.allocation_bytes(),
            )?) != case.compact_input_bytes)
    {
        return Err("actual model allocation differs from forecast".into());
    }
    let mut endpoints = BTreeMap::new();
    for mode in model::MODES {
        guard.check()?;
        let reserve = prepared.as_ref().map_or(0, |p| {
            200 * joint::persistence::bytes(&p.domain)
                + 1000 * p.domain.head_columns_bound as u64
                + 512 * p.domain.shape.rows() as u64
                + 128 * p.domain.parameters() as u64
        });
        let mut work = work::endpoint(reserve)?;
        let endpoint = if let Some(p) = &prepared {
            match p
                .budget
                .train(&p.domain, &p.records, pairs::policy(mode), &mut work)
            {
                Ok(trained) => {
                    work::release_reserve(&mut work)?;
                    let residual = trained.parameters.check(&p.domain, 1e-8, &mut work)?;
                    let tensor = tensor::save(
                        &dir.join(format!("{}.bin", model::name(mode))),
                        &p.domain,
                        &trained.parameters,
                        &trained.head,
                        &guard,
                        memory::remaining_model(p, &trained.parameters, &trained.head)?,
                        &mut work,
                    )?;
                    Endpoint {
                        mode,
                        model: Some(model::State::from_trained(
                            trained, &p.domain, tensor, work.used, residual,
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
            Endpoint {
                mode,
                model: None,
                refusal: Some(
                    case.source_refusal
                        .clone()
                        .or_else(|| case.input_refusal.clone())
                        .ok_or("unsupported case has no explicit source or model refusal")?,
                ),
                charged_work: 0,
            }
        };
        endpoints.insert(model::name(mode).into(), endpoint);
    }
    let value = Stage {
        case: name.into(),
        support: case.support,
        preparation_work: preparation.total()?,
        preparation_phases: preparation,
        heldout_pools_read: 0,
        endpoints,
    };
    validate_stage(&value, case)?;
    disk::json(&dir.join("training.json"), &value, JSON_LIMIT, &guard)?;
    Ok(())
}
pub fn validate_stage(value: &Stage, case: &Case) -> Result<()> {
    memory::stage(value)?;
    if value.case != case.name
        || value.support != case.support
        || value.heldout_pools_read != 0
        || value.preparation_work != value.preparation_phases.total()?
        || value.endpoints.len() != model::MODES.len()
    {
        return Err("complete training stage changed".into());
    }
    if case.source_refusal.is_none() && case.forecast.is_some() {
        if Some(value.preparation_phases.selection.used) != case.selection_operations
            || value.preparation_phases.materialization.used == 0
        {
            return Err("training omitted a complete preparation pass".into());
        }
    } else if value.preparation_work != 0 {
        return Err("unprepared input has spurious preparation charges".into());
    }
    for mode in model::MODES {
        let endpoint = value
            .endpoints
            .get(model::name(mode))
            .ok_or("missing training endpoint")?;
        if endpoint.mode != mode || endpoint.charged_work > work::ENDPOINT_LIMIT {
            return Err("endpoint policy/work changed".into());
        }
        match (&endpoint.model, &endpoint.refusal) {
            (Some(state), None)
                if case.source_refusal.is_none()
                    && case.forecast.is_some()
                    && case.compact_input_bytes.is_some()
                    && case.input_refusal.is_none()
                    && case.shape == Some(state.shape)
                    && case
                        .audit
                        .as_ref()
                        .is_some_and(|a| a.unknown_rows == state.unknown_rows)
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
                if state.tensor.bytes != tensor::length(state.shape, state.unknown_rows)?
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
        endpoints: model::MODES.len() * inv.cases.len(),
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
