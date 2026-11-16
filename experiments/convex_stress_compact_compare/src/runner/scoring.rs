//! Score and independently read back every original record after the global
//! training barrier. No target enters the Predictor interface.
use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub raw: disk::Artifact,
    pub metrics: evaluation::Metrics,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub mode: joint::training::Mode,
    pub refusal: Option<String>,
    pub splits: BTreeMap<String, Split>,
    pub charged_work: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub case: String,
    pub training_seal: disk::Artifact,
    pub endpoints: BTreeMap<String, Endpoint>,
    pub preparation_phases: preparation::Phases,
    pub preparation_work: u64,
}
pub fn validate_score(
    score: &Score,
    staged: &stage::Stage,
    case: &Case,
    seal: &disk::Artifact,
) -> Result<()> {
    stage::validate_stage(staged, case)?;
    memory::score(score)?;
    if score.case != case.name
        || score.training_seal != *seal
        || score.endpoints.len() != model::MODES.len()
        || score.preparation_work != score.preparation_phases.total()?
    {
        return Err("score is not bound to all sealed endpoints".into());
    }
    if staged.endpoints.values().any(|e| e.model.is_some()) {
        if score.preparation_phases != staged.preparation_phases {
            return Err("scoring preparation differs from complete sealed training inputs".into());
        }
    } else if score.preparation_work != 0 {
        return Err("refused endpoints have spurious scoring preparation".into());
    }
    for mode in model::MODES {
        let label = model::name(mode);
        let saved = score.endpoints.get(label).ok_or("missing scoring mode")?;
        let before = &staged.endpoints[label];
        if saved.mode != mode
            || saved.refusal != before.refusal
            || saved.charged_work > work::ENDPOINT_LIMIT
        {
            return Err("scoring policy/refusal differs from training".into());
        }
        if before.model.is_none() {
            if !saved.splits.is_empty() || saved.charged_work != 0 {
                return Err("refused model cannot receive score credit".into());
            }
        } else {
            if saved.splits.len() != 2 {
                return Err("missing original split".into());
            }
            for (name, n) in [
                ("train", case.training_records),
                ("validation", case.validation_records),
            ] {
                let split = saved
                    .splits
                    .get(name)
                    .ok_or("missing original split metrics")?;
                split.metrics.validate(n)?;
                if split.raw.bytes == 0
                    || split.raw.bytes > n as u64 * (2048 + 48 * case.vocabulary as u64)
                {
                    return Err("raw output exceeds its original-row forecast".into());
                }
            }
        }
    }
    Ok(())
}

pub fn run(root: &Path, workspace: &Path, name: &str, replay: bool) -> Result<()> {
    let (inv, guard) = load(root, workspace)?;
    execute(root, name, replay, inv, guard, resources::process_limit()?)
}
pub(crate) fn execute(
    root: &Path,
    name: &str,
    replay: bool,
    inv: Inventory,
    guard: resources::Guard,
    process_limit: u64,
) -> Result<()> {
    let seal = stage::require_seal(root, &inv)?;
    let case = case(&inv, name)?;
    let dir = root.join("cases").join(name);
    let saved: Option<Score> = if replay {
        Some(read(&dir.join("score.json"))?)
    } else {
        None
    };
    let stage: stage::Stage = read(&dir.join("training.json"))?;
    stage::validate_stage(&stage, case)?;
    if let Some(saved) = &saved {
        memory::score(saved)?;
    }
    let mut preparation = preparation::Phases::new()?;
    let mut input = None;
    let mut expected = None;
    let mut dataset = None;
    let prepared = if stage.endpoints.values().any(|e| e.model.is_some()) {
        input = Some(training(root, &inv, name)?);
        expected = Some(selection(root, &inv, name)?);
        let data: Dataset = checked(root, &inv, &format!("inputs/data/{name}.json"))?;
        data.validate(input.as_ref().ok_or("missing training")?, case)?;
        dataset = Some(data);
        Some(preparation::prepare(
            input.as_ref().ok_or("missing training")?,
            expected.as_ref().ok_or("missing selection")?,
            process_limit,
            memory::caller(&inv, &guard)?
                + memory::dataset(dataset.as_ref().ok_or("missing data")?),
            &mut preparation,
        )?)
    } else {
        None
    };
    let mut endpoints = BTreeMap::new();
    for mode in model::MODES {
        guard.check()?;
        let label = model::name(mode);
        let endpoint = &stage.endpoints[label];
        let mut work = work::endpoint(0)?;
        let mut splits = BTreeMap::new();
        if let Some(state) = &endpoint.model {
            let prepared = prepared.as_ref().ok_or("missing supported original data")?;
            let data = dataset.as_ref().ok_or("missing complete dataset")?;
            let (parameters, head) = tensor::load(
                &dir.join(format!("{label}.bin")),
                &prepared.domain,
                &state.tensor,
                state.tensor_body_sha256,
                prepared
                    .budget
                    .model_room(&prepared.domain, &prepared.records)?,
                &mut work,
            )?;
            let remaining = memory::remaining_model(prepared, &parameters, &head)?;
            state.check(&prepared.domain, &parameters, &head, remaining, &mut work)?;
            let mut predictor = model::Predictor::new(
                &head,
                &prepared.domain,
                &prepared.selection,
                case.training_records + case.validation_records,
                remaining,
                &mut work,
            )?;
            for (split, rows) in [
                ("train", &data.training.rows),
                ("validation", &data.validation),
            ] {
                let path = dir.join(format!("{label}-{split}.jsonl"));
                let (raw, metrics) = if let Some(saved) = &saved {
                    let expected = saved
                        .endpoints
                        .get(label)
                        .and_then(|x| x.splits.get(split))
                        .ok_or("missing complete saved endpoint")?;
                    let actual = evaluation::replay(
                        &path,
                        &expected.raw,
                        rows,
                        case.vocabulary,
                        &mut predictor,
                        &mut work,
                    )?;
                    if actual != expected.metrics {
                        return Err("independent raw metric recount differs".into());
                    }
                    (expected.raw.clone(), actual)
                } else {
                    evaluation::score(
                        &path,
                        rows,
                        case.vocabulary,
                        &mut predictor,
                        &mut work,
                        &guard,
                    )?
                };
                if split == "train"
                    && metrics
                        .ordinary_cross_entropy
                        .is_some_and(|ce| (ce - state.final_ce).abs() > 1e-7)
                {
                    return Err("raw observed CE differs from sealed training objective".into());
                }
                splits.insert(split.into(), Split { raw, metrics });
            }
        }
        endpoints.insert(
            label.into(),
            Endpoint {
                mode,
                refusal: endpoint.refusal.clone(),
                splits,
                charged_work: work.used,
            },
        );
    }
    let score = Score {
        case: name.into(),
        training_seal: seal,
        endpoints,
        preparation_work: preparation.total()?,
        preparation_phases: preparation,
    };
    validate_score(&score, &stage, case, &score.training_seal)?;
    // Keep the original input/selection alive through every nested model call;
    // their capacities remain part of Prepared::budget until this point.
    drop((input, expected, dataset, prepared));
    if let Some(saved) = saved {
        if saved != score {
            return Err("complete scoring replay differs".into());
        }
        disk::json(
            &dir.join("replay.json"),
            &json!({"case":name,"score":disk::hash(&dir.join("score.json"))?,
            "all_original_rows_recounted":true,"complete_head_predictions_recomputed":true,
            "target_realizability_certified":false}),
            JSON_LIMIT,
            &guard,
        )?;
    } else {
        disk::json(&dir.join("score.json"), &score, JSON_LIMIT, &guard)?;
    }
    Ok(())
}
