use super::*;
use crate::model::{Model, Prior};

pub fn goals(metric: &Value) -> Result<(bool, bool)> {
    let targets = metric["targets"]
        .as_u64()
        .ok_or("missing raw denominator")?;
    let correct = metric["correct"].as_u64().ok_or("missing raw numerator")?;
    let examples = metric["examples"]
        .as_u64()
        .ok_or("missing exact denominator")?;
    let exact = metric["exact"].as_u64().ok_or("missing exact numerator")?;
    if targets == 0 || examples == 0 || correct > targets || exact > examples {
        return Err("invalid measured denominators".into());
    }
    Ok((
        100u128 * correct as u128 >= 99u128 * targets as u128,
        100u128 * exact as u128 >= 99u128 * examples as u128,
    ))
}

pub fn score(root: &Path) -> Result<()> {
    let inv = inventory(root)?;
    seal(root)?;
    for case in cases() {
        let prepared = read_prepared(root, &case)?;
        let data = rebuild(&inv, &case)?;
        if prepared.metadata != data_metadata(&inv, &case)? {
            return Err("prepared case metadata changed".into());
        }
        let mut endpoints = vec![];
        if let Some(stage) = &prepared.stage {
            if prepared.error.is_some() || stage.endpoints.len() != 6 {
                return Err("inconsistent prepared stage".into());
            }
            for (arm, prior) in [Prior::Coordinates, Prior::Categorical]
                .into_iter()
                .enumerate()
            {
                let corpus = model::corpus(&stage.discovery, &data.task.splits["train"], prior)?;
                let binding = corpus.fingerprint()?;
                let specification = model::specification(
                    &stage.discovery,
                    &data.task.splits["train"],
                    &corpus,
                    prior,
                )?;
                for (index, mode) in [
                    joint::training::Mode::Forward,
                    joint::training::Mode::Full,
                    joint::training::Mode::Mini,
                ]
                .into_iter()
                .enumerate()
                {
                    let endpoint = &stage.endpoints[3 * arm + index];
                    if endpoint.prior != prior || endpoint.policy.mode != mode {
                        return Err("endpoint order or mode changed".into());
                    }
                    if let Some(error) = &endpoint.error {
                        if endpoint.model.is_some() || endpoint.training.is_some() {
                            return Err("refused endpoint contains trained state".into());
                        }
                        endpoints.push(
                            json!({"prior":prior,"mode":mode,"error":error,"evaluation":null,
                            "original_success":false,"all_heldout99":false,"training":null}),
                        );
                        continue;
                    }
                    let training = endpoint
                        .training
                        .as_ref()
                        .ok_or("missing joint training trace")?;
                    let model = endpoint.model.as_ref().ok_or("missing complete model")?;
                    let expected = Model {
                        prior,
                        programs: vec![
                            stage
                                .discovery
                                .selected
                                .clone()
                                .ok_or("missing selected program")?,
                        ],
                        head: training.head.clone(),
                        feature_bases: training.domain.constraints().feature_basis.clone(),
                    };
                    if *model != expected
                        || training.specification != specification
                        || training.policy != endpoint.policy
                        || training.corpus_sha256 != binding
                        || training.original_records != stage.discovery.input.records
                        || training.domain
                            != weighted::domain::Domain::new(&corpus, &specification)?
                    {
                        return Err(
                            "saved model, selected anchors or original training binding differs"
                                .into(),
                        );
                    }
                    let file = format!("{case}-{prior:?}-{mode:?}.raw.jsonl");
                    let raw_path = root.join(&file);
                    let mut evaluation = evaluation::evaluate(
                        model,
                        &stage.discovery.input,
                        &corpus,
                        &data.task.splits,
                        &raw_path,
                    )?;
                    let readback = crate::audit::verify(
                        model,
                        &stage.discovery.input,
                        &corpus,
                        &data.task.splits,
                        &raw_path,
                        &evaluation,
                    )?;
                    evaluation["readback"] = readback;
                    let splits = evaluation["splits"]
                        .as_object()
                        .ok_or("missing complete measured splits")?;
                    if splits.len() != data.task.splits.len()
                        || !splits.contains_key(&data.task.selection)
                    {
                        return Err("incomplete original split evaluation".into());
                    }
                    let mut heldout = 0;
                    let mut all_heldout99 = true;
                    for (name, value) in splits {
                        let (raw, exact) = goals(&value["metrics"])?;
                        if !name.starts_with("train") {
                            heldout += 1;
                            all_heldout99 &= raw && exact;
                        }
                    }
                    if heldout == 0 {
                        return Err("empty heldout criterion".into());
                    }
                    let original_success = goals(&splits[&data.task.selection]["metrics"])?.1;
                    endpoints.push(json!({"prior":prior,"mode":mode,"error":null,"evaluation":evaluation,
                        "original_success":original_success,"all_heldout99":all_heldout99,"heldout_splits":heldout,
                        "training":{"work":training.work,"initial_ce":training.initial_ce,"final_ce":training.final_ce,
                            "initial_proposal":training.initial_proposal,"policy":training.policy,
                            "original_records":training.original_records,"distinct_groups":training.distinct_groups,
                            "corpus_sha256":training.corpus_sha256,"anchors":training.specification.anchors,
                            "final_normalized_violation":training.domain.violation(&training.head.theta),
                            "complete_max_column_charge":training.head.max_column_charge(),"compact_charge":joint::charge(&training.head.theta)}}));
                }
            }
        } else {
            let error = prepared
                .error
                .as_ref()
                .ok_or("missing whole-case refusal")?;
            for prior in [Prior::Coordinates, Prior::Categorical] {
                for mode in [
                    joint::training::Mode::Forward,
                    joint::training::Mode::Full,
                    joint::training::Mode::Mini,
                ] {
                    endpoints.push(
                        json!({"prior":prior,"mode":mode,"error":error,"evaluation":null,
                        "original_success":false,"all_heldout99":false,"training":null}),
                    );
                }
            }
        }
        let result = json!({"case":case,"data":prepared.metadata,"original_selection":data.task.selection,
            "stage_sha256":artifacts::digest(&root.join(format!("{case}.stage.json")))?,"error":prepared.error,
            "all_anchor_diagnostics":prepared.stage.as_ref().map(|s|&s.all_anchor_diagnostics),
            "corpus_bindings":prepared.stage.as_ref().map(|s|&s.corpus_bindings),
            "complete_scalar_forecast":prepared.stage.as_ref().map(|s|s.complete_scalar_forecast),
            "buffer_forecast":prepared.stage.as_ref().map(|s|s.buffer_forecast),"endpoints":endpoints});
        artifacts::write(&root.join(format!("{case}.metrics.json")), &result)?;
        println!("{case}: all original splits scored");
    }
    Ok(())
}
