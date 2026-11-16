//! Full denominators and inherited historical scope remain explicit. New
//! catalog credit requires every mode on the original heldout split.
use super::*;
fn passing(score: &scoring::Score, mode: joint::training::Mode) -> bool {
    score
        .endpoints
        .get(model::name(mode))
        .and_then(|x| x.splits.get("validation"))
        .is_some_and(|s| s.metrics.raw99 && s.metrics.exact99 && s.metrics.original_success)
}
pub fn build(root: &Path, inv: &Inventory) -> Result<Value> {
    let seal = stage::require_seal(root, inv)?;
    let data: Value = checked(root, inv, "inputs/data-inventory.json")?;
    let baseline: Value = checked(root, inv, "inputs/original35-metrics.json")?;
    let old_verification: Value =
        checked(root, inv, "inputs/catalog77-historical-verification.json")?;
    if old_verification["catalog_all_modes_all_heldout99_cases"] != 77
        || old_verification["catalog_total_cases"] != 165
    {
        return Err("historical coverage changed".into());
    }
    let mut records = vec![];
    let mut scores = vec![];
    let (mut additions, mut nonconstant, mut constants, mut regressions) =
        (vec![], vec![], vec![], vec![]);
    for case in &inv.cases {
        let dir = root.join("cases").join(&case.name);
        let score: scoring::Score = read(&dir.join("score.json"))?;
        let staged: stage::Stage = read(&dir.join("training.json"))?;
        scoring::validate_score(&score, &staged, case, &seal)?;
        let replay: Value = read(&dir.join("replay.json"))?;
        if score.case != case.name
            || score.training_seal != seal
            || score.endpoints.len() != 3
            || replay["case"] != case.name
            || replay["score"] != serde_json::to_value(disk::hash(&dir.join("score.json"))?)?
            || replay["all_original_rows_recounted"] != true
            || replay["complete_head_predictions_recomputed"] != true
            || replay["target_realizability_certified"] != false
        {
            return Err("complete original scoring/replay is missing or changed".into());
        }
        let all = model::MODES.iter().all(|&m| passing(&score, m));
        if all && !case.existing35 {
            additions.push(case.name.clone());
            if case.constant_control {
                constants.push(case.name.clone());
            } else {
                nonconstant.push(case.name.clone());
            }
        }
        if !all && case.existing35 {
            regressions.push(case.name.clone());
        }
        let historical = data["cases"]
            .as_array()
            .ok_or("history")?
            .iter()
            .find(|r| r["case"] == case.name)
            .ok_or("missing historical case")?;
        let mut changes = vec![];
        if case.existing35 {
            for arm in baseline.as_array().ok_or("both baseline policies")? {
                for mode in model::MODES {
                    let before = arm["choices"]
                        .as_array()
                        .ok_or("baseline choices")?
                        .iter()
                        .find(|r| {
                            r["task"] == case.name
                                && r["training_mode"] == model::original_mode(mode)
                        })
                        .ok_or("missing original baseline endpoint")?;
                    let after = &score.endpoints[model::name(mode)];
                    changes.push(json!({"initializer":arm["initializer"],"mode":mode,"before":before,
                        "after_original_success":passing(&score,mode),"after_validation":after.splits.get("validation").map(|x|&x.metrics),
                        "after_refusal":after.refusal}));
                }
            }
        }
        records.push(json!({"case":case,"all_modes_original_heldout99":all,"score":score,
            "historical_transformer":historical["historical_gpt"],"changes_from_original35":changes}));
        scores.push(score);
    }
    let modes:Vec<_>=model::MODES.iter().map(|&mode| {
        let endpoints:Vec<_>=scores.iter().map(|s|&s.endpoints[model::name(mode)]).collect();
        let measured:Vec<_>=endpoints.iter().filter_map(|e|e.splits.get("validation")).collect();
        let available_targets:usize=inv.cases.iter().map(|c|c.validation_records).sum();
        let correct:usize=measured.iter().map(|s|s.metrics.raw_correct).sum();
        let measured_targets:usize=measured.iter().map(|s|s.metrics.targets).sum();
        json!({"mode":mode,"original_cases":54,"available_datasets":48,"source_unavailable":6,
            "unscored_endpoints":endpoints.iter().filter(|e|e.refusal.is_some()).count(),
            "empty_compatible_support":inv.cases.iter().filter(|c|c.source_refusal.is_none()&&c.support==0).count(),
            "training_refusals":inv.cases.iter().zip(&endpoints).filter(|(c,e)|c.support>0&&e.refusal.is_some()).count(),
            "original_successes":scores.iter().filter(|s|passing(s,mode)).count(),
            "raw99":measured.iter().filter(|s|s.metrics.raw99).count(),"exact99":measured.iter().filter(|s|s.metrics.exact99).count(),
            "available_original_validation_targets":available_targets,"measured_validation_targets":measured_targets,
            "raw_correct":correct,"accuracy_with_unanswered_available_targets_counted_wrong":correct as f64/available_targets as f64,
            "prediction_refusals":measured.iter().map(|s|s.metrics.refusals).sum::<usize>()})
    }).collect();
    Ok(
        json!({"complete":true,"producer_commit":inv.producer_commit,"protocol":inv.protocol,"data_seed":1,
        "cases":records,"modes":modes,"new_all_modes_all_heldout99_cases":additions,
        "legacy_probe_memory_refusals":inv.cases.iter().filter(|c|c.support>0&&c.audit.as_ref().is_some_and(|a|a.complete_buffer_forecast.is_none())).count(),
        "packed_comparison_memory_refusals":inv.cases.iter().filter(|c|c.support>0&&c.comparison_refusal.is_some()).count(),
        "complete_packed_models_admitted":inv.cases.iter().filter(|c|c.comparison_buffer_bytes.is_some()).count(),
        "new_nonconstant_cases":nonconstant,"new_constant_controls":constants,"regressed_old_controls":regressions,
        "baseline_historical_verified_catalog_cases":77,"candidate_catalog_union_cases":77+additions.len(),
        "candidate_catalog_with_revised_cohort":77+additions.len()-regressions.len(),"catalog_total_cases":165,
        "independent_repeat_required":true,"original_transformer_retrained":false,"target_realizability_certified":false,
        "scope":"Original seed1 arithmetic cohort; all54 cases retained. Coverage unions separately frozen historical cohorts; it is not one universal checkpoint or three-seed evidence. Historical raw outputs were deleted at user request after summary preservation."}),
    )
}
fn collect(root: &Path, path: &Path, files: &mut BTreeMap<String, disk::Artifact>) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err("artifact symlink is not allowed".into());
    }
    if metadata.is_dir() {
        for child in fs::read_dir(path)? {
            collect(root, &child?.path(), files)?;
        }
    } else {
        let key = path
            .strip_prefix(root)?
            .to_str()
            .ok_or("artifact path encoding")?
            .to_string();
        files.insert(key, disk::hash(path)?);
    }
    Ok(())
}
pub fn manifest(root: &Path) -> Result<BTreeMap<String, disk::Artifact>> {
    let mut files = BTreeMap::new();
    for path in [
        "inputs",
        "cases",
        "inventory.json",
        "training-seal.json",
        "report.json",
    ] {
        collect(root, &root.join(path), &mut files)?;
    }
    Ok(files)
}
fn bindings(root: &Path, inv: &Inventory, files: &BTreeMap<String, disk::Artifact>) -> Result<()> {
    let seal: stage::Seal = read(&root.join("training-seal.json"))?;
    for (name, artifact) in inv.inputs.iter().chain(&seal.training) {
        if files.get(name) != Some(artifact) {
            return Err("current input/training artifact differs from its seal".into());
        }
    }
    if files.get("inventory.json") != Some(&seal.inventory) {
        return Err("inventory differs from pre-heldout seal".into());
    }
    for case in &inv.cases {
        let score: scoring::Score = read(&root.join(format!("cases/{}/score.json", case.name)))?;
        for (mode, endpoint) in &score.endpoints {
            for (split, saved) in &endpoint.splits {
                let name = format!("cases/{}/{mode}-{split}.jsonl", case.name);
                if files.get(&name) != Some(&saved.raw) {
                    return Err("current raw output differs from its recounted score".into());
                }
            }
        }
    }
    Ok(())
}
pub fn run(root: &Path, workspace: &Path) -> Result<()> {
    let (inv, guard) = load(root, workspace)?;
    let result = build(root, &inv)?;
    disk::json(&root.join("report.json"), &result, JSON_LIMIT, &guard)?;
    let files = manifest(root)?;
    bindings(root, &inv, &files)?;
    disk::json(&root.join("manifest.json"), &files, JSON_LIMIT, &guard)?;
    Ok(())
}
pub fn verify(workspace: &Path, original: &Path, repeated: &Path, output: &Path) -> Result<()> {
    let mut manifests = vec![];
    for root in [original, repeated] {
        let inv: Inventory = read(&root.join("inventory.json"))?;
        if inv.protocol != protocol()? {
            return Err("verification producer protocol differs".into());
        }
        inputs::validate(root, &inv)?;
        let saved: BTreeMap<String, disk::Artifact> = read(&root.join("manifest.json"))?;
        let actual = manifest(root)?;
        bindings(root, &inv, &actual)?;
        if saved != actual || read::<Value>(&root.join("report.json"))? != build(root, &inv)? {
            return Err("complete artifact manifest or original-case recount changed".into());
        }
        manifests.push(actual);
    }
    if manifests[0] != manifests[1] {
        return Err("independent full training, tensors, raw rows or reports differ".into());
    }
    let report: Value = read(&original.join("report.json"))?;
    let value = json!({"independently_reproduced":true,"artifacts":manifests[0].len(),
        "report":manifests[0]["report.json"],"manifest":disk::hash(&original.join("manifest.json"))?,
        "data_seed":1,"modes":report["modes"],"catalog_union_cases":report["candidate_catalog_union_cases"],
        "catalog_with_revised_cohort":report["candidate_catalog_with_revised_cohort"],
        "new_all_modes_all_heldout99_cases":report["new_all_modes_all_heldout99_cases"],
        "new_nonconstant_cases":report["new_nonconstant_cases"],"new_constant_controls":report["new_constant_controls"],
        "regressed_old_controls":report["regressed_old_controls"],"catalog_total_cases":165,
        "historical_raw_artifacts_reverified":false,"new_complete_raw_artifacts_verified":true});
    let parent = output.parent().ok_or("verification output parent")?;
    let guard = joint::resources::Guard::new(workspace, parent, JSON_LIMIT)?;
    let mut file = guard.writer(output, JSON_LIMIT)?;
    use std::io::Write;
    serde_json::to_writer(&mut file, &value)?;
    file.write_all(b"\n")?;
    file.finish()?;
    Ok(())
}
