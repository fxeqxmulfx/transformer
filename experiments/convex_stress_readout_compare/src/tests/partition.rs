use super::*;

#[test]
fn declared_partition_grouped_basis_and_raw_expansion_are_bound_independently() -> Result<()> {
    let f = Fixture::new()?;
    let x = input("fixture-0")?;
    let task = "fixture-0";
    let index = catalogue().iter().position(|&p| p == 97).unwrap();
    let path = f
        .request
        .staging
        .root
        .join(task)
        .join(format!("stage-{index:02}-p97.json"));
    let original: Value = read(&path)?;
    let original_discovery: discovery::Report =
        serde_json::from_value(original["discovery"].clone())?;
    assert!(original_discovery.grouped.as_ref().unwrap().code.is_some());
    assert!(original_discovery.raw_code.is_some());
    assert!(!original_discovery.raw_label_identification);
    assert!(!original_discovery.unknown_target_compatibility_proved);
    let mut wrong = input(task)?;
    wrong.readout.as_mut().unwrap().swap(0, 1);
    let record = imported::case(&f.request.staging, &f.stage, task, &x)?;
    assert!(
        imported::load(
            &f.request.staging,
            task,
            &record,
            &record["result"]["stages"][index],
            &wrong,
            false
        )
        .is_err()
    );

    // Rehash each altered saved report, including removal of the head, so
    // rejection must come from the discovery certificate or partition binding.
    for kind in 0..5 {
        let mut changed = original.clone();
        changed["head"] = Value::Null;
        changed["head_error"] = json!("simulated head failure after saved discovery");
        match kind {
            0 => changed["discovery"]["raw_code"]["classes"][0][0] = json!(96),
            1 => {
                changed["discovery"]["grouped"]["request"]["input"]["observations"][0][1] =
                    json!(96)
            }
            2 => changed["discovery"]["request"]["answers"] = json!(98),
            3 => changed["discovery"]["work"]["expanded_coordinates"] = json!(0),
            _ => changed["discovery"]["unknown_target_compatibility_proved"] = json!(true),
        }
        overwrite(&path, &changed);
        let mut record = record.clone();
        let mut saved = record["result"]["stages"][index].clone();
        saved["sha256"] = json!(digest(&path)?);
        saved["bytes"] = json!(fs::metadata(&path)?.len());
        saved["retained"]["head_error"] = changed["head_error"].clone();
        record["result"]["stages"][index] = saved.clone();
        assert!(imported::load(&f.request.staging, task, &record, &saved, &x, true).is_err());
    }
    overwrite(&path, &original);
    let partition_path = f.request.staging.root.join(task).join("readout.json");
    let mut partition = x.partition();
    partition["classes"][0] = json!(0);
    overwrite(&partition_path, &partition);
    let result_path = f.request.staging.root.join(task).join("result.json");
    let mut result: Value = read(&result_path)?;
    result["readout_sha256"] = json!(digest(&partition_path)?);
    overwrite(&result_path, &result);
    let mut stage = f.stage.clone();
    stage["records"][0]["result"] = result;
    let mut artifacts = vec![];
    parents::tree(&f.request.staging.root.join(task), &mut artifacts)?;
    stage["records"][0]["artifacts"] = json!(artifacts);
    assert!(imported::case(&f.request.staging, &stage, task, &x).is_err());
    Ok(())
}

#[test]
fn standalone_success_does_not_override_parent_first_ce_selection() -> Result<()> {
    let all: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
    let parent = fixtures::parent(&all);
    let offer = comparison::Offer {
        task: all[0].clone(),
        mode: training::Mode::Full,
        prime: 97,
        train_ce: 1.,
        model_sha256: "a".repeat(64),
        stage_sha256: "b".repeat(64),
    };
    let selected = comparison::select(&parent, std::slice::from_ref(&offer), &all)?;
    let standalone = json!({"task":all[0],"training_mode":"full_batch","reference":offer.reference(),
        "status":"evaluated","train_ce":1.,"primary_split":"validation","offer_eligible":true,
        "metrics":{"validation":{"sequence_accuracy":1.}},"primary_pass":true,"all_splits_pass":true});
    let evaluated = BTreeMap::from([(offer.reference(), standalone)]);
    let arms = comparison::render(&parent, &selected, &evaluated)?;
    assert_eq!(
        diagnostics::standalone(&evaluated)["by_mode"][0]["primary_passes"],
        1
    );
    assert_eq!(arms[0]["choices"][0]["primary_pass"], false);
    assert_eq!(
        diagnostics::changes(&parent, &arms)?["arms"][0]["replacements"],
        0
    );

    // Conversely an observed-CE improvement can hurt unseen performance; the
    // diagnostic must retain that regression without changing the selector.
    let mut changed = arms;
    changed[0]["choices"][3]["reference"] = json!("changed-fixture");
    changed[0]["choices"][3]["primary_pass"] = json!(false);
    changed[0]["choices"][3]["metrics"]["validation"]["sequence_accuracy"] = json!(0.);
    let diagnostics = diagnostics::changes(&parent, &changed)?;
    assert_eq!(diagnostics["arms"][0]["primary_regressions"], 1);
    assert!(
        diagnostics["arms"][0]["changed_choices"][0]["metric_deltas"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["regression"] == true)
    );
    Ok(())
}
