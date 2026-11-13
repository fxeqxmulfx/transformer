use crate::*;
fn replace(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec(value)?)?;
    Ok(())
}
#[test]
fn complete_synthetic_tree_preserves105_both_arms336_slots_and35_transformers() -> Result<()> {
    let (temp, suite, request) = fixtures::tree()?;
    let out = temp.0.join("comparison");
    import::prepare_with(&out, &request, &suite)?;
    let replay = import::replay_with(&out, &request, &suite)?;
    assert_eq!(replay["replayed"], true);
    let report: Value = read(&out.join("summary.json"))?;
    assert_eq!(rows(&report, "depth_outcomes")?.len(), 336);
    assert_eq!(rows(&report, "transformer_endpoints")?.len(), 35);
    assert_eq!(
        report["constructor_accounting"]["depth_original"]["memory_refusals"],
        18
    );
    assert_eq!(
        report["constructor_accounting"]["consequences"]["process_failures"],
        1
    );
    assert_eq!(
        report["constructor_accounting"]["transformer"]["original_cap_failures"],
        1
    );
    assert_eq!(report["new_gradient_updates"], 0);
    assert_eq!(report["choose_between_initializers"], false);
    let p = protocol()?;
    for arm in rows(&report, "arms")? {
        assert_eq!(rows(arm, "choices")?.len(), 105);
        let mut changed = 0;
        for r in rows(arm, "choices")? {
            if p["retained_tasks"].get(text(r, "task")?).is_some() {
                assert!(text(r, "reference")?.starts_with("depth_retained:"));
                changed += 1;
            } else if r["status"] != "no_model" {
                assert_eq!(r["reference"], r["prior_reference"]);
            }
        }
        assert_eq!(changed, 12);
        assert_eq!(arm["by_mode"][2]["no_model"], 1);
    }
    for (name, pin) in &request.parents {
        assert_eq!(
            fs::read(out.join(format!("{name}-summary.json")))?,
            fs::read(pin.root.join("summary.json"))?
        );
    }
    assert!(import::prepare_with(&out, &request, &suite).is_err());
    Ok(())
}
#[test]
fn selection_uses_ce_earlier_order_and_keeps_policies_separate() -> Result<()> {
    let (_temp, suite, request) = fixtures::tree()?;
    let mut loaded = import::load(&request, &suite)?;
    let before = selection::select(&suite, &loaded.prior, &loaded.retained)?;
    for offer in loaded.prior.iter_mut().chain(&mut loaded.retained) {
        offer.metrics = json!({"validation":{"loss":-1e100,"sequence_accuracy":0.}});
        offer.saved = json!({"unrelated_heldout_score":99});
    }
    assert_eq!(
        before,
        selection::select(&suite, &loaded.prior, &loaded.retained)?
    );
    for offer in &mut loaded.retained {
        offer.train_ce = Some(if offer.policy.as_deref() == Some("minimum_ce") {
            1. + 5e-13
        } else {
            0.2
        });
    }
    let selection = selection::select(&suite, &loaded.prior, &loaded.retained)?;
    for arm in rows(&selection, "arms")? {
        for row in rows(arm, "choices")? {
            if protocol()?["retained_tasks"]
                .get(text(row, "task")?)
                .is_some()
            {
                if arm["initializer"] == "minimum_ce" {
                    assert_eq!(row["selected"], row["prior_reference"]);
                } else {
                    assert!(text(row, "selected")?.contains("--Residual--"));
                }
            }
        }
    }
    Ok(())
}
#[test]
fn missing_duplicate_nonfinite_and_wrong_mode_candidates_are_rejected() -> Result<()> {
    let (_temp, suite, request) = fixtures::tree()?;
    let loaded = import::load(&request, &suite)?;
    let mut prior = loaded.prior.clone();
    prior.pop();
    assert!(selection::select(&suite, &prior, &loaded.retained).is_err());
    prior.push(prior[0].clone());
    assert!(selection::select(&suite, &prior, &loaded.retained).is_err());
    let mut heads = loaded.retained.clone();
    heads[1] = heads[0].clone();
    assert!(selection::select(&suite, &loaded.prior, &heads).is_err());
    heads = loaded.retained.clone();
    heads[0].train_ce = Some(f64::NAN);
    assert!(selection::select(&suite, &loaded.prior, &heads).is_err());
    heads = loaded.retained.clone();
    heads[0].mode = "unexpected".into();
    assert!(selection::select(&suite, &loaded.prior, &heads).is_err());
    Ok(())
}
#[test]
fn changed_unused_model_cache_and_transformer_checkpoint_are_rejected() -> Result<()> {
    let (temp, suite, request) = fixtures::tree()?;
    let loaded = import::load(&request, &suite)?;
    let candidates: Vec<_> = loaded
        .evidence
        .0
        .keys()
        .filter(|p| {
            p.file_name().is_some_and(|n| {
                n == "model.bin" || n == "features.bin" || n == "before-000001.bin"
            })
        })
        .fold(BTreeMap::new(), |mut map, p| {
            map.entry(p.file_name().unwrap().to_owned())
                .or_insert(p.clone());
            map
        })
        .into_values()
        .collect();
    assert_eq!(candidates.len(), 3);
    for path in candidates {
        let old = fs::read(&path)?;
        fs::write(&path, b"corrupt artifact")?;
        assert!(import::prepare_with(&temp.0.join("rejected"), &request, &suite).is_err());
        fs::write(&path, old)?;
    }
    Ok(())
}
#[test]
fn changed_data_and_failed_review_cannot_be_admitted() -> Result<()> {
    let (temp, suite, mut request) = fixtures::tree()?;
    let pin = &request.parents["depth_retained"];
    let case = &depth::expected(true)?[0];
    let id = format!(
        "{}--{}--joint--{}--d{}--retained-{}",
        text(case, "mode")?,
        text(case, "task")?,
        text(case, "transport")?,
        case["depth"],
        text(case, "retained_initializer")?
    );
    let path = pin.root.join(id).join("metadata.json");
    let old = fs::read(&path)?;
    let mut metadata: Value = read(&path)?;
    metadata["data_fingerprints"]["train"] = json!("0".repeat(64));
    replace(&path, &metadata)?;
    assert!(import::prepare_with(&temp.0.join("changed-data"), &request, &suite).is_err());
    fs::write(&path, old)?;
    fs::write(&pin.review.exit, b"1\n")?;
    assert!(import::prepare_with(&temp.0.join("failed-review"), &request, &suite).is_err());
    fs::write(&pin.review.exit, b"0\n")?;
    request.parents.remove("transformer");
    assert!(import::prepare_with(&temp.0.join("missing-parent"), &request, &suite).is_err());
    Ok(())
}
#[test]
fn rehashed_heldout_choice_and_report_mutations_do_not_replay() -> Result<()> {
    let (temp, suite, request) = fixtures::tree()?;
    let out = temp.0.join("comparison");
    import::prepare_with(&out, &request, &suite)?;
    let manifest_path = out.join("manifest.json");
    let original: Value = read(&manifest_path)?;
    let selection_path = out.join("selection.json");
    let mut selection: Value = read(&selection_path)?;
    let unchanged = selection.clone();
    selection["arms"][0]["choices"][0]["selected"] = Value::Null;
    replace(&selection_path, &selection)?;
    let mut manifest = original.clone();
    manifest["selection_sha256"] = json!(digest(&selection_path)?);
    replace(&manifest_path, &manifest)?;
    assert!(import::replay_with(&out, &request, &suite).is_err());
    replace(&selection_path, &unchanged)?;
    replace(&manifest_path, &original)?;
    let report_path = out.join("summary.json");
    let mut report: Value = read(&report_path)?;
    report["arms"][0]["full35_pass"] = json!(true);
    replace(&report_path, &report)?;
    manifest = original;
    manifest["summary_sha256"] = json!(digest(&report_path)?);
    replace(&manifest_path, &manifest)?;
    assert!(import::replay_with(&out, &request, &suite).is_err());
    Ok(())
}
#[test]
fn altered_complete_parent_and_replay_seal_are_detected() -> Result<()> {
    let (temp, suite, request) = fixtures::tree()?;
    let out = temp.0.join("comparison");
    import::prepare_with(&out, &request, &suite)?;
    let path = out.join("consequences-summary.json");
    let old = fs::read(&path)?;
    fs::write(&path, b"{}")?;
    assert!(import::replay_with(&out, &request, &suite).is_err());
    fs::write(&path, old)?;
    let manifest: Value = read(&out.join("manifest.json"))?;
    let artifact = rows(&manifest, "artifacts")?
        .iter()
        .find(|r| {
            r["path"]
                .as_str()
                .is_some_and(|s| s.ends_with(".replay.json"))
        })
        .ok_or("fixture replay missing")?;
    fs::write(Path::new(text(artifact, "path")?), b"{}")?;
    assert!(import::replay_with(&out, &request, &suite).is_err());
    Ok(())
}
