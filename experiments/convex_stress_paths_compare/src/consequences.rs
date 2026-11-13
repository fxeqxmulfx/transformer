//! Preserve the whole105-choice parent and every new constructor outcome.
use crate::*;
fn exit(root: &Path, name: &str, tracker: &mut Tracker) -> Result<i32> {
    let path = root.join(name);
    tracker.bind(&path, None)?;
    Ok(fs::read_to_string(path)?.trim().parse()?)
}
fn partial(path: &Path, tracker: &mut Tracker) -> Result<()> {
    if path.exists() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                partial(&entry.path(), tracker)?;
            } else {
                tracker.bind(&entry.path(), None)?;
            }
        }
    }
    Ok(())
}
pub(crate) fn load(
    root: &Path,
    inv: &Value,
    summary: &Value,
    suite: &Value,
    loaded: &mut Loaded,
) -> Result<()> {
    check(
        summary["producer_commit"] == inv["producer_commit"]
            && summary["full35_result"] == true
            && summary["choices_compared"] == 105
            && summary["tasks"] == 23
            && rows(summary, "records")?.len() == 23
            && rows(inv, "tasks")?.len() == 23,
        "incomplete consequence parent",
    )?;
    loaded
        .evidence
        .bind(&root.join("parent.json"), Some(text(inv, "parent_sha256")?))?;
    let mut unique = BTreeSet::new();
    let mut failures = 0;
    for record in rows(summary, "records")? {
        let name = artifacts::local(text(record, "task")?)?;
        task(suite, name)?;
        check(
            unique.insert(name) && rows(inv, "tasks")?.iter().filter(|r| *r == name).count() == 1,
            "duplicate or undeclared consequence task",
        )?;
        let code = exit(root, &format!("{name}.run.exit"), &mut loaded.evidence)?;
        let replay_code = if code == 0 {
            Some(exit(
                root,
                &format!("{name}.replay.exit"),
                &mut loaded.evidence,
            )?)
        } else {
            None
        };
        if record["status"] == "process_failure" {
            check(
                (code != 0 || replay_code != Some(0))
                    && record["run_exit"] == code
                    && record["replay_exit"] == serde_json::to_value(replay_code)?
                    && record["prior_choices_retained"] == 3,
                "failed consequence process was relabeled",
            )?;
            loaded.evidence.bind(
                &root.join(format!("{name}.log")),
                Some(text(record, "run_log_sha256")?),
            )?;
            partial(&root.join(name), &mut loaded.evidence)?;
            failures += 1;
            continue;
        }
        check(
            record["status"] == "complete" && code == 0 && replay_code == Some(0),
            "unreplayed consequence outcome",
        )?;
        let dir = root.join(name);
        let result = loaded.evidence.load(&dir.join("result.json"))?;
        let replay = loaded
            .evidence
            .load(&root.join(format!("{name}.replay.json")))?;
        check(
            result == record["result"]
                && replay == record["seal"]
                && replay["replayed"] == true
                && replay["all_view_reports_replayed"] == true
                && replay["raw_latent_ce_metrics_audit"] == true
                && result["ce_gradients"] == 0
                && replay["task"] == name,
            "consequence numerical review record differs",
        )?;
        loaded.evidence.bind(
            &dir.join("result.json"),
            Some(text(&replay, "result_sha256")?),
        )?;
        check(
            result["metadata"] == loaded.evidence.load(&dir.join("metadata.json"))?
                && result["metadata"]["source_fingerprint"] == inv["source_fingerprint"]
                && result["metadata"]["producer_commit"] == inv["producer_commit"]
                && replay["source_fingerprint"] == inv["source_fingerprint"],
            "consequence provenance changed",
        )?;
        data(
            suite,
            name,
            &result["metadata"]["input"]["data_fingerprints"],
        )?;
        let selection = loaded.evidence.load(&dir.join("selection.json"))?;
        check(
            selection == result["selection"],
            "consequence selection differs",
        )?;
        loaded.evidence.bind(
            &dir.join("selection.json"),
            Some(text(&replay, "selection_sha256")?),
        )?;
        for (file, key) in [
            ("input.json", "input_sha256"),
            ("parents.json", "parents_sha256"),
            ("bank.json", "bank_sha256"),
        ] {
            loaded
                .evidence
                .bind(&dir.join(file), Some(text(&selection, key)?))?;
        }
        for view in rows(&selection, "views")? {
            let artifact = loaded.evidence.bind(
                &dir.join(artifacts::local(text(view, "file")?)?),
                Some(text(view, "sha256")?),
            )?;
            check(view["bytes"] == artifact.bytes, "dense view size changed")?;
        }
        let bank = loaded.evidence.load(&dir.join("bank.json"))?;
        for candidate in rows(&bank, "candidates")? {
            check(
                !candidate["value"]["model"].is_null()
                    && text(candidate, "model_sha256")?.len() == 64,
                "actual complex model is absent from the sealed bank",
            )?;
        }
    }
    check(
        summary["process_failures"] == failures,
        "consequence failure accounting changed",
    )?;
    for (index, row) in rows(summary, "choices")?.iter().enumerate() {
        let selected = &row["selected"];
        check(
            matches!(row["status"].as_str(), Some("selected" | "no_model"))
                && (row["status"] == "no_model") == selected.is_null(),
            "invalid prior terminal choice",
        )?;
        let ce = if selected.is_null() {
            None
        } else {
            let ce = number(selected, "train_ce")?;
            check(
                (number(&selected["metrics"]["train"], "loss")? - ce).abs() <= 1e-10,
                "prior ordinary CE differs from replay metrics",
            )?;
            Some(ce)
        };
        loaded.prior.push(Offer {
            reference: format!("consequences:choices:{index}"),
            task: text(row, "task")?.into(),
            mode: text(row, "training_mode")?.into(),
            policy: None,
            transport: None,
            train_ce: ce,
            primary: text(row, "selection")?.into(),
            metrics: selected["metrics"].clone(),
            saved: selected.clone(),
        });
    }
    selection::validate_prior(suite, &loaded.prior)?;
    loaded.accounting.insert("consequences".into(),json!({"process_failures":failures,"tasks":23,
        "constructor_accounting":summary["constructor_accounting"],"prior_costs":"complete consequence summary and its complete_parent",
        "inherited105_model_bindings":"exact reviewed parent; unchanged earlier models inherit their earlier replay evidence"}));
    Ok(())
}
