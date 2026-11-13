use super::*;
pub(super) fn metadata(inv: &Value, x: &Inputs) -> Value {
    json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],
        "producer_commit":inv["producer_commit"],"protocol":protocol().expect("embedded protocol"),
        "input":x.record,"parent_sha256":inv["parent_sha256"],"training_gradient_evaluations":0,
        "standalone_forward_construction":true,"one_dense_view_at_a_time":true})
}
pub(super) fn append(
    bank: &mut Bank,
    report: &view::Report,
    artifact: artifacts::Artifact,
) -> Result<()> {
    bank.views.push(ViewRecord {
        index: report.input.view,
        artifact,
        disposition: report.slot.disposition.clone(),
        candidates: report.candidates.len(),
        selected: report.selected,
        work: report.work.clone(),
    });
    for (index, c) in report.candidates.iter().enumerate() {
        let duplicate_of = bank
            .candidates
            .iter()
            .position(|prior| prior.value.model.output_labels == c.model.output_labels);
        bank.candidates.push(Candidate {
            view: report.input.view,
            candidate: index,
            duplicate_of,
            model_sha256: hash(&serde_json::to_vec(&c.model)?),
            output_sha256: hash(&serde_json::to_vec(&c.model.output_labels)?),
            value: c.clone(),
        });
    }
    Ok(())
}
pub fn run(root: &Path, name: &str) -> Result<()> {
    let (inv, parent) = inventory::inventory(root)?;
    let x = inputs(name)?;
    let parents = parent::choices(&parent, name, &x.record)?;
    drop(parent);
    execute(root, name, &inv, parents, x)
}
pub(super) fn execute(
    root: &Path,
    name: &str,
    inv: &Value,
    parents: Vec<Value>,
    x: Inputs,
) -> Result<()> {
    check_input(inv, name, &x)?;
    let dir = root.join(name);
    fs::create_dir(&dir)?;
    let meta = metadata(inv, &x);
    write(&dir.join("metadata.json"), &meta)?;
    write(&dir.join("input.json"), &x.input)?;
    write(&dir.join("parents.json"), &parents)?;
    let slots = view::catalogue(&x.input)?;
    let mut bank = Bank {
        views: vec![],
        candidates: vec![],
    };
    let started = std::time::Instant::now();
    for index in 0..slots.len() {
        let request = view::Request {
            input: x.input.clone(),
            view: index,
        };
        let report = view::run(request.clone())?;
        view::check(&request, &report)?;
        let file = format!("view-{index}.json");
        write(&dir.join(&file), &report)?;
        append(&mut bank, &report, artifacts::record(&dir, &file)?)?;
        // The dense report is dropped here, before constructing another view.
    }
    let seconds = started.elapsed().as_secs_f64();
    write(&dir.join("bank.json"), &bank)?;
    let references = selection::select(&bank, &parents)?;
    let seal = selection::seal(&dir, &bank, &references)?;
    write(&dir.join("selection.json"), &seal)?;
    // No held-out target is inspected by the constructor or the selection rule.
    let evaluated = evaluate::evaluate(&bank, &x, &parents, &references)?;
    write(
        &dir.join("result.json"),
        &json!({"metadata":meta,"selection":seal,
        "selection_sha256":digest(&dir.join("selection.json"))?,"construction_seconds":seconds,
        "views":bank.views,"candidate_count":bank.candidates.len(),"ce_gradients":0,"evaluation":evaluated}),
    )
}
pub fn replay(root: &Path, name: &str) -> Result<Value> {
    let (inv, parent) = inventory::inventory(root)?;
    let x = inputs(name)?;
    let parents = parent::choices(&parent, name, &x.record)?;
    drop(parent);
    replay_with(root, name, &inv, &parents, x)
}
pub(super) fn replay_with(
    root: &Path,
    name: &str,
    inv: &Value,
    parents: &[Value],
    x: Inputs,
) -> Result<Value> {
    check_input(inv, name, &x)?;
    let dir = root.join(name);
    let result: Value = read(&dir.join("result.json"))?;
    if read::<view::Input>(&dir.join("input.json"))? != x.input
        || read::<Value>(&dir.join("metadata.json"))? != metadata(inv, &x)
        || result["metadata"] != metadata(inv, &x)
        || read::<Vec<Value>>(&dir.join("parents.json"))? != parents
        || result["ce_gradients"] != 0
    {
        return Err("original input/parent/metadata changed".into());
    }
    let mut bank = Bank {
        views: vec![],
        candidates: vec![],
    };
    for index in 0..view::catalogue(&x.input)?.len() {
        let file = format!("view-{index}.json");
        let report: view::Report = read(&dir.join(&file))?;
        let request = view::Request {
            input: x.input.clone(),
            view: index,
        };
        view::check(&request, &report)?;
        view::replay(&request, &report)?;
        append(&mut bank, &report, artifacts::record(&dir, &file)?)?;
    }
    let references = selection::select(&bank, parents)?;
    let seal = selection::seal(&dir, &bank, &references)?;
    if read::<Bank>(&dir.join("bank.json"))? != bank
        || read::<Value>(&dir.join("selection.json"))? != seal
        || result["selection"] != seal
        || result["selection_sha256"] != digest(&dir.join("selection.json"))?
        || result["views"] != serde_json::to_value(&bank.views)?
        || result["candidate_count"] != bank.candidates.len()
        || result["evaluation"] != evaluate::evaluate(&bank, &x, parents, &references)?
    {
        return Err("retained views, full model bank, CE seal or original metrics changed".into());
    }
    Ok(
        json!({"task":name,"replayed":true,"source_fingerprint":fingerprint(),"parent_sha256":inv["parent_sha256"],
        "result_sha256":digest(&dir.join("result.json"))?,"selection_sha256":result["selection_sha256"],
        "references":references,"all_view_reports_replayed":true,"raw_latent_ce_metrics_audit":true,
        "unconditional_ground_truth_identification":false}),
    )
}
