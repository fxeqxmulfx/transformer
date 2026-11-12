//! Reconstruct coding and saved endpoint; require sealed raw audits in summary.
use super::*;
use crate::{Mode, Model, Solver, budgets, choose_by_train, ideal_loss};

pub fn replay(root: &Path, id: &str) -> Result<Value> {
    let c = case(id)?;
    let x = inputs(&c)?;
    with_inputs(root, id, x)
}

pub(super) fn with_inputs(root: &Path, id: &str, x: Inputs) -> Result<Value> {
    let c = case(id)?;
    let inv = inventory(root)?;
    check_inventory_task(&inv, &c.task, &x)?;
    let p = protocol()?;
    let out = root.join(id);
    let bytes = fs::read(out.join("result.json"))?;
    let r: Value = serde_json::from_slice(&bytes)?;
    let count = x.task.splits["train"].len();
    let batch = x.row["case"]["batch"].as_u64().unwrap() as usize;
    let per_epoch = count.div_ceil(batch);
    let cadence = (x.row["case"]["observe"].as_u64().unwrap() as usize).min(per_epoch);
    if r["case"] != serde_json::to_value(&c)?
        || r["id"] != id
        || r["source_fingerprint"] != fingerprint()
        || r["protocol"] != p
        || r["producer_commit"] != inv["producer_commit"]
        || r["vocab"] != x.task.vocab
        || r["data_fingerprints"] != x.row["data_fingerprints"]
        || r["selection"] != x.task.selection
        || r["discovery"] != serde_json::to_value(&x.discovery)?
        || r["coding"] != serde_json::to_value(&x.coding)?
        || r["train_prefixes"] != count
        || r["proposal_examples_presented"] != count
        || r["explicit_bytes_upper_bound"] != x.explicit_bytes
        || r["batch_size"] != batch
        || r["updates_per_epoch"] != per_epoch
        || r["seed"] != p["seed"]
    {
        return Err("saved protocol/data/coding mismatch".into());
    }
    let seal = |audit: Value| json!({"replayed":true,"id":id,"source_fingerprint":fingerprint(),"result_sha256":hash(&bytes),"status":r["status"],"model_sha256":r["model_sha256"],"audit":audit});
    if x.coding.is_none() {
        if r["status"] != "proposal_ineligible"
            || r["updates"] != 0
            || r["training_gradient_evaluations"] != 0
            || r["examples_presented"] != 0
            || out.join("model.json").exists()
            || out.join("diagnostic_state.json").exists()
        {
            return Err("invalid rejected proposal record".into());
        }
        return Ok(seal(json!({"rejected_proposal_reproduced":true})));
    }
    if r["status"] != "complete" {
        return Err("incomplete cell".into());
    }
    let model_bytes = fs::read(out.join("model.json"))?;
    let state = fs::read(out.join("diagnostic_state.json"))?;
    if r["model_sha256"] != hash(&model_bytes) || r["diagnostic_state_sha256"] != hash(&state) {
        return Err("saved file digest mismatch".into());
    }
    let model: Model = serde_json::from_slice(&model_bytes)?;
    model.validate_shape()?;
    let (radius, nuclear_budget) = budgets(&c.budget, x.task.vocab)?;
    if Some(&model.coding) != x.coding.as_ref()
        || model.table.vocab != x.task.vocab
        || (model.table.radius, model.table.nuclear_budget) != (radius, nuclear_budget)
        || number(&r, "radius")? != radius
        || number(&r, "nuclear_budget")? != nuclear_budget
    {
        return Err("saved model configuration mismatch".into());
    }
    let mode = Mode::parse(&c.arm)?;
    let mut solver = Solver::new(
        &model.table,
        x.samples,
        mode,
        p["frequency_workers"].as_u64().unwrap() as usize,
    )?;
    solver.restore_diagnostic_state(serde_json::from_slice(&state)?)?;
    let (snapshot, alpha, shadow_norms) = solver.snapshot()?;
    if snapshot.probabilities != model.table.probabilities {
        return Err("endpoint does not match saved splitting state".into());
    }
    audit::close(&json!(alpha), &r["uniform_mixture_alpha"], 1e-12)?;
    audit::close(
        &serde_json::to_value(shadow_norms)?,
        &r["shadow_norms"],
        1e-8,
    )?;
    audit::close(
        &json!(solver.full_residual()?),
        &r["full_operator_residual"],
        1e-9,
    )?;
    audit::close(
        &json!(solver.fit_defect(&model.table)),
        &r["fit_defect"],
        1e-12,
    )?;
    if metrics_all(&model, &x.task)? != r["metrics"] {
        return Err("saved metrics mismatch".into());
    }
    let updates = r["updates"].as_u64().ok_or("missing updates")? as usize;
    let presented = match mode {
        Mode::Full => updates * count,
        Mode::Mini => updates / per_epoch * count + ((updates % per_epoch) * batch).min(count),
        Mode::Forward => count,
    };
    let excess = number(&r["metrics"]["train"], "loss")? - ideal_loss(x.task.vocab);
    if updates == 0
        || updates > c.limit(&p)
        || (updates != c.limit(&p) && !updates.is_multiple_of(cadence))
        || r["examples_presented"] != presented
        || r["training_gradient_evaluations"] != if mode == Mode::Forward { 0 } else { presented }
        || (number(&r, "epochs")? - presented as f64 / count as f64).abs() > 1e-12
        || match r["stop_reason"].as_str() {
            Some("budget") => updates != c.limit(&p),
            Some("ideal_ce_excess") => excess > number(&p, "ideal_ce_tolerance")?,
            _ => true,
        }
    {
        return Err("invalid stopping or exposure metadata".into());
    }
    // Release optimizer work buffers before the independent raw-coordinate audit.
    drop(solver);
    let raw_audit = audit::validate(&model, &x.task, &r)?;
    Ok(seal(raw_audit))
}

pub fn summarize(root: &Path) -> Result<Value> {
    inventory(root)?;
    let p = protocol()?;
    let mut records = vec![];
    for c in cases()? {
        let id = c.id();
        let out = root.join(&id);
        let bytes = fs::read(out.join("result.json"))?;
        let r: Value = serde_json::from_slice(&bytes)?;
        let seal = read(&root.join(format!("{id}.replay.json")))?;
        if r["case"] != serde_json::to_value(&c)?
            || r["source_fingerprint"] != fingerprint()
            || r["protocol"] != p
            || seal["replayed"] != true
            || seal["id"] != id
            || seal["source_fingerprint"] != fingerprint()
            || seal["result_sha256"] != hash(&bytes)
            || seal["status"] != r["status"]
            || seal["model_sha256"] != r["model_sha256"]
        {
            return Err("missing or stale replay seal".into());
        }
        if r["status"] == "complete" {
            if seal["audit"]["raw_coordinate_audit"] != true
                || r["model_sha256"] != hash(&fs::read(out.join("model.json"))?)
                || r["diagnostic_state_sha256"]
                    != hash(&fs::read(out.join("diagnostic_state.json"))?)
            {
                return Err("unverified saved endpoint".into());
            }
        } else if r["status"] != "proposal_ineligible"
            || seal["audit"]["rejected_proposal_reproduced"] != true
        {
            return Err("unresolved cell".into());
        }
        records.push((c, r));
    }
    let mut choices = vec![];
    for task in p["tasks"].as_array().unwrap() {
        for arm in p["arms"].as_array().unwrap() {
            let candidates: Vec<_> = records
                .iter()
                .filter(|(c, r)| {
                    c.task == task.as_str().unwrap()
                        && c.arm == arm.as_str().unwrap()
                        && r["status"] == "complete"
                })
                .collect();
            let losses = candidates
                .iter()
                .map(|(_, r)| number(&r["metrics"]["train"], "loss"))
                .collect::<Result<Vec<_>>>()?;
            let n = candidates.first().ok_or("no completed candidate")?.1["vocab"]
                .as_u64()
                .unwrap() as usize;
            let winner =
                choose_by_train(&losses, ideal_loss(n), number(&p, "ideal_ce_tolerance")?)?;
            let (c, r) = candidates[winner];
            choices.push(json!({"task":task,"arm":arm,"selected_case":c,"selected_model":format!("{}/model.json",c.id()),"model_sha256":r["model_sha256"],
            "train_ce":losses[winner],"ideal_ce_excess":losses[winner]-ideal_loss(n),"metrics":r["metrics"],"selection":r["selection"],
            "candidates":candidates.iter().zip(&losses).map(|((c,r), ce)| json!({"id":c.id(),"train_ce":ce,"updates":r["updates"],"proposal_examples_presented":r["proposal_examples_presented"],"training_examples_presented":r["examples_presented"],"total_seconds":r["total_seconds_including_data_and_proposal"]})).collect::<Vec<_>>()}));
        }
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":p,"cells":records.len(),"complete":records.iter().filter(|(_, r)| r["status"]=="complete").count(),
        "proposal_ineligible":records.iter().filter(|(_, r)| r["status"]=="proposal_ineligible").count(),"choices":choices,"full35_result":false}),
    )
}
