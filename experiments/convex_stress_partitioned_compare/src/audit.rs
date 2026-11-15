//! Retain every old/new raw logit, not just aggregate improvements. Independent
//! inverse Fourier products check arbitrary complete post-CE coefficients.
use super::*;
use std::io::BufRead;
pub(crate) fn run(
    model: &model::Model,
    rows: &[data::Example],
    path: &Path,
    replay: bool,
    retained332: Option<&Path>,
) -> Result<Value> {
    costs::examples(rows, model.prior.outcome.model.input.vocabulary)?;
    let output = if replay {
        path.with_extension("replay.jsonl")
    } else {
        path.to_owned()
    };
    let mut file = std::io::BufWriter::new(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&output)?,
    );
    let (mut correct, mut previous, mut changed, mut fixed, mut regressed, mut covered) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut old_ce, mut new_ce, mut residual) = (0f64, 0f64, 0f64);
    let mut old_by_label = BTreeMap::<usize, (u64, u64, u64)>::new();
    let mut retained = retained332
        .map(fs::File::open)
        .transpose()?
        .map(|f| BufReader::new(f).lines());
    let (mut retained_correct, mut retained_fixed, mut retained_regressed, mut retained_ce) =
        (0u64, 0u64, 0u64, 0f64);
    let (mut older_correct, mut older_fixed, mut older_regressed, mut older_ce) =
        (0u64, 0u64, 0u64, 0f64);
    for (id, e) in rows.iter().enumerate() {
        let cell = model.cell(&e.tokens);
        let (role, role_name) = model.routed(cell);
        let old = model.old_logits(cell)?;
        let new = model.logits(cell)?;
        let independent = model.independent(cell)?;
        let label = e.targets[1] as usize;
        let before = model::argmax(&old)?;
        let after = model::argmax(&new)?;
        let r = new
            .iter()
            .zip(&independent)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        residual = residual.max(r);
        let scale = new.iter().map(|v| v.abs()).fold(1f64, f64::max);
        if r > 1e-10 * scale || (!role.covered(cell) && old != new) {
            return Err("complete product or unsupported parent preservation audit failed".into());
        }
        correct += u64::from(after == label);
        previous += u64::from(before == label);
        changed += u64::from(before != after);
        fixed += u64::from(before != label && after == label);
        regressed += u64::from(before == label && after != label);
        covered += u64::from(role.covered(cell));
        let before_ce = fresh::cross_entropy(&old, label);
        let after_ce = fresh::cross_entropy(&new, label);
        let mut older = Value::Null;
        let reference = if let Some(lines) = retained.as_mut() {
            let value: Value = serde_json::from_str(&lines.next().ok_or("short332 raw audit")??)?;
            if value["record_id"] != id
                || value["tokens"] != json!(e.tokens)
                || value["target"] != label
                || value["field"] != json!(e.field)
                || value["role"] != role_name
                || value["old_logits"] != json!(old)
            {
                return Err("retained332 raw record or stronger319 logits differ".into());
            }
            let logits: Vec<f64> = serde_json::from_value(value["new_logits"].clone())?;
            if logits.len() != new.len() {
                return Err("retained332 raw vocabulary differs".into());
            }
            let prediction = model::argmax(&logits)?;
            let ce = fresh::cross_entropy(&logits, label);
            if value["new_prediction"] != prediction || value["new_ce"].as_f64() != Some(ce) {
                return Err("retained332 prediction or CE changed".into());
            }
            retained_correct += u64::from(prediction == label);
            retained_fixed += u64::from(prediction != label && after == label);
            retained_regressed += u64::from(prediction == label && after != label);
            retained_ce += ce;
            older = value["retained330"].clone();
            let old_logits: Vec<f64> = serde_json::from_value(older["logits"].clone())?;
            if old_logits.len() != new.len() {
                return Err("retained330 raw vocabulary differs".into());
            }
            let old_prediction = model::argmax(&old_logits)?;
            let old_loss = fresh::cross_entropy(&old_logits, label);
            if older["prediction"] != old_prediction || older["ce"].as_f64() != Some(old_loss) {
                return Err("retained330 prediction or CE changed".into());
            }
            older_correct += u64::from(old_prediction == label);
            older_fixed += u64::from(old_prediction != label && after == label);
            older_regressed += u64::from(old_prediction == label && after != label);
            older_ce += old_loss;
            json!({"logits":logits,"prediction":prediction,"ce":ce})
        } else {
            Value::Null
        };
        old_ce += before_ce;
        new_ce += after_ce;
        let entry = old_by_label.entry(label).or_default();
        entry.0 += 1;
        entry.1 += u64::from(before == label);
        entry.2 += u64::from(after == label);
        serde_json::to_writer(
            &mut file,
            &json!({"record_id":id,"tokens":e.tokens,"target":label,"field":e.field,
            "role":role_name,"covered":role.covered(cell),"old_logits":old,"new_logits":new,
            "old_prediction":before,"new_prediction":after,"old_ce":before_ce,"new_ce":after_ce,
            "independent_residual":r,"retained332":reference,"retained330":older}),
        )?;
        file.write_all(b"\n")?;
    }
    if retained
        .as_mut()
        .is_some_and(|lines| lines.next().is_some())
    {
        return Err("extra332 raw audit records".into());
    }
    file.flush()?;
    file.get_ref().sync_all()?;
    drop(file);
    let sha = digest(&output)?;
    if replay {
        if digest(path)? != sha {
            return Err("complete raw-logit replay differs".into());
        }
        fs::remove_file(output)?;
    }
    if correct + regressed != previous + fixed {
        return Err("transition conservation failed".into());
    }
    if retained332.is_some()
        && (correct + retained_regressed != retained_correct + retained_fixed
            || correct + older_regressed != older_correct + older_fixed)
    {
        return Err("retained comparison transition conservation failed".into());
    }
    Ok(
        json!({"targets":rows.len(),"correct":correct,"parent_correct":previous,"changed":changed,"fixed":fixed,
        "regressed":regressed,"errors":rows.len() as u64-correct,"parent_errors":rows.len() as u64-previous,
        "covered_queries":covered,"fallback_queries":rows.len() as u64-covered,"old_ce":old_ce/rows.len() as f64,
        "new_ce":new_ce/rows.len() as f64,"independent_max_residual":residual,"raw_logits_sha256":sha,
        "raw_logits_file":path.file_name().unwrap().to_str().unwrap(),"classes":old_by_label,
        "old_raw_audit_calls":rows.len(),"new_raw_audit_calls":rows.len(),"independent_new_calls":rows.len(),
        "retained330":retained332.map(|_|json!({"targets":rows.len(),"correct":older_correct,
            "fixed":older_fixed,"regressed":older_regressed,"ce":older_ce/rows.len() as f64})),
        "retained332":retained332.map(|_|json!({"targets":rows.len(),"correct":retained_correct,
            "fixed":retained_fixed,"regressed":retained_regressed,"ce":retained_ce/rows.len() as f64}))}),
    )
}
