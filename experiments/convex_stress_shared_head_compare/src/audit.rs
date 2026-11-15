//! Retain every old/new raw logit, not just aggregate improvements. Independent
//! inverse Fourier products check arbitrary complete post-CE coefficients.
use super::*;
pub(crate) fn run(
    model: &model::Model,
    rows: &[data::Example],
    path: &Path,
    replay: bool,
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
            "independent_residual":r}),
        )?;
        file.write_all(b"\n")?;
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
    Ok(
        json!({"targets":rows.len(),"correct":correct,"parent_correct":previous,"changed":changed,"fixed":fixed,
        "regressed":regressed,"errors":rows.len() as u64-correct,"parent_errors":rows.len() as u64-previous,
        "covered_queries":covered,"fallback_queries":rows.len() as u64-covered,"old_ce":old_ce/rows.len() as f64,
        "new_ce":new_ce/rows.len() as f64,"independent_max_residual":residual,"raw_logits_sha256":sha,
        "raw_logits_file":path.file_name().unwrap().to_str().unwrap(),"classes":old_by_label,
        "old_raw_audit_calls":rows.len(),"new_raw_audit_calls":rows.len(),"independent_new_calls":rows.len()}),
    )
}
