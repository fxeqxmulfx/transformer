use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// Direct full-vocabulary derivatives and exhaustive L1 oracle from saved raw
/// coefficients. Does not call the optimizer's objective, gradient or oracle.
fn raw_tensor_oracle(head: &Head, examples: &[data::Example]) -> Result<(f64, f64, usize)> {
    let v = head.bank.vocab;
    let mut gradient = vec![0.; head.coefficients.len()];
    let (mut ce, mut n) = (0., 0);
    let mut logits = vec![0.; v];
    let mut residual = vec![0.; v];
    for example in examples {
        for (position, &label) in example.targets.iter().enumerate() {
            if label < 0 {
                continue;
            }
            let features = head.bank.encode(&example.tokens[..=position])?;
            logits.fill(0.);
            for &k in &features {
                for (z, w) in logits
                    .iter_mut()
                    .zip(&head.coefficients[k * v..(k + 1) * v])
                {
                    *z += w;
                }
            }
            let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mass = logits.iter().map(|z| (z - maximum).exp()).sum::<f64>();
            ce += maximum - logits[label as usize] + mass.ln();
            n += 1;
            for (c, (r, z)) in residual.iter_mut().zip(&logits).enumerate() {
                *r = (z - maximum).exp() / mass - f64::from(c == label as usize);
            }
            for k in features {
                for (g, r) in gradient[k * v..(k + 1) * v].iter_mut().zip(&residual) {
                    *g += r;
                }
            }
        }
    }
    if n == 0 {
        return Err("empty raw head audit".into());
    }
    let mut maxima = vec![0f64; v];
    let mut dot = 0.;
    for (k, g) in gradient.iter_mut().enumerate() {
        *g /= n as f64;
        dot += *g * head.coefficients[k];
        maxima[k % v] = maxima[k % v].max(g.abs());
    }
    Ok((
        ce / n as f64,
        dot + head.radius * maxima.iter().sum::<f64>(),
        n,
    ))
}

pub fn replay_cell(root: &Path, id: &str) -> Result<Value> {
    let (manifest, case) = selected_case(root, id)?;
    let directory = root.join(id);
    let metadata = read_json(directory.join("metadata.json"))?;
    let result = read_json(directory.join("result.json"))?;
    let result_hash =
        checkpoint::digest(BufReader::new(File::open(directory.join("result.json"))?))?;
    let encoded_case = serde_json::to_value(&case)?;
    if metadata["source_fingerprint"] != fingerprint()
        || result["source_fingerprint"] != fingerprint()
        || metadata["case"] != encoded_case
        || result["case"] != encoded_case
        || metadata["producer_commit"] != manifest["producer_commit"]
    {
        return Err("saved cohort provenance mismatch".into());
    }
    let row = meta::selected(&case.task)?;
    let task = data::build(&row)?;
    let n = task.splits["train"]
        .iter()
        .map(|e| e.targets.iter().filter(|y| **y >= 0).count())
        .sum::<usize>();
    let bank = case.bank(&task)?;
    let allocation = case.allocation(&bank, n)?;
    if metadata["bank"] != serde_json::to_value(&bank)?
        || metadata["data_fingerprints"] != row["data_fingerprints"]
        || metadata["training_examples"] != task.splits["train"].len()
        || metadata["training_prefixes"] != n
        || metadata["explicit_buffer_bytes"] != allocation
        || metadata["radius"] != 32. * task.vocab as f64
    {
        return Err("original data or fixed bank does not replay".into());
    }
    let expected = manifest["allocations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == id)
        .ok_or("missing allocation forecast")?;
    if expected["bytes"] != allocation || expected["eligible"] != (allocation <= head::BUFFER_LIMIT)
    {
        return Err("frozen allocation forecast differs from replay".into());
    }
    if result["status"] == "ineligible_memory_budget" {
        if allocation <= head::BUFFER_LIMIT
            || result["explicit_buffer_bytes"] != allocation
            || result["heldout_scored"] != false
            || directory.join("model.bin").exists()
        {
            return Err("invalid memory rejection record".into());
        }
        return Ok(
            json!({"status":"replay_pass","source_fingerprint":fingerprint(),"case":case,"memory_rejection":true,"result_sha256":result_hash}),
        );
    }
    if result["status"] != "complete" || result["producer_commit"] != manifest["producer_commit"] {
        return Err("incomplete model result".into());
    }
    let hash = checkpoint::digest(BufReader::new(File::open(directory.join("model.bin"))?))?;
    if result["model_sha256"] != hash {
        return Err("cohort model hash mismatch".into());
    }
    let model = Model::load(&case.architecture, &directory.join("model.bin"))?;
    if serde_json::to_value(model.bank())? != metadata["bank"]
        || model.radius() != 32. * task.vocab as f64
        || result["parameters"] != model.parameters()
        || result["paths"] != model.bank().paths()
    {
        return Err("saved model configuration mismatch".into());
    }
    let norm = model.validate()?;
    if (norm - meta::number(&result, "max_class_l1")?).abs() > 1e-8 {
        return Err("saved class norm does not replay".into());
    }
    let metrics = convex_stress_hull::all_metrics(&model, &task)?;
    if metrics != result["metrics"]
        || result["primary_split"] != task.selection
        || result["primary_pass"] != convex_stress_hull::passes(&metrics[&task.selection])
        || result["all_splits_pass"]
            != metrics
                .as_object()
                .unwrap()
                .iter()
                .filter(|(s, _)| s.as_str() != "train")
                .all(|(_, m)| convex_stress_hull::passes(m))
    {
        return Err("original split metrics do not replay".into());
    }
    let ce = metrics["train"]["loss"].as_f64().ok_or("missing CE")?;
    if (ce - meta::number(&result, "train_ce")?).abs() > 1e-10 {
        return Err("cached CE and original split CE disagree".into());
    }
    let updates = result["gradient_updates"]
        .as_u64()
        .ok_or("missing update count")?;
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let cap = if case.mode == "full_batch" {
        protocol["full_batch_updates"].as_u64().unwrap()
    } else {
        protocol["minibatch_updates"].as_u64().unwrap()
    };
    if case.mode == "forward_only" {
        if updates != 0
            || [
                "optimizer_backward_examples",
                "optimizer_backward_prefixes",
                "diagnostic_backward_prefixes",
                "line_search_forward_prefixes",
            ]
            .iter()
            .any(|k| result[k] != 0)
            || !result["full_gradient_linear_gap"].is_null()
        {
            return Err("forward-only record contains backward work".into());
        }
    } else if updates == 0
        || updates > cap
        || (updates < cap && meta::number(&result, "train_ce")? > 1e-8)
    {
        return Err("optimizer cap or training-only stop does not replay".into());
    }
    let mut raw = Value::Null;
    if let Model::Tensor(head) = &model {
        if case.mode != "forward_only" {
            let (direct_ce, gap, count) = raw_tensor_oracle(head, &task.splits["train"])?;
            let gap_error = (gap - meta::number(&result, "full_gradient_linear_gap")?).abs();
            if count != n
                || (direct_ce - meta::number(&result, "train_ce")?).abs() > 1e-10
                || gap_error > 1e-7 * (1. + gap.abs())
            {
                return Err("raw full-data CE/oracle does not replay".into());
            }
            raw = json!({"raw_ce":direct_ce,"raw_gap":gap,"gap_error":gap_error,"prefixes":count});
        }
    } else if !result["full_gradient_linear_gap"].is_null() {
        return Err("factor baseline has an unsupported global gap claim".into());
    }
    Ok(
        json!({"status":"replay_pass","source_fingerprint":fingerprint(),"case":case,"model_sha256":hash,"result_sha256":result_hash,
        "train_ce":ce,"max_class_l1":norm,"raw_tensor_audit":raw,"original_metrics_equal":true}),
    )
}

/// The only selection inputs are depth, status and original training CE.
pub fn choose_depth(records: &[(Case, Value)]) -> Result<&(Case, Value)> {
    let mut eligible: Vec<_> = records
        .iter()
        .filter(|(_, r)| r["status"] == "complete")
        .collect();
    eligible.sort_by_key(|(c, _)| c.depth);
    if eligible.is_empty() {
        return Err("no completed depth candidate".into());
    }
    for (_, r) in &eligible {
        if meta::number(r, "train_ce")? < 0. {
            return Err("negative ordinary training CE".into());
        }
    }
    if let Some(record) = eligible
        .iter()
        .find(|(_, r)| r["train_ce"].as_f64().unwrap() <= 1e-8)
    {
        return Ok(record);
    }
    Ok(eligible
        .into_iter()
        .min_by(|(a, x), (b, y)| {
            x["train_ce"]
                .as_f64()
                .unwrap()
                .total_cmp(&y["train_ce"].as_f64().unwrap())
                .then(a.depth.cmp(&b.depth))
        })
        .unwrap())
}

pub fn summarize(root: &Path) -> Result<Value> {
    inventory(root)?;
    let mut groups: BTreeMap<String, Vec<(Case, Value)>> = BTreeMap::new();
    let mut complete = 0;
    let mut ineligible = 0;
    let mut primary = 0;
    let mut all_splits = 0;
    let mut seconds = 0.;
    let mut represented = BTreeSet::new();
    for case in cases()? {
        let id = case.id();
        let result = read_json(root.join(&id).join("result.json"))?;
        let replay = read_json(root.join(format!("{id}.replay.json")))?;
        let result_hash = checkpoint::digest(BufReader::new(File::open(
            root.join(&id).join("result.json"),
        )?))?;
        if replay["status"] != "replay_pass"
            || replay["source_fingerprint"] != fingerprint()
            || replay["case"] != serde_json::to_value(&case)?
            || result["source_fingerprint"] != fingerprint()
            || result["case"] != serde_json::to_value(&case)?
            || replay["result_sha256"] != result_hash
        {
            return Err("cohort has unverified records".into());
        }
        match result["status"].as_str() {
            Some("complete") => {
                complete += 1;
                primary += usize::from(result["primary_pass"] == true);
                all_splits += usize::from(result["all_splits_pass"] == true);
                seconds += meta::number(&result["timing_seconds"], "total")?;
            }
            Some("ineligible_memory_budget") => ineligible += 1,
            _ => return Err("cohort includes incomplete outcomes".into()),
        }
        represented.insert((
            case.task.clone(),
            case.architecture.clone(),
            format!("{:?}", case.transport),
            case.mode.clone(),
        ));
        groups
            .entry(format!(
                "{}--{}--{}--{:?}",
                case.task, case.mode, case.architecture, case.transport
            ))
            .or_default()
            .push((case, result));
    }
    if represented.len() != 72 || complete + ineligible != 288 {
        return Err("incomplete comparison scope".into());
    }
    let mut choices = vec![];
    for (key, records) in groups {
        if records.len() != 4 {
            return Err("missing candidate depth".into());
        }
        let (case, result) = choose_depth(&records)?;
        let row = meta::selected(&case.task)?;
        choices.push(json!({"group":key,"selected_id":case.id(),"depth":case.depth,"task":case.task,"mode":case.mode,
            "architecture":case.architecture,"transport":case.transport,"train_ce":result["train_ce"],"primary_split":result["primary_split"],
            "primary_pass":result["primary_pass"],"all_splits_pass":result["all_splits_pass"],"metrics":result["metrics"],
            "historical_transformer_seed1":row["historical_seed1"],"comparison_is_fresh_matched_transformer_run":false}));
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"cells":288,"complete":complete,"memory_ineligible":ineligible,
        "candidate_primary_passes":primary,"candidate_all_splits_passes":all_splits,"summed_candidate_seconds":seconds,
        "selected_primary_passes":choices.iter().filter(|c|c["primary_pass"]==true).count(),
        "selected_all_splits_passes":choices.iter().filter(|c|c["all_splits_pass"]==true).count(),"choices":choices}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_saved_tensor_audit_matches_the_full_empirical_oracle() -> Result<()> {
        let bank = Bank {
            vocab: 3,
            depth: 1,
            history_slots: 8,
            masks: vec![crate::features::Mask::All],
            transport: Transport::Fourier,
            execution: Execution::Composed,
        };
        let mut model = Head::zero(bank, 10.)?;
        for (k, w) in model.coefficients.iter_mut().enumerate() {
            *w = 0.01 * (k as f64 - 30.);
        }
        let example = data::Example {
            task: "test".into(),
            tokens: vec![0, 1, 2, 0],
            targets: vec![1, 2, -100, 1],
            changes: vec![false; 4],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None,
        };
        let cache = Cache::build(&model.bank, std::slice::from_ref(&example))?;
        let mut gradient = vec![0.; model.coefficients.len()];
        let (loss, count) = head::objective(&model, &cache, &[0], Some(&mut gradient))?;
        let gap = head::linear_gap(&model, &gradient)?;
        let (direct_loss, direct_gap, direct_count) = raw_tensor_oracle(&model, &[example])?;
        assert_eq!(count, direct_count);
        assert_eq!(loss, direct_loss);
        assert_eq!(gap, direct_gap);
        Ok(())
    }
    #[test]
    fn depth_selection_ignores_all_heldout_outcomes_and_preserves_the_frozen_order() -> Result<()> {
        let mut records:Vec<_>=[4,0,2,1].into_iter().map(|depth| {
            (Case{task:"test".into(),depth,architecture:"joint".into(),transport:Transport::Fourier,mode:"full_batch".into()},
                json!({"status":"complete","train_ce":if depth>=2 {1e-9}else{0.1},"primary_pass":depth==4}))
        }).collect();
        assert_eq!(choose_depth(&records)?.0.depth, 2);
        for (_, r) in &mut records {
            r["primary_pass"] = Value::Bool(false);
            r["metrics"] = json!({"validation":{"loss":-1e50}});
        }
        assert_eq!(choose_depth(&records)?.0.depth, 2);
        for (_, r) in &mut records {
            r["train_ce"] = json!(0.5);
        }
        assert_eq!(choose_depth(&records)?.0.depth, 0);
        records[1].1["status"] = json!("ineligible_memory_budget");
        assert_eq!(choose_depth(&records)?.0.depth, 1);
        Ok(())
    }
    #[test]
    fn inventory_covers_every_task_mode_depth_transport_and_architecture_once() -> Result<()> {
        let cases = cases()?;
        assert_eq!(cases.len(), 288);
        assert_eq!(
            cases.iter().map(Case::id).collect::<BTreeSet<_>>().len(),
            288
        );
        for task in [
            "gpt-probe-key-value-binding",
            "basis-depth-easy",
            "basis-depth-hard",
            "basis-recall-easy",
        ] {
            assert_eq!(cases.iter().filter(|c| c.task == task).count(), 72);
        }
        Ok(())
    }
}
