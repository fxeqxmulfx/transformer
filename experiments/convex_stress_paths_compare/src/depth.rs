//! Verify every depth slot, including refused allocations and unused models.
use crate::*;
fn id(case: &Value) -> Result<String> {
    let mut id = format!(
        "{}--{}--{}--{}--d{}",
        text(case, "mode")?,
        text(case, "task")?,
        text(case, "architecture")?,
        text(case, "transport")?,
        case["depth"].as_u64().ok_or("missing depth")?
    );
    if let Some(policy) = case["retained_initializer"].as_str() {
        id.push_str(&format!("--retained-{policy}"));
    }
    Ok(id)
}
pub(crate) fn expected(retained: bool) -> Result<Vec<Value>> {
    let p = protocol()?;
    let mut result = vec![];
    for (task, maximum) in p["retained_tasks"]
        .as_object()
        .ok_or("missing depth tasks")?
    {
        let depths = if retained {
            vec![maximum.as_u64().ok_or("invalid maximum depth")?]
        } else {
            vec![0, 1, 2, 4]
        };
        let policies = if retained {
            vec![Some(POLICIES[0]), Some(POLICIES[1])]
        } else {
            vec![None]
        };
        let architectures = if retained {
            vec!["joint"]
        } else {
            vec!["joint", "independent", "additive"]
        };
        for depth in depths {
            for policy in &policies {
                for architecture in &architectures {
                    for transport in TRANSPORTS {
                        for mode in MODES {
                            let mut case = json!({"task":task,"depth":depth,"architecture":architecture,"transport":transport,"mode":mode});
                            if let Some(policy) = policy {
                                case["retained_initializer"] = json!(policy);
                            }
                            result.push(case);
                        }
                    }
                }
            }
        }
    }
    Ok(result)
}
pub(crate) fn load(
    root: &Path,
    inv: &Value,
    summary: &Value,
    suite: &Value,
    retained: bool,
    loaded: &mut Loaded,
) -> Result<()> {
    let expected = expected(retained)?;
    let declared: BTreeMap<_, _> = rows(inv, "cases")?
        .iter()
        .map(|c| Ok((id(c)?, c)))
        .collect::<Result<_>>()?;
    check(
        declared.len() == expected.len()
            && rows(inv, "cases")?.len() == expected.len()
            && summary["cells"] == expected.len(),
        "depth case coverage changed",
    )?;
    let (mut complete, mut refused, mut seconds) = (0, 0, 0.);
    let mut covered = BTreeSet::new();
    for case in expected {
        let key = id(&case)?;
        check(
            declared.get(&key).is_some_and(|c| **c == case),
            "depth case differs from frozen scope",
        )?;
        let dir = root.join(&key);
        let result = loaded.evidence.load(&dir.join("result.json"))?;
        let metadata = loaded.evidence.load(&dir.join("metadata.json"))?;
        let replay = loaded
            .evidence
            .load(&root.join(format!("{key}.replay.json")))?;
        loaded.evidence.bind(
            &dir.join("result.json"),
            Some(text(&replay, "result_sha256")?),
        )?;
        for v in [&result, &metadata, &replay] {
            check(
                v["source_fingerprint"] == inv["source_fingerprint"] && v["case"] == case,
                "depth source or case changed",
            )?;
        }
        check(
            metadata["producer_commit"] == inv["producer_commit"]
                && replay["status"] == "replay_pass",
            "depth producer or replay changed",
        )?;
        data(suite, text(&case, "task")?, &metadata["data_fingerprints"])?;
        let mut record = json!({"source":if retained {"depth_retained"} else {"depth_original"},"id":key,"case":case,
            "status":result["status"],"result":loaded.evidence.bind(&dir.join("result.json"),None)?,
            "replay":loaded.evidence.bind(&root.join(format!("{key}.replay.json")),None)?});
        if result["status"] == "ineligible_memory_budget" {
            check(
                !retained
                    && replay["memory_rejection"] == true
                    && result["heldout_scored"] == false
                    && !dir.join("model.bin").exists(),
                "invalid depth memory refusal",
            )?;
            refused += 1;
        } else {
            check(
                result["status"] == "complete"
                    && result["producer_commit"] == inv["producer_commit"]
                    && replay["original_metrics_equal"] == true
                    && replay["model_sha256"] == result["model_sha256"],
                "depth model is not numerically replayed",
            )?;
            record["model"] = serde_json::to_value(
                loaded
                    .evidence
                    .bind(&dir.join("model.bin"), Some(text(&result, "model_sha256")?))?,
            )?;
            let ce = number(&result, "train_ce")?;
            check(
                (ce - number(&result["metrics"]["train"], "loss")?).abs() <= 1e-10
                    && (ce - number(&replay, "train_ce")?).abs() <= 1e-10,
                "depth ordinary CE differs from original replay",
            )?;
            let (primary, all) =
                report::passes(&result["metrics"], text(&result, "primary_split")?)?;
            check(
                result["primary_pass"] == primary && result["all_splits_pass"] == all,
                "depth success flags differ from original metrics",
            )?;
            if case["mode"] == "forward_only" {
                for k in [
                    "gradient_updates",
                    "optimizer_backward_examples",
                    "optimizer_backward_prefixes",
                    "diagnostic_backward_prefixes",
                    "line_search_forward_prefixes",
                ] {
                    check(result[k] == 0, "forward endpoint contains derivative work")?;
                }
            }
            seconds += number(&result["timing_seconds"], "total")?;
            complete += 1;
            if retained {
                let a = loaded.evidence.bind(
                    &dir.join("features.bin"),
                    Some(text(&result["feature_cache"], "sha256")?),
                )?;
                check(
                    result["feature_cache"]["bytes"] == a.bytes
                        && replay["feature_cache_audit"]["sha256"] == a.sha256,
                    "depth feature cache binding differs",
                )?;
                record["feature_cache"] = serde_json::to_value(a)?;
                let matching: Vec<_> = rows(summary, "choices")?
                    .iter()
                    .filter(|r| r["selected_id"] == key)
                    .collect();
                check(
                    matching.len() == 1,
                    "missing or duplicate retained summary endpoint",
                )?;
                let choice = matching[0];
                for k in [
                    "metrics",
                    "primary_split",
                    "train_ce",
                    "primary_pass",
                    "all_splits_pass",
                ] {
                    check(
                        choice[k] == result[k],
                        "retained summary differs from replayed result",
                    )?;
                }
                covered.insert(key.clone());
                loaded.retained.push(Offer {
                    reference: format!("depth_retained:{key}"),
                    task: text(&case, "task")?.into(),
                    mode: text(&case, "mode")?.into(),
                    policy: Some(text(&case, "retained_initializer")?.into()),
                    transport: Some(text(&case, "transport")?.into()),
                    train_ce: Some(ce),
                    primary: text(&result, "primary_split")?.into(),
                    metrics: result["metrics"].clone(),
                    saved: result.clone(),
                });
            }
        }
        loaded.depth_records.push(record);
    }
    check(
        summary["complete"] == complete
            && summary["memory_ineligible"] == refused
            && (number(summary, "summed_candidate_seconds")? - seconds).abs()
                <= 1e-6 * (1. + seconds),
        "depth accounting changed",
    )?;
    if retained {
        check(
            covered.len() == 48
                && rows(summary, "choices")?.len() == 48
                && summary["choice_between_retained_initializers"] == false,
            "retained initializer policies were collapsed",
        )?;
    }
    loaded.accounting.insert(if retained {"depth_retained"} else {"depth_original"}.into(),json!({
        "complete":complete,"memory_refusals":refused,"candidate_seconds":seconds,"all_costs":"each saved result.json; complete parent summary","charged_once":true}));
    Ok(())
}
