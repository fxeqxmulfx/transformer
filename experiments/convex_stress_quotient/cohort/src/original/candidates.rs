use super::*;
/// Only declared-readout-compatible candidates enter the score comparison.
/// Every incompatible candidate remains in its original stage and evaluation.
pub(super) fn offers(name: &str, ledger: &Ledger) -> Result<Vec<comparison::Offer>> {
    let mut offers = vec![];
    for stage in &ledger.stages {
        let rows = stage.retained["endpoints"]
            .as_array()
            .ok_or("missing retained stage endpoints")?;
        if stage.retained["candidate_readout_compatible"] != true {
            continue;
        }
        if rows.len() != 3 || stage.retained["code_admitted"] != true {
            return Err("compatible stage lacks actual endpoints".into());
        }
        for row in rows {
            offers.push(comparison::Offer {
                task: name.into(),
                mode: serde_json::from_value(row["mode"].clone())?,
                prime: stage.prime,
                train_ce: row["train_ce"].as_f64().ok_or("missing actual raw CE")?,
                model_sha256: row["model_sha256"]
                    .as_str()
                    .ok_or("missing model hash")?
                    .into(),
                stage_sha256: stage.sha256.clone(),
            });
        }
    }
    Ok(offers)
}
pub(super) fn evaluate(
    root: &Path,
    name: &str,
    x: &inputs::Inputs,
    ledger: &Ledger,
) -> Result<Value> {
    let mut evaluated = vec![];
    for (index, stage) in ledger.stages.iter().enumerate() {
        if stage.retained["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?
            .is_empty()
        {
            continue;
        }
        let attempt: Attempt = read(&root.join(&stage.file))?;
        if attempt.request != x.config.request(stage.prime)
            || ledger::stage(root, index, &attempt)? != *stage
        {
            return Err("saved original model or stage binding differs before evaluation".into());
        }
        let report = attempt.report.ok_or("model stage has no report")?;
        for (mode, e) in report.endpoints.iter().enumerate() {
            let model = &e.training.model;
            model.validate()?;
            let model_hash = hash(&serde_json::to_vec(model)?);
            let record = &stage.retained["endpoints"][mode];
            if record["model_sha256"] != model_hash || record["train_ce"] != e.audit.train_ce {
                return Err("actual model hash or raw CE differs from retained ledger".into());
            }
            let mut actual = serde_json::Map::new();
            for (split, examples) in &x.task.splits {
                actual.insert(split.clone(), metrics::evaluate(model, examples)?);
            }
            let actual = Value::Object(actual);
            let loss = actual["train"]["loss"]
                .as_f64()
                .filter(|x| x.is_finite())
                .ok_or("missing original CE")?;
            if (loss - e.audit.train_ce).abs() > 1e-8 {
                return Err("usual raw CE differs from original evaluator".into());
            }
            let (primary, all) = passes(&actual, &x.task.selection)?;
            let offer = comparison::Offer {
                task: name.into(),
                mode: e.training.policy.mode,
                prime: stage.prime,
                train_ce: e.audit.train_ce,
                model_sha256: model_hash,
                stage_sha256: stage.sha256.clone(),
            };
            evaluated.push(json!({"reference":offer.reference(),"task":name,"training_mode":offer.mode(),
                "prime":stage.prime,"stage":stage.file,"stage_sha256":stage.sha256,"model_sha256":offer.model_sha256,
                "train_ce":offer.train_ce,"primary_split":x.task.selection,"metrics":actual,
                "primary_pass":primary,"all_splits_pass":all,"audit":e.audit,
                "offer_eligible":report.candidate_readout_compatible==Some(true),
                "eligibility":"declared_original_readout_constant_on_candidate_code_fibers",
                "code_certified":e.certificate.certified,
                "certified_queries":e.certificate.queries.iter().filter(|q|q.decision==certificate::Decision::Passed).count(),
                "raw_label_identification":false,"target_readout_compatibility_proved":false}));
        }
    }
    Ok(json!(evaluated))
}
pub(super) fn verify_stage_bytes(root: &Path, ledger: &Ledger) -> Result<()> {
    let primes = component::catalogue();
    if ledger.stages.len() != primes.len() {
        return Err("incomplete catalogue".into());
    }
    for (index, (stage, prime)) in ledger.stages.iter().zip(primes).enumerate() {
        let file = format!("stage-{index:02}-p{prime}.json");
        if stage.file != file
            || stage.prime != prime
            || fs::metadata(root.join(&file))?.len() != stage.bytes
            || digest(&root.join(file))? != stage.sha256
        {
            return Err("retained original stage bytes changed".into());
        }
    }
    Ok(())
}
/// Bind every saved evaluation, including an ineligible model, to its stage.
/// Numerical metrics are rerun by replay_case; this check also protects the
/// lightweight parent-preserving summary against altered field bindings.
pub(super) fn check_evaluations(
    task: &str,
    input: &Value,
    ledger: &Ledger,
    outcomes: &[Value],
) -> Result<()> {
    let mut expected = BTreeSet::new();
    for stage in &ledger.stages {
        for row in stage.retained["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?
        {
            let offer = comparison::Offer {
                task: task.into(),
                mode: serde_json::from_value(row["mode"].clone())?,
                prime: stage.prime,
                train_ce: row["train_ce"].as_f64().ok_or("missing training CE")?,
                model_sha256: row["model_sha256"]
                    .as_str()
                    .ok_or("missing model hash")?
                    .into(),
                stage_sha256: stage.sha256.clone(),
            };
            if !expected.insert(offer.reference()) {
                return Err("repeated stage mode".into());
            }
            let values: Vec<_> = outcomes
                .iter()
                .filter(|v| v["reference"] == offer.reference())
                .collect();
            if values.len() != 1 {
                return Err("missing or repeated constructed endpoint".into());
            }
            let v = values[0];
            let loss = v["metrics"]["train"]["loss"]
                .as_f64()
                .filter(|v| v.is_finite() && *v >= 0.)
                .ok_or("invalid raw training CE")?;
            if v["task"] != task
                || v["training_mode"] != offer.mode()
                || v["prime"] != stage.prime
                || v["stage"] != stage.file
                || v["stage_sha256"] != stage.sha256
                || v["model_sha256"] != offer.model_sha256
                || v["train_ce"] != offer.train_ce
                || (loss - offer.train_ce).abs() > 1e-8
                || v["audit"] != row["audit"]
                || v["primary_split"] != input["primary_split"]
                || v["offer_eligible"] != (stage.retained["candidate_readout_compatible"] == true)
                || v["code_certified"] != row["code_certified"]
                || v["certified_queries"] != row["certified_queries"]
                || v["raw_label_identification"] != false
                || v["target_readout_compatibility_proved"] != false
            {
                return Err(
                    "constructed endpoint differs from its original stage/readout/metric binding"
                        .into(),
                );
            }
        }
    }
    if outcomes.len() != expected.len() {
        return Err("extra constructed endpoint".into());
    }
    Ok(())
}
