use crate::*;
pub fn run(config: Configuration, root: &Path, producer: &str) -> Result<()> {
    if root.exists() {
        return Err("catalogue destination already exists".into());
    }
    let meta = metadata(&config, producer)?;
    fs::create_dir(root)?;
    write(&root.join("configuration.json"), &config)?;
    write(&root.join("metadata.json"), &meta)?;
    let mut out = Ledger {
        metadata: meta,
        stages: vec![],
        seconds_per_stage: vec![],
        selected_stages: [None; 3],
    };
    for (index, prime) in component::catalogue().into_iter().enumerate() {
        let started = std::time::Instant::now();
        let attempt = Attempt::run(config.request(prime));
        if let Some(report) = &attempt.report {
            binding::check(&attempt.request, report)?;
        }
        write(
            &root.join(format!("stage-{index:02}-p{prime}.json")),
            &attempt,
        )?;
        out.stages.push(ledger::stage(root, index, &attempt)?);
        out.seconds_per_stage.push(started.elapsed().as_secs_f64());
        // The complete field/model/certificate report is released here.
    }
    out.selected_stages = ledger::select(&out.stages)?;
    write(&root.join("ledger.json"), &out)
}
pub(crate) fn retained(root: &Path) -> Result<(Configuration, Ledger)> {
    let config: Configuration = read(&root.join("configuration.json"))?;
    let saved: Ledger = read(&root.join("ledger.json"))?;
    let producer = saved.metadata["producer_commit"]
        .as_str()
        .ok_or("missing producer commit")?;
    if saved.metadata != metadata(&config, producer)?
        || read::<Value>(&root.join("metadata.json"))? != saved.metadata
        || saved.stages.len() != component::catalogue().len()
        || saved.seconds_per_stage.len() != saved.stages.len()
        || saved
            .seconds_per_stage
            .iter()
            .any(|s| !s.is_finite() || *s < 0.)
    {
        return Err(
            "original configuration, source/binary, stage coverage or timings changed".into(),
        );
    }
    Ok((config, saved))
}
pub fn replay(root: &Path) -> Result<Value> {
    let (config, saved) = retained(root)?;
    let mut stages = vec![];
    for (index, prime) in component::catalogue().into_iter().enumerate() {
        let file = format!("stage-{index:02}-p{prime}.json");
        if saved.stages[index].prime != prime
            || saved.stages[index].file != file
            || digest(&root.join(&file))? != saved.stages[index].sha256
        {
            return Err("original stage order or sealed bytes changed".into());
        }
        let attempt: Attempt = read(&root.join(&file))?;
        attempt.check(&config.request(prime))?;
        stages.push(ledger::stage(root, index, &attempt)?);
    }
    if stages != saved.stages || ledger::select(&stages)? != saved.selected_stages {
        return Err(
            "actual stage work, model/audit binding or observed-CE selection differs".into(),
        );
    }
    Ok(json!({"replayed":true,"source_fingerprint":fingerprint(),
        "ledger_sha256":digest(&root.join("ledger.json"))?,"stages":stages.len(),
        "selected_stages":saved.selected_stages,
        "admitted_codes":stages.iter().filter(|s|s.retained["code_admitted"]==true).count(),
        "returned_errors":stages.iter().filter(|s|s.retained["returned_error"].is_string()).count(),
        "all_original_stage_reports_replayed":true,"raw_label_identification":false,
        "original_benchmark_comparison":false}))
}
