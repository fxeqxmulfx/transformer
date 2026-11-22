//! Matched removals of the triple transport read, scored on the frozen task splits.
use super::*;

pub fn run(name: &str, seed: u32, out: &Path) -> Result<()> {
    if !architecture::TRIPLE_TRANSPORT {
        return Err("read removal requires triple transport".into());
    }
    let (row, task) = load(name)?;
    let cfg = task_config(&row, &task, false)?;
    let dir = out.join(name).join(format!("seed{seed}"));
    let (saved, original) = embedding::verified_weights(&row, seed, &cfg, &dir)?;
    let mut engine = Engine::new(cfg.clone(), &original)?;
    let mut score = |weights: &[f32]| -> Result<Value> {
        engine.parameters.write(weights)?;
        let mut splits = BTreeMap::new();
        for (split, examples) in &task.splits {
            if split == &task.selection || split == "test" || split.starts_with("test/") {
                let counts = metrics::evaluate(&mut engine, examples, true)?;
                splits.insert(
                    split,
                    json!({"correct":counts["exact"],"total":examples.len()}),
                );
            }
        }
        Ok(json!(splits))
    };
    let baseline = score(&original)?;
    if baseline[&task.selection]["correct"] != saved["observation"]["correct"]
        || baseline[&task.selection]["total"] != saved["observation"]["total"]
    {
        return Err("triple-transport baseline differs from the saved held-out observation".into());
    }
    let mut results = BTreeMap::new();
    let mut weights = original.clone();
    for (name, removed) in [
        ("without_context", &[0][..]),
        ("without_direct", &[1][..]),
        ("without_constant", &[2][..]),
        ("without_transport", &[3][..]),
        ("context_only", &[1, 2, 3][..]),
        ("structural_only", &[0, 1][..]),
    ] {
        weights.copy_from_slice(&original);
        crate::slot_checks::disable_reads(&cfg, &mut weights, removed);
        results.insert(name, score(&weights)?);
    }
    println!(
        "{}",
        json!({"case":name,"seed":seed,"model":MODEL,"weights_sha256":saved["weights_sha256"],
        "kernite_version":trained_with(&saved)?,"scored_with_kernite":KERNITE_VERSION,
        "step":saved["observation"]["step"],"selection":task.selection,"baseline":baseline,"ablations":results})
    );
    Ok(())
}
