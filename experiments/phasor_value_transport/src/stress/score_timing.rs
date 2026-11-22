//! Compare weight preparation on the same engine, split and greedy decoder.
use super::*;
use std::time::Instant;

pub fn run(name: &str, seed: u32, out: &Path) -> Result<()> {
    let (row, task) = load(name)?;
    let cfg = task_config(&row, &task, true)?;
    let batch = cfg.max_batch;
    let dir = out.join(name).join(format!("seed{seed}"));
    let (saved, weights) = embedding::verified_weights(&row, seed, &cfg, &dir)?;
    let held = &task.splits[&task.selection];
    let mut engine = Engine::new(cfg, &weights)?;
    let reference = metrics::evaluate_batches(&mut engine, held, true, |_, _| Ok(()))?;
    if reference["exact"] != saved["observation"]["correct"]
        || reference != metrics::evaluate(&mut engine, held, true)?
    {
        return Err("warmup scores differ from the saved observation or fixed scoring".into());
    }
    let mut uncached = Vec::new();
    let mut fixed = Vec::new();
    for pair in 0..8 {
        for reuse in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            let scores = if reuse {
                metrics::evaluate(&mut engine, held, true)?
            } else {
                metrics::evaluate_batches(&mut engine, held, true, |_, _| Ok(()))?
            };
            let elapsed = start.elapsed().as_secs_f64();
            if scores != reference {
                return Err("repeat scoring differs from the complete reference metrics".into());
            }
            if reuse {
                fixed.push(elapsed);
            } else {
                uncached.push(elapsed);
            }
        }
    }
    println!(
        "{}",
        json!({"case":name,"seed":seed,"model":MODEL,"kernite_version":KERNITE_VERSION,
            "weights_sha256":saved["weights_sha256"],"correct":reference["exact"],"total":held.len(),
            "max_batch":batch,"uncached_seconds":uncached,"fixed_seconds":fixed})
    );
    Ok(())
}
