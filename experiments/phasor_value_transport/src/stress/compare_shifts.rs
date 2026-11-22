//! Replay two independently trained Phasors and measure their aligned class
//! probabilities. The complete pair of label tables belongs to the evaluator.
use super::*;
use class_translation::{ScoredPair, probabilities};

struct Table {
    result: Value,
    vocab: usize,
    training: Vec<ScoredPair>,
    held_out: Vec<ScoredPair>,
}

fn scores(e: &mut Engine, examples: &[Example]) -> Result<Vec<ScoredPair>> {
    let p = e.config.vocab;
    let mut scored = Vec::with_capacity(examples.len());
    for batch in examples.chunks(e.config.max_batch) {
        let mut pairs = Vec::with_capacity(batch.len());
        for example in batch {
            let ([a, b], [-100, target]) = (example.tokens.as_slice(), example.targets.as_slice())
            else {
                return Err(
                    "class-table comparison requires two inputs and a single final target".into(),
                );
            };
            if !example.prompt.is_empty()
                || [*a, *b, *target]
                    .iter()
                    .any(|&x| !(0..p as i64).contains(&x))
            {
                return Err("invalid source row in class-table comparison".into());
            }
            pairs.push(([*a as usize, *b as usize], *target as usize));
        }
        let tokens: Vec<u32> = pairs
            .iter()
            .flat_map(|(pair, _)| pair.map(|x| x as u32))
            .collect();
        e.forward(&tokens, &vec![u32::MAX; tokens.len()], batch.len(), 2)?;
        let logits = e.logits.read()?;
        let predicted = e.predictions.read()?;
        for (i, (input, target)) in pairs.into_iter().enumerate() {
            let pos = 2 * i + 1;
            scored.push(ScoredPair {
                input,
                target,
                predicted: predicted[pos] as usize,
                probabilities: probabilities(&logits[pos * p..(pos + 1) * p])?,
            });
        }
    }
    Ok(scored)
}

fn table(name: &str, seed: u32, updates: usize, out: &Path) -> Result<Table> {
    let (result, bytes) = benchmark::replay_catalog(name, seed, updates, out)?;
    let (row, task) = catalog::load(name)?;
    let mut cfg = task_config(&row, &task, false)?;
    cfg.head = architecture::HEAD;
    let mut e = snapshot::load(&bytes, cfg)?;
    let training = scores(&mut e, &task.splits["train"])?;
    let held_out = scores(&mut e, &task.splits[&task.selection])?;
    let correct = held_out.iter().filter(|x| x.predicted == x.target).count();
    let expected = &result["scores"][&task.selection];
    if expected["correct"] != correct || expected["total"] != held_out.len() {
        return Err("all-masked inference differs from the fixed-budget held-out score".into());
    }
    Ok(Table {
        result,
        vocab: task.vocab,
        training,
        held_out,
    })
}

pub fn run(base: &str, other: &str, seed: u32, updates: usize, out: &Path) -> Result<()> {
    let reference = table(base, seed, updates, out)?;
    let candidate = table(other, seed, updates, out)?;
    let common = |v: &Value| {
        let mut protocol = v["protocol"].as_object().cloned().unwrap_or_default();
        protocol.remove("case");
        protocol.remove("suite_sha256");
        protocol
    };
    if reference.vocab != candidate.vocab || common(&reference.result) != common(&candidate.result)
    {
        return Err(
            "class comparison requires matched initialization, inputs and training protocols"
                .into(),
        );
    }
    let p = reference.vocab;
    let mut shifts = vec![None; p.checked_mul(p).ok_or("excessive comparison vocabulary")?];
    let training =
        class_translation::split(&reference.training, &candidate.training, p, &mut shifts)?;
    let held = class_translation::split(&reference.held_out, &candidate.held_out, p, &mut shifts)?;
    let summary = json!({"reference_case":base,"candidate_case":other,"model":MODEL,
        "kernite_version":KERNITE_VERSION,"seed":seed,"updates":updates,
        "reference_weights_sha256":reference.result["weights_sha256"],
        "candidate_weights_sha256":candidate.result["weights_sha256"],
        "initial_weights_sha256":reference.result["protocol"]["initial_weights_sha256"],
        "ordered_inputs_sha256":reference.result["protocol"]["ordered_inputs_sha256"],
        "row_plus_column_shift":class_translation::separable(&shifts,p)?,
        "training":training,"held_out":held});
    println!("{summary}");
    Ok(())
}
