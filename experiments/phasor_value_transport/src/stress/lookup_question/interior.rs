//! Score each intervention against its matching original on the frozen model.
use super::*;

pub fn run(seed: u32, out: &Path, hops: usize, passes: Option<usize>) -> Result<()> {
    let (row, mut task) = data()?;
    let mut cfg = task_config(&row, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let trained = cfg.head.peer_read().and_then(|r| r.path).map(|p| p.passes);
    if let Some(n) = passes {
        hop_sweep::set_budget(&mut cfg, n)?;
    }
    data::complete_hops(&mut task)?;
    let group = data::interior::group(&task, hops)?;
    let mut engine = Engine::for_scoring(cfg, &weights)?;
    let base = &group.arms["original"];
    let predictions = hop_sweep::generated(&mut engine, base)?;
    let letters = data::vocabulary();
    let letter = |x: Option<i64>| {
        x.and_then(|v| usize::try_from(v).ok())
            .and_then(|v| letters.get(v))
            .is_some_and(|word| data::LETTERS.contains(&word.as_str()))
    };
    let mut scores = BTreeMap::new();
    for (name, examples) in &group.arms {
        let modified = if name == "original" {
            predictions.clone()
        } else {
            hop_sweep::generated(&mut engine, examples)?
        };
        let mut correct = 0;
        let mut both = 0;
        let mut changed = 0;
        let mut gold_changed = 0;
        for (i, b) in modified.into_iter().enumerate() {
            let a = predictions[i];
            let ok = b == Some(examples[i].answer[0]);
            correct += usize::from(ok);
            both += usize::from(ok && a == Some(base[i].answer[0]));
            changed += usize::from(letter(a) && letter(b) && a != b);
            gold_changed += usize::from(base[i].answer != examples[i].answer);
        }
        scores.insert(
            name,
            json!({"correct":correct,"both_correct":both,"different_valid_letters":changed,
            "reference_changed":gold_changed,"total":examples.len()}),
        );
    }
    println!(
        "{}",
        json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "weights_sha256":weight_digest(&weights),"training_protocol_sha256":row["catalog_sha256"],
        "trained_passes":trained,"evaluated_passes":passes.or(trained),"hops":hops,"split_sha256":group.digests,"scores":scores})
    );
    Ok(())
}
