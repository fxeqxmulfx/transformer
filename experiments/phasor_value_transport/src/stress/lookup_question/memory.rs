//! Free-generation controls for completed chains, missing edges and conflicting later writes.
use super::*;

pub fn run(seed: u32, out: &Path, hops: usize, edge: usize, passes: Option<usize>) -> Result<()> {
    let (row, mut task) = data()?;
    let mut cfg = task_config(&row, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let trained = cfg.head.peer_read().and_then(|r| r.path).map(|p| p.passes);
    if let Some(n) = passes {
        hop_sweep::set_budget(&mut cfg, n)?;
    }
    data::complete_hops(&mut task)?;
    let family = data::memory::family(&task, hops, edge)?;
    cfg.max_length = family
        .arms
        .values()
        .flatten()
        .map(|e| e.prompt.len())
        .max()
        .ok_or("empty memory probe")?;
    cfg.max_batch = task_config(&row, &task, false)?.max_batch;
    let mut engine = Engine::for_scoring(cfg, &weights)?;
    let predictions = family
        .arms
        .iter()
        .map(|(&k, e)| Ok((k, hop_sweep::generated(&mut engine, e)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let base = &family.arms["original"];
    let mut scores = BTreeMap::new();
    for (&name, examples) in &family.arms {
        let mut exact = 0;
        let mut both = 0;
        let mut last_key = 0;
        let mut original = 0;
        for (i, &p) in predictions[name].iter().enumerate() {
            let hit = p == Some(examples[i].answer[0]);
            exact += usize::from(hit);
            both += usize::from(hit && predictions["original"][i] == Some(base[i].answer[0]));
            last_key += usize::from(p == Some(family.last_key_letters[i]));
            original += usize::from(p == Some(base[i].answer[0]));
        }
        let mut result = json!({"last_reachable_key_choice":last_key,"intact_answer_choice":original,"total":examples.len()});
        if name != "missing_reached_edge" {
            result["correct"] = json!(exact);
            result["both_correct_with_original"] = json!(both);
        }
        scores.insert(name, result);
    }
    let mut crossed = BTreeMap::new();
    for (a, b) in [
        ("old_active_old_tail", "old_active_new_tail"),
        ("new_active_old_tail", "new_active_new_tail"),
        ("old_active_old_tail", "new_active_old_tail"),
        ("old_active_new_tail", "new_active_new_tail"),
        ("old_overwritten_inside", "old_with_same_row_outside"),
        ("new_overwritten_inside", "new_with_same_row_outside"),
    ] {
        let both = (0..base.len())
            .filter(|&i| {
                predictions[a][i] == Some(family.arms[a][i].answer[0])
                    && predictions[b][i] == Some(family.arms[b][i].answer[0])
            })
            .count();
        crossed.insert(format!("{a}/{b}"), both);
    }
    println!(
        "{}",
        json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "weights_sha256":weight_digest(&weights),"training_protocol_sha256":row["catalog_sha256"],
        "trained_passes":trained,"evaluated_passes":passes.or(trained),"hops":hops,"edited_edge":edge,
        "split_sha256":family.digests,"scores":scores,"both_correct_by_pair":crossed})
    );
    Ok(())
}
