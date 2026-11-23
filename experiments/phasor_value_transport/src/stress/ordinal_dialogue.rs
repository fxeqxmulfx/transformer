//! A synthetic eight-occurrence dialogue with a generated one-word answer.
use super::*;
mod data;
pub(super) use data::{NAME, data as dataset};
pub fn score(seed: u32, out: &Path) -> Result<()> {
    let (row, task) = dataset()?;
    let cfg = task_config(&row, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let mut engine = Engine::for_scoring(cfg, &weights)?;
    let mut scores = BTreeMap::new();
    for (split, rows) in task.splits.iter().filter(|(s, _)| s.as_str() != "train") {
        let mut exact = Vec::new();
        let metric = metrics::evaluate_with_completions(&mut engine, rows, true, |i, tokens| {
            if i != exact.len() {
                return Err("changed ordinal example order".into());
            }
            let gold = rows[i].answer[0];
            exact.push(matches!(tokens,[v] if i64::from(*v)==gold));
            Ok(())
        })?;
        let mut ordinal = [0; 8];
        for (i, &success) in exact.iter().enumerate() {
            ordinal[i % 8] += usize::from(success);
        }
        let (transcripts, rest) = exact.as_chunks::<8>();
        if !rest.is_empty() || metric["exact"] != ordinal.iter().sum::<usize>() {
            return Err("ordinal count disagrees with free generation".into());
        }
        let all = transcripts
            .iter()
            .filter(|r| r.iter().all(|&ok| ok))
            .count();
        scores.insert(
            split,
            json!({"correct":metric["exact"],"total":rows.len(),"by_ordinal":ordinal,
            "all_eight_correct":all,"transcripts":transcripts.len()}),
        );
    }
    if scores[&task.selection]["correct"] != point["correct"] {
        return Err("ordinal held-out replay changed".into());
    }
    println!(
        "{}",
        json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "protocol_sha256":row["catalog_sha256"],"weights_sha256":weight_digest(&weights),"scores":scores})
    );
    Ok(())
}
