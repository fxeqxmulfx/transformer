//! Matched read budgets on the same checkpoint and same actual chain length.
//! For each start, one swap of destination names in the final table changes its answer.
use super::*;
use std::collections::BTreeSet;

fn pairs(hops: usize, task: &str) -> Result<(Vec<Example>, Vec<Example>)> {
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i64))
        .collect();
    let mut rng = random::Random::new(3_107_419 + hops as u32);
    let mut original = Vec::new();
    let mut rewired = Vec::new();
    for set_index in 0..20 {
        let set = sample(hops, &mut rng)?;
        original.extend(set.rows(&ids, task));
        for lane in 0..5 {
            let mut changed = set.clone();
            let end = set.options[set.answers[lane]];
            let last = changed.tables.last_mut().ok_or("empty chain")?;
            let at = last
                .iter()
                .position(|(_, v)| *v == end)
                .ok_or("endpoint absent")?;
            let other = (at + 1 + set_index % 4) % last.len();
            let next = last[other].1;
            last[at].1 = next;
            last[other].1 = end;
            // Recompute this start's answer using the printed map at every level.
            let mut key = set.starts[lane];
            for t in &changed.tables {
                key = t
                    .iter()
                    .find(|(k, _)| *k == key)
                    .ok_or("incomplete edited path")?
                    .1;
            }
            changed.answers[lane] = set
                .options
                .iter()
                .position(|&v| v == key)
                .ok_or("missing edited option")?;
            if changed.answers[lane] == set.answers[lane] {
                return Err("last-link edit kept the answer".into());
            }
            rewired.push(changed.rows(&ids, task).into_iter().nth(lane).unwrap());
        }
    }
    Ok((original, rewired))
}

pub fn run(
    seed: u32,
    out: &Path,
    hops: usize,
    budgets: &str,
    mode: &str,
    case: Option<&str>,
) -> Result<()> {
    let budgets: Vec<usize> = budgets
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<BTreeSet<_>, _>>()?
        .into_iter()
        .collect();
    if budgets.is_empty() || budgets.contains(&0) || !budgets.contains(&hops) {
        return Err("provide positive read budgets including the chain length".into());
    }
    let study =
        Training::named(case.unwrap_or(NAME)).ok_or("unknown disjoint-layer training study")?;
    let (row, task) = dataset(study)?;
    let mut cfg = task_config(&row, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let trained = cfg.head.peer_read().and_then(|r| r.path).map(|r| r.passes);
    let (original, edited) = pairs(hops, study.name)?;
    let n = original.len();
    let hashes = [digest(&original)?, digest(&edited)?];
    let examples: Vec<_> = original.into_iter().chain(edited).collect();
    let mut alias_details = None;
    let preds = match mode {
        "isolated-leading" | "isolated-refined" => {
            let audit = lookup_question::isolated::aliases::run(
                &cfg,
                &weights,
                &task.splits["train"],
                &examples,
                &budgets,
                mode == "isolated-refined",
            )?;
            alias_details = Some(audit.details);
            audit.predictions
        }
        "isolated" => {
            lookup_question::isolated::budget_predictions(&cfg, &weights, &examples, &budgets)?
        }
        "full" => {
            cfg.max_batch = task_config(&row, &task, false)?.max_batch;
            cfg.max_length = cfg.max_length.max(examples[0].prompt.len());
            let mut outputs = Vec::new();
            for &depth in &budgets {
                lookup_question::hop_sweep::set_budget(&mut cfg, depth)?;
                let mut engine = Engine::for_scoring(cfg.clone(), &weights)?;
                outputs.push(lookup_question::hop_sweep::generated(
                    &mut engine,
                    &examples,
                )?);
            }
            outputs
        }
        _ => return Err("choose full, isolated, isolated-leading or isolated-refined".into()),
    };
    let reference = &preds[budgets.binary_search(&hops).unwrap()];
    let letters: BTreeSet<_> = examples[..n].iter().map(|e| e.answer[0]).collect();
    let valid = |pred: &[Option<i64>]| {
        pred.iter()
            .flatten()
            .filter(|id| letters.contains(id))
            .count()
    };
    let mut results = BTreeMap::new();
    for (&depth, pred) in budgets.iter().zip(&preds) {
        let mut paired = [0; 4];
        let mut changed = 0;
        let mut same = 0;
        let mut histogram = [0usize; 6];
        for i in 0..n {
            let a = pred[i] == Some(examples[i].answer[0]);
            let b = pred[n + i] == Some(examples[n + i].answer[0]);
            paired[2 * usize::from(a) + usize::from(b)] += 1;
            changed += usize::from(pred[i] != pred[n + i]);
            same += usize::from(pred[i] == reference[i]);
        }
        for (set, answers) in examples[..n]
            .as_chunks::<5>()
            .0
            .iter()
            .zip(pred[..n].as_chunks::<5>().0)
        {
            let right = set
                .iter()
                .zip(answers)
                .filter(|(e, a)| **a == Some(e.answer[0]))
                .count();
            histogram[right] += 1;
        }
        let [neither, edited_only, original_only, both] = paired;
        results.insert(
            depth,
            json!({
                "correct": original_only + both, "edited_correct": edited_only + both,
                "both_correct": both, "neither_correct": neither, "prediction_changed": changed,
                "valid_letters": valid(&pred[..n]), "edited_valid_letters": valid(&pred[n..]),
                "same_original_output_as_hops_reads": same, "sets_by_correct_answers": histogram,
                "prediction_sha256": hash(&serde_json::to_vec(pred)?)
            }),
        );
    }
    let mut result = json!({"study":study.name,"mode":mode,"model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "trained_reads":trained,"hops":hops,"pairs":n,"weights_sha256":weight_digest(&weights),"protocol_sha256":row["catalog_sha256"],
        "original_split_sha256":hashes[0],"edited_split_sha256":hashes[1],"budgets":results});
    if let Some(audit) = alias_details {
        result["role_aliases"] = audit;
    }
    println!("{result}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn last_link_pairs_keep_the_start_choices_and_earlier_tables() {
        let h = 40;
        let (a, b) = pairs(h, NAME).unwrap();
        assert_eq!(
            digest(&a).unwrap(),
            digest(&probes(h, NAME).unwrap()).unwrap()
        );
        for (x, y) in a.iter().zip(&b) {
            assert_ne!(x.answer, y.answer);
            let differences: Vec<_> = x
                .prompt
                .iter()
                .zip(&y.prompt)
                .enumerate()
                .filter(|(_, (u, v))| u != v)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(differences.len(), 2);
            assert!(
                differences
                    .iter()
                    .all(|&i| i >= 1 + 16 * (h - 1) && i < 1 + 16 * h)
            );
            assert_eq!(x.prompt[differences[0]], y.prompt[differences[1]]);
            assert_eq!(x.prompt[differences[1]], y.prompt[differences[0]]);
        }
    }
}
