//! Matched read budgets on the same checkpoint and same actual chain length.
//! For each start, one swap of destination names in the chosen table changes its answer.
use super::*;
use std::collections::BTreeSet;

fn follow(tables: &[Vec<(i64, i64)>], mut key: i64) -> Result<i64> {
    for table in tables {
        key = table
            .iter()
            .find(|(source, _)| *source == key)
            .ok_or("incomplete table chain")?
            .1;
    }
    Ok(key)
}

fn pairs(hops: usize, edge: usize, task: &str) -> Result<(Vec<Example>, Vec<Example>)> {
    if !(1..=hops).contains(&edge) {
        return Err("the edited table must lie on the displayed chain".into());
    }
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
            let source = follow(&set.tables[..edge - 1], set.starts[lane])?;
            let table = &mut changed.tables[edge - 1];
            let at = table
                .iter()
                .position(|(key, _)| *key == source)
                .ok_or("missing source at intervention")?;
            let other = (at + 1 + set_index % 4) % table.len();
            let before = table[at].1;
            table[at].1 = table[other].1;
            table[other].1 = before;
            let key = follow(&changed.tables, set.starts[lane])?;
            changed.answers[lane] = set
                .options
                .iter()
                .position(|&v| v == key)
                .ok_or("missing edited option")?;
            if changed.answers[lane] == set.answers[lane] {
                return Err("the edited link kept the answer".into());
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
    edited_edge: Option<usize>,
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
    let (original, edited) = pairs(hops, edited_edge.unwrap_or(hops), study.name)?;
    let n = original.len();
    let hashes = [digest(&original)?, digest(&edited)?];
    let examples: Vec<_> = original.into_iter().chain(edited).collect();
    let mut alias_details = None;
    let walk = if matches!(mode, "isolated-jumps" | "isolated-refined-jumps") {
        crate::typed_lookup::Walk::JumpTables
    } else {
        crate::typed_lookup::Walk::Sequential
    };
    let preds = match mode {
        "isolated-leading" | "isolated-refined" | "isolated-refined-jumps" => {
            let audit = lookup_question::isolated::aliases::run(
                &cfg,
                &weights,
                &task.splits["train"],
                &examples,
                &budgets,
                mode != "isolated-leading",
                walk,
            )?;
            alias_details = Some(audit.details);
            audit.predictions
        }
        "isolated" | "isolated-jumps" => {
            lookup_question::isolated::budget_predictions(&cfg, &weights, &examples, &budgets, walk)?
        }
        "full" | "requested" => {
            cfg.max_batch = task_config(&row, &task, false)?.max_batch;
            cfg.max_length = cfg.max_length.max(examples[0].prompt.len());
            let mut outputs = Vec::new();
            for &depth in &budgets {
                lookup_question::hop_sweep::set_budget(&mut cfg, depth)?;
                let mut engine = Engine::for_scoring(cfg.clone(), &weights)?;
                let predictions = if mode == "requested" {
                    let embeddings = &weights[..cfg.width * cfg.vocab];
                    crate::first_generation::requested_rollout(&mut engine, embeddings, &examples)?
                        .into_iter()
                        .map(|tokens| match tokens.as_slice() {
                            [token] => Some(i64::from(*token)),
                            _ => None,
                        })
                        .collect()
                } else {
                    lookup_question::hop_sweep::generated(&mut engine, &examples)?
                };
                outputs.push(predictions);
            }
            outputs
        }
        _ => {
            return Err(
                "choose full, requested, isolated, isolated-jumps, isolated-leading, isolated-refined or isolated-refined-jumps".into(),
            );
        }
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
    if let Some(edge) = edited_edge {
        result["edited_edge"] = json!(edge);
    }
    println!("{result}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn edits_keep_the_start_choices_and_all_other_tables() {
        let h = 40;
        assert!(pairs(h, 0, NAME).is_err());
        assert!(pairs(h, h + 1, NAME).is_err());
        for edge in [1, h / 2, h] {
            let (a, b) = pairs(h, edge, NAME).unwrap();
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
                        .all(|&i| i > 16 * (edge - 1) && i <= 16 * edge)
                );
                assert_eq!(x.prompt[differences[0]], y.prompt[differences[1]]);
                assert_eq!(x.prompt[differences[1]], y.prompt[differences[0]]);
            }
        }
    }
}
