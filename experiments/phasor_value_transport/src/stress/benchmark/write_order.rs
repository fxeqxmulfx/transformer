//! Matched write-order decisions from the original exact-sequence scorer.
use super::*;
use basis::wire;

struct Group {
    owners: Vec<usize>,
    early: Vec<Example>,
    late: Vec<Example>,
}

fn decisions(engine: &mut Engine, rows: &[Example]) -> Result<Vec<bool>> {
    rows.iter()
        .map(|row| {
            let score =
                metrics::evaluate_batches(engine, std::slice::from_ref(row), true, |_, _| Ok(()))?;
            match (score["examples"].as_u64(), score["exact"].as_u64()) {
                (Some(1), Some(0)) => Ok(false),
                (Some(1), Some(1)) => Ok(true),
                _ => Err("expected one decision from the source exact-sequence scorer".into()),
            }
        })
        .collect()
}

fn matched(owners: &[usize], source: &[bool], early: &[bool], late: &[bool]) -> Result<Value> {
    if owners.len() != early.len() || owners.len() != late.len() {
        return Err("write-order decisions differ in length".into());
    }
    let mut codes = Vec::with_capacity(owners.len());
    for ((&owner, &a), &b) in owners.iter().zip(early).zip(late) {
        let original = source.get(owner).ok_or("missing original decision")?;
        codes.push((u8::from(*original) << 2) | (u8::from(a) << 1) | u8::from(b));
    }
    let correct = |mask| codes.iter().filter(|&&c| c & mask == mask).count();
    Ok(
        json!({"pairs":codes.len(),"original_correct":correct(4),"early_correct":correct(2),
        "late_correct":correct(1),"both_orders_correct":correct(3),"original_and_both_correct":correct(7),
        "decision_sha256":hash(&codes)}),
    )
}

pub fn target_writes(name: &str, seed: u32, updates: usize, out: &Path) -> Result<()> {
    let (row, task, _) = catalog::load_declared(name)?;
    let original = task
        .splits
        .get("test")
        .ok_or("missing original recall test")?;
    let (spec, rows) = recall_test(name, original).map_err(|e| e.to_string())?;
    let panel = recall_writes::panel(name, &spec, &rows).map_err(|e| e.to_string())?;
    let mut cfg = task_config(&row, &task, false)?;
    cfg.max_batch = 1;
    let mut fingerprints = json!({"source_test":wire::split_hash(&spec, "test", &rows)});
    let mut groups = BTreeMap::new();
    for (name, group) in panel.groups {
        fingerprints[name] = json!({
            "early":wire::split_hash(&panel.spec, "test", &group.early),
            "late":wire::split_hash(&panel.spec, "test", &group.late),
            "source_indices":hash(&serde_json::to_vec(&group.owners)?)
        });
        cfg.max_length = group
            .early
            .iter()
            .map(|e| e.tokens.len())
            .fold(cfg.max_length, usize::max);
        groups.insert(
            name,
            Group {
                owners: group.owners,
                early: group
                    .early
                    .into_iter()
                    .map(catalog::sequence_example)
                    .collect(),
                late: group
                    .late
                    .into_iter()
                    .map(catalog::sequence_example)
                    .collect(),
            },
        );
    }
    let result = measure(Source::DeclaredTarget, name, seed, updates, out, false)?;
    let dir = directory(out, MODEL, name, seed, updates).join("declared-target");
    let checkpoint = |file: &str, step: &Value, digest: &Value, scores: &Value| -> Result<Value> {
        let bytes = verified_file(&dir.join(file), digest)?;
        let mut engine = checkpoint_engine(cfg.clone(), &bytes)?;
        engine.with_fixed_slot_weights(|engine| {
            let baseline = decisions(engine, original)?;
            let observed =
                json!({"correct":baseline.iter().filter(|b| **b).count(),"total":baseline.len()});
            if scores["test"] != observed {
                return Err("write-order scorer disagrees with original checkpoint test".into());
            }
            let mut paired = BTreeMap::new();
            for (&name, group) in &groups {
                let early = decisions(engine, &group.early)?;
                let late = decisions(engine, &group.late)?;
                paired.insert(name, matched(&group.owners, &baseline, &early, &late)?);
            }
            Ok(json!({"step":step,"weights_sha256":digest,"test":observed,"groups":paired}))
        })
    };
    let (first, last) = at_target_checkpoints(&result, checkpoint)?;
    println!(
        "{}",
        json!({"protocol":result["protocol"],"scored_with_kernite":KERNITE_VERSION,
        "probe":"recall-write-order-v1","fingerprints":fingerprints,"first":first,"final":last})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_correctness_uses_the_original_row_of_each_write() -> Result<()> {
        let report = matched(
            &[1, 0, 1, 0],
            &[false, true],
            &[true, true, true, false],
            &[true, true, false, true],
        )?;
        assert_eq!(report["pairs"], 4);
        assert_eq!(report["both_orders_correct"], 2);
        assert_eq!(report["original_and_both_correct"], 1);
        assert_eq!(report["original_correct"], 2);
        assert_eq!(report["early_correct"], 3);
        assert_eq!(report["late_correct"], 3);
        assert!(matched(&[2], &[], &[true], &[true]).is_err());
        assert!(matched(&[0], &[true], &[true], &[]).is_err());
        Ok(())
    }
}
