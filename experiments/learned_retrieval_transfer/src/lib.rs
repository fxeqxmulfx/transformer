//! Source-derived original multi-query data and raw sequence scoring. The
//! predictor and both update rules are the task404 implementation.
pub use baseline::{Result, data, hash, random::Random};
use learned_retrieval::cohort::{self, Protocol, Study};
use serde_json::{Value, json};

// Reuse the four reviewed original data and fingerprint modules, with the
// same Example and RNG types as the learner.
#[path = "../../convex_stress_sequences/src/dataset.rs"]
mod dataset;
#[path = "../../convex_stress_sequences/src/prefix.rs"]
mod prefix;
#[path = "../../convex_stress_sequences/src/retrieval.rs"]
mod retrieval;
#[path = "../../convex_stress_sequences/src/wire.rs"]
mod wire;
pub const CATALOG: &str = include_str!("../../convex_stress_catalog/protocol.json");
const DEFINITION: &str = include_str!("../protocol.json");

pub struct Case {
    pub study: Study,
    pub task: data::Task,
    pub criteria: Value,
}

fn inputs(case: &str) -> Result<(Protocol, dataset::Dataset, Value)> {
    let definition: Value = serde_json::from_str(DEFINITION)?;
    let expected = definition["cases"][case]
        .as_str()
        .ok_or("case outside frozen transfer")?;
    if definition["catalog_sha256"] != hash(CATALOG.as_bytes())
        || definition["model_protocol_sha256"] != hash(cohort::PROTOCOL.as_bytes())
    {
        return Err("frozen catalog or parent training protocol changed".into());
    }
    let data = dataset::build(case)?;
    let metadata = data.metadata();
    if hash(&serde_json::to_vec(&metadata)?) != expected {
        return Err(
            format!("source-derived data or tensor fingerprints changed for {case}").into(),
        );
    }
    let mut p = cohort::protocol()?;
    p.name = definition["name"]
        .as_str()
        .ok_or("missing protocol name")?
        .into();
    p.case = case.into();
    p.context = usize::try_from(definition["context"].as_u64().ok_or("missing context")?)?;
    p.suite_sha256 = hash(CATALOG.as_bytes());
    Ok((p, data, definition))
}
pub fn case(name: &str) -> Result<Case> {
    let (p, data, definition) = inputs(name)?;
    let pins = json!({"original":data.original_fingerprints,"tensor":data.tensor_fingerprints});
    let study = Study::new(p, DEFINITION, pins)?;
    Ok(Case {
        study,
        task: data.task,
        criteria: definition,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use learned_retrieval::evaluation::prefix_split;

    #[test]
    fn reviewed_splits_and_training_settings_match() -> Result<()> {
        let previous = cohort::protocol()?;
        assert!(inputs("unregistered").is_err());
        for case in ["suite-mqar", "suite-lookup"] {
            let (p, d, definition) = inputs(case)?;
            assert_eq!(d.task.splits.len(), 11);
            assert_eq!(d.task.selection, "validation/novel");
            assert_eq!(p.context, 96);
            assert_eq!(definition["target"], 0.95);
            assert_eq!(
                p.initial(d.task.vocab, 1)?.weights[..2 * d.task.vocab * 16],
                previous.initial(d.task.vocab, 1)?.weights[..2 * d.task.vocab * 16]
            );
            assert_eq!(
                (p.batch, p.updates, p.seeds, p.modes),
                (32, 1024, previous.seeds.clone(), previous.modes.clone())
            );
            for (name, rows) in &d.task.splits {
                let expected = if name.starts_with("train") {
                    128
                } else if name.starts_with("test") {
                    64
                } else {
                    32
                };
                assert_eq!(rows.len(), expected);
                assert!(
                    rows.iter()
                        .all(|row| row.targets.iter().filter(|&&y| y >= 0).count() == 2)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn raw_and_exact_use_both_supervised_positions() -> Result<()> {
        let (p, mut d, _) = inputs("suite-mqar")?;
        let m = p.initial(d.task.vocab, 0)?;
        let mut rows = d
            .task
            .splits
            .remove("validation")
            .ok_or("missing fixture")?;
        rows.truncate(2);
        let mut expected = vec![];
        for (i, row) in rows.iter_mut().enumerate() {
            let mut ordinal = 0;
            for (pos, label) in row.targets.iter_mut().enumerate() {
                if *label < 0 {
                    continue;
                }
                let logits = m.logits(&row.tokens[..=pos])?;
                let winner = baseline::channel::argmax(&logits);
                *label = ((winner + usize::from(i == 1 && ordinal == 1)) % m.vocabulary) as i64;
                expected.extend(logits.iter().flat_map(|x| x.to_le_bytes()));
                ordinal += 1;
            }
        }
        let (report, raw) = prefix_split(&m, &rows)?;
        assert_eq!(raw, expected);
        assert_eq!(report["metrics"]["correct"], 3);
        assert_eq!(report["metrics"]["exact"], 1);
        assert_eq!(report["metrics"]["token_accuracy"], 0.75);
        assert_eq!(report["metrics"]["sequence_accuracy"], 0.5);
        assert_eq!(report["raw99"], false);
        rows[0].targets.push(0);
        assert!(prefix_split(&m, &rows).is_err());
        Ok(())
    }
}
