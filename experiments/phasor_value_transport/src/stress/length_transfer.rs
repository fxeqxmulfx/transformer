//! Rebuild the longest original length probe before changing the source length.
//! BlockCut and CraspLengthScore give the label at every supervised prefix.
use super::*;
pub(super) fn add(task: &mut Task, row: &Value, length: usize) -> basis::Result<(String, String)> {
    let (spec, reference) =
        basis::length::specification(row["name"].as_str().ok_or("missing source name")?)?;
    let anchor = format!("test/length-{reference}");
    let count = task
        .splits
        .get(&anchor)
        .ok_or("missing original test length")?
        .len();
    let expected = row["data_fingerprints"][&anchor]
        .as_str()
        .ok_or("missing source length fingerprint")?;
    let (digest, rows) = basis::length::verified_rows(spec, reference, count, expected, length)?;
    let split = format!("test/depth-length-{length}");
    task.splits.insert(
        split.clone(),
        rows.into_iter().map(catalog::sequence_example).collect(),
    );
    Ok((split, digest))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transfer_reproduces_source_fingerprints_and_extends_both_oracles() {
        for (name, anchor, length) in [("language-l2", 128, 256), ("suite-crasp-depth-1", 64, 128)]
        {
            let (row, mut task) = load(name).unwrap();
            let original = format!("test/length-{anchor}");
            let original_hash = row["data_fingerprints"][&original].as_str().unwrap();
            let (base, digest) = add(&mut task, &row, anchor).unwrap();
            assert_eq!(digest, original_hash);
            assert!(
                task.splits[&base]
                    .iter()
                    .zip(&task.splits[&original])
                    .all(|(a, b)| a.tokens == b.tokens
                        && a.targets == b.targets
                        && a.changes == b.changes)
            );
            let (long, hash) = add(&mut task, &row, length).unwrap();
            assert_ne!(hash, original_hash);
            assert_eq!(task.splits[&long].len(), task.splits[&original].len());
            assert!(
                task.splits[&long]
                    .iter()
                    .all(|e| e.tokens.len() == length && e.targets.len() == length)
            );
            let mut altered = row.clone();
            altered["data_fingerprints"][&original] = json!("wrong");
            assert!(add(&mut task, &altered, length).is_err());
        }
    }
}
