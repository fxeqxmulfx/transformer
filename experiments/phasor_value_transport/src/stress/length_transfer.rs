//! Rebuild the longest original length probe before changing the source length.
//! BlockCut and CraspLengthScore give the label at every supervised prefix.
use super::*;
use basis::previous as remaining_synthetic;

fn source(name: &str) -> basis::Result<(Value, usize)> {
    let canonical = match name {
        "basis-depth-easy" => Some("language-l2"),
        "basis-depth-hard" => Some("language-l4"),
        _ => None,
    };
    if let Some(language) = canonical {
        let mut spec = basis::dataset::specification(language)?;
        spec["neutral_fraction"] = json!(0.25);
        return Ok((spec, 128));
    }
    if basis::CASES.contains(&name) {
        let spec = basis::dataset::specification(name)?;
        if spec["task"] == "blocks" {
            return Ok((spec, 128));
        }
    } else if remaining_synthetic::CASES.contains(&name) {
        let (spec, lengths) = remaining_synthetic::dataset::specification(name)?;
        if spec["task"] == "crasp" {
            return Ok((
                spec,
                lengths.into_iter().max().ok_or("missing source lengths")?,
            ));
        }
    }
    Err("depth-length requires a Basis depth, block language or C-RASP checkpoint".into())
}

fn test_rows(spec: &Value, count: usize) -> basis::Result<Vec<basis::data::Example>> {
    if spec["task"] == "blocks" {
        return basis::generator::generate(spec, "test", count, &Default::default(), false);
    }
    let rows = remaining_synthetic::dataset::generate(spec, "test", count)?;
    let depth = remaining_synthetic::spec::number(spec, "formula_depth")?;
    let seed = remaining_synthetic::spec::number(spec, "formula_seed")?.try_into()?;
    let program = remaining_synthetic::crasp::select(depth, seed)?;
    for row in &rows {
        let word = row.tokens.strip_prefix(&[1]).ok_or("missing C-RASP BOS")?;
        let labels = row
            .targets
            .strip_prefix(&[-100])
            .ok_or("wrong C-RASP BOS mask")?;
        let recounted = program.formula.independent(word)?;
        if !labels
            .iter()
            .copied()
            .eq(recounted.into_iter().map(|b| if b { 16 } else { 15 }))
        {
            return Err(
                "independent inclusive recount differs from the generated C-RASP targets".into(),
            );
        }
    }
    Ok(rows)
}

pub(super) fn add(task: &mut Task, row: &Value, length: usize) -> basis::Result<(String, String)> {
    let (mut spec, reference) = source(row["name"].as_str().ok_or("missing source name")?)?;
    spec["min_length"] = Value::Null;
    let anchor = format!("test/length-{reference}");
    let count = task
        .splits
        .get(&anchor)
        .ok_or("missing original test length")?
        .len();
    let mut rows_at = |size: usize| -> basis::Result<_> {
        spec["length"] = json!(size);
        let rows = test_rows(&spec, count)?;
        Ok((basis::wire::split_hash(&spec, "test", &rows), rows))
    };
    let (digest, original) = rows_at(reference)?;
    if row["data_fingerprints"][&anchor] != digest {
        return Err(
            format!("generator does not reproduce the original {anchor} fingerprint").into(),
        );
    }
    let (digest, rows) = if length == reference {
        (digest, original)
    } else {
        rows_at(length)?
    };
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
