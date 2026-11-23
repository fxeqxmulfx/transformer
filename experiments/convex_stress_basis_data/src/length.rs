//! Rebuild frozen block-language and C-RASP length tests, with their original data fingerprints.
use crate::previous as remaining_synthetic;
use serde_json::{Value, json};

pub fn specification(name: &str) -> crate::Result<(Value, usize)> {
    let canonical = match name {
        "basis-depth-easy" => Some("language-l2"),
        "basis-depth-hard" => Some("language-l4"),
        _ => None,
    };
    if let Some(language) = canonical {
        let mut spec = crate::dataset::specification(language)?;
        spec["neutral_fraction"] = json!(0.25);
        return Ok((spec, 128));
    }
    if crate::CASES.contains(&name) {
        let spec = crate::dataset::specification(name)?;
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

fn test_rows(spec: &Value, count: usize) -> crate::Result<Vec<crate::data::Example>> {
    if spec["task"] == "blocks" {
        return crate::generator::generate(spec, "test", count, &Default::default(), false);
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

/// Reproduce the recorded reference split before drawing the requested length.
pub fn verified_rows(
    mut spec: Value,
    reference: usize,
    count: usize,
    expected: &str,
    length: usize,
) -> crate::Result<(String, Vec<crate::data::Example>)> {
    spec["min_length"] = Value::Null;
    let mut rows_at = |size: usize| -> crate::Result<_> {
        spec["length"] = json!(size);
        let rows = test_rows(&spec, count)?;
        Ok((crate::wire::split_hash(&spec, "test", &rows), rows))
    };
    let (digest, original) = rows_at(reference)?;
    if expected != digest {
        return Err(format!(
            "generator does not reproduce the original length-{reference} fingerprint"
        )
        .into());
    }
    let (digest, rows) = if length == reference {
        (digest, original)
    } else {
        rows_at(length)?
    };
    Ok((digest, rows))
}
