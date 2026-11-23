//! Paired relative-delay probes from the frozen GPT and Basis MQAR test inputs.
//! Basis.recallAnswer_insert_gap proves that the added tokens preserve every answer.
use basis::{Result, data::Example, generator::number, independent, wire};
use serde_json::{Value, json};

pub fn source_spec(name: &str) -> Result<Value> {
    if name == "suite-mqar" {
        let catalog: Value = serde_json::from_str(basis::sequences::CATALOG)?;
        let mut spec = catalog["first_cohort_definitions"][name].clone();
        let fields = spec
            .as_object_mut()
            .ok_or("missing original MQAR specification")?;
        fields.remove("longer_lengths");
        if fields.remove("overwrites") != Some(json!(0)) {
            return Err("the source MQAR generator expects zero overwrites".into());
        }
        return Ok(spec);
    }
    let mut spec = basis::dataset::specification("basis-recall-hard")?;
    match name {
        "basis-recall-hard" => {}
        "basis-recall-easy" => {
            spec["pairs"] = json!(8);
            spec["min_length"] = json!(32);
            spec.as_object_mut()
                .ok_or("invalid MQAR specification")?
                .remove("overwrites");
        }
        _ => return Err("recall-gap requires an MQAR checkpoint".into()),
    }
    Ok(spec)
}

pub fn delay(name: &str, rows: &[Example], gap: usize) -> Result<(String, Vec<Example>)> {
    let source = source_spec(name)?;
    let maximum = number(&source, "length")?
        .checked_add(gap)
        .ok_or("recall length overflow")?;
    if rows.is_empty() || maximum > 8192 || rows.len() > 4_194_304 / maximum {
        return Err("recall gap exceeds the source split dimensions".into());
    }
    let table_end = 1 + 2 * number(&source, "pairs")?;
    let symbols = number(&source, "symbols")? as u32;
    let mut extended = source.clone();
    extended["length"] = json!(maximum);
    extended["min_length"] = json!(number(&source, "min_length")? + gap);
    extended["query_gap"] = json!(number(&source, "query_gap")? + gap);
    let mut delayed = Vec::with_capacity(rows.len());
    for (i, original) in rows.iter().enumerate() {
        independent::check(&source, original)?;
        // The generator identity omits the requested gap. Longer probes add to
        // the exact same per-row filler stream, preserving all sampled queries.
        let mut rng = wire::addressed_rng(&json!(["phasor-recall-gap-v1", name, "test", i]));
        let mut tokens = original.tokens[..table_end].to_vec();
        tokens.extend((0..gap).map(|_| i64::from(36 + symbols + rng.below(symbols))));
        tokens.extend_from_slice(&original.tokens[table_end..]);
        let mut targets = original.targets[..table_end].to_vec();
        targets.resize(table_end + gap, -100);
        targets.extend_from_slice(&original.targets[table_end..]);
        let row = wire::example("mqar", tokens, targets);
        independent::check(&extended, &row)?;
        delayed.push(row);
    }
    Ok((wire::split_hash(&extended, "test", &delayed), delayed))
}
