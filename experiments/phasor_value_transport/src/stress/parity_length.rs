//! Fixed-length generalization of the catalog's scratchpad-free parity test.
//! Transformer.Basis.parityFunction_prompt and parityFunction_answer give
//! the required label and EOS on every nonempty bit word.
use basis::{Result, data::Example};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub fn rows(length: usize) -> Result<(String, Vec<Example>)> {
    let protocol: Value = serde_json::from_str(parity::PROTOCOL)?;
    let mut spec = protocol["definitions"]["suite-parity-none"].clone();
    spec.as_object_mut()
        .and_then(|s| s.remove("longer_lengths"))
        .filter(|lengths| lengths == &json!([32, 64]))
        .ok_or("unexpected source parity probe lengths")?;
    spec["min_length"] = Value::Null;
    spec["length"] = json!(length);
    spec["number_limit"] = json!(length);
    let rows = parity::dataset::generate(&spec, "test", 256)?;
    for row in &rows {
        basis::independent::check(&spec, row)?;
    }
    let distinct: BTreeSet<_> = rows.iter().map(|r| &r.prompt).collect();
    if distinct.len() != rows.len() {
        return Err("duplicate extended parity test prompt".into());
    }
    Ok((basis::wire::split_hash(&spec, "test", &rows), rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_previous_phasor_32_bit_test() -> Result<()> {
        let (digest, examples) = rows(32)?;
        assert_eq!(
            digest,
            "3da61f3d536035881e62a31013fb914cb95dfcf47a6218be45b0f61ab0e94848"
        );
        assert_eq!(examples.len(), 256);
        assert!(examples.iter().all(|row| row.prompt.len() == 34 && row.limit == 2));
        assert!(rows(0).is_err());
        Ok(())
    }
}
