//! Bound the exact unchanged task343 JSON schema, rather than its older generic
//! 64-byte-per-value/16KiB-header allowance. No compression or evidence removal.
use crate::*;
use std::collections::BTreeMap;

pub(crate) fn query_line(
    vocabulary: usize,
    programs: usize,
    prefix: usize,
    split: &str,
) -> Result<u128> {
    // Tokens/predictions are in0..1024: at most4 digits plus a comma.
    // Pinned serde_json1.0.154 delegates finite floats to zmij1.0.23's24-byte
    // buffer; allow32 bytes including separators. Each
    // program contributes one Boolean and one span residual (40 bytes together).
    // The remaining keys, scalar fields, punctuation and newline fit in1024.
    Ok(1024
        + serde_json::to_string(split)?.len() as u128
        + 5 * prefix as u128
        + 32 * vocabulary as u128
        + 40 * programs as u128)
}
pub fn endpoint(
    vocabulary: usize,
    programs: usize,
    splits: &BTreeMap<String, Vec<dataset::data::Example>>,
) -> Result<u64> {
    evaluation::forecast(vocabulary, programs, splits)?;
    let mut bytes = 0u128;
    for (name, rows) in splits {
        for e in rows {
            let h = e.tokens.len().max(e.prompt.len() + e.limit);
            let calls = e.targets.iter().filter(|&&y| y >= 0).count() + e.limit;
            bytes += calls as u128 * query_line(vocabulary, programs, h, name)?;
            // One result per original row: up to three token arrays of length h,
            // all scalar score fields, escaped split name and punctuation.
            bytes += 1024 + serde_json::to_string(name)?.len() as u128 + 15 * h as u128;
        }
    }
    Ok(u64::try_from(bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn independent_json_serialization_fits_extreme_finite_value_allowances() {
        let values = [
            f64::MAX,
            -f64::MAX,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(1),
            -0.,
            1.2345678901234567e-120,
        ];
        for value in values {
            assert!(serde_json::to_string(&value).unwrap().len() < 32);
        }
        let split = "escaped\"\\\nα";
        let query = model::Query {
            logits: (0..1024).map(|i| values[i % values.len()]).collect(),
            defined: vec![false; 64],
            span_residuals: (0..64).map(|i| values[i % values.len()]).collect(),
            active_observable_coverage: false,
            target_realizability_certified: false,
        };
        let line = json!({"split":split,"row":19999,"phase":"teacher","position":4095,
            "tokens":vec![1023;4096],"target":1023,"predicted":1023,"ce":f64::MAX,
            "query":query,"independent_residual":-f64::MAX});
        assert!(
            (serde_json::to_vec(&line).unwrap().len() as u128)
                < query_line(1024, 64, 4096, split).unwrap()
        );
    }
}
