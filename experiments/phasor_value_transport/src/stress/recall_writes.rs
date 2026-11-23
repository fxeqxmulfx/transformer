//! The MQAR write-order intervention from Basis.RecallWriteIntervention.
//! Generate each pair from the original test row and validate both orders
//! with the independent source oracle before exposing them to evaluation.
use basis::{Result, data::Example, generator::number, independent, wire};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct Group {
    pub owners: Vec<usize>,
    pub early: Vec<Example>,
    pub late: Vec<Example>,
}

pub(super) struct Panel {
    pub groups: BTreeMap<&'static str, Group>,
    pub spec: Value,
}

fn insert(row: &Example, position: usize, key: i64, value: i64) -> Example {
    let mut tokens = row.tokens.clone();
    let mut targets = row.targets.clone();
    tokens.splice(position..position, [key, value]);
    targets.splice(position..position, [-100, -100]);
    wire::example("mqar", tokens, targets)
}

pub(super) fn panel(name: &str, source: &Value, rows: &[Example]) -> Result<Panel> {
    let pairs = number(source, "pairs")?;
    let end = 1 + 2 * pairs;
    let symbols = number(source, "symbols")? as i64;
    let mut spec = source.clone();
    spec["pairs"] = json!(pairs + 1);
    spec["overwrites"] = json!(source["overwrites"].as_u64().unwrap_or(0) + 1);
    for field in ["length", "min_length"] {
        spec[field] = json!(number(source, field)? + 2);
    }
    let mut groups = BTreeMap::<_, Group>::new();
    for (index, row) in rows.iter().enumerate() {
        independent::check(source, row)?;
        // A different value at either end keeps every adjacent rewrite valid,
        // including when a key already had several values in the source table.
        let mut bounds = BTreeMap::new();
        for &[key, value] in row.tokens[1..end].as_chunks::<2>().0 {
            bounds.entry(key).or_insert((value, value)).1 = value;
        }
        for (key, (first, last)) in bounds {
            let candidates: Vec<_> = (36 + symbols..36 + 2 * symbols)
                .filter(|&v| v != first && v != last)
                .collect();
            if candidates.is_empty() {
                return Err("no valid conflicting value for this source key".into());
            }
            let mut rng =
                wire::addressed_rng(&json!(["phasor-mqar-write-order-v1", name, index, key]));
            let value = candidates[rng.below(candidates.len() as u32) as usize];
            let early = insert(row, 1, key, value);
            let mut late = insert(row, end, key, value);
            let query = row.tokens[end..].iter().position(|&t| t == key);
            if let Some(offset) = query {
                late.targets[end + 2 + offset] = value;
                late = wire::example("mqar", late.tokens, late.targets);
            }
            independent::check(&spec, &early)?;
            independent::check(&spec, &late)?;
            let group = groups
                .entry(if query.is_some() {
                    "queried"
                } else {
                    "unqueried"
                })
                .or_default();
            group.owners.push(index);
            group.early.push(early);
            group.late.push(late);
        }
    }
    Ok(Panel { groups, spec })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stress::recall_gap;

    #[test]
    fn moving_a_write_preserves_the_bag_and_changes_only_its_query() -> Result<()> {
        for name in ["suite-mqar", "basis-recall-easy", "basis-recall-hard"] {
            let spec = recall_gap::source_spec(name)?;
            let rows = basis::generator::generate(&spec, "test", 64, &Default::default(), false)?;
            let panel = panel(name, &spec, &rows)?;
            if name == "suite-mqar" {
                assert_eq!(panel.groups["queried"].owners.len(), 128);
                assert_eq!(panel.groups["unqueried"].owners.len(), 128);
            }
            let end = 1 + 2 * number(&spec, "pairs")?;
            for (kind, group) in panel.groups {
                for ((i, early), late) in group.owners.into_iter().zip(group.early).zip(group.late)
                {
                    let source = &rows[i];
                    let key = early.tokens[1];
                    assert_eq!(&early.tokens[3..], &source.tokens[1..]);
                    assert_eq!(&early.targets[3..], &source.targets[1..]);
                    assert_eq!(&late.tokens[1..end], &source.tokens[1..end]);
                    assert_eq!(&early.tokens[end + 2..], &late.tokens[end + 2..]);
                    assert_eq!(&early.tokens[1..3], &late.tokens[end..end + 2]);
                    let diffs: Vec<_> = early
                        .targets
                        .iter()
                        .zip(&late.targets)
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .collect();
                    assert_eq!(diffs.len(), usize::from(kind == "queried"));
                    for (j, (_, new)) in diffs {
                        assert_eq!(late.tokens[j], key);
                        assert_eq!(*new, early.tokens[2]);
                    }
                }
            }
        }
        Ok(())
    }
}
