//! Answer-only prediction on the original grok modular-division split.
use crate::{
    Result,
    data::{Example, Task},
    division_equations as source,
    suite::Shape,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) const NAME: &str = "grok-division-p97";
pub(crate) const TRAIN_HASH: &str =
    "53e9d2fb9336a74529e88bf582a2fb37483c1f7e79ab8f1fd670de565baf0eef";

pub(crate) fn shape() -> Shape {
    Shape {
        width: 64,
        heads: 4,
        layers: 6,
        vocab: 239,
        max_batch: 512,
        max_length: source::ANSWER_POS + 1,
    }
}

pub(crate) fn example(a: usize, b: usize, label: usize) -> Example {
    let tokens = [
        source::EOS,
        source::NUM0 + a,
        source::DIV,
        source::NUM0 + b,
        source::EQ,
    ];
    let mut targets = vec![-100; tokens.len()];
    targets[source::ANSWER_POS] = (source::NUM0 + label) as i64;
    Example {
        task: NAME.to_owned(),
        tokens: tokens.map(|t| t as i64).to_vec(),
        targets,
        changes: vec![false; tokens.len()],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}

pub(crate) fn load() -> Result<(Value, Task)> {
    let (open, hidden) = source::split(0, 4656);
    let hashes = [source::fingerprint(&open), source::fingerprint(&hidden)];
    if hashes[0] != TRAIN_HASH
        || hashes[1] != "1f679f635bd75862e28f1f6dec8cc121a2963f2fa258a0d8d291df0c8b9acc7d"
    {
        return Err("original division equation fingerprints changed".into());
    }
    let mut splits = BTreeMap::new();
    for (split, equations) in [("train", open), ("hidden", hidden)] {
        splits.insert(
            split.to_owned(),
            equations
                .iter()
                .map(|e| example(e.num as usize, e.den as usize, e.quo as usize))
                .collect(),
        );
    }
    let spec = shape();
    let row = json!({"name":NAME,"tier":"division","split_sha256":hashes,"case":{
        "width":spec.width,"heads":spec.heads,"layers":spec.layers,"batch":spec.max_batch,"observe":256,"max_updates":8192}});
    Ok((
        row,
        Task {
            splits,
            vocab: spec.vocab,
            selection: "hidden".to_owned(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_the_quotient_is_supervised_and_it_is_absent_from_the_prompt() -> Result<()> {
        let (_, task) = load()?;
        for examples in task.splits.values() {
            assert_eq!(examples.len(), 4656);
            for e in examples {
                assert_eq!(e.tokens.len(), 5);
                assert_eq!(e.tokens[0], source::EOS as i64);
                assert_eq!(e.tokens[2], source::DIV as i64);
                assert_eq!(e.tokens[4], source::EQ as i64);
                assert_eq!(&e.targets[..4], &[-100; 4]);
                let a = e.tokens[1] - source::NUM0 as i64;
                let b = e.tokens[3] - source::NUM0 as i64;
                let q = e.targets[4] - source::NUM0 as i64;
                assert!(b > 0 && b < 97);
                assert_eq!(b * q % 97, a);
            }
        }
        Ok(())
    }
}
