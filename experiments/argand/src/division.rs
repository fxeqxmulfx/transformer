//! Answer-only prediction on the original grok modular-division split.
use crate::{
    Result,
    data::{Example, Task},
    division_equations as source,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) const NAME: &str = "grok-division-p97";

pub(crate) fn load() -> Result<(Value, Task)> {
    let (open, hidden) = source::split(0, 4656);
    let hashes = [source::fingerprint(&open), source::fingerprint(&hidden)];
    if hashes[0] != "53e9d2fb9336a74529e88bf582a2fb37483c1f7e79ab8f1fd670de565baf0eef"
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
                .map(|e| {
                    let pos = source::ANSWER_POS;
                    let tokens = e.tokens()[..=pos].iter().map(|&t| t as i64).collect();
                    let mut targets = vec![-100; pos + 1];
                    targets[pos] = e.target(pos) as i64;
                    Example {
                        task: NAME.to_owned(),
                        tokens,
                        targets,
                        changes: vec![false; pos + 1],
                        prompt: vec![],
                        answer: vec![],
                        limit: 0,
                        field: None,
                        row_index: None,
                    }
                })
                .collect(),
        );
    }
    let row = json!({"name":NAME,"tier":"division","split_sha256":hashes,"case":{
        "width":64,"heads":4,"layers":6,"batch":512,"observe":256,"max_updates":8192}});
    Ok((
        row,
        Task {
            splits,
            vocab: 239,
            selection: "hidden".to_owned(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use convex_relabel_graph_audit::graph::{Pair, analyze};

    /// The ideal character learner uses an explicit prior on the output
    /// codes. The neural trainer has no access to this test or its factors.
    #[test]
    fn original_mask_identifies_unit_characters_and_the_zero_row() -> Result<()> {
        let (open, hidden) = source::split(0, 4656);
        let units: Vec<_> = open
            .iter()
            .filter(|e| e.num > 0)
            .map(|e| Pair {
                row: e.num as usize - 1,
                column: e.den as usize - 1,
                label: e.quo as usize - 1,
            })
            .collect();
        let mut coding = vec![usize::MAX; 96];
        let mut residue = 1;
        for power in 0..96 {
            coding[residue - 1] = power;
            residue = 5 * residue % 97;
        }
        let graph = analyze(96, &units, &coding).map_err(|error| error.to_string())?;
        assert_eq!(graph.observed_cells, 4611);
        assert!(graph.connected && graph.observed_phase_consistent);
        assert!(graph.perfect_matching.is_some());
        let predictions = graph.completion_labels.ok_or("no unit completion")?;
        let mut zero_examples = open.iter().filter(|e| e.num == 0);
        let zero = zero_examples.next().ok_or("zero row was unobserved")?.quo;
        assert!(zero_examples.all(|e| e.quo == zero));
        for e in hidden {
            let pred = if e.num == 0 {
                zero
            } else {
                (1 + predictions[e.num as usize - 1 + 96 * (e.den as usize - 1)]) as u8
            };
            assert_eq!(pred, e.quo);
        }
        Ok(())
    }
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
