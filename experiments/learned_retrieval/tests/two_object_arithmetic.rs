//! CM112 on the original arithmetic tensors. Each label is supervised at the
//! second operand. Group sizes bound every deterministic first-operand readout.
use learned_retrieval::{Result, cohort};
use serde_json::Value;

#[test]
fn original_modular_addition_exceeds_the_two_object_head() -> Result<()> {
    let suite: Value = serde_json::from_str(cohort::SUITE)?;
    let source = suite["tasks"]
        .as_array()
        .ok_or("original cases")?
        .iter()
        .find(|row| row["name"] == "arithmetic-add")
        .ok_or("original addition case")?;
    let task = baseline::data::build(source)?;
    assert_eq!(task.selection, "validation");
    assert_eq!(task.vocab, 97);
    let model = cohort::protocol()?.initial(task.vocab, 1)?;
    let by_first: Vec<_> = (0..task.vocab)
        .map(|a| model.logits(&[a as i64, 0]))
        .collect::<Result<_>>()?;
    assert_ne!(by_first[0], by_first[1]);
    assert_eq!(task.splits.len(), 2);
    for (name, expected) in [("train", 4704), ("validation", 4705)] {
        let rows = &task.splits[name];
        assert_eq!(rows.len(), expected);
        let mut counts = vec![vec![0usize; task.vocab]; task.vocab];
        for row in rows {
            let [first, second] = row.tokens.as_slice() else {
                return Err("expected two opaque operands".into());
            };
            let [-100, y] = row.targets.as_slice() else {
                return Err("expected one masked label".into());
            };
            let a = usize::try_from(*first)?;
            let b = usize::try_from(*second)?;
            let answer = usize::try_from(*y)?;
            assert_eq!(answer, (a + b) % task.vocab);
            assert_eq!(model.logits(&row.tokens)?, by_first[a]);
            counts[a][answer] += 1;
        }
        let best: usize = counts
            .iter()
            .map(|group| group.iter().max().copied().unwrap_or(0))
            .sum();
        assert_eq!(best, 97);
        println!(
            "{name}: any fixed prediction per first operand is correct on at most {best}/{expected}"
        );
    }
    Ok(())
}
