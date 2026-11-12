use super::*;
use crate::{
    cohort::model::Model,
    features::{Execution, Mask, Transport},
};

#[test]
fn shared_model_training_loop_keeps_dense_and_streamed_updates_identical() -> Result<()> {
    let bank = Bank {
        vocab: 3,
        depth: 2,
        history_slots: 16,
        masks: vec![Mask::All],
        transport: Transport::Fourier,
        execution: Execution::Cumulative,
    };
    let examples: Vec<_> = [[0, 1, 2], [2, 0, 1], [0, 1, 2]]
        .into_iter()
        .map(|tokens| data::Example {
            task: "test".into(),
            tokens: tokens.to_vec(),
            targets: vec![2, -100, 0],
            changes: vec![false; 3],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None,
        })
        .collect();
    let path = std::env::temp_dir().join(format!(
        "retained-loop-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let mut dense = Observations::build(&bank, &examples, None)?;
    let mut stream = Observations::build(&bank, &examples, Some(&path))?;
    let mut a = Head::zero(bank.clone(), 96.)?;
    let mut b = Head::zero(bank, 96.)?;
    let mut ga = vec![0.; a.coefficients.len()];
    let mut gb = ga.clone();
    assert_eq!(
        dense.initialize(&mut a, &mut ga, ForwardChoice::MinimumCe)?,
        stream.initialize(&mut b, &mut gb, ForwardChoice::MinimumCe)?
    );
    let (mut a, mut b) = (Model::Tensor(a), Model::Tensor(b));
    for batch in [vec![0, 1, 2], vec![2, 0], vec![1], vec![2, 2, 0]] {
        assert_eq!(dense.prefixes(&batch)?, stream.prefixes(&batch)?);
        assert_eq!(
            a.objective(&mut dense, &batch)?,
            b.objective(&mut stream, &batch)?
        );
        assert_eq!(
            a.update(&mut dense, &batch, &mut ga)?,
            b.update(&mut stream, &batch, &mut gb)?
        );
        assert_eq!(ga, gb);
        let (Model::Tensor(x), Model::Tensor(y)) = (&a, &b) else {
            unreachable!()
        };
        assert_eq!(x.coefficients, y.coefficients);
    }
    drop(stream);
    std::fs::remove_file(path)?;
    Ok(())
}
