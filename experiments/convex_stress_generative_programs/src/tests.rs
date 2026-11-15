use crate::*;
use serde_json::json;
use std::collections::BTreeMap;

#[path = "../../convex_stress_nested_programs/src/tests/gradients.rs"]
mod gradients;

fn row(word: &[i64]) -> data::Example {
    let prompt: Vec<i64> = std::iter::once(6)
        .chain(word.iter().copied())
        .chain([18])
        .collect();
    // Unknown nonidentity decoder, intentionally different from input symbols.
    let answer: Vec<i64> = word
        .iter()
        .map(|&t| match t {
            2 => 4,
            3 => 5,
            4 => 2,
            _ => unreachable!(),
        })
        .chain([17])
        .collect();
    let tokens = prompt
        .iter()
        .chain(answer[..answer.len() - 1].iter())
        .copied()
        .collect();
    let targets: Vec<_> = std::iter::repeat_n(-100, prompt.len() - 1)
        .chain(answer.iter().copied())
        .collect();
    let mut previous = None;
    let changes = targets
        .iter()
        .map(|&t| {
            let changed = t >= 0 && previous != Some(t);
            if t >= 0 {
                previous = Some(t);
            }
            changed
        })
        .collect();
    data::Example {
        task: "fixture-only".into(),
        tokens,
        targets,
        changes,
        prompt,
        limit: answer.len(),
        answer,
        field: None,
        row_index: None,
    }
}
fn training_rows() -> Vec<data::Example> {
    let mut out = vec![];
    for n in 2..=4 {
        for mut v in 0..3usize.pow(n) {
            let word: Vec<_> = (0..n)
                .map(|_| {
                    let t = 2 + (v % 3) as i64;
                    v /= 3;
                    t
                })
                .collect();
            out.push(row(&word));
        }
    }
    out
}
fn fixture() -> Result<Input> {
    dataset::input(20, &training_rows())
}

#[test]
fn independent_rank_interpreter_includes_repetition_missing_and_exhaustion() -> Result<()> {
    let bank = program::catalogue(&[1, 2, 3, 4, 9], 12)?;
    assert_eq!(bank.len(), 5 * 4 * 2 * 4);
    for prefix in [
        vec![1, 4, 2, 4, 3, 9],
        vec![1, 4, 2, 4, 3, 9, 8, 8],
        vec![9],
        vec![1, 2],
        vec![1, 2, 9, 2, 9, 4],
    ] {
        for p in &bank {
            let a = p.run(&prefix);
            assert_eq!(p.interpret(&prefix), (a.state, a.buffer_symbols));
        }
    }
    let make = |order| program::Program {
        delimiter: 9,
        skip: 1,
        latest: false,
        order,
    };
    for (order, expected) in [
        (program::Order::Index, [4, 2, 4, 3]),
        (program::Order::ReverseIndex, [3, 4, 2, 4]),
        (program::Order::AscendingRank, [2, 3, 4, 4]),
        (program::Order::DescendingRank, [4, 4, 3, 2]),
    ] {
        let p = make(order);
        let mut prefix = vec![1, 4, 2, 4, 3, 9];
        for token in expected {
            assert_eq!(p.run(&prefix).state, token);
            prefix.push(8);
        }
        assert_eq!(p.run(&prefix).state, STATES - 2);
        assert_eq!(p.run(&[1, 4]).state, STATES - 1);
    }
    assert!(program::catalogue(&[1, 1], 12).is_err());
    assert!(program::catalogue(&[1, 2], 127).is_err());
    Ok(())
}

#[test]
fn causal_adapter_keeps_original_multiplicities_and_never_reads_later_tokens() -> Result<()> {
    let e = row(&[2, 3, 4]);
    let input = dataset::input(20, &[e.clone(), e.clone()])?;
    assert_eq!(input.observations.len(), 8);
    assert_eq!(&input.observations[..4], &input.observations[4..]);
    for (i, o) in input.observations[..4].iter().enumerate() {
        assert_eq!(
            o.tokens,
            e.tokens[..e.prompt.len() + i]
                .iter()
                .map(|&t| t as usize)
                .collect::<Vec<_>>()
        );
        assert_eq!(o.label, e.answer[i] as usize);
    }
    let mut changed = e.clone();
    changed.task = "unrelated-name".into();
    changed.changes.fill(false);
    assert_eq!(
        dataset::input(20, &[changed.clone()])?,
        dataset::input(20, &[e])?
    );
    changed.targets[0] = 2;
    assert!(dataset::input(20, &[changed]).is_err());
    Ok(())
}

#[test]
fn training_learns_decoder_and_eos_and_supports_unseen_long_prefixes() -> Result<()> {
    let input = fixture()?;
    let found = discovery::run(input.clone())?;
    discovery::check(&input, &found)?;
    assert!(found.error.is_none());
    assert!(!found.retained.is_empty());
    let model = model::Model::new(&found)?;
    model.head.audit(&model.weights)?;
    assert!(
        model
            .head
            .decoders
            .iter()
            .all(|d| d[STATES - 2] == Some(17) && d[2] == Some(4) && d[3] == Some(5))
    );
    let word: Vec<_> = (0..64).map(|i| 2 + (i % 3) as i64).collect();
    let long = row(&word);
    let generated = generative::scoring::rollout(&long.prompt, long.limit, 20, |prefix| {
        Ok(argmax(&metrics::Predictor::logits(&model, prefix)))
    })?;
    assert!(generative::scoring::score(&long.answer, &generated)?.exact);
    let splits: BTreeMap<_, _> = [("test".into(), vec![long])].into();
    let measured = evaluation::evaluate(&model, &input, &splits)?;
    assert_eq!(measured["splits"]["test"]["goals"]["both99"], true);
    assert_eq!(measured["splits"]["test"]["generated_accepted_queries"], 65);
    assert_eq!(measured["target_realizability_certified"], false);
    let mut corrupted = model.clone();
    corrupted.head.coefficients[0][0] += 0.1;
    assert!(corrupted.validate().is_err());
    Ok(())
}

#[test]
fn all_modes_use_original_ce_and_real_tails_with_zero_forward_updates() -> Result<()> {
    let input = fixture()?;
    let model = model::Model::new(&discovery::run(input.clone())?)?;
    for mut p in training::policies()? {
        p.iterations = if p.mode == training::Mode::Forward {
            0
        } else {
            1
        };
        p.batch = 31;
        let out = training::run(&model, &input, p.clone())?;
        training::check(&model, &input, &out)?;
        assert!(out.final_ce < 1e-7);
        if p.mode == training::Mode::Forward {
            assert!(out.steps.is_empty());
            assert_eq!(out.work.gradient_batches, 0);
            assert_eq!(out.work.projected_steps, 0);
            assert_eq!(out.work.cache_products, 0);
            assert_eq!(out.model, model);
        } else if p.mode == training::Mode::Mini {
            assert_eq!(
                out.steps.last().unwrap().count,
                input.observations.len() % 31
            );
            assert_eq!(out.work.gradient_examples, input.observations.len());
        }
    }
    Ok(())
}

#[test]
fn dictionaries_keep_unknown_states_and_conflicts_without_claiming_targets() -> Result<()> {
    let input = fixture()?;
    let found = discovery::run(input.clone())?;
    let model = model::Model::new(&found)?;
    let states: Vec<_> = model.states(&[6, 11, 18]).iter().map(|s| s.state).collect();
    assert!(model.labels(&states).iter().any(Option::is_none));
    // Numerical identity completion is an explicit prior, never acceptance
    // or a learned label. The partial dictionary must still report unknown.
    assert_eq!(argmax(&model.logits_states(&states)), 11);
    let mut contradicted = input;
    let mut duplicate = contradicted.observations[0].clone();
    duplicate.label = 19;
    contradicted.observations.push(duplicate);
    let no = discovery::run(contradicted)?;
    assert!(no.retained.is_empty() && no.error.is_some());
    let mut changed = found.clone();
    changed.retained[0].decoder[2] = Some(19);
    assert!(discovery::check(&found.input, &changed).is_err());
    Ok(())
}

#[test]
fn training_boundary_excludes_dataset_names_and_heldout_content() -> Result<()> {
    let mut rows = training_rows();
    let original = runner::construct(20, &rows, json!({"name":"old","heldout":[1]}))?;
    for e in &mut rows {
        e.task = "renamed".into();
        e.changes.fill(false);
    }
    let changed = runner::construct(20, &rows, json!({"name":"new","heldout":[99,100]}))?;
    assert_eq!(original.input, changed.input);
    assert_eq!(original.model, changed.model);
    assert_eq!(original.error, changed.error);
    Ok(())
}
