use crate::*;
use program::{Emit, Program, Side};

fn specimen(emit: Emit, hints: bool, bits: &[bool], start: usize) -> data::Example {
    let mut prompt = vec![1];
    for (i, &b) in bits.iter().enumerate() {
        if hints {
            prompt.push((start + i) as i64);
        }
        prompt.push(if b { 7 } else { 5 });
    }
    prompt.push(18);
    let labels = [55, 57, 56];
    let mut answer = if emit == Emit::Final {
        vec![]
    } else {
        vec![labels[0]]
    };
    for i in 0..bits.len() {
        if emit == Emit::Final || emit == Emit::Selected && !bits[i] {
            continue;
        }
        if hints {
            answer.push((start + i) as i64);
        }
        let count = bits[..=i].iter().filter(|&&b| b).count();
        answer.push(labels[count % 3]);
    }
    if emit == Emit::Final {
        answer.push(labels[bits.iter().filter(|&&b| b).count() % 3]);
    }
    answer.push(17);
    let tokens = prompt
        .iter()
        .chain(answer[..answer.len() - 1].iter())
        .copied()
        .collect();
    let targets = std::iter::repeat_n(-100, prompt.len() - 1)
        .chain(answer.iter().copied())
        .collect();
    let mut changes = vec![false; prompt.len() - 1];
    changes.extend(
        answer
            .iter()
            .enumerate()
            .map(|(i, &y)| i == 0 || answer[i - 1] != y),
    );
    let limit = if emit == Emit::Final {
        2
    } else {
        (if hints { 2 } else { 1 }) * bits.len() + 2
    };
    data::Example {
        task: "synthetic-count-three".into(),
        tokens,
        targets,
        changes,
        prompt,
        answer,
        limit,
        field: None,
        row_index: None,
    }
}
fn fixtures(emit: Emit, hints: bool) -> Vec<data::Example> {
    (0usize..48)
        .map(|mask| {
            let n = 4 + mask % 4;
            let bits: Vec<_> = (0..n).map(|i| mask & (1 << i) != 0).collect();
            specimen(emit, hints, &bits, 80 + mask % 4)
        })
        .collect()
}

#[test]
fn two_interpreters_agree_on_every_candidate_and_malformed_prefix() -> Result<()> {
    let rows = fixtures(Emit::Selected, true);
    let input = dataset::input(180, &rows)?;
    let roles = program::infer(&input)?;
    assert_eq!(roles.delimiters, vec![18]);
    let mut altered = input.clone();
    for row in &mut altered.observations {
        row.label = (row.label + 29) % 180;
    }
    assert_eq!(roles, program::infer(&altered)?);
    let bank = program::catalogue(&roles)?;
    for p in &bank {
        for tokens in [
            &rows[19].prompt[..],
            &rows[19].tokens[..],
            &[1, 7, 5][..],
            &[9, 9, 7][..],
        ] {
            let tokens: Vec<_> = tokens.iter().map(|&t| t as usize).collect();
            let a = p.run(&tokens);
            assert_eq!((a.state, a.buffer_symbols), p.interpret(&tokens));
            assert!(a.state < STATES);
        }
    }
    let p = Program {
        delimiter: 9,
        counter: 7,
        modulus: 3,
        skip: 1,
        stride: 2,
        offset: 1,
        latest: false,
        emit: Emit::Selected,
        copy: Some(Side::Before),
    };
    assert_eq!(p.run(&[1, 80, 5, 81, 7, 82, 7, 9, 55]).state, 81);
    assert_eq!(p.run(&[1, 80, 5, 81, 7, 82, 7, 9, 55, 81]).state, 241);
    Ok(())
}

#[test]
fn training_only_discovery_learns_all_formats_and_categorical_decoders() -> Result<()> {
    for (emit, hints) in [
        (Emit::Final, false),
        (Emit::All, false),
        (Emit::All, true),
        (Emit::Selected, true),
    ] {
        let rows = fixtures(emit, hints);
        let input = dataset::input(180, &rows)?;
        let report = discovery::run(input.clone())?;
        discovery::check(&input, &report)?;
        assert!(
            report.error.is_none(),
            "{emit:?}/{hints}: {:?}",
            report.error
        );
        assert!(!report.retained.is_empty());
        let model = model::Model::new(&report)?;
        assert_eq!(model.programs, vec![report.retained[0].program.clone()]);
        assert_eq!(model.programs[0].modulus, 3);
        assert_eq!(model.programs[0].counter, 7);
        let query = specimen(
            emit,
            hints,
            &[
                true, false, true, true, false, false, true, true, true, true, false,
            ],
            120,
        );
        let predicted = generative::scoring::rollout(&query.prompt, query.limit, 180, |prefix| {
            let state = model.programs[0]
                .run(&prefix.iter().map(|&t| t as usize).collect::<Vec<_>>())
                .state;
            Ok(argmax(&model.logits_states(&[state])))
        })?;
        assert!(generative::scoring::score(&query.answer, &predicted)?.exact);
        if hints {
            assert!(model.head.decoders[0][120..131].iter().all(Option::is_none));
        }
        let mut renamed = input.clone();
        for row in &mut renamed.observations {
            row.label = match row.label {
                55 => 60,
                57 => 59,
                56 => 58,
                x => x,
            };
        }
        let renamed_report = discovery::run(renamed)?;
        assert_eq!(
            report.attempts,
            renamed_report
                .attempts
                .iter()
                .map(|a| {
                    let mut a = a.clone();
                    if let Some((i, j, s, x, y)) = a.conflict {
                        let restore = |v| match v {
                            60 => 55,
                            59 => 57,
                            58 => 56,
                            t => t,
                        };
                        a.conflict = Some((i, j, s, restore(x), restore(y)));
                    }
                    a
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            report.retained[0].program,
            renamed_report.retained[0].program
        );
    }
    Ok(())
}

#[test]
fn complete_complex_head_and_real_batch_tail_are_audited() -> Result<()> {
    let rows = fixtures(Emit::Final, false);
    let input = dataset::input(180, &rows)?;
    let report = discovery::run(input.clone())?;
    let initial = model::Model::new(&report)?;
    let audit = initial.head.audit(&initial.weights)?;
    assert_eq!(audit.real_head_coordinates, 2 * STATES * 180);
    assert!(audit.max_inversion_error < 1e-8 && audit.max_column_l1 <= 16384.);
    for mode in [
        training::Mode::Full,
        training::Mode::Mini,
        training::Mode::Forward,
    ] {
        let p = training::Policy {
            mode,
            iterations: usize::from(mode != training::Mode::Forward),
            batch: 7,
            learning_rate: 0.1,
        };
        let out = training::run(&initial, &input, p)?;
        training::check(&initial, &input, &out)?;
        assert!(out.final_ce < 1e-6);
        if mode == training::Mode::Mini {
            assert_eq!(
                out.steps.last().unwrap().count,
                input.observations.len() % 7
            );
        }
        if mode == training::Mode::Forward {
            assert_eq!(out.work.gradient_batches, 0);
            assert_eq!(out.work.projected_steps, 0);
        }
    }
    let mut corrupt = initial;
    corrupt.head.coefficients[0][0] += 0.25;
    assert!(corrupt.validate().is_err());
    Ok(())
}

#[test]
fn count_bounds_and_missing_roles_refuse_before_construction() -> Result<()> {
    let rows = fixtures(Emit::All, true);
    let mut input = dataset::input(180, &rows)?;
    input.vocabulary = 241;
    assert!(program::infer(&input).is_err());
    input.vocabulary = 180;
    input.observations.clear();
    assert!(program::infer(&input).is_err());
    let mut roles = program::infer(&dataset::input(180, &rows)?)?;
    roles.counters = (0..9).collect();
    assert!(program::catalogue(&roles).is_err());
    assert!(evaluation::forecast(1, 241, &Default::default()).is_err());
    Ok(())
}

#[test]
fn complete_binding_and_generation_use_the_same_model_on_new_lengths() -> Result<()> {
    let rows = fixtures(Emit::Final, false);
    let input = dataset::input(100, &rows)?;
    let bound = binding::run(input.clone())?;
    binding::check(&input, &bound)?;
    let probes = vec![
        specimen(Emit::Final, false, &[true; 15], 0),
        specimen(Emit::Final, false, &[true; 14], 0),
    ];
    let data = dataset::Dataset {
        name: "generic-counter-fixture".into(),
        task: data::Task {
            vocab: 100,
            selection: "validation".into(),
            splits: [("train".into(), rows), ("validation".into(), probes)]
                .into_iter()
                .collect(),
        },
        original_fingerprints: Default::default(),
        tensor_fingerprints: Default::default(),
        empty_novel_splits: vec![],
    };
    let stage = runner::Stage {
        dataset: serde_json::json!({"fixture":true}),
        input,
        model: Some(bound),
        error: None,
    };
    let measured = runner::measurements(&stage, &data)?;
    let modes = measured["modes"].as_array().unwrap();
    assert_eq!(modes.len(), 3);
    for mode in modes {
        assert!(mode["error"].is_null(), "{}", mode["error"]);
        assert_eq!(
            mode["primary"]["both99"],
            true,
            "{}",
            serde_json::to_string(mode)?
        );
    }
    Ok(())
}

#[test]
fn prospective_original_shapes_fit_the_declared_search_and_scoring_caps() -> Result<()> {
    for case in generative::CASES {
        let data = dataset::build(case)?;
        let input = dataset::input(data.task.vocab, &data.task.splits["train"])?;
        let roles = program::infer(&input)?;
        let candidates = program::catalogue(&roles)?.len();
        let terms: u64 = input
            .observations
            .iter()
            .map(|s| 64 * s.tokens.len() as u64 + 256)
            .sum();
        let search = roles.scalar_work_forecast + candidates as u64 * terms;
        let score = evaluation::forecast(1, data.task.vocab, &data.task.splits)?;
        let training = training::policies()?
            .iter()
            .map(|p| p.forecast(1, data.task.vocab, input.observations.len()))
            .collect::<Result<Vec<_>>>()?;
        assert!(search <= WORK_LIMIT, "{case}: search {search}");
        assert!(
            score.scalar_work_forecast <= evaluation::SCORING_WORK_LIMIT,
            "{case}: score {}",
            score.scalar_work_forecast
        );
        eprintln!(
            "{case}: records={} training={training:?}",
            input.observations.len()
        );
        eprintln!(
            "{case}: candidates={candidates} search={search} score={} calls={}",
            score.scalar_work_forecast, score.maximum_prediction_calls
        );
    }
    Ok(())
}
