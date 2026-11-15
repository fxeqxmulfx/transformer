//! Hand-written fixtures independent of the benchmark target generators.
use crate::{
    joint::training::{Mode, Policy},
    model::Prior,
    observations::Observation,
    prefix::{Formula, Term},
    program::{Atom, Program},
    *,
};

fn words(length: usize) -> Vec<Vec<usize>> {
    (0..3usize.pow(length as u32))
        .map(|mut n| {
            (0..length)
                .map(|_| {
                    let c = [9, 10, 23][n % 3];
                    n /= 3;
                    c
                })
                .collect()
        })
        .collect()
}

#[test]
fn full_bank_matches_separate_interpreter_on_valid_and_refused_inputs() {
    let prompts = [
        vec![1, 100, 100, 18],
        vec![1, 105, 101, 18],
        vec![1, 101, 105, 18],
        vec![1, 100, 18],
        vec![1, 104, 18],
        vec![1, 26, 27, 28, 20, 26, 26, 35, 18],
        vec![1, 26, 26, 26, 20, 26, 26, 26, 18],
        vec![
            1, 110, 26, 111, 27, 112, 28, 20, 110, 26, 111, 26, 112, 35, 18,
        ],
        vec![1, 110, 26, 111, 27, 20, 110, 26, 112, 27, 18],
        vec![1, 35, 35, 20, 35, 35, 18],
        vec![1, 999, 20, 26, 18],
        vec![1, 26, 20, 26],
        vec![18],
        vec![],
    ];
    for p in program::numeric_bank() {
        for prompt in &prompts {
            let mut x = prompt.clone();
            for rank in 0..12 {
                assert_eq!(p.run(&x), p.interpret(&x), "{p:?} on {x:?}");
                x.push(998 - rank);
            }
        }
    }
    let more_b = Formula::Less(
        Box::new(Term::Count(Box::new(Formula::Symbol(9)))),
        Box::new(Term::Count(Box::new(Formula::Symbol(10)))),
    );
    let nested = Program::Prefix(Formula::And(
        Box::new(Formula::Not(Box::new(Formula::Symbol(23)))),
        Box::new(Formula::Less(
            Box::new(Term::Negate(Box::new(Term::Count(Box::new(more_b))))),
            Box::new(Term::Add(
                Box::new(Term::Constant(-2)),
                Box::new(Term::Constant(1)),
            )),
        )),
    ));
    for n in 1..=6 {
        for x in words(n) {
            assert_eq!(nested.run(&x), nested.interpret(&x));
            let with_bos: Vec<_> = std::iter::once(1).chain(x.iter().copied()).collect();
            assert_eq!(nested.run(&with_bos), nested.run(&x));
            assert_eq!(nested.interpret(&with_bos), nested.interpret(&x));
        }
    }
}

fn sparse_rows() -> Vec<Observation> {
    [
        (20, vec![1, 100, 102, 18], 100),
        (71, vec![1, 100, 102, 18, 99, 99], 102),
        (96, vec![1, 100, 102, 18], 100),
    ]
    .into_iter()
    .map(|(id, tokens, label)| Observation { id, tokens, label })
    .collect()
}
fn range() -> Program {
    Program::Range {
        reverse: false,
        inclusive: true,
    }
}
fn policy(mode: Mode) -> Policy {
    Policy {
        mode,
        epochs: usize::from(mode != Mode::Forward),
        batch: 2,
        learning_rate: 1e5,
        initial_total_mass: 10000.,
        projection_cycles: 20000,
        projection_tolerance: 1e-10,
    }
}

#[test]
fn observed_joint_decoder_identifies_an_unseen_class_but_not_a_missing_role() -> Result<()> {
    let rows = sparse_rows();
    let d = discovery::run(&rows, observations::coordinates(229)?, vec![range()])?;
    assert_eq!(d.compatible.len(), 1);
    assert!(d.compatible[0].maximum_anchor_residual < 1e-12);
    let x = [1, 100, 102, 18, 99];
    let missing_role = [1, 100, 102, 18, 99, 99, 99];
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let trained = model::train(&d, &rows, Prior::Coordinates, policy(mode), 1e8, 1.)?;
        let q = trained.model.query(&x)?;
        assert_eq!(joint::argmax(&q.logits), 101);
        assert!(q.active_observable_coverage);
        assert!(!q.target_realizability_certified);
        assert!(
            !trained
                .model
                .query(&missing_role)?
                .active_observable_coverage
        );
        let independent = trained.model.independent_logits(&x)?;
        assert!(
            q.logits
                .iter()
                .zip(&independent)
                .all(|(a, b)| (a - b).abs() < 1e-9)
        );
        assert!(trained.model.head.max_column_charge() <= 1e8 + 1e-7);
        trained.replay(&d, &rows)?;
        let encoded = serde_json::to_vec(&trained)?;
        assert_eq!(trained, serde_json::from_slice::<model::Trained>(&encoded)?);
        match mode {
            Mode::Forward => assert!(trained.training.steps.is_empty()),
            Mode::Full => assert_eq!(trained.training.steps[0].record_ids, vec![20, 71, 96]),
            Mode::Mini => {
                assert_eq!(trained.training.steps[0].record_ids, vec![20, 71]);
                assert_eq!(trained.training.steps[1].record_ids, vec![96]);
            }
        }
        if mode != Mode::Forward {
            assert!(trained.training.work.effective_updates > 0);
            assert!(trained.training.final_ce < trained.training.initial_ce);
        }
        assert!(trained.model.query(&[229]).is_err());
        let mut damaged = trained.clone();
        damaged.training.initial_ce += 1.;
        assert!(damaged.replay(&d, &rows).is_err());
        let control = model::train(&d, &rows, Prior::Categorical, policy(mode), 1e8, 1.)?;
        let prediction = control.model.query(&x)?;
        assert_eq!(joint::argmax(&prediction.logits), 0);
        assert!(!prediction.defined[0]);
        assert!(!prediction.active_observable_coverage);
    }
    Ok(())
}

#[test]
fn incompatible_labels_and_resource_limits_fail_explicitly() -> Result<()> {
    let mut rows = sparse_rows();
    rows[2].label = 17;
    let d = discovery::run(&rows, observations::coordinates(229)?, vec![range()])?;
    assert!(d.compatible.is_empty());
    assert!(
        d.attempts[0]
            .error
            .as_ref()
            .unwrap()
            .contains("zero program mass")
    );
    assert!(
        model::train(
            &d,
            &rows,
            Prior::Coordinates,
            policy(Mode::Forward),
            1e8,
            1.
        )
        .is_err()
    );
    rows[2].id = rows[0].id;
    assert!(observations::validate(&rows, 229).is_err());
    assert!(
        Atom::Quantity(i64::MIN)
            .features(&observations::coordinates(229)?)
            .is_err()
    );
    let bank = vec![range(); 8193];
    assert!(discovery::run(&sparse_rows(), observations::coordinates(229)?, bank).is_err());
    Ok(())
}

#[test]
fn bounded_prefix_search_preserves_multiple_fitting_syntaxes_and_raw_label_transport() -> Result<()>
{
    // 117 records exercise a partial bitset word and repeated original inputs.
    let rows: Vec<_> = (2..=4)
        .flat_map(words)
        .enumerate()
        .map(|(i, x)| {
            let a = x.iter().filter(|&&t| t == 9).count();
            let b = x.iter().filter(|&&t| t == 10).count();
            Observation {
                id: 3 * i + 5,
                tokens: x,
                label: if a < b { 16 } else { 15 },
            }
        })
        .collect();
    assert_eq!(rows.len(), 117);
    let p = search::Policy {
        depth: 1,
        beam: 16,
        retained: 8,
        max_candidates: 20000,
    };
    let found = search::run(&rows, 36, p.clone())?;
    assert_eq!(found.stages.len(), 1);
    assert_eq!(found.stages[0].visited, 10080);
    assert!(found.programs.len() > 1);
    let d = discovery::run(
        &rows,
        observations::coordinates(36)?,
        found.programs.clone(),
    )?;
    assert_eq!(d.compatible.len(), found.programs.len());
    for program in &found.programs {
        for r in &rows {
            assert_eq!(program.run(&r.tokens), program.interpret(&r.tokens));
        }
    }
    let swapped: Vec<_> = rows
        .iter()
        .map(|r| Observation {
            label: 31 - r.label,
            ..r.clone()
        })
        .collect();
    let transported = search::run(&swapped, 36, p.clone())?;
    assert_eq!(found, transported);
    let with_bos: Vec<_> = rows
        .iter()
        .map(|r| Observation {
            tokens: std::iter::once(1).chain(r.tokens.iter().copied()).collect(),
            ..r.clone()
        })
        .collect();
    let bos_search = search::run(&with_bos, 36, p.clone())?;
    assert_eq!(found.programs, bos_search.programs);
    assert_eq!(found.stages, bos_search.stages);
    let bos_discovery = discovery::run(
        &with_bos,
        observations::coordinates(36)?,
        bos_search.programs,
    )?;
    assert_eq!(bos_discovery.compatible, d.compatible);
    let cap = search::Policy {
        max_candidates: 1,
        ..p
    };
    assert!(search::run(&rows, 36, cap).is_err());
    let mut skipped = rows.clone();
    skipped[0].tokens[0] = 18;
    assert!(
        search::run(&skipped, 36, search::Policy::default())?
            .programs
            .is_empty()
    );
    Ok(())
}
