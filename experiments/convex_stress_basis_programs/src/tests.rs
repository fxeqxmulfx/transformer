use crate::*;
use model::Prior;
use program::Program;

#[test]
fn independent_grammar_interpreters_cover_rewrites_run_changes_and_generated_positions() {
    let programs = [
        Program::Walk {
            hops: 1,
            offset: 1,
            latest: true,
        },
        Program::Walk {
            hops: 1,
            offset: 1,
            latest: false,
        },
        Program::RunsEq {
            first: 9,
            second: 10,
            runs: 2,
        },
        Program::Residue {
            symbol: 22,
            modulus: 2,
        },
        Program::Lag(0),
    ];
    let words = [
        vec![1, 36, 292, 37, 293, 36, 294, 36],
        vec![1, 9, 9, 11, 10, 10, 9],
        vec![1, 21, 22, 22, 18],
        vec![1, 21, 22, 22, 18, 24],
        vec![1, 18, 18],
        vec![1, 18, 24, 25],
    ];
    for p in &programs {
        for word in &words {
            for n in 1..=word.len() {
                assert_eq!(p.run(&word[..n]), p.interpret(&word[..n]));
            }
        }
    }
    assert_eq!(programs[0].run(&words[0]), Some(Atom::Copy(294)));
    assert_eq!(programs[1].run(&words[0]), Some(Atom::Copy(292)));
    assert_eq!(programs[2].run(&words[1][..6]), Some(Atom::Quantity(1)));
    assert_eq!(programs[2].run(&words[1]), Some(Atom::Quantity(0)));
    assert_eq!(programs[3].run(&words[2]), Some(Atom::Quantity(0)));
    assert_eq!(programs[3].run(&words[3]), Some(Atom::Role(0)));
    assert!(
        Program::Walk {
            hops: usize::MAX,
            offset: usize::MAX,
            latest: true
        }
        .run(&words[0])
        .is_none()
    );
}

#[test]
fn causal_borrowed_records_preserve_masks_ids_and_a_nonconstant_language_fit() -> Result<()> {
    let rows = vec![
        basis::wire::example(
            "blocks",
            vec![1, 9, 9, 10, 10, 9],
            vec![-100, 15, 15, 16, 16, 15],
        ),
        basis::wire::example("blocks", vec![1, 10, 10, 9, 10], vec![-100, 15, 15, 15, 15]),
        basis::wire::example("blocks", vec![1, 9, 10, 9, 10], vec![-100, 15, 16, 15, 15]),
    ];
    let report = discovery::run(36, &rows)?;
    assert_eq!(report.input.records, 13);
    assert_eq!(report.retained_original_records, 13);
    assert!(report.selected.as_ref().unwrap().exact);
    let observations: Vec<_> = input::observations(&rows).collect();
    assert_eq!(observations[5].id, 5);
    assert_eq!(observations[5].row, 1);
    assert_eq!(observations[5].position, 1);
    assert_eq!(observations[5].tokens, &[1, 10]);
    for prior in [Prior::Coordinates, Prior::Categorical] {
        let c = model::corpus(&report, &rows, prior)?;
        assert_eq!(c.occurrences().len(), 13);
        assert_eq!(c.groups().len(), 2);
        assert!(matches!(
            model::specification(&report, &rows, &c, prior)?.anchors,
            weighted::domain::Anchors::All
        ));
    }
    let mut bad = rows;
    bad[0].changes[1] = false;
    assert!(input::inspect(36, &bad).is_err());
    Ok(())
}

#[test]
fn full_noisy_training_pool_keeps_corrupted_labels_and_all_six_joint_endpoints() -> Result<()> {
    let (v, rows) = input::training_only("basis-parity-noise20")?;
    let stage = fit::construct(v, &rows)?;
    assert_eq!(stage.discovery.input.records, 1024);
    let selected = stage.discovery.selected.as_ref().unwrap();
    assert!(!selected.exact);
    assert_eq!(
        selected.frequencies.iter().map(|f| f.count).sum::<u64>(),
        1024
    );
    assert!(
        selected
            .frequencies
            .iter()
            .filter(|f| matches!(f.atom, Atom::Quantity(_)))
            .count()
            > 2
    );
    assert_eq!(stage.endpoints.len(), 6);
    assert_eq!(stage.all_anchor_diagnostics.len(), 2);
    assert!(
        stage
            .all_anchor_diagnostics
            .iter()
            .all(|d| d["feasible"] == false)
    );
    for endpoint in &stage.endpoints {
        assert!(endpoint.error.is_none(), "{:?}", endpoint.error);
        let c = model::corpus(&stage.discovery, &rows, endpoint.prior)?;
        let out = endpoint.training.as_ref().unwrap();
        assert_eq!(out.original_records, 1024);
        assert_eq!(out.work.gradient_records, 1024 * endpoint.policy.epochs);
        assert!(
            matches!(&out.specification.anchors,weighted::domain::Anchors::Groups(groups) if !groups.is_empty() && groups.len()<c.groups().len())
        );
        weighted::training::replay(out, &c)?;
        let model = endpoint.model.as_ref().unwrap();
        for r in input::observations(&rows).take(10) {
            let q = model.query(r.tokens)?;
            let z = model.independent_logits(r.tokens)?;
            assert_eq!(joint::argmax(&q.logits), joint::argmax(&z));
            assert!(q.logits.iter().zip(&z).all(|(a, b)| (a - b).abs() < 1e-8));
            assert!(!q.target_realizability_certified);
        }
        if endpoint.policy.mode == joint::training::Mode::Mini {
            assert_eq!(out.steps[1].batch.records(), 3);
        }
    }
    Ok(())
}

#[test]
fn changed_source_and_absent_categorical_entries_are_not_silently_identified() -> Result<()> {
    let (v, rows) = input::training_only("basis-parity-noise20")?;
    let report = discovery::run(v, &rows)?;
    let candidate = report.selected.as_ref().unwrap();
    let (features, known) = model::feature(
        candidate,
        Some(Atom::Quantity(100)),
        &report.coordinates,
        Prior::Categorical,
    )?;
    assert!(!known);
    assert_eq!(features[0], report.coordinates[0]);
    let mut changed = rows;
    changed[0].targets[0] = 0;
    assert!(model::corpus(&report, &changed, Prior::Coordinates).is_err());
    Ok(())
}
