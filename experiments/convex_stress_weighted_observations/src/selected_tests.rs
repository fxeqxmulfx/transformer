//! CM61's actual noisy witness, selected projection, and complete CE accounting.
use crate::{
    corpus::{Corpus, Limits},
    domain::{Anchors, Domain, HeadSpec},
    *,
};
use joint::{
    data::Record,
    training::{Mode, Policy},
};

fn corpus() -> Result<Corpus> {
    Corpus::build(
        Limits {
            records: 3,
            groups: 3,
            programs: 1,
            vocabulary: 5,
        },
        [
            ([1., 0., 0., 0., 0.], 0),
            ([1., 0., 0., 0., 0.], 4),
            ([0., 0., 0., 0., 1.], 4),
        ]
        .into_iter()
        .enumerate()
        .map(|(id, (h, label))| {
            Ok(Record {
                id,
                features: vec![h],
                label,
            })
        }),
    )
}
fn spec(anchors: Anchors) -> HeadSpec {
    HeadSpec {
        coordinates: (0..5).map(|c| c as f64 / 4.).collect(),
        budgets: vec![16.],
        radius: 16.,
        minimum_mass: 1.,
        anchors,
    }
}
fn policy(mode: Mode) -> Policy {
    Policy {
        mode,
        epochs: if mode == Mode::Forward { 0 } else { 8 },
        batch: 2,
        learning_rate: 1.,
        initial_total_mass: 4.,
        projection_cycles: 10000,
        projection_tolerance: 1e-10,
    }
}

#[test]
fn selecting_copy_anchors_restores_feasibility_and_keeps_every_noisy_ce_observation() -> Result<()>
{
    let corpus = corpus()?;
    let mut bounded = policy(Mode::Forward);
    bounded.projection_cycles = 128;
    assert!(training::fit(&corpus, &spec(Anchors::All), bounded).is_err());
    let selected = spec(Anchors::Groups(vec![2]));
    let d = Domain::new(&corpus, &selected)?;
    assert!(d.violation(&[[0.5, 0., 0., 0., 1., 1.]]) < 1e-12);
    assert_eq!(d.constraints().anchors[0].len(), 2);
    // The content direction is deliberately not certified by these anchors.
    assert!(d.constraints().span_residual(&[[1., 0., 0., 0., 0.]])?[0] > 0.99);
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let out = training::fit(&corpus, &selected, policy(mode))?;
        training::replay(&out, &corpus)?;
        assert_eq!(out.original_records, 3);
        assert_eq!(out.distinct_groups, 3);
        assert_eq!(
            corpus.groups().iter().map(|r| r.label).collect::<Vec<_>>(),
            vec![0, 4, 4]
        );
        assert!(out.domain.violation(&out.head.theta) <= 1e-10);
        let mut independent = 0.;
        for row in corpus.groups() {
            independent += joint::ce(&out.head.independent_logits(&row.features)?, row.label);
        }
        assert!((out.final_ce - independent / 3.).abs() < 1e-12);
        if mode == Mode::Forward {
            assert_eq!(out.work.gradient_records, 0);
            assert_eq!(out.work.effective_updates, 0);
        } else {
            assert_eq!(out.work.gradient_records, 24);
            assert!(out.work.effective_updates > 0);
            assert!(out.final_ce < out.initial_ce - 1e-3);
        }
        if mode == Mode::Mini {
            assert_eq!(out.steps[1].batch.start, 2);
            assert_eq!(out.steps[1].batch.records(), 1);
            assert_eq!(out.steps[0].batch.counts, vec![(0, 1), (1, 1)]);
        }
        let mut changed = out.clone();
        changed.specification.anchors = Anchors::Groups(vec![]);
        assert!(training::replay(&changed, &corpus).is_err());
    }
    Ok(())
}

#[test]
fn selected_projection_satisfies_variational_checks_and_an_empty_selection_uses_no_fake_rows()
-> Result<()> {
    let corpus = corpus()?;
    let empty = Domain::new(&corpus, &spec(Anchors::Groups(vec![])))?;
    assert_eq!(empty.constraints().anchors[0].len(), 1);
    let selected = Domain::new(&corpus, &spec(Anchors::Groups(vec![2])))?;
    let input = vec![[3., -2., 0.5, 1., -1., 2.]];
    let a = empty.project(&input, 10000, 1e-10)?;
    let b = selected.project(&input, 10000, 1e-10)?;
    for (x, y) in a.theta.iter().flatten().zip(b.theta.iter().flatten()) {
        assert!((x - y).abs() < 1e-8);
    }
    assert!(empty.violation(&a.theta) < 1e-10);
    for beta in [-1., 0., 0.5, 1., 2.] {
        let feasible = [[beta, 0., 0., 0., 1., 1.]];
        assert!(empty.violation(&feasible) < 1e-12);
        let inner: f64 = input
            .iter()
            .flatten()
            .zip(a.theta.iter().flatten())
            .zip(feasible.iter().flatten())
            .map(|((x, y), z)| (x - y) * (z - y))
            .sum();
        assert!(inner < 1e-7, "projection variational residual {inner}");
    }
    assert!(empty.project(&input, 0, 1e-10).is_err());
    assert!(empty.project(&[[f64::NAN; 6]], 100, 1e-10).is_err());
    for groups in [vec![2, 2], vec![2, 0], vec![3]] {
        assert!(Domain::new(&corpus, &spec(Anchors::Groups(groups))).is_err());
    }
    if let Domain::Selected(mut changed) = selected {
        changed.constraints.basis[0][0][0] += 0.01;
        assert!(changed.validate().is_err());
    } else {
        panic!("expected explicit selected domain");
    }
    Ok(())
}

#[test]
fn selecting_all_clean_groups_reproduces_the_original_domain_projection() -> Result<()> {
    let c = Corpus::build(
        Limits {
            records: 3,
            groups: 3,
            programs: 1,
            vocabulary: 5,
        },
        (0..3).map(|i| {
            Ok(Record {
                id: i,
                features: vec![[i as f64 / 2., 1., 0., 0., 0.]],
                label: i + 1,
            })
        }),
    )?;
    let a = Domain::new(&c, &spec(Anchors::All))?;
    let b = Domain::new(&c, &spec(Anchors::Groups(vec![0, 1, 2])))?;
    assert_eq!(a.constraints(), b.constraints());
    for input in [
        vec![[0., 0., 0., 0., 0., 4.]],
        vec![[1., -2., 3., -4., 5., 6.]],
    ] {
        assert_eq!(
            a.project(&input, 10000, 1e-10)?,
            b.project(&input, 10000, 1e-10)?
        );
    }
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let a = training::fit(&c, &spec(Anchors::All), policy(mode))?;
        let b = training::fit(&c, &spec(Anchors::Groups(vec![0, 1, 2])), policy(mode))?;
        assert_eq!(a.head, b.head);
        assert_eq!(a.steps, b.steps);
        assert_eq!(a.work, b.work);
    }
    Ok(())
}

#[test]
fn a_declared_joint_initialization_is_projected_and_replayed_without_extra_anchors() -> Result<()> {
    let c = corpus()?;
    let mut specification = spec(Anchors::Groups(vec![2]));
    specification.radius = 128.;
    let initial = vec![[2., 0., 0., 0., 4., 4.]];
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let out = training::fit_from(&c, &specification, policy(mode), initial.clone())?;
        assert_eq!(out.initial_proposal, initial);
        assert_eq!(out.domain.constraints().anchors[0].len(), 2);
        assert!(out.domain.violation(&out.initial_projection.theta) < 1e-10);
        assert_eq!(out.work.gradient_records, 3 * policy(mode).epochs);
        training::replay(&out, &c)?;
        if mode == Mode::Forward {
            for (a, b) in initial
                .iter()
                .flatten()
                .zip(out.head.theta.iter().flatten())
            {
                assert!((a - b).abs() < 1e-10);
            }
            assert_eq!(out.work.effective_updates, 0);
        }
        let mut bad = out;
        bad.initial_proposal[0][0] += 0.5;
        assert!(training::replay(&bad, &c).is_err());
    }
    let mut wrong = initial;
    wrong[0][5] = 3.;
    assert!(training::fit_from(&c, &specification, policy(Mode::Full), wrong).is_err());
    assert!(
        training::fit_from(&c, &specification, policy(Mode::Full), vec![[f64::NAN; 6]]).is_err()
    );
    Ok(())
}
