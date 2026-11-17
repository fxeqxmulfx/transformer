use super::{Rule, project};
use crate::{C, source, update};
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use rotary::{
    atom::{Atom, Family, State},
    model::Model,
};

fn distance(a: State, b: State) -> f64 {
    (a[0] - b[0]).norm_sqr() + (a[1] - b[1]).norm_sqr()
}
fn inner(a: State, b: State) -> f64 {
    (a[0].conj() * b[0] + a[1].conj() * b[1]).re
}

#[test]
fn complete_projection_covers_zero_inside_dominant_and_two_active_cases() {
    let zero = [C::default(); 2];
    assert_eq!(project(zero).unwrap(), zero);
    let inside = [C::new(0.1, 0.2), C::new(-0.2, 0.1)];
    assert_eq!(project(inside).unwrap(), inside);
    for (input, expected) in [
        (
            [C::new(0., 2.), C::new(1., 0.)],
            [C::new(0., 1.), C::default()],
        ),
        (
            [C::new(1., 0.), C::new(0., 2.)],
            [C::default(), C::new(0., 1.)],
        ),
        (
            [C::new(3., 4.), C::default()],
            [C::new(0.6, 0.8), C::default()],
        ),
        (
            [C::new(0., 1.), C::new(1., 0.)],
            [C::new(0., 0.5), C::new(0.5, 0.)],
        ),
        (
            [C::new(0., 1.2), C::new(0.8, 0.)],
            [C::new(0., 0.7), C::new(0.3, 0.)],
        ),
    ] {
        assert!(distance(project(input).unwrap(), expected) < 1e-28);
    }
    let z = [C::new(2., 0.), C::new(1., 0.)];
    assert!(distance(z, project(z).unwrap()) < distance(z, update::project_row(z).unwrap()));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(project([C::new(bad, 0.), C::default()]).is_err());
    }
}

#[test]
fn complex_projection_satisfies_independent_variational_and_distance_checks() {
    for i in 0..256 {
        let t = i as f64 + 1.;
        let z = [
            C::from_polar(((i * 17) % 31) as f64 / 7., 0.37 * t),
            C::from_polar(((i * 23) % 29) as f64 / 9., -0.51 * t),
        ];
        let p = project(z).unwrap();
        assert!(p[0].norm() + p[1].norm() <= 1. + 1e-14);
        let residual = [z[0] - p[0], z[1] - p[1]];
        // The dual of the complete complex row-l1 ball is the maximum magnitude.
        // This checks the variational condition independently of the branch formula.
        let gap = residual[0].norm().max(residual[1].norm()) - inner(residual, p);
        assert!(gap.abs() < 2e-13, "variational gap {gap} at {i}");
        for a in 0..=4 {
            for b in 0..=4 - a {
                let v = [
                    C::from_polar(a as f64 / 4., 0.29 * t + a as f64),
                    C::from_polar(b as f64 / 4., 0.91 * t - b as f64),
                ];
                assert!(inner(residual, [v[0] - p[0], v[1] - p[1]]) <= 2e-13);
                assert!(distance(z, p) <= distance(z, v) + 2e-13);
            }
        }
    }
}

fn cm101_model() -> Model {
    // The reviewed nonconstant Lean counterexample is only an execution control.
    let mut atom = Atom::identity_start(2, Family::Rotary);
    atom.frequency = Some([0., 0.]);
    atom.initial = [C::new(1., 0.), C::new(0.5, 0.)];
    atom.transition[0] = [
        [C::new(-1., 0.), C::default()],
        [C::default(), C::new(-1., 0.)],
    ];
    atom.output = vec![
        [C::new(-2. / 3., 0.), C::new(-1. / 3., 0.)],
        [C::new(2. / 3., 0.), C::new(1. / 3., 0.)],
    ];
    let mut model = Model::zero(2, 1., Family::Rotary).unwrap();
    model.mix(atom, 1.).unwrap();
    model
}

#[test]
fn both_modes_escape_actual_radial_ce_failure_with_the_complete_new_projection() {
    let records = [
        Record {
            prefix: &[0],
            target: 0,
        },
        Record {
            prefix: &[1],
            target: 1,
        },
    ];
    let initial = cm101_model();
    let content = source::content_hash(&initial).unwrap();
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let config = update::Config {
            mode,
            query_step: 1e-4,
            learning_rate: 4.,
            line_iterations: 64,
            projection: Rule::Radial,
        };
        let mut old = initial.clone();
        let unchanged = update::step(&mut old, &records, config).unwrap();
        assert!((unchanged.after_ce - unchanged.centre_ce).abs() < 1e-12);
        for (a, b) in old.atoms[1].output.iter().zip(&initial.atoms[1].output) {
            assert!(distance(*a, *b) < 1e-26);
        }
        let mut fixed = initial.clone();
        let receipt = update::step(
            &mut fixed,
            &records,
            update::Config {
                projection: Rule::Euclidean,
                ..config
            },
        )
        .unwrap();
        assert!(receipt.after_ce < receipt.centre_ce - 1e-3);
        assert!(receipt.after_ce < unchanged.after_ce - 1e-3);
        assert_eq!(source::content_hash(&fixed).unwrap(), content);
        assert_eq!(receipt.row_coordinates_processed, 16);
        assert_eq!(
            receipt.total_ce_evaluations,
            142 + if mode == Mode::ForwardMini { 8 } else { 0 }
        );
        for r in &records {
            let before = cross_entropy(&initial.logits(r.prefix).unwrap(), r.target).unwrap();
            let after = cross_entropy(&fixed.logits(r.prefix).unwrap(), r.target).unwrap();
            assert!(after < before - 1e-3);
            assert_eq!(fixed.next_object(r.prefix).unwrap(), r.target);
        }
    }
}
