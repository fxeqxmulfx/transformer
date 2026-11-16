//! Training controls that actually retain large shared class complements.
//! Compare against the complete pre-CM70 model, including every saved update.
use super::*;

fn constant() -> (Domain, Vec<Record>) {
    let rows = (0..130)
        .map(|i| Record {
            states: vec![4 * (i % 2), 4 * ((i / 2) % 2)],
            label: 0,
        })
        .collect::<Vec<_>>();
    let shape = Shape {
        programs: 2,
        states: 8,
        classes: 97,
    };
    (
        Domain::new(shape, &rows, 1., 32., &mut work()).unwrap(),
        rows,
    )
}
fn nonuniform(d: &Domain) -> legacy::Parameters {
    let mut q = d.initial(8., &mut work()).unwrap();
    for &(p, s) in &d.unknown {
        let at = d.offsets[p * 8 + s];
        q.values[at..at + 97].fill(4. / 98.);
        // The stored baseline is the first unobserved class, 1. Keep its
        // common value; class 96 is an explicit nonuniform exception.
        q.values[at + 96] = 8. / 98.;
    }
    q
}

#[test]
fn shared_columns_preserve_complete_full_mini_forward_endpoints_steps_and_checkpoints() {
    let (d, rows) = constant();
    let packed = Packed::from_records(d.shape, &rows, LIMIT, &mut work()).unwrap();
    let old_packed = LegacyPacked::from_records(d.shape, &rows, &mut work()).unwrap();
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        for rate in [3., 1e6] {
            let cfg = config(mode, rate);
            let old = nonuniform(&d);
            let compact = Parameters::from_legacy(&d, &old, LIMIT, &mut work()).unwrap();
            let (mut a_work, mut b_work) = (work(), work());
            let a = legacy::training::train_from_source(
                &d,
                LegacySource::Packed(&old_packed),
                cfg.clone(),
                old,
                &mut a_work,
            )
            .unwrap();
            let b = train_from_source(
                &d,
                Source::Packed(&packed),
                cfg,
                compact,
                LIMIT,
                &mut b_work,
            )
            .unwrap();
            assert!(b.head.evaluated_columns(&[0, 4]).unwrap() <= 6);
            result(&a, &b);
            assert_eq!(a_work.used, b_work.used);
            let (mut old_bytes, mut new_bytes) = (Vec::new(), Vec::new());
            let old_hash =
                legacy::persistence::write(&d, &a.parameters, &mut old_bytes, &mut a_work).unwrap();
            let new_hash =
                crate::persistence::write(&d, &b.parameters, &mut new_bytes, &mut b_work).unwrap();
            assert_eq!(old_hash, new_hash);
            assert_eq!(old_bytes, new_bytes);
            assert_eq!(a_work.used, b_work.used);
            if mode == Mode::Mini {
                assert_eq!(b.steps.last().unwrap().first_record, 127);
                assert_eq!(b.steps.last().unwrap().end_record_exclusive, 130);
            }
            if mode == Mode::Forward {
                assert!(b.steps.is_empty());
            }
        }
    }
}

#[test]
fn shared_plans_rebuild_after_every_actual_minibatch_gradient_and_projection() {
    let (d, rows) = constant();
    let mut old = nonuniform(&d);
    let mut compact = Parameters::from_legacy(&d, &old, LIMIT, &mut work()).unwrap();
    let (mut a_work, mut b_work) = (work(), work());
    for range in [0..127, 127..130] {
        let a = legacy::head::Head::build(&d, &old, &mut a_work).unwrap();
        let b = Head::build(&d, &compact, LIMIT, &mut b_work).unwrap();
        assert!(b.evaluated_columns(&[0, 4]).unwrap() <= 6);
        coefficients(&a, &b);
        for s in 0..8 {
            assert_eq!(
                bits(a.logits(&[s, 7 - s], &mut a_work).unwrap()),
                bits(b.logits(&[s, 7 - s], &mut b_work).unwrap())
            );
        }
        let (la, ga) = legacy::training::gradient_source(
            &d,
            &a,
            LegacySource::Expanded(&rows),
            range.clone(),
            &mut a_work,
        )
        .unwrap();
        let (lb, gb) =
            gradient_source(&d, &b, Source::Expanded(&rows), range, LIMIT, &mut b_work).unwrap();
        assert_eq!(la.to_bits(), lb.to_bits());
        assert_eq!(bits(ga.values.iter().copied()), bits(gb.values().copied()));
        let ta = legacy::projection::Target::step(&d, &old, &ga, 3., &mut a_work).unwrap();
        let tb = Target::step(&d, &compact, &gb, 3., LIMIT, &mut b_work).unwrap();
        old = legacy::projection::project(&d, &ta, &mut a_work)
            .unwrap()
            .parameters;
        compact = projection::project(&d, &tb, LIMIT, &mut b_work)
            .unwrap()
            .parameters;
        assert_eq!(
            bits(old.values.iter().copied()),
            bits(compact.values().copied())
        );
        assert_eq!(a_work.used, b_work.used);
    }
    coefficients(
        &legacy::head::Head::build(&d, &old, &mut a_work).unwrap(),
        &Head::build(&d, &compact, LIMIT, &mut b_work).unwrap(),
    );
    assert_eq!(a_work.used, b_work.used);
}
