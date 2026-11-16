use super::*;
use legacy::records::Packed;
const LIMIT: u64 = 32 * 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}
fn fixture(count: usize) -> (Domain, Vec<Record>) {
    let base = [(0, 0, 0), (1, 0, 0), (2, 1, 2), (3, 2, 1)];
    let rows: Vec<_> = (0..count)
        .map(|i| {
            let (a, b, label) = base[i % 4];
            Record {
                states: vec![a, b],
                label,
            }
        })
        .collect();
    let domain = Domain::new(
        Shape {
            programs: 2,
            states: 8,
            classes: 5,
        },
        &rows,
        1.,
        32.,
        &mut work(),
    )
    .unwrap();
    (domain, rows)
}
fn config(mode: Mode, rate: f64) -> Config {
    Config {
        mode,
        epochs: match mode {
            Mode::Forward => 0,
            Mode::Full => 4,
            Mode::Mini => 1,
        },
        batch_size: 127,
        learning_rate: rate,
        initial_mass: 8.,
    }
}
fn initial(d: &Domain, nonuniform: bool) -> legacy::Parameters {
    let mut q = d.initial(8., &mut work()).unwrap();
    if nonuniform {
        for &(p, s) in &d.unknown {
            let at = d.offsets[p * d.shape.states + s];
            if s % 2 == 0 {
                for c in 0..5 {
                    q.values[at + c] = q.values[p] * (c + 1) as f64 / 15.;
                }
            }
        }
    }
    q
}
fn coefficients(old: &legacy::head::Head, new: &Head) {
    let sh = new.shape();
    for p in 0..sh.programs {
        for f in 0..sh.states {
            for c in 0..sh.classes {
                let a = old.coefficient(p, f, c).unwrap();
                let b = new.coefficient(p, f, c).unwrap();
                assert_eq!(
                    (a.0.to_bits(), a.1.to_bits()),
                    (b.0.to_bits(), b.1.to_bits())
                );
            }
        }
    }
}
fn step(a: &Step, b: &Step) {
    assert_eq!(
        (
            a.epoch,
            a.first_record,
            a.end_record_exclusive,
            a.uniform_unknown_programs
        ),
        (
            b.epoch,
            b.first_record,
            b.end_record_exclusive,
            b.uniform_unknown_programs
        )
    );
    assert_eq!(
        bits([
            a.before_ce,
            a.after_ce,
            a.before_mass,
            a.after_mass,
            a.parameter_change_l2,
            a.maximum_feasibility_residual,
            a.maximum_kkt_residual
        ]),
        bits([
            b.before_ce,
            b.after_ce,
            b.before_mass,
            b.after_mass,
            b.parameter_change_l2,
            b.maximum_feasibility_residual,
            b.maximum_kkt_residual
        ])
    );
}
fn result(a: &legacy::training::Trained, b: &Trained) {
    assert_eq!(
        bits(a.parameters.values.iter().copied()),
        bits(b.parameters.values().copied())
    );
    assert_eq!(
        (a.initial_ce.to_bits(), a.final_ce.to_bits()),
        (b.initial_ce.to_bits(), b.final_ce.to_bits())
    );
    assert_eq!(a.operations, b.operations);
    assert_eq!(a.config, b.config);
    assert_eq!(a.steps.len(), b.steps.len());
    for (a, b) in a.steps.iter().zip(&b.steps) {
        step(a, b);
    }
    assert_eq!(
        bits(a.audit.column_l1.iter().copied()),
        bits(b.audit.column_l1.iter().copied())
    );
    assert_eq!(
        a.audit.maximum_inverse_residual.to_bits(),
        b.audit.maximum_inverse_residual.to_bits()
    );
    assert_eq!(a.audit.logical_coefficients, b.audit.logical_coefficients);
    assert!(!b.target_realizability_certified && !b.exact_real_projection_certified);
    coefficients(&a.head, &b.head);
    let (mut ba, mut bb) = (
        legacy::head::PredictionBudget::new(64).unwrap(),
        legacy::head::PredictionBudget::new(64).unwrap(),
    );
    for x in 0..8 {
        for y in 0..8 {
            let states = [x, y];
            assert_eq!(
                bits(a.head.logits(&states, &mut work()).unwrap()),
                bits(b.head.logits(&states, &mut work()).unwrap())
            );
            let pa = a.head.predict(&states, &mut ba, &mut work()).unwrap();
            let pb = b.head.predict(&states, &mut bb, &mut work()).unwrap();
            assert_eq!(bits(pa.logits), bits(pb.logits));
            assert_eq!(
                (pa.winner, pa.maximizers, pa.certificate),
                (pb.winner, pb.maximizers, pb.certificate)
            );
        }
    }
    assert_eq!(ba, bb);
}

#[test]
fn all_three_policies_match_complete_training_with_original_duplicates_and_tails() {
    for count in [128, 130] {
        let (d, rows) = fixture(count);
        let packed = Packed::from_records(d.shape, &rows, &mut work()).unwrap();
        for source in [Source::Expanded(&rows), Source::Packed(&packed)] {
            for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
                for rate in [3., 1e6] {
                    for nonuniform in [false, true] {
                        let cfg = config(mode, rate);
                        let old = initial(&d, nonuniform);
                        let compact =
                            Parameters::from_legacy(&d, &old, LIMIT, &mut work()).unwrap();
                        let (mut wa, mut wb) = (work(), work());
                        let a = legacy::training::train_from_source(
                            &d,
                            source,
                            cfg.clone(),
                            old,
                            &mut wa,
                        )
                        .unwrap();
                        let b =
                            train_from_source(&d, source, cfg, compact, LIMIT, &mut wb).unwrap();
                        result(&a, &b);
                        assert_eq!(wa.used, wb.used);
                        if mode == Mode::Mini {
                            assert_eq!(b.steps.len(), 2);
                            assert_eq!(b.steps[1].first_record, 127);
                            assert_eq!(b.steps[1].end_record_exclusive, count);
                        }
                        if mode != Mode::Forward {
                            assert!(b.final_ce < b.initial_ce);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_intermediate_parameter_and_head_matches_actual_full_and_mini_updates() {
    let (d, rows) = fixture(130);
    for mode in [Mode::Full, Mode::Mini] {
        let cfg = config(mode, 3.);
        let mut old = initial(&d, true);
        let mut compact = Parameters::from_legacy(&d, &old, LIMIT, &mut work()).unwrap();
        let (mut wa, mut wb) = (work(), work());
        let batch = if mode == Mode::Mini { 127 } else { rows.len() };
        for _ in 0..cfg.epochs {
            for first in (0..rows.len()).step_by(batch) {
                let range = first..(first + batch).min(rows.len());
                let a = legacy::head::Head::build(&d, &old, &mut wa).unwrap();
                let b = Head::build(&d, &compact, LIMIT, &mut wb).unwrap();
                coefficients(&a, &b);
                let (la, ga) = legacy::training::gradient_source(
                    &d,
                    &a,
                    Source::Expanded(&rows),
                    range.clone(),
                    &mut wa,
                )
                .unwrap();
                let (lb, gb) =
                    gradient_source(&d, &b, Source::Expanded(&rows), range, LIMIT, &mut wb)
                        .unwrap();
                assert_eq!(la.to_bits(), lb.to_bits());
                assert_eq!(bits(ga.values.iter().copied()), bits(gb.values().copied()));
                let ta =
                    legacy::projection::Target::step(&d, &old, &ga, cfg.learning_rate, &mut wa)
                        .unwrap();
                let tb =
                    Target::step(&d, &compact, &gb, cfg.learning_rate, LIMIT, &mut wb).unwrap();
                old = legacy::projection::project(&d, &ta, &mut wa)
                    .unwrap()
                    .parameters;
                compact = projection::project(&d, &tb, LIMIT, &mut wb)
                    .unwrap()
                    .parameters;
                assert_eq!(
                    bits(old.values.iter().copied()),
                    bits(compact.values().copied())
                );
                assert_eq!(wa.used, wb.used);
            }
        }
        coefficients(
            &legacy::head::Head::build(&d, &old, &mut wa).unwrap(),
            &Head::build(&d, &compact, LIMIT, &mut wb).unwrap(),
        );
    }
}

#[test]
fn compact_initialization_gradient_tail_and_rejections_remain_original() {
    let (d, rows) = fixture(130);
    let cfg = config(Mode::Mini, 3.);
    let (mut wa, mut wb) = (work(), work());
    let a = legacy::training::train(&d, &rows, cfg.clone(), &mut wa).unwrap();
    let b = train(&d, &rows, cfg.clone(), LIMIT, &mut wb).unwrap();
    result(&a, &b);
    assert_eq!(wa.used, wb.used);
    let legacy_q = d.initial(8., &mut work()).unwrap();
    let q = Parameters::initial(&d, 8., LIMIT, &mut work()).unwrap();
    let old = legacy::head::Head::build(&d, &legacy_q, &mut work()).unwrap();
    let new = Head::build(&d, &q, LIMIT, &mut work()).unwrap();
    for range in [0..127, 127..130] {
        let a = legacy::training::gradient_source(
            &d,
            &old,
            Source::Expanded(&rows),
            range.clone(),
            &mut work(),
        )
        .unwrap();
        let b =
            gradient_source(&d, &new, Source::Expanded(&rows), range, LIMIT, &mut work()).unwrap();
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(bits(a.1.values), bits(b.1.values().copied()));
    }
    let mut refused = work();
    assert!(train(&d, &rows, cfg.clone(), 1, &mut refused).is_err());
    assert_eq!(refused.used, 0, "storage refusal precedes initialization");
    assert!(train(&d, &rows, cfg.clone(), LIMIT, &mut Work::new(1).unwrap()).is_err());
    assert!(train(&d, &[], cfg.clone(), LIMIT, &mut work()).is_err());
    let mut bad = rows.clone();
    bad[0].label = 4;
    assert!(gradient(&d, &new, &bad, LIMIT, &mut work()).is_err());
    assert!(train(&d, &bad, cfg.clone(), LIMIT, &mut work()).is_err());
    assert!(
        gradient_source(
            &d,
            &new,
            Source::Expanded(&rows),
            129..131,
            LIMIT,
            &mut work()
        )
        .is_err()
    );
    let mut labels = d.labels.clone();
    labels[7] = Some(4);
    let extra = Domain::from_labels(d.shape, labels, 1., 32., &mut work()).unwrap();
    assert!(train(&extra, &rows, cfg, LIMIT, &mut work()).is_err());
}

#[test]
fn simultaneous_training_buffers_are_charged_beyond_the_final_model() {
    let (d, rows) = fixture(130);
    let cfg = config(Mode::Mini, 3.);
    let ample = train(&d, &rows, cfg.clone(), LIMIT, &mut work()).unwrap();
    let owned = ample.parameters.allocation_bytes() + ample.head.allocation_bytes();
    assert!(owned < 8192);
    assert!(train(&d, &rows, cfg.clone(), owned + 100, &mut work()).is_err());
    let mut limited = work();
    let error = train(&d, &rows, cfg.clone(), 12_288, &mut limited).unwrap_err();
    assert!(error.to_string().contains("buffers"));
    assert!(
        limited.used > 0,
        "the initial model fits, but simultaneous update buffers do not"
    );
    result(
        &legacy::training::train(&d, &rows, cfg.clone(), &mut work()).unwrap(),
        &train(&d, &rows, cfg, 65_536, &mut work()).unwrap(),
    );
}
