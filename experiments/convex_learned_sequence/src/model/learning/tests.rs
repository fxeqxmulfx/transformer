use super::{MiniLimits, MiniPlan, Record};
use crate::domain::DomainLimits;
use crate::model::{ComplexModel, ModelLimits, Renaming};
use crate::{ProjectionLimits, Rational, Refusal};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};

fn ratio(n: i64, d: i64) -> Rational {
    Rational::new(BigInt::from(n), BigInt::from(d))
}
fn model(objects: usize) -> ComplexModel {
    ComplexModel::zero(
        objects,
        1,
        ModelLimits {
            domain: DomainLimits {
                max_prefixes: 8,
                max_patterns: 256,
                max_real_parameters: 100_000,
                max_row_entries: 1000,
            },
            projection: ProjectionLimits {
                max_active_sets: 256,
                max_matrix_entries: 10_000,
                max_coefficient_bits: 4096,
            },
            max_working_bytes: 1 << 30,
        },
    )
    .unwrap()
}
fn limits() -> MiniLimits {
    MiniLimits {
        max_records: 100,
        max_pullback_terms: 10_000_000,
        max_projection_candidates: 10_000_000,
        max_working_bytes: 1 << 30,
    }
}
fn record(object: usize, target: usize) -> Record {
    Record {
        tokens: vec![object],
        target,
    }
}
fn bias_witness() -> ComplexModel {
    let base = model(2);
    let mut point = base.parameters().to_vec();
    point[base.parameter_index(7, 1, 0, 0, 0).unwrap()] = ratio(3, 1);
    point[base.parameter_index(7, 1, 0, 1, 0).unwrap()] = ratio(3, 1);
    base.with_parameters(point).unwrap()
}

#[test]
fn cm82_nonzero_ce_slope_keeps_repeated_records_and_short_tail_normalization() {
    let model = bias_witness();
    let records = [record(1, 1), record(0, 0), record(1, 1)];
    let result = model.mini_gradient(&records, None, limits()).unwrap();
    assert_eq!(result.coordinates.len(), 192);
    assert_eq!(result.plan.records, 3);
    assert_eq!(result.plan.passes, 1);
    assert_eq!(result.plan.pullback_terms, 3 * 192);
    assert!((result.mean_loss - 2.0_f64.ln()).abs() < 1e-15);
    let at = |s, k, d| result.coordinates[model.parameter_index(7, 1, s, k, d).unwrap()].clone();
    assert_eq!(at(0, 0, 0), ratio(-1, 6));
    assert_eq!(at(0, 1, 0), ratio(1, 6));
    assert_eq!(at(1, 0, 0), ratio(1, 6));
    assert_eq!(at(1, 1, 0), ratio(-1, 6));
    assert_eq!(at(0, 0, 1), ratio(1, 6));
    assert_eq!(at(0, 0, 2), ratio(-1, 3));
    assert!(result.coordinates[..24].iter().all(Rational::is_zero));
    let unique = model.mini_gradient(&records[..2], None, limits()).unwrap();
    let tail = model.mini_gradient(&records[2..], None, limits()).unwrap();
    let coordinate = model.parameter_index(7, 1, 0, 0, 0).unwrap();
    assert_eq!(unique.coordinates[coordinate], ratio(0, 1));
    assert_eq!(tail.coordinates[coordinate], ratio(-1, 2));
    assert_eq!(tail.plan.records, 1);
    let reordered = [records[1].clone(), records[0].clone(), records[2].clone()];
    assert_eq!(
        model
            .mini_gradient(&reordered, None, limits())
            .unwrap()
            .coordinates,
        result.coordinates
    );
}

#[test]
fn both_shared_passes_match_actual_ce_finite_differences_in_all_selected_cell_coordinates() {
    let base = model(3);
    let mut point = base.parameters().to_vec();
    for class in 0..3 {
        for sign in 0..2 {
            for component in 0..2 {
                point[base.parameter_index(15, class, sign, component, 0).unwrap()] = ratio(3, 1);
            }
        }
    }
    point[base.parameter_index(15, 1, 0, 0, 0).unwrap()] = ratio(4, 1);
    let model = base.with_parameters(point).unwrap();
    let renaming = Renaming::new(vec![0, 2, 1]).unwrap();
    let records = [record(1, 1)];
    let gradient = model
        .mini_gradient(&records, Some(&renaming), limits())
        .unwrap();
    let separate = model.mini_gradient(&records, None, limits()).unwrap();
    assert_ne!(gradient.coordinates, separate.coordinates);
    assert_eq!(gradient.plan.passes, 2);
    assert_eq!(gradient.plan.pullback_terms, 2 * 768);
    let step = ratio(1, 100_000);
    for class in 0..3 {
        for sign in 0..2 {
            for component in 0..2 {
                for coordinate in 0..4 {
                    let index = model
                        .parameter_index(15, class, sign, component, coordinate)
                        .unwrap();
                    let mut plus = model.parameters().to_vec();
                    let mut minus = model.parameters().to_vec();
                    plus[index] += &step;
                    minus[index] -= &step;
                    let plus_loss = model
                        .with_parameters(plus)
                        .unwrap()
                        .paired_forward(&[1], &renaming)
                        .unwrap()
                        .cross_entropy(1)
                        .unwrap();
                    let minus_loss = model
                        .with_parameters(minus)
                        .unwrap()
                        .paired_forward(&[1], &renaming)
                        .unwrap()
                        .cross_entropy(1)
                        .unwrap();
                    let measured = (plus_loss - minus_loss) / (2.0 * step.to_f64().unwrap());
                    let analytic = gradient.coordinates[index].to_f64().unwrap();
                    assert!(
                        (measured - analytic).abs() < 2e-9,
                        "index {index}: {measured} vs {analytic}"
                    );
                }
            }
        }
    }
}

#[test]
fn mini_updates_complete_complex_state_and_replays_all_projection_certificates() {
    let old = model(3);
    let original = old.parameters().to_vec();
    let records = [record(1, 1), record(0, 0), record(1, 1)];
    let renaming = Renaming::new(vec![0, 2, 1]).unwrap();
    for shared in [None, Some(&renaming)] {
        let step = old
            .mini_step(&records, shared, &ratio(1, 16), limits())
            .unwrap();
        let next = &step.projected.model;
        assert_eq!(next.parameters().len(), 768);
        assert_eq!(step.proposal.len(), 768);
        assert_eq!(step.projected.certificates.len(), 192);
        next.verify_projection(&step.proposal, &step.projected.certificates)
            .unwrap();
        assert_eq!(old.parameters(), original);
        assert!(next.recovered().iter().any(|n| n.amplitude > 0.0));
        assert!(
            next.recovered()
                .iter()
                .any(|n| n.representation.iter().any(|z| z.im != 0.0))
        );
        let after = next.mini_gradient(&records, shared, limits()).unwrap();
        assert!(after.mean_loss < step.gradient.mean_loss);
        let mut corrupt = step.projected.certificates.clone();
        corrupt.pop();
        assert_eq!(
            next.verify_projection(&step.proposal, &corrupt),
            Err(Refusal::Shape)
        );
    }
}

#[test]
fn mini_refuses_incomplete_inputs_and_complete_work_or_memory_overflow_atomically() {
    let model = model(3);
    let saved = model.parameters().to_vec();
    let records = [record(1, 1)];
    assert!(matches!(
        model.mini_gradient(&[], None, limits()),
        Err(Refusal::EmptyBatch)
    ));
    assert!(matches!(
        model.mini_gradient(&[record(3, 0)], None, limits()),
        Err(Refusal::InvalidObject)
    ));
    assert!(matches!(
        model.mini_gradient(&[record(0, 3)], None, limits()),
        Err(Refusal::InvalidObject)
    ));
    assert!(matches!(
        model.mini_gradient(
            &[Record {
                tokens: vec![0, 1],
                target: 0
            }],
            None,
            limits()
        ),
        Err(Refusal::ContextTooLong)
    ));
    let wrong = Renaming::new(vec![1, 0]).unwrap();
    assert!(matches!(
        model.mini_gradient(&records, Some(&wrong), limits()),
        Err(Refusal::InvalidRenaming)
    ));
    assert!(matches!(
        model.mini_step(&records, None, &ratio(0, 1), limits()),
        Err(Refusal::LearningRate)
    ));
    for restricted in [
        MiniLimits {
            max_records: 0,
            ..limits()
        },
        MiniLimits {
            max_pullback_terms: 767,
            ..limits()
        },
        MiniLimits {
            max_projection_candidates: 192 * 16 - 1,
            ..limits()
        },
        MiniLimits {
            max_working_bytes: 1,
            ..limits()
        },
    ] {
        assert!(matches!(
            model.mini_step(&records, None, &ratio(1, 16), restricted),
            Err(Refusal::Budget)
        ));
    }
    assert!(matches!(
        MiniPlan::new(&model, usize::MAX, true, limits()),
        Err(Refusal::Overflow)
    ));
    let plan = MiniPlan::new(&model, 1, false, limits()).unwrap();
    assert_eq!(plan.projection_candidates, 192 * 16);
    assert!(plan.reserved_bytes > model.memory_plan().reserved_bytes);
    assert_eq!(model.parameters(), saved);
}
