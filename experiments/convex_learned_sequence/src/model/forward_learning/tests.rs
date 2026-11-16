use super::{ComplexModel, ForwardConfig, ForwardLimits, ForwardPlan, Record, Renaming};
use crate::domain::DomainLimits;
use crate::geometry::GeometryLimits;
use crate::model::{ModelLimits, learning::MiniLimits};
use crate::{ProjectionLimits, Rational, Refusal, checked_dot};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};

fn ratio(n: i64, d: i64) -> Rational {
    Rational::new(BigInt::from(n), BigInt::from(d))
}
fn model(objects: usize, context: usize) -> ComplexModel {
    ComplexModel::zero(
        objects,
        context,
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
fn limits() -> ForwardLimits {
    ForwardLimits {
        geometry: GeometryLimits {
            max_normal_projections: 8,
            max_active_candidates: 4096,
            max_rational_entries: 10_000,
            max_working_bytes: 1 << 30,
        },
        max_records: 100,
        max_geometry_builds: 512,
        max_active_candidates: 1_000_000,
        max_batch_evaluations: 20_000,
        max_forward_passes: 2_000_000,
        max_forward_terms: 20_000_000_000,
        max_constraint_terms: 20_000_000_000,
        max_working_bytes: 1 << 30,
    }
}
fn mini_limits() -> MiniLimits {
    MiniLimits {
        max_records: 100,
        max_pullback_terms: 10_000_000,
        max_projection_candidates: 1_000_000,
        max_working_bytes: 1 << 30,
    }
}
fn config() -> ForwardConfig {
    ForwardConfig {
        rate: ratio(1, 64),
        mixture: ratio(1, 4),
        query_cap: ratio(1, 10_000),
    }
}
fn records() -> [Record; 3] {
    [
        Record {
            tokens: vec![1],
            target: 1,
        },
        Record {
            tokens: vec![0],
            target: 0,
        },
        Record {
            tokens: vec![1],
            target: 1,
        },
    ]
}

#[test]
fn complete_forward_mini_matches_cm82_repeated_batch_slope_and_replays_all_queries() {
    let old = model(2, 1);
    let batch = records();
    let updated = old
        .forward_mini_step(&batch, None, &config(), limits())
        .unwrap();
    let estimate = &updated.estimate;
    assert_eq!(estimate.coordinates.len(), 192);
    assert_eq!(estimate.cells.len(), 64);
    assert_eq!(
        estimate
            .cells
            .iter()
            .map(|c| c.directions.len())
            .sum::<usize>(),
        192
    );
    for (index, cell) in estimate.cells.iter().enumerate() {
        let first = old
            .parameter_index(cell.mask, cell.class, cell.sign, cell.component, 0)
            .unwrap();
        assert_eq!(index * 3, first);
        assert_eq!(cell.directions.len(), 3);
        assert!(
            cell.directions
                .iter()
                .all(|q| q.step > Rational::zero() && q.step <= config().query_cap)
        );
    }
    // This TEST comparison is not called by the forward-only implementation.
    let analytic = estimate
        .mixed
        .mini_gradient(&batch, None, mini_limits())
        .unwrap();
    for (numeric, exact_formula) in estimate.coordinates.iter().zip(&analytic.coordinates) {
        assert!((numeric.to_f64().unwrap() - exact_formula.to_f64().unwrap()).abs() < 2e-8);
    }
    let bias = old.parameter_index(7, 1, 0, 0, 0).unwrap();
    assert!((estimate.coordinates[bias].to_f64().unwrap() + 1.0 / 6.0).abs() < 2e-8);
    assert_eq!(updated.projected.certificates.len(), 64);
    assert!(updated.final_loss < estimate.mixed_loss);
    assert_eq!(updated.projected.model.forward(&[1]).unwrap().object(), 1);
    assert_eq!(updated.projected.model.forward(&[0]).unwrap().object(), 0);
    assert!(
        updated
            .projected
            .model
            .recovered()
            .iter()
            .any(|n| n.amplitude > 0.0)
    );
    assert!(
        updated
            .projected
            .model
            .recovered()
            .iter()
            .any(|n| n.representation.iter().any(|w| w.im != 0.0))
    );
    assert!(old.parameters().iter().all(Rational::is_zero));
    old.verify_forward_update(&batch, None, &updated, limits())
        .unwrap();
}

#[test]
fn shared_forward_mini_uses_actual_nonuniform_ce_and_records_mixture_cost_and_short_tail() {
    let zero = model(3, 1);
    let batch = records();
    let renaming = Renaming::new(vec![0, 2, 1]).unwrap();
    let old = zero
        .mini_step(&batch, Some(&renaming), &ratio(1, 8), mini_limits())
        .unwrap()
        .projected
        .model;
    let mut cfg = config();
    cfg.mixture = ratio(1, 2);
    let tail = &batch[2..];
    let updated = old
        .forward_mini_step(tail, Some(&renaming), &cfg, limits())
        .unwrap();
    let estimate = &updated.estimate;
    assert_eq!(estimate.plan.records, 1);
    assert_eq!(estimate.plan.passes_per_record, 2);
    assert_eq!(estimate.coordinates.len(), 768);
    assert_eq!(estimate.cells.len(), 192);
    assert_eq!(
        estimate
            .cells
            .iter()
            .map(|c| c.directions.len())
            .sum::<usize>(),
        768
    );
    assert!(estimate.mixed_loss > estimate.old_loss);
    assert!(updated.final_loss < estimate.mixed_loss);
    let analytic = estimate
        .mixed
        .mini_gradient(tail, Some(&renaming), mini_limits())
        .unwrap();
    for (numeric, exact_formula) in estimate.coordinates.iter().zip(&analytic.coordinates) {
        assert!((numeric.to_f64().unwrap() - exact_formula.to_f64().unwrap()).abs() < 5e-8);
    }
    old.verify_forward_update(tail, Some(&renaming), &updated, limits())
        .unwrap();
}

#[test]
fn forced_equalities_keep_all_coordinates_and_recover_complete_nonorthogonal_cell_work() {
    let old = model(2, 2);
    let family = old.domain.plan();
    let mut selected = None;
    for mask in 0..family.patterns {
        let geometry = old.forward_geometry(mask, limits().geometry).unwrap();
        let basis = &geometry.basis().vectors;
        if geometry.certificate().forced.iter().any(|&x| x) && basis.len() < family.coordinates {
            let nonorthogonal = basis.iter().enumerate().any(|(i, a)| {
                basis.iter().enumerate().any(|(j, b)| {
                    i != j && !checked_dot(a, b, old.limits.projection).unwrap().is_zero()
                })
            });
            if nonorthogonal {
                selected = Some((mask, geometry));
                break;
            }
        }
    }
    let (mask, geometry) =
        selected.expect("complete raw family contains a nonorthogonal forced-equality cone");
    let first = old.parameter_index(mask, 1, 0, 0, 0).unwrap();
    let mut parameters = old.parameters.clone();
    parameters[first..first + family.coordinates].clone_from_slice(geometry.centre());
    let mixed = old.with_parameters(parameters).unwrap();
    let batch = [
        Record {
            tokens: vec![1, 0],
            target: 1,
        },
        Record {
            tokens: vec![0, 1],
            target: 0,
        },
    ];
    let (direction, probes) = mixed
        .forward_cell(&geometry, first, &batch, None, &config().query_cap)
        .unwrap();
    assert_eq!(direction.len(), 5);
    assert_eq!(probes.len(), geometry.basis().vectors.len());
    let pairings: Vec<_> = probes.iter().map(|p| p.pairing.clone()).collect();
    geometry.verify_pairings(&direction, &pairings).unwrap();
    let analytic = mixed.mini_gradient(&batch, None, mini_limits()).unwrap();
    for (basis, probe) in geometry.basis().vectors.iter().zip(&probes) {
        let actual_work = checked_dot(
            basis,
            &analytic.coordinates[first..first + 5],
            old.limits.projection,
        )
        .unwrap();
        assert!((actual_work.to_f64().unwrap() - probe.pairing.to_f64().unwrap()).abs() < 1e-7);
    }
}

#[test]
fn replay_rejects_missing_or_forged_queries_directions_mixture_and_projection() {
    let old = model(2, 1);
    let batch = [Record {
        tokens: vec![1],
        target: 1,
    }];
    let valid = old
        .forward_mini_step(&batch, None, &config(), limits())
        .unwrap();
    let reject = |candidate: &super::ForwardUpdate| {
        assert!(
            old.verify_forward_update(&batch, None, candidate, limits())
                .is_err()
        );
    };
    let mut bad = valid.clone();
    bad.estimate.cells.pop();
    reject(&bad);
    let mut bad = valid.clone();
    bad.estimate.cells[0].directions.pop();
    reject(&bad);
    let mut bad = valid.clone();
    bad.estimate.cells[0].component = 1;
    reject(&bad);
    let mut bad = valid.clone();
    bad.estimate.cells[0].directions[0].plus_loss += 0.01;
    reject(&bad);
    let mut bad = valid.clone();
    bad.estimate.cells[0].directions[0].step *= ratio(1, 2);
    reject(&bad);
    let mut bad = valid.clone();
    bad.estimate.cells[0].directions[0].pairing += ratio(1, 1);
    reject(&bad);
    let mut bad = valid.clone();
    bad.estimate.coordinates[0] += ratio(1, 1);
    reject(&bad);
    let mut bad = valid.clone();
    bad.estimate.mixed = old.clone();
    reject(&bad);
    let mut bad = valid.clone();
    bad.proposal[0] += ratio(1, 1);
    reject(&bad);
    let mut bad = valid.clone();
    bad.projected.certificates.pop();
    reject(&bad);
    let mut bad = valid.clone();
    bad.final_loss += 0.01;
    reject(&bad);
    let mut bad = valid.clone();
    bad.estimate.plan.records += 1;
    reject(&bad);
    assert!(old.parameters().iter().all(Rational::is_zero));
}

#[test]
fn forward_admission_counts_whole_family_and_refuses_all_incomplete_or_oversized_requests() {
    let model = model(3, 1);
    let batch = records();
    let cfg = config();
    let plan = ForwardPlan::new(&model, 3, true, limits()).unwrap();
    assert_eq!(plan.geometry_builds, 32);
    assert_eq!(plan.active_candidates, 32 * 4 * 16 + 192 * 16);
    assert_eq!(plan.max_directions, 768);
    assert_eq!(plan.batch_evaluations, 2 * 768 + 3);
    assert_eq!(plan.forward_passes, (2 * 768 + 3) * 3 * 2);
    assert!(plan.reserved_bytes > model.memory_plan().reserved_bytes);
    assert!(matches!(
        ForwardPlan::new(&model, usize::MAX, true, limits()),
        Err(Refusal::Overflow)
    ));
    for restricted in [
        ForwardLimits {
            max_records: 0,
            ..limits()
        },
        ForwardLimits {
            max_geometry_builds: 31,
            ..limits()
        },
        ForwardLimits {
            max_active_candidates: 5119,
            ..limits()
        },
        ForwardLimits {
            max_batch_evaluations: 1538,
            ..limits()
        },
        ForwardLimits {
            max_forward_passes: 0,
            ..limits()
        },
        ForwardLimits {
            max_forward_terms: 0,
            ..limits()
        },
        ForwardLimits {
            max_constraint_terms: 0,
            ..limits()
        },
        ForwardLimits {
            max_working_bytes: 1,
            ..limits()
        },
        ForwardLimits {
            geometry: GeometryLimits {
                max_normal_projections: 0,
                ..limits().geometry
            },
            ..limits()
        },
    ] {
        assert!(matches!(
            model.forward_mini_step(&batch, None, &cfg, restricted),
            Err(Refusal::Budget)
        ));
    }
    assert!(matches!(
        model.forward_mini_step(&[], None, &cfg, limits()),
        Err(Refusal::EmptyBatch)
    ));
    assert!(matches!(
        model.forward_mini_step(
            &[Record {
                tokens: vec![3],
                target: 0
            }],
            None,
            &cfg,
            limits()
        ),
        Err(Refusal::InvalidObject)
    ));
    assert!(matches!(
        model.forward_mini_step(
            &[Record {
                tokens: vec![0],
                target: 3
            }],
            None,
            &cfg,
            limits()
        ),
        Err(Refusal::InvalidObject)
    ));
    assert!(matches!(
        model.forward_mini_step(
            &[Record {
                tokens: vec![0, 0],
                target: 0
            }],
            None,
            &cfg,
            limits()
        ),
        Err(Refusal::ContextTooLong)
    ));
    let wrong = Renaming::new(vec![1, 0]).unwrap();
    assert!(matches!(
        model.forward_mini_step(&batch, Some(&wrong), &cfg, limits()),
        Err(Refusal::InvalidRenaming)
    ));
    let mut bad = cfg.clone();
    bad.rate = ratio(0, 1);
    assert!(matches!(
        model.forward_mini_step(&batch, None, &bad, limits()),
        Err(Refusal::LearningRate)
    ));
    for amount in [ratio(0, 1), ratio(2, 1)] {
        let mut bad = cfg.clone();
        bad.mixture = amount;
        assert!(matches!(
            model.forward_mini_step(&batch, None, &bad, limits()),
            Err(Refusal::QueryStep)
        ));
    }
    let mut bad = cfg;
    bad.query_cap = ratio(0, 1);
    assert!(matches!(
        model.forward_mini_step(&batch, None, &bad, limits()),
        Err(Refusal::QueryStep)
    ));
    assert!(model.parameters().iter().all(Rational::is_zero));
}
