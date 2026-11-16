use crate::domain::DomainLimits;
use crate::geometry::{ConeGeometry, GeometryLimits};
use crate::model::{ComplexModel, ModelLimits, Output, Renaming};
use crate::{ProjectionLimits, Rational, Refusal, checked_dot};
use num_bigint::BigInt;
use num_traits::{One, Zero};

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

fn same_output(left: Output, right: Output) {
    assert_eq!(left.object(), right.object());
    assert_eq!(left.complex_scores().len(), right.complex_scores().len());
    assert_eq!(left.probabilities().len(), right.probabilities().len());
    for (a, b) in left.complex_scores().iter().zip(right.complex_scores()) {
        assert_eq!(
            (a.re.to_bits(), a.im.to_bits()),
            (b.re.to_bits(), b.im.to_bits())
        );
    }
    for (a, b) in left.probabilities().iter().zip(right.probabilities()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for target in 0..left.probabilities().len() {
        assert_eq!(
            left.cross_entropy(target).unwrap().to_bits(),
            right.cross_entropy(target).unwrap().to_bits()
        );
    }
}

fn reconstructed(
    old: &ComplexModel,
    first: usize,
    value: &[Rational],
) -> Result<ComplexModel, Refusal> {
    let mut parameters = old.parameters().to_vec();
    parameters[first..first + value.len()].clone_from_slice(value);
    old.with_parameters(parameters)
}

fn compare_query(old: &ComplexModel, first: usize, value: &[Rational], rename: &Renaming) {
    let query = old.component_query(first, value).unwrap();
    // Public full reconstruction is the slow reference: all rational cells
    // are checked and every complete complex neuron is recovered anew.
    let slow = reconstructed(old, first, value).unwrap();
    for tokens in old.domain().prefixes() {
        same_output(
            query.forward(tokens).unwrap(),
            slow.forward(tokens).unwrap(),
        );
        same_output(
            query.paired_forward(tokens, rename).unwrap(),
            slow.paired_forward(tokens, rename).unwrap(),
        );
    }
}

/// For one-position inputs, choose every original row's response directly,
/// then solve bias + lookup = response. This is an independent exact fixture.
fn vector(mask: usize, cell: usize, objects: usize) -> Vec<Rational> {
    let response = |row| {
        let magnitude = ratio(1 + ((cell + 3 * row) % 7) as i64, 128);
        if mask & (1usize << row) != 0 {
            magnitude
        } else {
            -magnitude
        }
    };
    let bias = response(0);
    std::iter::once(bias.clone())
        .chain((1..=objects).map(|row| response(row) - &bias))
        .collect()
}

#[test]
fn every_cell_matches_full_reconstruction_for_all_complex_outputs_and_classes() {
    let zero = model(3, 1);
    let plan = zero.domain().plan();
    assert_eq!((plan.patterns, plan.real_parameters), (16, 768));
    let mut parameters = Vec::new();
    for cell in 0..plan.real_parameters / plan.coordinates {
        parameters.extend(vector(cell / (plan.objects * 4), cell, plan.objects));
    }
    let old = zero.with_parameters(parameters).unwrap();
    let old_parameters = old.parameters().to_vec();
    let old_recovered = old.recovered().to_vec();
    let renaming = Renaming::new(vec![0, 2, 1]).unwrap();
    let mut checked = 0;
    for mask in 0..plan.patterns {
        for class in 0..plan.objects {
            for sign in 0..2 {
                for component in 0..2 {
                    let first = old
                        .parameter_index(mask, class, sign, component, 0)
                        .unwrap();
                    let boundary = vec![Rational::zero(); plan.coordinates];
                    let interior = vector(mask, first + 5, plan.objects);
                    for value in [&boundary, &interior] {
                        compare_query(&old, first, value, &renaming);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checked, 384);
    assert_eq!(old.parameters(), old_parameters);
    assert_eq!(old.recovered(), old_recovered);
}

#[test]
fn degenerate_nonorthogonal_queries_keep_all_rows_and_whole_neurons() {
    let zero = model(2, 2);
    let family = zero.domain().plan();
    assert_eq!((family.patterns, family.real_parameters), (128, 5120));
    let limits = GeometryLimits {
        max_normal_projections: 8,
        max_active_candidates: 4096,
        max_rational_entries: 10_000,
        max_working_bytes: 1 << 30,
    };
    let (mask, geometry) = (0..family.patterns)
        .find_map(|mask| {
            let g = ConeGeometry::new(
                zero.domain().cone(mask, zero.limits.projection).unwrap(),
                limits,
            )
            .unwrap();
            let nonorthogonal = g.basis().vectors.iter().enumerate().any(|(i, a)| {
                g.basis().vectors.iter().enumerate().any(|(j, b)| {
                    i != j && !checked_dot(a, b, zero.limits.projection).unwrap().is_zero()
                })
            });
            (g.certificate().forced.iter().any(|&forced| forced) && nonorthogonal)
                .then_some((mask, g))
        })
        .expect("full two-position domain has a nonorthogonal forced-equality cone");
    let mut parameters = zero.parameters().to_vec();
    for class in 0..2 {
        for sign in 0..2 {
            for component in 0..2 {
                let first = zero
                    .parameter_index(mask, class, sign, component, 0)
                    .unwrap();
                parameters[first..first + family.coordinates].clone_from_slice(geometry.centre());
            }
        }
    }
    let old = zero.with_parameters(parameters).unwrap();
    let renaming = Renaming::new(vec![1, 0]).unwrap();
    for direction in &geometry.basis().vectors {
        let query = geometry
            .symmetric_queries(geometry.centre(), direction, &ratio(1, 100))
            .unwrap();
        for class in 0..2 {
            for sign in 0..2 {
                for component in 0..2 {
                    let first = old
                        .parameter_index(mask, class, sign, component, 0)
                        .unwrap();
                    compare_query(&old, first, &query.plus, &renaming);
                    compare_query(&old, first, &query.minus, &renaming);
                }
            }
        }
    }
}

#[test]
fn malformed_cells_and_input_do_not_bypass_query_validation() {
    let old = model(3, 1);
    let zero = vec![Rational::zero(); 4];
    for first in [1, 767, 768, usize::MAX] {
        assert!(matches!(
            old.component_query(first, &zero),
            Err(Refusal::Shape)
        ));
    }
    for value in [vec![], vec![Rational::zero(); 3], vec![Rational::zero(); 5]] {
        assert!(matches!(
            old.component_query(0, &value),
            Err(Refusal::Shape)
        ));
    }
    let query = old.component_query(0, &zero).unwrap();
    assert_eq!(query.forward(&[3]), Err(Refusal::InvalidObject));
    assert_eq!(query.forward(&[0, 0]), Err(Refusal::ContextTooLong));
    assert_eq!(
        query.forward(&[]).unwrap().cross_entropy(3),
        Err(Refusal::InvalidObject)
    );
    assert_eq!(
        query.paired_forward(&[0], &Renaming::new(vec![0, 1]).unwrap()),
        Err(Refusal::InvalidRenaming)
    );
}

#[test]
fn every_row_and_original_arithmetic_and_numeric_refusal_remain_enforced() {
    let old = model(3, 1);
    let first = old.parameter_index(15, 1, 0, 0, 0).unwrap();
    let tiny = Rational::new(BigInt::one(), BigInt::one() << 1100);
    let huge = Rational::from_integer(BigInt::one() << 2048);
    let over_budget = Rational::from_integer(BigInt::one() << 4097);
    let cases = [
        (
            vec![ratio(0, 1), ratio(0, 1), ratio(0, 1), ratio(-1, 1)],
            Refusal::Infeasible,
        ),
        (
            vec![tiny, ratio(0, 1), ratio(0, 1), ratio(0, 1)],
            Refusal::NumericalUnderflow,
        ),
        (
            vec![huge, ratio(0, 1), ratio(0, 1), ratio(0, 1)],
            Refusal::NonFinite,
        ),
        (
            vec![over_budget, ratio(0, 1), ratio(0, 1), ratio(0, 1)],
            Refusal::Budget,
        ),
    ];
    for (value, expected) in cases {
        assert_eq!(
            old.component_query(first, &value).err(),
            Some(expected.clone())
        );
        assert_eq!(reconstructed(&old, first, &value).err(), Some(expected));
    }
    let value = vec![ratio(1000, 1), ratio(0, 1), ratio(0, 1), ratio(0, 1)];
    let query = old.component_query(first, &value).unwrap();
    assert_eq!(query.forward(&[]), Err(Refusal::NumericalUnderflow));
    assert_eq!(
        query.forward(&[]),
        reconstructed(&old, first, &value).unwrap().forward(&[])
    );
    assert!(old.parameters().iter().all(Rational::is_zero));
}
