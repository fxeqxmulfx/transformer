use super::{Complex64, ComplexModel, ModelLimits, PhysicalNeuron, Renaming};
use crate::domain::DomainLimits;
use crate::{ProjectionLimits, Rational, Refusal};
use num_bigint::BigInt;
use num_traits::{One, Zero};

fn integer(value: i64) -> Rational {
    Rational::from_integer(BigInt::from(value))
}
fn vector(values: &[i64]) -> Vec<Rational> {
    values.iter().map(|&x| integer(x)).collect()
}
fn limits() -> ModelLimits {
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
        max_working_bytes: 1024 * 1024 * 1024,
    }
}
fn close(left: Complex64, right: Complex64) {
    assert!((left.re - right.re).abs() < 1e-11, "{left:?} != {right:?}");
    assert!((left.im - right.im).abs() < 1e-11, "{left:?} != {right:?}");
}

#[test]
fn zero_model_keeps_the_entire_array_and_original_object_output() {
    let model = ComplexModel::zero(3, 1, limits()).unwrap();
    assert_eq!(model.parameters().len(), 768);
    assert_eq!(model.recovered().len(), 96);
    for tokens in model.domain().prefixes() {
        let output = model.forward(tokens).unwrap();
        assert_eq!(output.object(), 0);
        assert_eq!(output.complex_scores(), &[Complex64::new(0.0, 0.0); 3]);
        assert_eq!(output.probabilities().len(), 3);
        for target in 0..3 {
            assert!((output.probabilities()[target] - 1.0 / 3.0).abs() < 1e-15);
            assert!((output.cross_entropy(target).unwrap() - 3.0_f64.ln()).abs() < 1e-15);
        }
    }
}

#[test]
fn cm76_different_component_masks_preserve_whole_complex_predictions() {
    let base = ComplexModel::zero(2, 1, limits()).unwrap();
    let model = base
        .with_physical(&[
            PhysicalNeuron {
                real: vector(&[0, -2, 5]),
                imaginary: vector(&[0, 7, -6]),
                class_readouts: vector(&[0, 1]),
            },
            PhysicalNeuron {
                real: vector(&[0, -1, 2]),
                imaginary: vector(&[0, -4, 3]),
                class_readouts: vector(&[0, 1]),
            },
        ])
        .unwrap();
    close(
        model.forward(&[0]).unwrap().complex_scores()[1],
        Complex64::new(-7.0, 7.0),
    );
    close(
        model.forward(&[1]).unwrap().complex_scores()[1],
        Complex64::new(4.0, 10.0),
    );
    assert_eq!(model.forward(&[0]).unwrap().object(), 0);
    assert_eq!(model.forward(&[1]).unwrap().object(), 1);
    assert!(
        model
            .recovered()
            .iter()
            .any(|n| n.representation.iter().any(|w| w.im != 0.0))
    );
    assert!(model.recovered().iter().any(|n| n.amplitude != 0.0));
    assert_eq!(base.parameters().len(), model.parameters().len());
}

#[test]
fn signed_class_readouts_and_zero_prefix_remain_in_the_actual_forward() {
    let base = ComplexModel::zero(2, 1, limits()).unwrap();
    let model = base
        .with_physical(&[PhysicalNeuron {
            real: vector(&[2, 0, 0]),
            imaginary: vector(&[1, 0, 0]),
            class_readouts: vector(&[-2, 3]),
        }])
        .unwrap();
    for tokens in model.domain().prefixes() {
        let output = model.forward(tokens).unwrap();
        close(output.complex_scores()[0], Complex64::new(-2.0, -6.0));
        close(output.complex_scores()[1], Complex64::new(3.0, 9.0));
        assert!((output.cross_entropy(1).unwrap() - (-5.0_f64).exp().ln_1p()).abs() < 1e-13);
    }
}

fn paired_witness() -> ComplexModel {
    ComplexModel::zero(3, 1, limits())
        .unwrap()
        .with_physical(&[
            PhysicalNeuron {
                real: vector(&[0, 0, 2, 0]),
                imaginary: vector(&[0, 0, 1, 0]),
                class_readouts: vector(&[0, 4, 0]),
            },
            PhysicalNeuron {
                real: vector(&[0, 2, 0, 0]),
                imaginary: vector(&[0, 1, 0, 0]),
                class_readouts: vector(&[2, 0, 0]),
            },
        ])
        .unwrap()
}

#[test]
fn cm78_feasible_witness_returns_original_seen_control_and_unseen_objects() {
    let model = paired_witness();
    let rename = Renaming::new(vec![0, 2, 1]).unwrap();
    let expected_loss = (2.0 * (-2.0_f64).exp()).ln_1p();
    for object in 0..3 {
        let output = model.paired_forward(&[object], &rename).unwrap();
        assert_eq!(output.object(), object);
        close(output.complex_scores()[object], Complex64::new(2.0, 6.0));
        assert!((output.cross_entropy(object).unwrap() - expected_loss).abs() < 1e-13);
        assert!(output.cross_entropy(object).unwrap() < 2.0_f64.ln());
    }
}

#[test]
fn paired_forward_checks_whole_complex_equivariance_and_all_class_ce() {
    let model = paired_witness();
    let rename = Renaming::new(vec![0, 2, 1]).unwrap();
    for tokens in model.domain().prefixes() {
        let renamed: Vec<_> = tokens.iter().map(|&x| rename.permutation()[x]).collect();
        let first = model.paired_forward(tokens, &rename).unwrap();
        let second = model.paired_forward(&renamed, &rename).unwrap();
        for class in 0..3 {
            close(
                first.complex_scores()[class],
                second.complex_scores()[rename.permutation()[class]],
            );
            assert!(
                (first.cross_entropy(class).unwrap()
                    - second.cross_entropy(rename.permutation()[class]).unwrap())
                .abs()
                    < 1e-13
            );
        }
    }
}

#[test]
fn full_model_projection_keeps_all_cells_and_replays_every_certificate() {
    let old = ComplexModel::zero(3, 1, limits()).unwrap();
    let proposal: Vec<_> = (0..768).map(|i| integer((i % 7) as i64 - 3)).collect();
    let updated = old.project_parameters(&proposal).unwrap();
    assert_eq!(updated.model.parameters().len(), old.parameters().len());
    assert_eq!(updated.certificates.len(), 192);
    assert!(
        updated
            .certificates
            .iter()
            .all(|c| c.point.len() == 4 && c.multipliers.len() == 4)
    );
    updated
        .model
        .verify_projection(&proposal, &updated.certificates)
        .unwrap();
    assert!(old.parameters().iter().all(Rational::is_zero));
    assert!(updated.model.parameters().iter().any(|x| !x.is_zero()));
    assert!(updated.model.recovered().iter().any(|n| n.amplitude > 0.0));
    let mut forged = updated.certificates.clone();
    forged.pop();
    assert_eq!(
        updated.model.verify_projection(&proposal, &forged),
        Err(Refusal::Shape)
    );
    let mut forged = updated.certificates.clone();
    forged[0].point[0] += Rational::one();
    assert!(updated.model.verify_projection(&proposal, &forged).is_err());
}

#[test]
fn invalid_array_or_projection_does_not_mutate_the_old_model() {
    let old = paired_witness();
    let parameters = old.parameters().to_vec();
    let before = old.forward(&[1]).unwrap();
    let mut infeasible = parameters.clone();
    let index = old.parameter_index(0, 0, 0, 0, 0).unwrap();
    infeasible[index] = integer(1);
    assert!(matches!(
        old.with_parameters(infeasible),
        Err(Refusal::Infeasible)
    ));
    assert!(matches!(old.project_parameters(&[]), Err(Refusal::Shape)));
    assert_eq!(old.parameters(), &parameters);
    assert_eq!(old.forward(&[1]).unwrap(), before);
}

#[test]
fn model_resource_and_numerical_failures_are_explicit() {
    let limited = ModelLimits {
        max_working_bytes: 1,
        ..limits()
    };
    assert!(matches!(
        ComplexModel::zero(3, 1, limited),
        Err(Refusal::Budget)
    ));
    let base = ComplexModel::zero(2, 1, limits()).unwrap();
    let mut physical = PhysicalNeuron {
        real: vector(&[1, 0, 0]),
        imaginary: vector(&[0, 0, 0]),
        class_readouts: vector(&[0, 1]),
    };
    physical.real[0] = Rational::new(BigInt::one(), BigInt::from(10).pow(400));
    assert!(matches!(
        base.with_physical(&[physical.clone()]),
        Err(Refusal::NumericalUnderflow)
    ));
    physical.real[0] = Rational::from_integer(BigInt::from(10).pow(400));
    assert!(matches!(
        base.with_physical(&[physical.clone()]),
        Err(Refusal::NonFinite)
    ));
    physical.real[0] = integer(1000);
    let model = base.with_physical(&[physical]).unwrap();
    assert!(matches!(
        model.forward(&[]),
        Err(Refusal::NumericalUnderflow)
    ));
}

#[test]
fn invalid_objects_context_targets_and_renamings_are_refused() {
    let model = ComplexModel::zero(3, 1, limits()).unwrap();
    assert!(matches!(model.forward(&[3]), Err(Refusal::InvalidObject)));
    assert!(matches!(
        model.forward(&[0, 1]),
        Err(Refusal::ContextTooLong)
    ));
    assert_eq!(
        model.forward(&[]).unwrap().cross_entropy(3),
        Err(Refusal::InvalidObject)
    );
    assert!(matches!(
        Renaming::new(vec![1, 2, 0]),
        Err(Refusal::InvalidRenaming)
    ));
    assert!(matches!(
        Renaming::new(vec![0, 0, 2]),
        Err(Refusal::InvalidRenaming)
    ));
    assert!(matches!(
        Renaming::new(vec![0, 3, 1]),
        Err(Refusal::InvalidRenaming)
    ));
    let wrong = Renaming::new(vec![1, 0]).unwrap();
    assert!(matches!(
        model.paired_forward(&[], &wrong),
        Err(Refusal::InvalidRenaming)
    ));
}
