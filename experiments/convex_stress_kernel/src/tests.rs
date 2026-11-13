use crate::*;
use discovery::{Input, Policy, Report, Request};
mod head;
pub(crate) fn request(
    n: usize,
    vocabulary: usize,
    prime: u16,
    observations: Vec<[usize; 2]>,
) -> Request {
    Request {
        prime,
        input: Input {
            n,
            vocabulary,
            observations,
            policy: Policy::default(),
        },
    }
}
pub(crate) fn diagonal() -> Result<Report> {
    discovery::run(request(3, 7, 7, vec![[0, 2], [4, 4]]))
}
/// CM34's binary alias grid plus an independent (2,2) coefficient direction.
pub(crate) fn fixture() -> Result<Report> {
    discovery::run(request(
        3,
        7,
        2,
        vec![[0, 0], [1, 1], [3, 1], [4, 2], [8, 1]],
    ))
}
/// CM30/CM34's eight observations leave(2,2) unseen, with two fitting raw aliases.
pub(crate) fn alias_fixture() -> Result<Report> {
    discovery::run(request(
        3,
        5,
        2,
        vec![
            [0, 0],
            [1, 3],
            [2, 0],
            [3, 2],
            [4, 1],
            [5, 2],
            [6, 0],
            [7, 3],
        ],
    ))
}
#[test]
fn actual_two_direction_kernel_retains_original_rows_and_refuses_false_scalar_matches() -> Result<()>
{
    let d = diagonal()?;
    discovery::replay(&d.request, &d)?;
    let code = d.code.as_ref().unwrap();
    let field = d.field.as_ref().unwrap();
    assert_eq!(field.problem.system.rows.len(), 4);
    assert_eq!(field.free.len(), 2);
    assert_eq!(d.basis_audit.as_ref().unwrap().dimension, 2);
    assert_eq!(d.work.observed_vector_checks, 2);
    assert_eq!(code.dimension, 2);
    let query = code.query(1, 0).unwrap();
    assert_ne!(query, code.label(2).unwrap());
    assert_ne!(query, code.label(4).unwrap());
    // RREF chooses column/class free coordinates, a different basis from the
    // Lean diagonal's row/column coordinates. Its second coordinate falsely
    // matches the true-diagonal class while the complete vectors disagree.
    assert!(
        query
            .iter()
            .zip(code.label(4).unwrap())
            .any(|(a, b)| a == b)
    );
    let e = states::Encoding::new(3, 7, code)?;
    assert_eq!(e.keys.len(), 4);
    assert_eq!(e.unmatched_active_cells, 2);
    assert_eq!(e.unknown_axis_cells, 5);
    assert!(!e.matched(1));
    assert!(e.matched(0) && e.matched(4));
    Ok(())
}
#[test]
fn sheared_complete_basis_preserves_all_queries_aliases_states_and_actual_logits() -> Result<()> {
    let d = fixture()?;
    let field = d.field.as_ref().unwrap();
    let mut basis = field.space.as_ref().unwrap().directions.clone();
    assert_eq!(basis.len(), 2);
    for k in 0..basis[0].len() {
        basis[1][k] = (basis[0][k] + basis[1][k]) % 2;
    }
    discovery::check_basis(&field.problem, field, &basis)?;
    let changed = discovery::Code::from_directions(d.names.clone(), 2, &basis)?;
    let old = d.code.as_ref().unwrap();
    assert_ne!(&changed, old);
    for i in 0..3 {
        for j in 0..3 {
            for &c in &d.names.classes {
                assert_eq!(
                    old.query(i, j).as_deref() == old.label(c),
                    changed.query(i, j).as_deref() == changed.label(c)
                );
            }
        }
    }
    for &a in &d.names.classes {
        for &b in &d.names.classes {
            assert_eq!(
                old.label(a) == old.label(b),
                changed.label(a) == changed.label(b)
            );
        }
    }
    let initial = initialize(&d)?;
    let mut alternate = initial.model.clone();
    alternate.encoding = states::Encoding::new(3, 7, &changed)?;
    assert_eq!(
        alternate.encoding.cell_blocks,
        initial.model.encoding.cell_blocks
    );
    assert_eq!(alternate.encoding.classes, initial.model.encoding.classes);
    alternate.validate()?;
    for cell in 0..9 {
        assert_eq!(alternate.logits_cell(cell), initial.model.logits_cell(cell));
    }
    Ok(())
}
#[test]
fn dropped_dependent_forged_and_wrong_gauge_directions_fail_completeness_checks() -> Result<()> {
    let d = diagonal()?;
    let field = d.field.as_ref().unwrap();
    let basis = &field.space.as_ref().unwrap().directions;
    let mut dependent = basis.clone();
    dependent[1] = dependent[0].clone();
    let mut wrong = basis.clone();
    wrong[0][0] = 1;
    for bad in [vec![], basis[..1].to_vec(), dependent, wrong] {
        assert!(discovery::check_basis(&field.problem, field, &bad).is_err());
    }
    for kind in 0..10 {
        let mut bad = d.clone();
        match kind {
            0 => {
                bad.field
                    .as_mut()
                    .unwrap()
                    .space
                    .as_mut()
                    .unwrap()
                    .directions
                    .pop();
            }
            1 => {
                bad.field.as_mut().unwrap().free.pop();
            }
            2 => bad.field.as_mut().unwrap().augmented[0][0] = 0,
            3 => bad.code.as_mut().unwrap().columns[0][0] = 1,
            4 => bad.basis_audit.as_mut().unwrap().complete = false,
            5 => bad.work.observed_vector_checks -= 1,
            6 => bad.names.classes.reverse(),
            7 => bad.raw_label_identification = true,
            8 => bad.target_readout_compatibility_proved = true,
            _ => bad.request.input.observations[0][1] = 5,
        }
        assert!(
            discovery::check(&d.request, &bad).is_err(),
            "mutation {kind}"
        );
    }
    Ok(())
}
#[test]
fn all_observations_are_required_and_every_refusal_is_retained() -> Result<()> {
    let original = diagonal()?.request;
    for kind in 0..4 {
        let mut request = original.clone();
        let reason = match kind {
            0 => {
                request.input.policy.equations = 1;
                "all_observed_equation_cap"
            }
            1 => {
                request.input.policy.elimination.nullity = 1;
                "nullity_cap"
            }
            2 => {
                request.input.policy.elimination.scalar_updates = 0;
                "scalar_update_cap"
            }
            _ => {
                request.input.policy.elimination.row_operations = 0;
                "row_operation_cap"
            }
        };
        let d = discovery::run(request.clone())?;
        assert_eq!(d.reason, reason);
        assert!(d.code.is_none());
        discovery::replay(&request, &d)?;
        let p = training::policies(2, 1)?;
        let b = binding::run(&d, p.clone())?;
        assert!(b.initialization.is_none() && b.initialization_refusal.is_some());
        assert!(
            b.endpoints
                .iter()
                .all(|e| e.training.is_none() && e.refusal.is_some())
        );
        binding::check(&d, &p, &b)?;
        let mut bad = d.clone();
        bad.reason = "complete_kernel_vector_code".into();
        assert!(discovery::check(&request, &bad).is_err());
    }
    Ok(())
}
#[test]
fn zero_dimension_is_a_complete_kernel_without_raw_identification() -> Result<()> {
    let d = discovery::run(request(3, 5, 3, vec![[0, 0], [1, 1], [3, 1], [4, 0]]))?;
    assert_eq!(d.code.as_ref().unwrap().dimension, 0);
    assert_eq!(d.basis_audit.as_ref().unwrap().dimension, 0);
    discovery::replay(&d.request, &d)?;
    let initial = initialize(&d)?;
    assert_eq!(initial.default_states, 1);
    assert_eq!(
        initial
            .model
            .audit(&d.request.input.observations)?
            .train_errors,
        0
    );
    assert!(!d.raw_label_identification && !d.target_readout_compatibility_proved);
    Ok(())
}
#[test]
fn categorical_names_do_not_supply_arithmetic_or_unknown_raw_classes() -> Result<()> {
    let d = alias_fixture()?;
    assert!(d.code.as_ref().unwrap().label(4).is_none());
    let mut request = d.request.clone();
    let rename = |x| (2 * x + 1) % 5;
    let permute = |x| (2 * x + 1) % 3;
    for sample in &mut request.input.observations {
        sample[0] = permute(sample[0] % 3) + 3 * permute(sample[0] / 3);
        sample[1] = rename(sample[1]);
    }
    let other = discovery::run(request)?;
    discovery::check(&other.request, &other)?;
    for i in 0..3 {
        for j in 0..3 {
            for &c in &d.names.classes {
                assert_eq!(
                    d.code.as_ref().unwrap().query(i, j).as_deref()
                        == d.code.as_ref().unwrap().label(c),
                    other
                        .code
                        .as_ref()
                        .unwrap()
                        .query(permute(i), permute(j))
                        .as_deref()
                        == other.code.as_ref().unwrap().label(rename(c))
                );
            }
        }
    }
    Ok(())
}
#[test]
fn invalid_requests_fail_before_matrix_construction() -> Result<()> {
    let d = diagonal()?;
    for kind in 0..6 {
        let mut r = d.request.clone();
        match kind {
            0 => r.input.n = usize::MAX,
            1 => r.prime = 4,
            2 => r.input.observations.push(r.input.observations[0]),
            3 => r.input.observations[0][1] = r.input.vocabulary,
            4 => r.input.policy.elimination.nullity = 9,
            _ => r.input.observations.clear(),
        }
        assert!(discovery::run(r).is_err());
    }
    Ok(())
}
