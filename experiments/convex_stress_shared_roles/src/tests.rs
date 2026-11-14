//! CM40 witnesses use actual small finite systems and native complex states.
//! Exhaustive coordinate enumeration is independent of elimination and of the
//! shared-role row builder. No original held-out examples are used here.
use crate::*;
use std::collections::BTreeSet;
mod modes;
mod mutations;

fn diagonal(prime: u16) -> discovery::Request {
    discovery::Request {
        input: kernel::discovery::Input {
            n: 3,
            vocabulary: usize::from(prime),
            observations: vec![[0, 0], [4, 0]],
            policy: Default::default(),
        },
        readout: (0..usize::from(prime)).collect(),
        answers: usize::from(prime),
        prime,
    }
}
fn records(request: &discovery::Request) -> Vec<records::Record> {
    request
        .input
        .observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record { id, cell, label })
        .collect()
}
fn digits(mut value: usize, p: usize, length: usize) -> Vec<u16> {
    (0..length)
        .map(|_| {
            let digit = (value % p) as u16;
            value /= p;
            digit
        })
        .collect()
}
#[test]
fn all_small_diagonal_kernels_equal_the_independent_enumeration() -> Result<()> {
    let mut checked = 0;
    for p in [2_u16, 3] {
        for a in 0..usize::from(p) {
            for b in 0..usize::from(p) {
                let mut request = diagonal(p);
                request.input.observations[0][1] = a;
                request.input.observations[1][1] = b;
                let out = discovery::run(request.clone())?;
                discovery::replay(&request, &out)?;
                let field = out.field.as_ref().unwrap();
                let basis = &field.space.as_ref().unwrap().directions;
                let variables = if a == b { 5 } else { 6 };
                let modulus = usize::from(p);
                let direct: BTreeSet<_> = (0..modulus.pow(variables as u32))
                    .map(|value| digits(value, modulus, variables))
                    .filter(|v| {
                        let x: Vec<_> = v.iter().map(|&x| i32::from(x)).collect();
                        let m = i32::from(p);
                        x[0] == 0
                            && x[4] == 0
                            && (x[0] + x[2] - x[4]).rem_euclid(m) == 0
                            && (x[1] + x[3] - x[if a == b { 4 } else { 5 }]).rem_euclid(m) == 0
                            && (x[0] - x[2] - x[1] + x[3]).rem_euclid(m) == 0
                    })
                    .collect();
                let span: BTreeSet<_> = (0..modulus.pow(basis.len() as u32))
                    .map(|value| {
                        let scalars = digits(value, modulus, basis.len());
                        (0..variables)
                            .map(|k| {
                                (basis
                                    .iter()
                                    .zip(&scalars)
                                    .map(|(v, &s)| usize::from(v[k]) * usize::from(s))
                                    .sum::<usize>()
                                    % modulus) as u16
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                assert_eq!(span, direct);
                let code = out.grouped_code.as_ref().unwrap();
                for i in 0..2 {
                    for j in 0..2 {
                        for (c, label) in code.names.classes.iter().enumerate() {
                            let forced = direct.iter().all(|v| {
                                (i32::from(v[i]) + i32::from(v[2 + j]) - i32::from(v[4 + c]))
                                    .rem_euclid(i32::from(p))
                                    == 0
                            });
                            assert_eq!(code.query(i, j).as_deref() == code.label(*label), forced);
                        }
                    }
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 13);
    Ok(())
}
#[test]
fn strict_new_coverage_and_characteristic_two_refusal() -> Result<()> {
    let request = diagonal(3);
    let old = kernel::discovery::run(kernel::discovery::Request {
        input: request.input.clone(),
        prime: request.prime,
    })?;
    let old = old.code.unwrap();
    assert_ne!(old.query(0, 1).as_deref(), old.label(0));
    let shared = discovery::run(request)?;
    let code = shared.raw_code.as_ref().unwrap();
    assert_eq!(code.query(0, 1).as_deref(), code.label(0));
    // Lean's independent-role world fits these observations but has cross
    // answer -1. Support is conditional identification, not an unconditional
    // target certificate. This arithmetic is only the counterexample fixture.
    let independent = |i: usize, j: usize| (i + 3 - j) % 3;
    assert_eq!(independent(0, 0), 0);
    assert_eq!(independent(1, 1), 0);
    let model = construct::initialize(&shared)?.model;
    let logits = model.logits_state(model.location(3).0, model.location(3).1);
    assert!(model.known(3) && logits[0] > logits[independent(0, 1)]);
    let two = discovery::run(diagonal(2))?;
    let code = two.raw_code.as_ref().unwrap();
    assert_ne!(code.query(0, 1).as_deref(), code.label(0));
    assert_eq!(code.query(2, 1), None);
    let mut inconsistent = diagonal(2);
    inconsistent.input.observations[1][1] = 1;
    let refused = discovery::run(inconsistent.clone())?;
    assert_eq!(refused.reason, "incompatible_shared_group_codes");
    assert!(refused.raw_code.is_none());
    discovery::check(&inconsistent, &refused)?;
    Ok(())
}
#[test]
fn free_common_offset_and_complete_basis_changes_preserve_support() -> Result<()> {
    let mut request = diagonal(3);
    request.input.observations = vec![[3, 0], [1, 0]];
    let out = discovery::run(request.clone())?;
    let code = out.grouped_code.as_ref().unwrap();
    assert_eq!(code.dimension, 1);
    let offsets: Vec<_> = out
        .alignments
        .iter()
        .map(|&[i, j]| (i32::from(code.rows[i][0]) - i32::from(code.columns[j][0])).rem_euclid(3))
        .collect();
    assert!(offsets[0] != 0 && offsets.iter().all(|&x| x == offsets[0]));
    request.input.observations = vec![[0, 0], [4, 1], [8, 2]];
    let out = discovery::run(request.clone())?;
    let field = out.field.as_ref().unwrap();
    let basis = &field.space.as_ref().unwrap().directions;
    assert_eq!(basis.len(), 2);
    let mut alternate = basis.clone();
    for (x, &y) in alternate[0].iter_mut().zip(&basis[1]) {
        *x = (*x + y) % 3;
    }
    kernel::discovery::check_basis(&discovery::problem(&request)?, field, &alternate)?;
    let original = out.raw_code.as_ref().unwrap();
    let mut changed = original.clone();
    for v in changed
        .rows
        .iter_mut()
        .chain(&mut changed.columns)
        .chain(&mut changed.classes)
    {
        v[0] = (v[0] + v[1]) % 3;
    }
    let before = kernel::states::Encoding::new(3, 3, original)?;
    let after = kernel::states::Encoding::new(3, 3, &changed)?;
    assert_eq!(before.cell_blocks, after.cell_blocks);
    assert_eq!(before.classes, after.classes);
    assert_eq!(before.unmatched_active_cells, after.unmatched_active_cells);
    // Reordering observations changes the first row/class gauge anchors.
    request.input.observations.rotate_left(1);
    let shifted = discovery::run(request)?;
    let shifted = shifted.raw_code.unwrap();
    for i in 0..3 {
        for j in 0..3 {
            for label in 0..3 {
                assert_eq!(
                    original.query(i, j).as_deref() == original.label(label),
                    shifted.query(i, j).as_deref() == shifted.label(label)
                );
            }
        }
    }
    Ok(())
}
