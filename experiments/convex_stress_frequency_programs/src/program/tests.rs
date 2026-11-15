use super::*;

#[test]
fn complete_canonical_bank_matches_independent_rendering_and_tie_order() -> Result<()> {
    let roles = Roles {
        vocabulary: 16,
        delimiters: vec![2],
        minimal_prefixes: 1,
        scalar_work_forecast: 0,
    };
    let bank = catalogue(&roles)?;
    assert_eq!(bank.len(), 6240);
    assert!(bank.windows(2).all(|w| w[0] < w[1]));
    for p in &bank {
        for word in [
            &[7, 4, 7, 5, 4][..],
            &[5, 7, 4, 7, 7, 4][..],
            &[3, 3, 3][..],
        ] {
            for rank in 0..=2 * word.len() + 4 {
                let tokens: Vec<_> = std::iter::once(1)
                    .chain(word.iter().copied())
                    .chain(std::iter::once(2))
                    .chain(std::iter::repeat_n(9, rank))
                    .collect();
                assert_eq!(
                    p.run(&tokens).atom,
                    p.interpret(&tokens),
                    "{p:?}, {tokens:?}"
                );
            }
        }
    }
    let mut p = bank[0].clone();
    p.skip = 1;
    p.basis = Basis::Distinct;
    p.emit = Emit::Copy;
    p.order = Order::Descending;
    assert_eq!(p.run(&[1, 7, 4, 7, 5, 4, 2]).atom, Atom::Copy(7));
    assert_eq!(p.run(&[1, 7, 4, 7, 5, 4, 2, 9]).atom, Atom::Copy(4));
    p.order = Order::Ascending;
    assert_eq!(p.run(&[1, 7, 4, 7, 5, 4, 2]).atom, Atom::Copy(5));
    assert_eq!(p.run(&[1, 7, 4, 7, 5, 4, 2, 9]).atom, Atom::Copy(7));
    Ok(())
}

#[test]
fn delimiter_roles_use_only_minimal_training_prefixes() -> Result<()> {
    let mut input = Input {
        vocabulary: 16,
        context: 20,
        observations: vec![
            Observation {
                tokens: vec![1, 7, 4, 2],
                label: 9,
            },
            Observation {
                tokens: vec![1, 7, 4, 2, 9],
                label: 3,
            },
            Observation {
                tokens: vec![1, 5, 7, 2],
                label: 8,
            },
        ],
    };
    let roles = infer(&input)?;
    assert_eq!(roles.delimiters, vec![2]);
    assert_eq!(roles.minimal_prefixes, 2);
    for row in &mut input.observations {
        row.label = 15;
    }
    assert_eq!(infer(&input)?, roles);
    let p = &catalogue(&roles)?[0];
    assert_eq!(p.run(&[1, 7, 4]).atom, Atom::Role(2));
    assert_eq!(p.interpret(&[1, 7, 4]), Atom::Role(2));
    Ok(())
}
