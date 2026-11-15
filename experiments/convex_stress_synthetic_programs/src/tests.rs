use crate::{
    prefix::{Formula, Term},
    program::{Atom, Emission, Operator, Program},
    *,
};

#[test]
fn range_and_radix_operators_return_latent_states_without_output_labels() {
    let range = Program::Range {
        reverse: false,
        inclusive: true,
    };
    let mut prefix = vec![1, 102, 104, 18];
    for q in 2..=4 {
        assert_eq!(range.run(&prefix), Some(Atom::Quantity(q)));
        prefix.push(999);
    }
    assert_eq!(range.run(&prefix), Some(Atom::Role(0)));
    // Generic input operands012 and009. Emitted history values intentionally
    // disagree with reference answers; only causal history length is used.
    for reverse in [false, true] {
        let p = Program::Radix {
            operator: Operator::Sum,
            reverse,
            width_offset: 0,
            emission: Emission::Digits,
        };
        let mut x = vec![1, 26, 27, 28, 20, 26, 26, 35, 18];
        for q in if reverse { [1, 2, 0] } else { [0, 2, 1] } {
            assert_eq!(p.run(&x), Some(Atom::Quantity(q)));
            x.push(998);
        }
        assert_eq!(p.run(&x), Some(Atom::Role(0)));
    }
    assert_eq!(program::numeric_bank().len(), 302);
}

#[test]
fn hinted_radix_copy_is_input_derived_and_is_not_a_supplied_output_role() {
    let p = Program::Radix {
        operator: Operator::Sum,
        reverse: true,
        width_offset: 0,
        emission: Emission::HintDigit,
    };
    let mut x = vec![
        1, 110, 26, 111, 27, 112, 28, 20, 110, 26, 111, 26, 112, 35, 18,
    ];
    for atom in [
        Atom::Copy(112),
        Atom::Quantity(1),
        Atom::Copy(111),
        Atom::Quantity(2),
        Atom::Copy(110),
        Atom::Quantity(0),
        Atom::Role(0),
    ] {
        assert_eq!(p.run(&x), Some(atom));
        x.push(997);
    }
    assert!(p.run(&[1, 26, 20, 26, 18]).is_none());
}

#[test]
fn independent_prefix_grammar_uses_inclusive_counts_and_nested_causal_prefixes() -> Result<()> {
    let more_b = Formula::Less(
        Box::new(Term::Count(Box::new(Formula::Symbol(9)))),
        Box::new(Term::Count(Box::new(Formula::Symbol(10)))),
    );
    let word = [9, 10, 10, 23, 9, 10];
    let expected: Vec<_> = (1..=word.len())
        .map(|n| {
            word[..n].iter().filter(|&&x| x == 9).count()
                < word[..n].iter().filter(|&&x| x == 10).count()
        })
        .collect();
    assert_eq!(more_b.evaluate(&word)?, expected);
    let twice = Formula::Less(
        Box::new(Term::Constant(1)),
        Box::new(Term::Count(Box::new(more_b))),
    );
    let manual: Vec<_> = (1..=word.len())
        .map(|n| expected[..n].iter().filter(|&&x| x).count() > 1)
        .collect();
    assert_eq!(twice.evaluate(&word)?, manual);
    assert!(twice.evaluate(&[9, 999]).is_err());
    Ok(())
}
