//! Independent grammar cardinalities/ranks, two interpreters, and a complete
//! constant-control work forecast. No original training or heldout labels.
use super::*;
use family::{Input, Leaf};
use pairs::family::Program as Pair;

fn coefficient(x: i8) -> usize {
    [-2, -1, 1, 2].iter().position(|y| *y == x).unwrap()
}
fn input(x: Input) -> usize {
    match x {
        Input::A => 0,
        Input::B => 1,
        Input::Sum => 2,
        Input::Product => 3,
        Input::Distance => 4,
    }
}
fn leaf_rank(leaf: Leaf) -> usize {
    assert!([-1, 1].contains(&leaf.sign));
    let source = match leaf.source {
        Pair::Monomial { a, b } => {
            assert!((-3..=3).contains(&a) && (-3..=3).contains(&b));
            ((a + 3) * 7 + b + 3) as usize
        }
        Pair::Sum {
            left,
            right,
            coefficient: c,
        } => {
            let basis = [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]];
            let i = basis.iter().position(|x| *x == left).unwrap();
            let j = basis.iter().position(|x| *x == right).unwrap();
            assert!(i < j);
            49 + (i * (9 - i) / 2 + j - i - 1) * 4 + coefficient(c)
        }
        Pair::Integer(_) => panic!("integer primitive is outside the declared leaf grammar"),
    };
    2 * source + usize::from(leaf.sign == 1)
}
fn mark(seen: &mut [bool], index: usize) {
    assert!(!seen[index], "duplicate grammar descriptor {index}");
    seen[index] = true;
}

#[test]
fn complete_grammar_has_the_old_prefix_and_every_declared_cartesian_member() {
    let bank = programs();
    let old = wide::family::programs();
    assert_eq!(
        PROGRAMS,
        299_551 + 1_820 * 64 + 15 * 178 * 178 + 4_767 * 15 + 10
    );
    assert_eq!(bank.len(), PROGRAMS);
    for (a, b) in bank.iter().zip(old) {
        assert_eq!(*a, Program::Old(b));
    }
    let pairs: std::collections::BTreeMap<_, _> = pairs::family::programs()
        .iter()
        .enumerate()
        .map(|(i, p)| (serde_json::to_vec(p).unwrap(), i))
        .collect();
    assert_eq!(pairs.len(), 4_767);
    let mut four = vec![false; 16usize.pow(4) * 64];
    let mut branch = vec![false; 15 * 178 * 178];
    let mut recombine = vec![false; 4_767 * 15];
    let mut digit = [false; 10];
    for &program in &bank[wide::PROGRAMS..] {
        match program {
            Program::FourTerms {
                terms,
                coefficients,
            } => {
                let t = terms.map(|[a, b]| {
                    assert!((0..=3).contains(&a) && (0..=3).contains(&b));
                    (a * 4 + b) as usize
                });
                assert!(t.windows(2).all(|x| x[0] < x[1]));
                let c = coefficients.map(coefficient);
                let index = (((t[0] * 16 + t[1]) * 16 + t[2]) * 16 + t[3]) * 64
                    + c[0] * 16
                    + c[1] * 4
                    + c[2];
                mark(&mut four, index);
            }
            Program::Branch {
                control,
                modulus,
                threshold,
                low,
                high,
            } => {
                assert!([2, 3].contains(&modulus) && (1..modulus).contains(&threshold));
                let gate = if modulus == 2 {
                    0
                } else {
                    usize::from(threshold)
                };
                mark(
                    &mut branch,
                    ((input(control) * 3 + gate) * 178 + leaf_rank(low)) * 178 + leaf_rank(high),
                );
            }
            Program::Recombine {
                source,
                control,
                modulus,
            } => {
                assert!([2, 3, 4].contains(&modulus));
                let source = pairs[&serde_json::to_vec(&source).unwrap()];
                mark(
                    &mut recombine,
                    source * 15 + input(control) * 3 + usize::from(modulus) - 2,
                );
            }
            Program::Digit { source, high } => {
                mark(&mut digit, input(source) * 2 + usize::from(high))
            }
            Program::Old(_) => panic!("undeclared old descriptor outside the prefix"),
        }
    }
    assert_eq!(four.iter().filter(|x| **x).count(), 116_480);
    assert!(branch.into_iter().all(|x| x));
    assert!(recombine.into_iter().all(|x| x));
    assert!(digit.into_iter().all(|x| x));
}

#[test]
fn all_compositions_match_independent_interpretation_and_integer_boundaries() {
    let bank = programs();
    let evaluator = Evaluator::new(3).unwrap();
    for &tree in &bank {
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(
                    evaluator.evaluate(tree, a, b),
                    evaluator.reference(tree, a, b),
                    "{tree:?} at {a},{b}"
                );
            }
        }
    }
    for p in [17, 193, 1021] {
        let evaluator = Evaluator::new(p).unwrap();
        for i in 0..4096 {
            let tree = bank[(i * 7919) % bank.len()];
            for (a, b) in [(0, 0), (0, p - 1), (p - 1, 0), (p - 1, p - 1), (2, 3)] {
                assert_eq!(
                    evaluator.evaluate(tree, a, b),
                    evaluator.reference(tree, a, b)
                );
            }
        }
        for &tree in &bank[bank.len() - 10..] {
            for (a, b) in [(0, 0), (p - 1, p - 1), (p - 1, 1), (1, p - 1)] {
                assert_eq!(
                    evaluator.evaluate(tree, a, b),
                    evaluator.reference(tree, a, b)
                );
            }
        }
        let product_high = Program::Digit {
            source: Input::Product,
            high: true,
        };
        assert_eq!(evaluator.evaluate(product_high, p - 1, p - 1), p - 2);
        assert_eq!(evaluator.reference(product_high, p - 1, p - 1), p - 2);
    }
}

#[test]
fn full_constant_probe_fits_the_declared_work_budget_without_dropping_candidates() {
    let evaluator = Evaluator::new(193).unwrap();
    let bank = programs();
    let records = 18_528u128;
    let scalar = 16_384 * PROGRAMS as u128
        + 4_096 * records
        + 67_108_864
        + 1_024 * 196 * PROGRAMS as u128
        + records
            * bank
                .iter()
                .map(|p| 768 + evaluator.reference_cost(*p))
                .sum::<u128>();
    assert!(
        scalar <= MAX_WORK as u128,
        "complete constant probe needs {scalar} operations"
    );
    assert_eq!(bank.len(), PROGRAMS);
    eprintln!("complete p193 constant-probe forecast: {scalar} operations");
}
