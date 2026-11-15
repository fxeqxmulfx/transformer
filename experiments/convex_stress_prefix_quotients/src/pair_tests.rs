use crate::*;
use std::collections::{BTreeMap, BTreeSet};
#[test]
fn indexed_pairs_match_independent_exhaustive_evaluation_with_tail_bits() {
    let mut rows = vec![];
    for n in 1..=2u32 {
        for mut code in 0..3usize.pow(n) {
            let mut word = vec![1];
            for _ in 0..n {
                word.push([9, 10, 23][code % 3]);
                code /= 3;
            }
            let a = word.iter().filter(|&&x| x == 9).count();
            let b = word.iter().filter(|&&x| x == 10).count();
            rows.push(Observation {
                id: rows.len(),
                tokens: word,
                label: if a < b { 19 } else { 7 },
            });
        }
    }
    let pool = pool::Pool::new(&rows).unwrap();
    let base = bank::Bank::initial(&pool, 8).unwrap();
    let expanded = bank::Bank::expanded(&pool, &base, 32768).unwrap();
    assert!(expanded.refusal.is_none() && expanded.bank.rows.len() >= 73);
    let mut left = bank::Bank::new();
    let mut right = bank::Bank::new();
    for (i, c) in expanded.bank.rows.iter().take(73).enumerate() {
        right
            .add(&pool, &c.bits, c.nodes, 73, || c.formula.clone())
            .unwrap();
        if i < 17 {
            left.add(&pool, &c.bits, c.nodes, 17, || c.formula.clone())
                .unwrap();
        }
    }
    assert_eq!(right.rows.len(), 73);
    let observed: BTreeMap<_, _> = pool
        .row_nodes
        .iter()
        .zip(&rows)
        .map(|(&i, r)| (i, r.label == 19))
        .collect();
    let extras: Vec<_> = (0..3)
        .map(|s| pool.counts(&pool.signature(&grammar::literal(s)).unwrap()))
        .collect();
    let mut matching_total = 0;
    for extra in [None, Some(0)] {
        for offset in [-2, 0, 1, 2] {
            for negated in [false, true] {
                for gate in [None, Some(4)] {
                    let c = transitions::Config {
                        extra,
                        offset,
                        negated,
                        gate,
                        flipped: false,
                    };
                    let mut expected = BTreeSet::new();
                    let mut count = 0u64;
                    let mut bounds_meter = latent::Meter {
                        work: 0,
                        limit: 40_000_000_000_000,
                    };
                    let right_bounds = count_bounds::partners(
                        &pool,
                        &observed,
                        &left,
                        &c,
                        &extras,
                        true,
                        &mut bounds_meter,
                    )
                    .unwrap();
                    let left_bounds = count_bounds::partners(
                        &pool,
                        &observed,
                        &right,
                        &c,
                        &extras,
                        false,
                        &mut bounds_meter,
                    )
                    .unwrap();
                    for a in &left.rows {
                        for b in &right.rows {
                            let f =
                                grammar::make(&a.formula, &b.formula, extra, offset, negated, gate);
                            let p = Program::Prefix(f.clone());
                            if rows.iter().all(|r| {
                                p.interpret(&r.tokens)
                                    == Some(prior::program::Atom::Quantity(i64::from(
                                        r.label == 19,
                                    )))
                            }) {
                                assert!(
                                    a.counts
                                        .iter()
                                        .enumerate()
                                        .all(|(i, &v)| left_bounds.allows(i, v))
                                );
                                assert!(
                                    b.counts
                                        .iter()
                                        .enumerate()
                                        .all(|(i, &v)| right_bounds.allows(i, v))
                                );
                                count += 1;
                                expected.insert((f.nodes(), f));
                            }
                        }
                    }
                    let mut meter = latent::Meter {
                        work: 0,
                        limit: 40_000_000_000_000,
                    };
                    let (actual, stats) = pairs::run(
                        &pool,
                        &observed,
                        (&left, &right),
                        &c,
                        &extras,
                        32,
                        &mut meter,
                    )
                    .unwrap();
                    assert!(!stats.work_exhausted);
                    assert_eq!(stats.matching_pairs, count);
                    assert_eq!(
                        actual,
                        expected
                            .into_iter()
                            .take(32)
                            .map(|(_, f)| f)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(stats.possible_pairs, 17 * 73);
                    matching_total += count;
                }
            }
        }
    }
    assert!(
        matching_total > 0,
        "independent comparison must exercise actual matching pairs"
    );
    let c = transitions::Config {
        extra: None,
        offset: 0,
        negated: false,
        gate: None,
        flipped: false,
    };
    let mut meter = latent::Meter { work: 0, limit: 0 };
    let (actual, stats) = pairs::run(
        &pool,
        &observed,
        (&left, &right),
        &c,
        &extras,
        32,
        &mut meter,
    )
    .unwrap();
    assert!(actual.is_empty() && stats.work_exhausted);
    assert_eq!(meter.work, 0);
}

#[test]
fn truncated_child_banks_can_supply_fully_checked_training_fits() {
    let target = grammar::make(
        &grammar::literal(0),
        &grammar::literal(1),
        None,
        0,
        false,
        None,
    );
    let mut rows = vec![];
    for mut code in 0..27usize {
        let word: Vec<_> = (0..3)
            .map(|_| {
                let s = [9, 10, 23][code % 3];
                code /= 3;
                s
            })
            .collect();
        let truth = target.evaluate(&word).unwrap();
        for (i, b) in truth.into_iter().enumerate() {
            rows.push(Observation {
                id: rows.len(),
                tokens: [vec![1], word[..=i].to_vec()].concat(),
                label: if b { 19 } else { 7 },
            });
        }
    }
    let pool = pool::Pool::new(&rows).unwrap();
    let children = bank::Bank::initial(&pool, 8).unwrap();
    let report = deep::run(&pool, &rows, &children, 8, 32, 0).unwrap();
    assert!(!report.programs.is_empty(), "{:?}", report.refused);
    assert!(
        report
            .attempts
            .iter()
            .any(|a| a.left.capped || a.right.as_ref().is_some_and(|r| r.capped))
    );
    for p in report.programs {
        let mut decoder = BTreeMap::new();
        for r in &rows {
            let atom = p.interpret(&r.tokens).unwrap();
            assert_eq!(Some(atom), p.run(&r.tokens));
            if let Some(old) = decoder.insert(atom, r.label) {
                assert_eq!(old, r.label);
            }
        }
    }
}

#[test]
fn bounded_latent_bank_matches_independent_filter_of_complete_signature_bank() {
    let mut rows = vec![];
    for mut code in 0..27usize {
        let word: Vec<_> = (0..3)
            .map(|_| {
                let s = [9, 10, 23][code % 3];
                code /= 3;
                s
            })
            .collect();
        for i in 0..3 {
            let p = &word[..=i];
            let a = p.iter().filter(|&&s| s == 9).count();
            let b = p.iter().filter(|&&s| s == 10).count();
            rows.push(Observation {
                id: rows.len(),
                tokens: [vec![1], p.to_vec()].concat(),
                label: if a < b { 19 } else { 7 },
            });
        }
    }
    let pool = pool::Pool::new(&rows).unwrap();
    let children = bank::Bank::initial(&pool, 8).unwrap();
    let full = bank::Bank::expanded(&pool, &children, 32768).unwrap();
    assert!(full.refusal.is_none());
    let observed = pool
        .row_nodes
        .iter()
        .zip(&rows)
        .map(|(&i, r)| (i, r.label == 19))
        .collect();
    let extras: Vec<_> = (0..3)
        .map(|s| pool.counts(&pool.signature(&grammar::literal(s)).unwrap()))
        .collect();
    let c = transitions::Config {
        extra: None,
        offset: 0,
        negated: false,
        gate: None,
        flipped: false,
    };
    let mut meter = latent::Meter {
        work: 0,
        limit: 40_000_000_000_000,
    };
    let bounds =
        count_bounds::partners(&pool, &observed, &children, &c, &extras, true, &mut meter).unwrap();
    let expected: BTreeSet<_> = full
        .bank
        .rows
        .iter()
        .filter(|candidate| {
            pool.nodes.iter().enumerate().all(|(i, _)| {
                let count = candidate
                    .formula
                    .evaluate(&pool.word(i))
                    .unwrap()
                    .into_iter()
                    .filter(|b| *b)
                    .count() as i16;
                bounds.allows(i, count)
            })
        })
        .map(|c| c.bits.clone())
        .collect();
    assert!(!expected.is_empty() && expected.len() < full.bank.rows.len());
    let (bounded, stats) =
        latent::synthesize_bounded(&pool, &children, &vec![], 32768, &mut meter, Some(&bounds))
            .unwrap();
    assert!(!stats.capped && !stats.work_exhausted && stats.count_bound_rejections > 0);
    assert_eq!(
        bounded
            .rows
            .into_iter()
            .map(|c| c.bits)
            .collect::<BTreeSet<_>>(),
        expected
    );
}
