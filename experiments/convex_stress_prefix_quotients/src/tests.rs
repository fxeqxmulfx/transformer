use crate::*;
fn rows(f: &Formula, length: usize) -> Vec<Observation> {
    let mut out = vec![];
    for n in 1..=length {
        for mut code in 0..3usize.pow(n as u32) {
            let word: Vec<_> = (0..n)
                .map(|_| {
                    let t = [9, 10, 23][code % 3];
                    code /= 3;
                    t
                })
                .collect();
            let label = if *f.evaluate(&word).unwrap().last().unwrap() {
                19
            } else {
                7
            };
            out.push(Observation {
                id: out.len(),
                tokens: [vec![1], word].concat(),
                label,
            });
        }
    }
    out
}
#[test]
fn closure_missing_prefixes_duplicates_and_count_cache_match_full_interpreter() {
    let f = grammar::make(
        &grammar::literal(0),
        &grammar::literal(1),
        None,
        1,
        true,
        Some(4),
    );
    let mut input = rows(&f, 4);
    input.retain(|r| r.tokens.len() == 5);
    for (i, r) in input.iter_mut().enumerate() {
        r.id = i;
    }
    let mut duplicate = input[0].clone();
    duplicate.id = input.len();
    input.push(duplicate);
    let pool = pool::Pool::new(&input).unwrap();
    assert_eq!(pool.nodes.len(), 3 + 9 + 27 + 81);
    assert_eq!(pool.row_nodes[0], *pool.row_nodes.last().unwrap());
    let bits = pool.signature(&f).unwrap();
    let counts = pool.counts(&bits);
    for (i, &count) in counts.iter().enumerate() {
        let word = pool.word(i);
        assert_eq!(
            usize::try_from(count).unwrap(),
            f.evaluate(&word)
                .unwrap()
                .into_iter()
                .filter(|x| *x)
                .count()
        );
        assert!(count as usize <= word.len());
        assert!(pool.nodes[i].parent.is_none_or(|p| p < i));
    }
}
#[test]
fn nested_training_fit_and_independent_interpretation() {
    let a = grammar::make(
        &grammar::literal(0),
        &grammar::literal(1),
        None,
        0,
        false,
        Some(2),
    );
    let b = grammar::make(
        &grammar::literal(1),
        &grammar::literal(2),
        None,
        0,
        false,
        None,
    );
    let target = grammar::make(&a, &b, None, 0, false, None);
    let input = rows(&target, 4);
    assert!(input.iter().any(|r| r.label == 7));
    assert!(input.iter().any(|r| r.label == 19));
    let found = search::run(&input, 36, search::Policy::default()).unwrap();
    assert!(!found.programs.is_empty(), "{:?}", found.refusal);
    assert!(found.stages.last().unwrap().root.row_comparisons > 0);
    for p in found.programs {
        let mut map = std::collections::BTreeMap::new();
        for r in &input {
            let atom = p.run(&r.tokens).unwrap();
            assert_eq!(Some(atom), p.interpret(&r.tokens));
            if let Some(old) = map.insert(atom, r.label) {
                assert_eq!(old, r.label);
            }
        }
    }
}
#[test]
fn contradictions_and_resource_refusals_are_explicit() {
    let mut input = rows(&grammar::literal(0), 1);
    let mut conflict = input[0].clone();
    conflict.id = input.len();
    conflict.label = 7;
    input.push(conflict);
    let report = search::run(&input, 36, search::Policy::default()).unwrap();
    assert!(report.programs.is_empty());
    assert!(report.refusal.unwrap().contains("conflicting"));
    assert!(
        search::run(
            &input,
            36,
            search::Policy {
                max_classes: 32769,
                ..Default::default()
            }
        )
        .is_err()
    );
    let pool = pool::Pool::new(&input).unwrap();
    let bank = bank::Bank::initial(&pool, 8).unwrap();
    assert!(bank::Bank::expanded(&pool, &bank, 1).is_err());
}

#[test]
fn interval_pruning_matches_exhaustive_fixed_grammar() {
    let target = grammar::make(
        &grammar::literal(0),
        &grammar::literal(2),
        Some(1),
        -1,
        true,
        Some(3),
    );
    let input = rows(&target, 2);
    let pool = pool::Pool::new(&input).unwrap();
    let bank = bank::Bank::initial(&pool, 8).unwrap();
    let wanted: Vec<_> = pool
        .row_nodes
        .iter()
        .zip(&input)
        .filter(|(i, _)| **i % 2 == 0)
        .map(|(&i, r)| (i, r.label == 19))
        .collect();
    let mut meter = latent::Meter {
        work: 0,
        limit: 40_000_000_000_000,
    };
    let (found, stats) = latent::synthesize(&pool, &bank, &wanted, 32768, &mut meter).unwrap();
    assert!(!stats.capped);
    let mut expected = std::collections::BTreeSet::new();
    let mut add = |f: Formula| {
        let signature = pool.signature(&f).unwrap();
        if wanted.iter().all(|&(i, b)| pool::bit(&signature, i) == b) {
            expected.insert(signature);
        }
    };
    for c in &bank.rows {
        add(c.formula.clone());
    }
    for a in &bank.rows {
        for b in &bank.rows {
            for extra in [None, Some(0), Some(1), Some(2)] {
                for offset in -2..=2 {
                    for negated in [false, true] {
                        for gate in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)] {
                            add(grammar::make(
                                &a.formula, &b.formula, extra, offset, negated, gate,
                            ));
                        }
                    }
                }
            }
        }
    }
    let actual: std::collections::BTreeSet<_> = found.rows.into_iter().map(|c| c.bits).collect();
    assert_eq!(actual, expected);
    assert!(stats.row_comparisons > 0 && stats.signature_positions > 0);
}

#[test]
fn transition_conditions_preserve_nested_witnesses_and_ignore_unobserved_parents() {
    let a = grammar::make(
        &grammar::literal(0),
        &grammar::literal(1),
        None,
        0,
        false,
        None,
    );
    let b = grammar::make(
        &grammar::literal(1),
        &grammar::literal(2),
        Some(0),
        1,
        false,
        Some(4),
    );
    let left = grammar::make(&a, &b, None, 0, false, None);
    let right = grammar::make(&b, &a, Some(2), -1, true, None);
    let mut nonconstant = 0;
    for extra in [None, Some(0), Some(2)] {
        for offset in -2..=2 {
            for negated in [false, true] {
                for gate in [None, Some(2), Some(4)] {
                    let target = grammar::make(&left, &right, extra, offset, negated, gate);
                    let input = rows(&target, 4);
                    if !input.iter().any(|r| r.label == 7) || !input.iter().any(|r| r.label == 19) {
                        continue;
                    }
                    nonconstant += 1;
                    let pool = pool::Pool::new(&input).unwrap();
                    let branches = transitions::branches(&pool, &input).unwrap();
                    let l = pool.signature(&left).unwrap();
                    let r = pool.signature(&right).unwrap();
                    assert!(
                        branches.iter().any(|branch| {
                            let c = &branch.config;
                            let f =
                                grammar::make(&left, &right, c.extra, c.offset, c.negated, c.gate);
                            let compatible = input.iter().all(|row| {
                                let value = *f
                                    .evaluate(prior::prefix::payload(&row.tokens).unwrap())
                                    .unwrap()
                                    .last()
                                    .unwrap();
                                value == ((row.label == 19) ^ c.flipped)
                            });
                            compatible
                                && branch.left.iter().all(|&(i, b)| pool::bit(&l, i) == b)
                                && branch.right.iter().all(|&(i, b)| pool::bit(&r, i) == b)
                        }),
                        "no sound branch for extra={extra:?} offset={offset} negated={negated} gate={gate:?}"
                    );
                }
            }
        }
    }
    assert!(nonconstant >= 20);
    let mut input = rows(&left, 4);
    input.retain(|r| r.tokens.len() == 5);
    for (i, r) in input.iter_mut().enumerate() {
        r.id = i;
    }
    let pool = pool::Pool::new(&input).unwrap();
    for b in transitions::branches(&pool, &input).unwrap() {
        assert!(b.left.is_empty() && b.right.is_empty());
    }
}

#[test]
fn capped_search_keeps_partial_work_evidence() {
    let a = grammar::make(
        &grammar::literal(0),
        &grammar::literal(1),
        None,
        0,
        false,
        None,
    );
    let b = grammar::make(
        &grammar::literal(1),
        &grammar::literal(2),
        None,
        0,
        false,
        None,
    );
    let input = rows(&grammar::make(&a, &b, None, 0, false, None), 4);
    let report = search::run(
        &input,
        36,
        search::Policy {
            max_classes: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(report.programs.is_empty());
    assert!(report.refusal.unwrap().contains("class cap"));
    assert_eq!(report.expansions.len(), 1);
    assert!(report.expansions[0].visited > 8);
    assert_eq!(report.expansions[0].classes, 8);
    let pool = pool::Pool::new(&input).unwrap();
    let bank = bank::Bank::initial(&pool, 8).unwrap();
    let mut meter = latent::Meter { work: 0, limit: 0 };
    let (partial, stats) =
        latent::synthesize(&pool, &bank, &vec![(0, true)], 32, &mut meter).unwrap();
    assert!(partial.rows.is_empty() && stats.work_exhausted && !stats.capped);
    assert_eq!(stats.pairs, 0);
    assert_eq!(meter.work, 0);
    for c in &bank.rows {
        assert_eq!(c.nodes, c.formula.nodes());
    }
}
