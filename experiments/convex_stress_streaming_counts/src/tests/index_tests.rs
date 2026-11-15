use super::*;
use crate::index::{Index, Side};
use std::collections::BTreeMap;

#[test]
fn both_index_orientations_match_independent_interpreters_with_padded_tail() {
    let rows = rows(2);
    let pool = pool::Pool::new(&rows).unwrap();
    let base = bank::Bank::initial(&pool, 8).unwrap();
    let expanded = bank::Bank::expanded(&pool, &base, 32768).unwrap();
    assert!(expanded.refusal.is_none());
    let mut anchor = bank::Bank::new();
    for c in expanded.bank.rows.iter().take(73) {
        anchor
            .add(&pool, &c.bits, c.nodes, 73, || c.formula.clone())
            .unwrap();
    }
    assert_eq!(anchor.rows.len(), 73);
    let observed: BTreeMap<_, _> = pool
        .row_nodes
        .iter()
        .zip(&rows)
        .map(|(&i, r)| (i, r.label == 19))
        .collect();
    let extras: Vec<_> = (0..3)
        .map(|s| pool.counts(&pool.signature(&grammar::literal(s)).unwrap()))
        .collect();
    let mut matches = 0;
    for side in [Side::Left, Side::Right] {
        for extra in [None, Some(0)] {
            for offset in [-2, 0, 1, 2] {
                for negated in [false, true] {
                    for gate in [None, Some(4)] {
                        for flipped in [false, true] {
                            let config = transitions::Config {
                                extra,
                                offset,
                                negated,
                                gate,
                                flipped,
                            };
                            let mut meter = meter();
                            let mut index = Index::new(
                                &pool, &observed, &anchor, side, &config, &extras, &mut meter,
                            )
                            .unwrap();
                            for q in expanded.bank.rows.iter().take(17) {
                                let expected: Vec<_> = anchor
                                    .rows
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, a)| {
                                        let (left, right) = if side == Side::Left {
                                            (&a.formula, &q.formula)
                                        } else {
                                            (&q.formula, &a.formula)
                                        };
                                        let p = Program::Prefix(grammar::make(
                                            left, right, extra, offset, negated, gate,
                                        ));
                                        rows.iter().all(|r| {
                                            p.interpret(&r.tokens)
                                                == Some(prior::program::Atom::Quantity(i64::from(
                                                    (r.label == 19) ^ flipped,
                                                )))
                                        })
                                    })
                                    .map(|(i, _)| i)
                                    .collect();
                                let actual = index.query(&q.counts, q.nodes, &mut meter).unwrap();
                                assert_eq!(actual, expected, "{side:?} {config:?}");
                                matches += actual.len();
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(matches > 0);
    let config = transitions::Config {
        extra: None,
        offset: 0,
        negated: false,
        gate: None,
        flipped: false,
    };
    let mut empty_meter = latent::Meter { work: 0, limit: 0 };
    assert!(
        Index::new(
            &pool,
            &observed,
            &anchor,
            Side::Left,
            &config,
            &extras,
            &mut empty_meter
        )
        .is_err()
    );
    assert_eq!(empty_meter.work, 0);
}

#[test]
fn missing_observations_and_inactive_gates_invent_no_constraints() {
    let rows = rows(2);
    let pool = pool::Pool::new(&rows).unwrap();
    let anchor = bank::Bank::initial(&pool, 8).unwrap();
    let extras: Vec<_> = (0..3)
        .map(|s| pool.counts(&pool.signature(&grammar::literal(s)).unwrap()))
        .collect();
    let mut config = transitions::Config {
        extra: None,
        offset: -2,
        negated: true,
        gate: Some(0),
        flipped: false,
    };
    let observed: BTreeMap<_, _> = pool
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.symbol == 10)
        .map(|(i, _)| (i, false))
        .collect();
    let mut meter = meter();
    let mut index = Index::new(
        &pool,
        &observed,
        &anchor,
        Side::Right,
        &config,
        &extras,
        &mut meter,
    )
    .unwrap();
    assert_eq!(
        index
            .query(&anchor.rows[0].counts, anchor.rows[0].nodes, &mut meter)
            .unwrap(),
        (0..anchor.rows.len()).collect::<Vec<_>>()
    );
    config.flipped = true;
    let mut impossible = Index::new(
        &pool,
        &observed,
        &anchor,
        Side::Right,
        &config,
        &extras,
        &mut meter,
    )
    .unwrap();
    assert!(
        impossible
            .query(&anchor.rows[0].counts, anchor.rows[0].nodes, &mut meter)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn cheaper_equivalent_query_must_be_retested_after_actual_size_refusal() {
    let rows = vec![
        Observation {
            id: 0,
            tokens: vec![1, 9],
            label: 19,
        },
        Observation {
            id: 1,
            tokens: vec![1, 9, 10],
            label: 7,
        },
    ];
    let pool = pool::Pool::new(&rows).unwrap();
    let mut anchor = bank::Bank::new();
    let a = grammar::constants(true);
    anchor
        .add(&pool, &pool.signature(&a).unwrap(), a.nodes(), 1, || a)
        .unwrap();
    let cheap = grammar::literal(0);
    let mut expensive = cheap.clone();
    for _ in 0..120 {
        expensive = Formula::Not(Box::new(expensive));
    }
    assert_eq!(expensive.nodes(), 121);
    let bits = pool.signature(&cheap).unwrap();
    assert_eq!(pool.signature(&expensive).unwrap(), bits);
    let counts = pool.counts(&bits);
    let observed = pool
        .row_nodes
        .iter()
        .zip(&rows)
        .map(|(&i, r)| (i, r.label == 19))
        .collect();
    let config = transitions::Config {
        extra: None,
        offset: 1,
        negated: false,
        gate: None,
        flipped: false,
    };
    let mut meter = meter();
    let mut index = Index::new(
        &pool,
        &observed,
        &anchor,
        Side::Left,
        &config,
        &[],
        &mut meter,
    )
    .unwrap();
    let mut cache = cache::Cache::new(1);
    assert!(
        index
            .query(&counts, expensive.nodes(), &mut meter)
            .unwrap()
            .is_empty()
    );
    cache.record_refusal(&bits, expensive.nodes());
    assert!(cache.skip(&bits, expensive.nodes()));
    assert!(!cache.skip(&bits, cheap.nodes()));
    let ids = index.query(&counts, cheap.nodes(), &mut meter).unwrap();
    assert_eq!(ids, [0]);
    assert_eq!(index.stats.size_rejections, 1);
    assert_eq!(cache.cheaper_retests, 1);
    let program = Program::Prefix(index.assemble(&cheap, &ids, 1).pop().unwrap());
    for r in rows {
        assert_eq!(
            program.interpret(&r.tokens),
            Some(prior::program::Atom::Quantity(i64::from(r.label == 19)))
        );
    }
    cache.record_refusal(&[12345], 5);
    assert_eq!(cache.evictions, 1);
    assert!(!cache.skip(&bits, expensive.nodes()));
    assert_eq!(cache.peak_entries, 1);
}
