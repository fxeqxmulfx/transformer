use super::*;
use std::collections::BTreeMap;

#[test]
fn uncapped_stream_matches_full_small_grammar_and_reaches_past_truncation() {
    let rows = rows(3);
    let pool = pool::Pool::new(&rows).unwrap();
    let children = bank::Bank::initial(&pool, 8).unwrap();
    let full = bank::Bank::expanded(&pool, &children, 32768).unwrap();
    assert!(full.refusal.is_none());
    let bounds = count_bounds::Ranges {
        low: vec![0; pool.nodes.len()],
        high: pool.nodes.iter().map(|n| n.depth as i16).collect(),
    };
    // A partial observation admits both true and false unconstrained positions.
    let wanted = vec![(pool.row_nodes[0], true)];
    let expected: BTreeMap<_, _> = full
        .bank
        .rows
        .iter()
        .filter(|c| pool::bit(&c.bits, wanted[0].0))
        .map(|c| (c.bits.clone(), c.nodes))
        .collect();
    let mut actual = BTreeMap::new();
    let mut meter = meter();
    let mut stats = stream::Stats::default();
    stream::enumerate(
        &pool,
        &children,
        &wanted,
        &bounds,
        &mut meter,
        &mut stats,
        |q, _| {
            let formula = (q.make)();
            assert_eq!(formula.nodes(), q.nodes);
            assert_eq!(pool.signature(&formula).unwrap(), q.bits);
            assert_eq!(pool.counts(q.bits), q.counts);
            actual
                .entry(q.bits.to_vec())
                .and_modify(|old: &mut usize| *old = (*old).min(q.nodes))
                .or_insert(q.nodes);
            Ok(false)
        },
    )
    .unwrap();
    assert!(stats.complete && !stats.found);
    assert_eq!(actual, expected);
    let (truncated, capped) = latent::synthesize(&pool, &children, &wanted, 8, &mut meter).unwrap();
    assert!(capped.capped);
    let missing = expected
        .keys()
        .find(|bits| truncated.rows.iter().all(|c| &c.bits != *bits))
        .unwrap();
    let mut found = stream::Stats::default();
    let mut cache = cache::Cache::new(2);
    stream::enumerate(
        &pool,
        &children,
        &wanted,
        &bounds,
        &mut meter,
        &mut found,
        |q, _| {
            if cache.skip(q.bits, q.nodes) {
                return Ok(false);
            }
            if q.bits == missing {
                return Ok(true);
            }
            cache.record_refusal(q.bits, q.nodes);
            Ok(false)
        },
    )
    .unwrap();
    assert!(found.found && !found.complete && cache.evictions > 0);
    assert!(found.visited_queries > truncated.rows.len() as u64);
    let mut empty_meter = latent::Meter { work: 0, limit: 0 };
    let mut refused = stream::Stats::default();
    assert!(
        stream::enumerate(
            &pool,
            &children,
            &wanted,
            &bounds,
            &mut empty_meter,
            &mut refused,
            |_, _| panic!("work-refused query must not run")
        )
        .is_err()
    );
    assert!(!refused.complete && !refused.found);
    assert_eq!(empty_meter.work, 0);
}
