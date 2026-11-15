use crate::*;

#[test]
fn later_simpler_completed_roots_replace_earlier_ones_without_merging_semantics() -> Result<()> {
    let a = Formula::Symbol(9);
    let b = Formula::Symbol(10);
    let c = Formula::Not(Box::new(Formula::Not(Box::new(a.clone()))));
    let mut keep = retained::Retained::new(2)?;
    keep.offer(vec![c.clone()], 10)?;
    keep.offer(vec![b.clone(), c], 20)?;
    keep.offer(vec![a.clone()], 30)?;
    keep.offer(vec![a.clone()], 40)?;
    assert_eq!(keep.formulas(), [a.clone(), b]);
    assert_eq!(keep.stats.first_fit_work, Some(10));
    assert_eq!(keep.stats.first_fit_minimum_nodes, Some(3));
    assert_eq!(keep.stats.minimum_nodes, Some(1));
    assert_eq!(keep.stats.successful_queries, 4);
    assert_eq!(keep.stats.fitting_roots_offered, 5);
    assert!(!keep.stats.interrupted_after_fit);
    assert!(keep.offer(vec![], 50).is_err());
    assert!(retained::Retained::new(0).is_err());
    assert!(retained::Retained::new(33).is_err());
    let mut huge = a;
    for _ in 0..127 {
        huge = Formula::Not(Box::new(huge));
    }
    assert!(keep.offer(vec![huge], 60).is_err());
    Ok(())
}

fn rows() -> Vec<Observation> {
    (1..=3)
        .flat_map(|n| {
            (0..3usize.pow(n)).map(move |mut code| {
                let mut tokens = vec![1];
                for _ in 0..n {
                    tokens.push([9, 10, 23][code % 3]);
                    code /= 3;
                }
                let a = tokens.iter().filter(|&&x| x == 9).count();
                let b = tokens.iter().filter(|&&x| x == 10).count();
                (tokens, if a < b { 16 } else { 15 })
            })
        })
        .enumerate()
        .map(|(id, (tokens, label))| Observation { id, tokens, label })
        .collect()
}

#[test]
fn actual_stream_keeps_completed_fits_when_a_later_query_exhausts_budget() -> Result<()> {
    let rows = rows();
    let pool = pool::Pool::new(&rows)?;
    let children = bank::Bank::initial(&pool, 8)?;
    let mut policy = search::Policy::default();
    policy.quotient.max_classes = 8;
    policy.quotient.latent_classes = 8;
    policy.quotient.retained = 4;
    policy.cache_entries = 2;
    let full = deep::run(&pool, &rows, &children, &policy, 0)?;
    assert!(full.retention.successful_queries > 1);
    assert!(!full.programs.is_empty());
    assert_eq!(full, deep::run(&pool, &rows, &children, &policy, 0)?);
    let first = full.retention.first_fit_work.ok_or("fixture never fits")?;
    let limit = first + full.final_audit_reserved_work + 1;
    let interrupted = deep::run_with_limit(&pool, &rows, &children, &policy, 0, limit)?;
    assert!(interrupted.retention.interrupted_after_fit);
    assert!(interrupted.stopped_reason.is_some());
    assert!(interrupted.refused.is_none());
    assert!(!interrupted.programs.is_empty());
    assert!(interrupted.charged_scalar_bound <= limit);
    for p in &interrupted.programs {
        let mut mapping = std::collections::BTreeMap::new();
        for r in &rows {
            let atom = p.run(&r.tokens).ok_or("undefined retained program")?;
            assert_eq!(Some(atom), p.interpret(&r.tokens));
            if let Some(old) = mapping.insert(atom, r.label) {
                assert_eq!(old, r.label);
            }
        }
    }
    assert!(deep::run_with_limit(&pool, &rows, &children, &policy, 0, 0).is_err());
    let refused = deep::run_with_limit(
        &pool,
        &rows,
        &children,
        &policy,
        0,
        full.final_audit_reserved_work + 1,
    )?;
    assert!(refused.programs.is_empty() && refused.refused.is_some());
    assert!(!refused.retention.interrupted_after_fit);
    Ok(())
}

#[test]
fn budget_classification_does_not_swallow_invalid_constraints_or_products() {
    assert!(deep::is_budget_stop("transition search work cap exhausted"));
    assert!(deep::is_budget_stop("left anchor work cap exhausted"));
    assert!(deep::is_budget_stop("right anchor work cap exhausted"));
    assert!(!deep::is_budget_stop(
        "independent streamed interpreter mismatch"
    ));
    assert!(!deep::is_budget_stop("streamed AST size differs"));
    assert!(!deep::is_budget_stop("stream count-range shape differs"));
}

#[test]
fn reused_negative_cache_retests_cheaper_arrivals_and_evicted_entries() {
    let mut cache = cache::Cache::new(1);
    cache.record_refusal(&[3], 7);
    assert!(cache.skip(&[3], 7));
    assert!(!cache.skip(&[3], 6));
    cache.record_refusal(&[3], 6);
    assert!(cache.skip(&[3], 7));
    cache.record_refusal(&[7], 2);
    assert!(!cache.skip(&[3], 100));
    assert_eq!(cache.cheaper_retests, 1);
    assert_eq!(cache.evictions, 1);
}
