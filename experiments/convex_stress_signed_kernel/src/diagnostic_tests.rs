use crate::*;

#[test]
fn compressed_negative_logits_use_every_raw_class() {
    let row = diagnostic::Row {
        labels: vec![2, 4],
        target: 1,
        logits: vec![],
    };
    for z in [vec![-3., 1.], vec![-3., -1.], vec![0., 0.]] {
        let mut full = vec![0.; 7];
        full[2] = z[0];
        full[4] = z[1];
        let (ce, hit, prob) = row.statistics(&z, 7);
        assert!((ce - kernel::training::ce(&full, 4)).abs() < 1e-14);
        let winner = (0..7)
            .max_by(|&a, &b| full[a].total_cmp(&full[b]).then(b.cmp(&a)))
            .unwrap();
        assert_eq!(hit, winner == 4);
        let top = full.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = full.iter().map(|v| (v - top).exp()).sum();
        assert!((prob[0] - (full[2] - top).exp() / sum).abs() < 1e-14);
        assert!((prob[1] - (full[4] - top).exp() / sum).abs() < 1e-14);
    }
}

#[test]
fn actual_canonical_complex_products_match_signed_diagnostic_and_slopes() -> Result<()> {
    let m = tests::make()?;
    let mut work = Work::default();
    let (fits, _) = kernels::fit_bank(
        m.parent(),
        m.bank(),
        &model::active_blocks(m.parent()),
        &mut work,
    )?;
    let cache = diagnostic::Cache::new(m.parent(), &fits, &mut work)?;
    let mut checked = 0;
    let mut negative = 0;
    for (record, row) in m.records().iter().zip(&cache.rows) {
        let (block, state) = m.source().location(record.cell);
        for (k, fit) in fits[block].iter().enumerate().skip(1) {
            let Some(fit) = fit else {
                continue;
            };
            let weights = fit.excluded(state, &mut work)?;
            negative += usize::from(weights.iter().any(|&w| w < 0.));
            let mut full = vec![0.; m.source().vocabulary];
            for (i, &w) in weights.iter().enumerate() {
                for (z, actual) in full.iter_mut().zip(m.source().logits_state(block, i)) {
                    *z += w * actual;
                }
            }
            for (c, actual) in full.iter().enumerate() {
                let compressed = row
                    .labels
                    .binary_search(&c)
                    .map_or(0., |i| row.logits[k][i]);
                assert!((compressed - actual).abs() < 1e-9);
            }
            checked += 1;
        }
    }
    assert!(checked > 0 && negative > 0);
    let weights = vec![1. / m.bank.len() as f64; m.bank.len()];
    for (start, count) in [(0, 45), (40, 5)] {
        let eval = cache.evaluate(&weights, start, count, true);
        for index in [1, 10, 48] {
            let eps = 1e-6;
            let mut plus = weights.clone();
            plus[index] += eps;
            plus[0] -= eps;
            let mut minus = weights.clone();
            minus[index] -= eps;
            minus[0] += eps;
            let difference = (cache.evaluate(&plus, start, count, false).loss
                - cache.evaluate(&minus, start, count, false).loss)
                / (2. * eps);
            assert!((difference - (eval.slopes[index] - eval.slopes[0])).abs() < 2e-6);
        }
        let losses = cache.line_scores(&weights, 10, start, count);
        for (&loss, gamma) in losses.iter().zip(LINE_SEARCH) {
            let mut line: Vec<_> = weights.iter().map(|a| (1. - gamma) * a).collect();
            line[10] += gamma;
            assert!((loss - cache.evaluate(&line, start, count, false).loss).abs() < 1e-12);
        }
    }
    assert_eq!(digest(&cache)?, m.diagnostic_sha256);
    Ok(())
}
