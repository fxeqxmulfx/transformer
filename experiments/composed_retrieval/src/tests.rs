use super::{
    Result, data,
    model::{Model, WIDTH},
    operator::Tables,
    query,
    train::{self, Sample},
};
use baseline::{
    channel::{argmax, cross_entropy},
    data::Example,
};
use std::f64::consts::{PI, TAU};

fn probabilities(scores: Vec<f64>) -> Vec<f64> {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<_> = scores.iter().map(|&v| (v - max).exp()).collect();
    let z: f64 = exp.iter().sum();
    exp.iter().map(|p| p / z).collect()
}
/// Literal CM113 phase cosines, independent nested sums over both source
/// and destination indices. It shares neither routing factors nor readouts.
fn literal(m: &Model, xs: &[usize]) -> Vec<f64> {
    let l = m.validate().unwrap();
    let w = &m.weights;
    let decode = |v: usize, turn: Option<usize>, c: usize| {
        (0..WIDTH)
            .map(|d| {
                (w[v + d] + turn.map_or(0., |q| w[l.phase.start + q * WIDTH + d])
                    - w[l.value.start + c * WIDTH + d])
                    .cos()
            })
            .sum::<f64>()
            / WIDTH as f64
    };
    let q = xs.last().copied();
    let x = &xs[..xs.len().saturating_sub(1)];
    let n = x.len();
    if n == 0 {
        return (0..m.vocabulary)
            .map(|c| m.capacity * decode(l.start.start, q, c))
            .collect();
    }
    let q = q.unwrap();
    let link = |j: usize| probabilities((0..n).map(|k| w[l.offset(j, k)]).collect());
    let jump = |o: usize| {
        let key = probabilities(
            x.iter()
                .enumerate()
                .map(|(j, &v)| {
                    let mean = (0..WIDTH)
                        .map(|d| {
                            (w[l.key.start + o * WIDTH + d] - w[l.key.start + v * WIDTH + d]).cos()
                        })
                        .sum::<f64>()
                        / WIDTH as f64;
                    w[l.scale] * mean
                        + w[l.bias.start + if j == 0 { m.vocabulary } else { x[j - 1] }]
                        + w[l.recency] * j as f64
                })
                .collect(),
        );
        (0..n)
            .map(|k| (0..n).map(|j| key[j] * link(j)[k]).sum::<f64>())
            .collect::<Vec<_>>()
    };
    let first = jump(q);
    let second: Vec<_> = x.iter().map(|&o| jump(o)).collect();
    (0..m.vocabulary)
        .map(|c| {
            let mut one = 0.;
            let mut two = 0.;
            let mut rotated = 0.;
            for (k, &id) in x.iter().enumerate() {
                one += first[k] * decode(l.value.start + WIDTH * id, None, c);
                rotated += first[k] * decode(l.value.start + WIDTH * id, Some(q), c);
                for (j, &v) in x.iter().enumerate() {
                    two += first[k] * second[k][j] * decode(l.value.start + WIDTH * v, None, c);
                }
            }
            m.capacity
                * ((1. - w[l.product]) * ((1. - w[l.depth]) * one + w[l.depth] * two)
                    + w[l.product] * rotated)
        })
        .collect()
}
fn fixture() -> Result<Model> {
    let mut m = Model::new(5, 7, &data::protocol()?, 3)?;
    let l = m.validate()?;
    for i in l.bias.start..l.scale {
        m.weights[i] = (i as f64 * 0.97).sin();
    }
    m.weights[l.recency] = -0.14;
    Ok(m)
}
#[test]
fn operator_matches_literal_paths_boundaries_and_renamings() -> Result<()> {
    let m = fixture()?;
    let t = Tables::new(&m, &[1, 2, 3, 4, 5, 6])?;
    let mut permuted = m.clone();
    let l = m.validate()?;
    let perm = |v: usize| (v + 2) % 5;
    for o in 0..5 {
        for bank in [&l.key, &l.value, &l.phase] {
            for d in 0..WIDTH {
                permuted.weights[bank.start + perm(o) * WIDTH + d] =
                    m.weights[bank.start + o * WIDTH + d];
            }
        }
        permuted.weights[l.bias.start + perm(o)] = m.weights[l.bias.start + o];
    }
    let other = Tables::new(&permuted, &[1, 2, 3, 4, 5, 6])?;
    for x in [
        &[][..],
        &[2],
        &[1, 3],
        &[2, 1, 2, 0, 2],
        &[0, 1, 1, 2, 0, 2, 4],
    ] {
        let z = t.logits(&m, &t.pass(&m, x).hidden);
        let renamed: Vec<_> = x.iter().map(|&o| perm(o)).collect();
        let z2 = other.logits(&permuted, &other.pass(&permuted, &renamed).hidden);
        for (c, &v) in literal(&m, x).iter().enumerate() {
            assert!((z[c] - v).abs() < 2e-13 && (z2[perm(c)] - v).abs() < 2e-13);
            assert!(v.abs() <= m.capacity + 1e-13);
        }
    }
    assert!(m.objects(&[-1]).is_err() && m.objects(&[5]).is_err() && m.objects(&[0; 8]).is_err());
    Ok(())
}
#[test]
fn all_gradients_and_feasible_queries_match_independent_complete_ce() -> Result<()> {
    let mut m = fixture()?;
    let l = m.validate()?;
    let xs = [
        Sample {
            prefix: vec![],
            target: 2,
        },
        Sample {
            prefix: vec![1],
            target: 3,
        },
        Sample {
            prefix: vec![0, 1],
            target: 0,
        },
        Sample {
            prefix: vec![0, 1, 1, 2, 0],
            target: 2,
        },
        Sample {
            prefix: vec![2, 4, 0, 3],
            target: 1,
        },
    ];
    let batch: Vec<_> = xs.iter().collect();
    let coords: Vec<_> = (0..m.weights.len()).collect();
    let h = 1e-5;
    let mut t = Tables::new(&m, &[1, 3, 4])?;
    let analytic = train::analytic(&m, &t, &batch);
    let oracle = |v: &Model| {
        batch
            .iter()
            .map(|s| cross_entropy(&literal(v, &s.prefix), s.target))
            .sum::<f64>()
            / batch.len() as f64
    };
    for boundary in [false, true] {
        if boundary {
            m.weights[l.depth] = 0.;
            m.weights[l.product] = 1.;
        }
        let original = m.clone();
        let forward = query::differences(&mut m, &mut t, &batch, &coords, h, None)?;
        assert_eq!(m, original);
        for &i in &coords {
            let hi = l.project(i, original.weights[i] + h);
            let lo = l.project(i, original.weights[i] - h);
            m.weights[i] = hi;
            let plus = oracle(&m);
            m.weights[i] = lo;
            let minus = oracle(&m);
            m.weights[i] = original.weights[i];
            let truth = (plus - minus) / (hi - lo);
            assert!(
                (forward[i] - truth).abs() < 5e-9,
                "forward {i}: {} vs {truth}",
                forward[i]
            );
            if !boundary {
                assert!(
                    (analytic[i] - truth).abs() < 5e-8,
                    "analytic {i}: {} vs {truth}",
                    analytic[i]
                );
            }
        }
    }
    Ok(())
}
#[test]
fn log_odds_queries_match_the_literal_model_and_chain_rule() -> Result<()> {
    let mut m = fixture()?;
    let l = m.validate()?;
    let row = Sample {
        prefix: vec![0, 1, 1, 2, 0],
        target: 2,
    };
    let batch = [&row];
    let step = 0.0001;
    for z in [-4., 0., 9_f64.ln(), 4.] {
        let h = crate::model::sigmoid(z);
        m.weights[l.depth] = h;
        let mut t = Tables::new(&m, &[4])?;
        let analytic = train::analytic(&m, &t, &batch)[l.depth] * h * (1. - h);
        let actual = query::differences(&mut m, &mut t, &batch, &[l.depth], step, Some(z))?[0];
        let trial = |z: f64| {
            let mut trial = m.clone();
            trial.weights[l.depth] = 1. / (1. + (-z).exp());
            cross_entropy(&literal(&trial, &row.prefix), row.target)
        };
        let independent = (trial(z + step) - trial(z - step)) / (2. * step);
        assert!((actual - independent).abs() < 2e-10);
        assert!((analytic - independent).abs() < 5e-9);
        assert_eq!(m.weights[l.depth], h);
    }
    Ok(())
}
#[test]
fn inhabits_both_lean_capacity_witnesses() -> Result<()> {
    let p = data::protocol()?;
    let mut m = Model::new(97, 2, &p, 7)?;
    let l = m.validate()?;
    m.weights[l.product] = 1.;
    for o in 0..97 {
        for d in 0..WIDTH {
            m.weights[l.value.start + o * WIDTH + d] = TAU * o as f64 / 97.;
            m.weights[l.phase.start + o * WIDTH + d] = TAU * o as f64 / 97.;
        }
    }
    let t = Tables::new(&m, &[1])?;
    for a in 0..97 {
        for b in 0..97 {
            assert_eq!(
                argmax(&t.logits(&m, &t.pass(&m, &[a, b]).hidden)),
                (a + b) % 97
            );
        }
    }
    let mut m = Model::new(3, 5, &p, 1)?;
    let l = m.validate()?;
    m.weights.fill(0.);
    m.weights[l.scale] = 64.;
    m.weights[l.bias.start] = -64.;
    m.weights[l.offset(0, 1)] = 64.;
    m.weights[l.depth] = 31. / 32.;
    m.weights[l.product] = 1. / 32.;
    for bank in [&l.key, &l.value, &l.phase] {
        for o in 0..3 {
            for d in 0..WIDTH {
                m.weights[bank.start + o * WIDTH + d] = if o == 2 || (o == 1 && d % 2 == 0) {
                    PI
                } else {
                    0.
                };
            }
        }
    }
    let t = Tables::new(&m, &[4])?;
    let pass = t.pass(&m, &[0, 1, 1, 2, 0]);
    assert_eq!(argmax(&t.logits(&m, &pass.s1)), 1);
    assert_eq!(argmax(&t.logits(&m, &pass.hidden)), 2);
    Ok(())
}
#[test]
fn original_fingerprints_masks_and_optimizer_clocks() -> Result<()> {
    let mut p = data::protocol()?;
    for (id, s) in &p.cases {
        let (task, _) = data::original(&p, id)?;
        assert_eq!(s.train, task.splits["train"].len());
        let m = Model::new(task.vocab, s.context, &p, 0)?;
        let n = if id.starts_with("suite-") { 2 } else { 1 };
        assert_eq!(
            train::samples(&m, &task.splits["train"])?
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
            n * s.train
        );
    }
    p.updates = 7;
    let rows = vec![
        Example {
            task: "fixture".into(),
            tokens: vec![1, 2, 3],
            targets: vec![-100, 3, 4],
            changes: vec![false, true, true],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None
        };
        3
    ];
    let mut replay = None;
    for mode in ["mini", "forward"] {
        let mut m = Model::new(5, 4, &p, 0)?;
        let r = train::fit(&mut m, &rows, 2, mode, 0, &p, || Ok(()))?;
        assert_eq!(r["examples"], 11);
        assert_eq!(r["targets"], 22);
        assert!(r["min_visits"].as_u64().unwrap() > 0);
        if mode == "mini" || p.coordinates_per_table.is_none() {
            assert_eq!(r["min_visits"], p.updates);
            assert_eq!(r["max_visits"], p.updates);
            let queries = if mode == "forward" {
                2 * m.weights.len() * p.updates
            } else {
                0
            };
            assert_eq!(r["forward_loss_queries"], queries);
        }
        assert_eq!(
            r["forward_loss_queries"].as_u64().unwrap() > 0,
            mode == "forward"
        );
        if let Some(d) = replay {
            assert_eq!(r["batch_sha256"], d);
        }
        replay = Some(r["batch_sha256"].clone());
    }
    Ok(())
}

#[test]
#[cfg(feature = "arithmetic-cohort")]
fn every_previously_available_arithmetic_record_matches_the_original_adapter() -> Result<()> {
    let p = data::protocol()?;
    let suite: serde_json::Value = serde_json::from_str(learned_retrieval::cohort::SUITE)?;
    let mut matched = 0;
    for row in suite["tasks"].as_array().ok_or("original tasks")? {
        if row["case"]["kind"] != "arithmetic" {
            continue;
        }
        let (task, pins) = data::original(&p, row["name"].as_str().ok_or("case name")?)?;
        let old = baseline::data::build(row)?;
        assert_eq!(pins, row["data_fingerprints"]);
        assert_eq!((old.vocab, &old.selection), (task.vocab, &task.selection));
        assert_eq!(
            old.splits.keys().collect::<Vec<_>>(),
            task.splits.keys().collect::<Vec<_>>()
        );
        for (name, original) in &old.splits {
            let actual = &task.splits[name];
            assert_eq!(actual.len(), original.len());
            for (a, b) in actual.iter().zip(original) {
                assert_eq!(
                    (&a.tokens, &a.targets, &a.changes),
                    (&b.tokens, &b.targets, &b.changes)
                );
                assert_eq!(
                    (&a.task, &a.prompt, &a.answer, a.limit, a.field, a.row_index),
                    (&b.task, &b.prompt, &b.answer, b.limit, b.field, b.row_index)
                );
            }
        }
        matched += 1;
    }
    assert_eq!(matched, 22);
    Ok(())
}
