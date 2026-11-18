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
    let offset = |c: usize| w[l.output_offsets.clone()].get(c).copied().unwrap_or(0.);
    let radius = |c: usize| w[l.radii.clone()].get(c).copied().unwrap_or(1.);
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
            .map(|c| m.capacity * decode(l.start.start, q, c) + offset(c))
            .collect();
    }
    let q = q.unwrap();
    let link = |j: usize| {
        probabilities(
            (0..n)
                .map(|k| {
                    if !m.signed_profile {
                        return w[l.offset(j, k)];
                    }
                    let d = k as isize - j as isize;
                    if d == 0 {
                        return 0.;
                    }
                    let side = usize::from(d > 0);
                    let near = w[l.relation.start + side];
                    if d.unsigned_abs() == 1 {
                        near
                    } else {
                        near - w[l.relation.start + 2 + side] * (d.unsigned_abs() - 1) as f64
                    }
                })
                .collect(),
        )
    };
    let kernel = |o: usize, v: usize| {
        (0..WIDTH)
            .map(|d| (w[l.key.start + o * WIDTH + d] - w[l.key.start + v * WIDTH + d]).cos())
            .sum::<f64>()
            / WIDTH as f64
    };
    let jump = |o: usize, i: usize| {
        let key = probabilities(
            x.iter()
                .enumerate()
                .map(|(j, &v)| {
                    let comparison = match (l.predecessor, i, j) {
                        (Some(t), 1.., 1..) => w[t] * kernel(x[i - 1], x[j - 1]),
                        _ => 0.,
                    };
                    w[l.scale] * kernel(o, v)
                        + comparison
                        + w[l.bias.start + if j == 0 { m.vocabulary } else { x[j - 1] }]
                        + w[l.recency] * j as f64
                })
                .collect(),
        );
        (0..n)
            .map(|k| (0..n).map(|j| key[j] * link(j)[k]).sum::<f64>())
            .collect::<Vec<_>>()
    };
    let first = jump(q, n);
    let second: Vec<_> = x.iter().enumerate().map(|(i, &o)| jump(o, i)).collect();
    (0..m.vocabulary)
        .map(|c| {
            let mut one = 0.;
            let mut two = 0.;
            let mut rotated = 0.;
            for (k, &id) in x.iter().enumerate() {
                one += first[k] * decode(l.value.start + WIDTH * id, None, c);
                rotated += first[k]
                    * radius(id)
                    * radius(q)
                    * decode(l.value.start + WIDTH * id, Some(q), c);
                for (j, &v) in x.iter().enumerate() {
                    two += first[k] * second[k][j] * decode(l.value.start + WIDTH * v, None, c);
                }
            }
            m.capacity
                * ((1. - w[l.product]) * ((1. - w[l.depth]) * one + w[l.depth] * two)
                    + w[l.product] * rotated)
                + offset(c)
        })
        .collect()
}
fn fixture() -> Result<Model> {
    let mut m = Model::new(5, 7, &data::protocol()?, 3)?;
    let l = m.validate()?;
    for i in l.bias.start..l.scale {
        m.weights[i] = (i as f64 * 0.97).sin();
        if l.is_profile_decay(i) {
            m.weights[i] = m.weights[i].abs() + 0.25;
        }
    }
    m.weights[l.recency] = -0.14;
    for (c, b) in m.weights[l.output_offsets.clone()].iter_mut().enumerate() {
        *b = (c as f64 + 1.).sin() * 3.;
    }
    for (i, r) in m.weights[l.radii.clone()].iter_mut().enumerate() {
        *r = 0.2 + i as f64 * 0.13;
    }
    if let Some(t) = l.predecessor {
        m.weights[t] = 0.7;
    }
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
        if !l.output_offsets.is_empty() {
            permuted.weights[l.output_offsets.start + perm(o)] =
                m.weights[l.output_offsets.start + o];
        }
        if !l.radii.is_empty() {
            permuted.weights[l.radii.start + perm(o)] = m.weights[l.radii.start + o];
        }
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
            assert!(v.abs() <= m.capacity + m.output_offset_bound.unwrap_or(0.) + 1e-13);
        }
    }
    assert!(m.objects(&[-1]).is_err() && m.objects(&[5]).is_err() && m.objects(&[0; 8]).is_err());
    Ok(())
}
#[test]
fn all_gradients_and_feasible_queries_match_independent_complete_ce() -> Result<()> {
    check_gradients(fixture()?)
}
fn check_gradients(mut m: Model) -> Result<()> {
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
            prefix: vec![2, 2],
            target: 1,
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
            if let Some(bound) = m.output_offset_bound {
                m.weights[l.output_offsets.start] = -bound;
                m.weights[l.output_offsets.start + 1] = bound;
            }
            if !l.radii.is_empty() {
                m.weights[l.radii.start] = 0.;
                m.weights[l.radii.start + 1] = 1.;
            }
            if l.signed_profile {
                m.weights[l.relation.start + 2] = 0.;
                t.refresh(&m, l.relation.start + 2);
            }
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

#[cfg(feature = "radii-cohort")]
#[test]
fn predecessor_scores_match_the_parent_at_zero_and_the_literal_paths_elsewhere() -> Result<()> {
    let mut p = data::protocol()?;
    p.predecessor_match = false;
    let mut parent = Model::new(5, 7, &p, 3)?;
    let pair = Model::new(5, 2, &p, 3)?;
    p.predecessor_match = true;
    assert_eq!(Model::new(5, 2, &p, 3)?, pair);
    let mut m = Model::new(5, 7, &p, 3)?;
    let l = m.validate()?;
    let i = l.predecessor.unwrap();
    assert_eq!(i, parent.weights.len());
    assert_eq!(&m.weights[..i], &parent.weights);
    m.weights[l.depth] = 0.37;
    m.weights[l.product] = 0.42;
    for i in l.relation.clone() {
        let a = (i as f64 * 0.37).sin();
        m.weights[i] = if l.is_profile_decay(i) {
            0.25 + a.abs()
        } else {
            a
        };
    }
    for (j, r) in m.weights[l.radii.clone()].iter_mut().enumerate() {
        *r = 0.25 + 0.1 * j as f64;
    }
    parent.weights.copy_from_slice(&m.weights[..i]);
    let inputs = [&[][..], &[2], &[1, 0], &[1, 1, 2], &[0, 1, 1, 1, 0]];
    let lengths = [1, 2, 4];
    let baseline = Tables::new(&parent, &lengths)?;
    for t in [0., -0.7, 0.7] {
        m.weights[i] = t;
        let table = Tables::new(&m, &lengths)?;
        for xs in inputs {
            let actual = table.logits(&m, &table.pass(&m, xs).hidden);
            if t == 0. || xs.len() <= 2 {
                assert_eq!(
                    actual,
                    baseline.logits(&parent, &baseline.pass(&parent, xs).hidden)
                );
            } else if xs == [0, 1, 1, 1, 0] {
                let old = baseline.logits(&parent, &baseline.pass(&parent, xs).hidden);
                assert!(actual.iter().zip(old).any(|(z, v)| (z - v).abs() > 1e-6));
            }
            for (v, expected) in actual.iter().zip(literal(&m, xs)) {
                assert!((v - expected).abs() < 2e-13);
            }
        }
        check_gradients(m.clone())?;
    }
    m.weights[l.key.start..l.key.start + WIDTH].fill(0.);
    m.weights[l.key.start + WIDTH..l.key.start + 2 * WIDTH].fill(PI);
    for t in [-0.6, 0., 0.4] {
        m.weights[i] = t;
        let table = Tables::new(&m, &[4])?;
        let rows = table.pass(&m, &[0, 1, 1, 1, 0]).attention;
        let odds = rows[5] * rows[15] / (rows[7] * rows[13]);
        assert!((odds.ln() - 4. * t).abs() < 2e-13);
    }
    Ok(())
}

#[cfg(feature = "radii-cohort")]
#[test]
fn both_optimizers_train_the_shared_predecessor_coefficient() -> Result<()> {
    let mut p = data::protocol()?;
    crate::prefix::configure_from(&mut p, data::REWRITE_PROTOCOL, true)?;
    p.predecessor_match = true;
    p.updates = 3;
    let (task, _) = data::original(&p, "suite-copy-repeat")?;
    let rows = &task.splits["train"][..2];
    for mode in &p.modes {
        let mut m = Model::new(task.vocab, p.cases["suite-copy-repeat"].context, &p, 0)?;
        let i = m.validate()?.predecessor.unwrap();
        let receipt = train::fit(&mut m, rows, 2, mode, 0, &p, || Ok(()))?;
        assert!(m.weights[i].abs() > 1e-8);
        assert_eq!(receipt["min_visits"], p.updates);
        assert_eq!(
            receipt["forward_loss_queries"].as_u64().unwrap() > 0,
            mode == "forward"
        );
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
    if l.signed_profile {
        m.weights[l.relation.start + 1] = 64.;
        m.weights[l.relation.start + 3] = 64.;
    } else {
        m.weights[l.offset(0, 1)] = 64.;
    }
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
    let manifest: serde_json::Value = serde_json::from_str(data::PROTOCOL)?;
    for (id, s) in &p.cases {
        let (task, pins) = data::original(&p, id)?;
        assert_eq!(s.train, task.splits["train"].len());
        let m = Model::new(task.vocab, s.context, &p, 0)?;
        let targets = if let Some(n) = manifest["cases"][id]["training_targets"].as_u64() {
            usize::try_from(n)?
        } else if id.starts_with("suite-") {
            pins["splits"]["train"]["targets"]
                .as_u64()
                .ok_or("source target count")? as usize
        } else {
            s.train
        };
        assert_eq!(
            train::samples(&m, &task.splits["train"])?
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
            targets
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
#[cfg(all(feature = "arithmetic-cohort", not(feature = "prefix-cohort")))]
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

#[test]
#[cfg(all(
    feature = "arithmetic-cohort",
    feature = "logodds-depth",
    not(feature = "prefix-cohort")
))]
fn arithmetic_cohort_uses_the_original_log_odds_updates() -> Result<()> {
    let p = data::protocol()?;
    assert_eq!(
        p.cases.len(),
        if cfg!(feature = "output-offsets") {
            51
        } else {
            48
        }
    );
    assert_eq!(p.initial_log_odds, Some(9_f64.ln()));
    assert_eq!(p.coordinates_per_table, None);
    let m = Model::new(p.cases["arithmetic-add"].vocabulary, 2, &p, 0)?;
    assert_eq!(
        m.weights[m.validate()?.depth],
        crate::model::sigmoid(9_f64.ln())
    );
    let shell: serde_json::Value = serde_json::from_str(data::PROTOCOL)?;
    assert_eq!(
        shell["training_protocol_sha256"],
        baseline::hash(data::TRAINING_PROTOCOL.as_bytes())
    );
    Ok(())
}

#[cfg(feature = "output-offsets")]
#[test]
fn old_models_and_zero_offsets_give_the_same_logits() -> Result<()> {
    let mut p = data::protocol()?;
    p.learned_product_radii = false;
    let m = Model::new(5, 6, &p, 0)?;
    let l = m.validate()?;
    assert_eq!(m.output_offset_bound, Some(16.));
    assert_eq!(&m.weights[l.output_offsets.clone()], &[0.; 5]);
    p.output_offset_bound = None;
    let old = Model::new(5, 6, &p, 0)?;
    let bytes = serde_json::to_vec(&old)?;
    assert!(!String::from_utf8_lossy(&bytes).contains("output_offset_bound"));
    let replay: Model = serde_json::from_slice(&bytes)?;
    assert_eq!(&m.weights[..l.output_offsets.start], &replay.weights);
    for x in [&[][..], &[1], &[1, 0], &[4, 1, 2, 4, 1]] {
        let scores = |model: &Model| -> Result<_> {
            let t = Tables::new(model, &[x.len().saturating_sub(1)])?;
            Ok(t.logits(model, &t.pass(model, x).hidden))
        };
        assert_eq!(scores(&m)?, scores(&replay)?);
    }
    let mut bad = m.clone();
    bad.weights[l.output_offsets.start] = 17.;
    assert!(bad.validate().is_err());
    bad.weights.pop();
    assert!(bad.validate().is_err());
    Ok(())
}

#[cfg(feature = "radii-cohort")]
#[test]
fn old_coordinates_and_unit_radius_initial_scores_agree_exactly() -> Result<()> {
    let mut p = data::protocol()?;
    let model = Model::new(5, 7, &p, 2)?;
    let l = model.validate()?;
    assert_eq!(&model.weights[l.radii.clone()], &[1.; 5]);
    p.learned_product_radii = false;
    let old: Model = serde_json::from_slice(&serde_json::to_vec(&Model::new(5, 7, &p, 2)?)?)?;
    assert_eq!(&model.weights[..l.radii.start], &old.weights);
    let current = Tables::new(&model, &[1, 2, 3])?;
    let reference = Tables::new(&old, &[1, 2, 3])?;
    for prefix in [&[][..], &[2], &[4, 4], &[0, 2, 1], &[0, 3, 2, 3]] {
        assert_eq!(
            current.logits(&model, &current.pass(&model, prefix).hidden),
            reference.logits(&old, &reference.pass(&old, prefix).hidden)
        );
    }
    let mut bad = model;
    bad.weights[l.radii.start] = 1.1;
    assert!(bad.validate().is_err());
    Ok(())
}

#[cfg(feature = "profile-cohort")]
#[test]
fn signed_profile_allocates_only_visible_coordinates_and_starts_at_parent_logits() -> Result<()> {
    let p = data::protocol()?;
    let mut parent = p.clone();
    parent.signed_profile = false;
    for (context, count) in [(2, 0), (3, 2), (4, 4), (9, 4)] {
        let current = Model::new(5, context, &p, 1)?;
        let old = Model::new(5, context, &parent, 1)?;
        let l = current.validate()?;
        assert_eq!(l.relation.len(), count);
        assert!(!String::from_utf8_lossy(&serde_json::to_vec(&old)?).contains("signed_profile"));
        let current_tables = Tables::new(&current, &(0..context).collect::<Vec<_>>())?;
        let old_tables = Tables::new(&old, &(0..context).collect::<Vec<_>>())?;
        let input: Vec<_> = (0..context).map(|i| i % 5).collect();
        for length in 0..=context {
            let x = &input[..length];
            assert_eq!(
                current_tables.logits(&current, &current_tables.pass(&current, x).hidden),
                old_tables.logits(&old, &old_tables.pass(&old, x).hidden)
            );
        }
        if count == 4 {
            let mut invalid = current;
            invalid.weights[l.relation.start + 3] = -0.1;
            assert!(invalid.validate().is_err());
        }
    }
    Ok(())
}

#[cfg(feature = "profile-cohort")]
#[test]
fn arithmetic_control_keeps_fitted_common_coordinates_in_both_optimizers() -> Result<()> {
    let mut p = data::protocol()?;
    p.updates = 4;
    let mut parent = p.clone();
    parent.signed_profile = false;
    let (task, _) = data::original(&p, "arithmetic-add")?;
    let rows = &task.splits["train"][..8];
    for mode in &p.modes {
        let mut a = Model::new(task.vocab, 2, &parent, 0)?;
        let mut b = Model::new(task.vocab, 2, &p, 0)?;
        let receipt_a = train::fit(&mut a, rows, 4, mode, 0, &parent, || Ok(()))?;
        let receipt_b = train::fit(&mut b, rows, 4, mode, 0, &p, || Ok(()))?;
        let old = a.validate()?;
        let new = b.validate()?;
        assert!(new.relation.is_empty());
        assert_eq!(old.relation.len(), 1);
        assert_eq!(
            &a.weights[..old.relation.start],
            &b.weights[..new.relation.start]
        );
        assert_eq!(
            &a.weights[old.relation.end..],
            &b.weights[new.relation.end..]
        );
        assert_eq!(receipt_a["after"], receipt_b["after"]);
        assert_eq!(receipt_a["batch_sha256"], receipt_b["batch_sha256"]);
    }
    Ok(())
}
