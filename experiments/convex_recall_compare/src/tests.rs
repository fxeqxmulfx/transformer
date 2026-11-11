use crate::{convex::*, data, engine, layout::Dims, model, reference};

#[test]
fn arithmetic_labels_and_copy_limit() {
    for p in [17, 97] {
        let d = data::division(0, p);
        let mut ids = std::collections::HashSet::new();
        let mut copyable = 0;
        for s in d.train.iter().chain(&d.test) {
            assert_eq!(s.tokens.len(), 4);
            assert_eq!(s.targets[0] * s.tokens[2] % p, s.tokens[0]);
            assert!((1..p).contains(&s.tokens[2]));
            assert!(ids.insert(s.id));
            copyable += usize::from(s.values.contains(&s.targets[0]));
        }
        assert_eq!(ids.len(), p * (p - 1));
        assert_eq!(copyable, 3 * p - 4);
    }
}

#[test]
fn dictionary_data_separate_and_are_held_out() {
    let d = data::mqar(3, 8, 16, 512, 512);
    assert!(data::separates(&d));
    let train: std::collections::HashSet<_> = d.train.iter().map(|s| &s.values).collect();
    assert_eq!(train.len(), d.train.len());
    for s in &d.test {
        assert!(!train.contains(&s.values));
        for (i, &q) in s.queries.iter().enumerate() {
            let position = s.tokens[..16].chunks_exact(2).find(|p| p[0] == q).unwrap();
            assert_eq!(position[1], s.targets[i]);
            assert_eq!(s.tokens[16 + i], q);
        }
    }
    let again = data::mqar(3, 8, 16, 512, 512);
    assert_eq!(data::fingerprint(&d.train), data::fingerprint(&again.train));
}

#[test]
fn convex_gradients_are_numerical_derivatives() {
    for kind in [Kind::Matching, Kind::Copy, Kind::Table] {
        let d = if kind == Kind::Matching { data::mqar(1, 3, 4, 12, 4) } else { data::division(1, 7) };
        let mut c = Convex::new(kind, &d);
        let batch: Vec<_> = d.train[..3].iter().collect();
        let (_, grad) = c.loss_gradient(&batch);
        for (i, &g) in grad.iter().enumerate() {
            let old = c.p[i];
            c.p[i] = old + 1e-5;
            let plus = c.loss_gradient(&batch).0;
            c.p[i] = old - 1e-5;
            let minus = c.loss_gradient(&batch).0;
            c.p[i] = old;
            assert!((g - (plus - minus) / 2e-5).abs() < 1e-8, "{kind:?} coordinate {i}");
        }
    }
}

#[test]
fn sparse_table_update_is_the_full_batch_gradient() {
    let d = data::division(2, 7);
    let mut c = Convex::new(Kind::Table, &d);
    let batch = [&d.train[0], &d.train[1], &d.train[0]];
    let (_, g) = c.loss_gradient(&batch);
    let before = c.p.clone();
    c.step(&batch);
    for i in 0..c.p.len() {
        assert!((c.p[i] - (before[i] - g[i])).abs() < 1e-12);
    }
    for s in &d.test {
        assert!(c.logits(s).iter().all(|&x| x == 0.0));
    }
}

#[test]
fn projection_and_matching_recover_the_lean_optimizer() {
    let mut row = [-3.0, 2.0, 1.0, -0.5];
    project_simplex(&mut row);
    assert_eq!(row, [0.0, 1.0, 0.0, 0.0]);
    let d = data::mqar(9, 4, 4, 32, 32);
    assert!(data::separates(&d));
    let mut c = Convex::new(Kind::Matching, &d);
    let batch: Vec<_> = d.train.iter().collect();
    for _ in 0..100 {
        c.step(&batch);
    }
    assert!(c.identity_error().unwrap() < 1e-12);
    let ideal = ((1.0f64).exp() + d.vocab as f64 - 1.0).ln() - 1.0;
    assert!((c.loss_gradient(&batch).0 - ideal).abs() < 1e-12);
    for s in &d.test {
        let z = c.logits(s);
        for (i, &t) in s.targets.iter().enumerate() {
            for v in 0..d.vocab {
                assert_eq!(z[i * d.vocab + v], f64::from(v == t));
            }
        }
    }
}

#[test]
fn longer_transformer_matches_independent_double_reference() {
    let d = Dims { vocab: 8, d: 8, heads: 4, layers: 2, dff: 32, ctx: 24 };
    let p = engine::initialize(d, 5);
    let pd: Vec<_> = p.iter().map(|&x| x as f64).collect();
    let tokens: Vec<_> = (0..24).map(|i| (i * 3) % 8).collect();
    let sup = [16, 20, 23];
    let targets = [1, 2, 7];
    let mut example = reference::Example { tokens: tokens.clone(), targets: vec![0; 24] };
    for (&pos, &target) in sup.iter().zip(&targets) {
        example.targets[pos] = target;
    }
    let (loss, grad) = reference::loss_and_gradient(d, &pd, &[example], &sup);
    let pe = d.positions();
    let mut wt = model::Transposed::new(d);
    wt.update(d, &p);
    let m = model::Model { dims: d, p: &p, pe: &pe, wt: &wt, sup: &sup };
    let mut w = model::Work::default();
    m.forward(&mut w, &tokens);
    let logits = w.logits.clone();
    let losses = m.backward(&mut w, &tokens, &targets, 1.0 / targets.len() as f32);
    assert!((losses.iter().sum::<f64>() / 3.0 - loss).abs() < 2e-6);
    let error = w.grad.iter().zip(&grad).map(|(&x, y)| (x as f64 - y).abs()).fold(0.0, f64::max);
    assert!(error < 1e-5, "gradient maximum error {error}");
    let mut changed = tokens;
    changed[23] = (changed[23] + 1) % 8;
    m.forward(&mut w, &changed);
    assert_eq!(&logits[..16], &w.logits[..16], "future query token affected earlier logits");
}
