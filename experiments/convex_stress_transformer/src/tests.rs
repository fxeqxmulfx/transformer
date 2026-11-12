use crate::{adam, data::Example, engine, layout::Dims, memory, model, reference};

fn dims(layers: usize) -> Dims {
    Dims {
        vocab: 8,
        d: 8,
        heads: 4,
        layers,
        dff: 32,
        ctx: 24,
    }
}
fn example(n: usize, mask: &[usize], shift: usize) -> Example {
    let mut targets = vec![-100; n];
    for &i in mask {
        targets[i] = ((i + shift) % 8) as i64;
    }
    Example {
        task: "fixture".into(),
        tokens: (0..n).map(|i| ((i * 3 + shift) % 8) as i64).collect(),
        targets,
        changes: vec![false; n],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
fn net(layers: usize, microbatch: usize) -> engine::Transformer {
    engine::Transformer::new(dims(layers), 5, microbatch, adam::Hyper::source(0.1)).unwrap()
}
fn maximum(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y).abs())
        .fold(0., f64::max)
}

#[test]
fn one_two_and_six_layers_match_independent_double_reference() {
    for layers in [1, 2, 6] {
        let mut net = net(layers, 4);
        let examples = [
            example(24, &[0, 7, 23], 1),
            example(12, &[4, 7], 2),
            example(24, &[0, 7, 23], 3),
        ];
        let targets: usize = examples
            .iter()
            .map(|e| e.targets.iter().filter(|&&x| x >= 0).count())
            .sum();
        let pd: Vec<_> = net.p.iter().map(|&x| x as f64).collect();
        let mut total_loss = 0.;
        let mut total_grad = vec![0.; pd.len()];
        for e in &examples {
            let sup: Vec<_> = e
                .targets
                .iter()
                .enumerate()
                .filter_map(|(i, &y)| (y >= 0).then_some(i))
                .collect();
            let rd = Dims {
                ctx: e.tokens.len(),
                ..net.dims
            };
            let input = reference::Example {
                tokens: e.tokens.iter().map(|&x| x as usize).collect(),
                targets: e.targets.iter().map(|&x| x.max(0) as usize).collect(),
            };
            let full_logits = reference::logits(rd, &pd, &input.tokens);
            let actual = net.logits(&e.tokens, &sup).unwrap();
            for (row, &i) in sup.iter().enumerate() {
                let err = actual[row * rd.vocab..][..rd.vocab]
                    .iter()
                    .zip(&full_logits[i * rd.vocab..][..rd.vocab])
                    .map(|(x, y)| (x - y).abs())
                    .fold(0., f64::max);
                assert!(err < 4e-6, "{layers} layer logits: {err}");
            }
            let (loss, grad) = reference::loss_and_gradient(rd, &pd, &[input], &sup);
            let weight = sup.len() as f64 / targets as f64;
            total_loss += weight * loss;
            for (g, x) in total_grad.iter_mut().zip(grad) {
                *g += weight * x;
            }
        }
        let (loss, gradient, actual_targets) = net
            .loss_gradient(&examples.iter().collect::<Vec<_>>())
            .unwrap();
        assert_eq!(actual_targets, targets);
        assert!((loss - total_loss).abs() < 4e-6, "{layers} layer CE");
        let error = maximum(&gradient, &total_grad);
        assert!(error < 3e-5, "{layers} layer gradient: {error}");
    }
}

#[test]
fn two_layer_port_preserves_original_logits_gradients_and_initialization() {
    let d = dims(2);
    let p = engine::initialize(d, 5).unwrap();
    assert_eq!(p, convex_recall_compare::engine::initialize(d, 5));
    let pe = d.positions();
    let mut old_wt = convex_recall_compare::model::Transposed::new(d);
    let mut wt = model::Transposed::new(d);
    old_wt.update(d, &p);
    wt.update(d, &p);
    let sup = [0, 8, 23];
    let tokens: Vec<_> = (0..48).map(|i| (i * 3 + 1) % 8).collect();
    let labels = [1, 3, 6, 2, 0, 7];
    let old = convex_recall_compare::model::Model {
        dims: d,
        p: &p,
        pe: &pe,
        wt: &old_wt,
        sup: &sup,
    };
    let new = model::Model {
        dims: d,
        p: &p,
        pe: &pe,
        wt: &wt,
        sup: &sup,
    };
    let mut a = convex_recall_compare::model::Work::default();
    let mut b = model::Work::default();
    old.forward(&mut a, &tokens);
    new.forward(&mut b, &tokens);
    assert_eq!(a.logits, b.logits);
    assert_eq!(
        old.backward(&mut a, &tokens, &labels, 1. / 6.),
        new.backward(&mut b, &tokens, &labels, 1. / 6.)
    );
    assert_eq!(a.grad, b.grad);
}

#[test]
fn streamed_gradients_keep_whole_batch_weighting_and_adam_state() {
    let examples = [
        example(24, &[0, 7, 23], 1),
        example(12, &[4, 7], 2),
        example(24, &[0, 7, 23], 3),
        example(24, &[0, 7, 23], 6),
    ];
    let refs = examples.iter().collect::<Vec<_>>();
    for layers in [1, 2, 6] {
        let mut one = net(layers, 1);
        let mut many = net(layers, 32);
        for step in 1..=12 {
            let (la, ga, ca) = one.loss_gradient(&refs).unwrap();
            let (lb, gb, cb) = many.loss_gradient(&refs).unwrap();
            assert_eq!(ca, cb);
            assert!((la - lb).abs() < 1e-5);
            assert!(maximum(&ga, &gb.iter().map(|&x| x as f64).collect::<Vec<_>>()) < 1e-5);
            one.step(&refs, step).unwrap();
            many.step(&refs, step).unwrap();
            for (a, b) in [(&one.p, &many.p), (&one.m, &many.m), (&one.v, &many.v)] {
                assert!(maximum(a, &b.iter().map(|&x| x as f64).collect::<Vec<_>>()) < 1e-5);
            }
        }
    }
}

#[test]
fn causal_mask_and_loss_masks_prevent_future_token_leakage() {
    for layers in [1, 2, 6] {
        let mut net = net(layers, 4);
        let mut e = example(24, &[0, 7], 2);
        let logits = net.logits(&e.tokens, &[0, 7]).unwrap();
        let (loss, grad, _) = net.loss_gradient(&[&e]).unwrap();
        for x in &mut e.tokens[8..] {
            *x = (*x + 1) % 8;
        }
        assert_eq!(logits, net.logits(&e.tokens, &[0, 7]).unwrap());
        let (after, g, _) = net.loss_gradient(&[&e]).unwrap();
        assert_eq!(loss, after);
        assert_eq!(grad, g);
        let all = net.logits(&e.tokens, &(0..24).collect::<Vec<_>>()).unwrap();
        for i in [0, 7, 23] {
            let prefix = net.logits(&e.tokens[..=i], &[i]).unwrap();
            for (a, b) in all[i * 8..][..8].iter().zip(prefix) {
                assert!((a - b).abs() < 2e-6);
            }
        }
    }
}

#[test]
fn memory_preflight_and_malformed_masks_fail_before_model_allocation() {
    for layers in [1, 2, 6] {
        assert!(memory::check(dims(layers), 32).unwrap() < memory::BUFFER_LIMIT);
    }
    let large = Dims {
        ctx: 8192,
        d: 1024,
        dff: 4096,
        layers: 32,
        ..dims(2)
    };
    assert!(memory::bytes(large, 1).unwrap() > memory::BUFFER_LIMIT);
    assert!(engine::Transformer::new(large, 0, 1, adam::Hyper::source(0.1)).is_err());
    assert!(
        memory::bytes(
            Dims {
                d: usize::MAX,
                ..large
            },
            32
        )
        .is_err()
    );
    assert!(memory::bytes(dims(2), usize::MAX).is_err());
    let mut net = net(2, 4);
    for mask in [vec![-100; 24], vec![-1; 24], vec![8; 24], vec![0; 23]] {
        let mut e = example(24, &[0], 1);
        e.targets = mask;
        assert!(net.loss_gradient(&[&e]).is_err());
    }
    assert!(net.loss_gradient(&[]).is_err());
    assert!(net.logits(&[0, 1], &[1, 0]).is_err());
    assert!(net.logits(&[8], &[0]).is_err());
    assert!(net.step(&[&example(24, &[0], 1)], 0).is_err());
}

#[test]
fn original_metrics_agree_with_independent_reference_predictions() {
    use crate::metrics::{self, Predictor};
    struct Double {
        dims: Dims,
        p: Vec<f64>,
    }
    impl Predictor for Double {
        fn vocab(&self) -> usize {
            self.dims.vocab
        }
        fn logits(&self, prefix: &[i64]) -> Vec<f64> {
            let dims = Dims {
                ctx: prefix.len(),
                ..self.dims
            };
            let tokens: Vec<_> = prefix.iter().map(|&x| x as usize).collect();
            reference::logits(dims, &self.p, &tokens)[(prefix.len() - 1) * dims.vocab..].to_vec()
        }
    }
    let mut net = net(6, 4);
    let double = Double {
        dims: net.dims,
        p: net.p.iter().map(|&x| x as f64).collect(),
    };
    let mut generative = example(8, &[5, 6, 7], 2);
    generative.prompt = generative.tokens[..6].to_vec();
    generative.answer = generative.targets[5..].to_vec();
    generative.limit = 3;
    let mut field = example(4, &[3], 3);
    field.field = Some((12, 20));
    for examples in [
        vec![example(12, &[0, 4, 9], 2)],
        vec![generative],
        vec![field],
    ] {
        let mut actual = metrics::evaluate(
            &engine::Predictor(std::cell::RefCell::new(&mut net)),
            &examples,
        )
        .unwrap();
        let mut expected = metrics::evaluate(&double, &examples).unwrap();
        fn remove_close_losses(a: &mut serde_json::Value, b: &mut serde_json::Value) {
            for key in ["loss", "example_loss"] {
                assert!((a[key].as_f64().unwrap() - b[key].as_f64().unwrap()).abs() < 4e-6);
                a.as_object_mut().unwrap().remove(key);
                b.as_object_mut().unwrap().remove(key);
            }
            if a.get("teacher_forced").is_some() {
                remove_close_losses(&mut a["teacher_forced"], &mut b["teacher_forced"]);
            }
        }
        remove_close_losses(&mut actual, &mut expected);
        assert_eq!(actual, expected);
    }
}
