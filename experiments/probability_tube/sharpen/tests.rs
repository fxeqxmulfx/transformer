//! Oracle and synthetic ties copied from probability_tube/tests/encoder.rs at 7f6660d1.
//! Adapted from flat_approx/fixtures/reference.rs at 62638521. We retain stage
//! values to check each radius, not only the trivially bounded final features.
use probability_sharp::Checkpoint;
use probability_tube::encoder::{Caps, Shape};

fn length(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |a, b| a.hypot(*b))
}
fn distance(x: &[f64], y: &[f64]) -> f64 {
    length(&x.iter().zip(y).map(|(a, b)| a - b).collect::<Vec<_>>())
}
fn rms(x: &[f64]) -> Vec<f64> {
    let denominator =
        (x.iter().map(|v| v.powi(2)).sum::<f64>() / x.len() as f64 + f32::EPSILON as f64).sqrt();
    x.iter().map(|v| v / denominator).collect()
}
fn multiply(x: &[f64], a: &[f64], rows: usize) -> Vec<f64> {
    let mut result = vec![0.0; rows];
    for r in 0..rows {
        for c in 0..x.len() {
            result[r] += a[r * x.len() + c] * x[c];
        }
    }
    result
}
fn oracle(s: Shape, p: &[f64], a: usize, b: usize) -> Vec<Vec<f64>> {
    let (w, h, e) = (s.width, s.hidden_pairs, s.embeddings());
    let mut gather = vec![0.0; w];
    for k in 0..w / 2 {
        gather[2 * k] = p[a * w + 2 * k] - p[b * w + 2 * k + 1];
        gather[2 * k + 1] = p[a * w + 2 * k + 1] + p[b * w + 2 * k];
    }
    let x = rms(&gather);
    let inner = rms(&x);
    let gv = multiply(&inner, &p[e..e + 4 * h * w], 4 * h);
    let gate = gv[..2 * h].to_vec();
    let value = gv[2 * h..].to_vec();
    let mut product = vec![0.0; 2 * h];
    for k in 0..h {
        let (a, b, u, v) = (gv[k], gv[k + h], gv[k + 2 * h], gv[k + 3 * h]);
        product[k] = (a * u - b * v) / 2.0_f64.sqrt();
        product[k + h] = (a * v + b * u) / 2.0_f64.sqrt();
    }
    let raw = multiply(&product, &p[e + 4 * h * w..], w);
    let branch = rms(&raw);
    let pre: Vec<_> = x.iter().zip(&branch).map(|(x, y)| x + y).collect();
    let feature = rms(&pre);
    vec![
        gather, x, inner, gate, value, product, raw, branch, pre, feature,
    ]
}

// Synthetic deterministic test data, NOT the missing trainer initializer.
fn parameters(s: Shape, phase: f64) -> Vec<f64> {
    let mut p: Vec<_> = (0..s.count())
        .map(|i| ((i as f64 + 1.0) * 0.37 + phase).sin() * 0.2)
        .collect();
    // Enforce source complex-linear aliases; output rows/hidden halves have
    // different layouts. Expanded moment aliases have the same first-moment
    // signs and unsigned second-moment copies (optimizer_efficiency/layout.rs).
    for start in [s.gate().start, s.value().start] {
        for j in 0..s.hidden_pairs {
            for i in 0..s.width / 2 {
                let (a, b) = (
                    p[start + j * s.width + 2 * i],
                    p[start + (j + s.hidden_pairs) * s.width + 2 * i],
                );
                p[start + j * s.width + 2 * i + 1] = -b;
                p[start + (j + s.hidden_pairs) * s.width + 2 * i + 1] = a;
            }
        }
    }
    let start = s.output().start;
    for i in 0..s.width / 2 {
        for j in 0..s.hidden_pairs {
            let (a, b) = (
                p[start + 2 * i * s.hidden() + j],
                p[start + (2 * i + 1) * s.hidden() + j],
            );
            p[start + 2 * i * s.hidden() + j + s.hidden_pairs] = -b;
            p[start + (2 * i + 1) * s.hidden() + j + s.hidden_pairs] = a;
        }
    }
    p
}
fn caps(s: Shape, radius: f64) -> Caps {
    Caps {
        rows: vec![radius; s.classes],
        gate: radius,
        value: radius,
        output: radius,
    }
}
fn perturb(s: Shape, old: &[f64], budget: f64, phase: f64) -> Vec<f64> {
    let mut d = parameters(s, phase);
    for block in (0..s.classes)
        .map(|c| s.row(c))
        .chain([s.gate(), s.value(), s.output()])
    {
        let factor = budget / length(&d[block.clone()]);
        for k in block {
            d[k] = old[k] + factor * d[k];
        }
    }
    d
}
fn bounded(actual: f64, upper: f64) {
    assert!(
        actual <= upper + 2e-11 * (1.0 + upper),
        "{actual} > {upper}"
    );
}
fn softmax(logits: &[f64], u: usize) -> f64 {
    let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (logits[u] - m).exp() / logits.iter().map(|v| (v - m).exp()).sum::<f64>()
}

#[test]
fn whole_source_sharp_bounds_enclose_independent_tied_perturbations() {
    let s = Shape {
        classes: 5,
        width: 8,
        hidden_pairs: 7,
    };
    let old = parameters(s, 0.3);
    let checkpoint = Checkpoint::new(s, &old);
    let mut strict = 0;
    for budget in [0., 1e-8, 1e-5, 0.01, 1.] {
        let r = caps(s, budget);
        for (a, b) in [(0, 0), (1, 3), (4, 2)] {
            let f = checkpoint.evaluate(a, b, &r);
            assert!(f.radii[9] <= f.base.feature().radius);
            strict += usize::from(f.radii[9] < 0.99 * f.base.feature().radius);
            for phase in [-2., 0., 1.7] {
                let next = perturb(s, &old, budget, phase);
                let before = oracle(s, &old, a, b);
                let after = oracle(s, &next, a, b);
                for (i, (x, y)) in before.iter().zip(&after).enumerate() {
                    bounded(distance(x, y), f.radii[i]);
                    bounded(distance(x, &f.base.stages[i].old), 0.);
                }
                let logits = multiply(after.last().unwrap(), &next[..s.embeddings()], s.classes);
                let errors: Vec<_> = (0..s.classes).map(|c| 0.02 * c as f64).collect();
                let future: Vec<_> = logits
                    .iter()
                    .enumerate()
                    .map(|(c, z)| z + errors[c] + 0.001 * (c as f64).cos())
                    .collect();
                for u in 0..s.classes {
                    bounded(
                        softmax(&future, u),
                        f.probability_ceiling(u, &errors, 0.002),
                    );
                    for c in 0..s.classes {
                        let motion = future[u]
                            - future[c]
                            - (f.base.old_logits[u] + errors[u] - f.base.old_logits[c] - errors[c]);
                        bounded(motion.abs(), f.contrast_budget(u, c, 0.002));
                    }
                }
            }
        }
    }
    assert!(
        strict > 0,
        "the new formula must improve a nonzero synthetic case"
    );
}

#[test]
fn original_dimensions_keep_every_layer_and_parameter_block() {
    let s = Shape::ORIGINAL;
    let old = parameters(s, 0.8);
    let checkpoint = Checkpoint::new(s, &old);
    assert!(checkpoint.operators.iter().all(|b| b.factor <= b.frobenius));
    let r = caps(s, 1e-6);
    let f = checkpoint.evaluate(7, 91, &r);
    let next = perturb(s, &old, 1e-6, -0.7);
    let before = oracle(s, &old, 7, 91);
    let after = oracle(s, &next, 7, 91);
    for (i, (x, y)) in before.iter().zip(&after).enumerate() {
        bounded(distance(x, y), f.radii[i]);
    }
    assert!(f.radii[9] < f.base.feature().radius && f.radii[9] < 2. * 128f64.sqrt());
}
