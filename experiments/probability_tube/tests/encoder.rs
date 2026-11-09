//! Independent scalar source equations for validation, never a training runner.
//! Adapted from flat_approx/fixtures/reference.rs at 62638521. We retain stage
//! values to check each radius, not only the trivially bounded final features.
use probability_tube::{
    adam::Checkpoint,
    encoder::{self, Caps, Shape},
};

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
fn verify(s: Shape, old: &[f64], next: &[f64], r: &Caps, a: usize, b: usize) {
    // Construction completes before the separate future forward evaluation.
    let f = encoder::forecast(s, old, a, b, r);
    let old_values = oracle(s, old, a, b);
    let next_values = oracle(s, next, a, b);
    for ((stage, before), after) in f.stages.iter().zip(old_values).zip(&next_values) {
        bounded(distance(&stage.old, &before), 0.0);
        bounded(distance(&before, after), stage.radius);
    }
    let logits = multiply(
        next_values.last().unwrap(),
        &next[..s.embeddings()],
        s.classes,
    );
    let old_error: Vec<_> = (0..s.classes).map(|c| 0.03 * c as f64).collect();
    let next_error: Vec<_> = (0..s.classes)
        .map(|c| old_error[c] + 0.001 * (c as f64).cos())
        .collect();
    let executed: Vec<_> = logits.iter().zip(&next_error).map(|(x, e)| x + e).collect();
    for u in 0..s.classes {
        for c in 0..s.classes {
            let motion = ((logits[u] + next_error[u]) - (logits[c] + next_error[c]))
                - ((f.old_logits[u] + old_error[u]) - (f.old_logits[c] + old_error[c]));
            bounded(motion.abs(), f.contrast_budget(u, c, 0.002));
        }
        bounded(
            softmax(&executed, u),
            f.probability_ceiling(u, &old_error, 0.002),
        );
    }
}

#[test]
fn checkpoint_only_bounds_enclose_every_stage_of_nonzero_tied_motion() {
    let s = Shape {
        classes: 5,
        width: 8,
        hidden_pairs: 3,
    };
    let old = parameters(s, 0.3);
    for budget in [0.0, 1e-8, 1e-5, 0.01, 1.0] {
        let r = caps(s, budget);
        for phase in [-2.0, 0.0, 1.7] {
            let next = perturb(s, &old, budget, phase);
            for a in 0..s.classes {
                for b in 0..s.classes {
                    verify(s, &old, &next, &r, a, b);
                }
            }
        }
    }
    let f = encoder::forecast(s, &old, 1, 3, &caps(s, 1e-8));
    assert!(
        f.feature().radius < 0.01,
        "small-motion bound must beat the diameter"
    );
}

#[test]
fn original_dimensions_and_near_zero_normalizations_are_covered() {
    let s = Shape::ORIGINAL;
    assert_eq!(s.count(), 274304);
    let old = parameters(s, 0.5);
    let r = caps(s, 1e-7);
    verify(s, &old, &perturb(s, &old, 1e-7, 1.0), &r, 4, 4);
    let s = Shape {
        classes: 3,
        width: 4,
        hidden_pairs: 2,
    };
    for scale in [0.0, 1e-10, 1e-5] {
        let old: Vec<_> = parameters(s, 0.5).iter().map(|x| scale * x).collect();
        verify(s, &old, &perturb(s, &old, 1e-8, 1.0), &caps(s, 1e-8), 1, 2);
    }
}

#[test]
fn warm_adam_tubes_cover_all_parameters_and_encoder_motion() {
    let s = Shape {
        classes: 3,
        width: 4,
        hidden_pairs: 2,
    };
    let old = parameters(s, 0.3);
    let first = parameters(s, 0.7);
    let second: Vec<_> = first.iter().map(|m| m * m + 0.001).collect();
    let state = Checkpoint {
        parameters: &old,
        first: &first,
        second: &second,
        step: 17,
    };
    let rates: Vec<_> = (0..24)
        .map(|h| if h % 5 == 0 { 0.0 } else { 0.0001 })
        .collect();
    let (mut p, mut m, mut v) = (old.clone(), first.clone(), second.clone());
    for (h, &eta) in rates.iter().enumerate() {
        let r = state.motion(s, &rates[..=h], 0.1);
        let gradients = parameters(s, 0.19 * h as f64);
        for k in 0..p.len() {
            let g = gradients[k] * (if h % 3 == 0 { 1e4 } else { 1e-4 });
            m[k] = 0.9 * m[k] + 0.1 * g;
            v[k] = 0.95 * v[k] + 0.05 * g * g;
            let t = (state.step + h + 1) as f64;
            let direction = (m[k] / (1.0 - 0.9_f64.powf(t)))
                / ((v[k] / (1.0 - 0.95_f64.powf(t))).sqrt() + 1e-8);
            p[k] = (1.0 - eta * 0.1) * p[k] - eta * direction;
        }
        for (block, cap) in (0..s.classes).map(|c| (s.row(c), r.rows[c])).chain([
            (s.gate(), r.gate),
            (s.value(), r.value),
            (s.output(), r.output),
        ]) {
            bounded(distance(&p[block.clone()], &old[block]), cap);
        }
        verify(s, &old, &p, &r, 0, 2);
    }
    let zero = vec![0.0; s.count()];
    let initial = Checkpoint {
        parameters: &old,
        first: &zero,
        second: &zero,
        step: 0,
    };
    assert_eq!(initial.motion(s, &[], 0.1).output, 0.0);
    assert!(initial.motion(s, &[1e-4], 0.1).output > 0.0);
}

#[test]
#[should_panic(expected = "zero variance with nonzero moment")]
fn invalid_saved_moments_are_rejected() {
    let s = Shape {
        classes: 1,
        width: 2,
        hidden_pairs: 1,
    };
    let p = vec![1.0; s.count()];
    let v = vec![0.0; s.count()];
    Checkpoint {
        parameters: &p,
        first: &p,
        second: &v,
        step: 1,
    }
    .motion(s, &[0.001], 0.1);
}
