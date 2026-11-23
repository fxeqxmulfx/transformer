use super::*;

#[test]
fn soft_read_uses_the_write_prior_and_a_true_scalar_distance() {
    let mut m = Cells::new(2);
    Write::apply(&mut m, 2., &[5., 1.], 0.5).unwrap();
    Write::apply(&mut m, 0., &[1., 1.], 1.).unwrap();
    let r = Read::at(&m, 0.);
    let weak = 0.5 * (-4f64).exp();
    assert!((f64::from(r.value[0]) - (1. + 5. * weak) / (1. + weak)).abs() < 0.000001);
    let [n, gap, mass, _] = r.quality.unwrap();
    assert_eq!(n, 2.);
    assert!((gap - (4. + 2f64.ln())).abs() < 1e-9 && mass > 0.99);
}

fn values(n: usize, shift: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + shift) * 1.77).sin() * 0.7 + 0.13)
        .collect()
}
#[test]
fn shared_steps_differentiate_the_hidden_gate_the_addresses_and_each_value() {
    let (d, len) = (4, 7);
    let mut x = values(d * len, 0.7);
    let mut h = values(d * len, 1.4);
    let y = values(d * len, 3.7);
    let mut w = values(parameters(d), 2.9);
    w[MAPS * d + OUT_BIAS] = 0.6;
    let mut dx = vec![0.; x.len()];
    let mut dh = vec![0.; h.len()];
    let mut dw = vec![0.; w.len()];
    let tape = Tape::new(&x, &h, &w, d, len, 4, true);
    tape.backward(&h, &w, &y, &mut dx, &mut dh, &mut dw);
    let objective = |x: &[f32], h: &[f32], w: &[f32]| {
        let mut out = vec![0.; x.len()];
        Tape::new(x, h, w, d, len, 4, false).add_output(&mut out);
        dot(&out, &y)
    };
    let eps = 0.0003;
    for (kind, analytical) in [dx, dh, dw].iter().enumerate() {
        for (i, &derivative) in analytical.iter().enumerate() {
            match kind {
                0 => x[i] += eps,
                1 => h[i] += eps,
                _ => w[i] += eps,
            }
            let plus = objective(&x, &h, &w);
            match kind {
                0 => x[i] -= 2. * eps,
                1 => h[i] -= 2. * eps,
                _ => w[i] -= 2. * eps,
            }
            let minus = objective(&x, &h, &w);
            match kind {
                0 => x[i] += eps,
                1 => h[i] += eps,
                _ => w[i] += eps,
            }
            let numeric = (plus - minus) / (2. * eps);
            assert!(
                (numeric - derivative).abs() < 0.002 + 0.012 * derivative.abs(),
                "source {kind} coordinate {i}: numeric {numeric} analytic {derivative}"
            );
        }
    }
}
#[test]
fn deep_frozen_reads_are_prefix_causal_and_reset_at_batch_boundaries() {
    let (d, len) = (3, 7);
    let x = values(2 * d * len, 0.3);
    let h = values(2 * d * len, 1.7);
    let w = values(parameters(d), 2.1);
    for n in [1, 10, 20, 40] {
        let mut full = vec![0.; x.len()];
        Tape::new(&x, &h, &w, d, len, n, false).add_output(&mut full);
        let mut training = vec![0.; x.len()];
        Tape::new(&x, &h, &w, d, len, n, true).add_output(&mut training);
        assert_eq!(training, full);
        for b in 0..2 {
            let start = b * d * len;
            for l in 1..=len {
                let end = start + l * d;
                let mut separate = vec![0.; l * d];
                Tape::new(&x[start..end], &h[start..end], &w, d, l, n, false)
                    .add_output(&mut separate);
                assert_eq!(separate, full[start..end]);
            }
        }
    }
}
