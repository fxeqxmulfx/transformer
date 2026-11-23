use super::*;

fn nearest(row: &[f32]) -> Option<usize> {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0)
}
fn make(keys: &[usize], d: usize) -> Vec<f32> {
    keys.iter()
        .flat_map(|&k| (0..d).map(move |j| if j == k { 1. } else { 0. }))
        .collect()
}
fn choose(w: &mut [f32], cue: usize, main: usize, tail: usize) {
    let code = paths::code(cue, main, tail);
    w[code as usize % BINS] = if code & BINS as u32 == 0 { 1. } else { -1. };
}

#[test]
fn repeated_reads_use_the_latest_link_then_a_different_final_relation() {
    let d = 12;
    // Both occurrences of the same directed relation at address 0 are live.
    // Only the later write participates. The terminal is read in reverse.
    let keys = [0, 1, 9, 11, 0, 1, 2, 11, 2, 1, 3, 10, 4, 5, 3, 11, 7, 0];
    let x = make(&keys, d);
    let mut w = vec![0.; parameters(d)];
    choose(&mut w, 7, 2, 11);
    for (mode, exists) in [(1, false), (2, true), (5, true)] {
        let tape = Tape::new(&x, &x, &w, d, keys.len(), mode, nearest);
        let mut out = vec![0.; x.len()];
        tape.add_output(&mut out);
        let last = row(&out, keys.len() - 1, d);
        assert_eq!(last[4], f32::from(exists));
        assert_eq!(last.iter().copied().sum::<f32>(), f32::from(exists));
        for n in 1..keys.len() {
            let mut prefix = vec![0.; n * d];
            Tape::new(&x[..n * d], &x[..n * d], &w, d, n, mode, nearest).add_output(&mut prefix);
            assert_eq!(prefix, out[..n * d]);
        }
    }
}

#[test]
fn signed_path_sum_and_gate_differentiate_through_dense_values() {
    let d = 6;
    let x = make(&[0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 4, 3, 2, 5, 4, 3, 1, 2], d);
    let length = 9;
    let h: Vec<_> = (0..x.len())
        .map(|i| ((i * 23 % 67) as f32 - 32.) / 83.)
        .collect();
    let y: Vec<_> = (0..x.len())
        .map(|i| ((i * 17 % 83) as f32 - 40.) / 71.)
        .collect();
    let w: Vec<_> = (0..parameters(d))
        .map(|i| ((i * 17 % 113) as f32 - 50.) / 299.)
        .collect();
    let tape = Tape::new(&x, &h, &w, d, length, 2, nearest);
    let mut dx = vec![0.; x.len()];
    let mut dh = vec![0.; x.len()];
    let mut dw = vec![0.; w.len()];
    tape.backward(&h, &w, &y, &mut dx, &mut dh, &mut dw);
    let objective = |a: &[f32], b: &[f32], c: &[f32]| {
        let mut out = vec![0.; a.len()];
        Tape::new(a, b, c, d, length, 2, nearest).add_output(&mut out);
        out.iter()
            .zip(&y)
            .map(|(&v, &k)| f64::from(v) * f64::from(k))
            .sum::<f64>()
    };
    for (id, (base, pullback)) in [(&x, &dx), (&h, &dh), (&w, &dw)].into_iter().enumerate() {
        let direction: Vec<_> = (0..base.len())
            .map(|i| ((i * 97 % 193) as f32 - 96.) / 107.)
            .collect();
        let at = |eps: f32| {
            let z: Vec<_> = base
                .iter()
                .zip(&direction)
                .map(|(a, b)| a + eps * b)
                .collect();
            match id {
                0 => objective(&z, &h, &w),
                1 => objective(&x, &z, &w),
                _ => objective(&x, &h, &z),
            }
        };
        let analytic = dot(pullback, &direction);
        let numeric = (at(0.0007) - at(-0.0007)) / 0.0014;
        assert!(
            (numeric - f64::from(analytic)).abs() < 0.002 + 0.002 * f64::from(analytic).abs(),
            "input {id}, analytic {analytic}, numeric {numeric}"
        );
    }
    assert!(dw[..BINS].iter().any(|v| v.abs() > 0.01));
    assert!(dh.iter().any(|v| v.abs() > 0.0001));
}
