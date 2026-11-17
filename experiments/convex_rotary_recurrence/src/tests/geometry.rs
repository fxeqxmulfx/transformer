use crate::{
    C,
    atom::{Atom, Family, State, operator},
    model::Model,
};
fn fixture(family: Family) -> Atom {
    let mut a = Atom::zero(3, family);
    a.initial = [C::new(0.2, 0.7), C::new(-0.3, 0.1)];
    a.transition = (0..3)
        .map(|k| {
            let s = (k + 1) as f64 / 4.;
            [
                [C::new(0.3 * s, -0.4), C::new(0.2, 0.1 * s)],
                [C::new(-0.1, 0.2 * s), C::new(0.3, -0.1 * s)],
            ]
        })
        .collect();
    a.output = vec![
        [C::new(0.3, 0.2), C::new(-0.2, 0.1)],
        [C::new(0.1, -0.4), C::new(0.1, 0.2)],
        [C::new(-0.3, 0.1), C::new(0.4, 0.1)],
    ];
    if family == Family::Rotary {
        a.frequency = Some([0.31, -0.73]);
    }
    a
}
fn rotate(z: &mut [f64; 4], theta: [f64; 2], position: f64) {
    for j in 0..2 {
        let (s, c) = (theta[j] * position).sin_cos();
        let (x, y) = (z[2 * j], z[2 * j + 1]);
        z[2 * j] = c * x - s * y;
        z[2 * j + 1] = s * x + c * y;
    }
}
#[test]
fn full_complex_execution_matches_independent_real_four_matrix_product() {
    for family in [Family::Plain, Family::Rotary] {
        let a = fixture(family);
        a.validate(3, family).unwrap();
        let prefix = [2, 0, 2, 1, 1];
        let mut z = [
            a.initial[0].re,
            a.initial[0].im,
            a.initial[1].re,
            a.initial[1].im,
        ];
        for (t, &o) in prefix.iter().enumerate() {
            if let Some(theta) = a.frequency {
                rotate(&mut z, theta, -(t as f64));
            }
            let mut matrix = [[0.; 4]; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let u = a.transition[o as usize][i][j];
                    matrix[2 * i][2 * j] = u.re;
                    matrix[2 * i][2 * j + 1] = -u.im;
                    matrix[2 * i + 1][2 * j] = u.im;
                    matrix[2 * i + 1][2 * j + 1] = u.re;
                }
            }
            z = std::array::from_fn(|i| matrix[i].iter().zip(z).map(|(u, v)| u * v).sum());
            if let Some(theta) = a.frequency {
                rotate(&mut z, theta, t as f64);
            }
        }
        for (v, u) in a.complex(&prefix).unwrap().iter().zip(&a.output) {
            let re = u[0].re * z[0] - u[0].im * z[1] + u[1].re * z[2] - u[1].im * z[3];
            let im = u[0].im * z[0] + u[0].re * z[1] + u[1].im * z[2] + u[1].re * z[3];
            assert!((v.re - re).abs() < 1e-15 && (v.im - im).abs() < 1e-15);
        }
        let mut real = a.clone();
        for z in real
            .initial
            .iter_mut()
            .chain(real.transition.iter_mut().flatten().flatten())
            .chain(real.output.iter_mut().flatten())
        {
            z.im = 0.;
        }
        assert_ne!(real.complex(&prefix).unwrap(), a.complex(&prefix).unwrap());
        assert!(a.state(&[-1]).is_err());
        assert!(a.state(&[3]).is_err());
    }
}
#[test]
fn norm_budgets_and_signed_queries_are_actual_complete_models() {
    for family in [Family::Plain, Family::Rotary] {
        let mut m = Model::zero(3, 16., family).unwrap();
        m.mix(fixture(family), 0.8).unwrap();
        let h = 1e-4;
        m.shrink_for_queries(h).unwrap();
        for prefix in [vec![], vec![0, 1, 2], vec![2; 1000]] {
            let base = m.logits(&prefix).unwrap();
            assert!(base.iter().all(|x| x.abs() <= 16. + 1e-10));
            for c in 0..3 {
                for sign in [-1., 1.] {
                    let mut q = m.clone();
                    q.weights[0] -= h;
                    q.atoms.push(Atom::bias(3, family, c, sign).unwrap());
                    q.weights.push(h);
                    q.validate().unwrap();
                    for (j, (actual, centre)) in
                        q.logits(&prefix).unwrap().iter().zip(&base).enumerate()
                    {
                        assert!(
                            (actual - centre - if c == j { sign * h } else { 0. }).abs() < 1e-13
                        );
                    }
                }
            }
        }
        let mut invalid = fixture(family);
        invalid.transition[0][1] = [C::new(0.6, 0.); 2];
        assert!(m.mix(invalid, 0.1).is_err());
        let mut wrong = fixture(family);
        wrong.frequency = if family == Family::Plain {
            Some([0.; 2])
        } else {
            None
        };
        assert!(m.mix(wrong, 0.1).is_err());
    }
}
#[test]
fn opaque_renaming_and_zero_frequency_preserve_actual_execution() {
    let rename = [2, 0, 1];
    for family in [Family::Plain, Family::Rotary] {
        let a = fixture(family);
        let mut b = a.clone();
        for (i, &j) in rename.iter().enumerate() {
            b.transition[j] = a.transition[i];
            b.output[j] = a.output[i];
        }
        let p = [0, 2, 0, 1];
        let q: Vec<_> = p.iter().map(|&o| rename[o as usize] as i64).collect();
        let x = a.complex(&p).unwrap();
        let y = b.complex(&q).unwrap();
        for c in 0..3 {
            assert_eq!(x[c], y[rename[c]]);
        }
    }
    let a = fixture(Family::Plain);
    let mut b = a.clone();
    b.frequency = Some([0.; 2]);
    assert_eq!(
        a.complex(&[2, 1, 0, 2]).unwrap(),
        b.complex(&[2, 1, 0, 2]).unwrap()
    );
}
#[test]
fn rotary_positions_distinguish_order_even_when_content_commutes() {
    // CM98 existence witness is confined to tests, never a training initializer.
    let mut a = Atom::identity_start(2, Family::Rotary);
    a.frequency = Some([std::f64::consts::FRAC_PI_2, 0.]);
    a.initial = [C::new(1., 0.), C::default()];
    a.transition[0] = [
        [C::default(), C::new(1., 0.)],
        [C::new(1., 0.), C::default()],
    ];
    a.output = vec![
        [C::default(), C::new(1., 0.)],
        [C::default(), C::new(0., 1.)],
    ];
    let z: State = [C::new(0.3, 0.4), C::new(-0.2, 0.7)];
    assert_eq!(
        operator(a.transition[0], operator(a.transition[1], z)),
        operator(a.transition[1], operator(a.transition[0], z))
    );
    let mut m = Model::zero(2, 1., Family::Rotary).unwrap();
    m.mix(a.clone(), 1.).unwrap();
    assert_eq!(m.next_object(&[0, 1]).unwrap(), 0);
    assert_eq!(m.next_object(&[1, 0]).unwrap(), 1);
    let first = a.complex(&[0, 1]).unwrap();
    let second = a.complex(&[1, 0]).unwrap();
    assert!((first[0].re - first[1].re - 1.).abs() < 1e-15);
    assert!((second[1].re - second[0].re - 1.).abs() < 1e-15);
    a.frequency = Some([0.; 2]);
    assert_eq!(a.complex(&[0, 1]).unwrap(), a.complex(&[1, 0]).unwrap());
}
