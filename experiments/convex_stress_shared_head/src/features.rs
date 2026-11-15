//! A declared ordinal/quadratic bias, fixed during complete-head CE training.
//! CM51 normalizes signed rows; numerical Fourier products are audited separately.
use crate::*;

pub fn point(n: usize, cell: usize) -> [f64; 2] {
    let scale = 2. / (n - 1) as f64;
    [
        scale * (cell / n) as f64 - 1.,
        scale * (cell % n) as f64 - 1.,
    ]
}
pub fn row(n: usize, cell: usize) -> ([f64; STATES], f64) {
    let [u, v] = point(n, cell);
    let raw = [1., u, v, u * u, u * v, v * v, 0., 0.];
    let scale = raw.iter().map(|a| a.abs()).sum::<f64>().max(1.);
    (raw.map(|a| a / scale), scale)
}
pub fn phase(i: usize, f: usize) -> C {
    C::from_polar(
        1. / (STATES as f64).sqrt(),
        std::f64::consts::TAU * ((i * f) % STATES) as f64 / STATES as f64,
    )
}
pub fn encode(n: usize, cell: usize) -> [C; STATES] {
    let (weights, _) = row(n, cell);
    std::array::from_fn(|f| (0..STATES).map(|i| weights[i] * phase(i, f)).sum())
}
/// An independent inverse transform of the complete learned head. This
/// verifies arbitrary post-CE coefficients, not only initializer templates.
pub fn polynomial(model: &Model, block: usize, class: usize) -> [f64; STATES] {
    std::array::from_fn(|i| {
        (0..STATES)
            .map(|f| {
                let p = phase(i, f);
                let at = (block * WIDTH + 2 * f) * model.input.vocabulary + class;
                p.re * model.coefficients[at]
                    - p.im * model.coefficients[at + model.input.vocabulary]
            })
            .sum()
    })
}
