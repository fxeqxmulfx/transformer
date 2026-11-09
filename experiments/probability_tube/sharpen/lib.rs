//! PA1-PA2 in review/flat_spectrum.md: pointwise complex-product amplification.
//! The old source graph and the original bound remain unchanged. Only Gram
//! construction expands operations outward; full propagation remains FP64.
#[path = "../../forecast_tightening/src/matrix.rs"]
pub mod matrix;
use probability_tube::encoder::{self, Caps, EPSILON, Shape, norm};

/// Maximum logical-pair amplitude in the source split real/imaginary layout.
pub fn pair_amplitude(x: &[f64]) -> f64 {
    assert!(!x.is_empty() && x.len().is_multiple_of(2));
    assert!(x.iter().all(|v| v.is_finite()));
    let h = x.len() / 2;
    (0..h)
        .map(|k| matrix::norm_upper(&[x[k], x[k + h]]))
        .fold(0., f64::max)
}
/// PA1's finite product radius, including both moving-factor interaction.
pub fn product_radius(g: &[f64], u: &[f64], rg: f64, ru: f64) -> f64 {
    assert_eq!(g.len(), u.len());
    assert!([rg, ru].iter().all(|v| v.is_finite() && *v >= 0.));
    (rg * pair_amplitude(u) + pair_amplitude(g) * ru + rg * ru) / 2f64.sqrt()
}
fn rms_radius(old_norm: f64, width: usize, r: f64) -> f64 {
    let width = width as f64;
    let s = (old_norm * old_norm + width * EPSILON).sqrt();
    let mut result = (2. * width.sqrt()).min(2. * width.sqrt() * r / s);
    if r < s {
        result = result.min(width.sqrt() * r / (s * (s - r)).sqrt());
    }
    result
}
/// Cached-norm form of PA1. Norm order: gather, input, inner, gate, value,
/// product, branch, residual, final feature. Returned order matches EM4 stages.
pub fn radii(
    shape: Shape,
    n: &[f64; 9],
    pair: [f64; 2],
    caps: &Caps,
    input: [usize; 2],
    operators: [f64; 3],
) -> [f64; 10] {
    shape.validate();
    assert_eq!(caps.rows.len(), shape.classes);
    assert!(
        n.iter()
            .chain(&pair)
            .chain(&operators)
            .chain(&caps.rows)
            .chain([&caps.gate, &caps.value, &caps.output])
            .all(|v| v.is_finite() && *v >= 0.)
    );
    let r0 = caps.rows[input[0]] + caps.rows[input[1]];
    let r1 = rms_radius(n[0], shape.width, r0);
    let r2 = rms_radius(n[1], shape.width, r1);
    let rg = (operators[0] + caps.gate) * r2 + caps.gate * n[2];
    let ru = (operators[1] + caps.value) * r2 + caps.value * n[2];
    let rh = (rg * pair[1] + pair[0] * ru + rg * ru) / 2f64.sqrt();
    let rb = (operators[2] + caps.output) * rh + caps.output * n[5];
    let rn = rms_radius(n[6], shape.width, rb);
    let residual = r1 + rn;
    [
        r0,
        r1,
        r2,
        rg,
        ru,
        rh,
        rb,
        rn,
        residual,
        rms_radius(n[7], shape.width, residual),
    ]
}
/// Checkpoint-only matrix preparation. No future-state argument is accepted.
pub struct Checkpoint<'a> {
    shape: Shape,
    parameters: &'a [f64],
    pub operators: [matrix::Bound; 3],
}
impl<'a> Checkpoint<'a> {
    pub fn new(shape: Shape, parameters: &'a [f64]) -> Self {
        shape.validate();
        assert_eq!(parameters.len(), shape.count());
        let operators = [
            matrix::bound(&parameters[shape.gate()], shape.hidden(), shape.width),
            matrix::bound(&parameters[shape.value()], shape.hidden(), shape.width),
            matrix::bound(&parameters[shape.output()], shape.width, shape.hidden()),
        ];
        Self {
            shape,
            parameters,
            operators,
        }
    }
    pub fn evaluate(&self, a: usize, b: usize, caps: &Caps) -> Forecast<'a> {
        let base = encoder::forecast(self.shape, self.parameters, a, b, caps);
        let n = [0, 1, 2, 3, 4, 5, 6, 8, 9].map(|i| norm(&base.stages[i].old));
        let pair = [
            pair_amplitude(&base.stages[3].old),
            pair_amplitude(&base.stages[4].old),
        ];
        let mut radii = radii(
            self.shape,
            &n,
            pair,
            caps,
            [a, b],
            self.operators.map(|m| m.factor),
        );
        radii[9] = radii[9].min(base.feature().radius);
        Forecast {
            base,
            radii,
            pair_amplitudes: pair,
            shape: self.shape,
            readout: &self.parameters[..self.shape.embeddings()],
            row_motion: caps.rows.clone(),
        }
    }
}
pub struct Forecast<'a> {
    pub base: encoder::Forecast,
    pub radii: [f64; 10],
    pub pair_amplitudes: [f64; 2],
    shape: Shape,
    readout: &'a [f64],
    row_motion: Vec<f64>,
}
impl Forecast<'_> {
    pub fn contrast_budget(&self, u: usize, c: usize, execution_change: f64) -> f64 {
        if u == c {
            return 0.;
        }
        let contrast: Vec<_> = self.readout[self.shape.row(u)]
            .iter()
            .zip(&self.readout[self.shape.row(c)])
            .map(|(a, b)| a - b)
            .collect();
        probability_tube::contrast_radius(
            norm(&contrast),
            norm(&self.base.feature().old),
            self.row_motion[u] + self.row_motion[c],
            self.radii[9],
            execution_change,
        )
    }
    pub fn probability_ceiling(
        &self,
        u: usize,
        old_execution: &[f64],
        execution_change: f64,
    ) -> f64 {
        assert_eq!(old_execution.len(), self.shape.classes);
        let logits: Vec<_> = self
            .base
            .old_logits
            .iter()
            .zip(old_execution)
            .map(|(x, e)| x + e)
            .collect();
        let budgets: Vec<_> = (0..self.shape.classes)
            .map(|c| self.contrast_budget(u, c, execution_change))
            .collect();
        probability_tube::rival_ceiling(&logits, u, &budgets)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dispersed_old_pairs_remove_dimension_from_linear_amplification() {
        let mut g = vec![0.; 64];
        g[..32].fill(1.);
        let exact = 0.25 / 2f64.sqrt();
        // Perturb one real gate coordinate only: the inequality is attained.
        assert!((product_radius(&g, &g, 0.25, 0.) - exact).abs() < 1e-14);
        assert!(product_radius(&g, &g, 0.25, 0.) < 0.18 * 0.25 * norm(&g) / 2f64.sqrt());
        // With both old factors zero the mixed term is the entire product.
        assert!((product_radius(&[0.; 4], &[0.; 4], 2., 3.) - 6. / 2f64.sqrt()).abs() < 1e-14);
    }
}
