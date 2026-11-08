//! AI1-AI3: source 14cff3b5, engine::update/product::moment/ops::adamw.
//! FP64 evaluation on saved FP32 buffers; no optimizer step or bitwise CUDA claim.
use crate::{geometry, gradient::energy};
use serde_json::{Value, json};

pub struct Direction {
    pub mean: Vec<f64>,
    pub diagonal: Vec<f64>,
    pub dmax: f64,
}
pub fn corrected(first: &[f64], second: &[f64], step: usize) -> Direction {
    assert!(step > 0 && first.len() == second.len());
    let b1 = 1. - 0.9f64.powf(step as f64);
    let b2root = (1. - 0.98f64.powf(step as f64)).sqrt();
    let mean = first.iter().map(|m| m / b1).collect();
    let diagonal: Vec<_> = second
        .iter()
        .map(|&v| {
            assert!(v.is_finite() && v >= 0.);
            1. / (v.sqrt() / b2root + 1e-8)
        })
        .collect();
    let dmax = diagonal.iter().copied().fold(0., f64::max);
    Direction {
        mean,
        diagonal,
        dmax,
    }
}
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
fn multiply(x: &[f64], y: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(a, b)| a * b).collect()
}
fn subtract(x: &[f64], y: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}
pub fn audit(
    d: &Direction,
    g: &[f64],
    q: &[f64],
    width: usize,
    loss: f64,
    feature_energy: f64,
) -> Value {
    let projected = geometry::project(&multiply(&d.diagonal, &d.mean), q, width);
    let a = multiply(&d.diagonal, &geometry::project(g, q, width));
    let b = subtract(&geometry::project(&multiply(&d.diagonal, g), q, width), &a);
    let c = geometry::project(&multiply(&d.diagonal, &subtract(&d.mean, g)), q, width);
    let error = (0..g.len())
        .map(|j| (projected[j] - a[j] - b[j] - c[j]).abs())
        .fold(0., f64::max);
    let scale = projected
        .iter()
        .chain(&a)
        .chain(&b)
        .chain(&c)
        .map(|v| v.abs())
        .fold(0., f64::max);
    let (actual, ea, eb, ec) = (energy(&projected), energy(&a), energy(&b), energy(&c));
    let cross = 2. * (dot(&a, &b) + dot(&a, &c) + dot(&b, &c));
    let bound = 3. * (ea + eb + ec);
    let ce_term = d.dmax.powi(2) * 2. * loss * feature_energy;
    let objective_bound = 3. * (ce_term + eb + ec);
    let dqm = multiply(&d.diagonal, &geometry::project(&d.mean, q, width));
    let mix = subtract(&projected, &dqm);
    let e_error = (actual - (ea + eb + ec + cross)).abs();
    json!({"projected_saved_direction_energy":actual,"scaled_projected_gradient_energy":ea,
        "gradient_commutator_energy":eb,"moment_discrepancy_energy":ec,"cross_terms":cross,
        "term_energy_sum":ea+eb+ec,"three_term_energy_bound":bound,"objective_energy_bound":objective_bound,
        "loss_controlled_term_bound":ce_term,"dmax":d.dmax,
        "relative_cancellation":if ea+eb+ec>0. {Some(actual/(ea+eb+ec))}else{None},
        "diagonal_projected_moment_energy":energy(&dqm),"moment_commutator_energy":energy(&mix),
        "moment_commutator_cross":2.*dot(&dqm,&mix),
        "identity_max_error":error,"identity_holds":error<=1e-8*(1.+scale),
        "energy_identity_error":e_error,"energy_identity_holds":e_error<=1e-8*(1.+actual+ea+eb+ec),
        "three_term_bound_holds":actual<=bound+1e-8*(1.+bound),
        "loss_term_bound_holds":ea<=ce_term+1e-8*(1.+ce_term),
        "objective_bound_holds":actual<=objective_bound+1e-8*(1.+objective_bound)})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_buffer_factorization_matches_independent_scalar_updates() {
        let (mut m, mut v, mut p) = (0f64, 0f64, 2f64);
        for (i, g) in [-0.5f64, 2., -1., 0.1].iter().enumerate() {
            m = 0.9 * m + 0.1 * g;
            v = 0.98 * v + 0.02 * g * g;
            let t = i + 1;
            let lr = 0.001 * (t as f64 / 10.).min(1.);
            let direct = (1. - lr) * p
                - lr * (m / (1. - 0.9f64.powi(t as i32)))
                    / ((v / (1. - 0.98f64.powi(t as i32))).sqrt() + 1e-8);
            let d = corrected(&[m], &[v], t);
            let factored = (1. - lr) * p - lr * d.diagonal[0] * d.mean[0];
            assert!((direct - factored).abs() < 1e-14);
            p = direct;
        }
        let d = corrected(&[1.], &[0.], 1);
        assert_eq!(d.diagonal[0], 1e8);
    }
    #[test]
    fn rotated_nullspace_mixes_but_an_isotropic_scale_commutes() {
        let q = [0.8, -0.4, -0.4, 0.2];
        let g = [-0.5, -1.];
        let d = Direction {
            mean: g.to_vec(),
            diagonal: vec![2. / 3., 0.5],
            dmax: 2. / 3.,
        };
        let a = audit(&d, &g, &q, 2, 2., 5.);
        assert!((a["gradient_commutator_energy"].as_f64().unwrap() - 1. / 180.).abs() < 1e-14);
        assert!(a["scaled_projected_gradient_energy"].as_f64().unwrap() < 1e-28);
        assert_eq!(a["moment_discrepancy_energy"], 0.);
        let iso = Direction {
            mean: g.to_vec(),
            diagonal: vec![2.; 2],
            dmax: 2.,
        };
        assert_eq!(
            audit(&iso, &g, &q, 2, 2., 5.)["gradient_commutator_energy"],
            0.
        );
    }
}
