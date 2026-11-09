//! Orthogonal class decomposition fixed by observed training-label support.
use serde_json::{Value, json};
pub const NAMES: [&str; 3] = ["active_contrasts", "unused_contrasts", "between_groups"];
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
pub fn energy(a: &[f64]) -> f64 {
    dot(a, a)
}
pub struct Partition {
    pub active: Vec<bool>,
    pub width: usize,
}
impl Partition {
    pub fn new(active: Vec<bool>, width: usize) -> Self {
        assert!(active.iter().any(|&a| a) && active.iter().any(|&a| !a));
        Self { active, width }
    }
    pub fn project(&self, e: &[f64]) -> [Vec<f64>; 3] {
        assert_eq!(e.len(), self.active.len() * self.width);
        let na = self.active.iter().filter(|&&a| a).count() as f64;
        let nu = self.active.len() as f64 - na;
        let mut means = [vec![0.; self.width], vec![0.; self.width]];
        for (c, row) in e.chunks_exact(self.width).enumerate() {
            let group = usize::from(!self.active[c]);
            let n = if group == 0 { na } else { nu };
            for (m, &v) in means[group].iter_mut().zip(row) {
                *m += v / n;
            }
        }
        let mut out = std::array::from_fn(|_| vec![0.; e.len()]);
        for (c, row) in e.chunks_exact(self.width).enumerate() {
            let group = usize::from(!self.active[c]);
            for (k, &v) in row.iter().enumerate() {
                out[group][c * self.width + k] = v - means[group][k];
                out[2][c * self.width + k] =
                    means[group][k] - (na * means[0][k] + nu * means[1][k]) / (na + nu);
            }
        }
        out
    }
    pub fn energies(&self, e: &[f64]) -> [f64; 3] {
        self.project(e).each_ref().map(|v| energy(v))
    }
    pub fn metadata(&self) -> Value {
        let active: Vec<_> = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .map(|(c, _)| c)
            .collect();
        json!({"active":active,"unused_count":self.active.len()-active.len(),"projection_names":NAMES})
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orthogonal_class_parts_have_known_nonzero_energies() {
        let p = Partition::new(vec![true, true, false, false], 2);
        let x = [1., 2., 3., 2., 5., -1., 9., -1.];
        let v = p.project(&x);
        assert_eq!(v[0], [-1., 0., 1., 0., 0., 0., 0., 0.]);
        assert_eq!(v[1], [0., 0., 0., 0., -2., 0., 2., 0.]);
        assert_eq!(v.each_ref().map(|v| energy(v)), [2., 8., 34.]);
        for i in 0..3 {
            for j in 0..i {
                assert_eq!(dot(&v[i], &v[j]), 0.);
            }
        }
        assert_eq!(
            energy(
                &(0..8)
                    .map(|k| v.iter().map(|v| v[k]).sum())
                    .collect::<Vec<_>>()
            ),
            44.
        );
    }
    #[test]
    fn unused_partial_ce_gradient_is_dissipative() {
        let p = Partition::new(vec![true, true, false, false], 2);
        let e = [1., 0.5, -2., 1., 3., -1., -1., 2.];
        let (_, g) = crate::gradient::loss_gradient(&[2., -1., -0.5, 3.], &[0, 1], &e, 2, 4);
        let ep = p.project(&e);
        let gp = p.project(&g);
        assert!(dot(&ep[1], &gp[1]) > 0.1);
        let next: Vec<_> = e.iter().zip(&g).map(|(e, g)| e - 0.01 * g).collect();
        assert!(p.energies(&next)[1] < energy(&ep[1]));
    }
}
