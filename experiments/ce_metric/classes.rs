//! Class projection ported from class_geometry/src/classes.rs at 72f30df3.
//! Only its array operations are needed; no GPU or linear-algebra backend.
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
}
