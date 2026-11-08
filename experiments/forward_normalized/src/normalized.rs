//! Independent-coordinate normalized update from PROTOCOL.md. GPU scratch
//! contains warp energies and [parameter energy, gradient energy, factor,
//! zero-norm update count]; no parameter transfer implements training.
use crate::{
    coordinates as coord,
    domain::{
        Result,
        model::{Arm, H, W},
    },
};
use kernite_cuda::{Buffer, Cuda};
pub const A_COUNT: usize = 2 * H * W;
pub fn alpha(step: usize) -> f64 {
    0.01 * (step as f64 / 10.).min(1.)
}
pub struct Normalizer {
    pub partial: Buffer<f64>,
    pub stats: Buffer<f64>,
    pub count: usize,
}
impl Normalizer {
    pub fn new(cuda: &Cuda, arm: Arm) -> Result<Self> {
        let count = if arm == Arm::Swiglu {
            A_COUNT
        } else {
            A_COUNT / 2
        };
        Ok(Self {
            partial: cuda.alloc(2 * count / 32)?,
            stats: cuda.alloc(4)?,
            count,
        })
    }
    pub fn bytes(&self) -> usize {
        8 * (self.partial.len() + self.stats.len())
    }
}
/// Independent CPU oracle, used only before training. It traverses the
/// owner vector sequentially rather than the GPU's warp reduction tree.
pub fn reference(arm: Arm, p: &[f32], g: &[f32], step: usize) -> (Vec<f32>, [f64; 3]) {
    let mut a = coord::owners(arm, &p.iter().map(|&v| v as f64).collect::<Vec<_>>());
    let gradient = coord::pullback(arm, &g.iter().map(|&v| v as f64).collect::<Vec<_>>());
    let range = coord::groups()[1].1.clone();
    let ps = coord::dot(&a[range.clone()], &a[range.clone()]);
    let gs = coord::dot(&gradient[range.clone()], &gradient[range.clone()]);
    let factor = if ps == 0. || gs == 0. {
        0.
    } else {
        alpha(step) * ps.sqrt() / gs.sqrt()
    };
    let mut expected = p.to_vec();
    if factor != 0. {
        for i in range {
            a[i] = ((a[i] - factor * gradient[i]) as f32) as f64;
        }
        let expanded = coord::expand(arm, &a);
        for i in coord::expanded_groups(arm)[1].1.clone() {
            expected[i] = expanded[i] as f32;
        }
    }
    (expected, [ps, gs, factor])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn warmup_and_relative_norm_are_independent_of_adamw() {
        assert_eq!(alpha(1), 0.001);
        assert_eq!(alpha(10), 0.01);
        assert_eq!(alpha(4000), 0.01);
        let mut p = vec![0.; coord::COUNT];
        let mut g = p.clone();
        let a = coord::groups()[1].1.start;
        p[a] = 3.;
        p[a + 1] = 4.;
        g[a] = 4.;
        g[a + 1] = -3.;
        let (q, stats) = reference(Arm::Swiglu, &p, &g, 10);
        assert_eq!(stats, [25., 25., 0.01]);
        let displacement = ((q[a] - 3.) as f64).hypot((q[a + 1] - 4.) as f64);
        assert!((displacement - 0.05).abs() < 1e-6);
        assert_eq!(reference(Arm::Swiglu, &p, &vec![0.; p.len()], 10).0, p);
    }
}
