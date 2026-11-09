//! EM5: full-parameter warm-Adam tubes, from GradientEvidence.warm_adam_tube.
//! Exact-real formulas evaluated in FP64; no CUDA rounding guarantee is implied.
//! Unlike future_forcing/forecast.rs, every model coordinate is retained.
use crate::encoder::{Caps, Shape, norm};
use std::ops::Range;

const CONE: f64 = 19.0 / 14.0;
const MEMORY: f64 = 81.0 / 95.0;

/// A saved optimizer state; construction never receives later gradients.
pub struct Checkpoint<'a> {
    pub parameters: &'a [f64],
    pub first: &'a [f64],
    pub second: &'a [f64],
    pub step: usize,
}

impl Checkpoint<'_> {
    fn block_radius(&self, indices: Range<usize>, rates: &[f64], decay: f64) -> f64 {
        let initial_norm = norm(&self.parameters[indices.clone()]);
        let dim = indices.len() as f64;
        let ratios: f64 = indices
            .map(|i| {
                let (m, v) = (self.first[i], self.second[i]);
                assert!(v > 0.0 || m == 0.0, "zero variance with nonzero moment");
                if v == 0.0 { 0.0 } else { m * m / v }
            })
            .sum();
        assert!(ratios.is_finite());
        let (mut scale, mut radius) = (1.0, 0.0);
        for (h, &eta) in rates.iter().enumerate() {
            let retention = 1.0 - eta * decay;
            let memory = MEMORY.powf((h + 1) as f64);
            // Convex combination avoids subtraction of nearly equal cone sums.
            let cone_sum = memory * ratios + (1.0 - memory) * CONE * dim;
            let time = (self.step + h + 1) as f64;
            let bias = (1.0 - 0.95_f64.powf(time)).sqrt() / (1.0 - 0.9_f64.powf(time));
            radius = retention * radius + eta * cone_sum.sqrt() * bias;
            scale *= retention;
        }
        // Move the center from the damped checkpoint back to the checkpoint.
        (1.0 - scale).abs() * initial_norm + radius
    }

    /// Per-row and per-matrix displacement caps for the full plain encoder.
    /// The known future schedule is allowed; future parameters/gradients are not.
    /// Tied expanded entries may repeat: counting each copy is conservative.
    pub fn motion(&self, shape: Shape, rates: &[f64], decay: f64) -> Caps {
        shape.validate();
        assert_eq!(self.parameters.len(), shape.count());
        assert_eq!(self.first.len(), shape.count());
        assert_eq!(self.second.len(), shape.count());
        assert!(
            self.parameters
                .iter()
                .chain(self.first)
                .chain(self.second)
                .all(|x| x.is_finite())
        );
        assert!(self.second.iter().all(|x| *x >= 0.0));
        assert!(decay.is_finite() && decay >= 0.0);
        assert!(
            rates
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0 && x * decay <= 1.0)
        );
        Caps {
            rows: (0..shape.classes)
                .map(|c| self.block_radius(shape.row(c), rates, decay))
                .collect(),
            gate: self.block_radius(shape.gate(), rates, decay),
            value: self.block_radius(shape.value(), rates, decay),
            output: self.block_radius(shape.output(), rates, decay),
        }
    }
}
