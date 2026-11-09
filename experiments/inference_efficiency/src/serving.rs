//! Host top-1 serving. Inputs and cache keys are operands; targets never enter.
use crate::{
    bounds::Bound,
    device::{self, Weights},
    domain::Result,
    gather_gpu,
    infrastructure::engine::Engine,
    inputs::require,
    io::Transfers,
};
use kernite_cuda::Buffer;
pub const METHODS: [&str; 4] = ["source", "cached-weights", "reuse", "tight-reuse"];
pub fn winner(logits: &[f32]) -> usize {
    (1..logits.len()).fold(0, |best, c| if logits[c] > logits[best] { c } else { best })
}
pub fn predictions(logits: &[f32]) -> Vec<usize> {
    logits.chunks_exact(97).map(winner).collect()
}
pub struct Decision {
    pub predictions: Vec<usize>,
    pub accepted: Vec<bool>,
    pub fallback: Vec<usize>,
}
pub fn decide(pairs: &[[u32; 2]], logits: &[f32], bounds: &Bound) -> Result<Decision> {
    device::valid_pairs(pairs)?;
    require(
        logits.len() == pairs.len() * 97 && logits.iter().all(|x| x.is_finite()),
        "invalid approximation",
    )?;
    let mut out = Decision {
        predictions: Vec::with_capacity(pairs.len()),
        accepted: vec![],
        fallback: vec![],
    };
    for (i, pair) in pairs.iter().enumerate() {
        let q = (97 * pair[0] + pair[1]) as usize;
        let row = &logits[i * 97..(i + 1) * 97];
        let w = winner(row);
        let accepted = (0..97).filter(|&c| c != w).all(|c| {
            let gap = row[w] as f64 - row[c] as f64;
            let limit = bounds.contrasts[w * 97 + c] * (bounds.radii[q] + 2e-3) + 4e-3;
            gap > limit + 1e-8 * (1. + gap.abs().max(limit.abs()))
        });
        out.predictions.push(w);
        out.accepted.push(accepted);
        if !accepted {
            out.fallback.push(i);
        }
    }
    Ok(out)
}
pub struct Service<'a, 'c> {
    pub engine: &'a mut Engine,
    pub version: &'a str,
    pub old_version: &'a str,
    pub weights: &'a Weights,
    pub features: &'a Buffer<f32>,
    pub bounds: &'a [Bound; 2],
    pub gather: &'a gather_gpu::kernels::Kernels,
    pub io: &'a Transfers<'c>,
}
impl Service<'_, '_> {
    pub fn approximate(
        &mut self,
        pairs: &[[u32; 2]],
        method: usize,
    ) -> Result<(Vec<f32>, Decision)> {
        require(method < 2, "invalid reuse method")?;
        let b = &self.bounds[method];
        device::same_version(&b.from, self.old_version)?;
        device::same_version(&b.to, self.version)?;
        let logits = self.engine.reused_logits(
            pairs,
            self.features,
            self.weights,
            self.version,
            self.gather,
            self.io,
        )?;
        let decision = decide(pairs, &logits, b)?;
        Ok((logits, decision))
    }
    pub fn finish(&mut self, pairs: &[[u32; 2]], mut d: Decision) -> Result<Vec<usize>> {
        if !d.fallback.is_empty() {
            let missing: Vec<_> = d.fallback.iter().map(|&i| pairs[i]).collect();
            let exact =
                self.engine
                    .inference(&missing, Some(self.weights), self.version, self.io)?;
            for (&i, p) in d.fallback.iter().zip(predictions(&exact)) {
                d.predictions[i] = p;
            }
        }
        Ok(d.predictions)
    }
    pub fn request(&mut self, pairs: &[[u32; 2]], method: usize) -> Result<Vec<usize>> {
        match method {
            0 | 1 => Ok(predictions(&self.engine.inference(
                pairs,
                if method == 0 {
                    None
                } else {
                    Some(self.weights)
                },
                self.version,
                self.io,
            )?)),
            2 | 3 => {
                let (_, decision) = self.approximate(pairs, method - 2)?;
                self.finish(pairs, decision)
            }
            _ => Err("invalid serving method".into()),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties_and_inconclusive_bounds_require_exact_fallback_without_targets() {
        let bound = Bound {
            from: "a".into(),
            to: "b".into(),
            radii: vec![0.; 9409],
            contrasts: vec![1.; 97 * 97],
            seconds: 0.,
            gram_seconds: 0.,
            factors: [1., 1.],
        };
        let mut logits = vec![0.; 97 * 3];
        logits[8] = 1.;
        logits[97 + 4] = 0.005;
        let d = decide(&[[0, 0], [96, 96], [1, 2]], &logits, &bound).unwrap();
        assert_eq!(d.predictions, [8, 4, 0]);
        assert_eq!(d.accepted, [true, false, false]);
        assert_eq!(d.fallback, [1, 2]);
        logits[0] = f32::NAN;
        assert!(decide(&[[0, 0], [96, 96], [1, 2]], &logits, &bound).is_err());
    }
}
