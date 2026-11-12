//! CM19 branch_endpoint and certified_rows_blend, proved at 7d299b04.
//! A feasible phase anchor preserves certified outputs without fixing their
//! ambiguous latent states. No interval arithmetic or positive weight guarantee.
use crate::*;
use projection::{Norms, Projector, meets};
use propagation::Verified;

pub struct Endpoint {
    pub model: Model,
    pub iterate_weight: f64,
    pub norms: Vec<Norms>,
    pub norm_evaluations: u64,
}

pub struct Builder {
    pub(crate) verified: Verified,
    pub(crate) anchor: Model,
    pub(crate) projector: Projector,
    anchor_norms: Vec<Norms>,
}

impl Builder {
    pub fn new(verified: &Verified, decoder: usize) -> Result<Self> {
        let anchor = Model::anchor(verified, decoder)?;
        let mut projector = Projector::new(anchor.shape)?;
        let anchor_norms = projector.norms(&anchor.latent)?;
        if !meets(anchor.shape, &anchor_norms) {
            return Err("phase anchor failed the numerical spectral audit".into());
        }
        Ok(Self {
            verified: verified.clone(),
            anchor,
            projector,
            anchor_norms,
        })
    }
    fn model(&self, point: Vec<f64>) -> Result<Model> {
        let result = Model::new(self.anchor.shape, self.anchor.decoder.clone(), point)?;
        result.certified(&self.verified)?;
        Ok(result)
    }
    pub fn build(&mut self, point: &[f64], rounds: usize) -> Result<Endpoint> {
        if rounds > 40 {
            return Err("at most 40 endpoint repair rounds are allowed".into());
        }
        if point.len() != self.anchor.shape.entries() {
            return Err("incorrect endpoint table length".into());
        }
        let candidate = self.model(point.to_vec())?;
        let norms = self.projector.norms(&candidate.latent)?;
        if meets(candidate.shape, &norms) {
            return Ok(Endpoint {
                model: candidate,
                iterate_weight: 1.,
                norms,
                norm_evaluations: 1,
            });
        }
        let mut result = Endpoint {
            model: self.anchor.clone(),
            iterate_weight: 0.,
            norms: self.anchor_norms.clone(),
            norm_evaluations: 1,
        };
        let mut upper = 1.;
        for _ in 0..rounds {
            let alpha = (result.iterate_weight + upper) * 0.5;
            let mixed = point
                .iter()
                .zip(&self.anchor.latent)
                .map(|(&x, &a)| alpha * x + (1. - alpha) * a)
                .collect();
            // No latent row or decoded prediction is replaced after mixing.
            let model = self.model(mixed)?;
            let norms = self.projector.norms(&model.latent)?;
            result.norm_evaluations += 1;
            if meets(model.shape, &norms) {
                result.model = model;
                result.iterate_weight = alpha;
                result.norms = norms;
            } else {
                upper = alpha;
            }
        }
        Ok(result)
    }
}
