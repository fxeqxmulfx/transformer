//! CM18 `certified_restriction`: spectral endpoints must keep certified rows.
//! A uniform mixture would break the equalities. Instead use the symmetric
//! phase anchor from the checked certificate. Both endpoints of a proposed
//! segment have the certified rows, and every returned table is checked for
//! probability rows, the equalities and numerical spectral feasibility.
//! This finite floating-point search is not an interval feasibility proof.
use crate::{Certificate, Model, Result, Training, Verified, check};
use convex_stress_paired::projection::{Norms, Projector};

pub const SPECTRAL_SLACK: f64 = 1e-10;
pub const EQUALITY_SLACK: f64 = 1e-12;

pub struct Endpoint {
    pub model: Model,
    /// Weight on the solver's projected iterate; the remainder is the anchor.
    pub iterate_weight: f64,
    pub norms: Vec<Norms>,
    /// Norm passes in this snapshot, separate from update and setup costs.
    pub norm_evaluations: u64,
}

pub struct EndpointBuilder {
    projector: Projector,
    anchor: Model,
    anchor_norms: Vec<Norms>,
    verified: Verified,
}

fn meets_budgets(model: &Model, norms: &[Norms]) -> bool {
    norms.iter().all(|n| {
        n.operator.is_finite()
            && n.nuclear.is_finite()
            && n.operator <= model.radius * (1. + SPECTRAL_SLACK)
            && n.nuclear <= model.nuclear_budget * (1. + SPECTRAL_SLACK)
    })
}

/// Check actual decoded coordinates, with no post-hoc replacement of predictions.
pub fn check_equalities(model: &Model, verified: &Verified) -> Result<()> {
    model.validate_shape()?;
    if verified.labels().len() != model.shape.cells() {
        return Err("certificate equality table has the wrong input side".into());
    }
    for (cell, label) in verified.labels().iter().enumerate() {
        if let Some(label) = label {
            let start = cell * model.shape.vocab;
            if model.output[start..start + model.shape.vocab]
                .iter()
                .enumerate()
                .any(|(c, &p)| (p - f64::from(c == *label)).abs() > EQUALITY_SLACK)
            {
                return Err("endpoint violates a certified output equality".into());
            }
        }
    }
    Ok(())
}

impl EndpointBuilder {
    pub fn new(training: &Training, certificate: &Certificate) -> Result<Self> {
        let verified = check(training, certificate)?;
        let anchor = verified.anchor(training)?;
        let projector = Projector::new(training.shape())?;
        let anchor_norms = anchor.validate(&projector)?;
        check_equalities(&anchor, &verified)?;
        if !meets_budgets(&anchor, &anchor_norms) {
            return Err("phase anchor failed the numerical spectral check".into());
        }
        Ok(Self {
            projector,
            anchor,
            anchor_norms,
            verified,
        })
    }

    pub fn projector(&self) -> &Projector {
        &self.projector
    }

    pub fn anchor(&self) -> &Model {
        &self.anchor
    }

    pub fn verified(&self) -> &Verified {
        &self.verified
    }

    fn model(&self, half: Vec<f64>) -> Result<Model> {
        let model = Model::new(
            self.anchor.shape,
            self.anchor.decoder.clone(),
            self.anchor.radius,
            self.anchor.nuclear_budget,
            half,
        )?;
        check_equalities(&model, &self.verified)?;
        Ok(model)
    }

    /// Both segment endpoints must already satisfy the equalities. Re-copying
    /// their common fixed rows removes only floating-point interpolation error;
    /// the norms are computed afterwards on these actual returned coordinates.
    fn blend(&self, half: &[f64], alpha: f64) -> Result<Model> {
        let mut out: Vec<_> = half
            .iter()
            .zip(&self.anchor.latent_half)
            .map(|(&x, &a)| alpha * x + (1. - alpha) * a)
            .collect();
        let orbits = self.anchor.shape.orbits;
        for (cell, label) in self.verified.labels().iter().enumerate() {
            if label.is_some() {
                let row = cell * orbits..(cell + 1) * orbits;
                out[row.clone()].copy_from_slice(&self.anchor.latent_half[row]);
            }
        }
        self.model(out)
    }

    /// Bisection on a segment from a checked feasible point. A returned model
    /// always has its own norm measurements; no positive weight is guaranteed.
    /// When rounds = 0 and the iterate is infeasible, return the checked anchor.
    pub fn build(&self, half: &[f64], rounds: usize) -> Result<Endpoint> {
        if rounds > 40 {
            return Err("endpoint repair exceeds the declared 40-round work cap".into());
        }
        let candidate = self.model(half.to_vec())?;
        let norms = self.projector.norms(&candidate.latent_half)?;
        if meets_budgets(&candidate, &norms) {
            return Ok(Endpoint {
                model: candidate,
                iterate_weight: 1.,
                norms,
                norm_evaluations: 1,
            });
        }
        let mut accepted = Endpoint {
            model: self.anchor.clone(),
            iterate_weight: 0.,
            norms: self.anchor_norms.clone(),
            norm_evaluations: 1,
        };
        let mut upper = 1.;
        for _ in 0..rounds {
            let alpha = (accepted.iterate_weight + upper) / 2.;
            let model = self.blend(half, alpha)?;
            let norms = self.projector.norms(&model.latent_half)?;
            accepted.norm_evaluations += 1;
            if meets_budgets(&model, &norms) {
                accepted.model = model;
                accepted.iterate_weight = alpha;
                accepted.norms = norms;
            } else {
                upper = alpha;
            }
        }
        Ok(accepted)
    }
}
