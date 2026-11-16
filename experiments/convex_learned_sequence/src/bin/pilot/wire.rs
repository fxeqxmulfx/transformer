//! Strict lossless trace serialization and replay through the independent
//! native update verifiers. Float values are stored as their exact bit patterns.

use super::settings::*;
use convex_learned_sequence::Rational;
use convex_learned_sequence::model::forward_learning::{
    CellProbes, DirectionProbe, ForwardEstimate, ForwardPlan, ForwardUpdate,
};
use convex_learned_sequence::model::learning::{MiniUpdate, Record};
use convex_learned_sequence::model::{ComplexModel, ProjectedModel};
use convex_learned_sequence::projection::Projection;
use serde::{Deserialize, Serialize};

fn strings(values: &[Rational]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}
fn rationals(values: &[String]) -> Result<Vec<Rational>> {
    if values.len() > 768 || values.iter().any(|s| s.len() > 4096) {
        return Err("oversized rational array".into());
    }
    values.iter().map(|s| Ok(s.parse()?)).collect()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Certificate {
    point: Vec<String>,
    multipliers: Vec<String>,
}
fn certificates(values: &[Projection]) -> Vec<Certificate> {
    values
        .iter()
        .map(|p| Certificate {
            point: strings(&p.point),
            multipliers: strings(&p.multipliers),
        })
        .collect()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Probe {
    step: String,
    plus: u64,
    minus: u64,
    pairing: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    mask: usize,
    class: usize,
    sign: usize,
    component: usize,
    probes: Vec<Probe>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub mode: String,
    pub index: usize,
    pub epoch: usize,
    pub start: usize,
    pub count: usize,
    pub old_loss: u64,
    pub mixed_loss: Option<u64>,
    pub final_loss: u64,
    pub point: Vec<String>,
    pub mixed: Option<Vec<String>>,
    pub direction: Vec<String>,
    pub proposal: Vec<String>,
    pub certificates: Vec<Certificate>,
    pub cells: Vec<Cell>,
    pub plan: String,
}
impl Step {
    pub fn mini(
        index: usize,
        epoch: usize,
        start: usize,
        count: usize,
        update: &MiniUpdate,
        final_loss: f64,
    ) -> Self {
        Self {
            mode: "mini".into(),
            index,
            epoch,
            start,
            count,
            old_loss: update.gradient.mean_loss.to_bits(),
            mixed_loss: None,
            final_loss: final_loss.to_bits(),
            point: strings(update.projected.model.parameters()),
            mixed: None,
            direction: strings(&update.gradient.coordinates),
            proposal: strings(&update.proposal),
            certificates: certificates(&update.projected.certificates),
            cells: Vec::new(),
            plan: format!("{:?}", update.gradient.plan),
        }
    }
    pub fn forward(
        index: usize,
        epoch: usize,
        start: usize,
        count: usize,
        update: &ForwardUpdate,
    ) -> Self {
        Self {
            mode: "forward".into(),
            index,
            epoch,
            start,
            count,
            old_loss: update.estimate.old_loss.to_bits(),
            mixed_loss: Some(update.estimate.mixed_loss.to_bits()),
            final_loss: update.final_loss.to_bits(),
            point: strings(update.projected.model.parameters()),
            mixed: Some(strings(update.estimate.mixed.parameters())),
            direction: strings(&update.estimate.coordinates),
            proposal: strings(&update.proposal),
            certificates: certificates(&update.projected.certificates),
            cells: update
                .estimate
                .cells
                .iter()
                .map(|c| Cell {
                    mask: c.mask,
                    class: c.class,
                    sign: c.sign,
                    component: c.component,
                    probes: c
                        .directions
                        .iter()
                        .map(|q| Probe {
                            step: q.step.to_string(),
                            plus: q.plus_loss.to_bits(),
                            minus: q.minus_loss.to_bits(),
                            pairing: q.pairing.to_string(),
                        })
                        .collect(),
                })
                .collect(),
            plan: format!("{:?}", update.estimate.plan),
        }
    }
    pub fn replay(&self, old: &ComplexModel, batch: &[Record]) -> Result<ComplexModel> {
        if self.count != batch.len()
            || self.certificates.len() != 192
            || self.direction.len() != 768
            || self.proposal.len() != 768
            || self.point.len() != 768
        {
            return Err("incomplete complete-model trace".into());
        }
        let next = old.with_parameters(rationals(&self.point)?)?;
        let proposal = rationals(&self.proposal)?;
        let direction = rationals(&self.direction)?;
        let certs = self
            .certificates
            .iter()
            .map(|c| {
                Ok(Projection {
                    point: rationals(&c.point)?,
                    multipliers: rationals(&c.multipliers)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        match self.mode.as_str() {
            "mini" => {
                if self.mixed.is_some() || self.mixed_loss.is_some() || !self.cells.is_empty() {
                    return Err("unexpected Mini fields".into());
                }
                let gradient = old.mini_gradient(batch, Some(&rename()), mini_limits())?;
                if self.plan != format!("{:?}", gradient.plan)
                    || self.old_loss != gradient.mean_loss.to_bits()
                    || direction != gradient.coordinates
                {
                    return Err("Mini gradient mismatch".into());
                }
                for ((point, g), proposed) in old.parameters().iter().zip(&direction).zip(&proposal)
                {
                    if point - ratio(1, 32) * g != *proposed {
                        return Err("Mini proposal mismatch".into());
                    }
                }
                next.verify_projection(&proposal, &certs)?;
                if mean_loss(&next, batch)?.to_bits() != self.final_loss {
                    return Err("Mini CE mismatch".into());
                }
            }
            "forward" => {
                if self.cells.len() != 192 {
                    return Err("incomplete forward cell trace".into());
                }
                let plan = ForwardPlan::new(old, batch.len(), true, forward_limits())?;
                if self.plan != format!("{plan:?}") {
                    return Err("forward plan mismatch".into());
                }
                let mixed = old.with_parameters(rationals(
                    self.mixed.as_ref().ok_or("missing mixed point")?,
                )?)?;
                let cells = self
                    .cells
                    .iter()
                    .map(|c| {
                        if c.probes.len() > 4 {
                            return Err("oversized basis trace".into());
                        }
                        let directions = c
                            .probes
                            .iter()
                            .map(|q| {
                                if q.step.len() > 4096 || q.pairing.len() > 4096 {
                                    return Err("oversized probe scalar".into());
                                }
                                Ok(DirectionProbe {
                                    step: q.step.parse()?,
                                    plus_loss: f64::from_bits(q.plus),
                                    minus_loss: f64::from_bits(q.minus),
                                    pairing: q.pairing.parse()?,
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(CellProbes {
                            mask: c.mask,
                            class: c.class,
                            sign: c.sign,
                            component: c.component,
                            directions,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let update = ForwardUpdate {
                    config: forward_config(),
                    estimate: ForwardEstimate {
                        mixed,
                        coordinates: direction,
                        cells,
                        old_loss: f64::from_bits(self.old_loss),
                        mixed_loss: f64::from_bits(self.mixed_loss.ok_or("missing mixture loss")?),
                        plan,
                    },
                    proposal,
                    projected: ProjectedModel {
                        model: next.clone(),
                        certificates: certs,
                    },
                    final_loss: f64::from_bits(self.final_loss),
                };
                old.verify_forward_update(batch, Some(&rename()), &update, forward_limits())?;
            }
            _ => return Err("unknown training mode".into()),
        }
        Ok(next)
    }
}
