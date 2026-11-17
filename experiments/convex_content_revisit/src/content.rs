//! Persistent finite-CE content search, not a global convex parameter oracle.
//! CM97/98 supply the complete family, CM102 the row projection, CM106 its
//! changed-state accounting. Only matrices absent from the current raw batch skip.
use crate::{C, Result};
use bilinear::{model::cross_entropy, train::Record};
use rotary::{
    atom::{Atom, Family},
    model::Model,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy)]
pub enum Coordinate {
    Initial(usize),
    Matrix(usize, usize, usize),
    Frequency(usize),
}
impl Coordinate {
    fn kind(self) -> usize {
        match self {
            Self::Initial(_) => 0,
            Self::Matrix(..) => 1,
            Self::Frequency(_) => 2,
        }
    }
    pub fn apply(self, atom: &Atom, delta: C) -> Result<Atom> {
        let mut a = atom.clone();
        match self {
            Self::Initial(k) => {
                let z = a.initial[k] + delta;
                let n = z.norm();
                if !n.is_finite() {
                    return Err("nonfinite initial content trial".into());
                }
                a.initial[k] = if n > 1. { z / n } else { z };
            }
            Self::Matrix(o, i, j) => {
                a.transition[o][i][j] += delta;
                a.transition[o][i] = readouts::projection::project(a.transition[o][i])?;
            }
            Self::Frequency(k) => {
                let f = a.frequency.as_mut().ok_or("frequency absent")?;
                f[k] += delta.re;
                if !f[k].is_finite() || delta.im != 0. {
                    return Err("invalid frequency content trial".into());
                }
            }
        }
        Ok(a)
    }
}
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Config {
    pub radius: f64,
    pub frequency_radius: f64,
}
#[derive(Serialize, Deserialize)]
pub struct Receipt {
    pub component: usize,
    pub observed_objects: usize,
    pub coordinate_trials: [usize; 3],
    pub accepted_coordinates: [usize; 3],
    pub total_ce_evaluations: usize,
    pub before_ce: f64,
    pub after_ce: f64,
    pub maximum_actual_logit_error: f64,
    pub maximum_operator_norm_excess: f64,
    pub content_changed: bool,
}
pub fn step(
    model: &mut Model,
    records: &[Record<'_>],
    component: usize,
    config: Config,
) -> Result<Receipt> {
    model.validate()?;
    if records.is_empty()
        || component == 0
        || component >= model.atoms.len()
        || model.family != Family::Rotary
        || !config.radius.is_finite()
        || config.radius <= 0.
        || !config.frequency_radius.is_finite()
        || config.frequency_radius <= 0.
    {
        return Err("invalid persistent content step".into());
    }
    let before_hash = readouts::source::content_hash(model)?;
    let base = records
        .iter()
        .map(|r| model.logits(r.prefix))
        .collect::<Result<Vec<_>>>()?;
    let atom = &model.atoms[component];
    let weight = model.weights[component];
    let mut fixed = Vec::with_capacity(records.len());
    let mut before = 0.;
    for (r, scores) in records.iter().zip(&base) {
        before += cross_entropy(scores, r.target)?;
        let old = atom.complex(r.prefix)?;
        fixed.push(
            scores
                .iter()
                .zip(old)
                .map(|(&z, a)| z - weight * a.re)
                .collect::<Vec<_>>(),
        );
    }
    before /= records.len() as f64;
    let evaluate = |a: &Atom| -> Result<f64> {
        a.validate(model.objects, model.family)?;
        let mut ce = 0.;
        for (r, other) in records.iter().zip(&fixed) {
            let logits = other
                .iter()
                .zip(a.complex(r.prefix)?)
                .map(|(&z, a)| z + weight * a.re)
                .collect::<Vec<_>>();
            ce += cross_entropy(&logits, r.target)?;
        }
        Ok(ce / records.len() as f64)
    };
    let mut observed = vec![false; model.objects];
    for r in records {
        for &o in r.prefix {
            observed[usize::try_from(o)?] = true;
        }
    }
    let mut coordinates = vec![Coordinate::Initial(0), Coordinate::Initial(1)];
    for (o, &present) in observed.iter().enumerate() {
        if present {
            for i in 0..2 {
                for j in 0..2 {
                    coordinates.push(Coordinate::Matrix(o, i, j));
                }
            }
        }
    }
    coordinates.extend([Coordinate::Frequency(0), Coordinate::Frequency(1)]);
    let mut best = atom.clone();
    let mut best_ce = before;
    let mut trials = [0usize; 3];
    let mut accepted = [0usize; 3];
    for coordinate in coordinates {
        let kind = coordinate.kind();
        let radius = if kind == 2 {
            config.frequency_radius
        } else {
            config.radius
        };
        let deltas = [
            C::new(radius, 0.),
            C::new(-radius, 0.),
            C::new(0., radius),
            C::new(0., -radius),
        ];
        let mut chosen = best.clone();
        let mut chosen_ce = best_ce;
        for &delta in &deltas[..if kind == 2 { 2 } else { 4 }] {
            let trial = coordinate.apply(&best, delta)?;
            let ce = evaluate(&trial)?;
            trials[kind] += 1;
            if ce < chosen_ce {
                chosen = trial;
                chosen_ce = ce;
            }
        }
        if chosen_ce < best_ce {
            accepted[kind] += 1;
        }
        best = chosen;
        best_ce = chosen_ce;
    }
    let mut next = model.clone();
    next.atoms[component] = best;
    let norm_excess = next.validate()?;
    let mut actual_ce = 0.;
    let mut error = 0f64;
    for (r, other) in records.iter().zip(&fixed) {
        let logits = next.logits(r.prefix)?;
        let part = next.atoms[component].complex(r.prefix)?;
        for ((&actual, &other), p) in logits.iter().zip(other).zip(part) {
            error = error.max((actual - (other + weight * p.re)).abs());
        }
        actual_ce += cross_entropy(&logits, r.target)?;
    }
    actual_ce /= records.len() as f64;
    if error > 2e-12 || (actual_ce - best_ce).abs() > 2e-12 || actual_ce > before + 2e-12 {
        return Err("actual content-step logits or CE disagree".into());
    }
    let changed = readouts::source::content_hash(&next)? != before_hash;
    *model = next;
    Ok(Receipt {
        component,
        observed_objects: observed.iter().filter(|&&x| x).count(),
        coordinate_trials: trials,
        accepted_coordinates: accepted,
        total_ce_evaluations: (trials.iter().sum::<usize>() + 2) * records.len(),
        before_ce: before,
        after_ce: actual_ce,
        maximum_actual_logit_error: error,
        maximum_operator_norm_excess: norm_excess,
        content_changed: changed,
    })
}
