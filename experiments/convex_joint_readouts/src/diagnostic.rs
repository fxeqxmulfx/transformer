//! Floating CM104 joint support and CE gap. The analytic diagnostic is not a
//! forward-only update; it compares the old and enlarged conditional domains.
use crate::{
    C, Result,
    capacity::{self, Table},
    coefficient,
};
use bilinear::train::{Mode, Record};
use rotary::{atom::row, model::Model};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Support {
    pub selected: usize,
    pub component_supports: Vec<f64>,
    pub linear_cost: f64,
    pub approximate_gap: f64,
    pub predicted_change: f64,
    pub support_identity_error: f64,
    pub change_identity_error: f64,
}
pub fn direction(model: &Model, coefficient: &Table) -> Result<(Model, Support)> {
    model.validate()?;
    capacity::masses(model, coefficient)?;
    let mut support = Vec::with_capacity(model.atoms.len());
    for block in coefficient {
        support.push(
            block
                .iter()
                .map(|a| a[0].norm().max(a[1].norm()))
                .sum::<f64>(),
        );
    }
    let selected = (1..support.len()).fold(
        0,
        |best, j| if support[j] > support[best] { j } else { best },
    );
    let mut target = model.clone();
    target.weights.fill(0.);
    target.weights[selected] = model.budget;
    for (j, atom) in target.atoms.iter_mut().enumerate() {
        for (c, out) in atom.output.iter_mut().enumerate() {
            *out = if j == selected {
                optimality::best_row(coefficient[j][c])?
            } else {
                [C::default(); 2]
            };
        }
    }
    target.validate()?;
    let u = capacity::weighted(model);
    let v = capacity::weighted(&target);
    let (mut linear, mut minimum, mut change) = (0., 0., 0.);
    for ((us, vs), a) in u.iter().zip(&v).zip(coefficient) {
        for ((u, v), a) in us.iter().zip(vs).zip(a) {
            linear += row(*u, *a).re;
            minimum += row(*v, *a).re;
            change += row([v[0] - u[0], v[1] - u[1]], *a).re;
        }
    }
    let bound = model.budget * support[selected];
    let gap = linear + bound;
    if [linear, minimum, change, gap, bound]
        .iter()
        .any(|x| !x.is_finite())
        || gap < -1e-9
        || (minimum + bound).abs() > 1e-9
        || (change + gap).abs() > 1e-9
    {
        return Err("invalid full joint support diagnostic".into());
    }
    Ok((
        target,
        Support {
            selected,
            component_supports: support,
            linear_cost: linear,
            approximate_gap: gap,
            predicted_change: change,
            support_identity_error: (minimum + bound).abs(),
            change_identity_error: (change + gap).abs(),
        },
    ))
}

pub struct Scan<'a> {
    model: &'a Model,
    fixed: optimality::Scan<'a>,
    sum: Table,
    records: usize,
    error: f64,
}
pub struct Analysis {
    pub coefficient: Table,
    pub best: Model,
    pub support: Support,
    pub fixed: optimality::Summary,
    pub maximum_actual_logit_error: f64,
}
impl<'a> Scan<'a> {
    pub fn new(model: &'a Model) -> Result<Self> {
        Ok(Self {
            model,
            fixed: optimality::Scan::new(model)?,
            sum: capacity::zero(model),
            records: 0,
            error: 0.,
        })
    }
    pub fn observe(&mut self, records: &[Record<'_>]) -> Result<()> {
        let b = coefficient::batch(self.model, records, Mode::Mini, 1e-4)?;
        self.fixed.observe(records)?;
        for (total, part) in self
            .sum
            .iter_mut()
            .flatten()
            .flatten()
            .zip(b.sum.iter().flatten().flatten())
        {
            *total += *part;
        }
        self.records += records.len();
        self.error = self.error.max(b.maximum_logit_error);
        Ok(())
    }
    pub fn finish(mut self) -> Result<Analysis> {
        coefficient::mean(&mut self.sum, self.records)?;
        let (best, support) = direction(self.model, &self.sum)?;
        let fixed = self.fixed.finish()?.summary;
        if support.approximate_gap + 1e-9 < fixed.approximate_gap {
            return Err("larger feasible domain has smaller support gap".into());
        }
        Ok(Analysis {
            coefficient: self.sum,
            best,
            support,
            fixed,
            maximum_actual_logit_error: self.error,
        })
    }
}
