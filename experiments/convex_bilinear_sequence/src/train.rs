//! CM88 complete Mini and forward-only Mini pullbacks through learned logits.
//! Forward mode calls only ordinary CE at two finite bias queries per class
//! and record. CM88 proves reuse equals every responding coordinate query at
//! its response-scaled step; zero responses give exact zero differences.
use crate::{
    C, Result,
    model::{Model, cross_entropy},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Mode {
    Mini,
    ForwardMini,
}

pub struct Record<'a> {
    pub prefix: &'a [i64],
    pub target: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub mode: Mode,
    pub records: usize,
    pub mean_ce_before: f64,
    pub bias_loss_queries: usize,
    pub maximum_forward_residual: f64,
    pub reconstruction_error: f64,
    pub balance_error: f64,
    pub stored_columns: usize,
    pub reserved_bytes: u64,
}

/// Forward-only scalar queries: no analytic residual or CE derivative.
pub fn forward_scalars(logits: &[f64], target: usize, h: f64) -> Result<Vec<f64>> {
    if !h.is_finite() || h <= 0. {
        return Err("nonpositive finite-query step".into());
    }
    let mut changed = logits.to_vec();
    let mut differences = Vec::with_capacity(logits.len());
    for c in 0..logits.len() {
        changed[c] = logits[c] + h;
        let plus = cross_entropy(&changed, target)?;
        changed[c] = logits[c] - h;
        let minus = cross_entropy(&changed, target)?;
        changed[c] = logits[c];
        let difference = (plus - minus) / (2. * h);
        if !difference.is_finite() {
            return Err("nonfinite loss-query difference".into());
        }
        differences.push(difference);
    }
    Ok(differences)
}

fn analytic_scalars(logits: &[f64], target: usize) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let probabilities: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let total: f64 = probabilities.iter().sum();
    probabilities
        .into_iter()
        .enumerate()
        .map(|(c, x)| x / total - f64::from(c == target))
        .collect()
}

pub fn step(
    model: &Model,
    records: &[Record<'_>],
    mode: Mode,
    rate: f64,
    h: f64,
) -> Result<(Model, Receipt)> {
    if records.is_empty() || !rate.is_finite() || rate <= 0. {
        return Err("empty minibatch or invalid rate".into());
    }
    let shape = model.joint.shape;
    let inputs: Vec<_> = records
        .iter()
        .map(|record| shape.addresses(record.prefix))
        .collect::<Result<_>>()?;
    let active: BTreeSet<_> = model
        .joint
        .columns
        .keys()
        .copied()
        .chain(
            inputs
                .iter()
                .flat_map(|input| input.columns.iter().map(|&(d, _)| d)),
        )
        .collect();
    let reserved_bytes = shape.reservation(active.len())?;
    if reserved_bytes > model.max_bytes {
        return Err("complete update memory reservation refused".into());
    }
    let width = shape.objects * shape.queries;
    let mut gradient: BTreeMap<usize, Vec<C>> = BTreeMap::new();
    let mut total_loss = 0.;
    let mut maximum_forward_residual = 0f64;
    for (record, input) in records.iter().zip(&inputs) {
        let actual = model.complex(input)?;
        let direct = model.joint.joint_complex(input);
        for (a, b) in actual.iter().zip(&direct) {
            let residual = (*a - *b).norm();
            if !residual.is_finite() || residual > 2e-8 * (1. + b.norm()) {
                return Err("actual learned forward disagrees with complete joint identity".into());
            }
            maximum_forward_residual = maximum_forward_residual.max(residual);
        }
        let logits: Vec<_> = actual.iter().map(|x| x.re).collect();
        total_loss += cross_entropy(&logits, record.target)?;
        let scalars = match mode {
            Mode::Mini => analytic_scalars(&logits, record.target),
            Mode::ForwardMini => forward_scalars(&logits, record.target, h)?,
        };
        for &(d, count) in &input.columns {
            let column = gradient
                .entry(d)
                .or_insert_with(|| vec![C::default(); width]);
            for &q in &input.queries {
                for (c, &scalar) in scalars.iter().enumerate() {
                    let value = scalar * count as f64 / records.len() as f64;
                    column[c * shape.queries + q] += C::new(value, -value);
                }
            }
        }
    }
    let mut joint = model.joint.clone();
    for (d, gradient) in gradient {
        let column = joint
            .columns
            .entry(d)
            .or_insert_with(|| vec![C::default(); width]);
        for (x, g) in column.iter_mut().zip(gradient) {
            *x -= rate * g;
        }
    }
    let next = Model::recover(joint, model.max_bytes)?;
    let bias_loss_queries = match mode {
        Mode::Mini => 0,
        Mode::ForwardMini => records
            .len()
            .checked_mul(shape.objects)
            .and_then(|n| n.checked_mul(2))
            .ok_or("query count overflow")?,
    };
    let receipt = Receipt {
        mode,
        records: records.len(),
        mean_ce_before: total_loss / records.len() as f64,
        bias_loss_queries,
        maximum_forward_residual,
        reconstruction_error: next.reconstruction_error,
        balance_error: next.balance_error,
        stored_columns: next.joint.columns.len(),
        reserved_bytes,
    };
    Ok((next, receipt))
}
