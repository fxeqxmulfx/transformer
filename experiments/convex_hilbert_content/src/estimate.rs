//! Finite feasible coordinate pairs, followed by one simultaneous retained update.
//! Mini linearizes CE in complete logits; ForwardMini evaluates only CE pairs.
//! These are numerical constrained-coordinate estimates, not certified gradients.
use crate::{Result, block::Block, model::combine};
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use serde::Serialize;
#[derive(Serialize)]
pub struct Estimates {
    pub values: Vec<Vec<f64>>,
    pub mean_ce: f64,
    pub ce_evaluations: usize,
    pub logit_queries: usize,
    pub coordinate_pairs: [usize; 3],
}
pub fn batch(
    block: &Block,
    bases: &[Vec<f64>],
    records: &[Record<'_>],
    mass: f64,
    mode: Mode,
    h: f64,
) -> Result<Estimates> {
    block.validate()?;
    if records.is_empty() || bases.len() != records.len() || !h.is_finite() || h <= 0. {
        return Err("invalid complete finite-query batch".into());
    }
    let mut states = Vec::new();
    let mut residuals = Vec::new();
    let mut ce = 0.;
    for (r, base) in records.iter().zip(bases) {
        let z = block.atom.state(r.prefix)?;
        let logits = combine(base, block, z, mass)?;
        ce += cross_entropy(&logits, r.target)?;
        if mode == Mode::Mini {
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut s: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
            let total: f64 = s.iter().sum();
            for (c, v) in s.iter_mut().enumerate() {
                *v = *v / total - f64::from(c == r.target);
            }
            residuals.push(s);
        }
        states.push(z);
    }
    let mut result = Estimates {
        values: vec![vec![0.; block.dimensions()]; records.len()],
        mean_ce: ce / records.len() as f64,
        ce_evaluations: records.len(),
        logit_queries: 0,
        coordinate_pairs: [0; 3],
    };
    for k in 0..block.dimensions() {
        let category = block.category(k)?;
        let plus = block.perturb(k, h)?;
        let minus = block.perturb(k, -h)?;
        for (i, (r, base)) in records.iter().zip(bases).enumerate() {
            if !block.responds(k, r.prefix)? {
                continue;
            }
            let zp = if category == 2 {
                states[i]
            } else {
                plus.atom.state(r.prefix)?
            };
            let zm = if category == 2 {
                states[i]
            } else {
                minus.atom.state(r.prefix)?
            };
            let lp = combine(base, &plus, zp, mass)?;
            let lm = combine(base, &minus, zm, mass)?;
            result.values[i][k] = match mode {
                Mode::Mini => lp
                    .iter()
                    .zip(&lm)
                    .zip(&residuals[i])
                    .map(|((&p, &m), &s)| s * (p - m) / (2. * h))
                    .sum(),
                Mode::ForwardMini => {
                    result.ce_evaluations += 2;
                    (cross_entropy(&lp, r.target)? - cross_entropy(&lm, r.target)?) / (2. * h)
                }
            };
            result.coordinate_pairs[category] += 1;
            result.logit_queries += 2;
        }
    }
    if result.values.iter().flatten().any(|x| !x.is_finite()) {
        return Err("nonfinite coordinate estimate".into());
    }
    Ok(result)
}
