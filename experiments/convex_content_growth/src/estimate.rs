//! Complete finite parameter pairs with CM109 reuse only for output-row queries.
use crate::{C, Result, query::Centre};
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use hilbert::{block::Block, model::combine};
use rotary::atom::{State, row};
use serde::Serialize;
#[derive(Serialize)]
pub struct Estimates {
    pub values: Vec<Vec<f64>>,
    pub mean_ce: f64,
    pub full_ce_evaluations: usize,
    pub cached_ce_evaluations: usize,
    pub finite_logit_queries: usize,
    pub coordinate_pairs: [usize; 3],
    pub audit_comparisons: usize,
    pub maximum_cached_ce_error: f64,
}
pub fn output_pair(block: &Block, k: usize, h: f64) -> Result<(usize, State, State)> {
    if block.category(k)? != 2 || !h.is_finite() || h <= 0. {
        return Err("invalid complete output query".into());
    }
    let offset = k - 4 - 8 * block.objects();
    let c = offset / 4;
    let channel = (offset % 4) / 2;
    let mut plus = block.atom.output[c];
    let mut minus = plus;
    let delta = if k.is_multiple_of(2) {
        C::new(h, 0.)
    } else {
        C::new(0., h)
    };
    plus[channel] += delta;
    minus[channel] -= delta;
    Ok((
        c,
        block.geometry.output(plus)?,
        block.geometry.output(minus)?,
    ))
}
pub fn batch(
    block: &Block,
    bases: &[Vec<f64>],
    records: &[Record<'_>],
    mass: f64,
    mode: Mode,
    h: f64,
    audit: bool,
) -> Result<Estimates> {
    block.validate()?;
    if records.is_empty() || bases.len() != records.len() || !h.is_finite() || h <= 0. {
        return Err("invalid complete cached query batch".into());
    }
    let mut states = Vec::new();
    let mut logits = Vec::new();
    let mut centres = Vec::new();
    let mut ce = 0.;
    for (r, base) in records.iter().zip(bases) {
        let z = block.atom.state(r.prefix)?;
        let l = combine(base, block, z, mass)?;
        let centre = Centre::new(&l, r.target)?;
        ce += centre.loss;
        states.push(z);
        logits.push(l);
        centres.push(centre);
    }
    let mut result = Estimates {
        values: vec![vec![0.; block.dimensions()]; records.len()],
        mean_ce: ce / records.len() as f64,
        full_ce_evaluations: records.len(),
        cached_ce_evaluations: 0,
        finite_logit_queries: 0,
        coordinate_pairs: [0; 3],
        audit_comparisons: 0,
        maximum_cached_ce_error: 0.,
    };
    for k in 0..block.dimensions() {
        let category = block.category(k)?;
        if category == 2 {
            let (c, up, um) = output_pair(block, k, h)?;
            for (i, r) in records.iter().enumerate() {
                let plus = bases[i][c] + mass * row(up, states[i]).re;
                let minus = bases[i][c] + mass * row(um, states[i]).re;
                let dp = plus - logits[i][c];
                let dm = minus - logits[i][c];
                let check =
                    audit && (i == 0 || (k - 4 - 8 * block.objects()) == i % (4 * block.objects()));
                let increments = if mode == Mode::ForwardMini || check {
                    result.cached_ce_evaluations += 2;
                    Some((centres[i].increment(c, dp)?, centres[i].increment(c, dm)?))
                } else {
                    None
                };
                result.values[i][k] = match mode {
                    Mode::Mini => {
                        (centres[i].shares[c] - f64::from(c == r.target)) * (plus - minus)
                            / (2. * h)
                    }
                    Mode::ForwardMini => {
                        let (a, b) = increments.ok_or("missing finite losses")?;
                        (a - b) / (2. * h)
                    }
                };
                if check {
                    let (a, b) = increments.ok_or("missing checked finite losses")?;
                    for (value, inc) in [(plus, a), (minus, b)] {
                        let mut changed = logits[i].clone();
                        changed[c] = value;
                        let full = cross_entropy(&changed, r.target)?;
                        result.maximum_cached_ce_error = result
                            .maximum_cached_ce_error
                            .max((full - (centres[i].loss + inc)).abs());
                        result.full_ce_evaluations += 1;
                        result.audit_comparisons += 1;
                    }
                }
                result.coordinate_pairs[2] += 1;
                result.finite_logit_queries += 2;
            }
        } else {
            let plus = block.perturb(k, h)?;
            let minus = block.perturb(k, -h)?;
            for (i, (r, base)) in records.iter().zip(bases).enumerate() {
                if !block.responds(k, r.prefix)? {
                    continue;
                }
                let lp = combine(base, &plus, plus.atom.state(r.prefix)?, mass)?;
                let lm = combine(base, &minus, minus.atom.state(r.prefix)?, mass)?;
                result.values[i][k] = match mode {
                    Mode::Mini => lp
                        .iter()
                        .zip(&lm)
                        .enumerate()
                        .map(|(c, (&p, &m))| {
                            (centres[i].shares[c] - f64::from(c == r.target)) * (p - m) / (2. * h)
                        })
                        .sum(),
                    Mode::ForwardMini => {
                        result.full_ce_evaluations += 2;
                        (cross_entropy(&lp, r.target)? - cross_entropy(&lm, r.target)?) / (2. * h)
                    }
                };
                result.coordinate_pairs[category] += 1;
                result.finite_logit_queries += 2;
            }
        }
    }
    if result.maximum_cached_ce_error > 2e-12
        || result.values.iter().flatten().any(|x| !x.is_finite())
    {
        return Err("cached complete CE query or estimate disagrees".into());
    }
    Ok(result)
}
