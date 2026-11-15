//! Full-vocabulary CE and joint derivatives, CM60 Head.lean at819d4547.
//! The multiplicity weights are original record counts, never learned values.
use crate::{
    corpus::{Batch, Corpus},
    *,
};
use joint::head::Head;

fn check(corpus: &Corpus, head: &Head, batch: &Batch) -> Result<()> {
    corpus.check_batch(batch)?;
    if head.theta.len() != corpus.limits().programs
        || head.coordinates.len() != corpus.limits().vocabulary
    {
        return Err("head and grouped corpus dimensions differ".into());
    }
    Ok(())
}

pub fn score(corpus: &Corpus, head: &Head, batch: &Batch) -> Result<f64> {
    check(corpus, head, batch)?;
    let mut loss = 0.;
    for &(g, count) in &batch.counts {
        let row = &corpus.groups()[g as usize];
        loss += count as f64 * joint::ce(&head.logits(&row.features)?, row.label);
    }
    let mean = loss / batch.records() as f64;
    if !mean.is_finite() {
        return Err("nonfinite weighted CE".into());
    }
    Ok(mean)
}

pub fn gradient(corpus: &Corpus, head: &Head, batch: &Batch) -> Result<(f64, Vec<Parameters>)> {
    check(corpus, head, batch)?;
    let mut result = vec![[0.; 6]; head.theta.len()];
    let mut loss = 0.;
    for &(g, count) in &batch.counts {
        let row = &corpus.groups()[g as usize];
        let logits = head.logits(&row.features)?;
        loss += count as f64 * joint::ce(&logits, row.label);
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut residual: Vec<_> = logits.iter().map(|z| (z - max).exp()).collect();
        let mass: f64 = residual.iter().sum();
        for r in &mut residual {
            *r /= mass;
        }
        residual[row.label] -= 1.;
        let slope: f64 = residual
            .iter()
            .zip(&head.coordinates)
            .map(|(r, e)| r * 2. * e)
            .sum();
        let offset: f64 = residual
            .iter()
            .zip(&head.coordinates)
            .map(|(r, e)| -r * e * e)
            .sum();
        let weight =
            count as f64 / (batch.records() as f64 * joint::head::normalizer(&row.features));
        for (out, h) in result.iter_mut().zip(&row.features) {
            for j in 0..5 {
                out[j] += weight * slope * h[j];
            }
            out[5] += weight * offset;
        }
    }
    let mean = loss / batch.records() as f64;
    if !mean.is_finite() || result.iter().flatten().any(|x| !x.is_finite()) {
        return Err("nonfinite weighted CE derivative".into());
    }
    Ok((mean, result))
}
