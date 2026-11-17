//! Port of joint::update::line_search at f5ae34ef; unchanged finite minibatch CE search.
use crate::Result;
use bilinear::{model::cross_entropy, train::Record};
pub fn line_search(
    base: &[Vec<f64>],
    target: &[Vec<f64>],
    records: &[Record<'_>],
    iterations: usize,
    ceiling: f64,
) -> Result<(f64, f64)> {
    let evaluate = |eta: f64| -> Result<f64> {
        let mut loss = 0.;
        for ((x, y), r) in base.iter().zip(target).zip(records) {
            let scores: Vec<_> = x
                .iter()
                .zip(y)
                .map(|(&x, &y)| (1. - eta) * x + eta * y)
                .collect();
            loss += cross_entropy(&scores, r.target)?;
        }
        Ok(loss / records.len() as f64)
    };
    let (mut lo, mut hi) = (0., ceiling);
    let ratio = (5f64.sqrt() - 1.) / 2.;
    let (mut x, mut y) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
    let (mut fx, mut fy) = (evaluate(x)?, evaluate(y)?);
    let (mut best_eta, mut best_loss) = (0., evaluate(0.)?);
    let end = evaluate(ceiling)?;
    if end < best_loss {
        best_eta = ceiling;
        best_loss = end;
    }
    for _ in 0..iterations {
        for (eta, loss) in [(x, fx), (y, fy)] {
            if loss < best_loss {
                best_eta = eta;
                best_loss = loss;
            }
        }
        if fx < fy {
            hi = y;
            y = x;
            fy = fx;
            x = hi - ratio * (hi - lo);
            fx = evaluate(x)?;
        } else {
            lo = x;
            x = y;
            fx = fy;
            y = lo + ratio * (hi - lo);
            fy = evaluate(y)?;
        }
    }
    Ok((best_eta, best_loss))
}
