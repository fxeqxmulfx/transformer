//! The ordinary CE sum and empirical forward initializer on streamed rows.
//! No loss derivatives are formed when the gradient argument is absent.
use super::StreamCache;
use crate::{
    Result,
    head::{self, ForwardChoice, Head},
};

pub fn objective(
    model: &Head,
    cache: &mut StreamCache,
    examples: &[usize],
    mut gradient: Option<&mut [f64]>,
) -> Result<(f64, usize)> {
    if model.bank != *cache.bank()
        || gradient
            .as_ref()
            .is_some_and(|g| g.len() != model.coefficients.len())
    {
        return Err("incompatible streamed objective inputs".into());
    }
    if let Some(g) = gradient.as_deref_mut() {
        g.fill(0.);
    }
    let v = model.bank.vocab;
    let mut logits = vec![0.; v];
    let mut residual = vec![0.; v];
    let mut total = 0.;
    let count = cache.visit(examples, |features, label| {
        model.encoded_logits(features, &mut logits);
        total += if gradient.is_some() {
            head::ce_gradient(&logits, label, &mut residual)
        } else {
            head::cross_entropy(&logits, label)
        };
        if let Some(g) = gradient.as_deref_mut() {
            for &k in features {
                let start = k as usize * v;
                for (x, r) in g[start..start + v].iter_mut().zip(&residual) {
                    *x += r;
                }
            }
        }
        Ok(())
    })?;
    if count == 0 || !total.is_finite() {
        return Err("nonfinite or empty streamed CE".into());
    }
    if let Some(g) = gradient {
        for x in g {
            *x /= count as f64;
        }
    }
    Ok((total / count as f64, count))
}

/// Count all original supervision, including repetitions. The same scratch
/// buffer can subsequently hold a gradient; no third dense tensor is needed.
pub fn forward_initialize(
    model: &mut Head,
    cache: &mut StreamCache,
    counts: &mut [f64],
    choice: ForwardChoice,
) -> Result<usize> {
    if model.bank != *cache.bank() || counts.len() != model.coefficients.len() {
        return Err("incompatible streamed initializer inputs".into());
    }
    counts.fill(0.);
    let v = model.bank.vocab;
    cache.visit(&cache.full_order(), |features, label| {
        for &k in features {
            counts[k as usize * v + label] += 1.;
        }
        Ok(())
    })?;
    head::initialize_counts(model, cache.observations(), counts, choice)
}
