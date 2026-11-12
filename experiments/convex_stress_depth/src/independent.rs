//! Independent layer routing and a shared learned decoder: the nonconvex
//! factorization embedded by CD8 `liftHead_logits` and `liftHead_budget` at
//! 6f71b458, with normalization from CD5 `productWeights_mem`. Views belong to
//! the terminal feature index, so the router is shared across all input views.
use crate::{
    Result,
    features::{Bank, Execution},
    head::{self, Cache, Head},
    metrics::Predictor,
};

#[derive(Clone)]
pub struct Independent {
    pub bank: Bank,
    pub radius: f64,
    pub probabilities: Vec<f64>,
    pub decoder: Vec<f64>,
}
impl Independent {
    pub fn new(bank: Bank, radius: f64) -> Result<Self> {
        bank.validate()?;
        if bank.execution != Execution::Composed
            || !radius.is_finite()
            || radius <= 0.
            || bank.allocation_bytes(0)? > head::BUFFER_LIMIT
        {
            return Err("invalid independent baseline configuration".into());
        }
        Ok(Self {
            probabilities: vec![0.5; bank.depth],
            decoder: vec![0.; bank.masks.len() * bank.stride() * bank.vocab],
            bank,
            radius,
        })
    }
    /// The shared forward initializer is supported on a single full path and
    /// therefore has an exact independent factorization. Reject a general head.
    pub fn from_vertex(model: &Head, channel: usize) -> Result<Self> {
        model.validate()?;
        if channel >= 2 * model.bank.paths() {
            return Err("invalid initializer channel".into());
        }
        let mut result = Self::new(model.bank.clone(), model.radius)?;
        let paths = model.bank.paths_per_view();
        let path = channel / 2 % paths;
        for (layer, p) in result.probabilities.iter_mut().enumerate() {
            *p = ((path >> (model.bank.depth - layer - 1)) & 1) as f64;
        }
        let stride = model.bank.stride();
        let v = model.bank.vocab;
        for (k, row) in model.coefficients.chunks_exact(v).enumerate() {
            let selected = k / stride % paths;
            if selected != path {
                if row.iter().any(|w| *w != 0.) {
                    return Err("initializer is not a single-path tensor".into());
                }
                continue;
            }
            let local = (k / (stride * paths)) * stride + k % stride;
            result.decoder[local * v..(local + 1) * v].copy_from_slice(row);
        }
        result.validate()?;
        Ok(result)
    }
    pub fn routes(&self) -> Vec<f64> {
        (0..self.bank.paths_per_view())
            .map(|p| {
                self.probabilities
                    .iter()
                    .enumerate()
                    .map(|(layer, q)| {
                        if p >> (self.bank.depth - layer - 1) & 1 == 1 {
                            *q
                        } else {
                            1. - q
                        }
                    })
                    .product()
            })
            .collect()
    }
    pub fn route_derivatives(&self) -> Vec<Vec<f64>> {
        (0..self.bank.paths_per_view())
            .map(|p| {
                (0..self.bank.depth)
                    .map(|derivative| {
                        self.probabilities
                            .iter()
                            .enumerate()
                            .map(|(layer, q)| {
                                let bit = p >> (self.bank.depth - layer - 1) & 1;
                                if layer == derivative {
                                    if bit == 1 { 1. } else { -1. }
                                } else if bit == 1 {
                                    *q
                                } else {
                                    1. - q
                                }
                            })
                            .product()
                    })
                    .collect()
            })
            .collect()
    }
    pub fn validate(&self) -> Result<f64> {
        self.bank.validate()?;
        if self.bank.execution != Execution::Composed
            || self.probabilities.len() != self.bank.depth
            || self
                .probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || !self.radius.is_finite()
            || self.radius <= 0.
            || self.decoder.len() != self.bank.masks.len() * self.bank.stride() * self.bank.vocab
            || self.decoder.iter().any(|d| !d.is_finite())
        {
            return Err("invalid independent routing or decoder".into());
        }
        if (self.routes().iter().sum::<f64>() - 1.).abs() > 1e-12 {
            return Err("independent routes do not normalize".into());
        }
        let mut norms = vec![0.; self.bank.vocab];
        for row in self.decoder.chunks_exact(self.bank.vocab) {
            for (sum, w) in norms.iter_mut().zip(row) {
                *sum += w.abs();
            }
        }
        let maximum = norms.into_iter().fold(0., f64::max);
        if maximum > self.radius * (1. + 1e-10) {
            return Err("independent decoder violates L1 budget".into());
        }
        Ok(maximum)
    }
    fn coordinates(&self, k: usize) -> (usize, usize) {
        let stride = self.bank.stride();
        let paths = self.bank.paths_per_view();
        (
            k / stride % paths,
            (k / (stride * paths)) * stride + k % stride,
        )
    }
    pub fn encoded_logits(&self, features: &[u32], routes: &[f64], out: &mut [f64]) {
        out.fill(0.);
        let v = self.bank.vocab;
        for &k in features {
            let (path, local) = self.coordinates(k as usize);
            let weight = routes[path];
            if weight == 0. {
                continue;
            }
            for (z, d) in out
                .iter_mut()
                .zip(&self.decoder[local * v..(local + 1) * v])
            {
                *z += weight * d;
            }
        }
    }
    /// Materialize the embedding only for inspection/tests, never for training.
    pub fn lift(&self) -> Result<Head> {
        self.validate()?;
        let mut head = Head::zero(self.bank.clone(), self.radius)?;
        let routes = self.routes();
        for (k, row) in head
            .coefficients
            .chunks_exact_mut(self.bank.vocab)
            .enumerate()
        {
            let (p, h) = self.coordinates(k);
            for (w, d) in row
                .iter_mut()
                .zip(&self.decoder[h * self.bank.vocab..(h + 1) * self.bank.vocab])
            {
                *w = routes[p] * d;
            }
        }
        head.validate()?;
        Ok(head)
    }
}
impl Predictor for Independent {
    fn vocab(&self) -> usize {
        self.bank.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let features: Vec<_> = self
            .bank
            .encode(prefix)
            .expect("invalid baseline prefix")
            .into_iter()
            .map(|k| k as u32)
            .collect();
        let mut out = vec![0.; self.bank.vocab];
        self.encoded_logits(&features, &self.routes(), &mut out);
        out
    }
}

pub fn objective(
    model: &Independent,
    cache: &Cache,
    examples: &[usize],
    mut decoder_gradient: Option<&mut [f64]>,
    mut route_gradient: Option<&mut [f64]>,
) -> Result<(f64, usize)> {
    if examples.is_empty()
        || cache.dimension != model.bank.dimension()
        || cache.vocab != model.bank.vocab
        || decoder_gradient
            .as_ref()
            .is_some_and(|g| g.len() != model.decoder.len())
        || route_gradient
            .as_ref()
            .is_some_and(|g| g.len() != model.probabilities.len())
    {
        return Err("incompatible independent objective inputs".into());
    }
    if let Some(g) = decoder_gradient.as_deref_mut() {
        g.fill(0.);
    }
    if let Some(g) = route_gradient.as_deref_mut() {
        g.fill(0.);
    }
    let routes = model.routes();
    let derivatives = if route_gradient.is_some() {
        model.route_derivatives()
    } else {
        vec![]
    };
    let mut logits = vec![0.; cache.vocab];
    let mut residual = vec![0.; cache.vocab];
    let (mut loss, mut count) = (0., 0);
    for &e in examples {
        if e >= cache.example_offsets.len() - 1 {
            return Err("invalid independent example index".into());
        }
        for i in cache.example_offsets[e]..cache.example_offsets[e + 1] {
            let features = cache.features(i);
            model.encoded_logits(features, &routes, &mut logits);
            if decoder_gradient.is_none() && route_gradient.is_none() {
                loss += head::cross_entropy(&logits, cache.labels[i]);
            } else {
                loss += head::ce_gradient(&logits, cache.labels[i], &mut residual);
                for &k in features {
                    let (p, h) = model.coordinates(k as usize);
                    if let Some(g) = decoder_gradient.as_deref_mut()
                        && routes[p] != 0.
                    {
                        for (d, r) in g[h * cache.vocab..(h + 1) * cache.vocab]
                            .iter_mut()
                            .zip(&residual)
                        {
                            *d += routes[p] * r;
                        }
                    }
                    if let Some(g) = route_gradient.as_deref_mut()
                        && derivatives[p].iter().any(|d| *d != 0.)
                    {
                        let dot: f64 = model.decoder[h * cache.vocab..(h + 1) * cache.vocab]
                            .iter()
                            .zip(&residual)
                            .map(|(d, r)| d * r)
                            .sum();
                        for (q, d) in g.iter_mut().zip(&derivatives[p]) {
                            *q += dot * d;
                        }
                    }
                }
            }
            count += 1;
        }
    }
    if count == 0 || !loss.is_finite() {
        return Err("invalid independent loss".into());
    }
    if let Some(g) = decoder_gradient {
        for d in g {
            *d /= count as f64;
        }
    }
    if let Some(g) = route_gradient {
        for q in g {
            *q /= count as f64;
        }
    }
    Ok((loss / count as f64, count))
}

/// Block-coordinate projected gradient: one decoder pass, then one routing
/// pass and at most eight forward backtracking checks on that same batch.
/// Routing is nonconvex; no global optimality certificate is claimed.
pub fn step(
    model: &mut Independent,
    cache: &Cache,
    examples: &[usize],
    scratch: &mut [f64],
) -> Result<(usize, usize)> {
    let (_, count) = objective(model, cache, examples, Some(scratch), None)?;
    for (d, g) in model.decoder.iter_mut().zip(scratch) {
        *d -= *g / cache.width as f64;
    }
    head::project_columns(&mut model.decoder, model.bank.vocab, model.radius);
    if model.bank.depth == 0 {
        model.validate()?;
        return Ok((count, 0));
    }
    let mut gradient = vec![0.; model.bank.depth];
    let (base, _) = objective(model, cache, examples, None, Some(&mut gradient))?;
    let previous = model.probabilities.clone();
    let mut rate = 0.1;
    let mut evaluations = 0;
    for attempt in 0..8 {
        for ((q, old), g) in model.probabilities.iter_mut().zip(&previous).zip(&gradient) {
            *q = (old - rate * g).clamp(0., 1.);
        }
        evaluations += 1;
        if objective(model, cache, examples, None, None)?.0 <= base + 1e-14 {
            break;
        }
        if attempt == 7 {
            model.probabilities.copy_from_slice(&previous);
        }
        rate *= 0.5;
    }
    model.validate()?;
    Ok((2 * count, evaluations))
}

#[cfg(test)]
mod tests;
