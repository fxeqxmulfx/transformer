//! CM44 actual complex products. The table caches products of the complete
//! tensor, not labels. Construction and loading independently check the identity.
use crate::*;

pub fn forecast(n: usize, q: usize, vocab: usize, records: usize) -> Result<Bounds> {
    let source_buffer_bytes = tagged::model::memory_bound(n, q, vocab)?;
    if !(1..=crate::records::MAX_RECORDS).contains(&records) {
        return Err("original record count outside bounded domain".into());
    }
    let buffer_bytes = source_buffer_bytes
        + 128 * (q * q) as u64
        + 256 * (q * vocab) as u64
        + 512 * records as u64
        + 64 * (n * n) as u64
        + 64 * 1024 * 1024;
    let construction_scalar_work = 64 * (q * q * vocab + records * vocab + q * q + n * n) as u64;
    if buffer_bytes > 56 * 1024 * 1024 * 1024 || construction_scalar_work > WORK_CAP {
        return Err("decoder head exceeds memory or scalar-work forecast".into());
    }
    Ok(Bounds {
        source_buffer_bytes,
        buffer_bytes,
        construction_scalar_work,
        complex_features: q * q,
        real_coefficients: 2 * q * vocab,
        cached_logits: q * vocab,
        new_optimizer_updates: 0,
        new_gradient_batches: 0,
    })
}

pub(crate) fn code(q: usize) -> Vec<C> {
    (0..q)
        .flat_map(|state| {
            (0..q).map(move |f| {
                C::from_polar(
                    1. / (q as f64).sqrt(),
                    std::f64::consts::TAU * ((state * f) % q) as f64 / q as f64,
                )
            })
        })
        .collect()
}

pub(crate) fn initializer(source: &tagged::Model, code: &[C], margin: f64) -> Vec<f64> {
    let v = source.vocab;
    let mut coefficients = vec![0.; 2 * source.q * v];
    for (state, &label) in source.decoder.iter().enumerate() {
        for f in 0..source.q {
            let p = code[state * source.q + f];
            coefficients[2 * f * v + label] += margin * p.re;
            coefficients[2 * f * v + v + label] -= margin * p.im;
        }
    }
    coefficients
}

pub(crate) fn product(phi: &[C], coefficients: &[f64], vocab: usize) -> Vec<f64> {
    let mut logits = vec![0.; vocab];
    for (f, p) in phi.iter().enumerate() {
        let h = 2 * f * vocab;
        for (c, z) in logits.iter_mut().enumerate() {
            *z += p.re * coefficients[h + c] - p.im * coefficients[h + vocab + c];
        }
    }
    logits
}

impl Model {
    pub fn source(&self) -> &tagged::Model {
        &self.source
    }
    pub fn records(&self) -> &[records::Record] {
        &self.records
    }
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    pub fn radius(&self) -> f64 {
        self.radius
    }
    pub fn features(&self, cell: usize) -> Result<&[C]> {
        let state = *self
            .source
            .latent_labels
            .get(cell)
            .ok_or("invalid input cell")?;
        Ok(&self.code[state * self.source.q..(state + 1) * self.source.q])
    }
    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        let state = *self
            .source
            .latent_labels
            .get(cell)
            .ok_or("invalid input cell")?;
        Ok(self.table[state * self.source.vocab..(state + 1) * self.source.vocab].to_vec())
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        if !(2..=PREDICTION_LIMIT).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= self.source.vocab)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= self.source.n)
        {
            return Err("prefix outside the declared phase model domain".into());
        }
        self.logits_cell(
            prefix[prefix.len() - 1] as usize + self.source.n * prefix[prefix.len() - 2] as usize,
        )
    }
}

impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.source.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.logits(prefix)
            .expect("validated decoder-head prediction domain")
    }
}
