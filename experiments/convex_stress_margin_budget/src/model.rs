//! Complete original complex products, with an exact reference fast path.
//! Splitting the reference/residual product fixes the native arithmetic order.
use crate::*;

pub(crate) fn reference_mass(weights: &[f64]) -> f64 {
    (1. - RADIUS) + RADIUS * (weights[0] + REFERENCE_SCALE * weights[1])
}

pub(crate) fn is_anchor(weights: &[f64]) -> bool {
    weights[0] == 1. && weights[1..].iter().all(|a| *a == 0.)
}

pub(crate) fn row(
    source: &kernel::Model,
    cell: usize,
    weights: &[f64],
    maps: &[Vec<Weight>],
) -> FeatureRow {
    let b = &source.blocks[source.location(cell).0];
    let mut mass = vec![0.; b.states()];
    for (&alpha, map) in weights[2..].iter().zip(maps) {
        if alpha != 0. {
            for w in map {
                mass[w.state] += RADIUS * alpha * w.mass;
            }
        }
    }
    let mut residual = vec![C::new(0., 0.); b.states()];
    for (state, &mass) in mass.iter().enumerate() {
        if mass != 0. {
            for (f, p) in residual.iter_mut().enumerate() {
                *p += mass * b.code[state * b.states() + f];
            }
        }
    }
    FeatureRow {
        reference_mass: reference_mass(weights),
        residual,
    }
}

pub(crate) fn logits(
    source: &kernel::Model,
    cell: usize,
    row: &FeatureRow,
    reference: &[f64],
) -> Vec<f64> {
    let mut result: Vec<_> = reference.iter().map(|z| row.reference_mass * z).collect();
    // This path preserves signed zero in pure positive reference rescaling.
    if row.residual.iter().all(|p| p.re == 0. && p.im == 0.) {
        return result;
    }
    let b = &source.blocks[source.location(cell).0];
    let v = source.vocabulary;
    let mut residual = vec![0.; v];
    for (f, p) in row.residual.iter().enumerate() {
        let h = 2 * (b.offset + f) * v;
        for (c, z) in residual.iter_mut().enumerate() {
            *z += p.re * source.coefficients[h + c] - p.im * source.coefficients[h + v + c];
        }
    }
    for (z, r) in result.iter_mut().zip(residual) {
        *z += r;
    }
    result
}

pub(crate) fn rows(
    parent: &parent::Model,
    weights: &[Vec<f64>],
    gates: &[usize],
) -> Result<Vec<Option<FeatureRow>>> {
    let mut result = vec![None; parent.source().n * parent.source().n];
    for (cell, value) in result.iter_mut().enumerate() {
        if parent
            .parent()
            .parent()
            .context()
            .neighbors(cell)?
            .is_none()
            || is_anchor(&weights[gates[cell]])
        {
            continue;
        }
        let sorted = bank::distances(parent.source(), cell);
        let maps: Vec<_> = parent
            .bank()
            .iter()
            .map(|c| bank::weights(c, &sorted))
            .collect();
        *value = Some(row(parent.source(), cell, &weights[gates[cell]], &maps));
    }
    Ok(result)
}

impl Model {
    pub fn features(&self, cell: usize) -> Result<(usize, Vec<C>)> {
        let row = self
            .rows
            .get(cell)
            .ok_or("margin context cell outside domain")?;
        let (block, mut phi) = self.parent.features(cell)?;
        if let Some(row) = row {
            for (p, r) in phi.iter_mut().zip(&row.residual) {
                *p = row.reference_mass * *p + r;
            }
        }
        Ok((block, phi))
    }

    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        let row = self
            .rows
            .get(cell)
            .ok_or("margin context cell outside domain")?;
        let reference = self.parent.logits_cell(cell)?;
        Ok(match row {
            None => reference,
            Some(row) => logits(self.source(), cell, row, &reference),
        })
    }

    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        let source = self.source();
        if !(2..=scale::context::PREDICTION_LIMIT).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= source.vocabulary)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= source.n)
        {
            return Err("prefix outside the margin-context domain".into());
        }
        self.logits_cell(
            prefix[prefix.len() - 1] as usize + source.n * prefix[prefix.len() - 2] as usize,
        )
    }
}

impl kernel::metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.source().vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.logits(prefix)
            .expect("validated margin-context prediction domain")
    }
}
