//! CM46 actual gated products. Same complete source tensor, fixed block gate,
//! anchored local bank weights; no additional output decoder or task formula.
use crate::*;

pub(crate) fn effective(anchor: &[f64], local: &[f64]) -> Vec<f64> {
    anchor
        .iter()
        .zip(local)
        .map(|(a, b)| (1. - RADIUS) * a + RADIUS * b)
        .collect()
}

pub(crate) fn rows(
    parent: &parent::Model,
    weights: &[Vec<f64>],
) -> Result<Vec<Option<Vec<Weight>>>> {
    let source = parent.source();
    let mut result = vec![None; source.n * source.n];
    let effective: Vec<_> = weights
        .iter()
        .map(|row| effective(&parent.selection().weights, row))
        .collect();
    for (cell, row) in result.iter_mut().enumerate() {
        if parent.parent().context().neighbors(cell)?.is_none() {
            continue;
        }
        let (group, _) = source.location(cell);
        let b = &source.blocks[group];
        let sorted = bank::distances(source, cell);
        let mut mass = vec![0.; b.states()];
        for (context, &alpha) in parent.bank().iter().zip(&effective[group]) {
            if alpha == 0. {
                continue;
            }
            for w in bank::weights(context, &sorted) {
                mass[w.state] += alpha * w.mass;
            }
        }
        if mass.iter().any(|m| !m.is_finite() || *m < 0.)
            || mass.iter().sum::<f64>() > 1. + 1e-12
            || mass.iter().all(|m| *m == 0.)
        {
            return Err("invalid gated context feature row".into());
        }
        *row = Some(
            mass.into_iter()
                .enumerate()
                .filter(|(_, mass)| *mass > 0.)
                .map(|(state, mass)| Weight { state, mass })
                .collect(),
        );
    }
    Ok(result)
}

impl Model {
    fn unchanged(&self, cell: usize) -> bool {
        self.selection.weights[self.gates[cell]] == self.anchor
    }

    pub fn features(&self, cell: usize) -> Result<(usize, Vec<C>)> {
        let row = self
            .rows
            .get(cell)
            .ok_or("gated context cell outside domain")?;
        if row.is_none() || self.unchanged(cell) {
            return self.parent.features(cell);
        }
        let block = self.gates[cell];
        let b = &self.source().blocks[block];
        let mut phi = vec![C::new(0., 0.); b.states()];
        for w in row.as_ref().unwrap() {
            for (f, p) in phi.iter_mut().enumerate() {
                *p += w.mass * b.code[w.state * b.states() + f];
            }
        }
        Ok((block, phi))
    }

    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        let row = self
            .rows
            .get(cell)
            .ok_or("gated context cell outside domain")?;
        if row.is_none() || self.unchanged(cell) {
            return self.parent.logits_cell(cell);
        }
        let (block, phi) = self.features(cell)?;
        let b = &self.source().blocks[block];
        let v = self.source().vocabulary;
        let mut logits = vec![0.; v];
        for (f, p) in phi.iter().enumerate() {
            let h = 2 * (b.offset + f) * v;
            for (c, z) in logits.iter_mut().enumerate() {
                *z += p.re * self.source().coefficients[h + c]
                    - p.im * self.source().coefficients[h + v + c];
            }
        }
        Ok(logits)
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
            return Err("prefix outside the gated-context domain".into());
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
            .expect("validated gated-context prediction domain")
    }
}
