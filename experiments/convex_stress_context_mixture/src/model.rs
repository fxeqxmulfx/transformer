//! CM45 actual feature mixture. All real and imaginary source coefficients
//! remain charged and unchanged; no canonical-vote shortcut is used at inference.
use crate::*;

pub(crate) fn rows(
    parent: &parent::Model,
    bank: &[Context],
    weights: &[f64],
) -> Result<Vec<Option<Vec<Weight>>>> {
    let source = parent.source();
    let mut rows = vec![None; source.n * source.n];
    for (cell, row) in rows.iter_mut().enumerate() {
        if parent.context().neighbors(cell)?.is_none() {
            continue;
        }
        let (block, _) = source.location(cell);
        let b = &source.blocks[block];
        let sorted = bank::distances(source, cell);
        let mut mass = vec![0.; b.states()];
        for (context, &alpha) in bank.iter().zip(weights) {
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
            return Err("invalid actual context mixture row".into());
        }
        *row = Some(
            mass.into_iter()
                .enumerate()
                .filter(|(_, mass)| *mass > 0.)
                .map(|(state, mass)| Weight { state, mass })
                .collect(),
        );
    }
    Ok(rows)
}

impl Model {
    fn baseline(&self) -> bool {
        self.selection.weights[0] == 1.
    }

    pub fn features(&self, cell: usize) -> Result<(usize, Vec<C>)> {
        let row = self.rows.get(cell).ok_or("mixture cell outside domain")?;
        if row.is_none() || self.baseline() {
            return self.parent.features(cell);
        }
        let (block, _) = self.source().location(cell);
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
        let row = self.rows.get(cell).ok_or("mixture cell outside domain")?;
        if row.is_none() || self.baseline() {
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
        if !(2..=parent::context::PREDICTION_LIMIT).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= source.vocabulary)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= source.n)
        {
            return Err("prefix outside the context-mixture prediction domain".into());
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
            .expect("validated context-mixture prediction domain")
    }
}
