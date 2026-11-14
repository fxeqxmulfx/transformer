//! Actual signed complex features precede the unchanged complete head product.
//! Earlier geometric rows and every observed state retain exact parent logits.
use crate::*;

pub(crate) fn eligible(parent: &parent::Model, cell: usize) -> bool {
    let source = parent.source();
    let (block, state) = source.location(cell);
    state >= source.blocks[block].observed.len()
        && source.known(cell)
        && parent.rows()[cell].is_none()
        && parent.parent().rows()[cell].is_none()
        && parent.parent().parent().rows()[cell].is_none()
        && matches!(
            parent.parent().parent().geometry()[block].disposition,
            regions::Disposition::Separated | regions::Disposition::OverlapRefused
        )
}

pub(crate) fn active_blocks(parent: &parent::Model) -> Vec<bool> {
    let mut active = vec![false; parent.source().blocks.len()];
    for cell in 0..parent.source().n.pow(2) {
        if eligible(parent, cell) {
            active[parent.source().location(cell).0] = true;
        }
    }
    active
}

pub(crate) fn rows(
    parent: &parent::Model,
    fits: &kernels::Fits,
    selection: &Selection,
    work: &mut Work,
) -> Result<(Vec<Option<Row>>, Statistics)> {
    let source = parent.source();
    let mut rows = vec![None; source.n.pow(2)];
    let mut stats = Statistics::default();
    for (cell, row) in rows.iter_mut().enumerate() {
        if !eligible(parent, cell) {
            stats.retained_parent_cells += 1;
            continue;
        }
        stats.eligible_cells += 1;
        let block = source.location(cell).0;
        let mut parent_mass = selection.weights[0];
        let mut weights = vec![0.; source.blocks[block].observed.len()];
        let mut used_kernel = false;
        for (fit, &alpha) in fits[block].iter().zip(&selection.weights).skip(1) {
            if alpha == 0. {
                continue;
            }
            if let Some(fit) = fit {
                used_kernel = true;
                let local = fit.query(kernels::point(source.n, cell), work)?;
                for (w, a) in weights.iter_mut().zip(local) {
                    *w += alpha * a;
                }
            } else {
                parent_mass += alpha;
            }
        }
        if !used_kernel {
            stats.retained_baseline_rows += 1;
            continue;
        }
        // An outward check also covers rounding when fixed bank rows are mixed.
        let mut complete = Vec::with_capacity(weights.len() + 1);
        complete.push(parent_mass);
        complete.extend(weights);
        work.normalizations += 1;
        let normalization_scale = solver::normalize(&mut complete)?;
        let parent_mass = complete.remove(0);
        if parent_mass < 0. {
            return Err("negative parent feature coefficient".into());
        }
        stats.negative_weight_rows += usize::from(complete.iter().any(|&w| w < 0.));
        stats.changed_feature_rows += 1;
        work.prediction_rows += 1;
        *row = Some(Row {
            block,
            parent_mass,
            weights: complete,
            normalization_scale,
        });
    }
    Ok((rows, stats))
}

impl Model {
    pub fn features(&self, cell: usize) -> Result<(usize, Vec<C>)> {
        let Some(row) = self.rows.get(cell).ok_or("signed cell outside domain")? else {
            return self.parent.features(cell);
        };
        let b = &self.source().blocks[row.block];
        let mut phi = vec![C::new(0., 0.); b.states()];
        if row.parent_mass != 0. {
            let (block, parent) = self.parent.features(cell)?;
            if block != row.block || parent.len() != phi.len() {
                return Err("signed parent block differs".into());
            }
            for (p, value) in phi.iter_mut().zip(parent) {
                *p = row.parent_mass * value;
            }
        }
        for (state, &mass) in row.weights.iter().enumerate() {
            if mass == 0. {
                continue;
            }
            for (f, value) in phi.iter_mut().enumerate() {
                *value += mass * b.code[state * b.states() + f];
            }
        }
        Ok((row.block, phi))
    }

    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        if self
            .rows
            .get(cell)
            .ok_or("signed cell outside domain")?
            .is_none()
        {
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
        if !(2..=gated::scale::context::PREDICTION_LIMIT).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= source.vocabulary)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= source.n)
        {
            return Err("prefix outside signed kernel domain".into());
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
            .expect("validated signed kernel prediction domain")
    }
}
