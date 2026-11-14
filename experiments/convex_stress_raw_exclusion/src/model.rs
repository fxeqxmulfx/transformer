//! CM49 uses geometry only to select a label within an explicitly assumed pool.
//! Positive prediction rows use all observed sources of the survivor. All
//! unsupported, ambiguous and already-covered inputs retain the exact parent.
use crate::*;

pub(crate) fn point(source: &kernel::Model, cell: usize, state: usize) -> Result<Point> {
    Ok(Point {
        x: i64::try_from(cell % source.n)?,
        y: i64::try_from(cell / source.n)?,
        state,
    })
}

pub(crate) fn eligible(parent: &parent::Model, cell: usize) -> bool {
    let source = parent.source();
    let (block, state) = source.location(cell);
    state >= source.blocks[block].observed.len()
        && source.known(cell)
        && parent.rows()[cell].is_none()
        && parent.geometry()[block].disposition == parent::Disposition::Separated
}

/// Bind the geometric certificate to the actual routed block and original
/// input coordinates before verifying its pool and prediction weights.
pub(crate) fn verify_cell(
    parent: &parent::Model,
    cell: usize,
    row: &Row,
    work: &mut Work,
) -> Result<()> {
    let source = parent.source();
    if cell >= source.n.pow(2) || !eligible(parent, cell) || row.block != source.location(cell).0 {
        return Err("exclusion row differs from the eligible original source block".into());
    }
    let block = &source.blocks[row.block];
    let points = block
        .observed
        .iter()
        .enumerate()
        .map(|(i, &cell)| point(source, cell, i))
        .collect::<Result<Vec<_>>>()?;
    exclusion::verify_row(
        row,
        &points,
        &block.observed_labels,
        point(source, cell, points.len())?,
        work,
    )
}

pub(crate) fn build_rows(
    parent: &parent::Model,
    work: &mut Work,
) -> Result<(Vec<Option<Row>>, Statistics)> {
    let source = parent.source();
    let points: Vec<Vec<_>> = source
        .blocks
        .iter()
        .map(|b| {
            b.observed
                .iter()
                .enumerate()
                .map(|(i, &cell)| point(source, cell, i))
                .collect::<Result<_>>()
        })
        .collect::<Result<_>>()?;
    let mut rows = vec![None; source.n.pow(2)];
    let mut stats = Statistics::default();
    for (cell, row) in rows.iter_mut().enumerate() {
        if !eligible(parent, cell) {
            stats.retained_parent_cells += 1;
            continue;
        }
        stats.eligible_cells += 1;
        let (block, _) = source.location(cell);
        let labels = &source.blocks[block].observed_labels;
        let query = point(source, cell, points[block].len())?;
        let pool: Vec<_> = parent.geometry()[block]
            .hulls
            .iter()
            .map(|h| h.label)
            .collect();
        let result = exclusion::decide(&points[block], labels, query, &pool, work)?;
        match result.survivors.as_slice() {
            [label] => {
                let weights: Vec<_> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, y)| **y == *label)
                    .map(|(state, _)| RationalWeight {
                        state,
                        numerator: 1,
                    })
                    .collect();
                let r = Row {
                    block,
                    label: *label,
                    pool,
                    exclusions: result.exclusions,
                    denominator: u64::try_from(weights.len())?,
                    weights,
                };
                verify_cell(parent, cell, &r, work)?;
                *row = Some(r);
                stats.identified_cells += 1;
            }
            [] => stats.inconsistent_cells += 1,
            _ => stats.ambiguous_cells += 1,
        }
    }
    Ok((rows, stats))
}

impl Model {
    pub fn features(&self, cell: usize) -> Result<(usize, Vec<C>)> {
        let Some(row) = self.rows.get(cell).ok_or("exclusion cell outside domain")? else {
            return self.parent.features(cell);
        };
        let b = &self.source().blocks[row.block];
        let mut phi = vec![C::new(0., 0.); b.states()];
        for weight in &row.weights {
            let mass = weight.numerator as f64 / row.denominator as f64;
            for (f, p) in phi.iter_mut().enumerate() {
                *p += mass * b.code[weight.state * b.states() + f];
            }
        }
        Ok((row.block, phi))
    }

    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        if self
            .rows
            .get(cell)
            .ok_or("exclusion cell outside domain")?
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
        if !(2..=parent::parent::scale::context::PREDICTION_LIMIT).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= source.vocabulary)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= source.n)
        {
            return Err("prefix outside the exclusion input domain".into());
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
            .expect("validated exclusion prediction domain")
    }
}
