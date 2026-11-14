//! The same-label certificate supplies a fixed stochastic source feature row.
//! Complete complex multiplication remains the prediction path. All original
//! observed, unmatched, overlapping and uncovered contexts use the exact parent.
use crate::*;
use std::collections::BTreeMap;

fn point(source: &kernel::Model, cell: usize, state: usize) -> Result<Point> {
    Ok(Point {
        x: i64::try_from(cell % source.n)?,
        y: i64::try_from(cell / source.n)?,
        state,
    })
}

fn points(source: &kernel::Model, block: usize) -> Result<Vec<Point>> {
    source.blocks[block]
        .observed
        .iter()
        .enumerate()
        .map(|(state, &cell)| point(source, cell, state))
        .collect()
}

pub(crate) fn build_geometry(
    source: &kernel::Model,
    work: &mut GeometryWork,
) -> Result<Vec<BlockGeometry>> {
    let mut result = Vec::with_capacity(source.blocks.len());
    for (block, b) in source.blocks.iter().enumerate() {
        let mut classes: BTreeMap<usize, Vec<Point>> = BTreeMap::new();
        for (p, &label) in points(source, block)?.into_iter().zip(&b.observed_labels) {
            classes.entry(label).or_default().push(p);
        }
        let mut item = BlockGeometry {
            disposition: Disposition::Empty,
            hulls: vec![],
            overlap: None,
        };
        if classes.len() < 2 {
            if classes.len() == 1 {
                item.disposition = Disposition::SingleLabel;
            }
            result.push(item);
            continue;
        }
        item.disposition = Disposition::Separated;
        for (label, points) in classes {
            item.hulls.push(Hull::build(label, points, work)?);
        }
        'pairs: for i in 0..item.hulls.len() {
            for j in (i + 1)..item.hulls.len() {
                if geometry::overlaps(&item.hulls[i], &item.hulls[j], work)? {
                    item.disposition = Disposition::OverlapRefused;
                    item.overlap = Some([item.hulls[i].label, item.hulls[j].label]);
                    break 'pairs;
                }
            }
        }
        result.push(item);
    }
    Ok(result)
}

pub(crate) fn build_rows(
    source: &kernel::Model,
    geometry: &[BlockGeometry],
    work: &mut GeometryWork,
) -> Result<(Vec<Option<Row>>, Statistics)> {
    let original_points = (0..source.blocks.len())
        .map(|b| points(source, b))
        .collect::<Result<Vec<_>>>()?;
    let mut rows = vec![None; source.n * source.n];
    let mut stats = Statistics {
        separated_blocks: geometry
            .iter()
            .filter(|g| g.disposition == Disposition::Separated)
            .count(),
        overlap_refused_blocks: geometry
            .iter()
            .filter(|g| g.disposition == Disposition::OverlapRefused)
            .count(),
        ..Statistics::default()
    };
    for (cell, row) in rows.iter_mut().enumerate() {
        let (block, state) = source.location(cell);
        let b = &source.blocks[block];
        if state < b.observed.len() {
            stats.observed_cells += 1;
            continue;
        }
        if !source.known(cell) {
            stats.unsupported_cells += 1;
            continue;
        }
        match geometry[block].disposition {
            Disposition::Empty | Disposition::SingleLabel => {
                stats.inactive_cells += 1;
                continue;
            }
            Disposition::OverlapRefused => {
                stats.overlap_refused_cells += 1;
                continue;
            }
            Disposition::Separated => {}
        }
        let query = point(source, cell, state)?;
        let mut certificate = None;
        for hull in &geometry[block].hulls {
            if let Some(candidate) = hull.certify(query, work)? {
                if certificate.is_some() {
                    return Err("separated class hulls gave nonunique coverage".into());
                }
                geometry::verify(
                    &candidate,
                    query,
                    &original_points[block],
                    &b.observed_labels,
                    work,
                )?;
                certificate = Some(candidate);
            }
        }
        if let Some(certificate) = certificate {
            stats.covered_cells += 1;
            *row = Some(Row { block, certificate });
        } else {
            stats.uncovered_cells += 1;
        }
    }
    Ok((rows, stats))
}

impl Model {
    pub fn features(&self, cell: usize) -> Result<(usize, Vec<C>)> {
        let row = self
            .rows
            .get(cell)
            .ok_or("raw-region cell outside domain")?;
        let Some(row) = row else {
            return self.parent.features(cell);
        };
        let b = &self.source().blocks[row.block];
        let mut phi = vec![C::new(0., 0.); b.states()];
        for weight in &row.certificate.weights {
            let mass = weight.numerator as f64 / row.certificate.denominator as f64;
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
            .ok_or("raw-region cell outside domain")?
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
        if !(2..=parent::scale::context::PREDICTION_LIMIT).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= source.vocabulary)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= source.n)
        {
            return Err("prefix outside the raw-region input domain".into());
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
            .expect("validated raw-region prediction domain")
    }
}
