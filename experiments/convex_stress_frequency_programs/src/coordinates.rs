//! Supplied class geometry and training-only two-anchor fit, CM56 Basic at
//! b5f1c9d7. Floating residual checks are numerical checks, not exact proofs.
use crate::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub id: usize,
    pub feature: f64,
    pub label: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fit {
    pub coordinates: Vec<f64>,
    pub slope: f64,
    pub intercept: f64,
    pub anchor_ids: [usize; 2],
    pub max_training_residual: f64,
}

pub fn validate_coordinates(coordinates: &[f64]) -> Result<()> {
    if !(2..=1024).contains(&coordinates.len())
        || coordinates.iter().any(|e| !e.is_finite() || e.abs() > 1.)
    {
        return Err("invalid bounded supplied class geometry".into());
    }
    let mut sorted = coordinates.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.windows(2).any(|w| w[1] - w[0] < 1e-8) {
        return Err("noninjective or numerically unresolved class coordinates".into());
    }
    Ok(())
}

pub fn validate_records(rows: &[Record], vocabulary: usize) -> Result<()> {
    if !(2..=200_000).contains(&rows.len())
        || rows
            .iter()
            .any(|r| r.label >= vocabulary || !r.feature.is_finite() || r.feature.abs() > 4096.)
        || rows.iter().map(|r| r.id).collect::<BTreeSet<_>>().len() != rows.len()
    {
        return Err("invalid distinct record IDs, features or labels".into());
    }
    Ok(())
}

impl Fit {
    pub fn new(coordinates: Vec<f64>, rows: &[Record]) -> Result<Self> {
        validate_coordinates(&coordinates)?;
        validate_records(rows, coordinates.len())?;
        let first = &rows[0];
        let second = rows
            .iter()
            .find(|r| r.feature != first.feature)
            .ok_or("two distinct observed feature anchors required")?;
        let slope = (coordinates[second.label] - coordinates[first.label])
            / (second.feature - first.feature);
        let intercept = coordinates[first.label] - slope * first.feature;
        if !slope.is_finite() || !intercept.is_finite() {
            return Err("nonfinite fitted decoder".into());
        }
        let mut residual: f64 = 0.;
        for r in rows {
            let center = slope * r.feature + intercept;
            let error = (center - coordinates[r.label]).abs();
            if !error.is_finite() || error > 1e-10 {
                return Err("observations contradict the affine coordinate decoder".into());
            }
            // A tolerance is not allowed to hide a wrong or tied class winner.
            if coordinates
                .iter()
                .enumerate()
                .any(|(c, e)| c != r.label && (center - e).abs() <= error)
            {
                return Err("observed decoder has an ambiguous raw winner".into());
            }
            residual = residual.max(error);
        }
        Ok(Self {
            coordinates,
            slope,
            intercept,
            anchor_ids: [first.id, second.id],
            max_training_residual: residual,
        })
    }

    pub fn validate(&self, rows: &[Record]) -> Result<()> {
        if *self != Self::new(self.coordinates.clone(), rows)? {
            return Err("saved coordinate fit differs from original training records".into());
        }
        Ok(())
    }

    /// CM56's complete sufficient weighted cost per unit shared margin.
    pub fn cost_per_margin(&self) -> f64 {
        4. * self.slope.abs() + 4. * self.intercept.abs() + 2.
    }
}
