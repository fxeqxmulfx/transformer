//! CM10: fixed grouped spectral coordinates and ordinary original-observation CE.
//! Quotient.lean proves conditional convexity and the exact-fit CE certificate.
//! The training-only discrete grouping proposal has no recovery guarantee.
pub use convex_stress_budgets::{
    C, Model as Table, Result, Sample, check_layout, choose_by_train, data, gradient, hash,
    ideal_loss, linalg, loss, metrics, projection, random, samples as original_samples,
    solver::Mode,
};
pub mod grouping;
pub mod solver;
pub use grouping::Groups;
use serde::{Deserialize, Serialize};

/// Propose groups using training observations only, then freeze their maps.
pub fn prepare_model(
    task: &data::Task,
    budget: &str,
    overlap: usize,
) -> Result<(Model, Vec<Sample>)> {
    check_layout(task)?;
    let n = task.vocab;
    let raw = original_samples(&Table::new(n, n as f64, n as f64)?, &task.splits["train"])?;
    let groups = Groups::discover(n, &raw, overlap)?;
    let mapped = groups.map_samples(&raw)?;
    let (r, b) = budgets(budget, &groups)?;
    Ok((
        Model {
            groups,
            table: Table::new(n, r, b)?,
        },
        mapped,
    ))
}

/// Fixed group/vocabulary dimensions determine budgets; no target spectrum is read.
pub fn budgets(label: &str, groups: &Groups) -> Result<(f64, f64)> {
    groups.validate(groups.rows.len())?;
    let (r, c) = (groups.row_count as f64, groups.column_count as f64);
    match label {
        "rank_one" => Ok(((r * c).sqrt(), (r * c).sqrt())),
        "full_rank" => {
            let radius = (groups.rows.len() as f64).sqrt();
            Ok((radius, r.min(c) * radius))
        }
        _ => Err("unknown grouped spectral budget".into()),
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub groups: Groups,
    pub table: Table,
}

impl Model {
    pub fn validate_shape(&self) -> Result<()> {
        self.table.validate_shape()?;
        self.groups.validate(self.table.vocab)?;
        let n = self.table.vocab;
        for col in 0..n {
            for row in 0..n {
                if (row >= self.groups.row_count || col >= self.groups.column_count)
                    && self.table.probabilities[(row + n * col) * n..(row + n * col + 1) * n]
                        .iter()
                        .any(|p| (p - 1. / n as f64).abs() > 1e-10)
                {
                    return Err("inactive grouped coordinate is not uniform".into());
                }
            }
        }
        Ok(())
    }

    pub fn cell(&self, prefix: &[i64]) -> usize {
        self.groups.cell(self.table.cell(prefix))
    }
}

impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.table.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let n = self.table.vocab;
        let start = self.cell(prefix) * n;
        self.table.probabilities[start..start + n].to_vec()
    }
}

#[cfg(test)]
mod tests;
