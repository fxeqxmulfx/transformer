//! CM24 / Lean 75ca9276: balanced and missing-axis proposals added to the
//! complete CM21-23 report. The empirical search is not a joint convex oracle.
use crate::{
    C, Model, Result, feature_balance, interval, interval_audit, interval_search, refinement,
    refinement_audit, refinement_search, search,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
mod retained;
mod run;
pub use run::run;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub variants: Vec<interval::Options>,
    pub total_iterations: usize,
    pub model_audits: usize,
    pub refinement: refinement_search::Policy,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            variants: [0., 0.1, 0.3]
                .iter()
                .map(|&epsilon| interval::Options {
                    model_tolerance: epsilon,
                    iterations: 4096,
                    ..interval::Options::default()
                })
                .collect(),
            total_iterations: 73_728,
            model_audits: 24,
            refinement: refinement_search::Policy::default(),
        }
    }
}
impl Policy {
    fn validate(&self, n: usize) -> Result<()> {
        if self.variants.is_empty()
            || self.variants.len() > 4
            || !(1..=131_072).contains(&self.total_iterations)
            || !(1..=32).contains(&self.model_audits)
            || !(1..=128).contains(&self.refinement.rounds)
            || !(1..=32).contains(&self.refinement.seeds)
            || !(2..=8192).contains(&self.refinement.assignments)
            || self.variants.iter().enumerate().any(|(i, o)| {
                self.variants[..i]
                    .iter()
                    .any(|p| p.model_tolerance == o.model_tolerance)
            })
        {
            return Err("invalid bounded balancing policy".into());
        }
        for &options in &self.variants {
            interval::validate(&interval::Input {
                n,
                rank: 1,
                features: vec![C::default(); n],
                rows: [0, 1],
                options,
            })?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Subspace {
    pub exact_attempt: usize,
    pub reason: String,
    pub balanced: Option<feature_balance::Balanced>,
    pub audit: Option<Value>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub subspace: usize,
    /// None is the explicitly feasible uniform density, with no optimality claim.
    pub variant: Option<usize>,
    pub reason: String,
    pub extraction: Option<interval_audit::Artifact>,
    pub iterations_charged: usize,
    pub left: Option<Vec<C>>,
    pub right: Option<Vec<C>>,
    pub row_clock: Option<feature_balance::Clock>,
    pub column_clock: Option<feature_balance::Clock>,
    pub model: Option<Model>,
    pub model_audit: Option<Value>,
    pub refinement_reason: Option<String>,
    pub refinement: Option<refinement::Outcome>,
    pub refinement_audit: Option<Value>,
    pub assignments_charged: usize,
}
impl Attempt {
    fn new(subspace: usize, variant: Option<usize>) -> Self {
        Self {
            subspace,
            variant,
            reason: "missing_balanced_subspace".into(),
            extraction: None,
            iterations_charged: 0,
            left: None,
            right: None,
            row_clock: None,
            column_clock: None,
            model: None,
            model_audit: None,
            refinement_reason: None,
            refinement: None,
            refinement_audit: None,
            assignments_charged: 0,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cost {
    pub observation_coverage_visits: usize,
    pub feature_energy_terms: usize,
    pub repeated_sketches: usize,
    pub repeated_sketch_observations: usize,
    pub centering_cells: usize,
    pub uniform_column_products: usize,
    pub iterations: usize,
    pub iterations_charged: usize,
    pub right_phase_products: usize,
    pub decoder_observations: usize,
    pub model_audits: usize,
    pub seeds_run: usize,
    pub assignments_charged: usize,
    pub endpoint_audits: usize,
    pub latent_frequency_audit_allowance: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub format: u32,
    pub parent: refinement_search::Report,
    pub policy: Policy,
    pub observed_rows: Vec<bool>,
    pub observed_columns: Vec<bool>,
    pub subspaces: Vec<Subspace>,
    pub attempts: Vec<Attempt>,
    pub selected: Option<String>,
    pub cost: Cost,
    pub buffer_bound_bytes: u64,
}
/// All earlier candidates first, then each balanced seed and its refinement.
/// Counts and raw audits are retained even when a candidate loses by observed CE.
pub fn candidates(out: &Report) -> Result<Vec<(String, Model, f64)>> {
    let mut all = retained::candidates(&out.parent)?;
    for (i, a) in out.attempts.iter().enumerate() {
        if let Some(audit) = &a.model_audit {
            all.push((
                format!("balanced-{i}"),
                a.model.clone().ok_or("missing balanced seed")?,
                audit["train_ce"].as_f64().ok_or("missing seed CE")?,
            ));
        }
        if let Some(audit) = &a.refinement_audit
            && let Some(ce) = audit["endpoint"]["train_ce"].as_f64()
        {
            let o = a.refinement.as_ref().ok_or("missing balanced refinement")?;
            all.push((
                format!("balanced-refined-{i}"),
                o.steps[o.best.ok_or("missing best balanced state")?]
                    .state
                    .model(o.input.seed.vocab)?,
                ce,
            ));
        }
    }
    Ok(all)
}
pub fn replay(
    original: &interval_search::Input,
    prior: &[Model],
    parent: &refinement_search::Report,
    saved: &Report,
) -> Result<()> {
    if saved.format != 1
        || &saved.parent != parent
        || run(original, prior, parent.clone(), saved.policy.clone())? != *saved
    {
        return Err("balanced original input, parent or full state/work replay mismatch".into());
    }
    Ok(())
}
