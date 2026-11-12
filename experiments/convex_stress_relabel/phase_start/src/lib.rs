//! CM14, Lean ae9a35ed: a categorical phase completion is feasible and CE optimal.
//! This constructs the probability table from training only. Unknown truth may
//! violate the phase structure. A supplied class coding is fixed, not optimized
//! here; its discovery and this extra observation pass must both be charged.
use convex_relabel_graph_audit::graph::{self, Pair, Structure};
use convex_stress_relabel::{Coding, Model, Result, Sample, Table, memory_bound};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rejection {
    Disconnected,
    NoPerfectMatching,
    InconsistentPhases,
}

pub struct Proposal {
    pub model: Option<Model>,
    pub structure: Structure,
    pub rejection: Option<Rejection>,
    pub examples_presented: usize,
    pub explicit_bytes_upper_bound: u64,
}

/// Restrict admission to the connected matching case of CM14. The class map
/// and every observed phase equation are checked before dense allocation.
/// The resulting model uses the unchanged rank_one radius/budget, both N.
/// Inference is the existing table lookup and output reindexing, not a phase
/// formula evaluated on input-token values. No loss derivative is constructed.
pub fn propose(n: usize, training: &[Sample], coding: &Coding) -> Result<Proposal> {
    let explicit_bytes = memory_bound(n)?;
    if explicit_bytes > 2 * 1024 * 1024 * 1024 {
        return Err("phase start exceeds the explicit memory cap".into());
    }
    if training.is_empty() || training.len() > n * n {
        return Err("phase start requires between one and N squared observed cells".into());
    }
    coding.validate(n)?;
    let pairs: Vec<_> = training
        .iter()
        .map(|x| Pair {
            row: x.cell % n,
            column: x.cell / n,
            label: x.label,
        })
        .collect();
    let structure = graph::analyze(n, &pairs, &coding.encode)?;
    let rejection = if !structure.connected {
        Some(Rejection::Disconnected)
    } else if structure.perfect_matching.is_none() {
        Some(Rejection::NoPerfectMatching)
    } else if !structure.observed_phase_consistent {
        Some(Rejection::InconsistentPhases)
    } else {
        None
    };
    let model = if rejection.is_none() {
        let labels = structure
            .completion_labels
            .as_ref()
            .ok_or("missing phase completion")?;
        let mut table = Table {
            vocab: n,
            radius: n as f64,
            nuclear_budget: n as f64,
            probabilities: vec![0.; n * n * n],
        };
        for (cell, &label) in labels.iter().enumerate() {
            table.probabilities[cell * n + coding.encode[label]] = 1.;
        }
        let model = Model {
            coding: coding.clone(),
            table,
        };
        model.validate_shape()?;
        // Check the literal original training labels in the stored coordinates.
        if training
            .iter()
            .any(|x| model.table.probabilities[x.cell * n + coding.encode[x.label]] != 1.)
        {
            return Err("phase candidate does not interpolate its training observations".into());
        }
        Some(model)
    } else {
        None
    };
    Ok(Proposal {
        model,
        structure,
        rejection,
        examples_presented: training.len(),
        explicit_bytes_upper_bound: explicit_bytes,
    })
}

#[cfg(test)]
mod tests;
