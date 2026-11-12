//! CM11: certify a sharper endpoint in the unchanged grouped spectral family.
//! No optimizer iteration, task operator or held-out label enters this step.
//! Keep original endpoints already at ideal CE, or when the proposal fails to
//! improve training CE. The saved predictor remains a fixed lookup of a table.
pub use convex_stress_sharing::{
    C, Groups, Model, Result, Sample, Table, budgets, choose_by_train, data, hash, ideal_loss,
    linalg, loss, metrics, prepare_model, projection,
};
use projection::{Norms, Projector};
use serde::{Deserialize, Serialize};

pub fn winner(p: &[f64]) -> usize {
    let mut best = 0;
    for c in 1..p.len() {
        if p[c].total_cmp(&p[best]).is_gt() {
            best = c;
        }
    }
    best
}

/// Mean ordinary CE retains every original sample, including grouped repeats.
pub fn training_ce(model: &Model, samples: &[Sample]) -> Result<f64> {
    let n = model.table.vocab;
    if samples.is_empty() || samples.iter().any(|s| s.cell >= n * n || s.label >= n) {
        return Err("invalid original training observations".into());
    }
    Ok(samples
        .iter()
        .map(|s| {
            loss(
                &model.table.probabilities[s.cell * n..(s.cell + 1) * n],
                s.label,
            )
        })
        .sum::<f64>()
        / samples.len() as f64)
}

pub fn observed_correct(model: &Model, samples: &[Sample]) -> usize {
    let n = model.table.vocab;
    samples
        .iter()
        .filter(|s| winner(&model.table.probabilities[s.cell * n..(s.cell + 1) * n]) == s.label)
        .count()
}

/// Direct argmax readout on active cells; padded cells remain exactly uniform.
pub fn decoded(model: &Model) -> Result<Model> {
    model.validate_shape()?;
    let n = model.table.vocab;
    let mut rounded = model.clone();
    rounded.table.probabilities.fill(1. / n as f64);
    for col in 0..model.groups.column_count {
        for row in 0..model.groups.row_count {
            let start = (row + n * col) * n;
            let class = winner(&model.table.probabilities[start..start + n]);
            rounded.table.probabilities[start..start + n].fill(0.);
            rounded.table.probabilities[start + class] = 1.;
        }
    }
    Ok(rounded)
}

pub fn same_decisions(a: &Model, b: &Model) -> Result<bool> {
    a.validate_shape()?;
    b.validate_shape()?;
    if a.table.vocab != b.table.vocab || a.groups != b.groups {
        return Ok(false);
    }
    let n = a.table.vocab;
    Ok((0..a.groups.column_count).all(|col| {
        (0..a.groups.row_count).all(|row| {
            let start = (row + n * col) * n;
            winner(&a.table.probabilities[start..start + n])
                == winner(&b.table.probabilities[start..start + n])
        })
    }))
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub selected: String,
    pub reason: String,
    pub original_training_ce: f64,
    pub proposed_training_ce: f64,
    pub selected_training_ce: f64,
    pub original_correct: usize,
    pub original_examples: usize,
    pub mixture_alpha: f64,
    pub decoded_norms: Vec<Norms>,
    pub proposed_norms: Vec<Norms>,
    pub selected_norms: Vec<Norms>,
    pub proposed_excess_bound: Option<f64>,
    pub all_active_decisions_unchanged: bool,
}

pub struct Outcome {
    pub proposed: Model,
    pub selected: Model,
    pub certificate: Certificate,
}

pub fn polish(
    model: &Model,
    samples: &[Sample],
    workers: usize,
    ideal_tolerance: f64,
    improvement_tolerance: f64,
) -> Result<Outcome> {
    if !ideal_tolerance.is_finite()
        || ideal_tolerance < 0.
        || !improvement_tolerance.is_finite()
        || improvement_tolerance < 0.
    {
        return Err("invalid endpoint tolerances".into());
    }
    model.validate_shape()?;
    let n = model.table.vocab;
    let mut projector = Projector::new(n, workers)?;
    let original_norms = model.table.validate(&mut projector)?;
    let original_ce = training_ce(model, samples)?;
    let correct = observed_correct(model, samples);
    let rounded = decoded(model)?;
    let (table, alpha, decoded_norms) = projector.feasible(
        &rounded.table.probabilities,
        model.table.radius,
        model.table.nuclear_budget,
    )?;
    if !(0. < alpha && alpha <= 1.) {
        return Err("nonpositive sharpening mixture".into());
    }
    let proposed = Model {
        groups: model.groups.clone(),
        table,
    };
    proposed.validate_shape()?;
    let proposed_norms = proposed.table.validate(&mut projector)?;
    if !same_decisions(model, &proposed)? {
        return Err("sharpening changed a decoded class".into());
    }
    let proposed_ce = training_ce(&proposed, samples)?;
    let ideal = ideal_loss(n);
    if original_ce < ideal - 1e-10 || proposed_ce < ideal - 1e-10 {
        return Err("CE below the probability-logit lower bound".into());
    }
    let bound = if correct == samples.len() {
        let bound = (1. - alpha) * ((n as f64).ln() - ideal);
        if proposed_ce - ideal > bound + 1e-10 {
            return Err("decoded CE bound violated".into());
        }
        Some(bound)
    } else {
        None
    };
    let reason = if original_ce - ideal <= ideal_tolerance {
        "already_ideal"
    } else if correct != samples.len() {
        "observed_decoding_error"
    } else if proposed_ce >= original_ce - improvement_tolerance {
        "no_training_ce_improvement"
    } else {
        "training_ce_improved"
    };
    let changed = reason == "training_ce_improved";
    let selected = if changed {
        proposed.clone()
    } else {
        model.clone()
    };
    let certificate = Certificate {
        selected: if changed { "sharpened" } else { "original" }.into(),
        reason: reason.into(),
        original_training_ce: original_ce,
        proposed_training_ce: proposed_ce,
        selected_training_ce: if changed { proposed_ce } else { original_ce },
        original_correct: correct,
        original_examples: samples.len(),
        mixture_alpha: alpha,
        decoded_norms,
        proposed_norms: proposed_norms.clone(),
        selected_norms: if changed {
            proposed_norms
        } else {
            original_norms
        },
        proposed_excess_bound: bound,
        all_active_decisions_unchanged: true,
    };
    Ok(Outcome {
        proposed,
        selected,
        certificate,
    })
}

#[cfg(test)]
mod tests;
