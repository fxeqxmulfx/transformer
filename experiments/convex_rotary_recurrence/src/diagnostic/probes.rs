//! Existing coarse frequency grid and every finer dyadic step through2^-20.
//! Every response keeps all original records/classes and refits the complete readout.
//! Independent actual mixture-model CE checks the proposal comparison calculation.
use crate::{
    Result,
    atom::Atom,
    model::Model,
    proposal::Search,
    train::{self, Config},
};
use bilinear::{experiment as original, model::cross_entropy, train::Record};
use serde_json::{Value, json};
use std::path::Path;

fn actual_loss(centre: &Model, atom: Atom, records: &[Record<'_>]) -> Result<f64> {
    let alpha = (1. / centre.budget).min(0.1);
    let mut point = centre.clone();
    point.mix(atom, alpha)?;
    let mut loss = 0.;
    for r in records {
        loss += cross_entropy(&point.logits(r.prefix)?, r.target)?;
    }
    Ok(loss / records.len() as f64)
}

pub(super) fn measure(
    centre: &Model,
    atom: &Atom,
    records: &[Record<'_>],
    config: Config,
) -> Result<Value> {
    if records.is_empty() || atom.frequency.is_none() {
        return Err("empty or nonrotary diagnostic".into());
    }
    centre.validate()?;
    let base: Vec<_> = records
        .iter()
        .map(|r| centre.logits(r.prefix))
        .collect::<Result<_>>()?;
    let signals: Vec<_> = base
        .iter()
        .zip(records)
        .map(|(l, r)| train::scalars(l, r.target, config.mode, config.query_step))
        .collect::<Result<_>>()?;
    let mut search = Search {
        family: centre.family,
        frequency_start: config.frequency_start,
        base: &base,
        signals: &signals,
        records,
        budget: centre.budget,
        loss_queries: 0,
        coordinate_trials: [0; 3],
    };
    let (reference, reference_ce) = search.improve(atom.clone(), 0, false)?;
    let actual_reference = actual_loss(centre, reference.clone(), records)?;
    let reference_states: Vec<_> = records
        .iter()
        .map(|r| reference.state(r.prefix))
        .collect::<Result<_>>()?;
    let mut directions = vec![];
    for coordinate in 0..2 {
        for sweep in 0..22 {
            let radius = 2f64.powi(1 - sweep);
            for sign in [-1., 1.] {
                let mut candidate = reference.clone();
                candidate.frequency.as_mut().ok_or("frequency")?[coordinate] += sign * radius;
                let fixed_readout_ce = actual_loss(centre, candidate.clone(), records)?;
                let (candidate, ce) = search.improve(candidate, 0, false)?;
                let actual_ce = actual_loss(centre, candidate.clone(), records)?;
                let discrepancy = (actual_ce - ce).abs();
                if discrepancy > 2e-12 {
                    return Err(
                        "actual feasible mixture CE differs from component comparison".into(),
                    );
                }
                let mut change = 0f64;
                for (record, before) in records.iter().zip(&reference_states) {
                    let after = candidate.state(record.prefix)?;
                    for j in 0..2 {
                        change = change.max((after[j] - before[j]).norm());
                    }
                }
                directions.push(json!({"coordinate":coordinate,"delta":sign*radius,"radius":radius,
                    "original_grid":sweep<6,"component_ce":ce,"component_improvement":reference_ce-ce,
                    "actual_model_ce":actual_ce,"actual_model_improvement":actual_reference-actual_ce,
                    "fixed_readout_model_ce":fixed_readout_ce,"model_ce_discrepancy":discrepancy,
                    "maximum_state_change":change,"maximum_operator_norm_excess":candidate.validate(centre.objects,centre.family)?}));
            }
        }
    }
    let best = |coarse: bool| {
        directions
            .iter()
            .filter(|d| d["original_grid"] == coarse)
            .min_by(|a, b| {
                a["component_ce"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["component_ce"].as_f64().unwrap())
            })
            .cloned()
    };
    let unchanged = serde_json::to_value(&reference)? == serde_json::to_value(atom)?;
    Ok(
        json!({"scope":"last selected component on its original final minibatch",
        "records":records.len(),"classes":centre.objects,"frequency":reference.frequency,
        "component_comparison_ce":reference_ce,"actual_model_ce":actual_reference,
        "refit_matches_saved_component":unchanged,"best_original_grid":best(true),"best_finer_grid":best(false),
        "component_loss_queries":search.loss_queries,"actual_model_loss_queries":(1+2*directions.len())*records.len(),
        "directions":directions,"heldout_labels_used":false,"model_update_accepted":false}),
    )
}

pub(super) fn run(
    centre: &Model,
    atom: &Atom,
    records: &[Record<'_>],
    config: Config,
    saved_ce: f64,
    output: &Path,
) -> Result<()> {
    let report = measure(centre, atom, records, config)?;
    if report["component_comparison_ce"].as_f64() != Some(saved_ce)
        || report["refit_matches_saved_component"] != true
    {
        return Err(
            "final component criterion does not exactly reconstruct the recorded search".into(),
        );
    }
    original::write_json(&output.join("probes.json"), &report)?;
    println!(
        "{}",
        json!({"baseline":report["component_comparison_ce"],
        "best_original_grid":report["best_original_grid"],"best_finer_grid":report["best_finer_grid"]})
    );
    Ok(())
}
