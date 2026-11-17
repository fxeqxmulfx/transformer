//! Read-only actual CM105/106 error decomposition after every endpoint seals.
//! Current analytic residuals here are never used by forward-only training.
use crate::{Result, cache::Cache, experiment, history, score, source};
use bilinear::{
    experiment as original, guard,
    train::{Mode, Record},
};
use joint::capacity;
use rotary::model::Model;
use serde_json::{Value, json};
use std::{fs, path::Path};

pub fn analyze(
    model: &Model,
    cache: &Cache,
    records: &[Record<'_>],
    batch_size: usize,
    mut admit: impl FnMut(usize) -> Result<()>,
) -> Result<Value> {
    cache.validate(model)?;
    if records.is_empty() || batch_size == 0 || records.len() != cache.identities.len() {
        return Err("diagnostic record shape".into());
    }
    let mut a = capacity::zero(model);
    let (
        mut ce,
        mut signal_error,
        mut state_error,
        mut max_state_change,
        mut norm_excess,
        mut logit_error,
    ) = (0., 0., 0., 0f64, 0f64, 0f64);
    let (mut correct, mut offset) = (0usize, 0usize);
    for batch in records.chunks(batch_size) {
        admit(batch.len())?;
        let b = retained::coefficient::batch(model, batch, Mode::Mini, 1e-4)?;
        ce += b.mean_ce * batch.len() as f64;
        logit_error = logit_error.max(b.maximum_logit_error);
        for (i, r) in batch.iter().enumerate() {
            let id = offset + i;
            if retained::index::identity(r) != cache.identities[id] {
                return Err("diagnostic occurrence identity".into());
            }
            let mut d = 0f64;
            for (current, old) in b.states[i].iter().zip(&cache.states[id]) {
                for k in 0..2 {
                    d = d.max((current[k] - old[k]).norm());
                    norm_excess = norm_excess.max(current[k].norm() - 1.);
                }
            }
            max_state_change = max_state_change.max(d);
            signal_error += b.signals[i]
                .iter()
                .zip(&cache.signals[id])
                .map(|(r, s)| (r - s).abs())
                .sum::<f64>();
            state_error += cache.signals[id].iter().map(|s| s.abs()).sum::<f64>() * d;
            for (rows, state) in a.iter_mut().zip(&b.states[i]) {
                for (a, &s) in rows.iter_mut().zip(&b.signals[i]) {
                    for k in 0..2 {
                        a[k] += state[k] * s;
                    }
                }
            }
            let z = &b.logits[i];
            let predicted = (1..z.len()).fold(0, |best, c| if z[c] > z[best] { c } else { best });
            correct += usize::from(predicted == r.target);
        }
        offset += batch.len();
    }
    let n = records.len() as f64;
    joint::coefficient::mean(&mut a, records.len())?;
    signal_error /= n;
    state_error /= n;
    ce /= n;
    let e = signal_error + state_error;
    let bound = retained::diagnostic::error_bound(model, &a, &cache.mean)?;
    let cache_error = cache.recomputation_error(model)?;
    let peak_excess = (bound.coefficient_error_peak - e).max(0.);
    let correction = 2. * model.budget * e;
    let upper = -bound.estimated_gap + correction;
    let direction_excess = (bound.estimated_direction_actual_change - upper).max(0.);
    let gap_excess = ((bound.actual_gap - bound.estimated_gap).abs() - correction).max(0.);
    if cache_error > 1e-10
        || norm_excess > 2e-10
        || peak_excess > 1e-9
        || direction_excess > 1e-9
        || gap_excess > 1e-9
    {
        return Err("complete historical-product or CM106 bound failed".into());
    }
    Ok(
        json!({"records":records.len(),"correct":correct,"raw_accuracy":correct as f64/n,"mean_ce":ce,
        "coefficient_sha256":original::hash(&serde_json::to_vec(&a)?),"cm105":bound,
        "cm106":{"signal_error_mean":signal_error,"state_error_mean":state_error,"combined_error":e,
            "correction":correction,"direction_upper":upper,"strict_negative_condition":bound.estimated_gap>correction,
            "peak_bound_excess":peak_excess,"gap_bound_excess":gap_excess,"direction_bound_excess":direction_excess,
            "corrected_loss_lower_bound":ce-bound.estimated_gap-correction},
        "maximum_state_change":max_state_change,"maximum_state_norm_excess":norm_excess,
        "maximum_actual_logit_error":logit_error,"cache_recomputation_error":cache_error,
        "analytic_diagnostic_ce_evaluations":2*records.len(),"updates_accepted":0}),
    )
}
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, output: &Path) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    score::sealed(cohort)?;
    fs::create_dir(output)?;
    for arm in experiment::ARMS {
        for variant in experiment::VARIANTS {
            for mode in experiment::MODES {
                let path = experiment::cell(cohort, arm, variant, mode);
                let (model, _, task) = score::load(workspace, storage, &path)?;
                let cache: Cache = serde_json::from_slice(&fs::read(path.join("cache.json"))?)?;
                let history: Vec<history::Entry> = fs::read_to_string(path.join("history.jsonl"))?
                    .lines()
                    .map(serde_json::from_str)
                    .collect::<std::result::Result<_, _>>()?;
                let (base, _, _, _) = source::load(workspace, storage, variant, mode)?;
                let training = &task.splits["train"];
                let indices: Vec<_> = (0..training.len()).collect();
                let records = original::records(training, &indices)?;
                guard::admit(
                    model
                        .reservation(records.len())?
                        .checked_mul(8)
                        .ok_or("history reservation")?,
                    workspace,
                    storage,
                )?;
                let meta = original::score::read(&path.join("metadata.json"))?;
                if history.len() != 512
                    || history.last().ok_or("history")?.content_sha256
                        != readouts::source::content_hash(&model)?
                    || readouts::source::content_hash(&base)? != meta["initial_content_sha256"]
                {
                    return Err("history endpoint or initial content changed".into());
                }
                let checked = history::verify(&base, &records, &cache, &history)?;
                let report = analyze(&model, &cache, &records, 32, |n| {
                    guard::admit(
                        model
                            .reservation(n)?
                            .checked_mul(8)
                            .ok_or("diagnostic reservation")?,
                        workspace,
                        storage,
                    )?;
                    Ok(())
                })?;
                original::write_json(
                    &output.join(format!("{arm}-{variant}-{mode}.json")),
                    &json!({
            "source_fingerprint":experiment::fingerprint(),"arm":arm,"variant":variant,"mode":mode,
            "endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),
            "historical_records_replayed":checked,"history_versions":history.len(),"analysis":report,
            "claim":"training-only floating CM105/106 checks, not finite-rounding or unseen-answer certificates"}),
                )?;
            }
        }
    }
    Ok(())
}
