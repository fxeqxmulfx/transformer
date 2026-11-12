//! Save the constructed endpoint before separate raw-coordinate replay.
use super::*;
use std::time::Instant;
pub fn run(root: &Path, id: &str) -> Result<()> {
    let started = Instant::now();
    let c = case(id)?;
    let inv = inventory(root)?;
    let x = inputs(&c)?;
    execute(root, c, inv, x, started)
}
pub(super) fn execute(root: &Path, c: Case, inv: Value, x: Inputs, started: Instant) -> Result<()> {
    check_task(&inv, &c, &x)?;
    let n = x.task.vocab;
    let count = x.task.splits["train"].len();
    let id = c.id();
    let out = root.join(&id);
    fs::create_dir(&out)?;
    let mut record = json!({"case":c,"id":id,"source_fingerprint":fingerprint(),"protocol":protocol()?,
        "producer_commit":inv["producer_commit"],"parents":inv["parents"],"vocab":n,
        "data_fingerprints":x.row["data_fingerprints"],"selection":x.task.selection,
        "discovery":x.discovery,"coding":x.coding,"explicit_bytes_upper_bound":x.explicit_bytes,
        "training_examples":count,"coding_examples_presented":count,"updates":0,
        "training_gradient_evaluations":0,"optimizer_examples_presented":0});
    write_new(&out.join("metadata.json"), &record)?;
    let (model, proposal) = construct(&x)?;
    record["proposal"] = proposal;
    record["construction_seconds_including_data"] = started.elapsed().as_secs_f64().into();
    let Some(model) = model else {
        record["status"] = "proposal_ineligible".into();
        record["total_seconds_including_data_and_proposal"] =
            started.elapsed().as_secs_f64().into();
        return write_new(&out.join("result.json"), &record);
    };
    let check_started = Instant::now();
    let norms = {
        let mut projector = cold::projection::Projector::new(n, 1)?;
        model.table.validate(&mut projector)?
    };
    record["spectral_check_seconds"] = check_started.elapsed().as_secs_f64().into();
    let metrics_started = Instant::now();
    let values = cold::cohort::metrics_all(&model, &x.task)?;
    let excess = number(&values["train"], "loss")? - cold::ideal_loss(n);
    let mut fit = 0_f64;
    for sample in &x.samples {
        for c in 0..n {
            fit = fit.max(
                (model.table.probabilities[sample.cell * n + c] - f64::from(c == sample.label))
                    .abs(),
            );
        }
    }
    if fit != 0. || excess.abs() > 1e-10 {
        return Err("constructed endpoint is not the observed ideal fit".into());
    }
    record["metrics_seconds"] = metrics_started.elapsed().as_secs_f64().into();
    record["status"] = "complete".into();
    record["stop_reason"] = "exact_observed_fit_constructor".into();
    record["radius"] = model.table.radius.into();
    record["nuclear_budget"] = model.table.nuclear_budget.into();
    record["moment_norms"] = serde_json::to_value(norms)?;
    record["fit_defect"] = fit.into();
    record["ideal_ce_excess"] = excess.into();
    record["metrics"] = values;
    let save_started = Instant::now();
    write_new(&out.join("model.json"), &model)?;
    record["model_sha256"] = hash(&fs::read(out.join("model.json"))?).into();
    record["save_seconds"] = save_started.elapsed().as_secs_f64().into();
    record["total_seconds_including_data_and_proposal"] = started.elapsed().as_secs_f64().into();
    write_new(&out.join("result.json"), &record)
}
