//! Compare the same four controls under two independently frozen depth
//! coordinates. Only original training rows enter the endpoint slope checks.
#[path = "shared/audit.rs"]
mod archived;
#[path = "depth-signals/measurement.rs"]
mod measurement;
use baseline::{data::Example, hash};
use bilinear::guard;
use composed_retrieval::{Result, data, model::Model, operator::Tables, train};
use learned_retrieval::cohort::read;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn training_slope(m: &Model, rows: &[Example], eps: f64, z: Option<f64>) -> Result<Value> {
    let rows = train::samples(m, rows)?;
    let rows: Vec<_> = rows.iter().flatten().collect();
    let lengths: BTreeSet<_> = rows
        .iter()
        .map(|s| s.prefix.len().saturating_sub(1))
        .collect();
    let t = Tables::new(m, &lengths.into_iter().collect::<Vec<_>>())?;
    let view = measurement::measure(m, &t, &rows, eps)?;
    let raw_slope = z.map(|_| view.slope * view.depth * (1. - view.depth));
    Ok(json!({"physical":view,"log_odds_slope":raw_slope}))
}
fn final_splits(report: &Value, cell: &str) -> Result<Value> {
    let scored = &report["cells"][format!("{cell}-final")];
    let mut splits = BTreeMap::new();
    for (id, s) in scored["splits"].as_object().ok_or("original splits")? {
        let m = &s["metrics"];
        splits.insert(
            id,
            json!({"objects":[m["correct"],m["targets"]],"exact":[m["exact"],m["examples"]]}),
        );
    }
    Ok(
        json!({"original":scored["original_success"],"strict":scored["strict_all_heldout"],"splits":splits}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [workspace, storage, control, trial] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE MATCHED_FOUR_OUTPUT LOG_ODDS_FOUR_OUTPUT".into());
    };
    let (workspace, storage) = (Path::new(workspace), Path::new(storage));
    guard::admit(1 << 30, workspace, storage)?;
    let p = data::protocol()?;
    p.initial_log_odds.ok_or("compile for the log-odds study")?;
    let cohorts = [
        ("control", Path::new(control), data::DENSE_PROTOCOL),
        ("logodds", Path::new(trial), data::PROTOCOL),
    ];
    let mut totals = BTreeMap::new();
    for (name, root, definition) in cohorts {
        let audit = archived::audit(workspace, storage, root)?;
        if audit["summary"]["measured"] != 4 {
            return Err("incomplete four-case control".into());
        }
        for arm in ["original", "repeat"] {
            let seal = read(&root.join(arm).join("run/sealed/manifest.json"))?;
            if seal["identity"]["protocol_sha256"] != hash(definition.as_bytes()) {
                return Err("incorrect frozen coordinate".into());
            }
        }
        totals.insert(name, audit["summary"].clone());
    }
    let mut cases = BTreeMap::new();
    for id in p.cases.keys() {
        let (task, pins) = data::original(&p, id)?;
        let mut cells = BTreeMap::<String, Value>::new();
        for (name, root, _) in cohorts {
            let dir = root.join("original/run").join(id);
            let report = read(&dir.join("scores/report.json"))?;
            for mode in &p.modes {
                for seed in &p.seeds {
                    let cell = format!("{mode}-{seed}");
                    let path = dir.join(&cell);
                    let receipt = read(&path.join("receipt.json"))?;
                    let model: Model = serde_json::from_slice(&fs::read(path.join("model.json"))?)?;
                    let l = model.validate()?;
                    let training = &receipt["training"];
                    let matched = json!([
                        training["batch_sha256"],
                        training["targets"],
                        training["examples"],
                        training["updates"],
                        training["forward_loss_queries"],
                        training["min_visits"],
                        training["max_visits"]
                    ]);
                    let entry = cells
                        .entry(cell.clone())
                        .or_insert_with(|| json!({"matched":matched}));
                    if entry["matched"] != matched || receipt["data_fingerprints"] != pins {
                        return Err("unmatched training data or visits".into());
                    }
                    let z = training["depth_log_odds"].as_f64();
                    if (name == "logodds") != z.is_some()
                        || z.is_some_and(|z| {
                            composed_retrieval::model::sigmoid(z) != model.weights[l.depth]
                        })
                    {
                        return Err("model depth disagrees with training coordinate".into());
                    }
                    let slope = if id == "suite-lookup" {
                        let s = training_slope(&model, &task.splits["train"], p.query_step, z)?;
                        let ce = s["physical"]["ce"].as_f64().ok_or("replayed training CE")?;
                        if (ce
                            - training["after"]["ce"]
                                .as_f64()
                                .ok_or("sealed training CE")?)
                        .abs()
                            > 1e-10
                            || s["physical"]["targets"] != training["after"]["targets"]
                        {
                            return Err("training observations differ".into());
                        }
                        s
                    } else {
                        Value::Null
                    };
                    entry[name] = json!({"depth":model.weights[l.depth],"product":model.weights[l.product],
                    "log_odds":z,"training_depth_signal":slope,"scores":final_splits(&report,&cell)?});
                }
            }
        }
        cases.insert(id, cells);
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"summary":totals,"cases":cases}),
    )?;
    println!();
    Ok(())
}
