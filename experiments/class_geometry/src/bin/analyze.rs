//! Derived exact attribution of active-class energy share over every update.
//! Uses every frozen slot and the original 0/5000/10000 checkpoint windows.
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn n(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn coverage(row: &Value, family: &str, rank: u64) -> f64 {
    n(&row["coverage"]["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["family"] == family && r["rank"] == rank)
        .unwrap()["coverage"])
}
fn first(rows: &[Value], field: &str, threshold: f64) -> Value {
    rows.iter()
        .find(|r| n(&r[field]) >= threshold)
        .map(|r| r["step"].clone())
        .unwrap_or(Value::Null)
}
/// Aggregate every recorded fold, including deficient ranks and conditioning.
fn conditioning(root: &Path, id: u64, probes: &[Value]) -> Result<Value> {
    let mut groups: [Vec<Value>; 2] = std::array::from_fn(|_| vec![]);
    for probe in probes {
        let step = probe["step"].as_u64().unwrap();
        let frozen: Value = serde_json::from_reader(BufReader::new(File::open(
            root.join(format!("seed{id}/step-{step:06}/frozen.json")),
        )?))?;
        for (index, family) in ["hidden", "readout"].into_iter().enumerate() {
            for group in frozen["geometry"][family].as_array().unwrap() {
                for fold in group["folds"].as_array().unwrap() {
                    groups[index].push(fold["geometry"].clone());
                }
            }
        }
    }
    let mut output = vec![];
    for (family, folds) in ["hidden", "readout"].into_iter().zip(groups) {
        for rank in [0u64, 32, 33, 128] {
            let rows: Vec<_> = folds
                .iter()
                .map(|fold| {
                    fold["ranks"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|row| row["requested_rank"] == rank)
                        .unwrap()
                })
                .collect();
            let range = |field: &str| {
                let values: Vec<_> = rows.iter().map(|r| n(&r[field])).collect();
                json!([
                    values.iter().copied().fold(f64::INFINITY, f64::min),
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                ])
            };
            output.push(json!({"family":family,"rank":rank,"fold_count":folds.len(),
                "achieved_rank":range("achieved_rank"),
                "retained_readout_energy":range("retained_readout_energy"),
                "retained_training_energy":range("retained_training_energy"),
                "coefficient_norm":range("max_coefficient_norm"),
                "orthogonality_penalty":range("max_orthogonality_penalty"),
                "smallest_training_singular":folds.iter().map(|f|n(&f["smallest_retained_training_singular_value"])).fold(f64::INFINITY,f64::min)}));
        }
    }
    Ok(json!(output))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(args.len() == 1, "usage: analyze OUTPUT_DIRECTORY")?;
    let root = Path::new(&args[0]);
    let report: Value =
        serde_json::from_reader(BufReader::new(File::open(root.join("report.json"))?))?;
    require(report["status"] == "pass", "cohort failed")?;
    let mut output = vec![];
    for seed in report["seeds"].as_array().unwrap() {
        let id = seed["seed"].as_u64().unwrap();
        let file = File::open(root.join(format!("seed{id}/updates.jsonl")))?;
        let mut terms = [[0.; 7]; 2];
        let mut positive = [0; 2];
        let mut changes = [0.; 2];
        let mut max_error = 0f64;
        let mut initial = 0.;
        let mut final_share = 0.;
        let mut increases = 0;
        let mut updates = 0;
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let r: Value = serde_json::from_str(&line?)?;
            require(r["step"] == index + 1, "wrong update order")?;
            let s = n(&r["before"][0]);
            let v = n(&r["before"][1]) + n(&r["before"][2]);
            let next_s = n(&r["after"][0]);
            let next_v = n(&r["after"][1]) + n(&r["after"][2]);
            require(s + v > 0. && next_s + next_v > 0., "zero centered energy")?;
            let denominator = (s + v) * (next_s + next_v);
            let components = std::array::from_fn::<_, 7, _>(|j| {
                let ds = if j < 6 {
                    n(&r["radial"][0][j])
                } else {
                    n(&r["quadratic"][0])
                };
                let dv = if j < 6 {
                    n(&r["radial"][1][j]) + n(&r["radial"][2][j])
                } else {
                    n(&r["quadratic"][1]) + n(&r["quadratic"][2])
                };
                (v * ds - s * dv) / denominator
            });
            let change = next_s / (next_s + next_v) - s / (s + v);
            let window = usize::from(index >= 5000);
            for (j, value) in components.into_iter().enumerate() {
                terms[window][j] += value;
            }
            changes[window] += change;
            positive[window] += usize::from(change > 1e-12);
            increases += usize::from(change > 1e-12);
            max_error = max_error.max((components.iter().sum::<f64>() - change).abs());
            if index == 0 {
                initial = s / (s + v);
            }
            final_share = next_s / (next_s + next_v);
            max_error =
                max_error.max((initial + terms.iter().flatten().sum::<f64>() - final_share).abs());
            updates += 1;
        }
        require(updates == 10000, "incomplete update table")?;
        require(max_error < 1e-8, "share attribution failed to telescope")?;
        let rows:Vec<_>=seed["geometry"].as_array().unwrap().iter().map(|r|json!({"step":r["step"],"accuracy":r["metrics"]["validation"]["accuracy"],"loss":r["metrics"]["validation"]["loss"],"hidden32":coverage(r,"hidden",32),"readout32":coverage(r,"readout",32),"readout33":coverage(r,"readout",33),"readout128":coverage(r,"readout",128),"structural":r["structural"]})).collect();
        let late: Vec<_> = rows
            .iter()
            .filter(|r| r["step"].as_u64().unwrap() >= 5000)
            .collect();
        require(late.len() == 21, "missing late window")?;
        let late_min = json!({"accuracy":late.iter().map(|r|n(&r["accuracy"])).fold(f64::INFINITY,f64::min),"readout32":late.iter().map(|r|n(&r["readout32"])).fold(f64::INFINITY,f64::min),"readout33":late.iter().map(|r|n(&r["readout33"])).fold(f64::INFINITY,f64::min),"active_share":late.iter().map(|r|n(&r["structural"]["active_fraction"])).fold(f64::INFINITY,f64::min)});
        let crossings:Vec<_>=[0.5,0.9,0.99].into_iter().map(|level|json!({"level":level,"accuracy":first(&rows,"accuracy",level),"hidden32":first(&rows,"hidden32",level),"readout32":first(&rows,"readout32",level),"readout33":first(&rows,"readout33",level)})).collect();
        let conditioning = conditioning(root, id, seed["geometry"].as_array().unwrap())?;
        output.push(json!({"seed":id,"replication":!([7,11].contains(&id)),"initial_active_share":initial,"final_active_share":final_share,"share_increase_updates":increases,"max_share_identity_error":max_error,"conditioning":conditioning,
            "share_terms":["decay_radial","isotropic_readout","adaptive_readout_correction","tied_encoder","moment_memory","rounding","total_quadratic"],
            "whole_run_share_contributions":(0..7).map(|j|terms[0][j]+terms[1][j]).collect::<Vec<_>>(),
            "windows":[{"from":0,"to":5000,"share_change":changes[0],"contributions":terms[0],"increasing_updates":positive[0]},{"from":5000,"to":10000,"share_change":changes[1],"contributions":terms[1],"increasing_updates":positive[1]}],"late_min":late_min,"crossings":crossings,"endpoint":rows.last(),"rows":rows,"absolute_energy":seed["dynamics"]}));
    }
    let rep: Vec<_> = output.iter().filter(|r| r["replication"] == true).collect();
    let success:Vec<_>=[0.5,0.9,0.99].into_iter().map(|level|json!({"level":level,"replication_count":rep.len(),"endpoint_accuracy":rep.iter().filter(|r|n(&r["endpoint"]["accuracy"])>=level).count(),"endpoint_readout32":rep.iter().filter(|r|n(&r["endpoint"]["readout32"])>=level).count(),"all_late_accuracy":rep.iter().filter(|r|n(&r["late_min"]["accuracy"])>=level).count(),"all_late_readout32":rep.iter().filter(|r|n(&r["late_min"]["readout32"])>=level).count()})).collect();
    println!(
        "{}",
        json!({"status":report["status"],"seconds":report["seconds"],"source_sha256":report["source_sha256"],"seeds":output,"replication_threshold_counts":success})
    );
    Ok(())
}
