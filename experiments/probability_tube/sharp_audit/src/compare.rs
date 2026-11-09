//! Compare complete sealed inventories; this command may read verified future data.
use crate::{
    Result,
    evidence::{self, Ledger, require},
};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
fn sealed(root: &Path, ledger: &mut Ledger) -> Result<Value> {
    let bytes = ledger.read(&root.join("construction.json"))?;
    require(
        evidence::hash(&bytes) == fs::read_to_string(root.join("construction.sha256"))?,
        "construction seal",
    )?;
    let value: Value = serde_json::from_slice(&bytes)?;
    require(
        value["cases"].as_array().is_some_and(|c| c.len() == 144),
        "complete construction",
    )?;
    Ok(value)
}
fn verification(root: &Path, ledger: &mut Ledger) -> Result<(Value, f64)> {
    let mut last = Value::Null;
    let mut seconds = 0.;
    let mut attempt = 1;
    while root
        .join(format!("verification-{attempt:02}.json"))
        .exists()
    {
        last = ledger.json(&root.join(format!("verification-{attempt:02}.json")))?;
        require(last["status"] != "running", "verification still running")?;
        seconds += last["seconds"].as_f64().ok_or("verification time")?;
        attempt += 1;
    }
    require(
        last["cases"].as_array().is_some_and(|c| c.len() == 144),
        "verification missing/incomplete",
    )?;
    Ok((last, seconds))
}
fn payload(root: &Path, slot: &Value, ledger: &mut Ledger) -> Result<Value> {
    require(slot["status"] == "frozen", "unavailable forecast")?;
    let bytes =
        ledger.read(&root.join(slot["forecast"]["file"].as_str().ok_or("forecast file")?))?;
    require(
        evidence::hash(&bytes) == slot["forecast"]["sha256"],
        "forecast hash",
    )?;
    Ok(serde_json::from_slice(&bytes)?)
}
fn excess(a: &Value, b: &Value) -> Result<(f64, f64)> {
    let (a, b) = (
        a.as_array().ok_or("refined array")?,
        b.as_array().ok_or("original array")?,
    );
    require(
        a.len() == 4704 && b.len() == 4704,
        "complete query comparison",
    )?;
    let mut difference = f64::NEG_INFINITY;
    let mut scaled = f64::NEG_INFINITY;
    for (a, b) in a.iter().zip(b) {
        let a = a.as_f64().ok_or("refined bound")?;
        let b = b.as_f64().ok_or("original bound")?;
        difference = difference.max(a - b);
        scaled = scaled.max((a - b) / (1. + a.abs().max(b.abs())));
    }
    Ok((difference, scaled))
}
fn assess(original: &Value, refined: &Value, ov: &Value, rv: &Value) -> Result<Value> {
    for key in [
        "cohort",
        "seed",
        "anchor",
        "active",
        "alpha",
        "r",
        "eta",
        "s",
        "old_q",
        "old_query_probabilities",
        "training_inputs",
        "old_contrast_norm",
        "old_execution_checks",
        "metrics",
    ] {
        require(
            original[key] == refined[key],
            &format!("changed old-only input/metric: {key}"),
        )?;
    }
    for key in ["cohort", "seed", "anchor"] {
        require(
            original[key] == ov[key] && refined[key] == rv[key],
            &format!("verification case changed: {key}"),
        )?;
    }
    require(
        ov["status"] == "pass" && rv["status"] == "pass",
        "one or both case verifications failed",
    )?;
    require(
        ov["next_step"] == rv["next_step"],
        "next-step accounting changed",
    )?;
    let of = original["forecasts"]
        .as_array()
        .ok_or("original horizons")?;
    let rf = refined["forecasts"].as_array().ok_or("refined horizons")?;
    require(of.len() == rf.len(), "horizon count changed")?;
    let mut endpoints = vec![];
    for (i, (o, r)) in of.iter().zip(rf).enumerate() {
        require(
            o["horizon"] == r["horizon"] && o["parameter_caps"] == r["parameter_caps"],
            "horizon/parameter caps changed",
        )?;
        let (op, rp) = (&ov["endpoints"][i], &rv["endpoints"][i]);
        require(
            op["horizon"] == o["horizon"] && rp["horizon"] == r["horizon"],
            "endpoint order changed",
        )?;
        require(
            op["actual_q"] == rp["actual_q"] && op["full_potentials"] == rp["full_potentials"],
            "executed outcomes changed",
        )?;
        let (fe, fs) = excess(&r["feature_radii"], &o["feature_radii"])?;
        let (pe, _) = excess(&r["probability_ceilings"], &o["probability_ceilings"])?;
        let passed = fs <= 1e-8 && pe <= 1e-8;
        endpoints.push(
            json!({"horizon":r["horizon"],"status":if passed{"pass"}else{"fail"},
            "feature_excess":fe,"feature_scaled_excess":fs,"probability_excess":pe,
            "original_q":o["q"],"refined_q":r["q"],"actual_q":rp["actual_q"],
            "original_coefficients":o["coefficients"],"refined_coefficients":r["coefficients"],
            "full_potentials":rp["full_potentials"]}),
        );
    }
    let passed = endpoints.iter().all(|e| e["status"] == "pass");
    Ok(
        json!({"cohort":refined["cohort"],"seed":refined["seed"],"anchor":refined["anchor"],
        "status":if passed{"pass"}else{"fail"},"old_q":refined["old_q"],"metrics":refined["metrics"],"endpoints":endpoints}),
    )
}
pub fn run(original: &Path, refined: &Path) -> Result<()> {
    let started = Instant::now();
    let mut ledger = Ledger::default();
    let oc = sealed(original, &mut ledger)?;
    let rc = sealed(refined, &mut ledger)?;
    require(
        rc["source_sha256"] == crate::provenance::source_hash(),
        "comparison source differs from refined constructor",
    )?;
    require(oc["roots"] == rc["roots"], "training inputs changed")?;
    let (ov, _) = verification(original, &mut ledger)?;
    let (rv, verified_seconds) = verification(refined, &mut ledger)?;
    require(
        ov["construction_sha256"] == evidence::hash(&fs::read(original.join("construction.json"))?)
            && rv["construction_sha256"]
                == evidence::hash(&fs::read(refined.join("construction.json"))?),
        "verification seals",
    )?;
    let mut charged = rc["seconds"].as_f64().ok_or("construction time")? + verified_seconds;
    let mut attempt = 1;
    while refined
        .join(format!("comparison-{attempt:02}.json"))
        .exists()
    {
        let previous = ledger.json(&refined.join(format!("comparison-{attempt:02}.json")))?;
        require(
            previous["status"] != "running",
            "comparison already running",
        )?;
        charged += previous["seconds"]
            .as_f64()
            .ok_or("previous comparison time")?;
        attempt += 1;
    }
    let path = refined.join(format!("comparison-{attempt:02}.json"));
    let mut report = json!({"status":"running","original":original,"refined":refined,"charged_seconds":charged,"cases":rc["cases"]});
    evidence::write(&path, &report)?;
    for (i, slot) in rc["cases"].as_array().unwrap().iter().enumerate() {
        let result = (|| -> Result<Value> {
            require(
                charged + started.elapsed().as_secs_f64() < 1800.,
                "comparison cap",
            )?;
            let o = payload(original, &oc["cases"][i], &mut ledger)?;
            let r = payload(refined, slot, &mut ledger)?;
            assess(&o, &r, &ov["cases"][i], &rv["cases"][i])
        })();
        report["cases"][i] = match result {
            Ok(value) => value,
            Err(e) => {
                json!({"cohort":slot["cohort"],"seed":slot["seed"],"anchor":slot["anchor"],"status":"error","error":e.to_string(),"horizons":slot["horizons"]})
            }
        };
        evidence::write(&path, &report)?;
    }
    if let Err(e) = ledger.verify() {
        report["error"] = json!(e.to_string());
    }
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] != "pass")
        .count();
    let seconds = started.elapsed().as_secs_f64();
    let pass = failed == 0 && report["error"].is_null() && charged + seconds <= 1800.;
    report["status"] = json!(if pass { "pass" } else { "fail" });
    report["failed"] = json!(failed);
    report["seconds"] = json!(seconds);
    report["total_seconds"] = json!(charged + seconds);
    report["inputs"] = json!(ledger.0);
    evidence::write(&path, &report)?;
    require(pass, "comparison failed; complete inventory retained")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixtures() -> (Value, Value) {
        let forecast = json!({"cohort":"fixture","seed":7,"anchor":1000,
            "forecasts":[{"horizon":1,"parameter_caps":[1.,2.],
            "feature_radii":vec![2.;4704],"probability_ceilings":vec![0.4;4704],
            "q":0.4,"coefficients":[1.,3.]}]});
        let verification = json!({"cohort":"fixture","seed":7,"anchor":1000,
            "status":"pass","next_step":{"value":17},
            "endpoints":[{"horizon":1,"actual_q":0.2,"full_potentials":[4.,5.]}]});
        (forecast, verification)
    }
    #[test]
    fn matched_comparison_rejects_changed_inputs_and_retains_bound_excess() {
        let (o, ov) = fixtures();
        let mut r = o.clone();
        r["forecasts"][0]["feature_radii"] = json!(vec![1.; 4704]);
        r["forecasts"][0]["probability_ceilings"] = json!(vec![0.3; 4704]);
        assert_eq!(assess(&o, &r, &ov, &ov).unwrap()["status"], "pass");
        let mut changed = r.clone();
        changed["forecasts"][0]["parameter_caps"][0] = json!(2.);
        assert!(assess(&o, &changed, &ov, &ov).is_err());
        changed = r.clone();
        changed["r"] = json!([1., 2.]);
        assert!(assess(&o, &changed, &ov, &ov).is_err());
        let mut rv = ov.clone();
        rv["anchor"] = json!(4000);
        assert!(assess(&o, &r, &ov, &rv).is_err());
        rv = ov.clone();
        rv["endpoints"][0]["full_potentials"][0] = json!(6.);
        assert!(assess(&o, &r, &ov, &rv).is_err());
        r["forecasts"][0]["probability_ceilings"][4703] = json!(0.5);
        let failed = assess(&o, &r, &ov, &ov).unwrap();
        assert_eq!(failed["status"], "fail");
        assert!(
            failed["endpoints"][0]["probability_excess"]
                .as_f64()
                .unwrap()
                > 0.09
        );
    }
}
