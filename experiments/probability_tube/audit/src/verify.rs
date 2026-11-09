//! Reconstruct sealed forecasts through the old-only interface before future-state checks.
use crate::{
    Result,
    application::Backend,
    batch, checkpoints,
    classes::{Partition, energy},
    construct,
    evidence::{self, Ledger, require},
    gpu, guarded,
    infrastructure::storage,
    metric_analysis, prediction, provenance,
};
use probability_tube::encoder::Shape;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
fn array(v: &Value) -> Result<Vec<f64>> {
    Ok(serde_json::from_value(v.clone())?)
}
fn endpoint(
    old: &crate::application::State,
    next: &crate::application::State,
    old_forward: (&batch::Trace, &gpu::Snapshot),
    new_gpu: &gpu::Snapshot,
    f: &Value,
    forecast: &Value,
    rows: &[u32],
) -> Result<Value> {
    let (old_trace, old_gpu) = old_forward;
    let p: Vec<_> = next.parameters.iter().map(|v| *v as f64).collect();
    let trace = batch::forward(&p, rows);
    let execution = prediction::execution(&p, &trace, new_gpu)?;
    let active: Vec<bool> = serde_json::from_value(f["active"].clone())?;
    let caps = array(&forecast["parameter_caps"])?;
    require(caps.len() == 100, "block inventory")?;
    let mut parameter_excess = f64::NEG_INFINITY;
    let mut parameter_scaled = f64::NEG_INFINITY;
    for (i, block) in guarded::blocks(Shape::ORIGINAL).into_iter().enumerate() {
        let actual = block
            .map(|k| (next.parameters[k] as f64 - old.parameters[k] as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        parameter_excess = parameter_excess.max(actual - caps[i]);
        parameter_scaled = parameter_scaled.max((actual - caps[i]) / (1. + actual.max(caps[i])));
    }
    let features = array(&forecast["feature_radii"])?;
    let budgets = array(&forecast["contrast_budgets"])?;
    let probabilities = array(&forecast["probability_ceilings"])?;
    require(
        [features.len(), budgets.len(), probabilities.len()] == [4704; 3],
        "query inventory",
    )?;
    let q = prediction::probabilities(&new_gpu.logits, &active);
    let mut feature_excess = f64::NEG_INFINITY;
    let mut feature_scaled = f64::NEG_INFINITY;
    let mut contrast_excess = f64::NEG_INFINITY;
    let mut contrast_scaled = f64::NEG_INFINITY;
    let mut probability_excess = f64::NEG_INFINITY;
    for i in 0..4704 {
        let motion = trace.features[i * 128..(i + 1) * 128]
            .iter()
            .zip(&old_trace.features[i * 128..(i + 1) * 128])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        feature_excess = feature_excess.max(motion - features[i]);
        feature_scaled =
            feature_scaled.max((motion - features[i]) / (1. + motion.max(features[i])));
        let delta: Vec<_> = new_gpu.logits[i * 97..(i + 1) * 97]
            .iter()
            .zip(&old_gpu.logits[i * 97..(i + 1) * 97])
            .map(|(a, b)| a - b)
            .collect();
        let increase = delta
            .iter()
            .enumerate()
            .filter(|(c, _)| !active[*c])
            .map(|(_, d)| *d)
            .fold(f64::NEG_INFINITY, f64::max)
            - delta.iter().copied().fold(f64::INFINITY, f64::min);
        contrast_excess = contrast_excess.max(increase - budgets[i]);
        contrast_scaled =
            contrast_scaled.max((increase - budgets[i]) / (1. + increase.abs().max(budgets[i])));
        probability_excess = probability_excess.max(q[i] - probabilities[i]);
    }
    let part = Partition::new(active, 128);
    let next_readout = prediction::readout(next);
    let w = part.project(&next_readout.p)[1].clone();
    let m = part.project(&next_readout.m)[1].clone();
    let mut potentials = vec![];
    let s = f["s"].as_f64().ok_or("fixed step")?;
    for (i, r) in [vec![1.; 128], array(&f["r"])?].into_iter().enumerate() {
        let before = f["metrics"][i]["before"].as_f64().ok_or("old potential")?;
        let after = metric_analysis::potential(&w, &m, &r, s);
        let rmax = f["metrics"][i]["rmax"].as_f64().ok_or("weight cap")?;
        require(
            energy(&w) <= 1.9 * rmax * after + 1e-8 * (1. + energy(&w)),
            "ordinary energy conversion",
        )?;
        potentials.push(
            json!({"metric":f["metrics"][i]["metric"],"before":before,"after":after,
            "change":after-before,"increase":after-before>1e-8*(1.+before.abs().max(after.abs())),
            "relative_change":(after-before)/before,"ordinary_energy_bound":1.9*rmax*after}),
        );
    }
    let qmax = q.iter().copied().fold(0., f64::max);
    let passed = parameter_scaled <= 1e-8
        && feature_scaled <= 1e-8
        && contrast_scaled <= 1e-8
        && probability_excess <= 1e-8;
    Ok(
        json!({"horizon":forecast["horizon"],"status":if passed{"pass"}else{"fail"},"execution":execution,
        "parameter_excess":parameter_excess,"parameter_scaled_excess":parameter_scaled,
        "feature_excess":feature_excess,"feature_scaled_excess":feature_scaled,
        "contrast_excess":contrast_excess,"contrast_scaled_excess":contrast_scaled,
        "probability_excess":probability_excess,"actual_q":qmax,"forecast_q":forecast["q"],
        "old_q_excess_invalid_forecast":qmax-f["old_q"].as_f64().unwrap(),
        "coefficients":forecast["coefficients"],"full_potentials":potentials}),
    )
}
fn slot(
    out: &Path,
    root: &Path,
    spec: &Value,
    source: &Value,
    started: Instant,
    charged: f64,
    ledger: &mut Ledger,
) -> Result<Value> {
    let seed = spec["seed"].as_u64().ok_or("seed")?;
    let anchor = spec["anchor"].as_u64().ok_or("anchor")? as usize;
    let cohort = spec["cohort"].as_str().ok_or("cohort")?;
    require(spec["status"] == "frozen", "forecast unavailable")?;
    let file = out.join(spec["forecast"]["file"].as_str().ok_or("forecast file")?);
    let bytes = ledger.read(&file)?;
    require(
        evidence::hash(&bytes) == spec["forecast"]["sha256"],
        "forecast hash",
    )?;
    let f: Value = serde_json::from_slice(&bytes)?;
    let (old, data, plan) = construct::old(root, seed, anchor, ledger)?;
    let rebuilt = prediction::make(cohort, seed, anchor, &old, &data, &plan)?;
    require(f == rebuilt, "old-only forecast reconstruction")?;
    // Everything above uses old arrays only. Later states are opened below.
    let mut e = gpu::engine(&old, &data, &plan)?;
    let old_gpu = gpu::snapshot(&e, &data)?;
    let old_trace = batch::forward(
        &old.parameters.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &data.train,
    );
    let mut endpoints = vec![];
    let mut next_step = Value::Null;
    let seed_source = source["seeds"]
        .as_array()
        .ok_or("source seeds")?
        .iter()
        .find(|s| s["seed"] == seed)
        .ok_or("source seed")?;
    for forecast in f["forecasts"].as_array().ok_or("forecast horizons")? {
        let h = forecast["horizon"].as_u64().ok_or("horizon")? as usize;
        let result = (|| -> Result<Value> {
            require(
                charged + started.elapsed().as_secs_f64() < 1800.,
                "verification cap",
            )?;
            let path = root.join(format!("seed{seed}/checkpoints/step-{:06}.bin", anchor + h));
            let bytes = ledger.read(&path)?;
            let saved = seed_source["checkpoints"]
                .as_array()
                .ok_or("source checkpoints")?
                .iter()
                .find(|s| s["step"] == anchor + h)
                .ok_or("source endpoint")?;
            require(
                saved["status"] == "pass" && evidence::hash(&bytes) == saved["sha256"],
                "source endpoint hash",
            )?;
            let (step, next) = storage::decode(&bytes, crate::domain::model::Arm::Plain.count())?;
            require(step == anchor + h, "future checkpoint time")?;
            e.restore(&next)?;
            let actual = gpu::snapshot(&e, &data)?;
            let result = endpoint(
                &old,
                &next,
                (&old_trace, &old_gpu),
                &actual,
                &f,
                forecast,
                &data.train,
            )?;
            if h == 1 {
                e.restore(&old)?;
                let batch = gpu::minibatch(&e, &data, &plan, anchor)?;
                next_step = metric_analysis::assess(
                    &prediction::readout(&old),
                    &prediction::readout(&next),
                    &f,
                    &batch,
                )?;
            }
            Ok(result)
        })();
        endpoints.push(match result {
            Ok(value) => value,
            Err(e) => json!({"horizon":h,"status":"error","error":e.to_string()}),
        });
    }
    let passed = endpoints.iter().all(|e| e["status"] == "pass") && !next_step.is_null();
    Ok(
        json!({"cohort":cohort,"seed":seed,"anchor":anchor,"status":if passed{"pass"}else{"fail"},
        "old_q":f["old_q"],"metrics":f["metrics"],"endpoints":endpoints,"next_step":next_step}),
    )
}
pub fn run(out: &Path) -> Result<()> {
    let started = Instant::now();
    crate::algebra::threads();
    let mut ledger = Ledger::default();
    let construction = ledger.read(&out.join("construction.json"))?;
    require(
        evidence::hash(&construction) == fs::read_to_string(out.join("construction.sha256"))?,
        "construction seal",
    )?;
    let frozen: Value = serde_json::from_slice(&construction)?;
    require(
        frozen["source_sha256"] == provenance::source_hash(),
        "constructor/verifier source mismatch",
    )?;
    let mut charged = frozen["seconds"].as_f64().ok_or("construction time")?;
    let mut attempt = 1;
    while out.join(format!("verification-{attempt:02}.json")).exists() {
        let old = ledger.json(&out.join(format!("verification-{attempt:02}.json")))?;
        require(old["status"] != "running", "verification already running")?;
        charged += old["seconds"]
            .as_f64()
            .ok_or("previous verification time")?;
        attempt += 1;
    }
    let path = out.join(format!("verification-{attempt:02}.json"));
    let mut report = json!({"status":"running","construction_sha256":evidence::hash(&construction),"charged_seconds":charged,"cases":frozen["cases"]});
    evidence::write(&path, &report)?;
    let run = (|| -> Result<()> {
        require(charged < 1800., "audit budget exhausted")?;
        require(
            frozen["cases"].as_array().is_some_and(|a| a.len() == 144),
            "complete forecast inventory",
        )?;
        report["provenance"] = provenance::metadata()?;
        let roots: Vec<PathBuf> = serde_json::from_value(frozen["roots"].clone())?;
        require(roots.len() == 3, "source cohort inventory")?;
        let sources: Vec<_> = roots
            .iter()
            .map(|root| ledger.json(&root.join("report.json")))
            .collect::<Result<_>>()?;
        for (i, s) in sources.iter().enumerate() {
            require(
                s["status"] == "pass"
                    && s["cohort"] == construct::COHORTS[i]
                    && s["total_seconds"].as_f64().is_some_and(|s| s <= 7200.),
                "source cohort incomplete or capped",
            )?;
            require(
                s["seeds"].as_array().is_some_and(|ss| {
                    ss.len() == 8
                        && ss.iter().zip(checkpoints::SEEDS).all(|(s, seed)| {
                            s["seed"] == seed && s["status"] == "pass" && s["completed"] == 10000
                        })
                }),
                "source seed inventory",
            )?;
        }
        for (i, spec) in frozen["cases"].as_array().unwrap().iter().enumerate() {
            let value = slot(
                out,
                &roots[i / 48],
                spec,
                &sources[i / 48],
                started,
                charged,
                &mut ledger,
            );
            report["cases"][i] = match value {
                Ok(v) => v,
                Err(e) => {
                    json!({"cohort":spec["cohort"],"seed":spec["seed"],"anchor":spec["anchor"],"status":"error","error":e.to_string(),"horizons":spec["horizons"]})
                }
            };
            evidence::write(&path, &report)?;
            eprintln!(
                "verified slot={i} status={} cumulative={:.1}s",
                report["cases"][i]["status"],
                charged + started.elapsed().as_secs_f64()
            );
        }
        ledger.verify()?;
        Ok(())
    })();
    if let Err(e) = run {
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
    require(pass, "verification failed; complete inventory retained")
}
