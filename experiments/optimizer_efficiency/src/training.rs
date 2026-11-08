//! Fresh full-backpropagation pairs with all 41 canonical observations equal.
use crate::{
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::storage,
    inputs::require,
    ownership, quality,
    worker::Worker,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
fn matching(workers: &[Worker], step: usize) -> bool {
    workers.len() == 2
        && workers
            .iter()
            .all(|w| w.history.last().is_some_and(|r| r["step"] == step))
        && ["state_sha256", "train", "validation"].iter().all(|key| {
            workers[0].history.last().unwrap()[key] == workers[1].history.last().unwrap()[key]
        })
}
pub fn run(output: &Path, cuda: &Cuda) -> Result<Value> {
    let started = Instant::now();
    let mut groups = vec![];
    for seed in [29, 31] {
        for arm in Arm::ALL {
            let dir = output.join(format!("train-{}-seed{seed}", arm.name()));
            fs::create_dir(&dir)?;
            let mut snapshots = vec![ownership::snapshot(&dir, "before")];
            let shared = Instant::now();
            let data = Data::new(seed);
            let plan = Plan::new(seed, 4000);
            let shared_seconds = shared.elapsed().as_secs_f64();
            let reference = Plan::new(seed, 10000);
            require(
                plan.order == reference.order[..plan.order.len()]
                    && plan.boundaries == reference.boundaries[..=4000],
                "sampler prefix differs",
            )?;
            let identity = json!({"initial_parameters_sha256":storage::hash(&storage::floats(&model::initialize(arm,seed))),"train_sha256":storage::hash(&storage::words(&data.train)),"validation_sha256":storage::hash(&storage::words(&data.validation)),"sampler_prefix_sha256":storage::hash(&storage::words(&plan.order)),"shared_seconds_charged_per_method":shared_seconds});
            let mut workers = vec![];
            let mut reports = vec![];
            let mut matches = vec![];
            for compact in [false, true] {
                let name = if compact {
                    "compact-adamw"
                } else {
                    "source-adamw"
                };
                if started.elapsed().as_secs_f64() >= 900. {
                    reports
                        .push(json!({"method":name,"status":"cohort-capped","completed_steps":0}));
                    continue;
                }
                match Worker::new(arm,seed,compact,&data,&plan,shared_seconds,cuda){
                Ok(mut w)=>{if let Err(e)=w.evaluate(cuda){w.fail(e);}workers.push(w);}
                Err(e)=>reports.push(json!({"method":name,"status":"error","error":e.to_string(),"completed_steps":0})),
            }
            }
            let initial = matching(&workers, 0);
            matches.push(json!({"step":0,"passed":initial}));
            if !initial {
                for w in &mut workers {
                    if w.status == "running" {
                        w.fail("initial pair differs or unavailable");
                    }
                }
            }
            for chunk in 0..40 {
                let end = (chunk + 1) * 100;
                let count = workers.len();
                for j in 0..count {
                    let w = &mut workers[(chunk + j) % count];
                    if w.status != "running" {
                        continue;
                    }
                    if w.active() >= 120. {
                        w.status = "method-capped".into();
                        continue;
                    }
                    if started.elapsed().as_secs_f64() >= 900. {
                        w.status = "cohort-capped".into();
                        continue;
                    }
                    if let Err(e) = w.chunk(end, &plan, cuda).and_then(|()| w.evaluate(cuda)) {
                        w.fail(e);
                    }
                    if w.status == "running" && w.active() >= 120. {
                        w.status = "method-capped".into();
                    }
                    if w.status == "running" && w.step == 4000 {
                        w.status = "complete".into();
                    }
                }
                let matched = matching(&workers, end);
                matches.push(json!({"step":end,"passed":matched}));
                if !matched {
                    for w in &mut workers {
                        if w.status == "running" || w.status == "complete" {
                            w.fail("paired canonical states or metrics differ or unavailable");
                        }
                    }
                }
                if end % 400 == 0 {
                    snapshots.push(ownership::snapshot(&dir, &format!("step-{end:06}")));
                    for w in &workers {
                        let row = w.history.last().unwrap_or(&Value::Null);
                        eprintln!(
                            "{} seed={seed} {} step={} accuracy={} CE={} active={:.3}s {}",
                            arm.name(),
                            w.trainer.name(),
                            w.step,
                            row["validation"]["accuracy"],
                            row["validation"]["loss"],
                            w.active(),
                            w.status
                        );
                    }
                }
                storage::json_write(
                    &dir.join("progress.json"),
                    &json!({"workers":workers.iter().map(Worker::report).collect::<Vec<_>>(),"failures":reports,"matches":matches}),
                )?;
                if workers.iter().all(|w| w.status != "running") {
                    break;
                }
            }
            snapshots.push(ownership::snapshot(&dir, "after"));
            for w in &workers {
                let mut report = w.report();
                match w.save(&dir) {
                    Ok(v) => report["checkpoint"] = v,
                    Err(e) => {
                        report["status"] = json!("error");
                        report["checkpoint"] = json!({"valid":false,"error":e.to_string()});
                    }
                }
                reports.push(report);
            }
            quality::summarize(&mut reports);
            let group = json!({"arm":arm.name(),"seed":seed,"identity":identity,"matches":matches,"clean_device_timing":snapshots.iter().all(|s|s["clean_ownership"]==true),"gpu_snapshots":snapshots,"runs":reports});
            storage::json_write(&dir.join("result.json"), &group)?;
            groups.push(group);
            storage::json_write(
                &output.join("training.json"),
                &json!({"groups":groups,"seconds":started.elapsed().as_secs_f64()}),
            )?;
        }
    }
    let completed = groups
        .iter()
        .flat_map(|g| g["runs"].as_array().unwrap())
        .filter(|r| r["status"] == "complete")
        .count();
    let valid = completed == 16
        && groups.iter().all(|g| {
            g["matches"]
                .as_array()
                .is_some_and(|m| m.len() == 41 && m.iter().all(|m| m["passed"] == true))
                && g["runs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|r| r["checkpoint"]["valid"] == true)
        });
    Ok(
        json!({"groups":groups,"completed_runs":completed,"valid_completed_cohort":valid,"seconds":started.elapsed().as_secs_f64()}),
    )
}
