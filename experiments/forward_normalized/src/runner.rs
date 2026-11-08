//! Adapted from forward_training/runner.rs at c9995ba5. Fresh seeds, three
//! fixed rules, no prior fresh-seed trajectory, and identical initial metrics.
use crate::{
    config::{CADENCE, Kernels, STEPS},
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::storage,
    inputs::require,
    normalized_kernel, ownership,
    rule::Rule,
    summary,
    worker::Worker,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};

pub fn run(
    output: &Path,
    cuda: &Cuda,
    kernels: &Kernels,
    normalized: &normalized_kernel::kernels::Kernels,
) -> Result<Value> {
    let cohort = Instant::now();
    let mut groups = vec![];
    for seed in [17, 23] {
        for arm in Arm::ALL {
            let dir = output.join(format!("{}-seed{seed}", arm.name()));
            fs::create_dir(&dir)?;
            let mut snapshots = vec![ownership::snapshot(&dir, "before")];
            let shared = Instant::now();
            let data = Data::new(seed);
            let plan = Plan::new(seed, STEPS);
            let shared_seconds = shared.elapsed().as_secs_f64();
            let reference = Plan::new(seed, 10000);
            require(
                plan.order == reference.order[..plan.order.len()]
                    && plan.boundaries == reference.boundaries[..=STEPS],
                "sampler prefix differs",
            )?;
            let identity = json!({"initialization":"original model::initialize; zero source moments",
            "initial_parameters_sha256":storage::hash(&storage::floats(&model::initialize(arm,seed))),
            "train_sha256":storage::hash(&storage::words(&data.train)),
            "validation_sha256":storage::hash(&storage::words(&data.validation)),
            "sampler_prefix_sha256":storage::hash(&storage::words(&plan.order)),
            "shared_data_sampler_seconds_charged_per_method":shared_seconds});
            let mut workers = vec![];
            let mut reports = vec![];
            for method in Rule::ALL {
                if cohort.elapsed().as_secs_f64() >= 900. {
                    reports.push(json!({"method":method.name(),"status":"cohort-capped","completed_steps":0}));
                    continue;
                }
                match Worker::new(arm,seed,method,&data,&plan,shared_seconds,cuda) {
                Ok(mut worker)=>{
                    if let Err(e)=worker.evaluate(cuda) {worker.fail(e);}
                    workers.push(worker);
                }
                Err(e)=>reports.push(json!({"method":method.name(),"status":"error","error":e.to_string(),"completed_steps":0})),
            }
            }
            let initial = workers.first().and_then(|w| w.history.first());
            let initial_match = initial.is_some()
                && workers.len() == 3
                && workers.iter().all(|w| {
                    w.history.first().is_some_and(|r| {
                        initial.is_some_and(|i| {
                            r["train"] == i["train"] && r["validation"] == i["validation"]
                        })
                    })
                });
            if !initial_match {
                for worker in &mut workers {
                    worker.fail("initial metrics differ or initial state unavailable");
                }
            }
            for chunk in 0..STEPS / CADENCE {
                let end = (chunk + 1) * CADENCE;
                for j in 0..workers.len() {
                    let count = workers.len();
                    let worker = &mut workers[(chunk + j) % count];
                    if worker.status != "running" {
                        continue;
                    }
                    if worker.active() >= 120. {
                        worker.status = "method-capped".into();
                        continue;
                    }
                    if cohort.elapsed().as_secs_f64() >= 900. {
                        worker.status = "cohort-capped".into();
                        continue;
                    }
                    let result = worker
                        .chunk(end, &plan, cuda, kernels, normalized)
                        .and_then(|()| worker.evaluate(cuda));
                    if let Err(e) = result {
                        worker.fail(e);
                    }
                    if worker.status == "running" && worker.active() >= 120. {
                        worker.status = "method-capped".into();
                    }
                    if worker.status == "running" && worker.step == STEPS {
                        worker.status = "complete".into();
                    }
                }
                if end.is_multiple_of(400) {
                    snapshots.push(ownership::snapshot(&dir, &format!("step-{end:06}")));
                    for worker in &workers {
                        let last = worker.history.last().unwrap_or(&Value::Null).clone();
                        eprintln!(
                            "{} seed={} {} step={} validation={} CE={} active={:.3}s {}",
                            arm.name(),
                            seed,
                            worker.method.name(),
                            worker.step,
                            last["validation"]["accuracy"],
                            last["validation"]["loss"],
                            worker.active(),
                            worker.status
                        );
                    }
                }
                storage::json_write(
                    &dir.join("progress.json"),
                    &json!({"arm":arm.name(),"seed":seed,
                "workers":workers.iter().map(Worker::report).collect::<Vec<_>>(),"failures":reports}),
                )?;
                if workers.iter().all(|w| w.status != "running") {
                    break;
                }
            }
            snapshots.push(ownership::snapshot(&dir, "after"));
            for worker in &workers {
                let mut report = worker.report();
                match worker.save(&dir) {
                    Ok(checkpoint) => report["checkpoint"] = checkpoint,
                    Err(e) => {
                        report["checkpoint"] = json!({"error":e.to_string(),"valid":false});
                        report["status"] = json!("error");
                    }
                }
                reports.push(report);
            }
            summary::summarize(&mut reports);
            let clean = snapshots.iter().all(|s| s["clean_ownership"] == true);
            let group = json!({"arm":arm.name(),"seed":seed,"identity":identity,"clean_device_timing":clean,
            "initial_metrics_match":initial_match,
            "gpu_snapshots":snapshots,"runs":reports});
            storage::json_write(&dir.join("result.json"), &group)?;
            groups.push(group);
            storage::json_write(
                &output.join("training.json"),
                &json!({"groups":groups,"cohort_wall_seconds":cohort.elapsed().as_secs_f64()}),
            )?;
        }
    }
    let completed = groups
        .iter()
        .flat_map(|g| g["runs"].as_array().unwrap())
        .filter(|r| r["status"] == "complete")
        .count();
    let valid = completed == 24
        && groups.iter().all(|g| {
            g["initial_metrics_match"] == true
                && g["runs"].as_array().unwrap().iter().all(|r| {
                    r["checkpoint"]["valid"] == true
                        && r["history"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|r| r["state_valid"] == true)
                })
        });
    Ok(
        json!({"groups":groups,"cohort_wall_seconds":cohort.elapsed().as_secs_f64(),
        "completed_runs":completed,"valid_completed_cohort":valid}),
    )
}
