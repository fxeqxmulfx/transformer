//! Paired throughput workloads; canonical restore/export are diagnostic costs.
use crate::{
    application::Backend,
    domain::{
        Result,
        data::{Data, Plan},
        model::Arm,
    },
    infrastructure::storage,
    inputs::{self, Ledger, require},
    ownership, preflight, state,
    timer::{self, Events},
    trainer::Trainer,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
pub fn run(
    root: &Path,
    output: &Path,
    ledger: &mut Ledger,
    protocol: &str,
    cuda: &Cuda,
) -> Result<Value> {
    let started = Instant::now();
    let events = Events::new(cuda)?;
    let mut cases = vec![];
    let floor = (0..9)
        .map(|_| events.measure(|| Ok(())))
        .collect::<Result<Vec<_>>>()?;
    for seed in [7, 11] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 11000);
        for arm in Arm::ALL {
            for step in [0, 5000, 10000] {
                let dir = output.join(format!("bench-{}-seed{seed}-step{step}", arm.name()));
                fs::create_dir(&dir)?;
                let mut row = json!({"arm":arm.name(),"seed":seed,"step":step});
                row["gpu_before"] = ownership::snapshot(&dir, "before");
                let mut samples = vec![];
                let outcome = (|| -> Result<Value> {
                    require(
                        started.elapsed().as_secs_f64() < 600.,
                        "600-second paired benchmark cap",
                    )?;
                    let case = inputs::Case {
                        arm,
                        seed,
                        step,
                        data: &data,
                        plan: &plan,
                        protocol,
                        forecast_output: output,
                    };
                    let saved = inputs::checkpoint(ledger, root, &case)?;
                    let mut pair = [
                        Trainer::new(arm, &saved.state.parameters, &data, &plan, false, cuda)?,
                        Trainer::new(arm, &saved.state.parameters, &data, &plan, true, cuda)?,
                    ];
                    pair[0].restore(&saved.state)?;
                    let offset = plan.boundaries[step];
                    let gradient =
                        pair[0].prepare_gradient(offset, plan.boundaries[step + 1] - offset)?;
                    for round in 0..12 {
                        for which in 0..2 {
                            let update = (round + which) % 2 == 0;
                            let count = if update { 256 } else { 8 };
                            let mut hashes = [String::new(), String::new()];
                            for j in 0..2 {
                                let idx = (round + j) % 2;
                                let t = &mut pair[idx];
                                t.restore(&saved.state)?;
                                if update {
                                    t.set_gradient(&gradient)?;
                                }
                                let mut sample = events.measure(|| {
                                    for k in 1..=count {
                                        if update {
                                            t.update(step + k)?;
                                        } else {
                                            let offset = plan.boundaries[step + k - 1];
                                            t.train_step(
                                                offset,
                                                plan.boundaries[step + k] - offset,
                                                step + k,
                                            )?;
                                        }
                                    }
                                    Ok(())
                                })?;
                                require(state::valid(arm, &t.state()?), "invalid benchmark state")?;
                                hashes[idx] = preflight::hash(step + count, t)?;
                                sample["method"] = json!(t.name());
                                sample["round"] = json!(round);
                                sample["warmup"] = json!(round < 3);
                                sample["workload"] =
                                    json!(if update { "update" } else { "complete" });
                                sample["updates"] = json!(count);
                                sample["state_sha256"] = json!(hashes[idx]);
                                samples.push(sample);
                            }
                            require(hashes[0] == hashes[1], "paired benchmark state differs")?;
                        }
                    }
                    let mut summary = json!({});
                    for work in ["update", "complete"] {
                        for time in ["wall_seconds", "gpu_seconds"] {
                            for method in ["source-adamw", "compact-adamw"] {
                                summary[work][time][method] = timer::stats(
                                    samples
                                        .iter()
                                        .filter(|s| {
                                            s["warmup"] == false
                                                && s["workload"] == work
                                                && s["method"] == method
                                        })
                                        .map(|s| {
                                            s[time].as_f64().unwrap()
                                                / s["updates"].as_f64().unwrap()
                                        }),
                                );
                            }
                            let ratios = (3..12).map(|round| {
                                let get = |method| {
                                    samples
                                        .iter()
                                        .find(|s| {
                                            s["round"] == round
                                                && s["workload"] == work
                                                && s["method"] == method
                                        })
                                        .unwrap()[time]
                                        .as_f64()
                                        .unwrap()
                                };
                                get("source-adamw") / get("compact-adamw")
                            });
                            summary[work][time]["paired_speed_ratio"] = timer::stats(ratios);
                        }
                    }
                    Ok(
                        json!({"passed":true,"paired_state_checks":24,"summary":summary,
                    "memory":pair.iter().map(|t|json!({"method":t.name(),"allocated_payload_bytes":t.bytes(),"moment_bytes":t.moment_bytes()})).collect::<Vec<_>>()}),
                    )
                })();
                match outcome {
                    Ok(v) => {
                        row["passed"] = json!(true);
                        row["benchmark"] = v;
                    }
                    Err(e) => {
                        row["passed"] = json!(false);
                        row["error"] = json!(e.to_string());
                    }
                }
                row["samples"] = json!(samples);
                row["gpu_after"] = ownership::snapshot(&dir, "after");
                eprintln!(
                    "benchmark {} seed={seed} step={step} passed={}",
                    arm.name(),
                    row["passed"]
                );
                cases.push(row);
                storage::json_write(
                    &output.join("benchmark.json"),
                    &json!({"cases":cases,"seconds":started.elapsed().as_secs_f64()}),
                )?;
            }
        }
    }
    Ok(
        json!({"passed":cases.len()==24&&cases.iter().all(|r|r["passed"]==true),"cases":cases,"seconds":started.elapsed().as_secs_f64(),"empty_event_brackets":floor}),
    )
}
