//! Fixed source-only workloads from PROFILE_PROTOCOL.md; each round restores
//! its original state. Fixed-gradient updates are throughput probes only.
use crate::{
    application::Backend,
    domain::{
        Result,
        data::{Data, Plan},
        model::Arm,
    },
    infrastructure::{engine::Engine, storage},
    inputs::{self, Ledger, require},
    ownership, state,
    timer::{self, Events},
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
    let events = Events::new(cuda)?;
    let floor = (0..9)
        .map(|_| events.measure(|| Ok(())))
        .collect::<Result<Vec<_>>>()?;
    let started = Instant::now();
    let mut cases = vec![];
    for seed in [7, 11] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 11000);
        for arm in Arm::ALL {
            for step in [0, 5000, 10000] {
                let dir = output.join(format!("{}-seed{seed}-step{step}", arm.name()));
                fs::create_dir(&dir)?;
                let mut row = json!({"arm":arm.name(),"seed":seed,"step":step});
                row["gpu_before"] = ownership::snapshot(&dir, "before");
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<Value> {
                        require(
                            started.elapsed().as_secs_f64() < 600.,
                            "600-second profile cap",
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
                        require(state::valid(arm, &saved.state), "invalid saved state")?;
                        let mut engine = Engine::new(arm, &saved.state.parameters, &data, &plan)?;
                        engine.restore(&saved.state)?;
                        let offset = plan.boundaries[step];
                        let batch = plan.boundaries[step + 1] - offset;
                        engine.forward(false, offset, batch, true)?;
                        engine.backward(batch)?;
                        let gradient = engine.gradients.read()?;
                        let mut samples = vec![];
                        for round in 0..12 {
                            for which in 0..2 {
                                let update = (round + which) % 2 == 0;
                                let count = if update { 256 } else { 8 };
                                engine.restore(&saved.state)?;
                                if update {
                                    engine.gradients.write(&gradient)?;
                                }
                                let mut sample = events.measure(|| {
                                    for j in 1..=count {
                                        if update {
                                            engine.update(step + j)?;
                                        } else {
                                            let offset = plan.boundaries[step + j - 1];
                                            engine.train_step(
                                                offset,
                                                plan.boundaries[step + j] - offset,
                                                step + j,
                                            )?;
                                        }
                                    }
                                    Ok(())
                                })?;
                                let final_state = engine.state()?;
                                require(
                                    state::valid(arm, &final_state),
                                    "invalid final profile state",
                                )?;
                                sample["round"] = json!(round);
                                sample["warmup"] = json!(round < 3);
                                sample["workload"] =
                                    json!(if update { "update" } else { "complete" });
                                sample["updates"] = json!(count);
                                sample["state_sha256"] = json!(storage::hash(
                                    &storage::state_bytes(step + count, &final_state)
                                ));
                                samples.push(sample);
                            }
                        }
                        let mut summary = json!({});
                        for work in ["update", "complete"] {
                            for time in ["wall_seconds", "gpu_seconds"] {
                                summary[work][time] = timer::stats(
                                    samples
                                        .iter()
                                        .filter(|s| s["warmup"] == false && s["workload"] == work)
                                        .map(|s| {
                                            s[time].as_f64().unwrap()
                                                / s["updates"].as_f64().unwrap()
                                        }),
                                );
                            }
                            let hashes: Vec<_> = samples
                                .iter()
                                .filter(|s| s["workload"] == work)
                                .map(|s| &s["state_sha256"])
                                .collect();
                            require(
                                hashes.iter().all(|h| *h == hashes[0]),
                                "restored workload is not deterministic",
                            )?;
                        }
                        for time in ["wall_seconds", "gpu_seconds"] {
                            summary["update_over_complete"][time] = json!(
                                summary["update"][time]["median"].as_f64().unwrap()
                                    / summary["complete"][time]["median"].as_f64().unwrap()
                            );
                        }
                        Ok(
                            json!({"passed":true,"samples":samples,"summary":summary,"allocated_payload_bytes":engine.allocated_bytes(),"moment_bytes":engine.moment_bytes()}),
                        )
                    },
                ));
                match outcome {
                    Ok(Ok(value)) => {
                        row["passed"] = json!(true);
                        row["profile"] = value;
                    }
                    Ok(Err(e)) => {
                        row["passed"] = json!(false);
                        row["error"] = json!(e.to_string());
                    }
                    Err(_) => {
                        row["passed"] = json!(false);
                        row["error"] = json!("panic; see log");
                    }
                }
                row["gpu_after"] = ownership::snapshot(&dir, "after");
                eprintln!(
                    "profile {} seed={seed} step={step} passed={}",
                    arm.name(),
                    row["passed"]
                );
                cases.push(row);
                storage::json_write(
                    &output.join("progress.json"),
                    &json!({"cases":cases,"seconds":started.elapsed().as_secs_f64()}),
                )?;
            }
        }
    }
    Ok(
        json!({"passed":cases.len()==24&&cases.iter().all(|r|r["passed"]==true),"cases":cases,"seconds":started.elapsed().as_secs_f64(),"empty_event_brackets":floor}),
    )
}
