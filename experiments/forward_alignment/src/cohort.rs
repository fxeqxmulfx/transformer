//! Immutable inputs and exhaustive state traversal. Original and local pilot
//! artifacts are checked again, including all endpoint metrics and moments.
use crate::{
    config::{self, Kernels, Method, Prototypes},
    domain::{
        Result,
        data::{Data, Plan},
        model::Arm,
    },
    infrastructure::storage,
    inputs::{self, Ledger, require},
    measure::Cost,
    pins, probe,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

pub struct Context<'a> {
    pub root: &'a Path,
    pub pilot: &'a Path,
    pub output: &'a Path,
    pub ledger: &'a mut Ledger,
    pub protocol: &'a str,
    pub cuda: &'a Cuda,
    pub kernels: &'a Kernels,
}
pub fn run(context: &mut Context<'_>) -> Result<Value> {
    let started = Instant::now();
    let mut cost = Cost::default();
    let mut cases = vec![];
    let pilot = context.ledger.json(&context.pilot.join("report.json"))?;
    require(
        pilot["status"] == "complete"
            && pilot["source_sha256"] == pins::PILOT_SOURCE
            && pilot["input_verification"]["unchanged"] == true
            && pilot["training"]["valid_completed_cohort"] == true,
        "invalid pinned pilot report",
    )?;
    require(
        pilot["protocol"] == include_str!("../../forward_training/PROTOCOL.md"),
        "pilot protocol changed",
    )?;
    for seed in [7, 11] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 10001);
        let prefix = Plan::new(seed, 4000);
        for arm in Arm::ALL {
            let group = pilot["training"]["groups"]
                .as_array()
                .ok_or("missing pilot groups")?
                .iter()
                .find(|g| g["arm"] == arm.name() && g["seed"] == seed)
                .ok_or("missing arm/seed")?;
            for (field, hash) in [
                ("train_sha256", storage::hash(&storage::words(&data.train))),
                (
                    "validation_sha256",
                    storage::hash(&storage::words(&data.validation)),
                ),
                (
                    "sampler_prefix_sha256",
                    storage::hash(&storage::words(&prefix.order)),
                ),
            ] {
                require(
                    group["identity"][field] == hash,
                    "pilot data/sampler differs",
                )?;
            }
            let runs = group["runs"].as_array().ok_or("missing pilot runs")?;
            let local = runs
                .iter()
                .find(|r| r["method"] == "local-pair")
                .ok_or("missing local pilot")?;
            let prototypes = Prototypes::new(arm, seed);
            require(
                local["prototypes"]["vocabulary_sha256"]
                    == storage::hash(&storage::floats(&prototypes.vocabulary))
                    && local["prototypes"]["hidden_sha256"]
                        == storage::hash(&storage::floats(&prototypes.hidden)),
                "prototype changed",
            )?;
            for origin in std::iter::once("initial").chain(Method::ALL.map(Method::name)) {
                let start = Instant::now();
                let mut row = json!({"arm":arm.name(),"seed":seed,"origin":origin,"step":if origin=="initial" {0} else {4000}});
                let outcome =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
                        require(
                            started.elapsed().as_secs_f64() < 600.,
                            "600-second diagnostic cohort cap reached",
                        )?;
                        let (step, state, expected, hash) = if origin == "initial" {
                            let input = inputs::Case {
                                arm,
                                seed,
                                step: 0,
                                data: &data,
                                plan: &plan,
                                protocol: context.protocol,
                                forecast_output: context.output,
                            };
                            let saved = inputs::checkpoint(context.ledger, context.root, &input)?;
                            let hash = storage::hash(&storage::state_bytes(0, &saved.state));
                            (0, saved.state, saved.last_report, hash)
                        } else {
                            let source = runs
                                .iter()
                                .find(|r| r["method"] == origin)
                                .ok_or("missing endpoint rule")?;
                            require(
                                source["status"] == "complete" && source["completed_steps"] == 4000,
                                "incomplete endpoint",
                            )?;
                            let bytes = context.ledger.read(
                                &context
                                    .pilot
                                    .join(format!("{}-seed{seed}/{origin}.bin", arm.name())),
                            )?;
                            let hash = storage::hash(&bytes);
                            require(
                                source["checkpoint"]["sha256"] == hash,
                                "endpoint hash differs",
                            )?;
                            let (step, state) = storage::decode(&bytes, arm.count())?;
                            require(step == 4000, "wrong endpoint step")?;
                            let last = source["history"]
                                .as_array()
                                .and_then(|v| v.last())
                                .ok_or("missing endpoint report")?
                                .clone();
                            (step, state, last, hash)
                        };
                        require(
                            config::valid_state(arm, &state),
                            "invalid saved state/moment ties",
                        )?;
                        let case = probe::Case {
                            arm,
                            seed,
                            step,
                            data: &data,
                            plan: &plan,
                            state: &state,
                            expected: &expected,
                        };
                        let mut value =
                            probe::run(&case, context.cuda, context.kernels, &mut cost)?;
                        value["input_state_sha256"] = json!(hash);
                        Ok(value)
                    }));
                match outcome {
                    Ok(Ok(value)) => {
                        row["passed"] = value["passed"].clone();
                        row["diagnostic"] = value;
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
                row["seconds"] = json!(start.elapsed().as_secs_f64());
                eprintln!(
                    "{} seed={} origin={} passed={} {:.2}s",
                    arm.name(),
                    seed,
                    origin,
                    row["passed"],
                    start.elapsed().as_secs_f64()
                );
                cases.push(row);
                storage::json_write(
                    &context.output.join("progress.json"),
                    &json!({"cases":cases,"cost":cost.value(),"seconds":started.elapsed().as_secs_f64()}),
                )?;
            }
        }
    }
    Ok(
        json!({"passed":cases.len()==40&&cases.iter().all(|c|c["passed"]==true),"cases":cases,
        "cost":cost.value(),"seconds":started.elapsed().as_secs_f64()}),
    )
}
