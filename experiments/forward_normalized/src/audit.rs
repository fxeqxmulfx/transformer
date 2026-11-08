//! New update checks precede all fresh-seed training. Saved states are used
//! only to test moment preservation and never initialize candidate runs.
use crate::{
    application::State,
    config::Kernels,
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::storage,
    inputs::{self, Ledger, require},
    normalized_kernel,
    update_audit::{self, Case},
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::path::Path;
pub fn run(
    root: &Path,
    output: &Path,
    ledger: &mut Ledger,
    protocol: &str,
    cuda: &Cuda,
    kernels: &Kernels,
    normalized: &normalized_kernel::kernels::Kernels,
) -> Result<Value> {
    let edges = update_audit::edges(cuda, normalized)?;
    let mut records = vec![];
    for seed in [17, 23, 7] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 10001);
        for arm in Arm::ALL {
            let step = if seed == 7 { 5000 } else { 0 };
            let mut row = json!({"arm":arm.name(),"seed":seed,"step":step});
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
                    let state = if seed == 7 {
                        let case = inputs::Case {
                            arm,
                            seed,
                            step,
                            data: &data,
                            plan: &plan,
                            protocol,
                            forecast_output: output,
                        };
                        inputs::checkpoint(ledger, root, &case)?.state
                    } else {
                        State {
                            parameters: model::initialize(arm, seed),
                            first: vec![0.; arm.count()],
                            second: vec![0.; arm.count()],
                        }
                    };
                    update_audit::actual(
                        &Case {
                            arm,
                            seed,
                            step,
                            state: &state,
                            data: &data,
                            plan: &plan,
                        },
                        cuda,
                        kernels,
                        normalized,
                    )
                }));
            match outcome {
                Ok(Ok(value)) => {
                    row["passed"] = value["passed"].clone();
                    row["audit"] = value;
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
            eprintln!(
                "normalized preflight {} seed={seed} step={step} passed={}",
                arm.name(),
                row["passed"]
            );
            records.push(row);
        }
    }
    let passed = edges["passed"] == true && records.iter().all(|r| r["passed"] == true);
    let report = json!({"passed":passed,"edges":edges,"cases":records});
    storage::json_write(&output.join("preflight.json"), &report)?;
    require(passed, "normalized preflight failed; training not started")?;
    Ok(report)
}
