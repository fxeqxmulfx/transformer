//! Bitwise gates for compact moments, initial +0 aliases, warmup and epsilon.
use crate::{
    application::{Backend, State},
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::storage,
    inputs::{self, Ledger, require},
    layout, state,
    trainer::Trainer,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::path::Path;
pub fn hash(step: usize, t: &Trainer) -> Result<String> {
    Ok(storage::hash(&storage::state_bytes(step, &t.state()?)))
}
fn edges(cuda: &Cuda) -> Result<Value> {
    let data = Data::new(29);
    let plan = Plan::new(29, 1);
    let mut records = vec![];
    for arm in Arm::ALL {
        for tiny in [false, true] {
            let first: Vec<_> = (0..layout::COUNT)
                .map(|i| ((i % 11) as f32 - 5.) * 1e-9)
                .collect();
            let second = vec![if tiny { 1e-24 } else { 0. }; layout::COUNT];
            let s = State {
                parameters: model::initialize(arm, 29),
                first: layout::expand(arm, &first, false),
                second: layout::expand(arm, &second, true),
            };
            let mut g: Vec<_> = (0..arm.count())
                .map(|i| {
                    if tiny {
                        ((i % 13) as f32 - 6.) * 1e-12
                    } else {
                        0.
                    }
                })
                .collect();
            if tiny {
                g[arm.count() - 1] = 1e-11;
            }
            let mut pair = [
                Trainer::new(arm, &s.parameters, &data, &plan, false, cuda)?,
                Trainer::new(arm, &s.parameters, &data, &plan, true, cuda)?,
            ];
            for t in &mut pair {
                t.restore(&s)?;
                t.set_gradient(&g)?;
                t.update(1)?;
            }
            let hashes = [hash(1, &pair[0])?, hash(1, &pair[1])?];
            let valid = pair
                .iter()
                .map(|t| Ok(state::valid(arm, &t.state()?)))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .all(|v| v);
            records.push(json!({"arm":arm.name(),"kind":if tiny{"tiny-gradient"}else{"zero-gradient"},"hashes":hashes,"passed":valid&&hashes[0]==hashes[1]}));
        }
    }
    Ok(json!({"passed":records.iter().all(|r|r["passed"]==true),"cases":records}))
}
pub fn run(
    root: &Path,
    output: &Path,
    ledger: &mut Ledger,
    protocol: &str,
    cuda: &Cuda,
) -> Result<Value> {
    let edges = edges(cuda)?;
    let mut records = vec![];
    for seed in [7, 11] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 11000);
        for arm in Arm::ALL {
            for step in [0, 5000, 10000] {
                let mut row = json!({"arm":arm.name(),"seed":seed,"step":step});
                let outcome = (|| -> Result<Value> {
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
                    for t in &mut pair {
                        t.restore(&saved.state)?;
                    }
                    let initial_match = hash(step, &pair[0])? == hash(step, &pair[1])?;
                    let offset = plan.boundaries[step];
                    let batch = plan.boundaries[step + 1] - offset;
                    let mut warmup = vec![];
                    if step == 0 {
                        let g = pair[0].prepare_gradient(offset, batch)?;
                        pair[1].set_gradient(&g)?;
                        for clock in 1..=20 {
                            for t in &pair {
                                t.update(clock)?;
                            }
                            let hashes = [hash(clock, &pair[0])?, hash(clock, &pair[1])?];
                            warmup.push(json!({"clock":clock,"passed":hashes[0]==hashes[1],"hashes":hashes}));
                        }
                    }
                    let mut states = vec![];
                    let mut metrics = vec![];
                    for t in &mut pair {
                        t.restore(&saved.state)?;
                        t.train_step(offset, batch, step + 1)?;
                        require(state::valid(arm, &t.state()?), "invalid updated state")?;
                        states.push(hash(step + 1, t)?);
                        metrics.push(storage::report_value(&crate::application::Report {
                            step: step + 1,
                            train: t.evaluate(false)?,
                            validation: t.evaluate(true)?,
                        }));
                    }
                    let passed = initial_match
                        && states[0] == states[1]
                        && metrics[0] == metrics[1]
                        && warmup.iter().all(|w| w["passed"] == true);
                    Ok(
                        json!({"passed":passed,"initial_match":initial_match,"state_hashes":states,"metrics_match":metrics[0]==metrics[1],"metrics":metrics,"warmup":warmup}),
                    )
                })();
                match outcome {
                    Ok(v) => {
                        row["passed"] = v["passed"].clone();
                        row["audit"] = v;
                    }
                    Err(e) => {
                        row["passed"] = json!(false);
                        row["error"] = json!(e.to_string());
                    }
                }
                eprintln!(
                    "compact preflight {} seed={seed} step={step} passed={}",
                    arm.name(),
                    row["passed"]
                );
                records.push(row);
            }
        }
    }
    let report = json!({"passed":edges["passed"]==true&&records.iter().all(|r|r["passed"]==true),"edges":edges,"cases":records});
    storage::json_write(&output.join("preflight.json"), &report)?;
    Ok(report)
}
