//! Direction and scale are crossed exhaustively at a restored saved state.
//! Terminal gradients are diagnostic; no scale is selected for training.
use crate::{
    application::{Backend, State},
    config::{self, Aux, Kernels, Method, Prototypes},
    coordinates as coord, directions,
    domain::{
        Result,
        data::{Data, Plan},
        model::Arm,
    },
    finite,
    infrastructure::{engine::Engine, storage},
    measure::{self, Cost, Measured, Scope},
    reference::{self, Cache, Heads, Objective},
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};

pub struct Case<'a> {
    pub arm: Arm,
    pub seed: u64,
    pub step: usize,
    pub data: &'a Data,
    pub plan: &'a Plan,
    pub state: &'a State,
    pub expected: &'a Value,
    pub prior: &'a Value,
}
fn same_metrics(actual: &Value, expected: &Value) -> bool {
    Scope::ALL.iter().all(|scope| {
        let (a, b) = (&actual[scope.name()], &expected[scope.name()]);
        a["correct"] == b["correct"]
            && a["count"] == b["count"]
            && ["ce", "pair"]
                .iter()
                .all(|key| (a[key].as_f64().unwrap() - b[key].as_f64().unwrap()).abs() <= 1e-12)
    })
}
fn loss(before: f64, after: f64, prediction: f64) -> Value {
    let change = after - before;
    let threshold = 1e-7 * (1. + before.abs());
    let status = if change < -threshold {
        "improved"
    } else if change > threshold {
        "worsened"
    } else {
        "near-flat"
    };
    json!({"before":before,"after":after,"change":change,"threshold":threshold,"status":status,
        "prediction":prediction,"remainder":change-prediction})
}
fn outcomes(
    arm: Arm,
    before: &Value,
    after: &Value,
    measured: &[Measured],
    delta: &[f64],
) -> Value {
    let independent = coord::owners(arm, delta);
    let mut out = json!({});
    for (scope, m) in Scope::ALL.into_iter().zip(measured) {
        let name = scope.name();
        out[name]["correct_change"] = json!(
            after[name]["correct"].as_i64().unwrap() - before[name]["correct"].as_i64().unwrap()
        );
        for (j, objective) in ["ce", "pair"].into_iter().enumerate() {
            let g = coord::pullback(arm, &m.gradients[j]);
            let prediction = coord::dot(&g, &independent);
            let mut value = loss(
                before[name][objective].as_f64().unwrap(),
                after[name][objective].as_f64().unwrap(),
                prediction,
            );
            value["alignment"] = coord::compare(&g, &independent, true)["A"].clone();
            out[name][objective] = value;
        }
    }
    out
}

pub fn run(case: &Case<'_>, cuda: &Cuda, kernels: &Kernels, cost: &mut Cost) -> Result<Value> {
    let arm = case.arm;
    let p: Vec<_> = case.state.parameters.iter().map(|&v| v as f64).collect();
    let prototypes = Prototypes::new(arm, case.seed);
    let heads = Heads::from(&prototypes);
    let aux = Aux::new(cuda, arm, &prototypes, Method::Local)?;
    let mut engine = Engine::new(arm, &case.state.parameters, case.data, case.plan)?;
    engine.restore(case.state)?;
    let (offset, batch) = Scope::Batch.bounds(case.plan, case.step);
    engine.pilot_step(
        Method::Local,
        Some(&aux),
        kernels,
        offset,
        batch,
        case.step + 1,
    )?;
    cost.forward_batches += 1;
    cost.local_output += 1;
    cost.hidden_jacobians += 1;
    cost.vocabulary_jacobians += 1;
    cost.optimizer_steps += 1;
    let updated = engine.state()?;
    let gradient = engine.gradients.read()?;
    let source_step_replay = storage::hash(&storage::state_bytes(case.step + 1, &updated))
        == case.prior["updates"][3]["state_sha256"];
    let directions =
        directions::build(arm, &case.state.parameters, &updated.parameters, &gradient)?;
    engine.restore(case.state)?;
    let mut measured = vec![];
    let mut before = json!({});
    for scope in Scope::ALL {
        let m = measure::terminal(&engine, kernels, scope, case.plan, case.step, true, cost)?;
        before[scope.name()] = m.metrics.clone();
        measured.push(m);
    }
    let baseline_replay = measure::replay(&before["train"], &case.expected["train"])
        && measure::replay(&before["validation"], &case.expected["validation"])
        && same_metrics(&before, &case.prior["before"]);
    let prior_a = case.prior["local_block_probes"]
        .as_array()
        .ok_or("missing prior blocks")?
        .iter()
        .find(|b| b["group"] == "A")
        .ok_or("missing prior A block")?;
    let batch_rows = Scope::Batch.rows(case.data, case.plan, case.step);
    let old = Cache::new(arm, &p, &batch_rows);
    cost.cpu_forward_calls += 1;
    cost.cpu_forward_rows += batch;
    let local_before = reference::objective(Objective::A, arm, &p, &batch_rows, &old, &heads);
    cost.frozen_local_evaluations += 1;
    let mut numerical = vec![];
    let mut probes = vec![];
    let mut pairing_checks = vec![];
    let mut full_scale_replay = false;
    for direction in &directions.choices {
        let expanded = coord::expand(arm, &direction.independent);
        for (scope, m) in [Scope::Train, Scope::Validation]
            .into_iter()
            .zip(&measured[1..])
        {
            for mut row in finite::full(
                arm,
                &p,
                &scope.rows(case.data, case.plan, case.step),
                &expanded,
                m,
                cost,
            ) {
                row["scope"] = json!(scope.name());
                row["direction"] = json!(direction.name);
                numerical.push(row);
            }
        }
        for scale in directions::SCALES {
            let parameters =
                directions::displace(arm, &case.state.parameters, &direction.independent, scale);
            let valid = directions::valid(arm, &case.state.parameters, &parameters);
            let delta: Vec<_> = parameters
                .iter()
                .zip(&case.state.parameters)
                .map(|(&a, &b)| a as f64 - b as f64)
                .collect();
            let independent = coord::owners(arm, &delta);
            let requested: Vec<_> = direction.independent.iter().map(|x| scale * x).collect();
            let error: Vec<_> = independent
                .iter()
                .zip(&requested)
                .map(|(a, b)| a - b)
                .collect();
            let actual_norm = coord::norm(&independent);
            let requested_norm = coord::norm(&requested);
            let gn = coord::norm(&directions.gradient);
            let cosine = if actual_norm * gn > 0. {
                Some(-coord::dot(&directions.gradient, &independent) / (actual_norm * gn))
            } else {
                None
            };
            for (scope, m) in Scope::ALL.into_iter().zip(&measured) {
                for j in 0..2 {
                    let mut check = coord::pairing_check(arm, &m.gradients[j], &delta);
                    check["scope"] = json!(scope.name());
                    check["objective"] = json!(["ce", "pair"][j]);
                    check["direction"] = json!(direction.name);
                    check["scale"] = json!(scale);
                    pairing_checks.push(check);
                }
            }
            engine.parameters.write(&parameters)?;
            let after = measure::metrics_all(&engine, kernels, case.plan, case.step, cost)?;
            let changed: Vec<_> = parameters.iter().map(|&x| x as f64).collect();
            let local_after =
                reference::objective(Objective::A, arm, &changed, &batch_rows, &old, &heads);
            cost.frozen_local_evaluations += 1;
            if direction.name == "adamw" && scale == 1. {
                let a = coord::expanded_groups(arm)[1].1.clone();
                full_scale_replay =
                    directions::same_bits(&parameters[a.clone()], &updated.parameters[a])
                        && same_metrics(&after, &prior_a["after"])
                        && (local_before
                            - prior_a["frozen_local_objective"]["before"]
                                .as_f64()
                                .unwrap())
                        .abs()
                            <= 1e-12
                        && (local_after
                            - prior_a["frozen_local_objective"]["after"].as_f64().unwrap())
                        .abs()
                            <= 1e-12;
            }
            probes.push(json!({"direction":direction.name,"scale":scale,"valid_ties_and_fixed_E_B":valid,
                "parameters_sha256":storage::hash(&storage::floats(&parameters)),
                "requested_norm":requested_norm,"realized_norm":actual_norm,"rounding_error_norm":coord::norm(&error),
                "cosine_to_negative_local_gradient":cosine,"after":after,
                "terminal":outcomes(arm,&before,&after,&measured,&delta),
                "local_objective":loss(local_before,local_after,coord::dot(&directions.gradient,&independent))}));
        }
    }
    engine.restore(case.state)?;
    let passed = baseline_replay
        && source_step_replay
        && full_scale_replay
        && config::valid_state(arm, &updated)
        && directions.checks["passed"] == true
        && probes.iter().all(|r| r["valid_ties_and_fixed_E_B"] == true)
        && numerical.iter().all(|r| r["passed"] == true)
        && pairing_checks.iter().all(|r| r["passed"] == true);
    Ok(
        json!({"passed":passed,"baseline_replay":baseline_replay,"source_step_replay":source_step_replay,
        "adamw_full_scale_replay":full_scale_replay,"source_update_valid":config::valid_state(arm,&updated),
        "directions":directions.checks,"before":before,"local_objective_before":local_before,
        "probes":probes,"finite_differences":numerical,"coordinate_pairing_checks":pairing_checks}),
    )
}
