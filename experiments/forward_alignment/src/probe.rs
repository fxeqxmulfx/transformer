//! Counterfactuals always restore saved parameters and moments. Geometry,
//! first-order predictions and finite loss changes are distinct evidence.
use crate::{
    application::{Backend, State},
    config::{self, Aux, Kernels, Method, Prototypes},
    coordinates as coord,
    domain::{
        Result,
        data::{Data, Plan},
        model::Arm,
    },
    finite,
    infrastructure::{engine::Engine, storage},
    measure::{self, Cost, Measured, Scope},
    reference::{self, Cache, Heads, Objective},
    signals::Signals,
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
}
struct Update {
    name: &'static str,
    parameters: Vec<f32>,
    delta: Vec<f64>,
    valid: bool,
    sha: String,
}
fn changes(before: &Value, after: &Value) -> Value {
    let mut out = json!({});
    for scope in Scope::ALL {
        let name = scope.name();
        out[name] = json!({
            "ce":after[name]["ce"].as_f64().unwrap()-before[name]["ce"].as_f64().unwrap(),
            "pair":after[name]["pair"].as_f64().unwrap()-before[name]["pair"].as_f64().unwrap(),
            "correct":after[name]["correct"].as_i64().unwrap()-before[name]["correct"].as_i64().unwrap()});
    }
    out
}
fn alignments(arm: Arm, measured: &[Measured], expanded: &[f64], displacement: bool) -> Value {
    let direction = if displacement {
        coord::owners(arm, expanded)
    } else {
        coord::pullback(arm, expanded)
    };
    let mut out = json!({});
    for (scope, m) in Scope::ALL.into_iter().zip(measured) {
        for (j, name) in ["ce", "pair"].into_iter().enumerate() {
            out[scope.name()][name] = coord::compare(
                &coord::pullback(arm, &m.gradients[j]),
                &direction,
                displacement,
            );
        }
    }
    out
}
fn remainders(change: &Value, alignment: &Value, group: &str) -> Value {
    let mut out = json!({});
    for scope in Scope::ALL {
        for objective in ["ce", "pair"] {
            out[scope.name()][objective] = json!(
                change[scope.name()][objective].as_f64().unwrap()
                    - alignment[scope.name()][objective][group]["pairing"]
                        .as_f64()
                        .unwrap()
            );
        }
    }
    out
}

pub fn run(case: &Case<'_>, cuda: &Cuda, kernels: &Kernels, cost: &mut Cost) -> Result<Value> {
    let arm = case.arm;
    let p: Vec<_> = case.state.parameters.iter().map(|&v| v as f64).collect();
    let prototypes = Prototypes::new(arm, case.seed);
    let heads = Heads::from(&prototypes);
    let aux = Aux::new(cuda, arm, &prototypes, Method::Auxiliary)?;
    let mut engine = Engine::new(arm, &case.state.parameters, case.data, case.plan)?;
    engine.restore(case.state)?;
    let first = measure::terminal(
        &engine,
        kernels,
        Scope::Batch,
        case.plan,
        case.step,
        true,
        cost,
    )?;
    let (offset, batch) = Scope::Batch.bounds(case.plan, case.step);
    let signals = Signals::new(
        &mut engine,
        kernels,
        &aux,
        batch,
        offset,
        &first.gradients,
        cost,
    )?;
    let mut measured = vec![first];
    for scope in [Scope::Train, Scope::Validation] {
        measured.push(measure::terminal(
            &engine, kernels, scope, case.plan, case.step, true, cost,
        )?);
    }
    let mut before = json!({});
    for (scope, m) in Scope::ALL.into_iter().zip(&measured) {
        before[scope.name()] = m.metrics.clone();
    }
    let replay = measure::replay(&before["train"], &case.expected["train"])
        && measure::replay(&before["validation"], &case.expected["validation"]);
    let mut numerical = finite::short(&engine, kernels, &p, case.data, case.seed, cost)?;
    let mut gradients = json!({});
    for (name, g) in [
        ("ce-bp", &signals.ce),
        ("pair-bp", &signals.pair),
        ("auxiliary-bp", &signals.auxiliary),
        ("local-pair", &signals.local),
        ("vocabulary-E", &signals.vocabulary),
        ("readout-E", &signals.readout),
        ("input-E", &signals.input),
    ] {
        gradients[name] = alignments(
            arm,
            &measured,
            &g.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            false,
        );
    }
    let mut updates = vec![];
    for (i, name) in Signals::RULES.into_iter().enumerate() {
        engine.restore(case.state)?;
        engine.gradients.write(&signals.gradient(i))?;
        engine.update(case.step + 1)?;
        cost.optimizer_steps += 1;
        let state = engine.state()?;
        let delta = state
            .parameters
            .iter()
            .zip(&case.state.parameters)
            .map(|(&a, &b)| a as f64 - b as f64)
            .collect();
        let sha = storage::hash(&storage::state_bytes(case.step + 1, &state));
        updates.push(Update {
            name,
            valid: config::valid_state(arm, &state),
            parameters: state.parameters,
            delta,
            sha,
        });
    }
    engine.restore(case.state)?;
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
    let identical =
        storage::hash(&storage::state_bytes(case.step + 1, &engine.state()?)) == updates[3].sha;
    for (scope, m) in [Scope::Train, Scope::Validation]
        .into_iter()
        .zip(&measured[1..])
    {
        for mut row in finite::full(
            arm,
            &p,
            &scope.rows(case.data, case.plan, case.step),
            &updates[3].delta,
            m,
            cost,
        ) {
            row["scope"] = json!(scope.name());
            row["group"] = json!("normalized-local-displacement");
            numerical.push(row);
        }
    }
    let mut pairing_checks = vec![];
    let mut steps = vec![];
    for update in &updates {
        for (scope, m) in Scope::ALL.into_iter().zip(&measured) {
            for j in 0..2 {
                let mut check = coord::pairing_check(arm, &m.gradients[j], &update.delta);
                check["scope"] = json!(scope.name());
                check["objective"] = json!(["ce", "pair"][j]);
                check["rule"] = json!(update.name);
                pairing_checks.push(check);
            }
        }
        engine.parameters.write(&update.parameters)?;
        let after = measure::metrics_all(&engine, kernels, case.plan, case.step, cost)?;
        let change = changes(&before, &after);
        let alignment = alignments(arm, &measured, &update.delta, true);
        let contrast: Vec<_> = update
            .delta
            .iter()
            .zip(&updates[6].delta)
            .map(|(a, b)| a - b)
            .collect();
        steps.push(json!({"rule":update.name,"valid_state":update.valid,"state_sha256":update.sha,
            "alignment":alignment,"zero_gradient_contrast":alignments(arm,&measured,&contrast,true),
            "after":after,"change":change,"first_order_remainder":remainders(&change,&alignment,"all")}));
    }
    let batch_rows = Scope::Batch.rows(case.data, case.plan, case.step);
    let old = Cache::new(arm, &p, &batch_rows);
    cost.cpu_forward_calls += 1;
    cost.cpu_forward_rows += batch;
    let mut blocks = vec![];
    for ((name, range), objective) in coord::expanded_groups(arm).into_iter().take(3).zip([
        Objective::SharedE,
        Objective::A,
        Objective::B,
    ]) {
        let mut partial = case.state.parameters.clone();
        partial[range.clone()].copy_from_slice(&updates[3].parameters[range]);
        engine.parameters.write(&partial)?;
        let after = measure::metrics_all(&engine, kernels, case.plan, case.step, cost)?;
        let partial: Vec<_> = partial.iter().map(|&x| x as f64).collect();
        let local_before = reference::objective(objective, arm, &p, &batch_rows, &old, &heads);
        let local_after = reference::objective(objective, arm, &partial, &batch_rows, &old, &heads);
        cost.frozen_local_evaluations += 2;
        let change = changes(&before, &after);
        blocks.push(json!({"group":name,"after":after,"change":change,
            "first_order_remainder":remainders(&change,&steps[3]["alignment"],name),
            "frozen_local_objective":{"before":local_before,"after":local_after,"change":local_after-local_before}}));
    }
    let mut interaction = json!({});
    for scope in Scope::ALL {
        for loss in ["ce", "pair"] {
            interaction[scope.name()][loss] = json!(
                steps[3]["change"][scope.name()][loss].as_f64().unwrap()
                    - blocks
                        .iter()
                        .map(|b| b["change"][scope.name()][loss].as_f64().unwrap())
                        .sum::<f64>()
            );
        }
    }
    engine.restore(case.state)?;
    let passed = replay
        && identical
        && signals.checks["passed"] == true
        && updates.iter().all(|u| u.valid)
        && numerical.iter().all(|r| r["passed"] == true)
        && pairing_checks.iter().all(|r| r["passed"] == true);
    Ok(
        json!({"passed":passed,"source_replay":replay,"local_update_bitwise_match":identical,
        "signal_checks":signals.checks,"before":before,"gradient_alignment":gradients,"updates":steps,
        "local_block_probes":blocks,"local_cross_group_interaction":interaction,
        "finite_differences":numerical,"coordinate_pairing_checks":pairing_checks,
        "update_batch_rows":batch,"prototype0_sha256":storage::hash(&storage::floats(&prototypes.vocabulary)),
        "prototype1_sha256":storage::hash(&storage::floats(&prototypes.hidden))}),
    )
}
