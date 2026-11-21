//! E6: the learner of `convex_stress_schatten`, unchanged, on tasks `ab + u(a) + v(b) mod 97`.
//! This file is `convex_stress_schatten/src/main.rs` at `09fc84dd` with the task source
//! replaced: `selected` reads the 54 original cases of `convex_stress_arithmetic_data`, `task`
//! turns a case's frozen half split into the suite's arithmetic examples, and the data
//! fingerprints are the case's historical ones. Model, budgets, solver, arms, caps, cadence,
//! stopping and the training-only hull selection are the original code. `compare` is E6's
//! preregistered analysis (CM121 in `review/convex/moments.md`).
use convex_stress_schatten::{
    Model, Result, choose_by_train, data, hash, ideal_loss, metrics,
    projection::{Norms, Projector},
    random::Random,
    samples,
    solver::{Mode, Solver},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, io::Write, path::Path, time::Instant};

const PROTOCOL: &str = include_str!("../protocol.json");

fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../../convex_stress_schatten/src/lib.rs"),
        include_bytes!("../../convex_stress_schatten/src/projection.rs"),
        include_bytes!("../../convex_stress_schatten/src/solver.rs"),
        include_bytes!("../../convex_stress_moments/src/fft.rs"),
        include_bytes!("../../convex_stress_moments/src/lib.rs"),
        include_bytes!("../../convex_stress_moments/src/projection.rs"),
        include_bytes!("../../convex_stress_moments/src/solver.rs"),
        include_bytes!("../../convex_stress_spectral/src/lib.rs"),
        include_bytes!("../../convex_stress_spectral/src/linalg.rs"),
        include_bytes!("../../convex_stress_minibatch/src/lib.rs"),
        include_bytes!("../../convex_stress_minibatch/src/metrics.rs"),
        include_bytes!("../../convex_stress/src/channel.rs"),
        include_bytes!("../../convex_stress/src/features.rs"),
        include_bytes!("../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../convex_stress/src/vendor/blake.rs"),
        include_bytes!("../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../convex_stress_arithmetic_data/src/lib.rs"),
        include_bytes!("../../convex_stress_arithmetic_data/src/dataset.rs"),
        include_bytes!("../../convex_stress_arithmetic_data/src/oracle.rs"),
        arithmetic::CASES.as_bytes(),
        PROTOCOL.as_bytes(),
    ];
    let mut bytes = vec![];
    for source in sources {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
/// The original case `name` among the 54 arithmetic cases.
fn selected(name: &str) -> Result<Value> {
    serde_json::from_str::<Value>(arithmetic::CASES)?
        .as_array()
        .ok_or("no cases")?
        .iter()
        .find(|r| r["case"]["name"] == name)
        .cloned()
        .ok_or_else(|| "unknown task".into())
}
/// The case's historical fingerprints, which `arithmetic::dataset::build` checks.
fn frozen(row: &Value) -> &Value {
    &row["history"]["data_fingerprints"]
}
/// The case's frozen half split as the suite's arithmetic examples: tokens `[a, b]`, the label
/// supervised at the last position, exactly as `convex_stress`'s `data::build` writes them.
fn task(row: &Value) -> Result<data::Task> {
    let d = arithmetic::dataset::build(row)
        .map_err(|e| e.to_string())?
        .ok_or("case without a native oracle")?;
    let examples = |rows: &[[u32; 3]]| -> Vec<data::Example> {
        rows.iter()
            .map(|&[a, b, y]| data::Example {
                task: "arithmetic".into(),
                tokens: vec![a.into(), b.into()],
                targets: vec![-100, y.into()],
                changes: vec![false, false],
                prompt: vec![],
                answer: vec![],
                limit: 0,
                field: None,
                row_index: None,
            })
            .collect()
    };
    Ok(data::Task {
        splits: BTreeMap::from([
            ("train".into(), examples(&d.training.rows)),
            ("validation".into(), examples(&d.validation)),
        ]),
        vocab: d.training.vocabulary,
        selection: "validation".into(),
    })
}
fn budgets(label: &str, n: usize) -> Result<(f64, f64)> {
    match label {
        "full_rank" => Ok(((n as f64).sqrt(), n as f64 * (n as f64).sqrt())),
        "rank_one" => Ok((n as f64, n as f64)),
        _ => Err("unknown spectral budget".into()),
    }
}
fn allowed(p: &Value, key: &str, label: &str) -> bool {
    p[key].as_array().unwrap().iter().any(|v| v == label)
}
fn cap(p: &Value, arm: &str) -> usize {
    p[match arm {
        "full_batch" => "full_updates",
        "minibatch" => "minibatch_updates",
        _ => "forward_updates",
    }]
    .as_u64()
    .unwrap() as usize
}
fn metrics_all(model: &Model, task: &data::Task) -> Result<Value> {
    let mut out = serde_json::Map::new();
    for (name, examples) in &task.splits {
        out.insert(name.clone(), metrics::evaluate(model, examples)?);
    }
    Ok(Value::Object(out))
}
fn maximum(xs: &[Norms]) -> f64 {
    xs.iter().map(|s| s.operator).fold(0., f64::max)
}
fn max_nuclear(xs: &[Norms]) -> f64 {
    xs.iter().map(|s| s.nuclear).fold(0., f64::max)
}
fn number(v: &Value, name: &str) -> Result<f64> {
    v[name]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("missing finite {name}").into())
}

fn run(arm: &str, name: &str, r: &str, out: &Path) -> Result<()> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    if !allowed(&protocol, "arms", arm)
        || !allowed(&protocol, "tasks", name)
        || !allowed(&protocol, "budgets", r)
    {
        return Err("cell absent from frozen protocol".into());
    }
    let row = selected(name)?;
    let task = task(&row)?;
    let (radius, budget) = budgets(r, task.vocab)?;
    let initial = Model::new(task.vocab, radius, budget)?;
    let samples = samples(&initial, &task.splits["train"])?;
    let count = samples.len();
    let mode = Mode::parse(arm)?;
    let mut solver = Solver::new(
        &initial,
        samples,
        mode,
        protocol["frequency_workers"].as_u64().unwrap() as usize,
    )?;
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let per_epoch = count.div_ceil(batch);
    let cadence = (row["case"]["observe"].as_u64().unwrap() as usize).min(per_epoch);
    let limit = cap(&protocol, arm);
    let seed = protocol["seed"].as_u64().unwrap() as u32;
    let tolerance = number(&protocol, "ideal_ce_tolerance")?;
    fs::create_dir(out)?;
    let started = Instant::now();
    let mut result = json!({"task":name,"arm":arm,"budget_name":r,"radius":initial.radius,"nuclear_budget":budget,"seed":seed,
        "source_fingerprint":fingerprint(),"protocol":protocol,"data_fingerprints":frozen(&row),
        "train_prefixes":count,"vocab":task.vocab,"batch_size":batch,"updates_per_epoch":per_epoch,
        "selection":task.selection,"first_primary_99_update":null,"ideal_ce":ideal_loss(task.vocab)});
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    let mut trace = fs::File::create(out.join("trace.jsonl"))?;
    let mut order: Vec<_> = (0..count).collect();
    let mut rng = Random::new(seed + 10_000);
    let (mut updates, mut presented, mut minimum_alpha) = (0, 0, 1_f64);
    let (model, alpha, shadow_norms, stop_reason) = loop {
        if updates % cadence == 0 || updates == limit {
            let (model, alpha, norms) = solver.snapshot()?;
            minimum_alpha = minimum_alpha.min(alpha);
            let primary = metrics::evaluate(&model, &task.splits[&task.selection])?;
            let train = metrics::evaluate(&model, &task.splits["train"])?;
            let excess = number(&train, "loss")? - ideal_loss(task.vocab);
            if excess < -1e-10 {
                return Err("CE below proved lower bound".into());
            }
            let pass = primary["exact"].as_u64().unwrap() * 100
                >= primary["examples"].as_u64().unwrap() * 99;
            if pass && result["first_primary_99_update"].is_null() {
                result["first_primary_99_update"] = updates.into();
            }
            let record = json!({"update":updates,"seconds":started.elapsed().as_secs_f64(),"primary":primary,
                "train":train,"ideal_ce_excess":excess,"uniform_mixture_alpha":alpha,
                "shadow_max_operator_norm":maximum(&norms),"shadow_max_nuclear_norm":max_nuclear(&norms),"last_split_residual":solver.residual});
            writeln!(trace, "{record}")?;
            trace.flush()?;
            println!(
                "update={updates} primary={} train={} ce_excess={excess:.4e} alpha={alpha:.8}",
                primary["sequence_accuracy"], train["sequence_accuracy"]
            );
            if updates > 0 && excess <= tolerance {
                break (model, alpha, norms, "ideal_ce_excess");
            }
            if updates == limit {
                break (model, alpha, norms, "budget");
            }
        }
        let within = updates % per_epoch;
        if mode == Mode::Mini && within == 0 {
            rng.shuffle(&mut order);
        }
        let indices = if mode == Mode::Mini {
            &order[within * batch..((within + 1) * batch).min(count)]
        } else {
            &order
        };
        solver.step(indices)?;
        if mode != Mode::Forward || updates == 0 {
            presented += indices.len();
        }
        updates += 1;
    };
    let actual_norms = model.validate(&mut solver.projector)?;
    result["updates"] = updates.into();
    result["stop_reason"] = stop_reason.into();
    result["examples_presented"] = presented.into();
    result["epochs"] = (presented as f64 / count as f64).into();
    result["forward_training_gradient_evaluations"] = 0.into();
    result["uniform_mixture_alpha"] = alpha.into();
    result["minimum_observed_alpha"] = minimum_alpha.into();
    result["shadow_max_operator_norm"] = maximum(&shadow_norms).into();
    result["shadow_max_nuclear_norm"] = max_nuclear(&shadow_norms).into();
    result["moment_norms"] = serde_json::to_value(actual_norms)?;
    result["fit_defect"] = solver.fit_defect(&model).into();
    result["last_split_residual"] = solver.residual.into();
    result["full_operator_residual"] = solver.full_residual()?.into();
    result["metrics"] = metrics_all(&model, &task)?;
    result["ideal_ce_excess"] =
        (number(&result["metrics"]["train"], "loss")? - ideal_loss(task.vocab)).into();
    result["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    let bytes = serde_json::to_vec(&model)?;
    let state = serde_json::to_vec(solver.diagnostic_state())?;
    result["model_sha256"] = hash(&bytes).into();
    result["diagnostic_state_sha256"] = hash(&state).into();
    fs::write(out.join("model.json"), bytes)?;
    fs::write(out.join("diagnostic_state.json"), state)?;
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    println!(
        "complete {}",
        result["metrics"][&task.selection]["sequence_accuracy"]
    );
    Ok(())
}

fn replay(out: &Path) -> Result<()> {
    let report: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    if report["source_fingerprint"] != fingerprint() || report["protocol"] != protocol {
        return Err("source/protocol mismatch".into());
    }
    let name = report["task"].as_str().ok_or("no task")?;
    let arm = report["arm"].as_str().ok_or("no arm")?;
    let r = report["budget_name"].as_str().ok_or("no radius")?;
    if !allowed(&protocol, "tasks", name)
        || !allowed(&protocol, "arms", arm)
        || !allowed(&protocol, "budgets", r)
        || report["seed"] != protocol["seed"]
    {
        return Err("unplanned reported cell".into());
    }
    let row = selected(name)?;
    let task = task(&row)?;
    let bytes = fs::read(out.join("model.json"))?;
    let state = fs::read(out.join("diagnostic_state.json"))?;
    if report["model_sha256"] != hash(&bytes)
        || report["diagnostic_state_sha256"] != hash(&state)
        || report["data_fingerprints"] != *frozen(&row)
    {
        return Err("model/state/data fingerprint mismatch".into());
    }
    let model: Model = serde_json::from_slice(&bytes)?;
    model.validate_shape()?;
    if model.vocab != task.vocab
        || (model.radius, model.nuclear_budget) != budgets(r, task.vocab)?
        || number(&report, "radius")? != model.radius
        || number(&report, "nuclear_budget")? != model.nuclear_budget
    {
        return Err("model configuration mismatch".into());
    }
    let mut solver = Solver::new(
        &model,
        samples(&model, &task.splits["train"])?,
        Mode::parse(arm)?,
        protocol["frequency_workers"].as_u64().unwrap() as usize,
    )?;
    solver.restore_diagnostic_state(serde_json::from_slice(&state)?)?;
    let (snapshot, alpha, shadow_norms) = solver.snapshot()?;
    if snapshot.probabilities != model.probabilities {
        return Err("saved model differs from splitting-state output".into());
    }
    let norms = model.validate(&mut solver.projector)?;
    let metrics = metrics_all(&model, &task)?;
    let excess = number(&metrics["train"], "loss")? - ideal_loss(task.vocab);
    let residual = solver.full_residual()?;
    let saved_norms: Vec<Norms> = serde_json::from_value(report["moment_norms"].clone())?;
    if report["metrics"] != metrics
        || saved_norms.len() != norms.len()
        || saved_norms.iter().zip(&norms).any(|(a, b)| {
            !a.operator.is_finite()
                || !a.nuclear.is_finite()
                || (a.operator - b.operator).abs() > 1e-9
                || (a.nuclear - b.nuclear).abs() > 1e-8
        })
        || (number(&report, "uniform_mixture_alpha")? - alpha).abs() > 1e-12
        || (number(&report, "shadow_max_operator_norm")? - maximum(&shadow_norms)).abs() > 1e-9
        || (number(&report, "shadow_max_nuclear_norm")? - max_nuclear(&shadow_norms)).abs() > 1e-8
        || (number(&report, "full_operator_residual")? - residual).abs() > 1e-9
        || (number(&report, "ideal_ce_excess")? - excess).abs() > 1e-12
        || (number(&report, "fit_defect")? - solver.fit_defect(&model)).abs() > 1e-12
    {
        return Err("saved replay differs from reported measurements".into());
    }
    let update = report["updates"].as_u64().ok_or("no update")? as usize;
    let count = task.splits["train"].len();
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let per_epoch = count.div_ceil(batch);
    let cadence = (row["case"]["observe"].as_u64().unwrap() as usize).min(per_epoch);
    let presented = match arm {
        "full_batch" => update * count,
        "minibatch" => update / per_epoch * count + ((update % per_epoch) * batch).min(count),
        _ => count,
    };
    if report["selection"] != task.selection
        || report["vocab"] != task.vocab
        || report["train_prefixes"] != count
        || report["batch_size"] != batch
        || report["updates_per_epoch"] != per_epoch
        || report["examples_presented"] != presented
        || (update != cap(&protocol, arm) && !update.is_multiple_of(cadence))
    {
        return Err("invalid evaluation or training-exposure metadata".into());
    }
    if update == 0
        || update > cap(&protocol, arm)
        || match report["stop_reason"].as_str() {
            Some("budget") => update != cap(&protocol, arm),
            Some("ideal_ce_excess") => excess > number(&protocol, "ideal_ce_tolerance")?,
            _ => true,
        }
    {
        return Err("invalid stopping condition".into());
    }
    println!(
        "{}",
        json!({"replayed":true,"source_fingerprint":fingerprint(),"task":name,"arm":arm,"budget":r,
        "primary":metrics[&task.selection]["sequence_accuracy"],"full_operator_residual":residual,"ideal_ce_excess":excess})
    );
    Ok(())
}

fn hull_selection(cohort: &Path) -> Result<Value> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let tolerance = number(&protocol, "ideal_ce_tolerance")?;
    let mut choices = vec![];
    for task_value in protocol["tasks"].as_array().unwrap() {
        let name = task_value.as_str().unwrap();
        let row = selected(name)?;
        let task = task(&row)?;
        let ideal = ideal_loss(task.vocab);
        for arm_value in protocol["arms"].as_array().unwrap() {
            let arm = arm_value.as_str().unwrap();
            let mut candidates = vec![];
            for budget_value in protocol["budgets"].as_array().unwrap() {
                let budget_name = budget_value.as_str().unwrap();
                let dir = cohort.join(format!("{arm}--{name}--{budget_name}"));
                let report: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
                let bytes = fs::read(dir.join("model.json"))?;
                let model: Model = serde_json::from_slice(&bytes)?;
                if report["source_fingerprint"] != fingerprint()
                    || report["protocol"] != protocol
                    || report["task"] != name
                    || report["arm"] != arm
                    || report["budget_name"] != budget_name
                    || report["data_fingerprints"] != *frozen(&row)
                    || report["model_sha256"] != hash(&bytes)
                    || model.vocab != task.vocab
                    || (model.radius, model.nuclear_budget) != budgets(budget_name, task.vocab)?
                    || number(&report, "radius")? != model.radius
                    || number(&report, "nuclear_budget")? != model.nuclear_budget
                {
                    return Err("invalid hull candidate identity".into());
                }
                model.validate(&mut Projector::new(
                    model.vocab,
                    protocol["frequency_workers"].as_u64().unwrap() as usize,
                )?)?;
                // Decision inputs are recomputed solely from observed training rows.
                // Held-out metrics are read only after choose_by_train returns.
                let train = metrics::evaluate(&model, &task.splits["train"])?;
                let ce = number(&train, "loss")?;
                candidates.push((budget_name.to_owned(), model, report, ce));
            }
            let losses: Vec<_> = candidates.iter().map(|c| c.3).collect();
            let winner = choose_by_train(&losses, ideal, tolerance)?;
            let (budget_name, model, report, ce) = &candidates[winner];
            let all_metrics = metrics_all(model, &task)?;
            let comparison: Vec<_>=candidates.iter().map(|c|json!({"budget":c.0,"train_ce":c.3,
                "ideal_ce_excess":c.3-ideal,"updates":c.2["updates"],"elapsed_seconds":c.2["elapsed_seconds"],
                "examples_presented":c.2["examples_presented"],"model_sha256":c.2["model_sha256"]})).collect();
            choices.push(json!({"task":name,"arm":arm,"selected_budget":budget_name,
                "selected_model":format!("{arm}--{name}--{budget_name}/model.json"),"model_sha256":report["model_sha256"],
                "training_ce":ce,"ideal_ce_excess":ce-ideal,"ideal_ce_certificate":ce-ideal<=tolerance,
                "candidates":comparison,"selection":task.selection,"metrics":all_metrics}));
        }
    }
    Ok(json!({"source_fingerprint":fingerprint(),"protocol":protocol,"choices":choices}))
}

/// Labels on all `p²` cells, indexed `b + p a` as `Model::cell` indexes them, from both splits.
fn table(name: &str) -> Result<(usize, Vec<usize>)> {
    let task = task(&selected(name)?)?;
    let p = task.vocab;
    let mut labels = vec![usize::MAX; p * p];
    for e in task.splits.values().flatten() {
        labels[e.tokens[1] as usize + p * e.tokens[0] as usize] = e.targets[1] as usize;
    }
    if labels.contains(&usize::MAX) {
        return Err("table misses a cell".into());
    }
    Ok((p, labels))
}

/// E6's comparison with `arithmetic-mul` (CM121). For each task, the shift `s = f − ab` per cell
/// from the two label tables, whether it is `u(a) + v(b)`, and for every arm and budget the
/// largest `|P_f[a, b, c + s(a, b)] − P_mul[a, b, c]|`, the cells whose argmax differs after
/// the shift, and both reports' stop and held-out counts. Run after `select`. O(36 p³) time.
fn compare(cohort: &Path) -> Result<Value> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let reference = "arithmetic-mul";
    let (p, mul) = table(reference)?;
    let read = |arm: &str, name: &str, budget: &str| -> Result<(Value, Model)> {
        let dir = cohort.join(format!("{arm}--{name}--{budget}"));
        Ok((
            serde_json::from_slice(&fs::read(dir.join("result.json"))?)?,
            serde_json::from_slice(&fs::read(dir.join("model.json"))?)?,
        ))
    };
    let argmax = |row: &[f64]| (0..row.len()).fold(0, |m, c| if row[c] > row[m] { c } else { m });
    let hull: Value = serde_json::from_slice(&fs::read(cohort.join("hull.json"))?)?;
    let chosen = |name: &str, arm: &str| {
        hull["choices"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["task"] == name && c["arm"] == arm)
            .map(|c| c["selected_budget"].clone())
    };
    let mut out = vec![];
    for name in protocol["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
    {
        let (q, labels) = table(name)?;
        if q != p {
            return Err("vocabulary differs from the reference".into());
        }
        let shift: Vec<_> = (0..p * p)
            .map(|cell| (labels[cell] + p - mul[cell]) % p)
            .collect();
        let separable = (0..p).all(|a| {
            (0..p).all(|b| (shift[b + p * a] + shift[0] + 2 * p - shift[p * a] - shift[b]) % p == 0)
        });
        for arm in protocol["arms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap())
        {
            for budget in protocol["budgets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_str().unwrap())
            {
                let ((rf, mf), (rm, mm)) =
                    (read(arm, name, budget)?, read(arm, reference, budget)?);
                let (mut deviation, mut moved) = (0_f64, 0);
                for (cell, &s) in shift.iter().enumerate() {
                    let (f, m) = (
                        &mf.probabilities[cell * p..][..p],
                        &mm.probabilities[cell * p..][..p],
                    );
                    for c in 0..p {
                        deviation = deviation.max((f[(c + s) % p] - m[c]).abs());
                    }
                    moved += usize::from(argmax(f) != (argmax(m) + s) % p);
                }
                let v = |r: &Value, k: &str| r["metrics"]["validation"][k].clone();
                out.push(json!({"task":name,"arm":arm,"budget":budget,"shift_is_u_plus_v":separable,
                    "max_probability_deviation":deviation,"cells_with_moved_argmax":moved,
                    "selected":[chosen(name, arm), chosen(reference, arm)],
                    "updates":[rf["updates"], rm["updates"]],"stop_reason":[rf["stop_reason"], rm["stop_reason"]],
                    "heldout_exact":[v(&rf, "exact"), v(&rm, "exact")],"heldout_examples":v(&rf, "examples"),
                    "train_loss":[rf["metrics"]["train"]["loss"], rm["metrics"]["train"]["loss"]]}));
            }
        }
    }
    Ok(json!({"source_fingerprint":fingerprint(),"reference":reference,"comparisons":out}))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["fingerprint"] => {
            println!("{}", fingerprint());
            Ok(())
        }
        ["run", arm, task, radius, out] => run(arm, task, radius, Path::new(out)),
        ["replay", out] => replay(Path::new(out)),
        ["select", cohort] => {
            let report=hull_selection(Path::new(cohort))?;
            fs::write(Path::new(cohort).join("hull.json"),serde_json::to_vec_pretty(&report)?)?;
            println!("{}",json!({"selected":true,"choices":report["choices"].as_array().unwrap().len()}));
            Ok(())
        },
        ["replay-hull", cohort] => {
            let expected: Value=serde_json::from_slice(&fs::read(Path::new(cohort).join("hull.json"))?)?;
            if expected!=hull_selection(Path::new(cohort))? {return Err("hull selection replay mismatch".into());}
            println!("{}",json!({"hull_replayed":true,"choices":expected["choices"].as_array().unwrap().len()}));
            Ok(())
        },
        ["compare", cohort] => {
            let report = compare(Path::new(cohort))?;
            fs::write(Path::new(cohort).join("compare.json"), serde_json::to_vec_pretty(&report)?)?;
            println!("{}", json!({"compared":report["comparisons"].as_array().unwrap().len()}));
            Ok(())
        }
        _ => Err(
            "usage: convex-stress-schatten-mixed fingerprint | run ARM TASK BUDGET OUT | replay OUT | select COHORT | replay-hull COHORT | compare COHORT"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SUITE: &str = include_str!("../../convex_stress/suite.json");

    /// For the tasks in both catalogs, the bridged split is the 35-task suite's, example by
    /// example: so the learner sees exactly what `convex_stress_schatten` gives it.
    #[test]
    fn bridged_tasks_equal_the_suite_tasks() -> Result<()> {
        let suite: Value = serde_json::from_str(SUITE)?;
        for name in [
            "arithmetic-mul",
            "arithmetic-affine-product",
            "arithmetic-div",
        ] {
            let row = suite["tasks"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["name"] == name)
                .unwrap();
            let (old, new) = (data::build(row)?, task(&selected(name)?)?);
            assert_eq!((old.vocab, &old.selection), (new.vocab, &new.selection));
            assert_eq!(
                old.splits.keys().collect::<Vec<_>>(),
                new.splits.keys().collect::<Vec<_>>()
            );
            for (split, xs) in &old.splits {
                let ys = &new.splits[split];
                assert_eq!(xs.len(), ys.len());
                for (x, y) in xs.iter().zip(ys) {
                    assert_eq!(
                        (&x.task, &x.tokens, &x.targets, &x.changes),
                        (&y.task, &y.tokens, &y.targets, &y.changes)
                    );
                    assert!(x.prompt.is_empty() && x.answer.is_empty() && x.limit == 0);
                    assert!(x.field.is_none() && x.row_index.is_none());
                }
            }
        }
        Ok(())
    }

    /// The premise of E6 in the data: every planned task has mul's training cells in mul's order,
    /// and its labels are mul's shifted by `u(a) + v(b)`.
    #[test]
    fn planned_tasks_are_label_shifts_of_mul_on_the_same_cells() -> Result<()> {
        let protocol: Value = serde_json::from_str(PROTOCOL)?;
        let mul = task(&selected("arithmetic-mul")?)?;
        let (p, labels) = table("arithmetic-mul")?;
        for name in protocol["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
        {
            let other = task(&selected(name)?)?;
            for split in ["train", "validation"] {
                let cells = |t: &data::Task| {
                    t.splits[split]
                        .iter()
                        .map(|e| e.tokens.clone())
                        .collect::<Vec<_>>()
                };
                assert_eq!(cells(&mul), cells(&other), "{name}/{split}");
            }
            let (_, f) = table(name)?;
            let s = |a: usize, b: usize| (f[b + p * a] + p - labels[b + p * a]) % p;
            for a in 0..p {
                for b in 0..p {
                    assert_eq!(
                        (s(a, b) + s(0, 0)) % p,
                        (s(a, 0) + s(0, b)) % p,
                        "{name} at ({a}, {b})"
                    );
                }
            }
        }
        Ok(())
    }
}
