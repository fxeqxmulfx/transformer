use convex_stress_sharpen::{
    Model, Result, choose_by_train, data, hash, ideal_loss, metrics, polish, prepare_model,
    projection::Projector, training_ce,
};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};

const SUITE: &str = include_str!("../../convex_stress/suite.json");
const PROTOCOL: &str = include_str!("../protocol.json");
const PARENT_PROTOCOL: &str = include_str!("../../convex_stress_sharing/protocol.json");

fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("main.rs"),
        include_bytes!("lib.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../../convex_stress_sharing/src/lib.rs"),
        include_bytes!("../../convex_stress_sharing/src/grouping.rs"),
        include_bytes!("../../convex_stress_budgets/src/lib.rs"),
        include_bytes!("../../convex_stress_schatten/src/lib.rs"),
        include_bytes!("../../convex_stress_schatten/src/projection.rs"),
        include_bytes!("../../convex_stress_moments/src/lib.rs"),
        include_bytes!("../../convex_stress_moments/src/projection.rs"),
        include_bytes!("../../convex_stress_moments/src/fft.rs"),
        include_bytes!("../../convex_stress_spectral/src/lib.rs"),
        include_bytes!("../../convex_stress_spectral/src/linalg.rs"),
        include_bytes!("../../convex_stress_minibatch/src/lib.rs"),
        include_bytes!("../../convex_stress_minibatch/src/metrics.rs"),
        include_bytes!("../../convex_stress/src/channel.rs"),
        include_bytes!("../../convex_stress/src/features.rs"),
        include_bytes!("../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../convex_stress/src/vendor/blake.rs"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
        PARENT_PROTOCOL.as_bytes(),
    ];
    let mut bytes = vec![];
    for source in sources {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}

fn number(value: &Value, key: &str) -> Result<f64> {
    value[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("missing finite {key}").into())
}
fn allowed(protocol: &Value, key: &str, name: &str) -> bool {
    protocol[key].as_array().unwrap().iter().any(|x| x == name)
}
fn suite_row(name: &str) -> Result<Value> {
    serde_json::from_str::<Value>(SUITE)?["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == name)
        .cloned()
        .ok_or_else(|| "unknown task".into())
}
fn metrics_all(model: &Model, task: &data::Task) -> Result<Value> {
    let mut result = serde_json::Map::new();
    for (split, examples) in &task.splits {
        result.insert(split.clone(), metrics::evaluate(model, examples)?);
    }
    Ok(result.into())
}
fn fit_defect(model: &Model, samples: &[convex_stress_sharpen::Sample]) -> f64 {
    let n = model.table.vocab;
    samples
        .iter()
        .flat_map(|s| {
            model.table.probabilities[s.cell * n..(s.cell + 1) * n]
                .iter()
                .enumerate()
                .map(move |(c, p)| (p - f64::from(c == s.label)).abs())
        })
        .fold(0., f64::max)
}

fn prepare(parents: &Path) -> Result<Value> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let mut inventory = vec![];
    for task in protocol["tasks"].as_array().unwrap() {
        for arm in protocol["arms"].as_array().unwrap() {
            for budget in protocol["budgets"].as_array().unwrap() {
                let cell = format!(
                    "{}--{}--{}",
                    arm.as_str().unwrap(),
                    task.as_str().unwrap(),
                    budget.as_str().unwrap()
                );
                let dir = parents.join(&cell);
                let report_bytes = fs::read(dir.join("result.json"))?;
                let report: Value = serde_json::from_slice(&report_bytes)?;
                let bytes = fs::read(dir.join("model.json"))?;
                if report["source_fingerprint"] != protocol["parent_source_fingerprint"]
                    || report["protocol"] != serde_json::from_str::<Value>(PARENT_PROTOCOL)?
                    || report["status"] != "complete"
                    || report["task"] != *task
                    || report["arm"] != *arm
                    || report["budget_name"] != *budget
                    || report["model_sha256"] != hash(&bytes)
                    || report["data_fingerprints"]
                        != suite_row(task.as_str().unwrap())?["data_fingerprints"]
                {
                    return Err("invalid frozen parent inventory".into());
                }
                let model: Model = serde_json::from_slice(&bytes)?;
                model.validate_shape()?;
                inventory.push(json!({"cell":cell,"model_sha256":hash(&bytes),"result_sha256":hash(&report_bytes)}));
            }
        }
    }
    Ok(json!({"source_fingerprint":fingerprint(),"protocol":protocol,"parents":inventory}))
}

struct Output {
    selected: Model,
    proposed: Model,
    selected_report: Value,
    proposed_report: Value,
}

fn transform(parents: &Path, arm: &str, name: &str, budget: &str) -> Result<Output> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    if !allowed(&protocol, "tasks", name)
        || !allowed(&protocol, "arms", arm)
        || !allowed(&protocol, "budgets", budget)
    {
        return Err("unplanned endpoint".into());
    }
    let parent_dir = parents.join(format!("{arm}--{name}--{budget}"));
    let report_bytes = fs::read(parent_dir.join("result.json"))?;
    let parent: Value = serde_json::from_slice(&report_bytes)?;
    let parent_bytes = fs::read(parent_dir.join("model.json"))?;
    let parent_model: Model = serde_json::from_slice(&parent_bytes)?;
    let row = suite_row(name)?;
    if parent["source_fingerprint"] != protocol["parent_source_fingerprint"]
        || parent["protocol"] != serde_json::from_str::<Value>(PARENT_PROTOCOL)?
        || parent["status"] != "complete"
        || parent["task"] != name
        || parent["arm"] != arm
        || parent["budget_name"] != budget
        || parent["model_sha256"] != hash(&parent_bytes)
        || parent["data_fingerprints"] != row["data_fingerprints"]
    {
        return Err("invalid frozen parent identity".into());
    }
    let task = data::build(&row)?;
    if task.splits.values().flatten().any(|e| e.field.is_some()) {
        return Err("this pilot requires direct class decoding".into());
    }
    let (initial, samples) = prepare_model(
        &task,
        budget,
        protocol["minimum_overlap"].as_u64().unwrap() as usize,
    )?;
    if parent_model.groups != initial.groups
        || parent_model.table.vocab != task.vocab
        || (parent_model.table.radius, parent_model.table.nuclear_budget)
            != (initial.table.radius, initial.table.nuclear_budget)
    {
        return Err("parent differs from training-only group construction".into());
    }
    let outcome = polish(
        &parent_model,
        &samples,
        protocol["frequency_workers"].as_u64().unwrap() as usize,
        number(&protocol, "ideal_ce_tolerance")?,
        number(&protocol, "improvement_tolerance")?,
    )?;
    if (outcome.certificate.original_training_ce - number(&parent["metrics"]["train"], "loss")?)
        .abs()
        > 1e-12
    {
        return Err("parent training CE mismatch".into());
    }
    let mut report = parent.clone();
    let object = report.as_object_mut().unwrap();
    for key in [
        "diagnostic_state_sha256",
        "last_split_residual",
        "full_operator_residual",
        "uniform_mixture_alpha",
        "minimum_observed_alpha",
        "shadow_max_operator_norm",
        "shadow_max_nuclear_norm",
        "updates",
        "epochs",
        "examples_presented",
        "elapsed_seconds",
        "grouping_seconds",
        "total_seconds_including_data_and_grouping",
        "first_primary_99_update",
    ] {
        object.remove(key);
    }
    report["source_fingerprint"] = fingerprint().into();
    report["protocol"] = protocol;
    report["stage"] = "endpoint_sharpening".into();
    report["parent_cell"] = format!("{arm}--{name}--{budget}").into();
    report["parent_model_sha256"] = hash(&parent_bytes).into();
    report["parent_result_sha256"] = hash(&report_bytes).into();
    report["parent_training"] = json!({"updates":parent["updates"],"epochs":parent["epochs"],
        "examples_presented":parent["examples_presented"],"training_gradient_evaluations":parent["training_gradient_evaluations"],
        "elapsed_seconds":parent["elapsed_seconds"],"grouping_seconds":parent["grouping_seconds"],
        "total_seconds_including_data_and_grouping":parent["total_seconds_including_data_and_grouping"]});
    report["training_gradient_evaluations"] = 0.into();
    report["endpoint_training_ce_examples_evaluated"] = (2 * samples.len()).into();
    report["stop_reason"] = "endpoint_check".into();
    report["polish"] = serde_json::to_value(&outcome.certificate)?;
    let mut proposal_report = report.clone();
    for (model, result, norms, kind) in [
        (
            &outcome.selected,
            &mut report,
            &outcome.certificate.selected_norms,
            "selected",
        ),
        (
            &outcome.proposed,
            &mut proposal_report,
            &outcome.certificate.proposed_norms,
            "proposal",
        ),
    ] {
        result["endpoint_kind"] = kind.into();
        result["model_sha256"] = hash(&serde_json::to_vec(model)?).into();
        result["moment_norms"] = serde_json::to_value(norms)?;
        result["fit_defect"] = fit_defect(model, &samples).into();
        // Evaluate held-out labels only after the endpoint has been selected.
        result["metrics"] = metrics_all(model, &task)?;
        result["ideal_ce_excess"] =
            (number(&result["metrics"]["train"], "loss")? - ideal_loss(task.vocab)).into();
        for split in task.splits.keys() {
            for metric in [
                "correct",
                "exact",
                "targets",
                "examples",
                "sequence_accuracy",
                "balanced_accuracy",
            ] {
                if result["metrics"][split][metric] != parent["metrics"][split][metric] {
                    return Err("decoded endpoint changed a split correctness metric".into());
                }
            }
        }
    }
    Ok(Output {
        selected: outcome.selected,
        proposed: outcome.proposed,
        selected_report: report,
        proposed_report: proposal_report,
    })
}

fn run(parents: &Path, arm: &str, name: &str, budget: &str, out: &Path) -> Result<()> {
    let started = Instant::now();
    let mut output = transform(parents, arm, name, budget)?;
    output.selected_report["postprocessing_seconds"] = started.elapsed().as_secs_f64().into();
    output.proposed_report["postprocessing_seconds"] =
        output.selected_report["postprocessing_seconds"].clone();
    fs::create_dir(out)?;
    fs::create_dir(out.join("proposal"))?;
    for (dir, model, report) in [
        (out.to_path_buf(), output.selected, output.selected_report),
        (
            out.join("proposal"),
            output.proposed,
            output.proposed_report,
        ),
    ] {
        fs::write(dir.join("model.json"), serde_json::to_vec(&model)?)?;
        fs::write(dir.join("result.json"), serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(())
}

fn replay(parents: &Path, out: &Path) -> Result<()> {
    let reported: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    let arm = reported["arm"].as_str().ok_or("no arm")?;
    let task = reported["task"].as_str().ok_or("no task")?;
    let budget = reported["budget_name"].as_str().ok_or("no budget")?;
    let expected = transform(parents, arm, task, budget)?;
    for (dir, model, mut report) in [
        (
            out.to_path_buf(),
            expected.selected,
            expected.selected_report,
        ),
        (
            out.join("proposal"),
            expected.proposed,
            expected.proposed_report,
        ),
    ] {
        let saved: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
        let bytes = fs::read(dir.join("model.json"))?;
        report["postprocessing_seconds"] = number(&saved, "postprocessing_seconds")?.into();
        if report != saved || bytes != serde_json::to_vec(&model)? {
            return Err("endpoint replay differs".into());
        }
    }
    println!(
        "{}",
        json!({"replayed":true,"task":task,"arm":arm,"budget":budget})
    );
    Ok(())
}

fn selection(cohort: &Path) -> Result<Value> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let mut choices = vec![];
    for task_value in protocol["tasks"].as_array().unwrap() {
        let name = task_value.as_str().unwrap();
        let task = data::build(&suite_row(name)?)?;
        for arm in protocol["arms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
        {
            let mut candidates = vec![];
            for budget in protocol["budgets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
            {
                let dir = cohort.join(format!("{arm}--{name}--{budget}"));
                let report: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
                let bytes = fs::read(dir.join("model.json"))?;
                let model: Model = serde_json::from_slice(&bytes)?;
                let (initial, samples) = prepare_model(
                    &task,
                    budget,
                    protocol["minimum_overlap"].as_u64().unwrap() as usize,
                )?;
                if report["source_fingerprint"] != fingerprint()
                    || report["protocol"] != protocol
                    || report["status"] != "complete"
                    || report["endpoint_kind"] != "selected"
                    || report["task"] != name
                    || report["arm"] != arm
                    || report["budget_name"] != budget
                    || report["model_sha256"] != hash(&bytes)
                    || report["data_fingerprints"] != suite_row(name)?["data_fingerprints"]
                    || model.table.vocab != task.vocab
                    || model.groups != initial.groups
                    || (model.table.radius, model.table.nuclear_budget)
                        != (initial.table.radius, initial.table.nuclear_budget)
                {
                    return Err("invalid endpoint selection identity".into());
                }
                model.validate_shape()?;
                model.table.validate(&mut Projector::new(
                    task.vocab,
                    protocol["frequency_workers"].as_u64().unwrap() as usize,
                )?)?;
                let ce = training_ce(&model, &samples)?;
                candidates.push((budget.to_owned(), model, report, ce));
            }
            let ideal = ideal_loss(task.vocab);
            let index = choose_by_train(
                &candidates.iter().map(|x| x.3).collect::<Vec<_>>(),
                ideal,
                number(&protocol, "ideal_ce_tolerance")?,
            )?;
            let (budget, model, report, ce) = &candidates[index];
            let comparison: Vec<_> = candidates.iter().map(|c|json!({"budget":c.0,"training_ce":c.3,
                "polish":c.2["polish"]["selected"],"postprocessing_seconds":c.2["postprocessing_seconds"],
                "parent_training":c.2["parent_training"],"model_sha256":c.2["model_sha256"]})).collect();
            choices.push(json!({"task":name,"arm":arm,"selected_budget":budget,"training_ce":ce,
                "ideal_ce_excess":ce-ideal,"ideal_ce_certificate":ce-ideal<=number(&protocol,"ideal_ce_tolerance")?,
                "model_sha256":report["model_sha256"],"selected_model":format!("{arm}--{name}--{budget}/model.json"),
                "selection":task.selection,"metrics":metrics_all(model,&task)?,"candidates":comparison}));
        }
    }
    Ok(json!({"source_fingerprint":fingerprint(),"protocol":protocol,"choices":choices}))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"] => { println!("{}",fingerprint()); Ok(()) },
        ["prepare", parents] => {println!("{}",serde_json::to_string_pretty(&prepare(Path::new(parents))?)?); Ok(())},
        ["run", parents, arm, task, budget, out] => run(Path::new(parents),arm,task,budget,Path::new(out)),
        ["replay", parents, out] => replay(Path::new(parents),Path::new(out)),
        ["select", cohort] => { let result=selection(Path::new(cohort))?;
            fs::write(Path::new(cohort).join("hull.json"),serde_json::to_vec_pretty(&result)?)?;
            println!("{}",json!({"choices":result["choices"].as_array().unwrap().len()})); Ok(()) },
        ["replay-hull", cohort] => { let saved:Value=serde_json::from_slice(&fs::read(Path::new(cohort).join("hull.json"))?)?;
            if saved != selection(Path::new(cohort))? {return Err("selection replay differs".into());}
            println!("{}",json!({"hull_replayed":true,"choices":saved["choices"].as_array().unwrap().len()})); Ok(()) },
        _ => Err("usage: convex-stress-sharpen fingerprint | prepare PARENTS | run PARENTS ARM TASK BUDGET OUT | replay PARENTS OUT | select COHORT | replay-hull COHORT".into()),
    }
}
