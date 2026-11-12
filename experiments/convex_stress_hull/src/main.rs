use convex_stress_hull::{
    Result, all_metrics, data, hash, ideal_ce, meta, pair_eligibility, scale, teacher_ce,
    training::{self, Monitor},
};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
mod replay;
mod selection;

fn run(arm: &str, name: &str, candidate: &str, out: &Path) -> Result<()> {
    let p = meta::protocol()?;
    if !meta::allowed(&p, "arms", arm) || !meta::allowed(&p, "candidates", candidate) {
        return Err("unplanned candidate".into());
    }
    let row = meta::selected(name)?;
    let task = data::build(&row)?;
    let eligibility = pair_eligibility(task.vocab, &task.splits["train"]);
    fs::create_dir(out)?;
    let started = Instant::now();
    let mut report = json!({"task":name,"tier":row["tier"],"arm":arm,"candidate":candidate,
        "source_fingerprint":meta::fingerprint(),"protocol":p,"data_fingerprints":row["data_fingerprints"],
        "seed":p["seed"],"scale":scale(&task),"vocab":task.vocab,"selection":task.selection,
        "pair_ineligible_reason":eligibility,"ideal_ce":ideal_ce(task.vocab,scale(&task))});
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if candidate != "counts" && eligibility.is_some() {
        report["status"] = "ineligible".into();
        report["stop_reason"] = "unsupported_input_layout".into();
        report["updates"] = 0.into();
        report["model_sha256"] = Value::Null;
        report["metrics"] = Value::Null;
    } else {
        let mut monitor = Monitor {
            task: &task,
            file: fs::File::create(out.join("trace.jsonl"))?,
            started,
            first99: None,
        };
        let outcome = if candidate == "counts" {
            training::train_counts(arm, &row, &p, &mut monitor)?
        } else {
            training::train_moments(arm, candidate, &row, &p, &mut monitor)?
        };
        report
            .as_object_mut()
            .unwrap()
            .extend(outcome.report.as_object().unwrap().clone());
        report["first_primary_99_update"] = json!(monitor.first99);
        if let Some(model) = outcome.model {
            report["feasibility"] = model.validate(
                &task,
                candidate,
                p["frequency_workers"].as_u64().unwrap() as usize,
            )?;
            let ce = teacher_ce(&model, &task.splits["train"])?;
            report["training_ce"] = ce.into();
            report["ideal_ce_excess"] = (ce - ideal_ce(task.vocab, scale(&task))).into();
            report["metrics"] = all_metrics(&model, &task)?;
            let bytes = serde_json::to_vec(&model)?;
            report["model_sha256"] = hash(&bytes).into();
            fs::write(out.join("model.json"), bytes)?;
        } else {
            report["model_sha256"] = Value::Null;
            report["metrics"] = Value::Null;
        }
        if let Some(state) = outcome.state {
            let bytes = serde_json::to_vec(&state)?;
            report["diagnostic_state_sha256"] = hash(&bytes).into();
            fs::write(out.join("diagnostic_state.json"), bytes)?;
        }
    }
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&report)?)?;
    println!(
        "complete status={} primary={}",
        report["status"], report["metrics"][&task.selection]["sequence_accuracy"]
    );
    Ok(())
}

fn prepare() -> Result<()> {
    let p = meta::protocol()?;
    let mut tasks = vec![];
    for row in meta::rows()? {
        let task = data::build(&row)?;
        let reason = pair_eligibility(task.vocab, &task.splits["train"]);
        tasks.push(json!({"task":row["name"],"tier":row["tier"],"vocab":task.vocab,
            "train_examples":task.splits["train"].len(),"scale":scale(&task),"pair_ineligible_reason":reason,
            "data_fingerprints":row["data_fingerprints"]}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"source_fingerprint":meta::fingerprint(),"protocol":p,
        "tasks":tasks,"planned_candidate_records":315,"planned_choices":105})
        )?
    );
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"]=>{println!("{}",meta::fingerprint());Ok(())},
        ["prepare"]=>prepare(),
        ["run",arm,name,candidate,out]=>run(arm,name,candidate,Path::new(out)),
        ["replay",out]=>{println!("{}",replay::replay(Path::new(out))?);Ok(())},
        ["select",cohort]=>{
            let v=selection::select(Path::new(cohort))?;
            fs::write(Path::new(cohort).join("selected.json"),serde_json::to_vec_pretty(&v)?)?;
            println!("{}",json!({"choices":v["choices"].as_array().unwrap().len()}));Ok(())
        },
        ["replay-selection",cohort]=>{
            let expected:Value=serde_json::from_slice(&fs::read(Path::new(cohort).join("selected.json"))?)?;
            if expected!=selection::select(Path::new(cohort))?{return Err("selection replay mismatch".into());}
            println!("{}",json!({"selection_replayed":true,"choices":105}));Ok(())
        },
        ["compare-original",old,new]=>replay::compare_original(Path::new(old),Path::new(new)),
        _=>Err("usage: convex-stress-hull fingerprint | prepare | run ARM TASK CANDIDATE OUT | replay OUT | select COHORT | replay-selection COHORT | compare-original OLD NEW".into()),
    }
}
