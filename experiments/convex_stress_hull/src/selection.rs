use crate::replay;
use convex_stress_hull::{
    Result, all_metrics, ideal_ce, meta, passes, scale, spectral, teacher_ce,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

pub fn select(cohort: &Path) -> Result<Value> {
    let p = meta::protocol()?;
    let mut choices = vec![];
    for row in meta::rows()? {
        let name = row["name"].as_str().unwrap();
        for av in p["arms"].as_array().unwrap() {
            let arm = av.as_str().unwrap();
            let mut candidates = vec![];
            let mut retained = vec![];
            for cv in p["candidates"].as_array().unwrap() {
                let candidate = cv.as_str().unwrap();
                let dir = cohort.join(format!("{arm}--{name}--{candidate}"));
                let report: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
                let (_, task) = replay::identity(&report)?;
                if report["arm"] != arm
                    || report["task"] != name
                    || report["candidate"] != candidate
                {
                    return Err("candidate directory identity mismatch".into());
                }
                let mut entry = json!({"candidate":candidate,"status":report["status"],"updates":report["updates"],
                    "elapsed_seconds":report["elapsed_seconds"],"examples_presented":report["examples_presented"],
                    "model_sha256":report["model_sha256"]});
                match report["status"].as_str() {
                    Some("complete") => {
                        let model = replay::load(&dir, &report)?;
                        model.validate(
                            &task,
                            candidate,
                            p["frequency_workers"].as_u64().unwrap() as usize,
                        )?;
                        let ce = teacher_ce(&model, &task.splits["train"])?;
                        entry["training_ce"] = ce.into();
                        entry["ideal_ce_excess"] = (ce - ideal_ce(task.vocab, scale(&task))).into();
                        candidates.push((
                            candidate.to_owned(),
                            model,
                            task,
                            ce,
                            report["model_sha256"].clone(),
                        ));
                    }
                    Some("ineligible" | "fit_failure") => {
                        replay::replay(&dir)?;
                    }
                    _ => return Err("nonterminal candidate".into()),
                }
                retained.push(entry);
            }
            if candidates.is_empty() {
                choices.push(json!({"task":name,"tier":row["tier"],"arm":arm,"status":"no_model", "candidates":retained}));
                continue;
            }
            let ideal = ideal_ce(candidates[0].2.vocab, scale(&candidates[0].2));
            let tol = meta::number(&p, "ideal_ce_tolerance")?;
            let ce: Vec<_> = candidates.iter().map(|c| c.3).collect();
            let k = spectral::choose_by_train(&ce, ideal, tol)?;
            let (candidate, model, task, ce, hash) = &candidates[k];
            // No held-out metric is evaluated or read above this decision.
            let metrics = all_metrics(model, task)?;
            let all_pass = metrics.as_object().unwrap().values().all(passes);
            choices.push(json!({"task":name,"tier":row["tier"],"arm":arm,"status":"selected",
                "selected_candidate":candidate,"model_path":format!("{arm}--{name}--{candidate}/model.json"),
                "model_sha256":hash,"training_ce":ce,"ideal_ce_excess":ce-ideal,"ideal_ce_certificate":ce-ideal<=tol,
                "selection":task.selection,"metrics":metrics,"primary_pass":passes(&metrics[&task.selection]),
                "all_splits_pass":all_pass,"candidates":retained,"historical_transformer_seed1":row["historical_seed1"]}));
        }
    }
    let by_arm:Vec<_>=p["arms"].as_array().unwrap().iter().map(|arm|{
        let rows:Vec<_>=choices.iter().filter(|r|r["arm"]==*arm).collect();
        json!({"arm":arm,"tasks":rows.len(),"no_model":rows.iter().filter(|r|r["status"]=="no_model").count(),
            "primary_passes":rows.iter().filter(|r|r["primary_pass"]==true).count(),
            "all_splits_passes":rows.iter().filter(|r|r["all_splits_pass"]==true).count()})
    }).collect();
    Ok(
        json!({"source_fingerprint":meta::fingerprint(),"protocol":p,"choices":choices,"by_arm":by_arm}),
    )
}
