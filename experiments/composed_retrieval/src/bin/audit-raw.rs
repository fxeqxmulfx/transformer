//! Reconcile independent raw decoding with the full 165-case catalog.
//! This command scores no model and consumes only the sealed comparison.
use baseline::{data::Task, hash};
use bilinear::guard;
use composed_retrieval::{
    Result,
    data::{self, CaseSpec, Protocol},
};
use learned_retrieval::cohort::{raw_file, read};
use learned_retrieval_transfer::audit::decode_split;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn check_case(
    dir: &Path,
    id: &str,
    p: &Protocol,
    task: &Task,
    spec: &CaseSpec,
) -> Result<(Value, usize)> {
    let report = read(&dir.join("scores/report.json"))?;
    if report["case"] != id
        || report["selection"] != spec.selection
        || report["target"] != spec.target
        || report["protocol_sha256"] != hash(data::PROTOCOL.as_bytes())
        || report["metric"] != spec.metric
    {
        return Err("score report identity differs".into());
    }
    let supervised: BTreeSet<_> = task.splits["train"]
        .iter()
        .flat_map(|row| row.targets.iter().copied().filter(|&t| t >= 0))
        .collect();
    let mut cells = BTreeMap::new();
    let mut files = 0;
    for mode in &p.modes {
        for seed in &p.seeds {
            for phase in ["initial", "final"] {
                let key = format!("{mode}-{seed}-{phase}");
                let entry = &report["cells"][&key];
                let mut strict = true;
                if entry["splits"].as_object().ok_or("missing splits")?.len() != task.splits.len() {
                    return Err("unexpected splits".into());
                }
                for (name, rows) in &task.splits {
                    let r = &entry["splits"][name];
                    let metrics = &r["metrics"];
                    let raw = fs::read(raw_file(&dir.join("scores"), &key, name))?;
                    let groups = decode_split(
                        &raw,
                        rows,
                        task.vocab,
                        &supervised,
                        &json!({"raw_logits_sha256":r["raw_sha256"],"metrics":metrics}),
                    )?;
                    let exposure = &r["exposure"]["correct_total_by_exposure"];
                    for (i, field) in ["correct", "targets"].iter().enumerate() {
                        let count = |group: &str| {
                            exposure[group][i].as_u64().ok_or("missing exposure count")
                        };
                        if groups["never_supervised_answer"][field]
                            != count("input_only")? + count("absent")?
                            || groups["supervised_answer"][field] != count("seen_positive")?
                        {
                            return Err("answer-exposure breakdown differs".into());
                        }
                    }
                    let count = |key: &str| metrics[key].as_u64().ok_or("missing metric count");
                    let raw99 = 100 * count("correct")? >= 99 * count("targets")?;
                    let exact99 = 100 * count("exact")? >= 99 * count("examples")?;
                    if r["raw99"] != raw99 || r["exact99"] != exact99 {
                        return Err("raw or exact target differs".into());
                    }
                    if !name.starts_with("train") {
                        strict &= raw99 && exact99;
                    }
                    files += 1;
                }
                let original = entry["splits"][&spec.selection]["metrics"][&spec.metric]
                    .as_f64()
                    .ok_or("primary metric")?
                    >= spec.target;
                if original != entry["original_success"] || strict != entry["strict_all_heldout"] {
                    return Err("case criteria differ".into());
                }
                if phase == "final" {
                    cells.insert(
                        format!("{mode}-{seed}"),
                        json!({"original":original,"strict":strict}),
                    );
                }
            }
        }
    }
    if report["cells"]
        .as_object()
        .ok_or("missing score cells")?
        .len()
        != cells.len() * 2
    {
        return Err("unexpected score cells".into());
    }
    Ok((
        json!({"original":cells.values().all(|c|c["original"]==true),
        "strict":cells.values().all(|c|c["strict"]==true),"endpoints":cells}),
        files,
    ))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [workspace, storage, output] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE FROZEN_OUTPUT".into());
    };
    let (workspace, storage, output) =
        (Path::new(workspace), Path::new(storage), Path::new(output));
    guard::admit(1 << 30, workspace, storage)?;
    let text = catalog::checked_text("catalog", &workspace.join("stress_tests_for_gpt.md"))
        .map_err(|e| e.to_string())?;
    let inventory = catalog::catalog_rows(&text).map_err(|e| e.to_string())?;
    let definition: Value = serde_json::from_str(catalog::PROTOCOL)?;
    let controls = definition["controls"]
        .as_array()
        .ok_or("catalog controls")?;
    let p = data::protocol()?;
    let producer = fs::read_to_string(output.join("producer.commit"))?;
    let mut files = 0;
    let mut cases = BTreeMap::new();
    for (id, row) in inventory {
        let result = if let Some(spec) = p.cases.get(&id) {
            let (task, _) = data::original(&p, &id)?;
            let mut prior = None;
            for arm in ["original", "repeat"] {
                let root = output.join(arm).join("run");
                let seal = read(&root.join("sealed/manifest.json"))?;
                let identity = &seal["identity"];
                if identity["producer"] != producer.trim()
                    || identity["protocol_sha256"] != hash(data::PROTOCOL.as_bytes())
                    || identity["binary_sha256"]
                        != hash(&fs::read(output.join(arm).join("composed-retrieval"))?)
                {
                    return Err("unrecognized freeze".into());
                }
                for mode in &p.modes {
                    for seed in &p.seeds {
                        let cell = format!("{id}/{mode}-{seed}");
                        let receipt = fs::read(root.join(&cell).join("receipt.json"))?;
                        let r: Value = serde_json::from_slice(&receipt)?;
                        if seal["cells"][&cell] != hash(&receipt)
                            || r["identity"] != *identity
                            || r["endpoint_sha256"]
                                != hash(&fs::read(root.join(&cell).join("model.json"))?)
                        {
                            return Err("endpoint differs from seal".into());
                        }
                    }
                }
                let (out, n) = check_case(&root.join(&id), &id, &p, &task, spec)?;
                if prior.as_ref().is_some_and(|a| a != &out) {
                    return Err("independent arms disagree".into());
                }
                prior = Some(out);
                files += n;
            }
            prior
        } else {
            None
        };
        cases.insert(id.clone(),json!({"catalog_control":controls.iter().any(|v|v==&id),"historical_gpt_status":row["history"],"current":result}));
    }
    let measured = cases.values().filter(|x| !x["current"].is_null()).count();
    if measured != p.cases.len() {
        return Err("frozen case absent from original catalog".into());
    }
    let summary = json!({"catalog":cases.len(),"measured":measured,"unmeasured":cases.len()-measured,"raw_files":files,
        "original_pass_all":cases.values().filter(|x|x["current"]["original"]==true).count(),
        "strict_pass_all":cases.values().filter(|x|x["current"]["strict"]==true).count()});
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"producer":producer.trim(),"summary":summary,"cases":cases}),
    )?;
    println!();
    Ok(())
}
