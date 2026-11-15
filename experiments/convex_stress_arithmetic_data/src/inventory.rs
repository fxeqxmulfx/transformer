//! Bind complete historical inputs, then reconstruct every original split.
//! No training, model selection or accuracy credit occurs in this executable.
use crate::{CASES, DEFINITIONS, ORIGINAL35, PROTOCOL, Result, dataset, hash, reference};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
    process::Command,
};

pub fn cases() -> Result<Vec<Value>> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    if hash(CASES.as_bytes()) != p["cases_sha256"]
        || hash(DEFINITIONS.as_bytes()) != p["definitions_sha256"]
        || hash(ORIGINAL35.as_bytes()) != p["original35_sha256"]
    {
        return Err("immutable source extract changed".into());
    }
    let rows: Vec<Value> = serde_json::from_str(CASES)?;
    let names: BTreeSet<_> = rows.iter().map(|r| r["case"]["name"].as_str()).collect();
    if rows.len() != 54 || names.len() != 54 || names.contains(&None) {
        return Err("missing or repeated source case".into());
    }
    Ok(rows)
}
fn read(path: &Path) -> Result<Value> {
    if fs::metadata(path)?.len() > 64 * 1024 * 1024 {
        return Err("input file too large".into());
    }
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}
fn digest(path: &Path) -> Result<String> {
    if fs::metadata(path)?.len() > 64 * 1024 * 1024 {
        return Err("input file too large".into());
    }
    Ok(hash(&fs::read(path)?))
}
fn write(path: &Path, value: &Value) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
pub fn check_inputs(root: &Path) -> Result<()> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    for (name, expected) in p["input_sha256"].as_object().ok_or("source hashes")? {
        if digest(&root.join(name))? != *expected {
            return Err(format!("source hash differs: {name}").into());
        }
    }
    let plan = read(&root.join("plan.json"))?;
    let history = read(&root.join("arithmetic_seed1.json"))?;
    let definitions = read(&root.join("grokking_task_inventory.json"))?;
    if definitions["arithmetic_function_catalog"] != serde_json::from_str::<Value>(DEFINITIONS)? {
        return Err("definition extract differs from whole pinned source".into());
    }
    let mut expected = vec![];
    for case in plan["cases"]
        .as_array()
        .ok_or("plan cases")?
        .iter()
        .filter(|c| c["kind"] == "arithmetic")
    {
        let records: Vec<_> = history["runs"]
            .as_array()
            .ok_or("historical runs")?
            .iter()
            .filter(|r| r["case"] == case["name"])
            .collect();
        if records.len() != 1 {
            return Err("missing or duplicate historical case".into());
        }
        expected.push(json!({"case":case,"history":records[0]}));
    }
    if expected != cases()? {
        return Err("case extract differs from whole sources".into());
    }
    Ok(())
}
pub fn generate(root: &Path, producer: &str, binary: &str, create: bool) -> Result<Value> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let previous: Value = serde_json::from_str(ORIGINAL35)?;
    let old: BTreeSet<_> = previous["tasks"]
        .as_array()
        .ok_or("existing35")?
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    if create {
        fs::create_dir(root.join("data"))?;
    }
    let mut records = vec![];
    let (mut supported, mut existing, mut supported_additional) = (0, 0, 0);
    for row in cases()? {
        let name = row["case"]["name"].as_str().ok_or("case name")?;
        let original = old.contains(name);
        existing += usize::from(original);
        let mut record = json!({"case":name,"source_case":row["case"],"historical_gpt":row["history"],
            "existing35":original,"additional_case":!original,"model_result":null});
        if let Some(d) = dataset::build(&row)? {
            supported += 1;
            supported_additional += usize::from(!original);
            let file = format!("data/{name}.json");
            let value = serde_json::to_value(&d)?;
            if create {
                write(&root.join(&file), &value)?;
            } else {
                let mut expected = serde_json::to_vec(&value)?;
                expected.push(b'\n');
                if digest(&root.join(&file))? != hash(&expected) {
                    return Err(format!("raw source reconstruction differs: {file}").into());
                }
            }
            record["data"] = json!({"file":file,"sha256":digest(&root.join(&file))?,
                "train":{"rows":d.training.rows.len(),"fingerprint":reference::fingerprint(&d.training.rows)},
                "validation":{"rows":d.validation.len(),"fingerprint":reference::fingerprint(&d.validation)},
                "vocabulary":d.training.vocabulary,"historical_hashes_match":true});
            record["refusal"] = Value::Null;
        } else {
            record["refusal"] = p["refusal_reason"].clone();
        }
        records.push(record);
    }
    if supported != 48 || existing != 22 || supported_additional != 26 {
        return Err("complete cohort support/accounting changed".into());
    }
    Ok(
        json!({"producer_commit":producer,"binary_sha256":binary,"protocol":p,
        "cases":records,"attempted_cases":54,"supported_cases":supported,"refused_cases":54-supported,
        "existing_cases":existing,"additional_cases":54-existing,"supported_additional_cases":supported_additional,
        "new_model_evaluations":0,"new_accuracy_claims":0}),
    )
}
fn manifest(root: &Path) -> Result<Value> {
    let mut entries = BTreeMap::new();
    for dir in ["", "data", "inputs"] {
        for item in fs::read_dir(root.join(dir))? {
            let path = item?.path();
            if path.is_file() && path.file_name().unwrap() != "manifest.json" {
                let name = path
                    .strip_prefix(root)?
                    .to_str()
                    .ok_or("file path")?
                    .to_string();
                entries.insert(name, digest(&path)?);
            }
        }
    }
    Ok(serde_json::to_value(entries)?)
}
fn copy_inputs(input: &Path, output: &Path) -> Result<()> {
    check_inputs(input)?;
    fs::create_dir(output.join("inputs"))?;
    for name in [
        "plan.json",
        "arithmetic_seed1.json",
        "grokking_task_inventory.json",
    ] {
        fs::copy(input.join(name), output.join("inputs").join(name))?;
    }
    check_inputs(&output.join("inputs"))
}
pub fn prepare(input: &Path, output: &Path, producer: &str) -> Result<()> {
    let head = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    if !head.status.success()
        || !status.status.success()
        || !status.stdout.is_empty()
        || String::from_utf8(head.stdout)?.trim() != producer
    {
        return Err("clean committed source required".into());
    }
    check_inputs(input)?;
    fs::create_dir(output)?;
    copy_inputs(input, output)?;
    let inventory = generate(output, producer, &digest(&std::env::current_exe()?)?, true)?;
    write(&output.join("inventory.json"), &inventory)?;
    write(&output.join("manifest.json"), &manifest(output)?)
}
pub fn repeat(original: &Path, output: &Path) -> Result<()> {
    if manifest(original)? != read(&original.join("manifest.json"))? {
        return Err("original manifest changed".into());
    }
    let saved = read(&original.join("inventory.json"))?;
    let binary = digest(&std::env::current_exe()?)?;
    if saved["binary_sha256"] != binary
        || saved["protocol"] != serde_json::from_str::<Value>(PROTOCOL)?
    {
        return Err("repeat executable or protocol differs".into());
    }
    fs::create_dir(output)?;
    copy_inputs(&original.join("inputs"), output)?;
    let fresh = generate(
        output,
        saved["producer_commit"].as_str().ok_or("producer")?,
        &binary,
        true,
    )?;
    if fresh != saved {
        return Err("independent complete inventory differs".into());
    }
    write(&output.join("inventory.json"), &fresh)?;
    let files = manifest(output)?;
    if files != manifest(original)? {
        return Err("independent raw data differs".into());
    }
    write(&output.join("manifest.json"), &files)
}
pub fn verify(original: &Path, repeated: &Path, output: &Path) -> Result<()> {
    for root in [original, repeated] {
        check_inputs(&root.join("inputs"))?;
        if manifest(root)? != read(&root.join("manifest.json"))? {
            return Err("manifest tampering".into());
        }
        let saved = read(&root.join("inventory.json"))?;
        let binary = digest(&std::env::current_exe()?)?;
        let expected = generate(
            root,
            saved["producer_commit"].as_str().ok_or("producer")?,
            &binary,
            false,
        )?;
        if expected != saved {
            return Err("complete source recount differs".into());
        }
    }
    if manifest(original)? != manifest(repeated)? {
        return Err("independent manifests differ".into());
    }
    write(
        output,
        &json!({"independent_repeat_exact":true,"artifacts":manifest(original)?.as_object().unwrap().len(),
        "manifest_sha256":digest(&original.join("manifest.json"))?,"new_model_evaluations":0}),
    )
}
