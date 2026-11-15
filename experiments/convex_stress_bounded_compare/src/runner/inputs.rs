//! Bind the independently reviewed data/probe and retained baseline summaries.
//! Preparation hashes the heldout files but does not deserialize their labels.
use super::*;
use std::path::{Path, PathBuf};
pub struct Sources<'a> {
    pub data: &'a Path,
    pub probe: &'a Path,
    pub summaries: &'a Path,
}
fn manifest(path: &Path) -> Result<BTreeMap<String, String>> {
    let text = fs::read_to_string(path)?;
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let (hash, name) = line.split_once("  ").ok_or("malformed probe manifest")?;
        relative(name)?;
        if hash.len() != 64
            || !hash.bytes().all(|b| b.is_ascii_hexdigit())
            || out.insert(name.into(), hash.into()).is_some()
        {
            return Err("invalid or duplicated probe artifact".into());
        }
    }
    if out.len() != 99 {
        return Err("complete99-artifact probe required".into());
    }
    Ok(out)
}
fn specification(row: &Value, selected: Option<&pairs::selection::Selection>) -> Result<Case> {
    let name = row["case"].as_str().ok_or("case name")?.to_string();
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err("unsafe original case name".into());
    }
    let existing35 = row["existing35"]
        .as_bool()
        .ok_or("old-control membership")?;
    let vocabulary = row["source_case"]["data"]["prime"]
        .as_u64()
        .ok_or("original vocabulary")? as usize;
    let constant_control = protocol()?["additional_constant_controls"]
        .as_array()
        .ok_or("constant controls")?
        .iter()
        .any(|x| x == &name);
    if let Some(s) = selected {
        let data = &row["data"];
        let train = data["train"]["rows"].as_u64().ok_or("training count")? as usize;
        let validation = data["validation"]["rows"]
            .as_u64()
            .ok_or("validation count")? as usize;
        if s.vocabulary != vocabulary
            || s.records != train
            || s.family != pairs::FAMILY
            || s.family_sha256 != pairs::digest(&serde_json::to_vec(&pairs::family::programs())?)
            || s.total_candidates != pairs::PROGRAMS
            || s.retained.len() + s.conflicts.len() != pairs::PROGRAMS
            || !row["refusal"].is_null()
        {
            return Err("original training probe specification differs".into());
        }
        let shape = if s.retained.is_empty() {
            None
        } else {
            Some(s.shape())
        };
        let forecast = shape
            .map(|s| resources::case_budget(s, train, validation))
            .transpose()?;
        Ok(Case {
            name,
            existing35,
            constant_control,
            vocabulary,
            training_records: train,
            validation_records: validation,
            support: s.retained.len(),
            shape,
            source_refusal: None,
            forecast,
        })
    } else {
        if !row["data"].is_null() {
            return Err("available dataset omitted".into());
        }
        Ok(Case {
            name,
            existing35,
            constant_control,
            vocabulary,
            training_records: 0,
            validation_records: 0,
            support: 0,
            shape: None,
            source_refusal: Some(row["refusal"].as_str().ok_or("source refusal")?.into()),
            forecast: None,
        })
    }
}
fn add(
    files: &mut BTreeMap<String, (PathBuf, disk::Artifact)>,
    key: String,
    path: PathBuf,
    expected: &str,
) -> Result<()> {
    relative(&key)?;
    let artifact = disk::hash(&path)?;
    if artifact.sha256 != expected || artifact.bytes > JSON_LIMIT {
        return Err(format!("bound input differs: {key}").into());
    }
    if files.insert(key, (path, artifact)).is_some() {
        return Err("duplicate input binding".into());
    }
    Ok(())
}
fn total(cases: &[Case], inputs: &BTreeMap<String, disk::Artifact>) -> Result<u64> {
    let mut bytes = 1024u128.pow(3);
    for case in cases {
        bytes += case.forecast.map_or(2_000_000, |x| x.total_bytes) as u128;
    }
    bytes += inputs.values().map(|x| x.bytes as u128).sum::<u128>();
    Ok(u64::try_from(bytes)?)
}
pub fn plan(
    source: &Sources<'_>,
    producer: &str,
) -> Result<(Inventory, BTreeMap<String, PathBuf>)> {
    if producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("full frozen producer commit required".into());
    }
    let p = protocol()?;
    let refs = p["references"].as_object().ok_or("pinned references")?;
    let mut files = BTreeMap::new();
    for (name, expected) in refs {
        let path = match name.as_str() {
            "data-inventory.json" => source.data.join("inventory.json"),
            "data-verified.json" => source
                .data
                .parent()
                .ok_or("data parent")?
                .join("verified-v1.json"),
            "probe-manifest.sha256" => source.probe.join("manifest.sha256"),
            "probe-verified.json" => source
                .probe
                .parent()
                .ok_or("probe parent")?
                .join("verified-v1.json"),
            _ => source.summaries.join(name),
        };
        add(
            &mut files,
            format!("inputs/{name}"),
            path,
            expected.as_str().ok_or("digest")?,
        )?;
    }
    let probe = manifest(&source.probe.join("manifest.sha256"))?;
    let data: Value = read(&source.data.join("inventory.json"))?;
    let mut cases = vec![];
    for row in data["cases"].as_array().ok_or("complete original cases")? {
        if row["data"].is_null() {
            cases.push(specification(row, None)?);
            continue;
        }
        let name = row["case"].as_str().ok_or("case name")?;
        for (kind, original) in [("train", "training"), ("probe", "results")] {
            let file = format!("{original}/{name}.json");
            add(
                &mut files,
                format!("inputs/{kind}/{name}.json"),
                source.probe.join(&file),
                probe.get(&file).ok_or("missing probe input")?,
            )?;
        }
        let file = row["data"]["file"].as_str().ok_or("source data file")?;
        relative(file)?;
        add(
            &mut files,
            format!("inputs/data/{name}.json"),
            source.data.join(file),
            row["data"]["sha256"].as_str().ok_or("source digest")?,
        )?;
        let probe_result: Value = read(&source.probe.join(format!("results/{name}.json")))?;
        let selected: pairs::selection::Selection =
            serde_json::from_value(probe_result["selection"].clone())?;
        cases.push(specification(row, Some(&selected))?);
    }
    let inputs = files
        .iter()
        .map(|(key, (_, a))| (key.clone(), a.clone()))
        .collect();
    let inv = Inventory {
        protocol: p,
        producer_commit: producer.into(),
        single_run_budget: total(&cases, &inputs)?,
        cases,
        inputs,
    };
    validate_counts(&inv)?;
    Ok((
        inv,
        files
            .into_iter()
            .map(|(name, (path, _))| (name, path))
            .collect(),
    ))
}
fn validate_counts(inv: &Inventory) -> Result<()> {
    if inv.cases.len() != 54
        || inv
            .cases
            .iter()
            .map(|x| &x.name)
            .collect::<BTreeSet<_>>()
            .len()
            != 54
        || inv
            .cases
            .iter()
            .filter(|x| x.source_refusal.is_none())
            .count()
            != 48
        || inv.cases.iter().filter(|x| x.existing35).count() != 22
        || inv.cases.iter().filter(|x| x.constant_control).count() != 3
        || inv.single_run_budget != total(&inv.cases, &inv.inputs)?
    {
        return Err("complete54-case accounting changed".into());
    }
    Ok(())
}
pub fn validate(root: &Path, inv: &Inventory) -> Result<()> {
    if inv.protocol != protocol()?
        || inv.producer_commit.len() != 40
        || !inv.producer_commit.bytes().all(|b| b.is_ascii_hexdigit())
        || inv.inputs.len() != 8 + 3 * 48
    {
        return Err("immutable comparison producer/references changed".into());
    }
    for (name, expected) in inv.protocol["references"]
        .as_object()
        .ok_or("pinned references")?
    {
        let path = format!("inputs/{name}");
        let binding = inv.inputs.get(&path).ok_or("missing pinned reference")?;
        if binding.sha256 != expected.as_str().ok_or("reference hash")?
            || disk::hash(&root.join(path))? != *binding
        {
            return Err("independent input validation found a changed pinned reference".into());
        }
    }
    validate_counts(inv)?;
    let data: Value = checked(root, inv, "inputs/data-inventory.json")?;
    let probe = manifest(&root.join("inputs/probe-manifest.sha256"))?;
    let rows = data["cases"].as_array().ok_or("original source cases")?;
    if rows.len() != inv.cases.len() {
        return Err("original case count changed".into());
    }
    for (row, case) in rows.iter().zip(&inv.cases) {
        let actual = if row["data"].is_null() {
            specification(row, None)?
        } else {
            for (kind, original) in [("train", "training"), ("probe", "results")] {
                let file = format!("inputs/{kind}/{}.json", case.name);
                if inv.inputs.get(&file).ok_or("input binding")?.sha256
                    != *probe
                        .get(&format!("{original}/{}.json", case.name))
                        .ok_or("probe binding")?
                {
                    return Err("original ordered training/probe digest changed".into());
                }
            }
            if inv
                .inputs
                .get(&format!("inputs/data/{}.json", case.name))
                .ok_or("data binding")?
                .sha256
                != row["data"]["sha256"].as_str().ok_or("data digest")?
            {
                return Err("original heldout dataset digest changed".into());
            }
            specification(row, Some(&selection(root, inv, &case.name)?))?
        };
        if actual != *case {
            return Err("frozen case specification changed".into());
        }
    }
    Ok(())
}
pub fn prepare(source: &Sources<'_>, root: &Path, workspace: &Path, producer: &str) -> Result<()> {
    let (inv, files) = plan(source, producer)?;
    let guard = guard(root, workspace, &inv, 2)?;
    guard.check()?;
    fs::create_dir(root)?;
    for (key, path) in files {
        let target = root.join(&key);
        fs::create_dir_all(target.parent().ok_or("input parent")?)?;
        disk::copy(&path, &target, &inv.inputs[&key], &guard)?;
    }
    initialize(root, &inv, &guard, 2)?;
    validate(root, &inv)
}
fn initialize(root: &Path, inv: &Inventory, guard: &resources::Guard, copies: u64) -> Result<()> {
    fs::create_dir(root.join("cases"))?;
    disk::json(&root.join("inventory.json"), inv, JSON_LIMIT, guard)?;
    disk::json(
        &root.join("run-local.json"),
        &Local {
            copies_reserved: copies,
            binary_sha256: disk::hash(&std::env::current_exe()?)?.sha256,
        },
        4096,
        guard,
    )?;
    Ok(())
}
pub fn repeat(source: &Path, root: &Path, workspace: &Path) -> Result<()> {
    let inv: Inventory = read(&source.join("inventory.json"))?;
    if inv.protocol != protocol()? {
        return Err("repeat protocol differs".into());
    }
    validate(source, &inv)?;
    let guard = guard(root, workspace, &inv, 1)?;
    guard.check()?;
    fs::create_dir(root)?;
    for (key, artifact) in &inv.inputs {
        let target = root.join(key);
        fs::create_dir_all(target.parent().ok_or("input parent")?)?;
        disk::copy(&source.join(key), &target, artifact, &guard)?;
    }
    initialize(root, &inv, &guard, 1)
}
