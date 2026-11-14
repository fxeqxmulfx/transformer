//! Original evidence is externally pinned before new metrics. The native jq
//! projection keeps all chosen metrics and baselines while avoiding recursive
//! copies of already reviewed, immutable intermediate histories.
use super::*;
const PROJECT: &str = include_str!("../parent.jq");

pub(crate) fn manifest(request: &Request) -> Result<BTreeMap<String, String>> {
    let p = protocol()?;
    if digest(&request.completed_manifest)? != p["parent_completed_manifest_sha256"] {
        return Err("reviewed parent manifest changed".into());
    }
    fs::read_to_string(&request.completed_manifest)?
        .lines()
        .map(|line| {
            let (sha, path) = line.split_once("  ").ok_or("invalid completed manifest")?;
            if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid manifest hash".into());
            }
            Ok((path.to_owned(), sha.to_owned()))
        })
        .collect()
}

pub(crate) fn pinned(path: &Path, manifest: &BTreeMap<String, String>) -> Result<()> {
    let expected = manifest
        .get(path.to_str().ok_or("non-UTF8 evidence path")?)
        .ok_or("artifact absent from reviewed manifest")?;
    if &digest(path)? != expected {
        return Err(format!("reviewed artifact changed: {}", path.display()).into());
    }
    Ok(())
}

pub(crate) fn inventory(root: &Path) -> Result<Value> {
    let seal: Value = read(&root.join("prepare-seal.json"))?;
    if seal["inventory_sha256"] != digest(&root.join("inventory.json"))?
        || seal["parent_snapshot_sha256"] != digest(&root.join("parent-snapshot.json"))?
    {
        return Err("prepared snapshot or inventory changed".into());
    }
    let inv: Value = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint() || inv["protocol"] != protocol()? {
        return Err("prepared source or protocol changed".into());
    }
    Ok(inv)
}

pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if root.exists() {
        return Err("fresh output directory required".into());
    }
    if !request.parent_root.is_absolute()
        || request.producer_commit.len() != 40
        || !request
            .producer_commit
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
    {
        return Err("invalid original request".into());
    }
    let p = protocol()?;
    for (exit, commit, expected) in [
        (
            &request.parent_review_exit,
            &request.parent_review_commit,
            &p["parent_review_commit"],
        ),
        (
            &request.component_review_exit,
            &request.component_review_commit,
            &p["component_commit"],
        ),
    ] {
        if fs::read_to_string(exit)?.trim() != "0"
            || fs::read_to_string(commit)?.trim()
                != expected.as_str().ok_or("invalid review pin")?
        {
            return Err("exact-commit component/parent review required".into());
        }
    }
    for (file, key) in [
        ("summary.json", "parent_summary_sha256"),
        ("inventory.json", "parent_inventory_sha256"),
    ] {
        if digest(&request.parent_root.join(file))? != p[key] {
            return Err("frozen parent summary/inventory changed".into());
        }
    }
    let files = manifest(&request)?;
    let suite: Value = serde_json::from_str(SUITE)?;
    let tasks: Vec<String> = suite["tasks"]
        .as_array()
        .ok_or("missing original tasks")?
        .iter()
        .map(|r| {
            r["name"]
                .as_str()
                .map(String::from)
                .ok_or("missing task name")
        })
        .collect::<std::result::Result<_, _>>()?;
    if tasks.len() != 35 || tasks.iter().collect::<BTreeSet<_>>().len() != 35 {
        return Err("exactly35 original tasks required".into());
    }
    let mut cases = Vec::new();
    for task in &tasks {
        name(task)?;
        let path = request.parent_root.join(task).join("result.json");
        pinned(&path, &files)?;
        let result: Value = read(&path)?;
        let seal_path = request.parent_root.join(task).join("selection-seal.json");
        pinned(&seal_path, &files)?;
        if result["selection_seal_sha256"] != digest(&seal_path)? {
            return Err("reviewed context selection seal changed".into());
        }
        let seal: Value = read(&seal_path)?;
        let models = seal["models"]
            .as_array()
            .ok_or("missing parent model inventory")?;
        let mut eligible = Vec::new();
        for (i, m) in models.iter().enumerate() {
            let file = m["file"].as_str().ok_or("missing model file")?;
            let sha = m["model_sha256"].as_str().ok_or("missing model hash")?;
            if file != format!("model-{i}.json")
                || sha.len() != 64
                || !sha.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return Err("invalid original model filename/hash".into());
            }
            pinned(&request.parent_root.join(task).join(file), &files)?;
            let reference = m["reference"]
                .as_str()
                .ok_or("missing source context reference")?;
            let mut item = m.clone();
            item["reference"] = json!(format!("context:{reference}"));
            eligible.push(item);
        }
        cases.push(
            json!({"task":task,"parent_result_sha256":digest(&path)?,"eligible_models":eligible}),
        );
    }
    let projected = std::process::Command::new("jq")
        .args(["-c", "-S", PROJECT])
        .arg(request.parent_root.join("summary.json"))
        .output()?;
    if !projected.status.success() {
        return Err(format!(
            "parent projection failed: {}",
            String::from_utf8_lossy(&projected.stderr)
        )
        .into());
    }
    let snapshot: Value = serde_json::from_slice(&projected.stdout)?;
    for arm in snapshot["arms"].as_array().ok_or("missing parent arms")? {
        let choices = arm["choices"].as_array().ok_or("missing parent choices")?;
        if choices.len() != 105 {
            return Err("parent must retain105 choices per policy".into());
        }
    }
    fs::create_dir_all(root)?;
    write(&root.join("parent-snapshot.json"), &snapshot)?;
    write(
        &root.join("inventory.json"),
        &json!({"request":request,"protocol":p,"source_fingerprint":fingerprint(),
            "binary_sha256":digest(&std::env::current_exe()?)?,"suite_sha256":hash(SUITE.as_bytes()),"tasks":tasks,"cases":cases}),
    )?;
    write(
        &root.join("prepare-seal.json"),
        &json!({"inventory_sha256":digest(&root.join("inventory.json"))?,
        "parent_snapshot_sha256":digest(&root.join("parent-snapshot.json"))?}),
    )
}
