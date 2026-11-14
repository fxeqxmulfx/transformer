//! Pin reviewed selections and import exact source tensors before metrics.
//! The unchanged parent projection retains all outcomes, baselines and costs.
use super::*;
const PROJECT: &str = include_str!("../parent.jq");

pub(crate) fn manifest(request: &Request) -> Result<BTreeMap<String, String>> {
    read_manifest(
        &request.completed_manifest,
        &protocol()?["parent_completed_manifest_sha256"],
    )
}
fn read_manifest(path: &Path, expected: &Value) -> Result<BTreeMap<String, String>> {
    if digest(path)? != *expected {
        return Err("reviewed parent manifest changed".into());
    }
    fs::read_to_string(path)?
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
pub(crate) fn pinned(path: &Path, files: &BTreeMap<String, String>) -> Result<()> {
    let expected = files
        .get(path.to_str().ok_or("non-UTF8 path")?)
        .ok_or("artifact absent from reviewed manifest")?;
    if &digest(path)? != expected {
        return Err("reviewed artifact changed".into());
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
            return Err("exact-commit component and parent review required".into());
        }
    }
    for (file, key) in [
        ("summary.json", "parent_summary_sha256"),
        ("inventory.json", "parent_inventory_sha256"),
    ] {
        if digest(&request.parent_root.join(file))? != p[key] {
            return Err("frozen parent summary or inventory changed".into());
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
        return Err("exactly 35 original tasks required".into());
    }
    let projected = std::process::Command::new("jq")
        .args(["-c", "-S", PROJECT])
        .arg(request.parent_root.join("summary.json"))
        .output()?;
    if !projected.status.success() {
        return Err("reviewed parent projection failed".into());
    }
    let snapshot: Value = serde_json::from_slice(&projected.stdout)?;
    for arm in snapshot["arms"]
        .as_array()
        .ok_or("missing parent policies")?
    {
        if arm["choices"]
            .as_array()
            .ok_or("missing parent choices")?
            .len()
            != 105
        {
            return Err("all 105 parent choices per policy required".into());
        }
    }
    fs::create_dir_all(root.join("sources"))?;
    let mut containers = BTreeMap::new();
    let mut extracted = BTreeMap::<PathBuf, (String, String)>::new();
    let mut cases = Vec::new();
    for task in &tasks {
        name(task)?;
        let path = request.parent_root.join(task).join("result.json");
        pinned(&path, &files)?;
        let mut eligible = import::eligible(&snapshot, task)?;
        for entry in &mut eligible {
            let container = import::location(&request, task, entry)?;
            if !containers.contains_key(&container) {
                containers.insert(container.clone(), digest(&container)?);
            }
            let expected = entry["source_model_sha256"]
                .as_str()
                .ok_or("missing selected model hash")?
                .to_owned();
            let key = container.clone();
            if !extracted.contains_key(&key) {
                let file = format!("sources/{expected}.json");
                if containers[&container] != expected {
                    return Err("current selected model differs from source container".into());
                }
                if root.join(&file).exists() {
                    if digest(&root.join(&file))? != expected {
                        return Err("shared source bytes changed".into());
                    }
                } else {
                    fs::copy(&container, root.join(&file))?;
                }
                extracted.insert(key.clone(), (file, expected.clone()));
            }
            let (file, actual) = &extracted[&key];
            if actual != &expected {
                return Err("selected references disagree on source identity".into());
            }
            import::mode(&entry["training_mode"])?;
            entry["file"] = json!(file);
            entry["container"] = json!(container);
            entry["container_sha256"] = json!(containers[&container]);
        }
        cases.push(
            json!({"task":task,"parent_result_sha256":digest(&path)?,"eligible_models":eligible}),
        );
    }
    write(&root.join("parent-snapshot.json"), &snapshot)?;
    write(
        &root.join("inventory.json"),
        &json!({"request":request,"protocol":p,
        "source_fingerprint":fingerprint(),"binary_sha256":digest(&std::env::current_exe()?)?,
        "suite_sha256":hash(SUITE.as_bytes()),"tasks":tasks,"cases":cases}),
    )?;
    write(
        &root.join("prepare-seal.json"),
        &json!({
        "inventory_sha256":digest(&root.join("inventory.json"))?,
        "parent_snapshot_sha256":digest(&root.join("parent-snapshot.json"))?}),
    )
}
