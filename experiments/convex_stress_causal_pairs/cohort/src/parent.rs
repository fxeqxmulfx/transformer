//! Pin the completed quotient cohort and its independent complete-summary
//! replay. Deep gates invoke its frozen binary and hash every retained file.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Review {
    pub exit: PathBuf,
    pub commit: PathBuf,
    pub summary: PathBuf,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parent {
    pub root: PathBuf,
    pub binary: PathBuf,
    pub summary_sha256: String,
    pub inventory_sha256: String,
    pub review: Review,
}
const SNAPSHOTS: [&str; 3] = ["summary.json", "inventory.json", "request.json"];
pub(crate) fn load(pin: &Parent, suite: &Value, deep: bool) -> Result<Value> {
    let p = protocol()?;
    if !pin.root.is_absolute()
        || !pin.binary.is_absolute()
        || digest(&pin.root.join("summary.json"))? != pin.summary_sha256
        || digest(&pin.root.join("inventory.json"))? != pin.inventory_sha256
        || fs::read_to_string(&pin.review.exit)?.trim() != "0"
        || !commit(fs::read_to_string(&pin.review.commit)?.trim())
        || digest(&pin.review.summary)? != pin.summary_sha256
    {
        return Err("pinned completed quotient parent and independent review required".into());
    }
    let inv: Value = read(&pin.root.join("inventory.json"))?;
    let summary: Value = read(&pin.root.join("summary.json"))?;
    let all = tasks(suite)?;
    if inv["source_fingerprint"] != p["parent_source_fingerprint"]
        || inv["producer_commit"] != p["parent_producer_commit"]
        || inv["binary_sha256"] != digest(&pin.binary)?
        || inv["request_sha256"] != digest(&pin.root.join("request.json"))?
        || inv["tasks"] != json!(all)
        || summary["source_fingerprint"] != inv["source_fingerprint"]
        || summary["producer_commit"] != inv["producer_commit"]
        || summary["protocol"] != inv["protocol"]
        || summary["parent_sha256"] != inv["request"]["parent"]["summary_sha256"]
        || summary["tasks"] != 23
        || summary["records"].as_array().is_none_or(|a| a.len() != 23)
        || inv["planned"].as_array().is_none_or(|a| a.len() != 23)
    {
        return Err("quotient parent source, binary, scope or request changed".into());
    }
    let planned = inv["planned"].as_array().unwrap();
    let records = summary["records"].as_array().unwrap();
    let mut seen = BTreeSet::new();
    for record in records {
        let task = record["task"].as_str().ok_or("missing parent case")?;
        if !all.iter().any(|t| t == task)
            || !planned.contains(&json!(task))
            || !seen.insert(task)
            || !matches!(
                record["status"].as_str(),
                Some("complete" | "process_failure")
            )
        {
            return Err("missing or repeated quotient parent outcome".into());
        }
    }
    comparison::validate(&summary, &all)?;
    if deep {
        let output = std::process::Command::new(&pin.binary)
            .arg("summarize-original")
            .arg(&pin.root)
            .output()?;
        if !output.status.success() || hash(&output.stdout) != pin.summary_sha256 {
            return Err("frozen quotient summary no longer reproduces its reviewed bytes".into());
        }
    }
    Ok(summary)
}
pub(crate) fn evidence(path: &Path) -> Result<Value> {
    Ok(json!({"path":path,"sha256":digest(path)?,"bytes":fs::metadata(path)?.len()}))
}
pub(crate) fn tree(root: &Path, records: &mut Vec<Value>) -> Result<()> {
    let mut entries = fs::read_dir(root)?.collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.path());
    for e in entries {
        if e.file_type()?.is_dir() {
            tree(&e.path(), records)?;
        } else if e.file_type()?.is_file() {
            records.push(evidence(&e.path())?);
        } else {
            return Err("unexpected symbolic/non-file parent artifact".into());
        }
    }
    Ok(())
}
pub(crate) fn snapshot(pin: &Parent, root: &Path) -> Result<Value> {
    fs::create_dir(root)?;
    let mut records = vec![];
    for file in SNAPSHOTS {
        let source = pin.root.join(file);
        fs::copy(&source, root.join(file))?;
        let expected = digest(&source)?;
        if digest(&root.join(file))? != expected {
            return Err("parent changed during snapshot".into());
        }
        records.push(json!({"file":file,"sha256":expected,"bytes":fs::metadata(source)?.len()}));
    }
    Ok(json!(records))
}
pub(crate) fn check_snapshot(root: &Path, records: &Value) -> Result<()> {
    let rows = records.as_array().ok_or("missing parent snapshots")?;
    if rows.len() != SNAPSHOTS.len() {
        return Err("parent snapshot scope changed".into());
    }
    for (row, file) in rows.iter().zip(SNAPSHOTS) {
        if row["file"] != file
            || row["sha256"] != digest(&root.join(file))?
            || row["bytes"] != fs::metadata(root.join(file))?.len()
        {
            return Err("retained complete parent snapshot changed".into());
        }
    }
    Ok(())
}
