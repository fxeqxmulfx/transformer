//! Immutable foreign summaries and independent review, ported from b516af90.
//! The two frozen executables remain responsible for their own full summaries.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Review {
    pub exit: PathBuf,
    pub exit_sha256: String,
    pub commit: PathBuf,
    pub commit_sha256: String,
    pub summary: PathBuf,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub root: PathBuf,
    pub binary: PathBuf,
    pub summary_sha256: String,
    pub inventory_sha256: String,
    pub review: Review,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    Parent,
    Kernel,
}
impl Kind {
    fn key(self) -> &'static str {
        match self {
            Self::Parent => "parent",
            Self::Kernel => "staging",
        }
    }
}
pub(crate) fn evidence(path: &Path) -> Result<Value> {
    Ok(json!({"path":path,"sha256":digest(path)?,"bytes":fs::metadata(path)?.len()}))
}
pub(crate) fn tree(root: &Path, out: &mut Vec<Value>) -> Result<()> {
    let mut files = fs::read_dir(root)?.collect::<std::result::Result<Vec<_>, _>>()?;
    files.sort_by_key(|e| e.path());
    for f in files {
        if f.file_type()?.is_dir() {
            tree(&f.path(), out)?;
        } else if f.file_type()?.is_file() {
            out.push(evidence(&f.path())?);
        } else {
            return Err("non-file source artifact refused".into());
        }
    }
    Ok(())
}
pub(crate) fn load(
    pin: &Reference,
    kind: Kind,
    all: &[String],
    planned: &[String],
    deep: bool,
) -> Result<(Value, Value)> {
    let p = protocol()?;
    if convex_stress_kernel_cohort::fingerprint() != p["staging"]["source_fingerprint"]
        || convex_stress_refined_compare::fingerprint() != p["parent"]["source_fingerprint"]
    {
        return Err("frozen component sources have changed".into());
    }
    let (inv, summary) = load_with(pin, kind, &p[kind.key()], all, planned, deep)?;
    if kind == Kind::Parent {
        causal_stage(&summary)?;
    }
    Ok((inv, summary))
}
pub(crate) fn causal_stage(summary: &Value) -> Result<()> {
    let p = protocol()?;
    if summary["complete_parent"]["staged_construction"]["reference"]["summary_sha256"]
        != p["causal_stage_summary_sha256"]
        || summary["complete_parent"]["staged_construction"]["reference"]["inventory_sha256"]
            != p["causal_stage_inventory_sha256"]
        || summary["staging_sha256"] != p["refined_stage_summary_sha256"]
        || summary["staged_construction"]["inventory_sha256"] != p["refined_stage_inventory_sha256"]
    {
        return Err("original causal237 or refined243 stage changed".into());
    }
    Ok(())
}
pub(crate) fn load_with(
    pin: &Reference,
    kind: Kind,
    expected: &Value,
    all: &[String],
    planned: &[String],
    deep: bool,
) -> Result<(Value, Value)> {
    if !pin.root.is_absolute()
        || !pin.binary.is_absolute()
        || !sha(&pin.summary_sha256)
        || !sha(&pin.inventory_sha256)
        || digest(&pin.root.join("summary.json"))? != pin.summary_sha256
        || digest(&pin.root.join("inventory.json"))? != pin.inventory_sha256
        || !sha(&pin.review.exit_sha256)
        || !sha(&pin.review.commit_sha256)
        || digest(&pin.review.exit)? != pin.review.exit_sha256
        || digest(&pin.review.commit)? != pin.review.commit_sha256
        || fs::read_to_string(&pin.review.exit)?.trim() != "0"
        || !commit(fs::read_to_string(&pin.review.commit)?.trim())
        || digest(&pin.review.summary)? != pin.summary_sha256
        || digest(&pin.binary)? != expected["binary_sha256"]
    {
        return Err("complete foreign state, binary and independent review pins required".into());
    }
    let inv: Value = read(&pin.root.join("inventory.json"))?;
    let summary: Value = read(&pin.root.join("summary.json"))?;
    let producer = if kind == Kind::Parent {
        &inv["producer_commit"]
    } else {
        &inv["request"]["producer_commit"]
    };
    if inv["source_fingerprint"] != expected["source_fingerprint"]
        || summary["source_fingerprint"] != inv["source_fingerprint"]
        || inv["binary_sha256"] != expected["binary_sha256"]
        || *producer != expected["producer_commit"]
        || inv["tasks"] != json!(all)
        || summary["protocol"] != inv["protocol"]
    {
        return Err("foreign source, producer, protocol or35-task scope differs".into());
    }
    let names = if kind == Kind::Parent {
        comparison::validate(&summary, all)?;
        if summary["producer_commit"] != *producer
            || summary["tasks"] != 35
            || summary["staged_construction"].is_null()
        {
            return Err("complete imported refined parent required".into());
        }
        all
    } else {
        if summary["inventory_sha256"] != pin.inventory_sha256
            || summary["request"] != inv["request"]
            || summary["tasks"] != json!(all)
            || summary["planned"] != json!(planned)
            || inv["planned"] != json!(planned)
            || summary["phase"] != "training_only"
            || summary["heldout_evaluation"] != false
            || summary["parent_comparison"] != false
            || summary["raw_label_identification"] != false
            || summary["target_readout_compatibility_proved"] != false
            || summary["constructor_accounting"]["charged_once_across_both_arms"] != true
            || summary["constructor_accounting"]["all_discovery_work_in_stage_records"] != true
        {
            return Err("complete reviewed training-only stage required".into());
        }
        planned
    };
    let records = summary["records"]
        .as_array()
        .ok_or("missing foreign records")?;
    if records.len() != names.len() {
        return Err("foreign task outcomes omitted".into());
    }
    let mut seen = BTreeSet::new();
    for r in records {
        let task = r["task"].as_str().ok_or("missing foreign task")?;
        if !names.iter().any(|n| n == task)
            || !seen.insert(task)
            || !matches!(r["status"].as_str(), Some("complete" | "process_failure"))
        {
            return Err("unknown, repeated or missing foreign process outcome".into());
        }
    }
    if deep {
        let output = std::process::Command::new(&pin.binary)
            .arg("summarize")
            .arg(&pin.root)
            .output()?;
        if !output.status.success() || hash(&output.stdout) != pin.summary_sha256 {
            return Err("foreign frozen summary failed exact replay".into());
        }
    }
    Ok((inv, summary))
}
pub(crate) fn snapshot(pin: &Reference, destination: &Path) -> Result<Value> {
    fs::create_dir(destination)?;
    let mut rows = vec![];
    for file in ["summary.json", "inventory.json", "request.json"] {
        fs::copy(pin.root.join(file), destination.join(file))?;
        let sha = digest(&pin.root.join(file))?;
        if digest(&destination.join(file))? != sha {
            return Err("foreign snapshot changed while copying".into());
        }
        rows.push(
            json!({"file":file,"sha256":sha,"bytes":fs::metadata(destination.join(file))?.len()}),
        );
    }
    Ok(json!(rows))
}
pub(crate) fn check_snapshot(root: &Path, rows: &Value) -> Result<()> {
    let rows = rows.as_array().ok_or("missing source snapshots")?;
    if rows.len() != 3 {
        return Err("incomplete source snapshots".into());
    }
    for (row, file) in rows
        .iter()
        .zip(["summary.json", "inventory.json", "request.json"])
    {
        if row["file"] != file
            || row["sha256"] != digest(&root.join(file))?
            || row["bytes"] != fs::metadata(root.join(file))?.len()
        {
            return Err("foreign complete snapshot changed".into());
        }
    }
    Ok(())
}
