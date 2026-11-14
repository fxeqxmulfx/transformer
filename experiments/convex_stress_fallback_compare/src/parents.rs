//! Immutable foreign evidence, ported from d341fdf8. Source, executable,
//! reviewed summaries and actual predecessor bytes are independently bound.
use super::*;
pub use convex_stress_readout_compare::{Reference, Review};
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
    staging: bool,
    all: &[String],
    deep: bool,
) -> Result<(Value, Value)> {
    let p = protocol()?;
    if convex_stress_readout_compare::fingerprint() != p["parent"]["source_fingerprint"]
        || convex_stress_causal_pairs_cohort::fingerprint() != p["staging"]["source_fingerprint"]
    {
        return Err("frozen ancestor sources changed".into());
    }
    load_with(
        pin,
        staging,
        &p[if staging { "staging" } else { "parent" }],
        all,
        deep,
    )
}
pub(crate) fn load_with(
    pin: &Reference,
    staging: bool,
    expected: &Value,
    all: &[String],
    deep: bool,
) -> Result<(Value, Value)> {
    if !pin.root.is_absolute()
        || !pin.binary.is_absolute()
        || !pin.review.exit.is_absolute()
        || !pin.review.commit.is_absolute()
        || !pin.review.summary.is_absolute()
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
        return Err("foreign state, executable or independent review differs".into());
    }
    let inv: Value = read(&pin.root.join("inventory.json"))?;
    let summary: Value = read(&pin.root.join("summary.json"))?;
    if inv["source_fingerprint"] != expected["source_fingerprint"]
        || summary["source_fingerprint"] != inv["source_fingerprint"]
        || inv["producer_commit"] != expected["producer_commit"]
        || summary["producer_commit"] != inv["producer_commit"]
        || inv["binary_sha256"] != expected["binary_sha256"]
        || inv["tasks"] != json!(all)
        || inv["request"]["suite_sha256"] != hash(SUITE.as_bytes())
    {
        return Err("foreign source, producer or original suite differs".into());
    }
    if staging {
        if pin.summary_sha256 != expected["summary_sha256"]
            || pin.inventory_sha256 != expected["inventory_sha256"]
            || summary["inventory_sha256"] != pin.inventory_sha256
            || summary["tasks"] != json!(all)
            || summary["model_protocol"] != inv["model_protocol"]
            || summary["staging_protocol"] != inv["staging_protocol"]
            || summary["phase"] != "training_only"
            || summary["heldout_evaluation"] != false
            || summary["parent_comparison"] != false
        {
            return Err("original reviewed causal training stage differs".into());
        }
    } else {
        comparison::validate(&summary, all)?;
        if summary["protocol"] != inv["protocol"] || summary["tasks"] != 35 {
            return Err("complete reviewed readout comparison required".into());
        }
    }
    let records = summary["records"]
        .as_array()
        .ok_or("missing foreign outcomes")?;
    let mut seen = BTreeSet::new();
    if records.len() != 35 {
        return Err("foreign outcomes omitted".into());
    }
    for r in records {
        let task = r["task"].as_str().ok_or("missing foreign task")?;
        if !all.iter().any(|n| n == task)
            || !seen.insert(task)
            || !matches!(r["status"].as_str(), Some("complete" | "process_failure"))
        {
            return Err("unknown, repeated or missing foreign outcome".into());
        }
    }
    if deep {
        let output = std::process::Command::new(&pin.binary)
            .arg(if staging {
                "summarize-training"
            } else {
                "summarize"
            })
            .arg(&pin.root)
            .output()?;
        if !output.status.success() || hash(&output.stdout) != pin.summary_sha256 {
            return Err("foreign frozen summary failed exact replay".into());
        }
    }
    Ok((inv, summary))
}
pub(crate) fn attachment(parent: &Value, stage: &Value, pin: &Reference) -> Result<()> {
    let old = &comparison::causal(parent)["staged_construction"];
    if old["reference"]["root"] != json!(pin.root)
        || old["reference"]["summary_sha256"] != pin.summary_sha256
        || old["reference"]["inventory_sha256"] != pin.inventory_sha256
        || old["summary"] != *stage
    {
        return Err("selected causal ancestry is not the exact reviewed stage".into());
    }
    Ok(())
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
            return Err("foreign snapshot changed".into());
        }
    }
    Ok(())
}
pub(crate) fn checkpoint_path(root: &Path, saved: &str) -> Result<PathBuf> {
    let path = Path::new(saved);
    if !root.is_absolute()
        || path.components().any(|c| {
            !matches!(
                c,
                std::path::Component::Normal(_) | std::path::Component::RootDir
            )
        })
        || saved.is_empty()
    {
        return Err("checkpoint needs an absolute root and a normalized saved path".into());
    }
    // Reviewed ancestry contains absolute general-bank paths and original
    // root-relative checkpoints. Their saved hashes identify the exact bytes.
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    })
}
pub(crate) fn checkpoints(root: &Path, parent: &Value, all: &[String]) -> Result<Value> {
    if !root.is_absolute() {
        return Err("absolute checkpoint root required".into());
    }
    let quotient = &comparison::causal(parent)["complete_parent"];
    let mut rows = BTreeMap::new();
    for policy in POLICIES {
        for task in all {
            for m in MODES {
                let saved = &comparison::cell(quotient, policy, task, m)?["saved_endpoint"];
                if let (Some(relative), Some(expected)) =
                    (saved["model_path"].as_str(), saved["model_sha256"].as_str())
                {
                    if !sha(expected) {
                        return Err("invalid saved predecessor digest".into());
                    }
                    let path = checkpoint_path(root, relative)?;
                    let state = if path.exists() {
                        let e = evidence(&path)?;
                        if e["sha256"] != expected {
                            return Err("saved predecessor checkpoint hash changed".into());
                        }
                        json!({"status":"available","evidence":e})
                    } else {
                        json!({"status":"missing","path":path})
                    };
                    let value =
                        json!({"saved_path":relative,"expected_sha256":expected,"state":state});
                    if rows
                        .insert(relative.to_string(), value.clone())
                        .is_some_and(|old| old != value)
                    {
                        return Err("contradictory predecessor file binding".into());
                    }
                }
            }
        }
    }
    Ok(json!(rows.into_values().collect::<Vec<_>>()))
}
