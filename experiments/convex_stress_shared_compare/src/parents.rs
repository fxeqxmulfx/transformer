//! Exact approved evidence, ported from kernel_compare at d51aa422 and
//! fallback_compare at 3852e0b0. No recursive ancestor execution occurs here.
//! Unchanged metrics are historical reviewed values; changed states are replayed.
use super::*;
pub const KEYS: [&str; 4] = ["parent", "raw", "shared", "causal"];
pub(crate) struct Reviewed {
    pub summary: Value,
}

pub(crate) fn evidence(path: &Path) -> Result<Value> {
    Ok(json!({"path":path,"sha256":digest(path)?,"bytes":fs::metadata(path)?.len()}))
}
pub(crate) fn artifacts(rows: &Value) -> Result<()> {
    for row in rows
        .as_array()
        .ok_or("missing imported artifact inventory")?
    {
        let path = Path::new(
            row["path"]
                .as_str()
                .ok_or("missing imported artifact path")?,
        );
        if !path.is_absolute()
            || row["sha256"] != digest(path)?
            || row["bytes"] != fs::metadata(path)?.len()
        {
            return Err("reviewed imported artifact bytes changed".into());
        }
    }
    Ok(())
}
pub(crate) fn tree(root: &Path, rows: &mut Vec<Value>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    let mut entries = fs::read_dir(root)?.collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        if entry.file_type()?.is_dir() {
            tree(&entry.path(), rows)?;
        } else if entry.file_type()?.is_file() {
            rows.push(evidence(&entry.path())?);
        } else {
            return Err("non-file case artifact refused".into());
        }
    }
    Ok(())
}
pub(crate) fn checkpoint_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    if !root.is_absolute()
        || path.is_absolute()
        || path
            .components()
            .any(|p| !matches!(p, std::path::Component::Normal(_)))
    {
        return Err("checkpoint must be a relative path within its frozen root".into());
    }
    let result = root.join(path);
    if result.exists() && !result.canonicalize()?.starts_with(root.canonicalize()?) {
        return Err("checkpoint escapes its frozen root".into());
    }
    Ok(result)
}
pub(crate) fn sources() -> Result<()> {
    let p = protocol()?;
    for (key, actual) in [
        ("parent", convex_stress_refined_compare::fingerprint()),
        ("raw", convex_stress_kernel_cohort::fingerprint()),
        ("shared", convex_stress_shared_roles_cohort::fingerprint()),
        ("causal", convex_stress_causal_pairs_cohort::fingerprint()),
    ] {
        if p["parents"][key]["source_fingerprint"] != actual {
            return Err("compiled reviewed parent source differs".into());
        }
    }
    Ok(())
}
pub(crate) fn load_all(
    request: &Request,
    all: &[String],
    complete_files: bool,
) -> Result<BTreeMap<String, Reviewed>> {
    sources()?;
    if request.parents.len() != KEYS.len() {
        return Err("all four reviewed sources required".into());
    }
    let p = protocol()?;
    let planned = planned()?;
    let mut result = BTreeMap::new();
    for key in KEYS {
        let pin = request.parents.get(key).ok_or("reviewed source omitted")?;
        let item = load(pin, key, &p["parents"][key], all, &planned)?;
        if complete_files {
            for row in item.summary["records"]
                .as_array()
                .ok_or("missing reviewed cases")?
            {
                artifacts(&row["artifacts"])?;
            }
        }
        result.insert(key.into(), item);
    }
    Ok(result)
}
pub(crate) fn load(
    pin: &Reference,
    key: &str,
    expected: &Value,
    all: &[String],
    planned: &[String],
) -> Result<Reviewed> {
    for path in [
        &pin.root,
        &pin.binary,
        &pin.review.exit,
        &pin.review.commit,
        &pin.review.summary,
    ] {
        if !path.is_absolute() {
            return Err("absolute reviewed source paths required".into());
        }
    }
    if !sha(&pin.summary_sha256)
        || !sha(&pin.inventory_sha256)
        || !sha(&pin.review.exit_sha256)
        || !sha(&pin.review.commit_sha256)
        || pin.summary_sha256 != expected["summary_sha256"]
        || pin.inventory_sha256 != expected["inventory_sha256"]
        || digest(&pin.root.join("summary.json"))? != pin.summary_sha256
        || digest(&pin.root.join("inventory.json"))? != pin.inventory_sha256
        || digest(&pin.binary)? != expected["binary_sha256"]
        || digest(&pin.review.exit)? != pin.review.exit_sha256
        || digest(&pin.review.commit)? != pin.review.commit_sha256
        || fs::read_to_string(&pin.review.exit)?.trim() != "0"
        || fs::read_to_string(&pin.review.commit)?.trim() != expected["reviewed_commit"]
        || digest(&pin.review.summary)? != pin.summary_sha256
    {
        return Err("approved source, binary, summary or exact review differs".into());
    }
    let inv: Value = read(&pin.root.join("inventory.json"))?;
    let summary: Value = read(&pin.root.join("summary.json"))?;
    let producer = if key == "raw" || key == "shared" {
        &inv["request"]["producer_commit"]
    } else {
        &inv["producer_commit"]
    };
    if inv["source_fingerprint"] != expected["source_fingerprint"]
        || summary["source_fingerprint"] != inv["source_fingerprint"]
        || inv["binary_sha256"] != expected["binary_sha256"]
        || *producer != expected["producer_commit"]
        || inv["tasks"] != json!(all)
        || inv["request"]["suite_sha256"] != hash(SUITE.as_bytes())
    {
        return Err("reviewed producer, source, executable or original suite differs".into());
    }
    let names = match key {
        "parent" => {
            selection::select(&summary, &[], all)?;
            if summary["protocol"] != inv["protocol"]
                || summary["producer_commit"] != *producer
                || summary["tasks"] != 35
            {
                return Err("complete reviewed parent comparison required".into());
            }
            // Check the two immediate reviewed comparisons as well as the
            // preserved transformer/depth scope validated by the old selector.
            for p in [
                &summary,
                &summary["complete_parent"],
                &summary["complete_parent"]["complete_parent"],
            ] {
                for policy in POLICIES {
                    for task in all {
                        for mode in MODES {
                            let c = cell(p, policy, task, mode)?;
                            if c["status"] == "selected" {
                                let (primary, aux) = passes(
                                    &c["metrics"],
                                    c["primary_split"]
                                        .as_str()
                                        .ok_or("missing original primary split")?,
                                )?;
                                if c["primary_pass"] != primary || c["all_splits_pass"] != aux {
                                    return Err("reviewed parent success differs from original primary rule".into());
                                }
                            }
                        }
                    }
                }
            }
            all
        }
        "raw" | "shared" => {
            if summary["protocol"] != inv["protocol"]
                || summary["request"] != inv["request"]
                || inv["planned"] != json!(planned)
                || summary["planned"] != inv["planned"]
                || summary["inventory_sha256"] != pin.inventory_sha256
            {
                return Err("complete paired training scope changed".into());
            }
            if key == "shared"
                && (summary["shared_role_target_proved"] != false
                    || summary["original_record_multiplicity_retained"] != true)
            {
                return Err("shared-role target or original record scope changed".into());
            }
            planned
        }
        "causal" => {
            if summary["model_protocol"] != inv["model_protocol"]
                || summary["staging_protocol"] != inv["staging_protocol"]
                || summary["inventory_sha256"] != pin.inventory_sha256
            {
                return Err("complete causal training provenance changed".into());
            }
            all
        }
        _ => return Err("unknown reviewed source kind".into()),
    };
    if key != "parent"
        && (summary["phase"] != "training_only"
            || summary["heldout_evaluation"] != false
            || summary["parent_comparison"] != false
            || summary["tasks"] != json!(all))
    {
        return Err("original training-only evidence required".into());
    }
    let rows = summary["records"]
        .as_array()
        .ok_or("missing source case inventory")?;
    if rows.len() != names.len()
        || rows
            .iter()
            .map(|r| r["task"].as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != names.len()
        || rows.iter().any(|r| {
            !names.iter().any(|n| r["task"] == *n)
                || !matches!(r["status"].as_str(), Some("complete" | "process_failure"))
        })
    {
        return Err("reviewed case universe changed".into());
    }
    Ok(Reviewed { summary })
}
