//! Complete original/repeat artifacts and raw replay receipts are mandatory.
//! Verified task counts do not automatically add catalog coverage or seeds.
use crate::{
    resources::Guard,
    schema::{Replayed, Scored},
    *,
};
use serde_json::json;
use std::{collections::BTreeMap, fs, path::Path};
fn walk(
    root: &Path,
    at: &Path,
    out: &mut BTreeMap<String, io::Artifact>,
    depth: usize,
) -> Result<()> {
    if depth > 4 || out.len() > 512 {
        return Err("unexpected run directory inventory".into());
    }
    for entry in fs::read_dir(at)? {
        let path = entry?.path();
        let meta = fs::symlink_metadata(&path)?;
        if meta.is_dir() {
            walk(root, &path, out, depth + 1)?;
        } else if meta.is_file() {
            let name = path
                .strip_prefix(root)?
                .to_str()
                .ok_or("nonUTF8 artifact")?
                .to_string();
            out.insert(name, io::artifact(&path)?);
        } else {
            return Err("run contains a symlink or nonregular artifact".into());
        }
    }
    Ok(())
}
pub(crate) fn complete(
    root: &Path,
    inputs: &inputs::Inputs,
    guard: &Guard,
) -> Result<(BTreeMap<String, io::Artifact>, Vec<serde_json::Value>)> {
    let barrier = seal::require(root, inputs)?;
    let mut expected = seal::collect(root, inputs)?.artifacts;
    expected.insert("seal.json".into(), barrier.clone());
    let mut rows = vec![];
    for entry in &inputs.inventory.cases {
        let id = inputs::id(entry)?;
        let stage = stage::read(root, entry)?;
        let score_path = format!("{id}/scores/scores.json");
        let report: Scored = io::read(&root.join(&score_path))?;
        let replay_path = format!("{id}/replay.json");
        let replay: Replayed = io::read(&root.join(&replay_path))?;
        let score_artifact = io::artifact(&root.join(&score_path))?;
        if replay.name != id
            || replay.score != score_artifact
            || report.name != id
            || report.seal != barrier
            || report.refusal != stage.refusal
        {
            return Err("complete replay receipt or original outcome changed".into());
        }
        // Public verify loaded and validated the complete source inventory.
        // The barrier above binds these exact bytes, including every teacher.
        let data: Option<probes::dataset::Dataset> = if entry.artifact.is_some() {
            Some(io::read(&root.join(format!("data/{id}.json")))?)
        } else {
            None
        };
        let raw_calls = scoring::recompute(
            &root.join(id).join("scores"),
            &stage,
            &report,
            data.as_ref(),
            guard,
        )?;
        if replay.raw_calls != raw_calls {
            return Err("complete raw replay call total differs".into());
        }
        expected.insert(score_path, score_artifact);
        expected.insert(
            replay_path,
            io::artifact(&root.join(id).join("replay.json"))?,
        );
        if stage.refusal.is_some() {
            if !report.modes.is_empty() {
                return Err("refused case acquired an accuracy result".into());
            }
        } else {
            if report.modes.len() != MODES.len() {
                return Err("required scored mode missing".into());
            }
            let source_splits = entry.summary.as_ref().and_then(|v| v["splits"].as_object());
            for (mode, required) in report.modes.iter().zip(MODES) {
                if mode.mode != required || mode.splits.is_empty() {
                    return Err("mode order or split inventory differs".into());
                }
                if let Some(splits) = source_splits {
                    if mode.splits.len() != splits.len()
                        || mode
                            .splits
                            .iter()
                            .zip(splits.keys())
                            .any(|(a, b)| a.split != *b)
                    {
                        return Err("complete original split inventory changed".into());
                    }
                } else {
                    return Err("missing original split summary".into());
                }
                for (index, split) in mode.splits.iter().enumerate() {
                    let path = format!("{id}/scores/{}-{index}.raw", name(mode.mode)?);
                    if io::artifact(&root.join(&path))? != split.trace {
                        return Err("raw logit trace differs from replayed score".into());
                    }
                    expected.insert(path, split.trace.clone());
                }
            }
        }
        rows.push(
            json!({"name":id,"source_refusal":entry.refusal,"model_refusal":stage.refusal,
            "retained_programs":stage.family.as_ref().map(|f|f.partition.retained.len()),
            "transformer_reference":entry.source["history"]["final_test"],
            "modes":report.modes,"raw_replayed_calls":raw_calls}),
        );
    }
    let mut actual = BTreeMap::new();
    walk(root, root, &mut actual, 0)?;
    if actual != expected {
        return Err("complete artifact inventory contains missing or extra files".into());
    }
    let total: u64 = actual.values().map(|a| a.bytes).sum();
    if total > resources::RUN_MAX {
        return Err("complete run exceeds reserved disk forecast".into());
    }
    Ok((actual, rows))
}
pub fn verify(workspace: &Path, original: &Path, repeated: &Path, output: &Path) -> Result<()> {
    let guard = Guard::new(
        workspace,
        output.parent().ok_or("missing verification parent")?,
    )?;
    let a = inputs::load(original)?;
    let b = inputs::load(repeated)?;
    if a.header != b.header || a.inventory != b.inventory {
        return Err("independent complete source protocols differ".into());
    }
    let (first, rows) = complete(original, &a, &guard)?;
    guard.check()?;
    let (second, other) = complete(repeated, &b, &guard)?;
    if first != second || rows != other {
        return Err("independent full endpoints, raw outputs or scores differ".into());
    }
    io::json(
        output,
        &json!({"schema":1,"producer":a.header.producer,
        "complete_inventory_verified":true,"identical_all_artifacts":true,"artifacts":first,
        "cases":rows,"required_modes":MODES,"seeds":[1],"new_catalog_credit":false,
        "source_refusals":a.inventory.cases.iter().filter(|x|x.refusal.is_some()).count(),
        "raw99_and_exact99_distinct_from_original_numeric_success":true}),
        &guard,
    )?;
    Ok(())
}
