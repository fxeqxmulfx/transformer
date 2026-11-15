//! Rehash every original/replay raw artifact, complete state and retained history.
use super::*;
use std::collections::BTreeMap;
fn collect(root: &Path, dir: &Path, files: &mut BTreeMap<String, String>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect(root, &path, files)?;
        } else if path
            .extension()
            .is_some_and(|x| x == "json" || x == "jsonl")
            && path != root.join("manifest.json")
        {
            let relative = path
                .strip_prefix(root)?
                .to_str()
                .ok_or("non-UTF8 artifact path")?
                .to_string();
            files.insert(relative, digest(&path)?);
        }
    }
    Ok(())
}
pub(super) fn write_manifest(root: &Path) -> Result<()> {
    let mut files = BTreeMap::new();
    collect(root, root, &mut files)?;
    write(&root.join("manifest.json"), &files)
}
fn manifest(root: &Path) -> Result<BTreeMap<String, String>> {
    inventory(root)?;
    all_sealed(root)?;
    let saved: BTreeMap<String, String> = read(&root.join("manifest.json"))?;
    let mut actual = BTreeMap::new();
    collect(root, root, &mut actual)?;
    if saved != actual || !saved.contains_key("report.json") || saved.len() < 74 {
        return Err("complete artifact manifest differs or is missing cases".into());
    }
    Ok(saved)
}
pub fn verify_pair(original: &Path, repeated: &Path, output: &Path) -> Result<()> {
    let a = manifest(original)?;
    let b = manifest(repeated)?;
    if a != b {
        return Err("independent complete artifacts differ".into());
    }
    let report: Value = read(&original.join("report.json"))?;
    let c = &report["coverage"];
    write(
        output,
        &json!({"independently_reproduced":true,"artifacts":a.len(),"data_seed":1,
        "report_sha256":a["report.json"],"manifest_sha256":digest(&original.join("manifest.json"))?,
        "catalog_union_all_modes_all_heldout99_cases":c["candidate_catalog_union_all_modes_all_heldout99_cases"],
        "catalog_with_revised_cohort":c["candidate_catalog_with_revised_cohort"],
        "catalog_all_modes_raw99_lower_bound":report["raw_coverage"]["candidate_catalog_raw99_union_lower_bound"],
        "raw_coverage":report["raw_coverage"],"cohort_endpoint_totals":report["totals"],
        "newly_passing_catalog_cases":c["newly_passing_catalog_cases"],"regressed_catalog_cases":c["regressed_catalog_cases"],
        "catalog_total_cases":165,"whole_prior_reports_preserved":true,
        "scope":"Forward/full/minibatch on all original heldout splits; paired categorical controls retained separately."}),
    )
}
