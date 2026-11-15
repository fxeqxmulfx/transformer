//! Decisions are recomputed from measured integer counts. A single run's new
//! coverage is provisional until complete artifacts match a fresh independent run.
use super::*;
use std::collections::BTreeMap;

pub fn summary(root: &Path) -> Result<()> {
    let inv = inventory(root)?;
    seal(root)?;
    let mut rows = vec![];
    let mut new_verified_candidate = 0;
    let mut files: Vec<String> = [
        "inventory.json",
        "data345.json",
        "baseline344.json",
        "original35.json",
        "history.json",
        "train.seal.json",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    for case in cases() {
        let file = format!("{case}.metrics.json");
        let value: Value = artifacts::read(&root.join(&file))?;
        if value["case"] != case
            || value["data"] != data_metadata(&inv, &case)?
            || value["stage_sha256"] != artifacts::digest(&root.join(format!("{case}.stage.json")))?
        {
            return Err("complete case measurement binding differs".into());
        }
        let endpoints = value["endpoints"]
            .as_array()
            .ok_or("missing six endpoints")?;
        if endpoints.len() != 6 {
            return Err("incomplete six-endpoint comparison".into());
        }
        let mut all = true;
        for (i, e) in endpoints.iter().enumerate() {
            let prior = if i < 3 { "Coordinates" } else { "Categorical" };
            let mode = ["Forward", "Full", "Mini"][i % 3];
            if e["prior"] != prior || e["mode"] != mode {
                return Err("changed endpoint identity".into());
            }
            let success = if e["error"].is_null() {
                let evaluation = &e["evaluation"];
                if evaluation["readback"]["all_source_rows_recounted"] != true
                    || evaluation["readback"]["all_reported_integer_decisions_checked"] != true
                {
                    return Err("independent raw readback is incomplete".into());
                }
                let splits = evaluation["splits"]
                    .as_object()
                    .ok_or("missing split measurements")?;
                let mut heldout = 0;
                let mut measured = true;
                for (name, s) in splits {
                    let (raw, exact) = super::score::goals(&s["metrics"])?;
                    if !name.starts_with("train") {
                        heldout += 1;
                        measured &= raw && exact;
                    }
                }
                let selected = value["original_selection"]
                    .as_str()
                    .ok_or("missing original selection")?;
                if heldout == 0
                    || e["all_heldout99"] != measured
                    || e["original_success"]
                        != super::score::goals(
                            &splits.get(selected).ok_or("absent original selection")?["metrics"],
                        )?
                        .1
                {
                    return Err("summary decision differs from raw integer counts".into());
                }
                let raw = evaluation["raw_file"]
                    .as_str()
                    .ok_or("missing raw artifact")?;
                if raw != format!("{case}-{prior}-{mode}.raw.jsonl")
                    || evaluation["raw_sha256"] != artifacts::digest(&root.join(raw))?
                {
                    return Err("complete raw evaluation artifact differs".into());
                }
                files.push(raw.into());
                measured
            } else {
                if !e["evaluation"].is_null()
                    || e["all_heldout99"] != false
                    || e["original_success"] != false
                {
                    return Err("refused fit credited with measurements".into());
                }
                false
            };
            if prior == "Coordinates" {
                all &= success;
            }
        }
        new_verified_candidate += usize::from(all);
        files.extend([file, format!("{case}.stage.json")]);
        rows.push(value);
    }
    let references: Vec<_> = references()
        .iter()
        .map(|(file, key)| {
            json!({"file":file,"sha256":inv["protocol"][key],
        "whole_file_preserved":true})
        })
        .collect();
    artifacts::write(
        &root.join("report.json"),
        &json!({"complete":true,"new_cases":8,"data_seed":1,
        "producer_commit":inv["producer_commit"],"binary_sha256":inv["binary_sha256"],"cases":rows,
        "baseline_verified_catalog_cases":69,"new_all_modes_all_heldout99_cases":new_verified_candidate,
        "candidate_catalog_all_modes_all_heldout99_cases":69+new_verified_candidate,
        "independent_repeat_required":true,"references":references,"original_transformer_retrained":false,
        "scope":"All original Basis splits; observed noisy train and clean-train diagnostic remain distinct. Discrete discovery and anchor selection are outside joint convex CE."}),
    )?;
    files.push("report.json".into());
    files.sort();
    files.dedup();
    let manifest: BTreeMap<_, _> = files
        .into_iter()
        .map(|file| Ok((file.clone(), artifacts::digest(&root.join(&file))?)))
        .collect::<Result<_>>()?;
    artifacts::write(&root.join("manifest.json"), &manifest)
}

fn manifest(root: &Path) -> Result<BTreeMap<String, String>> {
    inventory(root)?;
    seal(root)?;
    let m: BTreeMap<String, String> = artifacts::read(&root.join("manifest.json"))?;
    if !m.contains_key("report.json") || m.len() < 23 {
        return Err("incomplete comparison manifest".into());
    }
    for (file, digest) in &m {
        if Path::new(file).components().count() != 1
            || artifacts::digest(&root.join(file))? != *digest
        {
            return Err("manifest artifact changed or is not a local filename".into());
        }
    }
    Ok(m)
}
pub fn verify_pair(original: &Path, review: &Path, output: &Path) -> Result<()> {
    let a = manifest(original)?;
    let b = manifest(review)?;
    if a != b {
        return Err("complete independent artifact manifests differ".into());
    }
    let report: Value = artifacts::read(&original.join("report.json"))?;
    let new = report["new_all_modes_all_heldout99_cases"]
        .as_u64()
        .ok_or("missing measured new-case count")?;
    artifacts::write(
        output,
        &json!({"independently_reproduced":true,"artifacts":a.len(),"data_seed":1,
        "report_sha256":a["report.json"],"manifest_sha256":artifacts::digest(&original.join("manifest.json"))?,
        "new_all_modes_all_heldout99_cases":new,"catalog_all_modes_all_heldout99_cases":69+new,
        "catalog_total_cases":165,"unmeasured_or_not_passing":165-69-new,
        "whole_prior_reports_preserved":true,"scope":"Coordinate model in forward/full/minibatch on every original heldout split; paired categorical controls retained separately."}),
    )
}
