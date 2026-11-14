//! Read-only task306 diagnosis of the reviewed CM48 result at 5dd63c6e.
//! No model, context, confidence or training trajectory is changed.
use convex_stress_raw_regions::{self as head, Result};
use convex_stress_relabel::{data, metrics};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};
mod counts;
mod geometry;
mod inspect;
#[cfg(test)]
mod tests;
const PROTOCOL: &str = include_str!("../protocol.json");
const SUITE: &str = include_str!("../../convex_stress/suite.json");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    root: PathBuf,
    review: PathBuf,
    manifest: PathBuf,
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(BufReader::new(fs::File::open(
        path,
    )?))?)
}
fn digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn pinned(path: &Path, manifest: &BTreeMap<String, String>) -> Result<()> {
    let key = path.to_str().ok_or("non-UTF8 source path")?;
    if manifest.get(key) != Some(&digest(path)?) {
        return Err("frozen artifact changed or absent from manifest".into());
    }
    Ok(())
}
fn run(request: Request) -> Result<Value> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    if !request.root.is_absolute()
        || fs::read_to_string(request.review.join("process.exit"))?.trim() != "0"
        || fs::read_to_string(request.review.join("reviewed.commit"))?.trim() != p["review_commit"]
        || convex_stress_raw_regions_compare::fingerprint() != p["source_fingerprint"]
        || digest(&request.manifest)? != p["manifest_sha256"]
    {
        return Err("exact reviewed original comparison required".into());
    }
    let manifest: BTreeMap<String, String> = fs::read_to_string(&request.manifest)?
        .lines()
        .map(|line| {
            let (sha, path) = line.split_once("  ").ok_or("invalid completed manifest")?;
            Ok((path.to_owned(), sha.to_owned()))
        })
        .collect::<Result<_>>()?;
    for path in manifest.keys() {
        pinned(Path::new(path), &manifest)?;
    }
    for (file, key) in [
        ("summary.json", "summary_sha256"),
        ("inventory.json", "inventory_sha256"),
    ] {
        if digest(&request.root.join(file))? != p[key] {
            return Err("original summary or inventory differs".into());
        }
    }
    let summary: Value = read(&request.root.join("summary.json"))?;
    if summary["source_fingerprint"] != p["source_fingerprint"] {
        return Err("summary source differs".into());
    }
    let totals = counts::summarize(&summary)?;
    let suite: Value = serde_json::from_str(SUITE)?;
    let mut endpoints = Vec::new();
    for record in summary["records"]
        .as_array()
        .ok_or("missing original records")?
    {
        let outcomes = record["outcomes"]
            .as_array()
            .ok_or("missing endpoint list")?;
        if outcomes.is_empty() {
            continue;
        }
        let task = record["task"].as_str().ok_or("missing task")?;
        if !task
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        {
            return Err("invalid task path".into());
        }
        let rows: Vec<_> = suite["tasks"]
            .as_array()
            .ok_or("missing original suite")?
            .iter()
            .filter(|r| r["name"] == task)
            .collect();
        if rows.len() != 1 {
            return Err("original data row missing or repeated".into());
        }
        let data = data::build(rows[0])?;
        let dir = request.root.join(task);
        let seal_path = dir.join("selection-seal.json");
        pinned(&seal_path, &manifest)?;
        let seal: Value = read(&seal_path)?;
        for row in outcomes {
            endpoints.push(inspect::endpoint(&dir, &data, row, &seal, &manifest)?);
        }
    }
    Ok(
        json!({"protocol":p,"root":request.root,"summary_sha256":digest(&request.root.join("summary.json"))?,
        "diagnostic_binary_sha256":digest(&std::env::current_exe()?)?,
        "primary_raw_totals":totals,"fresh_endpoint_diagnostics":endpoints,
        "inherited_imperfect_choices_remain_in_primary_totals":true,
        "new_model_or_predictions_selected":false,"new_gradient_batches":0,"new_optimizer_updates":0}),
    )
}
fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: convex-stress-raw-audit REQUEST.json")?;
    println!("{}", serde_json::to_string(&run(read(Path::new(&path))?)?)?);
    Ok(())
}
