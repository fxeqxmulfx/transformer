//! Adapt task347 at1371ae4f: preserve successes/regressions and distinguish
//! all-heldout raw99 from exact99 and their conjunction.
use super::*;
use std::collections::BTreeSet;

pub(super) fn totals(rows: &[Value]) -> Result<Vec<Value>> {
    let mut out = vec![];
    for prior in ["Coordinates", "Categorical"] {
        for mode in ["Forward", "Full", "Mini"] {
            let endpoints: Vec<_> = rows
                .iter()
                .flat_map(|r| r["results"]["endpoints"].as_array().into_iter().flatten())
                .filter(|e| e["prior"] == prior && e["mode"] == mode)
                .collect();
            if endpoints.len() != 13 {
                return Err("omitted case in thirteen-case denominator".into());
            }
            out.push(json!({"prior":prior,"mode":mode,"repeated_cases":13,
                "refusals":endpoints.iter().filter(|e| !e["error"].is_null()).count(),
                "fully_measured":endpoints.iter().filter(|e| e["error"].is_null() && e["measurement"].is_object()).count(),
                "original_successes":endpoints.iter().filter(|e| e["primary"]["original_success"]==true).count(),
                "primary_raw99":endpoints.iter().filter(|e| e["primary"]["raw99"]==true).count(),
                "primary_exact99":endpoints.iter().filter(|e| e["primary"]["exact99"]==true).count(),
                "primary_both99":endpoints.iter().filter(|e| e["primary"]["both99"]==true).count(),
                "test_both99":endpoints.iter().filter(|e| e["test"]["both99"]==true).count(),
                "all_heldout99":endpoints.iter().filter(|e| e["all_heldout99"]==true).count()}));
            let last = out.last_mut().ok_or("absent endpoint totals")?;
            last["all_heldout_raw99"] = json!(
                endpoints
                    .iter()
                    .filter(|e| e["all_heldout_raw99"] == true)
                    .count()
            );
            last["all_heldout_exact99"] = json!(
                endpoints
                    .iter()
                    .filter(|e| e["all_heldout_exact99"] == true)
                    .count()
            );
        }
    }
    Ok(out)
}
pub(super) fn passing_cases(rows: &[Value]) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    let mut passed = BTreeSet::new();
    for row in rows {
        let name = row["results"]["case"]
            .as_str()
            .ok_or("missing case identity")?;
        if !names.insert(name.to_string()) {
            return Err("duplicate case identity".into());
        }
        let endpoints = row["results"]["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?;
        if endpoints.len() != 6 {
            return Err("missing six endpoint identities".into());
        }
        for (i, e) in endpoints.iter().enumerate() {
            if e["prior"] != if i < 3 { "Coordinates" } else { "Categorical" }
                || e["mode"] != ["Forward", "Full", "Mini"][i % 3]
            {
                return Err("changed endpoint identity or ordering".into());
            }
        }
        if endpoints[..3].iter().all(|e| {
            e["error"].is_null() && e["measurement"].is_object() && e["all_heldout99"] == true
        }) {
            passed.insert(name.to_string());
        }
    }
    if names != cases().into_iter().collect() {
        return Err("complete original case set differs".into());
    }
    Ok(passed)
}
pub(super) fn coverage(previous: &BTreeSet<String>, current: &BTreeSet<String>) -> Value {
    let new: Vec<_> = current.difference(previous).collect();
    let regressed: Vec<_> = previous.difference(current).collect();
    json!({"baseline_verified_catalog_cases":77,"previous_cohort_passes":previous.len(),
        "current_cohort_passes":current.len(),"newly_passing_catalog_cases":new,
        "regressed_catalog_cases":regressed,"candidate_catalog_union_all_modes_all_heldout99_cases":77+new.len(),
        "candidate_catalog_with_revised_cohort":77-previous.len()+current.len(),
        "independent_repeat_required":true,"catalog_total_cases":165,"data_seed":1,
        "scope":"Union of separate frozen cohorts, not a shared all-task checkpoint or three seeds."})
}
pub(super) fn raw_passing_cases(rows: &[Value]) -> Result<BTreeSet<String>> {
    // Reuse complete identity/denominator validation, not its stricter decision.
    passing_cases(rows)?;
    let mut passed = BTreeSet::new();
    for row in rows {
        let endpoints = row["results"]["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?;
        if endpoints[..3].iter().all(|e| {
            e["error"].is_null() && e["measurement"].is_object() && e["all_heldout_raw99"] == true
        }) {
            passed.insert(
                row["results"]["case"]
                    .as_str()
                    .ok_or("missing case")?
                    .to_string(),
            );
        }
    }
    Ok(passed)
}
fn check_endpoint(e: &Value, selection: &str, dir: &Path, i: usize) -> Result<()> {
    if !e["error"].is_null() {
        if e["measurement"].is_object()
            || e["all_heldout99"] != false
            || e["all_heldout_raw99"] != false
            || e["all_heldout_exact99"] != false
        {
            return Err("refused endpoint credited with measurements".into());
        }
        return Ok(());
    }
    let m = &e["measurement"];
    let file = format!("raw-{i}.jsonl");
    if m["raw_file"] != file
        || m["raw_sha256"] != digest(&dir.join(&file))?
        || digest(&dir.join("replay").join(&file))? != m["raw_sha256"]
    {
        return Err("complete original/replay raw vectors differ".into());
    }
    let splits = m["splits"].as_object().ok_or("missing split metrics")?;
    let (mut heldout, mut all) = (0, true);
    let (mut raw99, mut exact99) = (true, true);
    for (name, s) in splits {
        let goals = score::goals(&s["metrics"])?;
        if s["goals"] != goals {
            return Err("saved decisions differ from integer counts".into());
        }
        if !name.starts_with("train") {
            heldout += 1;
            all &= goals["both99"] == true;
            raw99 &= goals["raw99"] == true;
            exact99 &= goals["exact99"] == true;
        }
    }
    if heldout == 0
        || e["all_heldout99"] != all
        || e["all_heldout_raw99"] != raw99
        || e["all_heldout_exact99"] != exact99
        || m["raw_readback"]["all_numerical_certificates_recomputed"] != true
        || e["primary"] != splits.get(selection).ok_or("absent original selection")?["goals"]
        || e["test"] != splits.get("test").ok_or("absent original test")?["goals"]
    {
        return Err("heldout/primary/test decision differs".into());
    }
    Ok(())
}
pub fn summary(root: &Path) -> Result<()> {
    let inv = inventory(root)?;
    all_sealed(root)?;
    let mut rows = vec![];
    for case in cases() {
        let dir = root.join(&case);
        let metric: Value = read(&dir.join("metrics.json"))?;
        let replay: Value = read(&dir.join("replay.json"))?;
        let seal: Value = read(&dir.join("seal.json"))?;
        if metric["results"]["case"] != case
            || metric["seal"] != seal
            || replay
                != json!({"case":case,"exact_replay":true,"state_sha256":seal["state_sha256"],"metrics_sha256":digest(&dir.join("metrics.json"))?})
            || digest(&dir.join("replay/metrics.json"))? != digest(&dir.join("metrics.json"))?
        {
            return Err("case state, metric or complete replay binding differs".into());
        }
        let selection = metric["results"]["selection"]
            .as_str()
            .ok_or("missing original selection")?;
        for (i, e) in metric["results"]["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?
            .iter()
            .enumerate()
        {
            check_endpoint(e, selection, &dir, i)?;
        }
        rows.push(metric);
    }
    let totals = totals(&rows)?;
    let current = passing_cases(&rows)?;
    let current_raw = raw_passing_cases(&rows)?;
    let previous: Value = read(&root.join("baseline347.json"))?;
    let old = passing_cases(
        previous["cases"]
            .as_array()
            .ok_or("missing inherited347 cases")?,
    )?;
    if old.len() != 11 {
        return Err("verified347 pass count differs".into());
    }
    drop(previous);
    let new_raw: Vec<_> = current_raw.difference(&old).collect();
    let regressed_raw: Vec<_> = old.difference(&current_raw).collect();
    let raw_coverage = json!({"baseline_verified_catalog_raw99_lower_bound":77,
        "current_cohort_all_modes_raw99_cases":current_raw.len(),
        "newly_raw_passing_catalog_cases":new_raw,"regressed_from_verified_cases":regressed_raw,
        "candidate_catalog_raw99_union_lower_bound":77+new_raw.len(),
        "candidate_catalog_raw99_with_revised_cohort_lower_bound":77-old.len()+current_raw.len(),
        "independent_repeat_required":true,"catalog_total_cases":165,"data_seed":1,
        "scope":"Primary Coordinates in all three modes on every original heldout split. Baseline77 is a verified lower bound, not a recount of every previous raw-only outcome."});
    let history: Vec<_> = references().into_iter().map(|(file, _, key)| json!({"file":file,"sha256":inv["protocol"][key],"whole_file_preserved":true})).collect();
    write(
        &root.join("report.json"),
        &json!({"inventory":inv,"complete":true,"repeated_cases":13,
        "cases":rows,"totals":totals,"coverage":coverage(&old,&current),"raw_coverage":raw_coverage,"references":history,
        "original_transformer_retrained":false,"target_realizability_certified":false,
        "entire165_evaluated":false,"ordinary_ce_is_teacher_forced":true}),
    )?;
    verify::write_manifest(root)
}
