use super::*;

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
            out.push(json!({"prior":prior,"mode":mode,"new_cases":13,
            "refusals":endpoints.iter().filter(|e|!e["error"].is_null()).count(),
            "fully_measured":endpoints.iter().filter(|e|e["error"].is_null() && e["measurement"].is_object()).count(),
            "original_successes":endpoints.iter().filter(|e|e["primary"]["original_success"]==true).count(),
            "primary_raw99":endpoints.iter().filter(|e|e["primary"]["raw99"]==true).count(),
            "primary_exact99":endpoints.iter().filter(|e|e["primary"]["exact99"]==true).count(),
            "primary_both99":endpoints.iter().filter(|e|e["primary"]["both99"]==true).count(),
            "test_both99":endpoints.iter().filter(|e|e["test"]["both99"]==true).count(),
            "all_heldout99":endpoints.iter().filter(|e|e["all_heldout99"]==true).count()}));
        }
    }
    Ok(out)
}
pub fn summary(root: &Path) -> Result<()> {
    let inv = inventory(root)?;
    all_sealed(root)?;
    let mut rows = vec![];
    for case in cases() {
        let dir = root.join(&case);
        let metric: Value = read(&dir.join("metrics.json"))?;
        let repeat: Value = read(&dir.join("replay.json"))?;
        let seal: Value = read(&dir.join("seal.json"))?;
        if metric["seal"] != seal
            || repeat
                != json!({"case":case,"exact_replay":true,
            "state_sha256":seal["state_sha256"],"metrics_sha256":digest(&dir.join("metrics.json"))?})
        {
            return Err("missing or changed complete reconstruction/replay".into());
        }
        for endpoint in metric["results"]["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?
        {
            if let Some(m) = endpoint["measurement"].as_object() {
                let file = m["raw_file"].as_str().ok_or("missing raw file")?;
                if digest(&dir.join(file))? != m["raw_sha256"] {
                    return Err("raw vectors changed after replay".into());
                }
            }
        }
        rows.push(metric);
    }
    let totals = totals(&rows)?;
    let source: Value = read(&root.join("reference-data.json"))?;
    let baseline_hash = artifacts::canonical_digest(&source["baseline339"])?;
    if baseline_hash != protocol()?["baseline339_report_sha256"] {
        return Err("complete inherited339 report changed".into());
    }
    let mut measured = 0;
    let mut passed = 0;
    for row in &rows {
        let endpoints: Vec<_> = row["results"]["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?
            .iter()
            .filter(|e| e["prior"] == "Coordinates")
            .collect();
        if endpoints.len() != 3 {
            return Err("coordinate mode count differs".into());
        }
        measured += usize::from(
            endpoints
                .iter()
                .all(|e| e["error"].is_null() && e["measurement"].is_object()),
        );
        passed += usize::from(endpoints.iter().all(|e| e["all_heldout99"] == true));
    }
    let mut combined = vec![];
    for new in &totals {
        let old = source["baseline339"]["cohort_totals"]
            .as_array()
            .ok_or("missing inherited cohort totals")?
            .iter()
            .find(|o| o["prior"] == new["prior"] && o["mode"] == new["mode"])
            .ok_or("missing inherited mode/prior")?;
        let mut sum = json!({"prior":new["prior"],"mode":new["mode"],"cases":36,"inherited_cases":23,"new_cases":13});
        for field in [
            "refusals",
            "original_successes",
            "primary_both99",
            "test_both99",
            "all_heldout99",
        ] {
            sum[field] = json!(
                old[field].as_u64().ok_or("invalid inherited total")?
                    + new[field].as_u64().ok_or("invalid new total")?
            );
        }
        combined.push(sum);
    }
    let report = json!({"inventory":inv,"cases":rows,"totals":totals,"cohort_totals":combined,"complete":true,
        "cohort_cases":36,"new_cases":13,"inherited_cases":23,"fully_measured_new_cases":measured,
        "new_all_modes_all_heldout99_cases":passed,"measured_catalog_cases":58+measured,
        "catalog_all_modes_all_heldout99_cases":58+passed,"remaining_unmeasured_catalog_cases":107-measured,
        "baseline339_exact_sha256":baseline_hash,"reference340":source,
        "original35_reference":{"file":"original35.json","sha256":protocol()?["original35_summary_sha256"],"new_evaluations":0},
        "entire165_evaluated":false,"target_realizability_certified":false});
    write(&root.join("report.json"), &report)
}
