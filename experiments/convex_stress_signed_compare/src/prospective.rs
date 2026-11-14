//! Use only the reviewed six training-only inspections to set prospective
//! case work and memory. No original evaluation metric or target is read.
use super::*;
pub fn forecast(path: &Path) -> Result<Value> {
    if digest(path)? != protocol()?["training_inspection_sha256"] {
        return Err("reviewed training-only inspections changed".into());
    }
    let rows: Vec<Value> = fs::read_to_string(path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    calculate(&rows)
}
pub(crate) fn calculate(rows: &[Value]) -> Result<Value> {
    if rows.len() != 6 {
        return Err("all six inspected source roles required".into());
    }
    let mut seen = BTreeSet::new();
    let mut bounds = BTreeMap::<String, (u64, u64)>::new();
    for row in rows {
        let mode = row["mode"].as_str().ok_or("missing inspected mode")?;
        let role = row["role"].as_str().ok_or("missing inspected role")?;
        if !["full", "mini", "forward"].contains(&mode)
            || !["prior", "supplement"].contains(&role)
            || !seen.insert((mode, role))
            || row["heldout_metrics_evaluated"] != false
        {
            return Err("changed training-only inspection scope".into());
        }
        let b: head::Bounds = serde_json::from_value(row["bounds"].clone())?;
        if b.total_scalar_work > head::WORK_LIMIT || b.buffer_bytes > head::BUFFER_LIMIT {
            return Err("component inspection exceeds frozen limits".into());
        }
        let totals = bounds.entry(mode.into()).or_default();
        totals.0 = totals
            .0
            .checked_add(b.buffer_bytes)
            .ok_or("buffer overflow")?;
        totals.1 = totals
            .1
            .checked_add(b.total_scalar_work)
            .ok_or("work overflow")?;
    }
    let mut construction = 0u64;
    let mut buffers = 0u64;
    for (bytes, work) in bounds.values() {
        buffers = buffers.max(*bytes);
        construction = construction
            .checked_add(*work)
            .ok_or("construction overflow")?;
    }
    let total = construction
        .checked_mul(3)
        .and_then(|v| v.checked_add(3 * 64_000_000_000))
        .ok_or("case work overflow")?;
    if buffers > CASE_BUFFER_LIMIT || total > CASE_WORK_CAP {
        return Err("prospective case limits refuse".into());
    }
    Ok(
        json!({"source_roles":6,"endpoints":3,"construction_passes":3,
        "construction_scalar_work_per_pass":construction,"metric_scalar_work_reserved":3*64_000_000_000u64,
        "case_scalar_work_forecast":total,"case_buffer_forecast_bytes":buffers,
        "case_scalar_work_limit":CASE_WORK_CAP,"case_buffer_limit_bytes":CASE_BUFFER_LIMIT,
        "heldout_metrics_evaluated":false,"reference_realizability_proved":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rows() -> Vec<Value> {
        ["full","mini","forward"].into_iter().flat_map(|mode|["prior","supplement"].map(move|role|json!({
            "mode":mode,"role":role,"heldout_metrics_evaluated":false,"bounds":{
            "parent_buffer_bytes":1,"buffer_bytes":3*1024*1024*1024u64,"inherited_scalar_work":1,
            "construction_scalar_work":1,"optimization_scalar_work":1,"total_scalar_work":200_000_000_000u64,
            "prediction_scalar_work_per_query":1,"cells":1,"eligible_cells":1,"active_blocks":1,
            "bank_size":49,"max_states":1,"max_gradient_batches":0}}))).collect()
    }
    #[test]
    fn all_roles_repeated_constructions_and_metric_caps_are_charged() -> Result<()> {
        let out = calculate(&rows())?;
        assert_eq!(out["case_scalar_work_forecast"], 3_792_000_000_000u64);
        assert_eq!(out["case_buffer_forecast_bytes"], 6 * 1024 * 1024 * 1024u64);
        for variant in 0..4 {
            let mut bad = rows();
            match variant {
                0 => {
                    bad.pop();
                }
                1 => bad[0] = bad[1].clone(),
                2 => bad[0]["heldout_metrics_evaluated"] = json!(true),
                _ => bad[0]["bounds"]["buffer_bytes"] = json!(56 * 1024 * 1024 * 1024u64),
            }
            assert!(calculate(&bad).is_err());
        }
        Ok(())
    }
}
