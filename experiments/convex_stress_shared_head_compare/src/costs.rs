//! Label-independent complete call forecasts. Native counters check actual calls.
use super::*;
pub(crate) fn examples(rows: &[data::Example], v: usize) -> Result<()> {
    if rows.is_empty()
        || rows.iter().any(|e| {
            e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || e.targets[1] as usize >= v
                || !e.prompt.is_empty()
                || e.tokens.iter().any(|&x| x < 0 || x as usize >= v)
        })
    {
        return Err("changed model requires original two-token single-target layout".into());
    }
    Ok(())
}
pub(crate) fn calls(task: &data::Task) -> Result<Value> {
    let (mut predictions, mut rows) = (0u64, 0u64);
    for split in task.splits.values() {
        examples(split, task.vocab)?;
        rows += split.len() as u64;
        for e in split {
            predictions += 1 + u64::from(e.field.is_some());
        }
    }
    Ok(
        json!({"old_metric_calls":predictions,"new_metric_calls":predictions,
        "old_raw_audit_calls":rows,"new_raw_audit_calls":rows,"independent_new_calls":rows,
        "raw_records":rows,"generation_calls":0,"layout":"two-token single-target; all original field calls included"}),
    )
}
pub(crate) fn records(task: &data::Task) -> Result<Vec<fresh::Record>> {
    let train = task.splits.get("train").ok_or("missing training records")?;
    examples(train, task.vocab)?;
    Ok(train
        .iter()
        .enumerate()
        .map(|(id, e)| fresh::Record {
            id,
            cell: e.tokens[0] as usize * task.vocab + e.tokens[1] as usize,
            label: e.targets[1] as usize,
        })
        .collect())
}
pub(crate) fn forecast(
    parent: &Value,
    sha: &str,
    task: &data::Task,
    mode: &str,
    bytes: u64,
    baseline_bytes: u64,
) -> Result<Value> {
    if bytes > 1 << 30 || baseline_bytes > 1 << 30 {
        return Err("JSON parse size refused".into());
    }
    let policy = policy(mode)?;
    let original = records(task)?;
    let mut roles = Vec::new();
    let (mut old_work, mut new_work, mut buffers, mut prediction) = (0u64, 0u64, 0u64, 0u64);
    for role in ["prior", "supplement"] {
        if parent[role].is_null() {
            continue;
        }
        let input = fresh::inspection::extract(parent, role, sha)?;
        if input.records != original || input.vocabulary != task.vocab || input.n != task.vocab {
            return Err("actual source records differ from original training pool".into());
        }
        let new = fresh::limits::forecast(&input, &policy)?;
        let old: old::Bounds = serde_json::from_value(parent[role]["bounds"].clone())?;
        old_work = old_work
            .checked_add(old.total_scalar_work)
            .ok_or("old cost overflow")?;
        new_work = new_work
            .checked_add(new.total_kernel_work)
            .ok_or("new cost overflow")?;
        buffers += old.buffer_bytes + 3 * new.buffer_bytes;
        prediction = prediction
            .max(old.prediction_scalar_work_per_query)
            .max(new.prediction_work);
        roles.push(json!({"role":role,"input_sha256":fresh::digest(&input)?,"old":old,"new":new}));
    }
    if roles.is_empty() {
        return Err("empty original endpoint".into());
    }
    // Conservative simultaneous JSON reserve, full old/new coefficients and
    // source validations. Hash/serialization work is reported separately.
    buffers += 32 * (bytes + baseline_bytes) + (2u64 << 30);
    let calls = calls(task)?;
    let total_calls = [
        "old_metric_calls",
        "new_metric_calls",
        "old_raw_audit_calls",
        "new_raw_audit_calls",
        "independent_new_calls",
    ]
    .iter()
    .map(|k| calls[k].as_u64().unwrap())
    .sum::<u64>();
    let metric_work = prediction
        .checked_mul(total_calls)
        .ok_or("metric cost overflow")?;
    let total = 2 * old_work + 3 * new_work + metric_work;
    if buffers > BUFFER_LIMIT || total > WORK_LIMIT {
        return Err(
            format!("complete endpoint forecast refused: {total} work, {buffers} bytes").into(),
        );
    }
    Ok(
        json!({"roles":roles,"calls":calls,"old_construction_work":old_work,"new_training_work":new_work,
        "old_construction_passes_reserved":2,"new_training_passes_reserved":3,
        "prediction_work_per_call":prediction,"metric_kernel_work":metric_work,"total_kernel_work":total,
        "buffer_bytes":buffers,"parent_bytes":bytes,"baseline_bytes":baseline_bytes}),
    )
}
