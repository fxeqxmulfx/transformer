//! Conservative complete-run accounting. Component preflights still run
//! before their allocations/loops. The multiplier covers the nested binder
//! replay tree: at most13 search reconstructions and7 training runs per tier,
//! plus independent interpreters and complete-table checks. Hashing, JSON
//! and dataset reconstruction are covered by whole-process time/RSS/I/O
//! measurements, not misrepresented as arithmetic-kernel operation counts.
use crate::*;
use serde_json::{Value, json};

fn total(value: &Value, keys: &[&str]) -> Result<u64> {
    let mut sum = 0u64;
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                let next = if keys.contains(&key.as_str()) {
                    value.as_u64().ok_or("noninteger work/buffer forecast")?
                } else {
                    total(value, keys)?
                };
                sum = sum.checked_add(next).ok_or("forecast sum overflow")?;
            }
        }
        Value::Array(values) => {
            for value in values {
                sum = sum
                    .checked_add(total(value, keys)?)
                    .ok_or("forecast sum overflow")?;
            }
        }
        _ => {}
    }
    Ok(sum)
}
pub fn report(stage: &runner::Stage, measured: &Value) -> Result<Value> {
    let source = serde_json::to_value(&stage.model)?;
    let construction = total(&source, &["search_forecast", "scalar_work_forecast"])?;
    let evaluation = total(measured, &["scalar_work_forecast"])?;
    let buffers = total(&source, &["buffer_forecast"])?;
    let complete = construction
        .checked_mul(32)
        .and_then(|n| evaluation.checked_mul(2).and_then(|s| n.checked_add(s)))
        .ok_or("complete run forecast overflow")?;
    Ok(
        json!({"construction_component_forecasts":construction,"construction_replay_factor":32,
        "scoring_all_modes_forecast":evaluation,"scoring_and_replay_factor":2,
        "complete_stage_score_replay_kernel_work_bound":complete,
        "sum_component_buffer_forecasts":buffers,"process_address_space_bytes":68_719_476_736u64,
        "scope":"arithmetic kernels in one complete stage/score/replay run, including all attempted parent families; fresh repeats are additional work; hashing/JSON/data reconstruction measured separately by process resources"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_attempts_are_charged_and_invalid_forecasts_refuse() -> Result<()> {
        let v = json!({"search_forecast":7,"parent":{"search_forecast":11},
            "endpoints":[{"scalar_work_forecast":13},{"scalar_work_forecast":17}],"buffer_forecast":29});
        assert_eq!(total(&v, &["search_forecast", "scalar_work_forecast"])?, 48);
        assert_eq!(total(&v, &["buffer_forecast"])?, 29);
        assert!(total(&json!({"search_forecast":-1}), &["search_forecast"]).is_err());
        assert!(
            total(
                &json!([{"search_forecast":u64::MAX},{"search_forecast":1}]),
                &["search_forecast"]
            )
            .is_err()
        );
        Ok(())
    }
}
