//! Original data reconstruction and shape-only resource checks. No training or
//! held-out prediction is performed by this module.
use crate::{Result, data, engine, layout::Dims, memory};
use serde_json::{Value, json};

pub const SUITE: &str = include_str!("../../convex_stress/suite.json");

pub fn rows() -> Result<Vec<Value>> {
    let suite: Value = serde_json::from_str(SUITE)?;
    let rows = suite["tasks"].as_array().ok_or("missing original tasks")?;
    if rows.len() != 35 {
        return Err("original stress inventory must contain 35 tasks".into());
    }
    Ok(rows.clone())
}

fn integer(v: &Value, key: &str) -> Result<usize> {
    usize::try_from(v[key].as_u64().ok_or("missing task dimension")?).map_err(Into::into)
}

pub fn dimensions(row: &Value, task: &data::Task) -> Result<Dims> {
    let c = &row["case"];
    let d = integer(c, "width")?;
    let ctx = task
        .splits
        .values()
        .flatten()
        .map(|e| {
            e.tokens
                .len()
                .max(e.prompt.len() + e.limit.saturating_sub(1))
        })
        .max()
        .ok_or("empty task")?;
    let dims = Dims {
        vocab: task.vocab,
        d,
        heads: integer(c, "heads")?,
        layers: integer(c, "layers")?,
        dff: d.checked_mul(4).ok_or("width overflow")?,
        ctx,
    };
    memory::validate(dims)?;
    Ok(dims)
}

pub fn inspect(row: &Value) -> Result<Value> {
    let task = data::build(row)?;
    let dims = dimensions(row, &task)?;
    let batch = integer(&row["case"], "batch")?;
    let mut splits = serde_json::Map::new();
    let mut data_bytes = 0usize;
    for (name, examples) in &task.splits {
        let mut targets = 0usize;
        for e in examples {
            targets += engine::validate_example(dims, e)?;
            data_bytes += std::mem::size_of::<data::Example>()
                + e.task.capacity()
                + 8 * (e.tokens.capacity()
                    + e.targets.capacity()
                    + e.prompt.capacity()
                    + e.answer.capacity())
                + e.changes.capacity();
        }
        splits.insert(
            name.clone(),
            json!({"examples":examples.len(),"supervised_targets":targets}),
        );
    }
    if task.splits["train"].len() != integer(row, "train_examples")? {
        return Err("original training count changed".into());
    }
    let plan = match memory::microbatch(dims, batch) {
        Ok(n) => {
            json!({"status":"admitted","microbatch":n,"buffer_bytes_upper_bound":memory::check(dims,n)?})
        }
        Err(e) => json!({"status":"rejected","reason":e.to_string()}),
    };
    Ok(
        json!({"task":row["name"],"tier":row["tier"],"data_fingerprints":row["data_fingerprints"],
        "primary_split":task.selection,"vocab":dims.vocab,"width":dims.d,"layers":dims.layers,
        "heads":dims.heads,"context":dims.ctx,"parameters":dims.len(),"optimizer_batch":batch,
        "dataset_vector_bytes":data_bytes,"splits":splits,"memory":plan}),
    )
}
