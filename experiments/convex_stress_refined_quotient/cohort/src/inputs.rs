//! The unchanged native generator verifies original fingerprints; only training
//! pairs and the original batch size are retained by this adapter.
use super::*;
pub(crate) struct Inputs {
    pub input: quotient::Input,
    pub policies: [training::Policy; 3],
    pub record: Value,
}
pub(crate) fn assemble(
    task: convex_stress_relabel::data::Task,
    name: &str,
    batch: usize,
    fingerprints: Value,
) -> Result<Inputs> {
    super::name(name)?;
    convex_stress_relabel::check_layout(&task)?;
    let input = quotient::Input {
        n: task.vocab,
        vocabulary: task.vocab,
        observations: task.splits["train"]
            .iter()
            .map(|e| {
                [
                    e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
                    e.targets[1] as usize,
                ]
            })
            .collect(),
        policy: quotient::Policy::default(),
    };
    let policies = training::policies(input.observations.len(), batch)?;
    let record = json!({"task":name,"vocabulary":task.vocab,"observations":input.observations.len(),
        "input_sha256":hash(&serde_json::to_vec(&input)?),"original_batch":batch,"policies":policies,
        "data_fingerprints":fingerprints,"construction_inputs":"original training pairs and raw labels"});
    Ok(Inputs {
        input,
        policies,
        record,
    })
}
pub(crate) fn build(suite: &Value, name: &str) -> Result<Inputs> {
    let rows: Vec<_> = suite["tasks"]
        .as_array()
        .ok_or("missing original tasks")?
        .iter()
        .filter(|r| r["name"] == name)
        .collect();
    if rows.len() != 1 {
        return Err("unknown or repeated original task".into());
    }
    let row = rows[0];
    let batch = row["case"]["batch"]
        .as_u64()
        .ok_or("missing original minibatch size")? as usize;
    assemble(
        convex_stress_relabel::data::build(row)?,
        name,
        batch,
        row["data_fingerprints"].clone(),
    )
}
pub(crate) fn check(inv: &Value, task: &str, x: &Inputs) -> Result<usize> {
    let inputs = inv["inputs"]
        .as_array()
        .ok_or("missing refined input inventory")?;
    let indices: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter(|(_, x)| x["task"] == task)
        .map(|(i, _)| i)
        .collect();
    if indices.len() != 1 || inputs[indices[0]] != x.record {
        return Err("refined original input or batch policy differs".into());
    }
    Ok(indices[0])
}
