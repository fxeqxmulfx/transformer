//! Original native data and declared metric groups, ported from49105730.
//! CM35 also uses this fixed output partition in observed kernel equations.
use super::*;
pub(crate) struct Inputs {
    pub task: data::Task,
    pub input: Option<kernel::discovery::Input>,
    pub policies: Option<[training::Policy; 3]>,
    pub readout: Option<Vec<usize>>,
    pub answers: Option<usize>,
    pub record: Value,
}
impl Inputs {
    pub(crate) fn partition(&self) -> Value {
        match (&self.readout, self.answers) {
            (Some(classes), Some(answers)) => json!({"classes":classes,"answers":answers}),
            _ => Value::Null,
        }
    }
    pub(crate) fn request(&self, prime: u16) -> Result<discovery::Request> {
        Ok(discovery::Request {
            input: self.input.clone().ok_or("missing original pair input")?,
            readout: self.readout.clone().ok_or("missing declared partition")?,
            answers: self.answers.ok_or("missing declared answer count")?,
            prime,
        })
    }
}
pub(crate) fn assemble(
    task: data::Task,
    name: &str,
    pair: bool,
    batch: usize,
    fingerprints: Value,
) -> Result<Inputs> {
    super::name(name)?;
    let (input, policies, readout, answers) = if pair {
        convex_stress_relabel::check_layout(&task)?;
        let train = task
            .splits
            .get("train")
            .filter(|v| !v.is_empty())
            .ok_or("missing original training pairs")?;
        let input = kernel::discovery::Input {
            n: task.vocab,
            vocabulary: task.vocab,
            policy: kernel::discovery::Policy::default(),
            observations: train
                .iter()
                .map(|e| {
                    [
                        e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
                        e.targets[1] as usize,
                    ]
                })
                .collect(),
        };
        let policies = training::policies(input.observations.len(), batch)?;
        let field = train[0].field.is_some();
        if train.iter().any(|e| e.field.is_some() != field) {
            return Err("mixed original metric readout".into());
        }
        let answers = if field { 97 } else { task.vocab };
        let q: Vec<usize> = if field {
            (0..task.vocab)
                .map(|c| (c as i64 - task.vocab as i64 / 2).rem_euclid(97) as usize)
                .collect()
        } else {
            (0..task.vocab).collect()
        };
        if q.iter().copied().collect::<BTreeSet<_>>().len() != answers {
            return Err("declared output partition is not surjective".into());
        }
        (Some(input), Some(policies), Some(q), Some(answers))
    } else {
        (None, None, None, None)
    };
    let record = json!({"task":name,"primary_split":task.selection,"vocabulary":task.vocab,
        "kernel_planned":pair,"original_batch":batch,"data_fingerprints":fingerprints,
        "input_sha256":input.as_ref().map(|x|serde_json::to_vec(x).map(|b|hash(&b))).transpose()?,"policies":policies,
        "readout":readout.as_ref().map(|classes|json!({"classes":classes,"answers":answers})),
        "readout_changes_features":pair,"readout_changes_ce":false,
        "construction_inputs":"original training pairs/raw labels and declared output partition"});
    Ok(Inputs {
        task,
        input,
        policies,
        readout,
        answers,
        record,
    })
}
pub(crate) fn build(suite: &Value, name: &str, planned: &[String]) -> Result<Inputs> {
    let rows: Vec<_> = suite["tasks"]
        .as_array()
        .ok_or("missing original tasks")?
        .iter()
        .filter(|r| r["name"] == name)
        .collect();
    if rows.len() != 1 {
        return Err("unknown or repeated original task".into());
    }
    assemble(
        data::build(rows[0])?,
        name,
        planned.iter().any(|t| t == name),
        rows[0]["case"]["batch"]
            .as_u64()
            .ok_or("missing original batch size")? as usize,
        rows[0]["data_fingerprints"].clone(),
    )
}
pub(crate) fn check(inv: &Value, task: &str, x: &Inputs) -> Result<()> {
    let records: Vec<_> = inv["inputs"]
        .as_array()
        .ok_or("missing input inventory")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect();
    if records.len() != 1 || *records[0] != x.record {
        return Err("original data, policy or declared metric changed".into());
    }
    Ok(())
}
