//! Ported readout_kernel/cohort at 5d4eee84 for CM40. Full original record
//! IDs/order/multiplicity are retained; only duplicate identical equations
//! are coalesced. The declared output partition is unchanged. No input
//! arithmetic rule or unseen answer is inspected by construction.
use super::*;
pub(crate) struct Inputs {
    pub input: kernel::discovery::Input,
    pub records: Vec<records::Record>,
    pub policies: [training::Policy; 3],
    pub readout: Vec<usize>,
    pub answers: usize,
    pub record: Value,
}
impl Inputs {
    pub(crate) fn partition(&self) -> Value {
        json!({"classes":self.readout,"answers":self.answers})
    }
    pub(crate) fn request(&self, prime: u16) -> discovery::Request {
        discovery::Request {
            input: self.input.clone(),
            readout: self.readout.clone(),
            answers: self.answers,
            prime,
        }
    }
}
pub(crate) fn assemble(
    task: convex_stress_relabel::data::Task,
    name: &str,
    batch: usize,
    fingerprints: Value,
) -> Result<Inputs> {
    super::name(name)?;
    kernel::limits::dimensions(task.vocab, task.vocab)?;
    let train = task
        .splits
        .get("train")
        .ok_or("missing original training split")?;
    if train.is_empty()
        || train.len() > records::MAX_RECORDS
        || train.iter().any(|e| {
            e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || e.targets[1] as usize >= task.vocab
                || !e.prompt.is_empty()
                || e.tokens.iter().any(|&x| x < 0 || x as usize >= task.vocab)
        })
    {
        return Err("unsupported bounded original training pair layout".into());
    }
    let mut input = kernel::discovery::Input {
        n: task.vocab,
        vocabulary: task.vocab,
        observations: vec![],
        policy: Default::default(),
    };
    let mut seen = std::collections::BTreeMap::new();
    let mut records = Vec::with_capacity(train.len());
    for (id, e) in train.iter().enumerate() {
        let cell = e.tokens[1] as usize + task.vocab * e.tokens[0] as usize;
        let label = e.targets[1] as usize;
        if let Some(&prior) = seen.get(&cell) {
            if prior != label {
                return Err("conflicting original raw targets for one input pair".into());
            }
        } else {
            seen.insert(cell, label);
            input.observations.push([cell, label]);
        }
        records.push(records::Record { id, cell, label });
    }
    records::validate(&input, &records)?;
    let policies = training::policies(records.len(), batch)?;
    let field = train[0].field.is_some();
    if train.iter().any(|e| e.field.is_some() != field) {
        return Err("mixed original metric readout".into());
    }
    let answers = if field { 97 } else { task.vocab };
    let readout: Vec<_> = (0..task.vocab)
        .map(|c| {
            if field {
                (c as i64 - task.vocab as i64 / 2).rem_euclid(97) as usize
            } else {
                c
            }
        })
        .collect();
    if readout
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != answers
    {
        return Err("declared original output partition is not surjective".into());
    }
    let record = json!({"task":name,"vocabulary":task.vocab,"observations":input.observations.len(),
        "original_records":records.len(),"records_sha256":hash(&serde_json::to_vec(&records)?),
        "record_multiplicity_retained":true,"input_pair_order":{"row":"last_token","column":"penultimate_token"},
        "input_sha256":hash(&serde_json::to_vec(&input)?),"original_batch":batch,"policies":policies,
        "readout":{"classes":readout,"answers":answers},"readout_changes_features":true,"readout_changes_ce":false,
        "data_fingerprints":fingerprints,"construction_inputs":"original training pairs/raw labels and declared output partition"});
    Ok(Inputs {
        input,
        records,
        policies,
        readout,
        answers,
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
        .ok_or("missing kernel input inventory")?;
    let indices: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter(|(_, x)| x["task"] == task)
        .map(|(i, _)| i)
        .collect();
    if indices.len() != 1 || inputs[indices[0]] != x.record {
        return Err("kernel original input or batch policy differs".into());
    }
    Ok(indices[0])
}
