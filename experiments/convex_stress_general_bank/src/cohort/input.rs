//! Preserve frozen data order; map it explicitly to the canonical gradient memory.
use super::*;
use std::collections::BTreeSet;
pub(super) struct Inputs {
    pub task: data::Task,
    pub row: Value,
    pub problem: Problem,
    pub shape: Shape,
    pub explicit_bytes: u64,
    pub canonical_indices: Vec<usize>,
}
pub(super) fn assemble(task: data::Task, row: Value) -> Result<Inputs> {
    let decoders = decoder_bank(task.vocab)?;
    let shape = Shape {
        side: task.vocab,
        states: 2 * task.vocab - 1,
        vocab: task.vocab,
    };
    shape.validate()?;
    convex_stress_relabel::check_layout(&task)?;
    let observations: Vec<_> = task.splits["train"]
        .iter()
        .map(|e| Observation {
            row: e.tokens[1] as usize,
            column: e.tokens[0] as usize,
            label: e.targets[1] as usize,
        })
        .collect();
    let problem = Problem {
        side: shape.side,
        states: shape.states,
        vocab: shape.vocab,
        decoders,
        observations,
    };
    let mut canonical = problem.observations.clone();
    canonical.sort();
    let canonical_indices = problem
        .observations
        .iter()
        .map(|t| {
            canonical
                .binary_search(t)
                .map_err(|_| "missing canonical sample")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let limits: search::Limits = serde_json::from_value(protocol()?["search"].clone())?;
    let mut labels = BTreeSet::new();
    let mut certificate = 0;
    for &anchor in &problem.observations {
        if labels.insert(anchor.label) {
            let c = propagation::Config {
                anchor,
                steps: limits.sweeps,
                max_checks: limits.producer_checks,
                max_branches: limits.branches,
                memory_limit: limits.memory_bytes,
            };
            certificate = certificate.max(propagation::preflight(&problem, c)?.memory_bound);
        }
    }
    let explicit_bytes = general::memory_bound(shape, problem.observations.len(), certificate)?
        .checked_add(3 * certificate)
        .and_then(|v| v.checked_add(128 * 1024 * 1024))
        .ok_or("cohort memory bound overflow")?;
    if explicit_bytes > RUN_MEMORY_BYTES {
        return Err("cohort forecast exceeds 64 GiB".into());
    }
    Ok(Inputs {
        task,
        row,
        problem,
        shape,
        explicit_bytes,
        canonical_indices,
    })
}
pub(super) fn inputs(c: &Case) -> Result<Inputs> {
    if !cases()?.contains(c) {
        return Err("unplanned case".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing original suite")?
        .iter()
        .find(|r| r["name"] == c.task)
        .ok_or("missing original task")?
        .clone();
    assemble(data::build(&row)?, row)
}
pub(super) fn task_record(c: &Case, x: &Inputs) -> Value {
    json!({"task":c.task,"shape":x.shape,"decoder_bank":x.problem.decoders,"training_examples":x.problem.observations.len(),
        "data_fingerprints":x.row["data_fingerprints"],"selection":x.task.selection,"explicit_bytes_upper_bound":x.explicit_bytes})
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Schedule {
    pub count: usize,
    pub batch: usize,
    pub per_epoch: usize,
    pub cadence: usize,
    pub limit: usize,
}
pub(super) fn schedule(c: &Case, x: &Inputs) -> Result<Schedule> {
    let count = x.problem.observations.len();
    let batch = count_value(&x.row["case"], "batch")?;
    if count == 0 || batch == 0 {
        return Err("empty schedule".into());
    }
    let per_epoch = count.div_ceil(batch);
    let cadence = count_value(&x.row["case"], "observe")?.min(per_epoch);
    let limit = count_value(
        &protocol()?,
        match mode(c)? {
            Mode::Full => "full_updates",
            Mode::Mini => "minibatch_updates",
            Mode::Forward => "forward_updates",
        },
    )?;
    if cadence == 0 || limit == 0 {
        return Err("empty cadence or update cap".into());
    }
    Ok(Schedule {
        count,
        batch,
        per_epoch,
        cadence,
        limit,
    })
}
fn count_value(v: &Value, k: &str) -> Result<usize> {
    super::count(v, k)
}
pub(super) fn batch(
    x: &Inputs,
    s: &Schedule,
    mode: Mode,
    update: usize,
    rng: &mut Random,
    order: &mut [usize],
) -> Vec<usize> {
    let within = update % s.per_epoch;
    if mode == Mode::Mini && within == 0 {
        rng.shuffle(order);
    }
    let selected = if mode == Mode::Mini {
        &order[within * s.batch..((within + 1) * s.batch).min(s.count)]
    } else {
        order
    };
    selected.iter().map(|&i| x.canonical_indices[i]).collect()
}
struct Predictor<'a>(&'a Model);
impl metrics::Predictor for Predictor<'_> {
    fn vocab(&self) -> usize {
        self.0.shape.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let m = self.0;
        let start = m.shape.cell(prefix[1] as usize, prefix[0] as usize) * m.shape.vocab;
        m.output[start..start + m.shape.vocab].to_vec()
    }
}
pub(super) fn evaluate(model: &Model, examples: &[data::Example]) -> Result<Value> {
    // Check split layout here, after training decisions, before indexing logits.
    for e in examples {
        if e.tokens.len() != 2
            || e.targets.len() != 2
            || e.targets[0] != -100
            || e.targets[1] < 0
            || e.targets[1] as usize >= model.shape.vocab
            || e.changes.len() != 2
            || !e.prompt.is_empty()
            || e.field.is_some()
            || e.tokens
                .iter()
                .any(|&t| t < 0 || t as usize >= model.shape.side)
        {
            return Err("unsupported pair evaluation layout".into());
        }
    }
    metrics::evaluate(&Predictor(model), examples)
}
pub(super) fn metrics_all(model: &Model, task: &data::Task) -> Result<Value> {
    let mut out = serde_json::Map::new();
    for (split, examples) in &task.splits {
        out.insert(split.clone(), evaluate(model, examples)?);
    }
    Ok(Value::Object(out))
}
pub(super) fn fit_defect(model: &Model, p: &Problem) -> f64 {
    p.observations
        .iter()
        .flat_map(|t| {
            let first = model.shape.cell(t.row, t.column) * model.shape.vocab;
            model.output[first..first + model.shape.vocab]
                .iter()
                .enumerate()
                .map(move |(c, &a)| (a - f64::from(c == t.label)).abs())
        })
        .fold(0., f64::max)
}
