//! Original categorical metrics with actual per-call routes and exact-input
//! coverage. These diagnostics are produced after structural selection seals.
use super::*;
struct Tracked<'a> {
    model: &'a fallback::Model,
    observed: &'a BTreeSet<Vec<i64>>,
    known: std::cell::RefCell<Vec<bool>>,
    exact: std::cell::RefCell<Vec<bool>>,
}
impl metrics::Predictor for Tracked<'_> {
    fn vocab(&self) -> usize {
        self.model.candidate().vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.known.borrow_mut().push(
            self.model
                .known(prefix)
                .expect("forecast validated categorical domain"),
        );
        self.exact.borrow_mut().push(self.observed.contains(prefix));
        self.model
            .logits(prefix)
            .expect("forecast validated fallback domain")
    }
}
fn causal_work(m: &component::model::Model) -> Result<u64> {
    let n = m
        .blocks
        .iter()
        .map(|b| b.states())
        .max()
        .ok_or("empty causal head")? as u64;
    Ok(8 * n * m.vocabulary as u64)
}
fn parent_work(parent: &fallback::Parent) -> Result<u64> {
    match parent {
        fallback::Parent::Causal(m) => causal_work(m),
        fallback::Parent::Counts(m) => {
            let value = serde_json::to_value(m)?;
            let shared = value["shared"]["active"]
                .as_array()
                .ok_or("missing shared keys")?
                .len();
            let local = value["local"]
                .as_object()
                .ok_or("missing local keys")?
                .values()
                .map(|v| {
                    v["active"]
                        .as_array()
                        .map(Vec::len)
                        .ok_or("missing local active keys")
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            let (v, context) = parent.shape();
            let items = (context + 4 * v + 1) as u64;
            Ok(16 * (shared + local + 1) as u64 * v as u64
                + 16 * items * (1 + items.ilog2() as u64))
        }
    }
}
pub(crate) fn forecast(task: &data::Task, model: &fallback::Model) -> Result<u64> {
    let (mut calls, mut tokens) = (0u64, 0u64);
    for examples in task.splits.values() {
        if examples.is_empty() {
            return Err("empty original metric split".into());
        }
        for e in examples {
            if e.tokens.is_empty()
                || e.tokens.len() > model.prediction_limit()
                || e.prompt
                    .len()
                    .checked_add(e.limit)
                    .is_none_or(|n| n > model.prediction_limit())
                || e.tokens.len() != e.targets.len()
                || e.changes.len() != e.targets.len()
                || e.tokens
                    .iter()
                    .chain(&e.prompt)
                    .any(|&t| t < 0 || t as usize >= task.vocab)
                || e.targets
                    .iter()
                    .any(|&t| t != -100 && (t < 0 || t as usize >= task.vocab))
            {
                return Err("original metric input exceeds common prediction domain".into());
            }
            for (i, _) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                calls += 1;
                tokens += (i + 1) as u64;
            }
            if e.field.is_some() {
                calls += 1;
                tokens += e.tokens.len() as u64;
            }
            if !e.prompt.is_empty() {
                calls += e.limit as u64;
                tokens += (e.limit * (e.prompt.len() + e.limit)) as u64;
            }
        }
    }
    let per_call = causal_work(model.candidate())? + parent_work(model.parent())?;
    let work = calls
        .checked_mul(per_call)
        .and_then(|n| n.checked_add(64 * tokens))
        .ok_or("metric forecast overflow")?;
    if work > 2_000_000_000 {
        return Err("both-head original metrics exceed2B scalar-work forecast".into());
    }
    Ok(work)
}
pub(crate) fn evaluate(x: &inputs::Inputs, bundle: &imported::Bundle) -> Result<Vec<Value>> {
    let observed: BTreeSet<Vec<i64>> = x
        .configuration
        .as_ref()
        .map(|c| {
            c.discovery
                .input
                .observations
                .iter()
                .map(|s| s.tokens.iter().map(|&t| t as i64).collect())
                .collect()
        })
        .unwrap_or_default();
    let mut rows = vec![];
    for imported in &bundle.rows {
        if imported["status"] != "wrapped" {
            continue;
        }
        let key = imported["reference"]
            .as_str()
            .ok_or("missing wrapper reference")?;
        let model = bundle.models.get(key).ok_or("missing actual wrapper")?;
        let mut row = imported.clone();
        let work = match forecast(&x.task, model) {
            Ok(w) => w,
            Err(e) => {
                row["status"] = json!("evaluation_refused");
                row["error"] = json!(e.to_string());
                rows.push(row);
                continue;
            }
        };
        let (mut actual, mut masks) = (serde_json::Map::new(), serde_json::Map::new());
        for (split, examples) in &x.task.splits {
            let tracked = Tracked {
                model,
                observed: &observed,
                known: std::cell::RefCell::new(vec![]),
                exact: std::cell::RefCell::new(vec![]),
            };
            actual.insert(split.clone(), metrics::evaluate(&tracked, examples)?);
            let known = tracked.known.into_inner();
            let exact = tracked.exact.into_inner();
            if known.len() != exact.len() {
                return Err("prediction masks lost alignment".into());
            }
            masks.insert(split.clone(),json!({"calls":known.len(),"supported_calls":known.iter().filter(|&&b|b).count(),
                "unsupported_calls":known.iter().filter(|&&b|!b).count(),"exact_observed_calls":exact.iter().filter(|&&b|b).count(),
                "supported_mask":known,"parent_route_mask":known.iter().map(|&b|!b).collect::<Vec<_>>(),"exact_observed_input_mask":exact,
                "order":"original teacher-forced, pooled-field and greedy prediction call order"}));
        }
        let actual = Value::Object(actual);
        let loss = actual["train"]["loss"]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("missing original CE")?;
        if (loss
            - imported["train_ce"]
                .as_f64()
                .ok_or("missing saved training CE")?)
        .abs()
            > 1e-8
        {
            return Err("ordinary raw CE changed in the original evaluator".into());
        }
        let (primary, all) = passes(&actual, &x.task.selection)?;
        row["status"] = json!("evaluated");
        row["metrics"] = actual;
        row["primary_split"] = json!(x.task.selection);
        row["primary_pass"] = json!(primary);
        row["all_splits_pass"] = json!(all);
        row["prediction_masks"] = json!(masks);
        row["metric_scalar_work_forecast"] = json!(work);
        row["unseen_identification"] = json!(false);
        rows.push(row);
    }
    Ok(rows)
}
