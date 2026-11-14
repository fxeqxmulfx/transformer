//! Reconstruct every fresh saved endpoint and reproduce its entire original
//! metric object before emitting per-example diagnostic classifications.
use super::*;
use serde::Serialize;
struct Pair {
    prior: head::Model,
    supplement: Option<head::Model>,
}
impl Pair {
    fn route(&self, cell: usize) -> (&head::Model, &'static str) {
        if !self.prior.source().known(cell)
            && let Some(m) = &self.supplement
            && m.source().known(cell)
        {
            (m, "supplement")
        } else {
            (&self.prior, "prior")
        }
    }
    fn selected(&self, prefix: &[i64]) -> (&head::Model, &'static str, usize) {
        let cell = prefix[prefix.len() - 1] as usize
            + self.prior.source().n * prefix[prefix.len() - 2] as usize;
        let (m, role) = self.route(cell);
        (m, role, cell)
    }
}
impl metrics::Predictor for Pair {
    fn vocab(&self) -> usize {
        self.prior.source().vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.selected(prefix)
            .0
            .logits(prefix)
            .expect("validated original pair")
    }
}
struct Old<'a>(&'a Pair);
impl metrics::Predictor for Old<'_> {
    fn vocab(&self) -> usize {
        self.0.prior.source().vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.0
            .selected(prefix)
            .0
            .parent()
            .logits(prefix)
            .expect("validated original parent")
    }
}
#[derive(Default, Serialize)]
struct Bucket {
    targets: usize,
    correct: usize,
    parent_correct: usize,
    changed: usize,
    fixed: usize,
    regressed: usize,
}

/// Exact original metric convention, including first-index and signed-zero ties.
pub(crate) fn argmax(z: &[f64]) -> Result<usize> {
    if z.is_empty() || z.iter().any(|x| !x.is_finite()) {
        return Err("invalid actual logits".into());
    }
    Ok((0..z.len())
        .max_by(|&a, &b| z[a].total_cmp(&z[b]).then(b.cmp(&a)))
        .unwrap())
}
fn kind(m: &head::Model, cell: usize) -> &'static str {
    let (block, state) = m.source().location(cell);
    if state < m.source().blocks[block].observed.len() {
        return "observed";
    }
    if !m.source().known(cell) {
        return "unsupported";
    }
    if m.rows()[cell].is_some() {
        return "covered";
    }
    match m.geometry()[block].disposition {
        head::Disposition::Empty | head::Disposition::SingleLabel => "inactive",
        head::Disposition::OverlapRefused => "overlap_refused",
        head::Disposition::Separated => "uncovered",
    }
}
fn split(model: &Pair, examples: &[data::Example], expected: &Value, old: &Value) -> Result<Value> {
    if metrics::evaluate(model, examples)? != *expected
        || metrics::evaluate(&Old(model), examples)? != *old
    {
        return Err("actual full original metrics differ from reviewed results".into());
    }
    let mut buckets = BTreeMap::<String, Bucket>::new();
    let mut errors = Vec::new();
    for (id, e) in examples.iter().enumerate() {
        let label = e.targets[1] as usize;
        let (m, role, cell) = model.selected(&e.tokens);
        let z = m.logits(&e.tokens)?;
        let prior = m.parent().logits(&e.tokens)?;
        let prediction = argmax(&z)?;
        let before = argmax(&prior)?;
        let category = kind(m, cell);
        let bucket = buckets.entry(format!("{role}:{category}")).or_default();
        bucket.targets += 1;
        bucket.correct += usize::from(prediction == label);
        bucket.parent_correct += usize::from(before == label);
        bucket.changed += usize::from(prediction != before);
        bucket.fixed += usize::from(before != label && prediction == label);
        bucket.regressed += usize::from(before == label && prediction != label);
        if prediction != label {
            let (block, state) = m.source().location(cell);
            let b = &m.source().blocks[block];
            let points: Vec<_> = b
                .observed
                .iter()
                .zip(&b.observed_labels)
                .map(|(&c, &y)| (((c % m.source().n) as i64, (c / m.source().n) as i64), y))
                .collect();
            errors.push(json!({"record":id,"cell":cell,"input":e.tokens,"target":label,"prediction":prediction,
                "parent_prediction":before,"role":role,"context":category,"block":block,"state":state,
                "raw_ce":head::kernel::training::ce(&z,label),"parent_raw_ce":head::kernel::training::ce(&prior,label),
                "target_geometry":geometry::describe(&points,label,((cell%m.source().n) as i64,(cell/m.source().n) as i64))?}));
        }
    }
    let correct: usize = buckets.values().map(|b| b.correct).sum();
    let parent_correct: usize = buckets.values().map(|b| b.parent_correct).sum();
    if expected["native_correct"] != correct
        || expected["targets"] != examples.len()
        || old["native_correct"] != parent_correct
    {
        return Err("diagnostic raw counts differ from original metrics".into());
    }
    Ok(
        json!({"full_metrics_reproduced":true,"targets":examples.len(),"correct":correct,"errors":errors.len(),
        "parent_correct":parent_correct,"buckets":buckets,"error_records":errors}),
    )
}
pub(crate) fn endpoint(
    dir: &Path,
    data: &data::Task,
    row: &Value,
    seal: &Value,
    manifest: &BTreeMap<String, String>,
) -> Result<Value> {
    let selected: Vec<_> = seal["models"]
        .as_array()
        .ok_or("missing sealed models")?
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s["reference"] == row["parent_reference"] && s["training_mode"] == row["training_mode"]
        })
        .collect();
    if selected.len() != 1 {
        return Err("missing or duplicate selected model".into());
    }
    let (index, selected) = selected[0];
    let path = dir.join(format!("model-{index}.json"));
    pinned(&path, manifest)?;
    if selected["file"] != format!("model-{index}.json")
        || selected["model_sha256"] != row["model_sha256"]
        || digest(&path)? != row["model_sha256"]
    {
        return Err("selected model identity differs".into());
    }
    let mut envelope: Value = read(&path)?;
    let mut load = |role: &str| -> Result<Option<head::Model>> {
        let value = envelope
            .as_object_mut()
            .ok_or("missing saved envelope")?
            .remove(role)
            .ok_or("missing source role")?;
        let info = &selected["model_info"][role];
        if value.is_null() {
            if !info.is_null() {
                return Err("saved source role disappeared".into());
            }
            Ok(None)
        } else {
            Ok(Some(head::Model::load(
                value,
                &serde_json::from_value(info["binding"].clone())?,
            )?))
        }
    };
    let prior = load("prior")?.ok_or("missing prior model")?;
    let supplement = load("supplement")?;
    let model = Pair { prior, supplement };
    let models: Vec<_> = std::iter::once(&model.prior)
        .chain(model.supplement.iter())
        .collect();
    let n = data.vocab;
    if models
        .iter()
        .any(|m| m.source().n != n || m.source().vocabulary != n)
    {
        return Err("original source domain differs".into());
    }
    for examples in data.splits.values() {
        if examples.is_empty()
            || examples.iter().any(|e| {
                e.tokens.len() != 2
                    || e.targets.len() != 2
                    || e.targets[0] != -100
                    || e.targets[1] < 0
                    || e.targets[1] as usize >= n
                    || !e.prompt.is_empty()
                    || e.tokens.iter().any(|&x| x < 0 || x as usize >= n)
            })
        {
            return Err("diagnosis requires original supervised pair layout".into());
        }
    }
    let states = models
        .iter()
        .flat_map(|m| m.source().blocks.iter().map(|b| b.states()))
        .max()
        .ok_or("missing source states")? as u64;
    let calls: u64 = data
        .splits
        .values()
        .flatten()
        .map(|e| 2 + u64::from(e.field.is_some()))
        .sum();
    let work = calls * 64 * states * (states + 2 * n as u64 + 10);
    if work > 128_000_000_000 {
        return Err("diagnostic prediction forecast exceeds128B".into());
    }
    let mut splits = serde_json::Map::new();
    for (name, examples) in &data.splits {
        splits.insert(
            name.clone(),
            split(
                &model,
                examples,
                &row["metrics"][name],
                &row["parent_metrics"][name],
            )?,
        );
    }
    pinned(&path, manifest)?;
    Ok(
        json!({"task":row["task"],"mode":row["training_mode"],"model_sha256":row["model_sha256"],
        "source_reconstruction_performed":true,"source_reconstruction_bounds":models.iter().map(|m|m.bounds()).collect::<Vec<_>>(),
        "additional_prediction_work_bound":work,"splits":splits}),
    )
}
