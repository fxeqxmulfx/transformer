//! Fresh scoring of actual selected raw or conservatively extended tensors.
//! Original CE, pooled-field and greedy metrics come from the unchanged suite.
use super::*;
use selected::Model;
struct Tracked<'a> {
    model: &'a Model,
    masks: std::cell::RefCell<Vec<[bool; 6]>>,
}
impl metrics::Predictor for Tracked<'_> {
    fn vocab(&self) -> usize {
        self.model.vocabulary()
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let prior = self.model.prior();
        let cell = prefix[prefix.len() - 1] as usize + prior.n * prefix[prefix.len() - 2] as usize;
        let active = prior.latent(cell).is_some();
        let known = prior.known(cell);
        let (block, state) = prior.location(cell);
        let observed = active && state < prior.blocks[block].observed.len();
        let (new_active, new_known, routed) = match self.model {
            Model::Raw(_) => (false, false, false),
            Model::Extended(m) => {
                let route = m
                    .route(cell)
                    .expect("preflight validated categorical range");
                (
                    route.supplement_axes_encoded,
                    route.supplement_supported,
                    route.source == extension::Source::Supplement,
                )
            }
        };
        self.masks
            .borrow_mut()
            .push([active, known, observed, new_active, new_known, routed]);
        self.model
            .logits(prefix)
            .expect("preflight validated full original prediction domain")
    }
}
fn head_work(m: &component::Model) -> Result<u64> {
    let states = m
        .blocks
        .iter()
        .map(|b| b.states())
        .max()
        .ok_or("empty saved head")? as u64;
    let linear = 64_u64
        .checked_mul((m.n + m.vocabulary * (m.encoding.code.dimension + 1)) as u64)
        .ok_or("code work overflow")?;
    (8_u64
        .checked_mul(states)
        .and_then(|n| n.checked_mul(m.vocabulary as u64)))
    .and_then(|n| n.checked_add(linear))
    .ok_or_else(|| "head work forecast overflow".into())
}
pub(crate) fn forecast(task: &data::Task, model: &Model) -> Result<u64> {
    if task.vocab != model.vocabulary() || task.vocab != model.prior().n {
        return Err("original pair axes and raw vocabulary differ".into());
    }
    let (mut calls, mut tokens) = (0_u64, 0_u64);
    for examples in task.splits.values() {
        if examples.is_empty() {
            return Err("empty original metric split".into());
        }
        for e in examples {
            if e.tokens.is_empty()
                || e.tokens.len() > 4096
                || e.tokens.len() != e.targets.len()
                || e.tokens.len() != e.changes.len()
                || e.prompt.len().checked_add(e.limit).is_none_or(|n| n > 4096)
                || e.tokens
                    .iter()
                    .chain(&e.prompt)
                    .any(|&x| x < 0 || x as usize >= task.vocab)
                || e.targets
                    .iter()
                    .enumerate()
                    .any(|(i, &y)| y != -100 && (i == 0 || y < 0 || y as usize >= task.vocab))
                || (!e.prompt.is_empty() && e.prompt.len() < 2)
                || (e.field.is_some() && e.tokens.len() < 2)
            {
                return Err(
                    "original metric input exceeds the declared pair prediction domain".into(),
                );
            }
            for (i, _) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                calls = calls.checked_add(1).ok_or("prediction count overflow")?;
                tokens = tokens
                    .checked_add((i + 1) as u64)
                    .ok_or("prefix count overflow")?;
            }
            if e.field.is_some() {
                calls = calls.checked_add(1).ok_or("field call overflow")?;
                tokens = tokens
                    .checked_add(e.tokens.len() as u64)
                    .ok_or("field prefix overflow")?;
            }
            if !e.prompt.is_empty() {
                calls = calls
                    .checked_add(e.limit as u64)
                    .ok_or("generation call overflow")?;
                tokens = tokens
                    .checked_add((e.limit * (e.prompt.len() + e.limit)) as u64)
                    .ok_or("generation prefix overflow")?;
            }
        }
    }
    let mut per_call = head_work(model.prior())?;
    if let Model::Extended(m) = model {
        per_call = per_call
            .checked_add(head_work(m.supplement())?)
            .ok_or("combined head work overflow")?;
    }
    let work = calls
        .checked_mul(per_call)
        .and_then(|n| tokens.checked_mul(64).and_then(|p| n.checked_add(p)))
        .ok_or("metric work overflow")?;
    if work > 32_000_000_000 {
        return Err("original metric work exceeds the 32B forecast".into());
    }
    Ok(work)
}
pub(crate) fn evaluate(
    task: &str,
    x: &inputs::Inputs,
    bundle: &selected::Bundle,
) -> Result<Vec<Value>> {
    // Preflight every changed raw/extended endpoint before the first metric.
    let forecasts = bundle
        .models
        .iter()
        .map(|(key, m)| Ok((key.clone(), forecast(&x.task, m)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut rows = vec![];
    for (key, model) in &bundle.models {
        let choices = bundle.choices["choices"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["selected"] == *key)
            .collect::<Vec<_>>();
        let first = choices.first().ok_or("unselected raw model retained")?;
        if choices.iter().any(|r| {
            r["train_ce"] != first["train_ce"] || r["training_mode"] != first["training_mode"]
        }) {
            return Err("same model reference has conflicting training objective".into());
        }
        let (mut actual, mut masks) = (serde_json::Map::new(), serde_json::Map::new());
        for (split, examples) in &x.task.splits {
            let tracked = Tracked {
                model,
                masks: Default::default(),
            };
            actual.insert(split.clone(), metrics::evaluate(&tracked, examples)?);
            let calls = tracked.masks.into_inner();
            let count = |index: usize| calls.iter().filter(|m| m[index]).count();
            masks.insert(split.clone(),json!({"calls":calls.len(),"prior_active_calls":count(0),"prior_supported_calls":count(1),
                "exact_observed_calls":count(2),"supplement_active_calls":count(3),"supplement_supported_calls":count(4),"supplement_route_calls":count(5),
                "columns":["prior_active","prior_known","exact_observed","supplement_active","supplement_known","supplement_route"],
                "masks":calls,"order":"original teacher-forced, pooled-field and greedy prediction call order","unseen_correctness_certified":false}));
        }
        let actual = Value::Object(actual);
        let measured = actual["train"]["loss"]
            .as_f64()
            .ok_or("original raw CE omitted")?;
        let expected = first["train_ce"]
            .as_f64()
            .ok_or("selected raw CE omitted")?;
        if !measured.is_finite() || (measured - expected).abs() > 1e-8 {
            return Err("original raw training CE differs from selected history".into());
        }
        let (primary, aux) = passes(&actual, &x.task.selection)?;
        rows.push(json!({"task":task,"training_mode":first["training_mode"],"reference":key,"status":"evaluated",
            "train_ce":first["train_ce"],"actual_train_ce":measured,"primary_split":x.task.selection,
            "metrics":actual,"primary_pass":primary,"all_splits_pass":aux,"prediction_masks":masks,
            "metric_scalar_work_forecast":forecasts[key],"metric_provenance":"fresh_changed_model",
            "model_sha256":hash(&serde_json::to_vec(model)?),"raw_label_identification":false,"shared_role_target_proved":false}));
    }
    Ok(rows)
}
