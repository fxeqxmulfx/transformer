//! Original scoring of actual saved complex coefficients, including pooled
//! answers. Candidate readout compatibility does not identify an unknown target.
use super::*;
const MASK_MEANING: &str = "known: both categorical axes and latent support observed; observed: exact input in training dictionary; neither certifies an unseen label";
pub(crate) fn compatible(d: &quotient::Report, readout: &[usize]) -> bool {
    d.code.is_some()
        && readout.len() == d.request.input.vocabulary
        && d.code_fibers.iter().all(|fiber| {
            fiber
                .first()
                .is_none_or(|&first| fiber.iter().all(|&c| readout[c] == readout[first]))
        })
}
pub(crate) fn offers(
    task: &str,
    prime: u16,
    sha: &str,
    r: &binding::Report,
    eligible: bool,
) -> Result<Vec<comparison::Offer>> {
    let mut offers = vec![];
    if r.endpoints.len() != 3 {
        return Err("refined report lost a mode".into());
    }
    for (i, e) in r.endpoints.iter().enumerate() {
        match (&e.training, &e.audit, &e.refusal) {
            (Some(t), Some(a), None) => {
                if mode(t.policy.mode) != MODES[i]
                    || t.policy != r.policies[i]
                    || (t.final_ce - a.train_ce).abs() > 1e-8
                {
                    return Err("saved mode/CE/audit differs".into());
                }
                if eligible {
                    offers.push(comparison::Offer {
                        task: task.into(),
                        mode: t.policy.mode,
                        prime,
                        train_ce: a.train_ce,
                        model_sha256: hash(&serde_json::to_vec(&t.model)?),
                        stage_sha256: sha.into(),
                    });
                }
            }
            (None, None, Some(reason)) if !reason.is_empty() => {}
            _ => return Err("ambiguous actual mode availability".into()),
        }
    }
    Ok(offers)
}
struct Tracked<'a> {
    model: &'a component::Model,
    known: std::cell::RefCell<Vec<bool>>,
    observed: std::cell::RefCell<Vec<bool>>,
}
impl metrics::Predictor for Tracked<'_> {
    fn vocab(&self) -> usize {
        self.model.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let cell =
            prefix[prefix.len() - 1] as usize + self.model.n * prefix[prefix.len() - 2] as usize;
        self.known.borrow_mut().push(self.model.known(cell));
        let (block, state) = self.model.location(cell);
        self.observed
            .borrow_mut()
            .push(state + 1 < self.model.blocks[block].states());
        self.model.logits_cell(cell)
    }
}
fn forecast(task: &data::Task, model: &component::Model) -> Result<u64> {
    convex_stress_relabel::check_layout(task)?;
    if task.vocab != model.vocabulary
        || task.splits.values().any(|examples| {
            examples.is_empty()
                || examples.iter().any(|e| {
                    e.tokens.len() != 2
                        || e.targets.len() != 2
                        || e.changes.len() != 2
                        || e.targets[0] != -100
                        || e.targets[1] < 0
                        || e.targets[1] as usize >= task.vocab
                        || !e.prompt.is_empty()
                        || e.tokens.iter().any(|&x| x < 0 || x as usize >= model.n)
                })
        })
    {
        return Err("original metric split exceeds the saved pair-model domain".into());
    }
    let maximum = model
        .blocks
        .iter()
        .map(|b| b.states())
        .max()
        .ok_or("empty model")? as u64;
    let calls = task
        .splits
        .values()
        .map(|examples| {
            examples
                .iter()
                .map(|e| {
                    e.targets.iter().filter(|&&y| y >= 0).count() as u64
                        + u64::from(e.field.is_some())
                })
                .sum::<u64>()
        })
        .sum::<u64>();
    let work = 8 * calls * maximum * model.vocabulary as u64;
    if work > 32_000_000_000 {
        return Err("original metrics exceed32B scalar-work forecast".into());
    }
    Ok(work)
}
pub(crate) fn evaluate(
    task: &str,
    prime: u16,
    sha: &str,
    x: &inputs::Inputs,
    r: &binding::Report,
    eligible: bool,
) -> Result<Vec<Value>> {
    offers(task, prime, sha, r, eligible)?;
    let mut rows = vec![];
    for (i, e) in r.endpoints.iter().enumerate() {
        let Some(t) = &e.training else {
            rows.push(json!({"task":task,"training_mode":MODES[i],"prime":prime,"status":"refused","reason":e.refusal,"stage_sha256":sha}));
            continue;
        };
        t.model.validate()?;
        let work = forecast(&x.task, &t.model)?;
        let (mut actual, mut masks) = (serde_json::Map::new(), serde_json::Map::new());
        for (split, examples) in &x.task.splits {
            let tracked = Tracked {
                model: &t.model,
                known: std::cell::RefCell::new(vec![]),
                observed: std::cell::RefCell::new(vec![]),
            };
            actual.insert(split.clone(), metrics::evaluate(&tracked, examples)?);
            let known = tracked.known.into_inner();
            let observed = tracked.observed.into_inner();
            masks.insert(split.clone(),json!({"meaning":MASK_MEANING,"calls":known.len(),"known_calls":known.iter().filter(|&&k|k).count(),"unknown_calls":known.iter().filter(|&&k|!k).count(),"known_mask":known,"observed_calls":observed.iter().filter(|&&k|k).count(),"default_calls":observed.iter().filter(|&&k|!k).count(),"observed_mask":observed}));
        }
        let actual = Value::Object(actual);
        let a = e.audit.as_ref().ok_or("missing actual endpoint audit")?;
        if actual["train"]["loss"]
            .as_f64()
            .is_none_or(|loss| !loss.is_finite() || (loss - a.train_ce).abs() > 1e-8)
        {
            return Err("raw original CE differs from component audit".into());
        }
        let (primary, all) = passes(&actual, &x.task.selection)?;
        let offer = comparison::Offer {
            task: task.into(),
            mode: t.policy.mode,
            prime,
            train_ce: a.train_ce,
            model_sha256: hash(&serde_json::to_vec(&t.model)?),
            stage_sha256: sha.into(),
        };
        rows.push(json!({"task":task,"training_mode":MODES[i],"prime":prime,"reference":offer.reference(),"status":"evaluated",
            "model_sha256":offer.model_sha256,"stage_sha256":sha,"train_ce":a.train_ce,"primary_split":x.task.selection,
            "metrics":actual,"primary_pass":primary,"all_splits_pass":all,"audit":a,"known_inputs":masks,
            "metric_scalar_work_forecast":work,"offer_eligible":eligible,"raw_label_identification":false,"target_readout_compatibility_proved":false}));
    }
    Ok(rows)
}
pub(crate) fn check(
    task: &str,
    prime: u16,
    sha: &str,
    input: &Value,
    r: &binding::Report,
    eligible: bool,
    rows: &[Value],
) -> Result<()> {
    offers(task, prime, sha, r, eligible)?;
    if rows.len() != 3 {
        return Err("evaluation omitted a mode".into());
    }
    for (i, (row, e)) in rows.iter().zip(&r.endpoints).enumerate() {
        if row["task"] != task
            || row["training_mode"] != MODES[i]
            || row["prime"] != prime
            || row["stage_sha256"] != sha
        {
            return Err("evaluated task/mode/stage changed".into());
        }
        if let Some(t) = &e.training {
            let a = e.audit.as_ref().ok_or("missing endpoint audit")?;
            let primary = input["primary_split"]
                .as_str()
                .ok_or("missing primary split")?;
            let (p, all) = passes(&row["metrics"], primary)?;
            let reference = format!("refined-quotient:{task}:p{prime}:{}", MODES[i]);
            if row["status"] != "evaluated"
                || row["reference"] != reference
                || row["model_sha256"] != hash(&serde_json::to_vec(&t.model)?)
                || row["train_ce"] != a.train_ce
                || row["audit"] != json!(a)
                || row["primary_split"] != primary
                || row["primary_pass"] != p
                || row["all_splits_pass"] != all
                || row["offer_eligible"] != eligible
                || row["raw_label_identification"] != false
                || row["target_readout_compatibility_proved"] != false
                || row["metrics"]["train"]["loss"]
                    .as_f64()
                    .is_none_or(|l| !l.is_finite() || (l - a.train_ce).abs() > 1e-8)
                || row["metric_scalar_work_forecast"]
                    .as_u64()
                    .is_none_or(|w| w > 32_000_000_000)
            {
                return Err("actual endpoint metrics, CE or scope changed".into());
            }
            let masks = row["known_inputs"]
                .as_object()
                .ok_or("missing actual known-input masks")?;
            let metrics = row["metrics"].as_object().ok_or("missing metrics")?;
            if masks.len() != metrics.len() || metrics.keys().any(|s| !masks.contains_key(s)) {
                return Err("prediction mask split scope changed".into());
            }
            for m in masks.values() {
                let known: Vec<bool> = serde_json::from_value(m["known_mask"].clone())?;
                let observed: Vec<bool> = serde_json::from_value(m["observed_mask"].clone())?;
                if m["meaning"] != MASK_MEANING
                    || m["calls"] != known.len()
                    || m["known_calls"] != known.iter().filter(|&&k| k).count()
                    || m["unknown_calls"] != known.iter().filter(|&&k| !k).count()
                    || observed.len() != known.len()
                    || m["observed_calls"] != observed.iter().filter(|&&k| k).count()
                    || m["default_calls"] != observed.iter().filter(|&&k| !k).count()
                    || observed.iter().zip(&known).any(|(&o, &k)| o && !k)
                {
                    return Err("prediction mask counts changed".into());
                }
            }
        } else if row["status"] != "refused" || row["reason"] != json!(e.refusal) {
            return Err("mode refusal changed".into());
        }
    }
    Ok(())
}
