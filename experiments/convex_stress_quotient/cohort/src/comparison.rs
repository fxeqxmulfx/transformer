//! Training-score-only extension of both complete task226 initializer arms.
//! This selector contains no held-out metrics or target/readout/certificate rule.
use crate::*;
use std::collections::BTreeSet;
const MODES: [&str; 3] = ["full_batch", "minibatch", "forward_only"];
const POLICIES: [&str; 2] = ["minimum_ce", "first_small_ce"];
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    pub task: String,
    pub mode: training::Mode,
    pub prime: u16,
    pub train_ce: f64,
    pub model_sha256: String,
    pub stage_sha256: String,
}
impl Offer {
    pub fn mode(&self) -> &'static str {
        match self.mode {
            training::Mode::Full => MODES[0],
            training::Mode::Mini => MODES[1],
            training::Mode::Forward => MODES[2],
        }
    }
    pub fn reference(&self) -> String {
        format!("quotient:{}:p{}:{}", self.task, self.prime, self.mode())
    }
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("missing comparison {key}").into())
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing comparison {key}").into())
}
fn sha(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())
}
fn validate(parent: &Value, offers: &[Offer], tasks: &[String]) -> Result<()> {
    let names: BTreeSet<_> = tasks.iter().map(String::as_str).collect();
    if names.len() != 35
        || tasks.len() != 35
        || array(parent, "arms")?.len() != 2
        || array(parent, "depth_outcomes")?.len() != 336
        || array(parent, "transformer_endpoints")?.len() != 35
        || parent["all_parent_costs_charged_once"] != true
        || parent["choose_between_initializers"] != false
    {
        return Err("complete reviewed35-task/two-arm parent required".into());
    }
    let mut policies = BTreeSet::new();
    for arm in array(parent, "arms")? {
        let policy = text(arm, "initializer")?;
        if !POLICIES.contains(&policy)
            || !policies.insert(policy)
            || array(arm, "choices")?.len() != 105
        {
            return Err("missing or repeated complete initializer arm".into());
        }
        let mut cells = BTreeSet::new();
        for row in array(arm, "choices")? {
            let (name, mode) = (text(row, "task")?, text(row, "training_mode")?);
            if !names.contains(name) || !MODES.contains(&mode) || !cells.insert((name, mode)) {
                return Err("missing, unknown or repeated task/mode in parent".into());
            }
            match row["status"].as_str() {
                Some("selected")
                    if row["reference"].is_string()
                        && row["train_ce"]
                            .as_f64()
                            .is_some_and(|v| v.is_finite() && v >= 0.) => {}
                Some("no_model") if row["reference"].is_null() && row["train_ce"].is_null() => {}
                _ => return Err("invalid parent model availability or training CE".into()),
            }
        }
    }
    let mut keys = BTreeSet::new();
    for offer in offers {
        if !names.contains(offer.task.as_str())
            || !component::catalogue().contains(&offer.prime)
            || !offer.train_ce.is_finite()
            || offer.train_ce < 0.
            || !sha(&offer.model_sha256)
            || !sha(&offer.stage_sha256)
            || !keys.insert((&offer.task, offer.mode(), offer.prime))
        {
            return Err("invalid or duplicated new saved endpoint offer".into());
        }
    }
    Ok(())
}
pub fn select(parent: &Value, offers: &[Offer], tasks: &[String]) -> Result<Value> {
    validate(parent, offers, tasks)?;
    let mut arms = vec![];
    for policy in POLICIES {
        let arm = array(parent, "arms")?
            .iter()
            .find(|a| a["initializer"] == policy)
            .ok_or("missing initializer")?;
        let mut choices = vec![];
        for task in tasks {
            for mode in MODES {
                let old = array(arm, "choices")?
                    .iter()
                    .find(|c| c["task"] == *task && c["training_mode"] == mode)
                    .ok_or("missing old choice")?;
                let mut selected = old["reference"].clone();
                let mut best = old["train_ce"].as_f64().unwrap_or(f64::INFINITY);
                let mut candidates = vec![];
                if best.is_finite() {
                    candidates
                        .push(json!({"source":"parent","reference":selected,"train_ce":best}));
                }
                let mut applicable: Vec<_> = offers
                    .iter()
                    .filter(|o| o.task == *task && o.mode() == mode)
                    .collect();
                applicable.sort_by_key(|o| o.prime);
                for offer in applicable {
                    let reference = offer.reference();
                    candidates.push(json!({"source":"quotient","reference":reference,"train_ce":offer.train_ce,
                        "model_sha256":offer.model_sha256,"stage_sha256":offer.stage_sha256,"prime":offer.prime}));
                    if offer.train_ce < best - 1e-12 {
                        best = offer.train_ce;
                        selected = json!(reference);
                    }
                }
                choices.push(
                    json!({"task":task,"training_mode":mode,"prior_reference":old["reference"],
                    "selected":selected,"candidates":candidates}),
                );
            }
        }
        arms.push(json!({"initializer":policy,"choices":choices}));
    }
    Ok(
        json!({"arms":arms,"tolerance":1e-12,"select_between_initializers":false,
        "selection":"original_raw_label_training_ce_then_earlier_order","heldout_inputs":false}),
    )
}

#[cfg(test)]
mod tests;
