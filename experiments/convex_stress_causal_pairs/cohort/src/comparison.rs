//! Port of quotient-cohort's observed-CE selector at49105730. The parent now
//! includes its complete quotient outcomes; no held-out field enters a choice.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    pub task: String,
    pub mode: training::Mode,
    pub train_ce: f64,
    pub model_sha256: String,
    pub report_sha256: String,
}
impl Offer {
    pub fn reference(&self) -> String {
        format!("causal-pairs:{}:{}", self.task, mode(self.mode))
    }
}
fn sha(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
pub(crate) fn validate(parent: &Value, tasks: &[String]) -> Result<()> {
    let names: BTreeSet<_> = tasks.iter().map(String::as_str).collect();
    let base = &parent["complete_parent"];
    if names.len() != 35
        || tasks.len() != 35
        || base["depth_outcomes"]
            .as_array()
            .is_none_or(|a| a.len() != 336)
        || base["transformer_endpoints"]
            .as_array()
            .is_none_or(|a| a.len() != 35)
        || base["all_parent_costs_charged_once"] != true
        || parent["all35_transformer_endpoints_retained"] != true
        || parent["choose_between_initializers"] != false
        || parent["choices_per_arm"] != 105
        || parent["constructor_accounting"]["charged_once_across_both_arms"] != true
        || parent["constructor_accounting"]["all_prior_costs_retained_in_complete_parent"] != true
    {
        return Err("complete reviewed quotient parent and all ancestor costs required".into());
    }
    let arms = parent["arms"].as_array().ok_or("missing parent arms")?;
    if arms.len() != 2 {
        return Err("both initializer arms required".into());
    }
    let mut policies = BTreeSet::new();
    for arm in arms {
        let policy = arm["initializer"].as_str().ok_or("missing initializer")?;
        let rows = arm["choices"].as_array().ok_or("missing parent choices")?;
        if !POLICIES.contains(&policy) || !policies.insert(policy) || rows.len() != 105 {
            return Err("missing or repeated complete initializer".into());
        }
        let mut cells = BTreeSet::new();
        for row in rows {
            let (task, mode) = (
                row["task"].as_str().ok_or("missing task")?,
                row["training_mode"].as_str().ok_or("missing mode")?,
            );
            if !names.contains(task) || !MODES.contains(&mode) || !cells.insert((task, mode)) {
                return Err("missing, repeated or unknown parent task/mode".into());
            }
            match row["status"].as_str() {
                Some("selected")
                    if row["reference"].is_string()
                        && row["train_ce"]
                            .as_f64()
                            .is_some_and(|v| v.is_finite() && v >= 0.) => {}
                Some("no_model") if row["reference"].is_null() && row["train_ce"].is_null() => {}
                _ => return Err("invalid parent model or ordinary CE".into()),
            }
        }
    }
    Ok(())
}
pub fn select(parent: &Value, offers: &[Offer], tasks: &[String]) -> Result<Value> {
    validate(parent, tasks)?;
    let mut keys = BTreeSet::new();
    for o in offers {
        if !tasks.contains(&o.task)
            || !o.train_ce.is_finite()
            || o.train_ce < 0.
            || !sha(&o.model_sha256)
            || !sha(&o.report_sha256)
            || !keys.insert((&o.task, mode(o.mode)))
        {
            return Err("invalid or duplicate actual causal endpoint".into());
        }
    }
    let mut arms = vec![];
    for policy in POLICIES {
        let prior = parent["arms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["initializer"] == policy)
            .unwrap();
        let mut choices = vec![];
        for task in tasks {
            for mode in MODES {
                let old = prior["choices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|c| c["task"] == *task && c["training_mode"] == mode)
                    .unwrap();
                let mut selected = old["reference"].clone();
                let best = old["train_ce"].as_f64().unwrap_or(f64::INFINITY);
                let mut candidates = vec![];
                if best.is_finite() {
                    candidates
                        .push(json!({"source":"parent","reference":selected,"train_ce":best}));
                }
                if let Some(o) = offers
                    .iter()
                    .find(|o| o.task == *task && super::mode(o.mode) == mode)
                {
                    candidates.push(json!({"source":"causal_pairs","reference":o.reference(),"train_ce":o.train_ce,
                        "model_sha256":o.model_sha256,"report_sha256":o.report_sha256}));
                    if o.train_ce < best - 1e-12 {
                        selected = json!(o.reference());
                    }
                }
                choices.push(json!({"task":task,"training_mode":mode,"prior_reference":old["reference"],"selected":selected,"candidates":candidates}));
            }
        }
        arms.push(json!({"initializer":policy,"choices":choices}));
    }
    Ok(
        json!({"arms":arms,"tolerance":1e-12,"select_between_initializers":false,
        "selection":"original_raw_label_training_ce_then_parent","heldout_inputs":false}),
    )
}
