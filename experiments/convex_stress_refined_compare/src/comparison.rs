//! Parent-first ordinary-CE comparison, ported from49105730; both complete
//! initializer arms remain independent. All held-out fields are ignored here.
use super::*;
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
    pub fn reference(&self) -> String {
        format!(
            "refined-quotient:{}:p{}:{}",
            self.task,
            self.prime,
            mode(self.mode)
        )
    }
}
pub(crate) fn validate(parent: &Value, tasks: &[String]) -> Result<()> {
    let names: BTreeSet<_> = tasks.iter().map(String::as_str).collect();
    let quotient = &parent["complete_parent"];
    let base = &quotient["complete_parent"];
    if names.len() != 35
        || tasks.len() != 35
        || base["depth_outcomes"]
            .as_array()
            .is_none_or(|v| v.len() != 336)
        || base["transformer_endpoints"]
            .as_array()
            .is_none_or(|v| v.len() != 35)
        || base["all_parent_costs_charged_once"] != true
    {
        return Err("complete causal/quotient/depth/transformer ancestry required".into());
    }
    for p in [parent, quotient] {
        if p["all35_transformer_endpoints_retained"] != true
            || p["choose_between_initializers"] != false
            || p["choices_per_arm"] != 105
            || p["constructor_accounting"]["charged_once_across_both_arms"] != true
            || p["constructor_accounting"]["all_prior_costs_retained_in_complete_parent"] != true
        {
            return Err("complete inherited comparison scope and costs required".into());
        }
    }
    let arms = parent["arms"].as_array().ok_or("missing parent arms")?;
    let mut policies = BTreeSet::new();
    if arms.len() != 2 {
        return Err("both initializer arms required".into());
    }
    for arm in arms {
        let policy = arm["initializer"].as_str().ok_or("missing initializer")?;
        let rows = arm["choices"].as_array().ok_or("missing parent choices")?;
        if !POLICIES.contains(&policy) || !policies.insert(policy) || rows.len() != 105 {
            return Err("incomplete or repeated initializer".into());
        }
        let mut cells = BTreeSet::new();
        for r in rows {
            let (task, m) = (
                r["task"].as_str().ok_or("missing task")?,
                r["training_mode"].as_str().ok_or("missing mode")?,
            );
            if !names.contains(task) || !MODES.contains(&m) || !cells.insert((task, m)) {
                return Err("incomplete or repeated parent task/mode".into());
            }
            match r["status"].as_str() {
                Some("selected")
                    if r["reference"].is_string()
                        && r["train_ce"]
                            .as_f64()
                            .is_some_and(|v| v.is_finite() && v >= 0.) => {}
                Some("no_model") if r["reference"].is_null() && r["train_ce"].is_null() => {}
                _ => return Err("invalid original model/CE availability".into()),
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
            || !quotient::catalogue().contains(&o.prime)
            || !o.train_ce.is_finite()
            || o.train_ce < 0.
            || !sha(&o.model_sha256)
            || !sha(&o.stage_sha256)
            || !keys.insert((&o.task, mode(o.mode), o.prime))
        {
            return Err("invalid or duplicated refined offer".into());
        }
    }
    let mut arms = vec![];
    for policy in POLICIES {
        let parent_arm = parent["arms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["initializer"] == policy)
            .unwrap();
        let mut choices = vec![];
        for task in tasks {
            for m in MODES {
                let old = parent_arm["choices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["task"] == *task && r["training_mode"] == m)
                    .unwrap();
                let mut best = old["train_ce"].as_f64().unwrap_or(f64::INFINITY);
                let mut selected = old["reference"].clone();
                let mut candidates = vec![];
                if best.is_finite() {
                    candidates
                        .push(json!({"source":"parent","reference":selected,"train_ce":best}));
                }
                let mut applicable: Vec<_> = offers
                    .iter()
                    .filter(|o| o.task == *task && mode(o.mode) == m)
                    .collect();
                applicable.sort_by_key(|o| o.prime);
                for o in applicable {
                    candidates.push(json!({"source":"refined_quotient","reference":o.reference(),"train_ce":o.train_ce,"prime":o.prime,"model_sha256":o.model_sha256,"stage_sha256":o.stage_sha256}));
                    if o.train_ce < best - 1e-12 {
                        best = o.train_ce;
                        selected = json!(o.reference());
                    }
                }
                choices.push(json!({"task":task,"training_mode":m,"prior_reference":old["reference"],"selected":selected,"candidates":candidates}));
            }
        }
        arms.push(json!({"initializer":policy,"choices":choices}));
    }
    Ok(
        json!({"arms":arms,"tolerance":1e-12,"select_between_initializers":false,"heldout_inputs":false,"selection":"original_raw_ce_then_parent_and_prime_order"}),
    )
}
pub(crate) fn render(
    parent: &Value,
    selection: &Value,
    evaluated: &BTreeMap<String, Value>,
) -> Result<Vec<Value>> {
    let mut arms = vec![];
    for arm in selection["arms"]
        .as_array()
        .ok_or("missing selected arms")?
    {
        let prior = parent["arms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["initializer"] == arm["initializer"])
            .ok_or("missing initializer")?;
        let mut choices = vec![];
        for row in arm["choices"].as_array().ok_or("missing choices")? {
            let old = prior["choices"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["task"] == row["task"] && r["training_mode"] == row["training_mode"])
                .ok_or("missing prior cell")?;
            let chosen = if row["selected"] == old["reference"] {
                old.clone()
            } else {
                let key = row["selected"]
                    .as_str()
                    .ok_or("parent discarded without model")?;
                let new = evaluated.get(key).ok_or("selected model not evaluated")?;
                if new["task"] != row["task"]
                    || new["training_mode"] != row["training_mode"]
                    || new["status"] != "evaluated"
                    || new["offer_eligible"] != true
                {
                    return Err("selected model scope or eligibility differs".into());
                }
                json!({"task":row["task"],"training_mode":row["training_mode"],"status":"selected","reference":key,"train_ce":new["train_ce"],"primary_split":new["primary_split"],"metrics":new["metrics"],"primary_pass":new["primary_pass"],"all_splits_pass":new["all_splits_pass"],"saved_endpoint":new,"baseline":old})
            };
            if chosen["status"] == "selected" {
                let (primary, all) = passes(
                    &chosen["metrics"],
                    chosen["primary_split"]
                        .as_str()
                        .ok_or("missing primary split")?,
                )?;
                if chosen["primary_pass"] != primary || chosen["all_splits_pass"] != all {
                    return Err("declared success differs from actual metrics".into());
                }
            }
            choices.push(chosen);
        }
        let by_mode:Vec<_>=MODES.iter().map(|m|{let rows:Vec<_>=choices.iter().filter(|r|r["training_mode"]==*m).collect();
            json!({"training_mode":m,"cases":rows.len(),"primary_passes":rows.iter().filter(|r|r["primary_pass"]==true).count(),"all_split_passes":rows.iter().filter(|r|r["all_splits_pass"]==true).count(),"no_model":rows.iter().filter(|r|r["status"]=="no_model").count()})}).collect();
        arms.push(json!({"initializer":arm["initializer"],"full35_pass":choices.iter().all(|r|r["primary_pass"]==true),"all_split_diagnostic_pass":choices.iter().all(|r|r["all_splits_pass"]==true),"choices":choices,"by_mode":by_mode}));
    }
    Ok(arms)
}
