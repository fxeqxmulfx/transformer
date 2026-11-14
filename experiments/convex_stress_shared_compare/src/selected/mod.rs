//! Construct the fixed prospective choices from original training evidence.
//! Actual selected source histories are replayed before metrics are permitted.
use super::*;
mod supplement;
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", content = "state")]
pub(crate) enum Model {
    Raw(Box<component::Model>),
    Extended(Box<extension::Model>),
}
impl Model {
    pub fn vocabulary(&self) -> usize {
        match self {
            Self::Raw(m) => m.vocabulary,
            Self::Extended(m) => m.prior().vocabulary,
        }
    }
    pub fn prior(&self) -> &component::Model {
        match self {
            Self::Raw(m) => m,
            Self::Extended(m) => m.prior(),
        }
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        if !(2..=4096).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&x| x < 0 || x as usize >= self.vocabulary())
        {
            return Err("prefix outside the original raw-head domain".into());
        }
        let prior = self.prior();
        if prefix[prefix.len() - 2..]
            .iter()
            .any(|&x| x as usize >= prior.n)
        {
            return Err("pair token outside the saved categorical axes".into());
        }
        let cell = prefix[prefix.len() - 1] as usize + prior.n * prefix[prefix.len() - 2] as usize;
        match self {
            Self::Raw(m) => Ok(m.logits_cell(cell)),
            Self::Extended(m) => m.logits(prefix),
        }
    }
    pub fn buffer_bytes(&self) -> Result<u64> {
        match self {
            Self::Raw(m) => Ok(m.bounds()?.buffer_bytes),
            Self::Extended(m) => Ok(m.bounds().buffer_bytes),
        }
    }
}
pub(crate) struct Bundle {
    pub choices: Value,
    pub models: BTreeMap<String, Model>,
    pub causal: crate::causal::imported::Bundle,
    pub raw_record: Value,
    pub shared_records: Vec<Value>,
    pub case_buffer_forecast: u64,
}
pub(crate) fn build(
    request: &Request,
    reviewed: &BTreeMap<String, parents::Reviewed>,
    task: &str,
    x: &inputs::Inputs,
    causal_x: &crate::causal::inputs::Inputs,
    all: &[String],
) -> Result<Bundle> {
    let parent = &reviewed["parent"].summary;
    let raw_pin = &request.parents["raw"];
    let candidates = raw::candidates(raw_pin, &reviewed["raw"].summary, task, x)?;
    let selected = selection::select(parent, &candidates.offers, all)?;
    let mut staged = BTreeMap::new();
    let mut supplemental = BTreeMap::new();
    for policy in POLICIES {
        for m in MODES {
            let choice = cell(&selected, policy, task, m)?;
            if let Some(offer) = candidates
                .offers
                .iter()
                .find(|o| choice["selected"] == o.reference())
                && let std::collections::btree_map::Entry::Vacant(entry) = staged.entry(offer.prime)
            {
                let index = catalogue().iter().position(|&p| p == offer.prime).unwrap();
                let saved = &candidates.record["result"]["stages"][index];
                let stage = raw::imported::load(raw_pin, task, &candidates.record, saved, x, true)?;
                let d = stage
                    .attempt
                    .discovery
                    .as_ref()
                    .ok_or("selected raw discovery omitted")?;
                let h = stage
                    .attempt
                    .head
                    .as_ref()
                    .ok_or("selected raw history omitted")?;
                binding::replay(
                    d,
                    x.policies.as_ref().ok_or("original policies omitted")?,
                    h,
                )?;
                entry.insert(stage);
                supplemental.insert(
                    offer.prime,
                    supplement::load(
                        &request.parents["shared"],
                        &reviewed["shared"].summary,
                        task,
                        offer.prime,
                        x,
                    )?,
                );
            }
        }
    }
    let mut models = BTreeMap::new();
    let mut rows = vec![];
    let mut intermediate = parent.clone();
    for policy in POLICIES {
        for (index, m) in MODES.into_iter().enumerate() {
            let old = cell(parent, policy, task, m)?;
            let choice = cell(&selected, policy, task, m)?;
            let mut row = json!({"initializer":policy,"task":task,"training_mode":m,
            "prior_reference":old["reference"],"selected":choice["selected"],"train_ce":old["train_ce"],
            "raw_selection":choice,"transformation":{"status":"unchanged","reason":"selected model is not a raw kernel"}});
            if let Some(offer) = candidates
                .offers
                .iter()
                .find(|o| choice["selected"] == o.reference())
            {
                let raw = &staged[&offer.prime];
                let d = raw.attempt.discovery.as_ref().unwrap();
                let h = raw.attempt.head.as_ref().unwrap();
                let prior = h.endpoints[index]
                    .training
                    .as_ref()
                    .ok_or("selected raw endpoint omitted")?
                    .model
                    .clone();
                let extra = &supplemental[&offer.prime];
                let mut model = Model::Raw(Box::new(prior.clone()));
                row["train_ce"] = json!(offer.train_ce);
                row["transformation"] = json!({"status":"refused","reason":"same-prime/mode shared endpoint unavailable"});
                if let (Some(sd), Some(sh)) = (&extra.discovery, &extra.report)
                    && let Some(endpoint) = sh.endpoints[index].training.as_ref()
                {
                    match extension::Model::new(
                        prior,
                        endpoint.model.clone(),
                        d,
                        sd,
                        extra.records.clone(),
                    ) {
                        Ok(extended) if extended.coverage().newly_covered > 0 => {
                            row["selected"] =
                                json!(format!("shared-extension:{task}:p{}:{m}", offer.prime));
                            row["transformation"] = json!({"status":"extended","coverage":extended.coverage(),
                              "binding":extended.binding(),"bounds":extended.bounds(),"observed_ce_unchanged":true});
                            model = Model::Extended(Box::new(extended));
                        }
                        Ok(_) => {
                            row["transformation"] =
                                json!({"status":"unchanged","reason":"no newly covered input"})
                        }
                        Err(e) => {
                            row["transformation"] =
                                json!({"status":"refused","reason":e.to_string()})
                        }
                    }
                }
                let key = row["selected"].as_str().unwrap().to_owned();
                if let Some(previous) = models.insert(key, model.clone())
                    && serde_json::to_value(previous)? != serde_json::to_value(model)?
                {
                    return Err("same selected reference has different actual tensors".into());
                }
                // The causal transformation sees the already fixed raw selection.
                let arm = intermediate["arms"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|a| a["initializer"] == policy)
                    .unwrap();
                let entry = arm["choices"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|r| r["task"] == task && r["training_mode"] == m)
                    .unwrap();
                entry["reference"] = row["selected"].clone();
                entry["train_ce"] = row["train_ce"].clone();
            }
            rows.push(row);
        }
    }
    let causal = crate::causal::imported::build(
        &request.checkpoint_root,
        &request.parents["causal"],
        &reviewed["causal"].summary,
        &intermediate,
        task,
        causal_x,
    )?;
    for row in &mut rows {
        let matching = causal
            .rows
            .iter()
            .filter(|r| {
                r["initializer"] == row["initializer"] && r["training_mode"] == row["training_mode"]
            })
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err("causal structural outcome omitted or repeated".into());
        }
        let fixed = matching[0];
        if fixed["status"] == "wrapped" {
            if fixed["prior_reference"] != row["selected"] || fixed["train_ce"] != row["train_ce"] {
                return Err("causal wrapper changed the fixed ordinary-CE choice".into());
            }
            row["selected"] = fixed["reference"].clone();
            row["transformation"] = fixed.clone();
        } else if fixed["status"] == "refused" {
            row["transformation"] = fixed.clone();
        }
    }
    let raw_buffers = staged.values().try_fold(0_u64, |sum, s| -> Result<u64> {
        let h = s.attempt.head.as_ref().unwrap();
        let mut buffers = s.attempt.discovery.as_ref().unwrap().buffer_preflight_bytes;
        if let Some(initial) = &h.initialization {
            buffers = buffers
                .checked_add(initial.bounds.buffer_bytes)
                .ok_or("raw initializer buffer overflow")?;
        }
        for e in &h.endpoints {
            if let Some(t) = &e.training {
                buffers = buffers
                    .checked_add(t.model.bounds()?.buffer_bytes)
                    .ok_or("raw history buffer overflow")?;
            }
        }
        Ok(sum
            .checked_add(buffers)
            .ok_or("raw history buffer sum overflow")?)
    })?;
    let shared_buffers = supplemental.values().try_fold(0_u64, |sum, s| {
        sum.checked_add(
            s.report
                .as_ref()
                .and_then(|h| h.buffer_preflight_bytes)
                .unwrap_or(0),
        )
        .ok_or("shared history buffer sum overflow")
    })?;
    let model_buffers = models.values().try_fold(0_u64, |sum, m| -> Result<u64> {
        Ok(sum
            .checked_add(m.buffer_bytes()?)
            .ok_or("selected model buffer sum overflow")?)
    })?;
    let causal_buffers = causal.models.values().try_fold(0_u64, |sum, m| {
        sum.checked_add(m.bounds().buffer_bytes)
            .ok_or("causal buffer sum overflow")
    })?;
    let forecast = raw_buffers
        .checked_add(shared_buffers)
        .and_then(|n| n.checked_add(model_buffers))
        .and_then(|n| n.checked_add(causal_buffers))
        .ok_or("complete case buffer overflow")?;
    if forecast > 56 * 1024 * 1024 * 1024 {
        return Err("complete selected case exceeds 56 GiB buffer forecast".into());
    }
    Ok(Bundle {
        choices: json!({"choices":rows,"heldout_inputs":false,"no_new_optimization":true}),
        models,
        causal,
        raw_record: candidates.record,
        shared_records: supplemental.into_values().map(|s| s.evidence).collect(),
        case_buffer_forecast: forecast,
    })
}
