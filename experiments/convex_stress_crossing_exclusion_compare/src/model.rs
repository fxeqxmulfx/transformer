//! CM50 wraps each exact CM49 prior/supplement model. Original records, source,
//! mode and minibatch metadata are checked before fixed exclusion construction.
use super::*;

#[derive(Debug, PartialEq, Serialize)]
pub(crate) struct Model {
    prior: head::Model,
    supplement: Option<head::Model>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    prior: Value,
    supplement: Option<Value>,
}

fn examples(examples: &[data::Example], vocab: usize) -> Result<()> {
    if examples.is_empty()
        || examples.iter().any(|e| {
            e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || e.targets[1] as usize >= vocab
                || !e.prompt.is_empty()
                || e.tokens.iter().any(|&t| t < 0 || t as usize >= vocab)
        })
    {
        return Err("input outside original pair layout".into());
    }
    Ok(())
}

impl Model {
    pub(crate) fn construct(
        value: Value,
        task: &data::Task,
        entry: &Value,
        batch: usize,
    ) -> Result<Self> {
        let envelope: Envelope = serde_json::from_value(value)?;
        let train = task.splits.get("train").ok_or("missing original records")?;
        examples(train, task.vocab)?;
        let records: Vec<_> = train
            .iter()
            .enumerate()
            .map(|(id, e)| records::Record {
                id,
                cell: e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
                label: e.targets[1] as usize,
            })
            .collect();
        let mode = import::mode(&entry["training_mode"])?;
        let load = |value: Value| -> Result<head::parent::Model> {
            let binding = serde_json::from_value(value["binding"].clone())?;
            let parent = head::parent::Model::load(value, &binding)?;
            if parent.source().n != task.vocab
                || parent.source().vocabulary != task.vocab
                || parent.records() != records
                || parent.endpoint().mode != mode
                || parent.endpoint().mixture_batch != batch
                || entry["reference"]
                    != format!(
                        "raw-exclusion:raw-regions:gated-context:context-mixture:{}",
                        parent.endpoint().source_reference
                    )
            {
                return Err("context source and original records differ".into());
            }
            Ok(parent)
        };
        let prior = load(envelope.prior)?;
        let supplement = envelope.supplement.map(load).transpose()?;
        let bytes = std::iter::once(&prior)
            .chain(supplement.iter())
            .try_fold(0u64, |total, model| -> Result<u64> {
                Ok(total + head::forecast(model)?.buffer_bytes)
            })?;
        if bytes > CASE_BUFFER_LIMIT {
            return Err("complete endpoint exceeds preallocation forecast".into());
        }
        let construct = |parent| -> Result<head::Model> {
            let pin = head::digest(&parent)?;
            head::Model::new(parent, &pin)
        };
        Ok(Self {
            prior: construct(prior)?,
            supplement: supplement.map(construct).transpose()?,
        })
    }
    fn models(&self) -> impl Iterator<Item = &head::Model> {
        std::iter::once(&self.prior).chain(self.supplement.iter())
    }
    fn routed(&self, cell: usize) -> &head::Model {
        if !self.prior.source().known(cell)
            && let Some(m) = &self.supplement
            && m.source().known(cell)
        {
            m
        } else {
            &self.prior
        }
    }
    pub(crate) fn buffer_bytes(&self) -> u64 {
        self.models().map(|m| m.bounds().buffer_bytes).sum()
    }
    pub(crate) fn construction_work(&self) -> u64 {
        self.models().map(|m| m.bounds().total_scalar_work).sum()
    }
    pub(crate) fn new_updates(&self) -> usize {
        self.models().map(|m| m.work().optimizer_updates).sum()
    }
    pub(crate) fn info(&self) -> Value {
        let item = |m: &head::Model| {
            json!({"binding":m.binding(),"endpoint":m.endpoint(),
            "parent_geometry_sha256":head::digest(&m.parent().parent().geometry()).expect("serializable fixed geometry"),
            "rows_sha256":head::digest(&m.rows()).expect("serializable rational rows"),
            "statistics":m.statistics(),"work":m.work(),"bounds":m.bounds(),
            "head_radius":m.source().radius,"confidence_scale":1.,
            "target_convexity_verified":false,"candidate_pool_closure_proved":false,
            "unconditional_raw_label_identification":false})
        };
        json!({"prior":item(&self.prior),"supplement":self.supplement.as_ref().map(item),
            "new_optimizer_updates":self.new_updates(),
            "new_local_row_updates":0,
            "new_context_rows":self.models().map(|m|m.statistics().identified_cells).sum::<usize>(),
            "new_gradient_batches":self.models().map(|m|m.work().gradient_batches).sum::<usize>()})
    }
    pub(crate) fn forecast(&self, task: &data::Task) -> Result<u64> {
        let mut calls = 0;
        for split in task.splits.values() {
            examples(split, task.vocab)?;
            calls += 2 * split
                .iter()
                .map(|e| 1 + u64::from(e.field.is_some()))
                .sum::<u64>();
            calls += 2 * split.len() as u64;
        }
        // Both complete metric paths plus two additional complete products for
        // each raw-transition audit. Every route uses at most this forecast.
        let per_call = self
            .models()
            .map(|m| m.bounds().prediction_scalar_work_per_query)
            .max()
            .ok_or("empty context")?;
        let work = calls.checked_mul(per_call).ok_or("metric-work overflow")?;
        if work > 64_000_000_000 {
            return Err("context metric forecast exceeds64B work units".into());
        }
        Ok(work)
    }
    pub(crate) fn audit(&self, split: &[data::Example]) -> Result<Value> {
        examples(split, self.prior.source().vocabulary)?;
        let (mut correct, mut parent_correct, mut changed, mut fixed, mut regressed) =
            (0, 0, 0, 0, 0);
        let (mut identified, mut identified_correct) = (0, 0);
        for e in split {
            let n = self.prior.source().n;
            let cell = e.tokens[1] as usize + n * e.tokens[0] as usize;
            let m = self.routed(cell);
            let new = argmax(&m.logits(&e.tokens)?)?;
            let old = argmax(&m.parent().logits(&e.tokens)?)?;
            let target = e.targets[1] as usize;
            correct += usize::from(new == target);
            parent_correct += usize::from(old == target);
            changed += usize::from(new != old);
            fixed += usize::from(new == target && old != target);
            regressed += usize::from(new != target && old == target);
            if m.rows()[cell].is_some() {
                identified += 1;
                identified_correct += usize::from(new == target);
            }
        }
        Ok(
            json!({"targets":split.len(),"correct":correct,"parent_correct":parent_correct,
            "changed":changed,"fixed":fixed,"regressed":regressed,
            "errors":split.len()-correct,"parent_errors":split.len()-parent_correct,
            "newly_identified_queries":identified,"newly_identified_correct":identified_correct,
            "newly_identified_errors":identified-identified_correct}),
        )
    }
    pub(crate) fn load(value: Value, expected: &Self) -> Result<Self> {
        let saved: Envelope = serde_json::from_value(value)?;
        let prior = head::Model::load(saved.prior, expected.prior.binding())?;
        let supplement = match (saved.supplement, &expected.supplement) {
            (Some(value), Some(m)) => Some(head::Model::load(value, m.binding())?),
            (None, None) => None,
            _ => return Err("saved supplement presence changed".into()),
        };
        Ok(Self { prior, supplement })
    }
}

/// Original metrics choose the first maximum, with total-order signed zeros.
pub(crate) fn argmax(z: &[f64]) -> Result<usize> {
    if z.is_empty() || z.iter().any(|x| !x.is_finite()) {
        return Err("invalid actual logits".into());
    }
    Ok((0..z.len())
        .max_by(|&a, &b| z[a].total_cmp(&z[b]).then(b.cmp(&a)))
        .unwrap())
}

impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.prior.source().vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let n = self.prior.source().n;
        let cell = prefix[prefix.len() - 1] as usize + n * prefix[prefix.len() - 2] as usize;
        self.routed(cell)
            .logits(prefix)
            .expect("forecast validated original prediction")
    }
}
pub(crate) struct Parent<'a>(pub &'a Model);
impl metrics::Predictor for Parent<'_> {
    fn vocab(&self) -> usize {
        self.0.prior.source().vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let n = self.0.prior.source().n;
        let cell = prefix[prefix.len() - 1] as usize + n * prefix[prefix.len() - 2] as usize;
        self.0
            .routed(cell)
            .parent()
            .logits(prefix)
            .expect("forecast validated parent prediction")
    }
}
