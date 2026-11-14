//! CM43 wraps actual reviewed CM42 context heads. The CM39 prior/supplement
//! route, context weights and complete coefficient budgets are retained.
use super::*;

#[derive(Serialize)]
pub(crate) struct Model {
    prior: context::Model,
    supplement: Option<context::Model>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    prior: Value,
    supplement: Option<Value>,
}

impl Model {
    pub(crate) fn from_original(value: Value, task: &data::Task) -> Result<Self> {
        let envelope: Envelope = serde_json::from_value(value)?;
        let train = task
            .splits
            .get("train")
            .ok_or("missing original training records")?;
        if train.is_empty()
            || train.iter().any(|e| {
                e.tokens.len() != 2
                    || e.targets.len() != 2
                    || e.targets[0] != -100
                    || e.targets[1] < 0
                    || e.targets[1] as usize >= task.vocab
                    || !e.prompt.is_empty()
                    || e.tokens.iter().any(|&t| t < 0 || t as usize >= task.vocab)
            })
        {
            return Err("original input is not the reviewed pair layout".into());
        }
        let records: Vec<_> = train
            .iter()
            .enumerate()
            .map(|(id, e)| records::Record {
                id,
                cell: e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
                label: e.targets[1] as usize,
            })
            .collect();
        let load = |value: Value| -> Result<context::Model> {
            let binding = serde_json::from_value(value["binding"].clone())?;
            let reviewed = context::context::Model::load(value, &binding)?;
            if reviewed.source().n != task.vocab
                || reviewed.source().vocabulary != task.vocab
                || reviewed.records() != records
            {
                return Err("reviewed context and original training records differ".into());
            }
            let pin = context::digest(&reviewed)?;
            context::Model::new(reviewed, &pin)
        };
        let prior = load(envelope.prior)?;
        let supplement = envelope.supplement.map(load).transpose()?;
        let model = Self { prior, supplement };
        if model.buffer_bytes() > 56 * 1024 * 1024 * 1024 {
            return Err("both scaled context heads exceed56GiB forecast".into());
        }
        Ok(model)
    }
    fn routed(&self, cell: usize) -> &context::Model {
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
        self.prior.bounds().buffer_bytes
            + self
                .supplement
                .as_ref()
                .map_or(0, |m| m.bounds().buffer_bytes)
    }
    pub(crate) fn info(&self) -> Value {
        let item = |m: &context::Model| {
            json!({"binding":m.binding(),"selection":m.selection(),
            "context_selection":m.context().selection(),"bounds":m.bounds(),"radius":m.source().radius})
        };
        json!({"prior":item(&self.prior),"supplement":self.supplement.as_ref().map(item),"new_gradient_batches":0})
    }
    pub(crate) fn forecast(&self, task: &data::Task) -> Result<u64> {
        let max_states = std::iter::once(&self.prior)
            .chain(self.supplement.iter())
            .flat_map(|m| m.source().blocks.iter())
            .map(|b| b.states())
            .max()
            .ok_or("empty context")?;
        let mut calls = 0u64;
        for examples in task.splits.values() {
            if examples.is_empty() {
                return Err("empty original metric split".into());
            }
            for e in examples {
                if e.tokens.len() != 2
                    || e.targets.len() != 2
                    || e.targets[0] != -100
                    || e.targets[1] < 0
                    || e.targets[1] as usize >= task.vocab
                    || !e.prompt.is_empty()
                    || e.tokens.iter().any(|&t| t < 0 || t as usize >= task.vocab)
                {
                    return Err("metric input outside original pair domain".into());
                }
                calls += 1 + u64::from(e.field.is_some());
            }
        }
        let work = calls * 16 * max_states as u64 * (task.vocab + 10) as u64;
        if work > kernel::limits::WORK_CAP {
            return Err("scale metrics exceed32B work forecast".into());
        }
        Ok(work)
    }
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
            .expect("forecast validated original query")
    }
}
