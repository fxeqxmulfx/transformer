//! CM45 wraps each exact CM43 prior/supplement tensor. Training records and
//! original minibatch size are checked before diagnostic mixture selection.
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
        let endpoint = head::Endpoint {
            mode,
            source_reference: entry["reference"]
                .as_str()
                .ok_or("missing reference")?
                .to_owned(),
            source_gradient_training_run: mode != head::Mode::Forward,
            source_optimizer_updates: if mode == head::Mode::Forward {
                Some(0)
            } else {
                None
            },
            mixture_batch: batch,
        };
        let load = |value: Value| -> Result<head::Model> {
            let binding = serde_json::from_value(value["binding"].clone())?;
            let parent = head::parent::Model::load(value, &binding)?;
            if parent.source().n != task.vocab
                || parent.source().vocabulary != task.vocab
                || parent.records() != records
            {
                return Err("context source and original records differ".into());
            }
            let pin = head::digest(&parent)?;
            head::Model::new(parent, endpoint.clone(), &pin)
        };
        Ok(Self {
            prior: load(envelope.prior)?,
            supplement: envelope.supplement.map(load).transpose()?,
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
        self.models()
            .map(|m| m.selection().work.optimizer_updates)
            .sum()
    }
    pub(crate) fn info(&self) -> Value {
        let item = |m: &head::Model| {
            json!({"binding":m.binding(),"endpoint":m.endpoint(),"bank":m.bank(),
            "selection":m.selection(),"bounds":m.bounds(),"radius":m.source().radius})
        };
        json!({"prior":item(&self.prior),"supplement":self.supplement.as_ref().map(item),
            "new_optimizer_updates":self.new_updates(),
            "new_gradient_batches":self.models().map(|m|m.selection().work.gradient_batches).sum::<usize>()})
    }
    pub(crate) fn forecast(&self, task: &data::Task) -> Result<u64> {
        let s = self
            .models()
            .flat_map(|m| m.source().blocks.iter())
            .map(|b| b.states())
            .max()
            .ok_or("empty context")? as u64;
        let mut calls = 0;
        for split in task.splits.values() {
            examples(split, task.vocab)?;
            calls += split
                .iter()
                .map(|e| 1 + u64::from(e.field.is_some()))
                .sum::<u64>();
        }
        // Both old and new complete products, including every new mixed
        // state-frequency product and the parent's original context products.
        let work = calls * 32 * s * (2 * task.vocab as u64 + s + 10);
        if work > 64_000_000_000 {
            return Err("context metric forecast exceeds64B work units".into());
        }
        Ok(work)
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
