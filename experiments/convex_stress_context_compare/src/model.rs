//! Retain CM39's exact prior/supplement route, replacing only each supported
//! ambiguous raw head's fixed unseen feature row. Old unsupported defaults stay.
use super::*;

#[derive(Serialize)]
pub(crate) struct Model {
    prior: context::Model,
    supplement: Option<context::Model>,
}

#[derive(Deserialize)]
struct Envelope {
    kind: String,
    state: Value,
}

impl Model {
    pub(crate) fn from_original(value: Value, task: &data::Task) -> Result<Self> {
        let envelope: Envelope = serde_json::from_value(value)?;
        let (prior, supplement, saved_records, bindings) = match envelope.kind.as_str() {
            "Raw" => (
                serde_json::from_value::<kernel::Model>(envelope.state)?,
                None,
                None,
                None,
            ),
            "Extended" => {
                let mut state = envelope.state;
                let binding = state["binding"].take();
                (
                    serde_json::from_value(state["prior"].take())?,
                    Some(serde_json::from_value::<kernel::Model>(
                        state["supplement"].take(),
                    )?),
                    Some(serde_json::from_value::<Vec<records::Record>>(
                        state["records"].take(),
                    )?),
                    Some(binding),
                )
            }
            _ => return Err("eligible original envelope is not a raw/extended pair head".into()),
        };
        let train = task
            .splits
            .get("train")
            .ok_or("missing original training records")?;
        if prior.n != task.vocab
            || prior.vocabulary != task.vocab
            || train.iter().any(|e| {
                e.tokens.len() != 2
                    || e.targets.len() != 2
                    || e.targets[0] != -100
                    || e.targets[1] < 0
                    || !e.prompt.is_empty()
            })
        {
            return Err("original input is not the source's paired categorical layout".into());
        }
        let records: Vec<_> = train
            .iter()
            .enumerate()
            .map(|(id, e)| records::Record {
                id,
                cell: e.tokens[1] as usize + prior.n * e.tokens[0] as usize,
                label: e.targets[1] as usize,
            })
            .collect();
        if saved_records.as_ref().is_some_and(|r| r != &records) {
            return Err("saved original record IDs/labels changed".into());
        }
        let prior_sha = context::digest(&prior)?;
        let records_sha = context::digest(&records)?;
        if bindings
            .as_ref()
            .is_some_and(|b| b["prior_sha256"] != prior_sha || b["records_sha256"] != records_sha)
        {
            return Err("original extension prior/record binding differs".into());
        }
        let prior = context::Model::new(prior, records.clone(), &prior_sha)?;
        let supplement = if let Some(source) = supplement {
            let sha = context::digest(&source)?;
            if bindings
                .as_ref()
                .is_none_or(|b| b["supplement_sha256"] != sha)
            {
                return Err("original supplement binding differs".into());
            }
            let m = context::Model::new(source, records, &sha)?;
            if m.source().n != prior.source().n
                || m.source().vocabulary != prior.source().vocabulary
            {
                return Err("extension source domains differ".into());
            }
            Some(m)
        } else {
            None
        };
        let bytes = prior.bounds().buffer_bytes
            + supplement.as_ref().map_or(0, |m| m.bounds().buffer_bytes);
        if bytes > 56 * 1024 * 1024 * 1024 {
            return Err("both interpolated heads exceed56GiB forecast".into());
        }
        Ok(Self { prior, supplement })
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
        let item = |m: &context::Model| json!({"binding":m.binding(),"selection":m.selection(),"bounds":m.bounds(),"radius":m.source().radius});
        json!({"prior":item(&self.prior),"supplement":self.supplement.as_ref().map(item),"new_gradient_batches":0})
    }
    pub(crate) fn forecast(&self, task: &data::Task) -> Result<u64> {
        let max_states = std::iter::once(&self.prior)
            .chain(self.supplement.iter())
            .flat_map(|m| m.source().blocks.iter())
            .map(|b| b.states())
            .max()
            .ok_or("empty head")?;
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
                    return Err("metric input is outside original pair domain".into());
                }
                calls += 1 + u64::from(e.field.is_some());
            }
        }
        let work = calls * 16 * max_states as u64 * (task.vocab + 9) as u64;
        if work > kernel::limits::WORK_CAP {
            return Err("context metrics exceed32B work forecast".into());
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
