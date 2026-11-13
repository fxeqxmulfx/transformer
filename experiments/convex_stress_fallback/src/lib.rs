//! CM38 at16e34b2f: fixed observed-support routing with a frozen saved parent.
//! Both heads and their storage are charged. Only the candidate coordinates
//! may vary; raw original CE on covered batches has its existing derivative.
pub use causal::{C, Input, Result, metrics};
pub use convex_stress_causal_pairs as causal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
pub mod counts;
pub mod parent;
pub use parent::Parent;
#[cfg(test)]
mod tests;

const BUFFER_LIMIT: u64 = 56 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    pub real_slots: u64,
    pub stored_scalars: u64,
    pub radius: f64,
    pub buffer_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub candidate_sha256: String,
    pub parent_sha256: String,
    pub input_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Model {
    candidate: causal::model::Model,
    parent: Parent,
    binding: Binding,
    bounds: Bounds,
}

// Inference models can only be deserialized through load, which checks the
// external binding and reconstructs the fixed domain and charged bounds.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedModel {
    candidate: causal::model::Model,
    parent: Parent,
    binding: Binding,
    bounds: Bounds,
}

pub fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(convex_stress_counts::hash(&serde_json::to_vec(value)?))
}

fn check_prefix(prefix: &[i64], vocab: usize, context: usize) -> Result<()> {
    if prefix.is_empty()
        || prefix.len() > context
        || prefix.iter().any(|&t| t < 0 || t as usize >= vocab)
    {
        return Err("prefix outside the declared categorical domain".into());
    }
    Ok(())
}

fn input_buffer(input: &Input) -> Result<u64> {
    if input.observations.is_empty() || input.observations.len() > 200_000 {
        return Err("original record count exceeds the declared causal limit".into());
    }
    let tokens = input.observations.iter().try_fold(0usize, |n, s| {
        n.checked_add(s.tokens.len())
            .ok_or("original token count overflow")
    })?;
    if tokens > 20_000_000 {
        return Err("original input storage exceeds20M tokens".into());
    }
    Ok(64 * tokens as u64 + 128 * input.observations.len() as u64)
}

impl Model {
    pub fn new(candidate: causal::model::Model, parent: Parent, input: &Input) -> Result<Self> {
        let input_bytes = input_buffer(input)?;
        candidate.observations(input)?;
        // Bind the support to all original observations, including multiplicity
        // in the input hash. No unseen value is admitted by a forged dictionary.
        let queries: BTreeSet<_> = input
            .observations
            .iter()
            .map(|s| *s.tokens.last().unwrap())
            .collect();
        if queries != candidate.blocks.iter().map(|b| b.query).collect() {
            return Err("candidate query support differs from original observations".into());
        }
        let library = causal::features::library(input.vocabulary, input.context);
        for block in &candidate.blocks {
            let values: BTreeSet<_> = input
                .observations
                .iter()
                .filter(|s| s.tokens.last() == Some(&block.query))
                .map(|s| {
                    block
                        .decoder
                        .candidate
                        .original(&library, input.vocabulary, &s.tokens)
                })
                .collect();
            if values.into_iter().collect::<Vec<_>>() != block.decoder.values {
                return Err("candidate value support differs from original observations".into());
            }
        }
        let (v, context) = parent.shape();
        if v != input.vocabulary || context < input.context {
            return Err("saved parent does not cover the declared input domain".into());
        }
        let a = parent::causal_bounds(&candidate)?;
        let b = parent.bounds()?;
        let bounds = Bounds {
            real_slots: a.real_slots + b.real_slots,
            stored_scalars: a.stored_scalars + b.stored_scalars,
            radius: a.radius + b.radius,
            buffer_bytes: a.buffer_bytes + b.buffer_bytes + input_bytes,
        };
        if bounds.buffer_bytes > BUFFER_LIMIT || !bounds.radius.is_finite() {
            return Err("combined fallback exceeds declared buffer/budget bound".into());
        }
        let binding = Binding {
            candidate_sha256: digest(&candidate)?,
            parent_sha256: digest(&parent)?,
            input_sha256: digest(input)?,
        };
        Ok(Self {
            candidate,
            parent,
            binding,
            bounds,
        })
    }

    pub fn load(value: serde_json::Value, input: &Input, expected: &Binding) -> Result<Self> {
        let saved: SavedModel = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved fallback differs from its external binding".into());
        }
        let rebuilt = Self::new(saved.candidate, saved.parent, input)?;
        if rebuilt.binding != *expected || rebuilt.bounds != saved.bounds {
            return Err("saved fallback tensors, inputs or bounds changed".into());
        }
        Ok(rebuilt)
    }
    pub fn candidate(&self) -> &causal::model::Model {
        &self.candidate
    }
    pub fn parent(&self) -> &Parent {
        &self.parent
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }
    pub fn known(&self, prefix: &[i64]) -> Result<bool> {
        check_prefix(prefix, self.candidate.vocabulary, self.candidate.context)?;
        Ok(self
            .candidate
            .known(&prefix.iter().map(|&x| x as usize).collect::<Vec<_>>()))
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        if self.known(prefix)? {
            Ok(self
                .candidate
                .logits_tokens(&prefix.iter().map(|&x| x as usize).collect::<Vec<_>>()))
        } else {
            self.parent.logits(prefix)
        }
    }
    /// Derivative in the free candidate coordinates only. Record IDs retain
    /// repeated inputs and labels; a final shorter batch uses its own length.
    pub fn batch_gradient(&self, input: &Input, ids: &[usize]) -> Result<(f64, Vec<f64>)> {
        if digest(input)? != self.binding.input_sha256
            || ids.is_empty()
            || ids.iter().any(|&i| i >= input.observations.len())
            || ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len()
        {
            return Err("invalid original record-ID batch".into());
        }
        let samples = self.candidate.observations(input)?;
        let selected: Vec<_> = ids.iter().map(|&i| samples[i]).collect();
        causal::training::gradient(&self.candidate, &selected)
    }
}

impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.candidate.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.logits(prefix)
            .expect("validated fallback prediction domain")
    }
}
